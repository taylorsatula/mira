"""The single agent-spawn seam.

Every site that starts a background agent thread routes through
`spawn_subagent`: the dispatcher poll (`agents/sidebar.py`), the tool loop's
`subagent_tool` activate path, and the collapse handler's integration
curator (`cns/services/segment_collapse_handler.py`). The process-global
in-flight registry below is the only concurrency accounting for agent
threads — three writer contexts reach it concurrently (the APScheduler
poll thread, the tool loop's parallel executor in
`cns/services/tool_loop.py`, and the turn thread), so the cap check, the
duplicate-key guard, the insert, and the thread start are atomic under one
lock, and the liveness prune runs under the same lock.

`sidebar_dispatcher.max_concurrent_agents` is the cap for every spawn, not
just dispatcher-dispatched ones. At cap the seam refuses the spawn and the
caller decides what that means: the tool reports a capacity failure to the
model, the dispatcher leaves the item unmarked so the next poll
rediscovers it, and the collapse handler skips with a log (the memory
floor trigger re-samples untended memories later).

Blocking runs (`SidebarAgent.run_blocking`) take no slot: they spawn no
thread, and the caller's turn plus `blocking_agent_timeout_seconds` bound
them. Call `run_blocking` directly; never route it through this module.

Agent classes are imported lazily inside the thread body (the heavy-chain
rule, `agents/implementations/AGENTS.md`): agent modules pull tool modules
which pull heavy clients, so this module imports no agent class at module
level.
"""
import logging
import threading
from contextvars import copy_context
from typing import TYPE_CHECKING

from agents.specs import load_agent_class

if TYPE_CHECKING:
    from agents.specs import SubagentSpec
    from agents.sidebar import WorkItem
    from cns.integration.event_bus import EventBus
    from tools.repo import ToolRepository

logger = logging.getLogger(__name__)

_registry_lock = threading.Lock()
# Keyed "{interface_name}:{item_id}" — process-global across users, mirroring
# the dispatcher's historical in-flight map. Values are the spawned threads;
# unstarted and finished threads are pruned by liveness under the lock.
_in_flight: dict[str, threading.Thread] = {}


def _prune_dead_locked() -> None:
    """Drop non-live threads. Caller must hold _registry_lock."""
    for key in [k for k, t in _in_flight.items() if not t.is_alive()]:
        del _in_flight[key]


def is_in_flight(interface_name: str, item_id: str) -> bool:
    """Whether an agent thread is live for this interface/item pair."""
    with _registry_lock:
        _prune_dead_locked()
        return f"{interface_name}:{item_id}" in _in_flight


def in_flight_count() -> int:
    """Live agent-thread count against the process-global cap."""
    with _registry_lock:
        _prune_dead_locked()
        return len(_in_flight)


def publish_result(
    event_bus: 'EventBus',
    agent_type: str,
    task_id: str,
    status: str,
    data: dict,
) -> None:
    """Publish one status update to the unified results trinket.

    The single producer-side publisher: the seam (pending, crash), the
    agents' own continue paths (refine auto-dismiss), and subagent_tool's
    dismiss all emit through here so the event shape cannot fork.
    """
    from cns.core.events import UpdateTrinketEvent

    event_bus.publish(UpdateTrinketEvent.create(
        continuum_id='sidebar',
        target_trinket='SubagentResultsTrinket',
        context={
            'task_id': task_id,
            'status': status,
            'agent_type': agent_type,
            **data,
        },
    ))


def capacity_result() -> dict:
    """The at-cap refusal envelope shared by every tool-dispatched activate.

    `success: False` is deliberate — the tool loop counts it as a reported
    error, which is the truthful outcome: nothing was dispatched.
    """
    return {
        "success": False,
        "status": "capacity",
        "message": (
            f"Not dispatched: {in_flight_count()} subagents are "
            "already running (the process is at its concurrent-agent "
            "cap). Retry later — a fresh activate once a running "
            "task finishes."
        ),
    }


def _run_agent(
    spec: 'SubagentSpec',
    work_item: 'WorkItem',
    event_bus: 'EventBus',
    tool_repo: 'ToolRepository',
) -> None:
    """Thread body — construct and run the agent; publish a terminal
    failure if the thread dies outside the agent's own error handling."""
    try:
        agent_class = load_agent_class(spec)
        agent = agent_class(tool_repo)
        agent.run(work_item, event_bus)
    except Exception as e:
        logger.error(
            "Subagent thread crashed for %s task %s: %s",
            spec.agent_type, work_item.item_id[:8], e,
            exc_info=True,
        )
        # error_ttl_turns None marks a type with no results-feed lifecycle
        # (heartbeat: the activity row is its surface) — log, don't publish.
        if spec.error_ttl_turns is None:
            return
        try:
            publish_result(event_bus, spec.agent_type, work_item.item_id,
                           'failed', {
                               'error': str(e),
                               'error_type': 'ThreadCrash',
                           })
        except Exception:
            logger.exception(
                "Failed to publish crash status for %s task %s",
                spec.agent_type, work_item.item_id[:8],
            )


def spawn_subagent(
    spec: 'SubagentSpec',
    work_item: 'WorkItem',
    event_bus: 'EventBus',
    *,
    tool_repo: 'ToolRepository',
) -> bool:
    """Start an agent thread through the process-global registry.

    Returns True if the agent was spawned; False if the process is at the
    `max_concurrent_agents` cap or an agent is already in flight for this
    interface/item pair — the caller owns the refusal semantics (see the
    module docstring). The thread runs with a copy of the caller's context,
    so spawn from a thread with the user contextvar set.

    For `publishes_pending` specs a `pending` entry is published after the
    start; the trinket's pending handler creates the entry only when
    absent, so a pending that loses the race to the agent's own first
    update is dropped instead of overwriting it. A failed pending publish
    is logged, not raised — the agent is already running and its own
    updates still surface the result.
    """
    from config.config_manager import config

    cap = config.sidebar_dispatcher.max_concurrent_agents

    ctx = copy_context()
    thread = threading.Thread(
        target=ctx.run,
        args=(_run_agent, spec, work_item, event_bus, tool_repo),
        name=f"subagent-{spec.agent_type}-{work_item.item_id[:8]}",
        daemon=True,
    )

    with _registry_lock:
        _prune_dead_locked()
        if len(_in_flight) >= cap:
            return False
        key = f"{work_item.interface_name}:{work_item.item_id}"
        if key in _in_flight:
            return False
        _in_flight[key] = thread
        # Start under the lock: is_alive() is False before start(), so a
        # prune landing between insert and start would evict the entry —
        # defeating both the cap and the duplicate-key guard.
        try:
            thread.start()
        except Exception:
            del _in_flight[key]
            raise

    if spec.publishes_pending:
        try:
            publish_result(event_bus, spec.agent_type, work_item.item_id,
                           'pending', {
                               'subject': work_item.context.get('subject', ''),
                           })
        except Exception:
            logger.warning(
                "Pending publish failed for %s task %s — agent is running; "
                "its own updates will surface the result",
                spec.agent_type, work_item.item_id[:8], exc_info=True,
            )

    run_count = work_item.context.get('run_count', 1)
    logger.info(
        "Spawned %s for %s:%s (run %s)",
        spec.agent_type, work_item.interface_name,
        work_item.item_id[:8], run_count,
    )
    return True
