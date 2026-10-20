"""Subagent specifications: one row per subagent type.

The single pattern for the subagent surface: adding a model-invocable
subagent means an agent class, a rubric, and one row here. Every consumer
reads its policy from this table — `tools/implementations/subagent_tool.py`
(the operation enum and continue/dismiss validation),
`working_memory/trinkets/subagent_results_trinket.py` (per-type result
lifecycle), and `agents/spawn.py` (spawn policy and failure publication).

Discovery is a separate concern: `agents/triggers/registry.py:TRIGGER_CLASSES`
owns which agents the dispatcher polls for; this table owns invocation and
presentation.

The dispatch protocol: every invocable agent class implements the
classmethod `activate(tool_repo, event_bus, *, subject, context, blocking)`;
types with `continuable=True` additionally implement
`continue_task(tool_repo, event_bus, *, task_id, subject, context,
blocking) -> dict | None` — the class owns its own mechanics (sync vs async,
WorkItem context shape, blocking-cap application, refusal prose, result
envelopes) and returns None from `continue_task` when the task_id is not its
surface (the claim protocol — `subagent_tool` tries each continuable type and
falls through to its generic not-found envelope; non-continuable types are
refused at the spec flag and never reach the class). `subagent_tool` reaches
these classmethods through `load_agent_class(spec)` and holds no per-type
knowledge beyond the spec flags.

Import-light by contract (the heavy-chain rule, `agents/implementations/AGENTS.md`):
agent modules pull tool modules which pull heavy clients, so `agent_class_path`
is a lazy import string — no agent class is imported at module level here, and
dispatch-path consumers must keep it that way.
"""
from dataclasses import dataclass
import importlib


@dataclass(frozen=True)
class SubagentSpec:
    """Policy row for one subagent type.

    Fields:
        agent_type: the model-facing type name (matches the agent class's
            `agent_id`); keys `SUBAGENT_SPECS` and every entry the unified
            results trinket stores.
        interface_name: the type's primary `sidebar_activity` interface.
            Work items may carry mode-specific interfaces (the curator's
            `_integration`/`_floor`); spawn accounting always keys off the
            work item's own `interface_name`, never this field.
        model_route: the fixed LLM route the agent class runs on.
        invocable: whether the model can start this type via
            `subagent_tool` `activate`. False = service-dispatched only.
        blocking_only: the type runs synchronously on the caller's thread
            (`run_blocking`) and never spawns a background thread.
        continuable: whether `subagent_tool` `continue` may target this
            type. Phone-a-friend resumes its thread; forage refines a
            successful result. Refusal semantics for non-continuable types
            are the tool's.
        publishes_pending: `agents/spawn.py` publishes a `pending` entry to
            the unified results trinket at spawn time.
        dismissable: whether `subagent_tool` `dismiss` may target this
            type's entries.
        result_ttl_turns: turns a `success` entry stays in the unified
            results trinket. None = persists until dismissed or segment
            collapse.
        error_ttl_turns: turns a `timeout`/`failed` entry stays. None = the
            type never renders in the results trinket (and a spawn crash is
            logged, not published).
        agent_class_path: "module:ClassName" — imported lazily at spawn.
    """

    agent_type: str
    interface_name: str
    model_route: str
    invocable: bool
    blocking_only: bool
    continuable: bool
    publishes_pending: bool
    dismissable: bool
    result_ttl_turns: int | None
    error_ttl_turns: int | None
    agent_class_path: str


SUBAGENT_SPECS: dict[str, SubagentSpec] = {
    "talktoanyone": SubagentSpec(
        agent_type="talktoanyone",
        interface_name="talktoanyone",
        model_route="other",
        invocable=True,
        blocking_only=True,
        continuable=True,
        publishes_pending=False,
        dismissable=False,
        result_ttl_turns=None,
        error_ttl_turns=None,
        agent_class_path="agents.implementations.talktoanyone_agent:TalkToAnyoneAgent",
    ),
    "phoneafriend": SubagentSpec(
        agent_type="phoneafriend",
        interface_name="phoneafriend",
        model_route="other",
        invocable=True,
        blocking_only=True,
        continuable=True,
        publishes_pending=False,
        dismissable=False,
        result_ttl_turns=None,
        error_ttl_turns=None,
        agent_class_path="agents.implementations.phoneafriend_agent:PhoneAFriendAgent",
    ),
    "forage": SubagentSpec(
        agent_type="forage",
        interface_name="forage",
        model_route="batch",
        invocable=True,
        blocking_only=False,
        continuable=True,
        publishes_pending=True,
        dismissable=True,
        # success persists until dismissed or collapse; errors fade after 5
        result_ttl_turns=None,
        error_ttl_turns=5,
        agent_class_path="agents.implementations.forage_agent:ForageAgent",
    ),
    "whilethecatsaway": SubagentSpec(
        agent_type="whilethecatsaway",
        interface_name="whilethecatsaway",
        model_route="batch",
        invocable=True,
        blocking_only=False,
        continuable=False,
        publishes_pending=True,
        dismissable=True,
        # every status auto-expires — informational, not a working document
        result_ttl_turns=8,
        error_ttl_turns=8,
        agent_class_path="agents.implementations.whilethecatsaway_agent:WhileTheCatsAwayAgent",
    ),
    "memory_curator": SubagentSpec(
        agent_type="memory_curator",
        interface_name="memory_curator",
        model_route="batch",
        invocable=False,
        blocking_only=False,
        continuable=False,
        publishes_pending=False,
        dismissable=True,
        # success persists until the next run's stable task_id replaces it
        result_ttl_turns=None,
        error_ttl_turns=5,
        agent_class_path="agents.implementations.memory_curator_agent:MemoryCuratorAgent",
    ),
    # Dispatcher-dispatched only (never model-invocable, never in the results
    # trinket) — present because every spawn site routes through the seam.
    "heartbeat": SubagentSpec(
        agent_type="heartbeat",
        interface_name="heartbeat",
        model_route="fast",
        invocable=False,
        blocking_only=False,
        continuable=False,
        publishes_pending=False,
        dismissable=False,
        result_ttl_turns=None,
        error_ttl_turns=None,
        agent_class_path="agents.implementations.heartbeat_agent:HeartbeatAgent",
    ),
}


def load_agent_class(spec: SubagentSpec) -> type:
    """Import and return the agent class named by spec.agent_class_path."""
    module_path, _, class_name = spec.agent_class_path.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def spec_for_agent_class(agent_class: type) -> SubagentSpec:
    """Resolve the spec row for a SidebarAgent subclass.

    Keys on the class's `agent_id`, so trigger-held classes
    (HeartbeatAgent, MemoryCuratorAgent) resolve without imports here.
    """
    agent_id = getattr(agent_class, "agent_id", None)
    spec = SUBAGENT_SPECS.get(agent_id)
    if spec is None:
        raise KeyError(
            f"No SubagentSpec row for agent_id {agent_id!r} "
            f"({agent_class.__name__}) — add the row in agents/specs.py"
        )
    return spec
