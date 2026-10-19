"""
ForageAgent -- background research on the SidebarAgent base, and the only
type whose `continue` refines a completed briefing.

`build_initial_message` reads `subject`, `context`, and `previous_result`
from the work item; when `previous_result` is present the run is a
refinement of an earlier briefing, and that briefing is handed in as the
starting material so the agent deepens rather than repeats it. `activate`
and `continue_task` both build their work item through `_work_item`, which
writes `previous_result` only when there is one.

`continue_task` is the refine path: it claims a task_id against the results
trinket, refuses with a reason when the entry is still running or ended
without a result, and on a usable success entry dismisses that entry
(`publish_result`) before respawning -- so a refine never leaves two live
entries for one line of research under different ids. It returns None when
the task_id is not an active forage entry, which is the claim protocol's
"not mine" answer.

Blocking runs (`_run_blocking`) rewrite the instance's `max_iterations` from
`config.subagent_tool.forage_blocking_max_iterations` before `run_blocking`:
the class's own cap is calibrated against the background wall-clock budget,
and a blocking call inheriting it can outlive the caller's turn. The progress
bar `_iteration_status` renders rescales with the rewritten cap.

Completions publish to the unified results trinket through
`_get_completion_trinket` and `_build_completion_context`; `on_overwatch_update`
publishes the per-iteration `in_progress` entry, and a failed publish is
swallowed at debug so a trinket problem never ends a run.
"""
import logging
from typing import Any, TYPE_CHECKING

from agents.base import SidebarAgent, load_agent_prompt

if TYPE_CHECKING:
    from agents.sidebar import WorkItem
    from cns.integration.event_bus import EventBus
    from tools.repo import ToolRepository

logger = logging.getLogger(__name__)


class ForageAgent(SidebarAgent):
    agent_id = "forage"
    model_config_name = "batch"
    available_tools = ["continuum_tool", "memory_tool", "web_tool"]
    inherit_base_prompt = False
    max_iterations = 20
    # Runnable inline inside a bounded turn via subagent_tool's blocking
    # param — bounded by sidebar_dispatcher.blocking_agent_timeout_seconds, not
    # the 600s background override, so a blocking call fits inside any caller turn.
    blocking_supported = True
    # Overwatch observer on `fast`: keeps the per-iteration one-shot off
    # the local primary slot (single KV cache shared with the main chat).
    overwatch_model_config_name = "fast"

    _PROGRESS_BAR_WIDTH = 20

    def __init__(self, tool_repo: 'ToolRepository'):
        super().__init__(tool_repo)

    def get_agent_prompt(self, work_item: 'WorkItem') -> str:
        return load_agent_prompt("forage_system.txt")

    def _iteration_status(self, iteration: int) -> str:
        if iteration > self.max_iterations:
            return (
                "# Research Progress\n\n"
                f"Turn {iteration} (grace)\n\n"
                "This is the final turn. Call sidebar_tool complete_task "
                "now with whatever you have."
            )
        width = self._PROGRESS_BAR_WIDTH
        filled = round(width * iteration / self.max_iterations)
        filled = max(0, min(width, filled))
        bar = "█" * filled + "░" * (width - filled)
        pct = round(100 * iteration / self.max_iterations)
        return (
            "# Research Progress\n\n"
            f"Turn {iteration} of {self.max_iterations}  "
            f"[{bar}]  {pct}%\n\n"
            "This bar fills as iterations pass. Wind down exploration "
            "and consolidate findings before it reaches 100%. Call "
            "sidebar_tool complete_task with your briefing before the "
            "final turn; do not wait to be forced out."
        )

    def build_initial_message(self, work_item: 'WorkItem') -> str:
        subject = work_item.context.get('subject', '')
        context = work_item.context.get('context', '')
        previous_result = work_item.context.get('previous_result')

        if previous_result:
            return (
                f"**Previous forage result (you are refining this):**\n"
                f"{previous_result}\n\n"
                f"**Refinement:** {subject}\n\n"
                f"**Context:** {context}\n\n"
                "The previous search found the above. Refine, correct, or "
                "dig deeper based on the refinement instruction. Build on "
                "what was found -- don't repeat the same searches."
            )
        return (
            f"**Query:** {subject}\n\n"
            f"**Context:** {context}\n\n"
            "Search for information relevant to this query. "
            "Use the context to understand what would be most useful."
        )

    def _get_completion_trinket(self) -> str:
        return 'SubagentResultsTrinket'

    def _build_completion_context(
        self,
        status: str,
        summary: str,
        work_item: 'WorkItem',
    ) -> dict[str, Any]:
        status_map = {'success': 'success', 'timeout': 'timeout', 'failed': 'failed'}
        trinket_status = status_map.get(status, 'failed')

        context: dict = {
            'task_id': work_item.item_id,
            'status': trinket_status,
            'agent_type': 'forage',
            'subject': work_item.context.get('subject', ''),
            'iterations': len(self._trace['iterations']),
        }

        if trinket_status == 'success':
            context['result'] = summary
        else:
            context['error'] = summary
            context['error_type'] = 'AgentFailure'

        return context

    def on_overwatch_update(
        self,
        event_bus,
        work_item: 'WorkItem',
        iteration: int,
        summary: str,
    ) -> None:
        """Publish per-iteration progress to the unified results trinket."""
        from cns.core.events import UpdateTrinketEvent

        try:
            subject = work_item.context.get('subject', '')
            event_bus.publish(UpdateTrinketEvent.create(
                continuum_id='sidebar',
                target_trinket='SubagentResultsTrinket',
                context={
                    'task_id': work_item.item_id,
                    'status': 'in_progress',
                    'agent_type': 'forage',
                    'subject': subject,
                    'iteration': iteration,
                    'max_iterations': self.max_iterations,
                    'summary': summary,
                },
            ))
        except Exception as e:
            logger.debug(f"ForageAgent: overwatch publish failed: {e}")

    # The dispatch protocol (agents/specs.py): subagent_tool reaches these
    # classmethods through the spec table; they own forage's dispatch and
    # refine mechanics — WorkItem shape, blocking-cap application, refine
    # gating, auto-dismiss, and this type's refusal prose.

    @classmethod
    def activate(
        cls,
        tool_repo: 'ToolRepository',
        event_bus: 'EventBus',
        *,
        subject: str,
        context: str = '',
        blocking: bool = False,
    ) -> dict[str, Any]:
        """Dispatch a fresh forage: async through the spawn seam by default,
        blocking via run_blocking with the tight tool-config iteration cap."""
        import uuid

        work_item = cls._work_item(str(uuid.uuid4()), subject, context, None)
        if blocking:
            return cls._run_blocking(work_item, tool_repo, event_bus)
        return cls._spawn(tool_repo, event_bus, work_item, subject)

    @classmethod
    def continue_task(
        cls,
        tool_repo: 'ToolRepository',
        event_bus: 'EventBus',
        *,
        task_id: str,
        subject: str,
        context: str = '',
        blocking: bool = False,
    ) -> dict[str, Any] | None:
        """Refine a completed forage result, or refuse with the reason.

        Returns None when task_id is not an active forage entry (the claim
        protocol). A success entry is auto-dismissed and its result is
        passed to the new agent as starting context; `subject` is the
        refinement instruction.
        """
        import uuid
        from agents.spawn import publish_result

        working_memory = getattr(tool_repo, 'working_memory', None)
        trinket = (
            working_memory.get_trinket('SubagentResultsTrinket')
            if working_memory else None
        )
        entry = trinket.active_results.get(task_id) if trinket else None
        if entry is None:
            return None

        entry_type = entry.get('type', 'unknown')
        if entry_type in ('pending', 'in_progress'):
            return {
                'success': False,
                'task_id': task_id,
                'message': (
                    f"Refine not dispatched: forage task {task_id[:8]} is "
                    "still running. Wait for it to complete, or dismiss it "
                    "and dispatch a fresh forage."
                ),
            }
        if entry_type != 'success':
            return {
                'success': False,
                'task_id': task_id,
                'message': (
                    f"Refine not dispatched: forage task {task_id[:8]} "
                    f"ended without a usable result ({entry_type}). Dismiss "
                    "it and dispatch a fresh forage."
                ),
            }

        previous_result = entry.get('data', {}).get('result', '')
        if not previous_result:
            return {
                'success': False,
                'task_id': task_id,
                'message': (
                    f"Refine not dispatched: forage task {task_id[:8]} is "
                    "a success entry with no prior content to build on. "
                    "Dispatch a fresh forage instead."
                ),
            }

        # Resolved — auto-dismiss the prior entry, then respawn seeded with it
        publish_result(event_bus, 'forage', task_id, 'dismissed', {})
        work_item = cls._work_item(
            str(uuid.uuid4()), subject, context, previous_result
        )
        if blocking:
            return cls._run_blocking(work_item, tool_repo, event_bus)
        return cls._spawn(tool_repo, event_bus, work_item, subject)

    @classmethod
    def _work_item(
        cls,
        task_id: str,
        subject: str,
        context: str,
        previous_result: str | None,
    ) -> 'WorkItem':
        from agents.sidebar import WorkItem

        item_context: dict[str, Any] = {'subject': subject, 'context': context}
        if previous_result:
            item_context['previous_result'] = previous_result
        return WorkItem(
            item_id=task_id, interface_name='forage', context=item_context
        )

    @classmethod
    def _spawn(
        cls,
        tool_repo: 'ToolRepository',
        event_bus: 'EventBus',
        work_item: 'WorkItem',
        subject: str,
    ) -> dict[str, Any]:
        from agents.spawn import capacity_result, spawn_subagent
        from agents.specs import SUBAGENT_SPECS

        if not spawn_subagent(
            SUBAGENT_SPECS['forage'], work_item, event_bus,
            tool_repo=tool_repo,
        ):
            return capacity_result()
        return {
            'success': True,
            'task_id': work_item.item_id,
            'message': (
                f"Forage agent dispatched for: '{subject}'. Results will "
                "appear in your context window when ready."
            ),
        }

    @classmethod
    def _run_blocking(
        cls,
        work_item: 'WorkItem',
        tool_repo: 'ToolRepository',
        event_bus: 'EventBus',
    ) -> dict[str, Any]:
        from config.config_manager import config

        agent = cls(tool_repo)
        # The class's 20-iteration cap is calibrated for the 600s
        # background budget; the tighter config cap lets the run wind
        # down and brief inside the blocking wall-clock bound.
        agent.max_iterations = config.subagent_tool.forage_blocking_max_iterations
        record = agent.run_blocking(work_item, event_bus)
        if record['status'] == 'success':
            return {
                'success': True,
                'task_id': work_item.item_id,
                'status': 'success',
                'result': record['summary'],
            }
        return {
            'success': False,
            'task_id': work_item.item_id,
            'status': record['status'],
            'error': record['summary'],
        }
