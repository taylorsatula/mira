"""
ForageAgent -- Background research agent on the SidebarAgent base class.

Publishes to ForageTrinket via _get_completion_trinket() and
_build_completion_context() overrides.
"""
import logging
from typing import Any, TYPE_CHECKING

from agents.base import SidebarAgent, load_agent_prompt

if TYPE_CHECKING:
    from agents.sidebar import WorkItem
    from tools.repo import ToolRepository

logger = logging.getLogger(__name__)


class ForageAgent(SidebarAgent):
    agent_id = "forage"
    model_config_name = "batch"
    available_tools = ["continuum_tool", "memory_tool", "web_tool"]
    inherit_base_prompt = False
    max_iterations = 20
    # Runnable inline inside a bounded turn via forage_tool's blocking param —
    # bounded by sidebar_dispatcher.blocking_agent_timeout_seconds, not the
    # 600s background override, so a blocking call fits a heartbeat turn.
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
        query = work_item.context.get('query', '')
        context = work_item.context.get('context', '')
        previous_result = work_item.context.get('previous_result')

        if previous_result:
            return (
                f"**Previous forage result (you are refining this):**\n"
                f"{previous_result}\n\n"
                f"**Refinement:** {query}\n\n"
                f"**Context:** {context}\n\n"
                "The previous search found the above. Refine, correct, or "
                "dig deeper based on the refinement instruction. Build on "
                "what was found -- don't repeat the same searches."
            )
        return (
            f"**Query:** {query}\n\n"
            f"**Context:** {context}\n\n"
            "Search for information relevant to this query. "
            "Use the context to understand what would be most useful."
        )

    def _get_completion_trinket(self) -> str:
        return 'ForageTrinket'

    def _build_completion_context(
        self,
        status: str,
        summary: str,
        work_item: 'WorkItem',
    ) -> dict[str, Any]:
        query = work_item.context.get('query', '')
        status_map = {'success': 'success', 'timeout': 'timeout', 'failed': 'failed'}
        trinket_status = status_map.get(status, 'failed')

        context: dict = {
            'task_id': work_item.item_id,
            'status': trinket_status,
            'query': query,
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
        """Publish per-iteration progress to ForageTrinket."""
        from cns.core.events import UpdateTrinketEvent

        try:
            query = work_item.context.get('query', '')
            event_bus.publish(UpdateTrinketEvent.create(
                continuum_id='sidebar',
                target_trinket='ForageTrinket',
                context={
                    'task_id': work_item.item_id,
                    'status': 'in_progress',
                    'query': query,
                    'iteration': iteration,
                    'max_iterations': self.max_iterations,
                    'summary': summary,
                },
            ))
        except Exception as e:
            logger.debug(f"ForageAgent: overwatch publish failed: {e}")
