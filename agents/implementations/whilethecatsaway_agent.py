"""
WhileTheCatsAwayAgent -- Background curiosity-driven research agent.

Mira dispatches this when she wants to learn more about a topic outside
the pressure of an active conversation. Runs with a high iteration cap --
no rush, just open-ended exploration. Findings are stored as LT_Memory
entries via memory_tool so they surface naturally in future conversations.
"""
import logging
from typing import Any, TYPE_CHECKING

from agents.base import SidebarAgent, load_agent_prompt

if TYPE_CHECKING:
    from agents.sidebar import WorkItem
    from cns.integration.event_bus import EventBus
    from tools.repo import ToolRepository

logger = logging.getLogger(__name__)


class WhileTheCatsAwayAgent(SidebarAgent):
    agent_id = "whilethecatsaway"
    model_config_name = "batch"
    available_tools = ["web_tool", "memory_tool", "continuum_tool"]
    inherit_base_prompt = False
    max_iterations = 25

    def __init__(self, tool_repo: 'ToolRepository'):
        super().__init__(tool_repo)

    def get_agent_prompt(self, work_item: 'WorkItem') -> str:
        return load_agent_prompt("whilethecatsaway_system.txt")

    def build_initial_message(self, work_item: 'WorkItem') -> str:
        subject = work_item.context.get('subject', '')
        context = work_item.context.get('context', '')
        return (
            f"**Topic:** {subject}\n\n"
            f"**Context:** {context}\n\n"
            "Go learn about this. Take your time."
        )

    def _get_completion_trinket(self) -> str:
        return 'SubagentResultsTrinket'

    def _build_completion_context(
        self,
        status: str,
        summary: str,
        work_item: 'WorkItem',
    ) -> dict[str, Any]:
        topic = work_item.context.get('subject', '')
        status_map = {'success': 'success', 'timeout': 'timeout', 'failed': 'failed'}
        trinket_status = status_map.get(status, 'failed')

        context: dict = {
            'task_id': work_item.item_id,
            'status': trinket_status,
            'agent_type': 'whilethecatsaway',
            'subject': topic,
        }

        if trinket_status == 'success':
            context['result'] = summary
        else:
            context['error'] = summary

        return context

    # The dispatch protocol (agents/specs.py) — activate only; this type is
    # not continuable (the spec's continuable=False makes subagent_tool
    # refuse continue generically, before any classmethod is reached).

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
        """Dispatch open-ended background research through the spawn seam.

        `blocking` is accepted for protocol uniformity and ignored: this
        type is fire-and-forget only.
        """
        import uuid

        from agents.sidebar import WorkItem
        from agents.spawn import capacity_result, spawn_subagent
        from agents.specs import SUBAGENT_SPECS

        work_item = WorkItem(
            item_id=str(uuid.uuid4()),
            interface_name="whilethecatsaway",
            context={"subject": subject, "context": context},
        )
        if not spawn_subagent(
            SUBAGENT_SPECS['whilethecatsaway'], work_item, event_bus,
            tool_repo=tool_repo,
        ):
            return capacity_result()
        return {
            "success": True,
            "task_id": work_item.item_id,
            "message": (
                f"Background research dispatched for: '{subject}'. The "
                "agent will explore at its own pace and store findings "
                "as memories."
            ),
        }
