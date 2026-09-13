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
        topic = work_item.context.get('topic', '')
        context = work_item.context.get('context', '')
        return (
            f"**Topic:** {topic}\n\n"
            f"**Context:** {context}\n\n"
            "Go learn about this. Take your time."
        )

    def _get_completion_trinket(self) -> str:
        return 'WhileTheCatsAwayTrinket'

    def _build_completion_context(
        self,
        status: str,
        summary: str,
        work_item: 'WorkItem',
    ) -> dict[str, Any]:
        topic = work_item.context.get('topic', '')
        status_map = {'success': 'success', 'timeout': 'timeout', 'failed': 'failed'}
        trinket_status = status_map.get(status, 'failed')

        context: dict = {
            'task_id': work_item.item_id,
            'status': trinket_status,
            'topic': topic,
        }

        if trinket_status == 'success':
            context['result'] = summary
        else:
            context['error'] = summary

        return context
