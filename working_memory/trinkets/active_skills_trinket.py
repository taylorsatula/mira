"""
Active skills trinket — renders the bodies of skills the model invoked.

Activation state is in-memory and per-user (user_id -> skill name -> body
snapshot taken at activation). The body is snapshotted when invoke_skill
publishes the activation, so a later file edit or delete mid-segment cannot
silently change or blank the prompt section. All state is flushed on segment
collapse by WorkingMemory._flush_stateful_trinkets(); there is no TTL and no
deactivate operation — a re-invocation overwrites the earlier body.
"""

import html
import logging
from typing import Dict, Any, TYPE_CHECKING

from working_memory.trinkets.base import StatefulTrinket
from utils.user_context import get_current_user_id

if TYPE_CHECKING:
    from cns.integration.event_bus import EventBus
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)


class ActiveSkillsTrinket(StatefulTrinket):
    """Renders activated skill bodies into the non-cached system prompt."""

    variable_name = "active_skills"
    cache_policy = False

    def __init__(self, event_bus: 'EventBus', working_memory: 'WorkingMemory'):
        super().__init__(event_bus, working_memory)
        # Process-global singleton: user state lives only in this per-user dict.
        self._active: dict[str, dict[str, str]] = {}

    def handle_update_request(self, event) -> None:
        """Store the activation snapshot first, then delegate for render/publish."""
        context = event.context
        if context.get("action") == "activate":
            name = context["name"]
            body = context["body"]
            self._active.setdefault(get_current_user_id(), {})[name] = body
            logger.debug("Skill activated: %s", name)
        return super().handle_update_request(event)

    def _expire_items(self) -> bool:
        """No TTL — activations live until segment collapse."""
        return False

    def _clear_all_state(self) -> None:
        """Segment collapse flush: all activations die with the segment."""
        self._active.clear()

    def generate_content(self, context: Dict[str, Any]) -> str:
        bodies = self._active.get(get_current_user_id(), {})
        if not bodies:
            return ""

        lines = ["<active_skills>"]
        for name, body in bodies.items():
            lines.append(f'<skill name="{html.escape(name, quote=True)}">')
            lines.append(html.escape(body))
            lines.append("</skill>")
        lines.append("</active_skills>")
        return "\n".join(lines)
