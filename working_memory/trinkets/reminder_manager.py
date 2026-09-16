"""Reminder manager trinket for system prompt injection."""
import html
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from utils.timezone_utils import (
    convert_from_utc, format_datetime, format_relative_time,
    parse_utc_time_string, utc_now
)
from utils.user_context import get_user_preferences
from .base import EventAwareTrinket

if TYPE_CHECKING:
    from cns.integration.event_bus import EventBus
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CategoryConfig:
    """Configuration capturing the differences between reminder categories."""
    tag: str
    overdue_attrs: str
    overdue_warning: str
    overdue_action: str      # Empty string = no action element
    today_guidance: str      # Empty string = no guidance element


_USER_CONFIG = _CategoryConfig(
    tag="user_reminders",
    overdue_attrs=' urgent="true"',
    overdue_warning="⚠️ OVERDUE - REQUIRE IMMEDIATE ATTENTION",
    overdue_action="YOU MUST notify the user about these overdue reminders IMMEDIATELY.",
    today_guidance="Please remind the user about these during the continuum when appropriate.",
)

_INTERNAL_CONFIG = _CategoryConfig(
    tag="internal_reminders",
    overdue_attrs="",
    overdue_warning="⚠️ OVERDUE INTERNAL REMINDERS",
    overdue_action="",
    today_guidance="",
)


class ReminderManager(EventAwareTrinket):
    """
    Manages reminder information for the notification center.

    Fetches active reminders from the reminder tool when requested.
    """

    variable_name = "active_reminders"

    def __init__(self, event_bus: 'EventBus', working_memory: 'WorkingMemory'):
        """Initialize with cached reminder tool instance."""
        from tools.implementations.reminder_tool import ReminderTool
        self._reminder_tool = ReminderTool()
        super().__init__(event_bus, working_memory)

    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate reminder content by fetching from reminder tool.

        Args:
            context: Update context (unused for reminder manager)

        Returns:
            Formatted reminders section or empty string if no reminders

        Raises:
            Exception: If ReminderTool operations fail (infrastructure/filesystem issues)
        """
        # Fetch all four reminder sets
        categories: List[Tuple[_CategoryConfig, List[Dict], List[Dict]]] = []
        for config, cat_name in [(_USER_CONFIG, "user"), (_INTERNAL_CONFIG, "internal")]:
            overdue = self._collect_reminders(self._reminder_tool.run(
                operation="get_reminders", date_filter="overdue", category=cat_name
            ))
            today = self._collect_reminders(self._reminder_tool.run(
                operation="get_reminders", date_filter="today", category=cat_name
            ))
            categories.append((config, overdue, today))

        # Skip if all empty
        if not any(overdue or today for _, overdue, today in categories):
            logger.debug("No active reminders")
            return ""  # Legitimately empty - user has no reminders set

        reminder_info = self._format_reminders(categories)

        total = sum(len(o) + len(t) for _, o, t in categories)
        total_overdue = sum(len(o) for _, o, _ in categories)
        logger.debug(f"Generated reminder info with {total} reminders ({total_overdue} overdue)")
        return reminder_info

    def _collect_reminders(self, result: dict[str, Any]) -> List[Dict]:
        """Extract non-completed reminders from a single tool result."""
        if result.get("count", 0) == 0:
            return []
        return [
            r for r in result.get("reminders", [])
            if not r.get('completed', False)
        ]

    def _format_reminders(
        self,
        categories: List[Tuple[_CategoryConfig, List[Dict], List[Dict]]]
    ) -> str:
        """Format all reminder categories as XML."""
        parts = ["<active_reminders>"]
        parts.append("<instruction>Reminders require immediate action. When one is due or overdue, notify The User even if mid-conversation. They cannot see reminders unless you voice them. Non-negotiable.</instruction>")

        for config, overdue, today in categories:
            section = self._format_category(config, overdue, today)
            if section:
                parts.append(section)

        parts.append("</active_reminders>")
        return "\n".join(parts)

    def _format_category(
        self,
        config: _CategoryConfig,
        overdue: List[Dict],
        today: List[Dict]
    ) -> str:
        """Format a single reminder category (user or internal)."""
        if not overdue and not today:
            return ""

        user_tz = get_user_preferences().timezone
        parts = [f"<{config.tag}>"]

        if overdue:
            parts.append(f"<overdue{config.overdue_attrs}>")
            parts.append(f"<warning>{config.overdue_warning}</warning>")
            for reminder in overdue:
                date_obj = parse_utc_time_string(reminder["reminder_date"])
                relative_time = format_relative_time(date_obj)
                parts.append(self._format_reminder_xml(reminder, relative_time))
            if config.overdue_action:
                parts.append(f"<action>{config.overdue_action}</action>")
            parts.append("</overdue>")

        if today:
            parts.append("<today>")
            now = utc_now()
            for reminder in today:
                date_obj = parse_utc_time_string(reminder["reminder_date"])
                local_time = convert_from_utc(date_obj, user_tz)

                if date_obj > now:
                    relative_time = format_relative_time(date_obj)
                    time_str = format_datetime(local_time, 'short')
                    parts.append(self._format_reminder_xml(reminder, relative_time, time_str))
                else:
                    formatted_time = format_datetime(local_time, 'date_time')
                    parts.append(self._format_reminder_xml(reminder, formatted_time))

            if config.today_guidance:
                parts.append(f"<guidance>{config.today_guidance}</guidance>")
            parts.append("</today>")

        parts.append(f"</{config.tag}>")
        return "\n".join(parts)

    def _format_reminder_xml(self, reminder: Dict, due: str, time: str | None = None) -> str:
        """Format a single reminder as XML element."""
        attrs = [
            f'id="{html.escape(str(reminder["id"]), quote=True)}"',
            f'title="{html.escape(reminder["encrypted__title"], quote=True)}"',
            f'due="{html.escape(due, quote=True)}"'
        ]
        if time:
            attrs.append(f'time="{html.escape(time, quote=True)}"')

        if reminder.get('encrypted__description'):
            description = html.escape(reminder['encrypted__description'])
            return f"<reminder {' '.join(attrs)}>\n<details>{description}</details>\n</reminder>"
        return f"<reminder {' '.join(attrs)}/>"
