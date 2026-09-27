"""
AsyncActivityTrinket -- Renders sidebar agent activity into the system prompt.

SQLite-backed. Reads from sidebar_activity on every generate_content() call.
UpdateTrinketEvent is a re-render signal; state lives in SQLite, not memory.

EventAwareTrinket (not StatefulTrinket): no turn-scoped expiry, no in-memory
state to clear on collapse. Items persist until explicitly dismissed.
"""
import html
import logging
from typing import Dict, Any

from utils.timezone_utils import utc_now
from working_memory.trinkets.base import EventAwareTrinket

logger = logging.getLogger(__name__)


class AsyncActivityTrinket(EventAwareTrinket):
    """Renders sidebar agent activity feed into the system prompt."""

    variable_name = "async_activity"

    def generate_content(self, context: Dict[str, Any]) -> str:
        """Read active sidebar items from SQLite and render as XML.

        Summary text originates from untrusted input; escape < > before
        rendering into the system prompt XML.
        """
        from utils.userdata_manager import get_user_data_manager
        from utils.user_context import get_current_user_id

        db = get_user_data_manager(get_current_user_id())

        from agents.base import ensure_activity_schema
        ensure_activity_schema(db)

        rows = db.select(
            "sidebar_activity",
            where="status NOT IN ('dismissed', 'resolved')",
            order_by="updated_at DESC",
        )

        if not rows:
            return ""

        lines = self._render_items(rows)
        return "<async_activity>\n" + "\n".join(lines) + "\n</async_activity>"

    def _render_items(self, rows: list[dict]) -> list[str]:
        """Group items by interface and render one line per thread."""
        # Group by interface_name
        interfaces: dict[str, list[dict]] = {}
        for row in rows:
            iface = row['interface_name']
            interfaces.setdefault(iface, []).append(row)

        lines: list[str] = []
        for iface, items in interfaces.items():
            lines.append(f"[{html.escape(iface)}]")
            for item in items:
                summary = html.escape(item['summary'])
                status = item['status']
                updated = item['updated_at']

                age = self._format_age(updated)
                if status == 'escalated':
                    prefix = "ESCALATION: "
                elif status == 'conflict':
                    prefix = "RULE CONFLICT: "
                else:
                    prefix = ""
                escalation = ""
                if status == 'escalated' and item.get('escalation_reason'):
                    escalation = f" Reason: {html.escape(item['escalation_reason'])}"

                lines.append(
                    f"  {prefix}{summary}{escalation} [{age}]"
                )

        return lines

    def _format_age(self, updated_at: str) -> str:
        """Format the time since last update as human-readable age."""
        try:
            from datetime import datetime
            updated = datetime.fromisoformat(updated_at)
            now = utc_now()
            # SQLite datetime('now') is naive UTC
            if updated.tzinfo is None:
                from datetime import timezone
                updated = updated.replace(tzinfo=timezone.utc)
            delta = now - updated
            seconds = delta.total_seconds()

            if seconds < 60:
                return "just now"
            elif seconds < 3600:
                minutes = int(seconds / 60)
                return f"{minutes}m ago"
            elif seconds < 86400:
                hours = int(seconds / 3600)
                return f"{hours}h ago"
            else:
                days = int(seconds / 86400)
                return f"{days}d ago"
        except Exception:
            return "recently"
