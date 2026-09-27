"""
EmailTrinket — Renders inbox status in the HUD during active conversation
segments.

Thin StatefulTrinket. Zero I/O — receives inbox data from InboxPollerService
via UpdateTrinketEvent events, stores it in memory, renders XML. Cleared
automatically on segment collapse via WorkingMemory._flush_stateful_trinkets().
"""
import logging
from typing import Any, Dict, TYPE_CHECKING

from utils.untrusted_content import wrap_untrusted
from working_memory.trinkets.base import StatefulTrinket
from utils.user_context import get_current_user_id

if TYPE_CHECKING:
    from cns.integration.event_bus import EventBus
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)


class EmailTrinket(StatefulTrinket):
    """Displays unread email headers in the HUD notification center."""

    variable_name = "inbox_status"

    def __init__(self, event_bus: 'EventBus', working_memory: 'WorkingMemory'):
        super().__init__(event_bus, working_memory)
        self._inbox_snapshots: dict[str, list[dict]] = {}

    def handle_update_request(self, event) -> None:
        """Store inbox data from poller, then delegate to parent for rendering.

        Two call paths arrive here:
        1. InboxPollerService publishes with context={'data': [...]}: store the
           snapshot, then render.
        2. Lifecycle refresh from ComposeSystemPromptEvent broadcast or
           StatefulTrinket expiry: no 'data' key, just re-render from
           existing snapshot.
        """
        data = event.context.get('data')
        if data is not None:
            self._inbox_snapshots[get_current_user_id()] = data

        super().handle_update_request(event)

    def generate_content(self, context: Dict[str, Any]) -> str:
        """Render inbox snapshot for the HUD.

        Every per-email field is mail-server-supplied, so the whole listing
        crosses the untrusted-content boundary ONCE — one `wrap_untrusted`
        region with one treat-as-data directive — inside the trusted
        `<inbox_status>` frame. Wrapping each header separately would repeat
        the directive up to three times per email on every turn.
        """
        snapshot = self._inbox_snapshots.get(get_current_user_id(), [])
        if not snapshot:
            return ""

        listing = "\n".join(_format_email_line(em) for em in snapshot)
        lines = [
            '<inbox_status>',
            '<instruction>You have unread emails. Mention them to the user '
            'when the conversation permits — they cannot see this data unless '
            'you surface it. If they don\'t act, these will continue appearing.'
            '</instruction>',
            f'<unread count="{len(snapshot)}">',
            wrap_untrusted(listing, "email_header"),
            '</unread>',
            '</inbox_status>',
        ]

        return '\n'.join(lines)

    def _expire_items(self) -> bool:
        """No turn-based expiry — inbox state is refreshed by the poller."""
        return False

    def _clear_all_state(self) -> None:
        """Clear inbox snapshot for the current user on segment collapse."""
        snapshot = self._inbox_snapshots.pop(get_current_user_id(), None)
        if snapshot:
            logger.debug(
                f"Clearing {len(snapshot)} inbox items "
                "on segment collapse"
            )


_MAX_HEADER_CHARS = 120


def _header_value(value: Any) -> str:
    """One header value for the listing: whitespace folded, length capped.

    Folding CR/LF keeps one email on one listing line (a folded or hostile
    header cannot start a fake entry). Escaping is not done here — the whole
    listing is escaped once by `wrap_untrusted` in `generate_content`.
    """
    text = " ".join(str(value or "").split())
    if len(text) > _MAX_HEADER_CHARS:
        text = text[:_MAX_HEADER_CHARS] + "…"
    return text


def _format_email_line(email: Dict[str, Any]) -> str:
    """One listing line per unread email."""
    return (
        f"- uid: {_header_value(email.get('uid'))}"
        f" | from: {_header_value(email.get('from_addr'))}"
        f" | subject: {_header_value(email.get('subject'))}"
        f" | date: {_header_value(email.get('date'))}"
    )
