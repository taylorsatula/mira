"""
Renders the standing session record and rare steering for the HUD.

The Peanut Gallery is a standing record, not an event. Every fifth segment
turn the observer extends the record with one to three beats; the record is
append-only, never expires, and re-renders each turn through the compose
path. Steering is the exception: at most one directive per pass, rendered
only until the next pass (which either replaces it or clears it).

Both are session-scoped, cleared on segment collapse by the central flush
(WorkingMemory._flush_stateful_trinkets). No Valkey-side copy beyond the
base-class content persistence — the standing block is an in-memory,
session-scoped slot like every other trinket.

Rendering: every line and directive is escaped at the interpolation point
(html.escape), never spliced raw.
"""
import html
import logging
from typing import Any, Dict, List, Optional

from utils.user_context import get_current_user_id
from working_memory.trinkets.base import StatefulTrinket

logger = logging.getLogger(__name__)


class PeanutGalleryTrinket(StatefulTrinket):
    """
    Displays the standing session record and rare steering in the HUD.

    The record is append-only and never expires; the steering section is
    replaced or cleared on every observation pass. Both are cleared entirely
    on segment collapse.
    """

    variable_name = "peanutgallery_guidance"

    def __init__(self, event_bus, working_memory):
        super().__init__(event_bus, working_memory)

        self._user_records: Dict[str, List[str]] = {}
        self._user_steering: Dict[str, Optional[Dict[str, Any]]] = {}

        logger.info("PeanutGalleryTrinket initialized (standing record)")

    # -- state ----------------------------------------------------------------

    def _expire_items(self) -> bool:
        """
        Never expire. The record grows only on a pass; steering is cleared only
        by the next pass that emits none. Nothing to re-render here.
        """
        return False

    def _clear_all_state(self) -> None:
        """Clear record and steering for the current user on segment collapse."""
        user_id = get_current_user_id()
        self._user_records.pop(user_id, None)
        self._user_steering.pop(user_id, None)

    def get_session_log(self) -> List[str]:
        """Return the current standing record for the active user."""
        return list(self._user_records.get(get_current_user_id(), ()))

    def _append_record_line(self, line: str) -> bool:
        """
        Append one beat to the standing record. Skips blanks and exact duplicates.

        Returns True when the line was added.
        """
        line = line.strip()
        if not line:
            return False
        user_id = get_current_user_id()
        record = self._user_records.setdefault(user_id, [])
        if line in record:
            return False
        record.append(line)
        return True

    def _set_steering(self, text: Optional[str], critical: bool = False) -> None:
        """
        Set the active steering directive, or clear it when absent.

        Every observation pass re-decides this — the record states, the steering
        directs, and the next pass is the only authority over the steering's
        lifetime.
        """
        user_id = get_current_user_id()
        text = (text or "").strip()
        if not text:
            self._user_steering.pop(user_id, None)
            return
        self._user_steering[user_id] = {"text": text, "critical": bool(critical)}

    # -- update handling ------------------------------------------------------

    def handle_update_request(self, event) -> None:
        """Process incoming observation from the Peanut Gallery observer."""
        context = event.context or {}
        if context.get('action') == 'observe':
            added = 0
            for line in context.get('log_lines') or []:
                if self._append_record_line(line):
                    added += 1

            steering = context.get('steering')
            if isinstance(steering, dict):
                self._set_steering(steering.get('text'), steering.get('critical', False))
            else:
                self._set_steering(None)

            if added or steering:
                logger.info(
                    "Record: %d beat(s) appended (%d total), steering %s",
                    added, len(self.get_session_log()),
                    "set" if steering else "cleared",
                )

        super().handle_update_request(event)

    # -- rendering ------------------------------------------------------------

    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Render the standing record and any active steering directive.

        The record is the standing block: one line per beat, newest last. The
        steering directive — when present — is the exception and renders as its
        own tag. Both are escaped at the interpolation point.
        """
        user_id = get_current_user_id()
        record = self._user_records.get(user_id, [])
        steering = self._user_steering.get(user_id)

        if not record and not steering:
            return ""

        parts: List[str] = ['<mira:peanutgallery>']

        if record:
            parts.append('  <header>Session record (append-only — describes session shape, not a directive; may lag the conversation):</header>')
            parts.append('  <session_log>')
            for line in record:
                parts.append(f'    - {html.escape(line)}')
            parts.append('  </session_log>')

        if steering:
            text = html.escape(steering['text'])
            if steering.get('critical'):
                parts.append(f'  <steering critical="true">{text}</steering>')
            else:
                parts.append(f'  <steering>{text}</steering>')

        parts.append('</mira:peanutgallery>')
        return "\n".join(parts)
