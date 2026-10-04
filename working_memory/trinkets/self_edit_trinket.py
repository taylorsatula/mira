"""
SelfEditTrinket -- the outcome of MIRA's last self-edit restart, in the HUD.

Reads the outcome file on every render (utils/self_edit.py owns the format and
lifecycle): applied with the commit, or failed with the stash id and the tail
of the failed boot's log. It renders until a non-heartbeat turn for the user
completes (cns/services/self_edit_handler.py clears it), so the outcome is in
front of MIRA for the first turn after the restart — the heartbeat wake that
announces it, or the user's next message. Renders nothing when there is no
outcome for the current user or self-edit is inactive.
"""
import html
import logging
from typing import Any, Dict

from utils import self_edit
from utils.user_context import get_current_user_id
from working_memory.trinkets.base import EventAwareTrinket

logger = logging.getLogger(__name__)


class SelfEditTrinket(EventAwareTrinket):
    """Renders the pending self-edit restart outcome into the HUD."""

    variable_name = "self_edit_status"

    def generate_content(self, context: Dict[str, Any]) -> str:
        result = self_edit.result_for_user(str(get_current_user_id()))
        if result is None:
            return ""
        ref = html.escape(result["ref"])
        if result["status"] == "applied":
            body = (
                f"Your code change was applied: MIRA restarted on it and started "
                f"successfully (commit {ref[:12]}). Starting is not the same as "
                f"working — check that the change does what was asked before "
                f"telling the user it is done."
            )
        else:
            body = (
                f"Your code change FAILED: MIRA could not start on it, so the change "
                f"was stashed (stash id {ref[:12]}) and MIRA is running the previous "
                f"code. Tell the user what failed and ask whether to fix and retry "
                f"(selfedit_tool restore_stash, then edit and request_restart) or "
                f"drop it (selfedit_tool discard_stash)."
            )
        detail = result["detail"]
        lines = [f"<self_edit_status status=\"{html.escape(result['status'])}\">", body]
        if detail:
            lines.append(f"<boot_output>\n{html.escape(detail)}\n</boot_output>")
        lines.append(
            "Report this outcome to the user if you have not already. It stays here "
            "until your next reply to a message from the user."
        )
        lines.append("</self_edit_status>")
        return "\n".join(lines)
