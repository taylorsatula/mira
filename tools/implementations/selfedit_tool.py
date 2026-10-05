"""
selfedit_tool -- apply MIRA's own code edits by restarting.

MIRA edits its code tree with bash_tool; this tool owns the one step bash must
not take: the restart (contract in utils/self_edit.py). request_restart records
the requester and restarts once the current turn is over; the launcher boots
the edited tree as a trial, which is either committed or stashed with the
previous code started instead. The outcome arrives as an activity-feed item
(HUD and heartbeat). status reports the uncommitted edits and stashed ones.

Not in ESSENTIAL_TOOLS: it loads through invokeother_tool when the
self-modification skill asks for it, and is listed there only where rollback
is active.
"""
import logging
from typing import Any, Dict

from pydantic import BaseModel, Field

from tools.registry import registry
from tools.repo import Tool
from utils import self_edit

logger = logging.getLogger(__name__)

# Listings are bounded so a large edit cannot flood the tool result; the
# counts are always reported in full.
_LIST_LIMIT = 50


class SelfeditToolConfig(BaseModel):
    """Configuration for selfedit_tool."""

    # Defaults to whether rollback is active in this process: an install
    # without it never lists the tool in invokeother_tool's catalog, since
    # every operation would refuse there anyway.
    enabled: bool = Field(
        default_factory=self_edit.is_active,
        description="Whether the self-edit tool can be loaded",
    )


registry.register("selfedit_tool", SelfeditToolConfig)


class SelfeditTool(Tool):
    """Apply changes to MIRA's own code tree by restarting, and report their state."""

    name = "selfedit_tool"
    simple_description = "Apply changes to MIRA's own code by restarting; list pending and stashed changes."
    parallel_safe = False

    tool_schema = {
        "name": "selfedit_tool",
        "description": "Apply changes to MIRA's own code tree by restarting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["request_restart", "status"],
                    "description": (
                        "request_restart: restart MIRA once this turn is over so the "
                        "uncommitted edits to the code tree take effect. The edited code "
                        "boots as a trial: if it starts, it is committed; if it does not, "
                        "the edits are stashed and MIRA starts on the previous code. Either "
                        "way the outcome arrives as an activity item in your HUD after the "
                        "restart. Tell the user you are restarting before calling it; "
                        "nothing after this turn runs until MIRA is back. Refused when "
                        "there are no uncommitted edits. "
                        "status: the uncommitted edits in the code tree and the stashed "
                        "edits (stash ref, id, description)."
                    ),
                },
            },
            "required": ["operation"],
            "additionalProperties": False,
        },
    }

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        reason = self_edit.inactive_reason()
        if reason is not None:
            raise ValueError(f"Self-edit is unavailable on this install: {reason}")
        if operation == "request_restart":
            return self._request_restart()
        if operation == "status":
            return self._status()
        raise ValueError(f"Unknown operation: {operation}. Valid: request_restart, status")

    def _request_restart(self) -> Dict[str, Any]:
        changes = self_edit.uncommitted_changes()
        if not changes:
            raise ValueError(
                "There are no uncommitted edits in the code tree, so a restart would "
                "apply nothing. Make the edit with bash_tool first."
            )
        self_edit.schedule_restart_after_turn(self.user_id)
        return {
            "success": True,
            "message": (
                "Restart scheduled for when this turn is over. Finish your reply now; "
                "the outcome (applied, or failed and stashed) arrives as an activity "
                "item once MIRA is back."
            ),
            "uncommitted_edits": len(changes),
        }

    def _status(self) -> Dict[str, Any]:
        changes = self_edit.uncommitted_changes()
        stashes = [
            dict(zip(("ref", "id", "description"), line.split("\t", 2)))
            for line in self_edit.git("stash", "list", "--format=%gd%x09%H%x09%s").splitlines()
        ]
        return {
            "success": True,
            "uncommitted_edits": changes[:_LIST_LIMIT],
            "uncommitted_edit_count": len(changes),
            "stashes": stashes[:_LIST_LIMIT],
            "stash_count": len(stashes),
        }
