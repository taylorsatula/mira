"""
selfedit_tool -- apply or roll back MIRA's own code changes.

MIRA edits its code tree with bash_tool. This tool owns the operations that
touch the rollback mechanism (contract in utils/self_edit.py):

- request_restart: schedule a restart that applies the uncommitted edits. The
  restart fires only after the current turn is committed
  (cns/services/self_edit_handler.py); the launcher then runs the edited tree
  as a trial boot, which either commits it or stashes it and boots the last
  good code. The outcome reaches MIRA through the self-edit HUD section and the
  next heartbeat wake.
- status: the pending outcome, the uncommitted edits, and the stashed edits.
- restore_stash / discard_stash: stashed edits are MIRA's failed or set-aside
  changes; bash_tool refuses raw `git stash drop/pop/clear` in the tree, so
  these are the only paths that consume them.

Not in ESSENTIAL_TOOLS: it loads through invokeother_tool when a self-edit
skill asks for it. Every operation refuses with the reason when self-edit
rollback is inactive (dev checkout, Docker, unsupervised start).
"""
import inspect
import logging
import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from tools.registry import registry
from tools.repo import Tool
from utils import self_edit
from utils.timezone_utils import format_utc_iso, utc_now

logger = logging.getLogger(__name__)

# Uncommitted-change and stash listings are bounded so a large edit cannot
# flood the tool result; the counts are always reported in full.
_LIST_LIMIT = 50

_STASH_ID = re.compile(r"^[0-9a-f]{7,40}$")


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
    """Apply, inspect, and roll back changes to MIRA's own code tree."""

    name = "selfedit_tool"
    simple_description = "Apply or roll back changes to MIRA's own code (restart, status, stashed changes)."
    # restart and stash operations mutate the tree and the state directory
    parallel_safe = False

    tool_schema = {
        "name": "selfedit_tool",
        "description": "Apply or roll back changes to MIRA's own code tree.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["request_restart", "status", "restore_stash", "discard_stash"],
                    "description": (
                        "request_restart: restart MIRA after this turn is saved so the "
                        "uncommitted edits to the code tree take effect. The edited code "
                        "boots as a trial: if it starts, it is committed; if it does not, "
                        "the edits are stashed and MIRA starts on the previous code. Either "
                        "way the outcome appears in your HUD after the restart. Tell the "
                        "user you are restarting before calling it; nothing after this "
                        "turn runs until MIRA is back. Refused when there are no "
                        "uncommitted edits. "
                        "status: the outcome of the last restart (if not yet reported), "
                        "the uncommitted edits, and the stashed edits with their ids. "
                        "restore_stash: put a stashed edit back into the tree so it can be "
                        "fixed and retried; refused while the tree has uncommitted edits. "
                        "discard_stash: permanently delete a stashed edit."
                    ),
                },
                "stash_id": {
                    "type": "string",
                    "description": (
                        "Stash id (hex commit hash, at least 7 characters) exactly as "
                        "'status' lists it. Required for restore_stash and discard_stash; "
                        "ignored otherwise."
                    ),
                },
            },
            "required": ["operation"],
            "additionalProperties": False,
        },
    }

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        handlers = {
            "request_restart": self._request_restart,
            "status": self._status,
            "restore_stash": self._restore_stash,
            "discard_stash": self._discard_stash,
        }
        method = handlers.get(operation)
        if method is None:
            raise ValueError(f"Unknown operation: {operation}. Valid: {', '.join(handlers)}")
        reason = self_edit.inactive_reason()
        if reason is not None:
            raise ValueError(f"Self-edit is unavailable on this install: {reason}")
        try:
            accepted = set(inspect.signature(method).parameters)
            return method(**{k: v for k, v in kwargs.items() if k in accepted})
        except Exception as e:
            self.logger.error(f"selfedit_tool {operation} failed: {e}")
            raise

    # -- operations ---------------------------------------------------------

    def _request_restart(self) -> Dict[str, Any]:
        changes = self_edit.uncommitted_changes()
        if not changes:
            raise ValueError(
                "There are no uncommitted edits in the code tree, so a restart would "
                "apply nothing. Make the edit with bash_tool first."
            )
        self_edit.record_restart_request(self.user_id)
        return {
            "success": True,
            "message": (
                "Restart scheduled for when this turn is saved. Finish your reply now; "
                "the outcome (applied, or failed and stashed) appears in your HUD once "
                "MIRA is back."
            ),
            "uncommitted_edits": len(changes),
        }

    def _status(self) -> Dict[str, Any]:
        result = self_edit.result_for_user(self.user_id)
        changes = self_edit.uncommitted_changes()
        stashes = self._stashes()
        return {
            "success": True,
            "last_restart": result,
            "uncommitted_edits": changes[:_LIST_LIMIT],
            "uncommitted_edit_count": len(changes),
            "stashes": stashes[:_LIST_LIMIT],
            "stash_count": len(stashes),
        }

    def _restore_stash(self, stash_id: str = "") -> Dict[str, Any]:
        entry = self._find_stash(stash_id)
        if self_edit.uncommitted_changes():
            raise ValueError(
                "The code tree has uncommitted edits; restoring a stash on top of them "
                "would mix two changes. Apply them (request_restart) or set them aside "
                "with `git stash push -u -m \"<why>\"` in the code tree first."
            )
        try:
            self_edit.git("stash", "apply", entry["ref"])
        except RuntimeError as exc:
            # A half-applied stash (conflicts with code committed since) is set
            # aside as its own stash so the tree is clean again; the original
            # stash is untouched either way.
            set_aside = "none"
            if self_edit.uncommitted_changes():
                # A conflicted apply leaves unmerged index entries, which stash
                # push refuses ("needs merge"); staging records the files as
                # they are, conflict markers included, so they can be set aside.
                self_edit.git("add", "-A")
                self_edit.git(
                    "stash", "push", "--include-untracked", "--quiet", "-m",
                    f"self-edit set aside {format_utc_iso(utc_now())} "
                    f"(conflicted restore of {entry['id'][:12]})",
                )
                set_aside = self_edit.git("rev-parse", "stash@{0}").strip()
            # Only git's CONFLICT/error lines: its "use git restore ..." advice
            # names commands bash_tool refuses.
            conflicts = "; ".join(
                line.strip() for line in str(exc).splitlines()
                if line.strip().startswith(("CONFLICT", "error"))
            ) or str(exc).splitlines()[0]
            raise ValueError(
                f"Stash {entry['id'][:12]} does not apply cleanly to the current code "
                f"({conflicts}). The tree is clean again; the original stash is kept, "
                f"and the partial apply was set aside as stash {set_aside[:12]}."
            ) from exc
        # Drop by the id just applied, re-resolved: stash@{n} positions shift.
        self_edit.git("stash", "drop", self._find_stash(entry["id"])["ref"])
        return {
            "success": True,
            "message": (
                f"Stash {entry['id'][:12]} is back in the code tree as uncommitted edits. "
                "Fix it with bash_tool, then request_restart to try it again."
            ),
            "uncommitted_edits": self_edit.uncommitted_changes()[:_LIST_LIMIT],
        }

    def _discard_stash(self, stash_id: str = "") -> Dict[str, Any]:
        entry = self._find_stash(stash_id)
        self_edit.git("stash", "drop", entry["ref"])
        logger.warning("Self-edit stash %s discarded: %s", entry["id"], entry["description"])
        return {
            "success": True,
            "message": f"Stash {entry['id'][:12]} ({entry['description']}) was permanently deleted.",
        }

    # -- helpers ------------------------------------------------------------

    def _stashes(self) -> List[Dict[str, str]]:
        """Every stash in the code tree, newest first, keyed by its commit hash."""
        output = self_edit.git("stash", "list", "--format=%H%x09%gd%x09%s")
        stashes = []
        for line in output.splitlines():
            commit, ref, description = line.split("\t", 2)
            stashes.append({"id": commit, "ref": ref, "description": description})
        return stashes

    def _find_stash(self, stash_id: str) -> Dict[str, str]:
        stash_id = (stash_id or "").strip().lower()
        if not _STASH_ID.match(stash_id):
            raise ValueError(
                "stash_id is required: the hex id (at least 7 characters) exactly as "
                "'status' lists it."
            )
        matches = [entry for entry in self._stashes() if entry["id"].startswith(stash_id)]
        if not matches:
            raise ValueError(f"No stash with id {stash_id}; call 'status' for the current list.")
        if len(matches) > 1:
            raise ValueError(f"Stash id {stash_id} is ambiguous; use more characters.")
        return matches[0]
