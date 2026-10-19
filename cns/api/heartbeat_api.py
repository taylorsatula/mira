"""
Brief inspection and cancellation: wake-ups with missions, one user's own.

Reads enumerate and fetch rows out of `heartbeat_tool`'s per-user store. Cancel
rides `HeartbeatTool().run("cancel", ...)` — the tool's real entry point —
rather than opening the table directly, so the tool's state rules stay the only
ones in force. The read path imports the tool's private `ensure_briefs_table` and
`_brief_row_public`, so a rename in the tool is a rename here in the same commit.

Authenticated per user like every router in this directory: a caller only ever
sees its own briefs.
"""
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth.api import get_current_user
from auth.types import SessionData, APITokenContext
from utils.user_context import set_current_user_id

from .base import PropagatingHandler, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter()

_VALID_STATES = ("armed", "done", "bailed", "failed_permanently", "cancelled")


class BriefCancelBody(BaseModel):
    note: str


class HeartbeatBriefsHandler(PropagatingHandler):
    """GET/POST over heartbeat_tool's store for the authenticated user."""

    def process_request(
        self,
        *,
        user_id: str,
        brief_id: Optional[str] = None,
        state_filter: Optional[str] = None,
        cancel_note: Optional[str] = None,
    ) -> dict[str, Any]:
        from tools.implementations.heartbeat_tool import (
            ensure_briefs_table,
            _brief_row_public,
        )
        from utils.userdata_manager import get_user_data_manager

        set_current_user_id(user_id)
        db = get_user_data_manager(user_id)
        ensure_briefs_table(db)

        if brief_id is not None:
            row = db.fetchone("SELECT * FROM briefs WHERE id = :id", {"id": brief_id})
            if row is None:
                raise ValidationError(f"No brief with id {brief_id}")
            return {"brief": _brief_row_public(row)}

        if state_filter is not None:
            if state_filter not in _VALID_STATES:
                raise ValidationError(
                    f"state must be one of {', '.join(_VALID_STATES)}, got: {state_filter}"
                )
            rows = db.fetchall(
                "SELECT * FROM briefs WHERE state = :state ORDER BY next_wake_at ASC",
                {"state": state_filter},
            )
        else:
            rows = db.fetchall("SELECT * FROM briefs ORDER BY next_wake_at ASC")
        return {"briefs": [_brief_row_public(r) for r in rows], "count": len(rows)}


@router.get("/heartbeat/briefs")
def list_briefs(
    state: Optional[str] = None,
    current_user: SessionData | APITokenContext = Depends(get_current_user),
):
    handler = HeartbeatBriefsHandler()
    response = handler.handle_request(
        user_id=current_user.user_id, brief_id=None, state_filter=state
    )
    return response.to_dict()


@router.get("/heartbeat/briefs/{brief_id}")
def get_brief(
    brief_id: str,
    current_user: SessionData | APITokenContext = Depends(get_current_user),
):
    handler = HeartbeatBriefsHandler()
    response = handler.handle_request(
        user_id=current_user.user_id, brief_id=brief_id
    )
    return response.to_dict()


@router.post("/heartbeat/briefs/{brief_id}/cancel")
def cancel_brief(
    brief_id: str,
    body: BriefCancelBody,
    current_user: SessionData | APITokenContext = Depends(get_current_user),
):
    if not body.note or not body.note.strip():
        raise ValidationError("note is required on cancel")
    set_current_user_id(current_user.user_id)
    from tools.implementations.heartbeat_tool import HeartbeatTool

    tool = HeartbeatTool()
    result = tool.run("cancel", brief_id=brief_id, note=body.note.strip())
    return {"success": True, "data": result}

