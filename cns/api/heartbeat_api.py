"""
Heartbeat API -- inspect and manage briefs (wake-ups with missions).

Taylor-facing surface over the per-user briefs store and the heartbeat
control state: GET endpoints enumerate briefs and reminders for the future
web frontend; POST cancel kills a brief by id. The old cancel/resume/status
ops on the wake-cycle service are gone with it.
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
    note: Optional[str] = None


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


class ReminderViewHandler(PropagatingHandler):
    """Read-only reminder view: the sibling type (delivers to the human) on
    the same per-user store, for one-frontend inspection of both."""

    def process_request(self, *, user_id: str) -> dict[str, Any]:
        set_current_user_id(user_id)
        from tools.implementations.reminder_tool import ReminderTool

        tool = ReminderTool()
        result = tool.run("get_reminders", date_filter="all")
        return {"reminders": result.get("reminders", []), "count": result.get("count", 0)}


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
    body: Optional[BriefCancelBody] = None,
    current_user: SessionData | APITokenContext = Depends(get_current_user),
):
    set_current_user_id(current_user.user_id)
    from tools.implementations.heartbeat_tool import HeartbeatTool

    tool = HeartbeatTool()
    result = tool.run(
        "cancel", brief_id=brief_id, note=body.note if body else None
    )
    return {"success": True, "data": result}


@router.get("/reminders")
def list_reminders(
    current_user: SessionData | APITokenContext = Depends(get_current_user),
):
    handler = ReminderViewHandler()
    response = handler.handle_request(user_id=current_user.user_id)
    return response.to_dict()
