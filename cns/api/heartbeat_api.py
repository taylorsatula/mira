"""
Heartbeat control API — external cancel/resume/status for the wake cycle.

Taylor-facing control surface for cns/services/heartbeat_service.py. Cancel
pauses future ticks for the user (Valkey latch) and cancels any in-flight
heartbeat turn through the same cancel-event machinery the websocket halt
uses. Resume clears the latch. Status reports the latch, config, and recent
decisions from heartbeat_tool's store.
"""
import logging
from typing import Any

from fastapi import APIRouter, Depends

from auth.api import get_current_user
from auth.types import SessionData, APITokenContext
from utils.user_context import set_current_user_id

from .base import BaseHandler, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter()

_VALID_OPS = ("cancel", "resume", "status")


class HeartbeatControlHandler(BaseHandler):
    """One handler, three ops; op maps onto a heartbeat_service function."""

    def process_request(self, *, user_id: str, op: str) -> dict[str, Any]:
        from cns.services.heartbeat_service import (
            cancel_heartbeat,
            heartbeat_state,
            resume_heartbeat,
        )

        set_current_user_id(user_id)
        if op == "cancel":
            result = cancel_heartbeat(user_id)
        elif op == "resume":
            result = resume_heartbeat(user_id)
        else:
            result = heartbeat_state(user_id)
        return {"heartbeat": result}


@router.post("/heartbeat/{op}")
def heartbeat_control_endpoint(
    op: str,
    current_user: SessionData | APITokenContext = Depends(get_current_user),
):
    if op not in _VALID_OPS:
        raise ValidationError(f"Unknown heartbeat op: {op}. Valid: {', '.join(_VALID_OPS)}")
    handler = HeartbeatControlHandler()
    response = handler.handle_request(user_id=current_user.user_id, op=op)
    return response.to_dict()
