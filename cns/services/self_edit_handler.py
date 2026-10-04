"""
Self-edit event handling: restart after the requesting turn commits, clear a delivered outcome.

Two subscriptions (wired in cns/integration/factory.py only when self-edit
rollback is active — utils/self_edit.py owns the contract):

- ToolResultHistoryCommittedEvent: published from a post-commit callback, so
  its tool messages are already persisted. A committed, non-error
  selfedit_tool request_restart result — matched by the tool_name and
  tool_arguments metadata the orchestrator stamps on every tool message —
  starts the restart. The restart waits until the requesting user's turn lock
  is free: the turn's terminal frame is queued before the lock is released,
  and shutdown drains queued frames, so the user sees the end of the reply.
  The lock is then held, not released, so no new turn starts in the shutdown
  window; the startup Valkey flush clears it.
- TurnCompletedEvent: once a non-heartbeat turn completes for the user an
  outcome belongs to, that turn rendered the outcome in the HUD, so the
  outcome is delivered and cleared. Heartbeat turns never clear it: a
  keepsleeping wake shows the user nothing.
"""
import contextvars
import logging
import threading
import time

from cns.core.events import ToolResultHistoryCommittedEvent, TurnCompletedEvent
from cns.integration.event_bus import EventBus
from utils import self_edit
from utils.distributed_lock import UserRequestLock

logger = logging.getLogger(__name__)

_TOOL_NAME = "selfedit_tool"
_RESTART_OPERATION = "request_restart"

# Upper bound on waiting for the requesting turn to release its lock. A turn
# still holding it after this is wedged; the request is already committed, so
# the restart proceeds rather than waiting forever.
_LOCK_WAIT_SECONDS = 600
_LOCK_POLL_SECONDS = 0.5


class SelfEditHandler:
    """Event-bus subscriber for the self-edit restart and outcome delivery."""

    def __init__(self, event_bus: EventBus):
        self._lock = UserRequestLock(ttl=_LOCK_WAIT_SECONDS)
        self._restart_started = threading.Event()
        event_bus.subscribe("ToolResultHistoryCommittedEvent", self._handle_tool_results_committed)
        event_bus.subscribe("TurnCompletedEvent", self._handle_turn_completed)

    def _handle_tool_results_committed(self, event: ToolResultHistoryCommittedEvent) -> None:
        requested = any(
            message.metadata.get("tool_name") == _TOOL_NAME
            and (message.metadata.get("tool_arguments") or {}).get("operation") == _RESTART_OPERATION
            and not message.is_error
            for message in event.tool_messages
        )
        if not requested or self._restart_started.is_set():
            return
        self._restart_started.set()
        context = contextvars.copy_context()
        threading.Thread(
            target=context.run,
            args=(self._restart_when_turn_ends, str(event.user_id)),
            name="self-edit-restart",
            daemon=True,
        ).start()

    def _restart_when_turn_ends(self, user_id: str) -> None:
        # The restart is committed and the user was told MIRA is restarting:
        # every path below ends in the restart. A lock that cannot be waited on
        # only costs the tail of the reply.
        try:
            deadline = time.monotonic() + _LOCK_WAIT_SECONDS
            while time.monotonic() < deadline:
                if self._lock.acquire(user_id) is not None:
                    break
                time.sleep(_LOCK_POLL_SECONDS)
            else:
                logger.error(
                    "Self-edit restart: user %s's turn still held its lock after %ds; "
                    "restarting anyway — an in-flight turn, if any, is cut off",
                    user_id, _LOCK_WAIT_SECONDS,
                )
        except Exception:
            logger.error(
                "Self-edit restart: waiting on user %s's turn lock failed; restarting "
                "without waiting — the end of the reply may not reach the client",
                user_id, exc_info=True,
            )
        self_edit.request_process_restart()

    def _handle_turn_completed(self, event: TurnCompletedEvent) -> None:
        user_id = str(event.user_id)
        if self_edit.result_for_user(user_id) is None:
            return
        messages = event.continuum.messages
        final = messages[-1] if messages else None
        turn_id = final.metadata.get("turn_id") if final is not None else None
        stimulus = next(
            (
                message for message in reversed(messages)
                if message.role == "user" and message.metadata.get("turn_id") == turn_id
            ),
            None,
        )
        if stimulus is None or stimulus.metadata.get("heartbeat") == "true":
            return
        self_edit.clear_result()
        logger.info("Self-edit outcome delivered to user %s in turn %s; cleared", user_id, turn_id)
