"""Follow-up nudge: one short message when the user has gone quiet.

After each completed turn a one-shot APScheduler job is armed for the user,
`FOLLOW_UP_DELAY_SECONDS` after the turn commits. Any new user message cancels
it (`orchestrator.process_message` calls `cancel_follow_up`), and so does any
newer turn; a follow-up never arms a second timer, so one turn yields at most
one nudge. Pending timers are in-memory only: a restart drops them, which is
intended, because the message on the other side of a restart is stale anyway.

When the timer fires it re-checks the world under the same per-user lock the
chat transports use, because state may have moved in the two minutes: the
segment may have collapsed or paused, the user may have replied, or the model
may decide silence is correct. Every gate fails closed to silence. The model
is asked once, with no tools, and the reply must be a single JSON object;
anything other than a clean `{"follow_up": true, "message": ...}` persists
nothing.

A delivered follow-up is an ordinary assistant message stamped
`follow_up: true` (plus `segment_id`/`turn_id`), pushed through the same
`push_proactive_message_sync` bridge as the background-job notice. It is not
sent through `orchestrator.process_message` and does not call
`increment_segment_turn`: a nudge is not a user turn and must not move the
segment clock. The stamp is what stops double-nudging from the persisted
record rather than from this process's timers, so a restarted process that
re-arms after a crash still cannot nudge the same silence twice.
"""
import json
import logging
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING
from uuid import uuid4

from cns.core.events import TurnCompletedEvent
from cns.core.message import Message, MessageMetadata
from utils.scheduler_service import scheduler_service
from utils.timezone_utils import utc_now

if TYPE_CHECKING:
    from cns.integration.event_bus import EventBus

logger = logging.getLogger(__name__)

_JOB_ID_PREFIX = "follow_up:"
FOLLOW_UP_DELAY_SECONDS = 120
FOLLOW_UP_CONTEXT_MESSAGES = 6
FOLLOW_UP_MAX_MESSAGE_CHARS = 280

_request_lock = None


def _get_request_lock():
    """The per-user turn lock, sharing the transports' TTL.

    Same construction as the background-job notice: an imported structural
    bound, never a re-derived one.
    """
    global _request_lock
    if _request_lock is None:
        from cns.api.chat import _HTTP_TURN_LOCK_TTL_SECONDS
        from utils.distributed_lock import UserRequestLock

        _request_lock = UserRequestLock(ttl=_HTTP_TURN_LOCK_TTL_SECONDS)
    return _request_lock


def _job_id(user_id: str) -> str:
    return f"{_JOB_ID_PREFIX}{user_id}"


def cancel_follow_up(user_id: str) -> None:
    """Drop any pending nudge for this user. Missing job is the normal case."""
    try:
        scheduler_service.scheduler.remove_job(_job_id(user_id))
    except Exception:
        # apscheduler raises JobLookupError for an absent id; anything else is
        # worth a log line but never worth breaking the turn that called us.
        logger.debug("No pending follow-up to cancel for user %s", user_id)


def schedule_turn_follow_up(event: TurnCompletedEvent) -> None:
    """Schedule the one-shot nudge for the turn that just committed.

    Bound as a `TurnCompletedEvent` subscriber; runs in the post-commit
    callback on the request thread, so the user context is still set.
    """
    from config.config_manager import config

    if not getattr(config.system, "follow_up_enabled", False):
        return

    user_id = event.user_id
    messages = event.continuum.messages
    if not messages:
        return
    last = messages[-1]
    if last.role != "assistant" or last.metadata.get("partial_response"):
        return
    # Tool-call-only tails are not a place a human should be nudged from.
    if not isinstance(last.content, str) or not last.content.strip():
        return

    segment_id = last.metadata.get("segment_id")
    anchor_id = str(last.id)
    try:
        from apscheduler.triggers.date import DateTrigger

        scheduler_service.register_job(
            job_id=_job_id(user_id),
            func=partial(run_follow_up, user_id, segment_id, anchor_id),
            trigger=DateTrigger(run_date=utc_now() + timedelta(seconds=FOLLOW_UP_DELAY_SECONDS)),
            component="cns",
            description=f"one-shot follow-up nudge for user {user_id}",
            replace_existing=True,
        )
        logger.info(
            "Follow-up nudge armed for user %s in %ss (segment %s, anchor %s)",
            user_id,
            FOLLOW_UP_DELAY_SECONDS,
            segment_id,
            anchor_id,
        )
    except Exception:
        logger.exception("Failed to arm follow-up for user %s", user_id)


def run_follow_up(user_id: str, segment_id: str | None, anchor_id: str) -> None:
    """Scheduler entry point; runs in APScheduler's executor."""
    try:
        from utils.user_context import clear_user_context, set_current_user_id

        set_current_user_id(user_id)
        try:
            _fire(user_id, segment_id, anchor_id)
        finally:
            clear_user_context()
    except Exception:
        logger.exception("Follow-up for user %s failed", user_id)


def _fire(user_id: str, segment_id: str | None, anchor_id: str) -> None:
    from cns.infrastructure.continuum_pool import get_continuum_pool

    lock = _get_request_lock()
    token = lock.acquire(user_id)
    if token is None:
        logger.info("Follow-up for user %s dropped: turn lock held", user_id)
        return
    renewal_stop = lock.start_renewal(user_id, token)
    try:
        pool = get_continuum_pool()
        snapshot = _load_anchor(pool, user_id, segment_id, anchor_id)
        if snapshot is None:
            return
        continuum, sentinel, messages = snapshot
        message = _compose_follow_up(messages)
        if not message:
            return
        # The model call ran without the segment being re-checked; re-verify
        # before writing so a user message that landed mid-composition wins.
        snapshot = _load_anchor(pool, user_id, segment_id, anchor_id)
        if snapshot is None:
            return
        continuum, sentinel, messages = snapshot
        _persist_and_push(pool, continuum, user_id, sentinel, message)
    finally:
        renewal_stop.set()
        lock.release(user_id, token)


def _load_anchor(pool, user_id: str, segment_id: str | None, anchor_id: str):
    """Return (continuum, sentinel, messages) iff the anchor is still the tail.

    None means: no live segment, segment moved on, or the user has spoken
    since the timer was armed. All three mean silence.
    """
    continuum = pool.get_or_create()
    sentinel = pool.repository.find_active_segment(continuum.id, user_id)
    if sentinel is None:
        return None
    if sentinel.metadata.get("status") == "paused":
        return None
    current_segment = str(sentinel.metadata.get("segment_id") or "")
    if segment_id and current_segment and current_segment != str(segment_id):
        return None
    messages = pool.repository.load_segment_messages(
        continuum.id, user_id, sentinel.created_at
    )
    if not messages:
        return None
    for index, message in enumerate(messages):
        if str(message.id) == anchor_id:
            break
    else:
        return None
    anchor = messages[index]
    if anchor.role != "assistant" or anchor.metadata.get("follow_up"):
        return None
    for later in messages[index + 1:]:
        if later.role == "user":
            return None
    return continuum, sentinel, messages[: index + 1]


def _render_transcript(messages: list[Message]) -> str:
    lines = []
    for m in messages[-FOLLOW_UP_CONTEXT_MESSAGES:]:
        text = m.content if isinstance(m.content, str) else "[non-text content]"
        lines.append(f"{m.role}: {text.strip()}")
    return "\n\n".join(lines)


def _compose_follow_up(messages: list[Message]) -> str | None:
    """One `primary` call, no tools. Returns the nudge text or None for silence."""
    from clients.llm_provider import get_llm_provider
    from config.config_manager import config as app_config
    from config.prompts import load_prompt

    provider = get_llm_provider()
    transcript = _render_transcript(messages)
    try:
        response = provider.generate_response(
            messages=[{"role": "user", "content": transcript}],
            model_config="primary",
            system_prompt=(
                app_config.system_prompt
                + "\n\n"
                + load_prompt("follow_up_system.txt")
            ),
        )
        raw = provider.extract_text_content(response).strip()
    except Exception as exc:
        logger.warning("Follow-up call failed: %s", exc)
        return None

    decision = _parse_decision(raw)
    if decision is None:
        logger.info("Follow-up decision unparseable; staying silent (%r)", raw[:200])
        return None
    return decision


def _parse_decision(raw: str) -> str | None:
    """Strict JSON contract: {"follow_up": bool, "message": str?}."""
    text = raw.strip()
    if text.startswith("```"):
        # Tolerate a fenced block despite the prompt forbidding one.
        body = text.strip("`")
        text = body.split("\n", 1)[-1].strip() if "\n" in body else body
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    if data.get("follow_up") is not True:
        return None
    message = data.get("message")
    if not isinstance(message, str):
        return None
    message = message.strip()
    if not message:
        return None
    if len(message) > FOLLOW_UP_MAX_MESSAGE_CHARS:
        message = message[:FOLLOW_UP_MAX_MESSAGE_CHARS].rstrip()
    return message


def _persist_and_push(pool, continuum, user_id: str, sentinel, message: str) -> None:
    from cns.api.websocket_chat import push_proactive_message_sync
    from utils.timezone_utils import format_utc_iso

    turn_id = str(uuid4())
    metadata: MessageMetadata = {
        "segment_id": str(sentinel.metadata.get("segment_id") or ""),
        "turn_id": turn_id,
        "follow_up": True,
    }
    reply, _events = continuum.add_assistant_message(message, metadata=metadata)
    uow = pool.begin_work(continuum)
    uow.add_messages(reply)
    uow.mark_metadata_updated()
    uow.commit()
    push_proactive_message_sync(
        user_id=user_id,
        message_id=str(reply.id),
        turn_id=turn_id,
        content=message,
        created_at=format_utc_iso(reply.created_at),
    )
    logger.info("Follow-up delivered to user %s", user_id)


def install_follow_up(event_bus: "EventBus") -> None:
    """Subscribe the armer. Called once from the factory after the bus exists."""
    event_bus.subscribe("TurnCompletedEvent", schedule_turn_follow_up)
