"""
Asynchronous live compaction for active-chat provider context.

This service keeps one rolling compaction report per active segment in
Valkey. A background worker summarizes older messages into the report; each
provider request then carries the report in place of the messages it covers.
Segment collapse remains the durable memory pipeline; this service only
shapes provider requests.
"""
from __future__ import annotations

import html
import json
import logging
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from clients.llm_provider import LLMProvider
from clients.valkey_client import ValkeyClient, get_valkey_client
from cns.core.continuum import Continuum
from cns.core.message import (
    ContentBlock,
    Message,
    preprocess_content_blocks,
)
from cns.core.message_formatter import format_messages_for_api
from utils.timezone_utils import format_utc_iso, parse_utc_time_string, utc_now
from utils.user_context import get_current_segment_id, get_current_user_id
from config.config_manager import config

if TYPE_CHECKING:
    from cns.infrastructure.continuum_repository import ContinuumRepository

logger = logging.getLogger(__name__)

LIVE_CONTEXT_COMPACTION_KEY_PREFIX = "live_context_compaction_report"
LIVE_CONTEXT_COMPACTION_LOCK_PREFIX = "live_context_compaction_lock"
LIVE_CONTEXT_COMPACTION_LOCK_TTL_SECONDS = 300
TOOL_TEXT_TRUNCATION_LIMIT = 500


@dataclass(frozen=True)
class LiveContextCompactionArtifact:
    """The rolling compaction report for one active segment."""

    segment_id: str
    covered_start: datetime
    covered_end: datetime
    report_text: str
    compaction_count: int
    updated_at: datetime

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiveContextCompactionArtifact":
        """Parse a Valkey JSON payload into a typed artifact."""
        return cls(
            segment_id=str(data["segment_id"]),
            covered_start=parse_utc_time_string(str(data["covered_start"])),
            covered_end=parse_utc_time_string(str(data["covered_end"])),
            report_text=str(data["report_text"]),
            compaction_count=int(data["compaction_count"]),
            updated_at=parse_utc_time_string(str(data["updated_at"])),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the artifact to the canonical Valkey payload shape."""
        return {
            "segment_id": self.segment_id,
            "covered_start": format_utc_iso(self.covered_start),
            "covered_end": format_utc_iso(self.covered_end),
            "report_text": self.report_text,
            "compaction_count": self.compaction_count,
            "updated_at": format_utc_iso(self.updated_at),
        }


class LiveContextCompactionStore:
    """Read, replace, and clear the per-user live compaction artifact."""

    def __init__(self, valkey: ValkeyClient | None = None) -> None:
        self.valkey = valkey or get_valkey_client()

    def get(self, user_id: str) -> LiveContextCompactionArtifact | None:
        """Return the current artifact for user_id, or None when absent."""
        raw = self.valkey.get(self._artifact_key(user_id))
        if raw is None:
            return None
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Live context compaction artifact must be a JSON object")
        return LiveContextCompactionArtifact.from_dict(data)

    def store_replacement(
        self,
        *,
        user_id: str,
        segment_id: str,
        covered_start: datetime,
        covered_end: datetime,
        report_text: str,
        prior_artifact: LiveContextCompactionArtifact | None,
    ) -> LiveContextCompactionArtifact:
        """
        Replace the user's artifact with the segment's updated report.

        First compaction starts at count 1. Repeated compactions keep the
        original covered_start and increment compaction_count.
        """
        artifact = LiveContextCompactionArtifact(
            segment_id=segment_id,
            covered_start=(
                prior_artifact.covered_start
                if prior_artifact is not None
                else covered_start
            ),
            covered_end=covered_end,
            report_text=report_text,
            compaction_count=(
                prior_artifact.compaction_count + 1
                if prior_artifact is not None
                else 1
            ),
            updated_at=utc_now(),
        )
        self.valkey.set(self._artifact_key(user_id), json.dumps(artifact.to_dict()))
        return artifact

    def clear(self, user_id: str) -> bool:
        """Delete the user's live compaction artifact."""
        return self.valkey.delete(self._artifact_key(user_id))

    def acquire_lock(
        self,
        user_id: str,
        ttl_seconds: int = LIVE_CONTEXT_COMPACTION_LOCK_TTL_SECONDS,
    ) -> str | None:
        """Acquire the non-blocking per-user compaction lock.

        Mirrors utils.distributed_lock.DistributedLock: the lock value is a
        unique per-acquisition token, returned to the caller and required by
        release_lock. A constant value (e.g. "1") would make even a
        compare-and-delete release unsafe — a second holder's value would
        compare equal and the stale holder's release would delete the fresh
        lock.

        Returns the token when the lock was acquired, None when it is held.
        """
        token = str(uuid.uuid4())
        acquired = self.valkey.set(
            self._lock_key(user_id),
            token,
            nx=True,
            ex=ttl_seconds,
        )
        if acquired:
            return token
        return None

    def release_lock(self, user_id: str, token: str) -> bool:
        """Release the per-user compaction lock by token (compare-and-delete).

        A mismatch (the lock expired and was re-acquired by another holder)
        is a benign no-op: the fresh holder's lock survives. Returns True only
        when this owner's lock was deleted.
        """
        released = bool(
            self.valkey.compare_and_delete(self._lock_key(user_id), token)
        )
        if not released:
            logger.debug(
                "Live context compaction lock for user %s not released: "
                "token mismatch (expired and re-acquired by another holder)",
                user_id,
            )
        return released

    @staticmethod
    def _artifact_key(user_id: str) -> str:
        return f"{LIVE_CONTEXT_COMPACTION_KEY_PREFIX}:{user_id}"

    @staticmethod
    def _lock_key(user_id: str) -> str:
        return f"{LIVE_CONTEXT_COMPACTION_LOCK_PREFIX}:{user_id}"



def wrap_report(artifact: LiveContextCompactionArtifact) -> str:
    """Render the stored report inside the compaction tag with its header and recovery instruction."""
    covered_start = format_utc_iso(artifact.covered_start)
    covered_end = format_utc_iso(artifact.covered_end)
    lines = [
        f'<mira:compacted_active_context covered_start="{covered_start}" covered_end="{covered_end}">',
        (
            f"EARLIER ACTIVE CONTEXT FROM {covered_start} TO {covered_end} WAS COMPACTED "
            "BECAUSE THE CONTEXT WINDOW LIMIT WAS BEING APPROACHED."
        ),
        (
            f"This block is historical as of {covered_end}. "
            "Later visible messages are more recent and take precedence if they conflict with this block."
        ),
        "",
        "If exact wording, old tool output, or prior message order matters, use continuum_tool with:",
        "operation=search",
        "search_mode=messages",
        f"start_time={covered_start}",
        f"end_time={covered_end}",
        "",
        artifact.report_text,
        "</mira:compacted_active_context>",
    ]
    return "\n".join(lines)


def filter_messages_for_live_context(
    messages: Sequence[Message],
    artifact: LiveContextCompactionArtifact | None,
) -> list[Message]:
    """
    Return a request-local message list with the artifact's covered messages
    omitted. Segment boundaries and system notifications are always kept.
    """
    if artifact is None:
        return list(messages)

    return [
        message
        for message in messages
        if not _is_covered_ordinary_message(message, artifact)
    ]


def _is_covered_ordinary_message(
    message: Message,
    artifact: LiveContextCompactionArtifact,
) -> bool:
    if message.metadata.get("is_segment_boundary"):
        return False
    if message.metadata.get("system_notification"):
        return False
    return artifact.covered_start < message.created_at <= artifact.covered_end


@dataclass(frozen=True)
class LiveCompactionRange:
    """Selected DB-backed range for one live compaction run."""

    segment_id: str
    new_range_start: datetime
    artifact_covered_start: datetime
    covered_end: datetime
    messages: list[Message]
    prior_artifact: LiveContextCompactionArtifact | None


def format_messages_for_live_compaction(messages: Sequence[Message]) -> str:
    """
    Format messages with plain XML role tags for the live compactor LLM.

    Assistant tool calls render with their arguments so the compactor sees
    what was saved or set. The output intentionally contains no message IDs,
    timestamps, tool_call_id attributes, or error-status attributes.
    """
    parts: list[str] = []

    for message in messages:
        if message.metadata.get("system_notification"):
            continue
        if message.metadata.get("is_segment_boundary"):
            continue

        if message.role == "tool":
            content = _tool_result_text_for_compaction(message.content)
        elif message.role == "assistant":
            content = _assistant_text_for_compaction(message.content)
        else:
            content = _ordinary_message_text_for_compaction(message.content)

        if not content.strip():
            continue

        tag = message.role
        parts.append(f"<{tag}>{html.escape(content, quote=False)}</{tag}>")

    return "\n".join(parts)


def _ordinary_message_text_for_compaction(content: str | list[ContentBlock]) -> str:
    preprocessed = preprocess_content_blocks(content)
    text_parts = list(preprocessed.text_parts)
    if preprocessed.image_count:
        text_parts.insert(0, f"[{preprocessed.image_count} image(s) shared]")
    return " ".join(part for part in text_parts if part)


def _assistant_text_for_compaction(content: str | list[ContentBlock]) -> str:
    if isinstance(content, str):
        return content
    tool_calls = [
        block for block in content
        if isinstance(block, dict) and block.get("type") == "tool_call"
    ]
    other_blocks = [block for block in content if block not in tool_calls]
    text = _ordinary_message_text_for_compaction(other_blocks)
    call_texts = [
        "[Used tool: {name} {arguments}]".format(
            name=call.get("name", "unknown"),
            arguments=_truncate_tool_text(
                json.dumps(call.get("input", {}), ensure_ascii=False, default=str),
                "Tool arguments",
            ),
        )
        for call in tool_calls
    ]
    return " ".join(part for part in [text, *call_texts] if part)


def _tool_result_text_for_compaction(content: str | list[ContentBlock]) -> str:
    if isinstance(content, str):
        text = content
    else:
        text = _ordinary_message_text_for_compaction(content)
    return _truncate_tool_text(text, "Tool result")


def _truncate_tool_text(text: str, label: str) -> str:
    if len(text) <= TOOL_TEXT_TRUNCATION_LIMIT:
        return text
    return (
        text[:TOOL_TEXT_TRUNCATION_LIMIT]
        + f"\n[{label} truncated to {TOOL_TEXT_TRUNCATION_LIMIT} chars for live compaction]"
    )


class LiveContextCompactionService:
    """Selects active-segment ranges and refreshes the segment's rolling report."""

    def __init__(
        self,
        *,
        continuum_repo: "ContinuumRepository",
        store: LiveContextCompactionStore,
        llm_provider: LLMProvider,
    ) -> None:
        self.continuum_repo = continuum_repo
        self.store = store
        self.llm_provider = llm_provider
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="live-context-compaction",
        )
        self._load_prompts()

    def _load_prompts(self) -> None:
        from config.prompts.loader import load_prompt
        self._system_template = load_prompt("live_context_compaction_system.txt")
        self._user_template = load_prompt("live_context_compaction_user.txt")

    def should_schedule(self, prompt_tokens: int) -> bool:
        """Return True when the previous turn's full prompt size reached the compaction trigger."""
        return prompt_tokens >= config.api.compaction_trigger_tokens

    def schedule(self, continuum_id: str) -> bool:
        """
        Schedule one non-blocking compaction attempt for the active user.

        Returns True when a worker was submitted. Lock contention or optional
        scheduling failure returns False and leaves the active turn alone.
        """
        user_id = get_current_user_id()
        try:
            token = self.store.acquire_lock(user_id)
            if token is None:
                logger.debug("Live context compaction already running for user %s", user_id)
                return False
        except Exception:
            logger.error("Failed to acquire live context compaction lock", exc_info=True)
            return False

        ctx = copy_context()
        try:
            self._executor.submit(
                ctx.run,
                self._run_compaction_with_lock,
                continuum_id,
                user_id,
                token,
            )
            return True
        except Exception:
            self.store.release_lock(user_id, token)
            logger.error("Failed to schedule live context compaction", exc_info=True)
            return False

    def _run_compaction_with_lock(self, continuum_id: str, user_id: str, token: str) -> None:
        try:
            self.compact_once(continuum_id)
        except Exception:
            logger.error(
                "Live context compaction failed; the previous report stays in use and the next turn over the trigger retries",
                exc_info=True,
            )
        finally:
            try:
                self.store.release_lock(user_id, token)
            except Exception:
                logger.error("Failed to release live context compaction lock", exc_info=True)

    def compact_once(self, continuum_id: str) -> LiveContextCompactionArtifact | None:
        """Run one synchronous compaction attempt for the current user."""
        compaction_range = self.select_range(continuum_id)
        if compaction_range is None:
            return None

        new_history = format_messages_for_live_compaction(compaction_range.messages)
        if not new_history.strip():
            logger.debug("Live compaction selected an empty history range")
            return None

        messages = self._build_compactor_messages(compaction_range, new_history)
        response = self.llm_provider.generate_response(
            messages=messages,
            model_config="batch",
        )
        report_text = self.llm_provider.extract_text_content(response).strip()
        if not report_text:
            raise ValueError("Live context compactor returned an empty report")

        artifact = self.store.store_replacement(
            user_id=get_current_user_id(),
            segment_id=compaction_range.segment_id,
            covered_start=compaction_range.artifact_covered_start,
            covered_end=compaction_range.covered_end,
            report_text=report_text,
            prior_artifact=compaction_range.prior_artifact,
        )
        logger.info(
            "Stored live context compaction report for segment %s through %s (count=%d)",
            compaction_range.segment_id,
            format_utc_iso(artifact.covered_end),
            artifact.compaction_count,
        )
        return artifact

    def select_range(self, continuum_id: str) -> LiveCompactionRange | None:
        """Select the next DB-backed range while preserving recent raw turns."""
        user_id = get_current_user_id()
        active_sentinel = self.continuum_repo.find_active_segment(continuum_id, user_id)
        if active_sentinel is None:
            logger.debug("No active segment found for live context compaction")
            return None
        segment_id = active_sentinel.metadata.get("segment_id")
        if not segment_id:
            raise ValueError(
                f"Active segment sentinel {active_sentinel.id} has no segment_id; cannot tag a compaction report"
            )

        prior_artifact = self.store.get(user_id)
        if prior_artifact is not None and prior_artifact.segment_id != segment_id:
            self.store.clear(user_id)
            prior_artifact = None

        new_range_start = (
            prior_artifact.covered_end
            if prior_artifact is not None
            else active_sentinel.created_at
        )
        artifact_covered_start = (
            prior_artifact.covered_start
            if prior_artifact is not None
            else new_range_start
        )

        candidates = self.continuum_repo.load_messages_for_live_compaction_range(
            continuum_id=continuum_id,
            user_id=user_id,
            covered_start=new_range_start,
            covered_end=utc_now(),
        )
        user_messages = [message for message in candidates if message.role == "user"]
        if len(user_messages) <= config.api.compaction_raw_user_turns_to_preserve:
            return None

        preserve_from = user_messages[-config.api.compaction_raw_user_turns_to_preserve].created_at
        compactable_messages = [
            message for message in candidates
            if message.created_at < preserve_from
        ]
        if not compactable_messages:
            return None

        return LiveCompactionRange(
            segment_id=segment_id,
            new_range_start=new_range_start,
            artifact_covered_start=artifact_covered_start,
            covered_end=compactable_messages[-1].created_at,
            messages=compactable_messages,
            prior_artifact=prior_artifact,
        )

    def format_messages_for_api(self, continuum: Continuum) -> list[dict[str, object]]:
        """
        Return provider-formatted history with the segment's report in front.

        The stored report is read once per call, so the covered-message filter
        and the inserted report always agree. The report rides as an assistant
        scaffold message, which the overflow fallback keeps intact.
        """
        artifact = self._current_artifact()
        visible_messages = filter_messages_for_live_context(continuum.messages, artifact)
        formatted = format_messages_for_api(visible_messages)
        if artifact is None:
            return formatted
        report_message: dict[str, object] = {
            "role": "assistant",
            "content": [{"type": "text", "text": wrap_report(artifact)}],
            "_is_scaffold": True,
        }
        return [report_message, *formatted]

    def clear_report(self, user_id: str) -> None:
        """Delete the user's stored report (segment collapse)."""
        self.store.clear(user_id)

    def _current_artifact(self) -> LiveContextCompactionArtifact | None:
        try:
            artifact = self.store.get(get_current_user_id())
        except Exception:
            logger.error(
                "Live context compaction report unavailable; this request carries the full uncompacted segment",
                exc_info=True,
            )
            return None
        if artifact is None or artifact.segment_id != get_current_segment_id():
            return None
        return artifact

    def _build_compactor_messages(
        self,
        compaction_range: LiveCompactionRange,
        new_history: str,
    ) -> list[dict[str, object]]:
        covered_end = format_utc_iso(compaction_range.covered_end)
        system_prompt = self._system_template.replace("{covered_end}", covered_end)
        user_prompt = self._user_template.replace("{new_history}", new_history)

        messages: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
        ]
        if compaction_range.prior_artifact is not None:
            messages.append({
                "role": "assistant",
                "content": compaction_range.prior_artifact.report_text,
            })
        messages.append({"role": "user", "content": user_prompt})
        return messages
