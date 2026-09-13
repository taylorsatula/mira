"""Canonical OpenAI Chat Completions dialect.

Wire shape:
  - Thinking: top-level `reasoning_effort` field.
  - Reasoning surfacing: not exposed in the canonical OpenAI Chat Completions
    response (reasoning content is internal-only on most OpenAI models).
  - Cache fields: standard `prompt_tokens_details.cached_tokens` only.

When a caller passes `budget_tokens` without `effort`, this dialect applies
the inherited heuristic mapping and emits a TranslationNote at WARNING level -
neither OpenAI nor Groq publish per-effort token relationships, so the
translation is necessarily lossy.

`effort="none"` is forwarded as `reasoning_effort: "none"`, the explicit
no-reasoning signal, and is never omitted from the payload.
"""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse

from clients.llm.dialects.openai_chat_base import OpenAIChatBase
from clients.llm.thinking import TranslationNote
from clients.llm.types import (
    EFFORT_LEVEL_ORDER,
    DeliberationLevel,
    EffortLevel,
    Request,
    ThinkingConfig,
)


class OpenAIDialect(OpenAIChatBase):
    """OpenAI Chat Completions canonical dialect."""

    dialect_name = "openai"
    is_abstract = False
    native_thinking_fields = ("effort",)

    # Per-model effort ceilings. Models not listed support all effort levels.
    # A ceiling is always a deliberating level; "none" is never clamped upward.
    _MAX_EFFORT_PER_MODEL: dict[str, DeliberationLevel] = {
        # Add entries when a model ships with a documented effort cap.
    }

    def _convert_assistant_message(
        self,
        message: dict[str, Any],
        content: Any,
        request: Request,
    ) -> dict[str, Any]:
        converted = super()._convert_assistant_message(message, content, request)

        if self._is_local_endpoint() and isinstance(content, list):
            if any(
                isinstance(block, dict) and block.get("cache")
                for block in content
            ):
                converted["cache_checkpoint"] = True

        return converted

    def _is_local_endpoint(self) -> bool:
        host = urlparse(self.endpoint_url).hostname
        if host is None:
            return False
        if host == "localhost":
            return True
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        return address.is_loopback or address.is_private

    def _extract_reasoning_message(self, message) -> str:
        """Override base to extract reasoning_content from OpenAI-compatible
        providers (llama.cpp, local servers) that surface reasoning in the
        `reasoning_content` field."""
        return message.get("reasoning_content") or ""

    def _extract_reasoning_delta(self, delta) -> str:
        """Override base to extract reasoning_content from streaming deltas."""
        return delta.get("reasoning_content") or ""

    def _serialize_thinking(self, payload: dict[str, Any], thinking: ThinkingConfig) -> None:
        effort = thinking.effort
        if effort is None and thinking.budget_tokens is not None:
            applied = self._budget_to_effort_heuristic(thinking.budget_tokens)
            self._log_translation(TranslationNote(
                field="budget_tokens",
                requested=thinking.budget_tokens,
                applied=applied,
                reason=(
                    "OpenAI dialect lacks native budget support; applied "
                    "heuristic monotonic mapping"
                ),
            ))
            effort = applied
        elif effort is not None and thinking.budget_tokens is not None:
            # Both set; effort wins. Note the discarded budget so operators see it.
            self._log_translation(TranslationNote(
                field="budget_tokens",
                requested=thinking.budget_tokens,
                applied=None,
                reason="OpenAI dialect uses effort natively; budget discarded",
            ))

        if effort is None:
            return

        clamped = self._clamp_effort_for_model(payload.get("model"), effort, original=effort)
        # "none" goes on the wire as reasoning_effort="none" rather than as an
        # omitted parameter. Absent means "model default", and the default on this
        # family is a deliberating level (medium on gpt-5.5; llama.cpp's server -
        # the offline deployment target - documents reasoning_effort "none" as the
        # switch that disables thinking). A model that rejects the literal fails with
        # a provider 400, which is loud; omission would be silent.
        payload["reasoning_effort"] = clamped

    def _clamp_effort_for_model(
        self,
        model: object,
        requested: EffortLevel,
        *,
        original: EffortLevel,
    ) -> EffortLevel:
        if not isinstance(model, str):
            return requested
        ceiling = self._MAX_EFFORT_PER_MODEL.get(model)
        if ceiling is None:
            return requested
        # "none" sits below every ranked level, so no ceiling can apply to it -
        # and it is deliberately absent from EFFORT_LEVEL_ORDER, where index()
        # would raise.
        if requested == "none":
            return requested
        ranking = EFFORT_LEVEL_ORDER
        if ranking.index(requested) <= ranking.index(ceiling):
            return requested
        self._log_translation(TranslationNote(
            field="effort",
            requested=original,
            applied=ceiling,
            reason=f"model {model!r} caps effort at {ceiling!r}",
        ))
        return ceiling
