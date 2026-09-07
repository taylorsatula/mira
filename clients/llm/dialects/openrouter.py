"""OpenRouter dialect for the provider-neutral LLM boundary.

Wire shape:
  - Thinking: nested `reasoning: {effort, max_tokens}` block (both fields
    native — no translation needed).
  - Reasoning surfacing: `message.reasoning_details` (list of typed blocks)
    and `message.reasoning` (concatenated text). Streaming surfaces these in
    `delta.reasoning_details` / `delta.reasoning`.
  - Cache fields: `prompt_tokens_details.cache_write_tokens` plus the standard
    `cached_tokens` for reads.

OpenRouter's `reasoning_details` round-trips through the assistant message
on the next request — the dialect preserves it on outbound serialization
where OpenAI direct strips it.

`effort="none"` is forwarded as `reasoning: {effort: "none"}`, which OpenRouter
documents as "disables reasoning entirely". Omitting the block instead would
leave the upstream model's default in place.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any

from clients.llm.dialects.openai_chat_base import OpenAIChatBase
from clients.llm.thinking import TranslationNote
from clients.llm.types import ThinkingConfig


class OpenRouterDialect(OpenAIChatBase):
    """OpenRouter Chat Completions dialect (nested `reasoning` block)."""

    dialect_name = "openrouter"
    is_abstract = False
    native_thinking_fields = ("effort", "budget")

    def _serialize_thinking(self, payload: dict[str, Any], thinking: ThinkingConfig) -> None:
        block: dict[str, Any] = {}
        if thinking.effort is not None:
            # "none" is the gateway's own disable-reasoning value and is forwarded
            # unchanged. The gateway translates effort into the target model's
            # vocabulary, and a target that cannot disable reasoning (Claude via
            # OpenRouter rejects none) fails with a 400 rather than deliberating.
            block["effort"] = thinking.effort
        if thinking.budget_tokens is not None:
            if thinking.effort == "none":
                # reasoning.effort and reasoning.max_tokens are documented as "one
                # of the following (not both)", and a budget cannot coexist with
                # disabled reasoning. effort wins; the discard is logged.
                self._log_translation(TranslationNote(
                    field="budget_tokens",
                    requested=thinking.budget_tokens,
                    applied=None,
                    reason=(
                        "reasoning.effort 'none' disables reasoning; a reasoning "
                        "token budget cannot accompany it"
                    ),
                ))
            else:
                block["max_tokens"] = thinking.budget_tokens
        if block:
            payload["reasoning"] = block

    def _extract_reasoning_message(self, message: MappingABC[str, Any]) -> str:
        reasoning_details = message.get("reasoning_details")
        if isinstance(reasoning_details, list):
            text = self._reasoning_details_text(reasoning_details)
            if text:
                return text
        candidate = message.get("reasoning")
        if isinstance(candidate, str):
            return candidate
        return ""

    def _extract_reasoning_delta(self, delta: MappingABC[str, Any]) -> str:
        reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            return reasoning
        return ""

    def _extract_cache_usage(self, prompt_details: MappingABC[str, Any]) -> tuple[int, int]:
        return (
            self._optional_usage_token(prompt_details, "cache_write_tokens"),
            self._optional_usage_token(prompt_details, "cached_tokens"),
        )

    _accepted_round_trip_fields = frozenset({"reasoning_details"})
