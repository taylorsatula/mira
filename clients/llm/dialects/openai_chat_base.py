"""Shared OpenAI Chat Completions transport for OpenAI-family dialects.

This base class owns the genuinely shared mechanics: message conversion,
tool-schema serialization, streaming SSE framework, usage parsing, and HTTP
error normalization. Subclasses (OpenAIDialect, OpenRouterDialect, GroqDialect)
override hook methods for the parts that diverge:

  - _serialize_thinking: how to encode the caller's ThinkingConfig into the
    request payload (top-level reasoning_effort, nested reasoning block, etc).
  - _extract_reasoning_message / _extract_reasoning_delta: which fields the
    provider uses to surface reasoning content.
  - _extract_cache_usage: which prompt_tokens_details fields carry cache
    write/read counts.
  - _accepted_round_trip_fields: which provider-specific round-trip fields
    (e.g., OpenRouter's reasoning_details) to copy onto the wire. Declared
    on the base Dialect class; each dialect whitelists only the fields its
    provider natively accepts.

The base class is not directly instantiable — it raises NotImplementedError
from from_selection. Discovery in clients.llm.dialect_registry intentionally
skips this module by name.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, NoReturn, TYPE_CHECKING

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from clients.llm.artifacts import FileArtifactSink
from clients.llm.events import CompleteEvent, StreamEvent, TextEvent, ThinkingEvent, ToolDetectedEvent
from clients.llm.dialects.base import (
    Dialect,
    ProviderAuthError,
    ProviderContextOverflowError,
    ProviderProtocolError,
    ProviderRetryableError,
)
from clients.llm.types import (
    EffortLevel,
    ProviderMetadata,
    ReasoningArtifact,
    ReasoningEntry,
    Request,
    Result,
    StopReason,
    ThinkingConfig,
    ToolCall,
    ToolDefinition,
    Usage,
)
from utils import http_client, llm_tap

if TYPE_CHECKING:
    from clients.llm.resolver import ModelSelection

logger = logging.getLogger(__name__)


class ToolNotLoadedError(ProviderProtocolError):
    """Provider reported a tool call for a tool not included in the request."""

    def __init__(self, endpoint: str, mode: str, tool_name: str, original_message: str):
        self.tool_name = tool_name
        self.original_message = original_message
        super().__init__(
            endpoint,
            mode,
            f"Tool '{tool_name}' not loaded by provider: {original_message}",
        )


class OpenAIChatBase(Dialect):
    """Abstract base for OpenAI-compatible Chat Completions dialects.

    Not a concrete dialect — it has no dialect_name override and the registry
    skips this module during discovery.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        api_key: str | None = None,
        timeout: int = 60,
    ) -> None:
        if not isinstance(endpoint_url, str) or not endpoint_url.strip():
            raise ValueError(f"{type(self).__name__} endpoint_url must be a non-empty string")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise PermissionError(f"{type(self).__name__} api_key must be a non-empty string when provided")
        if type(timeout) is not int or timeout <= 0:
            raise ValueError(f"{type(self).__name__} timeout must be a positive integer")
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout
        self._active_response: httpx.Response | None = None
        self._partial_usage: Usage | None = None

    @classmethod
    def from_selection(
        cls,
        selection: "ModelSelection",
        *,
        api_key: str | None,
        timeout: int,
        artifact_sink: FileArtifactSink | None,
    ) -> "OpenAIChatBase":
        # Shared construction for OpenAI-family dialects: pull endpoint from
        # the selection, source API key from the argument or Vault. A selection
        # with no api_key_name requires no credential, so the lookup is skipped
        # and the request goes out without an Authorization header — this is the
        # offline install's local llama-server route.
        if not selection.endpoint_url:
            raise ValueError(
                f"{cls.__name__} requires endpoint_url on the ModelSelection"
            )
        if api_key is None and selection.api_key_name:
            from clients.vault_client import get_api_key

            api_key = get_api_key(selection.api_key_name)
        return cls(endpoint_url=selection.endpoint_url, api_key=api_key, timeout=timeout)

    def _log_request(self, request: Request, payload: dict[str, Any]) -> None:
        from utils.llm_tap import is_active as _tap_active, log_request as _tap_request

        if _tap_active():
            _tap_request(
                provider=self.dialect_name,
                endpoint=self.endpoint_url,
                model=request.model,
                body=payload,
            )

    def _log_response(self, request: Request, result: Result) -> None:
        from utils.llm_tap import is_active as _tap_active, log_response as _tap_response

        if _tap_active():
            _tap_response(
                provider=self.dialect_name,
                model=request.model,
                response_data=result,
                endpoint=self.endpoint_url,
            )

    # ------------------------------------------------------------------
    # Hooks: subclasses override these for dialect-specific behavior.
    # ------------------------------------------------------------------

    def _serialize_thinking(self, payload: dict[str, Any], thinking: ThinkingConfig) -> None:
        """Encode the caller's ThinkingConfig into the request payload.

        Default no-op — base does not assume any thinking knob. Subclasses
        override to set reasoning_effort, nested reasoning blocks, or both.
        """

    def _extract_reasoning_message(self, message: MappingABC[str, Any]) -> str:
        """Extract reasoning text from a non-streaming response message."""
        return ""

    def _extract_reasoning_delta(self, delta: MappingABC[str, Any]) -> str:
        """Extract reasoning text from a streaming delta. Default: empty."""
        return ""

    def _extract_cache_usage(self, prompt_details: MappingABC[str, Any]) -> tuple[int, int]:
        """Return (cache_write_tokens, cache_read_tokens) from prompt_tokens_details."""
        return (0, self._optional_usage_token(prompt_details, "cached_tokens"))

    @staticmethod
    def _budget_to_effort_heuristic(budget_tokens: int) -> EffortLevel:
        """Best-effort monotonic mapping of a budget hint to an effort category.

        Heuristic — neither OpenAI nor Groq publish per-effort token budgets, so
        this is a documented guess. Dialects whose provider DOES publish
        thresholds should override this method with the documented values.

        Only "low" and "high" are emitted: both exist in every provider on this
        family, while "medium"/"xhigh"/"max" are provider-specific. A zero budget
        cannot be honoured as "no reasoning" — thinking-only models have no such
        level — so it maps to the lowest deliberating level.
        """
        if budget_tokens <= 8192:
            return "low"
        return "high"

    # ------------------------------------------------------------------
    # Transport: non-streaming completion.
    # ------------------------------------------------------------------

    def complete(self, request: Request) -> Result:
        payload = self._build_payload(request, stream=False)
        headers = self._headers()
        self._log_request(request, payload)

        try:
            # Hold the per-call transport on the abort handle for the whole
            # in-flight window (9d29): httpx.post keeps the only closeable
            # handle inside the blocking call, so abort_active_stream() found
            # nothing to close and a stalled non-streaming worker kept its
            # connection open past the lifecycle's ProviderStallError until
            # this dialect's own timeout. Closing the client closes the
            # pool's checked-out connection, releasing the peer's socket at
            # abort time. The handle here is the httpx.Client, not the
            # httpx.Response stream() registers: a non-streaming response
            # object exists only after the blocked read completes, so the
            # client is the only handle available before the block. The
            # dialect instance is per-request (llm_provider
            # ._dialect_for_selection), so this client has no other
            # in-flight request to disturb.
            with httpx.Client(timeout=self.timeout) as client:
                self._active_response = client
                try:
                    response = client.post(
                        self.endpoint_url,
                        headers=headers,
                        json=payload,
                    )
                finally:
                    self._active_response = None
        except httpx.TimeoutException as error:
            raise ProviderRetryableError(
                self.endpoint_url, 504, "non-streaming", "Request timed out"
            ) from error
        except httpx.ConnectError as error:
            raise ProviderRetryableError(
                self.endpoint_url, 503, "non-streaming", f"Connection failed: {error}"
            ) from error
        except httpx.RequestError as error:
            # Network-layer failures must not escape the dialect boundary raw;
            # normalize to the provider hierarchy (anthropic.py's
            # _call_with_overload_retry pattern).
            raise ProviderProtocolError(
                self.endpoint_url, "non-streaming", f"Transport error: {error}"
            ) from error
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e, mode="non-streaming")

        result = self._parse_response(
            self._decode_response_json(response, "non-streaming"),
            request,
        )
        self._log_response(request, result)
        return result

    # ------------------------------------------------------------------
    # Transport: streaming SSE framework.
    # ------------------------------------------------------------------

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        payload = self._build_payload(request, stream=True)
        headers = self._headers()
        self._log_request(request, payload)

        accumulated_text = ""
        accumulated_reasoning = ""
        accumulated_reasoning_details: list[dict[str, Any]] = []
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: Usage | None = None
        detected_tool_ids: set[str] = set()
        saw_reasoning_delta = False
        saw_refusal = False
        saw_done = False

        try:
            with http_client.stream(
                "POST",
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            ) as response:
                self._active_response = response
                if response.status_code >= 400:
                    error_text = response.read().decode("utf-8", errors="replace")
                    self._raise_provider_http_error(
                        status=response.status_code,
                        envelope=parse_error_body(
                            response.status_code,
                            error_text,
                            endpoint=self.endpoint_url,
                            mode="streaming",
                            dialect_name=self.dialect_name,
                        ),
                        mode="streaming",
                    )

                for line in response.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if line == "data: [DONE]":
                        saw_done = True
                        break
                    if not line.startswith("data: "):
                        continue

                    try:
                        chunk = json.loads(line[6:])
                    except json.JSONDecodeError:
                        raise ProviderProtocolError(
                            self.endpoint_url,
                            "streaming",
                            f"Malformed SSE JSON chunk: {line[:200]}",
                        )
                    if not isinstance(chunk, dict):
                        raise ProviderProtocolError(
                            self.endpoint_url,
                            "streaming",
                            "SSE JSON chunk must be an object",
                        )

                    if "error" in chunk:
                        # An error chunk is terminal for the stream, after partial
                        # events may have been yielded — same mid-stream failure
                        # semantics as the truncation raise below. Route through the
                        # HTTP-path status mapping; the stream's own status (200)
                        # lands in the terminal ProviderProtocolError branch.
                        error_value = chunk["error"]
                        if not isinstance(error_value, dict):
                            raise ProviderProtocolError(
                                self.endpoint_url,
                                "streaming",
                                f"In-band stream error value must be an object, got {type(error_value).__name__}: {line[:200]!r}",
                            )
                        self._raise_provider_http_error(
                            status=response.status_code,
                            envelope=_envelope_from_error_object(
                                response.status_code,
                                error_value,
                                line[:200],
                                endpoint=self.endpoint_url,
                                mode="streaming",
                                dialect_name=self.dialect_name,
                            ),
                            mode="streaming",
                        )

                    if llm_tap.is_active():
                        llm_tap.log_stream_chunk(
                            provider=self.dialect_name,
                            endpoint=self.endpoint_url,
                            model=request.model,
                            chunk=chunk,
                        )

                    if chunk.get("usage"):
                        chunk_usage = self._parse_usage(chunk["usage"])
                        if usage is None:
                            usage = chunk_usage
                        else:
                            usage = Usage(
                                input_tokens=max(usage.input_tokens, chunk_usage.input_tokens),
                                output_tokens=max(usage.output_tokens, chunk_usage.output_tokens),
                                cache_creation_input_tokens=max(
                                    usage.cache_creation_input_tokens,
                                    chunk_usage.cache_creation_input_tokens,
                                ),
                                cache_read_input_tokens=max(
                                    usage.cache_read_input_tokens,
                                    chunk_usage.cache_read_input_tokens,
                                ),
                            )
                        self._partial_usage = usage

                    choices = chunk.get("choices")
                    if not choices:
                        continue

                    if not isinstance(choices, list):
                        raise ProviderProtocolError(self.endpoint_url, "streaming", "SSE choices must be a list")
                    choice = choices[0]
                    if not isinstance(choice, MappingABC):
                        raise ProviderProtocolError(self.endpoint_url, "streaming", "SSE choice must be an object")
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, MappingABC):
                        raise ProviderProtocolError(self.endpoint_url, "streaming", "SSE delta must be an object")

                    if delta.get("content"):
                        text = delta["content"]
                        if not isinstance(text, str):
                            raise ProviderProtocolError(
                                self.endpoint_url,
                                "streaming",
                                "SSE content delta must be a string",
                            )
                        accumulated_text += text
                        yield TextEvent(content=text)

                    # Provider refusal signal, parsed upstream of
                    # finish-reason normalization (ticket d554). Streaming
                    # refusals arrive as delta.refusal (SDK: ChoiceDelta.refusal,
                    # openai/types/chat/chat_completion_chunk.py:77) and can
                    # co-exist with finish_reason "stop", so the field itself
                    # drives the typed refusal. Terminality contract (d554,
                    # full statement at _parse_response's refusal parse): the
                    # refusal is terminal for the turn; already-streamed
                    # partial text stays in Result.text — the refusal payload
                    # is streamed to the user as text and accumulated into
                    # Result.text rather than discarded.
                    if delta.get("refusal") is not None:
                        refusal_delta = delta["refusal"]
                        if not isinstance(refusal_delta, str):
                            raise ProviderProtocolError(
                                self.endpoint_url,
                                "streaming",
                                "SSE refusal delta must be a string",
                            )
                        if refusal_delta:
                            saw_refusal = True
                            accumulated_text += refusal_delta
                            yield TextEvent(content=refusal_delta)

                    reasoning_text = self._extract_reasoning_delta(delta)
                    if reasoning_text:
                        saw_reasoning_delta = True
                        reasoning_delta = self._new_reasoning_text(accumulated_reasoning, reasoning_text)
                        if reasoning_delta:
                            accumulated_reasoning += reasoning_delta
                            yield ThinkingEvent(content=reasoning_delta)

                    if delta.get("reasoning_details"):
                        details = delta["reasoning_details"]
                        if not isinstance(details, list):
                            raise ProviderProtocolError(
                                self.endpoint_url,
                                "streaming",
                                "SSE reasoning_details delta must be a list",
                            )
                        self._accumulate_reasoning_details(
                            accumulated_reasoning_details,
                            details,
                            mode="streaming",
                        )
                        if not saw_reasoning_delta:
                            details_text = self._reasoning_details_text(details)
                            reasoning_delta = self._new_reasoning_text(accumulated_reasoning, details_text)
                            if reasoning_delta:
                                accumulated_reasoning += reasoning_delta
                                yield ThinkingEvent(content=reasoning_delta)

                    if delta.get("tool_calls"):
                        if not isinstance(delta["tool_calls"], list):
                            raise ProviderProtocolError(
                                self.endpoint_url,
                                "streaming",
                                "SSE tool_calls delta must be a list",
                            )
                        for tool_call_delta in delta["tool_calls"]:
                            if not isinstance(tool_call_delta, MappingABC):
                                raise ProviderProtocolError(
                                    self.endpoint_url,
                                    "streaming",
                                    "SSE tool_call delta must be an object",
                                )
                            index = tool_call_delta.get("index")
                            if type(index) is not int:
                                raise ProviderProtocolError(
                                    self.endpoint_url,
                                    "streaming",
                                    "SSE tool_call delta index must be an integer",
                                )
                            state = accumulated_tool_calls.setdefault(
                                index,
                                {"id": "", "name": "", "arguments": ""},
                            )
                            if tool_call_delta.get("id"):
                                if not isinstance(tool_call_delta["id"], str):
                                    raise ProviderProtocolError(
                                        self.endpoint_url,
                                        "streaming",
                                        "SSE tool_call id delta must be a string",
                                    )
                                state["id"] = tool_call_delta["id"]
                            function_delta = tool_call_delta.get("function") or {}
                            if not isinstance(function_delta, MappingABC):
                                raise ProviderProtocolError(
                                    self.endpoint_url,
                                    "streaming",
                                    "SSE tool_call function delta must be an object",
                                )
                            if function_delta.get("name"):
                                if not isinstance(function_delta["name"], str):
                                    raise ProviderProtocolError(
                                        self.endpoint_url,
                                        "streaming",
                                        "SSE tool_call name delta must be a string",
                                    )
                                state["name"] = function_delta["name"]
                            if function_delta.get("arguments"):
                                if not isinstance(function_delta["arguments"], str):
                                    raise ProviderProtocolError(
                                        self.endpoint_url,
                                        "streaming",
                                        "SSE tool_call arguments delta must be a string",
                                    )
                                state["arguments"] += function_delta["arguments"]

                            if state["id"] and state["name"] and state["id"] not in detected_tool_ids:
                                detected_tool_ids.add(state["id"])
                                yield ToolDetectedEvent(
                                    tool_name=state["name"],
                                    tool_id=state["id"],
                                )
        except httpx.TimeoutException as error:
            raise ProviderRetryableError(
                self.endpoint_url, 504, "streaming", "Request timed out"
            ) from error
        except httpx.ConnectError as error:
            raise ProviderRetryableError(
                self.endpoint_url, 503, "streaming", f"Connection failed: {error}"
            ) from error
        except httpx.RequestError as error:
            # Network-layer failures must not escape the dialect boundary raw;
            # normalize to the provider hierarchy (anthropic.py's
            # _call_with_overload_retry pattern).
            raise ProviderProtocolError(
                self.endpoint_url, "streaming", f"Transport error: {error}"
            ) from error

        self._active_response = None

        # A stream that ends without [DONE] and without a finish_reason was cut
        # mid-response (connection close, gateway fault). Parsing the partial
        # accumulation would report truncated tool arguments as "missing required
        # fields" or hand the user a silently truncated reply as complete.
        if not saw_done and finish_reason is None:
            raise ProviderProtocolError(
                self.endpoint_url,
                "streaming",
                "Stream ended without [DONE] and without finish_reason — response truncated",
            )

        if usage is None:
            logger.warning(
                "%s stream from %s ended without usage despite include_usage; "
                "billing will be skipped for model=%s",
                self.dialect_name,
                self.endpoint_url,
                request.model,
            )

        result = Result(
            text=accumulated_text,
            tool_calls=self._parse_stream_tool_calls(accumulated_tool_calls, request),
            reasoning=self._build_reasoning(
                reasoning_text=accumulated_reasoning,
                reasoning_details=accumulated_reasoning_details or None,
            ),
            usage=usage,
            stop_reason=(
                "refusal" if saw_refusal else self._normalize_finish_reason(finish_reason)
            ),
            provider_metadata=ProviderMetadata(
                dialect_name=self.dialect_name,
                endpoint_url=self.endpoint_url,
            ),
        )

        self._log_response(request, result)

        yield CompleteEvent(response=result)

    # ------------------------------------------------------------------
    # Live-stream hooks: cross-thread abort and partial-usage snapshot.
    # ------------------------------------------------------------------

    def abort_active_stream(self) -> None:
        """Close the in-flight streaming response from another thread.

        The worker sits blocked inside a socket read; closing the httpx
        response is the only unblock that works (generator.close() raises
        ValueError while the frame is executing, not suspended at a yield).
        """
        response = self._active_response
        if response is not None:
            response.close()

    def current_partial_usage(self) -> Usage | None:
        return self._partial_usage

    # ------------------------------------------------------------------
    # Payload construction.
    # ------------------------------------------------------------------

    def _build_payload(self, request: Request, *, stream: bool) -> dict[str, Any]:
        max_tokens = request.max_tokens
        if request.thinking.budget_tokens is not None:
            max_tokens += request.thinking.budget_tokens

        messages = []
        if request.system:
            messages.append(self._convert_system_prompt(request.system))
        sanitized = [self.sanitize_outbound_message(m) for m in request.messages]
        messages.extend(self._convert_messages(sanitized, request))

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if request.tools:
            payload["tools"] = self._convert_tools(request.tools)
        if request.thinking.active:
            self._serialize_thinking(payload, request.thinking)
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _convert_system_prompt(self, system: str | list[dict[str, Any]]) -> dict[str, str]:
        if isinstance(system, list):
            text = "".join(
                block.get("text", "")
                for block in system
                if block.get("type") == "text"
            )
            return {"role": "system", "content": text}
        return {"role": "system", "content": system}

    # ------------------------------------------------------------------
    # Message conversion.
    # ------------------------------------------------------------------

    def _convert_messages(
        self,
        messages: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        request: Request,
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "tool":
                tool_content = message.get("content")
                extracted_images: list[dict[str, Any]] = []
                tool_text = self._extract_text_from_content(tool_content, extracted_images)
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id", ""),
                    "content": tool_text,
                })
                if extracted_images:
                    openai_messages.append({
                        "role": "user",
                        "content": extracted_images,
                    })
                continue

            content = message.get("content")
            if role == "user":
                openai_messages.extend(self._convert_user_message(content))
            elif role == "assistant":
                openai_messages.append(self._convert_assistant_message(message, content, request))
        return openai_messages

    def _convert_user_message(self, content: Any) -> list[dict[str, Any]]:
        if not isinstance(content, list):
            return [{"role": "user", "content": content}]

        content_parts: list[dict[str, Any]] = []
        dropped_block_types: list[str] = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                content_parts.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "image":
                content_parts.append(self._convert_image_block(block))
            elif block_type == "file_ref":
                file_id = block.get("file_id", "unknown")
                content_parts.append({"type": "text", "text": f"[File upload not supported by this provider: {file_id}]"})
            elif block_type == "document":
                content_parts.append({"type": "text", "text": f"[Document not supported by this provider: {block.get('media_type', 'unknown')}]"})
            elif block_type == "reasoning":
                dropped_block_types.append("reasoning")
            else:
                content_parts.append(block)

        if dropped_block_types:
            logger.warning(
                "OpenAI-family dialect dropped %d block(s) of type(s) %s from user message — "
                "these types have no equivalent in OpenAI Chat Completions",
                len(dropped_block_types),
                list(dict.fromkeys(dropped_block_types)),
            )

        if not content_parts:
            # Entire user message would vanish (e.g., message with only reasoning blocks).
            # Log loudly so operators see the data loss instead of discovering it as
            # a broken conversation later.
            logger.error(
                "OpenAI-family dialect produced empty user message from content with %d block(s) — "
                "user turn is being dropped entirely. Dropped types: %s",
                len(dropped_block_types),
                dropped_block_types,
            )
            raise ProviderProtocolError(
                self.endpoint_url,
                "message-conversion",
                f"Cannot convert user message: all blocks stripped (types: {dropped_block_types}). "
                "OpenAI-family providers have no equivalent for these block types.",
            )

        has_image = any(part.get("type") == "image_url" for part in content_parts)
        if has_image:
            return [{"role": "user", "content": content_parts}]
        else:
            text_content = "".join(part.get("text", "") for part in content_parts if part.get("type") == "text")
            if text_content:
                return [{"role": "user", "content": text_content}]
            return []

    def _extract_text_from_content(
        self,
        content: Any,
        extracted_images: list[dict[str, Any]],
    ) -> str:
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if not isinstance(block, MappingABC):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "image":
                    extracted_images.append(self._convert_image_block(block))
            return "".join(text_parts)
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return json.dumps(content)
        return str(content if content is not None else "")

    def _convert_image_block(self, block: dict[str, Any]) -> dict[str, Any]:
        media_type = block.get("media_type")
        if not isinstance(media_type, str) or not media_type:
            raise ProviderProtocolError(
                self.endpoint_url,
                "message-conversion",
                "Image block media_type must be a non-empty string",
            )
        data = block.get("data")
        if not isinstance(data, str) or not data:
            raise ProviderProtocolError(
                self.endpoint_url,
                "message-conversion",
                "Image block data must be a non-empty string",
            )
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{data}",
                "detail": "auto",
            },
        }

    def _convert_assistant_message(
        self,
        message: dict[str, Any],
        content: Any,
        request: Request,
    ) -> dict[str, Any]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        if isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_call":
                    # Historical assistant tool call: serialize the stored input
                    # verbatim. Replay-time validation against the current
                    # request's tools would raise ToolNotLoadedError every turn
                    # for the rest of a conversation that used an ephemeral tool;
                    # argument validation belongs in the response-parse path.
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(
                                block.get("input") if "input" in block else {}
                            ),
                        },
                    })
        elif isinstance(content, str):
            text_parts.append(content)

        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                function = tool_call["function"]
                arguments = function.get("arguments") if "arguments" in function else None
                # Historical assistant tool call: a stored JSON-arguments string
                # is replayed verbatim; any other stored value is serialized as-is.
                # No validation against the current request's tools — replay-time
                # validation would raise ToolNotLoadedError every turn for the
                # rest of a conversation that used an ephemeral tool.
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments if arguments is not None else {})
                tool_calls.append({
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": arguments,
                    },
                })

        converted: dict[str, Any] = {"role": "assistant"}
        converted["content"] = "".join(text_parts) if text_parts else None
        if tool_calls:
            converted["tool_calls"] = tool_calls
        # Copy round-trip fields onto the wire. sanitize_outbound_message()
        # already stripped foreign metadata upstream, so only fields this
        # dialect accepts can be present.
        for field in self._accepted_round_trip_fields:
            if message.get(field):
                round_trip_value = list(message[field])
                if field == "reasoning_details":
                    coalesced: list[dict[str, Any]] = []
                    self._accumulate_reasoning_details(
                        coalesced,
                        round_trip_value,
                        mode="message-conversion",
                    )
                    round_trip_value = coalesced
                converted[field] = round_trip_value
        return converted

    def _convert_tools(self, tools: tuple[ToolDefinition, ...]) -> list[dict[str, Any]]:
        result = []
        for tool in tools:
            tool_dict: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.input_schema),
                },
            }
            if tool.provider_options:
                tool_dict["function"].update(dict(tool.provider_options))
            if tool.cache is not None:
                tool_dict["function"]["cache"] = tool.cache
            result.append(tool_dict)
        return result

    # ------------------------------------------------------------------
    # Response parsing.
    # ------------------------------------------------------------------

    def _parse_response(self, response: dict[str, Any], request: Request) -> Result:
        if (
            "choices" not in response
            or not isinstance(response["choices"], list)
            or not response["choices"]
        ):
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Response missing choices")
        if "usage" not in response:
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Response missing usage")

        choice = response["choices"][0]
        if not isinstance(choice, MappingABC):
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Response choice must be an object")
        message = choice.get("message") or {}
        if not isinstance(message, MappingABC) or not message:
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Response choice has empty message")

        reasoning_details = response.get("reasoning_details") or message.get("reasoning_details")
        reasoning_text = self._extract_reasoning_message(message)
        if not reasoning_text and isinstance(reasoning_details, list):
            reasoning_text = self._reasoning_details_text(reasoning_details)

        content = message.get("content")
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, MappingABC) and block.get("type") == "text"
            )
        else:
            raise ProviderProtocolError(
                self.endpoint_url,
                "non-streaming",
                "Response message content must be a string, list, or null",
            )

        # Provider refusal signal, parsed upstream of finish-reason
        # normalization (ticket d554). OpenAI structured-output refusals
        # populate message.refusal (SDK: ChatCompletionMessage.refusal,
        # openai/types/chat/chat_completion_message.py:63) and can arrive
        # with finish_reason "stop", so the field itself — not the finish
        # reason alone — must drive the typed refusal. Terminality contract
        # (d554): a refusal is TERMINAL for the turn — no tool loop, no
        # retry at this layer — and the already-produced partial text is
        # preserved: Result.text carries the content plus the surfaced
        # refusal payload; a filter that stops output mid-generation must
        # not discard what was generated before it.
        refusal = message.get("refusal")
        if refusal is not None:
            if not isinstance(refusal, str):
                raise ProviderProtocolError(
                    self.endpoint_url,
                    "non-streaming",
                    "Response message refusal must be a string",
                )
            if refusal:
                text = f"{text}\n\n{refusal}" if text else refusal

        tool_calls = message.get("tool_calls") or ()
        if not isinstance(tool_calls, (list, tuple)):
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Response tool_calls must be a list")

        return Result(
            text=text,
            tool_calls=tuple(
                self._parse_tool_call(tool_call, request)
                for tool_call in tool_calls
            ),
            reasoning=self._build_reasoning(
                reasoning_text=reasoning_text,
                reasoning_details=reasoning_details,
            ),
            usage=self._parse_usage(response["usage"]),
            # Refusal field beats the finish reason: a structured-output
            # refusal can carry finish_reason "stop" (see the contract
            # comment above); content_filter reaches "refusal" through
            # _normalize_finish_reason's mapping.
            stop_reason=(
                "refusal" if refusal else self._normalize_finish_reason(choice.get("finish_reason"))
            ),
            provider_metadata=ProviderMetadata(
                dialect_name=self.dialect_name,
                endpoint_url=self.endpoint_url,
            ),
        )

    def _reasoning_details_text(self, reasoning_details: list[Any]) -> str:
        parts = []
        for item in reasoning_details:
            if isinstance(item, MappingABC) and item.get("type") == "reasoning.text":
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)

    def _accumulate_reasoning_details(
        self,
        accumulated: list[dict[str, Any]],
        details: list[Any],
        *,
        mode: str,
    ) -> None:
        """Coalesce consecutive OpenRouter reasoning.text stream fragments."""
        for index, item in enumerate(details):
            if not isinstance(item, MappingABC):
                raise ProviderProtocolError(
                    self.endpoint_url,
                    mode,
                    f"reasoning_details[{index}] must be an object",
                )
            detail = dict(item)
            previous = accumulated[-1] if accumulated else None
            if (
                detail.get("type") == "reasoning.text"
                and previous is not None
                and previous.get("type") == "reasoning.text"
            ):
                previous_text = previous.get("text")
                detail_text = detail.get("text")
                if previous_text is not None and not isinstance(previous_text, str):
                    raise ProviderProtocolError(
                        self.endpoint_url,
                        mode,
                        "reasoning.text text must be a string or null",
                    )
                if detail_text is not None and not isinstance(detail_text, str):
                    raise ProviderProtocolError(
                        self.endpoint_url,
                        mode,
                        "reasoning.text text must be a string or null",
                    )
                previous["text"] = (previous_text or "") + (detail_text or "")
                if not previous.get("signature") and detail.get("signature"):
                    previous["signature"] = detail["signature"]
                if not previous.get("format") and detail.get("format"):
                    previous["format"] = detail["format"]
            else:
                accumulated.append(detail)

    def _new_reasoning_text(self, accumulated_reasoning: str, candidate: str) -> str:
        if not candidate:
            return ""
        if accumulated_reasoning and candidate.startswith(accumulated_reasoning):
            return candidate[len(accumulated_reasoning):]
        # Dedup is aimed at provider/transport-level duplicate deltas
        # (reconnection/replay artifacts), not model behavior. A model
        # emitting the same reasoning fragment repeatedly was assessed by
        # the human as an impossible situation, so any
        # "legitimate repetition" concern is out of scope by decision. If
        # genuine model-level repetition ever shows up in evidence, revisit
        # this decision rather than widening the predicate.
        if accumulated_reasoning.endswith(candidate):
            return ""
        return candidate

    def _build_reasoning(self, *, reasoning_text: str, reasoning_details: Any) -> ReasoningArtifact | None:
        if not reasoning_text and not reasoning_details:
            return None
        details = (
            tuple(item for item in reasoning_details if isinstance(item, dict))
            if isinstance(reasoning_details, list)
            else ()
        )
        entries = (ReasoningEntry(text=reasoning_text),) if reasoning_text else ()
        return ReasoningArtifact(entries=entries, provider_details=details)

    def _decode_response_json(self, response: httpx.Response, mode: str) -> dict[str, Any]:
        try:
            parsed = response.json()
        except ValueError as error:
            raise ProviderProtocolError(
                self.endpoint_url,
                mode,
                f"Provider returned non-JSON response: {response.text[:200]}",
            ) from error
        if not isinstance(parsed, dict):
            raise ProviderProtocolError(self.endpoint_url, mode, "Provider response JSON must be an object")
        return parsed

    # ------------------------------------------------------------------
    # Tool-call parsing.
    # ------------------------------------------------------------------

    def _parse_tool_arguments(
        self,
        raw_arguments: Any,
        *,
        tool_name: str,
        tool_id: str,
        request: Request,
        mode: str,
    ) -> dict[str, Any]:
        required = self._required_tool_fields(tool_name, request, mode=mode)
        if raw_arguments is None or raw_arguments == "":
            if required:
                raise ProviderProtocolError(
                    self.endpoint_url,
                    mode,
                    (
                        f"Tool call '{tool_id}' for '{tool_name}' omitted JSON arguments "
                        f"required by schema: {required}"
                    ),
                )
            return {}
        if not isinstance(raw_arguments, str):
            raise ProviderProtocolError(
                self.endpoint_url,
                mode,
                f"Tool call '{tool_id}' for '{tool_name}' arguments must be a JSON string",
            )
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            raise ProviderProtocolError(
                self.endpoint_url,
                mode,
                f"Tool call '{tool_id}' for '{tool_name}' contains malformed JSON arguments",
            ) from error
        return self._coerce_tool_input(
            parsed,
            tool_name=tool_name,
            tool_id=tool_id,
            request=request,
            mode=mode,
        )

    def _coerce_tool_input(
        self,
        value: Any,
        *,
        tool_name: str,
        tool_id: str,
        request: Request,
        mode: str,
    ) -> dict[str, Any]:
        tool = self._loaded_tool_definition(tool_name, request, mode=mode)
        required = self._required_tool_fields(tool_name, request, mode=mode)
        if value is None:
            if required:
                raise ProviderProtocolError(
                    self.endpoint_url,
                    mode,
                    (
                        f"Tool call '{tool_id}' for '{tool_name}' omitted input required by schema: "
                        f"{required}"
                    ),
                )
            return {}
        if not isinstance(value, MappingABC):
            raise ProviderProtocolError(
                self.endpoint_url,
                mode,
                f"Tool call '{tool_id}' for '{tool_name}' input must be a JSON object",
            )
        missing = [field for field in required if field not in value]
        if missing:
            raise ProviderProtocolError(
                self.endpoint_url,
                mode,
                f"Tool call '{tool_id}' for '{tool_name}' missing required fields: {missing}",
            )
        try:
            validator = Draft202012Validator(dict(tool.input_schema))
        except SchemaError as error:
            raise RuntimeError(
                f"Tool '{tool_name}' has an invalid input schema: {error.message}"
            ) from error

        validation_errors = sorted(
            validator.iter_errors(dict(value)),
            key=lambda error: (
                tuple(str(part) for part in error.absolute_path),
                error.message,
            ),
        )
        if validation_errors:
            error = validation_errors[0]
            location = ".".join(str(part) for part in error.absolute_path) or "input"
            raise ProviderProtocolError(
                self.endpoint_url,
                mode,
                f"Tool call '{tool_id}' for '{tool_name}' violates schema at {location}: {error.message}",
            )
        return dict(value)

    def _required_tool_fields(self, tool_name: str, request: Request, *, mode: str) -> tuple[str, ...]:
        tool = self._loaded_tool_definition(tool_name, request, mode=mode)
        required = tool.input_schema.get("required", [])
        return tuple(required) if isinstance(required, (list, tuple)) else ()

    def _loaded_tool_definition(self, tool_name: str, request: Request, *, mode: str) -> ToolDefinition:
        for tool in request.tools:
            if tool.name == tool_name:
                return tool
        raise ToolNotLoadedError(
            self.endpoint_url,
            mode,
            tool_name,
            f"Provider attempted to call tool '{tool_name}' that was not included in the request",
        )

    def _parse_tool_call(self, tool_call: dict[str, Any], request: Request) -> ToolCall:
        if not isinstance(tool_call, MappingABC):
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Tool call must be an object")
        function = tool_call.get("function") or {}
        if not isinstance(function, MappingABC):
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Tool call function must be an object")
        tool_id = tool_call.get("id")
        tool_name = function.get("name")
        if not isinstance(tool_id, str) or not tool_id.strip():
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Tool call missing id")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Tool call function missing name")
        invalid_reason = None
        try:
            tool_input = self._parse_tool_arguments(
                function.get("arguments") if "arguments" in function else None,
                tool_name=tool_name,
                tool_id=tool_id,
                request=request,
                mode="non-streaming",
            )
        except ToolNotLoadedError:
            # Subclasses ProviderProtocolError: must escape the broad schema-repair
            # catch below so the lifecycle's invokeother_tool recovery runs.
            raise
        except ProviderProtocolError as error:
            invalid_reason = str(error)
            tool_input = {}
        return ToolCall(
            id=tool_id,
            tool_name=tool_name,
            input=tool_input,
            invalid_reason=invalid_reason,
        )

    def _parse_stream_tool_calls(
        self,
        accumulated_tool_calls: dict[int, dict[str, Any]],
        request: Request,
    ) -> tuple[ToolCall, ...]:
        calls = []
        for index in sorted(accumulated_tool_calls):
            state = accumulated_tool_calls[index]
            if not state["id"] or not state["name"]:
                raise ProviderProtocolError(
                    self.endpoint_url,
                    "streaming",
                    f"Streamed tool call at index {index} ended without id and name",
                )
            invalid_reason = None
            try:
                tool_input = self._parse_tool_arguments(
                    state["arguments"],
                    tool_name=state["name"],
                    tool_id=state["id"],
                    request=request,
                    mode="streaming",
                )
            except ToolNotLoadedError:
                # Subclasses ProviderProtocolError: must escape the broad schema-repair
                # catch below so the lifecycle's invokeother_tool recovery runs.
                raise
            except ProviderProtocolError as error:
                invalid_reason = str(error)
                tool_input = {}
            calls.append(
                ToolCall(
                    id=state["id"],
                    tool_name=state["name"],
                    input=tool_input,
                    invalid_reason=invalid_reason,
                )
            )
        return tuple(calls)

    # ------------------------------------------------------------------
    # Usage parsing.
    # ------------------------------------------------------------------

    def _parse_usage(self, usage: dict[str, Any]) -> Usage:
        if not isinstance(usage, MappingABC):
            raise ProviderProtocolError(self.endpoint_url, "usage", "Usage payload must be an object")
        prompt_details = usage.get("prompt_tokens_details") or {}
        if not isinstance(prompt_details, MappingABC):
            raise ProviderProtocolError(self.endpoint_url, "usage", "prompt_tokens_details must be an object")
        prompt_tokens = self._required_usage_token(usage, "prompt_tokens")
        completion_tokens = self._required_usage_token(usage, "completion_tokens")
        cache_write, cache_read = self._extract_cache_usage(prompt_details)
        return Usage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        )

    def _required_usage_token(self, usage: MappingABC[str, Any], field_name: str) -> int:
        if field_name not in usage:
            raise ProviderProtocolError(self.endpoint_url, "usage", f"Usage payload missing {field_name}")
        value = usage[field_name]
        if type(value) is not int or value < 0:
            raise ProviderProtocolError(
                self.endpoint_url,
                "usage",
                f"Usage field {field_name} must be a non-negative integer",
            )
        return value

    def _optional_usage_token(self, usage: MappingABC[str, Any], field_name: str) -> int:
        value = usage.get(field_name, 0)
        if value is None:
            return 0
        if type(value) is not int or value < 0:
            raise ProviderProtocolError(
                self.endpoint_url,
                "usage",
                f"Usage field {field_name} must be a non-negative integer when provided",
            )
        return value

    def _normalize_finish_reason(self, finish_reason: str | None) -> StopReason:
        # Every documented value maps deliberately (ticket d554); no known
        # value falls to the end_turn default. Unknown values still fall
        # through with the unmapped warning below.
        #
        # Documented sources:
        #   - OpenAI SDK Choice.finish_reason literal set — installed openai
        #     package, types/chat/chat_completion.py:25 (Choice.finish_reason)
        #     and types/chat/chat_completion_chunk.py:100 (ChoiceDelta-bearing
        #     chunk): "stop", "length", "tool_calls", "content_filter",
        #     "function_call".
        #   - OpenRouter normalizes every provider to (OpenRouter API
        #     reference, openrouter.ai/docs/api_reference/overview): "stop",
        #     "length", "tool_calls", "content_filter", "error"; the raw
        #     provider value rides along in native_finish_reason.
        #
        # Per-value decisions:
        #   stop           -> end_turn   (SDK literal; normal completion)
        #   length         -> max_tokens (SDK literal; output hit the token
        #                                limit)
        #   tool_calls     -> tool_use   (SDK literal; the model invoked tools)
        #   content_filter -> refusal    (SDK literal + OpenRouter normalized;
        #                                a provider filter stopped output —
        #                                typed refusal, partial text kept;
        #                                see the terminality contract at
        #                                _parse_response's refusal parse)
        #   function_call  -> end_turn   (SDK literal; legacy function
        #                                calling. This transport parses only
        #                                message.tool_calls, so no ToolCall
        #                                exists to justify tool_use — end_turn
        #                                is the honest terminal state.)
        #   error          -> error      (OpenRouter normalized value; a
        #                                provider error reported in-band on a
        #                                200. Consumers verified to tolerate
        #                                StopReason "error": coerce_stop_reason
        #                                (STOP_REASONS member, clients/llm/
        #                                types.py:24-32), orchestrator metadata
        #                                (opaque str, cns/services/
        #                                orchestrator.py:1363), actions debug
        #                                log (cns/api/actions.py:2120), POST
        #                                probe dict output
        #                                (utils/power_on_self_test.py:1347) —
        #                                nothing branches on the value.)
        #   max_tokens     -> max_tokens (non-SDK spelling some OpenAI-family
        #                                providers emit instead of "length")
        #   safety         -> refusal    (native filter stop, e.g. Gemini
        #                                "SAFETY" — same hazard as
        #                                content_filter: a filter stopped
        #                                output)
        #   recitation     -> refusal    (native filter stop, e.g. Gemini
        #                                "RECITATION" — same hazard as
        #                                content_filter)
        mapping: dict[str, StopReason] = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "max_tokens": "max_tokens",
            "content_filter": "refusal",
            "function_call": "end_turn",
            "error": "error",
            "safety": "refusal",
            "recitation": "refusal",
        }
        normalized = finish_reason.lower() if isinstance(finish_reason, str) else None
        if normalized not in mapping and normalized is not None:
            logger.warning("Unmapped finish_reason='%s', defaulting to end_turn", finish_reason)
        return mapping.get(normalized, "end_turn")

    # ------------------------------------------------------------------
    # Error normalization.
    # ------------------------------------------------------------------

    def _handle_http_error(self, error: httpx.HTTPStatusError, *, mode: str) -> NoReturn:
        response = error.response
        fallback_text = str(error)
        if response is not None:
            self._raise_provider_http_error(
                status=response.status_code,
                envelope=parse_error_body(
                    response.status_code,
                    response.text,
                    endpoint=self.endpoint_url,
                    mode=mode,
                    dialect_name=self.dialect_name,
                ),
                mode=mode,
            )
        raise ProviderProtocolError(
            self.endpoint_url,
            mode,
            f"{self.dialect_name} provider HTTP error without response: {fallback_text}",
        )

    def _raise_provider_http_error(
        self,
        *,
        status: int,
        envelope: ProviderErrorEnvelope,
        mode: str,
    ) -> NoReturn:
        if status >= 400:
            logger.error("%s API %s error %d — envelope: %r", self.dialect_name, mode, status, envelope)
        error_message = envelope.message
        # 413 (Groq request-too-large) carries overflow payloads too; tool_use_failed stays 400-only.
        if status in (400, 413):
            error_code = envelope.code or ""
            if "context_length" in error_code or "reduce the length" in error_message.lower():
                raise ProviderContextOverflowError(self.endpoint_url, mode, error_message)
            if status == 400 and error_code == "tool_use_failed":
                match = re.search(r"attempted to call tool '(\w+)'", error_message)
                if match:
                    raise ToolNotLoadedError(self.endpoint_url, mode, match.group(1), error_message)

        if status in (401, 403):
            raise ProviderAuthError(self.endpoint_url, mode, error_message)
        if status == 429 or status >= 500:
            raise ProviderRetryableError(self.endpoint_url, status, mode, error_message)
        raise ProviderProtocolError(
            self.endpoint_url,
            mode,
            f"{self.dialect_name} provider API error {status}: {error_message}",
        )


@dataclass(frozen=True)
class ProviderErrorEnvelope:
    """Strictly validated provider error body.

    The single typed value every error-normalization path consumes: ``message``
    is fully formatted at parse time (including OpenRouter provider/description
    tags); ``code`` feeds overflow / tool-use classification.
    """

    message: str
    code: str | None = None


def parse_error_body(
    status: int,
    raw_text: str,
    *,
    endpoint: str,
    mode: str,
    dialect_name: str,
) -> ProviderErrorEnvelope:
    """Single boundary where untyped provider error bytes become a typed value.

    Exactly two branches: the documented JSON envelope
    (``{"error": {"message": <str>, ...}}``) or non-JSON text (proxy HTML /
    empty body — a known enumerated shape, not tolerated). A JSON object
    without an object-valued ``error`` field is degraded to a raw-text
    envelope (contract violation logged) so the status taxonomy stays
    reachable. Every other shape is a contract violation and raises
    ProviderProtocolError with a raw excerpt.
    """
    try:
        decoded = json.loads(raw_text)
    except json.JSONDecodeError:
        return ProviderErrorEnvelope(message=raw_text)
    if not isinstance(decoded, dict):
        raise ProviderProtocolError(
            endpoint,
            mode,
            f"{dialect_name} error {status} body is JSON but not an object: {raw_text[:200]!r}",
        )
    error_info = decoded.get("error")
    if not isinstance(error_info, dict):
        # Non-conformant but parseable JSON object (e.g. {"detail": ...},
        # {"error": "string"}): degrade to a raw-text envelope so the
        # status taxonomy (auth / retryable / overflow) in
        # _raise_provider_http_error still runs instead of the raise
        # happening from inside its argument list.
        logger.error(
            "%s error %d body violates error-envelope contract "
            "(expected object-valued 'error' field): %r",
            dialect_name,
            status,
            raw_text[:200],
        )
        return ProviderErrorEnvelope(message=raw_text, code=None)
    return _envelope_from_error_object(status, error_info, raw_text[:200], endpoint=endpoint, mode=mode, dialect_name=dialect_name)


def _envelope_from_error_object(
    status: int,
    error_info: dict[str, Any],
    raw_excerpt: str,
    *,
    endpoint: str,
    mode: str,
    dialect_name: str,
) -> ProviderErrorEnvelope:
    """Validate an already-decoded ``error`` object into an envelope.

    Shared by ``parse_error_body`` (HTTP error bodies) and the SSE in-band
    error-chunk path; the OpenRouter provider/description tag formatting is
    applied here, once, at parse time.
    """
    message = error_info.get("message")
    if not isinstance(message, str) or not message:
        raise ProviderProtocolError(
            endpoint,
            mode,
            f"{dialect_name} error {status} envelope must carry a non-empty string 'error.message': {raw_excerpt!r}",
        )
    code: str | None = None
    raw_code = error_info.get("code")
    if raw_code is not None:
        if isinstance(raw_code, bool) or not isinstance(raw_code, (str, int)):
            raise ProviderProtocolError(
                endpoint,
                mode,
                f"{dialect_name} error {status} 'error.code' must be a string: {raw_excerpt!r}",
            )
        code = str(raw_code)
    # Include additional context fields (e.g. OpenRouter provider/description)
    # that carry the real error reason beyond the top-level message string.
    extra: list[str] = []
    for key in ("provider", "description"):
        value = error_info.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ProviderProtocolError(
                endpoint,
                mode,
                f"{dialect_name} error {status} 'error.{key}' must be a string: {raw_excerpt!r}",
            )
        extra.append(f"{key}={value}")
    formatted = f"{message} [{', '.join(extra)}]" if extra else message
    return ProviderErrorEnvelope(message=formatted, code=code)
