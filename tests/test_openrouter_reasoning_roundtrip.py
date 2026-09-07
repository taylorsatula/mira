from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from clients.llm.dialects.openrouter import OpenRouterDialect
from clients.llm.events import CompleteEvent
from clients.llm.tool_messages import append_tool_result_messages, assistant_message_from_result
from clients.llm.types import (
    ReasoningArtifact,
    ReasoningEntry,
    Request,
    Result,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from utils import http_client, llm_tap


class _StreamingResponse:
    status_code = 200

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks

    def iter_lines(self) -> Iterator[str]:
        for chunk in self._chunks:
            yield f"data: {json.dumps(chunk)}"
        yield "data: [DONE]"


def test_stream_coalesces_reasoning_text_deltas_with_final_signature(monkeypatch) -> None:
    chunks = [
        {
            "choices": [{
                "delta": {
                    "reasoning_details": [{
                        "type": "reasoning.text",
                        "text": "Plan ",
                        "signature": None,
                        "id": "reasoning-text-1",
                        "format": "anthropic-claude-v1",
                        "index": 0,
                    }],
                },
                "finish_reason": None,
            }],
        },
        {
            "choices": [{
                "delta": {
                    "reasoning_details": [{
                        "type": "reasoning.text",
                        "text": "tool use.",
                        "signature": None,
                        "id": "reasoning-text-1",
                        "format": "anthropic-claude-v1",
                        "index": 0,
                    }],
                },
                "finish_reason": None,
            }],
        },
        {
            "choices": [{
                "delta": {
                    "reasoning_details": [{
                        "type": "reasoning.text",
                        "text": None,
                        "signature": "opaque-signature",
                        "id": "reasoning-text-1",
                        "format": "anthropic-claude-v1",
                        "index": 0,
                    }],
                    "content": "Done.",
                },
                "finish_reason": "stop",
            }],
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        },
    ]

    @contextmanager
    def fake_stream(*args: Any, **kwargs: Any) -> Iterator[_StreamingResponse]:
        yield _StreamingResponse(chunks)

    monkeypatch.setattr(http_client, "stream", fake_stream)

    dialect = OpenRouterDialect(
        endpoint_url="https://openrouter.test/api/v1/chat/completions",
        api_key="test-key",
    )
    request = Request(
        messages=({"role": "user", "content": "Use a tool."},),
        model="anthropic/claude-sonnet",
        max_tokens=1024,
    )

    events = list(dialect.stream(request))
    complete = next(event for event in events if isinstance(event, CompleteEvent))
    result = complete.response

    assert result.reasoning is not None
    assert result.reasoning.provider_details == ({
        "type": "reasoning.text",
        "text": "Plan tool use.",
        "signature": "opaque-signature",
        "id": "reasoning-text-1",
        "format": "anthropic-claude-v1",
        "index": 0,
    },)

    assistant_message = assistant_message_from_result(result)
    follow_up = request.with_messages((assistant_message,))
    outbound_message = dialect._build_payload(follow_up, stream=True)["messages"][0]

    assert outbound_message["reasoning_details"] == list(result.reasoning.provider_details)


def test_assistant_tool_message_omits_whitespace_only_text_before_tool_call() -> None:
    result = Result(
        text="\n\n",
        reasoning=ReasoningArtifact(entries=(ReasoningEntry(text="Need customer lookup."),)),
        tool_calls=(
            ToolCall(
                id="call_lookup_customer",
                tool_name="clients_tool",
                input={"operation": "query_customers", "search": "Taylor"},
            ),
        ),
    )

    assistant_message = assistant_message_from_result(result)

    assert assistant_message["content"] == [
        {"type": "reasoning", "text": "Need customer lookup."},
        {
            "type": "tool_call",
            "id": "call_lookup_customer",
            "name": "clients_tool",
            "input": {"operation": "query_customers", "search": "Taylor"},
        },
    ]


def test_stream_returns_schema_invalid_tool_call_for_local_retry(monkeypatch) -> None:
    captured_chunks: list[dict[str, Any]] = []
    chunks = [
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_missing_operation",
                        "function": {
                            "name": "catalog_tool",
                            "arguments": "{}",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        },
    ]

    @contextmanager
    def fake_stream(*args: Any, **kwargs: Any) -> Iterator[_StreamingResponse]:
        yield _StreamingResponse(chunks)

    monkeypatch.setattr(http_client, "stream", fake_stream)
    monkeypatch.setattr(llm_tap, "is_active", lambda: True)
    monkeypatch.setattr(
        llm_tap,
        "log_stream_chunk",
        lambda **record: captured_chunks.append(record),
    )

    dialect = OpenRouterDialect(
        endpoint_url="https://openrouter.test/api/v1/chat/completions",
        api_key="test-key",
    )
    request = Request(
        messages=({"role": "user", "content": "Create a service."},),
        model="poolside/laguna-xs-2.1",
        max_tokens=1024,
        tools=(
            ToolDefinition(
                name="catalog_tool",
                description="Manage service catalog.",
                input_schema={
                    "type": "object",
                    "properties": {"operation": {"type": "string"}},
                    "required": ["operation"],
                },
            ),
        ),
    )

    events = list(dialect.stream(request))
    complete = next(event for event in events if isinstance(event, CompleteEvent))

    assert len(complete.response.tool_calls) == 1
    tool_call = complete.response.tool_calls[0]
    assert tool_call.tool_name == "catalog_tool"
    assert dict(tool_call.input) == {}
    assert tool_call.invalid_reason is not None
    assert "missing required fields: ['operation']" in tool_call.invalid_reason
    assert captured_chunks[0]["chunk"]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_invalid_tool_call_continuation_becomes_repair_feedback() -> None:
    result = Result(
        text="",
        tool_calls=(
            ToolCall(
                id="call_missing_operation",
                tool_name="catalog_tool",
                input={},
                invalid_reason="missing required fields: ['operation']",
            ),
        ),
        stop_reason="tool_use",
    )
    tool_results = (
        ToolResult(
            tool_call_id="call_missing_operation",
            content="Error: Invalid tool call arguments for 'catalog_tool'",
            is_error=True,
        ),
    )

    messages = append_tool_result_messages([], result, tool_results)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "catalog_tool tool call was rejected" in messages[0]["content"]
    assert "missing required fields: ['operation']" in messages[0]["content"]

    dialect = OpenRouterDialect(
        endpoint_url="https://openrouter.test/api/v1/chat/completions",
        api_key="test-key",
    )
    request = Request(
        messages=tuple(messages),
        model="poolside/laguna-xs-2.1",
        max_tokens=1024,
        tools=(
            ToolDefinition(
                name="catalog_tool",
                description="Manage service catalog.",
                input_schema={
                    "type": "object",
                    "properties": {"operation": {"type": "string"}},
                    "required": ["operation"],
                },
            ),
        ),
    )

    outbound_messages = dialect._build_payload(request, stream=True)["messages"]

    assert outbound_messages == [{"role": "user", "content": messages[0]["content"]}]


def test_valid_tool_only_continuation_keeps_assistant_tool_call_pair() -> None:
    result = Result(
        text="",
        tool_calls=(
            ToolCall(
                id="call_query_services",
                tool_name="catalog_tool",
                input={"operation": "query_services"},
            ),
        ),
        stop_reason="tool_use",
    )
    tool_results = (
        ToolResult(
            tool_call_id="call_query_services",
            content='{"success": true, "services": []}',
        ),
    )

    messages = append_tool_result_messages([], result, tool_results)

    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == [
        {
            "type": "tool_call",
            "id": "call_query_services",
            "name": "catalog_tool",
            "input": {"operation": "query_services"},
        },
    ]
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "call_query_services",
        "content": '{"success": true, "services": []}',
    }


def test_outbound_coalesces_pre_fix_persisted_reasoning_fragments() -> None:
    dialect = OpenRouterDialect(
        endpoint_url="https://openrouter.test/api/v1/chat/completions",
        api_key="test-key",
    )
    request = Request(
        messages=({
            "role": "assistant",
            "content": "",
            "reasoning_details": [
                {
                    "type": "reasoning.text",
                    "text": "Plan ",
                    "id": "reasoning-text-1",
                    "format": "anthropic-claude-v1",
                    "index": 0,
                },
                {
                    "type": "reasoning.text",
                    "text": "tool use.",
                    "id": "reasoning-text-1",
                    "format": "anthropic-claude-v1",
                    "index": 0,
                },
                {
                    "type": "reasoning.text",
                    "text": None,
                    "signature": "opaque-signature",
                    "id": "reasoning-text-1",
                    "format": "anthropic-claude-v1",
                    "index": 0,
                },
            ],
        },),
        model="anthropic/claude-sonnet",
        max_tokens=1024,
    )

    outbound_message = dialect._build_payload(request, stream=True)["messages"][0]

    assert outbound_message["reasoning_details"] == [{
        "type": "reasoning.text",
        "text": "Plan tool use.",
        "signature": "opaque-signature",
        "id": "reasoning-text-1",
        "format": "anthropic-claude-v1",
        "index": 0,
    }]
