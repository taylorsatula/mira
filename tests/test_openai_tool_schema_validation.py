from clients.llm.dialects.openrouter import OpenRouterDialect
from clients.llm.types import Request, ToolDefinition


def test_streamed_tool_call_with_obsolete_enum_value_is_invalid() -> None:
    dialect = OpenRouterDialect(
        endpoint_url="https://openrouter.test/api/v1/chat/completions",
        api_key="test-key",
    )
    request = Request(
        messages=({"role": "user", "content": "Load the customer tool."},),
        model="openai/test-model",
        max_tokens=1024,
        tools=(
            ToolDefinition(
                name="invokeother_tool",
                description="Load an allowed tool.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "load": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["clients_tool"]},
                        },
                    },
                    "additionalProperties": False,
                },
            ),
        ),
    )

    tool_calls = dialect._parse_stream_tool_calls({
        0: {
            "id": "call-1",
            "name": "invokeother_tool",
            "arguments": '{"load":["crm_clients_tool"]}',
        },
    }, request)

    assert len(tool_calls) == 1
    assert dict(tool_calls[0].input) == {}
    assert tool_calls[0].invalid_reason is not None
    assert "crm_clients_tool" in tool_calls[0].invalid_reason


def test_streamed_tool_call_with_current_enum_value_is_accepted() -> None:
    dialect = OpenRouterDialect(
        endpoint_url="https://openrouter.test/api/v1/chat/completions",
        api_key="test-key",
    )
    request = Request(
        messages=({"role": "user", "content": "Load the customer tool."},),
        model="openai/test-model",
        max_tokens=1024,
        tools=(
            ToolDefinition(
                name="invokeother_tool",
                description="Load an allowed tool.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "load": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["clients_tool"]},
                        },
                    },
                    "additionalProperties": False,
                },
            ),
        ),
    )

    tool_calls = dialect._parse_stream_tool_calls({
        0: {
            "id": "call-1",
            "name": "invokeother_tool",
            "arguments": '{"load":["clients_tool"]}',
        },
    }, request)

    assert len(tool_calls) == 1
    assert dict(tool_calls[0].input) == {"load": ["clients_tool"]}
    assert tool_calls[0].invalid_reason is None
