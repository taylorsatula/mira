"""
Tests for generic provider real-time streaming.

TDD RED phase: These tests define the expected streaming behavior for
OpenAI-compatible providers (OpenRouter, Groq, Ollama). Tests are written
BEFORE implementation to drive the design.

Tests use REAL API calls to OpenRouter with Gemini 3 Flash model.
No mocking - we test actual streaming behavior.
"""
import pytest
import time

from clients.llm_provider import LLMProvider
from clients.vault_client import get_api_key
from cns.core.stream_events import (
    TextEvent, CompleteEvent, ToolDetectedEvent,
    ToolExecutingEvent, ToolCompletedEvent
)


@pytest.fixture(scope="module")
def anthropic_api_key():
    """Get Anthropic API key from Vault - required for LLMProvider init."""
    try:
        key = get_api_key("anthropic_key")
        if not key:
            pytest.skip("Anthropic API key not configured in Vault")
        return key
    except Exception as e:
        pytest.skip(f"Failed to retrieve Anthropic API key from Vault: {e}")


@pytest.fixture
def llm_provider(anthropic_api_key):
    """Create LLMProvider instance for testing."""
    provider = LLMProvider(
        api_key=anthropic_api_key,
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0.7,
        timeout=60,
        enable_prompt_caching=False
    )
    provider.extended_thinking = False
    return provider


class TestGenericProviderStreaming:
    """Test real-time streaming for generic OpenAI-compatible providers.

    All tests use real OpenRouter API calls.
    - Text-only tests use google/gemini-3-flash-preview (fast, clean streaming)
    - Tool tests use openai/gpt-4o-mini (Gemini has quirks with tool calls)

    Assertions are precise and strong - no hedging.
    """

    # Model for text-only tests (fast, cheap)
    TEXT_MODEL = "google/gemini-3-flash-preview"
    # Model for tool tests (Gemini requires thought_signature for tools)
    TOOL_MODEL = "openai/gpt-4o-mini"

    @pytest.fixture
    def openrouter_api_key(self):
        """Get OpenRouter API key from Vault."""
        try:
            key = get_api_key("openrouter_key")
            if not key:
                pytest.skip("OpenRouter API key not configured in Vault")
            return key
        except Exception as e:
            pytest.skip(f"Failed to retrieve OpenRouter API key: {e}")

    def test_stream_events_yields_text_incrementally(self, llm_provider, openrouter_api_key):
        """Verify text arrives as stream, not all at once.

        Precise assertion: If streaming works, the largest single chunk must be
        smaller than the total accumulated text. Non-streaming would deliver
        everything in one chunk.
        """
        messages = [{"role": "user", "content": "Count from 1 to 10, writing each number on a new line."}]

        text_events = []
        for event in llm_provider.stream_events(
            messages=messages,
            endpoint_url="https://openrouter.ai/api/v1/chat/completions",
            model_override=self.TEXT_MODEL,
            api_key_override=openrouter_api_key
        ):
            if isinstance(event, TextEvent):
                text_events.append(event)

        # Must have received text
        assert len(text_events) > 0, "No TextEvents received"

        # Calculate total text and largest chunk
        total_text = "".join(e.content for e in text_events)
        largest_chunk = max(len(e.content) for e in text_events)

        # Streaming proof: largest chunk is smaller than total
        # (Non-streaming would have largest_chunk == len(total_text))
        assert largest_chunk < len(total_text), (
            f"Text arrived in single chunk ({largest_chunk} chars), "
            f"not streamed incrementally. Total: {len(total_text)} chars"
        )

    def test_stream_events_yields_tool_detected_before_execution(self, llm_provider, openrouter_api_key):
        """Verify ToolDetectedEvent emitted BEFORE tool execution starts.

        Precise assertion: detected_time must be strictly less than executed_time.
        No conditionals - if tool wasn't called, test fails explicitly.
        """
        class TimingToolRepo:
            def __init__(self):
                self.executed_time = None

            def invoke_tool(self, name, params):
                self.executed_time = time.time()
                return {"status": "ok"}

        tool_repo = TimingToolRepo()
        llm_provider.tool_repo = tool_repo

        messages = [{"role": "user", "content": "You MUST call get_status tool. Do it now."}]
        tools = [{
            "name": "get_status",
            "description": "Get current status. You must call this tool.",
            "input_schema": {"type": "object", "properties": {}, "required": []}
        }]

        detected_time = None
        for event in llm_provider.stream_events(
            messages=messages,
            tools=tools,
            endpoint_url="https://openrouter.ai/api/v1/chat/completions",
            model_override=self.TOOL_MODEL,
            api_key_override=openrouter_api_key
        ):
            if isinstance(event, ToolDetectedEvent) and detected_time is None:
                detected_time = time.time()

        # Both times must be captured - no hedging
        assert detected_time is not None, "ToolDetectedEvent was never emitted"
        assert tool_repo.executed_time is not None, "Tool was never executed"
        assert detected_time < tool_repo.executed_time, (
            f"ToolDetectedEvent came AFTER execution: "
            f"detected={detected_time:.4f}, executed={tool_repo.executed_time:.4f}"
        )

    def test_stream_accumulates_tool_call_arguments(self, llm_provider, openrouter_api_key):
        """Verify tool call JSON arguments are correctly accumulated across chunks.

        Precise assertion: The received arguments must be valid JSON with the
        expected structure. Truncated JSON would fail to parse or have wrong keys.
        """
        received_input = None

        class CapturingToolRepo:
            def invoke_tool(self, name, params):
                nonlocal received_input
                received_input = params
                return {"results": ["result1", "result2"]}

        llm_provider.tool_repo = CapturingToolRepo()

        messages = [{
            "role": "user",
            "content": "Search for 'python streaming api' with a limit of 10 results. Use the search_web tool."
        }]
        tools = [{
            "name": "search_web",
            "description": "Search the web. Required parameters: query (string), limit (integer).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"}
                },
                "required": ["query", "limit"]
            }
        }]

        list(llm_provider.stream_events(
            messages=messages,
            tools=tools,
            endpoint_url="https://openrouter.ai/api/v1/chat/completions",
            model_override=self.TOOL_MODEL,
            api_key_override=openrouter_api_key
        ))

        # Tool must have been called - no hedging
        assert received_input is not None, "Tool was never invoked"

        # Arguments must have correct structure
        assert isinstance(received_input, dict), f"Expected dict, got {type(received_input)}"
        assert "query" in received_input, f"Missing 'query' in {received_input}"
        assert isinstance(received_input["query"], str), f"query must be string"
        assert len(received_input["query"]) > 0, "query cannot be empty"

    def test_stream_handles_multiple_concurrent_tool_calls(self, llm_provider, openrouter_api_key):
        """Verify multiple tool calls in single response are correctly parsed.

        Precise assertion: Both tools must be called. The streaming accumulator
        must correctly track multiple concurrent tool calls by their index.
        """
        calls_received = []

        class MultiToolRepo:
            def invoke_tool(self, name, params):
                calls_received.append(name)
                return {"status": f"done_{name}"}

        llm_provider.tool_repo = MultiToolRepo()

        messages = [{
            "role": "user",
            "content": (
                "You must call BOTH tools in a single response. "
                "Call get_weather for 'London' AND get_time for 'UTC'. "
                "Do not respond until you have called both tools."
            )
        }]
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a city. MUST be called.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            },
            {
                "name": "get_time",
                "description": "Get time for a timezone. MUST be called.",
                "input_schema": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"]
                }
            }
        ]

        list(llm_provider.stream_events(
            messages=messages,
            tools=tools,
            endpoint_url="https://openrouter.ai/api/v1/chat/completions",
            model_override=self.TOOL_MODEL,
            api_key_override=openrouter_api_key
        ))

        # Both tools must have been called
        assert len(calls_received) == 2, (
            f"Expected exactly 2 tool calls, got {len(calls_received)}: {calls_received}"
        )
        assert set(calls_received) == {"get_weather", "get_time"}, (
            f"Expected both get_weather and get_time, got: {calls_received}"
        )

    def test_stream_completes_with_complete_event(self, llm_provider, openrouter_api_key):
        """Verify streaming ends with CompleteEvent containing full response.

        Precise assertions:
        - Exactly one CompleteEvent at the end
        - Response has Anthropic-compatible structure (content, stop_reason, usage)
        - Accumulated text matches response content
        """
        messages = [{"role": "user", "content": "Say hello."}]

        events = list(llm_provider.stream_events(
            messages=messages,
            endpoint_url="https://openrouter.ai/api/v1/chat/completions",
            model_override=self.TEXT_MODEL,
            api_key_override=openrouter_api_key
        ))

        # Must have events
        assert len(events) > 0, "No events received"

        # Last event must be CompleteEvent
        assert isinstance(events[-1], CompleteEvent), (
            f"Last event should be CompleteEvent, got {type(events[-1]).__name__}"
        )

        # Exactly one CompleteEvent
        complete_events = [e for e in events if isinstance(e, CompleteEvent)]
        assert len(complete_events) == 1, f"Expected 1 CompleteEvent, got {len(complete_events)}"

        # Response must have Anthropic-compatible structure
        response = complete_events[0].response
        assert response is not None, "CompleteEvent.response is None"
        assert hasattr(response, 'content'), "Response missing 'content'"
        assert hasattr(response, 'stop_reason'), "Response missing 'stop_reason'"
        assert hasattr(response, 'usage'), "Response missing 'usage'"
        assert response.stop_reason in ("end_turn", "tool_use", "max_tokens"), (
            f"Invalid stop_reason: {response.stop_reason}"
        )

        # Response content must match streamed text
        text_events = [e for e in events if isinstance(e, TextEvent)]
        streamed_text = "".join(e.content for e in text_events)
        response_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        assert streamed_text == response_text, (
            f"Streamed text doesn't match response:\n"
            f"Streamed: {streamed_text!r}\n"
            f"Response: {response_text!r}"
        )
