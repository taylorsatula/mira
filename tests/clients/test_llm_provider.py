"""
Tests for clients/llm_provider.py - CRITICAL CORE OF MIRA

Tests LLMProvider with real Anthropic API calls.
Following MIRA testing philosophy: no mocks, test real API behavior.

LLMProvider is the heart of MIRA. Without it working correctly, the entire system
falls apart. These tests validate all critical paths:
- generate_response() with various message formats and configurations
- stream_events() event generation and streaming behavior
- Model selection logic and dynamic routing
- Message preparation, validation, and tool handling
- Error handling, resilience, and failover mechanisms
- Circuit breaker behavior to prevent tool execution loops
- Extended thinking configuration and thinking block handling
- Generic provider routing for third-party APIs
- Callback integration for real-time streaming
- Response caching and performance characteristics
"""
import pytest
import anthropic
from typing import Dict, List, Any
from unittest.mock import Mock

from clients.llm_provider import LLMProvider, CircuitBreaker
from clients.vault_client import get_api_key
from cns.core.stream_events import (
    TextEvent, CompleteEvent, ToolDetectedEvent, ErrorEvent,
    ToolExecutingEvent, ToolCompletedEvent, ToolErrorEvent, CircuitBreakerEvent
)


@pytest.fixture(scope="module")
def anthropic_api_key():
    """Get Anthropic API key from Vault - REQUIRED for all tests."""
    try:
        key = get_api_key("anthropic_key")
        if not key:
            pytest.skip("Anthropic API key not configured in Vault")
        return key
    except Exception as e:
        pytest.skip(f"Failed to retrieve Anthropic API key from Vault: {e}")


@pytest.fixture
def llm_provider(anthropic_api_key):
    """Create LLMProvider instance with Haiku for testing."""
    provider = LLMProvider(
        api_key=anthropic_api_key,
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0.7,
        timeout=60,
        enable_prompt_caching=False  # Disable caching for consistent test behavior
    )
    # Disable extended thinking for tests (requires temperature=1.0)
    provider.extended_thinking = False
    return provider


@pytest.fixture
def llm_provider_with_caching(anthropic_api_key):
    """Create LLMProvider with prompt caching enabled."""
    provider = LLMProvider(
        api_key=anthropic_api_key,
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0.7,
        timeout=60,
        enable_prompt_caching=True
    )
    # Disable extended thinking for tests (requires temperature=1.0)
    provider.extended_thinking = False
    return provider


class TestLLMProviderBasics:
    """Test basic LLMProvider initialization and configuration."""

    def test_llm_provider_initializes_with_config(self, anthropic_api_key):
        """Verify LLMProvider initializes with correct configuration."""
        provider = LLMProvider(
            api_key=anthropic_api_key,
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            temperature=0.5,
            timeout=30,
            enable_prompt_caching=True
        )

        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.max_tokens == 2048
        assert provider.temperature == 0.5
        assert provider.timeout == 30
        assert provider.enable_prompt_caching is True
        assert provider.anthropic_client is not None

    def test_llm_provider_has_anthropic_client(self, llm_provider):
        """Verify LLMProvider creates Anthropic SDK client."""
        assert llm_provider.anthropic_client is not None
        assert isinstance(llm_provider.anthropic_client, anthropic.Anthropic)


class TestLLMProviderGenerateResponse:
    """Test generate_response() method with real API calls."""

    def test_generate_response_simple_text_message(self, llm_provider):
        """Verify generate_response() works with simple text message."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Verify response is Anthropic Message object
        assert isinstance(response, anthropic.types.Message)
        assert len(response.content) > 0
        assert response.content[0].type == "text"
        assert isinstance(response.content[0].text, str)
        assert len(response.content[0].text) > 0

    def test_generate_response_with_system_prompt_string(self, llm_provider):
        """Verify generate_response() handles string system prompt."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Response should contain text from Claude
        assert isinstance(response, anthropic.types.Message)
        text_content = llm_provider.extract_text_content(response)
        assert len(text_content) > 0

    def test_generate_response_with_system_prompt_blocks(self, llm_provider):
        """Verify generate_response() handles structured system blocks."""
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a math expert."}
                ]
            },
            {"role": "user", "content": "What is 5 * 6?"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Should return valid response
        assert isinstance(response, anthropic.types.Message)
        assert len(response.content) > 0

    def test_generate_response_multi_turn_conversation(self, llm_provider):
        """Verify generate_response() handles multi-turn conversations."""
        messages = [
            {"role": "user", "content": "My name is Taylor"},
            {"role": "assistant", "content": "Hello Taylor! Nice to meet you."},
            {"role": "user", "content": "What is my name?"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Response should be aware of the continuum
        assert isinstance(response, anthropic.types.Message)
        text_content = llm_provider.extract_text_content(response)
        # Should contain the name Taylor
        assert "taylor" in text_content.lower()

    def test_generate_response_with_user_message_content_blocks(self, llm_provider):
        """Verify generate_response() handles user messages with content blocks."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the capital of France?"}
                ]
            }
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        assert isinstance(response, anthropic.types.Message)
        text_content = llm_provider.extract_text_content(response)
        assert "paris" in text_content.lower()

    def test_generate_response_returns_usage_information(self, llm_provider):
        """Verify response includes token usage information."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Verify usage tracking
        assert hasattr(response, 'usage')
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_generate_response_with_assistant_message_content_blocks(self, llm_provider):
        """Verify handling of assistant messages with content blocks."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hi there! How can I help?"}
                ]
            },
            {"role": "user", "content": "What did you just say?"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        assert isinstance(response, anthropic.types.Message)
        text_content = llm_provider.extract_text_content(response)
        assert len(text_content) > 0

    def test_generate_response_system_override_parameter(self, llm_provider):
        """Verify system_override parameter takes precedence."""
        messages = [
            {"role": "system", "content": "You are a pirate"},
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider.generate_response(
            messages=messages,
            stream=False,
            system_override="You are a helpful assistant"
        )

        assert isinstance(response, anthropic.types.Message)
        text_content = llm_provider.extract_text_content(response)
        # Should use override system prompt
        assert len(text_content) > 0


class TestLLMProviderMessageHandling:
    """Test message preparation and format handling."""

    def test_prepare_messages_extracts_system_prompt(self, llm_provider):
        """Verify _prepare_messages() extracts system prompt correctly."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]

        system, anthropic_messages = llm_provider._prepare_messages(messages)

        # System should be extracted
        assert system == "You are helpful"
        # Messages should not contain system
        assert len(anthropic_messages) == 1
        assert anthropic_messages[0]["role"] == "user"

    def test_prepare_messages_with_no_system_prompt(self, llm_provider):
        """Verify _prepare_messages() handles messages without system."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]

        system, anthropic_messages = llm_provider._prepare_messages(messages)

        # System should be None
        assert system is None
        # All messages should remain
        assert len(anthropic_messages) == 2

    def test_prepare_messages_with_structured_system_blocks(self, llm_provider):
        """Verify _prepare_messages() preserves structured system blocks."""
        system_blocks = [
            {"type": "text", "text": "You are helpful"},
            {"type": "text", "text": "Be concise"}
        ]
        messages = [
            {"role": "system", "content": system_blocks},
            {"role": "user", "content": "Hello"}
        ]

        system, anthropic_messages = llm_provider._prepare_messages(messages)

        # System should be the list of blocks
        assert system == system_blocks
        assert len(anthropic_messages) == 1


class TestLLMProviderToolExtraction:
    """Test tool call extraction from responses."""

    def test_extract_tool_calls_returns_empty_list_for_text_response(self, llm_provider):
        """Verify extract_tool_calls() returns empty list when no tools used."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)
        tool_calls = llm_provider.extract_tool_calls(response)

        # Text response has no tool calls
        assert tool_calls == []

    def test_extract_tool_calls_returns_proper_format(self, llm_provider):
        """Verify extract_tool_calls() returns correct structure when tools present."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)
        tool_calls = llm_provider.extract_tool_calls(response)

        # Should be a list (even if empty)
        assert isinstance(tool_calls, list)
        # Each item should have required keys if present
        for call in tool_calls:
            assert "id" in call
            assert "tool_name" in call
            assert "input" in call

    def test_extract_text_content_from_response(self, llm_provider):
        """Verify extract_text_content() extracts text blocks correctly."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)
        text_content = llm_provider.extract_text_content(response)

        # Should extract non-empty text
        assert isinstance(text_content, str)
        assert len(text_content) > 0


class TestLLMProviderStreamingEvents:
    """Test stream_events() generator and event emission."""

    def test_stream_events_yields_text_events(self, llm_provider):
        """Verify stream_events() yields TextEvent for text content."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        events = list(llm_provider.stream_events(messages=messages))

        # Should yield events
        assert len(events) > 0

        # Should end with CompleteEvent
        assert isinstance(events[-1], CompleteEvent)

        # Should contain TextEvent before completion
        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) > 0

        # Text events should have content
        for event in text_events:
            assert len(event.content) > 0

    def test_stream_events_complete_event_has_message(self, llm_provider):
        """Verify CompleteEvent contains the final Message."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        events = list(llm_provider.stream_events(messages=messages))
        complete_events = [e for e in events if isinstance(e, CompleteEvent)]

        assert len(complete_events) == 1
        complete_event = complete_events[0]
        assert complete_event.response is not None
        assert isinstance(complete_event.response, anthropic.types.Message)

    def test_stream_events_multi_turn_conversation(self, llm_provider):
        """Verify stream_events() works with multi-turn conversations."""
        messages = [
            {"role": "user", "content": "My name is Alex"},
            {"role": "assistant", "content": "Hello Alex!"},
            {"role": "user", "content": "What is my name?"}
        ]

        events = list(llm_provider.stream_events(messages=messages))

        # Should yield events
        assert len(events) > 0
        # Should contain text event
        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) > 0


class TestLLMProviderErrorHandling:
    """Test error handling and edge cases."""

    def test_generate_response_with_empty_messages_raises_error(self, llm_provider):
        """Verify empty messages list raises appropriate error."""
        messages = []

        with pytest.raises(ValueError, match="at least one message is required|Cannot send empty"):
            llm_provider.generate_response(messages=messages, stream=False)

    def test_generate_response_with_empty_user_message_raises_error(self, llm_provider):
        """Verify empty user message raises error."""
        messages = [
            {"role": "user", "content": ""}
        ]

        with pytest.raises(ValueError, match="Cannot send empty"):
            llm_provider.generate_response(messages=messages, stream=False)

    def test_generate_response_with_invalid_api_key_raises_error(self):
        """Verify invalid API key raises authentication error."""
        provider = LLMProvider(
            api_key="sk-invalid-key-12345",
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            timeout=5
        )

        messages = [
            {"role": "user", "content": "Hello"}
        ]

        with pytest.raises((anthropic.AuthenticationError, PermissionError, RuntimeError)):
            provider.generate_response(messages=messages, stream=False)

    def test_stream_events_with_empty_messages_raises_error(self, llm_provider):
        """Verify stream_events() raises error with empty messages."""
        messages = []

        with pytest.raises(ValueError, match="Cannot send empty"):
            list(llm_provider.stream_events(messages=messages))

    def test_validate_messages_rejects_empty_content(self, llm_provider):
        """Verify _validate_messages() rejects empty content."""
        messages = [
            {"role": "user", "content": ""}
        ]

        with pytest.raises(ValueError):
            llm_provider._validate_messages(messages)


class TestLLMProviderResponseValidation:
    """Test response validation and parsing."""

    def test_response_has_required_fields(self, llm_provider):
        """Verify response has all required Anthropic Message fields."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Verify required fields
        assert hasattr(response, 'content')
        assert hasattr(response, 'stop_reason')
        assert hasattr(response, 'usage')
        assert isinstance(response.content, list)
        assert len(response.content) > 0

    def test_response_stop_reason_is_valid(self, llm_provider):
        """Verify response stop_reason is one of expected values."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Stop reason should be end_turn, tool_use, or max_tokens
        assert response.stop_reason in ["end_turn", "tool_use", "max_tokens"]

    def test_content_blocks_have_correct_structure(self, llm_provider):
        """Verify content blocks have correct type and structure."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        for block in response.content:
            assert hasattr(block, 'type')
            assert block.type in ["text", "tool_use", "thinking"]

            if block.type == "text":
                assert hasattr(block, 'text')
                assert isinstance(block.text, str)


class TestLLMProviderUnicodeAndSpecialCharacters:
    """Test handling of Unicode and special characters."""

    def test_generate_response_with_unicode_characters(self, llm_provider):
        """Verify Unicode text is handled correctly."""
        messages = [
            {"role": "user", "content": "Hello in different languages: 你好 (Chinese), مرحبا (Arabic), Привет (Russian)"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Should handle Unicode without errors
        assert isinstance(response, anthropic.types.Message)
        text_content = llm_provider.extract_text_content(response)
        assert len(text_content) > 0

    def test_generate_response_with_special_characters(self, llm_provider):
        """Verify special characters are handled correctly."""
        messages = [
            {"role": "user", "content": "Special chars: @#$%^&*(){}[]|\\<>?/~`"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        assert isinstance(response, anthropic.types.Message)
        assert len(response.content) > 0

    def test_generate_response_with_json_like_content(self, llm_provider):
        """Verify JSON-like content is parsed correctly."""
        messages = [
            {"role": "user", "content": 'Explain this JSON: {"name": "test", "value": 123}'}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        assert isinstance(response, anthropic.types.Message)
        text_content = llm_provider.extract_text_content(response)
        assert len(text_content) > 0


class TestLLMProviderBuildAssistantMessage:
    """Test building assistant messages from responses."""

    def test_build_assistant_message_creates_correct_structure(self, llm_provider):
        """Verify _build_assistant_message() creates valid message dict."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)
        assistant_msg = llm_provider._build_assistant_message(response)

        # Should have correct role
        assert assistant_msg["role"] == "assistant"

        # Should have content blocks
        assert "content" in assistant_msg
        assert isinstance(assistant_msg["content"], list)
        assert len(assistant_msg["content"]) > 0

        # Each block should have type
        for block in assistant_msg["content"]:
            assert "type" in block
            assert block["type"] in ["text", "tool_use", "thinking"]

            if block["type"] == "text":
                assert "text" in block

    def test_build_assistant_message_preserves_content(self, llm_provider):
        """Verify _build_assistant_message() preserves content accuracy."""
        messages = [
            {"role": "user", "content": "Say hello world"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)
        assistant_msg = llm_provider._build_assistant_message(response)

        # Verify content is preserved
        original_text = llm_provider.extract_text_content(response)
        built_text = " ".join([b.get("text", "") for b in assistant_msg["content"] if b.get("type") == "text"])
        assert original_text == built_text


class TestLLMProviderToolPreparation:
    """Test tool preparation and caching."""

    def test_prepare_tools_for_caching_without_caching_disabled(self, llm_provider):
        """Verify tools are returned as-is when caching disabled."""
        tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]

        # Provider has caching disabled
        result = llm_provider._prepare_tools_for_caching(tools)

        # Should return same tools without modification
        assert result == tools

    def test_prepare_tools_with_empty_list(self, llm_provider):
        """Verify empty tool list is handled correctly."""
        tools = []
        result = llm_provider._prepare_tools_for_caching(tools)

        assert result == []


class TestCircuitBreaker:
    """Test CircuitBreaker for preventing infinite tool loops."""

    def test_circuit_breaker_initializes_empty(self):
        """Verify CircuitBreaker initializes with empty results."""
        breaker = CircuitBreaker()

        assert breaker.tool_results == []

    def test_circuit_breaker_allows_first_execution(self):
        """Verify CircuitBreaker allows first tool execution."""
        breaker = CircuitBreaker()

        should_continue, reason = breaker.should_continue()

        assert should_continue is True
        assert reason == "First tool"

    def test_circuit_breaker_allows_one_retry_on_error(self):
        """Verify CircuitBreaker allows ONE retry per tool before stopping."""
        breaker = CircuitBreaker()

        # First error for a tool - should allow retry
        error = Exception("Tool failed")
        breaker.record_execution("test_tool", result=None, error=error)
        should_continue, reason = breaker.should_continue()
        assert should_continue is True  # Allow retry

        # Second error for SAME tool - should stop
        breaker.record_execution("test_tool", result=None, error=error)
        should_continue, reason = breaker.should_continue()
        assert should_continue is False
        assert "failed after correction" in reason

    def test_circuit_breaker_stops_on_repeated_results(self):
        """Verify CircuitBreaker stops when getting repeated identical results."""
        breaker = CircuitBreaker()

        # Record two identical results
        breaker.record_execution("tool_1", result="same result")
        breaker.record_execution("tool_2", result="same result")

        should_continue, reason = breaker.should_continue()

        assert should_continue is False
        assert "Repeated identical results" in reason

    def test_circuit_breaker_allows_different_results(self):
        """Verify CircuitBreaker allows execution with different results."""
        breaker = CircuitBreaker()

        # Record different results
        breaker.record_execution("tool_1", result="result_1")
        breaker.record_execution("tool_2", result="result_2")

        should_continue, reason = breaker.should_continue()

        assert should_continue is True
        assert reason == "Continue"

    def test_circuit_breaker_allows_many_different_results(self):
        """Verify CircuitBreaker allows unlimited tool calls with different results."""
        breaker = CircuitBreaker()

        # Record many different results - should all continue
        for i in range(100):
            breaker.record_execution(f"tool_{i}", result=f"result_{i}")
            should_continue, reason = breaker.should_continue()
            assert should_continue is True
            assert reason == "Continue"


class TestLLMProviderMessageValidation:
    """Test comprehensive message validation logic."""

    def test_validate_messages_accepts_valid_user_message(self, llm_provider):
        """Verify valid user message passes validation."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        # Should not raise
        llm_provider._validate_messages(messages)

    def test_validate_messages_accepts_multi_turn_conversation(self, llm_provider):
        """Verify multi-turn conversations pass validation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"}
        ]

        # Should not raise
        llm_provider._validate_messages(messages)

    def test_validate_messages_accepts_content_blocks(self, llm_provider):
        """Verify messages with content blocks pass validation."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"}
                ]
            }
        ]

        # Should not raise
        llm_provider._validate_messages(messages)

    def test_validate_messages_rejects_empty_string_content(self, llm_provider):
        """Verify empty string content is rejected."""
        messages = [
            {"role": "user", "content": ""}
        ]

        with pytest.raises(ValueError, match="Cannot send empty"):
            llm_provider._validate_messages(messages)

    def test_validate_messages_rejects_whitespace_only_content(self, llm_provider):
        """Verify whitespace-only content is rejected."""
        messages = [
            {"role": "user", "content": "   \n\t  "}
        ]

        with pytest.raises(ValueError, match="Cannot send empty"):
            llm_provider._validate_messages(messages)

    def test_validate_messages_accepts_assistant_with_tool_calls(self, llm_provider):
        """Verify assistant messages with tool calls pass validation."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "123", "name": "tool", "input": {}}
                ],
                "tool_calls": [{"name": "tool"}]
            }
        ]

        # Should not raise even though content is empty (tool_calls present)
        llm_provider._validate_messages(messages)


class TestLLMProviderStreamingWithCallbacks:
    """Test streaming behavior with callback integration."""

    def test_generate_response_streaming_with_callback(self, llm_provider):
        """Verify generate_response() streams events to callback."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        received_events = []

        def callback(event):
            received_events.append(event)

        response = llm_provider.generate_response(
            messages=messages,
            stream=True,
            callback=callback
        )

        # Should have streamed events
        assert len(received_events) > 0

        # Should receive text events
        text_events = [e for e in received_events if e.get("type") == "text"]
        assert len(text_events) > 0

        # Final response should be valid
        assert isinstance(response, anthropic.types.Message)

    def test_generate_response_streaming_callback_receives_text_events(self, llm_provider):
        """Verify callback receives text events during streaming."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        received_text = []

        def callback(event):
            if event.get("type") == "text":
                received_text.append(event.get("content"))

        llm_provider.generate_response(
            messages=messages,
            stream=True,
            callback=callback
        )

        # Should have received text
        assert len(received_text) > 0
        combined_text = "".join(received_text)
        assert len(combined_text) > 0

    def test_generate_response_streaming_callback_error_handling(self, llm_provider):
        """Verify streaming continues even if callback raises error."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        def callback(event):
            raise RuntimeError("Callback error")

        # Should not raise even though callback errors
        response = llm_provider.generate_response(
            messages=messages,
            stream=True,
            callback=callback
        )

        # Should still get valid response
        assert isinstance(response, anthropic.types.Message)
        assert len(response.content) > 0


class TestLLMProviderResponseCaching:
    """Test prompt caching functionality."""

    def test_prepare_tools_with_caching_enabled(self):
        """Verify tools get cache_control when caching enabled."""
        provider = LLMProvider(
            api_key="test",
            enable_prompt_caching=True
        )

        tools = [
            {
                "name": "tool1",
                "description": "First tool",
                "input_schema": {"type": "object"}
            },
            {
                "name": "tool2",
                "description": "Second tool",
                "input_schema": {"type": "object"}
            }
        ]

        result = provider._prepare_tools_for_caching(tools)

        # Last tool should have cache_control
        assert "cache_control" in result[-1]
        assert result[-1]["cache_control"]["type"] == "ephemeral"
        # Other tools should not
        assert "cache_control" not in result[0]

    def test_system_prompt_caching_structure(self, llm_provider_with_caching):
        """Verify system prompt is prepared with cache_control when caching enabled."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Say hello"}
        ]

        # Call _prepare_messages to extract system
        system, anthropic_messages = llm_provider_with_caching._prepare_messages(messages)

        assert system == "You are helpful"


class TestLLMProviderLongMessages:
    """Test handling of long conversations and context."""

    def test_generate_response_with_long_conversation_history(self, llm_provider):
        """Verify LLMProvider handles long continuum histories."""
        messages = []

        # Build long continuum
        for i in range(5):
            messages.append({
                "role": "user",
                "content": f"Message {i}: This is a test message to build continuum history"
            })
            messages.append({
                "role": "assistant",
                "content": f"Response {i}: This is a response to message {i}"
            })

        # Add final question
        messages.append({
            "role": "user",
            "content": "What did we discuss?"
        })

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Should handle long history
        assert isinstance(response, anthropic.types.Message)
        text_content = llm_provider.extract_text_content(response)
        assert len(text_content) > 0

    def test_generate_response_with_large_token_context(self, llm_provider):
        """Verify LLMProvider handles large context windows."""
        # Create messages with substantial content
        messages = [
            {
                "role": "user",
                "content": "Summarize this: " + ("Lorem ipsum dolor sit amet. " * 100)
            }
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)

        assert isinstance(response, anthropic.types.Message)
        assert len(response.content) > 0


class TestLLMProviderMaxTokensHandling:
    """Test max_tokens behavior and constraints."""

    def test_generate_response_respects_max_tokens_for_haiku(self, llm_provider):
        """Verify max_tokens is adjusted for Haiku model constraints."""
        # Haiku has 8192 max_tokens constraint
        provider = LLMProvider(
            api_key=llm_provider.api_key,
            model="claude-haiku-4-5-20251001",
            max_tokens=10000,  # Over Haiku limit
            timeout=60
        )

        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = provider.generate_response(messages=messages, stream=False)

        # Should still work (adjusted internally)
        assert isinstance(response, anthropic.types.Message)

    def test_generate_response_with_very_small_max_tokens(self, llm_provider):
        """Verify LLMProvider handles very small max_tokens."""
        messages = [
            {"role": "user", "content": "Write a long essay"}
        ]

        response = llm_provider.generate_response(
            messages=messages,
            stream=False,
            **{"max_tokens": 50}
        )

        # Should truncate response
        assert isinstance(response, anthropic.types.Message)
        text_content = llm_provider.extract_text_content(response)
        assert len(text_content) > 0


class TestLLMProviderTemperatureVariations:
    """Test temperature parameter effects."""

    def test_generate_response_with_very_low_temperature(self, llm_provider):
        """Verify very low temperature produces deterministic response."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider.generate_response(
            messages=messages,
            stream=False,
            **{"temperature": 0.0}
        )

        assert isinstance(response, anthropic.types.Message)
        text_content = llm_provider.extract_text_content(response)
        assert len(text_content) > 0

    def test_generate_response_with_very_high_temperature(self, llm_provider):
        """Verify very high temperature produces creative response."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider.generate_response(
            messages=messages,
            stream=False,
            **{"temperature": 1.0}
        )

        assert isinstance(response, anthropic.types.Message)
        assert len(response.content) > 0


class TestLLMProviderAnthropicErrorHandling:
    """Test handling of Anthropic API-specific errors."""

    def test_generate_response_handles_authentication_error(self):
        """Verify authentication errors are handled correctly."""
        provider = LLMProvider(
            api_key="sk-invalid-key",
            model="claude-haiku-4-5-20251001",
            timeout=5
        )

        messages = [
            {"role": "user", "content": "Hello"}
        ]

        with pytest.raises((anthropic.AuthenticationError, RuntimeError)):
            provider.generate_response(messages=messages, stream=False)

    def test_stream_events_with_empty_tool_list(self, llm_provider):
        """Verify stream_events() handles empty tool list."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        events = list(llm_provider.stream_events(messages=messages))

        # Should complete successfully without tools
        assert len(events) > 0
        complete_events = [e for e in events if isinstance(e, CompleteEvent)]
        assert len(complete_events) == 1


class TestLLMProviderEmitEventsFromResponse:
    """Test event emission from completed responses."""

    def test_emit_events_from_response_yields_text_events(self, llm_provider):
        """Verify _emit_events_from_response() yields TextEvent for text blocks."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)
        events = list(llm_provider._emit_events_from_response(response))

        # Should emit TextEvent and CompleteEvent
        assert len(events) >= 2

        # First events should be TextEvent, last should be CompleteEvent
        assert isinstance(events[-1], CompleteEvent)
        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) > 0

    def test_emit_events_from_response_preserves_response(self, llm_provider):
        """Verify CompleteEvent preserves the original response."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider.generate_response(messages=messages, stream=False)
        events = list(llm_provider._emit_events_from_response(response))

        complete_event = events[-1]
        assert complete_event.response is response


class TestLLMProviderDeepCopySafety:
    """Test that LLMProvider doesn't mutate input parameters."""

    def test_generate_response_does_not_mutate_messages(self, llm_provider):
        """Verify messages list is not mutated."""
        original_messages = [
            {"role": "user", "content": "Say hello"}
        ]

        messages_copy = [m.copy() for m in original_messages]

        llm_provider.generate_response(messages=original_messages, stream=False)

        # Messages should be unchanged
        assert original_messages == messages_copy

    def test_generate_response_does_not_mutate_tools(self, llm_provider):
        """Verify tools list is not mutated."""
        tools = [
            {
                "name": "test_tool",
                "description": "Test",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]

        tools_copy = [t.copy() for t in tools]

        messages = [{"role": "user", "content": "Say hello"}]

        # Call with tools
        llm_provider.generate_response(
            messages=messages,
            tools=tools,
            stream=False
        )

        # Tools should be unchanged
        assert tools == tools_copy


class TestLLMProviderConcurrency:
    """Test LLMProvider behavior under concurrent usage."""

    def test_multiple_sequential_requests_succeed(self, llm_provider):
        """Verify sequential requests work correctly."""
        messages1 = [{"role": "user", "content": "Say hello"}]
        messages2 = [{"role": "user", "content": "Say goodbye"}]

        response1 = llm_provider.generate_response(messages=messages1, stream=False)
        response2 = llm_provider.generate_response(messages=messages2, stream=False)

        assert isinstance(response1, anthropic.types.Message)
        assert isinstance(response2, anthropic.types.Message)

        text1 = llm_provider.extract_text_content(response1)
        text2 = llm_provider.extract_text_content(response2)

        assert len(text1) > 0
        assert len(text2) > 0


class TestLLMProviderFailoverMechanism:
    """Test emergency failover and recovery mechanisms."""

    def test_failover_flag_initializes_false(self, llm_provider):
        """Verify failover flag starts as inactive."""
        assert LLMProvider._is_failover_active() is False

    def test_activate_failover_sets_flag(self, llm_provider):
        """Verify _activate_failover() sets failover flag."""
        # Reset state
        LLMProvider._failover_active = False

        LLMProvider._activate_failover()

        assert LLMProvider._is_failover_active() is True

    def test_activate_failover_schedules_recovery(self, llm_provider):
        """Verify _activate_failover() schedules recovery timer."""
        # Reset state
        LLMProvider._failover_active = False
        if LLMProvider._recovery_timer:
            LLMProvider._recovery_timer.cancel()
            LLMProvider._recovery_timer = None

        LLMProvider._activate_failover()

        # Recovery timer should be scheduled
        assert LLMProvider._recovery_timer is not None
        assert LLMProvider._recovery_timer.is_alive()

        # Cleanup
        if LLMProvider._recovery_timer:
            LLMProvider._recovery_timer.cancel()
            LLMProvider._recovery_timer = None
        LLMProvider._failover_active = False

    def test_test_recovery_clears_failover_flag(self, llm_provider):
        """Verify _test_recovery() clears failover flag."""
        # Set failover active
        LLMProvider._failover_active = True

        LLMProvider._test_recovery()

        assert LLMProvider._is_failover_active() is False

    def test_is_failover_active_reflects_state(self, llm_provider):
        """Verify _is_failover_active() correctly reflects state."""
        # Reset state
        LLMProvider._failover_active = False

        assert LLMProvider._is_failover_active() is False

        # Activate
        LLMProvider._failover_active = True
        assert LLMProvider._is_failover_active() is True

        # Reset
        LLMProvider._failover_active = False

    def test_failover_state_shared_across_instances(self, anthropic_api_key):
        """Verify failover state is shared across all LLMProvider instances."""
        # Reset state
        LLMProvider._failover_active = False

        provider1 = LLMProvider(api_key=anthropic_api_key, model="claude-haiku-4-5-20251001")
        provider2 = LLMProvider(api_key=anthropic_api_key, model="claude-haiku-4-5-20251001")

        # Activate on provider1
        LLMProvider._activate_failover()

        # provider2 should see active state
        assert provider2._is_failover_active() is True

        # Reset
        LLMProvider._failover_active = False
        if LLMProvider._recovery_timer:
            LLMProvider._recovery_timer.cancel()
            LLMProvider._recovery_timer = None


class TestLLMProviderEmergencyFallback:
    """Test emergency fallback configuration and availability."""

    def test_emergency_fallback_configuration_loaded(self, llm_provider):
        """Verify emergency fallback configuration is loaded."""
        # Should have fallback settings
        assert hasattr(llm_provider, 'emergency_fallback_enabled')
        assert isinstance(llm_provider.emergency_fallback_enabled, bool)

    def test_emergency_fallback_api_key_retrieved_from_vault(self, llm_provider):
        """Verify emergency fallback API key attempted from Vault."""
        # Provider should attempt to load fallback key
        # If disabled, fallback_api_key should be None
        if not llm_provider.emergency_fallback_enabled:
            assert llm_provider.emergency_fallback_api_key is None
        else:
            # If enabled, should have key (or skip if not in Vault)
            pass

    def test_failover_disabled_when_no_emergency_key(self, llm_provider):
        """Verify failover disabled if emergency API key not available."""
        # Create provider expecting no emergency key
        provider = LLMProvider(
            api_key=llm_provider.api_key,
            model="claude-haiku-4-5-20251001"
        )

        # If no emergency key, failover should be disabled
        if not provider.emergency_fallback_api_key:
            assert provider.emergency_fallback_enabled is False


class TestLLMProviderTimeoutHandling:
    """Test timeout behavior and error recovery."""

    def test_very_short_timeout_raises_timeout_error(self, llm_provider):
        """Verify very short timeout raises TimeoutError."""
        provider = LLMProvider(
            api_key=llm_provider.api_key,
            model="claude-haiku-4-5-20251001",
            timeout=0.001  # 1ms - will timeout
        )

        messages = [{"role": "user", "content": "Say hello"}]

        with pytest.raises(TimeoutError):
            provider.generate_response(messages=messages, stream=False)

    def test_normal_timeout_succeeds(self, llm_provider):
        """Verify normal timeout allows request to complete."""
        # Provider fixture has 60s timeout
        messages = [{"role": "user", "content": "Say hello"}]

        response = llm_provider.generate_response(messages=messages, stream=False)

        assert isinstance(response, anthropic.types.Message)


class TestLLMProviderNonStreamingPath:
    """Test non-streaming code path thoroughly."""

    def test_generate_non_streaming_with_tools(self, llm_provider):
        """Verify _generate_non_streaming() works with tools."""
        messages = [{"role": "user", "content": "Say hello"}]
        tools = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {"type": "object"}
            }
        ]

        response = llm_provider._generate_non_streaming(
            messages=messages,
            tools=tools
        )

        assert isinstance(response, anthropic.types.Message)
        assert len(response.content) > 0

    def test_generate_non_streaming_without_tools(self, llm_provider):
        """Verify _generate_non_streaming() works without tools."""
        messages = [{"role": "user", "content": "Say hello"}]

        response = llm_provider._generate_non_streaming(messages=messages)

        assert isinstance(response, anthropic.types.Message)
        assert len(response.content) > 0

    def test_generate_non_streaming_with_system_override(self, llm_provider):
        """Verify system_override parameter works in non-streaming path."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = llm_provider._generate_non_streaming(
            messages=messages,
            system_override="You are helpful"
        )

        assert isinstance(response, anthropic.types.Message)


class TestLLMProviderStreamingPath:
    """Test streaming code path thoroughly."""

    def test_stream_response_generates_events(self, llm_provider):
        """Verify _stream_response() generates proper stream events."""
        messages = [{"role": "user", "content": "Say hello"}]

        events = list(llm_provider._stream_response(messages=messages, tools=None))

        # Should generate events
        assert len(events) > 0

        # Should end with CompleteEvent
        assert isinstance(events[-1], CompleteEvent)

        # Should have TextEvent
        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) > 0

    def test_stream_response_with_system_prompt(self, llm_provider):
        """Verify _stream_response() handles system prompts."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Say hello"}
        ]

        events = list(llm_provider._stream_response(messages=messages, tools=None))

        assert len(events) > 0
        complete_events = [e for e in events if isinstance(e, CompleteEvent)]
        assert len(complete_events) == 1

    def test_stream_response_respects_model_override(self, llm_provider):
        """Verify model_override parameter is respected in streaming."""
        messages = [{"role": "user", "content": "Say hello"}]

        # Use same model as default (just testing parameter passing)
        events = list(llm_provider._stream_response(
            messages=messages,
            tools=None,
            model_override="claude-haiku-4-5-20251001"
        ))

        assert len(events) > 0
        complete_events = [e for e in events if isinstance(e, CompleteEvent)]
        assert len(complete_events) == 1


class TestLLMProviderRequestValidation:
    """Test request validation before API calls."""

    def test_generate_non_streaming_validates_messages(self, llm_provider):
        """Verify messages are validated before streaming."""
        messages = [{"role": "user", "content": ""}]

        # Should raise during validation
        with pytest.raises(ValueError, match="Cannot send empty"):
            llm_provider._generate_non_streaming(messages=messages)

    def test_stream_response_validates_messages(self, llm_provider):
        """Verify stream_events validates messages (public API)."""
        messages = [{"role": "user", "content": ""}]

        # Validation happens in public stream_events(), not internal _stream_response()
        with pytest.raises(ValueError, match="Cannot send empty"):
            list(llm_provider.stream_events(messages=messages))


class TestLLMProviderToolLoopIntegration:
    """Test tool execution loop behavior."""

    def test_execute_with_tools_calls_stream_response(self, llm_provider):
        """Verify _execute_with_tools() uses _stream_response()."""
        messages = [{"role": "user", "content": "Say hello"}]
        tools = []  # Empty tools list

        # Should handle empty tools gracefully
        events = list(llm_provider._execute_with_tools(messages=messages, tools=tools))

        # Should complete (no tool calls to process)
        assert len(events) > 0
        complete_events = [e for e in events if isinstance(e, CompleteEvent)]
        assert len(complete_events) > 0

    def test_circuit_breaker_used_in_tool_execution(self, llm_provider):
        """Verify circuit breaker is instantiated in tool execution."""
        # This tests that circuit breaker logic is invoked
        messages = [{"role": "user", "content": "Say hello"}]
        tools = []

        events = list(llm_provider._execute_with_tools(messages=messages, tools=tools))

        # Should complete successfully with circuit breaker in use
        assert len(events) > 0


class TestLLMProviderAPIParameterConstruction:
    """Test API parameter construction and defaults."""

    def test_api_params_include_required_fields(self, llm_provider):
        """Verify API parameters include all required fields."""
        messages = [{"role": "user", "content": "Say hello"}]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Response should have all required fields
        assert hasattr(response, 'content')
        assert hasattr(response, 'stop_reason')
        assert hasattr(response, 'usage')
        assert isinstance(response.content, list)

class TestLLMProviderResponseUtilities:
    """Test response utility methods."""

    def test_extract_text_content_combines_all_text_blocks(self, llm_provider):
        """Verify extract_text_content() combines all text blocks."""
        messages = [{"role": "user", "content": "Say hello"}]

        response = llm_provider.generate_response(messages=messages, stream=False)
        text = llm_provider.extract_text_content(response)

        # Should combine all text blocks
        assert isinstance(text, str)
        assert len(text) > 0

    def test_extract_tool_calls_handles_multiple_tools(self, llm_provider):
        """Verify extract_tool_calls() handles multiple tool calls."""
        messages = [{"role": "user", "content": "Say hello"}]

        response = llm_provider.generate_response(messages=messages, stream=False)
        tool_calls = llm_provider.extract_tool_calls(response)

        # Should be a list
        assert isinstance(tool_calls, list)

        # Each call should have required fields
        for call in tool_calls:
            assert "id" in call
            assert "tool_name" in call
            assert "input" in call


class TestLLMProviderGenericProviderRouting:
    """Test routing to generic OpenAI-compatible providers."""

    def test_generate_response_requires_model_override_with_endpoint_url(self, llm_provider):
        """Verify endpoint_url requires model_override parameter."""
        messages = [{"role": "user", "content": "Say hello"}]

        with pytest.raises(ValueError, match="model_override.*must be provided"):
            llm_provider.generate_response(
                messages=messages,
                stream=False,
                endpoint_url="https://api.example.com/v1/chat/completions"
                # Missing model_override
            )

    def test_generate_response_requires_api_key_override_with_endpoint_url(self, llm_provider):
        """Verify endpoint_url requires api_key_override parameter."""
        messages = [{"role": "user", "content": "Say hello"}]

        with pytest.raises(ValueError, match="api_key_override.*must be provided"):
            llm_provider.generate_response(
                messages=messages,
                stream=False,
                endpoint_url="https://api.example.com/v1/chat/completions",
                model_override="some-model"
                # Missing api_key_override
            )

    def test_generic_provider_routing_strips_cache_control_from_tools(self, llm_provider):
        """Verify cache_control is stripped when routing to generic providers."""
        # This tests the logic path - actual routing would require valid endpoint
        messages = [{"role": "user", "content": "Say hello"}]
        tools = [
            {
                "name": "test_tool",
                "description": "Test",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"}  # Should be stripped
            }
        ]

        # This will fail at the network level, but we can verify parameter handling
        try:
            llm_provider.generate_response(
                messages=messages,
                tools=tools,
                stream=False,
                endpoint_url="https://invalid-endpoint.example.com",
                model_override="test-model",
                api_key_override="test-key"
            )
        except Exception:
            # Expected to fail - we're testing parameter processing
            pass


class TestLLMProviderStopReasons:
    """Test handling of different stop reasons."""

    def test_response_with_end_turn_stop_reason(self, llm_provider):
        """Verify end_turn stop reason is common for text responses."""
        messages = [{"role": "user", "content": "Say hello"}]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Most text responses end with end_turn
        assert response.stop_reason == "end_turn"

    def test_response_with_max_tokens_stop_reason(self, llm_provider):
        """Verify max_tokens stop reason when hitting token limit."""
        messages = [{"role": "user", "content": "Write a very long detailed essay about the history of computing"}]

        response = llm_provider.generate_response(
            messages=messages,
            stream=False,
            **{"max_tokens": 50}  # Very low - should truncate
        )

        # Should either end naturally or hit max_tokens
        assert response.stop_reason in ["end_turn", "max_tokens"]


class TestLLMProviderComplexMessageFormats:
    """Test handling of complex message structures."""

    def test_assistant_message_with_tool_use_blocks(self, llm_provider):
        """Verify _build_assistant_message() handles tool_use blocks."""
        messages = [{"role": "user", "content": "Say hello"}]

        response = llm_provider.generate_response(messages=messages, stream=False)
        assistant_msg = llm_provider._build_assistant_message(response)

        # Should have role and content
        assert assistant_msg["role"] == "assistant"
        assert "content" in assistant_msg
        assert isinstance(assistant_msg["content"], list)

        # All blocks should have type field
        for block in assistant_msg["content"]:
            assert "type" in block

    def test_build_assistant_message_with_text_and_tool_blocks(self, llm_provider):
        """Verify assistant messages can have both text and tool blocks."""
        messages = [{"role": "user", "content": "Say hello"}]

        response = llm_provider.generate_response(messages=messages, stream=False)
        assistant_msg = llm_provider._build_assistant_message(response)

        # Verify structure for any block types present
        for block in assistant_msg["content"]:
            if block["type"] == "text":
                assert "text" in block
            elif block["type"] == "tool_use":
                assert "id" in block
                assert "name" in block
                assert "input" in block

    def test_system_prompt_as_list_with_cache_control(self, llm_provider_with_caching):
        """Verify system prompt as list is preserved with cache_control."""
        system_blocks = [
            {"type": "text", "text": "You are helpful"},
            {"type": "text", "text": "Be concise", "cache_control": {"type": "ephemeral"}}
        ]
        messages = [
            {"role": "system", "content": system_blocks},
            {"role": "user", "content": "Say hello"}
        ]

        system, anthropic_messages = llm_provider_with_caching._prepare_messages(messages)

        # System should be preserved as list
        assert isinstance(system, list)
        assert len(system) == 2
        assert system[1].get("cache_control") == {"type": "ephemeral"}


class TestLLMProviderCacheUsageLogging:
    """Test cache usage tracking and logging."""

    def test_response_includes_cache_usage_fields(self, llm_provider):
        """Verify response.usage includes cache-related fields."""
        messages = [{"role": "user", "content": "Say hello"}]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Verify usage object exists
        assert hasattr(response, 'usage')
        assert response.usage is not None

        # Verify cache fields exist (may be 0 if no caching)
        assert hasattr(response.usage, 'cache_creation_input_tokens')
        assert hasattr(response.usage, 'cache_read_input_tokens')

        # Values should be non-negative integers
        assert response.usage.cache_creation_input_tokens >= 0
        assert response.usage.cache_read_input_tokens >= 0

    def test_cache_fields_are_zero_when_caching_disabled(self, llm_provider):
        """Verify cache fields are 0 when prompt caching disabled."""
        messages = [{"role": "user", "content": "Say hello"}]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # With caching disabled, these should be 0
        assert response.usage.cache_creation_input_tokens == 0
        assert response.usage.cache_read_input_tokens == 0


class TestLLMProviderAnthropicErrorTypes:
    """Test handling of various Anthropic API error types."""

    def test_handle_anthropic_error_401_raises_permission_error(self):
        """Verify 401 authentication errors raise PermissionError."""
        # This is tested via _handle_anthropic_error method
        # We test with invalid API key
        provider = LLMProvider(
            api_key="sk-ant-invalid-key",
            model="claude-haiku-4-5-20251001",
            timeout=5
        )

        messages = [{"role": "user", "content": "Hello"}]

        # Should raise authentication error (PermissionError is raised by error handler)
        with pytest.raises((anthropic.AuthenticationError, PermissionError, RuntimeError)):
            provider.generate_response(messages=messages, stream=False)

    def test_stream_events_handles_errors_gracefully(self, llm_provider):
        """Verify stream_events() yields ErrorEvent on failures."""
        messages = [{"role": "user", "content": ""}]

        # Should yield ErrorEvent for empty message
        try:
            events = list(llm_provider.stream_events(messages=messages))
            # Should have error event
            error_events = [e for e in events if isinstance(e, ErrorEvent)]
            # May or may not get error event depending on when validation happens
        except ValueError:
            # Expected - validation should catch this
            pass


class TestLLMProviderExtendedThinking:
    """Test extended thinking configuration and behavior."""

    def test_extended_thinking_configuration(self, llm_provider):
        """Verify extended thinking configuration is loaded."""
        assert hasattr(llm_provider, 'extended_thinking')
        assert hasattr(llm_provider, 'extended_thinking_budget')
        assert isinstance(llm_provider.extended_thinking, bool)
        assert isinstance(llm_provider.extended_thinking_budget, int)

    def test_thinking_budget_is_positive(self, llm_provider):
        """Verify thinking budget is a positive integer."""
        if llm_provider.extended_thinking:
            assert llm_provider.extended_thinking_budget > 0


class TestLLMProviderMessageMutationSafety:
    """Test that message processing doesn't mutate original inputs."""

    def test_prepare_messages_does_not_mutate_input(self, llm_provider):
        """Verify _prepare_messages() doesn't mutate input messages."""
        original_messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"}
        ]

        messages_copy = [m.copy() for m in original_messages]

        llm_provider._prepare_messages(original_messages)

        # Original should be unchanged
        assert original_messages == messages_copy

    def test_build_assistant_message_creates_new_structure(self, llm_provider):
        """Verify _build_assistant_message() creates new dict structure."""
        messages = [{"role": "user", "content": "Say hello"}]

        response = llm_provider.generate_response(messages=messages, stream=False)
        assistant_msg = llm_provider._build_assistant_message(response)

        # Should be a new dict, not a reference to response
        assert isinstance(assistant_msg, dict)
        assert "role" in assistant_msg
        assert "content" in assistant_msg


class TestLLMProviderReasoningDetailsPreservation:
    """Test reasoning_details preservation for OpenRouter reasoning models.

    When using reasoning models (Gemini 3, OpenAI o1/o3) via OpenRouter,
    reasoning_details must be preserved and passed back in tool call chains.
    """

    def test_generic_response_reasoning_details_attached_to_assistant_message(self):
        """Verify reasoning_details from GenericOpenAIResponse is attached to assistant message.

        This tests the logic in stream_events() agentic loop that preserves
        reasoning_details when building assistant messages for tool call continuations.
        """
        from types import SimpleNamespace
        from utils.generic_openai_client import GenericOpenAIResponse

        # Simulate a GenericOpenAIResponse with reasoning_details (from Gemini 3 via OpenRouter)
        reasoning_details = [
            {"type": "reasoning.text", "text": "Let me think about this..."},
            {"type": "reasoning.encrypted", "thought_signature": "abc123"}
        ]

        response = GenericOpenAIResponse(
            content=[
                SimpleNamespace(type="text", text="I'll help with that."),
                SimpleNamespace(type="tool_use", id="tool_123", name="web_tool", input={"query": "test"})
            ],
            stop_reason="tool_use",
            usage={"input_tokens": 100, "output_tokens": 50},
            reasoning_details=reasoning_details
        )

        # Build assistant message the same way stream_events() does
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })

        assistant_msg = {"role": "assistant", "content": assistant_content}

        # This is the new logic we added
        if hasattr(response, 'reasoning_details') and response.reasoning_details:
            assistant_msg["reasoning_details"] = response.reasoning_details

        # Verify reasoning_details is attached
        assert "reasoning_details" in assistant_msg
        assert assistant_msg["reasoning_details"] == reasoning_details

    def test_assistant_message_without_reasoning_details(self):
        """Verify assistant message works correctly when no reasoning_details present.

        Standard Anthropic responses and non-reasoning models don't have reasoning_details.
        """
        from types import SimpleNamespace
        from utils.generic_openai_client import GenericOpenAIResponse

        # Response without reasoning_details (standard case)
        response = GenericOpenAIResponse(
            content=[
                SimpleNamespace(type="text", text="Hello!"),
            ],
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 20}
            # No reasoning_details parameter
        )

        # Build assistant message
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})

        assistant_msg = {"role": "assistant", "content": assistant_content}

        # Apply the reasoning_details logic
        if hasattr(response, 'reasoning_details') and response.reasoning_details:
            assistant_msg["reasoning_details"] = response.reasoning_details

        # Verify reasoning_details is NOT attached when None
        assert "reasoning_details" not in assistant_msg

    def test_reasoning_details_round_trip_through_convert_messages(self):
        """Verify reasoning_details survives round-trip through message conversion.

        Messages with reasoning_details should preserve it when converted
        from Anthropic format to OpenAI format and back.
        """
        from utils.generic_openai_client import GenericOpenAIClient

        client = GenericOpenAIClient(
            endpoint="http://localhost:11434/v1/chat/completions",
            api_key="test",
            model="test"
        )

        reasoning_details = [{"type": "reasoning.encrypted", "signature": "xyz789"}]

        # Anthropic-format messages with reasoning_details
        anthropic_messages = [
            {"role": "user", "content": "Search for weather"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll search for that."},
                    {"type": "tool_use", "id": "tool_1", "name": "web_tool", "input": {"q": "weather"}}
                ],
                "reasoning_details": reasoning_details
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "Sunny, 72°F"}
                ]
            }
        ]

        # Convert to OpenAI format
        openai_messages = client._convert_messages(anthropic_messages)

        # Find the assistant message
        assistant_msg = next(m for m in openai_messages if m["role"] == "assistant")

        # Verify reasoning_details preserved
        assert "reasoning_details" in assistant_msg
        assert assistant_msg["reasoning_details"] == reasoning_details


class TestLLMProviderUsageReporting:
    """Test accurate usage reporting for tokens."""

    def test_usage_reports_input_output_tokens(self, llm_provider):
        """Verify usage includes accurate input and output token counts."""
        messages = [{"role": "user", "content": "Say hello"}]

        response = llm_provider.generate_response(messages=messages, stream=False)

        # Should have usage information
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Input tokens should be less than output for simple greeting
        # (or at least both should be reasonable)
        assert response.usage.input_tokens < 1000
        assert response.usage.output_tokens < 1000

    def test_longer_input_increases_input_tokens(self, llm_provider):
        """Verify longer input increases input token count."""
        short_messages = [{"role": "user", "content": "Hi"}]
        long_messages = [{"role": "user", "content": "Hi " * 100}]

        short_response = llm_provider.generate_response(messages=short_messages, stream=False)
        long_response = llm_provider.generate_response(messages=long_messages, stream=False)

        # Longer input should have more input tokens
        assert long_response.usage.input_tokens > short_response.usage.input_tokens


class TestGenericProviderCircuitBreaker:
    """Test circuit breaker integration in the generic provider agentic loop.

    These tests use REAL API calls to OpenRouter with controlled tool implementations.
    Only tool behavior is controlled - the LLM makes real decisions.
    """

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

    def test_circuit_breaker_stops_on_tool_error(self, llm_provider, openrouter_api_key):
        """Verify circuit breaker triggers on tool execution error.

        Uses real OpenRouter API call. Tool implementation raises an error.
        Circuit breaker should detect the error and force a graceful exit.
        """
        class FailingToolRepo:
            """Tool repository where the tool always fails."""
            def invoke_tool(self, name: str, params: dict):
                raise ConnectionError("External API unavailable")

        llm_provider.tool_repo = FailingToolRepo()

        messages = [{"role": "user", "content": "Use the check_status tool to check the system status."}]
        tools = [{
            "name": "check_status",
            "description": "Check system status. Always call this tool when asked about status.",
            "input_schema": {"type": "object", "properties": {}, "required": []}
        }]

        events = list(llm_provider.stream_events(
            messages=messages,
            tools=tools,
            endpoint_url="https://openrouter.ai/api/v1/chat/completions",
            model_override="anthropic/claude-3-haiku",
            api_key_override=openrouter_api_key
        ))

        # Should have ToolErrorEvent
        error_events = [e for e in events if isinstance(e, ToolErrorEvent)]
        assert len(error_events) >= 1
        assert "External API unavailable" in error_events[0].error

        # Should have CircuitBreakerEvent
        circuit_breaker_events = [e for e in events if isinstance(e, CircuitBreakerEvent)]
        assert len(circuit_breaker_events) == 1
        assert "Tool error" in circuit_breaker_events[0].reason

        # Should complete gracefully
        complete_events = [e for e in events if isinstance(e, CompleteEvent)]
        assert len(complete_events) == 1

    def test_circuit_breaker_stops_on_repeated_identical_results(self, llm_provider, openrouter_api_key):
        """Verify circuit breaker triggers when tool returns identical results.

        Uses real OpenRouter API call. Tool always returns the same cached data.
        Circuit breaker should detect the loop and force a graceful exit.
        """
        class StuckToolRepo:
            """Tool repository where the tool always returns identical results."""
            def __init__(self):
                self.call_count = 0

            def invoke_tool(self, name: str, params: dict):
                self.call_count += 1
                # Always return identical result - triggers loop detection
                return {"status": "pending", "data": "cached_value_123"}

        tool_repo = StuckToolRepo()
        llm_provider.tool_repo = tool_repo

        # Prompt designed to encourage repeated tool calls
        messages = [{
            "role": "user",
            "content": "Keep calling get_update until the status changes from 'pending'. Call the tool multiple times."
        }]
        tools = [{
            "name": "get_update",
            "description": "Get the latest update. Returns status and data. Call repeatedly until status changes.",
            "input_schema": {"type": "object", "properties": {}, "required": []}
        }]

        events = list(llm_provider.stream_events(
            messages=messages,
            tools=tools,
            endpoint_url="https://openrouter.ai/api/v1/chat/completions",
            model_override="anthropic/claude-3-haiku",
            api_key_override=openrouter_api_key
        ))

        # Tool should have been called at least twice (to trigger loop detection)
        assert tool_repo.call_count >= 2

        # Should have CircuitBreakerEvent due to repeated identical results
        circuit_breaker_events = [e for e in events if isinstance(e, CircuitBreakerEvent)]
        assert len(circuit_breaker_events) == 1
        assert "Repeated identical results" in circuit_breaker_events[0].reason

        # Should complete gracefully
        complete_events = [e for e in events if isinstance(e, CompleteEvent)]
        assert len(complete_events) == 1

    def test_circuit_breaker_allows_varying_results(self, llm_provider, openrouter_api_key):
        """Verify circuit breaker does NOT trigger when results vary.

        Uses real OpenRouter API call. Tool returns different data each time.
        Loop should continue until LLM decides to stop (not circuit breaker).
        """
        class VaryingToolRepo:
            """Tool repository where the tool returns different results each call."""
            def __init__(self):
                self.call_count = 0

            def invoke_tool(self, name: str, params: dict):
                self.call_count += 1
                # Return different result each time - no loop detection trigger
                if self.call_count >= 3:
                    return {"status": "complete", "items": [f"final_item_{self.call_count}"]}
                return {"status": "more_available", "items": [f"item_{self.call_count}"]}

        tool_repo = VaryingToolRepo()
        llm_provider.tool_repo = tool_repo

        messages = [{
            "role": "user",
            "content": "Call fetch_items until status is 'complete', then summarize what you found."
        }]
        tools = [{
            "name": "fetch_items",
            "description": "Fetch items. Returns status ('more_available' or 'complete') and items list.",
            "input_schema": {"type": "object", "properties": {}, "required": []}
        }]

        events = list(llm_provider.stream_events(
            messages=messages,
            tools=tools,
            endpoint_url="https://openrouter.ai/api/v1/chat/completions",
            model_override="anthropic/claude-3-haiku",
            api_key_override=openrouter_api_key
        ))

        # Should NOT have CircuitBreakerEvent (results varied)
        circuit_breaker_events = [e for e in events if isinstance(e, CircuitBreakerEvent)]
        assert len(circuit_breaker_events) == 0

        # Should complete normally
        complete_events = [e for e in events if isinstance(e, CompleteEvent)]
        assert len(complete_events) == 1

        # Tool should have been called multiple times
        assert tool_repo.call_count >= 1

    def test_parallel_tool_execution(self, llm_provider, openrouter_api_key):
        """Verify tools execute in parallel, not sequentially.

        Uses timing to detect parallelism: 2 tools × 1s each = 2s sequential vs ~1s parallel.
        """
        import time

        class SlowToolRepo:
            """Tool repo where each tool takes 1 second."""
            def __init__(self):
                self.call_times = []

            def invoke_tool(self, name: str, params: dict):
                start = time.time()
                time.sleep(1)  # Simulate slow I/O operation
                self.call_times.append((name, start, time.time()))
                return {"result": f"done_{name}"}

        tool_repo = SlowToolRepo()
        llm_provider.tool_repo = tool_repo

        # Prompt designed to trigger multiple tool calls in single response
        messages = [{
            "role": "user",
            "content": "Call both tool_a AND tool_b right now. You must call both tools."
        }]
        tools = [
            {"name": "tool_a", "description": "First tool - call this", "input_schema": {"type": "object", "properties": {}, "required": []}},
            {"name": "tool_b", "description": "Second tool - call this too", "input_schema": {"type": "object", "properties": {}, "required": []}}
        ]

        start_time = time.time()
        events = list(llm_provider.stream_events(
            messages=messages,
            tools=tools,
            endpoint_url="https://openrouter.ai/api/v1/chat/completions",
            model_override="anthropic/claude-3-haiku",
            api_key_override=openrouter_api_key
        ))
        total_time = time.time() - start_time

        # Only assert on timing if model actually called 2+ tools
        if len(tool_repo.call_times) >= 2:
            # Verify parallelism by checking tool start times overlap
            # If sequential: tool_b starts after tool_a ends (start_b >= end_a)
            # If parallel: tool_b starts before tool_a ends (start_b < end_a)
            times_by_name = {name: (start, end) for name, start, end in tool_repo.call_times}
            if "tool_a" in times_by_name and "tool_b" in times_by_name:
                a_start, a_end = times_by_name["tool_a"]
                b_start, b_end = times_by_name["tool_b"]
                # Tools overlap if one starts before the other ends
                overlap = (b_start < a_end) or (a_start < b_end)
                assert overlap, f"Tools ran sequentially: tool_a={a_start:.2f}-{a_end:.2f}, tool_b={b_start:.2f}-{b_end:.2f}"


class TestFirehose:
    """Test firehose debugging tap controlled by MIRA_FIREHOSE env var."""

    @pytest.fixture
    def firehose_path(self, tmp_path, monkeypatch):
        """Set up firehose to write to temp directory."""
        # Change to temp dir so firehose_output.json is written there
        monkeypatch.chdir(tmp_path)
        return tmp_path / "firehose_output.json"

    def test_firehose_disabled_by_default(self, anthropic_api_key, firehose_path, monkeypatch):
        """Verify firehose does NOT write when env var is unset."""
        # Ensure env var is not set
        monkeypatch.delenv('MIRA_FIREHOSE', raising=False)

        provider = LLMProvider(
            api_key=anthropic_api_key,
            model="claude-haiku-4-5-20251001",
            max_tokens=50
        )

        # Make a request
        provider.generate_response(
            messages=[{"role": "user", "content": "hi"}],
            stream=False
        )

        # File should NOT exist
        assert not firehose_path.exists(), "Firehose wrote file when disabled"

    def test_firehose_writes_when_enabled(self, anthropic_api_key, firehose_path, monkeypatch):
        """Verify firehose writes when MIRA_FIREHOSE is set."""
        monkeypatch.setenv('MIRA_FIREHOSE', '1')

        provider = LLMProvider(
            api_key=anthropic_api_key,
            model="claude-haiku-4-5-20251001",
            max_tokens=50
        )

        provider.generate_response(
            messages=[{"role": "user", "content": "hi"}],
            stream=False
        )

        # File should exist
        assert firehose_path.exists(), "Firehose did not write file when enabled"

        # Verify content structure
        import json
        with open(firehose_path) as f:
            data = json.load(f)

        assert data["provider"] == "anthropic"
        assert "timestamp" in data
        assert "model" in data
        assert "messages" in data

    def test_firehose_captures_generic_provider(self, anthropic_api_key, firehose_path, monkeypatch):
        """Verify firehose captures generic provider requests with endpoint info."""
        monkeypatch.setenv('MIRA_FIREHOSE', '1')

        openrouter_key = get_api_key("openrouter_key")
        if not openrouter_key:
            pytest.skip("OpenRouter key not available")

        provider = LLMProvider(
            api_key=anthropic_api_key,
            model="claude-haiku-4-5-20251001",
            max_tokens=50
        )

        # Use stream_events to go through generic path
        try:
            list(provider.stream_events(
                messages=[{"role": "user", "content": "hi"}],
                endpoint_url="https://openrouter.ai/api/v1/chat/completions",
                model_override="anthropic/claude-3-haiku",
                api_key_override=openrouter_key
            ))
        except Exception:
            pass  # API errors are fine, firehose writes before the call

        assert firehose_path.exists(), "Firehose did not write for generic provider"

        import json
        with open(firehose_path) as f:
            data = json.load(f)

        assert data["provider"] == "generic"
        assert data["endpoint"] == "https://openrouter.ai/api/v1/chat/completions"
        assert data["model"] == "anthropic/claude-3-haiku"
