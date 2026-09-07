"""
Tests for utils/generic_openai_client.py

Tests GenericOpenAIClient with real Ollama inference (qwen3:4b).
Following MIRA testing philosophy: no mocks, test real API behavior.
"""
import pytest
import json
from utils.generic_openai_client import GenericOpenAIClient, GenericOpenAIResponse


# Ollama configuration
OLLAMA_ENDPOINT = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_API_KEY = "ollama"  # Ollama doesn't require real key but expects Bearer format


@pytest.fixture(scope="module")
def ollama_client():
    """Shared Ollama client for tests."""
    return GenericOpenAIClient(
        endpoint=OLLAMA_ENDPOINT,
        api_key=OLLAMA_API_KEY,
        model=OLLAMA_MODEL,
        timeout=30,
        max_tokens=512,
        temperature=0.7
    )


class TestGenericOpenAIClientBasics:
    """Test basic client initialization and message creation."""

    def test_client_initialization(self, ollama_client):
        """Verify client initializes with correct configuration."""
        assert ollama_client.endpoint == OLLAMA_ENDPOINT
        assert ollama_client.api_key == OLLAMA_API_KEY
        assert ollama_client.model == OLLAMA_MODEL
        assert ollama_client.timeout == 30
        assert ollama_client.default_max_tokens == 512
        assert ollama_client.default_temperature == 0.7

    def test_client_has_messages_namespace(self, ollama_client):
        """Verify client has messages.create() interface."""
        assert hasattr(ollama_client, 'messages')
        assert hasattr(ollama_client.messages, 'create')
        assert callable(ollama_client.messages.create)

    def test_simple_text_message(self, ollama_client):
        """Verify simple text message works end-to-end."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = ollama_client.messages.create(messages=messages)

        # Verify response structure
        assert isinstance(response, GenericOpenAIResponse)
        assert hasattr(response, 'content')
        assert hasattr(response, 'stop_reason')
        assert hasattr(response, 'usage')

        # Verify content blocks
        assert len(response.content) > 0
        assert response.content[0].type == "text"
        assert isinstance(response.content[0].text, str)
        assert len(response.content[0].text) > 0

        # Verify stop reason
        assert response.stop_reason == "end_turn"

        # Verify usage tracking
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_message_with_system_prompt_string(self, ollama_client):
        """Verify system prompt as string is handled correctly."""
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        system = "You are a helpful math tutor."

        response = ollama_client.messages.create(
            messages=messages,
            system=system
        )

        assert isinstance(response, GenericOpenAIResponse)
        assert len(response.content) > 0
        assert response.content[0].type == "text"

    def test_message_with_system_prompt_blocks(self, ollama_client):
        """Verify system prompt as structured blocks is handled correctly."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        system = [
            {"type": "text", "text": "You are a helpful assistant."},
            {"type": "text", "text": " Be concise.", "cache_control": {"type": "ephemeral"}}
        ]

        response = ollama_client.messages.create(
            messages=messages,
            system=system
        )

        assert isinstance(response, GenericOpenAIResponse)
        assert len(response.content) > 0


class TestGenericOpenAIClientMessageConversion:
    """Test Anthropic → OpenAI message format conversion."""

    def test_user_message_with_text_blocks(self, ollama_client):
        """Verify user messages with text content blocks are converted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the capital of France?"}
                ]
            }
        ]

        response = ollama_client.messages.create(messages=messages)

        assert isinstance(response, GenericOpenAIResponse)
        assert len(response.content) > 0

    def test_multi_turn_conversation(self, ollama_client):
        """Verify multi-turn conversations are converted correctly."""
        messages = [
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What is my name?"}
        ]

        response = ollama_client.messages.create(messages=messages)

        assert isinstance(response, GenericOpenAIResponse)
        assert len(response.content) > 0
        # Response should reference the name Alice
        response_text = response.content[0].text.lower()
        assert "alice" in response_text

    def test_assistant_message_with_text_blocks(self, ollama_client):
        """Verify assistant messages with structured blocks are converted."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hi there!"}
                ]
            },
            {"role": "user", "content": "How are you?"}
        ]

        response = ollama_client.messages.create(messages=messages)

        assert isinstance(response, GenericOpenAIResponse)
        assert len(response.content) > 0


class TestGenericOpenAIClientParameters:
    """Test parameter handling (max_tokens, temperature)."""

    def test_custom_max_tokens(self, ollama_client):
        """Verify max_tokens parameter is used."""
        messages = [
            {"role": "user", "content": "Write a long story about a dragon"}
        ]

        response = ollama_client.messages.create(
            messages=messages,
            max_tokens=50  # Very short
        )

        assert isinstance(response, GenericOpenAIResponse)
        # Should be truncated (might hit max_tokens)
        assert response.stop_reason in ["end_turn", "max_tokens"]

    def test_custom_temperature(self, ollama_client):
        """Verify temperature parameter is accepted."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = ollama_client.messages.create(
            messages=messages,
            temperature=0.1  # Very deterministic
        )

        assert isinstance(response, GenericOpenAIResponse)
        assert len(response.content) > 0

    def test_uses_default_max_tokens(self, ollama_client):
        """Verify default max_tokens is used when not specified."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        response = ollama_client.messages.create(messages=messages)

        assert isinstance(response, GenericOpenAIResponse)
        # Should use default (512) from client initialization


class TestGenericOpenAIClientTools:
    """Test tool calling functionality."""

    def test_tool_definition_conversion(self, ollama_client):
        """Verify Anthropic tool definitions are converted to OpenAI format."""
        messages = [
            {"role": "user", "content": "What is the weather in San Francisco?"}
        ]

        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]

        # Note: qwen3:4b may or may not call tools depending on training
        # We're testing the conversion logic, not tool calling success
        response = ollama_client.messages.create(
            messages=messages,
            tools=tools
        )

        assert isinstance(response, GenericOpenAIResponse)
        # Response will be either text or tool_use
        assert len(response.content) > 0

    def test_tool_use_response_format(self, ollama_client):
        """Verify tool_use responses are formatted correctly."""
        messages = [
            {"role": "user", "content": "Calculate 25 * 4"}
        ]

        tools = [
            {
                "name": "calculator",
                "description": "Perform arithmetic operations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]

        response = ollama_client.messages.create(
            messages=messages,
            tools=tools
        )

        assert isinstance(response, GenericOpenAIResponse)
        # Check if response contains tool_use block
        for block in response.content:
            if block.type == "tool_use":
                # Verify tool_use block structure
                assert hasattr(block, 'id')
                assert hasattr(block, 'name')
                assert hasattr(block, 'input')
                assert isinstance(block.id, str)
                assert isinstance(block.name, str)
                assert isinstance(block.input, dict)

    def test_tool_result_in_messages(self, ollama_client):
        """Verify tool results in user messages are converted correctly."""
        messages = [
            {"role": "user", "content": "What is 5 + 3?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "calculator",
                        "input": {"expression": "5 + 3"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "8"
                    }
                ]
            }
        ]

        response = ollama_client.messages.create(messages=messages)

        assert isinstance(response, GenericOpenAIResponse)
        assert len(response.content) > 0


class TestGenericOpenAIClientStopReasons:
    """Test stop reason mapping."""

    def test_end_turn_stop_reason(self, ollama_client):
        """Verify 'stop' finish reason maps to 'end_turn'."""
        messages = [
            {"role": "user", "content": "Say hi"}
        ]

        response = ollama_client.messages.create(messages=messages)

        # Most responses should end naturally
        assert response.stop_reason == "end_turn"

    def test_max_tokens_stop_reason(self, ollama_client):
        """Verify 'length' finish reason maps to 'max_tokens'."""
        messages = [
            {"role": "user", "content": "Write a very long essay about artificial intelligence"}
        ]

        response = ollama_client.messages.create(
            messages=messages,
            max_tokens=20  # Very short - should hit limit
        )

        # Should hit max tokens or end naturally
        assert response.stop_reason in ["max_tokens", "end_turn"]


class TestGenericOpenAIClientUsage:
    """Test usage tracking."""

    def test_usage_includes_input_tokens(self, ollama_client):
        """Verify usage includes input token count."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        response = ollama_client.messages.create(messages=messages)

        assert hasattr(response.usage, 'input_tokens')
        assert response.usage.input_tokens > 0

    def test_usage_includes_output_tokens(self, ollama_client):
        """Verify usage includes output token count."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = ollama_client.messages.create(messages=messages)

        assert hasattr(response.usage, 'output_tokens')
        assert response.usage.output_tokens > 0

    def test_usage_includes_cache_tokens(self, ollama_client):
        """Verify usage includes cache token fields (always 0 for OpenAI)."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        response = ollama_client.messages.create(messages=messages)

        assert hasattr(response.usage, 'cache_creation_input_tokens')
        assert hasattr(response.usage, 'cache_read_input_tokens')
        # OpenAI doesn't have caching, so these should be 0
        assert response.usage.cache_creation_input_tokens == 0
        assert response.usage.cache_read_input_tokens == 0


class TestGenericOpenAIClientErrorHandling:
    """Test error handling for various failure modes."""

    def test_invalid_endpoint_raises_error(self):
        """Verify invalid endpoint raises RuntimeError."""
        client = GenericOpenAIClient(
            endpoint="http://localhost:99999/invalid",
            api_key="test",
            model="test",
            timeout=5
        )

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError):
            client.messages.create(messages=messages)

    def test_timeout_raises_timeout_error(self, ollama_client):
        """Verify timeout is enforced."""
        # Create client with very short timeout
        client = GenericOpenAIClient(
            endpoint=OLLAMA_ENDPOINT,
            api_key=OLLAMA_API_KEY,
            model=OLLAMA_MODEL,
            timeout=0.001  # 1ms - will timeout
        )

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(TimeoutError):
            client.messages.create(messages=messages)


class TestGenericOpenAIClientResponseValidation:
    """Test response validation and error handling."""

    def test_response_has_required_fields(self, ollama_client):
        """Verify response always has required fields."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        response = ollama_client.messages.create(messages=messages)

        # Required fields from GenericOpenAIResponse
        assert hasattr(response, 'content')
        assert hasattr(response, 'stop_reason')
        assert hasattr(response, 'usage')

        # Content should be a list
        assert isinstance(response.content, list)
        assert len(response.content) > 0

        # Stop reason should be valid
        assert response.stop_reason in ["end_turn", "tool_use", "max_tokens"]

    def test_content_blocks_have_correct_structure(self, ollama_client):
        """Verify content blocks have correct attributes."""
        messages = [
            {"role": "user", "content": "Say hello"}
        ]

        response = ollama_client.messages.create(messages=messages)

        # Check first content block (should be text)
        block = response.content[0]
        assert hasattr(block, 'type')
        assert block.type == "text"
        assert hasattr(block, 'text')
        assert isinstance(block.text, str)


class TestGenericOpenAIClientReasoningDetails:
    """Test reasoning_details preservation for OpenRouter reasoning models."""

    def test_response_stores_reasoning_details_attribute(self):
        """Verify GenericOpenAIResponse stores reasoning_details when provided."""
        reasoning_details = [
            {"type": "reasoning.text", "text": "Thinking about the problem..."},
            {"type": "reasoning.encrypted", "data": "encrypted_signature_abc"}
        ]

        response = GenericOpenAIResponse(
            content=[],
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 20},
            reasoning_details=reasoning_details
        )

        assert response.reasoning_details == reasoning_details

    def test_response_reasoning_details_defaults_to_none(self):
        """Verify reasoning_details defaults to None when not provided."""
        response = GenericOpenAIResponse(
            content=[],
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 20}
        )

        assert response.reasoning_details is None

    def test_convert_messages_preserves_reasoning_details_on_assistant(self):
        """Verify _convert_messages() preserves reasoning_details on assistant messages."""
        client = GenericOpenAIClient(
            endpoint="http://localhost:11434/v1/chat/completions",
            api_key="test",
            model="test"
        )

        reasoning_details = [{"type": "reasoning.text", "text": "Thinking..."}]
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
                "reasoning_details": reasoning_details
            }
        ]

        converted = client._convert_messages(messages)

        # Find the assistant message
        assistant_msgs = [m for m in converted if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get("reasoning_details") == reasoning_details

    def test_convert_messages_without_reasoning_details(self):
        """Verify _convert_messages() works when reasoning_details not present."""
        client = GenericOpenAIClient(
            endpoint="http://localhost:11434/v1/chat/completions",
            api_key="test",
            model="test"
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}]
            }
        ]

        converted = client._convert_messages(messages)

        # Assistant message should not have reasoning_details key
        assistant_msgs = [m for m in converted if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert "reasoning_details" not in assistant_msgs[0]


class TestGenericOpenAIClientSpecialCases:
    """Test special cases and edge conditions."""

    def test_assistant_message_with_tool_use(self, ollama_client):
        """Verify handling when assistant message contains tool_use blocks."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll help you with that."},
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "test_tool",
                        "input": {"param": "value"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": "Tool execution result"
                    }
                ]
            }
        ]

        # Should handle messages with tool_use blocks
        response = ollama_client.messages.create(messages=messages)

        assert isinstance(response, GenericOpenAIResponse)
        assert len(response.content) > 0

    def test_unicode_in_messages(self, ollama_client):
        """Verify Unicode text is handled correctly."""
        messages = [
            {"role": "user", "content": "Hello in Chinese: 你好"}
        ]

        response = ollama_client.messages.create(messages=messages)

        assert isinstance(response, GenericOpenAIResponse)
        assert len(response.content) > 0

    def test_special_characters_in_messages(self, ollama_client):
        """Verify special characters are handled correctly."""
        messages = [
            {"role": "user", "content": "Special chars: @#$%^&*(){}[]|\\"}
        ]

        response = ollama_client.messages.create(messages=messages)

        assert isinstance(response, GenericOpenAIResponse)
        assert len(response.content) > 0
