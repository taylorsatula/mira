"""
Tests for the LLM Bridge module.

This module tests the LLMBridge class, including API interactions,
rate limiting, response handling, and error management.
"""
import pytest
import time
import json
from unittest.mock import patch, MagicMock

import anthropic
from api.llm_bridge import LLMBridge
from errors import APIError, ErrorCode


@pytest.fixture
def mock_anthropic():
    """Create a mocked Anthropic client."""
    with patch('anthropic.Anthropic') as mock_client:
        # Create a mock instance
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock the messages API
        mock_messages = MagicMock()
        mock_instance.messages = mock_messages
        
        # Mock the create method
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "This is a test response"
        mock_response.content = [mock_content_block]
        mock_messages.create.return_value = mock_response
        
        yield mock_instance


def test_llm_bridge_init(mock_anthropic, monkeypatch):
    """Test initialization of the LLM bridge."""
    # Mock the API key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    
    # Initialize the bridge
    bridge = LLMBridge()
    
    # Check initialization
    assert bridge.api_key == "test-api-key"
    assert bridge.model == "claude-3-7-sonnet-20250219"
    assert bridge.max_tokens == 1000
    assert bridge.temperature == 0.7
    assert bridge.max_retries == 3
    assert bridge.timeout == 60
    assert bridge.rate_limit_rpm == 10
    assert bridge.min_request_interval == 6.0  # 60 / 10
    assert bridge.last_request_time == 0.0


def test_rate_limiting(mock_anthropic, monkeypatch):
    """Test rate limiting functionality."""
    # Mock the API key and time
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    
    # Initialize with a small rate limit for testing
    bridge = LLMBridge()
    bridge.rate_limit_rpm = 2  # 2 requests per minute
    bridge.min_request_interval = 30.0  # 60 / 2
    
    # Mock sleep to check if it's called
    with patch('time.sleep') as mock_sleep:
        # Set last request time to now
        bridge.last_request_time = time.time()
        
        # Enforce rate limit
        bridge._enforce_rate_limit()
        
        # Sleep should be called with approximately the min interval
        mock_sleep.assert_called_once()
        sleep_time = mock_sleep.call_args[0][0]
        assert 20.0 <= sleep_time <= 30.0  # Allow some flexibility
        
        # Reset last request time to far in the past
        bridge.last_request_time = time.time() - 60
        mock_sleep.reset_mock()
        
        # Enforce rate limit again
        bridge._enforce_rate_limit()
        
        # Sleep should not be called
        mock_sleep.assert_not_called()


def test_generate_response(mock_anthropic, monkeypatch):
    """Test generating a response from the API."""
    # Mock the API key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    
    # Initialize the bridge
    bridge = LLMBridge()
    
    # Prepare test messages
    messages = [
        {"role": "user", "content": "Hello"}
    ]
    
    # Generate a response
    with patch('time.time', return_value=100.0):
        response = bridge.generate_response(
            messages=messages,
            system_prompt="You are a helpful assistant",
            temperature=0.5,
            max_tokens=500
        )
    
    # Check response
    assert response.content[0].text == "This is a test response"
    
    # Check API was called with correct parameters
    mock_anthropic.messages.create.assert_called_once()
    call_args = mock_anthropic.messages.create.call_args[1]
    assert call_args["model"] == "claude-3-7-sonnet-20250219"
    assert call_args["messages"] == messages
    assert call_args["max_tokens"] == 500
    assert call_args["temperature"] == 0.5
    assert call_args["system"] == "You are a helpful assistant"
    
    # Check rate limiting was enforced
    assert bridge.last_request_time == 100.0


def test_api_error_handling(mock_anthropic, monkeypatch):
    """Test handling of API errors."""
    # Mock the API key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    
    # Initialize the bridge
    bridge = LLMBridge()
    
    # Mock API errors
    mock_anthropic.messages.create.side_effect = [
        # Authentication error
        anthropic.APIError(status_code=401, message="Invalid API key"),
        # Rate limit error (should retry)
        anthropic.APIError(status_code=429, message="Rate limit exceeded"),
        # Success after retry
        MagicMock(content=[MagicMock(type="text", text="Success after retry")])
    ]
    
    # Prepare test messages
    messages = [{"role": "user", "content": "Hello"}]
    
    # Test authentication error
    with pytest.raises(APIError) as excinfo:
        bridge.generate_response(messages=messages)
    assert excinfo.value.code == ErrorCode.API_AUTHENTICATION_ERROR
    
    # Reset for next test
    mock_anthropic.messages.create.reset_mock()
    mock_anthropic.messages.create.side_effect = [
        # Rate limit error (should retry)
        anthropic.APIError(status_code=429, message="Rate limit exceeded"),
        # Success after retry
        MagicMock(content=[MagicMock(type="text", text="Success after retry")])
    ]
    
    # Test rate limit error with retry
    with patch('time.sleep') as mock_sleep:  # Mock sleep to avoid waiting
        response = bridge.generate_response(messages=messages)
    
    # Check retry logic
    assert mock_anthropic.messages.create.call_count == 2
    assert mock_sleep.called
    assert response.content[0].text == "Success after retry"


def test_overloaded_error_handling(mock_anthropic, monkeypatch):
    """Test handling of overloaded errors."""
    # Mock the API key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    
    # Initialize the bridge
    bridge = LLMBridge()
    
    # Prepare test messages
    messages = [{"role": "user", "content": "Hello"}]
    
    # Test dictionary format error
    overloaded_error = {
        "type": "error", 
        "error": {
            "type": "overloaded_error", 
            "message": "Overloaded"
        }
    }
    mock_anthropic.messages.create.side_effect = overloaded_error
    
    with pytest.raises(APIError) as excinfo:
        bridge.generate_response(messages=messages)
    
    assert excinfo.value.code == ErrorCode.API_RATE_LIMIT_ERROR
    assert "high traffic" in str(excinfo.value)
    
    # Test string format error
    mock_anthropic.messages.create.reset_mock()
    string_error = Exception("{'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}")
    mock_anthropic.messages.create.side_effect = string_error
    
    with pytest.raises(APIError) as excinfo:
        bridge.generate_response(messages=messages)
    
    assert excinfo.value.code == ErrorCode.API_RATE_LIMIT_ERROR
    assert "high traffic" in str(excinfo.value)


def test_streaming_overloaded_error_handling(mock_anthropic, monkeypatch):
    """Test handling of overloaded errors in streaming mode."""
    # Mock the API key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    
    # Initialize the bridge
    bridge = LLMBridge()
    
    # Prepare test messages
    messages = [{"role": "user", "content": "Hello"}]
    
    # Create a mock stream that will raise an overloaded error
    mock_stream = MagicMock()
    mock_context = MagicMock()
    mock_context.__enter__ = MagicMock(side_effect=Exception("{'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}"))
    mock_stream.__enter__ = mock_context.__enter__
    mock_stream.__exit__ = MagicMock()
    
    # Mock the stream method
    mock_anthropic.messages.stream.return_value = mock_stream
    
    # Test streaming error handling
    with pytest.raises(APIError) as excinfo:
        bridge.generate_response(messages=messages, stream=True)
    
    assert excinfo.value.code == ErrorCode.API_RATE_LIMIT_ERROR
    assert "high traffic" in str(excinfo.value)
    
    # Test with callback
    mock_callback = MagicMock()
    
    with pytest.raises(APIError) as excinfo:
        bridge.generate_response(messages=messages, stream=True, callback=mock_callback)
    
    assert excinfo.value.code == ErrorCode.API_RATE_LIMIT_ERROR
    assert "high traffic" in str(excinfo.value)
    assert not mock_callback.called  # Callback should not be called when there's an error


def test_extract_text_content(mock_anthropic, monkeypatch):
    """Test extracting text content from API responses."""
    # Mock the API key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    
    # Initialize the bridge
    bridge = LLMBridge()
    
    # Create test response with multiple content blocks
    response = MagicMock()
    response.content = [
        type('ContentBlock', (), {'type': 'text', 'text': 'First text block'}),
        type('ContentBlock', (), {'type': 'text', 'text': 'Second text block'})
    ]
    
    # Extract text
    text = bridge.extract_text_content(response)
    
    # Check extracted text
    assert text == "First text block Second text block"
    
    # Test with empty response
    empty_response = MagicMock()
    empty_response.content = []
    assert bridge.extract_text_content(empty_response) == ""


def test_extract_tool_calls(mock_anthropic, monkeypatch):
    """Test extracting tool calls from API responses."""
    # Mock the API key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    
    # Initialize the bridge
    bridge = LLMBridge()
    
    # Create test response with tool calls
    response = MagicMock()
    tool_call_1 = type('ToolCall', (), {
        'type': 'tool_use',
        'id': 'tool-1',
        'name': 'weather_tool',
        'input': {'location': 'New York'}
    })
    tool_call_2 = type('ToolCall', (), {
        'type': 'tool_use',
        'id': 'tool-2',
        'name': 'calculator',
        'input': {'expression': '2+2'}
    })
    response.content = [
        type('ContentBlock', (), {'type': 'text', 'text': 'Let me check that for you.'}),
        tool_call_1,
        tool_call_2
    ]
    
    # Extract tool calls
    tool_calls = bridge.extract_tool_calls(response)
    
    # Check extracted tool calls
    assert len(tool_calls) == 2
    assert tool_calls[0]['id'] == 'tool-1'
    assert tool_calls[0]['tool_name'] == 'weather_tool'
    assert tool_calls[0]['input'] == {'location': 'New York'}
    assert tool_calls[1]['id'] == 'tool-2'
    assert tool_calls[1]['tool_name'] == 'calculator'
    assert tool_calls[1]['input'] == {'expression': '2+2'}
    
    # Test with no tool calls
    text_only_response = MagicMock()
    text_only_response.content = [
        type('ContentBlock', (), {'type': 'text', 'text': 'Just text, no tools.'})
    ]
    assert bridge.extract_tool_calls(text_only_response) == []