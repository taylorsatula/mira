"""
Tests for the conversation management module.

This module tests the Conversation class, including message management,
response generation, tool call processing, and serialization.
"""
import pytest
import uuid
import time
from typing import Dict, Any, List

from conversation import Conversation, Message
from errors import ConversationError, ErrorCode


def test_conversation_creation():
    """Test creating a new conversation."""
    # Create with default parameters
    conversation = Conversation()
    assert conversation.conversation_id is not None
    assert conversation.system_prompt == "You are a helpful AI assistant with access to tools."
    assert conversation.messages == []
    
    # Create with custom parameters
    custom_id = str(uuid.uuid4())
    custom_prompt = "Custom system prompt"
    conversation = Conversation(
        conversation_id=custom_id,
        system_prompt=custom_prompt
    )
    assert conversation.conversation_id == custom_id
    assert conversation.system_prompt == custom_prompt


def test_message_addition():
    """Test adding messages to a conversation."""
    conversation = Conversation()
    
    # Add a user message
    message = conversation.add_message("user", "Hello")
    assert message.role == "user"
    assert message.content == "Hello"
    assert message.id is not None
    assert len(conversation.messages) == 1
    
    # Add an assistant message
    message = conversation.add_message("assistant", "Hi there!")
    assert message.role == "assistant"
    assert message.content == "Hi there!"
    assert len(conversation.messages) == 2
    
    # Add a message with metadata
    metadata = {"key": "value"}
    message = conversation.add_message("user", "Test", metadata)
    assert message.metadata == metadata


def test_invalid_role_handling():
    """Test handling of invalid message roles."""
    conversation = Conversation()
    
    # Should convert invalid roles to "user"
    message = conversation.add_message("invalid_role", "Content")
    assert message.role == "user"
    
    # Valid roles should remain unchanged
    message = conversation.add_message("assistant", "Content")
    assert message.role == "assistant"


def test_message_formatting(conversation):
    """Test formatting messages for the API."""
    # Add messages
    conversation.add_message("user", "Hello")
    conversation.add_message("assistant", "Hi there!")
    
    # Get formatted messages
    formatted = conversation.get_formatted_messages()
    
    # Check format
    assert len(formatted) == 2
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Hello"
    assert formatted[1]["role"] == "assistant"
    assert formatted[1]["content"] == "Hi there!"


def test_conversation_pruning():
    """Test pruning conversation history."""
    conversation = Conversation()
    conversation.max_history = 2  # Set small max_history for testing
    
    # Add messages beyond the limit
    for i in range(5):
        conversation.add_message("user", f"Message {i}")
    
    # Check that history was pruned
    assert len(conversation.messages) == 4  # max_history * 2
    assert conversation.messages[0].content == "Message 1"
    assert conversation.messages[-1].content == "Message 4"


def test_conversation_serialization(conversation):
    """Test conversation to/from dictionary conversion."""
    # Add some messages
    conversation.add_message("user", "Hello")
    conversation.add_message("assistant", "Hi there!")
    
    # Convert to dictionary
    data = conversation.to_dict()
    
    # Check dictionary
    assert data["conversation_id"] == conversation.conversation_id
    assert data["system_prompt"] == conversation.system_prompt
    assert len(data["messages"]) == 2
    
    # Create a new conversation from the dictionary
    new_conversation = Conversation.from_dict(data)
    
    # Check the new conversation
    assert new_conversation.conversation_id == conversation.conversation_id
    assert new_conversation.system_prompt == conversation.system_prompt
    assert len(new_conversation.messages) == 2
    assert new_conversation.messages[0].role == "user"
    assert new_conversation.messages[0].content == "Hello"


def test_generate_response(conversation, mock_llm_bridge):
    """Test generating responses."""
    # Generate a response
    response = conversation.generate_response("Hello")
    
    # Check response
    assert response == "This is a mock response from the LLM."
    
    # Check message was added
    assert len(conversation.messages) == 2
    assert conversation.messages[0].role == "user"
    assert conversation.messages[0].content == "Hello"
    assert conversation.messages[1].role == "assistant"
    assert conversation.messages[1].content == "This is a mock response from the LLM."
    
    # Check LLM was called with correct parameters
    assert len(mock_llm_bridge.calls) == 1
    assert mock_llm_bridge.calls[0]["messages"][0]["role"] == "user"
    assert mock_llm_bridge.calls[0]["messages"][0]["content"] == "Hello"
    assert mock_llm_bridge.calls[0]["system_prompt"] == conversation.system_prompt


def test_tool_call_processing(conversation, mock_llm_bridge, monkeypatch):
    """Test processing tool calls."""
    # Mock tool call extraction and tool repo
    def mock_extract_tool_calls(self, response):
        return [{
            "id": "tool-call-1",
            "tool_name": "test_tool",
            "input": {"param": "value"}
        }]
    
    # Mock tool invocation
    def mock_invoke_tool(self, tool_name, tool_params):
        return f"Result from {tool_name} with {tool_params}"
    
    # Apply mocks
    monkeypatch.setattr(mock_llm_bridge.__class__, "extract_tool_calls", mock_extract_tool_calls)
    monkeypatch.setattr(conversation.tool_repo, "invoke_tool", mock_invoke_tool)
    
    # Generate a response which will now contain tool calls
    response = conversation.generate_response("Use a tool")
    
    # Check that tool calls were processed
    assert len(mock_llm_bridge.calls) == 2  # Initial call + call after tool results
    
    # Check that tool result was added as a user message
    assert len(conversation.messages) == 3  # user message + assistant with tool call + user with tool result
    assert conversation.messages[1].metadata.get("has_tool_calls") is True


def test_conversation_clear(conversation):
    """Test clearing conversation history."""
    # Add messages
    conversation.add_message("user", "Hello")
    conversation.add_message("assistant", "Hi there!")
    
    # Clear history
    conversation.clear_history()
    
    # Check that history was cleared
    assert conversation.messages == []
    
    # Check that conversation ID and system prompt were preserved
    assert conversation.conversation_id is not None
    assert conversation.system_prompt == "Test system prompt"


def test_message_serialization():
    """Test message to/from dictionary conversion."""
    # Create a message
    message = Message(
        role="user",
        content="Hello",
        metadata={"key": "value"}
    )
    
    # Convert to dictionary
    data = message.to_dict()
    
    # Check dictionary
    assert data["role"] == "user"
    assert data["content"] == "Hello"
    assert data["id"] == message.id
    assert data["created_at"] == message.created_at
    assert data["metadata"] == {"key": "value"}
    
    # Create a new message from the dictionary
    new_message = Message.from_dict(data)
    
    # Check the new message
    assert new_message.role == message.role
    assert new_message.content == message.content
    assert new_message.id == message.id
    assert new_message.created_at == message.created_at
    assert new_message.metadata == message.metadata