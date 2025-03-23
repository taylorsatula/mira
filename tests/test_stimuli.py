"""
Tests for the stimulus handling module.

This module tests the stimulus handling functionality, including
stimulus creation, processing, and conversation integration.
"""
import pytest
import time
from typing import Dict, Any, List, Optional, Callable

from stimuli import (
    StimulusType, Stimulus, StimulusHandler,
    add_stimulus_to_conversation, process_stimulus,
    create_conversation_handler, is_stimulus_message,
    get_stimulus_metadata, format_stimulus_context
)
from errors import StimulusError, ErrorCode
from conversation import Conversation


def test_stimulus_type_enum():
    """Test the StimulusType enumeration."""
    # Check stimulus types
    assert StimulusType.MESSAGE.value == "message"
    assert StimulusType.NOTIFICATION.value == "notification"
    assert StimulusType.EVENT.value == "event"
    assert StimulusType.SCHEDULE.value == "schedule"
    assert StimulusType.SENSOR.value == "sensor"
    assert StimulusType.API.value == "api"
    assert StimulusType.CUSTOM.value == "custom"


def test_stimulus_creation():
    """Test creating a stimulus."""
    # Create a stimulus
    stimulus = Stimulus(
        type=StimulusType.NOTIFICATION,
        content="Test notification",
        source="test_source",
        metadata={"priority": "high"}
    )
    
    # Check properties
    assert stimulus.type == StimulusType.NOTIFICATION
    assert stimulus.content == "Test notification"
    assert stimulus.source == "test_source"
    assert stimulus.id is not None
    assert stimulus.created_at is not None
    assert stimulus.metadata == {"priority": "high"}


def test_stimulus_serialization():
    """Test stimulus to/from dictionary conversion."""
    # Create a stimulus
    original = Stimulus(
        type=StimulusType.EVENT,
        content="Test event",
        source="test_source",
        metadata={"category": "test"}
    )
    
    # Convert to dictionary
    data = original.to_dict()
    
    # Check dictionary
    assert data["type"] == StimulusType.EVENT.value
    assert data["content"] == "Test event"
    assert data["source"] == "test_source"
    assert data["id"] == original.id
    assert data["created_at"] == original.created_at
    assert data["metadata"] == {"category": "test"}
    
    # Create from dictionary
    recreated = Stimulus.from_dict(data)
    
    # Check recreated stimulus
    assert recreated.type == original.type
    assert recreated.content == original.content
    assert recreated.source == original.source
    assert recreated.id == original.id
    assert recreated.created_at == original.created_at
    assert recreated.metadata == original.metadata


def test_stimulus_formatting():
    """Test formatting a stimulus for prompts."""
    # Create a stimulus without metadata
    stimulus = Stimulus(
        type=StimulusType.NOTIFICATION,
        content="Test notification",
        source="test_source"
    )
    
    # Check formatting
    formatted = stimulus.format_for_prompt()
    assert formatted == "[NOTIFICATION from test_source]: Test notification"
    
    # Create a stimulus with metadata
    stimulus = Stimulus(
        type=StimulusType.SENSOR,
        content="Temperature: 25°C",
        source="temp_sensor",
        metadata={"location": "living_room", "unit": "celsius"}
    )
    
    # Check formatting with metadata
    formatted = stimulus.format_for_prompt()
    assert "SENSOR from temp_sensor" in formatted
    assert "Temperature: 25°C" in formatted
    assert "location=living_room" in formatted
    assert "unit=celsius" in formatted


def test_stimulus_handler_init():
    """Test initializing a stimulus handler."""
    handler = StimulusHandler()
    
    # Check initial state
    assert handler.conversations == []
    assert len(handler.handlers) == len(StimulusType)
    for stim_type in StimulusType:
        assert stim_type in handler.handlers
        assert handler.handlers[stim_type] == []


def test_handler_registration(stimulus_handler):
    """Test registering handlers for stimulus types."""
    # Create a test handler function
    called_with = []
    
    def test_handler(stimulus):
        called_with.append(stimulus)
    
    # Register for a specific type
    stimulus_handler.register_handler(StimulusType.NOTIFICATION, test_handler)
    
    # Check registration
    assert len(stimulus_handler.handlers[StimulusType.NOTIFICATION]) == 1
    
    # Process a stimulus
    test_stimulus = Stimulus(
        type=StimulusType.NOTIFICATION,
        content="Test",
        source="test"
    )
    stimulus_handler.process_stimulus(test_stimulus)
    
    # Check that handler was called
    assert len(called_with) == 1
    assert called_with[0] == test_stimulus
    
    # Process a different type
    other_stimulus = Stimulus(
        type=StimulusType.EVENT,
        content="Test event",
        source="test"
    )
    stimulus_handler.process_stimulus(other_stimulus)
    
    # Check that handler was not called again
    assert len(called_with) == 1


def test_stimulus_creation_and_processing(stimulus_handler):
    """Test creating and processing a stimulus."""
    # Create a test handler function
    processed_stimuli = []
    
    def test_handler(stimulus):
        processed_stimuli.append(stimulus)
    
    # Register handler
    stimulus_handler.register_handler(StimulusType.SENSOR, test_handler)
    
    # Create and process a stimulus
    result = stimulus_handler.create_and_process(
        stimulus_type=StimulusType.SENSOR,
        content="Temperature: 30°C",
        source="temp_sensor",
        metadata={"location": "kitchen"}
    )
    
    # Check result
    assert result.type == StimulusType.SENSOR
    assert result.content == "Temperature: 30°C"
    assert result.source == "temp_sensor"
    assert result.metadata == {"location": "kitchen"}
    
    # Check that handler was called
    assert len(processed_stimuli) == 1
    assert processed_stimuli[0] == result


def test_conversation_attachment(stimulus_handler, conversation):
    """Test attaching a conversation to a stimulus handler."""
    # Attach the conversation
    stimulus_handler.attach_conversation(conversation)
    
    # Check that conversation was added
    assert len(stimulus_handler.conversations) == 1
    assert stimulus_handler.conversations[0] == conversation
    
    # Check that handlers were registered for all stimulus types
    for stim_type in StimulusType:
        assert len(stimulus_handler.handlers[stim_type]) == 1
    
    # Process a stimulus
    stimulus = Stimulus(
        type=StimulusType.NOTIFICATION,
        content="Test notification",
        source="test_source"
    )
    stimulus_handler.process_stimulus(stimulus)
    
    # Check that the stimulus was added to the conversation
    assert len(conversation.messages) == 1
    assert conversation.messages[0].role == "user"
    assert "[NOTIFICATION from test_source]: Test notification" in conversation.messages[0].content
    assert conversation.messages[0].metadata.get("is_stimulus") is True


def test_add_stimulus_to_conversation(conversation):
    """Test adding a stimulus to a conversation."""
    # Create a stimulus
    stimulus = Stimulus(
        type=StimulusType.EVENT,
        content="Test event",
        source="test_source",
        metadata={"category": "test"}
    )
    
    # Add to conversation
    add_stimulus_to_conversation(stimulus, conversation)
    
    # Check that the stimulus was added as a message
    assert len(conversation.messages) == 1
    message = conversation.messages[0]
    
    # Check message content
    assert message.role == "user"
    assert "[EVENT from test_source]: Test event" in message.content
    
    # Check message metadata
    assert message.metadata.get("is_stimulus") is True
    assert message.metadata.get("stimulus_id") == stimulus.id
    assert message.metadata.get("stimulus_type") == StimulusType.EVENT.value
    assert message.metadata.get("stimulus_source") == "test_source"
    assert message.metadata.get("category") == "test"


def test_process_stimulus(conversation, mock_llm_bridge):
    """Test processing a stimulus through a conversation."""
    # Create a stimulus
    stimulus = Stimulus(
        type=StimulusType.MESSAGE,
        content="Test message",
        source="test_source"
    )
    
    # Create a callback tracker
    callback_calls = []
    
    def callback(stim, response):
        callback_calls.append((stim, response))
    
    # Process the stimulus
    response = process_stimulus(stimulus, conversation, callback)
    
    # Check that response was generated
    assert response == "This is a mock response from the LLM."
    
    # Check that the callback was called
    assert len(callback_calls) == 1
    assert callback_calls[0][0] == stimulus
    assert callback_calls[0][1] == response
    
    # Check conversation state
    assert len(conversation.messages) == 2
    assert conversation.messages[0].role == "user"
    assert conversation.messages[1].role == "assistant"


def test_is_stimulus_message():
    """Test checking if a message originated from a stimulus."""
    # Create a message class for testing
    class TestMessage:
        def __init__(self, metadata):
            self.metadata = metadata
    
    # Message from stimulus
    stimulus_message = TestMessage({"is_stimulus": True, "stimulus_id": "123"})
    assert is_stimulus_message(stimulus_message) is True
    
    # Regular message
    regular_message = TestMessage({})
    assert is_stimulus_message(regular_message) is False


def test_get_stimulus_metadata():
    """Test extracting stimulus metadata from a message."""
    # Create a message class for testing
    class TestMessage:
        def __init__(self, metadata):
            self.metadata = metadata
    
    # Message with stimulus metadata
    metadata = {
        "is_stimulus": True,
        "stimulus_id": "123",
        "stimulus_type": "notification",
        "stimulus_source": "test",
        "other_key": "value"
    }
    message = TestMessage(metadata)
    
    # Extract metadata
    result = get_stimulus_metadata(message)
    
    # Check result
    assert result["stimulus_id"] == "123"
    assert result["stimulus_type"] == "notification"
    assert result["stimulus_source"] == "test"
    assert "other_key" not in result
    
    # Regular message
    regular_message = TestMessage({})
    assert get_stimulus_metadata(regular_message) == {}


def test_format_stimulus_context():
    """Test formatting a list of stimuli for context."""
    # Create stimuli
    stimuli = [
        Stimulus(type=StimulusType.NOTIFICATION, content="Notification 1", source="source1"),
        Stimulus(type=StimulusType.EVENT, content="Event 1", source="source2"),
        Stimulus(type=StimulusType.SENSOR, content="Sensor 1", source="source3"),
    ]
    
    # Format context
    context = format_stimulus_context(stimuli)
    
    # Check format
    assert "Recent stimuli:" in context
    assert "[NOTIFICATION from source1]: Notification 1" in context
    assert "[EVENT from source2]: Event 1" in context
    assert "[SENSOR from source3]: Sensor 1" in context
    
    # Test with empty list
    assert format_stimulus_context([]) == ""
    
    # Test with limit
    context = format_stimulus_context(stimuli, max_count=2)
    assert "Notification 1" in context
    assert "Event 1" in context
    assert "Sensor 1" not in context  # Should be limited to 2