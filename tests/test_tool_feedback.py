"""
Test Tool Feedback Module

Tests for the tool feedback functionality.
"""
import os
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.tool_feedback import save_tool_feedback, get_feedback_summary
from conversation import Conversation, Message


@pytest.fixture
def mock_system():
    """Create a mock system with necessary components."""
    tool_repo = MagicMock()
    
    # Mock tool_relevance_engine
    tool_relevance_engine = MagicMock()
    tool_relevance_engine.tool_activation_history = {
        "calendar_tool": 1, 
        "email_tool": 2
    }
    
    # Mock classifier with similarity calculation
    classifier = MagicMock()
    classifier.calculate_text_similarity = MagicMock(return_value=0.85)
    tool_relevance_engine.classifier = classifier
    
    # Mock tool examples
    tool_relevance_engine.tool_examples = {
        "calendar_tool": {
            "examples": [
                {"query": "Schedule a meeting for tomorrow"},
                {"query": "Add an event to my calendar"}
            ]
        },
        "email_tool": {
            "examples": [
                {"query": "Send an email to John"},
                {"query": "Check my inbox for new messages"}
            ]
        }
    }
    
    # Mock LLM Bridge
    llm_bridge = MagicMock()
    llm_bridge.generate_text = MagicMock(return_value="Mock LLM analysis of the tool feedback.")
    
    return {
        "tool_repo": tool_repo,
        "tool_relevance_engine": tool_relevance_engine,
        "llm_bridge": llm_bridge
    }


@pytest.fixture
def mock_conversation():
    """Create a mock conversation with sample messages."""
    conversation = MagicMock(spec=Conversation)
    
    # Create sample messages
    messages = [
        Message(role="user", content="I need to schedule a meeting"),
        Message(role="assistant", content="I can help you schedule a meeting. What date and time?"),
        Message(role="user", content="Tomorrow at 2pm with the marketing team")
    ]
    
    conversation.messages = messages
    conversation.conversation_id = "test_conversation_123"
    
    return conversation


@patch("tools.tool_feedback.config")
@patch("json.dump")  # Mock json.dump to avoid serialization issues
def test_save_tool_feedback(mock_json_dump, mock_config, mock_system, mock_conversation, tmp_path):
    """Test saving tool feedback with LLM analysis."""
    # Set up mock config
    mock_config.paths.persistent_dir = str(tmp_path)
    
    # Mock json.dump to avoid serialization issues
    mock_json_dump.return_value = None
    
    # Set up mock LLM bridge response
    llm_bridge = mock_system['llm_bridge']
    llm_bridge.generate_response.return_value = {"content": "Mock response"}
    llm_bridge.extract_text_content.return_value = "Mock LLM analysis of the tool feedback."
    
    # Call the function
    with patch("builtins.open", mock_open()) as mock_file:
        success, analysis = save_tool_feedback(mock_system, "The calendar tool was helpful", mock_conversation)
    
    # Verify results
    assert success is True
    assert analysis == "Mock LLM analysis of the tool feedback."
    
    # Verify the file was opened for writing
    mock_file.assert_called_once()
    
    # Verify json.dump was called with the right data
    assert mock_json_dump.called
    args, _ = mock_json_dump.call_args
    feedback_data = args[0]  # First argument to json.dump
    
    # Check the content of what would have been saved
    assert feedback_data["feedback"] == "The calendar tool was helpful"
    assert "last_messages" in feedback_data
    assert "active_tools" in feedback_data
    assert "nearest_examples" in feedback_data
    assert "llm_analysis" in feedback_data


@patch("tools.tool_feedback.config")
def test_get_feedback_summary(mock_config, tmp_path):
    """Test getting feedback summary."""
    # Set up mock config
    mock_config.paths.persistent_dir = str(tmp_path)
    
    # Create feedback directory
    feedback_dir = tmp_path / "tool_feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    
    # Create some sample feedback files
    sample_feedback = {
        "timestamp": "2023-01-01T12:00:00",
        "feedback": "Sample feedback",
        "conversation_id": "test123",
        "last_messages": [],
        "active_tools": ["calendar_tool", "email_tool"],
        "nearest_examples": {}
    }
    
    # Save sample feedback files
    for i in range(3):
        with open(feedback_dir / f"feedback_2023010{i}.json", 'w') as f:
            json.dump(sample_feedback, f)
    
    # Call the function
    summary = get_feedback_summary()
    
    # Verify results
    assert summary["count"] == 3
    assert "calendar_tool" in summary["tools"]
    assert "email_tool" in summary["tools"]
    assert len(summary["feedback"]) == 3


def test_get_feedback_summary_empty():
    """Test getting feedback summary when no feedback exists."""
    with patch("tools.tool_feedback.config") as mock_config, \
         patch("pathlib.Path.exists", return_value=False):
         
        # Get summary when no feedback directory exists
        summary = get_feedback_summary()
        
        # Verify results
        assert summary["count"] == 0
        assert summary["feedback"] == []


def test_analyze_feedback_with_llm():
    """Test analyzing feedback with LLM."""
    from tools.tool_feedback import analyze_feedback_with_llm
    
    # Mock LLM Bridge
    llm_bridge = MagicMock()
    llm_bridge.generate_response = MagicMock(return_value={"content": "Mock response"})
    llm_bridge.extract_text_content = MagicMock(return_value="Analysis of tool feedback")
    
    # Create sample feedback data
    feedback_data = {
        "feedback": "The calendar tool was helpful",
        "last_messages": [
            {"role": "user", "content": "Schedule a meeting tomorrow"}
        ],
        "active_tools": ["calendar_tool"],
        "nearest_examples": {
            "calendar_tool": [
                {"query": "Add an event to my calendar", "similarity": 0.85}
            ]
        }
    }
    
    # Call the analyze function
    result = analyze_feedback_with_llm(feedback_data, llm_bridge)
    
    # Verify results
    assert "timestamp" in result
    assert "analysis" in result
    assert result["analysis"] == "Analysis of tool feedback"
    
    # Verify LLM was called with appropriate arguments
    llm_bridge.generate_response.assert_called_once()
    args = llm_bridge.generate_response.call_args[1]
    assert "system_prompt" in args
    assert "concrete suggestions" in args["system_prompt"].lower()
    assert "messages" in args
    assert "temperature" in args
    assert args["temperature"] == 0.2  # Check that we're using a very low temperature for precise response
    
    # Verify text extraction was called
    llm_bridge.extract_text_content.assert_called_once()