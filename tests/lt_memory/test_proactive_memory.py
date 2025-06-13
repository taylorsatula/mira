"""
Tests for lt_memory/proactive_memory.py

Testing stateless proactive memory functions with real infrastructure.
"""

import pytest
from datetime import datetime

from lt_memory.proactive_memory import (
    get_relevant_memories,
    format_relevant_memories,
    build_weighted_memory_context,
    _calculate_repetition_weight
)
from conversation import Message


@pytest.fixture
def sample_messages():
    """Real Message objects for testing conversation context."""
    return [
        Message(role="user", content="I need help with my Python project"),
        Message(role="assistant", content="I'd be happy to help! What specific issue are you facing?"),
        Message(role="user", content="I'm having trouble with database connections"),
        Message(role="assistant", content="Let me help you troubleshoot the database connection issue"),
        Message(role="user", content="The connection keeps timing out"),
        Message(role="assistant", content="Connection timeouts often indicate network or configuration issues")
    ]


class TestCalculateRepetitionWeight:
    """Test the _calculate_repetition_weight private function."""
    
    def test_single_message_gets_weight_one(self):
        """
        Test that a single message gets weight 1.
        
        REAL BUG THIS CATCHES: If _calculate_repetition_weight() fails with
        single messages, memory context building breaks for short conversations,
        preventing memory retrieval for users with minimal chat history.
        """
        weight = _calculate_repetition_weight(position=1, total_messages=1)
        assert weight == 1
    
    def test_most_recent_message_gets_highest_weight(self):
        """
        Test that the most recent message gets weight 3.
        
        REAL BUG THIS CATCHES: If the newest message doesn't get highest weight,
        recent context gets diluted in memory search, causing retrieval of 
        irrelevant old memories instead of contextually relevant recent ones.
        """
        # Most recent message (position = total_messages) should get weight 3
        weight = _calculate_repetition_weight(position=5, total_messages=5)
        assert weight == 3


class TestBuildWeightedMemoryContext:
    """Test the build_weighted_memory_context function."""
    
    def test_empty_messages_returns_empty_string(self):
        """
        Test that empty message list returns empty string.
        
        REAL BUG THIS CATCHES: If build_weighted_memory_context() crashes on
        empty input, memory search fails when no conversation exists yet,
        breaking new user experience and empty conversation scenarios.
        """
        result = build_weighted_memory_context([])
        assert result == ""
    
    def test_builds_context_with_real_messages(self, sample_messages):
        """
        Test that real messages produce weighted context string.
        
        REAL BUG THIS CATCHES: If build_weighted_memory_context() fails to 
        extract content from real Message objects, memory search gets no 
        context and returns random/irrelevant memories instead of relevant ones.
        """
        result = build_weighted_memory_context(sample_messages, context_window=6)
        
        # Should contain only user messages
        assert "connection keeps timing out" in result
        assert "having trouble with database connections" in result
        # Should NOT contain assistant messages
        assert "Let me help you troubleshoot" not in result
        # Should have weighted repetition (most recent user message appears more)
        assert result.count("connection keeps timing out") > result.count("having trouble with database connections")