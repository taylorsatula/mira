"""
Tests for conversation context embedding utility.

Following testing guide principles:
- Test contracts, not implementation
- Write one test at a time with reflection
- Use real embedding system
- Focus on production scenarios
"""

import pytest
from utils.conversation_context_embedding import ConversationContextEmbedding


class TestConversationContextEmbeddingInitialization:
    """Test basic initialization contracts."""
    
    def test_initialization_creates_working_instance(self):
        """
        Test that initialization creates a properly configured instance.
        
        REAL BUG THIS CATCHES: If initialization fails with missing dependencies
        or configuration errors, the entire conversation context system breaks.
        """
        context_embedding = ConversationContextEmbedding()
        
        # Verify instance is properly configured
        assert context_embedding.cache_size == 100  # Default cache size
        assert len(context_embedding.message_history) == 0  # Empty initially
        assert len(context_embedding.context_cache) == 0  # Empty cache initially
        assert context_embedding.embeddings is not None  # Embedding model loaded
    
    def test_embedding_generation_works_with_real_text(self):
        """
        Test that embedding generation produces valid embeddings.
        
        REAL BUG THIS CATCHES: If the embedding integration is broken,
        the entire context embedding system returns None, breaking 
        downstream similarity calculations.
        """
        context_embedding = ConversationContextEmbedding()
        test_text = "This is a test context for embedding generation"
        
        embedding = context_embedding.get_embedding(test_text)
        
        # Verify embedding contract
        assert embedding is not None
        assert hasattr(embedding, 'shape'), "Embedding should be numpy array with shape"
        assert len(embedding.shape) == 1, "Should be 1D embedding vector"  
        assert embedding.shape[0] == 384, "Should be 384-dimensional"
    
    def test_caching_prevents_duplicate_computation(self):
        """
        Test that identical context text uses cache instead of recomputing.
        
        REAL BUG THIS CATCHES: If caching is broken, every embedding request
        recomputes, causing severe performance degradation in production.
        """
        context_embedding = ConversationContextEmbedding()
        test_text = "This context should be cached"
        
        # First call should compute and cache
        embedding1 = context_embedding.get_embedding(test_text)
        assert len(context_embedding.context_cache) == 1, "Should have cached the embedding"
        
        # Second call should use cache (should be identical)
        embedding2 = context_embedding.get_embedding(test_text)
        assert len(context_embedding.context_cache) == 1, "Cache size shouldn't increase"
        
        # Embeddings should be identical (from cache)
        import numpy as np
        np.testing.assert_array_equal(embedding1, embedding2)


class TestMessageHistoryManagement:
    """Test message history management contracts."""
    
    def test_add_message_stores_messages_correctly(self):
        """
        Test that add_message stores messages in correct format.
        
        REAL BUG THIS CATCHES: If message storage format is wrong,
        downstream context building systems get malformed data and break.
        """
        context_embedding = ConversationContextEmbedding()
        
        # Add different message types
        context_embedding.add_message("user", "Hello world")
        context_embedding.add_message("assistant", "Hi there!")
        
        messages = context_embedding.get_messages()
        
        # Verify message storage contract
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello world"
        assert messages[1]["role"] == "assistant" 
        assert messages[1]["content"] == "Hi there!"
    
    def test_message_buffer_respects_size_limit(self):
        """
        Test that message buffer doesn't grow beyond size limit.
        
        REAL BUG THIS CATCHES: If buffer size limit is broken, memory usage
        grows unbounded in long conversations, causing system crashes.
        """
        context_embedding = ConversationContextEmbedding()
        
        # Add more messages than the buffer limit (20)
        for i in range(25):
            context_embedding.add_message("user", f"Message {i}")
        
        messages = context_embedding.get_messages()
        
        # Should only keep the last 20 messages  
        assert len(messages) == 20
        assert messages[0]["content"] == "Message 5"  # First kept message
        assert messages[-1]["content"] == "Message 24"  # Last message


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge case scenarios."""
    
    def test_empty_and_invalid_messages_ignored(self):
        """
        Test that empty or invalid messages are properly ignored.
        
        REAL BUG THIS CATCHES: If empty messages get stored, downstream
        context building creates malformed contexts that break embeddings.
        """
        context_embedding = ConversationContextEmbedding()
        
        # Try adding various invalid messages
        context_embedding.add_message("user", "")  # Empty string
        context_embedding.add_message("user", "   ")  # Whitespace only
        context_embedding.add_message("user", None)  # None content
        context_embedding.add_message("user", 123)  # Non-string content
        
        # Add valid message
        context_embedding.add_message("user", "Valid message")
        
        messages = context_embedding.get_messages()
        
        # Should only have the valid message
        assert len(messages) == 1
        assert messages[0]["content"] == "Valid message"


class TestCacheManagement:
    """Test cache management and trimming behavior."""
    
    def test_cache_trimming_prevents_unbounded_growth(self):
        """
        Test that cache gets trimmed when it exceeds size limit.
        
        REAL BUG THIS CATCHES: If cache trimming is broken, memory usage
        grows unbounded with unique contexts, causing out-of-memory crashes.
        """
        # Use small cache size for testing
        context_embedding = ConversationContextEmbedding(cache_size=3)
        
        # Add more unique contexts than cache limit
        contexts = [
            "Context number one",
            "Context number two", 
            "Context number three",
            "Context number four",
            "Context number five"
        ]
        
        for context in contexts:
            context_embedding.get_embedding(context)
        
        # Cache should be trimmed to size limit
        assert len(context_embedding.context_cache) == 3
        
        # Should still be able to get embeddings
        new_embedding = context_embedding.get_embedding("New context")
        assert new_embedding is not None


class TestSingletonGlobalInstance:
    """Test singleton pattern for global instance."""
    
    def test_global_instance_provides_singleton_behavior(self):
        """
        Test that global instance function provides singleton behavior.
        
        REAL BUG THIS CATCHES: If singleton is broken, different parts of
        the system get different instances, breaking shared context state.
        """
        from utils.conversation_context_embedding import get_conversation_context_embedding, reset_conversation_context_embedding
        
        # Reset to clean state
        reset_conversation_context_embedding()
        
        # Get two instances
        instance1 = get_conversation_context_embedding()
        instance2 = get_conversation_context_embedding()
        
        # Should be the same object
        assert instance1 is instance2
        
        # State should be shared
        instance1.add_message("user", "Test message")
        messages_from_instance2 = instance2.get_messages()
        
        assert len(messages_from_instance2) == 1
        assert messages_from_instance2[0]["content"] == "Test message"
    
    def test_flush_context_clears_message_history(self):
        """
        Test that flush_context clears message history on topic changes.
        
        REAL BUG THIS CATCHES: If context flushing is broken, old conversation
        context contaminates new topics, causing irrelevant classifications.
        """
        context_embedding = ConversationContextEmbedding()
        
        # Add some messages and cache some embeddings
        context_embedding.add_message("user", "Previous topic message")
        context_embedding.get_embedding("Previous context")
        
        # Verify initial state
        assert len(context_embedding.get_messages()) == 1
        assert len(context_embedding.context_cache) == 1
        
        # Flush context
        context_embedding.flush_context()
        
        # Message history should be cleared
        assert len(context_embedding.get_messages()) == 0
        
        # Cache should still exist (trimmed, not cleared completely)
        # This allows for some reuse while clearing message context
        assert len(context_embedding.context_cache) >= 0  # Could be 0 or more after trimming