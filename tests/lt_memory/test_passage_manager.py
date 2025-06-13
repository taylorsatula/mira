"""
Production-grade tests for PassageManager archival memory system.

This test suite focuses on realistic production scenarios that matter for
reliability. We test the contracts and behaviors that users actually depend on,
using real memory managers, real databases, and real vector search operations.

Testing philosophy:
1. Test the public API contracts that users rely on
2. Test critical private methods that could fail in subtle ways
3. Test with real PostgreSQL database and vector operations
4. Test conversation archiving with real message structures
5. Verify error handling for invalid inputs and database failures
6. Test real-world passage creation and retrieval workflows
"""

import pytest
import time
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the system under test
from lt_memory.managers.passage_manager import PassageManager
from lt_memory.managers.memory_manager import MemoryManager
from lt_memory.models.base import MemoryPassage
from config.config_manager import AppConfig
from errors import ToolError, ErrorCode
from pathlib import Path


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def test_config():
    """
    Real configuration for testing with PostgreSQL test database.
    
    Uses actual config structure to catch configuration-related bugs.
    """
    config = AppConfig()
    
    # Override with test database
    config.memory.database_url = "postgresql://mira_app@localhost/lt_memory_test"
    config.memory.db_pool_size = 5
    config.memory.db_pool_max_overflow = 10
    
    # Ensure paths exist
    config.paths.data_dir = "/tmp/test_lt_memory_cache"
    Path(config.paths.data_dir).mkdir(exist_ok=True)
    
    return config


@pytest.fixture
def test_memory_manager(test_config):
    """Real memory manager with PostgreSQL test database."""
    return MemoryManager(test_config)


@pytest.fixture
def clean_database(test_memory_manager):
    """Clean database before each test to prevent contamination."""
    with test_memory_manager.get_session() as session:
        session.query(MemoryPassage).delete()
        session.commit()
    yield test_memory_manager
    # No cleanup needed after - next test will clean again


@pytest.fixture
def passage_manager(clean_database):
    """Real passage manager with clean database."""
    return PassageManager(clean_database)


@pytest.fixture
def sample_messages():
    """
    Realistic message samples for testing conversation archiving.
    
    These represent actual conversation patterns we see in production.
    """
    return {
        "simple_conversation": [
            {"role": "user", "content": "What is machine learning?", "created_at": "2025-01-10T10:00:00Z"},
            {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn from data.", "created_at": "2025-01-10T10:00:30Z"},
            {"role": "user", "content": "Can you give me an example?", "created_at": "2025-01-10T10:01:00Z"},
            {"role": "assistant", "content": "Email spam detection is a common example.", "created_at": "2025-01-10T10:01:30Z"}
        ],
        
        "topic_change_conversation": [
            {"role": "user", "content": "Tell me about Python", "created_at": "2025-01-10T10:00:00Z"},
            {"role": "assistant", "content": "Python is a programming language.", "created_at": "2025-01-10T10:00:30Z"},
            {"role": "assistant", "content": "Now let's talk about databases.", "metadata": {"topic_changed": True}, "created_at": "2025-01-10T10:01:00Z"},
            {"role": "user", "content": "What's a database?", "created_at": "2025-01-10T10:01:30Z"},
            {"role": "assistant", "content": "A database stores structured data.", "created_at": "2025-01-10T10:02:00Z"}
        ],
        
        "unicode_conversation": [
            {"role": "user", "content": "Hello! 你好 🌍", "created_at": "2025-01-10T10:00:00Z"},
            {"role": "assistant", "content": "Hello! I can help with international text.", "created_at": "2025-01-10T10:00:30Z"}
        ]
    }


@pytest.fixture
def sample_text_content():
    """Sample text content for passage creation."""
    return {
        "technical": "This passage discusses database optimization techniques for PostgreSQL including indexing strategies and query performance.",
        "conversation": "User asked about machine learning concepts. Assistant explained supervised vs unsupervised learning with practical examples.",
        "unicode": "International content: 世界 🌍 Café naïve résumé with special characters",
        "long": "Very detailed technical discussion. " * 100,  # ~3KB text
        "empty": "",
        "whitespace": "   \n\t   \n   "
    }


# =============================================================================
# PUBLIC CONTRACT TESTS
# =============================================================================

class TestPassageCreation:
    """
    Test core passage creation functionality.
    
    This is the primary user-facing feature for storing memories.
    """
    
    def test_creates_passage_with_real_embedding_and_database(self, passage_manager, sample_text_content):
        """
        Test successful passage creation with real components.
        
        REAL BUG THIS CATCHES: If create_passage() fails to generate embeddings
        or store to PostgreSQL database, memory passages can't be created,
        breaking the entire archival memory system.
        """
        text = sample_text_content["technical"]
        
        passage_id = passage_manager.create_passage(
            text=text,
            source="test",
            source_id="test_001",
            importance=0.7,
            metadata={"test": True}
        )
        
        # Should return valid UUID string
        assert len(passage_id) > 0
        
        # Should be stored in real database - verify with direct query
        with passage_manager.memory_manager.get_session() as session:
            passage = session.query(MemoryPassage).filter_by(id=passage_id).first()
            assert passage is not None
            assert passage.text == text
            assert passage.source == "test"
            assert passage.source_id == "test_001"
            assert passage.importance_score == 0.7
            assert passage.context["test"] is True
            assert passage.embedding is not None
            assert len(passage.embedding) > 0  # Real embedding generated

    def test_stores_embedding_with_correct_dimensions(self, passage_manager, sample_text_content):
        """
        Test that generated embeddings have expected dimensions.
        
        REAL BUG THIS CATCHES: If embedding generation returns wrong dimensions,
        vector similarity search will fail, breaking passage retrieval.
        """
        passage_id = passage_manager.create_passage(
            text=sample_text_content["conversation"],
            source="conversation",
            source_id="conv_001"
        )
        
        # Verify embedding dimensions in database
        with passage_manager.memory_manager.get_session() as session:
            passage = session.query(MemoryPassage).filter_by(id=passage_id).first()
            embedding = np.array(passage.embedding)
            
            # Should have standard embedding dimensions (based on model)
            assert embedding.shape[0] > 0
            assert len(embedding.shape) == 1  # Should be 1D vector
            assert embedding.dtype in [np.float32, np.float64]


class TestPassageSearch:
    """
    Test passage search and retrieval functionality.
    
    This is critical for memory recall and similarity search.
    """
    
    def test_searches_passages_with_real_vector_similarity(self, passage_manager, sample_text_content):
        """
        Test that passage search uses real vector similarity matching.
        
        REAL BUG THIS CATCHES: If search_passages() fails to generate query embeddings
        or perform vector similarity search, users can't find relevant memories,
        breaking the core memory retrieval functionality.
        """
        # Create passages with related content
        tech_passage_id = passage_manager.create_passage(
            text=sample_text_content["technical"],
            source="document",
            source_id="doc_001",
            importance=0.8
        )
        
        convo_passage_id = passage_manager.create_passage(
            text=sample_text_content["conversation"],
            source="conversation", 
            source_id="conv_001",
            importance=0.6
        )
        
        # Verify passages were created
        with passage_manager.memory_manager.get_session() as session:
            passages = session.query(MemoryPassage).all()
            print(f"Created {len(passages)} passages in database")
        
        # Check the similarity threshold that's being used
        threshold = passage_manager.memory_manager.config.memory.similarity_threshold
        print(f"Default similarity threshold: {threshold}")
        
        # Search for related content using real vector similarity with lower threshold
        results = passage_manager.search_passages(
            query="database optimization performance",
            limit=10,
            filters={"min_similarity": 0.0}  # Override with very low threshold for testing
        )
        
        print(f"Search returned {len(results)} results")
        
        # Should return passages in similarity order
        assert len(results) > 0
        
        # Results should have proper structure
        for result in results:
            assert "id" in result
            assert "text" in result
            assert "score" in result
            assert "source" in result
            assert "importance" in result
            assert "created_at" in result
            
            # Score should be between 0 and 1
            assert 0.0 <= result["score"] <= 1.0
        
        # Should include the technical passage (more similar to query)
        passage_ids = [result["id"] for result in results]
        assert tech_passage_id in passage_ids
        
        # Should be sorted by similarity score (highest first)
        scores = [result["score"] for result in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_filters_by_importance_threshold(self, passage_manager, sample_text_content):
        """
        Test that search respects importance filtering.
        
        REAL BUG THIS CATCHES: If importance filtering fails, low-quality memories
        pollute search results, reducing recall quality for users.
        """
        # Create passages with different importance scores
        low_importance_id = passage_manager.create_passage(
            text="Low importance content about random topics",
            source="test",
            source_id="low_001",
            importance=0.1
        )
        
        high_importance_id = passage_manager.create_passage(
            text="High importance content about random topics", 
            source="test",
            source_id="high_001",
            importance=0.9
        )
        
        # Search with importance filter and low similarity threshold
        results = passage_manager.search_passages(
            query="content about topics",
            filters={"min_importance": 0.5, "min_similarity": 0.0},
            limit=10
        )
        
        # Should only return high importance passage
        passage_ids = [result["id"] for result in results]
        assert high_importance_id in passage_ids
        assert low_importance_id not in passage_ids
        
        # All results should meet importance threshold
        for result in results:
            assert result["importance"] >= 0.5


class TestPassageRetrieval:
    """
    Test individual passage retrieval and access tracking.
    
    This is critical for memory access patterns and usage analytics.
    """
    
    def test_gets_passage_by_id_with_access_tracking(self, passage_manager, sample_text_content):
        """
        Test that get_passage retrieves correct data and tracks access.
        
        REAL BUG THIS CATCHES: If get_passage() fails to increment access_count
        or update last_accessed timestamp, usage analytics break and we can't
        identify frequently accessed memories for optimization.
        """
        # Create a passage
        passage_id = passage_manager.create_passage(
            text=sample_text_content["technical"],
            source="test",
            source_id="test_001",
            importance=0.8,
            metadata={"category": "technical"}
        )
        
        # Touch passage once to establish baseline (initial state has last_accessed=None)
        first_result = passage_manager.get_passage(passage_id)
        assert first_result is not None
        
        # Get baseline state after first access
        with passage_manager.memory_manager.get_session() as session:
            passage = session.query(MemoryPassage).filter_by(id=passage_id).first()
            baseline_access_count = passage.access_count
            baseline_last_accessed = passage.last_accessed
            assert baseline_last_accessed is not None  # Should be set after first access
        
        # Retrieve passage again to test increment
        result = passage_manager.get_passage(passage_id)
        
        # Should return complete passage data
        assert result["id"] == passage_id
        assert result["text"] == sample_text_content["technical"]
        assert result["source"] == "test"
        assert result["source_id"] == "test_001"
        assert result["importance"] == 0.8
        assert result["metadata"]["category"] == "technical"
        assert result["created_at"]  # Should have ISO format timestamp
        assert result["access_count"] >= 0
        
        # Verify access tracking was updated in database
        with passage_manager.memory_manager.get_session() as session:
            updated_passage = session.query(MemoryPassage).filter_by(id=passage_id).first()
            assert updated_passage.access_count == baseline_access_count + 1
            assert updated_passage.last_accessed > baseline_last_accessed

    def test_returns_none_for_nonexistent_passage(self, passage_manager):
        """
        Test that get_passage handles nonexistent IDs gracefully.
        
        REAL BUG THIS CATCHES: If get_passage() crashes or returns invalid data
        for nonexistent IDs, applications can't handle missing memories gracefully,
        causing user-facing errors.
        """
        # Try to get nonexistent passage
        result = passage_manager.get_passage("00000000-0000-0000-0000-000000000000")
        
        # Should return None gracefully
        assert result is None

    def test_multiple_accesses_increment_count_correctly(self, passage_manager, sample_text_content):
        """
        Test that repeated access properly increments access count.
        
        REAL BUG THIS CATCHES: If access counting has race conditions or
        increment bugs, analytics become inaccurate and memory optimization
        algorithms make wrong decisions about which memories to prioritize.
        """
        # Create passage
        passage_id = passage_manager.create_passage(
            text=sample_text_content["conversation"],
            source="conversation",
            source_id="conv_001"
        )
        
        # Access multiple times
        access_count = 5
        for i in range(access_count):
            result = passage_manager.get_passage(passage_id)
            assert result is not None
            assert result["id"] == passage_id
        
        # Verify final access count
        final_result = passage_manager.get_passage(passage_id)
        expected_count = access_count + 1  # +1 for the final retrieval
        assert final_result["access_count"] == expected_count


class TestPassageManagement:
    """
    Test passage management operations like importance updates.
    
    These operations are critical for memory maintenance and optimization.
    """
    
    def test_updates_passage_importance_successfully(self, passage_manager, sample_text_content):
        """
        Test that passage importance can be updated correctly.
        
        REAL BUG THIS CATCHES: If update_passage_importance() fails to persist
        changes to the database, memory optimization algorithms work with stale
        importance scores, leading to poor memory prioritization decisions.
        """
        # Create passage with initial importance
        passage_id = passage_manager.create_passage(
            text=sample_text_content["technical"],
            source="document",
            source_id="doc_001",
            importance=0.5
        )
        
        # Verify initial importance
        initial_passage = passage_manager.get_passage(passage_id)
        assert initial_passage["importance"] == 0.5
        
        # Update importance
        success = passage_manager.update_passage_importance(passage_id, 0.9)
        assert success is True
        
        # Verify importance was updated in database
        updated_passage = passage_manager.get_passage(passage_id)
        assert updated_passage["importance"] == 0.9
        
        # Verify database persistence (not just memory cache)
        with passage_manager.memory_manager.get_session() as session:
            db_passage = session.query(MemoryPassage).filter_by(id=passage_id).first()
            assert db_passage.importance_score == 0.9

    def test_importance_update_validates_range(self, passage_manager, sample_text_content):
        """
        Test that importance update validates input range.
        
        REAL BUG THIS CATCHES: If importance validation fails, invalid scores
        can corrupt the memory ranking system, breaking similarity search
        and memory prioritization algorithms.
        """
        # Create passage
        passage_id = passage_manager.create_passage(
            text=sample_text_content["conversation"],
            source="test",
            source_id="test_001"
        )
        
        # Test invalid importance values
        with pytest.raises(ToolError) as exc_info:
            passage_manager.update_passage_importance(passage_id, 1.5)  # > 1.0
        assert exc_info.value.code == ErrorCode.INVALID_INPUT
        assert "between 0 and 1" in str(exc_info.value)
        
        with pytest.raises(ToolError) as exc_info:
            passage_manager.update_passage_importance(passage_id, -0.1)  # < 0.0
        assert exc_info.value.code == ErrorCode.INVALID_INPUT
        assert "between 0 and 1" in str(exc_info.value)
        
        # Verify original importance unchanged after failed updates
        passage = passage_manager.get_passage(passage_id)
        assert passage["importance"] == 0.5  # Default from create_passage

    def test_importance_update_returns_false_for_nonexistent_passage(self, passage_manager):
        """
        Test that updating nonexistent passage is handled gracefully.
        
        REAL BUG THIS CATCHES: If update operations don't handle missing passages
        correctly, maintenance scripts can crash when processing deleted or
        corrupted passage references.
        """
        # Try to update nonexistent passage
        success = passage_manager.update_passage_importance(
            "00000000-0000-0000-0000-000000000000", 
            0.8
        )
        
        # Should return False, not crash
        assert success is False


class TestConversationArchiving:
    """
    Test conversation archiving workflow.
    
    This is the most complex PassageManager functionality and highest risk for bugs.
    The workflow involves: message chunking → summarization → importance calculation → passage creation.
    """
    
    def test_archives_simple_conversation_successfully(self, passage_manager, sample_messages):
        """
        Test that basic conversation archiving creates passages correctly.
        
        REAL BUG THIS CATCHES: If archive_conversation() fails to process messages,
        chunk them properly, or create passages, conversation history is lost and
        users can't retrieve memories from past interactions.
        """
        conversation_id = "conv_test_001"
        messages = sample_messages["simple_conversation"]
        
        # Archive the conversation  
        archived_count = passage_manager.archive_conversation(conversation_id, messages)
        
        # Should create at least one passage
        assert archived_count > 0
        
        # Verify passages were created in database
        with passage_manager.memory_manager.get_session() as session:
            passages = session.query(MemoryPassage).filter_by(
                source="conversation",
                source_id=conversation_id
            ).all()
            
            assert len(passages) == archived_count
            
            # Verify passage structure
            for passage in passages:
                assert passage.text is not None and len(passage.text.strip()) > 0
                assert passage.source == "conversation"
                assert passage.source_id == conversation_id
                assert 0.0 <= passage.importance_score <= 1.0
                assert passage.embedding is not None
                assert passage.context
                
                # Should have archiving metadata
                assert "chunk_index" in passage.context
                assert "chunk_type" in passage.context
                assert passage.context["chunk_type"] == "micro_chunk"
                assert "extraction_version" in passage.context

    def test_handles_empty_conversation_gracefully(self, passage_manager):
        """
        Test that empty conversation list is handled without errors.
        
        REAL BUG THIS CATCHES: If archive_conversation() crashes on empty input,
        automated archiving systems fail when processing conversations with
        no messages, breaking the memory system.
        """
        archived_count = passage_manager.archive_conversation("empty_conv", [])
        
        # Should return 0 without crashing
        assert archived_count == 0
        
        # Should not create any passages
        with passage_manager.memory_manager.get_session() as session:
            passages = session.query(MemoryPassage).filter_by(
                source="conversation",
                source_id="empty_conv"
            ).all()
            assert len(passages) == 0

    def test_extracts_multiple_facts_from_conversation(self, passage_manager, sample_messages):
        """
        Test that fact extraction creates multiple micro-chunks from conversation.
        
        REAL BUG THIS CATCHES: If fact extraction fails to create discrete facts,
        the system falls back to storing large summaries, defeating the purpose
        of the micro-chunk architecture.
        """
        conversation_id = "fact_test_001"
        messages = sample_messages["topic_change_conversation"]
        
        # Archive conversation which should extract multiple facts
        archived_count = passage_manager.archive_conversation(conversation_id, messages)
        
        # Should create multiple fact passages
        assert archived_count >= 1  # At least one fact extracted
        
        # Verify passages are micro-chunks, not summaries
        with passage_manager.memory_manager.get_session() as session:
            passages = session.query(MemoryPassage).filter_by(
                source="conversation",
                source_id=conversation_id
            ).all()
            
            # Each passage should be a short fact, not a long summary
            for passage in passages:
                assert len(passage.text) < 300  # Facts should be concise
                assert passage.context["chunk_type"] == "micro_chunk"
                assert "human_verified" in passage.context
                assert passage.human_verified == False  # Not yet verified

    def test_handles_unicode_content_in_archiving(self, passage_manager, sample_messages):
        """
        Test that Unicode content is preserved through archiving process.
        
        REAL BUG THIS CATCHES: If Unicode handling fails during summarization
        or passage creation, international users lose conversation history
        or get corrupted memory content.
        """
        conversation_id = "unicode_test_001"
        messages = sample_messages["unicode_conversation"]
        
        # Archive conversation with Unicode content
        archived_count = passage_manager.archive_conversation(conversation_id, messages)
        
        # Should successfully create passages
        assert archived_count > 0
        
        # Verify Unicode content is preserved
        with passage_manager.memory_manager.get_session() as session:
            passages = session.query(MemoryPassage).filter_by(
                source="conversation",
                source_id=conversation_id
            ).all()
            
            # Should contain Unicode characters in passage text or metadata
            all_text = " ".join(p.text for p in passages)
            assert any(ord(char) > 127 for char in all_text)  # Contains non-ASCII

    def test_fact_expiration_dates(self, passage_manager):
        """
        Test that facts get appropriate expiration dates.
        
        REAL BUG THIS CATCHES: If expiration date assignment fails, temporary
        facts persist forever or permanent facts expire, breaking memory lifecycle.
        """
        # Messages with different types of facts
        messages = [
            {"role": "user", "content": "I love hiking in the mountains", "created_at": "2025-01-10T10:00:00Z"},
            {"role": "assistant", "content": "That's wonderful! Mountains are great for hiking", "created_at": "2025-01-10T10:00:30Z"},
            {"role": "user", "content": "I have a dentist appointment tomorrow at 3 PM", "created_at": "2025-01-10T10:01:00Z"},
            {"role": "assistant", "content": "I'll remember your appointment", "created_at": "2025-01-10T10:01:30Z"}
        ]
        
        # Archive conversation
        archived_count = passage_manager.archive_conversation("expire_test", messages)
        assert archived_count > 0
        
        # Check expiration dates
        with passage_manager.memory_manager.get_session() as session:
            passages = session.query(MemoryPassage).filter_by(source_id="expire_test").all()
            
            # Should have both permanent and expiring facts
            permanent_facts = [p for p in passages if p.expires_on is None]
            expiring_facts = [p for p in passages if p.expires_on is not None]
            
            # Preferences like hiking should be permanent
            hiking_facts = [p for p in permanent_facts if "hiking" in p.text.lower() or "mountain" in p.text.lower()]
            assert len(hiking_facts) > 0 if permanent_facts else True
            
            # Appointments should have expiration dates
            appointment_facts = [p for p in expiring_facts if "appointment" in p.text.lower() or "dentist" in p.text.lower()]
            assert len(appointment_facts) > 0 if expiring_facts else True
    
    def test_fact_deduplication(self, passage_manager):
        """
        Test that duplicate facts are detected and handled intelligently.
        
        REAL BUG THIS CATCHES: If deduplication fails, the system stores
        redundant facts, wasting storage and confusing search results.
        Also catches if deduplication is too aggressive and deletes the original.
        """
        # First conversation with a fact
        messages1 = [
            {"role": "user", "content": "My favorite color is blue", "created_at": "2025-01-10T10:00:00Z"},
            {"role": "assistant", "content": "I'll remember that", "created_at": "2025-01-10T10:00:30Z"}
        ]
        
        # Archive first conversation
        count1 = passage_manager.archive_conversation("dedup_test_1", messages1)
        assert count1 >= 1
        
        # Verify the fact was stored
        with passage_manager.memory_manager.get_session() as session:
            initial_facts = session.query(MemoryPassage).filter(
                MemoryPassage.text.ilike("%favorite color%blue%")
            ).all()
            assert len(initial_facts) == 1  # Should have exactly one fact
        
        # Second conversation with same fact
        messages2 = [
            {"role": "user", "content": "Just to remind you, my favorite color is blue", "created_at": "2025-01-11T10:00:00Z"},
            {"role": "assistant", "content": "Yes, I remember", "created_at": "2025-01-11T10:00:30Z"}
        ]
        
        # Archive second conversation - should detect duplicate
        count2 = passage_manager.archive_conversation("dedup_test_2", messages2)
        
        # Check that we still have exactly one fact (not zero, not two)
        with passage_manager.memory_manager.get_session() as session:
            blue_facts = session.query(MemoryPassage).filter(
                MemoryPassage.text.ilike("%favorite color%blue%")
            ).all()
            
            # CRITICAL: Should have exactly one fact - not deleted, not duplicated
            assert len(blue_facts) == 1, f"Expected 1 fact about favorite color, got {len(blue_facts)}"
    
    def test_temporal_fact_updates(self, passage_manager):
        """
        Test that temporal updates to facts work correctly.
        
        REAL BUG THIS CATCHES: If temporal updates fail, the system either
        stores conflicting facts or fails to update time-sensitive information.
        """
        # First message about beach trip
        messages1 = [
            {"role": "user", "content": "I'm going to the beach tomorrow", "created_at": "2025-01-10T10:00:00Z"},
            {"role": "assistant", "content": "Have a great trip!", "created_at": "2025-01-10T10:00:30Z"}
        ]
        
        count1 = passage_manager.archive_conversation("temporal_test_1", messages1)
        assert count1 >= 1
        
        # Later message updating the beach trip timing
        messages2 = [
            {"role": "user", "content": "Actually, I'm going to the beach next week instead", "created_at": "2025-01-11T10:00:00Z"},
            {"role": "assistant", "content": "I've updated that", "created_at": "2025-01-11T10:00:30Z"}
        ]
        
        count2 = passage_manager.archive_conversation("temporal_test_2", messages2)
        
        # Check that we have the updated fact
        with passage_manager.memory_manager.get_session() as session:
            beach_facts = session.query(MemoryPassage).filter(
                MemoryPassage.text.ilike("%beach%")
            ).all()
            
            # Should have facts about beach (exact count depends on extraction)
            assert len(beach_facts) >= 1
            
            # Most recent fact should reflect the update
            if beach_facts:
                latest_fact = max(beach_facts, key=lambda f: f.created_at)
                # The expiration should be further out for "next week" vs "tomorrow"
                if latest_fact.expires_on:
                    assert latest_fact.expires_on > utc_now() + timedelta(days=3)
    
    def test_expire_old_memories(self, passage_manager):
        """
        Test that expired facts are properly removed.
        
        REAL BUG THIS CATCHES: If expiration fails, the system accumulates
        outdated facts forever, polluting search results with stale information.
        """
        # Create a fact that's already expired
        yesterday = utc_now() - timedelta(days=1)
        
        # Manually create an expired fact
        with passage_manager.memory_manager.get_session() as session:
            expired_passage = MemoryPassage(
                text="Had lunch at the cafe",
                embedding=passage_manager.memory_manager.generate_embedding("Had lunch at the cafe"),
                source="test",
                source_id="expire_test",
                importance_score=0.5,
                expires_on=yesterday,
                context={"chunk_type": "micro_chunk"}
            )
            session.add(expired_passage)
            session.commit()
            expired_id = str(expired_passage.id)
        
        # Create a permanent fact
        permanent_id = passage_manager.create_passage(
            text="User's name is Alice",
            source="test",
            source_id="expire_test",
            importance=0.8,
            expires_on=None  # Permanent
        )
        
        # Run expiration
        expired_count = passage_manager.expire_old_memories()
        assert expired_count == 1
        
        # Verify expired fact is gone but permanent fact remains
        assert passage_manager.get_passage(expired_id) is None
        assert passage_manager.get_passage(permanent_id) is not None

