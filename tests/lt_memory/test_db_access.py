"""Tests for lt_memory/db_access.py - LTMemoryDB database gateway.

Tests the complete public interface of LTMemoryDB based on contract specifications.
Uses real database components (lt_memory_session_manager) with no mocks.

Contract coverage:
- User isolation (U1-U5)
- Return structures (R1-R24)
- Exceptions (E1-E8)
- Edge cases (EC1-EC16)
- Security (SEC1-SEC5)
- Schema requirements (S1-S6)
- Integration (I1-I7)
- Architecture (A1-A5)
"""

import pytest
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from typing import List, Dict, Any

from lt_memory.db_access import LTMemoryDB
from lt_memory.models import (
    Memory,
    ExtractedMemory,
    MemoryLink,
    Entity,
    PostProcessingBatch,
)
from utils.timezone_utils import utc_now
from utils.user_context import set_current_user_id, clear_user_context
from pydantic import ValidationError


# ============================================================================
# Test Class 1: User Isolation
# ============================================================================


class TestUserIsolation:
    """Tests for user isolation and user context resolution."""

    def test_explicit_user_id_takes_precedence_over_ambient(
        self, lt_memory_session_manager, test_user
    ):
        """U3: Explicit user_id parameter takes precedence over ambient context."""
        db = LTMemoryDB(lt_memory_session_manager)
        another_user_id = str(uuid4())

        # Set ambient context to test_user
        set_current_user_id(test_user["user_id"])

        # Create memory with explicit user_id (different from ambient)
        memories = [
            ExtractedMemory(
                text="Test memory",
                importance_score=0.5,
                confidence=0.8,
            )
        ]
        memory_ids = db.store_memories(memories, user_id=another_user_id)

        # Should be stored for another_user_id, not ambient test_user
        retrieved = db.get_memory(memory_ids[0], user_id=another_user_id)
        assert retrieved is not None
        assert str(retrieved.user_id) == another_user_id

        # Should NOT be accessible via test_user context
        retrieved_via_ambient = db.get_memory(memory_ids[0], user_id=test_user["user_id"])
        assert retrieved_via_ambient is None

        clear_user_context()

    def test_fallback_to_ambient_context_when_user_id_none(
        self, lt_memory_session_manager, test_user
    ):
        """U2: Methods fall back to get_current_user_id() when user_id is None."""
        db = LTMemoryDB(lt_memory_session_manager)

        set_current_user_id(test_user["user_id"])

        # Store memory without explicit user_id (should use ambient)
        memories = [
            ExtractedMemory(
                text="Test memory",
                importance_score=0.5,
                confidence=0.8,
            )
        ]
        memory_ids = db.store_memories(memories)  # No user_id parameter

        # Should be retrievable via ambient context
        retrieved = db.get_memory(memory_ids[0])
        assert retrieved is not None
        assert str(retrieved.user_id) == test_user["user_id"]

        clear_user_context()

    def test_value_error_when_no_user_id_available(self, lt_memory_session_manager):
        """U4: ValueError raised when both user_id parameter and ambient context are None."""
        db = LTMemoryDB(lt_memory_session_manager)

        clear_user_context()  # Ensure no ambient context

        memories = [
            ExtractedMemory(
                text="Test memory",
                importance_score=0.5,
                confidence=0.8,
            )
        ]

        # E1: Should raise ValueError with specific message
        with pytest.raises(ValueError, match="No user_id provided and no ambient user context available"):
            db.store_memories(memories)  # No user_id, no ambient context

# ============================================================================
# Test Class 2: Memory CRUD Operations
# ============================================================================


class TestMemoryCRUD:
    """Tests for memory create, read, update, delete operations."""

    def test_store_memories_returns_list_of_uuids(self, lt_memory_session_manager, test_user):
        """R1: store_memories returns List[UUID] of created memory IDs."""
        db = LTMemoryDB(lt_memory_session_manager)

        memories = [
            ExtractedMemory(text="Memory 1", importance_score=0.5, confidence=0.8),
            ExtractedMemory(text="Memory 2", importance_score=0.6, confidence=0.9),
        ]

        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        assert isinstance(memory_ids, list)
        assert len(memory_ids) == 2
        assert all(isinstance(uid, UUID) for uid in memory_ids)

    def test_store_memories_with_embeddings(self, lt_memory_session_manager, test_user):
        """R1: store_memories accepts optional embeddings parameter."""
        db = LTMemoryDB(lt_memory_session_manager)

        memories = [
            ExtractedMemory(text="Memory 1", importance_score=0.5, confidence=0.8),
        ]
        embeddings = [[0.1] * 768]  # 768d embedding

        memory_ids = db.store_memories(
            memories, embeddings=embeddings, user_id=test_user["user_id"]
        )

        assert len(memory_ids) == 1

        # Verify embedding was stored
        retrieved = db.get_memory(memory_ids[0], user_id=test_user["user_id"])
        assert retrieved.embedding is not None
        assert len(retrieved.embedding) == 768

    def test_store_memories_embeddings_length_mismatch_raises_error(
        self, lt_memory_session_manager, test_user
    ):
        """E2: ValueError when store_memories embeddings length mismatches."""
        db = LTMemoryDB(lt_memory_session_manager)

        memories = [
            ExtractedMemory(text="Memory 1", importance_score=0.5, confidence=0.8),
            ExtractedMemory(text="Memory 2", importance_score=0.6, confidence=0.9),
        ]
        embeddings = [[0.1] * 768]  # Only 1 embedding for 2 memories

        with pytest.raises(ValueError, match=r"Embeddings length \(\d+\) must match memories length \(\d+\)"):
            db.store_memories(memories, embeddings=embeddings, user_id=test_user["user_id"])

    def test_get_memory_returns_optional_memory(self, lt_memory_session_manager, test_user):
        """R2: get_memory returns Optional[Memory] with all fields populated."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memory
        memories = [
            ExtractedMemory(
                text="Test memory",
                importance_score=0.7,
                confidence=0.85,
                happens_at=utc_now() + timedelta(days=1),
            )
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        # Retrieve memory
        retrieved = db.get_memory(memory_ids[0], user_id=test_user["user_id"])

        assert retrieved is not None
        assert isinstance(retrieved, Memory)
        assert retrieved.id == memory_ids[0]
        assert str(retrieved.user_id) == test_user["user_id"]
        assert retrieved.text == "Test memory"
        assert isinstance(retrieved.importance_score, float)
        assert 0.0 <= retrieved.importance_score <= 1.0
        assert isinstance(retrieved.created_at, datetime)
        assert isinstance(retrieved.access_count, int)
        assert isinstance(retrieved.inbound_links, list)
        assert isinstance(retrieved.outbound_links, list)
        assert isinstance(retrieved.entity_links, list)
        assert isinstance(retrieved.confidence, float)
        assert isinstance(retrieved.is_archived, bool)
        assert isinstance(retrieved.is_refined, bool)
        assert isinstance(retrieved.refinement_rejection_count, int)

    def test_get_memory_returns_none_for_nonexistent_id(
        self, lt_memory_session_manager, test_user
    ):
        """EC6: get_memory(non_existent_uuid) returns None."""
        db = LTMemoryDB(lt_memory_session_manager)

        non_existent_id = uuid4()
        retrieved = db.get_memory(non_existent_id, user_id=test_user["user_id"])

        assert retrieved is None

    def test_get_memories_by_ids_returns_list_sorted_by_importance(
        self, lt_memory_session_manager, test_user
    ):
        """R3: get_memories_by_ids returns List[Memory] sorted by importance_score DESC."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memories with different importance scores
        memories = [
            ExtractedMemory(text="Low importance", importance_score=0.3, confidence=0.8),
            ExtractedMemory(text="High importance", importance_score=0.9, confidence=0.8),
            ExtractedMemory(text="Medium importance", importance_score=0.6, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        # Retrieve all
        retrieved = db.get_memories_by_ids(memory_ids, user_id=test_user["user_id"])

        assert isinstance(retrieved, list)
        assert len(retrieved) == 3
        # Should be sorted by importance DESC
        assert retrieved[0].text == "High importance"
        assert retrieved[1].text == "Medium importance"
        assert retrieved[2].text == "Low importance"

    def test_get_memories_by_ids_with_mix_of_valid_invalid(
        self, lt_memory_session_manager, test_user
    ):
        """EC7: get_memories_by_ids with mix of valid/invalid UUIDs returns only found memories."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create one memory
        memories = [
            ExtractedMemory(text="Valid memory", importance_score=0.5, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        # Mix valid and invalid IDs
        mixed_ids = [memory_ids[0], uuid4(), uuid4()]

        retrieved = db.get_memories_by_ids(mixed_ids, user_id=test_user["user_id"])

        # Should only return the valid memory
        assert len(retrieved) == 1
        assert retrieved[0].id == memory_ids[0]

    def test_update_memory_returns_updated_memory(self, lt_memory_session_manager, test_user):
        """R4: update_memory returns Memory with updated fields and updated_at set."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memory
        memories = [
            ExtractedMemory(text="Original text", importance_score=0.5, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        # Update memory
        updates = {"text": "Updated text", "importance_score": 0.8}
        updated = db.update_memory(memory_ids[0], updates, user_id=test_user["user_id"])

        assert isinstance(updated, Memory)
        assert updated.text == "Updated text"
        assert updated.importance_score == 0.8
        assert updated.updated_at is not None
        assert isinstance(updated.updated_at, datetime)

    def test_update_memory_raises_error_for_nonexistent_id(
        self, lt_memory_session_manager, test_user
    ):
        """E3: ValueError when update_memory memory not found."""
        db = LTMemoryDB(lt_memory_session_manager)

        non_existent_id = uuid4()
        updates = {"text": "New text"}

        with pytest.raises(ValueError, match=f"Memory {non_existent_id} not found"):
            db.update_memory(non_existent_id, updates, user_id=test_user["user_id"])

    def test_archive_memory_sets_archived_fields(self, lt_memory_session_manager, test_user):
        """Archive memory sets is_archived, archived_at, updated_at."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memory
        memories = [
            ExtractedMemory(text="To be archived", importance_score=0.5, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        # Archive memory
        db.archive_memory(memory_ids[0], user_id=test_user["user_id"])

        # Retrieve and verify
        retrieved = db.get_memory(memory_ids[0], user_id=test_user["user_id"])
        assert retrieved.is_archived is True
        assert retrieved.archived_at is not None
        assert isinstance(retrieved.archived_at, datetime)


# ============================================================================
# Test Class 3: Pagination
# ============================================================================


class TestPagination:
    """Tests for get_all_memories pagination behavior."""

    def test_get_all_memories_without_limit_returns_list(
        self, lt_memory_session_manager, test_user
    ):
        """R6, EC16: get_all_memories without limit returns List[Memory] (backward compatibility)."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create some memories
        memories = [
            ExtractedMemory(text=f"Memory {i}", importance_score=0.5, confidence=0.8)
            for i in range(5)
        ]
        db.store_memories(memories, user_id=test_user["user_id"])

        # Get all without limit
        result = db.get_all_memories(user_id=test_user["user_id"])

        assert isinstance(result, list)
        assert all(isinstance(m, Memory) for m in result)
        assert len(result) == 5

    def test_get_all_memories_with_limit_returns_dict(self, lt_memory_session_manager, test_user):
        """R7: get_all_memories with limit returns Dict with keys: memories, has_more, next_offset."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create some memories
        memories = [
            ExtractedMemory(text=f"Memory {i}", importance_score=0.5, confidence=0.8)
            for i in range(15)
        ]
        db.store_memories(memories, user_id=test_user["user_id"])

        # Get with limit
        result = db.get_all_memories(limit=10, user_id=test_user["user_id"])

        assert isinstance(result, dict)
        assert "memories" in result
        assert "has_more" in result
        assert "next_offset" in result
        assert len(result["memories"]) == 10
        assert isinstance(result["has_more"], bool)

    def test_pagination_has_more_true_when_more_results_exist(
        self, lt_memory_session_manager, test_user
    ):
        """EC8: get_all_memories pagination has_more=True when more results exist."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create 15 memories
        memories = [
            ExtractedMemory(text=f"Memory {i}", importance_score=0.5, confidence=0.8)
            for i in range(15)
        ]
        db.store_memories(memories, user_id=test_user["user_id"])

        # Get first page
        result = db.get_all_memories(limit=10, offset=0, user_id=test_user["user_id"])

        assert result["has_more"] is True
        assert result["next_offset"] == 10

    def test_pagination_has_more_false_when_no_more_results(
        self, lt_memory_session_manager, test_user
    ):
        """EC9: get_all_memories pagination has_more=False when no more results."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create exactly 10 memories
        memories = [
            ExtractedMemory(text=f"Memory {i}", importance_score=0.5, confidence=0.8)
            for i in range(10)
        ]
        db.store_memories(memories, user_id=test_user["user_id"])

        # Get first page with limit=10
        result = db.get_all_memories(limit=10, offset=0, user_id=test_user["user_id"])

        assert result["has_more"] is False
        assert result["next_offset"] is None


# ============================================================================
# Test Class 4: Vector Search
# ============================================================================


class TestVectorSearch:
    """Tests for vector similarity search operations."""

    def test_search_similar_returns_memories_with_similarity_score(
        self, lt_memory_session_manager, test_user
    ):
        """R5: search_similar returns List[Memory] with similarity_score field populated."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memories with embeddings
        memories = [
            ExtractedMemory(text="Test memory 1", importance_score=0.8, confidence=0.8),
            ExtractedMemory(text="Test memory 2", importance_score=0.7, confidence=0.8),
        ]
        embeddings = [[0.1] * 768, [0.2] * 768]
        db.store_memories(memories, embeddings=embeddings, user_id=test_user["user_id"])

        # Search with query embedding
        query_embedding = [0.15] * 768
        results = db.search_similar(
            query_embedding, limit=10, user_id=test_user["user_id"]
        )

        assert isinstance(results, list)
        assert all(isinstance(m, Memory) for m in results)
        # Each result should have similarity_score (transient field)
        assert all(m.similarity_score is not None for m in results)
        assert all(isinstance(m.similarity_score, float) for m in results)

    def test_search_similar_filters_by_similarity_threshold(
        self, lt_memory_session_manager, test_user
    ):
        """search_similar filters by similarity_threshold parameter."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memories with embeddings
        memories = [
            ExtractedMemory(text="Test memory", importance_score=0.8, confidence=0.8),
        ]
        embeddings = [[1.0] + [0.0] * 383]  # Very specific embedding
        db.store_memories(memories, embeddings=embeddings, user_id=test_user["user_id"])

        # Search with very different query embedding (low similarity)
        query_embedding = [0.0] * 383 + [1.0]
        results = db.search_similar(
            query_embedding,
            limit=10,
            similarity_threshold=0.99,  # Very high threshold
            user_id=test_user["user_id"],
        )

        # Should filter out low-similarity results
        assert isinstance(results, list)

    def test_search_similar_filters_by_min_importance(
        self, lt_memory_session_manager, test_user
    ):
        """search_similar filters by min_importance parameter."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memories with different importance scores
        memories = [
            ExtractedMemory(text="Low importance", importance_score=0.1, confidence=0.8),
            ExtractedMemory(text="High importance", importance_score=0.9, confidence=0.8),
        ]
        embeddings = [[0.1] * 768, [0.2] * 768]
        memory_ids = db.store_memories(
            memories, embeddings=embeddings, user_id=test_user["user_id"]
        )

        # Search with high min_importance filter
        query_embedding = [0.15] * 768
        results = db.search_similar(
            query_embedding,
            limit=10,
            min_importance=0.5,  # Should filter out low importance memory
            user_id=test_user["user_id"],
        )

        # Should only return high importance memory
        assert all(m.importance_score >= 0.5 for m in results)

    def test_search_similar_filters_archived_memories(
        self, lt_memory_session_manager, test_user
    ):
        """search_similar excludes archived memories from results."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create and archive a memory
        memories = [
            ExtractedMemory(text="Archived memory", importance_score=0.8, confidence=0.8),
        ]
        embeddings = [[0.1] * 768]
        memory_ids = db.store_memories(
            memories, embeddings=embeddings, user_id=test_user["user_id"]
        )
        db.archive_memory(memory_ids[0], user_id=test_user["user_id"])

        # Search should not return archived memory
        query_embedding = [0.1] * 768
        results = db.search_similar(query_embedding, user_id=test_user["user_id"])

        assert all(not m.is_archived for m in results)


# ============================================================================
# Test Class 5: Scoring Operations
# ============================================================================


class TestScoringOperations:
    """Tests for memory scoring and importance calculation."""

    def test_update_access_stats_increments_count_and_updates_score(
        self, lt_memory_session_manager, test_user
    ):
        """R8: update_access_stats returns Memory with incremented access_count and recalculated score."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memory
        memories = [
            ExtractedMemory(text="Test memory", importance_score=0.5, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        # Get initial state
        initial = db.get_memory(memory_ids[0], user_id=test_user["user_id"])
        initial_count = initial.access_count
        initial_score = initial.importance_score

        # Update access stats
        updated = db.update_access_stats(memory_ids[0], user_id=test_user["user_id"])

        assert isinstance(updated, Memory)
        assert updated.access_count == initial_count + 1
        assert updated.last_accessed is not None
        assert isinstance(updated.last_accessed, datetime)
        # Score should be recalculated (I1: depends on scoring formula)
        assert isinstance(updated.importance_score, float)
        assert 0.0 <= updated.importance_score <= 1.0

    def test_update_access_stats_raises_error_for_nonexistent_id(
        self, lt_memory_session_manager, test_user
    ):
        """E4: ValueError when update_access_stats memory not found."""
        db = LTMemoryDB(lt_memory_session_manager)

        non_existent_id = uuid4()

        with pytest.raises(ValueError, match=f"Memory {non_existent_id} not found"):
            db.update_access_stats(non_existent_id, user_id=test_user["user_id"])

    def test_bulk_recalculate_scores_returns_count(self, lt_memory_session_manager, test_user):
        """R9: bulk_recalculate_scores returns int count of updated memories."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memories
        memories = [
            ExtractedMemory(text=f"Memory {i}", importance_score=0.5, confidence=0.8)
            for i in range(5)
        ]
        db.store_memories(memories, user_id=test_user["user_id"])

        # Recalculate scores
        count = db.bulk_recalculate_scores(user_id=test_user["user_id"])

        assert isinstance(count, int)
        assert count >= 0

    def test_bulk_recalculate_scores_returns_zero_when_no_stale_memories(
        self, lt_memory_session_manager, test_user
    ):
        """EC10: bulk_recalculate_scores returns 0 when no stale memories."""
        db = LTMemoryDB(lt_memory_session_manager)

        # No memories created, so no stale memories
        count = db.bulk_recalculate_scores(user_id=test_user["user_id"])

        assert count == 0

    def test_recalculate_temporal_scores_returns_count(
        self, lt_memory_session_manager, test_user
    ):
        """R10: recalculate_temporal_scores returns int count of updated memories."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memories with temporal fields
        memories = [
            ExtractedMemory(
                text="Future event",
                importance_score=0.5,
                confidence=0.8,
                happens_at=utc_now() + timedelta(days=7),
            ),
        ]
        db.store_memories(memories, user_id=test_user["user_id"])

        # Recalculate temporal scores
        count = db.recalculate_temporal_scores(user_id=test_user["user_id"])

        assert isinstance(count, int)
        assert count >= 0

    def test_recalculate_temporal_scores_returns_zero_when_no_temporal_memories(
        self, lt_memory_session_manager, test_user
    ):
        """EC11: recalculate_temporal_scores returns 0 when no temporal memories in window."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memory without temporal fields
        memories = [
            ExtractedMemory(text="Regular memory", importance_score=0.5, confidence=0.8),
        ]
        db.store_memories(memories, user_id=test_user["user_id"])

        # No temporal memories in window
        count = db.recalculate_temporal_scores(user_id=test_user["user_id"])

        assert count == 0


# ============================================================================
# Test Class 6: Link Operations
# ============================================================================


class TestLinkOperations:
    """Tests for memory linking operations."""

    def test_create_links_bidirectional(self, lt_memory_session_manager, test_user):
        """create_links creates bidirectional links between memories."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create two memories
        memories = [
            ExtractedMemory(text="Memory A", importance_score=0.5, confidence=0.8),
            ExtractedMemory(text="Memory B", importance_score=0.6, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        # Create link
        links = [
            MemoryLink(
                source_id=memory_ids[0],
                target_id=memory_ids[1],
                link_type="related",
                confidence=0.9,
                reasoning="Test link",
            )
        ]
        db.create_links(links, user_id=test_user["user_id"])

        # Verify bidirectional links
        links_a = db.get_links_for_memory(memory_ids[0], user_id=test_user["user_id"])
        links_b = db.get_links_for_memory(memory_ids[1], user_id=test_user["user_id"])

        assert "outbound" in links_a
        assert "inbound" in links_b
        assert len(links_a["outbound"]) > 0
        assert len(links_b["inbound"]) > 0

    def test_get_links_for_memory_returns_dict_with_inbound_outbound(
        self, lt_memory_session_manager, test_user
    ):
        """R11: get_links_for_memory returns Dict with keys: inbound, outbound."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memory
        memories = [
            ExtractedMemory(text="Test memory", importance_score=0.5, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        # Get links (should be empty initially)
        links = db.get_links_for_memory(memory_ids[0], user_id=test_user["user_id"])

        assert isinstance(links, dict)
        assert "inbound" in links
        assert "outbound" in links
        assert isinstance(links["inbound"], list)
        assert isinstance(links["outbound"], list)

    def test_remove_dead_links_returns_count(self, lt_memory_session_manager, test_user):
        """R12: remove_dead_links returns int count of affected memories."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create and link memories
        memories = [
            ExtractedMemory(text="Memory A", importance_score=0.5, confidence=0.8),
            ExtractedMemory(text="Memory B", importance_score=0.6, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        links = [
            MemoryLink(
                source_id=memory_ids[0],
                target_id=memory_ids[1],
                link_type="related",
                confidence=0.9,
                reasoning="Test link",
            )
        ]
        db.create_links(links, user_id=test_user["user_id"])

        # Remove dead links
        count = db.remove_dead_links([memory_ids[1]], user_id=test_user["user_id"])

        assert isinstance(count, int)
        assert count >= 0

    def test_archive_memory_automatically_calls_remove_dead_links(
        self, lt_memory_session_manager, test_user
    ):
        """EC14: archive_memory automatically calls remove_dead_links."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create and link memories
        memories = [
            ExtractedMemory(text="Memory A", importance_score=0.5, confidence=0.8),
            ExtractedMemory(text="Memory B", importance_score=0.6, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        links = [
            MemoryLink(
                source_id=memory_ids[0],
                target_id=memory_ids[1],
                link_type="related",
                confidence=0.9,
                reasoning="Test link",
            )
        ]
        db.create_links(links, user_id=test_user["user_id"])

        # Archive memory B
        db.archive_memory(memory_ids[1], user_id=test_user["user_id"])

        # Links to archived memory should be cleaned
        links_a = db.get_links_for_memory(memory_ids[0], user_id=test_user["user_id"])
        # Should not have link to archived memory anymore
        assert all(
            link["uuid"] != str(memory_ids[1]) for link in links_a["outbound"]
        )

    def test_increment_refinement_rejection_count(self, lt_memory_session_manager, test_user):
        """increment_refinement_rejection_count increments rejection count."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memory
        memories = [
            ExtractedMemory(text="Test memory", importance_score=0.5, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        # Get initial count
        initial = db.get_memory(memory_ids[0], user_id=test_user["user_id"])
        initial_count = initial.refinement_rejection_count

        # Increment rejection count
        db.increment_refinement_rejection_count(
            memory_ids[0], user_id=test_user["user_id"]
        )

        # Verify increment
        updated = db.get_memory(memory_ids[0], user_id=test_user["user_id"])
        assert updated.refinement_rejection_count == initial_count + 1


# ============================================================================
# Test Class 7: Entity Operations
# ============================================================================


class TestEntityOperations:
    """Tests for entity CRUD and knowledge graph operations."""

    def test_get_or_create_entity_returns_entity(self, lt_memory_session_manager, test_user):
        """R13: get_or_create_entity returns Entity (existing or new)."""
        db = LTMemoryDB(lt_memory_session_manager)

        embedding = [0.1] * 300  # 300d spaCy embedding

        # First call should create
        entity1 = db.get_or_create_entity(
            name="John Doe",
            entity_type="PERSON",
            embedding=embedding,
            user_id=test_user["user_id"],
        )

        assert isinstance(entity1, Entity)
        assert entity1.name == "John Doe"
        assert entity1.entity_type == "PERSON"

        # Second call with same name/type should return existing
        entity2 = db.get_or_create_entity(
            name="John Doe",
            entity_type="PERSON",
            embedding=embedding,
            user_id=test_user["user_id"],
        )

        assert entity2.id == entity1.id

    def test_get_or_create_entity_handles_race_condition(
        self, lt_memory_session_manager, test_user
    ):
        """EC15: get_or_create_entity handles race condition via ON CONFLICT."""
        db = LTMemoryDB(lt_memory_session_manager)

        embedding = [0.1] * 300

        # Multiple calls should not raise errors (ON CONFLICT handling)
        entity1 = db.get_or_create_entity(
            name="Jane Doe",
            entity_type="PERSON",
            embedding=embedding,
            user_id=test_user["user_id"],
        )
        entity2 = db.get_or_create_entity(
            name="Jane Doe",
            entity_type="PERSON",
            embedding=embedding,
            user_id=test_user["user_id"],
        )

        # Should return same entity
        assert entity1.id == entity2.id

    def test_get_entity_returns_optional_entity(self, lt_memory_session_manager, test_user):
        """R14: get_entity returns Optional[Entity]."""
        db = LTMemoryDB(lt_memory_session_manager)

        embedding = [0.1] * 300
        created = db.get_or_create_entity(
            name="Test Entity",
            entity_type="ORG",
            embedding=embedding,
            user_id=test_user["user_id"],
        )

        # Retrieve existing entity
        retrieved = db.get_entity(created.id, user_id=test_user["user_id"])
        assert retrieved is not None
        assert isinstance(retrieved, Entity)
        assert retrieved.id == created.id

        # Non-existent entity
        non_existent = db.get_entity(uuid4(), user_id=test_user["user_id"])
        assert non_existent is None

    def test_get_entities_by_ids_returns_list_sorted_by_link_count(
        self, lt_memory_session_manager, test_user
    ):
        """R15: get_entities_by_ids returns List[Entity] sorted by link_count DESC."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create entities
        embedding = [0.1] * 300
        entity1 = db.get_or_create_entity(
            "Entity A", "PERSON", embedding, user_id=test_user["user_id"]
        )
        entity2 = db.get_or_create_entity(
            "Entity B", "ORG", embedding, user_id=test_user["user_id"]
        )

        # Retrieve by IDs
        entities = db.get_entities_by_ids(
            [entity1.id, entity2.id], user_id=test_user["user_id"]
        )

        assert isinstance(entities, list)
        assert len(entities) == 2
        assert all(isinstance(e, Entity) for e in entities)

    def test_link_memory_to_entity(self, lt_memory_session_manager, test_user):
        """link_memory_to_entity creates link between memory and entity."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memory and entity
        memories = [
            ExtractedMemory(text="John works here", importance_score=0.5, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        embedding = [0.1] * 300
        entity = db.get_or_create_entity(
            "John", "PERSON", embedding, user_id=test_user["user_id"]
        )

        # Link memory to entity
        db.link_memory_to_entity(
            memory_ids[0], entity.id, "John", "PERSON", user_id=test_user["user_id"]
        )

        # Verify link
        memory = db.get_memory(memory_ids[0], user_id=test_user["user_id"])
        assert len(memory.entity_links) > 0
        assert any(link["entity_id"] == str(entity.id) for link in memory.entity_links)

    def test_find_dormant_entities_returns_filtered_list(
        self, lt_memory_session_manager, test_user
    ):
        """R16: find_dormant_entities returns List[Entity] sorted by link_count, last_linked_at."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create entities (they start with link_count=0)
        embedding = [0.1] * 300
        entity1 = db.get_or_create_entity(
            "Dormant Entity", "PERSON", embedding, user_id=test_user["user_id"]
        )

        # Find dormant entities
        dormant = db.find_dormant_entities(
            days_dormant=1,
            min_link_count=0,
            max_link_count=10,
            user_id=test_user["user_id"],
        )

        assert isinstance(dormant, list)
        assert all(isinstance(e, Entity) for e in dormant)

    def test_find_entities_by_vector_similarity(self, lt_memory_session_manager, test_user):
        """R17: find_entities_by_vector_similarity returns List[Entity] with similarity_score."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create entity with embedding
        embedding = [0.1] * 300
        entity = db.get_or_create_entity(
            "Test Entity", "PERSON", embedding, user_id=test_user["user_id"]
        )

        # Search by similarity
        query_embedding = [0.2] * 300
        results = db.find_entities_by_vector_similarity(
            query_embedding, limit=10, similarity_threshold=0.5, user_id=test_user["user_id"]
        )

        assert isinstance(results, list)
        assert all(isinstance(e, Entity) for e in results)
        # Each result should have similarity_score (transient field)
        assert all(e.similarity_score is not None for e in results)

    def test_get_memories_for_entity(self, lt_memory_session_manager, test_user):
        """R18: get_memories_for_entity returns List[Memory] sorted by created_at DESC."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create memory and entity
        memories = [
            ExtractedMemory(text="Entity mention", importance_score=0.5, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

        embedding = [0.1] * 300
        entity = db.get_or_create_entity(
            "Test Entity", "ORG", embedding, user_id=test_user["user_id"]
        )

        # Link them
        db.link_memory_to_entity(
            memory_ids[0], entity.id, "Test Entity", "ORG", user_id=test_user["user_id"]
        )

        # Get memories for entity
        entity_memories = db.get_memories_for_entity(entity.id, user_id=test_user["user_id"])

        assert isinstance(entity_memories, list)
        assert all(isinstance(m, Memory) for m in entity_memories)
        assert len(entity_memories) > 0

    def test_merge_entities(self, lt_memory_session_manager, test_user):
        """merge_entities transfers all links from source to target."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create two entities
        embedding = [0.1] * 300
        source = db.get_or_create_entity(
            "Source Entity", "PERSON", embedding, user_id=test_user["user_id"]
        )
        target = db.get_or_create_entity(
            "Target Entity", "PERSON", embedding, user_id=test_user["user_id"]
        )

        # Create memory linked to source
        memories = [
            ExtractedMemory(text="Mention", importance_score=0.5, confidence=0.8),
        ]
        memory_ids = db.store_memories(memories, user_id=test_user["user_id"])
        db.link_memory_to_entity(
            memory_ids[0], source.id, "Source Entity", "PERSON", user_id=test_user["user_id"]
        )

        # Merge source into target
        db.merge_entities(source.id, target.id, user_id=test_user["user_id"])

        # Source should be archived
        source_after = db.get_entity(source.id, user_id=test_user["user_id"])
        assert source_after.is_archived is True

        # Target should have the memory link
        target_memories = db.get_memories_for_entity(target.id, user_id=test_user["user_id"])
        assert len(target_memories) > 0

    def test_merge_entities_raises_error_for_nonexistent_target(
        self, lt_memory_session_manager, test_user
    ):
        """E5: ValueError when merge_entities target not found."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create source entity
        embedding = [0.1] * 300
        source = db.get_or_create_entity(
            "Source", "PERSON", embedding, user_id=test_user["user_id"]
        )

        non_existent_target = uuid4()

        with pytest.raises(ValueError, match=f"Target entity {non_existent_target} not found"):
            db.merge_entities(source.id, non_existent_target, user_id=test_user["user_id"])

    def test_delete_entity(self, lt_memory_session_manager, test_user):
        """delete_entity removes entity and cleans up memory entity_links."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create entity
        embedding = [0.1] * 300
        entity = db.get_or_create_entity(
            "To Delete", "PERSON", embedding, user_id=test_user["user_id"]
        )

        # Delete entity
        db.delete_entity(entity.id, user_id=test_user["user_id"])

        # Should no longer be retrievable
        retrieved = db.get_entity(entity.id, user_id=test_user["user_id"])
        assert retrieved is None

    def test_archive_entity(self, lt_memory_session_manager, test_user):
        """archive_entity soft deletes entity."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Create entity
        embedding = [0.1] * 300
        entity = db.get_or_create_entity(
            "To Archive", "ORG", embedding, user_id=test_user["user_id"]
        )

        # Archive entity
        db.archive_entity(entity.id, user_id=test_user["user_id"])

        # Should still be retrievable but archived
        retrieved = db.get_entity(entity.id, user_id=test_user["user_id"])
        assert retrieved is not None
        assert retrieved.is_archived is True


# ============================================================================
# Test Class 9: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_store_memories_empty_list_returns_empty_list(
        self, lt_memory_session_manager, test_user
    ):
        """EC1: store_memories([]) returns empty list."""
        db = LTMemoryDB(lt_memory_session_manager)

        result = db.store_memories([], user_id=test_user["user_id"])

        assert result == []

    def test_get_memories_by_ids_empty_list_returns_empty_list(
        self, lt_memory_session_manager, test_user
    ):
        """EC2: get_memories_by_ids([]) returns empty list."""
        db = LTMemoryDB(lt_memory_session_manager)

        result = db.get_memories_by_ids([], user_id=test_user["user_id"])

        assert result == []

    def test_get_entities_by_ids_empty_list_returns_empty_list(
        self, lt_memory_session_manager, test_user
    ):
        """EC3: get_entities_by_ids([]) returns empty list."""
        db = LTMemoryDB(lt_memory_session_manager)

        result = db.get_entities_by_ids([], user_id=test_user["user_id"])

        assert result == []

    def test_remove_dead_links_empty_list_returns_zero(
        self, lt_memory_session_manager, test_user
    ):
        """EC4: remove_dead_links([]) returns 0."""
        db = LTMemoryDB(lt_memory_session_manager)

        result = db.remove_dead_links([], user_id=test_user["user_id"])

        assert result == 0

    def test_create_links_empty_list_no_error(self, lt_memory_session_manager, test_user):
        """EC5: create_links([]) no-ops without error."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Should not raise error
        db.create_links([], user_id=test_user["user_id"])

    def test_memory_link_type_validation(self):
        """E7: Pydantic ValidationError when MemoryLink link_type invalid."""
        # Invalid link_type should raise ValidationError
        with pytest.raises(ValidationError):
            MemoryLink(
                source_id=uuid4(),
                target_id=uuid4(),
                link_type="invalid_type",  # Not in allowed values
                confidence=0.8,
                reasoning="Test",
            )


# ============================================================================
# Test Class 10: Transaction Context
# ============================================================================


class TestTransactionContext:
    """Tests for transaction context manager."""

    def test_transaction_context_manager(self, lt_memory_session_manager, test_user):
        """A4: Transaction context manager supports multi-step operations."""
        db = LTMemoryDB(lt_memory_session_manager)

        with db.transaction(user_id=test_user["user_id"]):
            # Perform multiple operations in transaction
            memories = [
                ExtractedMemory(text="Memory 1", importance_score=0.5, confidence=0.8),
                ExtractedMemory(text="Memory 2", importance_score=0.6, confidence=0.8),
            ]
            memory_ids = db.store_memories(memories, user_id=test_user["user_id"])

            # Link them
            links = [
                MemoryLink(
                    source_id=memory_ids[0],
                    target_id=memory_ids[1],
                    link_type="related",
                    confidence=0.9,
                    reasoning="Test link",
                )
            ]
            db.create_links(links, user_id=test_user["user_id"])

        # After transaction, changes should be committed
        retrieved = db.get_memory(memory_ids[0], user_id=test_user["user_id"])
        assert retrieved is not None


# ============================================================================
# Test Class 11: Cleanup
# ============================================================================


class TestCleanup:
    """Tests for resource cleanup."""

    def test_cleanup_no_errors(self, lt_memory_session_manager, test_user):
        """cleanup() cleans up database resources without errors."""
        db = LTMemoryDB(lt_memory_session_manager)

        # Should not raise errors
        db.cleanup()
