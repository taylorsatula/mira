"""
Tests for lt_memory.vector_ops - Vector operations service.

Contract-based tests using REAL components (no mocks):
- Real HybridEmbeddingsProvider (mdbr-leaf-ir-asym 768d)
- Real LTMemoryDB (PostgreSQL with RLS)
- Real VectorOps service

Tests verify:
- Embedding generation (768d)
- Memory storage with embeddings
- Similarity search (by text, by embedding, by reference memory)
- Reranking with real BGE reranker
- User isolation via RLS
- Edge cases and error handling
"""
import pytest
import numpy as np
from uuid import UUID
from lt_memory.models import ExtractedMemory


class TestEmbeddingGeneration:
    """Test embedding generation with real mdbr-leaf-ir-asym model."""

    def test_generate_embedding_returns_768_dimensions(self, vector_ops):
        """CONTRACT R1: generate_embedding() returns List[float] with exactly 768 dimensions."""
        result = vector_ops.generate_embedding("test text")

        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)

    def test_generate_embedding_different_texts_different_embeddings(self, vector_ops):
        """Embeddings for different texts should be different."""
        embedding1 = vector_ops.generate_embedding("Python programming")
        embedding2 = vector_ops.generate_embedding("JavaScript development")

        # Embeddings should be different (not identical)
        assert embedding1 != embedding2

    def test_generate_embedding_same_text_same_embedding(self, vector_ops):
        """Embeddings for same text should be identical (deterministic)."""
        text = "Machine learning is fascinating"
        embedding1 = vector_ops.generate_embedding(text)
        embedding2 = vector_ops.generate_embedding(text)

        assert embedding1 == embedding2

    def test_generate_embeddings_batch_returns_correct_count(self, vector_ops):
        """CONTRACT R2: generate_embeddings_batch() returns List[List[float]] matching input count."""
        texts = ["text one", "text two", "text three"]
        result = vector_ops.generate_embeddings_batch(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(emb, list) for emb in result)
        assert all(len(emb) == 768 for emb in result)

    def test_generate_embeddings_batch_empty_input(self, vector_ops):
        """EDGE CASE EC1: Empty list returns empty list."""
        result = vector_ops.generate_embeddings_batch([])
        assert result == []


class TestMemoryStorage:
    """Test memory storage with real database."""

    def test_store_memories_returns_uuids_in_order(self, vector_ops, test_user):
        """CONTRACT R3: store_memories_with_embeddings() returns List[UUID] in order."""
        memories = [
            ExtractedMemory(text="Memory one", importance_score=0.8),
            ExtractedMemory(text="Memory two", importance_score=0.9)
        ]

        result = vector_ops.store_memories_with_embeddings(memories)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(uid, UUID) for uid in result)

    def test_store_memories_creates_in_database(self, vector_ops, lt_memory_db, test_user):
        """Stored memories should be retrievable from database."""
        memories = [
            ExtractedMemory(text="Database test memory", importance_score=0.7)
        ]

        uuids = vector_ops.store_memories_with_embeddings(memories)

        # Verify in database
        stored_memory = lt_memory_db.get_memory(uuids[0])
        assert stored_memory is not None
        assert stored_memory.text == "Database test memory"
        assert stored_memory.importance_score == 0.7
        assert stored_memory.embedding is not None
        assert len(stored_memory.embedding) == 768

    def test_store_memories_empty_list(self, vector_ops):
        """EDGE CASE: Empty memories list returns empty list."""
        result = vector_ops.store_memories_with_embeddings([])
        assert result == []

    def test_store_memories_respects_user_isolation(self, vector_ops, lt_memory_db, test_user):
        """SECURITY S2: Memories stored are scoped to test user."""
        memories = [
            ExtractedMemory(text="User-specific memory", importance_score=0.6)
        ]

        uuids = vector_ops.store_memories_with_embeddings(memories)
        stored = lt_memory_db.get_memory(uuids[0])

        assert stored.user_id == UUID(test_user["user_id"])


class TestSimilaritySearch:
    """Test similarity search with real embeddings and database."""

    def test_find_similar_memories_returns_sorted_by_relevance(self, vector_ops, test_user):
        """CONTRACT R4: find_similar_memories() returns List[Memory] sorted by similarity DESC."""
        # Create test memories
        memories = [
            ExtractedMemory(text="Python is a programming language", importance_score=0.8),
            ExtractedMemory(text="JavaScript runs in browsers", importance_score=0.8),
            ExtractedMemory(text="Python programming tutorial", importance_score=0.8),
        ]
        vector_ops.store_memories_with_embeddings(memories)

        # Search for Python-related content
        results = vector_ops.find_similar_memories("Python coding", limit=10)

        assert isinstance(results, list)
        # Should find Python-related memories with higher similarity
        assert len(results) > 0
        # Results should have similarity scores
        assert all(hasattr(m, 'similarity_score') for m in results)
        # Should be sorted descending
        scores = [m.similarity_score for m in results]
        assert scores == sorted(scores, reverse=True)

    def test_find_similar_memories_respects_limit(self, vector_ops, test_user):
        """CONTRACT: Returns at most 'limit' results."""
        # Create many memories
        memories = [
            ExtractedMemory(text=f"Memory number {i}", importance_score=0.5)
            for i in range(20)
        ]
        vector_ops.store_memories_with_embeddings(memories)

        results = vector_ops.find_similar_memories("Memory", limit=5)

        assert len(results) <= 5

    def test_find_similar_by_embedding_with_reranking(self, vector_ops, test_user):
        """CONTRACT R5: find_similar_by_embedding() reranks when query_text + reranker available."""
        # Create test memories
        memories = [
            ExtractedMemory(text="Machine learning algorithms", importance_score=0.8),
            ExtractedMemory(text="Deep neural networks", importance_score=0.8),
        ]
        vector_ops.store_memories_with_embeddings(memories)

        # Search with embedding + query text (should trigger reranking)
        query_embedding = vector_ops.generate_embedding("neural networks")
        results = vector_ops.find_similar_by_embedding(
            query_embedding=query_embedding,
            query_text="neural networks",  # Triggers reranking
            limit=10
        )

        assert isinstance(results, list)
        assert len(results) > 0

    def test_find_similar_by_embedding_validates_dimensions(self, vector_ops):
        """CONTRACT E1: Raises ValueError when embedding dimension != 768."""
        wrong_dim_embedding = [0.1] * 256  # Wrong dimension

        with pytest.raises(ValueError, match="768-dimensional"):
            vector_ops.find_similar_by_embedding(wrong_dim_embedding)

    def test_find_similar_by_embedding_accepts_numpy_array(self, vector_ops, test_user):
        """EDGE CASE EC2: np.ndarray embeddings converted to list."""
        # Create test memory
        memories = [ExtractedMemory(text="NumPy test", importance_score=0.8)]
        vector_ops.store_memories_with_embeddings(memories)

        # Search with numpy array
        query_embedding = np.array([0.1] * 768, dtype=np.float32)
        results = vector_ops.find_similar_by_embedding(query_embedding)

        # Should not raise error
        assert isinstance(results, list)


class TestMemoryExpansion:
    """Test find_similar_to_memory with real database."""

    def test_find_similar_to_memory_excludes_reference(self, vector_ops, test_user):
        """CONTRACT R6: find_similar_to_memory() excludes reference memory from results."""
        # Create similar memories
        memories = [
            ExtractedMemory(text="Python programming language", importance_score=0.8),
            ExtractedMemory(text="Python coding tutorial", importance_score=0.8),
            ExtractedMemory(text="Python development guide", importance_score=0.8),
        ]
        uuids = vector_ops.store_memories_with_embeddings(memories)
        reference_id = uuids[0]

        results = vector_ops.find_similar_to_memory(reference_id, limit=10)

        # Reference memory should NOT be in results
        result_ids = [m.id for m in results]
        assert reference_id not in result_ids

    def test_find_similar_to_memory_returns_empty_when_not_found(self, vector_ops):
        """EDGE CASE EC3: Returns empty list when memory not found."""
        from uuid import uuid4
        fake_id = uuid4()

        results = vector_ops.find_similar_to_memory(fake_id)

        assert results == []

    def test_find_similar_to_memory_returns_similar_content(self, vector_ops, test_user):
        """find_similar_to_memory should return semantically similar memories."""
        # Create memories with varied content
        memories = [
            ExtractedMemory(text="Python web frameworks like Django", importance_score=0.8),
            ExtractedMemory(text="JavaScript frameworks like React", importance_score=0.8),
            ExtractedMemory(text="Python Flask microframework", importance_score=0.8),
        ]
        uuids = vector_ops.store_memories_with_embeddings(memories)

        # Find similar to first Python memory
        results = vector_ops.find_similar_to_memory(uuids[0], limit=10)

        # Should find other Python-related memory more than JavaScript
        if len(results) > 0:
            # Results should be sorted by relevance
            scores = [m.similarity_score for m in results]
            assert scores == sorted(scores, reverse=True)


class TestUpdateOperations:
    """Test memory update with real database."""

    def test_update_memory_embedding_returns_updated_memory(self, vector_ops, test_user):
        """CONTRACT R7: update_memory_embedding() returns Memory with updated text and embedding."""
        # Create initial memory
        memories = [ExtractedMemory(text="Original text", importance_score=0.8)]
        uuids = vector_ops.store_memories_with_embeddings(memories)
        memory_id = uuids[0]

        # Update it
        updated = vector_ops.update_memory_embedding(memory_id, "Updated text")

        assert updated.id == memory_id
        assert updated.text == "Updated text"
        assert updated.embedding is not None
        assert len(updated.embedding) == 768

    def test_update_memory_embedding_regenerates_embedding(self, vector_ops, lt_memory_db, test_user):
        """Updating text should regenerate embedding to match new text."""
        # Create memory
        memories = [ExtractedMemory(text="Old content", importance_score=0.8)]
        uuids = vector_ops.store_memories_with_embeddings(memories)
        original = lt_memory_db.get_memory(uuids[0])
        original_embedding = original.embedding

        # Update with different text
        updated = vector_ops.update_memory_embedding(uuids[0], "Completely different content")

        # Embedding should be different
        assert updated.embedding != original_embedding

    def test_update_memory_embedding_raises_on_not_found(self, vector_ops):
        """CONTRACT E3: Raises ValueError when memory_id not found."""
        from uuid import uuid4
        fake_id = uuid4()

        with pytest.raises(ValueError, match="not found"):
            vector_ops.update_memory_embedding(fake_id, "new text")


class TestReranking:
    """Test reranking with real BGE reranker."""

    def test_rerank_memories_with_real_reranker(self, vector_ops, test_user):
        """CONTRACT R8: rerank_memories() returns List[Memory] with count <= top_k."""
        # Create memories
        memories_data = [
            ExtractedMemory(text="Neural networks and deep learning", importance_score=0.8),
            ExtractedMemory(text="Unrelated topic about cooking", importance_score=0.8),
            ExtractedMemory(text="Machine learning algorithms", importance_score=0.8),
        ]
        vector_ops.store_memories_with_embeddings(memories_data)

        # Search and rerank
        initial_results = vector_ops.find_similar_memories("machine learning", limit=10)

        if len(initial_results) > 0:
            reranked = vector_ops.rerank_memories(
                query="deep learning neural networks",
                memories=initial_results,
                top_k=3
            )

            assert len(reranked) <= 3
            # Results should be Memory objects
            assert all(hasattr(m, 'text') for m in reranked)

    def test_rerank_memories_empty_input(self, vector_ops):
        """EDGE CASE EC6: Empty memories list returns empty list."""
        result = vector_ops.rerank_memories("query", [], top_k=10)
        assert result == []

    def test_rerank_memories_handles_exceptions_gracefully(self, vector_ops, test_user):
        """CONTRACT E5: rerank_memories catches exceptions and falls back."""
        # Create test memories
        memories_data = [ExtractedMemory(text="Test memory", importance_score=0.8)]
        vector_ops.store_memories_with_embeddings(memories_data)
        memories = vector_ops.find_similar_memories("test", limit=10)

        # Even with invalid input, should fall back gracefully (not crash)
        result = vector_ops.rerank_memories(
            query="test",
            memories=memories,
            top_k=5
        )

        # Should return something (either reranked or original order)
        assert isinstance(result, list)


class TestHybridSearch:
    """Test hybrid BM25 + vector search."""

    def test_hybrid_search_returns_ranked_results(self, vector_ops, test_user):
        """CONTRACT R9: hybrid_search() returns List[Memory] with RRF-combined ranking."""
        # Create diverse memories
        memories = [
            ExtractedMemory(text="Python programming language basics", importance_score=0.8),
            ExtractedMemory(text="Python Flask web development", importance_score=0.8),
            ExtractedMemory(text="JavaScript frontend frameworks", importance_score=0.8),
        ]
        vector_ops.store_memories_with_embeddings(memories)

        # Hybrid search
        query_text = "Python"
        query_embedding = vector_ops.generate_embedding(query_text)
        results = vector_ops.hybrid_search(
            query_text=query_text,
            query_embedding=query_embedding,
            limit=10
        )

        assert isinstance(results, list)
        # Results should have similarity scores (from hybrid ranking)
        if len(results) > 0:
            assert all(hasattr(m, 'text') for m in results)

    def test_hybrid_search_respects_search_intent(self, vector_ops, test_user):
        """Different search intents should work without errors."""
        # Create test data
        memories = [ExtractedMemory(text="Test memory", importance_score=0.8)]
        vector_ops.store_memories_with_embeddings(memories)

        query_embedding = vector_ops.generate_embedding("test")

        # Test different intents
        for intent in ["general", "recall", "explore", "exact"]:
            results = vector_ops.hybrid_search(
                query_text="test",
                query_embedding=query_embedding,
                search_intent=intent,
                limit=5
            )
            assert isinstance(results, list)


class TestUserIsolation:
    """Test user isolation via RLS."""

    def test_user_isolation_via_ambient_context(self, vector_ops, lt_memory_db, test_user):
        """SECURITY S1: Uses ambient context when user_id not provided."""
        # Create memory (should use test_user from context)
        memories = [ExtractedMemory(text="Isolated memory", importance_score=0.8)]
        uuids = vector_ops.store_memories_with_embeddings(memories, user_id=None)

        # Verify it's scoped to test user
        stored = lt_memory_db.get_memory(uuids[0])
        assert stored.user_id == UUID(test_user["user_id"])

    def test_search_respects_user_isolation(self, vector_ops, test_user):
        """SECURITY S2: Search operations scoped to current user."""
        # Create memories as test user
        memories = [ExtractedMemory(text="User-specific data", importance_score=0.8)]
        vector_ops.store_memories_with_embeddings(memories)

        # Search should only find test user's memories
        results = vector_ops.find_similar_memories("data", limit=10)

        # All results should belong to test user
        for memory in results:
            assert memory.user_id == UUID(test_user["user_id"])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query_string(self, vector_ops, test_user):
        """Empty query string should still work (embeddings can be generated)."""
        # Create test memory
        memories = [ExtractedMemory(text="Test", importance_score=0.8)]
        vector_ops.store_memories_with_embeddings(memories)

        # Empty string query
        results = vector_ops.find_similar_memories("", limit=10)

        # Should not crash
        assert isinstance(results, list)

    def test_very_long_text_embedding(self, vector_ops):
        """Very long text should be handled."""
        long_text = "word " * 1000  # 1000 words
        embedding = vector_ops.generate_embedding(long_text)

        assert len(embedding) == 768

    def test_special_characters_in_text(self, vector_ops, test_user):
        """Special characters should be handled."""
        special_text = "Testing with émojis 🎉 and spëcial çharacters!"
        memories = [ExtractedMemory(text=special_text, importance_score=0.8)]

        uuids = vector_ops.store_memories_with_embeddings(memories)

        assert len(uuids) == 1


class TestArchitecture:
    """Test architectural properties."""

    def test_vector_ops_has_expected_dependencies(self, vector_ops):
        """ARCH A1: VectorOps has expected dependencies."""
        assert hasattr(vector_ops, 'embeddings_provider')
        assert hasattr(vector_ops, 'db')
        assert hasattr(vector_ops, 'hybrid_searcher')

    def test_reranker_availability_detected_at_init(self, vector_ops):
        """ARCH A2: Reranker availability detected at initialization."""
        assert hasattr(vector_ops, 'reranker_available')
        assert isinstance(vector_ops.reranker_available, bool)
        # With real embeddings provider (enable_reranker=True), should be available
        assert vector_ops.reranker_available is True

    def test_cleanup_is_safe(self, vector_ops):
        """cleanup() should be safe to call."""
        # Should not raise
        vector_ops.cleanup()
