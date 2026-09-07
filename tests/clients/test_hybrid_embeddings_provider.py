"""
Tests for clients/hybrid_embeddings_provider.py

Tests the hybrid provider using mdbr-leaf-ir-asym (768d) and BGE reranker.
Following MIRA testing philosophy: no mocks, test real model behavior.
"""
import pytest
import numpy as np
from clients.hybrid_embeddings_provider import (
    HybridEmbeddingsProvider,
    EmbeddingCache,
    get_hybrid_embeddings_provider
)


@pytest.fixture(scope="module")
def hybrid_provider():
    """Shared hybrid provider with all models loaded."""
    return get_hybrid_embeddings_provider(cache_enabled=False, enable_reranker=True)


@pytest.fixture(scope="module")
def hybrid_provider_with_cache():
    """Hybrid provider with caching enabled."""
    provider = HybridEmbeddingsProvider(cache_enabled=True, enable_reranker=True)
    yield provider
    provider.close()


@pytest.fixture(scope="module")
def hybrid_provider_no_reranker():
    """Hybrid provider without reranker for testing fallback behavior."""
    provider = HybridEmbeddingsProvider(cache_enabled=False, enable_reranker=False)
    yield provider
    provider.close()


class TestHybridEmbeddingsProviderRealtimeEncoding:
    """Test query encoding (768-dim)."""

    def test_encode_realtime_single_string_returns_768_dim(self, hybrid_provider):
        """Verify realtime encoding returns 768-dimensional embedding."""
        text = "This is a test sentence."

        embedding = hybrid_provider.encode_realtime(text)

        # Should be 768-dim fp16
        assert embedding.shape == (768,)
        assert embedding.dtype == np.float16

    def test_encode_realtime_list_returns_2d_array(self, hybrid_provider):
        """Verify realtime encoding returns 2D array for list."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]

        embeddings = hybrid_provider.encode_realtime(texts)

        # Should be 2D array (3, 768) with fp16
        assert embeddings.ndim == 2
        assert embeddings.shape == (3, 768)
        assert embeddings.dtype == np.float16

    def test_encode_realtime_embeddings_are_normalized(self, hybrid_provider):
        """Verify realtime embeddings are normalized."""
        text = "Test text for normalization check"

        embedding = hybrid_provider.encode_realtime(text)

        # Convert to fp32 for norm calculation
        embedding_fp32 = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding_fp32)

        # Should be normalized (within fp16 tolerance)
        assert abs(norm - 1.0) < 0.01  # fp16 has lower precision

    def test_encode_realtime_contains_finite_values(self, hybrid_provider):
        """Verify realtime embeddings contain only finite values."""
        text = "Test text"

        embedding = hybrid_provider.encode_realtime(text)

        assert np.all(np.isfinite(embedding))
        assert not np.any(np.isnan(embedding))
        assert not np.any(np.isinf(embedding))


class TestHybridEmbeddingsProviderDeepEncoding:
    """Test OpenAI deep encoding (1024-dim)."""

    def test_encode_deep_single_string_returns_1024_dim(self, hybrid_provider):
        """Verify deep encoding returns 1024-dimensional embedding."""
        text = "This is a test sentence."

        embedding = hybrid_provider.encode_deep(text)

        # Should be 1024-dim fp16
        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float16

    def test_encode_deep_list_returns_2d_array(self, hybrid_provider):
        """Verify deep encoding returns 2D array for list."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]

        embeddings = hybrid_provider.encode_deep(texts)

        # Should be 2D array (3, 1024) with fp16
        assert embeddings.ndim == 2
        assert embeddings.shape == (3, 1024)
        assert embeddings.dtype == np.float16

    def test_encode_deep_contains_finite_values(self, hybrid_provider):
        """Verify deep embeddings contain only finite values."""
        text = "Test text"

        embedding = hybrid_provider.encode_deep(text)

        assert np.all(np.isfinite(embedding))
        assert not np.any(np.isnan(embedding))
        assert not np.any(np.isinf(embedding))


class TestHybridEmbeddingsProviderCaching:
    """Test embedding caching behavior."""

    def test_realtime_cache_hit_returns_same_embedding(self, hybrid_provider_with_cache):
        """Verify cached realtime embeddings are retrieved correctly."""
        text = "This text should be cached for realtime"

        # First call should compute and cache
        embedding1 = hybrid_provider_with_cache.encode_realtime(text)

        # Second call should hit cache
        embedding2 = hybrid_provider_with_cache.encode_realtime(text)

        # Should be identical
        np.testing.assert_array_equal(embedding1, embedding2)

    def test_deep_cache_hit_returns_same_embedding(self, hybrid_provider_with_cache):
        """Verify cached deep embeddings are retrieved correctly."""
        text = "This text should be cached for deep encoding"

        # First call should compute and cache
        embedding1 = hybrid_provider_with_cache.encode_deep(text)

        # Second call should hit cache
        embedding2 = hybrid_provider_with_cache.encode_deep(text)

        # Should be identical
        np.testing.assert_array_equal(embedding1, embedding2)

    def test_realtime_and_deep_caches_are_separate(self, hybrid_provider_with_cache):
        """Verify realtime and deep caches don't interfere (different dimensions)."""
        text = "Same text for both models"

        realtime_emb = hybrid_provider_with_cache.encode_realtime(text)
        deep_emb = hybrid_provider_with_cache.encode_deep(text)

        # Both use same dimension (768d) but different encoding modes
        assert realtime_emb.shape == (768,)
        assert deep_emb.shape == (768,)
        # Should not be exactly equal (query vs document encoding)
        assert not np.array_equal(realtime_emb, deep_emb)


class TestHybridEmbeddingsProviderReranking:
    """Test BGE reranking functionality."""

    def test_rerank_returns_sorted_results_with_scores(self, hybrid_provider):
        """Verify rerank returns results sorted by relevance."""
        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
            "Neural networks are used in deep learning.",
        ]

        results = hybrid_provider.rerank(query, passages, top_k=3)

        # Should return list of (index, score, passage) tuples
        assert len(results) == 3
        assert all(isinstance(item, tuple) and len(item) == 3 for item in results)

        # First result should have highest score
        scores = [score for _, score, _ in results]
        assert scores[0] >= scores[1] >= scores[2]

        # First passage should be most relevant
        assert results[0][0] == 0
        assert results[0][2] == passages[0]

    def test_rerank_respects_top_k_parameter(self, hybrid_provider):
        """Verify rerank returns only top_k results."""
        query = "What is machine learning?"
        passages = [f"Passage {i}" for i in range(10)]

        results = hybrid_provider.rerank(query, passages, top_k=3)

        # Should return only 3 results
        assert len(results) == 3

    def test_rerank_with_empty_passages_returns_empty_list(self, hybrid_provider):
        """Verify rerank handles empty passage list gracefully."""
        query = "What is machine learning?"
        passages = []

        results = hybrid_provider.rerank(query, passages)

        assert results == []

    def test_rerank_scores_are_between_0_and_1(self, hybrid_provider):
        """Verify rerank scores are valid probabilities."""
        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
        ]

        results = hybrid_provider.rerank(query, passages)

        for idx, score, passage in results:
            assert 0.0 <= score <= 1.0

    def test_rerank_without_reranker_raises_error(self, hybrid_provider_no_reranker):
        """Verify rerank raises error when reranker not enabled."""
        query = "What is machine learning?"
        passages = ["Machine learning is AI.", "Weather is nice."]

        with pytest.raises(RuntimeError, match="Reranker not available"):
            hybrid_provider_no_reranker.rerank(query, passages)

    def test_rerank_with_none_query_raises_error(self, hybrid_provider):
        """Verify rerank raises error for None query."""
        passages = ["Machine learning is AI.", "Weather is nice."]

        with pytest.raises(ValueError, match="Query cannot be None"):
            hybrid_provider.rerank(None, passages)


class TestHybridEmbeddingsProviderSearchAndRerank:
    """Test two-stage search and rerank functionality."""

    def test_search_and_rerank_with_fast_model(self, hybrid_provider):
        """Verify two-stage search using fast model."""
        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
            "Neural networks are used in deep learning.",
            "Python is a programming language.",
            "Deep learning requires large datasets.",
        ]

        results = hybrid_provider.search_and_rerank(
            query, passages, embedding_model="fast", initial_top_k=3, final_top_k=2
        )

        # Should return 2 results after reranking
        assert len(results) == 2
        assert all(isinstance(item, tuple) and len(item) == 3 for item in results)

        # Results should be sorted by rerank score
        scores = [score for _, score, _ in results]
        assert scores[0] >= scores[1]

    def test_search_and_rerank_with_deep_model(self, hybrid_provider):
        """Verify two-stage search using deep model."""
        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
            "Neural networks are used in deep learning.",
        ]

        results = hybrid_provider.search_and_rerank(
            query, passages, embedding_model="deep", initial_top_k=3, final_top_k=2
        )

        # Should return 2 results
        assert len(results) == 2
        assert all(isinstance(item, tuple) and len(item) == 3 for item in results)

    def test_search_and_rerank_with_precomputed_embeddings(self, hybrid_provider):
        """Verify search works with precomputed passage embeddings."""
        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
        ]

        # Precompute embeddings
        passage_embeddings = hybrid_provider.encode_realtime(passages)

        results = hybrid_provider.search_and_rerank(
            query, passages,
            passage_embeddings=passage_embeddings,
            embedding_model="fast",
            initial_top_k=2,
            final_top_k=2
        )

        # Should return 2 results
        assert len(results) == 2

    def test_search_and_rerank_without_reranker_falls_back_to_similarity(self, hybrid_provider_no_reranker):
        """Verify search falls back to embedding similarity when reranker disabled."""
        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
            "Neural networks are used in deep learning.",
        ]

        results = hybrid_provider_no_reranker.search_and_rerank(
            query, passages, embedding_model="fast", initial_top_k=3, final_top_k=2
        )

        # Should return 2 results using embedding similarity
        assert len(results) == 2
        assert all(isinstance(item, tuple) and len(item) == 3 for item in results)

    def test_search_and_rerank_returns_original_indices(self, hybrid_provider):
        """Verify returned indices map to original passage list."""
        query = "What is machine learning?"
        passages = [
            "The weather is nice today.",  # Index 0
            "Machine learning is a subset of artificial intelligence.",  # Index 1
            "Neural networks are used in deep learning.",  # Index 2
        ]

        results = hybrid_provider.search_and_rerank(
            query, passages, embedding_model="fast", initial_top_k=3, final_top_k=2
        )

        # Verify indices point to correct passages
        for idx, score, passage_text in results:
            assert passages[idx] == passage_text


class TestHybridEmbeddingsProviderSingleton:
    """Test singleton pattern."""

    def test_get_hybrid_embeddings_provider_returns_singleton(self):
        """Verify singleton factory returns same instance."""
        provider1 = get_hybrid_embeddings_provider(cache_enabled=False)
        provider2 = get_hybrid_embeddings_provider(cache_enabled=False)

        # Should be the exact same object
        assert provider1 is provider2


class TestHybridEmbeddingsProviderLifecycle:
    """Test provider lifecycle."""

    def test_close_method_exists(self):
        """Verify close() method can be called."""
        provider = HybridEmbeddingsProvider(cache_enabled=False, enable_reranker=True)

        # Should not raise exception
        provider.close()


class TestEmbeddingCache:
    """Test Valkey-backed embedding cache."""

    def test_cache_with_unavailable_valkey_degrades_gracefully(self):
        """Verify cache handles Valkey unavailability gracefully."""
        cache = EmbeddingCache(key_prefix="test")

        # If Valkey unavailable, operations should not crash
        embedding = np.random.randn(768).astype(np.float16)

        # These should not raise exceptions
        cache.set("test text", embedding)
        result = cache.get("test text")

        # Either returns cached value or None (depending on Valkey availability)
        assert result is None or isinstance(result, np.ndarray)

    def test_cache_key_generation_is_consistent(self):
        """Verify cache keys are consistent for same text."""
        cache = EmbeddingCache(key_prefix="test")

        key1 = cache._get_cache_key("same text")
        key2 = cache._get_cache_key("same text")

        assert key1 == key2

    def test_cache_key_generation_differs_for_different_text(self):
        """Verify cache keys differ for different texts."""
        cache = EmbeddingCache(key_prefix="test")

        key1 = cache._get_cache_key("text one")
        key2 = cache._get_cache_key("text two")

        assert key1 != key2


class TestHybridEmbeddingsProviderDimensionConsistency:
    """Test that models maintain their dimension contracts."""

    def test_realtime_always_returns_768_dimensions(self, hybrid_provider):
        """Verify realtime model consistently returns 768-dim."""
        texts = ["Short", "Medium length text here", "Very long text " * 50]

        for text in texts:
            embedding = hybrid_provider.encode_realtime(text)
            assert embedding.shape == (768,)

    def test_deep_always_returns_768_dimensions(self, hybrid_provider):
        """Verify deep model consistently returns 768-dim."""
        texts = ["Short", "Medium length text here", "Very long text " * 50]

        for text in texts:
            embedding = hybrid_provider.encode_deep(text)
            assert embedding.shape == (768,)
