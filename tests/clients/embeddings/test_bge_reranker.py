"""
Tests for clients/embeddings/bge_reranker.py

Tests BGE reranking functionality with real ONNX inference.
Following MIRA testing philosophy: no mocks, test real model behavior.

Note: This file only tests reranking - embeddings come from mdbr-leaf-ir-asym.
"""
import pytest
import numpy as np
from clients.embeddings.bge_reranker import (
    BGEReranker,
    BGERerankerPool,
    get_bge_reranker
)


@pytest.fixture(scope="module")
def bge_reranker():
    """Shared BGE reranker for tests."""
    return BGEReranker(model_name="BAAI/bge-reranker-base", use_fp16=True)


@pytest.fixture(scope="module")
def bge_reranker_pool():
    """Shared BGE reranker pool with single instance for tests."""
    pool = BGERerankerPool(pool_size=1, model_name="BAAI/bge-reranker-base", use_fp16=True)
    yield pool
    pool.close()


class TestBGERerankerBasics:
    """Test basic reranking functionality."""

    def test_rerank_returns_sorted_indices_with_scores(self, bge_reranker):
        """Verify rerank returns indices sorted by relevance scores."""
        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
            "Neural networks are used in deep learning.",
        ]

        results = bge_reranker.rerank(query, passages, return_scores=True)

        # Should return list of (index, score) tuples
        assert len(results) == 3
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)

        # First result should have highest score
        indices = [idx for idx, _ in results]
        scores = [score for _, score in results]
        assert scores[0] >= scores[1] >= scores[2]

        # First passage should be most relevant
        assert indices[0] == 0

    def test_rerank_returns_only_indices_when_return_scores_false(self, bge_reranker):
        """Verify rerank returns only indices when return_scores=False."""
        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
        ]

        results = bge_reranker.rerank(query, passages, return_scores=False)

        # Should return list of indices only
        assert len(results) == 2
        assert all(isinstance(idx, int) for idx in results)

    def test_rerank_with_empty_passages_returns_empty_list(self, bge_reranker):
        """Verify rerank handles empty passage list gracefully."""
        query = "What is machine learning?"
        passages = []

        results = bge_reranker.rerank(query, passages)

        assert results == []

    def test_rerank_scores_are_between_0_and_1(self, bge_reranker):
        """Verify rerank scores are in valid probability range (sigmoid output)."""
        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
        ]

        results = bge_reranker.rerank(query, passages, return_scores=True)

        for idx, score in results:
            assert 0.0 <= score <= 1.0

    def test_compute_relevance_scores_returns_array_in_original_order(self, bge_reranker):
        """Verify compute_relevance_scores returns scores in original passage order."""
        query = "What is machine learning?"
        passages = [
            "The weather is nice today.",  # Low relevance
            "Machine learning is a subset of artificial intelligence.",  # High relevance
            "Neural networks are used in deep learning.",  # Medium relevance
        ]

        scores = bge_reranker.compute_relevance_scores(query, passages)

        # Should return numpy array with 3 scores
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (3,)

        # Scores should be in original order (not sorted)
        # Second passage (index 1) should have highest score
        assert scores[1] > scores[0]
        assert scores[1] > scores[2]


class TestBGERerankerBatchProcessing:
    """Test batch processing in reranking."""

    def test_rerank_with_large_passage_list(self, bge_reranker):
        """Verify reranking handles large passage lists with batching."""
        query = "What is machine learning?"
        passages = [f"Passage number {i} about various topics" for i in range(50)]

        results = bge_reranker.rerank(query, passages, batch_size=16, return_scores=True)

        # Should return all passages ranked
        assert len(results) == 50
        assert all(isinstance(item, tuple) for item in results)

    def test_rerank_batch_size_parameter(self, bge_reranker):
        """Verify batch_size parameter controls batching."""
        query = "What is machine learning?"
        passages = [f"Passage {i}" for i in range(10)]

        # Should work with different batch sizes
        results_small = bge_reranker.rerank(query, passages, batch_size=3, return_scores=True)
        results_large = bge_reranker.rerank(query, passages, batch_size=32, return_scores=True)

        # Both should return same number of results
        assert len(results_small) == 10
        assert len(results_large) == 10


class TestBGERerankerTextHandling:
    """Test handling of various text inputs."""

    def test_rerank_with_long_query(self, bge_reranker):
        """Verify reranker handles long queries (truncated at 512 tokens)."""
        query = " ".join(["word"] * 1000)  # Very long query
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
        ]

        # Should not crash
        results = bge_reranker.rerank(query, passages, return_scores=True)

        assert len(results) == 2
        assert all(isinstance(item, tuple) for item in results)

    def test_rerank_with_long_passages(self, bge_reranker):
        """Verify reranker handles long passages (truncated at 512 tokens)."""
        query = "What is machine learning?"
        passages = [
            " ".join(["Machine learning content"] * 200),  # Very long passage
            "The weather is nice today.",
        ]

        # Should not crash
        results = bge_reranker.rerank(query, passages, return_scores=True)

        assert len(results) == 2
        assert all(isinstance(item, tuple) for item in results)

    def test_rerank_with_unicode_text(self, bge_reranker):
        """Verify reranker handles Unicode text correctly."""
        query = "机器学习是什么？"
        passages = [
            "机器学习是人工智能的一个分支。",
            "The weather is nice today.",
        ]

        # Should not crash
        results = bge_reranker.rerank(query, passages, return_scores=True)

        assert len(results) == 2
        assert all(isinstance(item, tuple) for item in results)

    def test_rerank_with_special_characters(self, bge_reranker):
        """Verify reranker handles special characters."""
        query = "What is @#$%^&*()?"
        passages = [
            "Special characters: @#$%^&*()",
            "Normal text here.",
        ]

        # Should not crash
        results = bge_reranker.rerank(query, passages, return_scores=True)

        assert len(results) == 2
        assert all(isinstance(item, tuple) for item in results)


class TestBGERerankerLifecycle:
    """Test reranker lifecycle."""

    def test_close_clears_session_and_tokenizer(self):
        """Verify close() sets session and tokenizer to None."""
        reranker = BGEReranker(model_name="BAAI/bge-reranker-base")

        assert reranker.session is not None
        assert reranker.tokenizer is not None

        reranker.close()

        assert reranker.session is None
        assert reranker.tokenizer is None


class TestBGERerankerPoolSingleInstance:
    """Test reranker pool with single instance (thread-safe mode)."""

    def test_pool_size_1_uses_thread_lock(self):
        """Verify pool_size=1 creates single instance with thread lock."""
        pool = BGERerankerPool(pool_size=1, model_name="BAAI/bge-reranker-base")

        assert pool.pool_size == 1
        assert pool._reranker is not None
        assert pool._lock is not None
        assert pool.executor is None

        pool.close()

    def test_rerank_through_pool_with_single_instance(self, bge_reranker_pool):
        """Verify reranking works through pool with single instance."""
        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
        ]

        results = bge_reranker_pool.rerank(query, passages, return_scores=True)

        assert len(results) == 2
        assert all(isinstance(item, tuple) for item in results)

    def test_compute_relevance_scores_through_pool(self, bge_reranker_pool):
        """Verify compute_relevance_scores works through pool."""
        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
        ]

        scores = bge_reranker_pool.compute_relevance_scores(query, passages)

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (2,)
        assert np.all((scores >= 0.0) & (scores <= 1.0))

    def test_pool_context_manager(self):
        """Verify pool works as context manager."""
        with BGERerankerPool(pool_size=1, model_name="BAAI/bge-reranker-base") as pool:
            query = "Test query"
            passages = ["Passage one", "Passage two"]
            results = pool.rerank(query, passages)
            assert len(results) == 2

        # Pool should be closed after context exit
        assert pool._shutdown is True

    def test_rerank_after_close_raises_error(self):
        """Verify operations after close raise RuntimeError."""
        pool = BGERerankerPool(pool_size=1, model_name="BAAI/bge-reranker-base")
        pool.close()

        with pytest.raises(RuntimeError, match="has been shut down"):
            pool.rerank("query", ["passage"])

    def test_compute_relevance_scores_after_close_raises_error(self):
        """Verify compute_relevance_scores after close raises RuntimeError."""
        pool = BGERerankerPool(pool_size=1, model_name="BAAI/bge-reranker-base")
        pool.close()

        with pytest.raises(RuntimeError, match="has been shut down"):
            pool.compute_relevance_scores("query", ["passage"])


class TestBGERerankerPoolProcessPool:
    """Test reranker pool with multiple processes."""

    def test_pool_size_greater_than_1_uses_process_pool(self):
        """Verify pool_size>1 creates ProcessPoolExecutor."""
        pool = BGERerankerPool(pool_size=2, model_name="BAAI/bge-reranker-base")

        assert pool.pool_size == 2
        assert pool._reranker is None
        assert pool._lock is None
        assert pool.executor is not None

        pool.close()

    def test_rerank_through_process_pool(self):
        """Verify reranking works through process pool."""
        pool = BGERerankerPool(pool_size=2, model_name="BAAI/bge-reranker-base")

        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
        ]

        results = pool.rerank(query, passages, return_scores=True)

        assert len(results) == 2
        assert all(isinstance(item, tuple) for item in results)

        pool.close()

    def test_compute_relevance_scores_through_process_pool(self):
        """Verify compute_relevance_scores works through process pool."""
        pool = BGERerankerPool(pool_size=2, model_name="BAAI/bge-reranker-base")

        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is nice today.",
        ]

        scores = pool.compute_relevance_scores(query, passages)

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (2,)

        pool.close()


class TestBGERerankerSingleton:
    """Test singleton pattern for reranker pool."""

    def test_get_bge_reranker_returns_singleton(self):
        """Verify get_bge_reranker returns singleton instance."""
        pool1 = get_bge_reranker(pool_size=1)
        pool2 = get_bge_reranker(pool_size=1)

        # Should be the exact same object
        assert pool1 is pool2
