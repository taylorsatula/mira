"""
Tests for ONNX embedding model.

Following testing guide principles:
- Test contracts, not implementation
- Use real ONNX model for testing  
- Mock nothing - test real functionality
- Focus on production scenarios
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from utils.onnx_embeddings import ONNXEmbeddingModel
from errors import ToolError, ErrorCode


@pytest.fixture(scope="session")
def real_onnx_model():
    """
    Real ONNX model for testing - uses the actual model files.
    
    REAL BUG THIS CATCHES: If model loading, tokenizer setup, or ONNX 
    session creation has bugs, embedding generation will fail in production.
    """
    return ONNXEmbeddingModel()


@pytest.fixture
def sample_texts():
    """Sample texts covering real-world scenarios."""
    return [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning embeddings for semantic similarity",
        "Unicode test: Hello 世界 🌍",
        "Very long text that might exceed token limits: " + "word " * 200,
        "Short",
        "A" * 1000,  # Very long single token scenario
    ]


@pytest.fixture
def temp_model_dir():
    """Temporary directory for testing missing model scenarios."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestONNXEmbeddingModelInitialization:
    """Test model initialization contracts."""
    
    def test_model_loads_successfully_with_default_path(self):
        """
        Test model initialization with default path.
        
        REAL BUG THIS CATCHES: If default model path calculation or model
        loading fails, the entire embedding system breaks on startup.
        """
        model = ONNXEmbeddingModel()
        
        # Verify model is properly initialized
        assert model.session is not None
        assert model.tokenizer is not None
        assert model.embedding_dim > 0
        assert model.input_names is not None
        assert model.output_name is not None
        
        # Verify embedding dimension is reasonable for all-MiniLM-L6-v2
        assert model.embedding_dim == 384
    
    def test_thread_limit_configuration(self):
        """
        Test thread limit parameter affects model configuration.
        
        REAL BUG THIS CATCHES: If thread limit configuration fails,
        performance tuning won't work, affecting production throughput.
        """
        model = ONNXEmbeddingModel(thread_limit=2)
        
        # Model should initialize successfully with custom thread limit
        assert model.session is not None
        assert model.embedding_dim > 0
    
    def test_missing_model_file_raises_appropriate_error(self, temp_model_dir):
        """
        Test missing model file handling.
        
        REAL BUG THIS CATCHES: If missing model errors aren't clear,
        deployment issues become hard to diagnose.
        """
        nonexistent_path = str(temp_model_dir / "nonexistent.onnx")
        
        with pytest.raises(ToolError) as exc_info:
            ONNXEmbeddingModel(model_path=nonexistent_path)
        
        error = exc_info.value
        assert error.code == ErrorCode.FILE_NOT_FOUND
        assert error.permanent_failure is True
        assert "ONNX model not found" in str(error)
        assert "setup_onnx.py" in str(error)  # Helpful recovery guidance


class TestONNXEmbeddingModelEncoding:
    """Test embedding encoding contracts."""
    
    def test_single_text_encoding_produces_correct_format(self, real_onnx_model):
        """
        Test single text encoding contract.
        
        REAL BUG THIS CATCHES: If single text encoding returns wrong format
        or dimensions, downstream similarity calculations break.
        """
        text = "Hello world"
        embedding = real_onnx_model.encode(text)
        
        # Verify contract: single text → single embedding vector
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # Should be 1D for single text
        assert embedding.dtype == np.float32 or embedding.dtype == np.float64
        
        # Verify normalization (embeddings should be unit vectors by default)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6
    
    def test_batch_text_encoding_produces_correct_format(self, real_onnx_model, sample_texts):
        """
        Test batch text encoding contract.
        
        REAL BUG THIS CATCHES: If batch encoding returns wrong dimensions
        or inconsistent shapes, batch processing operations fail.
        """
        # Use non-empty texts for this test
        texts = [t for t in sample_texts if t.strip()]
        embeddings = real_onnx_model.encode(texts)
        
        # Verify contract: list of texts → 2D embedding matrix
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(texts), 384)
        assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64
        
        # Verify each embedding is normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
    
    def test_identical_texts_produce_identical_embeddings(self, real_onnx_model):
        """
        Test embedding consistency contract.
        
        REAL BUG THIS CATCHES: If embedding generation is non-deterministic,
        duplicate detection and caching systems break.
        """
        text = "Test consistency"
        
        embedding1 = real_onnx_model.encode(text)
        embedding2 = real_onnx_model.encode(text)
        
        # Should be exactly identical for same text
        np.testing.assert_array_equal(embedding1, embedding2)
    
    def test_different_texts_produce_different_embeddings(self, real_onnx_model):
        """
        Test embedding uniqueness contract.
        
        REAL BUG THIS CATCHES: If different texts produce identical embeddings,
        semantic search and similarity matching completely break.
        """
        text1 = "Machine learning"
        text2 = "Cooking recipes"
        
        embedding1 = real_onnx_model.encode(text1)
        embedding2 = real_onnx_model.encode(text2)
        
        # Should be significantly different
        similarity = np.dot(embedding1, embedding2)
        assert similarity < 0.9  # Should not be too similar
    
    def test_semantic_similarity_works_correctly(self, real_onnx_model):
        """
        Test semantic similarity contract.
        
        REAL BUG THIS CATCHES: If semantic relationships aren't captured,
        the entire embedding-based search system provides wrong results.
        """
        # Related texts should have higher similarity
        text1 = "cat animal pet"
        text2 = "dog animal pet"
        text3 = "computer programming code"
        
        emb1 = real_onnx_model.encode(text1)
        emb2 = real_onnx_model.encode(text2)
        emb3 = real_onnx_model.encode(text3)
        
        # Related texts should be more similar
        related_similarity = np.dot(emb1, emb2)
        unrelated_similarity = np.dot(emb1, emb3)
        
        assert related_similarity > unrelated_similarity
        assert related_similarity > 0.5  # Should be reasonably similar
    
    def test_embedding_dimension_accessor(self, real_onnx_model):
        """
        Test embedding dimension accessor contract.
        
        REAL BUG THIS CATCHES: If dimension accessor returns wrong value,
        vector storage systems allocate wrong memory, causing crashes.
        """
        dim = real_onnx_model.get_sentence_embedding_dimension()
        
        assert isinstance(dim, int)
        assert dim == 384
        
        # Verify it matches actual embedding dimensions
        embedding = real_onnx_model.encode("test")
        assert embedding.shape[0] == dim


class TestONNXEmbeddingModelErrorHandling:
    """Test error handling scenarios."""
    
    def test_empty_text_list_raises_appropriate_error(self, real_onnx_model):
        """
        Test empty input handling.
        
        REAL BUG THIS CATCHES: If empty input crashes instead of proper error,
        users get cryptic failures instead of actionable feedback.
        """
        with pytest.raises(ToolError) as exc_info:
            real_onnx_model.encode([])
        
        error = exc_info.value
        assert error.code == ErrorCode.PARAMETER_MISSING
        assert error.permanent_failure is True
        assert "No texts provided" in str(error)
    
    def test_none_input_handled_gracefully(self, real_onnx_model):
        """
        Test None input handling.
        
        REAL BUG THIS CATCHES: If None input causes crash, API calls with
        missing data break the entire system instead of returning clear errors.
        """
        with pytest.raises((ToolError, TypeError)):
            real_onnx_model.encode(None)
    
    def test_very_long_text_handled_gracefully(self, real_onnx_model):
        """
        Test extremely long text handling.
        
        REAL BUG THIS CATCHES: If very long texts cause memory issues or
        crashes, users processing large documents break the system.
        """
        # Create very long text (much longer than typical 512 token limit)
        very_long_text = "word " * 2000
        
        # Should handle gracefully with truncation
        embedding = real_onnx_model.encode(very_long_text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert np.isfinite(embedding).all()
    
    def test_unicode_text_handled_correctly(self, real_onnx_model):
        """
        Test Unicode text handling.
        
        REAL BUG THIS CATCHES: If Unicode causes encoding/tokenization failures,
        international users can't use the embedding system.
        """
        unicode_texts = [
            "Hello 世界 🌍",
            "Café naïve résumé",
            "Москва Россия",
            "العربية",
            "🚀🌟💫⭐✨"
        ]
        
        for text in unicode_texts:
            embedding = real_onnx_model.encode(text)
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (384,)
            assert np.isfinite(embedding).all()


class TestONNXEmbeddingModelBatchProcessing:
    """Test batch processing and performance characteristics."""
    
    def test_batch_size_parameter_affects_processing(self, real_onnx_model, sample_texts):
        """
        Test batch size parameter contract.
        
        REAL BUG THIS CATCHES: If batch size parameter doesn't work,
        performance tuning for large datasets becomes impossible.
        """
        texts = [t for t in sample_texts if t.strip()]
        
        # Test different batch sizes produce same results
        embeddings_batch1 = real_onnx_model.encode(texts, batch_size=1)
        embeddings_batch4 = real_onnx_model.encode(texts, batch_size=4)
        
        # Results should be identical regardless of batch size
        np.testing.assert_array_almost_equal(embeddings_batch1, embeddings_batch4, decimal=6)
    
    def test_large_batch_processing_performance(self, real_onnx_model):
        """
        Test large batch processing performance.
        
        REAL BUG THIS CATCHES: If large batches cause memory issues or
        unacceptable performance, production workloads fail.
        """
        # Create reasonably large batch
        texts = [f"Text number {i} with some content" for i in range(100)]
        
        start_time = time.time()
        embeddings = real_onnx_model.encode(texts, batch_size=32)
        elapsed = time.time() - start_time
        
        # Verify results
        assert embeddings.shape == (100, 384)
        assert np.isfinite(embeddings).all()
        
        # Performance should be reasonable (less than 30 seconds for 100 texts)
        assert elapsed < 30.0
    
    def test_concurrent_encoding_thread_safety(self, real_onnx_model):
        """
        Test concurrent access thread safety.
        
        REAL BUG THIS CATCHES: If concurrent encoding causes race conditions
        or crashes, multi-user production systems become unstable.
        """
        def encode_worker(worker_id):
            text = f"Worker {worker_id} processing text"
            embedding = real_onnx_model.encode(text)
            return (worker_id, embedding)
        
        # Run concurrent encoding operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(encode_worker, i) for i in range(20)]
            results = [f.result() for f in futures]
        
        # Verify all operations completed successfully
        assert len(results) == 20
        
        # Verify embeddings are valid and consistent
        for worker_id, embedding in results:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (384,)
            assert np.isfinite(embedding).all()
    
    def test_normalization_parameter_contract(self, real_onnx_model):
        """
        Test normalization parameter affects output correctly.
        
        REAL BUG THIS CATCHES: If normalization parameter doesn't work,
        similarity calculations give wrong results.
        """
        text = "Test normalization"
        
        normalized = real_onnx_model.encode(text, normalize=True)
        unnormalized = real_onnx_model.encode(text, normalize=False)
        
        # Normalized should be unit vector
        norm_normalized = np.linalg.norm(normalized)
        assert abs(norm_normalized - 1.0) < 1e-6
        
        # Unnormalized should have different norm
        norm_unnormalized = np.linalg.norm(unnormalized)
        assert abs(norm_unnormalized - 1.0) > 1e-3
        
        # Direction should be the same (normalized version of unnormalized)
        manual_normalized = unnormalized / norm_unnormalized
        np.testing.assert_array_almost_equal(normalized, manual_normalized, decimal=6)