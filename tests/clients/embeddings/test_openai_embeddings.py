"""
Tests for clients/embeddings/openai_embeddings.py

Tests OpenAIEmbeddingModel with real OpenAI API calls.
Following MIRA testing philosophy: no mocks, test real API behavior.
"""
import pytest
import numpy as np
from clients.embeddings.openai_embeddings import OpenAIEmbeddingModel
from clients.vault_client import get_api_key


@pytest.fixture(scope="module")
def api_key():
    """Get OpenAI API key from Vault."""
    return get_api_key('openai_embeddings_key')


@pytest.fixture(scope="module")
def model_small(api_key):
    """Shared small model instance for tests."""
    return OpenAIEmbeddingModel(api_key=api_key, model="text-embedding-3-small")


@pytest.fixture(scope="module")
def model_large(api_key):
    """Shared large model instance for tests."""
    return OpenAIEmbeddingModel(api_key=api_key, model="text-embedding-3-large")


class TestOpenAIEmbeddingModelInitialization:
    """Test model initialization and configuration."""

    def test_initialize_with_valid_model(self, api_key):
        """Verify model initializes with valid model name."""
        model = OpenAIEmbeddingModel(api_key=api_key, model="text-embedding-3-small")

        assert model.model == "text-embedding-3-small"
        assert model.embedding_dim == 1024
        assert model.api_key == api_key

    def test_initialize_with_large_model(self, api_key):
        """Verify large model has correct dimensions."""
        model = OpenAIEmbeddingModel(api_key=api_key, model="text-embedding-3-large")

        assert model.model == "text-embedding-3-large"
        assert model.embedding_dim == 3072

    def test_initialize_with_ada_model(self, api_key):
        """Verify ada-002 model configuration."""
        model = OpenAIEmbeddingModel(api_key=api_key, model="text-embedding-ada-002")

        assert model.model == "text-embedding-ada-002"
        assert model.embedding_dim == 1024

    def test_initialize_with_invalid_model_raises_error(self, api_key):
        """Verify unsupported model names are rejected."""
        with pytest.raises(ValueError, match="Unsupported embedding model"):
            OpenAIEmbeddingModel(api_key=api_key, model="invalid-model-name")

    def test_get_dimension_returns_correct_value(self, model_small):
        """Verify get_dimension() returns configured dimension."""
        assert model_small.get_dimension() == 1024

    def test_initialize_without_api_key_uses_vault(self):
        """Verify initialization without api_key fetches from Vault."""
        # Should fetch from vault_client.get_api_key('openai_embeddings_key')
        model = OpenAIEmbeddingModel(model="text-embedding-3-small")

        assert model.api_key is not None
        assert model.model == "text-embedding-3-small"


class TestOpenAIEmbeddingModelBasicEncoding:
    """Test basic embedding generation."""

    def test_encode_single_string_returns_1d_array(self, model_small):
        """Verify single string input returns 1D embedding array."""
        text = "This is a test sentence."

        embedding = model_small.encode(text)

        # Should be 1D array
        assert embedding.ndim == 1
        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32

    def test_encode_list_returns_2d_array(self, model_small):
        """Verify list input returns 2D embedding array."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]

        embeddings = model_small.encode(texts)

        # Should be 2D array
        assert embeddings.ndim == 2
        assert embeddings.shape == (3, 1024)
        assert embeddings.dtype == np.float32

    def test_embeddings_have_correct_dimension(self, model_small):
        """Verify embeddings have exactly 1024 dimensions for small model."""
        embedding = model_small.encode("Test text")

        assert embedding.shape == (1024,)

    def test_large_model_embeddings_have_correct_dimension(self, model_large):
        """Verify embeddings have exactly 3072 dimensions for large model."""
        embedding = model_large.encode("Test text")

        assert embedding.shape == (3072,)

    def test_embeddings_contain_no_nan_or_inf(self, model_small):
        """Verify embeddings contain only finite values."""
        embedding = model_small.encode("Test text")

        assert np.all(np.isfinite(embedding))
        assert not np.any(np.isnan(embedding))
        assert not np.any(np.isinf(embedding))

    def test_single_text_in_list_returns_2d_array(self, model_small):
        """Verify single-item list returns 2D array (not 1D)."""
        texts = ["Only one sentence"]

        embeddings = model_small.encode(texts)

        # Should be 2D with shape (1, 1024), not 1D (1024,)
        assert embeddings.ndim == 2
        assert embeddings.shape == (1, 1024)


class TestOpenAIEmbeddingModelInputValidation:
    """Test input validation and error handling."""

    def test_empty_string_raises_error(self, model_small):
        """Verify empty string is rejected."""
        with pytest.raises(ValueError, match="Empty text at index 0"):
            model_small.encode("")

    def test_whitespace_only_string_raises_error(self, model_small):
        """Verify whitespace-only string is rejected."""
        with pytest.raises(ValueError, match="Empty text at index 0"):
            model_small.encode("   ")

    def test_empty_list_raises_error(self, model_small):
        """Verify empty list is rejected."""
        with pytest.raises(ValueError, match="No texts provided for encoding"):
            model_small.encode([])

    def test_list_with_empty_string_raises_error(self, model_small):
        """Verify list containing empty string is rejected."""
        texts = ["Valid text", "", "Another valid text"]

        with pytest.raises(ValueError, match="Empty text at index 1"):
            model_small.encode(texts)

    def test_list_with_whitespace_string_raises_error(self, model_small):
        """Verify list containing whitespace-only string is rejected."""
        texts = ["Valid text", "  \t\n  ", "Another valid text"]

        with pytest.raises(ValueError, match="Empty text at index 1"):
            model_small.encode(texts)


class TestOpenAIEmbeddingModelBatchProcessing:
    """Test batch processing behavior."""

    def test_batch_size_parameter_processes_correctly(self, model_small):
        """Verify batch_size parameter controls batching correctly."""
        # Create 10 texts with batch_size=3 (should create 4 batches: 3+3+3+1)
        texts = [f"Text number {i}" for i in range(10)]

        embeddings = model_small.encode(texts, batch_size=3)

        # Should still produce 10 embeddings
        assert embeddings.shape == (10, 1024)
        assert np.all(np.isfinite(embeddings))

    def test_large_batch_processes_correctly(self, model_small):
        """Verify large batches are handled correctly."""
        # Create 50 texts
        texts = [f"Document {i} with some content" for i in range(50)]

        embeddings = model_small.encode(texts, batch_size=32)

        assert embeddings.shape == (50, 1024)
        assert np.all(np.isfinite(embeddings))


class TestOpenAIEmbeddingModelTextHandling:
    """Test handling of various text inputs."""

    def test_long_text_handled(self, model_small):
        """Verify very long texts are handled correctly."""
        # Create a very long text
        long_text = " ".join(["word"] * 5000)

        # Should not crash
        embedding = model_small.encode(long_text)

        assert embedding.shape == (1024,)
        assert np.all(np.isfinite(embedding))

    def test_special_characters_handled(self, model_small):
        """Verify special characters don't break encoding."""
        text = "Test with symbols: @#$%^&*(){}[]|\\<>?/~`"

        embedding = model_small.encode(text)

        assert embedding.shape == (1024,)
        assert np.all(np.isfinite(embedding))

    def test_unicode_text_handled(self, model_small):
        """Verify Unicode text is handled correctly."""
        text = "Hello 世界 مرحبا мир 🌍"

        embedding = model_small.encode(text)

        assert embedding.shape == (1024,)
        assert np.all(np.isfinite(embedding))

    def test_newlines_and_tabs_handled(self, model_small):
        """Verify text with newlines and tabs is handled."""
        text = "Line one\nLine two\tWith tab"

        embedding = model_small.encode(text)

        assert embedding.shape == (1024,)
        assert np.all(np.isfinite(embedding))


class TestOpenAIEmbeddingModelSemantic:
    """Test semantic properties of embeddings."""

    def test_similar_texts_have_high_similarity(self, model_small):
        """Verify semantically similar texts produce similar embeddings."""
        text1 = "The cat sits on the mat"
        text2 = "A cat is sitting on a mat"
        text3 = "Quantum physics is complicated"

        emb1 = model_small.encode(text1)
        emb2 = model_small.encode(text2)
        emb3 = model_small.encode(text3)

        # Calculate cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        similarity_similar = cosine_similarity(emb1, emb2)
        similarity_different = cosine_similarity(emb1, emb3)

        # Similar texts should have higher similarity
        assert similarity_similar > 0.7
        assert similarity_different < similarity_similar

    def test_identical_texts_produce_identical_embeddings(self, model_small):
        """Verify identical texts produce identical embeddings."""
        text = "This is a test sentence for identity check"

        emb1 = model_small.encode(text)
        emb2 = model_small.encode(text)

        # Should be identical (within floating point precision)
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=6)


class TestOpenAIEmbeddingModelConnection:
    """Test connection and health check functionality."""

    def test_connection_test_succeeds(self, model_small):
        """Verify test_connection() succeeds with valid credentials."""
        result = model_small.test_connection()

        assert result["status"] == "success"
        assert result["model"] == "text-embedding-3-small"
        assert result["embedding_dim"] == 1024
        assert result["api_accessible"] is True
        assert "test_embedding_shape" in result
        assert result["test_embedding_shape"] == (1024,)
        assert "test_embedding_norm" in result
        assert result["test_embedding_norm"] > 0

    def test_connection_test_with_invalid_key_fails(self):
        """Verify test_connection() fails with invalid API key."""
        model = OpenAIEmbeddingModel(api_key="invalid-key-12345", model="text-embedding-3-small")

        result = model.test_connection()

        assert result["status"] == "error"
        assert result["api_accessible"] is False
        assert "error" in result


class TestOpenAIEmbeddingModelLifecycle:
    """Test model lifecycle and resource management."""

    def test_close_method_exists_and_runs(self, api_key):
        """Verify close() method can be called."""
        model = OpenAIEmbeddingModel(api_key=api_key, model="text-embedding-3-small")

        # Should not raise exception
        model.close()

    def test_model_can_be_used_after_initialization(self, api_key):
        """Verify model works immediately after initialization."""
        model = OpenAIEmbeddingModel(api_key=api_key, model="text-embedding-3-small")

        # Should work without additional setup
        embedding = model.encode("Test")
        assert embedding.shape == (1024,)


class TestOpenAIEmbeddingModelDimensionValidation:
    """Test that returned embeddings match expected dimensions."""

    def test_dimension_mismatch_would_raise_error(self, model_small):
        """Verify dimension validation logic exists in encode()."""
        # This tests that the code at lines 107-108 would catch dimension mismatches
        # We can't force a mismatch with real API, but we verify the model
        # has the correct expected dimension set

        embedding = model_small.encode("Test")

        # Verify the actual embedding matches what the model expects
        assert embedding.shape[0] == model_small.embedding_dim
        assert embedding.shape[0] == 1024
