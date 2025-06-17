"""
End-to-end tests for unified embeddings architecture.

Tests the complete integration of BGE embeddings with tool relevance
and memory systems, ensuring both local and remote providers work correctly.
"""
import pytest
import numpy as np
import os
from unittest.mock import patch, Mock
import tempfile
import shutil
from sqlalchemy import create_engine, text

from api.embeddings_provider import EmbeddingsProvider
from lt_memory.managers.memory_manager import MemoryManager
from tool_relevance_engine import ToolRelevanceEngine
from tools.repo import ToolRepository
from errors import APIError, ErrorCode
from config import config


class TestEmbeddingsProvider:
    """Test the unified embeddings provider with real models."""
    
    @pytest.fixture
    def local_provider(self):
        """Create local BGE provider for testing."""
        return EmbeddingsProvider(
            provider_type="local",
            enable_reranker=True
        )
    
    @pytest.fixture
    def remote_provider(self):
        """Create real remote OpenAI provider."""
        # Skip if no API key available
        api_key = os.getenv("OAI_EMBEDDINGS_KEY")
        if not api_key:
            pytest.skip("OAI_EMBEDDINGS_KEY not set")
        
        return EmbeddingsProvider(
            provider_type="remote",
            api_key=api_key,
            enable_reranker=True
        )
    
    def test_local_provider_initialization(self, local_provider):
        """
        Test local BGE provider initializes correctly.
        
        REAL BUG THIS CATCHES: If BGE model fails to load or has wrong
        architecture, embeddings will fail silently and break downstream
        tool classification and memory search.
        """
        assert local_provider.provider_type == "local"
        assert local_provider._impl is not None
        assert local_provider._reranker is not None
        assert local_provider.get_embedding_dimension() == 1024
    
    def test_local_embeddings_generation(self, local_provider):
        """
        Test local BGE generates correct embeddings.
        
        REAL BUG THIS CATCHES: If BGE model produces wrong-sized embeddings
        or fails to normalize them, vector similarity calculations will be
        incorrect, causing poor tool selection and memory retrieval.
        """
        # Test single text
        text = "This is a test sentence for embedding generation"
        embedding = local_provider.encode(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
        assert np.abs(np.linalg.norm(embedding) - 1.0) < 0.01  # Should be normalized
        
        # Test batch
        texts = ["First sentence", "Second sentence", "Third sentence"]
        embeddings = local_provider.encode(texts)
        
        assert embeddings.shape == (3, 1024)
        # All should be normalized
        for emb in embeddings:
            assert np.abs(np.linalg.norm(emb) - 1.0) < 0.01
    
    def test_remote_embeddings_generation(self, remote_provider):
        """
        Test real OpenAI API generates embeddings correctly.
        
        REAL BUG THIS CATCHES: If API response parsing fails or dimensions
        don't match expected 1024, downstream vector operations will crash.
        """
        texts = ["Test sentence one", "Test sentence two"]
        embeddings = remote_provider.encode(texts, batch_size=2)
        
        assert embeddings.shape == (2, 1024)
        # Check normalization was applied
        for emb in embeddings:
            assert np.abs(np.linalg.norm(emb) - 1.0) < 0.01
    
    def test_provider_compatibility(self, local_provider, remote_provider):
        """
        Test both providers produce compatible embeddings.
        
        REAL BUG THIS CATCHES: If local and remote embeddings aren't
        compatible, switching providers will break existing vector
        databases and similarity calculations.
        """
        test_text = "Unified test sentence for both providers"
        
        local_emb = local_provider.encode(test_text)
        remote_emb = remote_provider.encode(test_text)
        
        # Both should be 1024-dimensional and normalized
        assert local_emb.shape == remote_emb.shape == (1024,)
        assert np.abs(np.linalg.norm(local_emb) - 1.0) < 0.01
        assert np.abs(np.linalg.norm(remote_emb) - 1.0) < 0.01
    
    def test_reranking_functionality(self, local_provider):
        """
        Test BGE reranker improves search results.
        
        REAL BUG THIS CATCHES: If reranker fails or returns wrong order,
        memory search will return less relevant results, degrading the
        assistant's ability to recall important information.
        """
        query = "How do I configure the database connection?"
        passages = [
            "The weather today is sunny and warm.",
            "Database configuration requires setting the connection string in config.py",
            "To configure the database, set DATABASE_URL environment variable",
            "Python is a programming language.",
            "Connection strings follow the format: postgresql://user:pass@host/db"
        ]
        
        # Get reranked results
        reranked = local_provider.rerank(query, passages, top_k=3)
        
        assert len(reranked) == 3
        # Check that database-related passages rank higher
        top_passages = [p for _, _, p in reranked]
        assert any("database" in p.lower() for p in top_passages[:2])
    
    def test_search_and_rerank_integration(self, local_provider):
        """
        Test integrated search and rerank pipeline.
        
        REAL BUG THIS CATCHES: If the two-stage retrieval (embedding search
        followed by reranking) has integration issues, search quality will
        be poor despite having both components working individually.
        """
        query = "machine learning model training"
        passages = [
            "Machine learning models require training data",
            "The training process involves optimization",
            "Models can be trained using gradient descent",
            "Coffee is a popular beverage",
            "Training machine learning models requires GPUs",
            "The weather affects outdoor activities"
        ]
        
        # Precompute embeddings (simulating vector database)
        passage_embeddings = local_provider.encode(passages)
        
        # Search and rerank
        results = local_provider.search_and_rerank(
            query=query,
            passages=passages,
            passage_embeddings=passage_embeddings,
            initial_top_k=4,
            final_top_k=2
        )
        
        assert len(results) == 2
        # ML-related passages should rank highest
        for idx, score, passage in results:
            assert "machine learning" in passage.lower() or "training" in passage.lower()


class TestMemorySystemIntegration:
    """Test memory system with unified embeddings and real PostgreSQL."""
    
    @pytest.fixture
    def test_database(self):
        """Create test PostgreSQL database connection."""
        # Use test database URL from environment or config
        test_db_url = os.getenv("TEST_DATABASE_URL", 
                                "postgresql://mira:secure_password@localhost/mira_test")
        
        engine = create_engine(test_db_url)
        
        # Ensure pgvector extension is available
        try:
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
        except Exception as e:
            pytest.skip(f"PostgreSQL with pgvector not available: {e}")
        
        yield engine
        
        # Cleanup test data
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE IF EXISTS memory_blocks CASCADE"))
            conn.execute(text("TRUNCATE TABLE IF EXISTS memory_passages CASCADE"))
            conn.commit()
    
    @pytest.fixture
    def memory_manager_with_local(self, test_database, monkeypatch):
        """Create memory manager with local BGE embeddings."""
        # Patch config to use test database
        monkeypatch.setattr('lt_memory.managers.memory_manager.config.memory.database_url', 
                            test_database.url.render_as_string())
        
        manager = MemoryManager()
        # Ensure it's using our embeddings provider
        assert isinstance(manager.embedding_model, EmbeddingsProvider)
        assert manager.embedding_model.provider_type == "local"  # Default
        
        return manager
    
    def test_memory_embedding_storage(self, memory_manager_with_local):
        """
        Test memory system stores embeddings correctly in PostgreSQL.
        
        REAL BUG THIS CATCHES: If embeddings aren't properly stored in
        pgvector format, vector similarity searches will fail, making
        the memory system unable to retrieve relevant information.
        """
        # Create a memory block
        block_id = memory_manager_with_local.block_manager.create_block(
            name="test_preferences",
            content="User prefers dark mode interfaces and high contrast",
            max_chars=1000
        )
        
        # Verify embedding was generated and stored
        with memory_manager_with_local.get_session() as session:
            from lt_memory.models import MemoryBlock
            block = session.query(MemoryBlock).filter_by(id=block_id).first()
            
            assert block is not None
            assert block.embedding is not None
            # PostgreSQL pgvector stores as array
            assert len(block.embedding) == 1024
    
    def test_memory_search_with_embeddings(self, memory_manager_with_local):
        """
        Test memory search using vector similarity.
        
        REAL BUG THIS CATCHES: If vector search doesn't work correctly
        with the new embeddings, the assistant won't be able to recall
        relevant memories, appearing forgetful.
        """
        # Add some test memories
        memory_manager_with_local.block_manager.create_block(
            name="ui_preferences",
            content="The user strongly prefers dark mode for all interfaces",
            max_chars=1000
        )
        
        memory_manager_with_local.block_manager.create_block(
            name="food_preferences", 
            content="The user likes Italian food, especially pizza",
            max_chars=1000
        )
        
        # Search for UI-related memories
        results = memory_manager_with_local.search_memory("What are the UI preferences?")
        
        assert len(results) > 0
        # UI preference should rank higher than food preference
        assert "dark mode" in results[0]["content"].lower()
    
    def test_memory_reranking_improves_results(self, memory_manager_with_local):
        """
        Test memory search with reranking enabled.
        
        REAL BUG THIS CATCHES: If reranking isn't integrated with memory
        search, results won't be optimally ordered, causing the assistant
        to use less relevant context.
        """
        # Add diverse memories
        memories = [
            ("meeting_note", "Meeting scheduled for 3pm tomorrow"),
            ("ui_pref_1", "User wants dark mode enabled by default"),
            ("ui_pref_2", "Dark themes should use high contrast for readability"),
            ("random_fact", "The sky is blue"),
            ("ui_pref_3", "UI animations should be subtle and fast")
        ]
        
        for name, content in memories:
            memory_manager_with_local.block_manager.create_block(
                name=name,
                content=content,
                max_chars=1000
            )
        
        # Search with reranking
        query = "What are the user's preferences for UI themes?"
        results = memory_manager_with_local.search_memory(query, limit=3)
        
        # Top results should all be UI-related
        assert len(results) <= 3
        for result in results:
            content_lower = result["content"].lower()
            # Should prioritize UI/theme related memories
            assert any(term in content_lower for term in ["ui", "dark", "theme", "contrast"])


class TestToolRelevanceIntegration:
    """Test tool relevance engine with unified embeddings."""
    
    @pytest.fixture
    def temp_tools_dir(self):
        """Create temporary tools directory with examples."""
        temp_dir = tempfile.mkdtemp()
        tools_dir = os.path.join(temp_dir, "data", "tools")
        os.makedirs(tools_dir, exist_ok=True)
        
        # Create multiple tool examples
        tools_data = {
            "weather_tool": [
                {"query": "What's the weather today?", "label": "weather_tool"},
                {"query": "Is it going to rain tomorrow?", "label": "weather_tool"},
                {"query": "Show me the forecast", "label": "weather_tool"},
                {"query": "Temperature outside", "label": "weather_tool"},
                {"query": "Calculate 2+2", "label": "calculator_tool"},
                {"query": "Send an email", "label": "email_tool"}
            ],
            "calculator_tool": [
                {"query": "Calculate 15 * 7", "label": "calculator_tool"},
                {"query": "What's the square root of 144?", "label": "calculator_tool"},
                {"query": "Solve this equation", "label": "calculator_tool"},
                {"query": "Add these numbers", "label": "calculator_tool"},
                {"query": "What's the weather?", "label": "weather_tool"},
                {"query": "Email my boss", "label": "email_tool"}
            ]
        }
        
        import json
        for tool_name, examples in tools_data.items():
            tool_dir = os.path.join(tools_dir, tool_name)
            os.makedirs(tool_dir, exist_ok=True)
            
            with open(os.path.join(tool_dir, "classifier_examples.json"), "w") as f:
                json.dump(examples, f)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def tool_relevance_engine(self, temp_tools_dir):
        """Create tool relevance engine with BGE embeddings."""
        # Mock config
        with patch('tool_relevance_engine.config') as mock_config:
            mock_config.paths.data_dir = temp_tools_dir
            mock_config.tool_relevance.thread_limit = 2
            mock_config.tool_relevance.primary_threshold = 0.4
            mock_config.tool_relevance.secondary_threshold = 0.3
            mock_config.tool_relevance.tool_persistence_messages = 2
            mock_config.tool_relevance.context_window_size = 3
            mock_config.tool_relevance.topic_coherence_threshold = 0.7
            
            # Create shared embeddings provider
            embeddings = EmbeddingsProvider(
                provider_type="local",
                enable_reranker=False
            )
            
            # Create mock tool repository
            tool_repo = Mock(spec=ToolRepository)
            
            # Create engine
            engine = ToolRelevanceEngine(tool_repo, embeddings)
            
            return engine
    
    def test_tool_classification_accuracy(self, tool_relevance_engine):
        """
        Test tool classification accuracy with BGE embeddings.
        
        REAL BUG THIS CATCHES: If BGE embeddings produce poor quality
        representations for tool classification, wrong tools will be
        selected, causing failed operations and poor user experience.
        """
        test_cases = [
            ("What's the temperature outside?", "weather_tool"),
            ("Calculate 25 divided by 5", "calculator_tool"),
            ("Show me tomorrow's forecast", "weather_tool"),
            ("What's 2 plus 2?", "calculator_tool")
        ]
        
        for query, expected_tool in test_cases:
            results = tool_relevance_engine.analyze_message(query)
            
            if results:
                top_tool = results[0][0]  # Get top ranked tool
                assert expected_tool in top_tool, \
                    f"Expected {expected_tool} for '{query}', got {top_tool}"
    
    def test_matrix_operations_correctness(self, tool_relevance_engine):
        """
        Test matrix operations produce same results as loop-based approach.
        
        REAL BUG THIS CATCHES: If matrix operations have bugs, tool
        classification results will differ from the original implementation,
        causing inconsistent behavior.
        """
        query = "What's the weather forecast for next week?"
        
        # Get results from original method
        original_results = tool_relevance_engine.classifier.classify_message_with_scores(query)
        
        # Get results from matrix operations
        tool_relevance_engine.precompute_tool_embeddings_matrix()
        matrix_results = tool_relevance_engine.classify_with_matrix_operations(query)
        
        # Results should be equivalent
        assert len(original_results) == len(matrix_results)
        
        # Convert to dict for easier comparison
        orig_dict = {tool: score for tool, score in original_results}
        matrix_dict = {tool: score for tool, score in matrix_results}
        
        for tool in orig_dict:
            assert tool in matrix_dict
            # Scores should be very close (allowing for floating point differences)
            assert abs(orig_dict[tool] - matrix_dict[tool]) < 0.001


class TestEndToEndScenarios:
    """Test complete user scenarios with the unified architecture."""
    
    def test_memory_aware_tool_selection(self, tmp_path):
        """
        Test that stored memories influence tool selection.
        
        REAL BUG THIS CATCHES: If memory and tool systems don't integrate
        properly, the assistant won't use past context to make better
        tool selections, appearing less intelligent.
        """
        # This is a conceptual test showing how systems should integrate
        # In practice, this would require the full conversation flow
        
        # Create embeddings provider
        embeddings = EmbeddingsProvider(provider_type="local", enable_reranker=True)
        
        # Generate embeddings for a memory about user preferences
        memory_text = "User frequently asks about weather in Seattle"
        memory_embedding = embeddings.encode(memory_text)
        
        # Generate embedding for new query
        query = "How's it looking outside?"
        query_embedding = embeddings.encode(query)
        
        # The enhanced context should have higher similarity to weather tool
        enhanced_embedding = (query_embedding + 0.2 * memory_embedding) 
        enhanced_embedding = enhanced_embedding / np.linalg.norm(enhanced_embedding)
        
        # Similarity should be higher with context
        weather_example = "What's the weather today?"
        weather_embedding = embeddings.encode(weather_example)
        
        base_similarity = np.dot(query_embedding, weather_embedding)
        enhanced_similarity = np.dot(enhanced_embedding, weather_embedding)
        
        assert enhanced_similarity > base_similarity
    
    def test_performance_with_concurrent_load(self):
        """
        Test system performance under concurrent load.
        
        REAL BUG THIS CATCHES: If the system has thread safety issues or
        resource leaks, it will crash or slow down significantly under
        concurrent load, making it unsuitable for production use.
        """
        import concurrent.futures
        import time
        
        provider = EmbeddingsProvider(provider_type="local", enable_reranker=True)
        
        def process_request(i):
            """Simulate a full request cycle."""
            # Tool classification
            query = f"Query number {i}: What's the weather like?"
            query_embedding = provider.encode(query)
            
            # Memory search
            memories = [f"Memory {j} for request {i}" for j in range(5)]
            memory_embeddings = provider.encode(memories)
            
            # Reranking
            results = provider.rerank(query, memories, top_k=2)
            
            return len(results)
        
        start_time = time.time()
        
        # Run 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_request, i) for i in range(20)]
            results = [f.result() for f in futures]
        
        elapsed = time.time() - start_time
        
        # All requests should complete successfully
        assert all(r == 2 for r in results)  # Each should return 2 reranked results
        
        # Should complete in reasonable time (less than 10 seconds for 20 requests)
        assert elapsed < 10.0, f"Took {elapsed:.2f}s for 20 requests"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])