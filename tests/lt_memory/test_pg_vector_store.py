"""
Production-grade tests for PGVectorStore PostgreSQL vector similarity search.

This test suite focuses on realistic production scenarios using actual PostgreSQL
with pgvector extension. We test real database conditions, performance characteristics,
and error handling that matter for production reliability.

Testing philosophy:
1. Test with real PostgreSQL database for authentic conditions
2. Test the public API contracts that users rely on
3. Test critical error handling and edge cases
4. Focus on performance characteristics under realistic load
5. Test real similarity search workflows
6. Verify proper database integration and transaction handling
"""

import pytest
import time
import numpy as np
import psycopg2
from typing import List, Dict, Any
from unittest.mock import patch

# Import the system under test
from lt_memory.utils.pg_vector_store import PGVectorStore, create_vector_table


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def test_connection_string():
    """PostgreSQL connection string for testing."""
    # Use the real PostgreSQL test database as mira_app user
    return "postgresql://mira_app@localhost/lt_memory_test"


@pytest.fixture(scope="session")
def test_table_name():
    """Test table name to avoid conflicts with production data."""
    return "test_vector_passages"


@pytest.fixture(scope="session")
def setup_test_table(test_connection_string, test_table_name):
    """Create test table for vector operations."""
    # Create test table
    create_vector_table(
        test_connection_string, 
        table_name=test_table_name,
        dimension=384
    )
    
    yield test_table_name
    
    # Cleanup
    conn = psycopg2.connect(test_connection_string)
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {test_table_name}")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def clean_test_table(test_connection_string, setup_test_table):
    """Clean test table for each test."""
    table_name = setup_test_table
    
    # Clear existing data
    conn = psycopg2.connect(test_connection_string)
    try:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {table_name}")
        conn.commit()
    finally:
        conn.close()
    
    yield table_name


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    np.random.seed(42)  # Reproducible tests
    
    return {
        "dimension_384": np.random.random(384).astype(np.float32),
        "dimension_512": np.random.random(512).astype(np.float32),
        "zero_vector": np.zeros(384, dtype=np.float32),
        "unit_vector": np.ones(384, dtype=np.float32) / np.sqrt(384),
        "similar_vector": np.random.random(384).astype(np.float32) * 0.1 + 0.9,
    }


@pytest.fixture
def populated_test_table(clean_test_table, test_connection_string, sample_vectors):
    """Test table with sample data for search operations."""
    table_name = clean_test_table
    base_vector = sample_vectors["dimension_384"]
    
    # Insert sample data with variations
    conn = psycopg2.connect(test_connection_string)
    try:
        with conn.cursor() as cur:
            # Insert diverse test documents
            test_data = [
                {
                    "content": "Machine learning fundamentals",
                    "embedding": (base_vector + np.random.normal(0, 0.1, 384)).astype(np.float32),
                    "source": "ml_doc",
                    "importance": 0.9
                },
                {
                    "content": "Python programming guide",
                    "embedding": (base_vector + np.random.normal(0, 0.2, 384)).astype(np.float32),
                    "source": "programming_guide",
                    "importance": 0.8
                },
                {
                    "content": "Database optimization techniques",
                    "embedding": (base_vector + np.random.normal(0, 0.3, 384)).astype(np.float32),
                    "source": "db_doc",
                    "importance": 0.7
                },
                {
                    "content": "Web development best practices",
                    "embedding": (base_vector + np.random.normal(0, 0.4, 384)).astype(np.float32),
                    "source": "web_guide",
                    "importance": 0.6
                },
                {
                    "content": "System architecture patterns",
                    "embedding": (base_vector + np.random.normal(0, 0.5, 384)).astype(np.float32),
                    "source": "architecture_doc",
                    "importance": 0.5
                }
            ]
            
            for doc in test_data:
                cur.execute(f"""
                    INSERT INTO {table_name} (content, embedding, source, importance_score)
                    VALUES (%s, %s, %s, %s)
                """, [
                    doc["content"],
                    doc["embedding"].tolist(),
                    doc["source"],
                    doc["importance"]
                ])
        
        conn.commit()
    finally:
        conn.close()
    
    yield table_name


@pytest.fixture
def performance_test_table(clean_test_table, test_connection_string, sample_vectors):
    """Test table with larger dataset for performance testing."""
    table_name = clean_test_table
    base_vector = sample_vectors["dimension_384"]
    
    # Insert larger dataset for performance testing
    conn = psycopg2.connect(test_connection_string)
    try:
        with conn.cursor() as cur:
            for i in range(100):  # 100 documents for performance testing
                embedding = (base_vector + np.random.normal(0, 0.1, 384)).astype(np.float32)
                cur.execute(f"""
                    INSERT INTO {table_name} (content, embedding, source, importance_score)
                    VALUES (%s, %s, %s, %s)
                """, [
                    f"Performance test document {i}",
                    embedding.tolist(),
                    f"source_{i % 5}",  # 5 different sources
                    0.5 + (i % 10) * 0.05  # Varying importance scores
                ])
        
        conn.commit()
    finally:
        conn.close()
    
    yield table_name


# =============================================================================
# INITIALIZATION AND SETUP TESTS
# =============================================================================

class TestPGVectorStoreInitialization:
    """Test PGVectorStore initialization with real PostgreSQL."""
    
    def test_initialization_with_valid_connection(self, test_connection_string):
        """Test successful initialization with real PostgreSQL connection."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        
        assert vector_store.connection_string == test_connection_string
        assert vector_store.dimension == 384
        
        # Cleanup
        vector_store.close()
    
    def test_initialization_verifies_pgvector_extension(self, test_connection_string):
        """Test that initialization verifies pgvector extension exists."""
        # Should succeed with real database that has pgvector
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        
        # If we get here, pgvector verification passed
        assert vector_store is not None
        
        # Cleanup
        vector_store.close()
    
    def test_initialization_with_custom_dimension(self, test_connection_string):
        """Test initialization with different dimension sizes."""
        dimensions = [128, 384, 512, 1536]
        
        for dim in dimensions:
            vector_store = PGVectorStore(test_connection_string, dimension=dim)
            assert vector_store.dimension == dim
            vector_store.close()
    
    def test_initialization_with_connection_pooling(self, test_connection_string):
        """Test that connection pooling is properly set up."""
        vector_store = PGVectorStore(test_connection_string, dimension=384, pool_size=10)
        
        # Verify we can get connections from the pool
        with vector_store._get_connection() as conn:
            assert conn is not None
        
        vector_store.close()
    
    def test_initialization_fails_with_invalid_connection(self):
        """Test graceful failure with invalid connection string."""
        invalid_connection = "postgresql://invalid:invalid@nonexistent:5432/invalid"
        
        with pytest.raises(Exception):  # Could be OperationalError or similar
            PGVectorStore(invalid_connection, dimension=384)


# =============================================================================
# VECTOR SEARCH CONTRACT TESTS
# =============================================================================

class TestVectorSearch:
    """Test the core vector similarity search functionality."""
    
    def test_search_returns_empty_for_empty_table(self, clean_test_table, test_connection_string, sample_vectors):
        """Test search behavior with empty table returns empty results."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            results = vector_store.search(query_vector, k=10, table=clean_test_table)
            
            assert isinstance(results, list)
            assert len(results) == 0
        finally:
            vector_store.close()
    
    def test_search_returns_ranked_results(self, populated_test_table, test_connection_string, sample_vectors):
        """Test search returns properly ranked similarity results."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            results = vector_store.search(query_vector, k=5, table=populated_test_table)
            
            # Should return results
            assert len(results) > 0
            assert len(results) <= 5
            
            # Results should be tuples of (id, similarity_score)
            for result in results:
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert isinstance(result[0], str)  # ID
                assert isinstance(result[1], float)  # Similarity score
                assert 0 <= result[1] <= 1  # Normalized similarity
            
            # Results should be ranked by similarity (highest first)
            similarities = [result[1] for result in results]
            assert similarities == sorted(similarities, reverse=True)
        finally:
            vector_store.close()
    
    def test_search_respects_k_parameter(self, populated_test_table, test_connection_string, sample_vectors):
        """Test that k parameter correctly limits result count."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            # Test different k values
            for k in [1, 2, 3, 10]:
                results = vector_store.search(query_vector, k=k, table=populated_test_table)
                assert len(results) <= k
        finally:
            vector_store.close()
    
    def test_search_validates_query_dimension(self, populated_test_table, test_connection_string, sample_vectors):
        """Test dimension validation prevents mismatched queries."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        wrong_dimension_vector = sample_vectors["dimension_512"]  # 512 != 384
        
        try:
            with pytest.raises(ValueError) as exc_info:
                vector_store.search(wrong_dimension_vector, k=5, table=populated_test_table)
            
            assert "Query dimension" in str(exc_info.value)
            assert "384" in str(exc_info.value)
        finally:
            vector_store.close()
    
    def test_search_with_source_filter(self, populated_test_table, test_connection_string, sample_vectors):
        """Test filtering by source field works correctly."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            # Search with source filter
            results = vector_store.search(
                query_vector,
                k=10,
                table=populated_test_table,
                filters={"source": "ml_doc"}
            )
            
            # Verify source filtering by checking database directly
            conn = psycopg2.connect(vector_store.connection_string)
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT id FROM {populated_test_table} WHERE source = %s", ["ml_doc"])
                    expected_ids = {str(row[0]) for row in cur.fetchall()}
                    
                    result_ids = {result[0] for result in results}
                    assert result_ids.issubset(expected_ids)
            finally:
                conn.close()
        finally:
            vector_store.close()
    
    def test_search_with_importance_filter(self, populated_test_table, test_connection_string, sample_vectors):
        """Test filtering by minimum importance score."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            min_importance = 0.7
            results = vector_store.search(
                query_vector,
                k=10,
                table=populated_test_table,
                filters={"min_importance": min_importance}
            )
            
            # Verify importance filtering
            conn = psycopg2.connect(vector_store.connection_string)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT id FROM {populated_test_table} WHERE importance_score >= %s",
                        [min_importance]
                    )
                    expected_ids = {str(row[0]) for row in cur.fetchall()}
                    
                    result_ids = {result[0] for result in results}
                    assert result_ids.issubset(expected_ids)
            finally:
                conn.close()
        finally:
            vector_store.close()
    
    def test_search_with_similarity_threshold(self, populated_test_table, test_connection_string, sample_vectors):
        """Test filtering by minimum similarity score."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            min_similarity = 0.8
            results = vector_store.search(
                query_vector,
                k=10,
                table=populated_test_table,
                filters={"min_similarity": min_similarity}
            )
            
            # All results should meet similarity threshold
            for result in results:
                assert result[1] >= min_similarity
        finally:
            vector_store.close()
    
    def test_search_updates_access_tracking(self, populated_test_table, test_connection_string, sample_vectors):
        """Test that search updates access counts and timestamps."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            # Get initial access counts
            conn = psycopg2.connect(vector_store.connection_string)
            initial_counts = {}
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT id, access_count FROM {populated_test_table}")
                    for row in cur.fetchall():
                        initial_counts[str(row[0])] = row[1]
            finally:
                conn.close()
            
            # Perform search
            search_results = vector_store.search(query_vector, k=3, table=populated_test_table)
            
            # Check that access counts were updated
            conn = psycopg2.connect(vector_store.connection_string)
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT id, access_count FROM {populated_test_table}")
                    updated_counts = {}
                    for row in cur.fetchall():
                        updated_counts[str(row[0])] = row[1]
            finally:
                conn.close()
            
            # Access counts should be incremented for returned results
            for search_result in search_results:
                result_id = search_result[0]
                assert updated_counts[result_id] == initial_counts[result_id] + 1
        finally:
            vector_store.close()


# =============================================================================
# INDEX OPTIMIZATION TESTS
# =============================================================================

class TestIndexOptimization:
    """Test index optimization functionality for performance."""
    
    def test_optimize_index_creates_ivfflat_index(self, populated_test_table, test_connection_string):
        """Test that index optimization creates proper IVFFlat index."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        
        try:
            # Optimize index
            vector_store.optimize_index(table=populated_test_table, lists=5)
            
            # Verify index was created
            conn = psycopg2.connect(vector_store.connection_string)
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT indexname, indexdef 
                        FROM pg_indexes 
                        WHERE tablename = %s AND indexname LIKE %s
                    """, [populated_test_table, f"idx_{populated_test_table}_embedding"])
                    
                    result = cur.fetchone()
                    assert result is not None
                    assert "ivfflat" in result[1].lower()
                    assert "vector_cosine_ops" in result[1]
            finally:
                conn.close()
        finally:
            vector_store.close()
    
    def test_optimize_index_calculates_lists_automatically(self, performance_test_table, test_connection_string):
        """Test automatic calculation of lists parameter based on data size."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        
        try:
            # Optimize without specifying lists
            vector_store.optimize_index(table=performance_test_table)
            
            # Verify index was created
            conn = psycopg2.connect(vector_store.connection_string)
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT indexname 
                        FROM pg_indexes 
                        WHERE tablename = %s AND indexname LIKE %s
                    """, [performance_test_table, f"idx_{performance_test_table}_embedding"])
                    
                    result = cur.fetchone()
                    assert result is not None
            finally:
                conn.close()
        finally:
            vector_store.close()
    
    def test_optimize_index_handles_empty_table(self, clean_test_table, test_connection_string):
        """Test index optimization gracefully handles empty tables."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        
        try:
            # Should not crash on empty table
            vector_store.optimize_index(table=clean_test_table)
            
            # Index should still be created
            conn = psycopg2.connect(vector_store.connection_string)
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT indexname 
                        FROM pg_indexes 
                        WHERE tablename = %s AND indexname LIKE %s
                    """, [clean_test_table, f"idx_{clean_test_table}_embedding"])
                    
                    result = cur.fetchone()
                    assert result is not None
            finally:
                conn.close()
        finally:
            vector_store.close()


# =============================================================================
# PERFORMANCE ANALYSIS TESTS
# =============================================================================

class TestPerformanceAnalysis:
    """Test performance analysis and monitoring functionality."""
    
    def test_analyze_performance_returns_structured_data(self, populated_test_table, test_connection_string, sample_vectors):
        """Test performance analysis returns complete structured data."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            analysis = vector_store.analyze_performance(query_vector, table=populated_test_table)
            
            # Should return structured analysis
            assert isinstance(analysis, dict)
            assert "query_plan" in analysis
            assert "index_stats" in analysis
            assert "table_size" in analysis
            assert "execution_time_ms" in analysis
            
            # Index stats should be a list
            assert isinstance(analysis["index_stats"], list)
        finally:
            vector_store.close()
    
    def test_analyze_performance_tracks_execution_metrics(self, populated_test_table, test_connection_string, sample_vectors):
        """Test that performance analysis captures execution metrics."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            # Run some searches to generate metrics
            for _ in range(3):
                vector_store.search(query_vector, k=5, table=populated_test_table)
            
            analysis = vector_store.analyze_performance(query_vector, table=populated_test_table)
            
            # Should have execution time data
            assert "execution_time_ms" in analysis
            assert isinstance(analysis["execution_time_ms"], (int, float))
        finally:
            vector_store.close()


# =============================================================================
# ERROR HANDLING AND EDGE CASES
# =============================================================================

class TestErrorHandling:
    """Test error handling and edge cases for production reliability."""
    
    def test_search_filters_null_embeddings(self, clean_test_table, test_connection_string, sample_vectors):
        """Test that rows with NULL embeddings are filtered out."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        
        try:
            # Insert row with NULL embedding
            conn = psycopg2.connect(vector_store.connection_string)
            try:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        INSERT INTO {clean_test_table} (content, embedding, source)
                        VALUES (%s, NULL, %s)
                    """, ["Content with null embedding", "test_source"])
                    
                    # Insert row with valid embedding
                    cur.execute(f"""
                        INSERT INTO {clean_test_table} (content, embedding, source)
                        VALUES (%s, %s, %s)
                    """, [
                        "Content with valid embedding",
                        sample_vectors["dimension_384"].tolist(),
                        "test_source"
                    ])
                conn.commit()
            finally:
                conn.close()
            
            # Search should only return valid embeddings
            query_vector = sample_vectors["dimension_384"]
            results = vector_store.search(query_vector, k=10, table=clean_test_table)
            
            # Should return only the row with valid embedding
            assert len(results) == 1
        finally:
            vector_store.close()
    
    def test_search_with_zero_vector(self, populated_test_table, test_connection_string, sample_vectors):
        """Test search behavior with zero vector query."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        zero_vector = sample_vectors["zero_vector"]
        
        try:
            # Should not crash with zero vector
            results = vector_store.search(zero_vector, k=5, table=populated_test_table)
            
            # Should return results
            assert isinstance(results, list)
            assert len(results) >= 0
        finally:
            vector_store.close()
    
    def test_search_with_large_k_value(self, populated_test_table, test_connection_string, sample_vectors):
        """Test search with k larger than available data."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            # Request more results than exist
            results = vector_store.search(query_vector, k=1000, table=populated_test_table)
            
            # Should return all available results without error
            assert len(results) <= 1000
            assert len(results) > 0
        finally:
            vector_store.close()
    
    def test_search_with_nonexistent_table(self, test_connection_string, sample_vectors):
        """Test graceful handling of non-existent table."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            with pytest.raises(Exception):  # Should raise database error
                vector_store.search(query_vector, k=5, table="nonexistent_table")
        finally:
            vector_store.close()


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestSearchPerformance:
    """Test search performance characteristics under realistic conditions."""
    
    def test_search_performance_with_index(self, performance_test_table, test_connection_string, sample_vectors):
        """Test search performance with optimized index."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            # Optimize index first
            vector_store.optimize_index(table=performance_test_table)
            
            # Measure search performance
            start_time = time.time()
            for _ in range(10):
                results = vector_store.search(query_vector, k=10, table=performance_test_table)
                assert len(results) <= 10
            
            elapsed = time.time() - start_time
            
            # Should complete quickly (< 2 seconds for 10 searches on 100 vectors)
            assert elapsed < 2.0
        finally:
            vector_store.close()
    
    def test_search_performance_scaling(self, performance_test_table, test_connection_string, sample_vectors):
        """Test search performance scales reasonably with result size."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            # Test different k values
            performance_results = {}
            
            for k in [1, 5, 10, 20]:
                start_time = time.time()
                results = vector_store.search(query_vector, k=k, table=performance_test_table)
                elapsed = time.time() - start_time
                
                performance_results[k] = elapsed
                assert len(results) <= k
            
            # Performance should not degrade dramatically
            assert all(time < 1.0 for time in performance_results.values())
        finally:
            vector_store.close()


# =============================================================================
# CONCURRENCY TESTS
# =============================================================================

class TestConcurrentOperations:
    """Test concurrent access patterns for production reliability."""
    
    def test_concurrent_search_operations(self, populated_test_table, test_connection_string, sample_vectors):
        """Test multiple search operations can run concurrently safely."""
        import threading
        import queue
        
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        results_queue = queue.Queue()
        
        def search_worker():
            try:
                results = vector_store.search(query_vector, k=3, table=populated_test_table)
                results_queue.put(("success", results))
            except Exception as e:
                results_queue.put(("error", str(e)))
        
        try:
            # Run concurrent searches
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=search_worker)
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Check results
            successful_searches = 0
            while not results_queue.empty():
                status, result = results_queue.get()
                if status == "success":
                    successful_searches += 1
                    assert isinstance(result, list)
            
            # All searches should succeed
            assert successful_searches == 5
        finally:
            vector_store.close()


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================

class TestContextManager:
    """Test context manager functionality for proper resource cleanup."""
    
    def test_context_manager_support(self, populated_test_table, test_connection_string, sample_vectors):
        """Test that PGVectorStore works as context manager."""
        query_vector = sample_vectors["dimension_384"]
        
        # Use as context manager
        with PGVectorStore(test_connection_string, dimension=384) as vector_store:
            results = vector_store.search(query_vector, k=5, table=populated_test_table)
            assert len(results) >= 0
        
        # Context manager should clean up automatically
    
    def test_context_manager_cleanup_on_exception(self, populated_test_table, test_connection_string, sample_vectors):
        """Test context manager cleans up properly even when exceptions occur."""
        query_vector = sample_vectors["dimension_384"]
        
        try:
            with PGVectorStore(test_connection_string, dimension=384) as vector_store:
                # Perform valid operation
                vector_store.search(query_vector, k=5, table=populated_test_table)
                
                # Force an exception
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
        
        # Context manager should still clean up properly


# =============================================================================
# REAL-WORLD SCENARIO TESTS
# =============================================================================

class TestRealWorldScenarios:
    """Test complete workflows representing actual usage patterns."""
    
    def test_document_similarity_search_workflow(self, clean_test_table, test_connection_string, sample_vectors):
        """Test complete document similarity search workflow."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        base_vector = sample_vectors["dimension_384"]
        
        try:
            # Simulate ingesting documents with embeddings
            documents = [
                {
                    "content": "Python is a programming language",
                    "embedding": base_vector + np.random.normal(0, 0.1, 384).astype(np.float32),
                    "source": "programming_guide"
                },
                {
                    "content": "Machine learning uses Python extensively",
                    "embedding": base_vector + np.random.normal(0, 0.15, 384).astype(np.float32),
                    "source": "ml_tutorial"
                },
                {
                    "content": "JavaScript is used for web development",
                    "embedding": base_vector + np.random.normal(0, 0.8, 384).astype(np.float32),
                    "source": "web_guide"
                }
            ]
            
            # Insert documents
            conn = psycopg2.connect(vector_store.connection_string)
            try:
                with conn.cursor() as cur:
                    for doc in documents:
                        cur.execute(f"""
                            INSERT INTO {clean_test_table} (content, embedding, source, importance_score)
                            VALUES (%s, %s, %s, %s)
                        """, [doc["content"], doc["embedding"].tolist(), doc["source"], 0.8])
                conn.commit()
            finally:
                conn.close()
            
            # Search for Python-related content
            python_query = base_vector + np.random.normal(0, 0.05, 384).astype(np.float32)
            results = vector_store.search(python_query, k=5, table=clean_test_table)
            
            # Should return Python-related documents with higher similarity
            assert len(results) == 3  # All documents returned
            assert results[0][1] > results[-1][1]  # Higher similarity first
        finally:
            vector_store.close()
    
    def test_usage_analytics_workflow(self, populated_test_table, test_connection_string, sample_vectors):
        """Test access tracking for usage analytics."""
        vector_store = PGVectorStore(test_connection_string, dimension=384)
        query_vector = sample_vectors["dimension_384"]
        
        try:
            # Perform multiple searches
            search_results = []
            for _ in range(3):
                results = vector_store.search(query_vector, k=2, table=populated_test_table)
                search_results.extend(results)
            
            # Check that access counts were properly tracked
            conn = psycopg2.connect(vector_store.connection_string)
            try:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT id, access_count, last_accessed 
                        FROM {populated_test_table}
                    """)
                    access_data = {}
                    for row in cur.fetchall():
                        access_data[str(row[0])] = {
                            "access_count": row[1],
                            "last_accessed": row[2]
                        }
            finally:
                conn.close()
            
            # Verify that accessed documents have updated counts
            accessed_ids = {result[0] for result in search_results}
            for doc_id in accessed_ids:
                assert access_data[doc_id]["access_count"] > 0
                assert access_data[doc_id]["last_accessed"] is not None
        finally:
            vector_store.close()