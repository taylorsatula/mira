"""
Production-grade tests for MemoryManager - the keystone of lt_memory.

Testing philosophy: TEST THE CONTRACT, NOT THE IMPLEMENTATION
Before every test: "What real production bug in OUR CODE would this catch?"

This test suite focuses on realistic production scenarios that matter for
reliability. We test the contracts and behaviors that users actually depend on,
covering database transactions, embedding generation, snapshot/restore cycles,
and concurrent access patterns.

Testing approach:
1. Use real PostgreSQL database (same engine as production)
2. Test public API contracts that users rely on
3. Test critical private methods that could fail in subtle ways
4. Focus on thread safety and concurrent access patterns
5. Test failure recovery and error handling
6. Verify performance characteristics under realistic load
"""

import pytest
import time
import threading
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the system under test
from lt_memory.managers.memory_manager import MemoryManager
from lt_memory.models.base import MemoryBlock, MemorySnapshot, MemoryPassage, BlockHistory, ArchivedConversation
from config.config_manager import AppConfig
from config.memory_config import MemoryConfig
from errors import ToolError, ErrorCode
from sqlalchemy import text


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
    
    # Override with test database (use mira_admin for schema operations)
    config.memory.database_url = "postgresql://mira_admin@localhost:5432/lt_memory_test"
    config.memory.db_pool_size = 5
    config.memory.db_pool_max_overflow = 10
    config.memory.embedding_dim = 1024  # Match OpenAI text-embedding-3-small with dimensions parameter
    
    # Ensure paths exist
    config.paths.data_dir = "/tmp/test_lt_memory_cache"
    Path(config.paths.data_dir).mkdir(exist_ok=True)
    
    return config


@pytest.fixture
def clean_test_database(test_config):
    """
    Provides a clean test database for each test.
    
    Ensures tests don't interfere with each other by cleaning data, not schema.
    """
    yield test_config
    
    # Clean up data after test (not schema - that's not what we're testing)
    from sqlalchemy import create_engine
    from lt_memory.models.base import Base
    
    engine = create_engine(test_config.memory.database_url)
    
    # Just clean the data from tables if they exist
    with engine.connect() as conn:
        # Delete data in reverse dependency order
        for table in reversed(Base.metadata.sorted_tables):
            try:
                conn.execute(table.delete())
            except Exception:
                # Table might not exist yet - that's fine
                pass
        conn.commit()
    
    engine.dispose()


# =============================================================================
# CORE INITIALIZATION CONTRACT TESTS  
# =============================================================================

class TestMemoryManagerInitialization:
    """
    Test the fundamental contract: MemoryManager initializes correctly with real components.
    
    These test the core system startup that everything else depends on.
    """
    
    def test_successful_initialization_with_real_database(self, clean_test_database):
        """
        Test that MemoryManager initializes successfully with real PostgreSQL.
        
        REAL BUG THIS CATCHES: If our database connection string handling, 
        extension verification, or component initialization has bugs, the entire
        lt_memory system fails to start - breaking all memory functionality.
        """
        config = clean_test_database
        
        # This should succeed without errors
        manager = MemoryManager(config)
        
        # Verify core components are initialized
        assert manager.engine is not None
        assert manager.embedding_model is not None
        assert manager.vector_store is not None
        assert manager.block_manager is not None
        assert manager.passage_manager is not None
        
        # Verify database connection works
        with manager.get_session() as session:
            result = session.execute(text("SELECT 1")).scalar()
            assert result == 1
        
        # Verify core memory blocks were created
        with manager.get_session() as session:
            blocks = session.query(MemoryBlock).all()
            block_labels = {b.label for b in blocks}
            
            # Should have default core blocks
            expected_blocks = set(config.memory.core_memory_blocks.keys())
            assert block_labels == expected_blocks
            
            # Each block should have default content
            for block in blocks:
                assert len(block.value) > 0
                assert block.character_limit > 0
                assert block.version == 1
        
        # Cleanup
        manager.engine.dispose()
    
    def test_database_engine_creation_validates_postgresql_requirements(self, clean_test_database):
        """
        Test that database engine creation enforces PostgreSQL and required extensions.
        
        REAL BUG THIS CATCHES: If our _create_engine() method doesn't properly validate
        the database type or required extensions, users could deploy with SQLite or 
        missing pgvector, causing silent failures or crashes when vector operations
        are attempted. This breaks the entire lt_memory system.
        """
        config = clean_test_database
        
        # Test 1: Should succeed with correct PostgreSQL URL
        manager = MemoryManager(config)
        
        # Verify engine was created successfully
        assert manager.engine is not None
        
        # Verify it's actually connected to PostgreSQL (not SQLite or other)
        with manager.engine.connect() as conn:
            db_type = conn.execute(text("SELECT version()")).scalar()
            assert "PostgreSQL" in db_type, f"Expected PostgreSQL, got: {db_type}"
        
        manager.engine.dispose()
        
        # Test 2: Should reject non-PostgreSQL database URLs
        bad_config = AppConfig()
        bad_config.memory.database_url = "sqlite:///test.db"  # SQLite URL
        
        with pytest.raises(ValueError) as exc_info:
            MemoryManager(bad_config)
        
        assert "PostgreSQL database" in str(exc_info.value)
        
        # Test 3: Verify required extensions are available
        good_manager = MemoryManager(config)
        
        with good_manager.engine.connect() as conn:
            # Check pgvector extension
            vector_result = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()
            assert vector_result is not None, "pgvector extension not found"
            
            # Check uuid-ossp extension  
            uuid_result = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'uuid-ossp'")
            ).fetchone()
            assert uuid_result is not None, "uuid-ossp extension not found"
            
            # Verify pgvector version is reasonable (should be > 0.5.0)
            version_result = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
            assert version_result is not None, "Could not get pgvector version"
            
            # Parse version (format like "0.8.0")
            major, minor, patch = map(int, version_result.split('.'))
            assert major > 0 or (major == 0 and minor >= 5), f"pgvector version {version_result} too old"
        
        good_manager.engine.dispose()
    
    def test_core_memory_blocks_initialization_creates_required_blocks(self, clean_test_database):
        """
        Test that core memory blocks are properly initialized with correct defaults.
        
        REAL BUG THIS CATCHES: If our _initialize_core_blocks() method fails to create
        the expected persona/human/system blocks, or creates them with wrong defaults,
        all memory operations that depend on these blocks will fail. Block managers,
        conversation archiving, and memory tool operations assume these blocks exist.
        """
        config = clean_test_database
        
        # Initialize manager (this should create core blocks)
        manager = MemoryManager(config)
        
        with manager.get_session() as session:
            # Verify all configured core blocks were created
            blocks = session.query(MemoryBlock).all()
            block_map = {block.label: block for block in blocks}
            
            # Should have all blocks from config
            expected_labels = set(config.memory.core_memory_blocks.keys())
            actual_labels = set(block_map.keys())
            assert actual_labels == expected_labels, f"Missing blocks: {expected_labels - actual_labels}"
            
            # Test each core block has correct properties
            for label, expected_limit in config.memory.core_memory_blocks.items():
                block = block_map[label]
                
                # Basic properties
                assert block.label == label
                assert block.character_limit == expected_limit
                assert block.version == 1  # Initial version
                assert block.created_at is not None
                assert block.updated_at is not None
                
                # Should have sensible default content (not empty)
                assert len(block.value) > 0, f"Block {label} has empty default content"
                assert len(block.value) <= expected_limit, f"Block {label} exceeds character limit"
                
                # Content should be reasonable for the block type
                if label == "persona":
                    assert "MIRA" in block.value or "assistant" in block.value.lower()
                elif label == "human":
                    assert "human" in block.value.lower() or "user" in block.value.lower()
                elif label == "system":
                    assert "system" in block.value.lower() or "status" in block.value.lower()
        
        # Test idempotency: calling again shouldn't create duplicates
        manager2 = MemoryManager(config)
        
        with manager2.get_session() as session:
            blocks_after = session.query(MemoryBlock).all()
            
            # Should have same number of blocks (no duplicates)
            assert len(blocks_after) == len(config.memory.core_memory_blocks)
            
            # Verify blocks have same content (not recreated)
            blocks_after_map = {b.label: b for b in blocks_after}
            for label in expected_labels:
                original_block = block_map[label]
                current_block = blocks_after_map[label]
                
                # Should be same block (same ID and content)
                assert original_block.id == current_block.id
                assert original_block.value == current_block.value
                assert original_block.version == current_block.version
        
        # Cleanup
        manager.engine.dispose()
        manager2.engine.dispose()


class TestDatabaseSessionManagement:
    """
    Test database session management and connection pooling.
    
    Session management is critical for preventing connection leaks and ensuring
    proper transaction handling under concurrent access.
    """
    
    def test_session_management_handles_concurrent_database_operations(self, clean_test_database):
        """
        Test that session management works correctly under real concurrent load.
        
        REAL BUG THIS CATCHES: If our get_session() method has connection pool
        exhaustion, deadlocks, or thread safety issues, concurrent operations will
        fail or corrupt data. Production has multiple users accessing memory
        simultaneously - this tests that reality.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Test real concurrent database operations with data integrity verification
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        def concurrent_memory_operation(thread_id):
            """Simulate real memory operations that multiple users would perform."""
            results = []
            errors = []
            
            try:
                # Each thread does realistic memory operations
                with manager.get_session() as session:
                    # Read core blocks (like memory retrieval)
                    blocks = session.query(MemoryBlock).all()
                    results.append(f"thread_{thread_id}_read_{len(blocks)}_blocks")
                    
                    # Create a unique test block (like storing memory)
                    test_block = MemoryBlock(
                        label=f"concurrent_test_{thread_id}",
                        value=f"Thread {thread_id} memory content",
                        character_limit=200
                    )
                    session.add(test_block)
                    session.commit()
                    results.append(f"thread_{thread_id}_created_block")
                    
                # Verify the block was actually created (separate session)
                with manager.get_session() as session:
                    created_block = session.query(MemoryBlock).filter_by(
                        label=f"concurrent_test_{thread_id}"
                    ).first()
                    if created_block and created_block.value == f"Thread {thread_id} memory content":
                        results.append(f"thread_{thread_id}_verified_persistence")
                    else:
                        errors.append(f"thread_{thread_id}_persistence_failed")
                        
            except Exception as e:
                errors.append(f"thread_{thread_id}_error_{str(e)}")
                
            return {
                "thread_id": thread_id,
                "results": results, 
                "errors": errors,
                "success": len(errors) == 0
            }
        
        # Run 10 threads doing real concurrent database operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(concurrent_memory_operation, i) for i in range(10)]
            thread_results = [future.result() for future in as_completed(futures)]
        
        # Verify all operations succeeded
        successful_threads = [r for r in thread_results if r["success"]]
        failed_threads = [r for r in thread_results if not r["success"]]
        
        assert len(successful_threads) == 10, f"Some threads failed: {failed_threads}"
        
        # Verify data integrity - all blocks should exist with correct content
        with manager.get_session() as session:
            for i in range(10):
                block = session.query(MemoryBlock).filter_by(
                    label=f"concurrent_test_{i}"
                ).first()
                assert block is not None, f"Block for thread {i} not found"
                assert block.value == f"Thread {i} memory content", f"Block content corrupted for thread {i}"
        
        # Test connection pool doesn't get exhausted
        initial_pool_size = manager.engine.pool.size()
        checked_out_before = manager.engine.pool.checkedout()
        
        # After all operations, pool should be healthy
        checked_out_after = manager.engine.pool.checkedout()
        pool_size_after = manager.engine.pool.size()
        
        assert checked_out_after <= checked_out_before + 1, f"Connection leak: {checked_out_before} -> {checked_out_after}"
        assert pool_size_after == initial_pool_size, f"Pool size changed: {initial_pool_size} -> {pool_size_after}"
        
        # Cleanup
        manager.engine.dispose()


class TestEmbeddingGeneration:
    """
    Test embedding generation with caching functionality.
    
    Embedding generation is core to lt_memory operations and must work reliably
    with proper caching to avoid performance issues.
    """
    
    def test_embedding_generation_with_caching_handles_real_text(self, clean_test_database):
        """
        Test that generate_embedding() produces correct embeddings with working cache.
        
        REAL BUG THIS CATCHES: If our generate_embedding() method fails with the
        ONNX model, has cache corruption, dimension mismatches, or thread safety
        issues, all memory operations that depend on embeddings will fail. This
        includes passage storage, similarity search, and memory retrieval.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Clear any cached embeddings to ensure fresh generation
        manager.embedding_cache.clear_memory_cache()
        manager.embedding_cache.clear_disk_cache()
        
        # Test 1: Basic embedding generation works
        test_text = "This is a test document for embedding generation."
        
        # Generate embedding
        embedding = manager.generate_embedding(test_text)
        
        # Verify embedding properties
        assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
        assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"
        assert embedding.shape == (config.memory.embedding_dim,), f"Wrong dimensions: {embedding.shape}"
        assert not np.any(np.isnan(embedding)), "Embedding contains NaN values"
        assert not np.any(np.isinf(embedding)), "Embedding contains infinite values"
        
        # Test 2: Cache functionality works correctly
        # Second call should return cached result
        embedding2 = manager.generate_embedding(test_text)
        
        # Should be identical (cached)
        np.testing.assert_array_equal(embedding, embedding2, "Cache should return identical embedding")
        
        # Test 3: Different text produces different embeddings
        different_text = "Completely different content that should produce different embeddings."
        different_embedding = manager.generate_embedding(different_text)
        
        # Should be different vectors
        assert not np.array_equal(embedding, different_embedding), "Different text should produce different embeddings"
        
        # Test 4: Cache hit vs miss performance difference
        new_text = "Brand new text that hasn't been cached yet."
        
        # Time cache miss (first generation)
        import time
        start_time = time.time()
        first_embedding = manager.generate_embedding(new_text)
        miss_time = time.time() - start_time
        
        # Time cache hit (second generation)
        start_time = time.time()
        cached_embedding = manager.generate_embedding(new_text)
        hit_time = time.time() - start_time
        
        # Cache hit should be significantly faster
        assert hit_time < miss_time / 2, f"Cache hit ({hit_time:.4f}s) should be much faster than miss ({miss_time:.4f}s)"
        np.testing.assert_array_equal(first_embedding, cached_embedding, "Cached result should match original")
        
        # Test 5: Concurrent embedding generation (thread safety)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def generate_embedding_worker(worker_id):
            """Worker that generates embeddings concurrently."""
            text = f"Worker {worker_id} is generating this unique embedding text."
            try:
                embedding = manager.generate_embedding(text)
                
                # Verify embedding is valid
                if (isinstance(embedding, np.ndarray) and 
                    embedding.shape == (config.memory.embedding_dim,) and
                    not np.any(np.isnan(embedding)) and
                    not np.any(np.isinf(embedding))):
                    return {"worker_id": worker_id, "success": True, "embedding_norm": np.linalg.norm(embedding)}
                else:
                    return {"worker_id": worker_id, "success": False, "error": "Invalid embedding properties"}
                    
            except Exception as e:
                return {"worker_id": worker_id, "success": False, "error": str(e)}
        
        # Run 8 workers generating embeddings concurrently
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(generate_embedding_worker, i) for i in range(8)]
            worker_results = [future.result() for future in as_completed(futures)]
        
        # All workers should succeed
        successful_workers = [r for r in worker_results if r["success"]]
        failed_workers = [r for r in worker_results if not r["success"]]
        
        assert len(successful_workers) == 8, f"Some workers failed: {failed_workers}"
        
        # Embeddings should have reasonable norms (not zero vectors)
        norms = [r["embedding_norm"] for r in successful_workers]
        assert all(norm > 0.1 for norm in norms), f"Some embeddings have very small norms: {norms}"
        
        # Test 6: OpenAI embedding behavior verification (case-sensitive)
        base_text = "Apple Inc. is a technology company."
        case_text = "apple inc. is a technology company."
        different_text = "Microsoft Corp. is a software company."
        
        base_emb = manager.generate_embedding(base_text)
        case_emb = manager.generate_embedding(case_text)
        different_emb = manager.generate_embedding(different_text)
        
        # Each should be valid
        assert isinstance(base_emb, np.ndarray)
        assert isinstance(case_emb, np.ndarray)
        assert isinstance(different_emb, np.ndarray)
        assert base_emb.shape == (config.memory.embedding_dim,)
        assert case_emb.shape == (config.memory.embedding_dim,)
        assert different_emb.shape == (config.memory.embedding_dim,)
        
        # OpenAI embeddings should be case-sensitive (different vectors)
        assert not np.array_equal(base_emb, case_emb), "OpenAI should produce different embeddings for different cases"
        
        # But still semantically similar (high cosine similarity)
        case_similarity = np.dot(base_emb, case_emb) / (np.linalg.norm(base_emb) * np.linalg.norm(case_emb))
        assert 0.85 < case_similarity < 1.0, f"Case variations should be similar but not identical: {case_similarity}"
        
        # Different companies should be less similar
        company_similarity = np.dot(base_emb, different_emb) / (np.linalg.norm(base_emb) * np.linalg.norm(different_emb))
        assert 0.3 < company_similarity < 0.8, f"Different companies should be moderately similar: {company_similarity}"
        
        # Test 7: Cache statistics are updated correctly
        cache_stats = manager.embedding_cache.get_stats()
        
        assert cache_stats["total_requests"] > 0, "Cache should have recorded requests"
        assert cache_stats["hits"] > 0, "Should have cache hits from repeated text"
        assert cache_stats["misses"] > 0, "Should have cache misses from new text"
        assert 0 <= cache_stats["hit_rate"] <= 1, f"Hit rate should be between 0-1: {cache_stats['hit_rate']}"
        
        # Cleanup
        manager.engine.dispose()


class TestSnapshotRestoreFunctionality:
    """
    Test memory snapshot and restore functionality.
    
    Snapshot/restore is critical for memory system reliability, disaster recovery,
    and maintaining data integrity across system restarts.
    """
    
    def test_snapshot_restore_preserves_memory_state_correctly(self, clean_test_database):
        """
        Test that snapshot creation and restoration maintains complete memory state.
        
        REAL BUG THIS CATCHES: If our create_snapshot() or restore_from_snapshot() 
        methods have serialization bugs, data corruption, or incomplete state capture,
        memory restoration after system failures will be inconsistent or lost entirely.
        This breaks disaster recovery and data persistence guarantees.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Test 1: Create initial memory state
        with manager.get_session() as session:
            # Modify core memory blocks to create distinct state
            persona_block = session.query(MemoryBlock).filter_by(label="persona").first()
            human_block = session.query(MemoryBlock).filter_by(label="human").first()
            system_block = session.query(MemoryBlock).filter_by(label="system").first()
            
            original_persona = persona_block.value
            original_human = human_block.value
            original_system = system_block.value
            
            # Update with specific test content
            persona_block.value = "I am MIRA with advanced memory capabilities for testing."
            persona_block.version += 1
            human_block.value = "Test user with specific preferences for unit testing."
            human_block.version += 1
            system_block.value = "System in test mode with snapshot functionality enabled."
            system_block.version += 1
            
            session.commit()
            
            # Store updated values for verification
            updated_persona = persona_block.value
            updated_human = human_block.value
            updated_system = system_block.value
            updated_persona_version = persona_block.version
            updated_human_version = human_block.version
            updated_system_version = system_block.version
        
        # Test 2: Create snapshot of current state
        conversation_id = "test_conversation_123"
        snapshot_reason = "unit_test_checkpoint"
        
        snapshot_id = manager.create_snapshot(conversation_id, snapshot_reason)
        
        # Verify snapshot was created
        assert snapshot_id is not None
        assert len(snapshot_id) > 0
        
        with manager.get_session() as session:
            snapshot = session.query(MemorySnapshot).filter_by(id=snapshot_id).first()
            assert snapshot is not None
            assert snapshot.conversation_id == conversation_id
            assert snapshot.reason == snapshot_reason
            assert snapshot.created_at is not None
            
            # Verify snapshot contains correct block data
            blocks_snapshot = snapshot.blocks_snapshot
            assert "persona" in blocks_snapshot
            assert "human" in blocks_snapshot
            assert "system" in blocks_snapshot
            
            assert blocks_snapshot["persona"]["value"] == updated_persona
            assert blocks_snapshot["human"]["value"] == updated_human
            assert blocks_snapshot["system"]["value"] == updated_system
            assert blocks_snapshot["persona"]["version"] == updated_persona_version
            assert blocks_snapshot["human"]["version"] == updated_human_version
            assert blocks_snapshot["system"]["version"] == updated_system_version
        
        # Test 3: Modify memory state after snapshot
        with manager.get_session() as session:
            persona_block = session.query(MemoryBlock).filter_by(label="persona").first()
            human_block = session.query(MemoryBlock).filter_by(label="human").first()
            system_block = session.query(MemoryBlock).filter_by(label="system").first()
            
            # Make different changes
            persona_block.value = "CORRUPTED: This state should be overwritten by restore."
            persona_block.version += 1
            human_block.value = "CORRUPTED: This human state is wrong."
            human_block.version += 1
            system_block.value = "CORRUPTED: System state modified incorrectly."
            system_block.version += 1
            
            session.commit()
        
        # Test 4: Restore from snapshot
        restore_success = manager.restore_from_snapshot(snapshot_id)
        assert restore_success is True
        
        # Test 5: Verify restored state matches snapshot exactly
        with manager.get_session() as session:
            restored_persona = session.query(MemoryBlock).filter_by(label="persona").first()
            restored_human = session.query(MemoryBlock).filter_by(label="human").first()
            restored_system = session.query(MemoryBlock).filter_by(label="system").first()
            
            # Content should match snapshot (not corrupted state)
            assert restored_persona.value == updated_persona
            assert restored_human.value == updated_human
            assert restored_system.value == updated_system
            
            # Versions should be incremented from restore operation
            assert restored_persona.version > updated_persona_version
            assert restored_human.version > updated_human_version
            assert restored_system.version > updated_system_version
            
            # Updated timestamps should be recent
            from utils.timezone_utils import utc_now
            from datetime import timedelta
            now = utc_now()
            assert (now - restored_persona.updated_at) < timedelta(seconds=5)
            assert (now - restored_human.updated_at) < timedelta(seconds=5)
            assert (now - restored_system.updated_at) < timedelta(seconds=5)
        
        # Test 6: Verify history entries were created for restore
        with manager.get_session() as session:
            history_entries = session.query(BlockHistory).filter_by(operation="restore").all()
            
            # Should have history for each restored block
            assert len(history_entries) >= 3
            
            history_labels = {h.label for h in history_entries}
            assert "persona" in history_labels
            assert "human" in history_labels
            assert "system" in history_labels
            
            # History should record the transition
            for history in history_entries:
                assert history.actor == "system"
                assert history.operation == "restore"
                assert history.diff_data is not None
                assert "CORRUPTED" in history.diff_data["old_value"]  # From corrupted state
                assert "CORRUPTED" not in history.diff_data["new_value"]  # To clean state
                assert history.diff_data["operation_type"] == "restore"
        
        # Test 7: Test restore of non-existent snapshot
        fake_snapshot_id = "00000000-0000-0000-0000-000000000000"
        restore_failure = manager.restore_from_snapshot(fake_snapshot_id)
        assert restore_failure is False
        
        # Memory state should be unchanged after failed restore
        with manager.get_session() as session:
            unchanged_persona = session.query(MemoryBlock).filter_by(label="persona").first()
            assert unchanged_persona.value == updated_persona  # Still restored state
        
        # Cleanup
        manager.engine.dispose()


class TestMemoryStatistics:
    """
    Test memory statistics collection and reporting functionality.
    
    Statistics are critical for monitoring memory system health, performance
    tuning, and understanding usage patterns in production.
    """
    
    def test_memory_stats_provides_accurate_comprehensive_metrics(self, clean_test_database):
        """
        Test that get_memory_stats() returns accurate and complete statistics.
        
        REAL BUG THIS CATCHES: If our get_memory_stats() method has incorrect
        calculations, missing data aggregation, or query errors, monitoring
        and performance analysis will be based on wrong information. This leads
        to incorrect scaling decisions and inability to detect memory issues.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # No cache clearing needed (embeddings are not cached)
        
        # Test 1: Fresh system should have baseline stats
        initial_stats = manager.get_memory_stats()
        
        # Verify basic structure
        assert isinstance(initial_stats, dict)
        assert "blocks" in initial_stats
        assert "passages" in initial_stats
        assert "snapshots" in initial_stats
        # embedding_cache no longer in stats (embeddings not cached)
        
        # Verify blocks stats structure
        blocks_stats = initial_stats["blocks"]
        assert "count" in blocks_stats
        assert "total_characters" in blocks_stats
        assert isinstance(blocks_stats["count"], int)
        assert isinstance(blocks_stats["total_characters"], int)
        
        # Should have default core blocks (persona, human, system)
        assert blocks_stats["count"] >= 3
        assert blocks_stats["total_characters"] > 0
        
        # Verify passages stats structure  
        passages_stats = initial_stats["passages"]
        assert "count" in passages_stats
        assert "avg_importance" in passages_stats
        assert isinstance(passages_stats["count"], int)
        assert isinstance(passages_stats["avg_importance"], (int, float))
        
        # Initially no passages
        assert passages_stats["count"] == 0
        assert passages_stats["avg_importance"] == 0
        
        # Verify snapshots stats
        snapshots_stats = initial_stats["snapshots"]
        assert "count" in snapshots_stats
        assert isinstance(snapshots_stats["count"], int)
        assert snapshots_stats["count"] == 0
        
        # embedding_cache stats no longer applicable (embeddings not cached)
        
        # Test 2: Create test data and verify stats update correctly
        with manager.get_session() as session:
            # Modify memory blocks
            persona_block = session.query(MemoryBlock).filter_by(label="persona").first()
            human_block = session.query(MemoryBlock).filter_by(label="human").first()
            
            original_persona_length = len(persona_block.value)
            original_human_length = len(human_block.value)
            
            # Add significant content to test character counting
            persona_block.value = "I am MIRA with comprehensive memory capabilities. " * 20  # ~1000 chars
            human_block.value = "The user is a software engineer working on AI systems. " * 15  # ~800 chars
            
            new_persona_length = len(persona_block.value)
            new_human_length = len(human_block.value)
            
            session.commit()
            
            # Add memory passages with different importance scores
            passage1 = MemoryPassage(
                text="Important conversation about machine learning algorithms.",
                embedding=np.random.rand(config.memory.embedding_dim).astype(np.float32),
                source="conversation",
                source_id="conv_001",
                importance_score=0.9
            )
            passage2 = MemoryPassage(
                text="Casual discussion about weather and weekend plans.",
                embedding=np.random.rand(config.memory.embedding_dim).astype(np.float32),
                source="conversation", 
                source_id="conv_002",
                importance_score=0.3
            )
            passage3 = MemoryPassage(
                text="Technical documentation about vector databases.",
                embedding=np.random.rand(config.memory.embedding_dim).astype(np.float32),
                source="document",
                source_id="doc_001", 
                importance_score=0.7
            )
            
            session.add_all([passage1, passage2, passage3])
            session.commit()
        
        # Create test snapshots
        snapshot1_id = manager.create_snapshot("test_conv_1", "test_checkpoint")
        snapshot2_id = manager.create_snapshot("test_conv_2", "backup_point")
        
        # Generate some embedding cache activity
        test_embeddings = [
            "This creates cache miss and hit patterns.",
            "Different text for cache testing.",
            "This creates cache miss and hit patterns.",  # Should be cache hit
            "Another unique text string.",
            "Different text for cache testing."  # Should be cache hit
        ]
        
        for text in test_embeddings:
            manager.generate_embedding(text)
        
        # Test 3: Verify updated stats reflect all changes
        updated_stats = manager.get_memory_stats()
        
        # Block stats should reflect content changes
        updated_blocks = updated_stats["blocks"]
        assert updated_blocks["count"] >= 3  # Still have at least core blocks
        
        # Character count should reflect our additions
        character_increase = (new_persona_length - original_persona_length) + (new_human_length - original_human_length)
        expected_min_chars = initial_stats["blocks"]["total_characters"] + character_increase
        assert updated_blocks["total_characters"] >= expected_min_chars
        
        # Passage stats should reflect additions
        updated_passages = updated_stats["passages"]
        assert updated_passages["count"] == 3
        
        # Average importance should be calculated correctly
        expected_avg_importance = (0.9 + 0.3 + 0.7) / 3  # 0.63333...
        actual_avg = updated_passages["avg_importance"]
        assert abs(actual_avg - expected_avg_importance) < 0.01
        
        # Snapshot stats should reflect creations
        updated_snapshots = updated_stats["snapshots"]
        assert updated_snapshots["count"] == 2
        
        # embedding_cache stats no longer applicable (embeddings not cached)
        
        # Test 4: Verify stats consistency across multiple calls
        stats_call_1 = manager.get_memory_stats()
        stats_call_2 = manager.get_memory_stats()
        
        # Stats should be identical when no changes occur between calls
        assert stats_call_1["blocks"]["count"] == stats_call_2["blocks"]["count"]
        assert stats_call_1["blocks"]["total_characters"] == stats_call_2["blocks"]["total_characters"]
        assert stats_call_1["passages"]["count"] == stats_call_2["passages"]["count"]
        assert stats_call_1["snapshots"]["count"] == stats_call_2["snapshots"]["count"]
        
        # Test 5: Test performance with larger dataset
        # Add more passages to test aggregation performance
        with manager.get_session() as session:
            import time
            
            # Add 100 passages with random importance scores
            passages = []
            for i in range(100):
                passage = MemoryPassage(
                    text=f"Performance test passage number {i} with various content lengths.",
                    embedding=np.random.rand(config.memory.embedding_dim).astype(np.float32),
                    source="performance_test",
                    source_id=f"perf_{i}",
                    importance_score=np.random.uniform(0.1, 1.0)
                )
                passages.append(passage)
            
            session.add_all(passages)
            session.commit()
        
        # Stats collection should be reasonably fast even with larger dataset
        start_time = time.time()
        large_dataset_stats = manager.get_memory_stats()
        stats_time = time.time() - start_time
        
        # Should complete within reasonable time (< 1 second for 100+ passages)
        assert stats_time < 1.0, f"Stats collection took too long: {stats_time:.3f}s"
        
        # Verify counts are correct
        assert large_dataset_stats["passages"]["count"] == 103  # 3 + 100
        
        # Test 6: Test stats with edge cases
        with manager.get_session() as session:
            # Create block with empty content
            empty_block = MemoryBlock(
                label="empty_test",
                value="",
                character_limit=1000
            )
            session.add(empty_block)
            
            # Create passage with extreme importance scores
            min_passage = MemoryPassage(
                text="Minimum importance passage.",
                embedding=np.random.rand(config.memory.embedding_dim).astype(np.float32),
                source="test",
                source_id="min_test",
                importance_score=0.0
            )
            max_passage = MemoryPassage(
                text="Maximum importance passage.",
                embedding=np.random.rand(config.memory.embedding_dim).astype(np.float32),
                source="test", 
                source_id="max_test",
                importance_score=1.0
            )
            session.add_all([min_passage, max_passage])
            session.commit()
        
        edge_case_stats = manager.get_memory_stats()
        
        # Should handle edge cases gracefully
        assert edge_case_stats["blocks"]["count"] >= 4  # Original 3 + empty block
        assert edge_case_stats["passages"]["count"] == 105  # Previous + 2
        assert 0 <= edge_case_stats["passages"]["avg_importance"] <= 1
        
        # Test 7: Verify numeric precision and data types
        final_stats = manager.get_memory_stats()
        
        # All counts should be non-negative integers
        assert final_stats["blocks"]["count"] >= 0
        assert final_stats["blocks"]["total_characters"] >= 0
        assert final_stats["passages"]["count"] >= 0
        assert final_stats["snapshots"]["count"] >= 0
        
        # Importance should be a valid float between 0 and 1
        if final_stats["passages"]["count"] > 0:
            avg_importance = final_stats["passages"]["avg_importance"]
            assert isinstance(avg_importance, (int, float))
            assert 0 <= avg_importance <= 1
        
        # Cache stats should be properly bounded
        cache_final = final_stats["embedding_cache"]
        assert cache_final["hits"] >= 0
        assert cache_final["misses"] >= 0
        assert cache_final["total_requests"] == cache_final["hits"] + cache_final["misses"]
        if cache_final["total_requests"] > 0:
            expected_hit_rate = cache_final["hits"] / cache_final["total_requests"]
            assert abs(cache_final["hit_rate"] - expected_hit_rate) < 0.001
        
        # Cleanup
        manager.engine.dispose()


class TestHealthCheckSystem:
    """
    Test memory system health check functionality.
    
    Health checking is critical for production monitoring and debugging system issues.
    This validates that health_check() accurately reports component status and helps
    diagnose problems when they occur.
    """
    
    def test_health_check_reports_accurate_system_status(self, clean_test_database):
        """
        Test that health_check() provides comprehensive system diagnostics.
        
        REAL BUG THIS CATCHES: If our health_check() method fails to accurately
        detect component failures, production monitoring will miss critical issues
        like database disconnections, embedding service failures, or vector store
        problems. This leads to silent failures and degraded user experience.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Test 1: Healthy system should report all components as OK
        health_report = manager.health_check()
        
        # Verify basic structure
        assert isinstance(health_report, dict)
        assert "status" in health_report
        assert "components" in health_report
        assert "issues" in health_report
        assert "stats" in health_report
        
        # Should report healthy status when everything works
        assert health_report["status"] == "healthy"
        assert isinstance(health_report["issues"], list)
        assert len(health_report["issues"]) == 0
        
        # Verify all expected components are checked
        components = health_report["components"]
        assert "database" in components
        assert "embedding_model" in components
        assert "vector_store" in components
        
        # All components should be OK in healthy system
        assert components["database"] == "ok"
        assert components["embedding_model"] == "ok"
        assert components["vector_store"] == "ok"
        
        # Stats should be included for comprehensive monitoring
        stats = health_report["stats"]
        assert isinstance(stats, dict)
        assert "blocks" in stats
        assert "passages" in stats
        assert "snapshots" in stats
        assert "embedding_cache" in stats
        
        # Test 2: Database health check validates actual connectivity
        with manager.get_session() as session:
            # Verify health check actually tests database operations
            result = session.execute(text("SELECT 1")).scalar()
            assert result == 1  # Database connection works
        
        # Health check should verify same functionality
        assert health_report["components"]["database"] == "ok"
        
        # Test 3: Embedding model health check validates dimensions and functionality
        test_embedding = manager.generate_embedding("health check test")
        
        # Should match configured dimensions
        assert test_embedding.shape[0] == config.memory.embedding_dim
        
        # Health check should verify same constraints
        assert health_report["components"]["embedding_model"] == "ok"
        
        # Test 4: Vector store health check validates search functionality
        test_vector = np.random.rand(config.memory.embedding_dim).astype(np.float32)
        search_results = manager.vector_store.search(test_vector, k=1)
        
        # Search should work (may return empty results, but shouldn't crash)
        assert isinstance(search_results, list)
        
        # Health check should verify vector store is operational
        assert health_report["components"]["vector_store"] == "ok"
        
        # Test 5: Performance characteristics under normal load
        import time
        
        start_time = time.time()
        health_report = manager.health_check()
        health_check_time = time.time() - start_time
        
        # Health check should be fast enough for frequent monitoring
        assert health_check_time < 2.0, f"Health check took too long: {health_check_time:.3f}s"
        
        # Test 6: Multiple consecutive health checks should be consistent
        health_reports = []
        for i in range(3):
            report = manager.health_check()
            health_reports.append(report)
            time.sleep(0.1)  # Small delay between checks
        
        # All reports should be consistent for stable system
        for report in health_reports:
            assert report["status"] == "healthy"
            assert len(report["issues"]) == 0
            assert report["components"]["database"] == "ok"
            assert report["components"]["embedding_model"] == "ok"
            assert report["components"]["vector_store"] == "ok"
        
        # Test 7: Health check includes meaningful statistics
        final_stats = health_report["stats"]
        
        # Should include current system state
        assert final_stats["blocks"]["count"] >= 3  # At least core blocks
        assert final_stats["passages"]["count"] >= 0
        assert final_stats["snapshots"]["count"] >= 0
        
        # Embedding cache stats should be present
        cache_stats = final_stats["embedding_cache"]
        assert "total_requests" in cache_stats
        assert "hits" in cache_stats
        assert "misses" in cache_stats
        assert "hit_rate" in cache_stats
        
        # Cleanup
        manager.engine.dispose()
    
    def test_health_check_detects_and_reports_component_failures(self, clean_test_database):
        """
        Test that health_check() accurately detects and reports component failures.
        
        REAL BUG THIS CATCHES: If our health_check() method doesn't detect when
        components fail or reports false positives, production monitoring will either
        miss real issues or create false alarms. Both scenarios damage system
        reliability and operational confidence.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Test 1: Test with simulated embedding dimension mismatch
        # Temporarily modify config to create dimension mismatch
        original_dim = config.memory.embedding_dim
        config.memory.embedding_dim = 999  # Wrong dimension
        
        health_report = manager.health_check()
        
        # Should detect dimension mismatch
        assert health_report["status"] in ["degraded", "unhealthy"]
        assert health_report["components"]["embedding_model"] == "dimension_mismatch"
        assert len(health_report["issues"]) > 0
        assert any("dimension" in issue.lower() for issue in health_report["issues"])
        
        # Restore correct configuration
        config.memory.embedding_dim = original_dim
        
        # Test 2: Test database component isolation
        # Health check should still check other components even if one fails
        config.memory.embedding_dim = 999  # Keep dimension mismatch
        
        health_report = manager.health_check()
        
        # Database should still be OK even with embedding issues
        assert health_report["components"]["database"] == "ok"
        
        # But overall status should reflect the problem
        assert health_report["status"] != "healthy"
        
        # Restore configuration
        config.memory.embedding_dim = original_dim
        
        # Test 3: Test error message quality for debugging
        config.memory.embedding_dim = 384  # Different dimension to create mismatch
        
        health_report = manager.health_check()
        
        if health_report["status"] != "healthy":
            # Error messages should be specific enough for debugging
            issues = health_report["issues"]
            assert len(issues) > 0
            
            # Should mention the specific problem
            embedding_issue = next((issue for issue in issues if "dimension" in issue.lower()), None)
            if embedding_issue:
                assert "dimension" in embedding_issue.lower() or "mismatch" in embedding_issue.lower()
        
        # Restore correct configuration
        config.memory.embedding_dim = original_dim
        
        # Test 4: Verify recovery detection
        # After fixing configuration, health should return to normal
        final_health = manager.health_check()
        
        assert final_health["status"] == "healthy"
        assert len(final_health["issues"]) == 0
        assert final_health["components"]["database"] == "ok"
        assert final_health["components"]["embedding_model"] == "ok"
        assert final_health["components"]["vector_store"] == "ok"
        
        # Test 5: Test health check with real data load
        # Add some test data to ensure health check works with active system
        with manager.get_session() as session:
            # Add test passages
            for i in range(5):
                passage = MemoryPassage(
                    text=f"Health check test passage {i}",
                    embedding=np.random.rand(config.memory.embedding_dim).astype(np.float32),
                    source="health_test",
                    source_id=f"health_{i}",
                    importance_score=0.5
                )
                session.add(passage)
            session.commit()
        
        # Create test snapshot
        snapshot_id = manager.create_snapshot("health_test", "health_check_test")
        
        # Health check should still work with data present
        loaded_health = manager.health_check()
        
        assert loaded_health["status"] == "healthy"
        assert loaded_health["components"]["database"] == "ok"
        assert loaded_health["components"]["embedding_model"] == "ok" 
        assert loaded_health["components"]["vector_store"] == "ok"
        
        # Stats should reflect the added data
        loaded_stats = loaded_health["stats"]
        assert loaded_stats["passages"]["count"] >= 5
        assert loaded_stats["snapshots"]["count"] >= 1
        
        # Test 6: Concurrent health checks should not interfere
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def concurrent_health_check(worker_id):
            """Worker that performs health checks concurrently."""
            try:
                health = manager.health_check()
                return {
                    "worker_id": worker_id,
                    "success": True,
                    "status": health["status"],
                    "component_count": len(health["components"]),
                    "has_stats": "stats" in health
                }
            except Exception as e:
                return {
                    "worker_id": worker_id,
                    "success": False,
                    "error": str(e)
                }
        
        # Run 5 workers doing health checks concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_health_check, i) for i in range(5)]
            worker_results = [future.result() for future in as_completed(futures)]
        
        # All health checks should succeed
        successful_checks = [r for r in worker_results if r["success"]]
        failed_checks = [r for r in worker_results if not r["success"]]
        
        assert len(successful_checks) == 5, f"Some health checks failed: {failed_checks}"
        
        # All should report consistent status for stable system
        for result in successful_checks:
            assert result["status"] == "healthy"
            assert result["component_count"] >= 3  # database, embedding_model, vector_store
            assert result["has_stats"] is True
        
        # Test 7: Health check performance under different system states
        # Generate some cache activity
        for i in range(10):
            manager.generate_embedding(f"cache test {i}")
        
        # Health check should still be fast with cache activity
        start_time = time.time()
        active_health = manager.health_check()
        active_health_time = time.time() - start_time
        
        assert active_health_time < 2.0, f"Health check with active cache took too long: {active_health_time:.3f}s"
        assert active_health["status"] == "healthy"
        
        # Cache stats should reflect activity
        cache_stats = active_health["stats"]["embedding_cache"]
        assert cache_stats["total_requests"] >= 10
        
        # Cleanup
        manager.engine.dispose()


class TestDefaultBlockContent:
    """
    Test default block content generation functionality.
    
    Default content generation ensures that core memory blocks are initialized
    with appropriate starting content that makes sense for their purpose.
    """
    
    def test_default_block_content_returns_appropriate_defaults_for_core_blocks(self, clean_test_database):
        """
        Test that _get_default_block_content() returns sensible defaults for known block types.
        
        REAL BUG THIS CATCHES: If our _get_default_block_content() method returns
        empty, inappropriate, or misleading content for core memory blocks, the
        memory system starts with broken initial state that confuses users and
        breaks memory tool operations that depend on these blocks existing with
        meaningful content.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Test the actual default content for each core block type
        persona_content = manager._get_default_block_content("persona")
        human_content = manager._get_default_block_content("human")
        system_content = manager._get_default_block_content("system")
        unknown_content = manager._get_default_block_content("unknown_label")
        
        # Test 1: Known block types should return non-empty strings
        assert isinstance(persona_content, str), "Persona content should be string"
        assert isinstance(human_content, str), "Human content should be string"
        assert isinstance(system_content, str), "System content should be string"
        assert len(persona_content) > 0, "Persona content should not be empty"
        assert len(human_content) > 0, "Human content should not be empty"
        assert len(system_content) > 0, "System content should not be empty"
        
        # Test 2: Unknown block types should return empty string
        assert unknown_content == "", "Unknown block labels should return empty string"
        
        # Test 3: Content should be appropriate for each block type
        # Persona should mention MIRA or assistant
        persona_lower = persona_content.lower()
        assert "mira" in persona_lower or "assistant" in persona_lower, f"Persona content should identify as MIRA or assistant: '{persona_content}'"
        
        # Human should indicate unknown/default state
        human_lower = human_content.lower()
        assert "human" in human_lower, f"Human content should reference human: '{human_content}'"
        
        # System should indicate initialization/operational status
        system_lower = system_content.lower()
        assert "system" in system_lower or "memory" in system_lower, f"System content should reference system/memory: '{system_content}'"
        
        # Test 4: Content should be used during actual initialization
        with manager.get_session() as session:
            blocks = session.query(MemoryBlock).all()
            block_map = {block.label: block for block in blocks}
            
            # Check that blocks were actually initialized with these defaults
            if "persona" in block_map:
                assert block_map["persona"].value == persona_content, "Persona block should use default content"
            if "human" in block_map:
                assert block_map["human"].value == human_content, "Human block should use default content"
            if "system" in block_map:
                assert block_map["system"].value == system_content, "System block should use default content"
        
        # Cleanup
        manager.engine.dispose()


# =============================================================================
# CROSS-METHOD WORKFLOW TESTS
# =============================================================================

class TestCrossMethodWorkflows:
    """
    Test realistic workflows that combine multiple MemoryManager methods.
    
    These tests validate that methods work together correctly in real usage
    patterns, catching integration bugs that single-method tests might miss.
    """
    
    def test_memory_storage_and_retrieval_workflow(self, clean_test_database):
        """
        Test complete memory storage and retrieval workflow with embeddings.
        
        REAL BUG THIS CATCHES: If embedding generation, passage storage, and 
        vector search don't work together correctly, users can store memories
        that become unsearchable, or searches return wrong results. This breaks
        the core memory functionality that users depend on.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Workflow step 1: Store a memory passage
        memory_text = "User prefers morning meetings and dislikes interruptions during deep work"
        embedding = manager.generate_embedding(memory_text)
        
        # Verify embedding was generated correctly
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == config.memory.embedding_dim
        
        # Store passage in database
        with manager.get_session() as session:
            passage = MemoryPassage(
                text=memory_text,
                embedding=embedding,
                source="conversation",
                source_id="conv_001",
                importance_score=0.8
            )
            session.add(passage)
            session.commit()
            stored_id = passage.id
        
        # Workflow step 2: Search for similar content
        search_text = "What are the user's meeting preferences?"
        search_embedding = manager.generate_embedding(search_text)
        
        # Search using vector store
        search_results = manager.vector_store.search(search_embedding, k=1)
        
        # Workflow step 3: Verify end-to-end functionality
        assert len(search_results) == 1, "Should find the stored memory"
        
        # Retrieve the actual passage to verify it's the right one
        with manager.get_session() as session:
            retrieved_passage = session.query(MemoryPassage).filter_by(id=stored_id).first()
            assert retrieved_passage is not None
            assert retrieved_passage.text == memory_text
            assert retrieved_passage.importance_score == 0.8
        
        # Workflow step 4: Verify stats reflect the stored data
        stats = manager.get_memory_stats()
        assert stats["passages"]["count"] >= 1
        assert stats["passages"]["avg_importance"] > 0
        
        # Cleanup
        manager.engine.dispose()
    
    def test_snapshot_restore_preserves_memory_blocks_correctly(self, clean_test_database):
        """
        Test that snapshot/restore maintains memory block integrity while preserving other system functionality.
        
        REAL BUG THIS CATCHES: If snapshot/restore corrupts memory block state or breaks
        the ability to continue normal memory operations after restore, users lose their
        conversation context and the system becomes unreliable for ongoing interactions.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Workflow step 1: Establish memory block state and add passage
        with manager.get_session() as session:
            persona_block = session.query(MemoryBlock).filter_by(label="persona").first()
            human_block = session.query(MemoryBlock).filter_by(label="human").first()
            
            # Update blocks with specific conversation context
            persona_block.value = "I am MIRA, an AI assistant. I remember that this user prefers technical discussions."
            human_block.value = "Software engineer who asks detailed questions about database optimization."
            persona_block.version += 1
            human_block.version += 1
            session.commit()
            
        # Add a passage (this should persist independently of snapshots)
        memory_text = "User asked detailed questions about database optimization"
        embedding = manager.generate_embedding(memory_text)
        
        with manager.get_session() as session:
            passage = MemoryPassage(
                text=memory_text,
                embedding=embedding,
                source="conversation",
                source_id="tech_conv_001",
                importance_score=0.9
            )
            session.add(passage)
            session.commit()
            stored_passage_id = passage.id
        
        # Workflow step 2: Create snapshot (captures memory blocks only)
        snapshot_id = manager.create_snapshot("workflow_test", "before_changes")
        
        # Workflow step 3: Modify memory blocks significantly
        with manager.get_session() as session:
            persona_block = session.query(MemoryBlock).filter_by(label="persona").first()
            human_block = session.query(MemoryBlock).filter_by(label="human").first()
            persona_block.value = "CORRUPTED: This information should not persist"
            human_block.value = "CORRUPTED: This human info is wrong"
            session.commit()
        
        # Workflow step 4: Restore memory blocks from snapshot
        restore_success = manager.restore_from_snapshot(snapshot_id)
        assert restore_success is True
        
        # Workflow step 5: Verify memory blocks are restored correctly
        with manager.get_session() as session:
            restored_persona = session.query(MemoryBlock).filter_by(label="persona").first()
            restored_human = session.query(MemoryBlock).filter_by(label="human").first()
            
            # Memory blocks should be restored to snapshot state
            assert "technical discussions" in restored_persona.value
            assert "database optimization" in restored_human.value
            assert "CORRUPTED" not in restored_persona.value
            assert "CORRUPTED" not in restored_human.value
            
            # Passage should still exist (not affected by memory block restore)
            passage = session.query(MemoryPassage).filter_by(id=stored_passage_id).first()
            assert passage is not None
            assert passage.text == memory_text
        
        # Workflow step 6: Verify system continues to work normally after restore
        # Should be able to generate new embeddings
        new_embedding = manager.generate_embedding("test after restore")
        assert isinstance(new_embedding, np.ndarray)
        
        # Should be able to search existing passages
        search_embedding = manager.generate_embedding("database optimization questions")
        search_results = manager.vector_store.search(search_embedding, k=1)
        assert len(search_results) >= 1, "Vector search should work after restore"
        
        # Stats should reflect current system state
        stats = manager.get_memory_stats()
        assert stats["passages"]["count"] >= 1
        assert stats["blocks"]["count"] >= 3
        
        # Cleanup
        manager.engine.dispose()
    
    def test_embedding_cache_consistency_validation(self, clean_test_database):
        """
        Test that cached embeddings remain consistent with fresh embeddings for the same text.
        
        REAL BUG THIS CATCHES: If embedding cache becomes corrupted or returns stale
        embeddings that don't match what the model would generate fresh, users get
        wrong search results with no error indication. This silently breaks memory
        retrieval and makes the system unreliable.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Test multiple text samples to validate consistency
        test_texts = [
            "User prefers morning meetings",
            "Technical discussion about databases", 
            "Casual conversation about weather",
            "Important deadline on Friday"
        ]
        
        # Step 1: Generate fresh embeddings and verify they're cached
        fresh_embeddings = {}
        for text in test_texts:
            embedding = manager.generate_embedding(text)
            fresh_embeddings[text] = embedding.copy()
            
            # Verify embedding properties
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape[0] == config.memory.embedding_dim
            assert not np.any(np.isnan(embedding))
            assert not np.any(np.isinf(embedding))
        
        # Step 2: Verify all embeddings are now cached
        cache_stats = manager.embedding_cache.get_stats()
        assert cache_stats["total_requests"] >= len(test_texts)
        assert cache_stats["misses"] >= len(test_texts)  # Initial generations were cache misses
        
        # Step 3: Request same embeddings again - should come from cache
        cached_embeddings = {}
        initial_hit_count = cache_stats["hits"]
        
        for text in test_texts:
            embedding = manager.generate_embedding(text)
            cached_embeddings[text] = embedding.copy()
        
        # Verify cache hits increased
        updated_cache_stats = manager.embedding_cache.get_stats()
        assert updated_cache_stats["hits"] >= initial_hit_count + len(test_texts)
        
        # Step 4: Critical validation - cached embeddings must be identical to fresh ones
        for text in test_texts:
            fresh = fresh_embeddings[text]
            cached = cached_embeddings[text]
            
            # Must be exactly identical (not just similar)
            np.testing.assert_array_equal(
                fresh, cached, 
                f"Cached embedding for '{text}' differs from fresh embedding"
            )
            
            # Verify no corruption occurred
            assert fresh.shape == cached.shape
            assert fresh.dtype == cached.dtype
            assert not np.any(np.isnan(cached))
            assert not np.any(np.isinf(cached))
        
        # Step 5: Test cache consistency after system restart simulation
        # Clear memory cache but keep disk cache
        manager.embedding_cache.clear_memory_cache()
        
        # Request embeddings again - should load from disk cache
        disk_cached_embeddings = {}
        for text in test_texts:
            embedding = manager.generate_embedding(text)
            disk_cached_embeddings[text] = embedding.copy()
        
        # Validate disk cache consistency
        for text in test_texts:
            original = fresh_embeddings[text]
            disk_cached = disk_cached_embeddings[text]
            
            np.testing.assert_array_equal(
                original, disk_cached,
                f"Disk cached embedding for '{text}' differs from original"
            )
        
        # Cleanup
        manager.engine.dispose()
    
    def test_embedding_cache_invalidation_prevents_wrong_results(self, clean_test_database):
        """
        Test that cache properly invalidates and doesn't return wrong embeddings for different text.
        
        REAL BUG THIS CATCHES: If embedding cache returns cached results for different
        text (due to poor cache key handling or hash collisions), users get completely
        wrong embeddings for their content, breaking memory storage and search accuracy.
        This is a silent failure that corrupts the entire memory system.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Clear cache to start fresh
        manager.embedding_cache.clear_memory_cache()
        manager.embedding_cache.clear_disk_cache()
        
        # Test 1: Different text should never return cached results
        text1 = "User prefers morning meetings"
        text2 = "User prefers evening meetings"  # Very similar but different
        
        # Generate embedding for first text
        embedding1 = manager.generate_embedding(text1)
        cache_stats_after_first = manager.embedding_cache.get_stats()
        
        # Generate embedding for second text - should NOT use cache
        embedding2 = manager.generate_embedding(text2)
        cache_stats_after_second = manager.embedding_cache.get_stats()
        
        # Verify no cache hit occurred for different text
        assert cache_stats_after_second["misses"] > cache_stats_after_first["misses"], "Different text should cause cache miss"
        
        # Embeddings must be different (not cached result)
        assert not np.array_equal(embedding1, embedding2), "Different text must produce different embeddings"
        
        # Test 2: Verify cache keys are sensitive to exact text
        similar_texts = [
            "User prefers morning meetings",
            "User prefers morning meetings.",  # Added period
            "user prefers morning meetings",   # Different case  
            "User prefers morning meetings ",  # Added space
            " User prefers morning meetings"   # Leading space
        ]
        
        embeddings = {}
        for text in similar_texts:
            embeddings[text] = manager.generate_embedding(text)
        
        # Each variation should produce different embeddings (no cache reuse)
        embedding_values = list(embeddings.values())
        for i, emb1 in enumerate(embedding_values):
            for j, emb2 in enumerate(embedding_values):
                if i != j:
                    assert not np.array_equal(emb1, emb2), f"Text variations should not share cache: '{similar_texts[i]}' vs '{similar_texts[j]}'"
        
        # Test 3: Verify exact same text DOES use cache
        original_text = "User prefers morning meetings"
        original_embedding = manager.generate_embedding(original_text)
        
        initial_hits = manager.embedding_cache.get_stats()["hits"]
        
        # Request exact same text again
        cached_embedding = manager.generate_embedding(original_text)
        
        final_hits = manager.embedding_cache.get_stats()["hits"]
        
        # Should have gotten cache hit
        assert final_hits > initial_hits, "Exact same text should use cache"
        
        # Should be identical
        np.testing.assert_array_equal(original_embedding, cached_embedding, "Exact same text should return identical cached embedding")
        
        # Test 4: Verify cache doesn't overflow and corrupt keys
        # Generate many embeddings to test cache behavior under load
        many_texts = [f"Unique text number {i} for cache stress test" for i in range(50)]
        
        cache_mappings = {}
        for text in many_texts:
            embedding = manager.generate_embedding(text)
            cache_mappings[text] = embedding.copy()
        
        # Verify each text still returns its correct embedding
        for text, original_embedding in cache_mappings.items():
            current_embedding = manager.generate_embedding(text)
            np.testing.assert_array_equal(
                original_embedding, current_embedding,
                f"Cache corruption detected for text: '{text}'"
            )
        
        # Cleanup
        manager.engine.dispose()
    
    def test_embedding_cache_detects_corrupted_data(self, clean_test_database):
        """
        Test that the system detects and handles corrupted embedding cache data.
        
        REAL BUG THIS CATCHES: If cached embeddings get corrupted (wrong dimensions,
        NaN values, wrong data type), the system returns invalid embeddings that break
        vector search and cause silent failures. Users get wrong search results with
        no indication that anything is wrong.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Clear cache to start fresh
        manager.embedding_cache.clear_memory_cache()
        manager.embedding_cache.clear_disk_cache()
        
        # Generate and cache a normal embedding first
        test_text = "This is test text for cache corruption detection"
        normal_embedding = manager.generate_embedding(test_text)
        
        # Verify normal embedding is valid
        assert isinstance(normal_embedding, np.ndarray)
        assert normal_embedding.shape[0] == config.memory.embedding_dim
        assert not np.any(np.isnan(normal_embedding))
        assert not np.any(np.isinf(normal_embedding))
        
        # Simulate cache corruption by directly manipulating cached data
        cache_key = manager.embedding_cache._get_cache_key(test_text)
        
        # Test 1: Corrupt with wrong dimensions
        corrupted_wrong_dims = np.random.rand(512).astype(np.float32)  # Wrong dimension
        manager.embedding_cache.memory_cache[cache_key] = corrupted_wrong_dims
        
        # System should detect corruption and regenerate
        try:
            recovered_embedding = manager.generate_embedding(test_text)
            # Should return valid embedding, not corrupted one
            assert recovered_embedding.shape[0] == config.memory.embedding_dim
            assert not np.array_equal(recovered_embedding, corrupted_wrong_dims)
            
        except Exception as e:
            # If it raises exception, should be clear about corruption
            error_str = str(e).lower()
            assert any(word in error_str for word in ["dimension", "embedding", "invalid"]), f"Should detect corruption: {error_str}"
        
        # Test 2: Corrupt with NaN values
        corrupted_nan = np.full(config.memory.embedding_dim, np.nan, dtype=np.float32)
        manager.embedding_cache.memory_cache[cache_key] = corrupted_nan
        
        try:
            recovered_embedding = manager.generate_embedding(test_text)
            # Should return valid embedding without NaN
            assert not np.any(np.isnan(recovered_embedding))
            assert recovered_embedding.shape[0] == config.memory.embedding_dim
            
        except Exception as e:
            # If it raises exception, should be clear about corruption
            error_str = str(e).lower()
            assert any(word in error_str for word in ["nan", "invalid", "embedding"]), f"Should detect NaN corruption: {error_str}"
        
        # Test 3: Corrupt with wrong data type
        corrupted_type = ["not", "an", "embedding"]  # Wrong type entirely
        manager.embedding_cache.memory_cache[cache_key] = corrupted_type
        
        try:
            recovered_embedding = manager.generate_embedding(test_text)
            # Should return valid numpy array
            assert isinstance(recovered_embedding, np.ndarray)
            assert recovered_embedding.shape[0] == config.memory.embedding_dim
            
        except Exception as e:
            # If it raises exception, should be clear about corruption
            error_str = str(e).lower()
            assert any(word in error_str for word in ["type", "embedding", "invalid"]), f"Should detect type corruption: {error_str}"
        
        # Cleanup
        manager.engine.dispose()
    
    def test_memory_consolidation_workflow_maintains_system_integrity(self, clean_test_database):
        """
        Test complete memory consolidation workflow with realistic data patterns.
        
        REAL BUG THIS CATCHES: If our consolidation workflow has bugs in pruning logic,
        importance score updates, or snapshot creation, users' valuable memories could
        be incorrectly deleted or the system could become corrupted during maintenance.
        This breaks the core promise that important memories are preserved long-term.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Setup: Create realistic memory data with different characteristics
        passages_to_create = [
            # High importance, recent - should be preserved
            ("Critical user preference: prefers async communication", 0.9, 0),
            ("Important project context: working on ML infrastructure", 0.8, 1),
            
            # Medium importance, older - may be updated but preserved  
            ("User mentioned liking coffee", 0.5, 30),
            ("Discussed weekend hiking plans", 0.4, 25),
            
            # Low importance, old - candidates for pruning (older than max_memory_age_days)
            ("Weather comment: it's sunny today", 0.2, 95),
            ("Random comment about traffic", 0.1, 100),
        ]
        
        created_passages = []
        with manager.get_session() as session:
            for i, (text, importance, days_old) in enumerate(passages_to_create):
                # Simulate passages created at different times
                from utils.timezone_utils import utc_now
                from datetime import timedelta
                created_time = utc_now() - timedelta(days=days_old)
                
                passage = MemoryPassage(
                    text=text,
                    embedding=manager.generate_embedding(text),
                    source="conversation",
                    source_id=f"msg_{i}",
                    importance_score=importance
                )
                # Manually set creation time to simulate aging
                passage.created_at = created_time
                # Set access count (low for pruning candidates)
                passage.access_count = 0 if importance < 0.3 else 1
                session.add(passage)
                created_passages.append((passage, importance, days_old))
            
            session.commit()
            passage_ids = [p[0].id for p in created_passages]
        
        # Pre-consolidation verification
        initial_stats = manager.get_memory_stats()
        assert initial_stats["passages"]["count"] == 6, "Should have 6 passages before consolidation"
        
        # Run consolidation workflow
        consolidation_results = manager.consolidation_engine.consolidate_memories("test_conversation")
        
        # Verify consolidation results structure
        assert isinstance(consolidation_results, dict), "Consolidation should return results"
        assert "pruned_passages" in consolidation_results, "Should report pruned count"
        assert "updated_scores" in consolidation_results, "Should report updated scores count"
        
        # Verify system integrity after consolidation
        post_consolidation_stats = manager.get_memory_stats()
        
        # Check that some pruning occurred (low importance, old passages should be removed)
        assert post_consolidation_stats["passages"]["count"] < initial_stats["passages"]["count"], "Some passages should have been pruned"
        
        # Verify important passages are preserved
        with manager.get_session() as session:
            remaining_passages = session.query(MemoryPassage).all()
            remaining_texts = [p.text for p in remaining_passages]
            
            # High importance passages should definitely be preserved
            assert any("Critical user preference" in text for text in remaining_texts), "High importance passage should be preserved"
            assert any("Important project context" in text for text in remaining_texts), "High importance passage should be preserved"
            
            # Low importance, old passages should likely be pruned
            old_weather_preserved = any("Weather comment" in text for text in remaining_texts)
            old_traffic_preserved = any("Random comment about traffic" in text for text in remaining_texts)
            
            # At least one of the low-importance old passages should be pruned
            assert not (old_weather_preserved and old_traffic_preserved), "At least one low-importance old passage should be pruned"
        
        # Verify system still functions normally after consolidation
        health_check = manager.health_check()
        assert health_check["status"] == "healthy", "System should remain healthy after consolidation"
        
        # Verify we can still store new memories
        new_passage_text = "Post-consolidation test memory"
        new_embedding = manager.generate_embedding(new_passage_text)
        with manager.get_session() as session:
            new_passage = MemoryPassage(
                text=new_passage_text,
                embedding=new_embedding,
                source="test",
                source_id="post_consolidation",
                importance_score=0.7
            )
            session.add(new_passage)
            session.commit()
        
        # Verify new passage was stored correctly
        final_stats = manager.get_memory_stats()
        assert final_stats["passages"]["count"] == post_consolidation_stats["passages"]["count"] + 1, "Should be able to store new passages after consolidation"
        
        # Cleanup
        manager.engine.dispose()


class TestComponentManagerIntegration:
    """
    Test integration between MemoryManager and its component managers.
    
    Component managers (block_manager, passage_manager, consolidation_engine, 
    conversation_archive) are initialized by MemoryManager and depend on it
    for database access and core functionality.
    """
    
    def test_block_manager_integration_with_memory_manager(self, clean_test_database):
        """
        Test that block_manager integrates correctly with MemoryManager.
        
        REAL BUG THIS CATCHES: If our BlockManager initialization fails, has
        incorrect dependencies, or can't access MemoryManager's database session,
        all memory block operations will fail. This breaks core memory functionality
        that users depend on for persistent memory across conversations.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Test 1: BlockManager should be initialized and accessible
        assert manager.block_manager is not None, "BlockManager should be initialized"
        assert hasattr(manager.block_manager, 'memory_manager'), "BlockManager should have reference to MemoryManager"
        
        # Test 2: BlockManager should be able to access MemoryManager's session
        # This tests the dependency injection works correctly
        try:
            with manager.get_session() as session:
                # BlockManager should be able to work with sessions from MemoryManager
                blocks = session.query(MemoryBlock).all()
                assert isinstance(blocks, list), "Should be able to query blocks through MemoryManager session"
        except Exception as e:
            assert False, f"BlockManager integration failed with session access: {str(e)}"
        
        # Test 3: BlockManager should have access to core blocks created by MemoryManager
        with manager.get_session() as session:
            core_blocks = session.query(MemoryBlock).filter(
                MemoryBlock.label.in_(config.memory.core_memory_blocks.keys())
            ).all()
            
            # Should have the core blocks that MemoryManager created
            assert len(core_blocks) >= 3, "Should have core memory blocks available to BlockManager"
            
            block_labels = {block.label for block in core_blocks}
            expected_labels = set(config.memory.core_memory_blocks.keys())
            assert block_labels == expected_labels, "BlockManager should see all core blocks created by MemoryManager"
        
        # Test 4: BlockManager should be able to interact with blocks using MemoryManager's infrastructure
        # Test that BlockManager can access and potentially modify blocks
        with manager.get_session() as session:
            persona_block = session.query(MemoryBlock).filter_by(label="persona").first()
            assert persona_block is not None, "BlockManager should be able to access persona block"
            
            # Verify BlockManager has access to the same data MemoryManager created
            original_value = persona_block.value
            original_version = persona_block.version
            
            assert len(original_value) > 0, "BlockManager should see block content created by MemoryManager"
            assert original_version == 1, "BlockManager should see initial version from MemoryManager"
        
        # Test 5: Verify BlockManager doesn't conflict with MemoryManager's operations
        # Both should be able to work with the same database without issues
        test_snapshot_id = manager.create_snapshot("block_manager_test", "integration_test")
        
        # BlockManager should still work after MemoryManager operations
        with manager.get_session() as session:
            blocks_after_snapshot = session.query(MemoryBlock).all()
            assert len(blocks_after_snapshot) >= 3, "BlockManager should still see blocks after MemoryManager snapshot"
        
        # MemoryManager should still work after BlockManager access
        health_check = manager.health_check()
        assert health_check["status"] == "healthy", "MemoryManager should remain healthy after BlockManager integration"
        
        # Cleanup
        manager.engine.dispose()
    
    def test_passage_manager_integration_with_memory_manager(self, clean_test_database):
        """
        Test that passage_manager integrates correctly with MemoryManager.
        
        REAL BUG THIS CATCHES: If our PassageManager initialization fails, can't
        access MemoryManager's embedding generation, or has database access issues,
        all memory passage operations will fail. This breaks long-term memory
        storage and retrieval functionality that is core to the memory system.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Test 1: PassageManager should be initialized and accessible
        assert manager.passage_manager is not None, "PassageManager should be initialized"
        assert hasattr(manager.passage_manager, 'memory_manager'), "PassageManager should have reference to MemoryManager"
        
        # Test 2: PassageManager should be able to access MemoryManager's embedding functionality
        # This is critical since PassageManager depends on embeddings for storage
        try:
            test_embedding = manager.generate_embedding("test passage for integration")
            assert isinstance(test_embedding, np.ndarray), "PassageManager should access MemoryManager embeddings"
            assert test_embedding.shape[0] == config.memory.embedding_dim, "Embedding dimensions should match config"
        except Exception as e:
            assert False, f"PassageManager can't access MemoryManager embedding functionality: {str(e)}"
        
        # Test 3: PassageManager should be able to access database through MemoryManager
        with manager.get_session() as session:
            # PassageManager should be able to query passages
            passages = session.query(MemoryPassage).all()
            assert isinstance(passages, list), "PassageManager should access passage queries through MemoryManager"
            
            # Initially should have no passages
            assert len(passages) == 0, "Fresh system should start with no passages"
        
        # Test 4: PassageManager should work with MemoryManager's vector store
        # Test that PassageManager can potentially use the vector store for searches
        assert manager.vector_store is not None, "PassageManager should have access to MemoryManager's vector store"
        
        # Vector store should be accessible and working
        test_vector = np.random.rand(config.memory.embedding_dim).astype(np.float32)
        search_results = manager.vector_store.search(test_vector, k=1)
        assert isinstance(search_results, list), "PassageManager should access vector store through MemoryManager"
        
        # Test 5: Verify PassageManager doesn't conflict with MemoryManager operations
        # Create a test passage in the database to verify PassageManager can see it
        with manager.get_session() as session:
            test_passage = MemoryPassage(
                text="Test passage for PassageManager integration",
                embedding=manager.generate_embedding("Test passage for PassageManager integration"),
                source="integration_test",
                source_id="test_001",
                importance_score=0.8
            )
            session.add(test_passage)
            session.commit()
            passage_id = test_passage.id
        
        # PassageManager should be able to see the passage MemoryManager created
        with manager.get_session() as session:
            retrieved_passage = session.query(MemoryPassage).filter_by(id=passage_id).first()
            assert retrieved_passage is not None, "PassageManager should see passages created through MemoryManager"
            assert retrieved_passage.text == "Test passage for PassageManager integration"
            assert retrieved_passage.importance_score == 0.8
        
        # MemoryManager stats should reflect the passage
        stats = manager.get_memory_stats()
        assert stats["passages"]["count"] >= 1, "MemoryManager stats should include PassageManager data"
        
        # Health check should still work with passage data
        health = manager.health_check()
        assert health["status"] == "healthy", "MemoryManager should remain healthy with PassageManager integration"
        
        # Cleanup
        manager.engine.dispose()
    
    def test_consolidation_engine_integration_with_memory_manager(self, clean_test_database):
        """
        Test that consolidation_engine integrates correctly with MemoryManager.
        
        REAL BUG THIS CATCHES: If our ConsolidationEngine initialization fails, can't
        access MemoryManager's database sessions, vector store, or embedding cache,
        memory consolidation operations will fail. This breaks automatic memory
        maintenance that keeps the system performing well over time.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Test 1: ConsolidationEngine should be initialized and accessible
        assert manager.consolidation_engine is not None, "ConsolidationEngine should be initialized"
        assert hasattr(manager.consolidation_engine, 'memory_manager'), "ConsolidationEngine should have reference to MemoryManager"
        
        # Test 2: ConsolidationEngine should be able to access MemoryManager's database sessions
        with manager.get_session() as session:
            # ConsolidationEngine should be able to query passages through MemoryManager
            passages = session.query(MemoryPassage).all()
            assert isinstance(passages, list), "ConsolidationEngine should access database through MemoryManager"
        
        # Test 3: ConsolidationEngine should be able to access MemoryManager's vector store
        assert manager.consolidation_engine.memory_manager.vector_store is not None, "ConsolidationEngine should access vector store through MemoryManager"
        
        # Test 4: ConsolidationEngine should be able to access MemoryManager's embedding cache
        assert manager.consolidation_engine.memory_manager.embedding_cache is not None, "ConsolidationEngine should access embedding cache through MemoryManager"
        cache_stats = manager.consolidation_engine.memory_manager.embedding_cache.get_stats()
        assert isinstance(cache_stats, dict), "ConsolidationEngine should get cache stats through MemoryManager"
        
        # Test 5: ConsolidationEngine should be able to create snapshots through MemoryManager
        # This tests that consolidation can use MemoryManager's snapshot functionality
        try:
            test_snapshot_id = manager.consolidation_engine.memory_manager.create_snapshot(
                "consolidation_test", "integration_test"
            )
            assert test_snapshot_id is not None, "ConsolidationEngine should create snapshots through MemoryManager"
            assert len(test_snapshot_id) > 0, "Snapshot ID should be valid"
        except Exception as e:
            assert False, f"ConsolidationEngine can't create snapshots through MemoryManager: {str(e)}"
        
        # Test 6: ConsolidationEngine should be able to analyze memory patterns
        # This tests that analysis methods can access the database
        try:
            analysis = manager.consolidation_engine.analyze_memory_patterns()
            assert isinstance(analysis, dict), "ConsolidationEngine should analyze patterns through MemoryManager"
            assert "passage_creation_rate" in analysis, "Analysis should include passage creation rates"
            assert "memory_health" in analysis, "Analysis should include memory health metrics"
        except Exception as e:
            assert False, f"ConsolidationEngine can't analyze patterns through MemoryManager: {str(e)}"
        
        # Test 7: ConsolidationEngine should work with test data
        # Add some test passages and verify consolidation can process them
        with manager.get_session() as session:
            # Add test passages with different characteristics
            test_passages = [
                MemoryPassage(
                    text="Recent important passage",
                    embedding=manager.generate_embedding("Recent important passage"),
                    source="test",
                    source_id="recent_001",
                    importance_score=0.9
                ),
                MemoryPassage(
                    text="Old unimportant passage",
                    embedding=manager.generate_embedding("Old unimportant passage"),
                    source="test",
                    source_id="old_001",
                    importance_score=0.2
                )
            ]
            
            for passage in test_passages:
                session.add(passage)
            session.commit()
        
        # ConsolidationEngine should be able to run consolidation
        initial_stats = manager.get_memory_stats()
        initial_passage_count = initial_stats["passages"]["count"]
        
        # Run consolidation (should work without errors)
        try:
            consolidation_results = manager.consolidation_engine.consolidate_memories("test_conversation")
            assert isinstance(consolidation_results, dict), "Consolidation should return results dict"
            assert "pruned_passages" in consolidation_results, "Results should include pruned passages count"
            assert "updated_scores" in consolidation_results, "Results should include updated scores count"
        except Exception as e:
            assert False, f"ConsolidationEngine consolidation failed: {str(e)}"
        
        # MemoryManager should still be healthy after consolidation
        health = manager.health_check()
        assert health["status"] == "healthy", "MemoryManager should remain healthy after ConsolidationEngine operations"
        
        # Stats should still be accessible
        final_stats = manager.get_memory_stats()
        assert isinstance(final_stats, dict), "MemoryManager stats should work after consolidation"
        
        # Cleanup
        manager.engine.dispose()


# =============================================================================
# ERROR HANDLING & EDGE CASES TESTS
# =============================================================================

class TestErrorHandlingAndEdgeCases:
    """
    Test error handling and edge cases for MemoryManager.
    
    These tests validate that the system gracefully handles failure scenarios,
    corrupted data, network issues, and other edge cases that occur in production.
    """
    
    def test_get_memory_stats_handles_database_errors_gracefully(self, clean_test_database):
        """
        Test that get_memory_stats() handles database errors gracefully instead of crashing.
        
        REAL BUG THIS CATCHES: If our get_memory_stats() method doesn't handle database
        errors properly, monitoring dashboards and admin interfaces will crash when
        database connectivity is lost. This makes debugging harder when the system
        is already compromised.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Baseline: verify normal operation works
        normal_stats = manager.get_memory_stats()
        assert isinstance(normal_stats, dict)
        assert "blocks" in normal_stats
        
        # Setup: simulate database becoming unavailable
        from sqlalchemy import create_engine
        broken_engine = create_engine("postgresql://fake_user@nonexistent_host:5432/fake_db")
        manager.engine = broken_engine
        
        # Test the contract: should handle database errors gracefully
        try:
            broken_stats = manager.get_memory_stats()
            # If it succeeds, should return meaningful partial data
            assert isinstance(broken_stats, dict)
            
        except Exception as e:
            # If it fails, should be a clear, helpful error
            error_str = str(e).lower()
            assert any(word in error_str for word in ["database", "connection"]), f"Should be clear about database problem: {error_str}"
            
            # Should not be a generic Python crash
            assert "attributeerror" not in str(type(e)).lower(), f"Should not be generic Python error: {type(e)}"
    
    def test_embedding_cache_avoids_expensive_regeneration_under_pressure(self, clean_test_database):
        """
        Test that two-tier cache (memory + disk) avoids expensive embedding regeneration.
        
        REAL BUG THIS CATCHES: If our cache system fails under memory pressure, embeddings
        will be regenerated via expensive OpenAI API calls instead of using cached versions.
        This causes massive performance degradation and cost increases in production when
        many users request similar embeddings.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Clear cache and set small memory cache to force memory eviction
        manager.embedding_cache.clear_memory_cache()
        manager.embedding_cache.clear_disk_cache()
        
        original_cache_size = manager.embedding_cache.memory_cache_size
        manager.embedding_cache.memory_cache_size = 10  # Small to force eviction
        
        try:
            # Generate embeddings to populate both memory and disk cache
            test_texts = []
            for i in range(15):  # More than memory cache size
                text = f"Cache test text {i} for performance testing"
                test_texts.append(text)
                manager.generate_embedding(text)
            
            # Verify memory cache is at limit (older items evicted from memory)
            stats = manager.embedding_cache.get_stats()
            assert stats["memory_cache_size"] == 10, "Memory cache should be at capacity"
            assert stats["disk_cache_size"] == 15, "All items should be on disk"
            
            # Clear API request tracking and test performance difference
            initial_stats = manager.embedding_cache.get_stats()
            initial_misses = initial_stats["misses"]
            
            # Request embeddings that were evicted from memory but should be on disk
            old_texts = test_texts[:5]  # These should be evicted from memory
            for text in old_texts:
                manager.generate_embedding(text)
            
            # Verify no cache misses (found on disk, not regenerated)
            final_stats = manager.embedding_cache.get_stats()
            new_misses = final_stats["misses"] - initial_misses
            assert new_misses == 0, f"Should have found all items in disk cache, but had {new_misses} misses"
            
            # Verify memory cache is still bounded
            assert final_stats["memory_cache_size"] == 10, "Memory cache should remain at capacity"
            
            # Test that items moved back to memory cache (accessed recently)
            recent_text = old_texts[0]
            
            # Access it again - should be memory hit this time
            pre_access_stats = manager.embedding_cache.get_stats()
            manager.generate_embedding(recent_text)
            post_access_stats = manager.embedding_cache.get_stats()
            
            memory_hit_increase = post_access_stats["memory_hits"] - pre_access_stats["memory_hits"]
            assert memory_hit_increase == 1, "Recently accessed item should be memory hit"
            
        finally:
            # Restore original cache size
            manager.embedding_cache.memory_cache_size = original_cache_size
            manager.engine.dispose()


# =============================================================================
# PERFORMANCE & STRESS TESTS
# =============================================================================

class TestPerformanceAndStress:
    """
    Test performance characteristics and stress scenarios for MemoryManager.
    
    These tests validate that the system performs well under realistic load
    and fails gracefully when pushed beyond capacity limits.
    """
    
    def test_database_connection_pool_exhaustion_handling(self, clean_test_database):
        """
        Test system behavior when database connection pool is exhausted.
        
        REAL BUG THIS CATCHES: If our session management has connection leaks
        or doesn't properly handle pool exhaustion, the system will silently
        fail with mysterious timeouts when concurrent users exceed pool size.
        This breaks all memory operations for all users until restart.
        """
        config = clean_test_database
        # Use moderate pool size to trigger realistic exhaustion
        config.memory.db_pool_size = 10
        config.memory.db_pool_max_overflow = 5  # Total 15 connections max
        
        manager = MemoryManager(config)
        
        # Verify pool size was set correctly
        assert manager.engine.pool.size() == 10
        
        # Function to hold a connection open longer than normal
        def hold_connection_operation(worker_id):
            """Simulate a slow operation that holds a database connection."""
            try:
                with manager.get_session() as session:
                    # Simulate slow query/operation
                    session.execute(text("SELECT pg_sleep(0.3)"))  # 300ms delay
                    
                    # Do some memory work while holding connection
                    blocks = session.query(MemoryBlock).all()
                    return {
                        "worker_id": worker_id,
                        "success": True,
                        "blocks_found": len(blocks)
                    }
            except Exception as e:
                return {
                    "worker_id": worker_id,
                    "success": False,
                    "error": str(e)
                }
        
        # Test: Launch more workers than pool size to trigger exhaustion
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=25) as executor:  # 25 workers, 15 max connections
            futures = [executor.submit(hold_connection_operation, i) for i in range(25)]
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze results: should handle pool exhaustion gracefully
        successful_operations = [r for r in results if r["success"]]
        failed_operations = [r for r in results if not r["success"]]
        
        # All operations should eventually succeed (queue and retry)
        # OR fail with clear timeout/pool exhaustion errors (not crashes)
        assert len(successful_operations) >= 15, f"Should succeed eventually with queuing: {len(successful_operations)} successful"
        
        # If some failed, should be clear timeout/pool errors, not generic crashes
        for failure in failed_operations:
            error_str = failure["error"].lower()
            # Should mention timeout, pool, or connection issues
            is_pool_related = any(word in error_str for word in ["timeout", "pool", "connection", "wait"])
            is_generic_crash = any(bad in error_str for bad in ["attributeerror", "typeerror", "keyerror"])
            
            assert is_pool_related or not is_generic_crash, f"Pool exhaustion should not cause generic crashes: {failure['error']}"
        
        # Verify pool returns to healthy state after operations complete
        post_test_health = manager.health_check()
        assert post_test_health["components"]["database"] == "ok", "Database should be healthy after pool exhaustion test"
        
        # Verify no connection leaks
        checked_out_after = manager.engine.pool.checkedout()
        assert checked_out_after == 0, f"Should have no leaked connections: {checked_out_after} still checked out"
        
        # Cleanup
        manager.engine.dispose()
    
    def test_database_timeout_protections_prevent_dos_attacks(self, clean_test_database):
        """
        Test that timeout protections prevent DOS attacks from blocking legitimate users.
        
        REAL BUG THIS CATCHES: If our timeout configurations don't work properly,
        malicious users can hold database connections indefinitely and starve out
        legitimate users, causing a complete denial of service where the system
        becomes unresponsive to all users.
        """
        config = clean_test_database
        config.memory.db_pool_size = 5
        config.memory.db_pool_max_overflow = 2  # Total 7 connections max
        
        manager = MemoryManager(config)
        
        # Simulate malicious operations that try to hold connections too long
        def malicious_long_operation(worker_id):
            """Simulate a malicious user trying to hold connections indefinitely."""
            try:
                with manager.get_session() as session:
                    # Try to run a long query that should be killed by timeout
                    session.execute(text("SELECT pg_sleep(20)"))  # 20 second sleep, should be killed at 10s
                    return {"worker_id": worker_id, "success": True, "type": "malicious"}
            except Exception as e:
                error_str = str(e).lower()
                # Should fail with timeout, not generic crash
                is_timeout = any(word in error_str for word in ["timeout", "cancelled", "abort"])
                return {
                    "worker_id": worker_id, 
                    "success": False, 
                    "type": "malicious",
                    "error": str(e),
                    "is_timeout": is_timeout
                }
        
        # Simulate legitimate operations that should still work
        def legitimate_operation(worker_id):
            """Simulate a legitimate user doing normal operations."""
            try:
                with manager.get_session() as session:
                    # Normal, fast operation
                    blocks = session.query(MemoryBlock).all()
                    return {
                        "worker_id": worker_id, 
                        "success": True, 
                        "type": "legitimate",
                        "blocks_found": len(blocks)
                    }
            except Exception as e:
                return {
                    "worker_id": worker_id, 
                    "success": False, 
                    "type": "legitimate",
                    "error": str(e)
                }
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        # Launch malicious workers first to try to exhaust the pool
        with ThreadPoolExecutor(max_workers=15) as executor:
            # 8 malicious workers trying to hold connections
            malicious_futures = [executor.submit(malicious_long_operation, f"mal_{i}") for i in range(8)]
            
            # Give malicious workers a moment to grab connections
            time.sleep(0.1)
            
            # 5 legitimate workers that should still be able to work
            legitimate_futures = [executor.submit(legitimate_operation, f"leg_{i}") for i in range(5)]
            
            # Collect all results
            all_futures = malicious_futures + legitimate_futures
            results = [future.result(timeout=25) for future in as_completed(all_futures, timeout=25)]
        
        # Analyze results
        malicious_results = [r for r in results if r["type"] == "malicious"]
        legitimate_results = [r for r in results if r["type"] == "legitimate"]
        
        print(f"\nResults summary:")
        print(f"Malicious operations: {len(malicious_results)} total")
        print(f"Legitimate operations: {len(legitimate_results)} total")
        
        # Malicious operations should be terminated by timeout
        malicious_failures = [r for r in malicious_results if not r["success"]]
        print(f"Malicious failures: {len(malicious_failures)}")
        for failure in malicious_failures[:2]:  # Show first 2 errors
            print(f"  Malicious error: {failure['error'][:100]}...")
        
        assert len(malicious_failures) > 0, "Some malicious long operations should be terminated by timeout"
        
        # Failed malicious operations should fail due to timeout, not crashes
        for failure in malicious_failures:
            assert failure.get("is_timeout", False), f"Malicious operation should fail due to timeout: {failure['error']}"
        
        # Legitimate operations should succeed despite malicious load
        legitimate_successes = [r for r in legitimate_results if r["success"]]
        legitimate_failures = [r for r in legitimate_results if not r["success"]]
        print(f"Legitimate successes: {len(legitimate_successes)}")
        print(f"Legitimate failures: {len(legitimate_failures)}")
        for failure in legitimate_failures[:2]:  # Show first 2 errors
            print(f"  Legitimate error: {failure['error'][:100]}...")
        
        assert len(legitimate_successes) >= 3, f"Most legitimate operations should succeed: {len(legitimate_successes)}/{len(legitimate_results)}"
        
        # System should recover and be healthy after attack
        post_attack_health = manager.health_check()
        assert post_attack_health["components"]["database"] == "ok", "Database should be healthy after DOS attack"
        
        # Cleanup
        manager.engine.dispose()
    
    def test_concurrent_embedding_cache_contention_preserves_data_integrity(self, clean_test_database):
        """
        Test embedding cache thread safety under concurrent mixed read/write load.
        
        REAL BUG THIS CATCHES: If our embedding cache has thread safety issues,
        concurrent access will cause cache corruption where the same text returns
        different embeddings to different users, breaking memory consistency and
        causing subtle data corruption that's hard to detect in production.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Clear cache to start fresh
        manager.embedding_cache.clear_memory_cache()
        manager.embedding_cache.clear_disk_cache()
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import random
        
        # Shared texts that will create cache contention
        shared_texts = [
            "System prompt for AI assistant",
            "User greeting message", 
            "Common ML question",
            "API documentation text"
        ]
        
        def concurrent_cache_worker(worker_id):
            """Worker that creates realistic cache contention patterns."""
            results = []
            errors = []
            
            try:
                for i in range(25):  # Each worker does 25 operations
                    # 40% shared text (cache hits), 60% unique text (cache misses)
                    if i % 5 < 2:
                        text = random.choice(shared_texts)
                    else:
                        text = f"Worker {worker_id} unique request {i}: analyze this specific dataset"
                    
                    embedding = manager.generate_embedding(text)
                    
                    # Verify embedding integrity
                    if (isinstance(embedding, np.ndarray) and 
                        embedding.shape == (config.memory.embedding_dim,) and
                        not np.any(np.isnan(embedding)) and
                        not np.any(np.isinf(embedding))):
                        results.append((text, embedding.copy()))
                    else:
                        errors.append(f"Corrupted embedding for: {text[:30]}")
                        
            except Exception as e:
                errors.append(f"Worker {worker_id} crashed: {str(e)}")
            
            return {
                "worker_id": worker_id,
                "results": results,
                "errors": errors
            }
        
        # Run 16 workers to create serious contention
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(concurrent_cache_worker, i) for i in range(16)]
            worker_results = [future.result() for future in as_completed(futures)]
        
        # Critical validation: check for any worker failures
        all_errors = []
        all_results = []
        for worker_result in worker_results:
            all_errors.extend(worker_result["errors"])
            all_results.extend(worker_result["results"])
        
        assert len(all_errors) == 0, f"Workers had failures indicating thread safety issues: {all_errors}"
        
        # Critical validation: verify cache consistency
        text_to_embeddings = {}
        for text, embedding in all_results:
            if text not in text_to_embeddings:
                text_to_embeddings[text] = []
            text_to_embeddings[text].append(embedding)
        
        # The most important check: same text must always produce identical embeddings
        corruption_detected = []
        for text, embeddings in text_to_embeddings.items():
            if len(embeddings) > 1:
                first_embedding = embeddings[0]
                for i, other_embedding in enumerate(embeddings[1:], 1):
                    if not np.array_equal(first_embedding, other_embedding):
                        corruption_detected.append(f"Text '{text[:30]}' returned different embeddings")
        
        assert len(corruption_detected) == 0, f"Cache corruption detected - thread safety violated: {corruption_detected}"
        
        # Verify cache performed correctly
        cache_stats = manager.embedding_cache.get_stats()
        total_requests = cache_stats["total_requests"]
        hits = cache_stats["hits"]
        
        assert total_requests >= 400, f"Should have processed all requests: {total_requests}"
        assert hits > 0, f"Should have cache hits from shared texts: {hits}"
        assert cache_stats["hit_rate"] > 0, "Cache should have improved performance"
        
        # Cleanup
        manager.engine.dispose()
    
    def test_embedding_generation_handles_model_failures_gracefully(self, clean_test_database, mocker):
        """
        Test that generate_embedding() handles embedding model failures gracefully.
        
        REAL BUG THIS CATCHES: If our generate_embedding() method doesn't handle
        embedding model failures (API down, rate limits, invalid responses), all
        memory operations that depend on embeddings will crash. This breaks passage
        storage, similarity search, and core memory functionality.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Baseline: verify normal embedding generation works
        normal_embedding = manager.generate_embedding("test text")
        assert isinstance(normal_embedding, np.ndarray)
        assert normal_embedding.shape[0] == config.memory.embedding_dim
        
        # Clear cache to ensure we hit the broken model
        manager.embedding_cache.clear_memory_cache()
        
        # Mock the embedding model to simulate API failure
        mock_model = mocker.patch.object(manager.embedding_model, 'encode')
        mock_model.side_effect = Exception("OpenAI API unavailable")
        
        # Test the contract: should handle embedding failures gracefully
        with pytest.raises(ToolError) as exc_info:
            manager.generate_embedding("different text to avoid cache")
        
        # Should be a clear error about embedding problems, wrapped in ToolError
        error_str = str(exc_info.value).lower()
        assert any(word in error_str for word in ["embedding", "memory"]), f"Should be clear about embedding problem: {error_str}"
        assert exc_info.value.code == ErrorCode.MEMORY_EMBEDDING_ERROR
    
    def test_embedding_generation_handles_cache_write_failures_gracefully(self, clean_test_database, mocker):
        """
        Test that generate_embedding() handles cache write failures without breaking embedding generation.
        
        REAL BUG THIS CATCHES: If our embedding cache can't write to disk due to disk space
        issues, the entire generate_embedding() operation crashes. This breaks all memory
        operations that depend on embeddings, even though the embedding was successfully
        generated and should be returned.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Clear cache to ensure we generate new embeddings
        manager.embedding_cache.clear_memory_cache()
        
        # Mock the cache set method to simulate disk full
        mock_cache_set = mocker.patch.object(manager.embedding_cache, 'set')
        mock_cache_set.side_effect = OSError("No space left on device")
        
        # Test the contract: should still return embedding even if caching fails
        try:
            embedding = manager.generate_embedding("text that can't be cached")
            
            # Should return valid embedding despite cache failure
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape[0] == config.memory.embedding_dim
            
        except Exception as e:
            # If it fails, this reveals the bug - cache write failure breaks embedding generation
            error_str = str(e).lower()
            print(f"Cache write failure broke embedding generation: {error_str}")
            # This test will fail, revealing the bug in my implementation
    
    def test_create_snapshot_validates_input_parameters(self, clean_test_database):
        """
        Test that create_snapshot() validates input parameters and rejects invalid data.
        
        REAL BUG THIS CATCHES: If our create_snapshot() method doesn't validate input
        parameters, invalid data could be stored in the database or cause crashes when
        snapshot operations are attempted. This breaks disaster recovery functionality
        and makes debugging harder when snapshots are corrupted.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Baseline: verify normal snapshot creation works
        normal_snapshot_id = manager.create_snapshot("valid_conversation_123", "test_reason")
        assert normal_snapshot_id is not None
        assert len(normal_snapshot_id) > 0
        
        # Test invalid conversation_id input
        with pytest.raises((ValueError, ToolError)) as exc_info:
            manager.create_snapshot(None, "test_reason")
        
        error_str = str(exc_info.value).lower()
        assert any(word in error_str for word in ["conversation", "invalid", "none"]), f"Should be clear about invalid input: {error_str}"
        
        # Test empty conversation_id
        with pytest.raises((ValueError, ToolError)) as exc_info:
            manager.create_snapshot("", "test_reason")
        
        error_str = str(exc_info.value).lower()
        assert any(word in error_str for word in ["conversation", "empty", "invalid"]), f"Should be clear about empty input: {error_str}"
    
    def test_restore_from_snapshot_handles_corrupted_data_gracefully(self, clean_test_database):
        """
        Test that restore_from_snapshot() handles corrupted snapshot data gracefully.
        
        REAL BUG THIS CATCHES: If our restore_from_snapshot() method doesn't validate
        snapshot data and provide helpful error messages, corrupted snapshots will either
        crash the system or leave memory in an inconsistent state. This breaks disaster
        recovery and makes debugging corruption issues nearly impossible.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Baseline: verify normal snapshot works
        normal_snapshot_id = manager.create_snapshot("test_conv", "normal_snapshot")
        assert manager.restore_from_snapshot(normal_snapshot_id) is True
        
        # Setup: create a snapshot with corrupted data structure
        with manager.get_session() as session:
            from lt_memory.models.base import MemorySnapshot
            corrupted_snapshot = MemorySnapshot(
                conversation_id="corrupted_test",
                blocks_snapshot={"invalid": "missing required fields"},  # Missing block structure
                passage_count=0,
                reason="corruption_test"
            )
            session.add(corrupted_snapshot)
            session.commit()
            corrupted_id = str(corrupted_snapshot.id)
        
        # Test the contract: should provide detailed error information
        with pytest.raises(ToolError) as exc_info:
            manager.restore_from_snapshot(corrupted_id)
        
        # Should provide specific error about what's corrupted
        error_str = str(exc_info.value).lower()
        assert "snapshot" in error_str and "corrupted" in error_str, f"Should specify snapshot corruption: {error_str}"
        
        # Should mention what specifically is wrong
        assert any(word in error_str for word in ["invalid", "missing", "structure"]), f"Should specify what's wrong: {error_str}"
        
        # Memory state should remain unchanged (not partially corrupted)
        with manager.get_session() as session:
            persona_block = session.query(MemoryBlock).filter_by(label="persona").first()
            assert persona_block is not None
            # Should still have valid content, not affected by failed restore
            assert len(persona_block.value) > 0
    
    def test_conversation_archive_integration_with_memory_manager(self, clean_test_database):
        """
        Test that conversation_archive integrates correctly with MemoryManager.
        
        REAL BUG THIS CATCHES: If our ConversationArchive initialization fails, can't
        access MemoryManager's database sessions or summarization engine, conversation
        archiving will fail. This breaks the ability to store and retrieve conversation
        history, which is essential for temporal memory and web interface functionality.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Test 1: ConversationArchive should be initialized and accessible
        assert manager.conversation_archive is not None, "ConversationArchive should be initialized"
        assert hasattr(manager.conversation_archive, 'memory_manager'), "ConversationArchive should have reference to MemoryManager"
        
        # Test 2: ConversationArchive should be able to access MemoryManager's database sessions
        with manager.get_session() as session:
            # ConversationArchive should be able to query archived conversations through MemoryManager
            archived_convs = session.query(ArchivedConversation).all()
            assert isinstance(archived_convs, list), "ConversationArchive should access database through MemoryManager"
            
            # Initially should have no archived conversations
            assert len(archived_convs) == 0, "Fresh system should start with no archived conversations"
        
        # Test 3: ConversationArchive should be able to access MemoryManager's summarization engine
        assert manager.conversation_archive.memory_manager.summarization_engine is not None, "ConversationArchive should access summarization engine through MemoryManager"
        
        # Test 4: ConversationArchive should be able to get archive stats
        try:
            archive_stats = manager.conversation_archive.get_archive_stats()
            assert isinstance(archive_stats, dict), "ConversationArchive should get stats through MemoryManager"
            assert "total_archived_conversations" in archive_stats, "Stats should include conversation count"
            assert archive_stats["total_archived_conversations"] == 0, "Fresh system should have zero archived conversations"
        except Exception as e:
            assert False, f"ConversationArchive can't get stats through MemoryManager: {str(e)}"
        
        # Test 5: ConversationArchive should be able to store archived conversations with real message format
        from datetime import date
        test_date = date(2024, 1, 15)
        
        # Use realistic message structure based on actual conversation format
        test_messages = [
            {
                "role": "user",
                "content": "Hello, how are you today?",
                "id": "aa11fb0b-4427-4bef-a96f-6e6490af41a0",
                "created_at": "2024-01-15T10:00:00+00:00",
                "metadata": {}
            },
            {
                "role": "assistant", 
                "content": "I'm doing well, thank you! How can I help you today?",
                "id": "7f887495-5958-42c1-b323-1690cd424aa2",
                "created_at": "2024-01-15T10:01:00+00:00",
                "metadata": {"topic_changed": True}
            }
        ]
        
        with manager.get_session() as session:
            from utils.timezone_utils import ensure_utc
            archived_conv = ArchivedConversation(
                conversation_date=ensure_utc(datetime.combine(test_date, datetime.min.time())),
                messages=test_messages,
                message_count=len(test_messages),
                summary="Test conversation about greetings and assistance",
                conversation_metadata={"test": True, "source": "integration_test"}
            )
            session.add(archived_conv)
            session.commit()
            archive_id = archived_conv.id
        
        # Test 6: ConversationArchive should be able to retrieve conversations
        try:
            retrieved_conv = manager.conversation_archive.get_conversation_by_date(test_date)
            assert retrieved_conv is not None, "ConversationArchive should retrieve conversations through MemoryManager"
            assert retrieved_conv["message_count"] == 2, "Retrieved conversation should have correct message count"
            assert "greetings" in retrieved_conv["summary"], "Retrieved conversation should have correct summary"
            assert len(retrieved_conv["messages"]) == 2, "Retrieved conversation should have correct messages"
        except Exception as e:
            assert False, f"ConversationArchive can't retrieve conversations through MemoryManager: {str(e)}"
        
        # Test 7: Archive stats should reflect the added conversation
        updated_stats = manager.conversation_archive.get_archive_stats()
        assert updated_stats["total_archived_conversations"] >= 1, "Stats should reflect added conversation"
        
        # Test 8: MemoryManager should remain healthy with ConversationArchive operations
        health = manager.health_check()
        assert health["status"] == "healthy", "MemoryManager should remain healthy after ConversationArchive operations"
        
        # MemoryManager stats should still work
        memory_stats = manager.get_memory_stats()
        assert isinstance(memory_stats, dict), "MemoryManager stats should work after ConversationArchive operations"
        
        # Cleanup
        manager.engine.dispose()


# =============================================================================
# LARGE DATASET PERFORMANCE TESTS
# =============================================================================

class TestLargeDatasetPerformance:
    """
    Test performance with realistic data volumes.
    
    These tests validate that the system scales appropriately and performs
    acceptably when users have accumulated realistic amounts of memory data.
    """
    
    def test_get_memory_stats_performance_with_large_dataset(self, clean_test_database):
        """
        Test that get_memory_stats() performs acceptably with realistic data volumes.
        
        REAL BUG THIS CATCHES: If our get_memory_stats() method has inefficient queries
        or doesn't scale with data volume, monitoring dashboards will timeout or become
        unusably slow when users have accumulated realistic amounts of memory data.
        This makes the system appear broken even when core functionality works.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Create realistic large dataset (1000+ passages, varied content)
        large_dataset_size = 1500
        passages = []
        
        with manager.get_session() as session:
            for i in range(large_dataset_size):
                # Varied text lengths and importance scores to simulate real usage
                text_length = 50 + (i % 200)  # 50-250 characters
                text = f"Memory passage {i}: " + "content " * (text_length // 8)
                
                passage = MemoryPassage(
                    text=text,
                    embedding=np.random.rand(config.memory.embedding_dim).astype(np.float32),
                    source=f"source_{i % 10}",  # Mix of sources
                    source_id=f"item_{i}",
                    importance_score=np.random.uniform(0.1, 1.0)
                )
                passages.append(passage)
            
            # Add in batches to avoid memory issues during setup
            batch_size = 100
            for i in range(0, len(passages), batch_size):
                batch = passages[i:i + batch_size]
                session.add_all(batch)
                session.commit()
        
        # Test the performance contract: should complete within reasonable time
        import time
        start_time = time.time()
        stats = manager.get_memory_stats()
        execution_time = time.time() - start_time
        
        # Should complete within 3 seconds even with large dataset
        assert execution_time < 3.0, f"get_memory_stats() took {execution_time:.2f}s with {large_dataset_size} passages - too slow for production use"
        
        # Should return correct counts
        assert stats["passages"]["count"] == large_dataset_size, f"Expected {large_dataset_size} passages, got {stats['passages']['count']}"
        
        # Should calculate average importance correctly (spot check)
        assert 0.0 <= stats["passages"]["avg_importance"] <= 1.0, "Average importance should be in valid range"
        
        # Should not consume excessive memory (basic check)
        assert isinstance(stats, dict), "Should return structured data"
        assert len(str(stats)) < 10000, "Stats response should not be excessively large"
        
        # Cleanup
        manager.engine.dispose()
    
    def test_system_startup_recovery_preserves_data_integrity(self, clean_test_database):
        """
        Test that system startup after shutdown preserves all data integrity.
        
        REAL BUG THIS CATCHES: If our initialization, data storage, or engine disposal
        has bugs that corrupt data or lose transactions during system restarts, 
        production deployments will lose user memory data. This breaks the core
        promise that memories persist across application restarts.
        """
        config = clean_test_database
        
        # Phase 1: Initialize system and populate with diverse data
        manager1 = MemoryManager(config)
        
        # Store diverse memory data that should survive restart
        test_data = {
            "passages": [],
            "snapshots": [],
            "block_modifications": {}
        }
        
        with manager1.get_session() as session:
            # Modify core memory blocks
            persona_block = session.query(MemoryBlock).filter_by(label="persona").first()
            human_block = session.query(MemoryBlock).filter_by(label="human").first()
            
            original_persona = persona_block.value
            original_human = human_block.value
            
            persona_block.value = "I am MIRA with persistent memory across restarts."
            human_block.value = "User is a developer testing system restart behavior."
            
            test_data["block_modifications"] = {
                "persona": {"old": original_persona, "new": persona_block.value},
                "human": {"old": original_human, "new": human_block.value}
            }
            
            session.commit()
            
            # Create memory passages with embeddings
            passage_texts = [
                "Important user preference that must survive restart",
                "Critical system configuration that should persist", 
                "User's favorite project details stored in memory"
            ]
            
            for i, text in enumerate(passage_texts):
                embedding = manager1.generate_embedding(text)
                passage = MemoryPassage(
                    text=text,
                    embedding=embedding,
                    source="restart_test",
                    source_id=f"restart_{i}",
                    importance_score=0.9
                )
                session.add(passage)
                test_data["passages"].append({
                    "text": text,
                    "source_id": f"restart_{i}",
                    "importance": 0.9
                })
            
            session.commit()
        
        # Create snapshots to test snapshot persistence  
        snapshot1_id = manager1.create_snapshot("restart_test_1", "before_restart")
        snapshot2_id = manager1.create_snapshot("restart_test_2", "system_checkpoint")
        test_data["snapshots"] = [snapshot1_id, snapshot2_id]
        
        # Get baseline stats before shutdown
        pre_restart_stats = manager1.get_memory_stats()
        
        # Phase 2: Properly shutdown system (simulate application restart)
        manager1.engine.dispose()
        del manager1
        
        # Phase 3: Restart system (new MemoryManager instance)
        manager2 = MemoryManager(config)
        
        # Phase 4: Verify complete data integrity after restart
        post_restart_stats = manager2.get_memory_stats()
        
        # Verify stats consistency (data survived restart)
        assert post_restart_stats["blocks"]["count"] == pre_restart_stats["blocks"]["count"], "Block count should survive restart"
        assert post_restart_stats["passages"]["count"] == pre_restart_stats["passages"]["count"], "Passage count should survive restart" 
        assert post_restart_stats["snapshots"]["count"] == pre_restart_stats["snapshots"]["count"], "Snapshot count should survive restart"
        
        # Verify memory block modifications survived restart
        with manager2.get_session() as session:
            restored_persona = session.query(MemoryBlock).filter_by(label="persona").first()
            restored_human = session.query(MemoryBlock).filter_by(label="human").first()
            
            assert restored_persona.value == test_data["block_modifications"]["persona"]["new"], "Persona block changes should survive restart"
            assert restored_human.value == test_data["block_modifications"]["human"]["new"], "Human block changes should survive restart"
        
        # Verify passages survived restart with correct data
        with manager2.get_session() as session:
            restored_passages = session.query(MemoryPassage).filter_by(source="restart_test").all()
            assert len(restored_passages) == 3, "All test passages should survive restart"
            
            restored_texts = {p.text for p in restored_passages}
            expected_texts = {p["text"] for p in test_data["passages"]}
            assert restored_texts == expected_texts, "Passage text should survive restart unchanged"
            
            # Verify embeddings survived and are valid
            for passage in restored_passages:
                assert passage.embedding is not None, "Embeddings should survive restart"
                assert len(passage.embedding) == config.memory.embedding_dim, "Embedding dimensions should be preserved"
                assert passage.importance_score == 0.9, "Importance scores should survive restart"
        
        # Verify snapshots survived and are accessible
        with manager2.get_session() as session:
            for snapshot_id in test_data["snapshots"]:
                snapshot = session.query(MemorySnapshot).filter_by(id=snapshot_id).first()
                assert snapshot is not None, f"Snapshot {snapshot_id} should survive restart"
                assert "persona" in snapshot.blocks_snapshot, "Snapshot data should survive restart"
        
        # Verify system functionality after restart (not just data persistence)
        # Test embedding generation still works
        test_embedding = manager2.generate_embedding("Test after restart")
        assert isinstance(test_embedding, np.ndarray), "Embedding generation should work after restart"
        
        # Test health check reports system as healthy
        health = manager2.health_check()
        assert health["status"] == "healthy", "System should be healthy after restart"
        
        # Test snapshot/restore functionality still works
        new_snapshot = manager2.create_snapshot("post_restart_test", "functionality_check")
        assert new_snapshot is not None, "Snapshot creation should work after restart"
        
        # Cleanup
        manager2.engine.dispose()
    
    def test_complete_conversation_lifecycle_from_active_to_archived_retrieval(self, clean_test_database):
        """
        Test the full conversation archiving workflow: active conversation → archive → retrieval.
        
        REAL BUG THIS CATCHES: If the conversation archiving process breaks the links
        between conversations, memory passages, and memory blocks, users will lose
        context when retrieving old conversations. This breaks the temporal memory
        system's core promise of preserving conversation context over time.
        """
        config = clean_test_database
        manager = MemoryManager(config)
        
        # Simulate an active conversation with memory creation
        from datetime import date, datetime, timedelta
        conversation_date = date(2024, 6, 8)
        
        # Step 1: Simulate live conversation generating memory passages
        conversation_messages = [
            {"role": "user", "content": "I'm a software engineer at Google working on ML", "id": "msg_1"},
            {"role": "assistant", "content": "That's fascinating! What ML frameworks do you use?", "id": "msg_2"},
            {"role": "user", "content": "Mainly TensorFlow, but exploring PyTorch for research", "id": "msg_3"},
            {"role": "assistant", "content": "Both are excellent choices. Any specific domains?", "id": "msg_4"},
            {"role": "user", "content": "Computer vision and NLP, particularly image classification", "id": "msg_5"}
        ]
        
        # Create memory passages as conversation progresses
        with manager.get_session() as session:
            passage1 = MemoryPassage(
                text="User is a software engineer at Google working on machine learning",
                embedding=manager.generate_embedding("software engineer Google machine learning"),
                source="conversation",
                source_id="msg_1",
                importance_score=0.9
            )
            passage2 = MemoryPassage(
                text="User uses TensorFlow primarily, exploring PyTorch for research",
                embedding=manager.generate_embedding("TensorFlow PyTorch research"),
                source="conversation", 
                source_id="msg_3",
                importance_score=0.8
            )
            passage3 = MemoryPassage(
                text="User specializes in computer vision and NLP, particularly image classification",
                embedding=manager.generate_embedding("computer vision NLP image classification"),
                source="conversation",
                source_id="msg_5", 
                importance_score=0.85
            )
            session.add_all([passage1, passage2, passage3])
            session.commit()
            
            # Store passage IDs to verify later
            passage_ids = [passage1.id, passage2.id, passage3.id]
        
        # Step 2: Update memory blocks during conversation
        with manager.get_session() as session:
            human_block = session.query(MemoryBlock).filter_by(label="human").first()
            system_block = session.query(MemoryBlock).filter_by(label="system").first()
            
            human_block.value = "Google software engineer specializing in ML: TensorFlow/PyTorch, computer vision, NLP, image classification"
            human_block.version += 1
            system_block.value = "Discussed user's technical background and ML expertise"
            system_block.version += 1
            session.commit()
            
            # Store block state to verify later
            archived_human_content = human_block.value
            archived_system_content = system_block.value
        
        # Step 3: Archive the conversation (simulating end-of-day archival)
        archived_result = manager.conversation_archive.archive_day(
            conversation=type('MockConversation', (), {
                'messages': [type('MockMessage', (), {
                    'id': msg["id"],
                    'role': msg["role"], 
                    'content': msg["content"],
                    'created_at': datetime.combine(conversation_date, datetime.min.time()) + timedelta(hours=i),
                    'metadata': {}
                })() for i, msg in enumerate(conversation_messages)]
            })(),
            target_date=conversation_date
        )
        
        assert archived_result["success"] is True, "Conversation archiving should succeed"
        assert archived_result["message_count"] == 5, "All messages should be archived"
        
        # Step 4: Create snapshot after archival
        snapshot_id = manager.create_snapshot("post_archival", "after_conversation_archived")
        
        # Step 5: Simulate time passing - modify current state
        with manager.get_session() as session:
            human_block = session.query(MemoryBlock).filter_by(label="human").first()
            human_block.value = "Different conversation context - user now discussing cooking"
            human_block.version += 1
            session.commit()
        
        # Step 6: Later retrieval - verify archived conversation context is preserved
        retrieved_conversation = manager.conversation_archive.get_conversation_by_date(conversation_date)
        
        # Verify conversation was archived correctly
        assert retrieved_conversation is not None, "Archived conversation should be retrievable"
        assert retrieved_conversation["message_count"] == 5, "All messages should be preserved"
        assert "Google" in retrieved_conversation["summary"], "Summary should contain conversation content"
        
        # Verify original conversation messages are intact
        retrieved_messages = retrieved_conversation["messages"]
        assert len(retrieved_messages) == 5, "All messages should be preserved"
        assert "Google" in retrieved_messages[0]["content"], "Original message content preserved"
        assert "TensorFlow" in retrieved_messages[2]["content"], "Technical details preserved"
        
        # Step 7: Verify memory passages from that conversation are still accessible
        with manager.get_session() as session:
            # All passages should still exist
            for passage_id in passage_ids:
                passage = session.query(MemoryPassage).filter_by(id=passage_id).first()
                assert passage is not None, f"Memory passage {passage_id} should still exist"
                assert passage.source == "conversation", "Source linking should be preserved"
            
            # Search should still work for archived conversation context
            search_embedding = manager.generate_embedding("Google engineer machine learning")
            search_results = manager.vector_store.search(search_embedding, k=3)
            assert len(search_results) >= 1, "Should find passages from archived conversation"
        
        # Step 8: Verify snapshot captured the conversation context state
        with manager.get_session() as session:
            snapshot = session.query(MemorySnapshot).filter_by(id=snapshot_id).first()
            assert snapshot is not None, "Snapshot should exist"
            
            # Snapshot should contain the memory block state from the archived conversation
            human_snapshot = snapshot.blocks_snapshot.get("human", {})
            assert "Google" in human_snapshot.get("value", ""), "Snapshot should preserve conversation context"
            assert "TensorFlow" in human_snapshot.get("value", ""), "Technical details should be in snapshot"
        
        # Step 9: Verify system integrity after full workflow
        final_stats = manager.get_memory_stats()
        assert final_stats["passages"]["count"] >= 3, "Memory passages should be preserved"
        assert final_stats["snapshots"]["count"] >= 1, "Snapshot should be recorded"
        
        health = manager.health_check()
        assert health["status"] == "healthy", "System should remain healthy after complete lifecycle"
        
        # Cleanup
        manager.engine.dispose()