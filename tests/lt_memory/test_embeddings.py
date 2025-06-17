"""
Production-grade tests for EmbeddingCache system.

This test suite focuses on realistic production scenarios that matter for
reliability. We test the contracts and behaviors that users actually depend on,
covering thread safety, persistence, and performance characteristics.

Testing philosophy:
1. Test the public API contracts that users rely on
2. Test critical private methods that could fail in subtle ways
3. Focus on thread safety and concurrent access patterns
4. Test persistence and corruption recovery
5. Verify performance characteristics under realistic load
6. Include stress tests to discover system limits
"""

import pytest
import time
import threading
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch, mock_open
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the system under test
from api.embeddings_provider import EmbeddingCache


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path):
    """
    Provides a temporary cache directory for testing.
    
    This ensures tests don't interfere with each other or leave artifacts.
    """
    return tmp_path / "test_cache"


@pytest.fixture
def sample_embeddings():
    """
    Provides realistic embedding samples for testing.
    
    These represent actual embedding dimensions used in production:
    - Small: 128-dim (lightweight models)
    - Medium: 512-dim (common transformer models)
    - Large: 1536-dim (OpenAI embeddings)
    """
    np.random.seed(42)  # Reproducible test data
    return {
        "small": np.random.rand(128).astype(np.float32),
        "medium": np.random.rand(512).astype(np.float32),
        "large": np.random.rand(1536).astype(np.float32)
    }


@pytest.fixture
def sample_texts():
    """
    Provides realistic text samples for testing.
    
    These cover common use cases and edge cases:
    - Standard text
    - Unicode content
    - Empty strings
    - Very long text
    """
    return {
        "standard": "This is a standard document for embedding.",
        "unicode": "Unicode text: 世界 🌍 Привет мир! Café naïve résumé",
        "empty": "",
        "long": "Very long text. " * 1000,  # ~16KB text
        "special_chars": "Text with\nnewlines\tand\rspecial chars!@#$%^&*()",
        "repeated": "Repeated text for cache testing"
    }


@pytest.fixture
def basic_cache(temp_cache_dir):
    """Basic cache instance for testing."""
    return EmbeddingCache(str(temp_cache_dir), memory_cache_size=5)


@pytest.fixture
def populated_cache(temp_cache_dir, sample_embeddings, sample_texts):
    """Pre-populated cache for testing complex scenarios."""
    cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=10)
    
    # Populate with known data
    for text_key, text in sample_texts.items():
        if text:  # Skip empty string for now
            embedding = sample_embeddings["medium"]
            cache.set(text, embedding)
    
    return cache


# =============================================================================
# PRIVATE METHOD UNIT TESTS
# =============================================================================

class TestCacheKeyGeneration:
    """
    Test hash-based cache key generation.
    
    These tests ensure the cache key system works reliably for all input types.
    """
    
    def test_identical_text_produces_same_key(self, basic_cache):
        """
        Test that identical text always produces the same cache key.
        
        This is fundamental for cache consistency.
        """
        text = "Test content"
        key1 = basic_cache._get_cache_key(text)
        key2 = basic_cache._get_cache_key(text)
        
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex length
    
    def test_different_text_produces_different_keys(self, basic_cache):
        """
        Test that different text produces different cache keys.
        
        This prevents cache collisions.
        """
        key1 = basic_cache._get_cache_key("Text one")
        key2 = basic_cache._get_cache_key("Text two")
        
        assert key1 != key2
    
    def test_unicode_text_handling(self, basic_cache, sample_texts):
        """
        Test that Unicode text is handled correctly.
        
        Production systems must support international users.
        """
        unicode_text = sample_texts["unicode"]
        key = basic_cache._get_cache_key(unicode_text)
        
        # Should produce valid key without errors
        assert isinstance(key, str)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)
    
    def test_empty_string_handling(self, basic_cache):
        """
        Test edge case: empty string input.
        
        Empty strings are valid input and should work.
        """
        key = basic_cache._get_cache_key("")
        
        assert isinstance(key, str)
        assert len(key) == 64
    
    def test_very_long_text_handling(self, basic_cache, sample_texts):
        """
        Test that very long text doesn't break key generation.
        
        Some embeddings use very long context windows.
        """
        long_text = sample_texts["long"]
        key = basic_cache._get_cache_key(long_text)
        
        assert isinstance(key, str)
        assert len(key) == 64
    
    def test_special_characters_in_text(self, basic_cache, sample_texts):
        """
        Test text with newlines, tabs, and special characters.
        
        Real text contains various control characters.
        """
        special_text = sample_texts["special_chars"]
        key = basic_cache._get_cache_key(special_text)
        
        assert isinstance(key, str)
        assert len(key) == 64


class TestCachePathGeneration:
    """
    Test file path generation logic.
    
    Path generation affects filesystem organization and performance.
    """
    
    def test_path_uses_subdirectories(self, basic_cache):
        """
        Test that paths use subdirectories for filesystem efficiency.
        
        Too many files in one directory hurts performance.
        """
        cache_key = "abcdef1234567890" * 4  # 64-char hex
        path = basic_cache._get_cache_path(cache_key)
        
        # Should use first 2 chars as subdirectory
        assert "ab" in str(path)
        assert path.name == f"{cache_key}.npy"
    
    def test_path_includes_full_hash(self, basic_cache):
        """
        Test that complete hash is in filename to prevent collisions.
        """
        cache_key = "1234567890abcdef" * 4
        path = basic_cache._get_cache_path(cache_key)
        
        assert cache_key in path.name
        assert path.suffix == ".npy"
    
    def test_path_is_within_cache_directory(self, basic_cache):
        """
        Test that paths don't escape cache directory.
        
        Security: prevent directory traversal issues.
        """
        cache_key = "a" * 64
        path = basic_cache._get_cache_path(cache_key)
        
        # Path should be within cache directory
        assert basic_cache.cache_dir in path.parents or path == basic_cache.cache_dir
    
    def test_consistent_paths_for_same_key(self, basic_cache):
        """
        Test that same key always gives same path.
        """
        cache_key = "consistent" + "0" * 54
        path1 = basic_cache._get_cache_path(cache_key)
        path2 = basic_cache._get_cache_path(cache_key)
        
        assert path1 == path2
    
    def test_different_keys_give_different_paths(self, basic_cache):
        """
        Test that different keys produce different paths.
        """
        key1 = "a" * 64
        key2 = "b" * 64
        path1 = basic_cache._get_cache_path(key1)
        path2 = basic_cache._get_cache_path(key2)
        
        assert path1 != path2


class TestMemoryCacheManagement:
    """
    Test LRU memory cache behavior.
    
    Memory cache management is critical for performance and memory usage.
    """
    
    def test_adds_new_embedding_to_cache(self, basic_cache, sample_embeddings):
        """
        Test basic addition to memory cache.
        """
        cache_key = "test_key"
        embedding = sample_embeddings["small"]
        
        basic_cache._add_to_memory_cache(cache_key, embedding)
        
        assert cache_key in basic_cache.memory_cache
        np.testing.assert_array_equal(basic_cache.memory_cache[cache_key], embedding)
    
    def test_evicts_oldest_when_at_capacity(self, temp_cache_dir, sample_embeddings):
        """
        Test that LRU eviction works correctly.
        
        This prevents memory leaks in production.
        """
        # Small cache for testing eviction
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=3)
        embedding = sample_embeddings["small"]
        
        # Fill cache to capacity
        cache._add_to_memory_cache("key1", embedding)
        cache._add_to_memory_cache("key2", embedding)
        cache._add_to_memory_cache("key3", embedding)
        
        assert len(cache.memory_cache) == 3
        
        # Add one more - should evict oldest
        cache._add_to_memory_cache("key4", embedding)
        
        assert len(cache.memory_cache) == 3
        assert "key1" not in cache.memory_cache  # Oldest evicted
        assert "key4" in cache.memory_cache  # Newest added
    
    def test_preserves_most_recent_embeddings(self, temp_cache_dir, sample_embeddings):
        """
        Test that LRU keeps frequently used items.
        """
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=2)
        embedding = sample_embeddings["small"]
        
        # Add items
        cache._add_to_memory_cache("key1", embedding)
        cache._add_to_memory_cache("key2", embedding)
        
        # Access key1 to make it most recent
        cache.memory_cache.move_to_end("key1")
        
        # Add new item - should evict key2, not key1
        cache._add_to_memory_cache("key3", embedding)
        
        assert "key1" in cache.memory_cache  # Recently used, kept
        assert "key2" not in cache.memory_cache  # Least recent, evicted
        assert "key3" in cache.memory_cache
    
    def test_embedding_data_integrity(self, basic_cache, sample_embeddings):
        """
        Test that stored embeddings match input data.
        
        Cache corruption would be silent and dangerous.
        """
        cache_key = "integrity_test"
        original_embedding = sample_embeddings["medium"]
        
        basic_cache._add_to_memory_cache(cache_key, original_embedding)
        retrieved_embedding = basic_cache.memory_cache[cache_key]
        
        np.testing.assert_array_equal(retrieved_embedding, original_embedding)
    
    def test_capacity_limit_enforcement(self, temp_cache_dir, sample_embeddings):
        """
        Test that cache never exceeds max size.
        
        Memory protection is critical.
        """
        max_size = 5
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=max_size)
        embedding = sample_embeddings["small"]
        
        # Add more than capacity
        for i in range(max_size + 3):
            cache._add_to_memory_cache(f"key{i}", embedding)
            assert len(cache.memory_cache) <= max_size
    
    def test_embedding_independence(self, basic_cache, sample_embeddings):
        """
        Test that cached embeddings are independent copies.
        
        Prevents accidental modification of cached data.
        """
        cache_key = "independence_test"
        original_embedding = sample_embeddings["small"].copy()
        
        basic_cache._add_to_memory_cache(cache_key, original_embedding)
        
        # Modify original
        original_embedding[0] = 999.0
        
        # Cached version should be unchanged
        cached_embedding = basic_cache.memory_cache[cache_key]
        assert cached_embedding[0] != 999.0


# =============================================================================
# PUBLIC METHOD CONTRACT TESTS
# =============================================================================

class TestCacheRetrieval:
    """
    Test embedding retrieval functionality.
    
    These test the primary user-facing functionality.
    """
    
    def test_returns_none_for_uncached_text(self, basic_cache):
        """
        Test cache miss behavior.
        
        Should return None for text that hasn't been cached.
        """
        result = basic_cache.get("uncached text")
        assert result is None
    
    def test_returns_cached_embedding_when_available(self, basic_cache, sample_embeddings):
        """
        Test cache hit behavior.
        
        Should return exact embedding that was stored.
        """
        text = "cached text"
        embedding = sample_embeddings["medium"]
        
        basic_cache.set(text, embedding)
        result = basic_cache.get(text)
        
        assert result is not None
        np.testing.assert_array_equal(result, embedding)
    
    def test_memory_cache_hit_updates_stats(self, basic_cache, sample_embeddings):
        """
        Test that memory cache hits are tracked in statistics.
        """
        text = "memory test"
        embedding = sample_embeddings["small"]
        
        # Store and immediately retrieve (memory hit)
        basic_cache.set(text, embedding)
        initial_memory_hits = basic_cache.stats["memory_hits"]
        
        basic_cache.get(text)
        
        assert basic_cache.stats["memory_hits"] == initial_memory_hits + 1
        assert basic_cache.stats["hits"] > 0
    
    def test_disk_cache_hit_loads_to_memory(self, basic_cache, sample_embeddings):
        """
        Test that disk hits populate memory cache.
        
        This optimizes future access.
        """
        text = "disk test"
        embedding = sample_embeddings["medium"]
        
        # Store to cache
        basic_cache.set(text, embedding)
        
        # Clear memory cache to force disk hit
        basic_cache.clear_memory_cache()
        
        # Should load from disk and populate memory
        result = basic_cache.get(text)
        
        assert result is not None
        np.testing.assert_array_equal(result, embedding)
        
        # Should now be in memory cache
        cache_key = basic_cache._get_cache_key(text)
        assert cache_key in basic_cache.memory_cache
    
    def test_corrupted_disk_file_handled_gracefully(self, basic_cache, sample_embeddings):
        """
        Test that corrupted files are removed and don't crash system.
        
        File corruption happens in production.
        """
        text = "corruption test"
        embedding = sample_embeddings["small"]
        
        # Store embedding
        basic_cache.set(text, embedding)
        
        # Clear memory cache
        basic_cache.clear_memory_cache()
        
        # Corrupt the disk file
        cache_key = basic_cache._get_cache_key(text)
        cache_path = basic_cache._get_cache_path(cache_key)
        with open(cache_path, 'w') as f:
            f.write("corrupted data")
        
        # Should handle corruption gracefully
        result = basic_cache.get(text)
        assert result is None
        
        # Corrupted file should be removed
        assert not cache_path.exists()
    
    def test_lru_ordering_maintained(self, temp_cache_dir, sample_embeddings):
        """
        Test that LRU ordering works correctly.
        
        Recently accessed items should stay in memory longer.
        """
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=2)
        embedding = sample_embeddings["small"]
        
        # Add two items
        cache.set("item1", embedding)
        cache.set("item2", embedding)
        
        # Access item1 to make it most recent
        cache.get("item1")
        
        # Add item3 - should evict item2, keep item1
        cache.set("item3", embedding)
        
        # item1 should still be in memory (recently accessed)
        key1 = cache._get_cache_key("item1")
        assert key1 in cache.memory_cache
        
        # item2 should be evicted
        key2 = cache._get_cache_key("item2")
        assert key2 not in cache.memory_cache
    
    def test_statistics_updated_correctly(self, basic_cache, sample_embeddings):
        """
        Test that hit/miss stats reflect actual usage.
        
        Performance monitoring depends on accurate stats.
        """
        embedding = sample_embeddings["medium"]
        
        initial_stats = basic_cache.get_stats()
        
        # Generate some hits and misses
        basic_cache.get("miss1")  # miss
        basic_cache.get("miss2")  # miss
        
        basic_cache.set("hit1", embedding)
        basic_cache.get("hit1")  # hit
        
        final_stats = basic_cache.get_stats()
        
        assert final_stats["misses"] == initial_stats["misses"] + 2
        assert final_stats["hits"] == initial_stats["hits"] + 1
        assert final_stats["total_requests"] > initial_stats["total_requests"]


class TestCacheStorage:
    """
    Test embedding storage functionality.
    
    Storage must be reliable and thread-safe.
    """
    
    def test_stores_embedding_to_memory_and_disk(self, basic_cache, sample_embeddings):
        """
        Test that both cache layers are populated on storage.
        """
        text = "storage test"
        embedding = sample_embeddings["large"]
        
        basic_cache.set(text, embedding)
        
        # Should be in memory cache
        cache_key = basic_cache._get_cache_key(text)
        assert cache_key in basic_cache.memory_cache
        
        # Should be on disk
        cache_path = basic_cache._get_cache_path(cache_key)
        assert cache_path.exists()
        
        # Disk file should contain correct data
        disk_embedding = np.load(cache_path)
        np.testing.assert_array_equal(disk_embedding, embedding)
    
    def test_creates_cache_directory_structure(self, temp_cache_dir, sample_embeddings):
        """
        Test that subdirectories are created as needed.
        """
        cache = EmbeddingCache(str(temp_cache_dir))
        embedding = sample_embeddings["medium"]
        
        # Use text that will create specific subdirectory
        text = "directory test"
        cache_key = cache._get_cache_key(text)
        expected_subdir = cache.cache_dir / cache_key[:2]
        
        cache.set(text, embedding)
        
        # Subdirectory should be created
        assert expected_subdir.exists()
        assert expected_subdir.is_dir()
    
    def test_overwrites_existing_cache_entry(self, basic_cache, sample_embeddings):
        """
        Test that same text gets updated embedding.
        """
        text = "overwrite test"
        embedding1 = sample_embeddings["small"]
        embedding2 = sample_embeddings["large"]
        
        # Store initial embedding
        basic_cache.set(text, embedding1)
        result1 = basic_cache.get(text)
        np.testing.assert_array_equal(result1, embedding1)
        
        # Overwrite with different embedding
        basic_cache.set(text, embedding2)
        result2 = basic_cache.get(text)
        np.testing.assert_array_equal(result2, embedding2)
    
    def test_maintains_numpy_array_format(self, basic_cache, sample_embeddings):
        """
        Test that stored data format is correct.
        
        Format consistency is critical for retrieval.
        """
        text = "format test"
        embedding = sample_embeddings["medium"]
        
        basic_cache.set(text, embedding)
        result = basic_cache.get(text)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == embedding.dtype
        assert result.shape == embedding.shape


class TestCacheManagement:
    """
    Test cache lifecycle and maintenance operations.
    """
    
    def test_clear_memory_cache_removes_all_items(self, populated_cache):
        """
        Test that memory cache clearing works completely.
        """
        # Ensure cache has items
        assert len(populated_cache.memory_cache) > 0
        
        populated_cache.clear_memory_cache()
        
        assert len(populated_cache.memory_cache) == 0
    
    def test_clear_disk_cache_removes_all_files(self, populated_cache):
        """
        Test that disk cache clearing works completely.
        """
        # Ensure disk cache has files
        cache_files = list(populated_cache.cache_dir.rglob("*.npy"))
        assert len(cache_files) > 0
        
        populated_cache.clear_disk_cache()
        
        # All cache files should be gone
        cache_files_after = list(populated_cache.cache_dir.rglob("*.npy"))
        assert len(cache_files_after) == 0
        
        # Directory should still exist
        assert populated_cache.cache_dir.exists()
    
    def test_get_stats_provides_accurate_metrics(self, basic_cache, sample_embeddings):
        """
        Test that statistics reflect actual cache state.
        """
        embedding = sample_embeddings["small"]
        
        # Generate known activity
        basic_cache.get("miss1")  # miss
        basic_cache.set("item1", embedding)
        basic_cache.get("item1")  # hit
        
        stats = basic_cache.get_stats()
        
        assert stats["misses"] >= 1
        assert stats["hits"] >= 1
        assert stats["total_requests"] >= 2
        assert 0 <= stats["hit_rate"] <= 1
        assert stats["memory_cache_size"] >= 0
    
    def test_save_stats_persists_metrics(self, basic_cache, sample_embeddings):
        """
        Test that statistics can be saved for analysis.
        """
        embedding = sample_embeddings["medium"]
        
        # Generate some activity
        basic_cache.set("stats_test", embedding)
        basic_cache.get("stats_test")
        
        # Save stats
        basic_cache.save_stats()
        
        # Stats file should exist
        stats_file = basic_cache.cache_dir / "cache_stats.json"
        assert stats_file.exists()
        
        # Should contain valid JSON
        import json
        with open(stats_file) as f:
            saved_stats = json.load(f)
        
        assert "hit_rate" in saved_stats
        assert "total_requests" in saved_stats
    
    def test_hit_rate_calculation_accuracy(self, basic_cache, sample_embeddings):
        """
        Test that hit rate calculation is mathematically correct.
        """
        embedding = sample_embeddings["small"]
        
        # Start fresh
        basic_cache.stats = {"hits": 0, "misses": 0, "memory_hits": 0, "disk_hits": 0}
        
        # Generate known pattern: 2 misses, 3 hits
        basic_cache.get("miss1")  # miss
        basic_cache.get("miss2")  # miss
        
        basic_cache.set("hit1", embedding)
        basic_cache.set("hit2", embedding)
        basic_cache.set("hit3", embedding)
        
        basic_cache.get("hit1")  # hit
        basic_cache.get("hit2")  # hit
        basic_cache.get("hit3")  # hit
        
        stats = basic_cache.get_stats()
        
        expected_hit_rate = 3 / (3 + 2)  # 3 hits / 5 total
        assert abs(stats["hit_rate"] - expected_hit_rate) < 0.01


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

class TestThreadSafety:
    """
    Test concurrent access patterns.
    
    Critical for production multi-threaded environments.
    """
    
    def test_concurrent_get_operations(self, populated_cache, sample_texts):
        """
        Test multiple threads reading simultaneously.
        
        Simulates real production load.
        """
        def get_embedding(text):
            return populated_cache.get(text)
        
        texts = list(sample_texts.values())[:3]  # Use subset for performance
        
        # Run concurrent gets
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_embedding, text) for text in texts * 5]
            results = [future.result() for future in as_completed(futures)]
        
        # All operations should complete without errors
        assert len(results) == 15
        # Some should be successful (non-None)
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) > 0
    
    def test_concurrent_set_operations(self, temp_cache_dir, sample_embeddings):
        """
        Test multiple threads writing simultaneously.
        """
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=20)
        embedding = sample_embeddings["medium"]
        
        def set_embedding(i):
            text = f"concurrent_text_{i}"
            cache.set(text, embedding)
            return text
        
        # Run concurrent sets
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(set_embedding, i) for i in range(20)]
            texts = [future.result() for future in as_completed(futures)]
        
        # All operations should complete
        assert len(texts) == 20
        
        # All embeddings should be retrievable
        for text in texts:
            result = cache.get(text)
            assert result is not None
            np.testing.assert_array_equal(result, embedding)
    
    def test_mixed_read_write_operations(self, temp_cache_dir, sample_embeddings):
        """
        Test realistic mixed access patterns with data integrity verification.
        
        Most production workloads mix reads and writes.
        """
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=10)
        embedding = sample_embeddings["small"]
        
        # Pre-populate some data
        expected_data = {}
        for i in range(5):
            text = f"existing_{i}"
            cache.set(text, embedding)
            expected_data[text] = embedding
        
        def mixed_operation(i):
            if i % 2 == 0:
                # Write operation - verify data integrity
                text = f"new_{i}"
                cache.set(text, embedding)
                
                # Verify write succeeded with correct data
                result = cache.get(text)
                if result is not None:
                    np.testing.assert_array_equal(result, embedding)
                    return "write_success"
                else:
                    return "write_failed"
            else:
                # Read operation - verify data integrity
                text = f"existing_{i % 5}"
                result = cache.get(text)
                
                if result is not None:
                    # Verify data integrity
                    np.testing.assert_array_equal(result, expected_data[text])
                    return "read_success"
                else:
                    return "read_miss"
        
        # Run mixed operations
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(mixed_operation, i) for i in range(20)]
            operations = [future.result() for future in as_completed(futures)]
        
        # All operations should complete successfully
        assert len(operations) == 20
        assert "write_success" in operations
        assert "read_success" in operations
        
        # No operations should fail (data integrity maintained)
        assert "write_failed" not in operations
    
    def test_statistics_consistency_under_load(self, temp_cache_dir, sample_embeddings):
        """
        Test that stats remain accurate under concurrent access.
        """
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=10)
        embedding = sample_embeddings["medium"]
        
        def cache_operation(i):
            # Mix of hits and misses
            cache.get(f"miss_{i}")  # miss
            cache.set(f"item_{i}", embedding)
            cache.get(f"item_{i}")  # hit
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(cache_operation, i) for i in range(10)]
            [future.result() for future in as_completed(futures)]
        
        stats = cache.get_stats()
        
        # Stats should be consistent
        assert stats["total_requests"] == stats["hits"] + stats["misses"]
        assert stats["hits"] >= 10  # At least one hit per operation
        assert stats["misses"] >= 10  # At least one miss per operation
    
    def test_no_data_corruption_under_concurrency(self, temp_cache_dir, sample_embeddings):
        """
        Test that embeddings are retrieved correctly under load.
        
        Data integrity is paramount.
        """
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=15)
        embeddings = sample_embeddings
        
        def store_and_retrieve(embedding_type, index):
            text = f"{embedding_type}_{index}"
            embedding = embeddings[embedding_type]
            
            # Store
            cache.set(text, embedding)
            
            # Retrieve and verify
            result = cache.get(text)
            if result is not None:
                np.testing.assert_array_equal(result, embedding)
                return True
            return False
        
        # Run concurrent operations with different embedding sizes
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for embedding_type in ["small", "medium", "large"]:
                for i in range(5):
                    future = executor.submit(store_and_retrieve, embedding_type, i)
                    futures.append(future)
            
            results = [future.result() for future in as_completed(futures)]
        
        # All operations should succeed
        assert all(results)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestCacheIntegration:
    """
    End-to-end workflow tests.
    
    These ensure the complete system works together correctly.
    """
    
    def test_complete_cache_lifecycle(self, temp_cache_dir, sample_embeddings):
        """
        Test store -> retrieve -> evict -> reload from disk.
        
        Simulates complete real usage pattern.
        """
        # Small cache for testing eviction
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=2)
        embeddings = sample_embeddings
        
        # Store embeddings
        cache.set("doc1", embeddings["small"])
        cache.set("doc2", embeddings["medium"])
        
        # Verify memory storage
        assert cache.get("doc1") is not None
        assert cache.get("doc2") is not None
        
        # Force eviction by adding more items
        cache.set("doc3", embeddings["large"])
        cache.set("doc4", embeddings["small"])
        
        # Some items should be evicted from memory but available on disk
        # Clear memory to test disk retrieval
        cache.clear_memory_cache()
        
        # Should reload from disk
        result1 = cache.get("doc1")
        result2 = cache.get("doc2")
        
        assert result1 is not None or result2 is not None  # At least one should be retrievable
    
    def test_cache_persistence_across_restarts(self, temp_cache_dir, sample_embeddings):
        """
        Test that disk cache survives cache object recreation.
        
        Server restarts shouldn't lose cache.
        """
        embedding = sample_embeddings["medium"]
        
        # Create cache and store data
        cache1 = EmbeddingCache(str(temp_cache_dir))
        cache1.set("persistent_doc", embedding)
        
        # Simulate restart by creating new cache instance
        cache2 = EmbeddingCache(str(temp_cache_dir))
        
        # Should retrieve from disk
        result = cache2.get("persistent_doc")
        assert result is not None
        np.testing.assert_array_equal(result, embedding)
    
    def test_memory_overflow_triggers_disk_usage(self, temp_cache_dir, sample_embeddings):
        """
        Test that LRU eviction + disk retrieval works together.
        """
        # Very small memory cache
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=2)
        embedding = sample_embeddings["small"]
        
        # Fill memory cache
        cache.set("mem1", embedding)
        cache.set("mem2", embedding)
        
        # Add more items to force eviction
        cache.set("mem3", embedding)
        cache.set("mem4", embedding)
        
        # Memory cache should be full
        assert len(cache.memory_cache) == 2
        
        # But all items should be retrievable (some from disk)
        results = []
        for key in ["mem1", "mem2", "mem3", "mem4"]:
            result = cache.get(key)
            if result is not None:
                results.append(key)
        
        # All items should be retrievable
        assert len(results) == 4
    
    def test_large_embedding_handling(self, basic_cache):
        """
        Test system handles realistically large embeddings.
        
        Some embeddings are very large vectors.
        """
        # Large embedding (simulates high-dimensional models)
        large_embedding = np.random.rand(4096).astype(np.float32)
        text = "large embedding test"
        
        # Should handle without errors
        basic_cache.set(text, large_embedding)
        result = basic_cache.get(text)
        
        assert result is not None
        assert result.shape == large_embedding.shape
        np.testing.assert_array_equal(result, large_embedding)


# =============================================================================
# NORMAL PERFORMANCE TESTS
# =============================================================================

class TestNormalPerformance:
    """
    Performance tests with realistic expectations that should pass.
    
    These establish baseline performance characteristics.
    """
    
    def test_memory_cache_hit_performance(self, populated_cache, sample_texts):
        """
        Test that memory hits are significantly faster than computation.
        """
        # Use a text that should be in memory cache
        text = next(iter(sample_texts.values()))
        
        # Warm up
        populated_cache.get(text)
        
        # Measure memory hit performance
        start_time = time.time()
        for _ in range(1000):
            result = populated_cache.get(text)
        elapsed = time.time() - start_time
        
        # Should be very fast (< 0.1 seconds for 1000 hits)
        assert elapsed < 0.1
    
    def test_concurrent_access_performance(self, temp_cache_dir, sample_embeddings):
        """
        Test that threading doesn't create major bottlenecks.
        """
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=20)
        embedding = sample_embeddings["small"]
        
        # Pre-populate
        for i in range(10):
            cache.set(f"perf_test_{i}", embedding)
        
        def access_cache(iterations):
            for i in range(iterations):
                cache.get(f"perf_test_{i % 10}")
        
        # Measure concurrent performance
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(access_cache, 50) for _ in range(4)]
            [future.result() for future in as_completed(futures)]
        elapsed = time.time() - start_time
        
        # 800 total operations should complete quickly
        assert elapsed < 2.0
    
    def test_cache_overhead_is_reasonable(self, basic_cache, sample_embeddings):
        """
        Test that memory/disk overhead is acceptable.
        """
        embedding = sample_embeddings["large"]  # Large embedding for overhead test
        text = "overhead test"
        
        # Measure storage overhead
        embedding_size = embedding.nbytes
        
        basic_cache.set(text, embedding)
        
        # Check disk file size
        cache_key = basic_cache._get_cache_key(text)
        cache_path = basic_cache._get_cache_path(cache_key)
        disk_size = cache_path.stat().st_size
        
        # Overhead should be minimal (< 10% for large embeddings)
        overhead_ratio = (disk_size - embedding_size) / embedding_size
        assert overhead_ratio < 0.1
    
    def test_cache_efficiency_under_realistic_load(self, temp_cache_dir, sample_embeddings):
        """
        Test performance characteristics match expectations.
        
        Cache should actually improve performance.
        """
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=50)
        embedding = sample_embeddings["medium"]
        
        # Pre-populate cache
        texts = [f"doc_{i}" for i in range(30)]
        for text in texts:
            cache.set(text, embedding)
        
        # Measure cache hit performance
        start_time = time.time()
        for _ in range(100):
            # Mix of cache hits and misses
            cache.get(texts[0])  # Should be fast (memory hit)
            cache.get("nonexistent")  # Miss
        
        total_time = time.time() - start_time
        
        # Should complete quickly (< 1 second for 200 operations)
        assert total_time < 1.0
        
        # Hit rate should be reasonable
        stats = cache.get_stats()
        assert stats["hit_rate"] > 0.4  # At least 40% hit rate


# =============================================================================
# STRESS PERFORMANCE TESTS
# =============================================================================

class TestStressPerformance:
    """
    Stress tests that push system limits - may fail, shows boundaries.
    
    These tests help us understand where the system breaks down.
    """
    
    def test_massive_concurrent_load(self, temp_cache_dir, sample_embeddings):
        """
        Test system under heavy concurrent load to find breaking point.
        
        Target: ~5GB of concurrent operations to stress threading.
        """
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=100)
        # Use large embedding: 1536 * 4 bytes = ~6KB each
        embedding = sample_embeddings["large"]
        
        def stress_operation(thread_id):
            operations_completed = 0
            errors = []
            # 200 operations per thread * 50 threads = 10,000 operations = ~60MB total
            for i in range(200):
                try:
                    cache.set(f"stress_{thread_id}_{i}", embedding)
                    result = cache.get(f"stress_{thread_id}_{i}")
                    if result is not None:
                        np.testing.assert_array_equal(result, embedding)
                        operations_completed += 1
                    else:
                        errors.append("retrieval_failed")
                except Exception as e:
                    errors.append(str(e))
                    if len(errors) > 10:  # Stop if too many errors
                        break
            return operations_completed, len(errors)
        
        # Heavy concurrent load: 50 threads doing 200 operations each
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(stress_operation, i) for i in range(50)]
            results = [future.result() for future in as_completed(futures)]
        elapsed = time.time() - start_time
        
        # Analyze results
        total_operations = sum(r[0] for r in results)
        total_errors = sum(r[1] for r in results)
        target_operations = 50 * 200
        success_rate = total_operations / target_operations if target_operations > 0 else 0
        
        print(f"Concurrent stress: {total_operations}/{target_operations} operations in {elapsed:.2f}s")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Total errors: {total_errors}")
        print(f"Operations per second: {total_operations/elapsed:.0f}")
        print(f"Data processed: {total_operations * 6 / 1024:.1f} MB")
        
        # Expect some degradation under heavy concurrent load
        assert success_rate > 0.7, f"System collapsed under load: {success_rate:.2%} success rate"
        if success_rate > 0.98:
            print("WARNING: No degradation found - consider increasing stress level")
    
    def test_memory_pressure_handling(self, temp_cache_dir):
        """
        Test behavior under memory pressure to find limits.
        
        Target: ~3GB of memory pressure to find breaking point.
        """
        # Large embeddings: 50K floats * 4 bytes = ~200KB each
        large_embedding = np.random.rand(50000).astype(np.float32)
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=2000)  # 2000 item memory cache
        
        successful_stores = 0
        memory_errors = 0
        other_errors = 0
        
        # Try to store 15,000 large embeddings = ~3GB total
        # Memory cache will hold 2000 = ~400MB, rest goes to disk
        for i in range(15000):
            try:
                cache.set(f"memory_pressure_{i}", large_embedding)
                
                # Verify we can retrieve it (tests both memory and disk)
                result = cache.get(f"memory_pressure_{i}")
                if result is not None:
                    successful_stores += 1
                else:
                    other_errors += 1
                    
                # Progress indicator
                if i % 1000 == 0:
                    print(f"Progress: {i}/15000 - Memory cache: {len(cache.memory_cache)}")
                    
            except MemoryError:
                memory_errors += 1
                print(f"MemoryError at iteration {i}")
                break
            except Exception as e:
                other_errors += 1
                print(f"Other error at iteration {i}: {e}")
                if successful_stores < 500:  # If we fail very early, something's wrong
                    break
        
        total_attempted = successful_stores + memory_errors + other_errors
        success_rate = successful_stores / total_attempted if total_attempted > 0 else 0
        
        print(f"Memory pressure test: {successful_stores}/{total_attempted} successful")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Memory errors: {memory_errors}")
        print(f"Other errors: {other_errors}")
        print(f"Final memory cache size: {len(cache.memory_cache)}")
        print(f"Estimated data stored: {successful_stores * 200 / 1024:.1f} MB")
        
        # We expect reasonable performance until memory limits
        assert successful_stores > 1000, f"Failed too early - only stored {successful_stores} embeddings"
        if successful_stores == 15000:
            print("WARNING: No memory pressure encountered - consider larger test")
    
    def test_disk_space_pressure(self, temp_cache_dir, sample_embeddings):
        """
        Test disk storage with large volume to find I/O limits.
        
        Target: ~10GB of disk operations to stress filesystem.
        """
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=20)  # Small memory cache
        embedding = sample_embeddings["large"]  # ~6KB each
        
        successful_disk_stores = 0
        disk_errors = 0
        
        # Target: ~10GB = 1.6M operations of 6KB each
        # Use 100K operations for reasonable test time = ~600MB
        target_operations = 100000
        
        start_time = time.time()
        for i in range(target_operations):
            try:
                cache.set(f"disk_pressure_{i}", embedding)
                
                # Clear memory periodically to force disk storage
                if i % 50 == 0:
                    cache.clear_memory_cache()
                
                # Verify disk storage works
                if i % 1000 == 0:
                    result = cache.get(f"disk_pressure_{i}")
                    if result is None:
                        disk_errors += 1
                
                successful_disk_stores += 1
                
                # Progress indicator
                if i % 10000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Progress: {i}/{target_operations} - {i/elapsed:.0f} ops/sec")
                    
            except OSError as e:  # Disk full or other disk issues
                print(f"Disk error at iteration {i}: {e}")
                break
            except Exception as e:
                print(f"Other error at iteration {i}: {e}")
                if successful_disk_stores < 1000:  # Early failure
                    break
        
        elapsed = time.time() - start_time
        success_rate = successful_disk_stores / target_operations
        
        # Check actual disk usage
        cache_files = list(cache.cache_dir.rglob("*.npy"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        print(f"Disk pressure test: {successful_disk_stores}/{target_operations} successful")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Disk errors: {disk_errors}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Operations per second: {successful_disk_stores/elapsed:.0f}")
        print(f"Total disk usage: {total_size / (1024*1024):.1f} MB")
        print(f"Files created: {len(cache_files)}")
        
        # Expect reasonable disk performance
        assert successful_disk_stores > 10000, f"Disk operations failed too early: {successful_disk_stores}"
        if successful_disk_stores == target_operations:
            print("WARNING: No disk pressure encountered - consider larger test")
    
    def test_extreme_embedding_sizes(self, temp_cache_dir):
        """
        Test progressively larger embeddings to find size limits.
        
        Target: Find the maximum single embedding size we can handle.
        """
        cache = EmbeddingCache(str(temp_cache_dir), memory_cache_size=3)
        
        # Test increasingly large embeddings - up to ~12GB total across all tests
        sizes = [
            100000,   # ~400KB
            500000,   # ~2MB  
            1000000,  # ~4MB
            2000000,  # ~8MB
            5000000,  # ~20MB
            10000000, # ~40MB
            20000000, # ~80MB
            50000000, # ~200MB
            100000000 # ~400MB - this should definitely fail!
        ]
        
        max_successful_size = 0
        failure_reason = None
        total_data_processed = 0
        
        for size in sizes:
            try:
                print(f"Testing {size:,}-dimension embedding ({size*4/(1024*1024):.1f} MB)...")
                
                # Create large embedding
                start_time = time.time()
                huge_embedding = np.random.rand(size).astype(np.float32)
                create_time = time.time() - start_time
                
                # Try to store it
                start_time = time.time()
                cache.set(f"huge_embedding_{size}", huge_embedding)
                store_time = time.time() - start_time
                
                # Try to retrieve it
                start_time = time.time()
                result = cache.get(f"huge_embedding_{size}")
                retrieve_time = time.time() - start_time
                
                if result is not None and len(result) == size:
                    max_successful_size = size
                    data_size_mb = size * 4 / (1024 * 1024)
                    total_data_processed += data_size_mb
                    print(f"✓ SUCCESS: {size:,}-dim ({data_size_mb:.1f} MB)")
                    print(f"  Create: {create_time:.2f}s, Store: {store_time:.2f}s, Retrieve: {retrieve_time:.2f}s")
                    
                    # Stop if we've processed enough data
                    if total_data_processed > 5000:  # 5GB limit
                        print("Stopping - processed enough data")
                        break
                else:
                    failure_reason = f"Retrieval failed for {size:,}-dimension embedding"
                    break
                    
            except MemoryError as e:
                failure_reason = f"MemoryError at {size:,}-dimension: {e}"
                break
            except Exception as e:
                failure_reason = f"Exception at {size:,}-dimension: {e}"
                break
        
        max_size_mb = max_successful_size * 4 / (1024 * 1024)
        
        print(f"\nEmbedding size test results:")
        print(f"Maximum successful size: {max_successful_size:,} dimensions ({max_size_mb:.1f} MB)")
        print(f"Total data processed: {total_data_processed:.1f} MB")
        print(f"Failure reason: {failure_reason}")
        
        # We expect to handle reasonable sizes but fail at extreme sizes
        assert max_successful_size >= 100000, f"Failed at small embedding size: {max_successful_size:,}"
        if max_successful_size >= 100000000:
            print("WARNING: No size limits found - handled 400MB+ embeddings!")


# =============================================================================
# REAL-WORLD SCENARIO TESTS
# =============================================================================

class TestRealWorldScenarios:
    """
    Tests based on actual usage patterns.
    
    These represent how the cache is actually used in production.
    """
    
    def test_document_embedding_workflow(self, temp_cache_dir):
        """
        Test embedding documents for similarity search.
        
        Common use case pattern.
        """
        cache = EmbeddingCache(str(temp_cache_dir))
        
        # Simulate document embedding workflow
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret visual information."
        ]
        
        # Simulate embedding generation (using random vectors)
        np.random.seed(42)
        embeddings = [np.random.rand(512).astype(np.float32) for _ in documents]
        
        # Store embeddings
        for doc, emb in zip(documents, embeddings):
            cache.set(doc, emb)
        
        # Simulate similarity search (retrieve all embeddings)
        retrieved_embeddings = []
        for doc in documents:
            emb = cache.get(doc)
            assert emb is not None
            retrieved_embeddings.append(emb)
        
        # All embeddings should be retrievable and correct
        assert len(retrieved_embeddings) == len(documents)
        for original, retrieved in zip(embeddings, retrieved_embeddings):
            np.testing.assert_array_equal(original, retrieved)
    
    def test_repeated_text_optimization(self, basic_cache, sample_embeddings):
        """
        Test that frequently accessed text gets performance benefit.
        """
        embedding = sample_embeddings["medium"]
        frequent_text = "This text is accessed frequently"
        rare_text = "This text is accessed rarely"
        
        # Store both
        basic_cache.set(frequent_text, embedding)
        basic_cache.set(rare_text, embedding)
        
        # Access frequent text multiple times
        for _ in range(10):
            basic_cache.get(frequent_text)
        
        # Access rare text once
        basic_cache.get(rare_text)
        
        # Frequent text should be in memory cache
        frequent_key = basic_cache._get_cache_key(frequent_text)
        assert frequent_key in basic_cache.memory_cache
        
        # Performance benefit should be measurable
        start_time = time.time()
        for _ in range(100):
            basic_cache.get(frequent_text)
        frequent_time = time.time() - start_time
        
        # Should be very fast due to memory caching
        assert frequent_time < 0.05
    
    def test_unicode_text_embedding_workflow(self, basic_cache, sample_embeddings):
        """
        Test international text handling end-to-end.
        
        Production systems must support international users.
        """
        embedding = sample_embeddings["small"]
        
        # International text samples
        international_texts = [
            "Hello world",  # English
            "Bonjour le monde",  # French
            "Hola mundo",  # Spanish
            "こんにちは世界",  # Japanese
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "你好世界",  # Chinese
            "🌍 🌎 🌏"  # Emoji
        ]
        
        # Store all international text
        for text in international_texts:
            basic_cache.set(text, embedding)
        
        # Retrieve all international text
        for text in international_texts:
            result = basic_cache.get(text)
            assert result is not None
            np.testing.assert_array_equal(result, embedding)
        
        # Cache should handle Unicode correctly
        stats = basic_cache.get_stats()
        assert stats["hits"] >= len(international_texts)