"""
Embedding utilities including caching for efficient vector operations.
"""

import os
import json
import hashlib
import numpy as np
from typing import Optional, Dict, Any
import logging
from pathlib import Path
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Thread-safe cache for embeddings to avoid recomputation.
    
    Uses file-based caching with in-memory LRU cache for frequently
    accessed embeddings.
    """
    
    def __init__(self, cache_dir: str, memory_cache_size: int = 1000):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory for persistent cache storage
            memory_cache_size: Number of embeddings to keep in memory
        """
        self.cache_dir = Path(cache_dir) / "embeddings_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory LRU cache
        self.memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.memory_cache_size = memory_cache_size
        self.lock = threading.Lock()
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "disk_hits": 0
        }
        
        logger.info(f"Initialized embedding cache at {self.cache_dir}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text using SHA-256."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        # Use subdirectories to avoid too many files in one directory
        return self.cache_dir / cache_key[:2] / f"{cache_key}.npy"
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding if exists.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        cache_key = self._get_cache_key(text)
        
        with self.lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                # Move to end (most recently used)
                self.memory_cache.move_to_end(cache_key)
                self.stats["hits"] += 1
                self.stats["memory_hits"] += 1
                return self.memory_cache[cache_key].copy()
        
        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                embedding = np.load(cache_path)
                
                # Add to memory cache
                with self.lock:
                    self._add_to_memory_cache(cache_key, embedding)
                    self.stats["hits"] += 1
                    self.stats["disk_hits"] += 1
                
                return embedding
                
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
                # Remove corrupted cache file
                cache_path.unlink(missing_ok=True)
        
        with self.lock:
            self.stats["misses"] += 1
        
        return None
    
    def set(self, text: str, embedding: np.ndarray) -> None:
        """
        Cache an embedding.
        
        Args:
            text: Text that was embedded
            embedding: The embedding vector
        """
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        # Ensure directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save to disk
            np.save(cache_path, embedding)
            
            # Add to memory cache
            with self.lock:
                self._add_to_memory_cache(cache_key, embedding)
                
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def _add_to_memory_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """Add embedding to memory cache with LRU eviction."""
        # Remove oldest if at capacity
        if len(self.memory_cache) >= self.memory_cache_size:
            self.memory_cache.popitem(last=False)
        
        self.memory_cache[cache_key] = embedding.copy()
    
    def clear_memory_cache(self) -> None:
        """Clear the in-memory cache."""
        with self.lock:
            self.memory_cache.clear()
            logger.info("Cleared in-memory embedding cache")
    
    def clear_disk_cache(self) -> None:
        """Clear the disk cache."""
        import shutil
        try:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared disk embedding cache")
        except Exception as e:
            logger.error(f"Failed to clear disk cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_hits = self.stats["hits"]
            total_requests = total_hits + self.stats["misses"]
            hit_rate = total_hits / total_requests if total_requests > 0 else 0
            
            return {
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "memory_cache_size": len(self.memory_cache),
                "disk_cache_size": sum(1 for _ in self.cache_dir.rglob("*.npy")),
                **self.stats
            }
    
    def save_stats(self) -> None:
        """Save cache statistics to file."""
        stats_path = self.cache_dir / "cache_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.get_stats(), f, indent=2)