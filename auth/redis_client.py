"""
Redis client for authentication system.
Handles async Redis connections for refresh tokens and rate limiting.
"""

import logging
from typing import Optional, Dict, Any, List
import redis.asyncio as redis
from .config import config

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis client for authentication operations."""
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self._connecting = False
    
    async def connect(self):
        """Connect to Redis."""
        if self.redis is not None:
            return
            
        if self._connecting:
            # Another connection attempt in progress
            return
        
        try:
            self._connecting = True
            self.redis = redis.from_url(
                config.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                health_check_interval=30,
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            await self.redis.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis = None
            raise
        finally:
            self._connecting = False
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.aclose()
            self.redis = None
            logger.info("Redis connection closed")
    
    async def _ensure_connected(self):
        """Ensure Redis connection is established."""
        if self.redis is None:
            await self.connect()
        
        # Test connection health
        try:
            await self.redis.ping()
        except Exception:
            logger.warning("Redis connection lost, reconnecting...")
            self.redis = None
            await self.connect()
    
    async def set_with_expiry(self, key: str, value: str, expiry_seconds: int) -> bool:
        """Set a key with expiration."""
        try:
            await self._ensure_connected()
            await self.redis.setex(key, expiry_seconds, value)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            # Fail closed - raise exception to indicate operation failed
            raise Exception(f"Redis operation failed: {e}")
    
    async def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        try:
            await self._ensure_connected()
            return await self.redis.get(key)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            # Fail closed - raise exception for critical operations
            raise Exception(f"Redis operation failed: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete a key."""
        try:
            await self._ensure_connected()
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            # Fail closed - raise exception for critical operations
            raise Exception(f"Redis operation failed: {e}")
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        try:
            await self._ensure_connected()
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                result = await self.redis.delete(*keys)
                return result
            return 0
        except Exception as e:
            logger.error(f"Redis delete pattern error: {e}")
            # Fail closed - raise exception for critical operations
            raise Exception(f"Redis operation failed: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            await self._ensure_connected()
            result = await self.redis.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            # Fail closed - raise exception for critical operations
            raise Exception(f"Redis operation failed: {e}")
    
    async def list_keys(self, pattern: str) -> List[str]:
        """List all keys matching pattern."""
        try:
            await self._ensure_connected()
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            return keys
        except Exception as e:
            logger.error(f"Redis list keys error: {e}")
            # Fail closed - raise exception for critical operations
            raise Exception(f"Redis operation failed: {e}")
    
    async def increment_with_expiry(self, key: str, expiry_seconds: int) -> int:
        """Increment a counter with expiry (for rate limiting)."""
        try:
            await self._ensure_connected()
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, expiry_seconds)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.error(f"Redis increment error: {e}")
            # Fail closed - raise exception for rate limiting operations
            raise Exception(f"Redis operation failed: {e}")


# Global Redis client instance
redis_client = RedisClient()


async def get_redis() -> RedisClient:
    """Get Redis client instance."""
    await redis_client.connect()
    return redis_client