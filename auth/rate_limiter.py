"""
Redis-backed distributed rate limiter.
"""

import time
import logging
from typing import Tuple
from .config import config
from .redis_client import get_redis

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis-backed sliding window rate limiter with automatic cleanup."""
    
    def __init__(self, max_requests: int = None, window_seconds: int = None):
        self.max_requests = max_requests or config.RATE_LIMIT_REQUESTS
        self.window_seconds = window_seconds or config.RATE_LIMIT_WINDOW
        self.key_prefix = "rate_limit:"
    
    def _get_redis_key(self, key: str) -> str:
        """Generate Redis key for rate limiting."""
        return f"{self.key_prefix}{key}"
    
    async def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        Check if request is allowed using Redis counter with sliding window.
        
        Args:
            key: Identifier (e.g., email, IP address)
            
        Returns:
            Tuple of (allowed, seconds_until_reset)
        """
        try:
            redis = await get_redis()
            
            # Use a simple counter approach for better performance
            counter_key = self._get_redis_key(key)
            
            # Get current count
            current_count = await redis.increment_with_expiry(counter_key, self.window_seconds)
            
            if current_count > self.max_requests:
                # Calculate time until reset
                ttl = await redis.redis.ttl(counter_key)
                seconds_until_reset = ttl if ttl > 0 else self.window_seconds
                return False, seconds_until_reset
            
            return True, 0
            
        except Exception as e:
            logger.error(f"Redis rate limiter error: {e}")
            # Fail closed for security - deny request if Redis is unavailable
            return False, self.window_seconds
    
    async def reset(self, key: str) -> bool:
        """Reset rate limit for a key."""
        try:
            redis = await get_redis()
            redis_key = self._get_redis_key(key)
            return await redis.delete(redis_key)
        except Exception as e:
            logger.error(f"Redis rate limiter reset error: {e}")
            return False