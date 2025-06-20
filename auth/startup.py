"""
Auth system startup and shutdown handlers.
"""

import logging
from .redis_client import redis_client
from .config import config

logger = logging.getLogger(__name__)


async def startup_auth_system():
    """Initialize auth system components."""
    try:
        # Validate configuration first - fail fast if env vars missing
        config.validate()
        logger.info("Auth configuration validated")
        
        # Initialize Redis connection
        await redis_client.connect()
        logger.info("Auth system initialized successfully")
        
    except ValueError as e:
        # Configuration validation failed
        logger.error(f"Auth configuration validation failed: {e}")
        raise
    except Exception as e:
        # Other initialization failures
        logger.error(f"Failed to initialize auth system: {e}")
        raise


async def shutdown_auth_system():
    """Cleanup auth system components."""
    try:
        # Close Redis connection
        await redis_client.disconnect()
        logger.info("Auth system shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during auth system shutdown: {e}")


# Background task to cleanup expired refresh tokens
async def cleanup_expired_tokens():
    """Background task to clean up expired refresh tokens."""
    try:
        redis = await redis_client.connect()
        if redis_client.redis:
            # This would run periodically to clean up expired tokens
            # For now, Redis handles expiration automatically
            pass
    except Exception as e:
        logger.error(f"Error cleaning up expired tokens: {e}")