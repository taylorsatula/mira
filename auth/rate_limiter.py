"""
Valkey-backed rate limiter for authentication requests.
"""

import hashlib
import logging
from typing import Tuple, Optional
from clients.valkey_client import get_valkey
from .config import config

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter using Valkey for distributed rate limiting."""

    def __init__(self, max_requests: int = None, window_seconds: int = None):
        self.max_requests = max_requests or config.RATE_LIMIT_REQUESTS
        self.window_seconds = window_seconds or config.RATE_LIMIT_WINDOW
        self.key_prefix = "rate_limit:"

        # IP-based rate limiting (more permissive than email-based)
        self.ip_max_requests = 10
        self.ip_window_seconds = 300  # 5 minutes

    def _get_key(self, identifier: str, key_type: str = "email") -> str:
        """Get Valkey key for identifier."""
        return f"{self.key_prefix}{key_type}:{identifier}"

    def is_allowed(self, email: str, ip_address: Optional[str] = None) -> Tuple[bool, int]:
        """
        Check if request is allowed under rate limit.

        Implements dual rate limiting:
        - Email-based: 3 requests per 5 minutes (prevents spam to specific user)
        - IP-based: 10 requests per 5 minutes (prevents distributed email spam from one IP)

        Args:
            email: Email address to rate limit
            ip_address: Optional IP address for IP-based rate limiting

        Returns:
            Tuple of (allowed: bool, retry_after: int seconds)

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        valkey = get_valkey()

        # Check email-based rate limit
        email_key = self._get_key(email, "email")
        email_count = valkey.increment_with_expiry(email_key, self.window_seconds)

        if email_count > self.max_requests:
            # Email rate limit exceeded
            ttl = valkey.ttl(email_key)
            seconds_until_reset = ttl if ttl > 0 else self.window_seconds
            email_hash = hashlib.sha256(email.encode()).hexdigest()[:16]
            logger.warning(f"Rate limit exceeded for email {email_hash} (retry after {seconds_until_reset}s)")
            return False, seconds_until_reset

        # Check IP-based rate limit (if IP provided)
        if ip_address:
            ip_key = self._get_key(ip_address, "ip")
            ip_count = valkey.increment_with_expiry(ip_key, self.ip_window_seconds)

            if ip_count > self.ip_max_requests:
                # IP rate limit exceeded
                ttl = valkey.ttl(ip_key)
                seconds_until_reset = ttl if ttl > 0 else self.ip_window_seconds
                logger.warning(f"Rate limit exceeded for IP {ip_address} (retry after {seconds_until_reset}s)")
                return False, seconds_until_reset

        return True, 0

    def reset(self, email: str, ip_address: Optional[str] = None) -> bool:
        """
        Reset rate limit for email and optionally IP.

        Args:
            email: Email address to reset
            ip_address: Optional IP address to reset

        Returns:
            True if any key was deleted, False if no keys existed

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        valkey = get_valkey()

        # Reset email rate limit
        email_key = self._get_key(email, "email")
        deleted_count = valkey.delete(email_key)

        # Reset IP rate limit if provided
        if ip_address:
            ip_key = self._get_key(ip_address, "ip")
            deleted_count += valkey.delete(ip_key)

        if deleted_count > 0:
            email_hash = hashlib.sha256(email.encode()).hexdigest()[:16]
            logger.info(f"Rate limit reset for email {email_hash}")

        return deleted_count > 0
