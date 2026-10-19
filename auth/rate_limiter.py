"""Fixed-window rate limiting on Valkey for the pre-authentication surfaces.

`is_allowed` advances two independent windows per call — one keyed by the email
address, one by the client IP — so a single mailbox cannot be hammered through
many requests and a single host cannot spray many mailboxes. Both are Valkey
counters created with a TTL, so the window is shared by every worker and expires
without a sweeper; the retry hint handed back is the key's remaining TTL.

Infrastructure failure propagates to the caller. A Valkey outage must fail the
request that needed the limit, never admit it: a limiter that reads as "allowed"
because its store was unreachable is switched off exactly when the deployment is
already unhealthy.

`reset` has no caller anywhere in the tree. It is dead surface — do not build on
it, and deleting it is safe.
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
        self.ip_window_seconds = 300

    def _get_key(self, identifier: str, key_type: str = "email") -> str:
        """Get Valkey key for identifier."""
        return f"{self.key_prefix}{key_type}:{identifier}"

    def is_allowed(self, email: str, ip_address: Optional[str] = None) -> Tuple[bool, int]:
        """
        Check if request is allowed under rate limit.

        Implements dual rate limiting:
        - Email-based: `max_requests` increments per `window_seconds` (both from
          `auth/config.py`), preventing repeated sends to one address
        - IP-based: `ip_max_requests` per `ip_window_seconds`, preventing one
          host from spraying many addresses

        Args:
            email: Email address to rate limit
            ip_address: Optional IP address for IP-based rate limiting

        Returns:
            Tuple of (allowed: bool, retry_after: int seconds)

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        valkey = get_valkey()

        email_key = self._get_key(email, "email")
        email_count = valkey.increment_with_expiry(email_key, self.window_seconds)

        if email_count > self.max_requests:
            ttl = valkey.ttl(email_key)
            seconds_until_reset = ttl if ttl > 0 else self.window_seconds
            email_hash = hashlib.sha256(email.encode()).hexdigest()[:16]
            logger.warning(f"Rate limit exceeded for email {email_hash} (retry after {seconds_until_reset}s)")
            return False, seconds_until_reset

        if ip_address:
            ip_key = self._get_key(ip_address, "ip")
            ip_count = valkey.increment_with_expiry(ip_key, self.ip_window_seconds)

            if ip_count > self.ip_max_requests:
                ttl = valkey.ttl(ip_key)
                seconds_until_reset = ttl if ttl > 0 else self.ip_window_seconds
                logger.warning(f"Rate limit exceeded for IP {ip_address} (retry after {seconds_until_reset}s)")
                return False, seconds_until_reset

        return True, 0

    # ACCEPTED-RESIDUAL: dead surface, deliberately kept; no caller anywhere in the tree.
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

        email_key = self._get_key(email, "email")
        deleted_count = valkey.delete(email_key)

        if ip_address:
            ip_key = self._get_key(ip_address, "ip")
            deleted_count += valkey.delete(ip_key)

        if deleted_count > 0:
            email_hash = hashlib.sha256(email.encode()).hexdigest()[:16]
            logger.info(f"Rate limit reset for email {email_hash}")

        return deleted_count > 0
