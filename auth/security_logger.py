"""
Security event logging with sanitized data.
"""

import logging
from typing import Dict, Any, Optional
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


class SecurityLogger:
    """Log security events without exposing sensitive data."""

    def __init__(self):
        self.logger = logger

    def _sanitize_email(self, email: str) -> str:
        """Sanitize email showing partial info for pattern detection."""
        if not email or '@' not in email:
            return "***"

        local, domain = email.split('@', 1)

        # Show first 2 and last 3 chars of local part
        if len(local) <= 5:
            # Short email, just mask middle
            if len(local) <= 2:
                sanitized_local = "*" * len(local)
            else:
                sanitized_local = local[0] + "*" * (len(local) - 2) + local[-1]
        else:
            # Show first 2 and last 3 chars
            sanitized_local = local[:2] + "*" * (len(local) - 5) + local[-3:]

        return f"{sanitized_local}@{domain}"

    def log_event(
        self,
        event_type: str,
        success: bool = True,
        user_id: str = None,
        email: str = None,
        ip_address: str = None,
        user_agent: str = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log a security event with user tracing capability.

        Args:
            event_type: Type of security event (e.g., 'auth.magic_link_request')
            success: Whether the event was successful
            user_id: User UUID for tracing malicious actors
            email: Email address (will be sanitized)
            ip_address: Client IP address
            user_agent: Client user agent string
            details: Additional event details
        """
        log_data = {
            "timestamp": utc_now().isoformat(),
            "event": event_type,
            "success": success,
            "user_id": user_id,
            "email_sanitized": self._sanitize_email(email) if email else None,
            "ip": ip_address,
            "user_agent": user_agent[:100] if user_agent else None,
            "details": details or {}
        }

        # Remove None values for cleaner logs
        log_data = {k: v for k, v in log_data.items() if v is not None}

        if success:
            self.logger.info(f"Security event: {event_type}", extra=log_data)
        else:
            self.logger.warning(f"Security event failed: {event_type}", extra=log_data)

        return log_data


# Singleton instance
security_logger = SecurityLogger()
