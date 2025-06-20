"""
Authentication exceptions.
"""

from enum import Enum
from typing import Optional, Dict, Any

class AuthErrorCode(Enum):
    """Error codes for authentication failures."""
    INVALID_TOKEN = "invalid_token"
    EXPIRED_TOKEN = "expired_token"
    USER_NOT_FOUND = "user_not_found"
    USER_ALREADY_EXISTS = "user_already_exists"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    EMAIL_SEND_FAILED = "email_send_failed"
    INVALID_CREDENTIALS = "invalid_credentials"
    SESSION_EXPIRED = "session_expired"
    WEBAUTHN_FAILED = "webauthn_failed"

class AuthError(Exception):
    """Base authentication exception."""
    
    def __init__(
        self, 
        code: AuthErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "error": self.code.value,
            "message": self.message,
            "details": self.details
        }