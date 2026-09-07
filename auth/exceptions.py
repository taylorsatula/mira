"""
Authentication exceptions.
"""

from typing import Optional, Any


class AuthError(Exception):
    """Authentication error with structured error codes."""

    def __init__(self, code: str, message: str, details: Optional[dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)
