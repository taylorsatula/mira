"""The one exception type `auth` raises, and the `code` string that crosses the HTTP boundary.

`AuthError.code` is not a log field. Every handler in `auth/api.py` maps it to a
status code and response envelope on its own, with no shared default, so an
unmapped code ships as whatever fallback the raising handler happens to carry —
401 on the credential ladder `get_current_user`, the 400 envelope on handlers
with an `AuthError` branch, 500 on the handlers with none. A new code is not
complete until every handler that can raise it maps it.
"""

from typing import Optional, Any


class AuthError(Exception):
    """Authentication error with structured error codes."""

    def __init__(self, code: str, message: str, details: Optional[dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)
