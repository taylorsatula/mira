"""
Security headers middleware for the FastAPI application.

Adds defense-in-depth HTTP response headers to every response.

The Content-Security-Policy header is gated behind `MIRA_CSP` and is off by default.
`script-src 'self'` — the narrowing that makes this middleware worth having — has no
in-tree client that satisfies it since the `web/` UI was removed, so it stays off by
default. The other five headers carry no such cost and are always sent.

    MIRA_CSP unset or "off"   five core headers, plus HSTS on HTTPS
    MIRA_CSP=strict           the same, plus STRICT_CONTENT_SECURITY_POLICY

Any other value raises when the middleware stack is built, before the first response is
served, so a misconfigured posture cannot quietly run without a policy. An unset
variable takes the default; a variable set to the empty string raises. Parsing is strict
for the same reason `auth/mode.py` is strict: the value governs the process's
defense-in-depth posture, and a typo must not select a different one.

The strict policy admits no third-party origin. `frame-src` is omitted rather than set
to `'none'`, so same-origin frames fall back to `default-src 'self'`. `style-src
'unsafe-inline'` stays for now: it was required by the removed `web/` UI's inline
`<style>` block and inline `style="…"` attributes, and no replacement client has landed.

What must change before strict can become the default
-----------------------------------------------------
1. The replacement browser client must keep scripts and handlers out of inline markup
   (`script-src 'self'` rejects both) and serve its scripts and styles from assets.
2. A card-processor origin must never come back. mira-OSS has no payments subsystem, so
   nothing may widen `script-src`, `connect-src` or `frame-src` for one.
"""

import os
from typing import Callable, Literal, Optional, get_args

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

CSPMode = Literal["off", "strict"]

#: Environment variable naming the CSP posture.
CSP_MODE_ENVIRONMENT_FIELD = "MIRA_CSP"

#: The posture assumed when the variable is unset. See the module docstring for what
#: has to change before "strict" can become the default.
DEFAULT_CSP_MODE: CSPMode = "off"

#: The complete set of accepted values, derived from the annotation so the type and the
#: parser cannot drift apart.
CSP_MODES: tuple[str, ...] = get_args(CSPMode)

#: The policy emitted when `MIRA_CSP=strict`.
STRICT_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self' ws: wss:; "
    "font-src 'self'; "
    "worker-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


def csp_mode() -> CSPMode:
    """Return the configured Content-Security-Policy posture.

    Raises:
        ValueError: If `MIRA_CSP` is set to anything other than exactly "off" or
            "strict".
    """
    raw_value = os.getenv(CSP_MODE_ENVIRONMENT_FIELD)
    if raw_value is None:
        return DEFAULT_CSP_MODE
    if raw_value not in CSP_MODES:
        raise ValueError(
            f"{CSP_MODE_ENVIRONMENT_FIELD} must be exactly one of "
            f"{', '.join(CSP_MODES)}: {raw_value!r}"
        )
    return raw_value  # type: ignore[return-value]


def content_security_policy() -> Optional[str]:
    """The CSP header value to emit, or None when the posture is off."""
    mode = csp_mode()
    return None if mode == "off" else STRICT_CONTENT_SECURITY_POLICY


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds security headers to all responses."""

    def __init__(self, app) -> None:
        super().__init__(app)
        # Resolved once, at construction: a malformed MIRA_CSP fails when the
        # middleware stack is built rather than per response, and the header cannot
        # change underneath a running process.
        self.content_security_policy = content_security_policy()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Core security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        # HSTS only for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Absent while MIRA_CSP is off.
        if self.content_security_policy:
            response.headers["Content-Security-Policy"] = self.content_security_policy

        return response
