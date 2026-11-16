"""
Remote gate: an optional shared-secret checkpoint for non-loopback peers.

Single-user MIRA auto-provisions a browser session for anyone who can reach
port 1993. That is the right default on loopback and the wrong one on a
tailnet or LAN, where "can reach the port" is no longer "is the operator".
The gate closes that gap with one shared secret: when Vault holds
`mira/gate` field `token`, every non-loopback HTTP and WebSocket request must
carry a `mira_gate` cookie derived from that token, or it gets a 401 (API
clients) or a redirect to `/remote-gate` (browser navigations). Loopback peers
are never gated, so local tooling, the installer's health checks, and the
Docker healthcheck are unaffected.

The gate is enabled by writing the secret and disabled by deleting it; both
take effect on the next process start (the token is read once per process and
cached). A Vault read failure fails closed: the gate engages with an
unguessable random token rather than opening the door.

The cookie carries HMAC-SHA256(token, purpose) rather than the token itself,
so the stored value is not the secret. Comparison is constant-time. The gate
page and its submit endpoint are the only exempt paths; everything else —
pages, assets, API, WebSocket — sits behind the check.
"""

import hashlib
import hmac
import logging
import secrets
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from cns.api.base import APIError, create_error_response, generate_request_id

logger = logging.getLogger(__name__)

GATE_PATH = "/remote-gate"
GATE_AUTH_PATH = "/v1/auth/remote-gate"
GATE_COOKIE = "mira_gate"
_COOKIE_PURPOSE = b"mira-remote-gate-v1"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})
_EXEMPT_PATHS = frozenset({GATE_PATH, GATE_AUTH_PATH})

# Process-lifetime cache. _UNSET = not yet read; None = gate disabled;
# a string = the configured token (or, after a Vault failure, an unguessable
# random one — fail closed).
_UNSET = object()
_cached_token: object = _UNSET


def _load_token() -> Optional[str]:
    global _cached_token
    if _cached_token is not _UNSET:
        return _cached_token  # type: ignore[return-value]
    try:
        from clients.vault_client import get_secret
        token = get_secret("mira/gate", "token", quiet_missing=True)
        _cached_token = token if token else None
        if _cached_token:
            logger.info("Remote gate enabled (secret present in Vault at mira/gate)")
        else:
            logger.warning("mira/gate token is empty; remote gate disabled")
    except (FileNotFoundError, KeyError):
        # Path or field absent: gate disabled, the normal state for installs
        # that never opted in.
        logger.info("Remote gate disabled (no mira/gate secret in Vault)")
        _cached_token = None
    except Exception:
        logger.error("Remote gate: Vault read failed; failing closed", exc_info=True)
        _cached_token = secrets.token_hex(32)
    return _cached_token  # type: ignore[return-value]


def _cookie_value(token: str) -> str:
    return hmac.new(token.encode("utf-8"), _COOKIE_PURPOSE, hashlib.sha256).hexdigest()


def _token_valid(token: str, presented: Optional[str]) -> bool:
    if not presented:
        return False
    return hmac.compare_digest(_cookie_value(token), presented)


def _parse_cookies(header: Optional[bytes]) -> dict[str, str]:
    cookies: dict[str, str] = {}
    if not header:
        return cookies
    for part in header.decode("latin-1").split(";"):
        name, sep, value = part.strip().partition("=")
        if sep:
            cookies[name.strip()] = value.strip()
    return cookies


def _is_loopback(host: Optional[str]) -> bool:
    # No peer information (in-process test transports) is treated as loopback:
    # there is no network peer to distrust.
    return host is None or host in _LOOPBACK_HOSTS


def _wants_html(headers: list[tuple[bytes, bytes]]) -> bool:
    for name, value in headers:
        if name.lower() == b"accept":
            return b"text/html" in value.lower()
    return False


def _gate_required_response(scope, receive, send):
    """Build the rejection for an unauthenticated non-loopback request."""
    headers = scope.get("headers") or []
    path = scope.get("path", "")
    if scope.get("method") == "GET" and _wants_html(headers):
        query = scope.get("query_string", b"").decode("latin-1")
        target = path + (f"?{query}" if query else "")
        response = RedirectResponse(f"{GATE_PATH}?next={quote(target, safe='/')}", status_code=303)
    else:
        response = JSONResponse(
            create_error_response(
                APIError("REMOTE_GATE_REQUIRED", "Remote access gate: authentication required"),
                generate_request_id(),
            ).to_dict(),
            status_code=401,
        )
    return response(scope, receive, send)


class RemoteGateMiddleware:
    """Pure-ASGI gate so WebSocket handshakes are covered too."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        host = client[0] if client else None
        if _is_loopback(host):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if scope["type"] == "http" and path in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        token = _load_token()
        if token is None:
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers") or []
        cookie_header = next((v for k, v in headers if k.lower() == b"cookie"), None)
        if _token_valid(token, _parse_cookies(cookie_header).get(GATE_COOKIE)):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return

        await _gate_required_response(scope, receive, send)


# --- Gate page and submit endpoint -------------------------------------------

router = APIRouter(include_in_schema=False)

_GATE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MIRA — Remote Access</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center;
         background: #0e0e10; color: #e8e6e3;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  form { display: flex; flex-direction: column; gap: 0.9rem; width: min(22rem, 90vw);
         padding: 2rem; border: 1px solid #2a2a2e; border-radius: 12px; background: #161618; }
  h1 { font-size: 1.1rem; font-weight: 600; margin: 0; letter-spacing: 0.02em; }
  p { margin: 0; font-size: 0.85rem; color: #a09d98; }
  input[type=password] { padding: 0.65rem 0.75rem; border-radius: 8px; border: 1px solid #2a2a2e;
         background: #0e0e10; color: inherit; font-size: 1rem; }
  input[type=password]:focus { outline: 2px solid #6b6bd6; border-color: transparent; }
  button { padding: 0.65rem; border: 0; border-radius: 8px; background: #6b6bd6; color: #fff;
           font-size: 1rem; cursor: pointer; }
  button:disabled { opacity: 0.6; cursor: default; }
  #error { color: #e06c75; font-size: 0.85rem; min-height: 1.2em; margin: 0; }
</style>
</head>
<body>
<form id="gate">
  <h1>MIRA</h1>
  <p>This install is gated. Enter the remote access secret to continue.</p>
  <input id="token" type="password" autocomplete="current-password" autofocus
         placeholder="Remote access secret" aria-label="Remote access secret">
  <button type="submit">Unlock</button>
  <p id="error" role="alert"></p>
</form>
<script>
const params = new URLSearchParams(window.location.search);
let next = params.get('next') || '/';
if (!next.startsWith('/') || next.startsWith('//') || next.includes('\\\\')) next = '/';
const form = document.getElementById('gate');
const error = document.getElementById('error');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  error.textContent = '';
  const button = form.querySelector('button');
  button.disabled = true;
  try {
    const res = await fetch('/v1/auth/remote-gate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({token: document.getElementById('token').value, next}),
    });
    if (res.ok) {
      const body = await res.json();
      window.location.assign(body.next || '/');
      return;
    }
    error.textContent = res.status === 429 ? 'Too many attempts. Wait a few minutes.' : 'Incorrect secret.';
  } catch {
    error.textContent = 'Could not reach the server.';
  } finally {
    button.disabled = false;
  }
});
</script>
</body>
</html>
"""


@router.get(GATE_PATH, include_in_schema=False)
def gate_page() -> HTMLResponse:
    return HTMLResponse(_GATE_PAGE)


class GateSubmission(BaseModel):
    token: str
    next: Optional[str] = None


def _safe_next(candidate: Optional[str]) -> str:
    if candidate and candidate.startswith("/") and not candidate.startswith("//") and "\\" not in candidate:
        return candidate
    return "/"


@router.post(GATE_AUTH_PATH, include_in_schema=False)
def submit_gate(submission: GateSubmission, request: Request):
    from auth.security_logger import security_logger
    from auth.service import get_auth_service

    client_ip = request.client.host if request.client else "unknown"
    auth_service = get_auth_service()

    allowed, retry_after = auth_service.rate_limiter.is_allowed(f"remote-gate:{client_ip}", client_ip)
    if not allowed:
        return JSONResponse(
            create_error_response(
                APIError("RATE_LIMIT_EXCEEDED", f"Too many attempts. Try again in {retry_after} seconds."),
                generate_request_id(),
            ).to_dict(),
            status_code=429,
        )

    token = _load_token()
    if token is None:
        # Gate disabled: nothing to check. Succeed so a stale gate page does
        # not strand the user.
        return JSONResponse({"ok": True, "next": _safe_next(submission.next)})

    if not hmac.compare_digest(token, submission.token):
        security_logger.log_event(
            "auth.remote_gate_failure",
            success=False,
            ip_address=client_ip,
            user_agent=request.headers.get("user-agent"),
        )
        return JSONResponse(
            create_error_response(
                APIError("REMOTE_GATE_DENIED", "Incorrect secret"),
                generate_request_id(),
            ).to_dict(),
            status_code=403,
        )

    security_logger.log_event(
        "auth.remote_gate_success",
        success=True,
        ip_address=client_ip,
        user_agent=request.headers.get("user-agent"),
    )
    cookie_settings = auth_service.get_cookie_settings()
    response = JSONResponse({"ok": True, "next": _safe_next(submission.next)})
    response.set_cookie(
        key=GATE_COOKIE,
        value=_cookie_value(token),
        max_age=cookie_settings.max_age,
        httponly=cookie_settings.httponly,
        secure=cookie_settings.secure,
        samesite=cookie_settings.samesite,
        path="/",
    )
    return response
