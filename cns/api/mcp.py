"""MCP check-in endpoint: a one-tool Model Context Protocol surface over the
existing chat turn path.

main.py mounts this at /v0/mcp only when ``config.system.mcp_enabled`` is set
(strict ``MIRA_MCP_ENABLED=1``), and imports this module lazily inside that
flag branch — disabled mode constructs nothing MCP-related at boot: no SDK
import, no app, no session manager.

Design contract (owned by ``mcp/AGENTS.md`` at the repo root):

- One tool: ``check_in(message, image?, image_type?, document?, document_type?)``.
- No turn logic lives here. The tool self-calls ``POST /v0/api/chat`` on this
  process's own listener, forwarding the client's Authorization header
  verbatim — the real endpoint performs authentication, user-context
  installation, the per-user request lock, validation, and the commit. One
  turn path exists; MCP rides it.
- Boundary auth: ``_AuthGate`` rejects unauthenticated requests with 401
  before any MCP protocol handling, using the header rung of the shared
  credential ladder (auth/service.py validators, same order and semantics as
  auth/api.py:get_current_user: session first, then issued API token, no
  activity extension; the cookie rung is not accepted here because a cookie
  session cannot ride the forwarded header to the self-call).
- The forwarded header reaches the tool through a contextvar set by the gate
  — the per-call context object for a value that must flow from the ASGI
  boundary to a tool the SDK invokes later in the same request's task.
"""

import json
import logging
from contextvars import ContextVar
from typing import Optional

import httpx
from fastapi.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Receive, Scope, Send

from auth.service import get_auth_service
from cns.api.chat import _BUSY_REJECTION_MESSAGE
from config.config_manager import config as app_config

logger = logging.getLogger("mira.mcp")

# httpx (connect, read): connect is local TCP; read must cover a legal turn.
# The server's own per-user lock TTL derives from
# (MAX_LOCAL_TOOL_CALLS_PER_TURN + 1) * max(api.timeout, provider_response_timeout) * 2
# (cns/api/chat.py) — a structural bound no fixed client timeout can cover.
# 600 s spans every realistic turn (a handful of model calls); genuinely
# pathological turns surface the truthful timeout error below instead of a
# silent hang.
_CHAT_CONNECT_TIMEOUT_S = 10.0
_CHAT_READ_TIMEOUT_S = 600.0

# The raw Authorization header of the current MCP request ("Bearer <token>"),
# set by _AuthGate for the duration of one request and forwarded verbatim to
# the self-call. Default None: the tool refuses to run outside an
# authenticated request.
_forwarded_authorization: ContextVar[Optional[str]] = ContextVar(
    "mcp_forwarded_authorization", default=None
)

_CHECK_IN_DESCRIPTION = (
    "Send a message to MIRA, a persistent AI companion that maintains long-term "
    "memories, a persona, and a conversation history for its user. This performs a "
    "complete conversational turn: the message is appended to the permanent "
    "conversation history, may be extracted into long-term memory, and the reply is "
    "MIRA's own composed response with all of its internal tooling and memory already "
    "applied. Only one turn runs at a time — a call made while another turn is in "
    "progress fails with a busy error. A turn can take tens of seconds to minutes.\n\n"
    "An optional image or document (not both) may accompany the message. image/document "
    "are base64-encoded bytes with no 'data:' prefix; the matching image_type/document_type "
    "MIME type is required alongside. Images: image/jpeg, image/png, image/gif, "
    "image/webp (max 5 MB decoded). Documents: PDF, DOCX, XLSX, TXT, CSV, JSON "
    "(max 32 MB decoded)."
)


def _self_call_base_url() -> str:
    """Base URL of this process's own listener for the self-call.

    ``api_server.host`` is a *bind* address: the bind-all wildcards ("0.0.0.0",
    "::") are not connectable names, and an IPv6 address needs brackets in a
    URL. The process is co-located with its listener, so a non-wildcard bind
    is used as-is (an operator binding one specific interface still serves
    loopback or that interface locally).
    """
    host = app_config.api_server.host
    if host in ("", "0.0.0.0", "::"):
        host = "127.0.0.1"
    elif ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{app_config.api_server.port}"


async def _check_in(
    message: str,
    image: Optional[str] = None,
    image_type: Optional[str] = None,
    document: Optional[str] = None,
    document_type: Optional[str] = None,
) -> str:
    """One complete chat turn with MIRA; returns MIRA's reply text verbatim."""
    # ToolError is the SDK's sanctioned anticipated-failure channel: its message
    # reaches the model as the tool's error content. Any other exception type is
    # treated as a crash and the model sees only "Error executing tool check_in".
    from mcp.server.mcpserver.exceptions import ToolError

    header = _forwarded_authorization.get()
    if header is None:
        # Unreachable through the mounted app (_AuthGate always sets it); a
        # bare call means the tool ran outside an authenticated request.
        raise RuntimeError("check_in invoked outside an authenticated MCP request")

    # Async, not sync: a sync tool runs in an anyio worker thread from the
    # same default limiter that sizes the sync-endpoint pool (main.py), and
    # its blocking self-call would hold a token while waiting for another —
    # a self-dependency that deadlocks the whole sync pool at saturation.
    # On the event loop the tool holds no token; only the inner /chat does.
    url = f"{_self_call_base_url()}/v0/api/chat"
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_CHAT_CONNECT_TIMEOUT_S, read=_CHAT_READ_TIMEOUT_S)
        ) as client:
            response = await client.post(
                url,
                headers={"Authorization": header},
                json={
                    "message": message,
                    "image": image,
                    "image_type": image_type,
                    "document": document,
                    "document_type": document_type,
                },
            )
    except httpx.TimeoutException as exc:
        raise ToolError(
            f"MIRA did not finish the turn within {_CHAT_READ_TIMEOUT_S:.0f} seconds; "
            "the HTTP call was abandoned but the turn may still be running server-side "
            "and the message was persisted. Retrying immediately will bounce with a "
            "busy error until the in-flight turn completes."
        ) from exc
    except httpx.HTTPError as exc:
        raise ToolError(f"MIRA is unreachable at {url}: {exc}") from exc

    try:
        envelope = response.json()
    except ValueError:
        # Non-JSON body: fall through to the error rows below, which report
        # the status and raw text truthfully — the generic row is the reporter.
        envelope = None

    if response.status_code == 200 and isinstance(envelope, dict) and envelope.get("success"):
        data = envelope.get("data") or {}
        reply = data.get("response")
        if not isinstance(reply, str):
            raise ToolError(
                f"MIRA returned a malformed success envelope (no data.response): {response.text[:500]}"
            )
        return reply

    error_message = ""
    if isinstance(envelope, dict) and isinstance(envelope.get("error"), dict):
        error_message = str(envelope["error"].get("message") or "")
    detail = error_message or response.text[:500]

    # The busy case is the one 400 whose remedy the caller needs spelled out;
    # keyed on chat.py's rejection constant (imported — no transcription).
    if response.status_code == 400 and _BUSY_REJECTION_MESSAGE in error_message:
        raise ToolError(
            "Another turn is already in progress for this MIRA user (one turn at a "
            "time); wait and retry shortly."
        )
    if response.status_code in (401, 403):
        raise ToolError(
            f"MIRA rejected the check-in token (HTTP {response.status_code}); mint a "
            f"fresh API token. {detail}"
        )
    raise ToolError(f"MIRA chat endpoint returned HTTP {response.status_code}: {detail}")


def _bearer_header_from_scope(scope: Scope) -> Optional[str]:
    """Return the raw Authorization header when it is a Bearer header."""
    for key, value in scope.get("headers", []):
        if key == b"authorization":
            text = value.decode("latin-1")
            if text.lower().startswith("bearer "):
                return text
    return None


async def _send_unauthorized(send: Send, message: str) -> None:
    body = json.dumps(
        {"success": False, "error": {"code": "UNAUTHORIZED", "message": message}}
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class _AuthGate:
    """Fail-closed auth boundary around the MCP app.

    Rejects any request without a valid Bearer token (401) before the MCP
    protocol handler sees it — an unauthenticated client cannot even
    initialize. Validation uses the shared ladder's header rung:
    auth/service.py:validate_session(extend_activity=False) first, then
    validate_api_token, exactly as auth/api.py:get_current_user does for the
    REST API.
    """

    def __init__(self, wrapped: ASGIApp) -> None:
        self._wrapped = wrapped

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._wrapped(scope, receive, send)
            return

        header = _bearer_header_from_scope(scope)
        if header is None:
            await _send_unauthorized(
                send,
                "Authentication required: present a MIRA API token as "
                "'Authorization: Bearer <token>'.",
            )
            return

        token = header.split(" ", 1)[1].strip()
        auth_service = get_auth_service()
        # Validation is synchronous Valkey/Postgres I/O — cross to the
        # threadpool so this per-request wrapper never blocks the event loop
        # that schedules WebSocket frame sends.
        if (
            await run_in_threadpool(auth_service.validate_session, token, False) is None
            and await run_in_threadpool(auth_service.validate_api_token, token) is None
        ):
            logger.warning("MCP /v0/mcp rejected a request with an invalid or expired token")
            await _send_unauthorized(send, "Invalid or expired token.")
            return

        reset = _forwarded_authorization.set(header)
        try:
            await self._wrapped(scope, receive, send)
        finally:
            _forwarded_authorization.reset(reset)


def mount(app) -> None:
    """Build and mount the MCP endpoint at /v0 (final path: /v0/mcp).

    Called by main.py only when config.system.mcp_enabled is set. The SDK is
    imported here (not at module top) so importing this module alone does not
    require the mcp package — but an enabled deployment without it fails
    loudly at boot, as required infrastructure should.

    The session manager is parked on app.state for main.py's lifespan: the
    mounted sub-app receives no lifespan events, so its task group must be
    entered explicitly before the server accepts requests on this surface.
    """
    from mcp.server.mcpserver import MCPServer
    from mcp.server.transport_security import TransportSecuritySettings

    server = MCPServer("mira")
    server.add_tool(_check_in, name="check_in", description=_CHECK_IN_DESCRIPTION)
    # DNS-rebinding protection (the SDK's default) validates the Host header
    # against a fixed host list — unworkable for an instance whose address is
    # DHCP-assigned per spawn, and redundant here: _AuthGate is the security
    # boundary (no request reaches the MCP protocol without a valid Bearer
    # token), and rebinding attacks target unauthenticated localhost servers.
    inner_app = server.streamable_http_app(
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    app.mount("/v0", _AuthGate(inner_app))
    app.state.mcp_session_manager = server.session_manager
    logger.info("MCP endpoint mounted at /v0/mcp (tool: check_in)")
