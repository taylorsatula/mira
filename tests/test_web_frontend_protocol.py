"""Strict WebSocket frame and connection-dispatch contracts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError


def import_websocket_protocol(monkeypatch: pytest.MonkeyPatch):
    """Import the protocol without requiring a live Vault during unit tests."""
    import importlib

    stub_vault(monkeypatch)
    return importlib.import_module("cns.api.websocket_chat")


def stub_vault(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the exact configuration fields imported protocol modules require."""
    import clients.vault_client as vault_client

    monkeypatch.setattr(vault_client, "get_database_url", lambda _name: "postgresql://test")
    monkeypatch.setattr(vault_client, "get_service_config", lambda field: {
        "email_gateway_url": "https://mail.example.test",
        "email_gateway_api_key": "test-key",
        "email_gateway_hmac_secret": "test-secret",
        "valkey_url": "valkey://localhost:6379",
        "app_url": "https://mira.example.test",
    }[field])


def test_strict_csp_admits_no_third_party_origin_and_no_inline_scripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`MIRA_CSP=strict` serves mira-OSS's real policy, and the default serves none.

    The recovered crm original asserted Stripe card-processor origins and a
    `frame-src` this repository never emits. D7 omits billing, so no payments
    origin may appear at all, and WP3-A made the policy opt-in with the default
    off, so the header is absent unless asked for. Both halves are asserted
    here against `auth/security_middleware.py`, which is the source of truth
    for the string.
    """
    stub_vault(monkeypatch)
    from auth.security_middleware import SecurityHeadersMiddleware

    def csp_for(mode: str | None) -> str | None:
        if mode is None:
            monkeypatch.delenv("MIRA_CSP", raising=False)
        else:
            monkeypatch.setenv("MIRA_CSP", mode)
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)

        @app.get("/")
        def index() -> dict[str, bool]:
            return {"ok": True}

        return TestClient(app).get("/").headers.get("content-security-policy")

    # Default posture: no policy header. The retained UI still carries inline
    # scripts and handlers, so a hard-enabled 'script-src self' would serve a
    # blank page (plan §5 D-7).
    assert csp_for(None) is None

    csp = csp_for("strict")
    assert csp is not None
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "script-src 'self' 'unsafe-inline'" not in csp
    assert "style-src 'self' 'unsafe-inline'" in csp
    assert "img-src 'self' data:" in csp
    assert "connect-src 'self' ws: wss:" in csp
    assert "font-src 'self'" in csp
    assert "worker-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'self'" in csp
    assert "form-action 'self'" in csp
    # No payments provider, so no third-party origin may be admitted for one.
    assert "stripe" not in csp.lower()
    assert "frame-src" not in csp


def test_client_frames_reject_unknown_fields_and_unpaired_attachments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = import_websocket_protocol(monkeypatch)
    message_id = str(uuid4())

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        protocol.validate_client_frame({
            "type": "message",
            "message_id": message_id,
            "content": "hello",
            "priority": "high",
        })

    with pytest.raises(ValidationError, match="provided together"):
        protocol.validate_client_frame({
            "type": "message",
            "message_id": message_id,
            "content": "hello",
            "image": "aW1hZ2U=",
        })

    with pytest.raises(ValidationError, match="image or a document"):
        protocol.validate_client_frame({
            "type": "message",
            "message_id": message_id,
            "content": "hello",
            "image": "aW1hZ2U=",
            "image_type": "image/png",
            "document": "ZG9jdW1lbnQ=",
            "document_type": "text/plain",
        })


@pytest.mark.parametrize(
    ("event", "payload", "message"),
    [
        ("tool_executing", {}, "requires arguments"),
        ("tool_completed", {}, "requires a result"),
        ("tool_error", {"result": "bad"}, "is_error=true"),
        ("tool_detected", {"is_error": True}, "is_error=false"),
    ],
)
def test_server_tool_frames_enforce_event_specific_payloads(
    event: str,
    payload: dict[str, object],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = import_websocket_protocol(monkeypatch)
    with pytest.raises(ValidationError, match=message):
        protocol.validate_server_frame({
            "type": "tool",
            "turn_id": str(uuid4()),
            "segment_id": str(uuid4()),
            "event": event,
            "tool_name": "jobs_tool",
            "tool_id": "call-1",
            **payload,
        })


@pytest.mark.asyncio
async def test_second_message_while_turn_is_active_gets_busy_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = import_websocket_protocol(monkeypatch)
    sent: list[dict[str, object]] = []
    blocker = asyncio.Event()
    active_turn = asyncio.create_task(blocker.wait())
    connection = SimpleNamespace(
        inbound=asyncio.Queue(),
        active_turn_task=active_turn,
        active_turn_id=uuid4(),
        cancel_event=None,
        send=lambda frame: _record_frame(sent, frame),
    )
    await connection.inbound.put(protocol.MessageFrame(
        type="message",
        message_id=uuid4(),
        content="This must not disappear",
    ))
    await connection.inbound.put(protocol.ClientDisconnected())

    handler = protocol.WebSocketChatHandler.__new__(protocol.WebSocketChatHandler)
    try:
        await handler._dispatch(connection, "user-1")
    finally:
        active_turn.cancel()
        await asyncio.gather(active_turn, return_exceptions=True)

    assert sent == [{
        "type": "protocol_error",
        "code": "TURN_BUSY",
        "message": "A server turn is already active",
    }]


async def _record_frame(sent: list[dict[str, object]], frame: dict[str, object]) -> None:
    sent.append(frame)
