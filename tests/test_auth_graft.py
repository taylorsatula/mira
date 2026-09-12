"""Focused contracts for the hosted-auth graft and target-specific adaptations.

WP3-B excision record (plan §6.3.5, WP0 triage tests/TRIAGE.md §A): the seven
CRM-only tests this file recovered from crm — demo-policy egress, CRM
workspace token provisioning, workspace repair, CRM+billing cleanup
ordering, the demo endpoint, the CRM page-mount surface, and CRM page
visibility — are deleted, not stubbed: `auth/demo_policy.py`,
`auth/crm_workspace.py`, `cns/api/demo.py`, and `billing/` do not exist in
mira-OSS (D7, D8, D12). Retained tests were minimally updated where the
migration moved a collaborator (`crm_workspace_service.delete_account` →
`provisioner.delete`, the dev identity → neutral placeholders, the dev
redirect → `/chat`) or inverted an assertion that encoded CRM's topology
rather than mira-OSS's two-mode identity model (`single` local-session
bootstrap vs `multi` public signup).
"""

from __future__ import annotations

import hashlib
import importlib
from datetime import timedelta
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def auth_modules():
    """Import the auth primitives; nothing here touches external services.

    The crm version patched Vault accessors with the private deployment's
    email-gateway and CRM field values; those key names belong to a
    security topology that is scrubbed from this repository (plan §11), and
    `auth.config` no longer reads them at all (WP3-A).
    """
    session = importlib.import_module("auth.session")
    types = importlib.import_module("auth.types")
    yield SimpleNamespace(
        session=session,
        types=types,
    )


class FakeValkey:
    def __init__(self) -> None:
        self.values: dict[str, dict] = {}
        self.deleted: list[str] = []

    def json_set_with_expiry(self, key: str, path: str, value: dict, expiry: int) -> bool:
        self.values[key] = value.copy()
        return True

    def json_get(self, key: str, path: str):
        value = self.values.get(key)
        return [value.copy()] if value is not None else None

    def delete(self, key: str) -> bool:
        self.deleted.append(key)
        return self.values.pop(key, None) is not None

    def scan(self, cursor: int, match: str, count: int):
        return 0, list(self.values)


def _user(auth_modules, *, subject_kind: str, demo_expires_at=None):
    from utils.timezone_utils import utc_now

    return auth_modules.types.UserRecord(
        id="2c20dc95-e1d7-49ce-9227-675f3f439843",
        email="person@example.com",
        first_name="Alex",
        last_name="Example",
        is_active=True,
        created_at=utc_now(),
        webauthn_credentials={},
        memory_manipulation_enabled=True,
        timezone="America/Chicago",
        subject_kind=subject_kind,
        demo_expires_at=demo_expires_at,
    )


def test_session_uses_hashed_key_and_rejects_expired_demo(auth_modules, monkeypatch):
    from utils.timezone_utils import utc_now

    fake_valkey = FakeValkey()
    monkeypatch.setattr(auth_modules.session, "get_valkey", lambda: fake_valkey)
    manager = auth_modules.session.SessionManager()
    monkeypatch.setattr(manager, "_generate_session_token", lambda: "raw-session-token")

    user = _user(
        auth_modules,
        subject_kind="demo",
        demo_expires_at=utc_now() - timedelta(seconds=1),
    )
    token = manager.create_session(user)
    expected_key = f"session:{hashlib.sha256(token.encode()).hexdigest()}"

    assert expected_key in fake_valkey.values
    assert token not in expected_key
    assert manager.validate_session(token) is None
    assert expected_key in fake_valkey.deleted
    assert manager._csrf_key(token) in fake_valkey.deleted


def test_logout_others_preserves_current_session(auth_modules, monkeypatch):
    fake_valkey = FakeValkey()
    monkeypatch.setattr(auth_modules.session, "get_valkey", lambda: fake_valkey)
    manager = auth_modules.session.SessionManager()
    current_token = "current-session"
    other_token = "other-session"
    user_id = "2c20dc95-e1d7-49ce-9227-675f3f439843"
    fake_valkey.values[manager._session_key(current_token)] = {"user_id": user_id}
    fake_valkey.values[manager._session_key(other_token)] = {"user_id": user_id}
    fake_valkey.values[manager._session_key("another-user")] = {"user_id": "someone-else"}
    fake_valkey.values[manager._csrf_key(other_token)] = {"token": "csrf"}

    assert manager.revoke_user_sessions_except(user_id, current_token) == 1
    assert manager._session_key(current_token) in fake_valkey.values
    assert manager._session_key(other_token) not in fake_valkey.values
    assert manager._csrf_key(other_token) not in fake_valkey.values
    assert manager._session_key("another-user") in fake_valkey.values


def test_member_provisioning_failure_compensates_before_magic_link(auth_modules):
    from auth.service import AuthService

    events: list[str] = []
    service = AuthService.__new__(AuthService)
    service.rate_limiter = SimpleNamespace(is_allowed=lambda email, ip: (True, 0))
    service.security_logger = SimpleNamespace(log_event=lambda *args, **kwargs: None)
    service.db = SimpleNamespace(
        create_user=lambda *args, **kwargs: "2c20dc95-e1d7-49ce-9227-675f3f439843"
    )
    service.provisioner = SimpleNamespace(
        delete=lambda user_id: events.append("teardown_account") or True
    )
    service._initialize_account = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("provisioning unavailable")
    )
    service.request_magic_link = lambda *args, **kwargs: events.append("magic_link")

    with pytest.raises(RuntimeError, match="provisioning unavailable"):
        service.create_user(
            "member@example.com",
            "Alex",
            "Example",
            "America/Chicago",
            "Get work done",
        )

    assert events == ["teardown_account"]


def test_local_session_creates_fully_provisioned_local_account(auth_modules):
    from auth.service import AuthService

    member = _user(auth_modules, subject_kind="member")
    events: list[str] = []
    created: dict[str, str] = {}
    service = AuthService.__new__(AuthService)
    service.db = SimpleNamespace(
        get_user_by_email=lambda email: events.append(f"lookup:{email}") or None,
        create_user=lambda **kwargs: created.update(kwargs) or events.append("create_user") or member.id,
        get_user_by_id=lambda user_id: member,
        update_user_login=lambda user_id: events.append("login") or True,
    )
    service.session_manager = SimpleNamespace(
        create_session=lambda user, idle_timeout, max_lifetime: events.append("session") or "dev-session"
    )
    service.security_logger = SimpleNamespace(log_event=lambda *args, **kwargs: events.append("audit"))
    service.provisioner = SimpleNamespace()
    service._initialize_account = lambda **kwargs: events.append("initialize_account")
    service.SESSION_IDLE_TIMEOUT = 60
    service.SESSION_MAX_LIFETIME = 120

    user, token = service.create_local_session()

    assert created["email"] == "user@localhost"
    assert token == "dev-session"
    assert events == [
        "lookup:user@localhost",
        "create_user",
        "initialize_account",
        "login",
        "session",
        "audit",
    ]


def test_local_session_repairs_an_existing_local_account(auth_modules):
    from auth.service import AuthService

    member = _user(auth_modules, subject_kind="member")
    events: list[str] = []
    service = AuthService.__new__(AuthService)
    service.db = SimpleNamespace(
        get_user_by_email=lambda email: events.append(f"lookup:{email}") or member,
        update_user_login=lambda user_id: events.append("login") or True,
    )
    service.provisioner = SimpleNamespace(
        ensure=lambda user_id, timezone: events.append(f"ensure:{user_id}:{timezone}"),
    )
    service.session_manager = SimpleNamespace(
        create_session=lambda user, idle_timeout, max_lifetime: events.append("session") or "dev-session"
    )
    service.security_logger = SimpleNamespace(log_event=lambda *args, **kwargs: events.append("audit"))
    service.SESSION_IDLE_TIMEOUT = 60
    service.SESSION_MAX_LIFETIME = 120

    user, token = service.create_local_session()

    assert user.id == member.id
    assert token == "dev-session"
    assert events == [
        "lookup:user@localhost",
        f"ensure:{member.id}:America/Chicago",
        "login",
        "session",
        "audit",
    ]


def test_local_session_route_serves_under_single_and_refuses_elsewhere(auth_modules, monkeypatch):
    from fastapi import HTTPException
    import auth.api as auth_api

    auth_service = SimpleNamespace(
        create_local_session=lambda: (_user(auth_modules, subject_kind="member"), "local-session"),
        get_cookie_settings=lambda: auth_modules.types.CookieSettings(
            samesite="strict", httponly=True, secure=False, max_age=120
        ),
    )
    # The route exists only under `single`: one unauthenticated GET on a
    # multi install would provision a stray row alongside public signups.
    monkeypatch.setenv("MIRA_AUTH_MODE", "single")
    response = auth_api.create_local_session(auth_service)
    assert response.status_code == 303
    assert response.headers["location"] == "/chat"
    assert "session=local-session" in response.headers["set-cookie"]
    assert "Secure" not in response.headers["set-cookie"]

    monkeypatch.setenv("MIRA_AUTH_MODE", "multi")
    with pytest.raises(HTTPException) as error:
        auth_api.create_local_session(auth_service)
    assert error.value.status_code == 404


@pytest.mark.integration
def test_fastapi_surface_mounts_multi_user_auth_and_single_mode_identity(auth_modules):
    # Integration-classified because importing `main` requires VAULT_ADDR —
    # the Vault client raises at import time (see the macOS launcher note in
    # deploy/finalize.sh), so the import is not a pure-Python input. The rest
    # of this file runs with no infrastructure.
    import sys

    sys.modules.pop("main", None)
    import main

    paths = {route.path for route in main.create_app().routes}
    assert "/v0/auth/signup" in paths
    assert "/v0/auth/magic-link" in paths
    assert "/v0/auth/verify" in paths
    assert "/v0/auth/session" in paths
    assert "/v0/auth/csrf" in paths
    assert "/v0/auth/logout-others" in paths
    assert "/v0/ws/chat" in paths
    assert "/chat" in paths
    assert "/settings/" in paths
    # Both modes authenticate through the session stack; the bearer-key
    # token endpoint was retired with the pre-backport single-user model,
    # and its replacement mounts exactly once.
    assert "/v0/auth/local/session" in paths
    assert "/oss-auth/token" not in paths
    # No demo admission surface (D12), no OAuth (Square omitted, D-16).
    assert not any(str(path).startswith("/v0/api/demo") for path in paths)
    assert not any(str(path).startswith("/v0/auth/oauth") for path in paths)


@pytest.mark.asyncio
async def test_websocket_cookie_session_establishes_typed_user_context(auth_modules):
    from cns.api.websocket_chat import WebSocketChatHandler
    from utils.timezone_utils import utc_now
    from utils.user_context import clear_user_context, get_current_user

    session_data = auth_modules.types.SessionData(
        user_id="2c20dc95-e1d7-49ce-9227-675f3f439843",
        email="demo+2c20dc95-e1d7-49ce-9227-675f3f439843@no.email.add",
        first_name="Demo",
        last_name="User",
        timezone="America/Chicago",
        subject_kind="demo",
        demo_expires_at=(utc_now() + timedelta(hours=24)).isoformat(),
        created_at=utc_now().isoformat(),
        last_activity=utc_now().isoformat(),
        max_expiry=(utc_now() + timedelta(days=45)).isoformat(),
    )
    sent: list[dict] = []
    websocket = SimpleNamespace(
        cookies={"session": "cookie-token"},
        receive_json=lambda: None,
        send_json=lambda body: None,
    )

    async def receive_json():
        return {"type": "auth"}

    async def send_json(body):
        sent.append(body)

    websocket.receive_json = receive_json
    websocket.send_json = send_json
    handler = WebSocketChatHandler.__new__(WebSocketChatHandler)
    handler.session_manager = SimpleNamespace(
        validate_session=lambda token: session_data if token == "cookie-token" else None
    )

    try:
        # The auth frame is consumed by the connection's single reader, which
        # hands its optional token over. A browser presents no frame token, so
        # the session cookie is the credential here.
        user_id = await handler.authenticate(websocket, None)
        assert user_id == session_data.user_id
        assert sent == []
        assert get_current_user()["subject_kind"] == "demo"
    finally:
        clear_user_context()


def test_local_bootstrap_and_demo_gc_contracts():
    """The contracts that keep the two-mode split safe to ship.

    The crm original asserted the *removal* of the OSS single-user surface;
    mira-OSS replaces the static shared key with the local-session bootstrap
    rather than keeping it, so those assertions now check for absence. The
    `conversation_llm` negative assertion is retained — the column is gone
    (D4) and nothing may resurrect it.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    main_source = (root / "main.py").read_text()
    gc_source = (root / "auth/account_gc.py").read_text()

    assert "ensure_single_user" not in main_source
    assert "oss_ui" not in main_source
    assert "users.subject_kind = 'demo'" in gc_source
    assert "users.demo_expires_at <= NOW()" in gc_source
    assert "conversation_llm = 'demo'" not in gc_source
    assert "demo_history:" not in gc_source
    # CRM is out of the cleanup path entirely (D7): the GC sweep must not
    # name a workspace table or the lifecycle client.
    assert "crm_workspaces" not in gc_source
    assert "CRMWorkspaceLifecycleService" not in gc_source


def test_active_user_helpers_pass_explicit_rls_identity():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    user_context = (root / "utils/user_context.py").read_text()
    portrait_service = (root / "cns/services/portrait_service.py").read_text()

    assert user_context.count("PostgresClient('mira_service', user_id=user_id)") >= 2
    assert portrait_service.count("PostgresClient('mira_service', user_id=user_id)") >= 2
