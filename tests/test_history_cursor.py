"""Opaque keyset history pagination contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from cns.api.base import ValidationError
from cns.infrastructure.continuum_repository import ContinuumRepository


def import_data_module(monkeypatch: pytest.MonkeyPatch):
    """Import the endpoint without requiring live Vault configuration."""
    import importlib
    import clients.vault_client as vault_client

    monkeypatch.setattr(vault_client, "get_database_url", lambda _name: "postgresql://test")
    monkeypatch.setattr(vault_client, "get_service_config", lambda field: {
        "email_gateway_url": "https://mail.example.test",
        "email_gateway_api_key": "test-key",
        "email_gateway_hmac_secret": "test-secret",
        "valkey_url": "valkey://localhost:6379",
        "app_url": "https://mira.example.test",
        "crm_base_url": "https://crm.example.test",
        "crm_lifecycle_service_secret": "test-lifecycle-secret",
    }[field])
    return importlib.import_module("cns.api.data")


def test_history_cursor_round_trips_timestamp_and_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    data_module = import_data_module(monkeypatch)
    created_at = datetime(2026, 7, 14, 10, 30, 45, 123456, tzinfo=UTC)
    message_id = UUID("00000000-0000-0000-0000-000000000042")

    cursor = data_module._encode_history_cursor(created_at, message_id)

    assert "+" not in cursor
    assert "/" not in cursor
    assert data_module._decode_history_cursor(cursor) == (created_at, message_id)


@pytest.mark.parametrize("cursor", ["", "not-base64", "e30", "W10"])
def test_invalid_history_cursor_is_a_validation_error(
    cursor: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_module = import_data_module(monkeypatch)
    with pytest.raises(ValidationError, match="history cursor|non-empty"):
        data_module._decode_history_cursor(cursor)


def test_repository_uses_tie_safe_exclusive_keyset_and_returns_chronological_page() -> None:
    created_at = datetime(2026, 7, 14, 10, 0, tzinfo=UTC)
    newest = UUID("00000000-0000-0000-0000-000000000003")
    middle = UUID("00000000-0000-0000-0000-000000000002")
    oldest = UUID("00000000-0000-0000-0000-000000000001")
    captured: dict[str, object] = {}

    class FakeDB:
        def execute_query(self, query: str, params: tuple[object, ...]) -> list[dict[str, object]]:
            captured["query"] = query
            captured["params"] = params
            return [
                _row(newest, created_at, "newest"),
                _row(middle, created_at, "middle"),
                _row(oldest, created_at, "lookahead"),
            ]

    repository = ContinuumRepository.__new__(ContinuumRepository)
    repository._get_client = lambda _user_id: FakeDB()
    before = (datetime(2026, 7, 15, tzinfo=UTC), UUID(int=0))

    page = repository.get_history("user-1", limit=2, before=before, message_type="all")

    assert "(created_at, id) < (%s, %s)" in captured["query"]
    assert "ORDER BY created_at DESC, id DESC" in captured["query"]
    assert captured["params"] == ("user-1", *before, 3)
    assert [message["id"] for message in page["messages"]] == [str(middle), str(newest)]
    assert page["has_more"] is True
    assert page["next_before"] == (created_at, middle)


def _row(message_id: UUID, created_at: datetime, content: str) -> dict[str, object]:
    return {
        "id": message_id,
        "role": "assistant",
        "content": content,
        "created_at": created_at,
        "metadata": {},
        "tool_call_id": None,
        "is_error": False,
    }
