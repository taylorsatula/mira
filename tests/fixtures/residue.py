"""Residue checks: turn "I cleaned up" into a raising witness.

Both assertions raise `AssertionError` when debris remains, so a probe that wraps
one in `harness.check` records a FAIL instead of a pass. Call them after the
fixture's teardown has run: inside a `with probe_user()` or `with scratch_database()`
block they still see the pre-teardown state.
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def assert_user_removed(user_id: str) -> None:
    """Raise when data/users/<user_id> still exists."""
    path = _REPO / "data/users" / user_id
    if path.exists():
        raise AssertionError(f"probe user residue: {path}")


def assert_no_keys(prefix: str, client) -> None:
    """Raise when any key under `prefix` remains; `client` is a real Valkey client (has scan_iter)."""
    keys = list(client.scan_iter(f"{prefix}*"))
    if keys:
        raise AssertionError(f"probe key residue under {prefix!r}: {keys[:10]}")
