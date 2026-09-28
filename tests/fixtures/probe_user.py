"""Throwaway-user lifecycle: set user context, work, then clear context and remove data/users/<id>.

Spec: tests/fixtures/AGENTS.md. Uses the sanctioned contextvar surface; creates no
DB row — for user-scoped filesystem and contextvar paths, not persistence rows.
"""
from __future__ import annotations

import contextlib
import shutil
import sys
import uuid
from pathlib import Path
from typing import Iterator

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def user_dir(user_id: str) -> Path:
    """Absolute path of a user's data directory, anchored to the repo root (cwd-independent)."""
    return _REPO / "data/users" / user_id


def cleanup(user_id: str) -> bool:
    """Clear user context and remove the user's data dir; return True when nothing remains."""
    from utils.user_context import clear_user_context

    clear_user_context()
    shutil.rmtree(user_dir(user_id), ignore_errors=True)
    return not user_dir(user_id).exists()


@contextlib.contextmanager
def probe_user() -> Iterator[str]:
    """Yield a fresh UUID with user context set; clear context and remove its data dir on exit."""
    from utils.user_context import set_current_user_id

    user_id = str(uuid.uuid4())
    set_current_user_id(user_id)
    try:
        yield user_id
    finally:
        cleanup(user_id)
