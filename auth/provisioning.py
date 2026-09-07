"""
The account-provisioning seam.

`AccountProvisioner` is the boundary between an account's lifecycle inside `auth` and
any sidecar state an account might owe to a system outside it. mira-OSS has no such
system: `NullProvisioner` is the default and the only implementation, provisioning is
nothing, and deletion is `local_teardown`.

The seam exists so `auth/service.py` can hold one account-lifecycle path without naming
a product-specific external service in it. It is the minimal excision mechanism, not a
plug-in surface. Do not extend it — no hooks, priorities, ordering, events or
registration. Per plan §0.2, mira-OSS and crm_mira are parting and crm_mira adapts to
mira-OSS at unification, so a seam built for that convergence would be dead weight in
the distributed artifact. Add a method here only when `auth` itself needs a second
implementation.

Nothing in this module reaches for Vault, Postgres or Valkey at import time; the first
database or Valkey touch happens inside `local_teardown`.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Protocol

from auth.session import SessionManager
from utils.database_session_manager import get_shared_session_manager
from utils.user_context import clear_user_context
from utils.userdata_manager import clear_manager_cache

logger = logging.getLogger(__name__)

# Anchored to the project root rather than the process working directory, matching the
# store this teardown removes (utils/userdata_manager.py:86-92). A CWD-relative
# "data/users" would delete the wrong directory under any supervisor whose
# WorkingDirectory is not the checkout. Path arithmetic only — nothing is created or
# stat'd at import.
DATA_USERS_ROOT = Path(__file__).resolve().parent.parent / "data" / "users"


class AccountProvisioner(Protocol):
    """What an account's lifecycle owes to anything outside `auth` itself.

    Structural typing only — callers annotate this and call three methods, and no
    `isinstance` check is possible against a non-runtime `Protocol`. Implementations
    must be idempotent, because account creation is retried after a partial failure;
    that is why `ensure` exists alongside `provision`.
    """

    def provision(self, user_id: str, timezone: str) -> None:
        """Create whatever sidecar state a new account needs. Raise on failure."""
        ...

    def ensure(self, user_id: str, timezone: str) -> None:
        """Repair or re-create sidecar state for an account that already exists."""
        ...

    def delete(self, user_id: str) -> bool:
        """Tear down an account completely. True once it is gone; False to retry."""
        ...


def local_teardown(user_id: str) -> bool:
    """Remove every trace of a local MIRA account.

    Ordering matters, and each step depends on the one before it: log the user out
    before their data directory goes away, and close the cached SQLite connection
    before the file it holds is unlinked.

    The final ``DELETE FROM users`` runs on the **admin** session. A user-scoped session
    would bind ``app.current_user_id`` to the account being deleted, and the caller may
    hold no RLS context at all — the scheduled garbage-collection pass does not.

    Returns:
        True once the row is gone.

    Raises:
        ValueError: If ``user_id`` is not a UUID. ``users.id`` is ``UUID``, and the value
            names a directory on the way to ``shutil.rmtree``; parsing and
            canonicalising at entry rejects anything that could name a path outside
            ``DATA_USERS_ROOT`` and makes the path component and the query parameter
            agree on one spelling.
        RuntimeError: If no user row was deleted, meaning the account vanished
            mid-cleanup and the earlier steps ran against an account that is already
            gone.
    """
    user_id = str(uuid.UUID(user_id))  # fail before any destructive step, not after

    SessionManager().revoke_user_sessions(user_id)
    clear_manager_cache(user_id)

    user_dir = DATA_USERS_ROOT / user_id
    if user_dir.exists():
        shutil.rmtree(user_dir)

    clear_user_context()

    with get_shared_session_manager().get_admin_session() as session:
        rows_deleted = session.execute_update(
            "DELETE FROM users WHERE id = %(user_id)s",
            {"user_id": user_id},
        )

    if rows_deleted == 0:
        raise RuntimeError(f"Account {user_id} disappeared during cleanup")

    logger.info("Tore down local account %s", user_id)
    return True


class NullProvisioner:
    """The OSS default: an account has no sidecar state beyond MIRA's own.

    Provisioning is genuinely nothing to do. Creating the account itself already has an
    owner — `auth/database.py:initialize_mira_account` creates the continuum and the
    welcome content, which is account work, not sidecar work. Deletion is the local
    teardown.
    """

    def provision(self, user_id: str, timezone: str) -> None:
        """No sidecar state to create."""

    def ensure(self, user_id: str, timezone: str) -> None:
        """No sidecar state to repair."""

    def delete(self, user_id: str) -> bool:
        """Delete the account locally. See `local_teardown`."""
        return local_teardown(user_id)
