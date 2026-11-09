"""
The global memory store: facts that hold for a person across every MIRA
instance they run, beside the per-user `memories` table this install alone
holds. This module is the application's one write path for `global_memories`,
local and remote.

Local mode is the default: `global_memories` in this install's database is the
whole store. Remote mode is on exactly when Vault `mira/services` carries a
`global_memory_url` field. The switch lives in Vault because Vault persists on
every install class, while `/opt/mira/systemone.env` is lost on every container
recreate. In remote mode a Postgres database at that address, whose table is
`deploy/global_memory_remote_schema.sql`, is the authority; the local table is
its mirror, and every read path keeps reading the local
`global_memories_runtime` view. Remote mode is refused on a multi-user install,
because every account there reads the same global rows.

`pull_remote_global_memories` runs once at boot. It reads every remote row,
re-embeds through this install's document encoder the rows whose text is new or
changed and the local rows that have no vector, upserts them locally, and
deletes local rows the remote no longer holds. The remote stores no vectors, so
instances with different embedding models can share one store.

`store_global_memories` is extraction's write for memories tagged global. It
inserts the local rows inside an open transaction, commits the remote insert,
then commits the local one, so a remote failure rolls the local rows back and
the segment is retried whole. Ids derive from the source segment and the text,
and both inserts skip an existing id, so a retry that re-extracts identical
text writes no second copy on either side.

Local writes go through the admin session because the application role holds
only SELECT on `global_memories_runtime` (`deploy/mira_service_schema.sql`
grants). Extraction tags a memory global only in single-user mode
(`memory_processor._resolve_memory_scope`).

Every remote `OperationalError` — a connection that cannot be opened, one that
drops mid-statement, a statement the server cancels — raises
`RemoteGlobalStoreUnreachable`; every other remote error (a missing table, a
refused privilege, a constraint) propagates as the psycopg error. Neither is a
`RuntimeError` or `ValueError`, so
`orchestrator._is_content_caused_extraction_failure` counts both as
infrastructure. The remote URL goes to libpq verbatim rather than through
`PostgresClient._parse_database_url`, so its query parameters (`sslmode` above
all) reach the connection.
"""
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, NamedTuple, Optional
from urllib.parse import urlparse
from uuid import UUID, uuid4, uuid5

import psycopg
from psycopg.rows import dict_row

from config import config
from lt_memory.models import ExtractedMemory
from lt_memory.vector_ops import VectorOps
from utils.database_session_manager import LTMemorySessionManager
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)

GLOBAL_MEMORY_URL_FIELD = "global_memory_url"

# libpq bounds for the remote: connect, then keepalive probes and an unacknowledged-send
# limit so a connection that dies mid-statement fails instead of blocking on the kernel.
_REMOTE_CONNECTION_KWARGS: Dict[str, Any] = {
    "connect_timeout": 10,
    "keepalives": 1,
    "keepalives_idle": 10,
    "keepalives_interval": 5,
    "keepalives_count": 3,
    "tcp_user_timeout": 30000,
}

# Changing this re-keys every future extracted global id; stored rows keep theirs.
_GLOBAL_MEMORY_ID_NAMESPACE = UUID("5b0f6c1e-8d2a-4c3e-9f41-7a6d2e9b8c10")

_REMOTE_SELECT_SQL = """
SELECT id, text, importance_score, happens_at, created_at, is_archived, archived_at
FROM global_memories
"""

_REMOTE_INSERT_SQL = """
INSERT INTO global_memories (id, text, importance_score, happens_at, created_at)
VALUES (%(id)s, %(text)s, %(importance_score)s, %(happens_at)s, %(created_at)s)
ON CONFLICT (id) DO NOTHING
"""

_REMOTE_PRIVILEGE_SQL = """
SELECT has_table_privilege('global_memories', 'SELECT') AS can_select,
       has_table_privilege('global_memories', 'INSERT') AS can_insert
"""

_LOCAL_INSERT_SQL = """
INSERT INTO global_memories (id, text, embedding, importance_score, happens_at, created_at)
VALUES (%(id)s, %(text)s, %(embedding)s, %(importance_score)s, %(happens_at)s, %(created_at)s)
ON CONFLICT (id) DO NOTHING
"""

_LOCAL_TEXT_SQL = """
SELECT id, text, embedding IS NULL AS missing_vector FROM global_memories
"""

# A NULL embedding means the row needed no new vector, so the stored one stands.
_LOCAL_MIRROR_UPSERT_SQL = """
INSERT INTO global_memories (
    id, text, embedding, importance_score, happens_at, created_at, is_archived, archived_at
) VALUES (
    %(id)s, %(text)s, %(embedding)s, %(importance_score)s, %(happens_at)s,
    %(created_at)s, %(is_archived)s, %(archived_at)s
)
ON CONFLICT (id) DO UPDATE SET
    text = EXCLUDED.text,
    embedding = COALESCE(EXCLUDED.embedding, global_memories.embedding),
    importance_score = EXCLUDED.importance_score,
    happens_at = EXCLUDED.happens_at,
    is_archived = EXCLUDED.is_archived,
    archived_at = EXCLUDED.archived_at
WHERE EXCLUDED.embedding IS NOT NULL
   OR (global_memories.text, global_memories.importance_score, global_memories.happens_at,
       global_memories.is_archived, global_memories.archived_at)
      IS DISTINCT FROM
      (EXCLUDED.text, EXCLUDED.importance_score, EXCLUDED.happens_at,
       EXCLUDED.is_archived, EXCLUDED.archived_at)
"""

_LOCAL_MIRROR_PRUNE_SQL = """
DELETE FROM global_memories WHERE id <> ALL(%(remote_ids)s::uuid[])
"""

_resolved_remote_url: Optional[str] = None
_remote_url_resolved = False


class RemoteGlobalStoreUnreachable(Exception):
    """The remote global store could not be reached, or the connection failed mid-statement.

    libpq reports a down server, an unreachable host, refused credentials, a
    missing database and a refused TLS setting the same way at connect time — an
    OperationalError with no SQLSTATE — so this one class covers all of them.
    """


class GlobalPullResult(NamedTuple):
    """Counts from one boot pull of the remote global store."""
    pulled: int
    embedded: int
    updated: int
    removed: int


def resolve_remote_global_memory_url() -> Optional[str]:
    """The remote store's address, or None in local mode; resolved once per process.

    Raises:
        ValueError: the Vault value is not a postgresql:// URL naming a host.
        RuntimeError: the field is set on an install not in single-user auth mode.
    """
    global _resolved_remote_url, _remote_url_resolved
    if _remote_url_resolved:
        return _resolved_remote_url

    from auth.mode import auth_mode
    from clients.vault_client import get_service_config

    try:
        url = get_service_config(GLOBAL_MEMORY_URL_FIELD)
    except KeyError:
        url = None

    if url is not None:
        parsed = urlparse(url)
        if parsed.scheme not in ("postgresql", "postgres") or not parsed.hostname:
            raise ValueError(
                f"Vault mira/services/{GLOBAL_MEMORY_URL_FIELD} must be a postgresql:// URL "
                f"naming a host (got scheme {parsed.scheme!r}, host {parsed.hostname!r}). "
                f"Setup: docs/GLOBAL_MEMORY.md."
            )
        if auth_mode() != "single":
            raise RuntimeError(
                f"Vault mira/services/{GLOBAL_MEMORY_URL_FIELD} is set, but the remote global "
                f"memory store serves one person and this install runs MIRA_AUTH_MODE="
                f"{auth_mode()}: every account would read that person's global memories. "
                f"Remove the field (rewrite the secret with `vault kv put` without it) or run single-user. "
                f"See docs/GLOBAL_MEMORY.md."
            )

    _resolved_remote_url = url
    _remote_url_resolved = True
    return url


@contextmanager
def _remote_connection(url: str) -> Iterator[psycopg.Connection]:
    """A connection to the remote store that commits on clean exit."""
    try:
        with psycopg.connect(
            url,
            **_REMOTE_CONNECTION_KWARGS,
            options=f"-c statement_timeout={config.database.statement_timeout_ms}",
            row_factory=dict_row,
        ) as remote:
            yield remote
    except psycopg.OperationalError as e:
        raise RemoteGlobalStoreUnreachable(
            f"Could not reach the remote global memory store (server down or unreachable, "
            f"connection lost, or it refused the credentials, database name, or TLS "
            f"settings): {e}"
        ) from e


def _required_remote_url() -> str:
    url = resolve_remote_global_memory_url()
    if url is None:
        raise RuntimeError(
            f"Remote global memory store called in local mode: Vault mira/services has no "
            f"'{GLOBAL_MEMORY_URL_FIELD}' field."
        )
    return url


def check_remote_global_store() -> Dict[str, Any]:
    """Advisory boot check: the remote table answers and grants this role SELECT and INSERT.

    Raises:
        RemoteGlobalStoreUnreachable: no remote connection.
        psycopg.Error: the table is missing.
        RuntimeError: the role lacks SELECT or INSERT on it.
    """
    url = resolve_remote_global_memory_url()
    if url is None:
        return {"mode": "local"}
    with _remote_connection(url) as remote:
        privileges = remote.execute(_REMOTE_PRIVILEGE_SQL).fetchone()
        rows = remote.execute("SELECT count(*) AS n FROM global_memories").fetchone()["n"]
    if not (privileges["can_select"] and privileges["can_insert"]):
        raise RuntimeError(
            f"The remote global memory role lacks a privilege on global_memories: {privileges}. "
            f"It needs SELECT and INSERT (deploy/global_memory_remote_schema.sql)."
        )
    return {"mode": "remote", "rows": rows}


def pull_remote_global_memories(
    vector_ops: VectorOps,
    session_manager: LTMemorySessionManager,
) -> GlobalPullResult:
    """Mirror the remote global store into this install's `global_memories`.

    Raises:
        RemoteGlobalStoreUnreachable: no remote connection; nothing local changed.
        psycopg.Error: a remote or local statement failed.
    """
    with _remote_connection(_required_remote_url()) as remote:
        remote_rows = remote.execute(_REMOTE_SELECT_SQL).fetchall()

    with session_manager.get_admin_session() as session:
        local = {row["id"]: row for row in session.execute_query(_LOCAL_TEXT_SQL)}

    needs_vector = [
        row for row in remote_rows
        if row["id"] not in local
        or local[row["id"]]["text"] != row["text"]
        or local[row["id"]]["missing_vector"]
    ]
    embeddings = vector_ops.generate_embeddings_batch([row["text"] for row in needs_vector])
    embedding_by_id = {row["id"]: embedding for row, embedding in zip(needs_vector, embeddings)}

    updated = 0
    with session_manager.get_admin_session() as session:
        for row in remote_rows:
            updated += session.execute_update(
                _LOCAL_MIRROR_UPSERT_SQL,
                {**row, "embedding": embedding_by_id.get(row["id"])},
            )
        removed = session.execute_update(
            _LOCAL_MIRROR_PRUNE_SQL,
            {"remote_ids": [row["id"] for row in remote_rows]},
        )

    if removed:
        logger.warning(
            "Global memory pull deleted %d local global memories that the remote store "
            "does not hold; in remote mode the remote is the only authority",
            removed,
        )
    return GlobalPullResult(
        pulled=len(remote_rows),
        embedded=len(needs_vector),
        updated=updated,
        removed=removed,
    )


def _global_memory_id(memory: ExtractedMemory) -> UUID:
    if memory.source_segment_id is None:
        return uuid4()
    return uuid5(_GLOBAL_MEMORY_ID_NAMESPACE, f"{memory.source_segment_id}\n{memory.text}")


def store_global_memories(
    memories: List[ExtractedMemory],
    embeddings: List[List[float]],
    session_manager: LTMemorySessionManager,
) -> List[UUID]:
    """Write extraction's global-tagged memories locally and, in remote mode, remotely.

    Returns:
        The rows' ids, shared by the remote and local copies.

    Raises:
        ValueError: `embeddings` does not pair one-to-one with `memories`.
        RemoteGlobalStoreUnreachable: remote mode and the remote failed; nothing stored locally.
        psycopg.Error: a remote or local write failed.
    """
    if len(memories) != len(embeddings):
        raise ValueError(
            f"Embeddings length ({len(embeddings)}) must match memories length ({len(memories)})"
        )
    if not memories:
        return []

    remote_url = resolve_remote_global_memory_url()
    created_at = utc_now()
    rows = [
        {
            "id": _global_memory_id(memory),
            "text": memory.text,
            "importance_score": memory.importance_score,
            "happens_at": memory.happens_at,
            "created_at": created_at,
        }
        for memory in memories
    ]

    with session_manager.get_admin_session() as session:
        for row, embedding in zip(rows, embeddings):
            session.execute_update(_LOCAL_INSERT_SQL, {**row, "embedding": embedding})
        if remote_url is not None:
            with _remote_connection(remote_url) as remote:
                with remote.cursor() as cursor:
                    cursor.executemany(_REMOTE_INSERT_SQL, rows)

    logger.info(
        "Stored %d global memories (%s)",
        len(rows),
        "remote and local" if remote_url is not None else "local",
    )
    return [row["id"] for row in rows]
