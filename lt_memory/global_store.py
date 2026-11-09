"""
The global memory store: facts that hold for a person across every MIRA
instance they run, beside the per-user `memories` table this install alone
holds. This module is the application's one write path for `global_memories`,
local and remote.

Local mode is the default: `global_memories` in this install's database is the
whole store. Remote mode (`config.system.global_memory_remote_enabled`) moves
the authority to a Postgres database at the address in Vault
`mira/services/global_memory_url`, whose table is
`deploy/global_memory_remote_schema.sql`; the local table becomes its mirror,
and every read path keeps reading the local `global_memories_runtime` view.

`pull_remote_global_memories` runs once at boot. It reads every remote row,
re-embeds through this install's document encoder only the rows whose text is
new or changed, upserts them locally, and deletes local rows the remote no
longer holds. The remote stores no vectors, so instances with different
embedding models can share one store.

`store_global_memories` is extraction's write for memories tagged global. In
remote mode the remote INSERT commits before the local row with the same id is
written, so a remote outage raises before anything is stored and the segment is
retried whole.

Local writes go through the admin session because the application role holds
only SELECT on `global_memories_runtime` (`deploy/mira_service_schema.sql`
grants). Extraction tags a memory global only in single-user mode
(`memory_processor._resolve_memory_scope`), because every account on an install
reads the same global rows.

A connection that cannot be opened raises `RemoteGlobalStoreUnreachable`;
anything that fails after connecting propagates as the psycopg error. Neither
is a `RuntimeError` or `ValueError`, so
`orchestrator._is_content_caused_extraction_failure` counts both as
infrastructure, never as the segment's own data. The remote URL goes to libpq
verbatim rather than through `PostgresClient._parse_database_url`, so its query
parameters (`sslmode` above all) reach the connection.
"""
import logging
from typing import List, NamedTuple
from urllib.parse import urlparse
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from config import config
from lt_memory.models import ExtractedMemory
from lt_memory.vector_ops import VectorOps
from utils.database_session_manager import LTMemorySessionManager
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)

GLOBAL_MEMORY_URL_FIELD = "global_memory_url"

REMOTE_CONNECT_TIMEOUT_SECONDS = 10

_REMOTE_SELECT_SQL = """
SELECT id, text, importance_score, happens_at, created_at, is_archived, archived_at
FROM global_memories
"""

_REMOTE_INSERT_SQL = """
INSERT INTO global_memories (id, text, importance_score, happens_at, created_at)
VALUES (%(id)s, %(text)s, %(importance_score)s, %(happens_at)s, %(created_at)s)
"""

_LOCAL_INSERT_SQL = """
INSERT INTO global_memories (id, text, embedding, importance_score, happens_at, created_at)
VALUES (%(id)s, %(text)s, %(embedding)s, %(importance_score)s, %(happens_at)s, %(created_at)s)
"""

# A NULL embedding means the text did not change, so the stored vector stands.
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
WHERE (global_memories.text, global_memories.importance_score, global_memories.happens_at,
       global_memories.is_archived, global_memories.archived_at)
      IS DISTINCT FROM
      (EXCLUDED.text, EXCLUDED.importance_score, EXCLUDED.happens_at,
       EXCLUDED.is_archived, EXCLUDED.archived_at)
"""

_LOCAL_MIRROR_PRUNE_SQL = """
DELETE FROM global_memories WHERE id <> ALL(%(remote_ids)s::uuid[])
"""


class RemoteGlobalStoreUnreachable(Exception):
    """No connection to the remote global store could be opened.

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


def remote_global_memory_url() -> str:
    """The remote store's address from Vault, checked to be a Postgres URL with a host.

    Raises:
        RuntimeError: Vault `mira/services` has no `global_memory_url` field.
        ValueError: the value is not a postgresql:// URL naming a host.
    """
    from clients.vault_client import get_service_config

    try:
        url = get_service_config(GLOBAL_MEMORY_URL_FIELD)
    except KeyError as e:
        raise RuntimeError(
            f"MIRA_GLOBAL_MEMORY_REMOTE_ENABLED=1 but Vault mira/services has no "
            f"'{GLOBAL_MEMORY_URL_FIELD}' field. Write it with `vault kv patch secret/mira/services "
            f"{GLOBAL_MEMORY_URL_FIELD}=postgresql://USER:PASSWORD@HOST:5432/DBNAME`, or set "
            f"MIRA_GLOBAL_MEMORY_REMOTE_ENABLED=0. Setup: docs/GLOBAL_MEMORY.md."
        ) from e

    parsed = urlparse(url)
    if parsed.scheme not in ("postgresql", "postgres") or not parsed.hostname:
        raise ValueError(
            f"Vault mira/services/{GLOBAL_MEMORY_URL_FIELD} must be a postgresql:// URL "
            f"naming a host (got scheme {parsed.scheme!r}, host {parsed.hostname!r})."
        )
    return url


def _connect_remote() -> psycopg.Connection:
    try:
        return psycopg.connect(
            remote_global_memory_url(),
            connect_timeout=REMOTE_CONNECT_TIMEOUT_SECONDS,
            options=f"-c statement_timeout={config.database.statement_timeout_ms}",
            row_factory=dict_row,
        )
    except psycopg.OperationalError as e:
        raise RemoteGlobalStoreUnreachable(
            f"Could not connect to the remote global memory store (server down or "
            f"unreachable, or it refused the credentials, database name, or TLS settings): {e}"
        ) from e


def pull_remote_global_memories(
    vector_ops: VectorOps,
    session_manager: LTMemorySessionManager,
) -> GlobalPullResult:
    """Mirror the remote global store into this install's `global_memories`.

    Raises:
        RemoteGlobalStoreUnreachable: no remote connection; nothing local changed.
        psycopg.Error: a remote or local statement failed.
    """
    with _connect_remote() as remote:
        remote_rows = remote.execute(_REMOTE_SELECT_SQL).fetchall()

    with session_manager.get_admin_session() as session:
        local_text = {
            row["id"]: row["text"]
            for row in session.execute_query("SELECT id, text FROM global_memories")
        }

    changed = [row for row in remote_rows if local_text.get(row["id"]) != row["text"]]
    embeddings = vector_ops.generate_embeddings_batch([row["text"] for row in changed])
    embedding_by_id = {row["id"]: embedding for row, embedding in zip(changed, embeddings)}

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
        embedded=len(changed),
        updated=updated,
        removed=removed,
    )


def store_global_memories(
    memories: List[ExtractedMemory],
    embeddings: List[List[float]],
    session_manager: LTMemorySessionManager,
) -> List[UUID]:
    """Write extraction's global-tagged memories: remote first in remote mode, then local.

    Returns:
        The new rows' ids, shared by the remote and local copies.

    Raises:
        ValueError: `embeddings` does not pair one-to-one with `memories`.
        RemoteGlobalStoreUnreachable: remote mode and no remote connection; nothing stored.
        psycopg.Error: a remote or local write failed.
    """
    if len(memories) != len(embeddings):
        raise ValueError(
            f"Embeddings length ({len(embeddings)}) must match memories length ({len(memories)})"
        )
    if not memories:
        return []

    created_at = utc_now()
    rows = [
        {
            "id": uuid4(),
            "text": memory.text,
            "importance_score": memory.importance_score,
            "happens_at": memory.happens_at,
            "created_at": created_at,
        }
        for memory in memories
    ]

    if config.system.global_memory_remote_enabled:
        with _connect_remote() as remote:
            with remote.cursor() as cursor:
                cursor.executemany(_REMOTE_INSERT_SQL, rows)

    with session_manager.get_admin_session() as session:
        for row, embedding in zip(rows, embeddings):
            session.execute_update(_LOCAL_INSERT_SQL, {**row, "embedding": embedding})

    logger.info(
        "Stored %d global memories (%s)",
        len(rows),
        "remote and local" if config.system.global_memory_remote_enabled else "local",
    )
    return [row["id"] for row in rows]
