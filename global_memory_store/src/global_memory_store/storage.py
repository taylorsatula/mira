"""Postgres storage for the store.

One table of memories, one one-row table recording the embedding identity the
database was initialized with. The vector column carries no index: the only
vector query is the duplicate check, which must be exact, and an approximate
index can miss a near neighbor.
"""

import hashlib
import re
from datetime import datetime
from typing import Any

import numpy as np
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

# CROSS-FILE-COUPLING: mirrors DEDUP_SIMILARITY_THRESHOLD in
# lt_memory/processing/memory_processor.py. The store and the instance must
# agree on what counts as a duplicate, or a fact stored once globally is
# re-extracted as a personal memory.
GLOBAL_DEDUP_SIMILARITY_THRESHOLD = 0.92
_DEDUP_MAX_DISTANCE = 1.0 - GLOBAL_DEDUP_SIMILARITY_THRESHOLD

# Fixed key for the writer-serializing advisory lock. Any constant works; it
# only has to be the same in every store process touching this database.
_ADVISORY_LOCK_KEY = 0x676C6F62616C6D65

MAX_TEXT_CHARS = 2000
_STATEMENT_TIMEOUT = "15s"
_POOL_TIMEOUT_SECONDS = 10.0

_WS_RUN = re.compile(r"\s+")

_MEMORY_COLUMNS = "id, seq, text, embedding, happens_at, created_at"


class StoreError(RuntimeError):
    """The store cannot serve: schema missing, identity mismatch, or a failed write."""


def _text_hash(text: str) -> str:
    normalized = _WS_RUN.sub(" ", text.casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _configure_connection(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.row_factory = dict_row
    conn.autocommit = True
    conn.execute(f"SET statement_timeout = '{_STATEMENT_TIMEOUT}'")


def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    embedding = row["embedding"]
    vector = embedding.to_list() if hasattr(embedding, "to_list") else np.asarray(embedding, dtype=np.float32).tolist()
    return {
        "id": str(row["id"]),
        "seq": row["seq"],
        "text": row["text"],
        "embedding": vector,
        "happens_at": row["happens_at"].isoformat() if row["happens_at"] else None,
        "created_at": row["created_at"].isoformat(),
    }


class Store:
    def __init__(self, database_url: str, dimensions: int):
        self.dimensions = dimensions
        self._check_extension(database_url)
        self._pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=4,
            timeout=_POOL_TIMEOUT_SECONDS,
            configure=_configure_connection,
            open=True,
        )

    @staticmethod
    def _check_extension(database_url: str) -> None:
        """Fail with the remedy before the pool's configure callback can mask it.

        The pool registers pgvector types per connection; without the extension
        that registration raises inside the pool's connect path and surfaces as
        a generic PoolTimeout. Checking here keeps the error actionable.
        """
        try:
            with psycopg.connect(database_url, autocommit=True, connect_timeout=10) as conn:
                row = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
        except psycopg.OperationalError as e:
            raise StoreError(f"cannot connect to the database: {e}") from e
        if row is None:
            raise StoreError(
                "the 'vector' extension is not installed in this database; "
                "run CREATE EXTENSION vector; as a superuser, then re-run init"
            )

    def close(self) -> None:
        self._pool.close()

    def init_schema(self, model: str) -> None:
        """Create the tables and record the embedding identity. Idempotent."""
        with self._pool.connection() as conn:
            row = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
            if row is None:
                raise StoreError(
                    "the 'vector' extension is not installed in this database; "
                    "run CREATE EXTENSION vector; as a superuser, then re-run init"
                )
            dims = sql.Literal(self.dimensions)
            conn.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        seq BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
                        text TEXT NOT NULL,
                        text_hash TEXT NOT NULL UNIQUE,
                        embedding vector({dims}) NOT NULL,
                        happens_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        contributor_user_id UUID NOT NULL
                    )
                    """
                ).format(dims=dims)
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_identity (
                    id BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL
                )
                """
            )
            self._check_identity(conn, model, allow_insert=True)

    def verify_ready(self, model: str) -> None:
        """Fail unless the tables exist and the identity row matches the config."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT to_regclass('memories') AS memories, to_regclass('embedding_identity') AS identity"
            ).fetchone()
            if row["memories"] is None or row["identity"] is None:
                raise StoreError("tables are missing; run `python -m global_memory_store init` first")
            self._check_identity(conn, model, allow_insert=False)

    def _check_identity(self, conn: psycopg.Connection, model: str, allow_insert: bool) -> None:
        row = conn.execute("SELECT model, dimensions FROM embedding_identity WHERE id").fetchone()
        if row is None:
            if not allow_insert:
                raise StoreError("embedding_identity is empty; run `python -m global_memory_store init`")
            conn.execute(
                "INSERT INTO embedding_identity (id, model, dimensions) VALUES (TRUE, %s, %s)",
                (model, self.dimensions),
            )
            return
        if row["model"] != model or row["dimensions"] != self.dimensions:
            raise StoreError(
                f"embedding identity mismatch: database was initialized with model={row['model']!r} "
                f"dimensions={row['dimensions']}, config says model={model!r} dimensions={self.dimensions}; "
                "refusing to serve"
            )

    def identity(self) -> tuple[str, int, int]:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT model, dimensions FROM embedding_identity WHERE id").fetchone()
            if row is None:
                raise StoreError("embedding_identity is empty; run `python -m global_memory_store init`")
            count = conn.execute("SELECT count(*) AS n FROM memories").fetchone()["n"]
            return row["model"], row["dimensions"], count

    def insert_memory(
        self,
        text: str,
        embedding: np.ndarray,
        happens_at: datetime | None,
        contributor_user_id: str,
    ) -> tuple[str, dict[str, Any]]:
        """Insert, or report the near-duplicate that already exists.

        Returns ("created" | "duplicate", memory row as a JSON-ready dict).
        The duplicate check and the insert run in one transaction holding an
        advisory lock, so concurrent writers cannot both insert the same fact.
        """
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY_LOCK_KEY,))
                duplicate = conn.execute(
                    f"""
                    SELECT {_MEMORY_COLUMNS}
                    FROM memories
                    WHERE (embedding <=> %s::vector) <= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                    """,
                    (embedding, _DEDUP_MAX_DISTANCE, embedding),
                ).fetchone()
                if duplicate is not None:
                    return "duplicate", _row_to_dict(duplicate)
                inserted = conn.execute(
                    f"""
                    INSERT INTO memories (text, text_hash, embedding, happens_at, contributor_user_id)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (text_hash) DO NOTHING
                    RETURNING {_MEMORY_COLUMNS}
                    """,
                    (text, _text_hash(text), embedding, happens_at, contributor_user_id),
                ).fetchone()
                if inserted is None:
                    existing = conn.execute(
                        f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE text_hash = %s",
                        (_text_hash(text),),
                    ).fetchone()
                    return "duplicate", _row_to_dict(existing)
                return "created", _row_to_dict(inserted)

    def list_memories(self, after: int, limit: int) -> tuple[list[dict[str, Any]], int | None]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT {_MEMORY_COLUMNS}
                FROM memories
                WHERE seq > %s
                ORDER BY seq
                LIMIT %s
                """,
                (after, limit),
            ).fetchall()
        memories = [_row_to_dict(row) for row in rows]
        next_after = memories[-1]["seq"] if len(memories) == limit else None
        return memories, next_after

    def delete_memory(self, memory_id: str) -> bool:
        with self._pool.connection() as conn:
            row = conn.execute(
                "DELETE FROM memories WHERE id = %s RETURNING id", (memory_id,)
            ).fetchone()
            return row is not None
