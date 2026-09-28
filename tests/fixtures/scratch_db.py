"""Throwaway Postgres database: create, apply the shipped schema, drop WITH (FORCE).

Spec: tests/fixtures/AGENTS.md. The schema is the real
`deploy/mira_service_schema.sql`, never a transcription. The admin connection is
used only for CREATE/DROP DATABASE and never reads or writes an application table.
"""
from __future__ import annotations

import contextlib
import uuid
from pathlib import Path
from typing import Iterator

import psycopg

_REPO = Path(__file__).resolve().parents[2]
_SCHEMA = _REPO / "deploy/mira_service_schema.sql"


def _name(prefix: str) -> str:
    return "".join(c for c in f"{prefix}_{uuid.uuid4().hex[:12]}" if c.isalnum() or c == "_")


@contextlib.contextmanager
def scratch_database(prefix: str = "mira_probe", **conn_kwargs) -> Iterator[str]:
    """Yield a fresh DB name provisioned from the shipped schema; drop it (FORCE) on exit.

    `conn_kwargs` are local admin libpq kwargs for the superuser connection that
    creates and drops the database (e.g. `user="postgres", host="localhost"`). The
    yielded database is reached by the caller's own `psycopg.connect(dbname=<name>,
    **conn_kwargs)`; the app's `PostgresClient` is for user-scoped SQL, not `CREATE DATABASE`.
    """
    if not _SCHEMA.is_file():
        raise FileNotFoundError(f"shipped schema not found: {_SCHEMA}")
    name = _name(prefix)
    admin = psycopg.connect(dbname="postgres", autocommit=True, **conn_kwargs)
    try:
        admin.execute(f'CREATE DATABASE "{name}"')
        conn = psycopg.connect(dbname=name, autocommit=True, **conn_kwargs)
        try:
            conn.execute(_SCHEMA.read_text())
        finally:
            conn.close()
        yield name
    finally:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()
