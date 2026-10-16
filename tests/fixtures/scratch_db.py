"""Throwaway Postgres database: create (owned by mira_admin), apply the shipped schema and the ownership bootstrap, drop WITH (FORCE).

Spec: tests/fixtures/AGENTS.md. The schema is the real
`deploy/mira_service_schema.sql`, never a transcription. The admin connection is
used only for CREATE/DROP DATABASE, role provisioning, and the pgvector
availability check; it never reads or writes an application table. The shipped
schema is psql-only (it opens with `\\if`/`\\set` guards), so the apply shells out
to `psql -v ... -f` with the five embedding_* variables — the same applier and
the same values seam (`describe_for_installer`) the deploy scripts use.
"""
from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Iterator, List

import psycopg

_REPO = Path(__file__).resolve().parents[2]
_SCHEMA = _REPO / "deploy/mira_service_schema.sql"
_BOOTSTRAP = _REPO / "deploy/bootstrap_adopt.sql"

# The schema's preconditions (its header, lines 4-11): mira_admin/mira_dbuser
# must exist before the apply, and pgvector must be installed. deploy/postgresql.sh
# provisions the roles for a real install; the fixture must not depend on that
# having happened, so it provisions them the same way (LOGIN, BYPASSRLS for
# mira_admin; no password — nothing connects as these roles in a scratch probe).
_ROLES: tuple[tuple[str, str], ...] = (
    ("mira_admin", "LOGIN BYPASSRLS"),
    ("mira_dbuser", "LOGIN"),
)


def _name(prefix: str) -> str:
    return "".join(c for c in f"{prefix}_{uuid.uuid4().hex[:12]}" if c.isalnum() or c == "_")


def _provision_roles(admin: psycopg.Connection) -> List[str]:
    """Create the schema's roles when absent; return the names created here."""
    created: List[str] = []
    for role, options in _ROLES:
        present = admin.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
        if present is None:
            admin.execute(f'CREATE ROLE "{role}" {options}')
            created.append(role)
    return created


def _require_pgvector(admin: psycopg.Connection) -> None:
    """The schema runs CREATE EXTENSION vector; the server must ship it."""
    available = admin.execute(
        "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
    ).fetchone()
    if available is None:
        raise RuntimeError(
            "pgvector is not installed on this server; the shipped schema cannot "
            "run CREATE EXTENSION vector (see docs/MANUAL_INSTALL.md)"
        )


def _embedding_args() -> List[str]:
    """psql -v arguments for the schema's guard, built the way the installer does.

    `deploy/lib/embedding_config.sh` resolves local-provider values by calling
    `clients.embeddings_provider.describe_for_installer`; the fixture uses
    that same seam rather than transcribing model or dimension literals. A local
    provider takes no endpoint URL and no API key, so those variables are empty.
    """
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    from clients.embeddings_provider import describe_for_installer

    model, dimensions = describe_for_installer(["describe", "local"]).split()
    return [
        "-v", "embedding_provider=local",
        "-v", f"embedding_model={model}",
        "-v", "embedding_endpoint_url=",
        "-v", "embedding_api_key_name=",
        "-v", f"embedding_dimensions={dimensions}",
    ]


def _psql_file(dbname: str, sql_file: Path, extra_args: List[str], **conn_kwargs) -> None:
    """Apply a shipped SQL file with psql as the superuser — psycopg cannot send psql meta-commands."""
    psql = shutil.which("psql")
    if psql is None:
        raise FileNotFoundError(
            "psql not on PATH: the shipped schema is psql-only "
            "(deploy/AGENTS.md — a raw client cannot apply it)"
        )
    conninfo = psycopg.conninfo.make_conninfo(dbname=dbname, **conn_kwargs)
    result = subprocess.run(
        [psql, "-X", "-v", "ON_ERROR_STOP=1", *extra_args,
         "-d", conninfo, "-f", str(sql_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"psql apply of {sql_file} failed (exit {result.returncode}):\n{result.stderr}"
        )


def _apply_schema(dbname: str, **conn_kwargs) -> None:
    """Apply the shipped schema, then the ownership bootstrap every installer runs after it."""
    _psql_file(dbname, _SCHEMA, _embedding_args(), **conn_kwargs)
    _psql_file(dbname, _BOOTSTRAP, [], **conn_kwargs)


@contextlib.contextmanager
def scratch_database(prefix: str = "mira_probe", **conn_kwargs) -> Iterator[str]:
    """Yield a fresh DB name provisioned like an install; drop it (FORCE) on exit.

    The database is owned by mira_admin and carries the shipped schema plus
    `deploy/bootstrap_adopt.sql` — every object mira_admin-owned and the step
    ledger present (empty), exactly the state the installers leave.

    `conn_kwargs` are local admin libpq kwargs for the superuser connection that
    creates and drops the database (e.g. `user="postgres", host="localhost"`). The
    yielded database is reached by the caller's own `psycopg.connect(dbname=<name>,
    **conn_kwargs)`; the app's `PostgresClient` is for user-scoped SQL, not `CREATE DATABASE`.
    """
    if not _SCHEMA.is_file():
        raise FileNotFoundError(f"shipped schema not found: {_SCHEMA}")
    name = _name(prefix)
    admin = psycopg.connect(dbname="postgres", autocommit=True, **conn_kwargs)
    created_roles: List[str] = []
    try:
        created_roles = _provision_roles(admin)
        _require_pgvector(admin)
        admin.execute(f'CREATE DATABASE "{name}" OWNER mira_admin')
        _apply_schema(name, **conn_kwargs)
        yield name
    finally:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        for role in created_roles:
            admin.execute(f'DROP ROLE IF EXISTS "{role}"')
        admin.close()
