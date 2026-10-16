"""
Versioned steps for the per-user SQLite store (``data/users/<id>/userdata.db``).

The ledger is ``PRAGMA user_version``: each step above it runs in its own
transaction, and the version is set inside that same transaction, so a crash
or a raise leaves the file at the last completed step. A file created fresh by
``utils/userdata_manager.py`` already has today's tables and is stamped at
``USERDATA_HEAD`` without running any step. Steps keep their introspection
guards, so a database whose shape already matches passes through unchanged.
Only MIRA's own tables are managed here; tables a tool creates for itself are
the tool's.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserdataStepContext:
    """What a step receives: the step transaction's cursor and the user's field encryptor."""

    cursor: sqlite3.Cursor
    encrypt_value: Callable[[Any], str]


def _pager_encrypted_columns(ctx: UserdataStepContext) -> None:
    """Add encrypted pager columns to existing per-user databases."""
    cursor = ctx.cursor
    cursor.execute("PRAGMA table_info(pager_devices)")
    device_columns = {row[1] for row in cursor.fetchall()}

    if "encrypted__name" not in device_columns:
        cursor.execute("ALTER TABLE pager_devices ADD COLUMN encrypted__name TEXT")
        logger.info("Migrated pager_devices: added encrypted__name column")
    if "encrypted__description" not in device_columns:
        cursor.execute("ALTER TABLE pager_devices ADD COLUMN encrypted__description TEXT")
        logger.info("Migrated pager_devices: added encrypted__description column")

    if "name" in device_columns:
        rows = cursor.execute(
            "SELECT id, name FROM pager_devices "
            "WHERE name IS NOT NULL AND encrypted__name IS NULL"
        ).fetchall()
        for row in rows:
            cursor.execute(
                "UPDATE pager_devices SET encrypted__name = :name WHERE id = :id",
                {"id": row["id"], "name": ctx.encrypt_value(row["name"])},
            )
    if "description" in device_columns:
        rows = cursor.execute(
            "SELECT id, description FROM pager_devices "
            "WHERE description IS NOT NULL AND encrypted__description IS NULL"
        ).fetchall()
        for row in rows:
            cursor.execute(
                "UPDATE pager_devices SET encrypted__description = :description WHERE id = :id",
                {"id": row["id"], "description": ctx.encrypt_value(row["description"])},
            )

    cursor.execute("PRAGMA table_info(pager_trust)")
    trust_columns = {row[1] for row in cursor.fetchall()}
    if "encrypted__trusted_name" not in trust_columns:
        cursor.execute("ALTER TABLE pager_trust ADD COLUMN encrypted__trusted_name TEXT")
        logger.info("Migrated pager_trust: added encrypted__trusted_name column")

    if "trusted_name" in trust_columns:
        rows = cursor.execute(
            "SELECT id, trusted_name FROM pager_trust "
            "WHERE trusted_name IS NOT NULL AND encrypted__trusted_name IS NULL"
        ).fetchall()
        for row in rows:
            cursor.execute(
                "UPDATE pager_trust SET encrypted__trusted_name = :trusted_name WHERE id = :id",
                {"id": row["id"], "trusted_name": ctx.encrypt_value(row["trusted_name"])},
            )


def _pager_plaintext_cleared(ctx: UserdataStepContext) -> None:
    """Drop legacy plaintext pager columns their encrypted twins have replaced.

    Step 1 copied name/description (pager_devices) and trusted_name
    (pager_trust) into their encrypted__ twins but left the plaintext
    originals behind in the same file. This step runs after step 1 in the
    chain, so by the time it executes every row's twin is populated — a
    file at a version past 1 got there through step 1, and a file below 1
    runs step 1 first in the same pass — so dropping the plaintext column
    cannot lose a value the chain could have preserved. (The legacy schema
    declares ``name`` NOT NULL, so clearing with UPDATE ... SET NULL is
    not an option; DROP COLUMN also leaves the table in today's shape:
    the fresh DDL has no plaintext pager columns.) Files without the
    legacy columns — today's DDL never creates them — pass through
    unchanged, and a plaintext column whose encrypted twin never landed
    is left alone rather than destroyed.
    """
    cursor = ctx.cursor
    cursor.execute("PRAGMA table_info(pager_devices)")
    device_columns = {row[1] for row in cursor.fetchall()}

    if "name" in device_columns and "encrypted__name" in device_columns:
        cursor.execute("ALTER TABLE pager_devices DROP COLUMN name")
        logger.info("Migrated pager_devices: dropped legacy plaintext name column")
    if "description" in device_columns and "encrypted__description" in device_columns:
        cursor.execute("ALTER TABLE pager_devices DROP COLUMN description")
        logger.info("Migrated pager_devices: dropped legacy plaintext description column")

    cursor.execute("PRAGMA table_info(pager_trust)")
    trust_columns = {row[1] for row in cursor.fetchall()}
    if "trusted_name" in trust_columns and "encrypted__trusted_name" in trust_columns:
        cursor.execute("ALTER TABLE pager_trust DROP COLUMN trusted_name")
        logger.info("Migrated pager_trust: dropped legacy plaintext trusted_name column")


def _domaindoc_columns(ctx: UserdataStepContext) -> None:
    """Add the archived, pinned, and encrypted__summary columns; set pinned for always-expanded sections."""
    cursor = ctx.cursor
    cursor.execute("PRAGMA table_info(domaindocs)")
    domaindoc_columns = {row[1] for row in cursor.fetchall()}

    if "archived" not in domaindoc_columns:
        cursor.execute(
            "ALTER TABLE domaindocs ADD COLUMN archived BOOLEAN DEFAULT FALSE"
        )
        logger.info("Migrated domaindocs: added archived column")

    cursor.execute("PRAGMA table_info(domaindoc_sections)")
    section_columns = {row[1] for row in cursor.fetchall()}

    if "pinned" not in section_columns:
        cursor.execute(
            "ALTER TABLE domaindoc_sections ADD COLUMN pinned BOOLEAN DEFAULT FALSE"
        )
        # Sections that were always expanded (top-level, sort_order 0) become pinned.
        cursor.execute(
            "UPDATE domaindoc_sections SET pinned = TRUE "
            "WHERE sort_order = 0 AND parent_section_id IS NULL"
        )
        logger.info("Migrated domaindoc_sections: added pinned column")

    if "encrypted__summary" not in section_columns:
        cursor.execute(
            "ALTER TABLE domaindoc_sections ADD COLUMN encrypted__summary TEXT DEFAULT NULL"
        )
        logger.info("Migrated domaindoc_sections: added encrypted__summary column")


def _domaindoc_versions_unique(ctx: UserdataStepContext) -> None:
    """
    Rebuild domaindoc_versions with per-section uniqueness.

    Version numbering is per section (cns/api/actions.py:_record_version
    numbers versions with MAX(version_num) filtered by section_id), so the
    uniqueness constraint is (domaindoc_id, section_id, version_num); the
    original (domaindoc_id, version_num) constraint rejected the second
    section's version 1. SQLite cannot ALTER a table constraint, so the
    table is rebuilt once when the old constraint is detected, copying
    existing rows verbatim — no row is renumbered or rewritten.
    """
    from utils.userdata_manager import _DOMAINDOC_DDL

    cursor = ctx.cursor
    cursor.execute("PRAGMA index_list(domaindoc_versions)")
    for index_row in cursor.fetchall():
        index_name = index_row[1]
        is_unique = bool(index_row[2])
        if not is_unique or '"' in index_name:
            continue
        cursor.execute(f'PRAGMA index_info("{index_name}")')
        columns = tuple(row[2] for row in cursor.fetchall())
        if columns != ("domaindoc_id", "version_num"):
            continue
        cursor.execute("ALTER TABLE domaindoc_versions RENAME TO domaindoc_versions_old")
        cursor.execute(_DOMAINDOC_DDL["domaindoc_versions"])
        cursor.execute(
            "INSERT INTO domaindoc_versions "
            "(id, domaindoc_id, section_id, version_num, operation, encrypted__diff_data, created_at) "
            "SELECT id, domaindoc_id, section_id, version_num, operation, "
            "encrypted__diff_data, created_at FROM domaindoc_versions_old ORDER BY id"
        )
        cursor.execute("DROP TABLE domaindoc_versions_old")
        logger.info(
            "Migrated domaindoc_versions: uniqueness is now "
            "(domaindoc_id, section_id, version_num) — per-section version numbering"
        )
        return


def _contacts_columns(ctx: UserdataStepContext) -> None:
    """Add optional contact fields to existing per-user databases."""
    cursor = ctx.cursor
    cursor.execute("PRAGMA table_info(contacts)")
    contact_columns = {row[1] for row in cursor.fetchall()}
    for column in (
        "encrypted__street",
        "encrypted__city",
        "encrypted__state",
        "encrypted__zip",
        "encrypted__pager_address",
    ):
        if column not in contact_columns:
            cursor.execute(f"ALTER TABLE contacts ADD COLUMN {column} TEXT")
            logger.info("Migrated contacts: added %s column", column)


# Append only: a shipped step is never renumbered, edited, or removed.
STEPS: tuple[tuple[int, str, Callable[[UserdataStepContext], None]], ...] = (
    (1, "pager_encrypted_columns", _pager_encrypted_columns),
    (2, "domaindoc_columns", _domaindoc_columns),
    (3, "domaindoc_versions_unique", _domaindoc_versions_unique),
    (4, "contacts_columns", _contacts_columns),
    (5, "pager_plaintext_cleared", _pager_plaintext_cleared),
)
USERDATA_HEAD = STEPS[-1][0]


def stamp_userdata_head(conn: sqlite3.Connection) -> None:
    """Mark a file whose tables were just created from today's DDL as current."""
    conn.execute(f"PRAGMA user_version = {USERDATA_HEAD}")


def run_userdata_migrations(conn: sqlite3.Connection, encrypt_value: Callable[[Any], str]) -> int:
    """
    Apply every step above the file's ``user_version``; returns the version reached.

    ``conn`` must use ``sqlite3.Row`` rows (the manager's connection does).

    Raises:
        RuntimeError: ``conn`` already holds an open transaction (a step's
            work and its version must commit together, alone).
        ValueError: The file is at a version this code does not know.
        Exception: Whatever a step raises; that step is rolled back and the
            file stays at the previous version.
    """
    if conn.in_transaction:
        raise RuntimeError("run_userdata_migrations needs a connection with no open transaction")
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > USERDATA_HEAD:
        raise ValueError(f"userdata.db is at version {current}; this code knows up to {USERDATA_HEAD}")
    for number, name, step in STEPS:
        if number <= current:
            continue
        conn.execute("BEGIN")
        try:
            step(UserdataStepContext(cursor=conn.cursor(), encrypt_value=encrypt_value))
            conn.execute(f"PRAGMA user_version = {number}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            logger.error("userdata step %d (%s) failed; the file stays at version %d", number, name, current)
            raise
        logger.info("userdata step %d (%s) applied", number, name)
        current = number
    return current
