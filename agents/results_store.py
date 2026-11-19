"""Durable ledger for agent runs.

`agent_results` is the source of truth for "what did the agents report".
The SubagentResultsTrinket renders from it; collapse no longer destroys
the only copy of a result. Rows age out of the default view after
`ARCHIVE_AFTER_DAYS` days (or once seen and older than the window) but are
never deleted; `recall` surfaces them forever.

Encrypted columns go through the manager's encrypting accessors
(`db.insert` / `db._decrypt_dict`), never raw SQL — a raw write puts
plaintext into an `encrypted__` column and a raw read returns ciphertext.

All functions take the per-user SQLite handle from the caller (they run
inside agent threads and the trinket, both of which have one).
"""
from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

from utils.timezone_utils import utc_now, format_utc_iso

ARCHIVE_AFTER_DAYS = 14

_DDL = """
    run_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    status TEXT NOT NULL,
    subject TEXT,
    task_id TEXT,
    encrypted__summary TEXT,
    encrypted__payload TEXT,
    created_at TEXT NOT NULL,
    seen_at TEXT,
    archived_at TEXT
"""
_INDEX_AGENT = "CREATE INDEX IF NOT EXISTS idx_agent_results_agent ON agent_results(agent, created_at)"
_INDEX_ACTIVE = "CREATE INDEX IF NOT EXISTS idx_agent_results_active ON agent_results(seen_at, archived_at, created_at)"


def ensure_results_table(db) -> None:
    db.create_table("agent_results", _DDL)
    db.execute(_INDEX_AGENT)
    db.execute(_INDEX_ACTIVE)


def record_run(
    db,
    *,
    agent: str,
    status: str,
    summary: str,
    subject: str | None = None,
    task_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> str:
    ensure_results_table(db)
    run_id = uuid.uuid4().hex
    db.insert("agent_results", {
        "run_id": run_id,
        "agent": agent,
        "status": status,
        "subject": subject,
        "task_id": task_id,
        "encrypted__summary": (summary or "")[:4000],
        "encrypted__payload": payload,
        "created_at": format_utc_iso(utc_now()),
    })
    maybe_archive_sweep(db)
    return run_id


_last_sweep = 0.0


def maybe_archive_sweep(db, *, force: bool = False) -> int:
    """Archive rows past the retention window. Never deletes.

    Called opportunistically from record_run, at most once an hour per
    process unless forced. Returns the number of rows archived.
    """
    global _last_sweep
    import time

    if not force and (time.time() - _last_sweep) < 3600:
        return 0
    _last_sweep = time.time()
    ensure_results_table(db)
    cutoff = format_utc_iso(utc_now() - timedelta(days=ARCHIVE_AFTER_DAYS))
    rows = db.execute(
        "SELECT run_id FROM agent_results "
        "WHERE archived_at IS NULL "
        "  AND (created_at < :cutoff OR (seen_at IS NOT NULL AND seen_at < :cutoff))",
        {"cutoff": cutoff},
    )
    run_ids = [r["run_id"] for r in rows]
    if not run_ids:
        return 0
    now = format_utc_iso(utc_now())
    for i in range(0, len(run_ids), 200):
        chunk = run_ids[i:i + 200]
        placeholders = ", ".join(f":p{j}" for j in range(len(chunk)))
        params = {f"p{j}": rid for j, rid in enumerate(chunk)}
        params["now"] = now
        db.execute(
            f"UPDATE agent_results SET archived_at = :now WHERE run_id IN ({placeholders})",
            params,
        )
    return len(run_ids)


def _decrypt_rows(db, rows: list[dict]) -> list[dict]:
    return [db._decrypt_dict(row) for row in rows]


def unarchived(db, *, per_agent_limit: int | None = None) -> list[dict]:
    """Non-archived rows, newest first per agent."""
    ensure_results_table(db)
    rows = db.execute(
        "SELECT * FROM agent_results WHERE archived_at IS NULL "
        "ORDER BY agent ASC, created_at DESC"
    )
    results = _decrypt_rows(db, rows)
    if per_agent_limit is not None:
        by_agent: dict[str, list[dict]] = {}
        for row in results:
            bucket = by_agent.setdefault(row["agent"], [])
            if len(bucket) < per_agent_limit:
                bucket.append(row)
        results = [row for bucket in by_agent.values() for row in bucket]
    return results


def counts_by_agent(db) -> dict[str, dict[str, int]]:
    """{agent: {"unarchived": n, "archived": n}} across the whole ledger."""
    ensure_results_table(db)
    rows = db.execute(
        "SELECT agent, "
        "  SUM(CASE WHEN archived_at IS NULL THEN 1 ELSE 0 END) AS unarchived, "
        "  SUM(CASE WHEN archived_at IS NOT NULL THEN 1 ELSE 0 END) AS archived "
        "FROM agent_results GROUP BY agent"
    )
    return {
        r["agent"]: {"unarchived": int(r["unarchived"]), "archived": int(r["archived"])}
        for r in rows
    }


def mark_seen(db, run_ids: list[str]) -> None:
    if not run_ids:
        return
    ensure_results_table(db)
    now = format_utc_iso(utc_now())
    for i in range(0, len(run_ids), 200):
        chunk = run_ids[i:i + 200]
        placeholders = ", ".join(f":p{j}" for j in range(len(chunk)))
        params = {f"p{j}": rid for j, rid in enumerate(chunk)}
        params["now"] = now
        db.execute(
            f"UPDATE agent_results SET seen_at = :now "
            f"WHERE run_id IN ({placeholders}) AND seen_at IS NULL",
            params,
        )


def list_archived(db, agent: str | None = None, limit: int = 50) -> list[dict]:
    ensure_results_table(db)
    if agent:
        rows = db.execute(
            "SELECT * FROM agent_results WHERE archived_at IS NOT NULL AND agent = :a "
            "ORDER BY created_at DESC LIMIT :n",
            {"a": agent, "n": limit},
        )
    else:
        rows = db.execute(
            "SELECT * FROM agent_results WHERE archived_at IS NOT NULL "
            "ORDER BY created_at DESC LIMIT :n",
            {"n": limit},
        )
    return _decrypt_rows(db, rows)


def recall(db, run_id: str) -> dict | None:
    """Fetch one row by run_id, archived or not. The permanent read path."""
    ensure_results_table(db)
    row = db.fetchone("SELECT * FROM agent_results WHERE run_id = :r", {"r": run_id})
    return db._decrypt_dict(row) if row else None


def archive_by_task_id(db, task_id: str) -> int:
    """Archive every unarchived row for a work-item id (dismiss path)."""
    ensure_results_table(db)
    rows = db.execute(
        "SELECT run_id FROM agent_results WHERE task_id = :t AND archived_at IS NULL",
        {"t": task_id},
    )
    ids = [r["run_id"] for r in rows]
    if not ids:
        return 0
    now = format_utc_iso(utc_now())
    for i in range(0, len(ids), 200):
        chunk = ids[i:i + 200]
        placeholders = ", ".join(f":p{j}" for j in range(len(chunk)))
        params = {f"p{j}": rid for j, rid in enumerate(chunk)}
        params["now"] = now
        db.execute(
            f"UPDATE agent_results SET archived_at = :now WHERE run_id IN ({placeholders})",
            params,
        )
    return len(ids)


def unarchive(db, run_id: str) -> bool:
    """Clear archived_at so the row returns to the default view.

    Returns True when the row exists (whether or not it was archived).
    """
    ensure_results_table(db)
    row = db.fetchone("SELECT run_id FROM agent_results WHERE run_id = :r", {"r": run_id})
    if row is None:
        return False
    db.execute(
        "UPDATE agent_results SET archived_at = NULL WHERE run_id = :r",
        {"r": run_id},
    )
    return True


def recent_for_agent(db, agent: str, limit: int = 5) -> list[dict]:
    """Newest unarchived rows for one agent, newest first."""
    ensure_results_table(db)
    rows = db.execute(
        "SELECT * FROM agent_results WHERE agent = :a AND archived_at IS NULL "
        "ORDER BY created_at DESC LIMIT :n",
        {"a": agent, "n": limit},
    )
    return _decrypt_rows(db, rows)
