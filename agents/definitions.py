"""User-authored agent definitions (the `agent_definitions` table).

Custom agents are rows, not classes: a template name, an allowlist of
tools, a prompt tophat authored in-session with Mira, and a state. Code
provides the four templates; everything user-specific lives here.

Blacklist: `heartbeat_tool` and `agent_tool` (NEVER_GRANTABLE_TOOLS) can
never be granted, regardless of what a create/update call carries. Agents
do not arm, cancel, or edit heartbeats — the scheduler is owned by the
user and by Mira in-session — and agents do not create other agents.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from utils.timezone_utils import utc_now, format_utc_iso

TEMPLATES = ("investigator", "implementer", "watcher", "general")

# Never grantable to an agent, regardless of request. heartbeat_tool would
# let a spawned agent re-arm or edit its own wake conditions; the gate is
# the one component that must stay user-owned. agent_tool would let a
# spawned agent mint further agents — user-authored agents are created by
# the user in-session, never by other agents.
NEVER_GRANTABLE_TOOLS = frozenset({"heartbeat_tool", "agent_tool"})

MAX_TOPHAT_CHARS = 8000
MAX_NAME_CHARS = 64
MAX_MIN_INTERVAL_SEC = 7 * 24 * 3600
MIN_MIN_INTERVAL_SEC = 60

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")

_DDL = """
    agent_id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    template TEXT NOT NULL,
    display_name TEXT,
    prompt_tophat TEXT NOT NULL,
    tools_json TEXT NOT NULL,
    model_route TEXT,
    max_iterations INTEGER,
    workdir TEXT,
    min_interval_sec INTEGER NOT NULL DEFAULT 3600,
    state TEXT NOT NULL DEFAULT 'active',
    run_count INTEGER NOT NULL DEFAULT 0,
    last_run_at TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
"""
_INDEX = "CREATE INDEX IF NOT EXISTS idx_agent_definitions_state ON agent_definitions(state)"


def ensure_definitions_table(db) -> None:
    db.create_table("agent_definitions", _DDL)
    db.execute(_INDEX)


def _now() -> str:
    return format_utc_iso(utc_now())


def validate_name(name: str) -> str:
    name = (name or "").strip().lower()
    if not _NAME_RE.match(name):
        raise ValueError(
            "name must be 2-64 chars, lowercase letters/digits/underscores, "
            "starting with a letter"
        )
    return name


def validate_template(template: str) -> str:
    template = (template or "").strip().lower()
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of: {', '.join(TEMPLATES)}")
    return template


def validate_tools(db_tool_names: set[str], requested: list[str]) -> list[str]:
    if not isinstance(requested, list) or not requested:
        raise ValueError("tools must be a non-empty list of tool names")
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in requested:
        name = str(raw).strip()
        if not name:
            continue
        if name in NEVER_GRANTABLE_TOOLS:
            raise ValueError(
                f"{name!r} is never grantable to an agent: agents may not arm, "
                f"cancel, or edit heartbeats, and may not create other agents. "
                f"Those surfaces stay with the user."
            )
        if name not in db_tool_names:
            raise ValueError(f"unknown tool {name!r}")
        if name not in seen:
            seen.add(name)
            cleaned.append(name)
    if not cleaned:
        raise ValueError("tools must contain at least one grantable tool")
    return cleaned


def validate_min_interval(value: Any) -> int:
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        raise ValueError("min_interval_sec must be an integer number of seconds")
    if seconds < MIN_MIN_INTERVAL_SEC or seconds > MAX_MIN_INTERVAL_SEC:
        raise ValueError(
            f"min_interval_sec must be between {MIN_MIN_INTERVAL_SEC} and {MAX_MIN_INTERVAL_SEC}"
        )
    return seconds


def validate_tophat(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("prompt_tophat must not be empty")
    if len(text) > MAX_TOPHAT_CHARS:
        raise ValueError(f"prompt_tophat exceeds {MAX_TOPHAT_CHARS} characters")
    return text


def validate_max_iterations(value: Any) -> int | None:
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError("max_iterations must be an integer or null")
    if n < 1 or n > 50:
        raise ValueError("max_iterations must be between 1 and 50")
    return n


def validate_model_route(value: str | None) -> str | None:
    if value is None:
        return None
    route = str(value).strip().lower()
    allowed = {"primary", "fast", "batch", "assessment", "other"}
    if route not in allowed:
        raise ValueError(f"model_route must be one of: {', '.join(sorted(allowed))}")
    return route


def create_definition(
    db,
    *,
    name: str,
    template: str,
    tophat: str,
    tools: list[str],
    tool_names: set[str],
    model_route: str | None = None,
    max_iterations: int | None = None,
    workdir: str | None = None,
    min_interval_sec: int = 3600,
    display_name: str | None = None,
) -> dict:
    ensure_definitions_table(db)
    name = validate_name(name)
    template = validate_template(template)
    tophat = validate_tophat(tophat)
    if template == "general" and not tools:
        # The general template is the catch-all: omitted tools means every
        # grantable tool, not none. Explicit lists are honored as given.
        tools = sorted(set(tool_names) - NEVER_GRANTABLE_TOOLS)
    tools = validate_tools(tool_names, tools)
    model_route = validate_model_route(model_route)
    max_iterations = validate_max_iterations(max_iterations)
    min_interval_sec = validate_min_interval(min_interval_sec)
    existing = db.fetchone("SELECT agent_id FROM agent_definitions WHERE name = :n", {"n": name})
    if existing:
        raise ValueError(f"an agent named {name!r} already exists")
    now = _now()
    agent_id = f"agt_{uuid.uuid4().hex[:16]}"
    db.execute(
        """INSERT INTO agent_definitions
               (agent_id, name, template, display_name, prompt_tophat, tools_json,
                model_route, max_iterations, workdir, min_interval_sec, state,
                run_count, consecutive_failures, created_at, updated_at)
           VALUES (:id, :name, :template, :display, :tophat, :tools,
                   :route, :maxiter, :workdir, :min_interval, 'active', 0, 0, :now, :now)""",
        {
            "id": agent_id, "name": name, "template": template,
            "display": display_name or name, "tophat": tophat,
            "tools": json.dumps(tools), "route": model_route,
            "maxiter": max_iterations, "workdir": (workdir or None),
            "min_interval": min_interval_sec, "now": now,
        },
    )
    return get_definition(db, agent_id)


def get_definition(db, key: str) -> dict | None:
    ensure_definitions_table(db)
    row = db.fetchone(
        "SELECT * FROM agent_definitions WHERE agent_id = :k OR name = :k",
        {"k": key},
    )
    return _row(row) if row else None


def _row(row) -> dict:
    d = dict(row)
    try:
        d["tools"] = json.loads(d.pop("tools_json", "[]") or "[]")
    except json.JSONDecodeError:
        d["tools"] = []
    return d


def list_definitions(db, *, include_retired: bool = True) -> list[dict]:
    ensure_definitions_table(db)
    if include_retired:
        rows = db.execute("SELECT * FROM agent_definitions ORDER BY created_at ASC")
    else:
        rows = db.execute(
            "SELECT * FROM agent_definitions WHERE state != 'retired' ORDER BY created_at ASC"
        )
    return [_row(r) for r in (rows or [])]


def update_definition(db, key: str, changes: dict[str, Any], tool_names: set[str]) -> dict:
    current = get_definition(db, key)
    if not current:
        raise ValueError(f"no agent {key!r}")
    updates: dict[str, Any] = {}
    if "name" in changes:
        updates["name"] = validate_name(changes["name"])
    if "template" in changes:
        updates["template"] = validate_template(changes["template"])
    if "prompt_tophat" in changes:
        updates["prompt_tophat"] = validate_tophat(changes["prompt_tophat"])
    if "tools" in changes:
        updates["tools_json"] = json.dumps(validate_tools(tool_names, changes["tools"]))
    if "display_name" in changes:
        updates["display_name"] = str(changes["display_name"])[:120]
    if "model_route" in changes:
        updates["model_route"] = validate_model_route(changes["model_route"])
    if "max_iterations" in changes:
        updates["max_iterations"] = validate_max_iterations(changes["max_iterations"])
    if "workdir" in changes:
        updates["workdir"] = (str(changes["workdir"]).strip() or None) if changes["workdir"] is not None else None
    if "min_interval_sec" in changes:
        updates["min_interval_sec"] = validate_min_interval(changes["min_interval_sec"])
    if not updates:
        raise ValueError("no recognized fields to update")
    updates["updated_at"] = _now()
    assignments = ", ".join(f"{k} = :{k}" for k in updates)
    updates["id"] = current["agent_id"]
    db.execute(f"UPDATE agent_definitions SET {assignments} WHERE agent_id = :id", updates)
    return get_definition(db, current["agent_id"])


def set_state(db, key: str, state: str) -> dict:
    ensure_definitions_table(db)
    if state not in ("active", "paused", "retired"):
        raise ValueError("state must be active, paused, or retired")
    current = get_definition(db, key)
    if not current:
        raise ValueError(f"no agent {key!r}")
    db.execute(
        "UPDATE agent_definitions SET state = :s, updated_at = :now WHERE agent_id = :id",
        {"s": state, "now": _now(), "id": current["agent_id"]},
    )
    return get_definition(db, current["agent_id"])


def record_run_outcome(db, agent_id: str, *, success: bool, max_consecutive_failures: int = 3) -> dict | None:
    """Update run counters after a run. Auto-pauses after repeated failures."""
    ensure_definitions_table(db)
    row = db.fetchone(
        "SELECT consecutive_failures, state FROM agent_definitions WHERE agent_id = :id",
        {"id": agent_id},
    )
    if not row:
        return None
    failures = 0 if success else int(row.get("consecutive_failures") or 0) + 1
    state = row.get("state") or "active"
    paused_now = False
    if not success and failures >= max_consecutive_failures and state == "active":
        state = "paused"
        paused_now = True
    db.execute(
        "UPDATE agent_definitions SET run_count = run_count + 1, last_run_at = :now, "
        "consecutive_failures = :cf, state = :state, updated_at = :now WHERE agent_id = :id",
        {"now": _now(), "cf": failures, "state": state, "id": agent_id},
    )
    return {"consecutive_failures": failures, "state": state, "auto_paused": paused_now}
