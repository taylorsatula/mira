"""
heartbeat_tool -- briefs: MIRA's authored wake-up-with-a-mission system.

A brief is a persisted payload the scheduler injects at a wake time:
type ('one-off' | 'scheduled'), mission, condition_check, bail_if, and a
schedule. The heartbeat trigger surfaces due briefs to the HeartbeatAgent
(agents/implementations/heartbeat_agent.py), which evaluates bail_if first,
then the condition, executes the mission when the condition holds, and
records each brief's outcome via the 'report' operation. A run that dies
without reporting never leaves its briefs silently armed: the agent's
on_completion passes them to `apply_run_failure`, which backs off and
retires them once the failure count reaches max_failures.

Storage split:
- Brief content lives in per-user SQLite (this tool's store, same database
  as reminders), with the user-authored text columns encrypted at rest via
  the encrypted__ prefix convention.
- The Postgres heartbeat_armed_users table carries only (user_id,
  earliest_wake_at) presence so the sidebar dispatcher can enumerate users
  with armed briefs (dormant users included) and the device-power binding
  can sleep the metal between wakes. State lives in SQLite; Postgres
  carries presence, not truth — `refresh_armed_marker` recomputes it after
  every state transition.
"""

import inspect
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import format_utc_iso, parse_utc_time_string, utc_now

logger = logging.getLogger(__name__)

BRIEFS_DDL = """
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK (type IN ('one-off', 'scheduled')),
    encrypted__mission TEXT NOT NULL,
    encrypted__condition_check TEXT NOT NULL,
    encrypted__bail_if TEXT,
    schedule TEXT NOT NULL,
    interval_seconds INTEGER NOT NULL DEFAULT 600,
    state TEXT NOT NULL DEFAULT 'armed'
        CHECK (state IN ('armed', 'done', 'bailed', 'failed_permanently', 'cancelled')),
    next_wake_at TEXT NOT NULL,
    consecutive_false INTEGER NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    encrypted__last_result TEXT,
    last_wake_at TEXT,
    author TEXT NOT NULL DEFAULT 'mira',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
"""

BRIEFS_STATE_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_briefs_state_wake ON briefs(state, next_wake_at)"
)

_MIN_INTERVAL_SECONDS = 60
# Decay/backoff both use base * 2^count capped at this exponent, then the max_interval_seconds config ceiling applies on top.
_MAX_DECAY_EXPONENT = 3
# Sub-daily scheduled cadence: the `schedule` column stores 'every:<seconds>'
# for every-briefs (daily briefs store their 'HH:MM' wall time). The cap keeps
# every-briefs off daily_at's territory; a wall-clock daily or slower schedule
# is daily_at's job. _MIN_INTERVAL_SECONDS bounds the floor.
_EVERY_PREFIX = "every:"
_MAX_EVERY_SECONDS = 23 * 3600
_NOTE_MAX_CHARS = 2000
_MISSION_MAX_CHARS = 4000


def ensure_briefs_table(db) -> None:
    """Create the briefs table and index if absent. Idempotent."""
    db.create_table("briefs", BRIEFS_DDL)
    db.execute(BRIEFS_STATE_INDEX_DDL)


def _brief_row_public(row: Dict[str, Any]) -> Dict[str, Any]:
    """Shape a raw briefs row for tool/API responses (id, state, timing, notes)."""
    return {
        "id": row.get("id"),
        "type": row.get("type"),
        "state": row.get("state"),
        "schedule": row.get("schedule"),
        "next_wake_at": row.get("next_wake_at"),
        "interval_seconds": row.get("interval_seconds"),
        "consecutive_false": row.get("consecutive_false"),
        "consecutive_failures": row.get("consecutive_failures"),
        "mission": row.get("encrypted__mission"),
        "condition_check": row.get("encrypted__condition_check"),
        "bail_if": row.get("encrypted__bail_if"),
        "last_result": row.get("encrypted__last_result"),
        "last_wake_at": row.get("last_wake_at"),
        "author": row.get("author"),
        "created_at": row.get("created_at"),
    }



def upsert_armed_user(user_id: str, wake_at_utc: str) -> None:
    """Record that this user has at least one armed brief due at wake_at_utc.

    Called on schedule and on every re-arm; the row carries the earliest
    pending wake so the dispatcher's eligibility union and the device-power
    binding can sleep until then.
    """
    from utils.database_session_manager import get_shared_session_manager

    session = get_shared_session_manager()
    with session.get_admin_session() as admin:
        admin.execute_query(
            """
            INSERT INTO heartbeat_armed_users (user_id, earliest_wake_at, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                earliest_wake_at = EXCLUDED.earliest_wake_at,
                updated_at = NOW()
            """,
            (user_id, wake_at_utc),
        )


def remove_armed_user(user_id: str) -> None:
    """Drop the armed-brief presence row (last armed brief left terminal state)."""
    from utils.database_session_manager import get_shared_session_manager

    session = get_shared_session_manager()
    with session.get_admin_session() as admin:
        admin.execute_query("DELETE FROM heartbeat_armed_users WHERE user_id = %s", (user_id,))


def refresh_armed_marker(db, user_id: str) -> None:
    """Recompute the Postgres presence row from the user's armed briefs.

    Called after any state transition: upserts the earliest armed wake time,
    or removes the row when nothing is armed.
    """
    rows = db.fetchall(
        "SELECT next_wake_at FROM briefs WHERE state = 'armed' ORDER BY next_wake_at ASC"
    )
    if rows:
        upsert_armed_user(user_id, rows[0]["next_wake_at"])
    else:
        remove_armed_user(user_id)



def _parse_wake_at(wake_at: str, daily_at: Optional[str], every_hours: Optional[float] = None) -> datetime:
    """Resolve a wake time: one-off takes wake_at (ISO or 'in N seconds');
    scheduled takes daily_at ('HH:MM', user-local, wall-clock daily) or
    every_hours (sub-daily fixed cadence) and fires next."""
    from utils.user_context import get_user_preferences

    now = utc_now()
    if daily_at is not None:
        import re

        match = re.fullmatch(r"(\d{1,2}):(\d{2})", daily_at.strip())
        if not match:
            raise ValueError(
                f"daily_at must be 'HH:MM' (24h, user-local), got: {daily_at!r}"
            )
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour > 23 or minute > 59:
            raise ValueError(f"daily_at out of range: {daily_at!r}")
        tz_name = get_user_preferences().timezone
        # DST-strict resolution, the codebase's sanctioned pair for
        # user-supplied exact wall times (reminder_tool, punchclock_tool):
        # local_datetime_to_utc_iso raises on ambiguous/nonexistent wall
        # times instead of silently guessing. Compute the candidate date in
        # user-local terms (today, rolling to tomorrow when the moment has
        # passed), then resolve that wall time strictly.
        from utils.timezone_utils import local_datetime_to_utc_iso, convert_from_utc

        local_now = convert_from_utc(now, tz_name)
        candidate_date = local_now.date()
        if (hour, minute) <= (local_now.hour, local_now.minute):
            candidate_date += timedelta(days=1)
        wall = f"{candidate_date.isoformat()}T{hour:02d}:{minute:02d}:00"
        return parse_utc_time_string(local_datetime_to_utc_iso(wall, tz_name))
    if every_hours is not None:
        try:
            hours = float(every_hours)
        except (TypeError, ValueError):
            raise ValueError(f"every_hours must be a number, got: {every_hours!r}")
        seconds = int(hours * 3600)
        if seconds < _MIN_INTERVAL_SECONDS or seconds > _MAX_EVERY_SECONDS:
            raise ValueError(
                f"every_hours must be between {_MIN_INTERVAL_SECONDS / 3600:g}h and "
                f"{_MAX_EVERY_SECONDS / 3600:g}h (>= 24h schedules use daily_at), "
                f"got: {every_hours!r}"
            )
        return now + timedelta(seconds=seconds)
    if wake_at is None:
        raise ValueError("one-off briefs require wake_at (ISO 8601 or 'in N seconds')")
    stripped = wake_at.strip()
    if stripped.lower().startswith("in "):
        try:
            seconds = int(stripped[3:].strip().split()[0])
        except (ValueError, IndexError):
            raise ValueError(f"could not parse relative wake_at: {wake_at!r}")
        if seconds <= 0:
            raise ValueError("relative wake_at must be positive seconds")
        return now + timedelta(seconds=seconds)
    try:
        parsed = parse_utc_time_string(stripped)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"could not parse wake_at {wake_at!r} as ISO 8601 or 'in N seconds': {exc}"
        )
    return parsed


def _next_recurring_occurrence(schedule: str) -> str:
    """Compute a scheduled brief's next wake as a UTC ISO timestamp.

    Dispatches on the stored schedule value: 'every:<seconds>' re-arms at
    now + seconds (a fixed cadence, deliberately not wall-clock-aligned);
    anything else is a daily_at 'HH:MM' wall time.
    """
    if schedule.startswith(_EVERY_PREFIX):
        seconds = int(schedule[len(_EVERY_PREFIX):])
        return format_utc_iso(utc_now() + timedelta(seconds=seconds))
    return format_utc_iso(_parse_wake_at(None, schedule))


def _clamped_interval(base_seconds: int, count: int) -> int:
    """interval * 2^min(count, cap), clamped by config ceiling."""
    from config.config_manager import config

    seconds = base_seconds * (2 ** min(count, _MAX_DECAY_EXPONENT))
    return min(seconds, config.heartbeat.max_interval_seconds)


def _max_rearm_seconds() -> int:
    """Ceiling for an agent-requested re-arm (report wake_in_seconds)."""
    from config.config_manager import config

    return int(config.heartbeat.max_interval_seconds)



class HeartbeatToolConfig(BaseModel):
    """Configuration for heartbeat_tool."""

    enabled: bool = Field(default=True, description="Whether the heartbeat tool is enabled")


registry.register("heartbeat_tool", HeartbeatToolConfig)


# Deliberate narrowing: HeartbeatAgent sees report only — never schedule, cancel,
# or inspect other briefs mid-run.
HEARTBEAT_AGENT_SCHEMA = {
    "name": "heartbeat_tool",
    "description": (
        "Record the outcome of a heartbeat brief you were dispatched to execute. "
        "Call once per brief carried by this work item, before complete_task."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "const": "report",
                "description": "Report a brief outcome.",
            },
            "brief_id": {
                "type": "string",
                "description": "The brf_ identifier of the brief, from the work item payload.",
            },
            "outcome": {
                "type": "string",
                "enum": ["fired", "false", "bailed"],
                "description": (
                    "fired: the condition held and you executed the mission. "
                    "false: the condition did not hold (re-arms on schedule, "
                    "optionally sooner via wake_in_seconds). "
                    "bailed: the bail_if condition held; the brief is moot, "
                    "report it without executing the mission."
                ),
            },
            "note": {
                "type": "string",
                "description": (
                    "What you found and did. Stored as the brief's last_result and "
                    "shown to the user, so write it to stand alone."
                ),
            },
            "wake_in_seconds": {
                "type": "integer",
                "description": (
                    "With outcome 'false' only: re-arm in this many seconds instead "
                    "of the decayed interval, minimum 60."
                ),
            },
        },
        "required": ["operation", "brief_id", "outcome", "note"],
        "additionalProperties": False,
    },
}


class HeartbeatTool(Tool):
    """Author, inspect, and report briefs (heartbeat wake-ups with missions)."""

    name = "heartbeat_tool"
    simple_description = "schedules briefs (wake-ups with a mission) and records their outcomes"
    parallel_safe = False  # briefs writes; sequential by default

    _parallel_safe_operations = frozenset({"list", "status"})

    @classmethod
    def is_call_parallel_safe(cls, tool_input: Dict[str, Any]) -> bool:
        return tool_input.get("operation") in cls._parallel_safe_operations

    tool_schema = {
        "name": "heartbeat_tool",
        "description": (
            "Schedule and manage briefs: persisted wake-ups carrying a mission, a "
            "condition to check at wake, and (one-offs) a bail_if early-exit. "
            "At the wake time the HeartbeatAgent is dispatched with the brief's "
            "payload: it evaluates bail_if first, then the condition, and executes "
            "the mission only when the condition holds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["schedule", "list", "cancel", "report", "status"],
                    "description": "Operation to perform.",
                },
                "type": {
                    "type": "string",
                    "enum": ["one-off", "scheduled"],
                    "description": (
                        "Brief type, for 'schedule'. one-off: fires once until its "
                        "condition holds, then dies (bail_if required). scheduled: "
                        "recurs perpetually via daily_at (>= 24h wall-clock) or "
                        "every_hours (sub-daily cadence), bail_if optional."
                    ),
                },
                "mission": {
                    "type": "string",
                    "description": (
                        "What to do when the condition holds. Self-contained: the "
                        "agent sees only this text and the condition, never the "
                        "conversation that authored them."
                    ),
                },
                "condition_check": {
                    "type": "string",
                    "description": (
                        "The check to perform at wake. Write it as a verifiable "
                        "predicate the agent can evaluate with tools."
                    ),
                },
                "bail_if": {
                    "type": "string",
                    "description": (
                        "Early-exit condition, checked BEFORE the condition at every "
                        "wake. When it holds, the brief terminates without running the "
                        "mission. Required for one-off briefs; optional for scheduled."
                    ),
                },
                "wake_at": {
                    "type": "string",
                    "description": (
                        "One-off only: first wake time. ISO 8601 datetime or 'in N "
                        "seconds'."
                    ),
                },
                "daily_at": {
                    "type": "string",
                    "description": (
                        "Scheduled only: 'HH:MM' in the user's local timezone, daily. "
                        "Mutually exclusive with every_hours."
                    ),
                },
                "every_hours": {
                    "type": "number",
                    "description": (
                        "Scheduled only: fixed cadence in hours, between 1 minute "
                        "(0.0167) and 23h. Fires every N hours from scheduling, not "
                        "wall-clock-aligned. Mutually exclusive with daily_at; use "
                        "daily_at for daily-or-slower wall-clock schedules."
                    ),
                },
                "interval_seconds": {
                    "type": "integer",
                    "description": (
                        "Re-arm cadence when the condition is false, seconds (default "
                        "600, minimum 60). Sustained false results decay toward "
                        "max_interval_seconds."
                    ),
                },
                "brief_id": {
                    "type": "string",
                    "description": (
                        "brf_ identifier, for 'cancel' and 'report'."
                    ),
                },
                "outcome": {
                    "type": "string",
                    "enum": ["fired", "false", "bailed"],
                    "description": "Brief outcome, for 'report'.",
                },
                "note": {
                    "type": "string",
                    "description": (
                        "Outcome note ('report') or reasoning ('schedule'/'cancel', "
                        "optional). For 'report', what was found and done; shown to "
                        "the user, so write it to stand alone."
                    ),
                },
                "state_filter": {
                    "type": "string",
                    "description": "Optional state filter for 'list': armed|done|bailed|failed_permanently|cancelled.",
                },
            },
            "required": ["operation"],
            "additionalProperties": False,
        },
    }

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        from utils.user_context import has_user_context

        if has_user_context():
            ensure_briefs_table(self.db)

    def _ensure_tables(self) -> None:
        ensure_briefs_table(self.db)

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        try:
            self._ensure_tables()
            handlers = {
                "schedule": self._schedule,
                "list": self._list,
                "cancel": self._cancel,
                "report": self._report,
                "status": self._status,
            }
            method = handlers.get(operation)
            if method is None:
                raise ValueError(
                    f"Unknown operation: {operation}. Valid: {', '.join(handlers)}"
                )
            accepted = set(inspect.signature(method).parameters)
            return method(**{k: v for k, v in kwargs.items() if k in accepted})
        except Exception as e:
            self.logger.error(f"Error in {operation}: {e}")
            raise

    def _schedule(
        self,
        type: Optional[str] = None,
        mission: Optional[str] = None,
        condition_check: Optional[str] = None,
        bail_if: Optional[str] = None,
        wake_at: Optional[str] = None,
        daily_at: Optional[str] = None,
        every_hours: Optional[float] = None,
        interval_seconds: Optional[int] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        from config.config_manager import config
        from utils.user_context import get_current_user_id

        if type not in ("one-off", "scheduled"):
            raise ValueError("type must be 'one-off' or 'scheduled'")
        if not mission or not mission.strip():
            raise ValueError("mission is required and must not be blank")
        if len(mission) > _MISSION_MAX_CHARS:
            raise ValueError(f"mission exceeds {_MISSION_MAX_CHARS} characters (got {len(mission)})")
        if not condition_check or not condition_check.strip():
            raise ValueError("condition_check is required and must not be blank")
        if len(condition_check) > _MISSION_MAX_CHARS:
            raise ValueError(
                f"condition_check exceeds {_MISSION_MAX_CHARS} characters (got {len(condition_check)})"
            )
        if type == "one-off" and (not bail_if or not bail_if.strip()):
            raise ValueError("one-off briefs require a non-empty bail_if")
        if bail_if and len(bail_if) > _MISSION_MAX_CHARS:
            raise ValueError(f"bail_if exceeds {_MISSION_MAX_CHARS} characters (got {len(bail_if)})")

        # Enforce per-type inputs before _parse_wake_at can AttributeError on
        # None; the one-off cross-pairings are rejected too — a one-off must
        # name its one wake moment, not borrow the scheduled machinery.
        if type == "one-off" and not wake_at:
            raise ValueError("one-off briefs require wake_at (ISO 8601 or 'in N seconds')")
        if type == "one-off" and daily_at:
            raise ValueError("one-off briefs take wake_at, not daily_at")
        if type == "one-off" and every_hours is not None:
            raise ValueError("one-off briefs take wake_at, not every_hours")
        if daily_at and every_hours is not None:
            raise ValueError("daily_at and every_hours are mutually exclusive")
        if type == "scheduled" and daily_at is None and every_hours is None:
            raise ValueError(
                "scheduled briefs require daily_at ('HH:MM' user-local) "
                "or every_hours (sub-daily cadence)"
            )

        base_interval = (
            interval_seconds
            if interval_seconds is not None
            else (int(every_hours * 3600) if type == "scheduled" and every_hours is not None
                  else config.heartbeat.default_interval_seconds)
        )
        if not isinstance(base_interval, int) or base_interval < _MIN_INTERVAL_SECONDS:
            raise ValueError(
                f"interval_seconds must be an integer >= {_MIN_INTERVAL_SECONDS}, got: {base_interval!r}"
            )

        # Resolve first wake time. A one-off's schedule field is its first
        # wake ISO timestamp; a scheduled brief's is its 'HH:MM' local time.
        wake_dt = _parse_wake_at(wake_at, daily_at, every_hours)
        if type == "one-off":
            schedule_value = format_utc_iso(wake_dt)
        elif every_hours is not None:
            schedule_value = f"{_EVERY_PREFIX}{int(float(every_hours) * 3600)}"
        else:
            schedule_value = daily_at.strip()

        brief_id = f"brf_{uuid4().hex[:12]}"
        now_iso = format_utc_iso(utc_now())
        row = {
            "id": brief_id,
            "type": type,
            "encrypted__mission": mission.strip(),
            "encrypted__condition_check": condition_check.strip(),
            "encrypted__bail_if": bail_if.strip() if bail_if else None,
            "schedule": schedule_value,
            "interval_seconds": base_interval,
            "state": "armed",
            "next_wake_at": format_utc_iso(wake_dt),
            "consecutive_false": 0,
            "consecutive_failures": 0,
            "encrypted__last_result": None,
            "last_wake_at": None,
            "author": "mira",
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        self.db.insert("briefs", row)
        refresh_armed_marker(self.db, str(get_current_user_id()))

        return {
            "success": True,
            "brief_id": brief_id,
            "type": type,
            "next_wake_at": row["next_wake_at"],
            "interval_seconds": base_interval,
            "message": (
                f"Brief {brief_id} armed ({type}). The HeartbeatAgent will wake "
                f"with its payload at {row['next_wake_at']}"
                + (
                    f" (every {every_hours}h)" if type == "scheduled" and every_hours is not None
                    else f" (daily at {schedule_value} local)" if type == "scheduled" else ""
                )
                + "."
            ),
        }

    def _list(self, state_filter: Optional[str] = None) -> Dict[str, Any]:
        valid_states = ("armed", "done", "bailed", "failed_permanently", "cancelled")
        if state_filter is not None and state_filter not in valid_states:
            raise ValueError(f"state_filter must be one of {valid_states}, got: {state_filter!r}")
        if state_filter:
            rows = self.db.fetchall(
                "SELECT * FROM briefs WHERE state = :state ORDER BY next_wake_at ASC",
                {"state": state_filter},
            )
        else:
            rows = self.db.fetchall("SELECT * FROM briefs ORDER BY next_wake_at ASC")
        return {
            "success": True,
            "briefs": [_brief_row_public(r) for r in rows],
            "count": len(rows),
        }

    def _status(self) -> Dict[str, Any]:
        rows = self.db.fetchall(
            "SELECT id, state, encrypted__last_result, last_wake_at, updated_at "
            "FROM briefs WHERE state IN ('done', 'bailed', 'failed_permanently') "
            "ORDER BY updated_at DESC LIMIT 10"
        )
        return {
            "success": True,
            "recent_outcomes": [
                {
                    "id": r.get("id"),
                    "state": r.get("state"),
                    "last_result": r.get("encrypted__last_result"),
                    "last_wake_at": r.get("last_wake_at"),
                    "updated_at": r.get("updated_at"),
                }
                for r in rows
            ],
            "count": len(rows),
        }

    def _cancel(self, brief_id: Optional[str] = None, note: Optional[str] = None) -> Dict[str, Any]:
        from utils.user_context import get_current_user_id

        if not brief_id:
            raise ValueError("brief_id is required for 'cancel'")
        row = self.db.fetchone("SELECT * FROM briefs WHERE id = :id", {"id": brief_id})
        if row is None:
            raise ValueError(f"No brief with id {brief_id}")
        now_iso = format_utc_iso(utc_now())
        self.db.execute(
            "UPDATE briefs SET state = 'cancelled', "
            "encrypted__last_result = :note, updated_at = :now WHERE id = :id",
            {"note": note or "cancelled by user/mira", "now": now_iso, "id": brief_id},
        )
        refresh_armed_marker(self.db, str(get_current_user_id()))
        return {
            "success": True,
            "brief_id": brief_id,
            "state": "cancelled",
            "message": f"Brief {brief_id} cancelled. It will not wake again.",
        }

    def _report(
        self,
        brief_id: Optional[str] = None,
        outcome: Optional[str] = None,
        note: Optional[str] = None,
        wake_in_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        from utils.user_context import get_current_user_id

        if not brief_id:
            raise ValueError("brief_id is required for 'report'")
        if outcome not in ("fired", "false", "bailed"):
            raise ValueError("outcome must be 'fired', 'false', or 'bailed'")
        if note is not None and len(note) > _NOTE_MAX_CHARS:
            raise ValueError(f"note exceeds {_NOTE_MAX_CHARS} characters (got {len(note)})")

        row = self.db.fetchone("SELECT * FROM briefs WHERE id = :id", {"id": brief_id})
        if row is None:
            raise ValueError(f"No brief with id {brief_id}")
        if row.get("state") != "armed":
            raise ValueError(
                f"Brief {brief_id} is '{row.get('state')}', not armed; refusing to report on it"
            )

        user_id = str(get_current_user_id())
        now = utc_now()
        now_iso = format_utc_iso(now)
        base_interval = int(row.get("interval_seconds") or 600)

        if outcome == "bailed":
            self.db.execute(
                "UPDATE briefs SET state = 'bailed', encrypted__last_result = :note, "
                "last_wake_at = :now, updated_at = :now WHERE id = :id",
                {"note": note or "bail_if held; brief is moot", "now": now_iso, "id": brief_id},
            )
            refresh_armed_marker(self.db, user_id)
            return {
                "success": True,
                "brief_id": brief_id,
                "state": "bailed",
                "message": f"Brief {brief_id} bailed. Its exit note: {note or 'bail_if held'}",
            }

        if outcome == "fired":
            if row.get("type") == "one-off":
                self.db.execute(
                    "UPDATE briefs SET state = 'done', encrypted__last_result = :note, "
                    "last_wake_at = :now, updated_at = :now WHERE id = :id",
                    {"note": note or "condition held; mission executed", "now": now_iso, "id": brief_id},
                )
                refresh_armed_marker(self.db, user_id)
                return {
                    "success": True,
                    "brief_id": brief_id,
                    "state": "done",
                    "message": f"Brief {brief_id} fired and is complete.",
                }
            # scheduled: re-arm at the next occurrence, reset counters
            next_wake = _next_recurring_occurrence(row.get("schedule"))
            self.db.execute(
                "UPDATE briefs SET encrypted__last_result = :note, next_wake_at = :nw, "
                "consecutive_false = 0, consecutive_failures = 0, last_wake_at = :now, "
                "updated_at = :now WHERE id = :id",
                {"note": note, "nw": next_wake, "now": now_iso, "id": brief_id},
            )
            refresh_armed_marker(self.db, user_id)
            return {
                "success": True,
                "brief_id": brief_id,
                "state": "armed",
                "next_wake_at": next_wake,
                "message": f"Brief {brief_id} fired; re-armed for {next_wake}.",
            }

        false_count = int(row.get("consecutive_false") or 0) + 1
        if wake_in_seconds is not None:
            if not isinstance(wake_in_seconds, int) or wake_in_seconds < _MIN_INTERVAL_SECONDS:
                raise ValueError(
                    f"wake_in_seconds must be an integer >= {_MIN_INTERVAL_SECONDS}, got: {wake_in_seconds!r}"
                )
            if wake_in_seconds > _max_rearm_seconds():
                raise ValueError(
                    f"wake_in_seconds exceeds the ceiling of {_max_rearm_seconds()} "
                    f"seconds (config.heartbeat.max_interval_seconds); rejected, not clamped"
                )
            delay = wake_in_seconds
        elif str(row.get("schedule") or "").startswith(_EVERY_PREFIX):
            # every-brief: the cadence IS the schedule. The false-path decay
            # exists to back off condition-wait briefs; applying it here would
            # clamp max_interval_seconds (default 4800s) UNDER a sub-daily
            # cadence and silently shorten the fixed loop. Keep the cadence.
            delay = base_interval
        else:
            delay = _clamped_interval(base_interval, false_count)
        next_wake = format_utc_iso(now + timedelta(seconds=delay))
        self.db.execute(
            "UPDATE briefs SET encrypted__last_result = :note, next_wake_at = :nw, "
            "consecutive_false = :cf, last_wake_at = :now, updated_at = :now WHERE id = :id",
            {
                "note": note or "condition did not hold",
                "nw": next_wake,
                "cf": false_count,
                "now": now_iso,
                "id": brief_id,
            },
        )
        refresh_armed_marker(self.db, user_id)
        return {
            "success": True,
            "brief_id": brief_id,
            "state": "armed",
            "next_wake_at": next_wake,
            "consecutive_false": false_count,
            "message": (
                f"Brief {brief_id} re-armed for {next_wake} "
                f"(decay step {false_count})."
            ),
        }



def apply_run_failure(brief_ids: List[str], failure_note: str) -> List[Dict[str, Any]]:
    """Advance the failure state machine for briefs whose agent run died
    without reporting them (failed/timed out/capped/rejected).

    Called by the HeartbeatAgent's on_completion with every carried brief
    that received no valid report. Backoff is interval * 2^failures; at
    max_failures the brief goes failed_permanently with an exit note.
    """
    from config.config_manager import config
    from utils.user_context import get_current_user_id

    from utils.userdata_manager import get_user_data_manager

    user_id = str(get_current_user_id())
    db = get_user_data_manager(user_id)
    ensure_briefs_table(db)

    max_failures = config.heartbeat.max_failures
    now = utc_now()
    now_iso = format_utc_iso(now)
    results = []
    for brief_id in brief_ids:
        row = db.fetchone("SELECT * FROM briefs WHERE id = :id", {"id": brief_id})
        if row is None or row.get("state") != "armed":
            continue
        failures = int(row.get("consecutive_failures") or 0) + 1
        base_interval = int(row.get("interval_seconds") or 600)
        if failures >= max_failures:
            db.execute(
                "UPDATE briefs SET state = 'failed_permanently', "
                "encrypted__last_result = :note, consecutive_failures = :cf, "
                "last_wake_at = :now, updated_at = :now WHERE id = :id",
                {
                    "note": f"{failure_note} (gave up after {failures} consecutive failed runs)",
                    "cf": failures,
                    "now": now_iso,
                    "id": brief_id,
                },
            )
            results.append({"brief_id": brief_id, "state": "failed_permanently"})
        elif str(row.get("schedule") or "").startswith(_EVERY_PREFIX):
            # same fixed-cadence rule as the false re-arm: failure backoff must
            # not clamp a sub-daily cadence below itself
            next_wake = format_utc_iso(now + timedelta(seconds=base_interval))
        else:
            delay = _clamped_interval(base_interval, failures)
            next_wake = format_utc_iso(now + timedelta(seconds=delay))
            db.execute(
                "UPDATE briefs SET encrypted__last_result = :note, "
                "consecutive_failures = :cf, next_wake_at = :nw, last_wake_at = :now, "
                "updated_at = :now WHERE id = :id",
                {
                    "note": failure_note,
                    "cf": failures,
                    "nw": next_wake,
                    "now": now_iso,
                    "id": brief_id,
                },
            )
            results.append({"brief_id": brief_id, "state": "armed", "next_wake_at": next_wake})
    refresh_armed_marker(db, user_id)
    return results
