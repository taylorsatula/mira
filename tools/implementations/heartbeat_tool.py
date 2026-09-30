"""
heartbeat_tool -- MIRA's keepsleeping/breakout decision record for the wake cycle.

The heartbeat scheduler (cns/services/heartbeat_service.py) wakes MIRA with a
synthetic stimulus; MIRA records its decision here via `confirm`. The service
reads the decision back from this table after the turn to drive breakout
pushes, and the digest in the next stimulus includes recent decisions.

The tool never wakes anything itself: it is the record-keeping half of the
wake cycle, plus a status view for normal turns.
"""
import inspect
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import format_utc_iso, utc_now
from utils.user_context import get_user_preferences

logger = logging.getLogger(__name__)

_VALID_DECISIONS = ("keepsleeping", "breakout")

# Single DDL source for the heartbeat log. The tool creates it on first use;
# cns/services/heartbeat_service.py calls ensure_heartbeat_log() before its
# read paths (digest, decision readback, state) so ordering never matters.
HEARTBEAT_LOG_DDL = """
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    wake_in_seconds INTEGER,
    created_at TEXT NOT NULL
"""

HEARTBEAT_LOG_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_heartbeat_log_tick ON heartbeat_log(tick_id)"
)

# Minimum accepted wake_in_seconds. Below this a "sleep" is a tight retry loop
# that burns a full LLM turn per iteration; the default interval (300s) is the
# floor for meaningful quiet periods.
_MIN_WAKE_SECONDS = 60


def ensure_heartbeat_log(db) -> None:
    """Create the heartbeat log table and index if absent (idempotent).

    Also adds wake_in_seconds to a table created before that column existed;
    the duplicate-column error from an already-migrated table is the expected
    no-op path."""
    db.create_table("heartbeat_log", HEARTBEAT_LOG_DDL)
    try:
        db.execute(
            "ALTER TABLE heartbeat_log ADD COLUMN wake_in_seconds INTEGER"
        )
    except Exception as exc:
        if "duplicate column name" not in str(exc):
            raise
    db.execute(HEARTBEAT_LOG_INDEX_DDL)


class HeartbeatToolConfig(BaseModel):
    """Configuration for heartbeat_tool."""

    enabled: bool = Field(default=True, description="Whether the heartbeat tool is enabled")
    status_limit: int = Field(default=10, ge=1, le=50, description="Max records returned by 'status'")


registry.register("heartbeat_tool", HeartbeatToolConfig)


class HeartbeatTool(Tool):
    """Record heartbeat wake-cycle decisions and expose their history."""

    name = "heartbeat_tool"
    simple_description = "records keepsleeping/breakout heartbeat decisions"
    parallel_safe = False  # confirm writes; sequential by default

    _parallel_safe_operations = frozenset({"status"})

    @classmethod
    def is_call_parallel_safe(cls, tool_input: Dict[str, Any]) -> bool:
        return tool_input.get("operation") in cls._parallel_safe_operations

    tool_schema = {
        "name": "heartbeat_tool",
        "description": "Record the keepsleeping/breakout decision for a heartbeat wake tick.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["confirm", "status"],
                    "description": "Operation to perform."
                },
                "tick_id": {
                    "type": "string",
                    "description": "Tick identifier in hb_ form, taken verbatim from the heartbeat stimulus message. Required for 'confirm'; ignored by 'status'."
                },
                "decision": {
                    "type": "string",
                    "enum": ["keepsleeping", "breakout"],
                    "description": "keepsleeping: nothing needs attention, end the turn after confirming. breakout: something needs attention or action, continue the turn. Required for 'confirm'; ignored by 'status'."
                },
                "reason": {
                    "type": "string",
                    "description": "One-line justification for the decision, under 300 characters; longer values are rejected. Required for 'confirm'; ignored by 'status'."
                },
                "wake_in_seconds": {
                    "type": "integer",
                    "description": "With decision 'keepsleeping' only — providing it with 'breakout' raises an error: sleep this many seconds before the next wake, with no heartbeat ticks in between. Minimum 60; values above the configured ceiling are rejected. Omit to sleep the default interval. Only for 'confirm'."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum records returned by 'status', 1-50, default 10. Only for 'status'."
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    }

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        from utils.user_context import has_user_context
        if has_user_context():
            self._ensure_tables()

    def _ensure_tables(self) -> None:
        ensure_heartbeat_log(self.db)

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        try:
            self._ensure_tables()
            handlers = {
                "confirm": self._confirm,
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

    def _confirm(
        self, tick_id: str, decision: str, reason: str,
        wake_in_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        from config.config_manager import config

        if not tick_id or not tick_id.startswith("hb_"):
            raise ValueError(
                "tick_id is required for 'confirm' and must be the hb_ identifier "
                "from the heartbeat stimulus message"
            )
        if decision not in _VALID_DECISIONS:
            raise ValueError(
                f"decision must be one of {', '.join(_VALID_DECISIONS)}, got: {decision}"
            )
        if not reason or not reason.strip():
            raise ValueError("reason is required for 'confirm' and must not be blank")
        if len(reason) > 300:
            raise ValueError(f"reason exceeds 300 characters (got {len(reason)})")

        requested_sleep: Optional[int] = None
        if wake_in_seconds is not None:
            if decision != "keepsleeping":
                raise ValueError(
                    "wake_in_seconds only applies with decision 'keepsleeping'; "
                    "a breakout turn is already awake"
                )
            max_sleep = config.heartbeat.max_sleep_seconds
            if not isinstance(wake_in_seconds, int) or wake_in_seconds < _MIN_WAKE_SECONDS:
                raise ValueError(
                    f"wake_in_seconds must be an integer of at least {_MIN_WAKE_SECONDS} "
                    f"seconds, got: {wake_in_seconds!r}"
                )
            if wake_in_seconds > max_sleep:
                raise ValueError(
                    f"wake_in_seconds exceeds the configured ceiling of {max_sleep} "
                    f"seconds ({max_sleep / 3600:.0f}h), got: {wake_in_seconds}"
                )
            requested_sleep = wake_in_seconds

        self.db.insert("heartbeat_log", {
            "tick_id": tick_id,
            "decision": decision,
            "reason": reason.strip(),
            "wake_in_seconds": requested_sleep,
            "created_at": format_utc_iso(utc_now()),
        })

        if decision == "keepsleeping":
            if requested_sleep is not None:
                guidance = (
                    f"Decision recorded. Sleeping for {requested_sleep} seconds with "
                    "no heartbeat ticks in between. End the turn now; your final "
                    "message must be exactly the single word: keepsleeping"
                )
            else:
                guidance = (
                    "Decision recorded. Nothing needs attention. End the turn now; "
                    "your final message must be exactly the single word: keepsleeping"
                )
        else:
            prefs = get_user_preferences()
            display_name = (prefs.first_name or "").strip() or "friend"
            guidance = (
                f"Decision recorded. Continue this turn normally: act on what needs "
                f"attention, then end with the message {display_name} should see. Write it "
                f"to stand alone; {display_name} may not have been watching."
            )
        return {
            "success": True,
            "decision": decision,
            "wake_in_seconds": requested_sleep,
            "message": guidance,
        }

    def _status(self, limit: Optional[int] = None) -> Dict[str, Any]:
        from config.config_manager import config

        max_records = config.heartbeat_tool.status_limit
        if limit is not None:
            max_records = max(1, min(int(limit), 50))
        rows = self.db.fetchall(
            "SELECT tick_id, decision, reason, wake_in_seconds, created_at FROM heartbeat_log "
            "ORDER BY created_at DESC LIMIT ?",
            (max_records,),
        )
        return {
            "success": True,
            "records": rows,
            "count": len(rows),
            "wake_mode": config.heartbeat.wake_mode,
            "interval_seconds": config.heartbeat.interval_seconds,
            "max_sleep_seconds": config.heartbeat.max_sleep_seconds,
        }
