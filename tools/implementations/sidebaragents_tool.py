"""
Sidebar Agents Tool -- Main conversation tool for managing sidebar activity.

Reads the same sidebar_activity and scratchpad SQLite tables that
sidebar_tool writes. Both use self.db (same UserDataManager per user).
"""
import logging
import re
from datetime import datetime
from typing import Dict, Any, TYPE_CHECKING

from pydantic import (
    BaseModel,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)

from agents.base import ensure_activity_schema
from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import utc_now

if TYPE_CHECKING:
    from working_memory.core import WorkingMemory


def _sqlite_now() -> str:
    """UTC timestamp in the same format the table's other writers use
    (SQLite datetime('now')) so TEXT updated_at values sort consistently."""
    return utc_now().strftime("%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)


# -------------------- SCRATCHPAD PARSER --------------------
# Scratchpad notes are model-authored free text (written via
# sidebar_tool.write_note) with no documented structure, so they are
# never returned to the main conversation as raw text. Each DB row is
# parsed into a ScratchpadNoteRecord -- a strict, typed, known shape --
# and any row that does not conform raises loudly with diagnostics
# instead of leaking free-form content into the tool envelope.

# Format used by the table's writers (SQLite datetime('now')).
_SCRATCHPAD_TS_FORMAT = "%Y-%m-%d %H:%M:%S"

# Control characters with no legitimate place in prose notes. Tab,
# newline, and carriage return are allowed; all other C0/C1 controls fail.
_NOTE_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')


class ScratchpadNoteRecord(BaseModel):
    """Known-shape record for one scratchpad note row.

    Strict types throughout: pydantic's default coercion is disabled so
    a malformed row fails validation instead of being silently repaired.
    """

    model_config = {"strict": True}

    note_id: StrictInt
    created_at: StrictStr
    note: StrictStr

    @field_validator("created_at")
    @classmethod
    def _created_at_is_sqlite_datetime(cls, v: str) -> str:
        try:
            datetime.strptime(v, _SCRATCHPAD_TS_FORMAT)
        except ValueError:
            raise ValueError(
                f"created_at {v!r} is not in the SQLite datetime format "
                f"'{_SCRATCHPAD_TS_FORMAT}'"
            )
        return v

    @field_validator("note")
    @classmethod
    def _note_is_prose(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("note is empty or whitespace-only")
        if _NOTE_CONTROL_CHARS.search(v):
            raise ValueError("note contains control characters")
        return v


def _parse_scratchpad_note(
    row: Dict[str, Any], thread_id: str
) -> ScratchpadNoteRecord:
    """Parse one scratchpad DB row into a known shape or fail loudly."""
    try:
        if row.get('thread_id') != thread_id:
            raise ValueError(
                f"row thread_id {row.get('thread_id')!r} does not match "
                f"requested thread {thread_id!r}"
            )
        return ScratchpadNoteRecord(
            note_id=row['id'],
            created_at=row['created_at'],
            note=row['note'],
        )
    except KeyError as exc:
        raise ValueError(
            f"scratchpad row for thread {thread_id!r} is missing field "
            f"{exc}; row keys: {sorted(row.keys())}"
        ) from exc
    except ValidationError as exc:
        raise ValueError(
            f"scratchpad note id {row.get('id', '<unknown>')} for thread "
            f"{thread_id!r} is not parseable into the known shape: {exc}"
        ) from exc


# -------------------- CONFIGURATION --------------------

class SidebarAgentsToolConfig(BaseModel):
    """Configuration for sidebaragents_tool."""
    enabled: bool = Field(
        default=True,
        description="Active in main conversation for managing sidebar activity",
    )


registry.register("sidebaragents_tool", SidebarAgentsToolConfig)


# -------------------- TOOL --------------------

class SidebarAgentsTool(Tool):
    """Review, inspect, and manage sidebar agent activity."""

    name = "sidebaragents_tool"

    tool_schema = {
        "name": "sidebaragents_tool",
        "description": (
            "Review and manage activity from sidebar agents (email, etc.).\n\n"
            "  list_activity(interface_name?): See all active items, "
            "optionally filtered by interface.\n"
            "  get_details(thread_id): Read the agent's working notes "
            "for a specific thread.\n"
            "  dismiss(thread_id): Remove an item from the activity feed.\n"
            "  resolve(thread_id): Mark a thread as resolved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "list_activity",
                        "get_details",
                        "dismiss",
                        "resolve",
                    ],
                },
                "interface_name": {
                    "type": "string",
                    "description": (
                        "Filter by interface. Optional for list_activity. "
                        "Example: 'email_watcher'."
                    ),
                },
                "thread_id": {
                    "type": "string",
                    "description": (
                        "Thread to inspect or act on. Required for "
                        "get_details, dismiss, resolve."
                    ),
                },
            },
            "required": ["operation"],
        },
    }

    def __init__(self, working_memory: 'WorkingMemory'):
        super().__init__()
        self._schema_ensured = False
        self.event_bus = working_memory.event_bus if working_memory else None

    def _ensure_schema(self) -> None:
        if self._schema_ensured:
            return
        ensure_activity_schema(self.db)
        self._schema_ensured = True

    def run(self, **params) -> Dict[str, Any]:
        self._ensure_schema()
        operation = params.get("operation")

        if operation == "list_activity":
            return self._list_activity(params)
        elif operation == "get_details":
            return self._get_details(params)
        elif operation == "dismiss":
            return self._dismiss(params)
        elif operation == "resolve":
            return self._resolve(params)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _list_activity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        interface_name = params.get("interface_name")

        if interface_name:
            rows = self.db.select(
                "sidebar_activity",
                where="interface_name = :iface AND status != 'dismissed'",
                params={'iface': interface_name},
                order_by="updated_at DESC",
            )
        else:
            rows = self.db.select(
                "sidebar_activity",
                where="status != 'dismissed'",
                order_by="updated_at DESC",
            )

        items = [
            {
                "thread_id": r['thread_id'],
                "interface_name": r['interface_name'],
                "agent_id": r['agent_id'],
                "summary": r['summary'],
                "status": r['status'],
                "escalation_reason": r.get('escalation_reason'),
                "run_count": r.get('run_count', 1),
                "updated_at": r['updated_at'],
            }
            for r in rows
        ]
        return {"items": items, "count": len(items)}

    def _get_details(self, params: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = params.get("thread_id")
        if not thread_id:
            raise ValueError("get_details requires thread_id")

        # Activity record
        activity_rows = self.db.select(
            "sidebar_activity",
            where="thread_id = :tid",
            params={'tid': thread_id},
        )
        activity = activity_rows[0] if activity_rows else None

        # Scratchpad content is model-authored free text with no
        # documented format: parse each row into a validated known shape
        # (raising loudly on anything unparseable) rather than returning
        # raw text with escaping applied.
        note_rows = self.db.select(
            "scratchpad",
            where="thread_id = :tid",
            params={'tid': thread_id},
            order_by="created_at ASC",
        )
        notes = [
            _parse_scratchpad_note(r, thread_id).model_dump()
            for r in note_rows
        ]

        result: Dict[str, Any] = {
            "thread_id": thread_id,
            "notes": notes,
        }
        if activity:
            result["activity"] = {
                "summary": activity['summary'],
                "status": activity['status'],
                "escalation_reason": activity.get('escalation_reason'),
                "updated_at": activity['updated_at'],
            }

        return result

    def _dismiss(self, params: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = params.get("thread_id")
        if not thread_id:
            raise ValueError("dismiss requires thread_id")

        rows_updated = self.db.update(
            'sidebar_activity',
            {'status': 'dismissed', 'updated_at': _sqlite_now()},
            'thread_id = :tid',
            {'tid': thread_id},
        )
        if rows_updated == 0:
            raise ValueError(
                f"dismiss failed: no sidebar activity thread '{thread_id}' found"
            )

        self._refresh_trinket()
        return {"success": True, "thread_id": thread_id, "action": "dismissed"}

    def _resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = params.get("thread_id")
        if not thread_id:
            raise ValueError("resolve requires thread_id")

        rows_updated = self.db.update(
            'sidebar_activity',
            {'status': 'resolved', 'updated_at': _sqlite_now()},
            'thread_id = :tid',
            {'tid': thread_id},
        )
        if rows_updated == 0:
            raise ValueError(
                f"resolve failed: no sidebar activity thread '{thread_id}' found"
            )

        self._refresh_trinket()
        return {"success": True, "thread_id": thread_id, "action": "resolved"}

    def _refresh_trinket(self) -> None:
        if self.event_bus is None:
            return
        from agents.base import _publish_trinket_refresh
        _publish_trinket_refresh(self.event_bus)
