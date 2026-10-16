"""
Subagent Tool — one surface for subagent work: activate, continue, dismiss,
inspect, resolve.

One invocation pattern over one task_id family for every subagent surface:
the synchronous outside-model consult, background research, and the
sidebar-activity lifecycle. The tool owns the schema and
config, verb routing, generic param validation, spec-flag switches, envelope
mapping, and the activity-management surface; every per-type mechanic (sync
vs async dispatch, WorkItem shape, blocking-cap application, refine gating,
result prose) lives in the agent classes and is reached through the spec
table's lazy class resolution — the dispatch protocol documented in
agents/specs.py.

Background spawns go through agents/spawn.py:spawn_subagent
(process-global cap); blocking runs call SidebarAgent.run_blocking()
directly — they take no in-flight slot and their result is this tool's
return value.

The activity-management operations read/write the same sidebar_activity
and scratchpad SQLite tables sidebar_tool writes, through self.db (same
per-user UserDataManager).
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
from agents.specs import SUBAGENT_SPECS
from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import utc_now

if TYPE_CHECKING:
    from tools.repo import ToolRepository
    from working_memory.core import WorkingMemory


def _sqlite_now() -> str:
    """UTC timestamp in the same format the table's other writers use
    (SQLite datetime('now')) so TEXT updated_at values sort consistently."""
    return utc_now().strftime("%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)

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


class SubagentToolConfig(BaseModel):
    """Configuration for subagent_tool."""

    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled",
    )
    forage_blocking_max_iterations: int = Field(
        default=12,
        description=(
            "Iteration cap for a blocking forage run. The class's 20-iteration "
            "cap is calibrated for the 600s background budget; a blocking run "
            "gets this tighter cap so it winds down and briefs inside the "
            "blocking wall-clock bound. Applied by ForageAgent's dispatch "
            "classmethods"
        ),
    )


registry.register("subagent_tool", SubagentToolConfig)

_INVOCABLE_AGENT_TYPES = [
    agent_type for agent_type, spec in SUBAGENT_SPECS.items() if spec.invocable
]


class SubagentTool(Tool):
    """Start, continue, and manage subagent work through one surface."""

    name = "subagent_tool"

    simple_description = """
    Start and manage subagent work: consult an outside model synchronously
    (phoneafriend), dispatch background research (forage, whilethecatsaway),
    continue or refine prior work, and inspect or close sidebar agent activity.
    """

    tool_schema = {
        "name": "subagent_tool",
        "description": (
            "Start, continue, and manage subagent work. One task_id family covers "
            "every surface.\n\n"
            "  activate (agent_type + subject): start work. phoneafriend consults "
            "an outside model synchronously — the full response returns in this "
            "tool result, and its task_id resumes the same consult thread. forage "
            "researches across conversations, memories, and web — async by default "
            "(results appear in your context window when ready) or blocking=true. "
            "whilethecatsaway explores a topic at leisure in the background and "
            "stores findings as memories.\n"
            "  continue (task_id + subject): resume — continue a phoneafriend "
            "consult thread with a follow-up question, or refine a completed "
            "forage result (the prior result is dismissed and passed to the new "
            "agent; subject is the refinement instruction).\n"
            "  dismiss (task_id): remove a result from your context window and "
            "mark its activity row dismissed.\n"
            "  inspect (task_id?, interface_name?): list sidebar agent activity, "
            "or with task_id read one thread's details and scratchpad notes.\n"
            "  resolve (task_id): mark an activity thread resolved — "
            "escalation-lifecycle closure, distinct from dismiss."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "activate",
                        "continue",
                        "dismiss",
                        "inspect",
                        "resolve",
                        "memory_inspect",
                        "memory_update",
                        "memory_deprecate_all",
                    ],
                    "description": (
                        "The action to perform. activate and continue start or "
                        "resume subagent work; dismiss, inspect, and resolve "
                        "manage activity and results."
                    ),
                },
                "agent_type": {
                    "type": "string",
                    "enum": _INVOCABLE_AGENT_TYPES,
                    "description": (
                        "activate only: which subagent type to start. "
                        "phoneafriend = synchronous outside-model consult; "
                        "forage = research across conversations, memories, and "
                        "web; whilethecatsaway = open-ended background research "
                        "stored as memories."
                    ),
                },
                "task_id": {
                    "type": "string",
                    "description": (
                        "Target identifier — one family for every subagent "
                        "surface. activate (forage/whilethecatsaway) returns a "
                        "fresh task_id; continue, dismiss, and inspect-with-"
                        "task_id target a task_id you received from this tool "
                        "or one shown in a <subagent_results> entry; resolve "
                        "targets an activity thread_id from the activity feed."
                    ),
                },
                "subject": {
                    "type": "string",
                    "description": (
                        "The work itself. activate: the inquiry (phoneafriend — "
                        "include every fact the outside model needs; it sees "
                        "only this and the thread's prior turns, never the main "
                        "conversation) or the research target (forage/"
                        "whilethecatsaway — be specific, it drives the search "
                        "strategy). continue: the follow-up question "
                        "(phoneafriend) or the refinement instruction (forage)."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Conversational motivation, for forage and "
                        "whilethecatsaway (activate and continue): what you and "
                        "the user are discussing and why this information would "
                        "be useful. The subagent uses it to judge relevance. "
                        "Not read by phoneafriend — put everything in subject."
                    ),
                },
                "blocking": {
                    "type": "boolean",
                    "description": (
                        "Run synchronously and return the result inside this "
                        "tool result instead of your context window later. "
                        "forage only — activate, or continue refining a forage "
                        "result. Wall-clock bounded by the blocking agent "
                        "timeout. Use when you need the findings within this "
                        "turn."
                    ),
                },
                "interface_name": {
                    "type": "string",
                    "description": (
                        "inspect only, optional: filter the activity list to "
                        "this interface. Example: 'forage'."
                    ),
                },
            },
            "required": ["operation"],
        },
    }

    # Read-modify-write over a shared identity: a continue consumes an entry
    # (auto-dismiss + respawn), a dismiss mutates entry + activity row.
    # Everything else is fresh-identity or pure-read.
    _SEQUENTIAL_OPERATIONS = frozenset({'continue', 'dismiss'})

    def __init__(
        self,
        tool_repo: 'ToolRepository',
        working_memory: 'WorkingMemory',
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.tool_repo = tool_repo
        self.working_memory = working_memory
        self._schema_ensured = False

        self.event_bus = None
        if self.working_memory:
            self.event_bus = self.working_memory.event_bus

    @classmethod
    def is_call_parallel_safe(cls, tool_input: Dict[str, Any]) -> bool:
        return tool_input.get('operation') not in cls._SEQUENTIAL_OPERATIONS

    def _ensure_schema(self) -> None:
        if self._schema_ensured:
            return
        ensure_activity_schema(self.db)
        self._schema_ensured = True

    def run(self, **kwargs) -> Dict[str, Any]:
        operation = kwargs.get("operation")

        if operation == "activate":
            return self._op_activate(kwargs)
        elif operation == "continue":
            return self._op_continue(kwargs)
        elif operation == "dismiss":
            self._ensure_schema()
            return self._op_dismiss(kwargs)
        elif operation == "inspect":
            self._ensure_schema()
            return self._op_inspect(kwargs)
        elif operation == "resolve":
            self._ensure_schema()
            return self._op_resolve(kwargs)
        elif operation == "memory_inspect":
            self._ensure_schema()
            return self._op_memory_inspect(kwargs)
        elif operation == "memory_update":
            self._ensure_schema()
            return self._op_memory_update(kwargs)
        elif operation == "memory_deprecate_all":
            self._ensure_schema()
            return self._op_memory_deprecate_all(kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # activate

    def _op_activate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from agents.specs import SUBAGENT_SPECS, load_agent_class

        agent_type = params.get("agent_type")
        subject = (params.get("subject") or "").strip()
        if not agent_type:
            raise ValueError("activate requires agent_type")
        spec = SUBAGENT_SPECS.get(agent_type)
        if spec is None or not spec.invocable:
            raise ValueError(
                f"activate: unknown or non-invocable agent_type {agent_type!r}"
            )
        if not subject:
            raise ValueError("activate requires subject")

        agent_class = load_agent_class(spec)
        return agent_class.activate(
            self.tool_repo,
            self.event_bus,
            subject=subject,
            context=params.get("context", ""),
            blocking=bool(params.get("blocking", False)),
        )

    # continue

    def _op_continue(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from agents.specs import SUBAGENT_SPECS, load_agent_class

        task_id = params.get("task_id")
        subject = (params.get("subject") or "").strip()
        if not task_id:
            raise ValueError("continue requires task_id")
        if not subject:
            raise ValueError(
                "continue requires subject (the follow-up or refinement)"
            )

        delegate = dict(
            task_id=task_id,
            subject=subject,
            context=params.get("context", ""),
            blocking=bool(params.get("blocking", False)),
        )

        entry = self._active_result(task_id)
        if entry is not None:
            agent_type = entry.get("data", {}).get("agent_type")
            spec = SUBAGENT_SPECS.get(agent_type)
            if spec is None:
                return {
                    "success": False,
                    "task_id": task_id,
                    "message": (
                        f"Continue failed: task {task_id[:8]} carries "
                        f"unknown agent_type {agent_type!r} — no continue "
                        "path exists."
                    ),
                }
            if not spec.continuable:
                return {
                    "success": False,
                    "task_id": task_id,
                    "message": (
                        f"Continue refused: {agent_type} results cannot be "
                        "continued. Wait for the result to arrive (it "
                        "expires on its own), or dismiss it and activate "
                        "a fresh task."
                    ),
                }
            result = load_agent_class(spec).continue_task(
                self.tool_repo, self.event_bus, **delegate
            )
            if result is not None:
                return result
        else:
            # No unified-trinket entry: the remaining continuable surface is
            # a blocking-only type's own task store (the consult thread),
            # reached generically through the spec flags.
            for spec in SUBAGENT_SPECS.values():
                if spec.blocking_only and spec.continuable:
                    result = load_agent_class(spec).continue_task(
                        self.tool_repo, self.event_bus, **delegate
                    )
                    if result is not None:
                        return result

        return {
            "success": False,
            "task_id": task_id,
            "message": (
                f"Continue failed: task {task_id[:8]} is not an active "
                "subagent result — it may be from a previous segment, "
                "dismissed, or expired. Start a fresh activate."
            ),
        }

    def _active_result(self, task_id: str) -> Dict[str, Any] | None:
        """Look up task_id in the unified results trinket, if registered."""
        if not self.working_memory:
            return None
        trinket = self.working_memory.get_trinket("SubagentResultsTrinket")
        if trinket is None:
            return None
        return trinket.active_results.get(task_id)

    # dismiss

    def _op_dismiss(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from agents.specs import SUBAGENT_SPECS
        from agents.spawn import publish_result

        task_id = params.get("task_id")
        if not task_id:
            raise ValueError("dismiss requires task_id")

        entry = self._active_result(task_id)
        entry_dismissed = False
        if entry is not None:
            agent_type = entry.get("data", {}).get("agent_type")
            spec = SUBAGENT_SPECS.get(agent_type)
            if spec is None or not spec.dismissable:
                return {
                    "success": False,
                    "task_id": task_id,
                    "message": (
                        f"Dismiss refused: {agent_type} results are not "
                        "dismissable."
                    ),
                }
            publish_result(self.event_bus, agent_type, task_id, "dismissed", {})
            entry_dismissed = True

        rows_updated = self.db.update(
            'sidebar_activity',
            {'status': 'dismissed', 'updated_at': _sqlite_now()},
            'thread_id = :tid',
            {'tid': task_id},
        )
        self._refresh_trinket()

        if rows_updated == 0 and not entry_dismissed:
            # The trinket removes only a present entry, so dismissing an
            # absent/stale task_id must not report success.
            return {
                "success": False,
                "task_id": task_id,
                "error": "not found",
            }
        return {
            "success": True,
            "task_id": task_id,
            "action": "dismissed",
            "message": f"Result {task_id[:8]} dismissed.",
        }

    # inspect

    def _op_inspect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task_id = params.get("task_id")
        if task_id:
            return self._get_details(task_id)
        return self._list_activity(params)

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

    def _get_details(self, task_id: str) -> Dict[str, Any]:
        activity_rows = self.db.select(
            "sidebar_activity",
            where="thread_id = :tid",
            params={'tid': task_id},
        )
        activity = activity_rows[0] if activity_rows else None

        note_rows = self.db.select(
            "scratchpad",
            where="thread_id = :tid",
            params={'tid': task_id},
            order_by="created_at ASC",
        )
        notes = [
            _parse_scratchpad_note(r, task_id).model_dump()
            for r in note_rows
        ]

        result: Dict[str, Any] = {
            "thread_id": task_id,
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

    # resolve

    def _op_resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        task_id = params.get("task_id")
        if not task_id:
            raise ValueError("resolve requires task_id")

        rows_updated = self.db.update(
            'sidebar_activity',
            {'status': 'resolved', 'updated_at': _sqlite_now()},
            'thread_id = :tid',
            {'tid': task_id},
        )
        if rows_updated == 0:
            raise ValueError(
                f"resolve failed: no sidebar activity thread '{task_id}' found"
            )

        self._refresh_trinket()
        return {"success": True, "thread_id": task_id, "action": "resolved"}

    # agent memory management (main context surface)

    def _op_memory_inspect(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        agent_type = params.get("agent_type") or params.get("agent_id")
        if not agent_type:
            raise ValueError("memory_inspect requires agent_type")

        rows = self.db.select(
            "agent_memory",
            where="agent_id = :agent_id",
            params={"agent_id": agent_type},
            order_by="created_at DESC",
        )
        items = []
        for row in rows:
            items.append(
                {
                    "id": row.get("id"),
                    "agent_id": row.get("agent_id"),
                    "content": row.get("encrypted__content"),
                    "justification": row.get("encrypted__justification"),
                    "status": row.get("status"),
                    "source_thread_id": row.get("source_thread_id"),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                }
            )
        return {"items": items, "count": len(items)}

    def _op_memory_update(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        memory_id = params.get("memory_id")
        if memory_id is None:
            raise ValueError("memory_update requires memory_id")

        data: Dict[str, Any] = {}
        if params.get("content") is not None:
            content = str(params["content"]).strip()
            if not content:
                raise ValueError("content cannot be empty")
            data["encrypted__content"] = content
        if params.get("justification") is not None:
            justification = str(params["justification"]).strip()
            if not justification:
                raise ValueError("justification cannot be empty")
            data["encrypted__justification"] = justification
        if params.get("status") is not None:
            status = str(params["status"])
            if status not in ("active", "deprecated"):
                raise ValueError("status must be active or deprecated")
            data["status"] = status
        if not data:
            raise ValueError(
                "memory_update requires at least one updatable field"
            )
        data["updated_at"] = _sqlite_now()

        updated = self.db.update(
            "agent_memory",
            data,
            "id = :mid",
            {"mid": int(memory_id)},
        )
        if updated == 0:
            raise ValueError(
                f"memory_update failed: memory {memory_id} not found"
            )
        self._refresh_trinket()
        return {
            "success": True,
            "memory_id": int(memory_id),
            "updated_rows": updated,
        }

    def _op_memory_deprecate_all(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        agent_type = params.get("agent_type") or params.get("agent_id")
        if not agent_type:
            raise ValueError(
                "memory_deprecate_all requires agent_type"
            )

        updated = self.db.update(
            "agent_memory",
            {
                "status": "deprecated",
                "updated_at": _sqlite_now(),
            },
            "agent_id = :agent_id",
            {"agent_id": agent_type},
        )
        self._refresh_trinket()
        return {
            "success": True,
            "agent_id": agent_type,
            "deprecated_rows": updated,
        }

    # shared publishers

    def _refresh_trinket(self) -> None:
        if self.event_bus is None:
            return
        from agents.base import _publish_trinket_refresh
        _publish_trinket_refresh(self.event_bus)
