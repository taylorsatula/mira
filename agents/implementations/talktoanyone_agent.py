"""
TalkToAnyoneAgent -- synchronous persona-driven consultation on the
SidebarAgent base.

A consult is one LLM call on the fixed `other` route (the designated
outside model), run blocking on the caller's thread via `run_blocking()`.
The complete_task summary is the only channel out of a blocking run, so it
must carry the full in-character response; the activate/continue classmethods
below map it onto the tool result and enforce the non-empty contract.

The persona is supplied by the caller in `subject`: a detailed persona
description followed by the question to ask. The rubric instructs the model
to adopt the persona fully and answer in character, rooted in that persona's
lived experience and knowledge. `build_initial_message` labels prior turns
"Persona" rather than "Outside voice" for the same reason -- the transcript
has to read as one continuous character.

The segment-scoped Valkey thread lives in this module (`create_thread` /
`load_thread` / `save_thread`): key
`talktoanyone:{user_id}:{segment_id}:{thread_id}` with a 24h TTL reset on
every save, owner and segment validated on load. `continue_task` claims a
task_id by loading its thread and returns None when that load raises -- an
unresolvable id must not silently degrade into a fresh ungrounded consult
under the same success envelope. `thread_id` is the consult's task_id and
the WorkItem's item_id, one identifier family, so the `sidebar_activity`
dedup row UPSERTs once per thread.

CROSS-FILE-COUPLING: `phoneafriend_agent.py` carries the same thread store —
record shape, validation order, TTL — under a different key prefix; a change
to this one belongs in both.
"""
import json
import logging
from typing import Any, TypedDict, TYPE_CHECKING
from uuid import uuid4

from agents.base import SidebarAgent, load_agent_prompt
from utils.timezone_utils import format_utc_iso, utc_now
from utils.user_context import get_current_user_id

if TYPE_CHECKING:
    from agents.sidebar import WorkItem
    from cns.integration.event_bus import EventBus
    from tools.repo import ToolRepository

logger = logging.getLogger(__name__)

KEY_PREFIX = "talktoanyone"
THREAD_TTL_SECONDS = 24 * 60 * 60


class ConsultThreadNotFound(ValueError):
    """The task_id names no live consult thread for this user and segment.

    Distinguishes the claim-protocol miss (a legitimate None from
    `continue_task`) from a Valkey outage, which must propagate: a broad
    ValueError catch turns an unreachable store into "not my task_id" and
    hands the caller a not-found envelope for a working thread.
    """


def thread_key(user_id: str, segment_id: str, thread_id: str) -> str:
    return f"{KEY_PREFIX}:{user_id}:{segment_id}:{thread_id}"


def create_thread(user_id: str, segment_id: str) -> dict[str, Any]:
    """Build a fresh, not-yet-persisted consult thread record."""
    now = format_utc_iso(utc_now())
    return {
        "thread_id": uuid4().hex,
        "owner_user_id": user_id,
        "segment_id": segment_id,
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }


def load_thread(user_id: str, segment_id: str, thread_id: str) -> dict[str, Any]:
    """Load a consult thread, validating owner user and segment."""
    from clients.valkey_client import get_valkey_client

    valkey = get_valkey_client()
    raw = valkey.get(thread_key(user_id, segment_id, thread_id))
    if raw is None:
        raise ConsultThreadNotFound(
            f"task_id {thread_id} is not an active talk-to-anyone "
            f"thread in this conversation segment"
        )
    thread = json.loads(raw)
    if thread.get("owner_user_id") != user_id:
        raise ConsultThreadNotFound(
            f"task_id {thread_id} is not an active talk-to-anyone "
            f"thread for the current user"
        )
    if thread.get("segment_id") != segment_id:
        raise ConsultThreadNotFound(
            f"task_id {thread_id} is not an active talk-to-anyone "
            f"thread in this conversation segment"
        )
    return thread


def save_thread(thread: dict[str, Any]) -> None:
    """Persist the thread and reset its TTL."""
    from clients.valkey_client import get_valkey_client

    thread["updated_at"] = format_utc_iso(utc_now())
    get_valkey_client().setex(
        thread_key(
            thread["owner_user_id"], thread["segment_id"], thread["thread_id"]
        ),
        THREAD_TTL_SECONDS,
        json.dumps(thread),
    )


class ConsultContext(TypedDict):
    """Work-item context shape for a talk-to-anyone consult.

    subject: the persona description plus the question to ask, everything
        the outside model needs; it never sees MIRA's context window,
        conversation history, memories, or system prompt.
    prior_messages: earlier turns of this thread, alternating user /
        assistant; empty on a fresh consult.
    """

    subject: str
    prior_messages: list[dict[str, str]]


class TalkToAnyoneAgent(SidebarAgent):
    """Consults a persona-driven outside model; blocking-only."""

    agent_id = "talktoanyone"
    model_config_name = "other"
    available_tools: list[str] = []
    inherit_base_prompt = False
    max_iterations = 1
    blocking_supported = True

    def __init__(self, tool_repo: 'ToolRepository'):
        super().__init__(tool_repo)

    def get_agent_prompt(self, work_item: 'WorkItem') -> str:
        return load_agent_prompt("talktoanyone_system.txt")

    def build_initial_message(self, work_item: 'WorkItem') -> str:
        ctx: ConsultContext = work_item.context  # type: ignore[assignment]
        subject = ctx.get('subject', '')
        prior_messages = ctx.get('prior_messages', [])

        lines = []
        for message in prior_messages:
            speaker = "User" if message.get('role') == 'user' else "Persona"
            lines.append(f"{speaker}: {message.get('content', '')}")
        transcript = "\n\n".join(lines)

        if transcript:
            return (
                f"Prior consult turns:\n\n{transcript}\n\n"
                f"New inquiry:\n\n{subject}"
            )
        return subject

    @classmethod
    def activate(
        cls,
        tool_repo: 'ToolRepository',
        event_bus: 'EventBus',
        *,
        subject: str,
        context: str = '',
        blocking: bool = False,
    ) -> dict[str, Any]:
        from utils.user_context import get_current_segment_id

        segment_id = get_current_segment_id() or "presegment"
        thread = create_thread(get_current_user_id(), segment_id)
        return cls._consult(tool_repo, event_bus, thread, subject)

    @classmethod
    def continue_task(
        cls,
        tool_repo: 'ToolRepository',
        event_bus: 'EventBus',
        *,
        task_id: str,
        subject: str,
        context: str = '',
        blocking: bool = False,
    ) -> dict[str, Any] | None:
        from utils.user_context import get_current_segment_id

        segment_id = get_current_segment_id() or "presegment"
        try:
            thread = load_thread(get_current_user_id(), segment_id, task_id)
        except ConsultThreadNotFound:
            return None
        return cls._consult(tool_repo, event_bus, thread, subject)

    @classmethod
    def _consult(
        cls,
        tool_repo: 'ToolRepository',
        event_bus: 'EventBus',
        thread: dict[str, Any],
        subject: str,
    ) -> dict[str, Any]:
        from agents.sidebar import WorkItem

        work_item = WorkItem(
            item_id=thread["thread_id"],
            interface_name="talktoanyone",
            context={
                "subject": subject,
                "prior_messages": list(thread["messages"]),
            },
        )
        record = cls(tool_repo).run_blocking(work_item, event_bus)

        if record["status"] != "success":
            return {
                "success": False,
                "task_id": thread["thread_id"],
                "status": record["status"],
                "error": record["summary"],
            }
        response = record["summary"].strip()
        if not response:
            raise RuntimeError(
                "the persona returned an empty talk-to-anyone response"
            )

        thread["messages"].append({"role": "user", "content": subject})
        thread["messages"].append({"role": "assistant", "content": response})
        save_thread(thread)

        return {
            "success": True,
            "task_id": thread["thread_id"],
            "response": response,
            "message": (
                f"The persona responded. Reuse task_id "
                f"{thread['thread_id']} with operation 'continue' to keep "
                "consulting this thread."
            ),
        }
