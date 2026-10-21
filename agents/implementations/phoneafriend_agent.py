"""
PhoneAFriendAgent -- synchronous outside-model consultation on the
SidebarAgent base.

A consult is one LLM call on the fixed `other` route (the designated
outside model), run blocking on the caller's thread via `run_blocking()`.
The complete_task summary is the only channel out of a blocking run, so it
must carry the full response; the activate/continue classmethods below map
it onto the tool result and enforce the non-empty contract.

The consult protocol lives here (the dispatch protocol from
agents/specs.py): `activate` resolves the segment, creates the thread, and
runs one consult; `continue_task` claims a task_id by loading its thread
(owner+segment validated) and returns None when the id is not an active
consult thread. The segment-scoped Valkey thread also lives in this module
(create_thread / load_thread / save_thread): key
`phoneafriend:{user_id}:{segment_id}:{thread_id}`, 24h TTL, owner and
segment validated on load. `thread_id` is the consult's task_id and the
WorkItem's item_id — one identifier family — so the `sidebar_activity` dedup
row (UNIQUE(interface_name, thread_id)) UPSERTs once per thread and updates
per consult.
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

KEY_PREFIX = "phoneafriend"
THREAD_TTL_SECONDS = 24 * 60 * 60


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
    """Load a consult thread, validating owner user and segment.

    Raises ValueError when the thread is absent in this segment or belongs
    to another user — an unresolvable task_id must not silently degrade
    into a fresh ungrounded consult under the same success envelope.
    """
    from clients.valkey_client import get_valkey_client

    valkey = get_valkey_client()
    raw = valkey.get(thread_key(user_id, segment_id, thread_id))
    if raw is None:
        raise ValueError(
            f"task_id {thread_id} is not an active phone-a-friend "
            f"thread in this conversation segment"
        )
    thread = json.loads(raw)
    if thread.get("owner_user_id") != user_id:
        raise ValueError(
            f"task_id {thread_id} is not an active phone-a-friend "
            f"thread for the current user"
        )
    if thread.get("segment_id") != segment_id:
        raise ValueError(
            f"task_id {thread_id} is not an active phone-a-friend "
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
    """Work-item context shape for a phone-a-friend consult.

    subject: the inquiry — the full question with every fact the outside
        model needs; it never sees MIRA's context window, conversation
        history, memories, or system prompt.
    prior_messages: earlier turns of this thread, alternating user /
        assistant; empty on a fresh consult.
    """

    subject: str
    prior_messages: list[dict[str, str]]


class PhoneAFriendAgent(SidebarAgent):
    """Consults the outside model; blocking-only."""

    agent_id = "phoneafriend"
    model_config_name = "other"
    available_tools: list[str] = []
    inherit_base_prompt = False
    max_iterations = 1
    blocking_supported = True

    def __init__(self, tool_repo: 'ToolRepository'):
        super().__init__(tool_repo)

    def get_agent_prompt(self, work_item: 'WorkItem') -> str:
        return load_agent_prompt("phoneafriend_system.txt")

    def build_initial_message(self, work_item: 'WorkItem') -> str:
        ctx: ConsultContext = work_item.context  # type: ignore[assignment]
        subject = ctx.get('subject', '')
        prior_messages = ctx.get('prior_messages', [])

        lines = []
        for message in prior_messages:
            speaker = "User" if message.get('role') == 'user' else "Outside voice"
            lines.append(f"{speaker}: {message.get('content', '')}")
        transcript = "\n\n".join(lines)

        if transcript:
            return (
                f"Prior consult turns:\n\n{transcript}\n\n"
                f"New inquiry:\n\n{subject}"
            )
        return subject

    # The dispatch protocol (agents/specs.py): subagent_tool reaches these
    # classmethods through the spec table; they own the whole consult
    # mechanics — segment resolution, thread lifecycle, blocking run,
    # envelope mapping, and this type's refusal prose.

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
        """Start a fresh consult thread and run it blocking.

        `context` and `blocking` are accepted for protocol uniformity and
        ignored: the consult is always synchronous, and everything the
        outside model sees must arrive in `subject`.
        """
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
        """Resume the consult thread named by task_id.

        Returns None when the id does not resolve to an active thread for
        this user and segment (the claim protocol) — an unresolvable id must
        not silently degrade into a fresh ungrounded consult under the same
        success envelope.
        """
        from utils.user_context import get_current_segment_id

        segment_id = get_current_segment_id() or "presegment"
        try:
            thread = load_thread(get_current_user_id(), segment_id, task_id)
        except ValueError:
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
        """Run one consult blocking; save the exchange on success.

        run_blocking returns failures, never raises — the status check comes
        first so a failed consult is never read as a success.
        """
        from agents.sidebar import WorkItem

        work_item = WorkItem(
            item_id=thread["thread_id"],
            interface_name="phoneafriend",
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
                "the outside model returned an empty phone-a-friend response"
            )

        thread["messages"].append({"role": "user", "content": subject})
        thread["messages"].append({"role": "assistant", "content": response})
        save_thread(thread)

        return {
            "success": True,
            "task_id": thread["thread_id"],
            "response": response,
            "message": (
                f"The outside model responded. Reuse task_id "
                f"{thread['thread_id']} with operation 'continue' to keep "
                "consulting this thread."
            ),
        }
