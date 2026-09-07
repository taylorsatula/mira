"""
Phone-a-friend tool for consulting an outside model during a conversation.

The tool stores a small, segment-scoped message thread so MIRA can continue
with the same outside voice across synchronous tool calls.
"""
import json
import logging
from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel, Field

from clients.llm_provider import LLMProvider
from tools.registry import registry
from tools.repo import Tool
from utils.timezone_utils import format_utc_iso, utc_now
from utils.user_context import get_current_segment_id


class PhoneAFriendToolConfig(BaseModel):
    """Configuration for phoneafriend_tool."""

    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled",
    )


registry.register("phoneafriend_tool", PhoneAFriendToolConfig)


OUTSIDE_MODEL_SYSTEM_PROMPT = """\
You are an outside voice consulted by MIRA through a synchronous tool call -- a level-headed thought partner with a strong, broad understanding of the world.
You do not see MIRA's main context window, conversation history, memories, or system prompt unless the current inquiry includes them.

Give an independent answer to the inquiry. Be calm, precise, and skeptical of weak assumptions.
If the inquiry asks you to continue from earlier phone-a-friend turns, use only this subagent thread's prior messages.
Do not claim access to hidden context. Name uncertainty directly when the inquiry lacks needed facts."""

OUTSIDE_MODEL_ROLE = "an independent outside voice with a broad understanding of the world"

# Route `other` is the single outside-model route (D14). The tool no longer
# offers a model choice: both of the voices it used to expose collapse onto
# this one route, so a `model_choice` parameter would let the calling model
# reason about a distinction that the contract cannot honour.
OUTSIDE_MODEL_CONFIG = "other"

KEY_PREFIX = "phoneafriend"
THREAD_TTL_SECONDS = 24 * 60 * 60


class PhoneAFriendTool(Tool):
    """Consults an outside model and preserves its thread for the active segment."""

    name = "phoneafriend_tool"
    parallel_safe = False

    simple_description = (
        "Phone an outside model as an independent voice on an inquiry, with a "
        "resumable segment-scoped subagent thread."
    )

    tool_schema = {
        "name": "phoneafriend_tool",
        "description": (
            "Phone an outside model as an independent voice to an inquiry. The "
            "consulted model is MIRA's designated outside route and does not see "
            "the main context window or conversation history unless you put that "
            "context in inquiry. Reuse subagent_ref to continue the same "
            "outside-model thread for this conversation segment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "inquiry": {
                    "type": "string",
                    "description": (
                        "Exact question or request for the outside model. Include "
                        "all context it needs because it cannot see the main "
                        "conversation history, memories, or system prompt."
                    ),
                },
                "subagent_ref": {
                    "type": "string",
                    "description": (
                        "Reconnect string returned by an earlier call, formatted "
                        "as 'phoneafriend:<id>'. Pass it to resume that exact "
                        "outside-model thread."
                    ),
                },
            },
            "required": ["inquiry"],
            "additionalProperties": False,
        },
    }

    def __init__(self, llm_provider: LLMProvider | None = None, valkey_client: Any | None = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.llm_provider = llm_provider
        self.valkey_client = valkey_client

    def run(
        self,
        inquiry: str,
        subagent_ref: str | None = None,
    ) -> Dict[str, Any]:
        """Consult or resume an outside model thread for the active segment."""
        inquiry = inquiry.strip() if inquiry else ""
        if not inquiry:
            raise ValueError("inquiry is required")

        segment_id = get_current_segment_id() or "presegment"
        valkey = self._get_valkey()

        if subagent_ref:
            thread = self._load_thread(valkey, segment_id, subagent_ref)
        else:
            thread = self._create_thread(segment_id)

        messages = thread["messages"]
        messages.append({"role": "user", "content": inquiry})

        llm_provider = self.llm_provider or LLMProvider()
        response = llm_provider.generate_response(
            messages=list(messages),
            model_config=OUTSIDE_MODEL_CONFIG,
            system_prompt=OUTSIDE_MODEL_SYSTEM_PROMPT,
        )
        response_text = llm_provider.extract_text_content(response).strip()
        if not response_text:
            raise RuntimeError("the outside model returned an empty phone-a-friend response")

        messages.append({"role": "assistant", "content": response_text})
        self._save_thread(valkey, thread, messages)

        return {
            "success": True,
            "model_role": OUTSIDE_MODEL_ROLE,
            "subagent_ref": thread["subagent_ref"],
            "segment_id": segment_id,
            "response": response_text,
            "message": (
                f"The outside model responded. Reuse subagent_ref "
                f"{thread['subagent_ref']} to continue this outside-model thread."
            ),
        }

    def _get_valkey(self) -> Any:
        if self.valkey_client is not None:
            return self.valkey_client
        from clients.valkey_client import get_valkey_client

        self.valkey_client = get_valkey_client()
        return self.valkey_client

    def _create_thread(self, segment_id: str) -> Dict[str, Any]:
        now = format_utc_iso(utc_now())
        thread_id = uuid4().hex
        thread = {
            "thread_id": thread_id,
            "subagent_ref": f"{KEY_PREFIX}:{thread_id}",
            "owner_user_id": self.user_id,
            "segment_id": segment_id,
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        return thread

    def _load_thread(self, valkey: Any, segment_id: str, subagent_ref: str) -> Dict[str, Any]:
        thread_id = self._parse_subagent_ref(subagent_ref)
        key = self._thread_key(segment_id, thread_id)
        raw = valkey.get(key)
        if raw is None:
            raise ValueError(
                f"subagent_ref {subagent_ref} is not active in this conversation segment"
            )
        thread = json.loads(raw)
        if thread.get("owner_user_id") != self.user_id:
            raise ValueError(
                f"subagent_ref {subagent_ref} is not active for the current user"
            )
        if thread.get("segment_id") != segment_id:
            raise ValueError(
                f"subagent_ref {subagent_ref} is not active in this conversation segment"
            )
        return thread

    def _save_thread(
        self,
        valkey: Any,
        thread: Dict[str, Any],
        messages: list[dict[str, str]],
    ) -> None:
        thread["messages"] = messages
        thread["updated_at"] = format_utc_iso(utc_now())
        valkey.setex(
            self._thread_key(thread["segment_id"], thread["thread_id"]),
            THREAD_TTL_SECONDS,
            json.dumps(thread),
        )

    def _parse_subagent_ref(self, subagent_ref: str) -> str:
        prefix = f"{KEY_PREFIX}:"
        if not subagent_ref.startswith(prefix):
            raise ValueError("subagent_ref must be formatted as 'phoneafriend:<id>'")
        thread_id = subagent_ref[len(prefix):].strip()
        if not thread_id:
            raise ValueError("subagent_ref must include a thread id")
        return thread_id

    def _thread_key(self, segment_id: str, thread_id: str) -> str:
        return f"{KEY_PREFIX}:{self.user_id}:{segment_id}:{thread_id}"
