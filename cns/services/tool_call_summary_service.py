"""
One-sentence display descriptions of tool calls, sent to the chat client's
tool row and never persisted.

``is_summarizable`` is the pure eligibility decision the WebSocket turn path
runs at ``tool_executing``; ``summarize_tool_call`` is the blocking fast-route
call that runs once a call qualifies. The summary describes the arguments, so
it stays truthful for failed calls too: a refusal or an error row describes
the attempt, not the outcome.

Every failure degrades to "no summary" — the caller logs and the tool row
falls back to its raw-arguments display — because a display decoration must
never block or fail the turn that triggered it. Nothing in this module writes
anywhere: the summary exists only inside the live frame sent to the client.
"""
import json
import logging
from typing import Optional

from clients.llm_provider import LLMProvider, get_llm_provider

logger = logging.getLogger(__name__)

# At or under one rendered row the raw arguments read faster than a generated
# sentence; the LLM pass is for compound calls that truncate into noise.
MIN_SERIALIZED_ARGUMENTS_LENGTH = 100
# Same shape as domaindoc_summary_service's 2000-char cap: enough of any
# command or argument blob for a faithful one-liner, never the full prompt
# cost of a giant one.
SUMMARY_INPUT_MAX_CHARS = 4000
# Untrusted-component bound: a misbehaving fast route cannot push a paragraph
# into a one-row display; the clients truncate at render anyway.
SUMMARY_MAX_CHARS = 200

_system_prompt: Optional[str] = None
_user_template: Optional[str] = None
_llm_provider: Optional[LLMProvider] = None


def _load_prompts() -> None:
    """Load prompt templates from config files (lazy, once per process)."""
    global _system_prompt, _user_template

    if _system_prompt is not None:
        return  # Already loaded

    from config.prompts.loader import load_prompt
    _system_prompt = load_prompt("tool_call_summary_system.txt")
    _user_template = load_prompt("tool_call_summary_user.txt")


def _get_llm_provider() -> LLMProvider:
    """Get or create the LLM provider singleton."""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = get_llm_provider()
    return _llm_provider


def is_summarizable(arguments: object) -> bool:
    """Pure eligibility decision, safe on any thread: a call is summarized
    when its serialized arguments are long enough to render as noise."""
    if not isinstance(arguments, dict):
        return False
    return len(json.dumps(arguments, ensure_ascii=False, default=str)) >= MIN_SERIALIZED_ARGUMENTS_LENGTH


def summarize_tool_call(tool_name: str, arguments: dict[str, object]) -> str:
    """Blocking fast-route call returning the one-sentence description.

    Raises on provider failure; the caller degrades. The caller must have
    checked ``is_summarizable`` — it is re-checked here so the eligibility
    rule lives in exactly one place.
    """
    if not is_summarizable(arguments):
        raise ValueError("summarize_tool_call called on a non-summarizable tool call")

    _load_prompts()
    assert _system_prompt is not None
    assert _user_template is not None

    serialized = json.dumps(arguments, ensure_ascii=False, default=str)
    if len(serialized) > SUMMARY_INPUT_MAX_CHARS:
        serialized = serialized[:SUMMARY_INPUT_MAX_CHARS] + " …"

    messages = [
        {"role": "system", "content": _system_prompt},
        {"role": "user", "content": _user_template.format(
            tool_name=tool_name,
            arguments=serialized,
        )},
    ]

    llm = _get_llm_provider()
    response = llm.generate_response(
        messages=messages,
        model_config="fast",
    )

    summary = llm.extract_text_content(response).strip()
    # The clients render one row: a multi-line or runaway reply degrades to
    # its first line, cut to the display bound.
    summary = summary.partition("\n")[0].strip()[:SUMMARY_MAX_CHARS].strip()
    if not summary:
        raise ValueError("fast route returned an empty tool-call summary")
    return summary
