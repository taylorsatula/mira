"""Provider-neutral local tool continuation message assembly."""

from __future__ import annotations

from typing import Any

from clients.llm.types import Result, ToolCall, ToolResult


def _selected_tool_calls(
    result: Result,
    include_tool_call_ids: set[str] | None,
) -> tuple[ToolCall, ...]:
    selected = (
        tuple(tc for tc in result.tool_calls if tc.id in include_tool_call_ids)
        if include_tool_call_ids is not None
        else result.tool_calls
    )
    return tuple(tc for tc in selected if tc.invalid_reason is None)


def assistant_message_from_result(
    result: Result,
    *,
    include_tool_call_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build the assistant message that records model-requested local tool calls.

    When include_tool_call_ids is provided, only tool calls whose IDs appear in
    the set are included. This prevents orphaned tool_call/tool-result pairs
    when some tool calls (e.g. server-side code_execution) don't have results.
    """
    tool_calls = _selected_tool_calls(result, include_tool_call_ids)

    content: list[dict[str, Any]] = []
    if result.reasoning:
        content.extend(result.reasoning.to_content_blocks())
    if result.text.strip():
        content.append({"type": "text", "text": result.text})
    content.extend(tool_call.to_message_block() for tool_call in tool_calls)

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if result.reasoning:
        signatures = result.reasoning.to_signatures()
        if signatures:
            message["thinking_signatures"] = signatures
        if result.reasoning.provider_details:
            message["reasoning_details"] = list(result.reasoning.provider_details)
    return message


def tool_result_messages(tool_results: tuple[ToolResult, ...]) -> list[dict[str, Any]]:
    """Build individual role="tool" messages for completed local tool results.

    Each tool result becomes its own role="tool" message rather than
    being packed into a single user-role message.
    """
    return [
        {
            "role": "tool",
            "tool_call_id": tr.tool_call_id,
            "content": tr.content,
            **({"is_error": True} if tr.is_error else {}),
        }
        for tr in tool_results
    ]


def invalid_tool_call_feedback_messages(
    result: Result,
    tool_results: tuple[ToolResult, ...],
) -> list[dict[str, Any]]:
    """Build ordinary messages for provider-rejected local tool calls.

    Invalid tool calls cannot be replayed as assistant tool_calls because doing
    so reintroduces schema-invalid arguments into the next provider request.
    They also cannot produce role="tool" messages without a matching assistant
    tool_call. Convert them into text feedback so the model can repair the call.
    """
    results_by_id = {tr.tool_call_id: tr for tr in tool_results}
    messages: list[dict[str, Any]] = []
    for tool_call in result.tool_calls:
        if tool_call.invalid_reason is None or tool_call.id not in results_by_id:
            continue
        tool_result = results_by_id[tool_call.id]
        content = (
            "[Automated system message: The previous "
            f"{tool_call.tool_name} tool call was rejected before execution "
            f"because its arguments were invalid: {tool_call.invalid_reason}. "
            "If the tool is still needed, issue a new tool call with valid "
            "arguments. Rejection details follow.]\n\n"
            f"{tool_result.content}"
        )
        messages.append({"role": "user", "content": content})
    return messages


def append_tool_result_messages(
    messages: list[dict[str, Any]],
    result: Result,
    tool_results: tuple[ToolResult, ...],
) -> list[dict[str, Any]]:
    """Append a provider-neutral local tool use/result pair to message history.

    Only includes tool calls that have matching results. Tool calls without
    results (e.g. server-side code_execution) are stripped from the assistant
    message to prevent provider 400 errors for orphaned tool_call/tool pairs.
    """
    valid_tool_call_ids = {
        tool_call.id
        for tool_call in result.tool_calls
        if tool_call.invalid_reason is None
    }
    valid_tool_results = tuple(
        tr for tr in tool_results if tr.tool_call_id in valid_tool_call_ids
    )
    invalid_feedback = invalid_tool_call_feedback_messages(result, tool_results)
    result_ids = {tr.tool_call_id for tr in valid_tool_results}
    assistant_msg = assistant_message_from_result(result, include_tool_call_ids=result_ids)

    assistant_content = assistant_msg.get("content")
    include_assistant = bool(
        assistant_msg.get("tool_calls")
        or assistant_msg.get("thinking_signatures")
        or assistant_msg.get("reasoning_details")
        or (isinstance(assistant_content, list) and bool(assistant_content))
        or (isinstance(assistant_content, str) and assistant_content.strip())
    )

    return [
        *messages,
        *([assistant_msg] if include_assistant else []),
        *tool_result_messages(valid_tool_results),
        *invalid_feedback,
    ]
