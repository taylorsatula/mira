"""Pure rendering layer for the MIRA TUI: display filters, block
renderers, the streaming-preview region state, and the Rich→ANSI bridge.

No I/O beyond the StringIO capture console; no prompt_toolkit imports.
Everything returns ``rich.RenderableType`` (or plain strings for the
preview region); the app owns emission.

Display filters mirror the web client's JS (``web/assets/javascript/
messaging.js``) so both clients display the same text.
"""

from __future__ import annotations

import json
import re
from io import StringIO
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ---------------------------------------------------------------------------
# Display filters (mirror of web/assets/javascript/messaging.js)
# ---------------------------------------------------------------------------

# JS: complete think blocks, any casing, dot-matches-newline, non-greedy.
# Built by concatenation: literal think-tag strings placed directly in a
# tool payload get HTML-entity-mangled on write.
_THINK_BLOCK_RE = re.compile(
    "<" + "think" + ">.*?</" + "think" + ">",
    re.IGNORECASE | re.DOTALL,
)
# JS: /<mira:([^>\/\s]+)(?:\s[^>]*)?>[\s\S]*?<\/mira:\1>|<mira:[^>]*\/>/gi
_MIRA_TAG_RE = re.compile(
    r"<mira:([^>/\s]+)(?:\s[^>]*)?>[\s\S]*?</mira:\1>|<mira:[^>]*/>",
    re.IGNORECASE | re.DOTALL,
)
# JS: /^\[\d{1,2}:\d{2}[ap]m\]\s*/i — ephemeral LLM-context timestamps.
_EPHEMERAL_TIMESTAMP_RE = re.compile(r"^\[\d{1,2}:\d{2}[ap]m\]\s*", re.IGNORECASE)

# JS streaming guards: an incomplete trailing tag must never flash.
_INCOMPLETE_TAG_RES = (
    re.compile(r"<think$"),
    re.compile(r"<mira:[^>]*$"),
    re.compile(r"<$"),
)


def filter_system_tags(text: str) -> str:
    """Remove think blocks, all <mira:...> namespaced tags (paired or
    self-closing), and a leading ephemeral `[5:47pm]` timestamp; trim.

    Returns its input unchanged when falsy, matching the web client.
    """
    if not text:
        return text
    stripped = _THINK_BLOCK_RE.sub("", text)
    stripped = _MIRA_TAG_RE.sub("", stripped)
    stripped = _EPHEMERAL_TIMESTAMP_RE.sub("", stripped)
    return stripped.strip()


def filter_streaming_text(text: str) -> str:
    """filter_system_tags, but first cut at an incomplete trailing tag so
    partial tags never flash while streaming."""
    if not text:
        return text
    safe_end = len(text)
    for pattern in _INCOMPLETE_TAG_RES:
        match = pattern.search(text)
        if match:
            safe_end = match.start()
            break
    return filter_system_tags(text[:safe_end])


def summarize_tool_result(raw: str) -> str:
    """Extract the human-relevant summary of a tool result row: a JSON
    object's `message` / `error` / `status` field if present, else the raw
    string. Truncated to ~200 chars with an ellipsis when longer."""
    summary = raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            summary = parsed.get("message") or parsed.get("error") or parsed.get("status") or raw
    except (ValueError, TypeError):
        pass
    summary = str(summary).replace("\n", " ")
    if len(summary) > 200:
        summary = summary[:200].rstrip() + "…"
    return summary


def format_content_blocks(content: str) -> str | None:
    """Detect a JSON content-block array stored as a string and render it as
    readable text. Returns None when `content` is not a content-block array
    (the caller renders the raw text), mirroring the web fallback."""
    if not content or not isinstance(content, str):
        return None
    trimmed = content.lstrip()
    if not trimmed.startswith("["):
        return None
    try:
        blocks = json.loads(trimmed)
    except ValueError:
        return None
    if not isinstance(blocks, list):
        return None

    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and block.get("text"):
            parts.append(str(block["text"]))
        elif block_type == "tool_use" and block.get("name"):
            entries: list[tuple[str, str]] = []
            tool_input = block.get("input")
            if isinstance(tool_input, dict):
                for key, value in tool_input.items():
                    entries.append(
                        (key, value if isinstance(value, str) else json.dumps(value))
                    )
            lines = [f"◆ {block['name']}"]
            last_entry = len(entries) - 1
            for index, (key, value) in enumerate(entries):
                elbow = "└─" if index == last_entry else "├─"
                lines.append(f"  {elbow} {key}: {value}")
            parts.append("\n".join(lines))
        # thinking / redacted_thinking / image / document / tool_result: skip.

    if not parts:
        return None
    return "\n\n".join(parts)


def looks_like_content_blocks(content: str) -> bool:
    """True when content is a JSON array of typed block dicts (the server's
    content-block storage form). Plain JSON answers (arrays of strings,
    objects without a "type" key) are NOT content blocks."""
    trimmed = content.lstrip()
    if not trimmed.startswith("["):
        return False
    try:
        parsed = json.loads(trimmed)
    except ValueError:
        return False
    if not isinstance(parsed, list) or not parsed:
        return False
    return all(
        isinstance(block, dict) and isinstance(block.get("type"), str)
        for block in parsed
    )


def user_content_text(content: str) -> str:
    """Extract display text for a user row whose content is a stringified
    content-block array (attachment-bearing rows): text blocks' `text`
    fields joined by newlines, or a placeholder when there are none.
    Image/document block data is never rendered."""
    if looks_like_content_blocks(content):
        try:
            blocks = json.loads(content)
        except ValueError:
            return content  # unreachable (looks_like verified); belt only
        texts = [
            str(block["text"])
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "text"
            and block.get("text")
        ]
        if texts:
            return "\n".join(texts)
        return "[image or document attachment]"
    return content


# ---------------------------------------------------------------------------
# ANSI bridge: Rich renderable -> ANSI string (for print_formatted_text and
# the prompt's bottom_toolbar). prompt_toolkit owns the cursor; this only
# produces text.
# ---------------------------------------------------------------------------

def render_ansi(renderable: Any, width: int) -> str:
    """Render one Rich renderable to an ANSI string at the given width."""
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=max(width, 20))
    console.print(renderable)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Block renderers (each returns a Rich renderable printed to scrollback)
# ---------------------------------------------------------------------------

def user_message(content: str) -> Panel:
    """A user chat message: full-width bordered panel, cyan border, text
    in the default foreground (same shape as the segment banner)."""
    return Panel(
        Text(filter_system_tags(content)),
        title=Text("You", style="cyan dim"),
        title_align="left",
        border_style="cyan",
    )


def tool_tree_lines(
    tool_name: str,
    input_entries: list[tuple[str, str]],
    result_summary: str | None,
    is_error: bool = False,
) -> list[Text]:
    """Build the `◆ name / ├─ arg / └─ result` ASCII tree as rich Text lines.

    The result line is red on error, green otherwise; argument lines are dim.
    With a result summary present the tree collapses to one status line.
    """
    if result_summary is not None:
        style = "red" if is_error else "green"
        glyph = "✗" if is_error else "◆"
        return [Text(f"{glyph} {tool_name} — result: {result_summary}", style=style)]
    lines: list[Text] = [Text(f"◆ {tool_name}", style="bold green")]
    for index, (key, value) in enumerate(input_entries):
        elbow = "└─" if index == len(input_entries) - 1 else "├─"
        lines.append(Text(f"  {elbow} {key}: {value}", style="dim"))
    return lines


def tool_result_line(result_raw: str, is_error: bool = False) -> Text:
    """A completed tool-result history row: the tool name is not
    recoverable from the row, so a generic collapsed summary line."""
    return tool_tree_lines("tool", [], summarize_tool_result(result_raw), is_error)[0]


def assistant_final(
    answer: str,
    *,
    thinking: str = "",
    tool_lines: list[Text] | None = None,
) -> Panel | None:
    """The final assistant block: a full-width bordered panel (green
    border), the dim reasoning trace above the answer separated by a
    blank line, the filtered answer text, and completed tool trees inside
    the panel. Returns None when the block would carry no content (empty
    turns leave nothing)."""
    display = filter_system_tags(answer or "")
    if not display and not thinking and not tool_lines:
        return None
    body_parts: list[Text] = []
    if thinking:
        body_parts.append(Text(thinking.strip(), style="dim"))
        body_parts.append(Text("\n\n"))
    if display:
        body_parts.append(Text(display))
    for line in tool_lines or []:
        body_parts.append(Text("\n"))
        body_parts.append(line)
    return Panel(
        Text.assemble(*body_parts) if body_parts else Text(""),
        title=Text("MIRA", style="green dim"),
        title_align="left",
        border_style="green",
    )


def segment_banner(
    title: str | None, summary: str, end_time: str | None
) -> Panel:
    """Pre-session summary banner: dim bordered panel headed "Where we left
    off last session" with the segment's display title and summary."""
    header = "Where we left off last session:"
    if title:
        header = f"{header} {title}"
    body = Text(summary, style="dim")
    if end_time:
        body = Text.assemble(
            body,
            Text("\n"),
            Text(f"(ended {end_time})", style="dim"),
        )
    return Panel(
        body,
        title=Text(header, style="cyan dim"),
        title_align="left",
        border_style="dim",
    )


def chrome_delimiter(width: int) -> Text:
    """One native-colored `─` delimiter row spanning the terminal width,
    for the input-field chrome above and below the prompt."""
    return Text("─" * max(width, 20))


def notice(text: str) -> Text:
    """A dim meta line (separators, reconnect notices, hints)."""
    return Text(text, style="dim")


def queued_notice(count: int, text: str) -> Text:
    """Confirmation that a follow-up was queued behind the active turn."""
    return Text(f"queued ({count}): {text}", style="dim cyan")


def error_block(text: str) -> Text:
    """A red error line for turn errors and send failures."""
    return Text(text, style="red")


def preview_text(fragments: list[tuple[str, str]]) -> Text:
    """Join the preview region's `(style, line)` fragments into one
    wrapped Text block for the prompt's bottom_toolbar."""
    parts: list[Text] = []
    for index, (style, line) in enumerate(fragments):
        if index:
            parts.append(Text("\n"))
        parts.append(Text(line, style=style) if style else Text(line))
    return Text.assemble(*parts)


def status_line(endpoint: str | None, state: str, detail: str | None) -> Text:
    """The idle toolbar: one dim `<endpoint> · <state> · <detail>` line
    (detail red on error states)."""
    parts: list[Text] = [Text(endpoint or "no endpoint", style="dim cyan")]
    parts.append(Text(f" · {state}", style="dim"))
    if detail:
        style = "red" if state in ("error", "protocol_error") else "dim"
        parts.append(Text(f" · {detail}", style=style))
    return Text.assemble(*parts)


# ---------------------------------------------------------------------------
# ActiveTurn: streaming-preview region state (the prompt's bottom_toolbar)
# ---------------------------------------------------------------------------

_PREVIEW_LINES = 12  # bounded preview: the LAST ~12 lines of the answer


class ActiveTurn:
    """Per-turn streaming state backing the preview region.

    The app creates this lazily on the first content-bearing event (so
    empty turns leave nothing) and drops it on the terminal event; the
    final answer is printed to scrollback from the authoritative terminal
    payload, not from this buffer.
    """

    def __init__(self, *, include_thinking: bool = False) -> None:
        self._include_thinking = include_thinking
        self._answer_raw = ""  # unfiltered buffer; display goes through filters
        self._thinking_raw = ""
        self._tools: dict[str, dict[str, Any]] = {}  # tool_id -> tool state
        self._tool_order: list[str] = []

    # -- streaming API --------------------------------------------------------

    def append_delta(self, text: str) -> None:
        self._answer_raw += text

    def append_thinking(self, text: str) -> None:
        self._thinking_raw += text

    def set_tool_event(
        self,
        *,
        event: str,
        tool_name: str,
        tool_id: str = "",
        arguments: dict[str, Any] | None = None,
        result: str | None = None,
        is_error: bool = False,
    ) -> None:
        """Update the tool state from a WS `tool` frame event.

        tool_detected/tool_executing mark the tool running;
        tool_completed/tool_error collapse it with the result summary.
        """
        tool = self._tools.get(tool_id)
        if tool is None:
            tool = {
                "name": tool_name,
                "arguments": {},
                "result": None,
                "is_error": False,
                "running": True,
            }
            self._tools[tool_id] = tool
            self._tool_order.append(tool_id)
        if event in ("tool_completed", "tool_error"):
            tool["running"] = False
            tool["is_error"] = is_error or event == "tool_error"
            if result is not None:
                tool["result"] = summarize_tool_result(result)
        else:  # tool_detected / tool_executing
            tool["running"] = True
            if arguments is not None:
                tool["arguments"].update(arguments)

    def apply_context_reset(self) -> None:
        """Server-side context wipe: discard all buffered answer and
        thinking text and tool trees; a fresh answer starts."""
        self._answer_raw = ""
        self._thinking_raw = ""
        self._tools = {}
        self._tool_order = []

    # -- readout ---------------------------------------------------------------

    def has_content(self) -> bool:
        return bool(self._answer_raw or self._thinking_raw or self._tools)

    def thinking_raw(self) -> str:
        return self._thinking_raw

    def streamed_answer(self) -> str:
        """The raw (unfiltered) streamed answer buffer — the fallback final
        text when a turn ends without an authoritative payload."""
        return self._answer_raw

    def completed_tool_lines(self) -> list[Text]:
        """Collapsed one-line summaries of every tool the turn used."""
        lines: list[Text] = []
        for tool_id in self._tool_order:
            tool = self._tools[tool_id]
            lines.append(
                tool_tree_lines(
                    tool["name"],
                    [
                        (key, value if isinstance(value, str) else json.dumps(value))
                        for key, value in tool["arguments"].items()
                    ],
                    tool["result"],
                    tool["is_error"],
                )[0]
            )
        return lines

    def toolbar_lines(self) -> list[tuple[str, str]]:
        """The bounded preview: `(style, line)` fragments for the prompt's
        bottom_toolbar — one running-tool line (first running tool), one
        thinking line (its last line), and the last ~12 lines of the
        filtered answer. Empty list while nothing has streamed yet."""
        fragments: list[tuple[str, str]] = []
        for tool_id in self._tool_order:
            tool = self._tools[tool_id]
            if tool["running"]:
                fragments.append(("green", f"running tool: {tool['name']}"))
                break
        if self._include_thinking and self._thinking_raw:
            last = self._thinking_raw.strip().splitlines()[-1]
            fragments.append(("dim green", last))
        display = filter_streaming_text(self._answer_raw)
        if display:
            lines = display.splitlines()[-_PREVIEW_LINES:]
            fragments.extend(("", line) for line in lines)
        return fragments


# ---------------------------------------------------------------------------
# History mapping (rows -> scrollback blocks)
# ---------------------------------------------------------------------------

def history_blocks(rows: list) -> list:
    """Map history rows (oldest->newest) to scrollback renderables.

    Ported from the first build's `_history_widgets`: the newest collapsed
    sentinel renders as the segment banner; active/paused sentinels render
    nothing; assistant content-block arrays render via
    `format_content_blocks` (pure-reasoning arrays are skipped); user
    attachment block-arrays render their text blocks or a placeholder.
    """
    blocks: list = []
    for row in rows:
        content = row.content if isinstance(row.content, str) else str(row.content)
        role = row.role
        if role == "user":
            blocks.append(user_message(user_content_text(content)))
        elif role == "assistant":
            if row.metadata.get("is_segment_boundary"):
                if row.metadata.get("status") == "collapsed":
                    blocks.append(
                        segment_banner(
                            title=row.metadata.get("display_title"),
                            summary=content,
                            end_time=row.metadata.get("segment_end_time"),
                        )
                    )
                # active/paused sentinel: never render an in-progress marker
            else:
                rendered = format_content_blocks(content)
                if rendered is not None:
                    blocks.append(assistant_final(rendered))
                elif looks_like_content_blocks(content):
                    pass  # block array with no text/tool_use (pure reasoning)
                else:
                    block = assistant_final(content)
                    if block is not None:
                        blocks.append(block)
        elif role == "tool":
            blocks.append(tool_result_line(content, is_error=bool(row.is_error)))
    return blocks
