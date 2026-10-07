"""History replay: REST history rows to scrollback blocks.

Pure — rows in, Rich renderables out, no I/O. Replayed text goes through
the same ``text.display_lines`` sanitize/tag-filter path as live turns and
uses the same ``transcript`` elements, so a replayed conversation looks like
a live one: a ``user`` row is the user's box, and the ``assistant`` and
``tool`` rows that share a ``turn_id`` (metadata) form one MIRA box — text,
a ``tool_line`` per tool row (``metadata.tool_name``, ``is_error``), the
tool-count footer, a paragraph break before text that follows earlier
text. Rows without a ``turn_id`` are a box each. Segment sentinels
(defensively — ``session_only`` pages exclude them), other roles, and rows
with no displayable text are skipped. Not recoverable from history: the
reasoning view and the ``stopped`` row of a halted turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from rich.console import RenderableType
from rich.text import Text

from tui import transcript
from tui.client import _is_sentinel
from tui.protocol import HistoryMessage
from tui.text import display_lines


def _display_text(content: object) -> str:
    """Extract displayable text from one row's content: a plain string, or a
    JSON-stringified array of provider content blocks (join the ``text``
    blocks; ``reasoning`` and ``tool_call`` blocks are not conversation)."""
    if not isinstance(content, str) or not content.strip():
        return ""
    try:
        parsed = json.loads(content)
    except ValueError:
        return content
    if not isinstance(parsed, list):
        return content
    parts = [
        block.get("text", "")
        for block in parsed
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(str(part) for part in parts if isinstance(part, str) and part)


@dataclass
class _Reply:
    """MIRA's box for one turn while its rows are read."""

    turn_id: str | None
    bodies: list[Text] = field(default_factory=list)
    tools: set[str] = field(default_factory=set)
    shown: bool = False  # text already in the box: the next text starts a paragraph


def _close(reply: _Reply | None, blocks: list[RenderableType]) -> None:
    if reply is None or not reply.bodies:
        return
    if reply.tools:
        reply.bodies.append(transcript.reply_footer(len(reply.tools)))
    blocks.extend(transcript.mira_box(reply.bodies))


def replay_blocks(rows: list[HistoryMessage]) -> list[RenderableType]:
    """Scrollback blocks for one chronological run of history rows."""
    blocks: list[RenderableType] = []
    reply: _Reply | None = None
    for row in rows:
        if _is_sentinel(row) or row.role not in ("user", "assistant", "tool"):
            continue
        if row.role == "user":
            _close(reply, blocks)
            reply = None
            text = _display_text(row.content)
            if text:
                blocks.append(transcript.you(text))
            continue
        turn_id = row.metadata.get("turn_id")
        if not isinstance(turn_id, str):
            turn_id = None
        if reply is None or turn_id is None or turn_id != reply.turn_id:
            _close(reply, blocks)
            reply = _Reply(turn_id)
        if row.role == "tool":
            name = row.metadata.get("tool_name")
            if isinstance(name, str):
                arguments = row.metadata.get("tool_arguments")
                ok = not row.is_error
                reply.bodies.append(transcript.tool_line(
                    name, ok, arguments if isinstance(arguments, dict) else None, None
                ))
                if not ok and isinstance(row.content, str) and row.content.strip():
                    reply.bodies.append(transcript.tool_error(row.content))
                reply.tools.add(row.tool_call_id or row.id)
            continue
        lines = display_lines(_display_text(row.content))
        if lines:
            reply.bodies.append(transcript.mira_lines([""] + lines if reply.shown else lines))
            reply.shown = True
    _close(reply, blocks)
    return blocks
