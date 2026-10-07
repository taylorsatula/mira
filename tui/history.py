"""History replay: REST history rows to scrollback blocks.

Pure — rows in, Rich renderables out, no I/O. Replayed text goes through
the same ``text.display_lines`` sanitize/tag-filter path as live turns and
uses the same ``transcript`` elements (``you``/``mira_reply``), so a replayed conversation is indistinguishable from a live
one. ``user`` and ``assistant`` rows are replayed; ``tool`` rows, segment
sentinels (defensively — ``session_only`` pages exclude them) and rows with
no displayable text are skipped.
"""

from __future__ import annotations

import json

from rich.console import RenderableType

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


def replay_blocks(rows: list[HistoryMessage]) -> list[RenderableType]:
    """Scrollback blocks for one chronological run of history rows."""
    blocks: list[RenderableType] = []
    for row in rows:
        if _is_sentinel(row) or row.role not in ("user", "assistant"):
            continue
        text = _display_text(row.content)
        if not text:
            continue
        if row.role == "user":
            blocks.append(transcript.you(text))
        else:
            lines = display_lines(text)
            if lines:
                blocks.extend(transcript.mira_reply(lines))
    return blocks
