"""Scrollback look of the MIRA TUI: every element is one ``rich.text.Text``.

Pure: Rich ``Text`` objects only, built from plain strings, never from
markup, so model or user text cannot raise ``MarkupError`` or pick up emoji
codes. All text passes through ``text.sanitize``, because Rich does not strip
ESC. A block's leading blank line is part of its Text (``"\\n"`` prefix);
spacing lives here and nowhere else.
"""

from __future__ import annotations

from rich.text import Text

from tui.text import sanitize


def banner(endpoint: str, base_url: str) -> Text:
    return Text(
        f"MIRA · {sanitize(endpoint)} ({sanitize(base_url)})"
        " · Enter send · Alt+Enter newline · Ctrl+C stop/quit · Ctrl+T thinking",
        style="dim",
    )


def you(text: str) -> Text:
    """The user's message as typed (no tag filtering)."""
    return Text.assemble("\n", ("You", "bold cyan"), "\n", sanitize(text))


def mira_label() -> Text:
    return Text.assemble("\n", ("MIRA", "bold green"))


def mira_lines(lines: list[str]) -> Text:
    return Text("\n".join(sanitize(line) for line in lines))


def thinking_lines(lines: list[str]) -> Text:
    """MIRA's reasoning stream (the Ctrl+T view), dimmed and italicized."""
    return Text("\n".join(sanitize(line) for line in lines), style="dim italic")


def tool_line(name: str, ok: bool) -> Text:
    if ok:
        return Text(f"· used {sanitize(name)}", style="dim")
    return Text(f"· {sanitize(name)} failed", style="dim red")


def reply_footer(tools: int) -> Text:
    """Dim tool-count line after a finished reply; omitted when no tools ran."""
    return Text(f"· {tools} tool" + ("" if tools == 1 else "s"), style="dim")


def notice(text: str) -> Text:
    return Text(sanitize(text), style="dim")


def alert(text: str) -> Text:
    return Text(sanitize(text), style="red")
