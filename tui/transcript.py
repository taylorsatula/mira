"""Scrollback look of the MIRA TUI.

Pure: Rich renderables built from plain strings, never from markup, so model
or user text cannot raise ``MarkupError`` or pick up emoji codes. All text
passes through ``text.sanitize``, because Rich does not strip ESC. A block's
leading blank line is part of the block; spacing lives here and nowhere else.

Conversation turns are open boxes inside a band narrower than the terminal
(``_band``), word-wrapped here at render time so the gutter repeats on every
wrapped row. MIRA's box opens with the ``:MIRA`` label on its first row,
continues with a ``╵`` gutter, and closes with a ``╰╴╴╴`` border along the
bottom left; the user's box opens with a ``╴╴╴╮`` border along the top right
and carries ``ME:`` / ``╵`` on the right edge. Both are printable top to
bottom without knowing their final height, which is what append-only
scrollback requires: the user's box prints whole, MIRA's prints row by row
as the reply streams and closes when the turn ends.

``mira_lines``, ``thinking_lines``, ``tool_line``, ``reply_footer``,
``notice`` and ``alert`` build body ``Text``; inside a turn the caller wraps
a body with ``mira_rows``, outside it a body prints on its own.
"""

from __future__ import annotations

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.text import Text

from tui.text import sanitize

MIRA_STYLE = "bright_green"
ME_STYLE = "bright_magenta"

MIRA_LABEL = ":MIRA "
MIRA_GUTTER = "╵     "  # same width as MIRA_LABEL: the ╵ sits under the ":"
ME_LABEL = "  ME:"
ME_GUTTER = "    ╵"  # same width as ME_LABEL: the ╵ sits under the ":"
BORDER_CHAR = "╴"

# The band is the turn area: the terminal width less a sixth, so turns never
# run to the window edge, capped so wide terminals keep readable line lengths.
BAND_MARGIN_DIVISOR = 6
BAND_MAX = 100
# Each border runs two thirds of the band from its own side; MIRA's (bottom
# left) and the user's (top right) overlap in the middle third.
BORDER_FRACTION = 2 / 3
# The user's text starts at the same indent as MIRA's.
ME_INDENT = len(MIRA_LABEL)
TAB_SIZE = 4


def _band(width: int) -> int:
    return max(1, min(BAND_MAX, width - width // BAND_MARGIN_DIVISOR))


def _border(band: int) -> int:
    return max(1, round(band * BORDER_FRACTION))


def _wrap(console: Console, body: Text, width: int) -> list[Text]:
    """Word-wrap ``body`` to ``width`` cells; blank lines stay as empty rows."""
    rows = body.wrap(console, max(1, width), overflow="fold", tab_size=TAB_SIZE)
    for row in rows:
        row.rstrip()
    return list(rows)


class _MiraRows:
    """Rows of MIRA's open box. ``opens`` puts the label on the first row
    (after a blank line); otherwise every row carries the gutter."""

    def __init__(self, body: Text, opens: bool) -> None:
        self._body = body
        self._opens = opens

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        band = _band(options.max_width)
        out = Text("\n" if self._opens else "")
        for i, row in enumerate(_wrap(console, self._body, band - len(MIRA_LABEL))):
            if i:
                out.append("\n")
            if i == 0 and self._opens:
                out.append(MIRA_LABEL, style=f"bold {MIRA_STYLE}")
            else:
                out.append(MIRA_GUTTER if row.plain else MIRA_GUTTER.rstrip(), style=MIRA_STYLE)
            out.append_text(row)
        yield out


class _MiraClose:
    """The bottom-left border that ends MIRA's box."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield Text("╰" + BORDER_CHAR * _border(_band(options.max_width)), style=MIRA_STYLE)


class _You:
    """The user's box: top-right border, then the message right-anchored in
    the band, left-aligned within its own width, ``ME:`` on the first row."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        band = _band(options.max_width)
        border = _border(band)
        rows = _wrap(console, Text(self._text), band - ME_INDENT - len(ME_LABEL))
        box = max(row.cell_len for row in rows)
        indent = " " * max(0, band - len(ME_LABEL) - box)
        out = Text.assemble(
            "\n",
            " " * max(0, band - 1 - border),
            (BORDER_CHAR * border + "╮", ME_STYLE),
        )
        for i, row in enumerate(rows):
            out.append("\n" + indent)
            out.append_text(row)
            out.append(" " * (box - row.cell_len))
            if i == 0:
                out.append(ME_LABEL, style=f"bold {ME_STYLE}")
            else:
                out.append(ME_GUTTER, style=ME_STYLE)
        yield out


def banner(endpoint: str, base_url: str) -> Text:
    return Text(
        f"MIRA · {sanitize(endpoint)} ({sanitize(base_url)})"
        " · Enter send · Alt+Enter newline · Ctrl+C stop/quit · Ctrl+T thinking",
        style="dim",
    )


def you(text: str) -> RenderableType:
    """The user's message as typed (no tag filtering), in its box."""
    return _You(sanitize(text).strip("\n"))


def mira_rows(body: Text, *, opens: bool) -> RenderableType:
    """``body`` as rows of MIRA's box; ``opens`` marks the box's first rows."""
    return _MiraRows(body, opens)


def mira_close() -> RenderableType:
    return _MiraClose()


def mira_reply(lines: list[str]) -> list[RenderableType]:
    """A whole MIRA box: proactive messages and replayed history."""
    return [mira_rows(mira_lines(lines), opens=True), mira_close()]


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
