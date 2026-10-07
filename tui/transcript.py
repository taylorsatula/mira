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
a body with ``mira_rows``, outside it a body goes through ``Flow.place``,
which wraps it to the band and spaces it off a box above it.
"""

from __future__ import annotations

import re

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.text import Text

from tui.text import sanitize

MIRA_STYLE = "bright_green"
ME_STYLE = "bright_magenta"
CODE_STYLE = "cyan"

MIRA_LABEL = ":MIRA "
MIRA_GUTTER = "╵     "  # same width as MIRA_LABEL: the ╵ sits under the ":"
ME_LABEL = "  ME:"
ME_GUTTER = "    ╵"  # same width as ME_LABEL: the ╵ sits under the ":"
BORDER_CHAR = "╴"

# The band is the turn area: five sixths of the terminal width, so turns
# never run to the window edge and scale with the window.
BAND_MARGIN_DIVISOR = 6
# Each border runs two thirds of the band from its own side; MIRA's (bottom
# left) and the user's (top right) overlap in the middle third.
BORDER_FRACTION = 2 / 3
# The user's text starts at the same indent as MIRA's.
ME_INDENT = len(MIRA_LABEL)
TAB_SIZE = 4

# Indentation plus a list marker: wrapped rows of the line hang under its text.
_HANG_RE = re.compile(r"[ \t]*(?:(?:[-*+•]|\d{1,3}[.)])[ \t]+)?")
_HEADING_RE = re.compile(r"#{1,6}[ \t]+")
# Inline markdown, left to right: `code` first so its contents stay literal,
# then **bold**, then *italic* (not a bullet "* ", not inside a word).
_INLINE_RE = re.compile(
    r"`([^`\n]+)`"
    r"|\*\*(?=\S)(.+?)(?<=\S)\*\*"
    r"|(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])"
)


def _band(width: int) -> int:
    return max(1, width - width // BAND_MARGIN_DIVISOR)


def _border(band: int) -> int:
    return max(1, round(band * BORDER_FRACTION))


def _wrap(console: Console, body: Text, width: int) -> list[Text]:
    """Word-wrap ``body`` to ``width`` cells; blank lines stay as empty rows."""
    rows: list[Text] = []
    for line in body.split("\n", allow_blank=True):
        line.expand_tabs(TAB_SIZE)
        rows.extend(_wrap_line(console, line, max(1, width)))
    return rows


def _wrap_line(console: Console, line: Text, width: int) -> list[Text]:
    """One logical line: the first row keeps its indentation; wrapped rows
    drop the spaces the break left and hang under the line's text."""
    first, *more = line.wrap(console, width, overflow="fold")
    first.rstrip()
    if not more:
        return [first]
    hang = _HANG_RE.match(line.plain).end()  # spaces and markers: one cell each
    if hang * 2 > width:
        hang = 0
    rows = [first]
    for row in line[len(first.plain):].wrap(console, width - hang, overflow="fold"):
        row = row[len(row.plain) - len(row.plain.lstrip(" ")):]
        row.rstrip()
        if row.plain:
            rows.append(Text(" " * hang) + row if hang else row)
    return rows


def _markdown(line: str) -> Text:
    """One line of model text with its inline markdown applied: headings
    bold, ``**bold**``, ``*italic*``, ```code``` styled and their markers
    dropped."""
    heading = _HEADING_RE.match(line)
    if heading:
        line = line[heading.end():]
    out = Text()
    pos = 0
    for match in _INLINE_RE.finditer(line):
        out.append(line[pos:match.start()])
        code, bold, italic = match.groups()
        if code is not None:
            out.append(code, style=CODE_STYLE)
        elif bold is not None:
            out.append(bold, style="bold")
        else:
            out.append(italic, style="italic")
        pos = match.end()
    out.append(line[pos:])
    if heading:
        out.stylize("bold")
    return out


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
    """The user's box: top-right border, then the message in the band,
    ``ME:`` on the first row. A one-row message sits against the label; a
    longer one fills the band from MIRA's text indent."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        band = _band(options.max_width)
        border = _border(band)
        width = max(1, band - ME_INDENT - len(ME_LABEL))
        rows = _wrap(console, Text(self._text), width)
        box = width if len(rows) > 1 else rows[0].cell_len
        indent = " " * max(0, band - len(ME_LABEL) - box)
        out = Text.assemble(
            "\n",
            " " * max(0, band - 1 - border),
            (BORDER_CHAR * border + "╮", ME_STYLE),
        )
        for i, row in enumerate(rows):
            out.append("\n" + indent)
            out.append_text(row)
            out.append(" " * max(0, box - row.cell_len))
            if i == 0:
                out.append(ME_LABEL, style=f"bold {ME_STYLE}")
            else:
                out.append(ME_GUTTER, style=ME_STYLE)
        yield out


class _Plain:
    """A body printed outside any box: wrapped to the band, after a blank
    line when ``gap``."""

    def __init__(self, body: Text, gap: bool) -> None:
        self._body = body
        self._gap = gap

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        rows = _wrap(console, self._body, _band(options.max_width))
        yield Text("\n").join([Text(), *rows] if self._gap else rows)


class Flow:
    """Places blocks in scrollback for one session: a body outside a box is
    wrapped to the band and gets a blank line when a box is right above it;
    consecutive bodies stay together. Keeps one fact across emits — whether
    the last block placed was a box."""

    def __init__(self) -> None:
        self._after_box = False

    def place(self, blocks: tuple[RenderableType, ...]) -> list[RenderableType]:
        placed: list[RenderableType] = []
        for block in blocks:
            if isinstance(block, Text):
                placed.append(_Plain(block, gap=self._after_box))
                self._after_box = False
            else:
                placed.append(block)
                self._after_box = True
        return placed


def banner(endpoint: str, base_url: str) -> Text:
    return Text(
        f"MIRA · {sanitize(endpoint)} · {sanitize(base_url)}\n"
        "Enter send · Alt+Enter newline · Ctrl+C stop/quit · Ctrl+T thinking",
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


def mira_box(bodies: list[Text]) -> list[RenderableType]:
    """A whole MIRA box: proactive messages and replayed history."""
    rows = [mira_rows(body, opens=i == 0) for i, body in enumerate(bodies)]
    return [*rows, mira_close()]


def mira_reply(lines: list[str]) -> list[RenderableType]:
    return mira_box([mira_lines(lines)])


def mira_lines(lines: list[str]) -> Text:
    return Text("\n").join(_markdown(sanitize(line)) for line in lines)


def thinking_lines(lines: list[str]) -> Text:
    """MIRA's reasoning stream (the Ctrl+T view), dimmed and italicized."""
    return Text("\n", style="dim italic").join(_markdown(sanitize(line)) for line in lines)


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
