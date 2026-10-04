"""Screen: owns prompt_toolkit and every byte the TUI writes to the terminal.

Layout is normal terminal scrollback above a bar pinned to the bottom row:
live region (in-progress paragraph, outbox lines, status line), a ``─`` rule,
the input box, a ``─`` rule. The box grows upward with its content.

Scrollback is append-only and has exactly one writer while the app runs: the
print step in ``Screen.emit`` (wait for pending cursor-position replies, then
synchronously erase the bar, write the ANSI, request a cursor position,
redraw). The step never leaves raw mode, so keys typed during an emit are
never echoed by the tty into scrollback (``run_in_terminal`` switches to
cooked mode, which turns ECHO on — the double-print bug). Nothing here
counts rows or moves the cursor by hand; prompt_toolkit's renderer measures
and erases the bar.

Callers hand over Rich ``Text`` blocks; ``Intent`` objects (``Submit``,
``Interrupt``, ``Quit``, ``ToggleThinking``) are posted to the caller's
inbox with ``put_nowait``. Contract: ``tui/FRONTEND_PLAN.md`` "### ``tui/screen.py``".
"""

from __future__ import annotations

import asyncio
import io
import sys
import time
from dataclasses import dataclass
from typing import Literal

from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    ConditionalContainer,
    Dimension,
    FormattedTextControl,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea
from rich.console import Console
from rich.text import Text

# Redraw cadence for the busy spinner and the elapsed-seconds counter.
REFRESH_INTERVAL = 0.5

# Height caps are functions of terminal rows so a short terminal keeps room
# for the fixed rows (status, two rules, one input row); each is at least 1.
PENDING_MAX_ROWS_CEILING = 8   # tail-follow window; older rows are already in memory, not lost
PENDING_ROWS_DIVISOR = 4
INPUT_MAX_ROWS_CEILING = 12    # beyond this the box scrolls inside itself
INPUT_ROWS_DIVISOR = 3
OUTBOX_MAX_ROWS_CEILING = 4    # one row per unsent message; overflow collapses to a count
OUTBOX_ROWS_DIVISOR = 6

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_STEP_SECONDS = 0.1  # frame advances per 0.1 s of wall time; the redraw cadence decides what is seen
ELLIPSIS = "…"
RULE_CHAR = "─"

STYLE = Style.from_dict(
    {
        "rule": "ansibrightblack",
        "status.idle": "ansibrightblack",
        "status.busy": "ansibrightblack",
        "status.alert": "ansired",
        "outbox": "ansicyan",
        "pending": "",
    }
)


@dataclass(frozen=True)
class Status:
    text: str
    tone: Literal["idle", "busy", "alert"] = "idle"
    since: float | None = None  # time.monotonic() start; busy -> spinner + live "· 14s"


@dataclass(frozen=True)
class Live:
    pending: str = ""  # ReplyStream.pending()
    sending: str | None = None  # sent, not yet accepted (turn_started)
    queued: tuple[str, ...] = ()  # waiting behind the current reply, oldest first
    status: Status = Status("")


@dataclass(frozen=True)
class Submit:
    text: str  # raw box text at Enter time (may be empty/blank)


@dataclass(frozen=True)
class Interrupt:
    pass


@dataclass(frozen=True)
class Quit:
    pass


@dataclass(frozen=True)
class ToggleThinking:
    pass


Intent = Submit | Interrupt | Quit | ToggleThinking


def _cap(ceiling: int, divisor: int, rows: int) -> int:
    return max(1, min(ceiling, rows // divisor))


def _clip(prefix: str, text: str, columns: int) -> str:
    """First line of ``text`` after ``prefix``, cut to ``columns`` cells with an ellipsis."""
    first, *rest = text.split("\n")
    line = prefix + first
    truncated = bool(rest)
    width = max(1, columns - 1)
    if get_cwidth(line) > width:
        out, used = [], 0
        for ch in line:
            w = get_cwidth(ch)
            if used + w > width - 1:
                break
            out.append(ch)
            used += w
        line = "".join(out)
        truncated = True
    return line + ELLIPSIS if truncated else line


class Screen:
    def __init__(self, inbox: asyncio.Queue) -> None:
        self._inbox = inbox
        self._real_out = sys.stdout
        self._live = Live()
        self._lock = asyncio.Lock()
        self._running = False  # True from the Application's pre_run hook (first frame follows in the same step) until teardown
        self._at_bottom = False  # push-to-bottom newlines already written
        self._closing = False
        self._finished = asyncio.Event()  # set once the Application has torn down

        self._console = Console(
            file=io.StringIO(),
            force_terminal=True,
            color_system="standard",
            markup=False,
            emoji=False,
            highlight=False,
            # no soft_wrap: Rich word-wraps at the live terminal width (set
            # per render) instead of letting the terminal hard-cut mid-word.
            # Scrollback therefore does not reflow on a later resize.
        )

        self._input = TextArea(
            multiline=True,
            wrap_lines=True,
            scrollbar=False,
            height=lambda: Dimension(min=1, max=_cap(INPUT_MAX_ROWS_CEILING, INPUT_ROWS_DIVISOR, self._rows())),
            dont_extend_height=True,
        )
        self._app = Application(
            layout=Layout(self._build_container(), focused_element=self._input),
            key_bindings=self._build_bindings(),
            style=STYLE,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
            refresh_interval=REFRESH_INTERVAL,
            output=create_output(stdout=self._real_out),
        )

    # --- public API -------------------------------------------------------

    async def emit(self, *blocks: Text, live: Live | None = None) -> None:
        """Print blocks to scrollback (and apply ``live``) in one print step."""
        data = self._render(blocks)
        if live is not None:
            self._live = live
        if not self._running or self._closing:
            if self._closing and self._running:
                await self._finished.wait()
            self._write_direct(data)
            return
        async with self._lock:
            app = self._app
            await app.renderer.wait_for_cpr_responses()
            if not self._running or self._closing:
                if self._running:
                    await self._finished.wait()
                self._write_direct(data)
                return
            # No await below: nothing may interleave between erase and redraw.
            app.renderer.erase()
            app.output.write_raw(data)
            app.output.flush()
            app._request_absolute_cursor_position()
            app._redraw()

    def set_live(self, live: Live) -> None:
        self._live = live
        self._app.invalidate()

    def consume_input(self, text: str) -> bool:
        """Compare-and-clear: remove ``text`` from the front of the box if it is there."""
        buf = self._input.buffer
        if not buf.text.startswith(text):
            return False
        rest = buf.text[len(text):]
        cursor = max(0, buf.cursor_position - len(text))
        buf.set_document(Document(rest, min(cursor, len(rest))), bypass_readonly=True)
        self._app.invalidate()
        return True

    def restore_input(self, text: str) -> None:
        """Box becomes ``text`` followed by a blank line and the previous content, cursor at end."""
        buf = self._input.buffer
        new = text + ("\n\n" + buf.text if buf.text else "")
        buf.set_document(Document(new, len(new)), bypass_readonly=True)
        self._app.invalidate()

    def clear_input(self) -> bool:
        buf = self._input.buffer
        if not buf.text:
            return False
        buf.set_document(Document("", 0), bypass_readonly=True)
        self._app.invalidate()
        return True

    async def run(self) -> None:
        if self._closing:
            return
        self._ensure_bottom()
        try:
            # Stray library writes are routed above the bar. The Application
            # already holds the real stdout, so patch_stdout does not affect it.
            with patch_stdout(raw=True):
                await self._app.run_async(pre_run=self._mark_running)
        finally:
            self._running = False
            self._finished.set()

    def close(self) -> None:
        self._closing = True
        if self._app.is_running and not self._app.is_done:
            self._app.exit()

    # --- layout -----------------------------------------------------------

    def _rows(self) -> int:
        return self._app.output.get_size().rows

    def _columns(self) -> int:
        return self._app.output.get_size().columns

    def _build_container(self) -> HSplit:
        def pending_fragments() -> StyleAndTextTuples:
            # Cursor fragment at the end makes the Window follow the tail.
            return [("class:pending", self._live.pending), ("[SetCursorPosition]", "")]

        def outbox_fragments() -> StyleAndTextTuples:
            live = self._live
            cols = self._columns()
            cap = _cap(OUTBOX_MAX_ROWS_CEILING, OUTBOX_ROWS_DIVISOR, self._rows())
            lines = []
            if live.sending is not None:
                lines.append(_clip("sending: ", live.sending, cols))
            lines.extend(_clip("queued: ", q, cols) for q in live.queued)
            if len(lines) > cap:
                hidden = len(lines) - (cap - 1)
                lines = lines[: cap - 1] + [f"+{hidden} more{ELLIPSIS}" if cap > 1 else f"{hidden} unsent{ELLIPSIS}"]
            return [("class:outbox", "\n".join(lines))]

        def status_fragments() -> StyleAndTextTuples:
            st = self._live.status
            text = st.text
            if st.tone == "busy":
                frame = SPINNER_FRAMES[int(time.monotonic() / SPINNER_STEP_SECONDS) % len(SPINNER_FRAMES)]
                text = f"{frame} {text}"
                if st.since is not None:
                    text += f" · {max(0, int(time.monotonic() - st.since))}s"
            return [(f"class:status.{st.tone}", text.replace("\n", " "))]

        def outbox_rows() -> int:
            live = self._live
            return (live.sending is not None) + len(live.queued)

        pending = ConditionalContainer(
            Window(
                FormattedTextControl(pending_fragments),
                wrap_lines=True,
                dont_extend_height=True,
                height=lambda: Dimension(
                    min=1, max=_cap(PENDING_MAX_ROWS_CEILING, PENDING_ROWS_DIVISOR, self._rows())
                ),
            ),
            filter=Condition(lambda: bool(self._live.pending)),
        )
        outbox = ConditionalContainer(
            Window(
                FormattedTextControl(outbox_fragments),
                wrap_lines=False,
                dont_extend_height=True,
                height=lambda: Dimension(
                    min=1,
                    max=min(
                        outbox_rows(), _cap(OUTBOX_MAX_ROWS_CEILING, OUTBOX_ROWS_DIVISOR, self._rows())
                    ),
                ),
            ),
            filter=Condition(lambda: outbox_rows() > 0),
        )
        status = Window(
            FormattedTextControl(status_fragments), height=1, dont_extend_height=True
        )

        def rule() -> Window:
            return Window(height=1, char=RULE_CHAR, style="class:rule", dont_extend_height=True)

        return HSplit([Window(), pending, outbox, status, rule(), self._input, rule()])

    def _build_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        box_empty = Condition(lambda: not self._input.buffer.text)

        @kb.add("enter")
        def _submit(event) -> None:
            self._post(Submit(self._input.buffer.text))

        @kb.add("escape", "enter")
        @kb.add("c-j")
        def _newline(event) -> None:
            event.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def _interrupt(event) -> None:
            self._post(Interrupt())

        @kb.add("c-d", filter=box_empty)
        def _quit(event) -> None:
            self._post(Quit())

        @kb.add("c-t")
        def _toggle_thinking(event) -> None:
            self._post(ToggleThinking())

        # Up/Down jump the cursor to the start/end of the box text, replacing
        # the built-in per-line movement (app-level bindings outrank the
        # prompt_toolkit defaults; the control binds neither key).
        @kb.add("up")
        def _to_start(event) -> None:
            event.current_buffer.cursor_position = 0

        @kb.add("down")
        def _to_end(event) -> None:
            buf = event.current_buffer
            buf.cursor_position = len(buf.text)

        return kb

    def _post(self, intent: Intent) -> None:
        try:
            self._inbox.put_nowait(intent)
        except asyncio.QueueFull:
            out = self._app.output
            out.bell()
            out.flush()

    # --- printing ---------------------------------------------------------

    def _render(self, blocks: tuple[Text, ...]) -> str:
        console = self._console
        console.width = self._columns()  # word-wrap scrollback at the live width
        buf = console.file
        assert isinstance(buf, io.StringIO)
        buf.seek(0)
        buf.truncate()
        for block in blocks:
            console.print(block)
        data = buf.getvalue()
        if data and not data.endswith("\n"):
            data += "\n"
        return data

    def _mark_running(self) -> None:
        # pre_run executes inside _run_async; the first CPR request and redraw
        # follow in the same task step with no await, so any emit that sees
        # _running == True runs after the first frame.
        self._running = True

    def _ensure_bottom(self) -> None:
        """Once, before Screen's first write: put the cursor on the bottom row so
        the bar never draws mid-screen and earlier output is never scrolled off."""
        if self._at_bottom:
            return
        self._at_bottom = True
        self._real_out.write("\n" * (self._rows() - 1))
        self._real_out.flush()

    def _write_direct(self, data: str) -> None:
        self._ensure_bottom()
        self._real_out.write(data)
        self._real_out.flush()

