"""Screen: owns prompt_toolkit and every byte the TUI writes to the terminal.

Layout is normal terminal scrollback above a bar pinned to the bottom row:
live region (in-progress paragraph, outbox lines, status line), a ``─`` rule,
the input box, a ``─`` rule. The box grows upward with its content.

Scrollback is append-only and has exactly one writer while the app runs: the
print step in ``Screen.emit`` (wait for pending cursor-position replies, then
synchronously erase the bar and write the ANSI; request a cursor position,
wait for its answer, then redraw). The step never leaves raw mode, so keys
typed during an emit are never echoed by the tty into scrollback
(``run_in_terminal`` switches to cooked mode, which turns ECHO on — the
double-print bug). Nothing here counts rows or moves the cursor by hand;
prompt_toolkit's renderer measures and erases the bar.

Callers hand over Rich renderables (``transcript`` elements), rendered at
the live terminal width for scrollback and the live region alike;
``Intent`` objects (``Submit``, ``Interrupt``, ``Quit``, ``ToggleThinking``)
are posted to the caller's inbox with ``put_nowait``. Contract: ``tui/FRONTEND_PLAN.md`` "### ``tui/screen.py``".
"""

from __future__ import annotations

import asyncio
import io
import sys
import time
from asyncio import Future
from dataclasses import dataclass
from typing import Literal

from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
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
from prompt_toolkit.renderer import CPR_Support
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea
from rich.console import Console, RenderableType

from tui.transcript import spinner

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

ELLIPSIS = "…"
RULE_CHAR = "─"

STYLE = Style.from_dict(
    {
        "rule": "ansibrightblack",
        "status.idle": "ansibrightblack",
        "status.busy": "ansibrightblack",
        "status.alert": "ansired",
        "outbox": "ansicyan",
    }
)


@dataclass(frozen=True)
class Status:
    text: str
    tone: Literal["idle", "busy", "alert"] = "idle"
    since: float | None = None  # time.monotonic() start; busy -> spinner + live "· 14s"


@dataclass(frozen=True)
class Live:
    pending: RenderableType | None = None  # in-progress reply rows (transcript.mira_rows)
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


def _console() -> Console:
    return Console(
        file=io.StringIO(),
        force_terminal=True,
        color_system="standard",
        markup=False,
        emoji=False,
        highlight=False,
    )


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
        # Terminal size at the last completed print step. A mismatch against
        # the live size (or a CPR still in flight) marks a resize since then:
        # the renderer's erase anchor and cached geometry no longer match the
        # physical screen.
        self._printed_size = None
        # Resize answer-epoch gate: _stale_cpr pre-resize cursor answers to
        # discard; _anchor_waiter resolves when the first post-resize answer
        # arrives, and until then _gated_redraw refuses to paint.
        self._stale_cpr = 0
        self._anchor_waiter: Future | None = None
        self._cpr_dead = False  # terminal never answered a cursor request
        self._orig_redraw = None
        self._orig_report_cursor = None

        # no soft_wrap: Rich word-wraps at the live terminal width (set per
        # render) instead of letting the terminal hard-cut mid-word.
        # Scrollback therefore does not reflow on a later resize.
        self._console = _console()
        # The live region renders through its own console: it is drawn from
        # inside the print step, after _render filled the scrollback buffer.
        self._live_console = _console()

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

    async def emit(self, *blocks: RenderableType, live: Live | None = None) -> None:
        """Print blocks to scrollback (and apply ``live``) in one print step."""
        if live is not None:
            self._live = live
        if not self._running or self._closing:
            if self._closing and self._running:
                await self._finished.wait()
            self._write_direct(self._render(blocks))
            return
        async with self._lock:
            app = self._app
            await app.renderer.wait_for_cpr_responses()
            if not self._running or self._closing:
                if self._running:
                    await self._finished.wait()
                self._write_direct(self._render(blocks))
                return
            # Re-anchor after a resize: once the terminal reflows, the
            # renderer's relative erase and cached geometry describe a
            # screen that no longer exists, and a write from that anchor
            # deposits rows where no later erase ever reaches. Scrollback is
            # append-only, so such rows are permanent. Settling the cursor
            # exchange and re-rendering here keeps both the width and the
            # anchor fresh at the atomic print step below.
            size = app.output.get_size()
            if app.renderer.waiting_for_cpr or self._printed_size != size:
                await self._await_cpr(self._request_cpr())
                size = app.output.get_size()
            data = self._render(blocks)
            # The erase→write stretch is synchronous; the redraw waits for a
            # post-write cursor answer. A repaint with no answer since the
            # erase (min_avail 0) anchors a minimum-height bar at the
            # physical cursor — mid-screen whenever the write did not end at
            # the bottom, i.e. right after a resize — and no later erase
            # reaches those rows; the next geometry-correcting repaint
            # scrolls them into append-only scrollback.
            app.renderer.erase()
            app.output.write_raw(data)
            app.output.flush()
            await self._await_anchor()
            app._redraw()
            self._printed_size = size

    async def _await_anchor(self) -> None:
        """Wait until a cursor answer has re-anchored the renderer since the
        last erase (``_min_available_height`` is 0 after a reset and nonzero
        after every true report), then let the caller redraw. A resize whose
        answer is still in flight keeps the loop on its waiter; terminals
        that never answer CPR degrade, after one bounded wait, to painting
        from best-known geometry — the library's own behavior."""
        renderer = self._app.renderer
        if renderer.cpr_support == CPR_Support.NOT_SUPPORTED or self._cpr_dead:
            return
        try:
            await asyncio.wait_for(self._request_cpr(), timeout=1.0)
        except asyncio.TimeoutError:
            self._cpr_dead = True  # terminal never answers: stop asking
            return
        while renderer._min_available_height == 0:
            waiter = self._anchor_waiter
            if waiter is None or waiter.done():
                break  # no resize in flight; paint from current state
            try:
                await asyncio.wait_for(waiter, timeout=1.0)
            except asyncio.TimeoutError:
                break

    async def _await_cpr(self, fut: Future) -> None:
        try:
            await asyncio.wait_for(fut, timeout=1.0)
        except asyncio.TimeoutError:
            self._cpr_dead = True

    def _request_cpr(self) -> Future:
        """Post a cursor-position request mid-lifetime and return its
        answer future. ``Application._request_absolute_cursor_position``
        asserts a freshly reset renderer (cursor at the draw origin); this
        is the same request, valid while a bar is on screen."""
        fut = Future()
        self._app.renderer._waiting_for_cpr_futures.append(fut)
        self._app.output.ask_for_cpr()
        return fut

    def _handle_resize(self) -> None:
        """SIGWINCH handler, installed over ``Application._on_resize`` by
        run(). prompt_toolkit's own handler erases, requests the cursor
        position, and redraws immediately — before the answer that anchors
        the renderer — so the repaint paints from stale geometry and its
        newlines-at-the-bottom scroll freshly painted bar rows into
        append-only scrollback. This variant discards the stale answers,
        erases, requests, and repaints only once a post-resize answer has
        re-anchored the renderer."""
        app = self._app
        renderer = app.renderer
        # Answer-epoch discipline. Every cursor request posted before this
        # resize has an answer in flight that was computed against
        # pre-reflow geometry; processed now, it poisons
        # ``_min_available_height`` with rows that no longer exist. Answers
        # arrive in request order, so the next ``_stale_cpr`` answers are
        # dropped by the ``report_absolute_cursor_row`` shadow, and the
        # first answer after them is post-resize truth.
        self._stale_cpr += len(renderer._waiting_for_cpr_futures)
        for f in renderer._waiting_for_cpr_futures:
            if not f.done():
                f.set_result(None)  # wake waiters; their redraws stay gated
        renderer._waiting_for_cpr_futures.clear()
        app.renderer.erase(leave_alternate_screen=False)
        app._request_absolute_cursor_position()
        if renderer.cpr_support != CPR_Support.NOT_SUPPORTED:
            self._anchor_waiter = Future()
            app.create_background_task(self._resize_repaint())
        else:
            app._redraw()  # no answers ever: paint from best-known geometry

    async def _resize_repaint(self) -> None:
        waiter = self._anchor_waiter
        if waiter is None:
            return
        try:
            await asyncio.wait_for(waiter, timeout=1.0)
        except asyncio.TimeoutError:
            pass  # answer never came; ungate and paint from current state
        if self._running and not self._closing:
            self._app._redraw()

    def _report_cursor(self, row: int) -> None:
        """Shadow of ``Renderer.report_absolute_cursor_row``, installed by
        run(). Drops pre-resize answers (see ``_handle_resize``), opens the
        resize gate, and re-arms the repaints the gate skipped."""
        if self._stale_cpr > 0:
            self._stale_cpr -= 1
            return
        self._orig_report_cursor(row)
        waiter = self._anchor_waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(None)
        self._app.invalidate()  # repaint what the gate skipped, now anchored

    def _gated_redraw(self, render_as_done: bool = False) -> None:
        """Shadow of ``Application._redraw``, installed by run(). Two gates,
        both satisfied by a true cursor answer and re-armed by its
        invalidate: (a) while a resize's answer is still in flight, no
        repaint may run — it would anchor on pre-resize geometry; (b) while
        no answer has arrived since the last erase (``_min_available_height``
        0, the post-reset state mid-print-step), a repaint would anchor a
        minimum-height bar at the physical cursor — mid-screen whenever the
        write did not end at the bottom — and no later erase reaches those
        rows. Terminals that cannot answer degrade to the library's paint-
        from-best-known behavior. Teardown (``render_as_done``) always passes."""
        if not render_as_done and not self._cpr_dead:
            renderer = self._app.renderer
            if renderer.cpr_support == CPR_Support.NOT_SUPPORTED:
                pass  # no answers ever: paint like the library does
            else:
                waiter = self._anchor_waiter
                if waiter is not None and not waiter.done():
                    return
                if renderer._min_available_height == 0:
                    return  # unanchored: skip; the answer repaints
        self._orig_redraw(render_as_done)

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
        # Shadow three library hooks with instance attributes (every internal
        # call site resolves them, and attach_winch_signal_handler removes
        # ours with its own): the SIGWINCH handler, the repaint entry point
        # (gates the 0.5s auto-refresh and every other repaint on the resize
        # answer epoch), and the cursor-answer sink (drops pre-resize
        # answers). Rationale in each method's docstring.
        app = self._app
        app._on_resize = self._handle_resize
        self._orig_redraw = app._redraw
        app._redraw = self._gated_redraw
        self._orig_report_cursor = app.renderer.report_absolute_cursor_row
        app.renderer.report_absolute_cursor_row = self._report_cursor
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
            # The same renderer and width as scrollback, so a line keeps its
            # wrap when it commits. Cursor fragment at the end makes the
            # Window follow the tail.
            pending = self._live.pending
            if pending is None:
                return []
            console = self._live_console
            console.width = self._columns()
            with console.capture() as capture:
                console.print(pending)
            # The renderable ends its last row with a newline; dropping it
            # keeps the cursor (and the status row) right under the text.
            rows = capture.get().removesuffix("\n")
            return [*to_formatted_text(ANSI(rows)), ("[SetCursorPosition]", "")]

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
                frame = spinner(time.monotonic())
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
            filter=Condition(lambda: self._live.pending is not None),
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

    def _render(self, blocks: tuple[RenderableType, ...]) -> str:
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

