"""ChatSession: the single consumer of the TUI's inbox and the owner of turn state.

Keyboard intents (from ``Screen``) and wire events (from ``MiraClient``) arrive
on one ``asyncio.Queue`` and are handled strictly in order, one at a time; this
module is the only reader. Every handler mutates state, then hands one print
step to ``Screen.emit`` (blocks plus a freshly derived ``Live``).

Invariant: a message lives in exactly one place — the input box, the live region
(``sending`` / ``queued``), or scrollback. A user message reaches scrollback only
at ``turn_started`` (the server accepted it); every path that cannot deliver it
returns it, with its queued siblings, to the input box. Nothing is dropped.

``Live`` is derived on read by ``_live()`` from the state below, never kept in
step by hand. Private inbox items (``ConnectFailed``, ``RetrySend``, ``AutoReconnectDue``,
``UpdateChecked``, ``ActionDone``/``ActionFailed``, ``ScreenExited``) let background tasks
report to the same single consumer.

A lost connection reconnects automatically with bounded backoff, so a server restart
comes back without the user pressing Enter and a message MIRA pushes after the
restart reaches the open session.

Contract: ``ui/tui/FRONTEND_PLAN.md`` "### ``tui/chat.py``" and its event table.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from rich.console import Group, RenderableType
from rich.text import Text

from . import commands, transcript
from .client import (
    AssistantDelta,
    AuthFailed,
    AuthOk,
    ClientError,
    ClientEvent,
    ContextReset,
    Disconnected,
    MiraClient,
    ModelError,
    Proactive,
    ProtocolError,
    ServerShutdown,
    Thinking,
    ToolUpdate,
    TurnComplete,
    TurnError,
    TurnStarted,
    TurnStopped,
)
from .history import replay_blocks
from .screen import Interrupt, Intent, Live, Quit, Screen, Status, Submit, SubmitNow, ToggleThinking
from .text import ReplyStream, display_lines


# Bounded-wait constants; the only unbounded await is the inbox read.

# One print step: CPR wait (library bound 1 s) plus the emit lock. Generous.
EMIT_TIMEOUT = 10.0
# Joining the frame pump after close(): must exceed client.CLOSE_TIMEOUT (5 s).
PUMP_JOIN_TIMEOUT = 8.0
SCREEN_CLOSE_TIMEOUT = 5.0
CONNECT_CANCEL_TIMEOUT = 5.0
# Background tasks posting private items; a longer wait means the consumer is wedged.
QUEUE_PUT_TIMEOUT = 5.0

# The server answers TURN_BUSY while any turn for this user holds its per-user
# lock. The WS transport frees the lock before queueing the terminal frame, so
# a follow-up send never races the just-finished turn; the steady poll covers
# the window where another turn still holds the lock. A bounced send waits in
# the live region and is resent until MAX_WAIT has passed since the FIRST
# bounce, then goes back to the box. Ctrl+C cancels the wait.
TURN_BUSY_RETRY_DELAYS = (0.5, 1.0, 2.0)
TURN_BUSY_POLL_SECONDS = 3.0
TURN_BUSY_MAX_WAIT_SECONDS = 600.0

# protocol_error codes the server sends without a message_id while a send is
# unanswered (cns/api/websocket_chat.py process_turn setup failures).
_UNATTRIBUTED_SEND_CODES = frozenset({"INVALID_MESSAGE", "TURN_SETUP_FAILED"})

# A lost connection retries on its own: quickly at first, then at a steady poll
# that covers a restart's supervisor delay plus the boot gate, until MAX_WAIT has
# passed since the loss. Enter retries at once.
AUTO_RECONNECT_DELAYS = (2.0, 4.0, 8.0)
AUTO_RECONNECT_POLL_SECONDS = 15.0
AUTO_RECONNECT_MAX_WAIT_SECONDS = 600.0

# Ctrl+C is conditional. Text in the box: clear it, nothing else. Empty
# box: stop what is in flight (turn, TURN_BUSY wait, pending send, connect).
# Only an idle empty box arms this window, and a second Ctrl+C inside it
# quits; every non-idle Ctrl+C handling disarms it.
QUIT_DOUBLE_TAP_SECONDS = 3.0

# Release notice: one read at startup, then daily. The server runs its own
# check on the same cadence and caches the verdict, so this only asks.
UPDATE_CHECK_INTERVAL_SECONDS = 24 * 60 * 60

_LOST_BEFORE_CONFIRM = (
    "connection lost before MIRA confirmed this message — check history before resending"
)

ConnState = Literal["connected", "connecting", "disconnected"]




@dataclass
class ConnectFailed:
    """connect() raised; AuthFailed alone cannot cover every failure path."""

    error: ClientError


@dataclass
class RetrySend:
    """A TURN_BUSY retry timer fired for this flight."""

    flight: _InFlight


@dataclass
class _AutoReconnect:
    """One automatic reconnect run, from a lost connection until it succeeds or gives up."""

    since: float  # monotonic time the connection was lost
    attempts: int = 0


@dataclass
class AutoReconnectDue:
    """An automatic reconnect timer fired for this run."""

    run: _AutoReconnect


@dataclass
class UpdateChecked:
    """A release check came back; ``latest`` is None when there is nothing to show."""

    latest: str | None


@dataclass
class ScreenExited:
    """The screen task ended without close(): fail loud, never run bar-less."""

    error: BaseException


@dataclass
class ActionDone:
    """A slash command's REST action returned a success envelope."""

    command: commands.Command
    data: dict


@dataclass
class ActionFailed:
    """A slash command's REST action failed: server rejection or transport."""

    command: commands.Command
    error: ClientError


Item = ClientEvent | Intent | ConnectFailed | RetrySend | AutoReconnectDue | UpdateChecked | ScreenExited | ActionDone | ActionFailed


@dataclass
class _InFlight:
    """A message handed to the client, not yet accepted by turn_started."""

    message_id: str | None  # None while send_message awaits or a TURN_BUSY retry waits
    text: str
    since: float
    retries: int = 0  # TURN_BUSY retries already scheduled
    first_bounce: float | None = None  # monotonic time of the first TURN_BUSY


@dataclass
class _Turn:
    turn_id: str
    stream: ReplyStream
    started: float
    tools: set[str] = field(default_factory=set)  # unique tool_ids that finished
    running: dict[str, transcript.RunningTool] = field(default_factory=dict)  # by tool_id, shown live
    halt_requested: bool = False
    activity: Literal["replying", "thinking", "tool"] = "replying"
    tool_name: str = ""
    thinking: ReplyStream = field(default_factory=ReplyStream)  # reasoning stream, shown when toggled on
    opened: bool = False  # MIRA's box has printed its labelled first row


class ChatSession:
    """Drives one chat: precondition is that ``client.connect()`` succeeded
    (its ``AuthOk`` is on ``inbox``)."""

    def __init__(
        self,
        client: MiraClient,
        screen: Screen,
        inbox: asyncio.Queue,
        endpoint_label: str,
        base_url: str,
        history_mode: str,
    ) -> None:
        self._client = client
        self._screen = screen
        self._inbox = inbox
        self._label = endpoint_label
        self._base_url = base_url
        self._history_mode = history_mode

        self._outbox: deque[str] = deque()
        self._in_flight: _InFlight | None = None
        self._turn: _Turn | None = None
        self._halt_on_start = False  # Ctrl+C before turn_started: halt on arrival
        self._interrupting_for_send = False  # Alt+Enter halted the turn: send the queue at its end, don't restore it
        self._deferred: list[str] = []  # proactive content held until the turn ends
        self._show_thinking = False  # Ctrl+T view toggle; forward-looking, no replay of what streamed while off
        self._flow = transcript.Flow()  # spacing between scrollback blocks across emits
        self._conn: ConnState = "connecting"  # AuthOk on the inbox flips it
        self._conn_since = time.monotonic()
        self._auto_reconnect: _AutoReconnect | None = None  # set while reconnecting on its own
        self._available_update: str | None = None  # newest release the server knows of, if any
        self._actions_in_flight: set[str] = set()  # command names with a REST action still running
        self._quit_armed_at: float | None = None  # monotonic time of the last idle Ctrl+C

        self._pump_task: asyncio.Task[None] | None = None
        self._connect_task: asyncio.Task[None] | None = None
        self._screen_task: asyncio.Task[None] | None = None
        self._background: set[asyncio.Task] = set()
        self._quitting = False


    async def run(self) -> int:
        """Run until the user quits; returns 0. Raises if the screen dies or a
        handler fails, after restoring the terminal."""
        # Screen is not running yet: this is a direct write above the bar-to-be.
        await self._emit(transcript.banner(self._label, self._base_url))
        await self._load_history()
        self._screen_task = asyncio.create_task(self._screen.run())
        self._screen_task.add_done_callback(self._on_screen_done)
        self._start_pump()
        self._spawn(self._update_loop())
        try:
            while True:
                # Intentionally unbounded: the idle wait of the whole app. It is
                # cancellable, and every other await in the handlers is bounded.
                item = await self._inbox.get()
                if await self._handle(item):
                    return 0
        finally:
            await self._teardown()
            await self._print_remaining()

    async def _load_history(self) -> None:
        """Startup replay into scrollback, before the bar exists (direct
        writes; the pump is not started yet, so wire frames arriving
        meanwhile wait in the socket). The endpoint's ``history_fetch`` mode
        selects how much; the default ``session_plus_summary`` puts the
        newest collapsed segment's summary above the active segment's
        earlier messages — and, when no active segment has messages, shows
        only the summary. A fetch failure degrades to an alert — the chat
        still works without history."""
        await self._emit(transcript.notice("loading earlier messages…"))
        try:
            rows = await self._client.fetch_history(self._history_mode)
        except ClientError as error:
            await self._emit(
                transcript.alert(
                    f"could not load history: {error.message} [{error.code}] — continuing without it"
                )
            )
            return
        blocks = replay_blocks(rows)
        if blocks:
            await self._emit(*blocks)
        else:
            await self._emit(transcript.notice("no earlier messages in this session yet"))

    async def _update_loop(self) -> None:
        """Advisory release check: one read at startup, then daily.

        A background task reporting through the single consumer like every
        other producer — the status row is derived on read, never written from
        here. An unanswerable check (server too old for the route, transport
        failure, no verdict yet) yields None and simply leaves no notice.
        """
        while True:
            latest = await self._client.fetch_available_update()
            await self._post_private(UpdateChecked(latest))
            await asyncio.sleep(UPDATE_CHECK_INTERVAL_SECONDS)


    def _busy(self) -> bool:
        return self._in_flight is not None or self._turn is not None

    def _status(self) -> Status:
        turn, flight = self._turn, self._in_flight
        if self._conn == "disconnected":
            if self._auto_reconnect is not None:
                return Status(
                    "connection lost — reconnecting automatically · Enter to retry now",
                    "busy",
                    self._auto_reconnect.since,
                )
            return Status("disconnected — press Enter to reconnect", "alert")
        if self._conn == "connecting":
            return Status("connecting…", "busy", self._conn_since)
        if turn is not None:
            if turn.halt_requested:
                return Status("stopping…", "busy", turn.started)
            text = {
                "replying": "MIRA is replying… · Ctrl+C clears input / stops · Alt+Enter interrupts and sends",
                "thinking": "MIRA is thinking… · Ctrl+C clears input / stops · Alt+Enter interrupts and sends",
                "tool": f"running {turn.tool_name}… · Ctrl+C clears input / stops · Alt+Enter interrupts and sends",
            }[turn.activity]
            return Status(text, "busy", turn.started)
        if flight is not None:
            if self._halt_on_start:
                return Status("stopping…", "busy", flight.since)
            if flight.message_id is None and flight.retries:
                return Status(
                    "MIRA is busy with another turn — will send when free · Ctrl+C clears input / cancels",
                    "busy",
                    flight.first_bounce,
                )
            return Status("sending…", "busy", flight.since)
        if self._available_update is not None:
            # Last on purpose: connection state and an active turn own the row
            # while present, and the notice reappears when the row frees up.
            # `mira update` is the safe action: it asks before any database
            # change, backs up first, and puts the install back on failure.
            return Status(
                f"MIRA {self._available_update} is available — run `mira update`",
                "alert",
            )
        return Status("")

    def _live(self) -> Live:
        turn, flight = self._turn, self._in_flight
        pending: RenderableType | None = None
        if turn is not None:
            # While reasoning is the live activity and the view is on, the
            # live region tail-follows the thinking stream, not the reply.
            thinking = self._show_thinking and turn.activity == "thinking"
            stream = turn.thinking if thinking else turn.stream
            text = stream.pending()
            if text:
                if stream.breaks_before_pending:
                    text = "\n" + text  # the blank row it commits with, so nothing jumps
                body = (transcript.thinking_lines if thinking else transcript.mira_lines)([text])
                pending = transcript.mira_rows(body, opens=not turn.opened)
            if turn.running:
                tools = transcript.tools_running(
                    list(turn.running.values()), opens=pending is None and not turn.opened
                )
                pending = tools if pending is None else Group(pending, tools)
        return Live(
            pending=pending,
            sending=flight.text if flight else None,
            queued=tuple(self._outbox),
            status=self._status(),
        )

    async def _emit(self, *blocks: RenderableType) -> None:
        placed = self._flow.place(blocks)
        await asyncio.wait_for(self._screen.emit(*placed, live=self._live()), EMIT_TIMEOUT)

    @staticmethod
    def _in_box(turn: _Turn, *bodies: Text) -> list[RenderableType]:
        """``bodies`` as rows of the turn's box; the turn's first row carries the label."""
        rows: list[RenderableType] = []
        for body in bodies:
            rows.append(transcript.mira_rows(body, opens=not turn.opened))
            turn.opened = True
        return rows

    def _meta(self, body: Text) -> RenderableType:
        """A notice or alert: inside MIRA's box while it is open, so the box
        stays unbroken; on its own otherwise."""
        turn = self._turn
        if turn is not None and turn.opened:
            return transcript.mira_rows(body, opens=False)
        return body

    def _refresh(self) -> None:
        self._screen.set_live(self._live())


    async def _handle(self, item: Item) -> bool:
        """Handle one inbox item; True means quit."""
        match item:
            case Submit(text=raw):
                return await self._on_submit(raw)
            case SubmitNow(text=raw):
                return await self._on_submit(raw, interrupt=True)
            case Interrupt():
                return await self._on_interrupt()
            case Quit():
                return True
            case ToggleThinking():
                self._show_thinking = not self._show_thinking
                await self._emit(
                    self._meta(
                        transcript.notice(
                            "showing MIRA's thinking · Ctrl+T hides it"
                            if self._show_thinking
                            else "hiding MIRA's thinking"
                        )
                    )
                )
            case AuthOk():
                await self._on_auth_ok()
            case AuthFailed():
                # Ignored by design: every connect failure also raises
                # ClientError out of connect(), arriving as ConnectFailed.
                pass
            case ConnectFailed(error=error):
                await self._on_connect_failed(error)
            case TurnStarted():
                await self._on_turn_started(item)
            case AssistantDelta():
                await self._on_delta(item)
            case Thinking():
                turn = await self._current(item.turn_id, "thinking")
                if turn is not None:
                    turn.activity = "thinking"
                    lines = (
                        turn.thinking.feed("", item.content) if self._show_thinking else []
                    )
                    if lines:
                        await self._emit(*self._in_box(turn, transcript.thinking_lines(lines)))
                    else:
                        self._refresh()
            case ToolUpdate():
                await self._on_tool(item)
            case ModelError():
                await self._on_model_error(item)
            case RetrySend():
                await self._on_retry_send(item)
            case AutoReconnectDue():
                await self._on_auto_reconnect_due(item)
            case UpdateChecked(latest=latest):
                await self._on_update_checked(latest)
            case ActionDone():
                await self._on_action_done(item)
            case ActionFailed():
                await self._on_action_failed(item)
            case ContextReset():
                await self._on_context_reset(item)
            case TurnComplete():
                await self._on_turn_complete(item)
            case TurnStopped():
                await self._on_turn_stopped(item)
            case TurnError():
                await self._on_turn_error(item)
            case Proactive():
                await self._on_proactive(item)
            case ProtocolError():
                await self._on_protocol_error(item)
            case ServerShutdown():
                await self._client.close()
                await self._lost_connection(
                    transcript.alert(f"server shutting down: {item.message} [{item.code}]")
                )
            case Disconnected():
                await self._lost_connection(
                    transcript.alert(f"connection lost: {item.reason}")
                )
            case ScreenExited(error=error):
                raise error
        return False


    async def _on_submit(self, raw: str, interrupt: bool = False) -> bool:
        text = raw.strip()
        if not text:
            if self._conn == "disconnected":
                await self._reconnect()
            return False
        match commands.parse(text):
            case commands.UnknownCommand(word=word):
                # Not sent to the model; the text stays in the box to fix.
                await self._emit_command_result(
                    transcript.notice(
                        f"unknown command: {word} — commands: {commands.help_list()}"
                    )
                )
                return False
            case commands.CommandHit(command=command, args=args):
                if args:
                    await self._emit_command_result(
                        transcript.notice(f"the /{command.name} command takes no arguments")
                    )
                    return False  # the text stays in the box
                if command.kind == "quit":
                    return True  # /exit: today's early return, before any effect
                if not self._screen.consume_input(raw):
                    return False  # duplicate Enter: the box no longer holds this text
                self._actions_in_flight.add(command.name)
                await self._emit_command_result(
                    transcript.notice(f"running /{command.name}…")
                )
                self._spawn(self._run_action(command))
                return False
            case commands.PlainMessage(text=message):
                if not self._screen.consume_input(raw):
                    return False  # duplicate Enter: the box no longer holds this text
                self._outbox.append(message)
                if self._conn == "disconnected":
                    await self._reconnect()
                elif self._conn == "connected" and not self._busy():
                    await self._send_next()
                elif interrupt and self._turn is not None:
                    # Alt+Enter mid-turn: end the turn so the message goes out
                    # as soon as the server confirms the halt, instead of
                    # queueing behind whatever MIRA was doing.
                    self._interrupting_for_send = True
                    await self._emit(
                        self._meta(transcript.notice("stopping this reply to send your message"))
                    )
                    if not self._turn.halt_requested:
                        await self._halt(self._turn)
                    else:
                        self._refresh()
                else:
                    self._refresh()
        return False


    async def _run_action(self, command: commands.Command) -> None:
        """Background producer: one bounded REST action, reported to the single
        consumer as a private item — the UpdateChecked pattern. Never renders,
        never blocks the consumer: a mid-turn /collapse can hold this task
        through a server-side summarization while turn frames keep flowing."""
        try:
            data = await self._client.post_action("continuum", command.action, {})
        except ClientError as error:
            await self._post_private(ActionFailed(command, error))
            return
        await self._post_private(ActionDone(command, data))

    async def _on_action_done(self, item: ActionDone) -> None:
        self._actions_in_flight.discard(item.command.name)
        await self._emit_command_result(
            transcript.notice(f"/{item.command.name} — {item.command.render(item.data)}")
        )

    async def _on_action_failed(self, item: ActionFailed) -> None:
        self._actions_in_flight.discard(item.command.name)
        error = item.error
        await self._emit_command_result(
            transcript.alert(f"/{item.command.name} failed: {error.message} [{error.code}]")
        )

    async def _emit_command_result(self, body: Text) -> None:
        """A command's notice/alert, placed so it can never land inside a
        paragraph still pending in the live region: idle it is a standalone
        row; mid-turn the pending reply and thinking paragraphs flush first
        (the _on_model_error discipline), then the row joins MIRA's open box."""
        turn = self._turn
        if turn is None:
            await self._emit(self._meta(body))
            return
        bodies: list[Text] = []
        thinking = turn.thinking.flush()  # the row must not land inside a paragraph
        if thinking:
            bodies.append(transcript.thinking_lines(thinking))
        lines = turn.stream.flush()  # the row must not land inside a paragraph
        if lines:
            bodies.append(transcript.mira_lines(lines))
        bodies.append(body)
        await self._emit(*self._in_box(turn, *bodies))

    async def _on_interrupt(self) -> bool:
        """One Ctrl+C, resolved against the box first, then the turn, then
        the session: text in the box clears it and nothing else (a draft
        never halts a turn); an empty box stops what is in flight; an idle
        empty box arms the quit window, and a second Ctrl+C inside it quits."""
        now = time.monotonic()
        armed = (
            self._quit_armed_at is not None
            and now - self._quit_armed_at < QUIT_DOUBLE_TAP_SECONDS
        )
        self._quit_armed_at = None
        if self._screen.clear_input():
            return False
        turn = self._turn
        if turn is not None:
            self._interrupting_for_send = False  # a Ctrl+C halt means restore, not send
            if not turn.halt_requested:
                await self._halt(turn)
            return False
        flight = self._in_flight
        if flight is not None:
            if flight.message_id is None:
                # Waiting out a TURN_BUSY retry (a send in progress cannot overlap
                # this handler): the server holds nothing to halt.
                self._restore_unsent()
                await self._emit(transcript.notice("send cancelled"))
                return False
            if not self._halt_on_start:
                self._halt_on_start = True
                self._refresh()
            return False
        if self._conn == "connecting":
            await self._cancel_connect()
            return False
        if armed:
            return True
        self._quit_armed_at = now
        await self._emit(
            transcript.notice(
                f"press Ctrl+C again within {QUIT_DOUBLE_TAP_SECONDS:.0f}s to exit"
            )
        )
        return False

    async def _halt(self, turn: _Turn) -> None:
        try:
            await self._client.send_halt(turn.turn_id)
        except ClientError as error:
            await self._emit(self._meta(transcript.alert(f"could not stop: {error.message} [{error.code}]")))
            return
        turn.halt_requested = True
        self._refresh()


    async def _send_next(self) -> None:
        text = self._outbox.popleft()
        flight = _InFlight(None, text, time.monotonic())
        self._in_flight = flight
        self._refresh()  # the text is in the live region before the await
        await self._transmit(flight)

    async def _transmit(self, flight: _InFlight) -> None:
        try:
            # Bounded inside the client (SEND_TIMEOUT).
            flight.message_id = await self._client.send_message(flight.text)
        except ClientError as error:
            self._restore_unsent()
            await self._emit(transcript.alert(f"not sent — {error.message} [{error.code}]"))

    async def _on_retry_send(self, item: RetrySend) -> None:
        flight = item.flight
        if self._in_flight is not flight or self._conn != "connected" or self._turn is not None:
            return  # cancelled, restored, or superseded while the timer ran
        await self._transmit(flight)

    async def _retry_after(self, flight: _InFlight, delay: float) -> None:
        await asyncio.sleep(delay)
        await self._post_private(RetrySend(flight))

    def _restore_unsent(self) -> bool:
        """Put the in-flight and queued texts back in the input box, oldest first."""
        texts = ([self._in_flight.text] if self._in_flight else []) + list(self._outbox)
        self._in_flight = None
        self._outbox.clear()
        self._halt_on_start = False
        if not texts:
            return False
        self._screen.restore_input("\n\n".join(texts))
        return True


    def _start_pump(self) -> None:
        self._pump_task = asyncio.create_task(self._client.run())

    def _on_screen_done(self, task: asyncio.Task[None]) -> None:
        if self._quitting:
            return
        if task.cancelled():
            error: BaseException = RuntimeError("the screen task was cancelled")
        else:
            error = task.exception() or RuntimeError("the screen exited without being closed")
        self._spawn(self._post_private(ScreenExited(error)))

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _post_private(self, item: Item) -> None:
        await asyncio.wait_for(self._inbox.put(item), QUEUE_PUT_TIMEOUT)

    async def _join_pump(self) -> None:
        task, self._pump_task = self._pump_task, None
        if task is None:
            return
        if not task.done():
            done, _ = await asyncio.wait({task}, timeout=PUMP_JOIN_TIMEOUT)
            if not done:
                task.cancel()
                await asyncio.wait({task}, timeout=PUMP_JOIN_TIMEOUT)
                await self._emit(
                    transcript.alert(
                        f"the connection did not close within {PUMP_JOIN_TIMEOUT:.0f}s; abandoned it"
                    )
                )
                return
        if not task.cancelled():
            task.result()  # client.run() promises no exception; surface one if it breaks that

    async def _reconnect(self) -> None:
        self._conn = "connecting"
        self._conn_since = time.monotonic()
        self._refresh()
        # The old pump's finally clears client._conn; it must be finished before
        # connect() installs the new socket.
        await self._client.close()
        await self._join_pump()
        self._connect_task = asyncio.create_task(self._connect())

    async def _connect(self) -> None:
        try:
            await self._client.connect()
        except ClientError as error:
            await self._post_private(ConnectFailed(error))
            return
        self._start_pump()

    async def _cancel_connect(self) -> None:
        task, self._connect_task = self._connect_task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=CONNECT_CANCEL_TIMEOUT)
        await self._client.close()
        await self._join_pump()
        self._conn = "disconnected"
        self._auto_reconnect = None
        restored = self._restore_unsent()
        notice = "connect cancelled — press Enter to reconnect"
        if restored:
            notice += "; unsent messages are back in the input box"
        await self._emit(transcript.notice(notice))

    async def _on_auth_ok(self) -> None:
        if self._conn != "connecting":
            return  # a connect the user already cancelled
        self._conn = "connected"
        if self._auto_reconnect is not None:
            self._auto_reconnect = None
            await self._emit(transcript.notice("reconnected"))
        if self._outbox and not self._busy():
            await self._send_next()
        else:
            self._refresh()

    async def _on_connect_failed(self, error: ClientError) -> None:
        if self._conn != "connecting":
            return  # a connect the user already cancelled
        self._connect_task = None
        self._conn = "disconnected"
        self._restore_unsent()
        run = self._auto_reconnect
        if run is not None and error.code != "AUTH_FAILED":
            if time.monotonic() - run.since < AUTO_RECONNECT_MAX_WAIT_SECONDS:
                self._schedule_auto_reconnect(run)
                self._refresh()
                return
            self._auto_reconnect = None
            await self._emit(
                transcript.alert(
                    f"could not reconnect for {int(AUTO_RECONNECT_MAX_WAIT_SECONDS)}s: "
                    f"{error.message} [{error.code}] — press Enter to retry"
                )
            )
            return
        self._auto_reconnect = None
        blocks = [transcript.alert(f"could not connect: {error.message} [{error.code}]")]
        if error.code == "AUTH_FAILED":
            blocks.append(transcript.notice("re-mint the API token: python3 -m ui.tui --login"))
        await self._emit(*blocks)

    async def _lost_connection(self, headline: Text) -> None:
        blocks: list[RenderableType] = []
        turn = self._turn
        if turn is not None:
            bodies = self._tail_bodies(turn) + [transcript.notice("reply cut off")]
            blocks.extend(self._in_box(turn, *bodies))
            blocks.append(transcript.mira_close())
        blocks.append(headline)
        flight = self._in_flight
        if flight is not None:
            blocks.append(
                transcript.notice(
                    _LOST_BEFORE_CONFIRM
                    if flight.message_id is not None
                    else "not sent — the message never went out; it is back in the input box"
                )
            )
        self._restore_unsent()
        self._turn = None
        self._interrupting_for_send = False
        self._conn = "disconnected"
        blocks.extend(self._take_deferred())
        if not self._quitting and self._auto_reconnect is None:
            self._auto_reconnect = _AutoReconnect(since=time.monotonic())
            self._schedule_auto_reconnect(self._auto_reconnect)
        await self._emit(*blocks)

    def _schedule_auto_reconnect(self, run: _AutoReconnect) -> None:
        n = run.attempts
        delay = AUTO_RECONNECT_DELAYS[n] if n < len(AUTO_RECONNECT_DELAYS) else AUTO_RECONNECT_POLL_SECONDS
        run.attempts += 1
        self._spawn(self._auto_reconnect_after(run, delay))

    async def _auto_reconnect_after(self, run: _AutoReconnect, delay: float) -> None:
        await asyncio.sleep(delay)
        await self._post_private(AutoReconnectDue(run))

    async def _on_auto_reconnect_due(self, item: AutoReconnectDue) -> None:
        if item.run is not self._auto_reconnect or self._conn != "disconnected":
            return  # reconnected, cancelled, or superseded while the timer ran
        await self._reconnect()

    async def _on_update_checked(self, latest: str | None) -> None:
        """Record one release-check result. Only a change repaints: the poll
        re-runs daily and the row already says what it said."""
        if latest == self._available_update:
            return
        self._available_update = latest
        self._refresh()


    async def _current(self, turn_id: str, kind: str) -> _Turn | None:
        turn = self._turn
        if turn is not None and turn.turn_id == turn_id:
            return turn
        await self._emit(self._meta(transcript.alert(f"protocol error: {kind} for a turn that is not active")))
        return None

    async def _on_turn_started(self, event: TurnStarted) -> None:
        flight = self._in_flight
        if flight is None or flight.message_id != event.message_id or self._turn is not None:
            await self._emit(
                transcript.alert("protocol error: turn_started for a message that is not in flight")
            )
            return
        now = time.monotonic()
        self._turn = _Turn(event.turn_id, ReplyStream(), now)
        self._in_flight = None
        # The user's box and the live region change in ONE print step: the
        # text is in the live region until this emit lands it in scrollback.
        await self._emit(transcript.you(flight.text))
        if self._halt_on_start:
            self._halt_on_start = False
            await self._halt(self._turn)

    async def _on_delta(self, event: AssistantDelta) -> None:
        turn = await self._current(event.turn_id, "assistant_delta")
        if turn is None:
            return
        bodies: list[Text] = []
        if turn.activity == "thinking":
            # The reply starts: reasoning's unfinished last line belongs above it.
            thinking = turn.thinking.flush()
            if thinking:
                bodies.append(transcript.thinking_lines(thinking))
        turn.activity = "replying"
        lines = turn.stream.feed(event.entry_id, event.content)
        if lines:
            bodies.append(transcript.mira_lines(lines))
        if bodies:
            await self._emit(*self._in_box(turn, *bodies))
        else:
            self._refresh()

    async def _on_tool(self, event: ToolUpdate) -> None:
        turn = await self._current(event.turn_id, "tool")
        if turn is None:
            return
        bodies: list[Text] = []
        thinking = turn.thinking.flush()  # a tool event ends the reasoning step too
        if thinking:
            bodies.append(transcript.thinking_lines(thinking))
        lines = turn.stream.flush()  # a tool event ends the provider step
        if lines:
            bodies.append(transcript.mira_lines(lines))
        if event.event in ("tool_detected", "tool_executing"):
            turn.activity = "tool"
            turn.tool_name = event.tool_name
            run = turn.running.setdefault(
                event.tool_id, transcript.RunningTool(event.tool_name, time.monotonic())
            )
            if event.arguments is not None:
                run.arguments = event.arguments
        else:
            run = turn.running.pop(event.tool_id, None)
            if turn.running:
                turn.tool_name = next(reversed(turn.running.values())).name
            else:
                turn.activity = "replying"
            turn.tools.add(event.tool_id)
            ok = event.event == "tool_completed" and not event.is_error
            arguments = event.arguments if event.arguments is not None else run and run.arguments
            seconds = time.monotonic() - run.started if run else None
            bodies.append(transcript.tool_line(event.tool_name, ok, arguments, seconds))
            if not ok and event.result and event.result.strip():
                bodies.append(transcript.tool_error(event.result))
        if bodies:
            await self._emit(*self._in_box(turn, *bodies))
        else:
            self._refresh()

    async def _on_model_error(self, event: ModelError) -> None:
        notice = transcript.notice(f"model error: {event.message} — retrying")
        turn = self._turn
        if turn is None or turn.turn_id != event.turn_id:
            await self._emit(notice)
            return
        bodies: list[Text] = []
        thinking = turn.thinking.flush()  # the notice must not land inside a paragraph
        if thinking:
            bodies.append(transcript.thinking_lines(thinking))
        lines = turn.stream.flush()  # the notice must not land inside a paragraph
        if lines:
            bodies.append(transcript.mira_lines(lines))
        await self._emit(*self._in_box(turn, *bodies, notice))

    async def _on_context_reset(self, event: ContextReset) -> None:
        turn = await self._current(event.turn_id, "context_reset")
        if turn is None:
            return
        shown = turn.stream.has_text
        turn.stream.discard()
        turn.thinking.discard()
        text = (
            "the server discarded the text above and is restarting the reply"
            if shown
            else "the server discarded the reply so far and is restarting it"
        )
        await self._emit(*self._in_box(turn, transcript.notice(text)))

    async def _on_turn_complete(self, event: TurnComplete) -> None:
        turn = await self._current(event.turn_id, "turn_complete")
        if turn is None:
            return
        thinking = turn.thinking.finish()
        lines = turn.stream.finish()
        shown = turn.stream.has_text
        if not shown:
            lines = display_lines(event.response)  # never when the stream displayed text
        bodies: list[Text] = []
        if thinking:
            bodies.append(transcript.thinking_lines(thinking))
        if lines:
            bodies.append(transcript.mira_lines(lines))
        elif not shown:
            bodies.append(transcript.notice("(empty reply)"))
        if turn.tools:
            bodies.append(transcript.reply_footer(len(turn.tools)))
        await self._end_reply(turn, bodies)

    async def _on_turn_stopped(self, event: TurnStopped) -> None:
        turn = await self._current(event.turn_id, "turn_stopped")
        if turn is None:
            return
        bodies = self._tail_bodies(turn)
        if turn.halt_requested:
            bodies.append(transcript.notice("stopped"))
        elif event.reason == "stall":
            bodies.append(transcript.alert(
                "the server stalled (its event loop stopped turning) "
                "and dropped this reply"
            ))
        else:
            bodies.append(transcript.notice("the server stopped this reply"))
        await self._end_reply(turn, bodies)

    async def _on_turn_error(self, event: TurnError) -> None:
        turn = await self._current(event.turn_id, "turn_error")
        if turn is None:
            return
        bodies = self._tail_bodies(turn)
        bodies.append(transcript.alert(f"error: {event.message} [{event.code}]"))
        await self._end_reply(turn, bodies)

    @staticmethod
    def _tail_bodies(turn: _Turn) -> list[Text]:
        bodies: list[Text] = []
        thinking = turn.thinking.finish()
        if thinking:
            bodies.append(transcript.thinking_lines(thinking))
        lines = turn.stream.finish()
        if lines:
            bodies.append(transcript.mira_lines(lines))
        return bodies

    async def _end_reply(self, turn: _Turn, bodies: list[Text]) -> None:
        """Common tail of every turn end: the last rows and the bottom border
        of MIRA's box, deferred proactives, then the queue — back to the box
        after a Ctrl+C halt, otherwise the next message is sent."""
        interrupting = self._interrupting_for_send
        self._interrupting_for_send = False
        self._turn = None
        blocks = self._in_box(turn, *bodies)
        blocks.append(transcript.mira_close())
        blocks.extend(self._take_deferred())
        if turn.halt_requested and self._outbox and not interrupting:
            self._restore_unsent()
            blocks.append(transcript.notice("queued messages returned to the input box"))
        await self._emit(*blocks)
        if self._outbox and self._conn == "connected":
            await self._send_next()

    def _take_deferred(self) -> list[RenderableType]:
        blocks: list[RenderableType] = []
        for content in self._deferred:
            lines = display_lines(content)
            if lines:
                blocks.extend(transcript.mira_reply(lines))
        self._deferred.clear()
        return blocks

    async def _on_proactive(self, event: Proactive) -> None:
        if self._busy():
            self._deferred.append(event.content)
            return
        lines = display_lines(event.content)
        if lines:
            await self._emit(*transcript.mira_reply(lines))

    async def _on_protocol_error(self, event: ProtocolError) -> None:
        flight = self._in_flight
        attributed = flight is not None and (
            (event.message_id is not None and event.message_id == flight.message_id)
            or (event.code in _UNATTRIBUTED_SEND_CODES and self._turn is None)
        )
        busy = attributed and event.code == "TURN_BUSY"
        waited = 0.0
        if busy and not self._halt_on_start:
            now = time.monotonic()
            if flight.first_bounce is None:
                flight.first_bounce = now
            waited = now - flight.first_bounce
        if busy and not self._halt_on_start and waited < TURN_BUSY_MAX_WAIT_SECONDS:
            n = flight.retries
            delay = TURN_BUSY_RETRY_DELAYS[n] if n < len(TURN_BUSY_RETRY_DELAYS) else TURN_BUSY_POLL_SECONDS
            flight.retries += 1
            flight.message_id = None  # the bounced id is dead; the retry mints a new one
            self._refresh()
            self._spawn(self._retry_after(flight, delay))
        elif attributed:
            self._restore_unsent()
            if busy:
                text = f"not sent — {event.message} [TURN_BUSY] (waited {int(waited)}s)"
            else:
                text = f"not sent — {event.message} [{event.code}]"
            await self._emit(self._meta(transcript.alert(text)))
        elif event.code == "NO_MATCHING_ACTIVE_TURN":
            await self._emit(self._meta(transcript.notice("nothing to stop — the reply had already ended")))
        else:
            await self._emit(self._meta(transcript.alert(f"protocol error: {event.message} [{event.code}]")))


    async def _teardown(self) -> None:
        """Bar first, then the socket: ``Screen.close`` erases the bar, the
        client closes quietly (no Disconnected), tasks are joined with bounds."""
        self._quitting = True
        self._screen.close()
        screen_task = self._screen_task
        if screen_task is not None:
            done, _ = await asyncio.wait({screen_task}, timeout=SCREEN_CLOSE_TIMEOUT)
            if not done:
                screen_task.cancel()
                await asyncio.wait({screen_task}, timeout=SCREEN_CLOSE_TIMEOUT)
        for task in list(self._background):
            task.cancel()  # pending retry timers
        connect_task, self._connect_task = self._connect_task, None
        if connect_task is not None and not connect_task.done():
            connect_task.cancel()
            await asyncio.wait({connect_task}, timeout=CONNECT_CANCEL_TIMEOUT)
        await self._client.close()
        await self._join_pump()
        if screen_task is not None and screen_task.done() and not screen_task.cancelled():
            screen_task.result()

    async def _print_remaining(self) -> None:
        """After the bar is gone: print whatever the user would otherwise lose —
        a partial reply, held proactive messages, unsent and unconfirmed texts."""
        blocks: list[RenderableType] = []
        turn = self._turn
        if turn is not None:
            bodies = self._tail_bodies(turn) + [transcript.notice("reply cut off")]
            blocks.extend(self._in_box(turn, *bodies))
            blocks.append(transcript.mira_close())
            self._turn = None
        blocks.extend(self._take_deferred())
        flight = self._in_flight
        if flight is not None:
            label = "not sent" if flight.message_id is None else "sent but not confirmed by MIRA"
            blocks.append(transcript.notice(f"{label}: {flight.text}"))
            self._in_flight = None
        blocks.extend(transcript.notice(f"not sent: {text}") for text in self._outbox)
        self._outbox.clear()
        blocks.extend(
            transcript.notice(f"/{name} was still running when you quit — its outcome is unknown")
            for name in sorted(self._actions_in_flight)
        )
        self._actions_in_flight.clear()
        if blocks:
            await self._emit(*blocks)
