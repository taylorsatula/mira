"""MiraClient: the complete I/O layer of the MIRA TUI.

Pure asyncio — no textual, no server-tree imports. Owns the WebSocket
connection (auth + frame pump) and the REST history pager, and surfaces
everything to the app as typed ``ClientEvent`` dataclasses on an
app-owned ``asyncio.Queue``. The UI never touches the wire.

Frame shapes are owned by ``tui/protocol.py`` (strict mirror of the
server); this module only maps validated frames to events. A frame that
fails validation is surfaced as ``ProtocolError`` — never silently
dropped, never allowed to crash the pump.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass

import httpx
import websockets

from tui.endpoints import EndpointConfig, HISTORY_FETCH_MODES
from tui.protocol import (
    AuthFrame,
    HaltFrame,
    HistoryMessage,
    MessageFrame,
    dump_outbound_frame,
    parse_history_response,
    parse_inbound_frame,
)

# --- Bounded-wait constants (no unbounded await anywhere in this module) ---

WS_OPEN_TIMEOUT = 15.0  # prompt-pinned: 15 s to establish the socket
AUTH_WAIT_TIMEOUT = 12.0  # server closes with AUTH_TIMEOUT at 10 s; cover it
SEND_TIMEOUT = 10.0
CLOSE_TIMEOUT = 5.0
HTTP_PAGE_TIMEOUT = 30.0
# Server tool frames carry the full, untruncated tool result; the websockets
# default (1 MiB) would close the socket with 1009 on one large result mid-turn.
MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_HISTORY_PAGES = 200  # paging bound; exceeding it fails loud, never truncates


class ClientError(Exception):
    """The only exception this module raises — wire/pager failures with
    the server's (or a client-side) code and a real message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


# --- Client events (pinned contract; the app is written against these) -----


@dataclass
class AuthOk:
    user_id: str


@dataclass
class AuthFailed:
    code: str
    message: str


@dataclass
class TurnStarted:
    turn_id: str
    message_id: str
    segment_id: str


@dataclass
class AssistantDelta:
    turn_id: str
    segment_id: str
    entry_id: str
    content: str


@dataclass
class Thinking:
    turn_id: str
    segment_id: str
    content: str


@dataclass
class ToolUpdate:
    turn_id: str
    segment_id: str
    event: str
    tool_name: str
    tool_id: str
    arguments: dict | None
    result: str | None
    is_error: bool | None


@dataclass
class ModelError:
    turn_id: str
    segment_id: str
    message: str


@dataclass
class ContextReset:
    turn_id: str
    segment_id: str


@dataclass
class TurnComplete:
    turn_id: str
    segment_id: str
    continuum_id: str
    response: str
    tools_used: list
    processing_time_ms: int
    emotion: str | None


@dataclass
class TurnStopped:
    turn_id: str
    segment_id: str
    reason: str


@dataclass
class TurnError:
    turn_id: str
    segment_id: str
    code: str
    message: str


@dataclass
class Proactive:
    message_id: str
    turn_id: str | None
    segment_id: str | None
    content: str
    created_at: str


@dataclass
class ServerShutdown:
    code: str
    message: str


@dataclass
class ProtocolError:
    code: str
    message: str
    message_id: str | None


@dataclass
class Disconnected:
    reason: str


ClientEvent = (
    AuthOk
    | AuthFailed
    | TurnStarted
    | AssistantDelta
    | Thinking
    | ToolUpdate
    | ModelError
    | ContextReset
    | TurnComplete
    | TurnStopped
    | TurnError
    | Proactive
    | ServerShutdown
    | ProtocolError
    | Disconnected
)


def _is_sentinel(message: HistoryMessage) -> bool:
    """A segment sentinel row. Live-verified: the server currently sends
    ``is_segment_boundary`` as JSON ``true``; the plan recorded a string
    ``"true"`` — accept both so neither representation slips through."""
    value = message.metadata.get("is_segment_boundary")
    return value is True or value == "true"


def _chronological(pages: list[list[HistoryMessage]]) -> list[HistoryMessage]:
    """Pages are fetched newest-window-first, rows within a page are
    oldest->newest; emit everything oldest->newest."""
    return [message for page in reversed(pages) for message in page]


class MiraClient:
    """WebSocket connection + REST history pager; emits typed events only."""

    def __init__(self, config: EndpointConfig, events: asyncio.Queue[ClientEvent]) -> None:
        self._config = config
        self._events = events
        self._conn: websockets.asyncio.client.ClientConnection | None = None
        self._closing = False  # close() initiated by us: run() returns quietly
        self._server_shutdown_seen = False
        self._active_turn_id: str | None = None  # set by TurnStarted, cleared by terminals

    # --- public API ---------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._conn is not None and not self._closing

    async def connect(self) -> None:
        """Open the socket, authenticate, and surface the outcome.

        On ANY auth-phase failure: ``AuthFailed`` lands on the queue and
        ``ClientError`` raises from here.
        """
        if self.connected:
            raise ClientError("ALREADY_CONNECTED", "MiraClient.connect() called twice")
        # close() sets _closing; a fresh connect on the same client is a
        # fresh lifecycle — without this reset, connected stays False
        # forever after /reconnect and _send_frame raises NOT_CONNECTED.
        self._closing = False
        # Likewise for the shutdown flag: a stale True would make run() swallow
        # the next unexpected drop.
        self._server_shutdown_seen = False
        ws_url = self._ws_url()
        try:
            conn = await asyncio.wait_for(
                websockets.connect(
                    ws_url, open_timeout=WS_OPEN_TIMEOUT, max_size=MAX_FRAME_BYTES
                ),
                timeout=WS_OPEN_TIMEOUT + 5.0,
            )
        except asyncio.TimeoutError:
            await self._fail_auth(
                "AUTH_CONNECTION_FAILED", f"Timed out opening WebSocket to {ws_url}"
            )
        except Exception as error:
            await self._fail_auth(
                "AUTH_CONNECTION_FAILED", f"Could not open WebSocket to {ws_url}: {error}"
            )
        try:
            user_id = await self._authenticate(conn)
        except ClientError as error:
            await conn.close()
            await self._fail_auth(error.code, error.message)
        except Exception as error:
            await conn.close()
            await self._fail_auth("AUTH_CONNECTION_FAILED", f"Auth phase failed: {error!r}")
        self._conn = conn
        await self._events.put(AuthOk(user_id=user_id))

    async def run(self) -> None:
        """Frame pump: read until the socket closes.

        Every frame is validated via protocol.parse_inbound_frame and mapped
        to a ClientEvent on the queue. Unexpected close -> Disconnected; a
        close initiated by close() or announced by server_shutdown -> quiet
        return. No exception escapes this coroutine.
        """
        if self._conn is None:
            raise ClientError("NOT_CONNECTED", "run() called before connect()")
        conn = self._conn
        try:
            # websockets' built-in ping_interval/ping_timeout (20 s) bounds a
            # dead connection; recv itself cannot stall past that.
            async for raw in conn:
                event = self._frame_to_event(raw)
                if event is not None:
                    await self._events.put(event)
            # The loop also ends normally on a clean (1000/1001) server close.
            if not self._closing and not self._server_shutdown_seen:
                code = conn.close_code
                reason = conn.close_reason
                await self._events.put(
                    Disconnected(
                        reason=f"server closed the connection (code {code}"
                        + (f", {reason}" if reason else "")
                        + ")"
                    )
                )
        except websockets.exceptions.ConnectionClosed as error:
            if not self._closing and not self._server_shutdown_seen:
                await self._events.put(Disconnected(reason=str(error) or error.__class__.__name__))
        except Exception as error:
            if not self._closing:
                await self._events.put(
                    Disconnected(reason=f"{error.__class__.__name__}: {error}")
                )
        finally:
            self._active_turn_id = None
            self._conn = None

    async def send_message(self, content: str) -> str:
        """Send one user message; returns its message_id."""
        if not content:
            raise ClientError("EMPTY_MESSAGE", "Message content is empty")
        if len(content) > 100_000:
            raise ClientError(
                "MESSAGE_TOO_LONG",
                f"Message is {len(content)} characters; the maximum is 100_000",
            )
        message_id = uuid.uuid4()
        frame = MessageFrame(
            type="message",
            message_id=message_id,
            content=content,
            include_thinking=self._config.include_thinking,
        )
        await self._send_frame(frame)
        return str(message_id)

    async def send_halt(self, turn_id: str | None = None) -> None:
        """Halt the active server turn.

        The protocol's HaltFrame requires a turn_id matching the server's
        active turn; when the caller omits it, the turn most recently
        announced by TurnStarted (and not yet terminated) is used.
        """
        resolved = turn_id or self._active_turn_id
        if resolved is None:
            raise ClientError(
                "NO_ACTIVE_TURN",
                "No turn_id given and no turn is active on this connection",
            )
        frame = HaltFrame(type="halt", turn_id=uuid.UUID(resolved))
        await self._send_frame(frame)

    async def fetch_history(self, mode: str, limit: int = 50) -> list[HistoryMessage]:
        """Fetch conversation history in one of the three modes.

        Rows are returned oldest->newest. Keyset pagination only; the fetch
        is capped at MAX_HISTORY_PAGES pages and fails loud past that.
        """
        if mode not in HISTORY_FETCH_MODES:
            raise ClientError(
                "UNKNOWN_HISTORY_MODE",
                f"Unknown history mode {mode!r}; expected one of {', '.join(HISTORY_FETCH_MODES)}",
            )
        if limit < 1:
            raise ClientError("BAD_HISTORY_LIMIT", f"limit must be >= 1, got {limit}")

        pages: list[list[HistoryMessage]] = []
        before: str | None = None
        session_rows: list[HistoryMessage] | None = None  # rows after the newest sentinel
        summary_sentinel: HistoryMessage | None = None
        found_boundary = False

        while True:
            page, has_more, next_before = await self._fetch_history_page(before, limit)
            pages.append(page)

            # Boundary/summary detection only applies to the session modes;
            # "all" pages until has_more is False — the boundary breaks
            # below must never truncate it.
            if mode != "all":
                if not found_boundary:
                    sentinel_indices = [i for i, m in enumerate(page) if _is_sentinel(m)]
                    if sentinel_indices:
                        found_boundary = True
                        last = sentinel_indices[-1]  # newest sentinel in the page
                        newer_pages = pages[:-1]  # every earlier page is newer than it
                        # Rows after the sentinel inside the boundary page are
                        # newer than the sentinel but OLDER than every previously
                        # fetched (newer) page, so they come first chronologically.
                        session_rows = page[last + 1 :] + _chronological(newer_pages)
                        if page[last].metadata.get("status") == "collapsed":
                            summary_sentinel = page[last]
                            break  # newest sentinel is itself the newest collapsed one
                        if mode == "session_only":
                            break
                        # The newest sentinel is active/paused: the newest collapsed
                        # one may still be in THIS page, below the boundary.
                        collapsed_here = [
                            i
                            for i in sentinel_indices
                            if page[i].metadata.get("status") == "collapsed"
                        ]
                        if collapsed_here:
                            summary_sentinel = page[collapsed_here[-1]]
                            break
                elif mode == "session_plus_summary" and summary_sentinel is None:
                    collapsed = [
                        i
                        for i, m in enumerate(page)
                        if _is_sentinel(m) and m.metadata.get("status") == "collapsed"
                    ]
                    if collapsed:
                        summary_sentinel = page[collapsed[-1]]
                        break

            if mode == "all" or not found_boundary or (
                mode == "session_plus_summary" and summary_sentinel is None
            ):
                if not has_more or next_before is None:
                    break  # history exhausted; whatever we need, there is no more
                if len(pages) >= MAX_HISTORY_PAGES:
                    raise ClientError(
                        "HISTORY_PAGE_BOUND",
                        f"History fetch exceeded {MAX_HISTORY_PAGES} pages of {limit} rows "
                        f"without satisfying mode {mode!r}; refusing to silently truncate",
                    )
                before = next_before
            else:
                break  # this mode found everything it needs

        if mode == "all":
            return _chronological(pages)
        rows = session_rows if found_boundary else _chronological(pages)
        if summary_sentinel is not None:
            return [summary_sentinel] + rows
        return rows

    async def close(self) -> None:
        """Close the socket cleanly; run() then returns without Disconnected."""
        self._closing = True
        conn = self._conn
        if conn is None:
            return
        try:
            await asyncio.wait_for(conn.close(), timeout=CLOSE_TIMEOUT)
        except Exception:
            pass  # socket already gone; run() handles the fallout
        self._conn = None

    # --- internals ----------------------------------------------------------

    def _ws_url(self) -> str:
        base = self._config.base_url
        if base.startswith("http://"):
            ws_base = "ws://" + base[len("http://"):]
        elif base.startswith("https://"):
            ws_base = "wss://" + base[len("https://"):]
        else:
            raise ClientError(
                "BAD_BASE_URL",
                f"base_url {base!r} must start with http:// or https://",
            )
        return ws_base.rstrip("/") + "/v0/ws/chat"

    async def _fail_auth(self, code: str, message: str) -> None:
        """Surface an auth failure on the queue, then raise. Never returns."""
        await self._events.put(AuthFailed(code=code, message=message))
        raise ClientError(code, message)

    async def _authenticate(self, conn: websockets.asyncio.client.ClientConnection) -> str:
        """Send the auth frame and wait for auth_success (bounded).

        The server closes with protocol_error AUTH_TIMEOUT after 10 s; our
        wait is bounded past that so the server's own error wins the race.
        """
        await asyncio.wait_for(
            conn.send(
                json.dumps(
                    dump_outbound_frame(AuthFrame(type="auth", token=self._config.api_key))
                )
            ),
            timeout=SEND_TIMEOUT,
        )
        while True:
            try:
                raw = await asyncio.wait_for(conn.recv(), timeout=AUTH_WAIT_TIMEOUT)
            except asyncio.TimeoutError:
                raise ClientError(
                    "AUTH_TIMEOUT", "Server did not answer authentication within 12 s"
                ) from None
            except websockets.exceptions.ConnectionClosed as error:
                raise ClientError(
                    "AUTH_CONNECTION_FAILED", f"Connection closed during auth: {error}"
                ) from None
            try:
                frame = parse_inbound_frame(json.loads(raw))
            except Exception as error:
                raise ClientError("UNPARSEABLE_FRAME", f"Bad frame during auth: {error}") from None
            if frame.type == "auth_success":
                return frame.user_id
            if frame.type == "protocol_error":
                raise ClientError(frame.code, frame.message)
            # pong / anything else during auth: keep waiting

    async def _send_frame(self, frame: object) -> None:
        conn = self._conn
        if conn is None or self._closing:
            raise ClientError("NOT_CONNECTED", "No open WebSocket connection")
        try:
            await asyncio.wait_for(
                conn.send(json.dumps(dump_outbound_frame(frame))),
                timeout=SEND_TIMEOUT,
            )
        except ClientError:
            raise
        except Exception as error:
            raise ClientError(
                "SEND_FAILED", f"Could not send frame: {error!r}"
            ) from error

    def _frame_to_event(self, raw: str | bytes) -> ClientEvent | None:
        """Validate one inbound wire message and map it to a ClientEvent.

        Returns None only for frames the event union does not cover (pong).
        Validation failures become ProtocolError, never a pump crash.
        """
        try:
            payload = json.loads(raw)
            frame = parse_inbound_frame(payload)
        except Exception as error:
            return ProtocolError(
                code="UNPARSEABLE_FRAME", message=str(error), message_id=None
            )

        kind = frame.type
        if kind == "pong":
            return None
        if kind == "auth_success":
            return AuthOk(user_id=frame.user_id)
        if kind == "protocol_error":
            return ProtocolError(
                code=frame.code,
                message=frame.message,
                message_id=str(frame.message_id) if frame.message_id is not None else None,
            )
        if kind == "server_shutdown":
            self._server_shutdown_seen = True
            return ServerShutdown(code=frame.code, message=frame.message)
        if kind == "turn_started":
            self._active_turn_id = str(frame.turn_id)
            return TurnStarted(
                turn_id=str(frame.turn_id),
                message_id=str(frame.message_id),
                segment_id=str(frame.segment_id),
            )
        if kind == "assistant_delta":
            return AssistantDelta(
                turn_id=str(frame.turn_id),
                segment_id=str(frame.segment_id),
                entry_id=str(frame.entry_id),
                content=frame.content,
            )
        if kind == "thinking":
            return Thinking(
                turn_id=str(frame.turn_id),
                segment_id=str(frame.segment_id),
                content=frame.content,
            )
        if kind == "tool":
            return ToolUpdate(
                turn_id=str(frame.turn_id),
                segment_id=str(frame.segment_id),
                event=frame.event,
                tool_name=frame.tool_name,
                tool_id=frame.tool_id,
                arguments=frame.arguments,
                result=None if frame.result is None else str(frame.result),
                is_error=frame.is_error,
            )
        if kind == "model_error":
            return ModelError(
                turn_id=str(frame.turn_id),
                segment_id=str(frame.segment_id),
                message=frame.message,
            )
        if kind == "context_reset":
            return ContextReset(turn_id=str(frame.turn_id), segment_id=str(frame.segment_id))
        if kind == "turn_complete":
            self._active_turn_id = None
            return TurnComplete(
                turn_id=str(frame.turn_id),
                segment_id=str(frame.segment_id),
                continuum_id=str(frame.continuum_id),
                response=frame.response,
                tools_used=frame.tools_used,
                processing_time_ms=frame.processing_time_ms,
                emotion=frame.emotion,
            )
        if kind == "turn_stopped":
            self._active_turn_id = None
            return TurnStopped(
                turn_id=str(frame.turn_id),
                segment_id=str(frame.segment_id),
                reason=frame.reason,
            )
        if kind == "turn_error":
            self._active_turn_id = None
            return TurnError(
                turn_id=str(frame.turn_id),
                segment_id=str(frame.segment_id),
                code=frame.code,
                message=frame.message,
            )
        if kind == "proactive_message":
            # The protocol mirror defines no segment_id on this frame.
            return Proactive(
                message_id=str(frame.message_id),
                turn_id=str(frame.turn_id) if frame.turn_id is not None else None,
                segment_id=None,
                content=frame.content,
                created_at=frame.created_at,
            )
        return ProtocolError(
            code="UNPARSEABLE_FRAME",
            message=f"Validated frame of unknown type {kind!r}",
            message_id=None,
        )

    async def _fetch_history_page(
        self, before: str | None, limit: int
    ) -> tuple[list[HistoryMessage], bool, str | None]:
        """One keyset page; raises ClientError on any failure — no defaults."""
        params: dict[str, str | int] = {
            "type": "history",
            "limit": limit,
            "message_type": "regular",
        }
        if before is not None:
            params["before"] = before
        url = self._config.base_url.rstrip("/") + "/v0/api/data"
        try:
            async with httpx.AsyncClient(timeout=HTTP_PAGE_TIMEOUT) as http:
                response = await http.get(
                    url,
                    params=params,
                    headers={"Authorization": f"Bearer {self._config.api_key}"},
                )
        except httpx.HTTPError as error:
            raise ClientError(
                "HISTORY_FETCH_FAILED", f"History request to {url} failed: {error}"
            ) from error

        if response.status_code // 100 != 2:
            server_message = self._extract_error_message(response)
            raise ClientError(
                "HISTORY_HTTP_ERROR",
                f"History fetch returned HTTP {response.status_code}: {server_message}",
            )
        try:
            envelope = parse_history_response(response.json())
        except Exception as error:
            raise ClientError(
                "HISTORY_BAD_ENVELOPE",
                f"History response failed schema validation: {error}",
            ) from error
        if not envelope.success or envelope.data is None:
            error = envelope.error
            raise ClientError(
                error.code if error else "HISTORY_FETCH_FAILED",
                error.message if error else "History fetch failed with no error payload",
            )
        return (
            envelope.data.messages,
            envelope.data.meta.has_more,
            envelope.data.meta.next_before,
        )

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        """Best-effort server error text for a non-2xx page; the status code
        is already in the raised message so failure here is not silent."""
        try:
            payload = response.json()
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
        except Exception:
            pass
        text = response.text.strip()
        return text[:500] if text else "(no body)"
