"""Authenticated WebSocket connection and strict turn protocol for MIRA chat.

Every frame crossing this boundary in either direction is validated against a
Pydantic model that forbids unknown fields, so a malformed or unspeakable frame
fails loudly at the edge instead of being half-applied. One socket reader task
and one socket writer task own the connection for its lifetime; the message
loop never awaits ``websocket.receive_json()`` anywhere else — a second reader
would silently swallow frames.

``AuthFrame.token`` lets non-browser clients authenticate the socket with an
issued API token; ``ThinkingFrame``/``ModelErrorFrame`` forward ``thinking`` and
``model_error`` to the browser, and ``TurnCompleteFrame`` carries the fields the
UI reads to close a turn. Account-billing handshakes are not part of this
protocol.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import logging
import threading
from collections.abc import Callable
from functools import partial
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError as PydanticValidationError,
    model_validator,
)

from auth.service import get_auth_service
from auth.session import SessionManager
from auth.types import APITokenContext
from cns.core.continuum import Continuum
from cns.core.message import ContentBlock
from cns.infrastructure.continuum_pool import get_continuum_pool
from cns.infrastructure.continuum_repository import get_continuum_repository
from cns.services.async_work_barrier import get_async_work_barrier
from cns.services.orchestrator import get_orchestrator
from config.config_manager import config as app_config
from utils.distributed_lock import UserRequestLock
from utils.document_processing import (
    MAX_DOCUMENT_SIZE_MB,
    SUPPORTED_DOCUMENT_FORMATS,
    ProcessedDocument,
    process_document,
)
from utils.image_compression import CompressedImage, compress_image
from utils.text_sanitizer import sanitize_message_content
from utils.timezone_utils import utc_now
from utils.user_context import (
    clear_user_context,
    set_cancel_event,
    set_cancel_reason,
    set_current_user_data,
    set_current_user_id,
)

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_IMAGE_FORMATS = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
MAX_IMAGE_SIZE_MB = 5
MAX_CONTENT_LENGTH = 100_000


# --- Inbound frames -------------------------------------------------------


class ProtocolModel(BaseModel):
    """Strict base for every WebSocket frame."""

    model_config = ConfigDict(extra="forbid")


class AuthFrame(ProtocolModel):
    """Handshake frame.

    ``token`` lets server-to-server clients present an issued API token on the
    first frame instead of relying on a browser session cookie. Browsers send no
    token and fall through to the cookie rung; there is no static shared key.
    When present it wins over the cookie; when absent the cookie ladder runs.
    """

    type: Literal["auth"]
    token: str | None = None


class MessageFrame(ProtocolModel):
    """One user message accepted for turn processing."""

    type: Literal["message"]
    message_id: UUID
    content: str = Field(min_length=1, max_length=MAX_CONTENT_LENGTH)
    include_thinking: bool = False
    image: str | None = None
    image_type: str | None = None
    document: str | None = None
    document_type: str | None = None

    @model_validator(mode="after")
    def validate_attachment_pairs(self) -> "MessageFrame":
        """Require each attachment body and MIME type together."""
        if (self.image is None) != (self.image_type is None):
            raise ValueError("image and image_type must be provided together")
        if (self.document is None) != (self.document_type is None):
            raise ValueError("document and document_type must be provided together")
        if self.image is not None and self.document is not None:
            raise ValueError("one message may contain an image or a document, not both")
        return self


class HaltFrame(ProtocolModel):
    """Stop the active server turn after any running tool returns."""

    type: Literal["halt"]
    turn_id: UUID


class PingFrame(ProtocolModel):
    """Client keepalive."""

    type: Literal["ping"]


ClientFrame = Annotated[AuthFrame | MessageFrame | HaltFrame | PingFrame, Field(discriminator="type")]
CLIENT_FRAME_ADAPTER = TypeAdapter(ClientFrame)


# --- Outbound frames ------------------------------------------------------


class AuthSuccessFrame(ProtocolModel):
    type: Literal["auth_success"]
    user_id: str


class ServerShutdownFrame(ProtocolModel):
    type: Literal["server_shutdown"]
    message: str


class PongFrame(ProtocolModel):
    type: Literal["pong"]


class ProtocolErrorFrame(ProtocolModel):
    type: Literal["protocol_error"]
    code: str
    message: str


class TurnStartedFrame(ProtocolModel):
    type: Literal["turn_started"]
    turn_id: UUID
    message_id: UUID
    segment_id: UUID


class AssistantDeltaFrame(ProtocolModel):
    type: Literal["assistant_delta"]
    turn_id: UUID
    segment_id: UUID
    entry_id: UUID
    content: str


class ThinkingFrame(ProtocolModel):
    """Reasoning stream, forwarded only when the request asked for it."""

    type: Literal["thinking"]
    turn_id: UUID
    segment_id: UUID
    content: str


class ToolFrame(ProtocolModel):
    type: Literal["tool"]
    turn_id: UUID
    segment_id: UUID
    event: Literal["tool_detected", "tool_executing", "tool_completed", "tool_error"]
    tool_name: str
    tool_id: str
    arguments: dict[str, object] | None = None
    result: object | None = None
    is_error: bool = False

    @model_validator(mode="after")
    def validate_event_payload(self) -> "ToolFrame":
        """Require the payload owned by the selected lifecycle event."""
        if self.event == "tool_executing" and self.arguments is None:
            raise ValueError("tool_executing requires arguments")
        if self.event == "tool_completed" and self.result is None:
            raise ValueError("tool_completed requires a result")
        if self.event == "tool_error" and not self.is_error:
            raise ValueError("tool_error requires is_error=true")
        if self.event != "tool_error" and self.is_error:
            raise ValueError(f"{self.event} requires is_error=false")
        return self


class ModelErrorFrame(ProtocolModel):
    """Notice that the model misused a tool and the turn is recovering."""

    type: Literal["model_error"]
    turn_id: UUID
    segment_id: UUID
    message: str


class TurnCompleteFrame(ProtocolModel):
    """Terminal frame for a successful turn.

    crm's version carries only ``turn_id`` and ``segment_id``, which the
    retained UI cannot close a turn with: it needs the continuum identity, the
    final text, and the tool/timing/emotion metadata. ``emotion`` stays
    because ``<mira:my_emotion>`` remains in the system prompt and
    ``orchestrator.process_message()`` still parses it.
    """

    type: Literal["turn_complete"]
    turn_id: UUID
    segment_id: UUID
    continuum_id: UUID
    response: str
    tools_used: list[str] = Field(default_factory=list)
    processing_time_ms: int = 0
    emotion: str | None = None


class TurnStoppedFrame(ProtocolModel):
    type: Literal["turn_stopped"]
    turn_id: UUID
    segment_id: UUID
    reason: Literal["halt", "disconnect"]


class TurnErrorFrame(ProtocolModel):
    type: Literal["turn_error"]
    turn_id: UUID
    segment_id: UUID
    code: str
    message: str


ServerFrame = Annotated[
    AuthSuccessFrame
    | ServerShutdownFrame
    | PongFrame
    | ProtocolErrorFrame
    | TurnStartedFrame
    | AssistantDeltaFrame
    | ThinkingFrame
    | ToolFrame
    | ModelErrorFrame
    | TurnCompleteFrame
    | TurnStoppedFrame
    | TurnErrorFrame,
    Field(discriminator="type"),
]
SERVER_FRAME_ADAPTER = TypeAdapter(ServerFrame)


class ClientDisconnected:
    """Inbound sentinel emitted by the sole socket reader."""


class StopWriter:
    """Outbound sentinel consumed by the sole socket writer."""


class WebSocketAuthError(Exception):
    """Raised when WebSocket authentication fails."""


def validate_client_frame(value: object) -> ClientFrame:
    """Validate one untrusted client frame."""
    return CLIENT_FRAME_ADAPTER.validate_python(value)


def validate_server_frame(value: object) -> ServerFrame:
    """Validate one server frame before it reaches the writer queue."""
    return SERVER_FRAME_ADAPTER.validate_python(value)


def get_friendly_error_message(error: Exception) -> str:
    """Convert infrastructure/provider failures into stable user-facing copy."""
    error_text = str(error).lower()
    if "usage limit" in error_text or "rate limit" in error_text:
        return (
            "I'm currently rate limited. Please try again in a few moments. "
            "If this persists, the API usage limits may have been reached."
        )
    if "authentication failed" in error_text or "401" in error_text:
        return "The API provider outside Mira had a problem. Please try again in a couple minutes."
    if "no allowed providers" in error_text or ("model" in error_text and "404" in error_text):
        return "The AI model I'm trying to use isn't available. Please contact support."
    if "connection" in error_text or "network" in error_text:
        return "I'm having trouble connecting to the AI service. Please try again."
    if "timeout" in error_text:
        return "The request took too long to process. Please try again with a simpler message."
    if any(code in error_text for code in ["500", "502", "503"]):
        return "The AI service is experiencing technical difficulties. Please try again in a few moments."
    return "I encountered an unexpected error while processing your message. Please try again."


class ChatConnection:
    """Own the only socket reader and writer tasks for one connection."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.inbound: asyncio.Queue[ClientFrame | ClientDisconnected] = asyncio.Queue(maxsize=32)
        self.outbound: asyncio.Queue[ServerFrame | StopWriter] = asyncio.Queue(maxsize=128)
        self.reader_task: asyncio.Task[None] | None = None
        self.writer_task: asyncio.Task[None] | None = None
        self.active_turn_task: asyncio.Task[None] | None = None
        self.active_turn_id: UUID | None = None
        self.cancel_event: threading.Event | None = None
        self.accepts_output = True

    def start(self) -> None:
        self.reader_task = asyncio.create_task(self._read_frames())
        self.writer_task = asyncio.create_task(self._write_frames())

    async def send(self, frame: object) -> None:
        if not self.accepts_output:
            return
        await self.outbound.put(validate_server_frame(frame))

    async def drain(self) -> None:
        await self.outbound.join()

    async def close(self) -> None:
        if self.reader_task is not None:
            self.reader_task.cancel()
            await asyncio.gather(self.reader_task, return_exceptions=True)
        if self.writer_task is not None and not self.writer_task.done():
            await self.outbound.put(StopWriter())
            try:
                await asyncio.wait_for(self.writer_task, timeout=2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self.writer_task.cancel()
        try:
            await self.websocket.close()
        except RuntimeError:
            pass

    async def _read_frames(self) -> None:
        try:
            while True:
                raw = await self.websocket.receive_json()
                try:
                    frame = validate_client_frame(raw)
                except PydanticValidationError as error:
                    await self.send({
                        "type": "protocol_error",
                        "code": "MALFORMED_FRAME",
                        "message": _validation_message(error),
                    })
                    continue
                await self.inbound.put(frame)
        except (WebSocketDisconnect, RuntimeError):
            await self.inbound.put(ClientDisconnected())
        except asyncio.CancelledError:
            return

    async def _write_frames(self) -> None:
        try:
            while True:
                frame = await self.outbound.get()
                try:
                    if isinstance(frame, StopWriter):
                        return
                    await self.websocket.send_json(frame.model_dump(mode="json", exclude_none=True))
                finally:
                    self.outbound.task_done()
        except (WebSocketDisconnect, RuntimeError):
            return
        except asyncio.CancelledError:
            return
        finally:
            self.accepts_output = False


_active_connections: dict[str, ChatConnection] = {}


async def close_all_connections() -> None:
    """Tell every active client about shutdown through its sole writer."""
    connections = list(_active_connections.values())
    for connection in connections:
        await connection.send({
            "type": "server_shutdown",
            "message": "Server is shutting down",
        })
    for connection in connections:
        try:
            await asyncio.wait_for(connection.drain(), timeout=2)
        except asyncio.TimeoutError:
            pass
        await connection.close()
    _active_connections.clear()


class WebSocketChatHandler:
    """Authenticate and dispatch one connection's validated frames."""

    def __init__(self) -> None:
        self.orchestrator = get_orchestrator()
        self.continuum_pool = get_continuum_pool()
        self.continuum_repo = get_continuum_repository()
        self.auth_service = get_auth_service()
        self.session_manager = SessionManager()
        self.user_request_lock = UserRequestLock(ttl=60)

    async def authenticate(self, websocket: WebSocket, token: str | None) -> str:
        """Resolve an identity for this socket and install the user context.

        Credential ladder: the auth-frame token wins, a browser session cookie
        is the fallback, and an issued API token is the second rung. There is
        no static shared key: every mode authenticates against the session/API-
        token stack.

        Both the id and the full typed context are installed. The context is
        what lets WS-originated work call ``get_current_user()`` instead of
        raising.
        """
        credential = token or websocket.cookies.get("session")
        if not credential:
            raise WebSocketAuthError("Missing authentication token")

        session_data = await run_in_threadpool(self.session_manager.validate_session, credential)
        if not session_data:
            api_token_data = await run_in_threadpool(
                self.auth_service.validate_api_token, credential
            )
            if api_token_data:
                session_data = APITokenContext(
                    user_id=api_token_data["user_id"],
                    token_type="api_token",
                    token_id=api_token_data["id"],
                    subject_kind=api_token_data["subject_kind"],
                    demo_expires_at=(
                        api_token_data["demo_expires_at"].isoformat()
                        if api_token_data["demo_expires_at"] is not None
                        else None
                    ),
                )

        if not session_data:
            raise WebSocketAuthError("Invalid or expired session")

        set_current_user_id(session_data.user_id)
        set_current_user_data(session_data.model_dump())
        return session_data.user_id

    async def handle_connection(self, connection: ChatConnection) -> None:
        """Authenticate first, then dispatch message, halt, ping, and disconnect."""
        try:
            first = await asyncio.wait_for(connection.inbound.get(), timeout=10)
        except asyncio.TimeoutError:
            await connection.send({
                "type": "protocol_error",
                "code": "AUTH_TIMEOUT",
                "message": "Authentication timeout",
            })
            return

        if isinstance(first, ClientDisconnected):
            return
        if not isinstance(first, AuthFrame):
            await connection.send({
                "type": "protocol_error",
                "code": "AUTH_REQUIRED",
                "message": "First message must be authentication",
            })
            return

        try:
            user_id = await self.authenticate(connection.websocket, first.token)
        except WebSocketAuthError as error:
            await connection.send({
                "type": "protocol_error",
                "code": "AUTH_FAILED",
                "message": str(error),
            })
            return

        if not await run_in_threadpool(self.user_request_lock.acquire, user_id):
            await connection.send({
                "type": "protocol_error",
                "code": "USER_CONNECTION_BUSY",
                "message": (
                    "MIRA is designed in a way where each user has a 'lock' on a "
                    "connection to the server. For some reason yours didn't expire "
                    "last time you disconnected. It will clear in 60 seconds. Please "
                    "refresh the page in one minute."
                ),
            })
            return

        await connection.send({"type": "auth_success", "user_id": user_id})
        try:
            await self._dispatch(connection, user_id)
        finally:
            if connection.active_turn_task is not None:
                if connection.cancel_event is not None:
                    set_cancel_reason("disconnect")
                    connection.cancel_event.set()
                await asyncio.gather(connection.active_turn_task, return_exceptions=True)
            await run_in_threadpool(self.user_request_lock.release, user_id)
            clear_user_context()

    async def _dispatch(self, connection: ChatConnection, user_id: str) -> None:
        while True:
            frame = await connection.inbound.get()
            if isinstance(frame, ClientDisconnected):
                if connection.cancel_event is not None:
                    set_cancel_reason("disconnect")
                    connection.cancel_event.set()
                return
            if isinstance(frame, AuthFrame):
                await connection.send({
                    "type": "protocol_error",
                    "code": "ALREADY_AUTHENTICATED",
                    "message": "Authentication is already complete",
                })
                continue
            if isinstance(frame, PingFrame):
                await connection.send({"type": "pong"})
                continue
            if isinstance(frame, HaltFrame):
                if connection.active_turn_id != frame.turn_id or connection.cancel_event is None:
                    await connection.send({
                        "type": "protocol_error",
                        "code": "NO_MATCHING_ACTIVE_TURN",
                        "message": "Halt turn_id does not match the active turn",
                    })
                    continue
                set_cancel_reason("halt")
                connection.cancel_event.set()
                continue
            if isinstance(frame, MessageFrame):
                if connection.active_turn_task is not None and not connection.active_turn_task.done():
                    await connection.send({
                        "type": "protocol_error",
                        "code": "TURN_BUSY",
                        "message": "A server turn is already active",
                    })
                    continue
                turn_id = uuid4()
                cancel_event = threading.Event()
                set_cancel_event(cancel_event)
                set_cancel_reason("halt")
                connection.active_turn_id = turn_id
                connection.cancel_event = cancel_event
                task = asyncio.create_task(
                    self.process_turn(connection, user_id, frame, turn_id, cancel_event)
                )
                connection.active_turn_task = task
                task.add_done_callback(lambda completed: self._clear_turn(connection, completed))

    @staticmethod
    def _clear_turn(connection: ChatConnection, completed: asyncio.Task[None]) -> None:
        if connection.active_turn_task is completed:
            connection.active_turn_task = None
            connection.active_turn_id = None
            connection.cancel_event = None

    async def process_turn(
        self,
        connection: ChatConnection,
        user_id: str,
        message: MessageFrame,
        turn_id: UUID,
        cancel_event: threading.Event,
    ) -> None:
        """Validate attachments, run the orchestrator, commit, then emit one terminal frame."""
        segment_id: UUID | None = None
        start_time = utc_now()
        try:
            content = sanitize_message_content(message.content.strip())
            if not content:
                raise ValueError("Message cannot be empty")
            compressed = self._prepare_image(message)
            processed_document = self._prepare_document(message)
            if not app_config.system_prompt:
                raise RuntimeError("System prompt not configured")

            set_cancel_event(cancel_event)
            context = contextvars.copy_context()
            # Give previous-turn tool-result compaction a bounded head start
            # before loading the hot continuum. The timeout is intentionally
            # fail-open; the timeout value itself is configuration.
            wait_for_background_work = partial(
                get_async_work_barrier().wait_for_user,
                user_id,
                timeout=app_config.api.async_work_barrier_timeout_seconds,
                source="tool_summarizer",
            )
            await run_in_threadpool(context.run, wait_for_background_work)
            continuum = await run_in_threadpool(context.run, self._get_user_continuum)
            segment = await run_in_threadpool(
                context.run,
                self.continuum_pool.repository.increment_segment_turn,
                continuum.id,
                user_id,
            )
            segment_id = UUID(segment.segment_id)
            await connection.send({
                "type": "turn_started",
                "turn_id": turn_id,
                "message_id": message.message_id,
                "segment_id": segment_id,
            })

            loop = asyncio.get_running_loop()

            def stream_to_connection(event: dict[str, object]) -> None:
                frame = self._server_frame_for_event(event, turn_id, segment_id)
                if frame is None:
                    return
                # R5's other half: include_thinking must select the reasoning
                # stream. A field the transport accepts and never reads is a lie
                # about what the client asked for.
                if frame["type"] == "thinking" and not message.include_thinking:
                    return
                future = asyncio.run_coroutine_threadsafe(connection.send(frame), loop)
                future.result()

            result = await run_in_threadpool(
                context.run,
                self._process_with_orchestrator,
                continuum,
                content,
                compressed,
                processed_document,
                stream_to_connection,
                segment.turn_number,
                message.message_id,
                turn_id,
            )
            processing_time_ms = int((utc_now() - start_time).total_seconds() * 1000)
            metadata = result["metadata"]
            if metadata.get("stopped"):
                await connection.send({
                    "type": "turn_stopped",
                    "turn_id": turn_id,
                    "segment_id": segment_id,
                    "reason": metadata["stop_reason"],
                })
            else:
                await connection.send({
                    "type": "turn_complete",
                    "turn_id": turn_id,
                    "segment_id": segment_id,
                    "continuum_id": result["continuum"].id,
                    "response": result["response"],
                    "tools_used": list(metadata.get("tools_used") or []),
                    "processing_time_ms": processing_time_ms,
                    "emotion": metadata.get("emotion"),
                })
        except (ValueError, PydanticValidationError) as error:
            if segment_id is None:
                await connection.send({
                    "type": "protocol_error",
                    "code": "INVALID_MESSAGE",
                    "message": str(error),
                })
            else:
                await connection.send({
                    "type": "turn_error",
                    "turn_id": turn_id,
                    "segment_id": segment_id,
                    "code": "TURN_VALIDATION_FAILED",
                    "message": get_friendly_error_message(error),
                })
        except Exception as error:
            logger.error("WebSocket turn failed for user %s: %s", user_id, error, exc_info=True)
            if segment_id is not None:
                await connection.send({
                    "type": "turn_error",
                    "turn_id": turn_id,
                    "segment_id": segment_id,
                    "code": "TURN_PROCESSING_FAILED",
                    "message": get_friendly_error_message(error),
                })
            else:
                await connection.send({
                    "type": "protocol_error",
                    "code": "TURN_SETUP_FAILED",
                    "message": get_friendly_error_message(error),
                })

    def _prepare_image(self, message: MessageFrame) -> CompressedImage | None:
        if message.image is None:
            return None
        assert message.image_type is not None
        if message.image_type not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format: {message.image_type}")
        decoded = base64.b64decode(message.image, validate=True)
        if len(decoded) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Image exceeds maximum size of {MAX_IMAGE_SIZE_MB}MB")
        return compress_image(decoded, message.image_type)

    def _prepare_document(self, message: MessageFrame) -> ProcessedDocument | None:
        if message.document is None:
            return None
        assert message.document_type is not None
        if message.document_type not in SUPPORTED_DOCUMENT_FORMATS:
            raise ValueError(
                "Unsupported document format. Supported: "
                + ", ".join(sorted(SUPPORTED_DOCUMENT_FORMATS))
            )
        decoded = base64.b64decode(message.document, validate=True)
        if len(decoded) > MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Document exceeds maximum size of {MAX_DOCUMENT_SIZE_MB}MB")
        return process_document(
            decoded,
            message.document_type,
            filename=f"document.{message.document_type.split('/')[-1]}",
        )

    @staticmethod
    def _server_frame_for_event(
        event: dict[str, object],
        turn_id: UUID,
        segment_id: UUID,
    ) -> dict[str, object] | None:
        """Translate one orchestrator stream event into a server frame.

        Forwards reasoning-stream and invalid-tool-call events in addition to
        the text and tool-event deltas — dropping either would lose the
        reasoning stream or the malformed-call signal.
        """
        event_type = event.get("type")
        if event_type == "text":
            return {
                "type": "assistant_delta",
                "turn_id": turn_id,
                "segment_id": segment_id,
                "entry_id": event["entry_id"],
                "content": event.get("content", ""),
            }
        if event_type == "thinking":
            return {
                "type": "thinking",
                "turn_id": turn_id,
                "segment_id": segment_id,
                "content": event.get("content", ""),
            }
        if event_type == "tool_event":
            frame: dict[str, object] = {
                "type": "tool",
                "turn_id": turn_id,
                "segment_id": segment_id,
                "event": event["event"],
                "tool_name": event["tool_name"],
                "tool_id": event["tool_id"],
            }
            for field_name in ("arguments", "result", "is_error"):
                if field_name in event:
                    frame[field_name] = event[field_name]
            return frame
        if event_type == "model_error":
            return {
                "type": "model_error",
                "turn_id": turn_id,
                "segment_id": segment_id,
                "message": (
                    "The AI model made an invalid tool call. Attempting to recover..."
                ),
            }
        return None

    def _process_with_orchestrator(
        self,
        continuum: Continuum,
        content: str,
        compressed: CompressedImage | None,
        processed_document: ProcessedDocument | None,
        stream_callback: Callable[[dict[str, object]], None],
        segment_turn_number: int,
        message_id: UUID,
        turn_id: UUID,
    ) -> dict[str, Any]:
        inference_content: str | list[ContentBlock]
        storage_content: str | list[ContentBlock] | None = None
        if compressed is not None:
            # Inference tier (1200px) for this call, storage tier (512px WebP)
            # for the durable row and every later turn.
            inference_content = [
                {"type": "text", "text": content},
                {"type": "image", "media_type": compressed.inference_media_type, "data": compressed.inference_base64},
            ]
            storage_content = [
                {"type": "text", "text": content},
                {"type": "image", "media_type": compressed.storage_media_type, "data": compressed.storage_base64},
            ]
        elif processed_document is not None:
            document_block: ContentBlock = {
                "type": "text",
                "text": f"[Document: {processed_document.media_type}]\n{processed_document.data}",
            }
            inference_content = [{"type": "text", "text": content}, document_block]
            storage_content = [{"type": "text", "text": content}, document_block]
        else:
            inference_content = content

        unit_of_work = self.continuum_pool.begin_work(continuum)
        try:
            continuum, response_text, metadata = self.orchestrator.process_message(
                continuum,
                inference_content,
                app_config.system_prompt,
                stream=True,
                stream_callback=stream_callback,
                unit_of_work=unit_of_work,
                storage_content=storage_content,
                segment_turn_number=segment_turn_number,
                message_id=message_id,
                turn_id=turn_id,
            )
        except Exception:
            # process_message stages the accepted user message before model
            # work begins, so a failure after acceptance must still commit what
            # the user was told was received. Commit the staged rows, then let
            # the exception propagate to the turn_error path.
            if unit_of_work.pending_messages:
                unit_of_work.commit()
            raise
        unit_of_work.commit()
        return {"continuum": continuum, "response": response_text, "metadata": metadata}

    def _get_user_continuum(self) -> Continuum:
        return self.continuum_pool.get_or_create()


def _validation_message(error: PydanticValidationError) -> str:
    first = error.errors(include_url=False)[0]
    location = ".".join(str(part) for part in first["loc"] if part not in {"auth", "message", "halt", "ping"})
    return f"Invalid {location or 'frame'}: {first['msg']}"


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket) -> None:
    """Accept one connection and run its reader, writer, and dispatcher.

    Protocol (all frames reject unknown fields):

    * client ``{type:"auth", token?}`` -> ``{type:"auth_success", user_id}``
    * client ``{type:"message", message_id, content, include_thinking?, image?,
      image_type?, document?, document_type?}`` -> ``turn_started``, then
      ``assistant_delta`` / ``thinking`` / ``tool`` / ``model_error``, then
      exactly one of ``turn_complete`` / ``turn_stopped`` / ``turn_error``
    * client ``{type:"halt", turn_id}`` -> the active turn ends in ``turn_stopped``
    * client ``{type:"ping"}`` -> ``{type:"pong"}``
    """
    await websocket.accept()
    connection_id = str(uuid4())
    connection = ChatConnection(websocket)
    _active_connections[connection_id] = connection
    connection.start()
    try:
        await WebSocketChatHandler().handle_connection(connection)
    finally:
        _active_connections.pop(connection_id, None)
        await connection.close()
