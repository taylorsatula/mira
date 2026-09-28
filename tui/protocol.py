"""Client-side mirror of MIRA's WebSocket turn-protocol frames and REST history models.

This module is a faithful, strict (``extra="forbid"``) mirror of the server's
frame schemas in ``cns/api/websocket_chat.py`` and the history response in
``cns/api/data.py`` (envelope from ``cns/api/base.py``). It is a pure client —
it imports nothing from the server tree — but the two sides MUST change
together: any frame-shape change on the server requires the identical change
here in the same commit, and drift ownership is recorded in ``tui/AGENTS.md``.
Strictness is the point: an unknown or mistyped field fails loudly at this
boundary instead of being half-applied by the UI.

Verified against the 2026-09-18 build plan and live history samples:

* History pages arrive oldest->newest; a request without ``before`` returns
  the newest window; ``meta.next_before`` pages further back; ``has_more``
  means older messages exist.
* Segment sentinels arrive inline: ``metadata.is_segment_boundary == "true"``
  (string), ``status`` active|paused|collapsed. A collapsed sentinel's
  ``content`` IS the session summary; the active sentinel's content is
  ``"[Segment in progress]"``.
* ``role == "tool"`` rows carry JSON tool results; assistant rows carry a
  stringified JSON array of content blocks. ``metadata`` stays
  ``dict[str, Any]`` here (server-defined free-form); rendering is the
  components' concern.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)

MAX_CONTENT_LENGTH = 100_000


class ProtocolModel(BaseModel):
    """Strict base for every WebSocket frame — mirrors the server's ProtocolModel."""

    model_config = ConfigDict(extra="forbid")


# --- Outbound frames (client -> server) ----------------------------------


class AuthFrame(ProtocolModel):
    """First frame within 10 s. The TUI always sends a token; the server
    also accepts a browser-session cookie, hence the optional type here."""

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
        """Require each attachment body and MIME type together, exclusively."""
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
    message_id: UUID | None = None


class PingFrame(ProtocolModel):
    """Client keepalive."""

    type: Literal["ping"]


OutboundFrame = Annotated[
    AuthFrame | MessageFrame | HaltFrame | PingFrame,
    Field(discriminator="type"),
]
OUTBOUND_FRAME_ADAPTER = TypeAdapter(OutboundFrame)


# --- Inbound frames (server -> client) -----------------------------------


class AuthSuccessFrame(ProtocolModel):
    type: Literal["auth_success"]
    user_id: str


class ServerShutdownFrame(ProtocolModel):
    type: Literal["server_shutdown"]
    code: str = "SERVER_SHUTDOWN"
    message: str


class PongFrame(ProtocolModel):
    type: Literal["pong"]


class ProtocolErrorFrame(ProtocolModel):
    """TURN_BUSY, NO_MATCHING_ACTIVE_TURN, MALFORMED_FRAME, AUTH_*, ..."""

    type: Literal["protocol_error"]
    code: str
    message: str
    # Lets the client attribute pre-turn_started errors to the send that
    # caused them.
    message_id: UUID | None = None


class TurnStartedFrame(ProtocolModel):
    type: Literal["turn_started"]
    turn_id: UUID
    message_id: UUID
    segment_id: UUID


class AssistantDeltaFrame(ProtocolModel):
    """Streamed text chunk — accumulate per entry_id."""

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
    arguments: dict[str, Any] | None = None
    result: Any | None = None
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


class ContextResetFrame(ProtocolModel):
    """Discard streamed-so-far text; the turn is regenerating from a reset."""

    type: Literal["context_reset"]
    turn_id: UUID
    segment_id: UUID


class ModelErrorFrame(ProtocolModel):
    """The model misused a tool; the turn is recovering."""

    type: Literal["model_error"]
    turn_id: UUID
    segment_id: UUID
    message: str


class TurnCompleteFrame(ProtocolModel):
    """Terminal frame for a successful turn."""

    type: Literal["turn_complete"]
    turn_id: UUID
    segment_id: UUID
    continuum_id: UUID
    response: str
    tools_used: list[str] = Field(default_factory=list)
    processing_time_ms: int = 0
    emotion: str | None = None


class TurnStoppedFrame(ProtocolModel):
    """Terminal frame for a halted/disconnected turn."""

    type: Literal["turn_stopped"]
    turn_id: UUID
    segment_id: UUID
    reason: Literal["halt", "disconnect"]


class TurnErrorFrame(ProtocolModel):
    """Terminal frame for a failed turn (TURN_VALIDATION_FAILED,
    TURN_PROCESSING_FAILED)."""

    type: Literal["turn_error"]
    turn_id: UUID
    segment_id: UUID
    code: str
    message: str


class ProactiveMessageFrame(ProtocolModel):
    """Server-initiated assistant message (heartbeat breakout) with no client
    turn behind it. Render as an assistant message."""

    type: Literal["proactive_message"]
    message_id: UUID
    turn_id: UUID | None = None
    content: str
    created_at: str


InboundFrame = Annotated[
    AuthSuccessFrame
    | ServerShutdownFrame
    | PongFrame
    | ProtocolErrorFrame
    | TurnStartedFrame
    | AssistantDeltaFrame
    | ThinkingFrame
    | ToolFrame
    | ContextResetFrame
    | ModelErrorFrame
    | TurnCompleteFrame
    | TurnStoppedFrame
    | TurnErrorFrame
    | ProactiveMessageFrame,
    Field(discriminator="type"),
]
INBOUND_FRAME_ADAPTER = TypeAdapter(InboundFrame)


def parse_inbound_frame(raw: dict[str, Any]) -> InboundFrame:
    """Validate one decoded server frame; raises pydantic.ValidationError on
    drift or malformed input — never silently drops or defaults."""
    return INBOUND_FRAME_ADAPTER.validate_python(raw)


def dump_outbound_frame(frame: OutboundFrame) -> dict[str, Any]:
    """Serialize one outbound frame to a JSON-ready dict (exclude_none parity
    with the server's writer)."""
    return TypeAdapter(OutboundFrame).dump_python(frame, mode="json", exclude_none=True)


# --- REST history models (GET /v0/api/data?type=history) -----------------


class HistoryMessage(BaseModel):
    """One row from the history endpoint.

    ``metadata`` is the sanctioned ``dict[str, Any]`` exception: the server
    defines its shape (segment sentinels, display titles) and the client
    reads known keys defensively.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    role: str
    content: Any
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str | None = None
    is_error: bool | None = None


class HistoryPageMeta(BaseModel):
    """Keyset pagination cursor. Pages arrive oldest->newest; ``next_before``
    pages further back in time; ``has_more`` true means older messages exist."""

    model_config = ConfigDict(extra="forbid")

    has_more: bool
    next_before: str | None = None


class HistoryData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[HistoryMessage]
    meta: HistoryPageMeta


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class HistoryResponse(BaseModel):
    """Response envelope from ``cns/api/base.py``: success carries ``data``;
    failure carries ``error``. Exactly one is present."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    data: HistoryData | None = None
    error: ErrorDetail | None = None

    @model_validator(mode="after")
    def validate_envelope(self) -> "HistoryResponse":
        if self.success != (self.data is not None):
            raise ValueError(
                "success=true requires data; success=false requires error instead of data"
            )
        if not self.success and self.error is None:
            raise ValueError("success=false requires an error payload")
        return self


HISTORY_RESPONSE_ADAPTER = TypeAdapter(HistoryResponse)


def parse_history_response(payload: dict[str, Any]) -> HistoryResponse:
    """Validate one decoded REST history envelope; raises on drift."""
    return HISTORY_RESPONSE_ADAPTER.validate_python(payload)
