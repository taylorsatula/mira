"""The provider-neutral stream-event union a live call yields, plus the cancellation exception.

Every subclass of `StreamEvent` fixes its own `type` discriminator string with
`init=False`, so a consumer can switch on `event.type` or on the class and the
two agree by construction. Two facts a consumer must hold: `GenerationCancelled`
subclasses `Exception`, not `StreamEvent` — a user halt arrives as a raise from
the iterator and can never appear as an event — and this module defines the
union only, while several members are produced outside the directory, so an
event type with no producer here is not dead code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clients.llm.types import Result


class GenerationCancelled(Exception):
    """Raised when the user cancels generation midstream."""


@dataclass
class StreamEvent:
    """Base event for all streaming events."""

    type: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TextEvent(StreamEvent):
    """Text content chunk from an LLM response."""

    content: str
    type: str = field(default="text", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ThinkingEvent(StreamEvent):
    """Thinking content chunk from an LLM response."""

    content: str
    type: str = field(default="thinking", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ToolDetectedEvent(StreamEvent):
    """Tool call detected in an LLM response."""

    tool_name: str
    tool_id: str
    type: str = field(default="tool_detected", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ToolExecutingEvent(StreamEvent):
    """Tool execution started."""

    tool_name: str
    tool_id: str
    arguments: dict[str, object]
    type: str = field(default="tool_executing", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ToolCompletedEvent(StreamEvent):
    """Tool execution completed successfully."""

    tool_name: str
    tool_id: str
    result: str | list[dict[str, object]]
    type: str = field(default="tool_completed", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ToolErrorEvent(StreamEvent):
    """Tool execution failed with an error result for model replay."""

    tool_name: str
    tool_id: str
    error: str
    result: str | list[dict[str, object]]
    type: str = field(default="tool_error", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class CompleteEvent(StreamEvent):
    """Stream completed with final response for the iterator yielding it."""

    response: "Result"
    type: str = field(default="complete", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ModelStepCompletedEvent(StreamEvent):
    """One model/provider step completed inside an orchestrated assistant turn.

    This is non-terminal for the user turn. TextEvent chunks emitted before this
    event are already part of the user-visible stream; the response carried here
    exists so the orchestrator can persist tool-call/reasoning replay metadata.
    """

    response: "Result"
    type: str = field(default="model_step_completed", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ErrorEvent(StreamEvent):
    """Stream error occurred."""

    error: str
    type: str = field(default="error", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ProviderNoticeEvent(StreamEvent):
    """Retry or fallback status from the provider boundary; never content.

    Emitted by LLMProvider's retry loop between attempts: the provider is
    unreachable and a retry is scheduled, or the call moved to the backup
    target. Safe to repeat, and never counted as stream content by
    stream_events' first-event gate.
    """

    message: str
    attempt: int
    retry_in_seconds: int
    using_fallback: bool = False
    type: str = field(default="provider_notice", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class CircuitBreakerEvent(StreamEvent):
    """Circuit breaker triggered during local tool execution."""

    reason: str
    type: str = field(default="circuit_breaker", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class FileArtifactEvent(StreamEvent):
    """File artifact produced by provider-side code execution."""

    file_id: str
    filename: str
    mime_type: str
    size_bytes: int
    type: str = field(default="file_artifact", init=False)
    timestamp: float = field(default_factory=time.time, init=False)
