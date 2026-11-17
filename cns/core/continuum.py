"""
The `Continuum` aggregate: frozen identity plus a mutable hot message cache.

Every `add_*` method appends to `_message_cache` and does nothing else. A
message becomes durable only when the same `Message` object is staged into
`UnitOfWork.add_messages` and committed — the aggregate holds no persistence
of its own. An append path that skips staging therefore shows its message
until the cache is next replaced wholesale, then loses it with no error
raised anywhere.

`apply_cache` is the only writer that replaces the list instead of appending
to it, and the pool calls it on both the cache-miss and cache-hit paths, so a
`Message` reference held across that call points at a message no longer in
the conversation.

The event lists the `add_*` methods return are always empty: a turn's events
are published by whoever completed the turn, never by the aggregate.
"""
from __future__ import annotations

import logging
from uuid import UUID, uuid4

from .message import Message, MessageMetadata, ContentBlock
from .state import ContinuumState
from .events import ContinuumEvent

logger = logging.getLogger(__name__)


class Continuum:
    """
    Continuum aggregate root.

    Manages continuum state and message cache. ContinuumState is frozen,
    but the message cache is mutable (appended during turns, replaced on reload).
    """

    def __init__(self, state: ContinuumState):
        """Initialize continuum with state."""
        self._state = state
        self._message_cache: list[Message] = []  # Hot cache of recent messages

    @classmethod
    def create_new(cls, user_id: str) -> Continuum:
        """Create a new continuum for user."""
        state = ContinuumState(
            id=uuid4(),
            user_id=user_id
        )
        return cls(state)
    
    @property
    def id(self) -> UUID:
        """Get continuum ID."""
        return self._state.id

    @property
    def user_id(self) -> str:
        """Get user ID."""
        return self._state.user_id

    @property
    def messages(self) -> list[Message]:
        """Get cached messages - must be initialized through ContinuumPool."""
        return self._message_cache

    def apply_cache(self, messages: list[Message]) -> None:
        """
        Apply an externally managed cache update.
        
        Used by segment cache loader to update the cache after operations
        like segment reconstruction and message loading.
        
        Args:
            messages: New message cache to apply
        """
        self._message_cache = messages
    
    def add_user_message(
        self,
        content: str | list[ContentBlock],
        *,
        message_id: UUID | None = None,
        metadata: MessageMetadata | None = None,
    ) -> tuple[Message, list[ContinuumEvent]]:
        """
        Add user message to continuum.

        The transport boundary supplies ``message_id`` and turn metadata so the
        durable row can be reconciled with the client's optimistic id and with
        the frames streamed for the same turn.

        Returns:
            Tuple of (created Message, list of domain events)
        """
        message = Message(
            content=content,
            role="user",
            id=message_id or uuid4(),
            metadata=metadata or {},
        )

        self._message_cache.append(message)

        return message, []

    def add_assistant_message(self, content: str, metadata: MessageMetadata | None = None) -> tuple[Message, list[ContinuumEvent]]:
        """
        Add assistant message to continuum.

        Returns:
            Tuple of (created Message, list of domain events)
        """
        if not content or not content.strip():
            raise ValueError("Assistant message content cannot be blank or empty")

        message = Message(content=content, role="assistant", metadata=metadata or {})

        self._message_cache.append(message)

        return message, []
    
    def add_tool_history(self, messages: list[Message]) -> None:
        """
        Insert tool history messages into the cache.

        Called after turn completion to persist tool use/result pairs
        so subsequent turns can see what tools returned.
        """
        self._message_cache.extend(messages)
