"""
The hot continuum message cache and the collapse-epoch counter beside it.

Two keys per user: the serialized message list and an integer epoch.
`invalidate_continuum()` deletes the list and bumps the epoch in one atomic
Valkey operation, so a writer still holding the pre-collapse epoch cannot
land a stale list between a delete and a separate bump.
`set_continuum_if_epoch()` writes the list only while the epoch still matches
what the loader captured; a mismatch means a collapse happened, and the
skipped write leaves the cache empty for the next load to rebuild from
PostgreSQL.

A cache miss is meaningful: the pool reads it as a new session boundary.
Invalidation is event-driven — segment collapse and a failed commit — so a
miss never means the entry expired.
"""
from __future__ import annotations

import json
import logging

from cns.core.message import Message
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)


class ValkeyMessageCache:
    """
    Manages continuum message cache in Valkey.

    Cache invalidation is event-driven (triggered by segment timeout),
    not TTL-based. Cache miss indicates new session requiring boundary marker.
    """

    def __init__(self):
        """
        Initialize Valkey continuum cache.

        Cache invalidation is event-driven via segment timeout, not TTL-based.
        """
        from clients.valkey_client import get_valkey_client
        self.valkey = get_valkey_client()

        self.key_prefix = "continuum"

        logger.info("ValkeyMessageCache initialized (event-driven invalidation)")
    
    def _get_key(self, user_id: str) -> str:
        """Generate cache key for user continuum messages."""
        return f"{self.key_prefix}:{user_id}:messages"

    def _get_epoch_key(self, user_id: str) -> str:
        """Generate the collapse-epoch counter key for a user."""
        return f"{self.key_prefix}:{user_id}:epoch"

    def _serialize_messages(self, messages: list[Message]) -> str:
        """
        Serialize messages to JSON for storage.

        Args:
            messages: List of Message objects

        Returns:
            JSON string representation
        """
        return json.dumps([msg.to_dict() for msg in messages])

    def _deserialize_messages(self, data: str) -> list[Message]:
        """
        Deserialize JSON data back to Message objects.

        Args:
            data: JSON string from Valkey

        Returns:
            List of Message objects
        """
        messages = []
        serialized = json.loads(data)

        for msg_dict in serialized:
            messages.append(Message.from_dict(msg_dict))

        return messages
    
    def get_continuum(self) -> list[Message] | None:
        """
        Get continuum messages from Valkey cache.

        Cache miss indicates a new session (invalidated by segment timeout).
        Epoch tracking for commit-time compare-and-set lives in get_epoch().

        Requires: Active user context (set via set_current_user_id during authentication)

        Returns:
            List of messages if cached, None if not found in cache

        Raises:
            ValkeyError: If Valkey infrastructure is unavailable
            RuntimeError: If no user context is set
        """
        user_id = get_current_user_id()
        key = self._get_key(user_id)
        data = self.valkey.get(key)

        if data:
            logger.debug(f"Found cached continuum for user {user_id}")
            return self._deserialize_messages(data)
        else:
            logger.debug(f"No cached continuum found for user {user_id}")
            return None

    def get_epoch(self) -> int:
        """Read the current collapse epoch for the ambient user (missing = 0)."""
        user_id = get_current_user_id()
        value = self.valkey.get(self._get_epoch_key(user_id))
        return int(value) if value else 0

    def set_continuum_if_epoch(self, messages: list[Message], epoch: int) -> bool:
        """
        Store continuum messages in Valkey only if the collapse epoch is
        unchanged since the continuum was loaded.

        When a segment collapse invalidated the cache between the load and
        this write, the passed epoch is stale and the write is skipped — the
        cache stays empty and the next get_or_create() reloads collapse-
        correct state from the database. A skipped write is not an error.

        Args:
            messages: List of messages to cache
            epoch: Collapse epoch captured with the loaded continuum

        Returns:
            True if the cache write landed, False if it was skipped on epoch mismatch

        Requires: Active user context (set via set_current_user_id during authentication)

        Raises:
            ValkeyError: If Valkey infrastructure is unavailable
            RuntimeError: If no user context is set
        """
        user_id = get_current_user_id()
        epoch_key = self._get_epoch_key(user_id)
        key = self._get_key(user_id)
        data = self._serialize_messages(messages)

        written = self.valkey.set_if_counter_matches(epoch_key, epoch, key, data)
        if written:
            logger.debug(f"Cached continuum for user {user_id} (epoch {epoch})")
        else:
            logger.debug(
                "Skipped continuum cache write for user %s — collapse epoch moved "
                "past %d; next load rebuilds from database", user_id, epoch
            )
        return written

    def invalidate_continuum(self) -> bool:
        """
        Invalidate continuum cache entry.

        Requires: Active user context (set via set_current_user_id during authentication)

        Returns:
            True if cache entry was invalidated, False if entry didn't exist

        Raises:
            ValkeyError: If Valkey infrastructure is unavailable
            RuntimeError: If no user context is set
        """
        user_id = get_current_user_id()
        messages_key = self._get_key(user_id)
        epoch_key = self._get_epoch_key(user_id)

        # One atomic op: no in-flight writer holding the pre-collapse epoch can land a stale write between the delete and the epoch bump.
        deleted_count, _ = self.valkey.delete_and_increment(messages_key, epoch_key)

        if deleted_count:
            logger.debug(f"Invalidated cached continuum for user {user_id}")

        return deleted_count > 0
