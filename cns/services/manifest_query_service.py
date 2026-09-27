"""
Manifest query service for retrieving conversation segment data.

Provides segment data for manifest display with Valkey caching
and event-driven invalidation. Formatting is handled by ManifestTrinket.
"""
from __future__ import annotations

import logging
from typing import TypedDict

from clients.valkey_client import get_valkey_client
from valkey.exceptions import ConnectionError as ValkeyConnectionError
from valkey.exceptions import TimeoutError as ValkeyTimeoutError
from cns.core.events import ManifestUpdatedEvent
from cns.integration.event_bus import EventBus
from cns.infrastructure.continuum_repository import ContinuumRepository, get_continuum_repository

logger = logging.getLogger(__name__)

# Manifest display settings
MANIFEST_DEPTH = 15       # Recent segments in manifest display
MANIFEST_CACHE_TTL = 3600  # Valkey cache TTL in seconds (1 hour)


class ManifestSegment(TypedDict):
    """Segment data returned by ManifestQueryService for manifest display."""
    id: str
    display_title: str
    status: str
    start_time: str | None
    end_time: str | None
    created_at: str | None


# Module-level singleton instance
_manifest_service_instance: ManifestQueryService | None = None


def initialize_manifest_query_service(event_bus: EventBus) -> ManifestQueryService:
    """
    Create and store singleton ManifestQueryService instance (called by factory).

    Args:
        event_bus: Event bus for cache invalidation

    Returns:
        The initialized ManifestQueryService instance
    """
    global _manifest_service_instance
    logger.info("Creating singleton ManifestQueryService instance")
    _manifest_service_instance = ManifestQueryService(event_bus)
    return _manifest_service_instance


def get_manifest_query_service() -> ManifestQueryService:
    """
    Get singleton ManifestQueryService instance.

    Returns:
        The initialized ManifestQueryService

    Raises:
        RuntimeError: If service not initialized (factory not run)
    """
    if _manifest_service_instance is None:
        raise RuntimeError(
            "ManifestQueryService not initialized. "
            "Ensure CNSIntegrationFactory has been initialized."
        )
    return _manifest_service_instance


class ManifestQueryService:
    """
    Retrieves conversation segment data for manifest display.

    Queries segment sentinels from messages table with Valkey caching.
    Formatting is delegated to ManifestTrinket following the trinket pattern.
    """

    def __init__(self, event_bus: EventBus, continuum_repository: ContinuumRepository | None = None):
        """
        Initialize manifest query service.

        Args:
            event_bus: Event bus for subscribing to ManifestUpdatedEvent
            continuum_repository: Continuum repository (uses singleton if not provided)
        """
        self.valkey = get_valkey_client()
        self.cache_ttl = MANIFEST_CACHE_TTL
        self.continuum_repository = continuum_repository or get_continuum_repository()

        # Subscribe to manifest update events for cache invalidation
        event_bus.subscribe('ManifestUpdatedEvent', self._handle_manifest_updated)
        logger.info("ManifestQueryService subscribed to ManifestUpdatedEvent")

    def _handle_manifest_updated(self, event: ManifestUpdatedEvent) -> None:
        """
        Handle manifest update event by invalidating cache.

        Args:
            event: ManifestUpdatedEvent with user_id
        """
        cache_key = f"manifest_segments:{event.user_id}"
        try:
            self.valkey.delete(cache_key)
            logger.debug(f"Invalidated manifest cache for user {event.user_id}")
        except Exception as e:
            logger.warning(f"Failed to invalidate manifest cache: {e}")

    def get_segments(self, user_id: str, limit: int = MANIFEST_DEPTH) -> list[ManifestSegment]:
        """
        Get segment data for manifest display.

        Returns raw segment dictionaries for the trinket to format.
        Results are cached in Valkey for performance.

        Args:
            user_id: User ID for query
            limit: Maximum number of segments (default: MANIFEST_DEPTH)

        Returns:
            List of segment dictionaries with keys:
            - id: Segment ID
            - display_title: Title for display
            - status: 'active' or 'collapsed'
            - start_time: ISO format start time
            - end_time: ISO format end time
            - created_at: ISO format creation time
        """
        # Try cache first
        cache_key = f"manifest_segments:{user_id}"
        try:
            import json
            cached = self.valkey.get(cache_key)
            if cached:
                logger.debug(f"Manifest cache hit for user {user_id}")
                cached_str = cached.decode('utf-8') if isinstance(cached, bytes) else cached
                return json.loads(cached_str)
        except (ValkeyConnectionError, ValkeyTimeoutError) as e:
            logger.warning(f"Valkey unavailable for manifest cache read: {e}")

        # Query segments from database
        segments = self._query_segments(user_id, limit)

        # Cache the result
        if segments:
            try:
                import json
                self.valkey.setex(cache_key, self.cache_ttl, json.dumps(segments))
                logger.debug(f"Cached manifest segments for user {user_id} (TTL={self.cache_ttl}s)")
            except (ValkeyConnectionError, ValkeyTimeoutError) as e:
                logger.warning(f"Valkey unavailable for manifest cache write: {e}")

        return segments

    def _query_segments(self, user_id: str, limit: int) -> list[ManifestSegment]:
        """
        Query segment boundary sentinels from messages table.

        Args:
            user_id: User ID for RLS
            limit: Maximum segments to return

        Returns:
            List of segment dictionaries with metadata

        Raises:
            RuntimeError: If database query fails
        """
        segment_messages = self.continuum_repository.find_all_segments(user_id, limit)

        segments = []
        for msg in segment_messages:
            metadata = msg.metadata

            # Use display_title from metadata for manifest tree, fallback to content for in-progress segments
            display_title = metadata.get('display_title')
            if not display_title:
                # Fallback for active segments or segments without display_title
                display_title = msg.content if msg.content else '[Segment in progress]'
                # Truncate if using full content as fallback
                if len(display_title) > 50:
                    display_title = display_title[:47] + "..."

            segments.append({
                'id': str(msg.id),
                'display_title': display_title,
                # 'collapsing' is a transient mid-claim state lasting seconds; the
                # segment is still open while the claim runs, so the UI sees 'active'.
                'status': (
                    'active'
                    if metadata.get('status') == 'collapsing'
                    else metadata.get('status', 'unknown')
                ),
                'start_time': metadata.get('segment_start_time'),
                'end_time': metadata.get('segment_end_time'),
                'created_at': msg.created_at.isoformat() if msg.created_at else None
            })

        # Reverse to chronological order (oldest first)
        segments.reverse()

        logger.debug(f"Queried {len(segments)} segments for user {user_id}")
        return segments
