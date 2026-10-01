"""
Continuum pool using Valkey for distributed caching.

Provides session detection and automatic expiration for continuums,
replacing the in-memory LRU pool with Valkey-based caching.
"""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable

from cns.core.continuum import Continuum
from cns.core.message import Message
from cns.infrastructure.continuum_repository import ContinuumRepository
from cns.infrastructure.valkey_message_cache import ValkeyMessageCache
from cns.core.segment_cache_loader import SegmentCacheLoader
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)


def _trim_content_blocks(content: list, limit: int) -> int:
    """
    Trim oversized list message content in place until it fits the limit.

    The durable form of list content is json.dumps(content) (see
    Message.to_db_tuple), so that serialization is what the limit is
    measured against. When it exceeds ``limit``, the largest string field
    in a block — the offending text or media payload — is truncated and
    marked. The list/block structure is preserved so the row is stored with
    content_type "json" and reloads as a content array; the whole list is
    never flattened to a repr.

    Mirrors the format-aware tool-result truncation upstream in the
    orchestrator: cut the payload, not the envelope.

    Blocks are trimmed in place so the continuum's in-memory copy and the
    persisted row stay identical — a cache reload serves exactly what
    was saved.

    Returns:
        The serialized length before trimming; 0 when no trim was needed.
    """
    original_len = len(json.dumps(content))
    if original_len <= limit:
        return 0
    while True:
        total = len(json.dumps(content))
        if total <= limit:
            break
        # Largest trimmable string field across all blocks.
        candidates = [
            (len(value), block, key)
            for block in content
            if isinstance(block, dict)
            for key, value in block.items()
            if isinstance(value, str)
        ]
        if not candidates:
            break  # no trimmable payload; keep the structure as-is
        field_len, block, key = max(candidates, key=lambda c: c[0])
        marker = (f"\n\n[Block truncated: {field_len:,} chars cut to fit "
                  f"the {limit:,} char persistence limit]")
        keep = field_len - (total - limit) - len(marker)
        if keep < 0:
            keep = 0
        if keep + len(marker) >= field_len:
            break  # field already at the marker floor — cannot shrink further
        block[key] = block[key][:keep] + marker
    return original_len


class UnitOfWork:
    """
    Unit of Work pattern for continuum operations.
    
    Accumulates changes during a continuum turn and commits them
    atomically to both database and cache.
    """
    
    def __init__(self, continuum: Continuum, pool: 'ContinuumPool',
                 cache_epoch: int = 0):
        """
        Initialize unit of work.

        Args:
            continuum: Continuum being modified
            pool: Parent continuum pool for persistence operations
            cache_epoch: Collapse epoch captured when the continuum was loaded;
                used to skip the cache write on commit if a collapse invalidated
                the cache in the meantime
        """
        self.continuum = continuum
        self.pool = pool
        self.cache_epoch = cache_epoch
        self.pending_messages: list[Message] = []
        self.metadata_updated = False
        self._post_commit_callbacks: list[Callable[[], None]] = []
        
    def add_messages(self, *messages: Message) -> None:
        """
        Queue messages for persistence.

        Enforces a per-message character limit as a safety net against
        oversized content bricking the conversation. String content is cut
        to the limit; list content is trimmed format-aware — the offending
        block payloads are truncated in place so the persisted row keeps
        the list/block structure (content_type "json") and the in-memory
        continuum copy stays identical to the durable one. Tool results
        have a tighter, format-aware limit upstream in the orchestrator.

        Args:
            *messages: One or more Message objects to persist
        """
        # Hard cap on any single message content before DB persistence.
        # Larger than tool result truncation (100k) since assistant messages
        # can legitimately be long.
        limit = 150_000

        for msg in messages:
            content = msg.content
            if isinstance(content, str):
                if len(content) > limit:
                    truncated = content[:limit]
                    truncated += f"\n\n[Message truncated: {len(content):,} chars exceeded {limit:,} char limit]"
                    msg = Message(
                        id=msg.id,
                        content=truncated,
                        role=msg.role,
                        created_at=msg.created_at,
                        metadata=msg.metadata,
                        tool_call_id=msg.tool_call_id,
                        is_error=msg.is_error
                    )
                    logger.warning(
                        "Truncated oversized %s message at persistence: %d -> %d chars",
                        msg.role, len(content), len(truncated)
                    )
            else:
                original_len = _trim_content_blocks(content, limit)
                if original_len:
                    logger.warning(
                        "Trimmed oversized %s message blocks at persistence: "
                        "%d -> %d serialized chars",
                        msg.role, original_len, len(json.dumps(content))
                    )
            self.pending_messages.append(msg)
        
    def mark_metadata_updated(self) -> None:
        """Mark that continuum metadata needs to be updated."""
        self.metadata_updated = True

    def add_post_commit_callback(self, callback: Callable[[], None]) -> None:
        """Run a callback only after database and cache persistence succeed."""
        self._post_commit_callbacks.append(callback)
        
    def commit(self) -> None:
        """
        Persist all accumulated changes atomically.

        Saves messages to database, updates cache, and persists metadata changes.
        Segment creation now happens automatically in repository.save_message().
        """
        if self.pending_messages:
            # Batch save to database. The batch lands in a single transaction,
            # so a mid-batch failure rolls back atomically — but the cached
            # copy must not survive the failure either: invalidate it so no
            # stale-cache/stale-disk divergence keeps serving the old whole.
            try:
                self.pool.repository.save_messages_batch(
                    self.pending_messages,
                    self.continuum.id,
                    self.continuum.user_id
                )
            except Exception:
                try:
                    self.pool.valkey_cache.invalidate_continuum()
                except Exception:
                    logger.critical(
                        "Valkey cache invalidation failed for continuum %s after a "
                        "failed message-batch save; stale cache may serve until the "
                        "next collapse forces a reload. The failed batch was rolled "
                        "back, so the database holds no partial turn.",
                        self.continuum.id,
                        exc_info=True,
                    )
                raise

            # Update Valkey cache once with current continuum state, but only
            # if no segment collapse invalidated it since the continuum was
            # loaded — a stale write would re-cache the pre-collapse message
            # set. A skipped write is not an error: the cache stays empty and
            # the next get_or_create() reloads collapse-correct state from DB.
            # The cache is an optional layer atop the durable DB write that just
            # landed: a cache failure must not fail the request, but a stale
            # cache entry must not be left serving either.
            try:
                self.pool.valkey_cache.set_continuum_if_epoch(
                    self.continuum.messages, self.cache_epoch
                )
            except Exception:
                logger.error(
                    "Valkey cache write failed after DB commit for continuum %s; "
                    "users may see stale conversation until the cache reloads. "
                    "The turn itself is safely persisted.",
                    self.continuum.id,
                    exc_info=True,
                )
                try:
                    self.pool.valkey_cache.invalidate_continuum()
                except Exception:
                    logger.critical(
                        "Valkey cache invalidation failed for continuum %s after a "
                        "failed cache write; stale cache may serve until the next "
                        "collapse forces a reload. The turn itself is safely persisted.",
                        self.continuum.id,
                        exc_info=True,
                    )

            logger.debug(f"Committed {len(self.pending_messages)} messages for continuum {self.continuum.id}")

        # Update metadata if needed
        if self.metadata_updated:
            self.pool.repository.update_continuum_metadata(self.continuum)
            logger.debug(f"Updated metadata for continuum {self.continuum.id}")

        callbacks = tuple(self._post_commit_callbacks)
        self._post_commit_callbacks.clear()
        for callback in callbacks:
            callback()

    def _get_real_messages(self) -> list[Message]:
        """
        Get conversation messages, excluding summaries and boundaries.

        Returns:
            List of actual conversation messages (user/assistant exchanges)
        """
        return [
            msg for msg in self.continuum.messages
            if not msg.metadata.get('system_notification')
            and not msg.metadata.get('is_segment_boundary')
        ]


class ContinuumPool:
    """
    Continuum pool backed by Valkey with TTL-based session management.
    
    Uses Valkey for distributed caching with automatic expiration,
    enabling clear session boundary detection when continuums expire.
    """
    
    def __init__(self, repository: ContinuumRepository,
                 session_loader: SegmentCacheLoader):
        """
        Initialize pool with repository and session loader.

        Args:
            repository: Repository for continuum persistence
            session_loader: Session cache loader for new sessions
        """
        self.repository = repository
        self.session_loader = session_loader
        self.valkey_cache = ValkeyMessageCache()
        # Per-user locks keyed by user_id: same-user get_or_create calls stay
        # serialized, while cross-user IO (Valkey/Postgres) does not block on
        # one another. The epoch compare-and-set remains the collapse guard.
        self._user_locks: dict[str, threading.Lock] = {}
        self._user_locks_guard = threading.Lock()
        # Collapse epoch captured at load, per user — consumed by begin_work()
        # so UnitOfWork.commit() can compare-and-set against the live epoch.
        self._loaded_epochs: dict[str, int] = {}
        
    def get_or_create(self) -> Continuum:
        """
        Get continuum from Valkey cache or create new one.

        Checks Valkey first - if not found, it's a new session.
        Uses ambient user context from set_current_user_id().

        Returns:
            Continuum instance with appropriate cache
        """
        user_id = get_current_user_id()

        # Serialize only same-user get_or_create calls; cross-user turns no
        # longer contend on one process-global lock.
        user_key = str(user_id)
        with self._user_locks_guard:
            user_lock = self._user_locks.setdefault(user_key, threading.Lock())

        with user_lock:
            # Check Valkey cache first. The epoch is captured BEFORE the message
            # read: any collapse that invalidates the messages we are about to
            # read also bumps the epoch past what we capture, so the commit's
            # compare-and-set skips the stale write.
            epoch = self.valkey_cache.get_epoch()
            cached_messages = self.valkey_cache.get_continuum()
            self._loaded_epochs[str(user_id)] = epoch

            # Get continuum structure from DB (must exist from signup)
            continuum = self.repository.get_continuum(user_id)
            if not continuum:
                raise RuntimeError(f"Continuum not found for user {user_id}. Continuum should be created during signup.")

            # No callback needed - using Unit of Work pattern

            if cached_messages is None:
                # NEW SESSION - continuum expired from Valkey
                logger.info(f"New session detected for user {user_id} - loading with session boundary")

                # Load session context (segment summaries + boundary)
                messages = self.session_loader.load_session_cache(
                    str(continuum.id), user_id
                )
                continuum.apply_cache(messages)

                # Epoch captured before the DB read — a collapse in between
                # means a newer cache entry exists; don't clobber it.
                if messages:
                    self.valkey_cache.set_continuum_if_epoch(messages, epoch)

            else:
                # CONTINUING SESSION - cache hit
                logger.debug(f"Continuing session for user {user_id}")

                # Apply cached messages to continuum
                continuum.apply_cache(cached_messages)

            return continuum
    
    def begin_work(self, continuum: Continuum) -> UnitOfWork:
        """
        Begin a unit of work for continuum operations.

        Args:
            continuum: Continuum to track changes for

        Returns:
            UnitOfWork instance for accumulating and committing changes
        """
        return UnitOfWork(
            continuum,
            self,
            cache_epoch=self._loaded_epochs.get(str(continuum.user_id), 0),
        )

    def invalidate(self) -> None:
        """
        Remove continuum from Valkey cache.

        Requires: Active user context (set via set_current_user_id during authentication)

        Raises:
            RuntimeError: If no user context is set
        """
        user_id = get_current_user_id()
        if self.valkey_cache.invalidate_continuum():
            logger.debug(f"Invalidated cached continuum for user {user_id}")
        else:
            logger.debug(f"No cached continuum to invalidate for user {user_id}")
    
# Global continuum pool instance
_continuum_pool: ContinuumPool | None = None


def initialize_continuum_pool(repository: ContinuumRepository,
                                session_loader: SegmentCacheLoader) -> ContinuumPool:
    """
    Initialize the global continuum pool with required dependencies.

    Must be called during application startup.

    Args:
        repository: Continuum repository
        session_loader: Session cache loader for new sessions

    Returns:
        Initialized ContinuumPool instance
    """
    global _continuum_pool
    _continuum_pool = ContinuumPool(repository, session_loader)
    logger.info("Continuum pool initialized with session cache loader")
    return _continuum_pool


def get_continuum_pool() -> ContinuumPool:
    """
    Get the global continuum pool instance.

    Raises:
        RuntimeError: If pool has not been initialized
    """
    if _continuum_pool is None:
        raise RuntimeError(
            "Continuum pool not initialized. Call initialize_continuum_pool() "
            "during application startup."
        )
    return _continuum_pool
