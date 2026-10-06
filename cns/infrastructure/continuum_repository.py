"""
Continuum repository for CNS.

Handles persistence and retrieval of continuums and messages
with RLS (Row Level Security) per user.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, NamedTuple, TypedDict
from uuid import UUID, uuid4

from cns.core.continuum import Continuum
from cns.core.state import ContinuumState
from cns.core.message import Message
from clients.postgres_client import PostgresClient
from utils.timezone_utils import utc_now, format_utc_iso, parse_utc_time_string
from utils.user_context import get_current_segment_id, set_current_segment_id

logger = logging.getLogger(__name__)

# How long a 'collapsing' claim stays exclusive before it is considered
# crashed/abandoned and may be re-claimed or re-swept (crash recovery: a
# process death between claim and save must not strand the segment).
# Shared by SegmentCollapseHandler's claim UPDATE and the admin timeout sweep
# below so the two arms can never disagree on the bound.
COLLAPSE_CLAIM_STALE_MINUTES = 10

# Content-caused extraction failures before a collapsed segment is abandoned
# (excluded) by the extraction sweep. Only failures caused by the segment's
# own data consume this budget — infrastructure/LLM outages do not
# (ExtractionOrchestrator.extract_unprocessed_segments classifies failures and
# counts only content-caused ones). When the budget is exhausted the sweep
# writes an 'extraction_abandoned' marker on the boundary message so the
# exclusion is visible to operators instead of silent.
EXTRACTION_MAX_CONTENT_FAILURES = 3


class HistoryResult(TypedDict):
    """Chronological keyset page returned by get_history()."""
    messages: list[dict[str, object]]
    has_more: bool
    next_before: tuple[datetime, UUID] | None



class FailedSegment(TypedDict):
    """A collapsed segment where memory extraction failed or hasn't been attempted."""
    message_id: str
    segment_id: str
    extraction_attempts: int
    extraction_content_failures: int


class ActiveSegmentRow(TypedDict):
    """Row from find_all_active_segments_admin() — UUIDs normalized to strings."""
    id: str
    continuum_id: str
    user_id: str
    metadata: dict[str, Any]
    created_at: datetime


class IncrementSegmentTurnResult(NamedTuple):
    """
    Return value from `increment_segment_turn`.

    `segment_id` is always populated. For continuing sessions it is read from
    the existing active sentinel; for new sessions it is freshly allocated in
    memory and the corresponding sentinel is NOT yet persisted — persistence
    is deferred to `save_messages_batch`, where the sentinel INSERT commits
    separately (autocommit) before the message-batch transaction opens.
    Sentinel and first message land in separate commits by design, so a
    message-batch failure leaves a residual orphan-sentinel window. The id
    is exposed to tools (and the deferred sentinel) via the
    `current_segment_id` contextvar, so no separate channel threads it through
    the persistence call stack.
    """
    turn_number: int
    segment_id: str


# Module-level singleton instance
_continuum_repo_instance: ContinuumRepository | None = None


def get_continuum_repository() -> 'ContinuumRepository':
    """
    Get or create singleton ContinuumRepository instance.

    Following the same module-singleton pattern as clients/valkey_client.py,
    this ensures we reuse the same repository instance and its database connection pool.

    Returns:
        Singleton ContinuumRepository instance
    """
    global _continuum_repo_instance
    if _continuum_repo_instance is None:
        logger.info("Creating singleton ContinuumRepository instance")
        _continuum_repo_instance = ContinuumRepository()
    return _continuum_repo_instance


class ContinuumRepository:
    """
    Repository for continuum persistence.

    Handles database operations for continuums and messages
    with automatic user isolation via RLS.
    """
    
    def __init__(self):
        """Initialize repository."""
        self._db_cache = {}
        
    def get_user_db_client(self, user_id: str) -> PostgresClient:
        """Get or create database client for user."""
        if user_id not in self._db_cache:
            self._db_cache[user_id] = PostgresClient("mira_service", user_id=user_id)
        return self._db_cache[user_id]
    
    def get_continuum(self, user_id: str) -> Continuum | None:
        """
        Get most recent continuum for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Most recent continuum or None if no continuums exist
        """
        try:
            db = self.get_user_db_client(user_id)
            
            # Get most recent continuum
            existing = db.execute_query(
                "SELECT * FROM continuums ORDER BY created_at DESC LIMIT 1"
            )
            
            if not existing:
                return None
                
            row = existing[0]
            # Parse JSON metadata if it's a string (asyncpg doesn't auto-parse)
            metadata = row.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata) if metadata else {}
            
            # Convert asyncpg UUID to standard UUID
            from uuid import UUID
            continuum_id = UUID(str(row['id'])) if not isinstance(row['id'], UUID) else row['id']
            
            state = ContinuumState(
                id=continuum_id,
                user_id=row['user_id'],
                metadata=metadata
            )
            
            # Create continuum
            continuum = Continuum(state)
            
            logger.debug(f"Retrieved existing continuum {continuum.id} for user {user_id}")
            return continuum
        except Exception as e:
            logger.error(f"Failed to get continuum for user {user_id}: {str(e)}")
            raise RuntimeError(f"Database operation failed: {str(e)}") from e
    
    def create_continuum(self, user_id: str) -> Continuum:
        """
        Create new continuum for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            New continuum instance
        """
        try:
            # Create new continuum
            continuum = Continuum.create_new(user_id)
            
            # Persist to database
            now = utc_now()
            db = self.get_user_db_client(user_id)
            db.execute_query(
                """
                INSERT INTO continuums (id, user_id, created_at, updated_at, metadata)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    continuum.id,  # Keep as UUID - PostgresClient will convert
                    user_id,
                    now,
                    now,
                    json.dumps(continuum._state.metadata)
                )
            )

            logger.info(f"Created new continuum {continuum.id} for user {user_id}")
            return continuum
        except Exception as e:
            logger.error(f"Failed to create continuum for user {user_id}: {str(e)}")
            raise RuntimeError(f"Database operation failed: {str(e)}") from e
    
    def save_message(self, message: Message, continuum_id: str | UUID, user_id: str) -> None:
        """
        Save message to database.

        Automatically creates segment boundary sentinel when second real message
        is saved (forming a complete user/assistant exchange).

        Args:
            message: Message to save
            continuum_id: Continuum ID (string or UUID)
            user_id: User ID
        """
        try:
            # Additional validation to prevent empty messages from being saved
            if isinstance(message.content, str) and not message.content.strip():
                logger.error(f"Blocking empty {message.role} message for continuum {continuum_id}")
                raise ValueError(f"Cannot save empty {message.role} message to database")

            db = self.get_user_db_client(user_id)

            # Convert continuum_id to UUID if it's a string
            if isinstance(continuum_id, str):
                from uuid import UUID as uuid_type
                continuum_id = uuid_type(continuum_id)

            # Check if this is a real conversation message (not segment/system notification)
            is_real_message = not (
                message.metadata.get('is_segment_boundary') or
                message.metadata.get('system_notification')
            )

            # Ensure active segment exists if this is a real message
            if is_real_message:
                self._ensure_active_segment(continuum_id, user_id, message.created_at, db)

            # Extract segment embedding if this is a segment boundary with embedding
            segment_embedding_value = None
            if message.metadata.get('is_segment_boundary') and message.metadata.get('segment_embedding_value') is not None:
                embedding_list = message.metadata['segment_embedding_value']
                # Format as PostgreSQL vector: '[0.1, 0.2, ...]'
                segment_embedding_value = '[' + ','.join(str(x) for x in embedding_list) + ']'

            # Stamp the content-type identity tag at write time so reloads
            # key on the tag, never on shape (a plain-text message that happens
            # to be valid JSON must reload as plain text). with_metadata()
            # returns a copy, so the in-memory message is untouched.
            content_type = "json" if not isinstance(message.content, str) else "text"

            # Get base message tuple
            base_tuple = message.with_metadata(content_type=content_type).to_db_tuple(continuum_id, user_id)

            # Upsert message with segment embedding
            # ON CONFLICT handles updates to existing messages (e.g., collapsed segment sentinels)
            db.execute_query(
                """
                INSERT INTO messages (id, continuum_id, user_id, role, content, metadata, created_at, tool_call_id, is_error, segment_embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    tool_call_id = EXCLUDED.tool_call_id,
                    is_error = EXCLUDED.is_error,
                    segment_embedding = EXCLUDED.segment_embedding
                """,
                base_tuple + (segment_embedding_value,)
            )

            # Track user activity day (upstream activity tracking for vacation-proof scoring)
            # Heartbeat stimuli are machine-sourced, not user activity.
            if message.role == "user" and message.metadata.get("heartbeat") != "true":
                try:
                    from utils.user_activity import increment_user_activity_day
                    logger.info(f"[SINGLE] Attempting to increment activity day for user {user_id}")
                    result = increment_user_activity_day(user_id)
                    logger.info(f"[SINGLE] Activity day incremented successfully for user {user_id}, result: {result}")
                except Exception as e:
                    logger.error(f"[SINGLE] Failed to increment activity day for user {user_id}: {e}", exc_info=True)
                # Note: if segment was paused, increment_segment_turn() (called at
                # API entry before message save) auto-resumes it to 'active'

        except Exception as e:
            logger.error(f"Failed to save message to continuum {continuum_id}: {str(e)}")
            raise RuntimeError(f"Database operation failed: {str(e)}") from e

    def _ensure_active_segment(self, continuum_id: UUID, user_id: str, current_message_time: datetime, db: PostgresClient) -> None:
        """
        Ensure active segment exists, creating one when the segment's first
        real message is saved.

        The sentinel is created whenever no active sentinel exists — which
        includes the segment's first real message — matching the turn-1
        initialization of create_segment_boundary_sentinel. The INSERT runs
        on the autocommitted connection, so it lands in a separate commit
        from the message batch that follows.

        Args:
            continuum_id: Continuum UUID
            user_id: User ID
            current_message_time: Timestamp of message being saved
            db: Database client
        """
        # Check if active segment already exists
        active_segment_query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'active'
            ORDER BY created_at DESC
            LIMIT 1
        """
        active_segments = db.execute_query(active_segment_query, (continuum_id,))

        if not active_segments:
            # No active segment - create one for this segment's first message
            # This ensures segment exists from turn 1, enabling proper turn counting

            # Find the most recent collapsed segment to determine time boundary
            last_segment_query = """
                SELECT metadata->>'segment_end_time' as segment_end_time
                FROM messages
                WHERE continuum_id = %s
                    AND metadata->>'is_segment_boundary' = 'true'
                    AND metadata->>'status' = 'collapsed'
                ORDER BY created_at DESC
                LIMIT 1
            """
            last_segment_row = db.execute_query(last_segment_query, (continuum_id,))

            # If there's a previous collapsed segment, only look at messages after it ended
            if last_segment_row and last_segment_row[0].get('segment_end_time'):
                last_segment_end = parse_utc_time_string(last_segment_row[0]['segment_end_time'])

                first_message_query = """
                    SELECT created_at FROM messages
                    WHERE continuum_id = %s
                        AND created_at > %s
                        AND COALESCE(metadata->>'is_segment_boundary', 'false') = 'false'
                        AND COALESCE(metadata->>'system_notification', 'false') = 'false'
                    ORDER BY created_at ASC
                    LIMIT 1
                """
                first_msg_row = db.execute_query(first_message_query, (continuum_id, last_segment_end))
            else:
                # No previous collapsed segment - this may be the absolute first message
                first_message_query = """
                    SELECT created_at FROM messages
                    WHERE continuum_id = %s
                        AND COALESCE(metadata->>'is_segment_boundary', 'false') = 'false'
                        AND COALESCE(metadata->>'system_notification', 'false') = 'false'
                    ORDER BY created_at ASC
                    LIMIT 1
                """
                first_msg_row = db.execute_query(first_message_query, (continuum_id,))

            # Use first existing message time, or current message time if this IS the first message
            first_message_time = first_msg_row[0]['created_at'] if first_msg_row else current_message_time

            # Create segment boundary sentinel (initializes segment_turn_count to 1)
            from cns.services.segment_helpers import create_segment_boundary_sentinel

            # The segment_id was allocated at API entry by increment_segment_turn
            # and exposed to tools via set_current_segment_id(). Source it from the
            # request contextvar so the persisted sentinel uses the same id the
            # tools saw this turn. When absent (e.g. fixture loaders with no request
            # context) the helper allocates a fresh uuid4. A stale contextvar id can
            # collide with an existing sentinel of this continuum (the contextvar
            # outlives the segment it named), so re-check before use.
            segment_id = get_current_segment_id()
            if segment_id:
                collisions = db.execute_query("""
                    SELECT id FROM messages
                    WHERE continuum_id = %s
                        AND metadata->>'is_segment_boundary' = 'true'
                        AND metadata->>'segment_id' = %s
                    LIMIT 1
                """, (continuum_id, segment_id))
                if collisions:
                    segment_id = str(uuid4())

            sentinel = create_segment_boundary_sentinel(
                first_message_time=first_message_time,
                continuum_id=str(continuum_id),
                segment_id=segment_id,
            )

            # Direct INSERT with conflict on the unique partial index
            # (idx_messages_active_segment_unique). If a concurrent call already
            # created an active sentinel for this continuum, DO NOTHING — the
            # race loser silently no-ops instead of creating a duplicate.
            # Content-type tag stamped like every other write path so the
            # sentinel reloads by identity, not shape.
            base_tuple = sentinel.with_metadata(content_type="text").to_db_tuple(continuum_id, user_id)
            db.execute_query(
                """
                INSERT INTO messages (id, continuum_id, user_id, role, content, metadata, created_at, tool_call_id, is_error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (continuum_id)
                    WHERE metadata->>'is_segment_boundary' = 'true'
                      AND metadata->>'status' = 'active'
                DO NOTHING
                """,
                base_tuple
            )

            logger.info(f"Created segment boundary sentinel for continuum {continuum_id}")
    
    def save_messages_batch(self, messages: list[Message], continuum_id: str | UUID, user_id: str) -> None:
        """
        Save multiple messages to database as one atomic batch.

        All message inserts run in a single database transaction
        (PostgresClient.execute_transaction): either the entire batch
        persists or none of it does — a mid-batch failure rolls back
        cleanly instead of leaving a partially persisted turn. The
        segment sentinel (via _ensure_active_segment) and activity-day
        tracking commit separately by design: the sentinel INSERT runs on
        the autocommitted connection before the message transaction opens,
        so a message-batch failure rolls the messages back but leaves the
        sentinel committed — the orphan-sentinel window is residual, not
        eliminated. UnitOfWork.commit
        invalidates the Valkey cache when this method raises so a stale
        cached copy cannot outlive the failure.

        Args:
            messages: List of messages to save
            continuum_id: Continuum ID (string or UUID)
            user_id: User ID
        """
        if not messages:
            return

        try:
            # Additional validation to prevent empty messages from being saved
            for message in messages:
                if isinstance(message.content, str) and not message.content.strip():
                    logger.error(f"Blocking empty {message.role} message for continuum {continuum_id}")
                    raise ValueError(f"Cannot save empty {message.role} message to database")

            db = self.get_user_db_client(user_id)

            # Convert continuum_id to UUID if it's a string
            if isinstance(continuum_id, str):
                from uuid import UUID as uuid_type
                continuum_id = uuid_type(continuum_id)

            # Check if any messages are real conversation messages
            real_messages = [
                msg for msg in messages
                if not (
                    msg.metadata.get('is_segment_boundary') or
                    msg.metadata.get('system_notification')
                )
            ]

            # Ensure active segment exists if we have real messages
            if real_messages:
                # Use the earliest real message timestamp
                earliest_timestamp = min(msg.created_at for msg in real_messages)
                self._ensure_active_segment(continuum_id, user_id, earliest_timestamp, db)
                logger.debug(f"Checked segment boundary for batch save with {len(real_messages)} real messages")

            # Insert every message as ONE database transaction: per-statement
            # autocommit here would leave a partially persisted turn on a
            # mid-batch failure while the cache keeps serving the old whole.
            insert_operations: list[tuple[str, tuple[object, ...]]] = []
            has_user_message = False
            for message in messages:
                # Extract segment embedding if this is a segment boundary with embedding
                segment_embedding_value = None
                if message.metadata.get('is_segment_boundary') and message.metadata.get('segment_embedding_value') is not None:
                    embedding_list = message.metadata['segment_embedding_value']
                    # Format as PostgreSQL vector: '[0.1, 0.2, ...]'
                    segment_embedding_value = '[' + ','.join(str(x) for x in embedding_list) + ']'

                # Stamp the content-type identity tag (see save_message) and
                # get the base message tuple
                content_type = "json" if not isinstance(message.content, str) else "text"
                base_tuple = message.with_metadata(content_type=content_type).to_db_tuple(continuum_id, user_id)

                insert_operations.append((
                    """
                    INSERT INTO messages (id, continuum_id, user_id, role, content, metadata, created_at, tool_call_id, is_error, segment_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    base_tuple + (segment_embedding_value,)
                ))
                # Heartbeat stimuli are machine-sourced, not user activity.
                if message.role == "user" and message.metadata.get("heartbeat") != "true":
                    has_user_message = True

            db.execute_transaction(insert_operations)

            # Track user activity day if batch contained user message
            if has_user_message:
                try:
                    from utils.user_activity import increment_user_activity_day
                    logger.info(f"Attempting to increment activity day for user {user_id}")
                    result = increment_user_activity_day(user_id)
                    logger.info(f"Activity day incremented successfully for user {user_id}, result: {result}")
                except Exception as e:
                    logger.error(f"Failed to increment activity day for user {user_id}: {e}", exc_info=True)

            logger.debug(f"Saved {len(messages)} messages to continuum {continuum_id}")

        except Exception as e:
            logger.error(f"Failed to save {len(messages)} messages to continuum {continuum_id}: {str(e)}")
            raise RuntimeError(f"Database batch operation failed: {str(e)}") from e

    def _parse_message_rows(self, rows: list[dict[str, Any]]) -> list[Message]:
        """Convert raw database rows into Message instances."""
        messages: list[Message] = []

        for row in rows:
            message_id = row.get("id")
            try:
                canonical_id = message_id if isinstance(message_id, UUID) else UUID(str(message_id))
            except (ValueError, TypeError):
                logger.warning(f"Skipping message row with invalid ID: {message_id}")
                continue

            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata) if metadata else {}
                except json.JSONDecodeError:
                    logger.debug("Failed to parse message metadata JSON; defaulting to empty dict")
                    metadata = {}
            metadata = metadata or {}

            # Reconstitute content by its stored identity tag, never by shape.
            # A plain-text user message that happens to be valid JSON must
            # reload as plain text; rows written before the tag existed carry
            # no tag and are treated as plain text.
            content = row.get("content")
            if metadata.get("content_type") == "json" and isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Message {canonical_id} tagged as JSON content failed to parse; keeping raw string"
                    )

            # Read tool_call_id/is_error from dedicated columns
            tool_call_id = row.get("tool_call_id")
            is_error = row.get("is_error", False)

            role = row.get("role")

            messages.append(
                Message(
                    id=canonical_id,
                    content=content,
                    role=role,
                    created_at=row.get("created_at"),
                    metadata=metadata,
                    tool_call_id=tool_call_id,
                    is_error=is_error,
                )
            )

        return messages

    def load_messages_with_metadata(self, 
                                  continuum_id: str, 
                                  user_id: str,
                                  metadata_filters: dict[str, str | bool],
                                  limit: int | None = None,
                                  order_desc: bool = True) -> list[Message]:
        """
        Load messages with flexible metadata filtering.

        This consolidates loading for segment boundaries, system notifications, and other metadata-based queries.

        Args:
            continuum_id: Continuum ID
            user_id: User ID for RLS
            metadata_filters: Dict of metadata key-value pairs to filter by
            limit: Maximum number of messages to return
            order_desc: If True, order by created_at DESC (most recent first)

        Returns:
            List of messages matching criteria

        Examples:
            # Load collapsed segment boundaries
            load_messages_with_metadata(conv_id, user_id, {
                'is_segment_boundary': 'true',
                'status': 'collapsed'
            }, limit=5)

            # Load system notifications
            load_messages_with_metadata(conv_id, user_id, {
                'system_notification': 'true'
            }, limit=3)
        """
        db = self.get_user_db_client(user_id)
        
        # Build query with metadata filters
        where_conditions = ["continuum_id = %s"]
        params = [continuum_id]
        
        # Add metadata conditions
        for key, value in metadata_filters.items():
            # Handle boolean values properly
            if isinstance(value, bool):
                value = 'true' if value else 'false'
            where_conditions.append(f"metadata->>'{key}' = %s")
            params.append(str(value))
        
        order_clause = "DESC" if order_desc else "ASC"
        query = f"""
            SELECT * FROM messages 
            WHERE {' AND '.join(where_conditions)}
            ORDER BY created_at {order_clause}
        """
        
        if limit:
            query += f" LIMIT {limit}"

        message_rows = db.execute_query(query, tuple(params))

        # Reverse if we queried DESC to get chronological order
        if order_desc and len(message_rows) > 1:
            message_rows.reverse()

        return self._parse_message_rows(message_rows)

    def load_tool_result_by_id(
        self,
        user_id: str,
        session_id: str,
        tool_result_id: str,
    ) -> str | None:
        """Load raw tool-result text by its session-scoped retrieval ID."""
        try:
            message_id = UUID(tool_result_id.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError("Malformed tool result identifier") from exc

        db = self.get_user_db_client(user_id)
        rows = db.execute_query(
            """
            SELECT content
            FROM messages
            WHERE id = %s
              AND user_id = %s
              AND role = 'tool'
              AND metadata->>'tool_result_session_id' = %s
              AND metadata->>'tool_result_id' = %s
            """,
            (message_id, user_id, session_id, tool_result_id),
        )
        if len(rows) > 1:
            raise RuntimeError(
                f"Ambiguous tool result ID in session {session_id}: {tool_result_id}"
            )
        if not rows:
            return None
        content = rows[0].get("content")
        if not isinstance(content, str):
            raise RuntimeError(
                f"Tool result {tool_result_id} does not contain persisted text"
            )
        return content

    def get_history(
        self,
        user_id: str,
        limit: int = 50,
        before: tuple[datetime, UUID] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        message_type: str = "regular",
    ) -> HistoryResult:
        """
        Get one chronological history page by exclusive keyset cursor.

        Offset pagination over `ORDER BY created_at DESC` is unstable under
        concurrent inserts: rows shift across the page boundary, producing
        duplicates and skips. `created_at` alone is not a total order either,
        because one turn's messages carry microsecond offsets that can tie.
        `(created_at, id)` is a proper keyset, so it is the boundary.

        Args:
            user_id: User ID for RLS
            limit: Maximum number of messages to return
            before: Exclusive `(created_at, id)` keyset boundary
            start_date: Optional start date filter
            end_date: Optional end date filter
            message_type: Type of messages to retrieve ("regular" or "all")

        Returns:
            Dictionary with messages, pagination info, and metadata
        """
        db = self.get_user_db_client(user_id)
        
        # Build query with optional date filtering
        where_conditions = ["user_id = %s"]
        params = [user_id]

        # Filter by message type
        if message_type == "all":
            # Include all messages, no additional filter
            pass
        else:
            # Default to regular messages (exclude system notifications and
            # keepsleeping heartbeat turns; breakout heartbeat turns are real
            # conversation the user must be able to read after the fact)
            where_conditions.append("COALESCE(metadata->>'system_notification', 'false') != 'true'")
            where_conditions.append(
                "NOT ("
                "COALESCE(metadata->>'heartbeat', 'false') = 'true' "
                "AND COALESCE(metadata->>'heartbeat_decision', 'keepsleeping') != 'breakout'"
                ")"
            )
            # Keepsleeping-turn exclusion, de-correlated on purpose: turn_id is a
            # per-turn UUID, so equality alone identifies the same turn and the
            # old `other.continuum_id = messages.continuum_id` predicate was pure
            # correlation — it forced a re-scan of `messages` per candidate row
            # (hundreds of loops × ~10 ms seq scan ≈ 6 s per history page on a
            # ~16k-row table). NOT EXISTS probes the partial index
            # idx_messages_keepsleeping_turn_id instead.
            where_conditions.append(
                "(metadata->>'turn_id' IS NULL OR NOT EXISTS ("
                "SELECT 1 FROM messages other "
                "WHERE other.metadata->>'turn_id' = messages.metadata->>'turn_id' "
                "AND other.metadata->>'heartbeat' = 'true' "
                "AND COALESCE(other.metadata->>'heartbeat_decision', 'keepsleeping') != 'breakout'))"
            )
        
        if start_date:
            where_conditions.append("created_at >= %s")
            params.append(start_date)
            
        if end_date:
            where_conditions.append("created_at <= %s")
            params.append(end_date)

        if before is not None:
            before_created_at, before_id = before
            where_conditions.append("(created_at, id) < (%s, %s)")
            params.extend([before_created_at, before_id])
        
        # Get messages with pagination
        query = f"""
            SELECT * FROM messages 
            WHERE {' AND '.join(where_conditions)}
            ORDER BY created_at DESC, id DESC
            LIMIT %s
        """
        params.append(limit + 1)  # Get one extra to check for more results
        
        message_rows = db.execute_query(query, tuple(params))
        
        # Check if there are more results
        has_more = len(message_rows) > limit
        if has_more:
            message_rows = message_rows[:limit]  # Drop the lookahead row

        next_before = None
        if has_more and message_rows:
            oldest = message_rows[-1]
            next_before = (oldest["created_at"], UUID(str(oldest["id"])))
        
        # Format messages for API - newest-first scan, chronological page out
        messages = []
        for row in reversed(message_rows):
            messages.append({
                "id": str(row['id']),
                "role": row['role'],
                "content": row['content'],
                "timestamp": format_utc_iso(row['created_at']),
                "metadata": row.get('metadata', {}),
                "tool_call_id": str(row['tool_call_id']) if row.get('tool_call_id') else None,
                "is_error": bool(row.get('is_error', False)),
            })
        
        return {
            "messages": messages,
            "has_more": has_more,
            "next_before": next_before,
        }
    
    def update_continuum_metadata(self, continuum: Continuum) -> None:
        """
        Update continuum metadata.

        Args:
            continuum: Continuum with updated metadata
        """
        db = self.get_user_db_client(continuum.user_id)

        db.execute_query(
            """
            UPDATE continuums
            SET metadata = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
            (json.dumps(continuum._state.metadata), continuum.id)
        )

    # =========================================================================
    # Segment Query Methods
    # =========================================================================

    def find_active_segment(self, continuum_id: str | UUID, user_id: str) -> Message | None:
        """
        Find non-collapsed segment sentinel for a continuum.

        Matches both 'active' and 'paused' segments — from the caller's
        perspective, a paused segment is still the current segment. The only
        consumer that distinguishes active from paused is the timeout service,
        which uses find_all_active_segments_admin() with its own filter.

        Args:
            continuum_id: Continuum ID
            user_id: User ID

        Returns:
            Active or paused segment sentinel, or None
        """
        db = self.get_user_db_client(user_id)

        query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' IN ('active', 'paused')
            ORDER BY created_at DESC
            LIMIT 1
        """

        rows = db.execute_query(query, (str(continuum_id),))
        messages = self._parse_message_rows(rows)

        return messages[0] if messages else None

    def increment_segment_turn(self, continuum_id: str | UUID, user_id: str) -> IncrementSegmentTurnResult:
        """
        Increment segment turn counter on the active or paused segment sentinel.

        Called at API entry point when a real user message arrives.
        This is the authoritative source for segment turn count.

        If the segment is paused, atomically resumes it to 'active' —
        the timeout clock restarts from this message.

        For continuing segments the sentinel already exists in the DB, so we
        read its segment_id from the RETURNING clause. For new segments we
        allocate a fresh segment_id in memory and expose it via the
        ``current_segment_id`` contextvar; sentinel persistence is deferred
        to ``save_messages_batch``, where the sentinel INSERT commits
        separately (autocommit) before the message-batch transaction —
        sentinel and first message land in separate commits by design, so
        a message-batch failure leaves a residual orphan-sentinel window.

        Args:
            continuum_id: Continuum ID
            user_id: User ID

        Returns:
            IncrementSegmentTurnResult with turn_number and segment_id.
        """
        db = self.get_user_db_client(user_id)

        # Atomically increment turn count, ensure status is 'active', and stamp
        # last_turn_at. Matches both 'active' and 'paused' segments — a paused
        # segment is auto-resumed when the user sends a message.
        # last_turn_at closes a race window: without it, the timeout service can
        # see the reactivated segment and compute inactivity from the last
        # *committed* message (which may be hours old). The stamp records fresh
        # activity before the new message is committed via uow.commit().
        now_iso = format_utc_iso(utc_now())
        query = """
            UPDATE messages
            SET metadata = jsonb_set(
                metadata - 'paused_at',
                '{segment_turn_count}',
                to_jsonb((metadata->>'segment_turn_count')::int + 1)
            ) || jsonb_build_object('status', 'active', 'last_turn_at', %s::text)
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' IN ('active', 'paused')
            RETURNING (metadata->>'segment_turn_count')::int as turn_count,
                      metadata->>'segment_id' as segment_id
        """

        rows = db.execute_returning(query, (now_iso, str(continuum_id)))

        if rows and rows[0].get('turn_count'):
            segment_id = rows[0]['segment_id']
            set_current_segment_id(segment_id)
            return IncrementSegmentTurnResult(rows[0]['turn_count'], segment_id)

        # No active/paused segment exists — allocate a fresh id in memory.
        # Sentinel persistence is deferred to save_messages_batch, where the
        # sentinel INSERT autocommits before the message-batch transaction:
        # separate commits by design, leaving a residual orphan-sentinel
        # window if the message batch fails.
        new_segment_id = str(uuid4())
        set_current_segment_id(new_segment_id)
        return IncrementSegmentTurnResult(1, new_segment_id)

    def set_heartbeat_wake_at(self, continuum_id: str | UUID, user_id: str, wake_at: str) -> bool:
        """
        Stamp the next heartbeat wake time on the active segment sentinel.

        The heartbeat service calls this after every confirm: keepsleeping
        stamps ``now + requested delay`` (or the default interval), breakout
        stamps ``now + default interval``. Two consumers read the stamp:
        the heartbeat dispatcher skips users whose wake time is in the future,
        and the segment timeout service defers collapse until wake_at plus a
        grace window so a long sleep never collapses the session out from
        under MIRA.

        A stale stamp is inert: both consumers only honor a future timestamp,
        so no cleanup pass is needed when MIRA breaks out or the heartbeat is
        disabled.

        Args:
            continuum_id: Continuum ID
            user_id: User ID
            wake_at: UTC ISO timestamp (format_utc_iso) of the next wake

        Returns:
            True if an active segment sentinel was stamped, False if none exists
        """
        db = self.get_user_db_client(user_id)

        query = """
            UPDATE messages
            SET metadata = jsonb_set(metadata, '{heartbeat_wake_at}', to_jsonb(%s::text))
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'active'
            RETURNING id
        """
        rows = db.execute_returning(query, (wake_at, str(continuum_id)))
        return bool(rows)

    def set_heartbeat_retry_at(self, continuum_id: str | UUID, user_id: str, retry_at: str) -> bool:
        """Stamp the heartbeat retry backoff and bump the consecutive-failure
        counter on the active segment sentinel.

        Called after a FAILED heartbeat turn. The backoff deliberately lives on
        its own metadata key, not heartbeat_wake_at: the dispatcher honors it
        for retry pacing, but the segment timeout service defers collapse only
        on heartbeat_wake_at — a real sleep commitment MIRA asked for — so a
        persistently failing heartbeat cannot defer the sweep forever. The
        counter bounds the pre-turn liveness stamp in heartbeat_service, which
        is the sweep's other deferral leg.

        Args:
            continuum_id: Continuum ID
            user_id: User ID
            retry_at: UTC ISO timestamp (format_utc_iso) of the next retry

        Returns:
            True if an active segment sentinel was stamped, False if none exists
        """
        db = self.get_user_db_client(user_id)

        query = """
            UPDATE messages
            SET metadata = jsonb_set(
                    jsonb_set(metadata, '{heartbeat_retry_at}', to_jsonb(%s::text)),
                    '{heartbeat_failures}',
                    to_jsonb(COALESCE((metadata->>'heartbeat_failures')::int, 0) + 1)
                )
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'active'
            RETURNING id
        """
        rows = db.execute_returning(query, (retry_at, str(continuum_id)))
        return bool(rows)

    def reset_heartbeat_failures(self, continuum_id: str | UUID, user_id: str) -> bool:
        """Zero the consecutive-failure counter on the active segment sentinel.

        Called after a successful heartbeat turn: re-arms the pre-turn
        liveness stamp (hang protection) that a failure streak had suspended.

        Args:
            continuum_id: Continuum ID
            user_id: User ID

        Returns:
            True if an active segment sentinel was stamped, False if none exists
        """
        db = self.get_user_db_client(user_id)

        query = """
            UPDATE messages
            SET metadata = jsonb_set(metadata, '{heartbeat_failures}', to_jsonb(0))
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'active'
            RETURNING id
        """
        rows = db.execute_returning(query, (str(continuum_id),))
        return bool(rows)

    def stamp_segment_liveness(self, continuum_id: str | UUID, user_id: str) -> bool:
        """
        Stamp last_turn_at on the active segment sentinel at heartbeat wake-turn dispatch.

        The segment timeout service's last_turn_at guard then covers the wake
        turn for one threshold window: a hung heartbeat turn does not get
        force-tombstoned as abandoned while it is still running.

        Args:
            continuum_id: Continuum ID
            user_id: User ID

        Returns:
            True if an active segment sentinel was stamped, False if none exists
        """
        db = self.get_user_db_client(user_id)

        query = """
            UPDATE messages
            SET metadata = jsonb_set(metadata, '{last_turn_at}', to_jsonb(%s::text))
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'active'
            RETURNING id
        """
        rows = db.execute_returning(query, (format_utc_iso(utc_now()), str(continuum_id)))
        return bool(rows)

    def pause_segment(self, continuum_id: str | UUID, user_id: str) -> bool:
        """
        Pause the active segment, making it invisible to the timeout service.

        A paused segment is excluded from timeout checks entirely. It resumes
        automatically when the user sends their next message (via
        increment_segment_turn matching paused status).

        Args:
            continuum_id: Continuum ID
            user_id: User ID

        Returns:
            True if segment was paused, False if no active segment found
        """
        db = self.get_user_db_client(user_id)

        query = """
            UPDATE messages
            SET metadata = metadata || jsonb_build_object(
                'status', 'paused',
                'paused_at', %s::text
            )
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'active'
            RETURNING id
        """

        rows = db.execute_returning(query, (format_utc_iso(utc_now()), str(continuum_id)))
        return bool(rows)

    def unpause_segment(self, continuum_id: str | UUID, user_id: str) -> bool:
        """
        Explicitly unpause a paused segment without sending a message.

        Typically unpausing happens automatically via increment_segment_turn(),
        but this allows the UI to unpause without requiring a message.

        Args:
            continuum_id: Continuum ID
            user_id: User ID

        Returns:
            True if segment was unpaused, False if no paused segment found
        """
        db = self.get_user_db_client(user_id)

        query = """
            UPDATE messages
            SET metadata = (metadata - 'paused_at') || '{"status": "active"}'::jsonb
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'paused'
            RETURNING id
        """

        rows = db.execute_returning(query, (str(continuum_id),))
        return bool(rows)

    def find_collapsed_segments(
        self,
        continuum_id: str | UUID,
        user_id: str,
        limit: int
    ) -> list[Message]:
        """
        Find recent collapsed segment sentinels for a continuum.

        Args:
            continuum_id: Continuum ID
            user_id: User ID
            limit: Maximum number of segments to return

        Returns:
            List of collapsed segment sentinels in chronological order (oldest first)
        """
        db = self.get_user_db_client(user_id)

        query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'collapsed'
            ORDER BY created_at DESC
            LIMIT %s
        """

        rows = db.execute_query(query, (str(continuum_id), limit))

        # Reverse to get chronological order (oldest first)
        if rows:
            rows.reverse()

        return self._parse_message_rows(rows)

    def find_segment_by_id(
        self,
        continuum_id: str | UUID,
        segment_id: str,
        user_id: str
    ) -> Message | None:
        """
        Find segment sentinel by segment_id.

        Args:
            continuum_id: Continuum ID
            segment_id: Segment UUID
            user_id: User ID

        Returns:
            Segment sentinel or None
        """
        db = self.get_user_db_client(user_id)

        query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'segment_id' = %s
            ORDER BY created_at DESC
            LIMIT 1
        """

        rows = db.execute_query(query, (str(continuum_id), segment_id))
        messages = self._parse_message_rows(rows)

        return messages[0] if messages else None

    def find_all_segments(self, user_id: str, limit: int) -> list[Message]:
        """
        Find all segment sentinels for a user across all continuums.

        Args:
            user_id: User ID
            limit: Maximum number of segments to return

        Returns:
            List of segment sentinels ordered by creation time (newest first)
        """
        db = self.get_user_db_client(user_id)

        query = """
            SELECT
                id,
                role,
                content,
                metadata,
                created_at
            FROM messages
            WHERE metadata->>'is_segment_boundary' = 'true'
            ORDER BY created_at DESC
            LIMIT %s
        """

        rows = db.execute_query(query, (limit,))
        return self._parse_message_rows(rows)

    def find_failed_extraction_segments(
        self,
        user_id: str,
        max_attempts: int = EXTRACTION_MAX_CONTENT_FAILURES,
    ) -> list[FailedSegment]:
        """
        Find collapsed segments where memory extraction failed or hasn't been attempted.

        Excludes segments whose CONTENT-caused failure count has reached
        max_attempts, to prevent infinite retry loops. Infrastructure/LLM
        outages are not counted toward that budget (see
        ExtractionOrchestrator.extract_unprocessed_segments), so a segment is
        abandoned only when its own data has repeatedly failed extraction —
        three provider blips must not permanently silence a segment. Budget-
        excluded segments carry metadata 'extraction_abandoned' = true,
        written by the sweep, so the state is operator-visible.

        Args:
            user_id: User ID
            max_attempts: Skip segments with this many or more content-caused
                extraction failures

        Returns:
            List of dicts with segment_id, message_id, extraction_attempts, and
            extraction_content_failures
        """
        db = self.get_user_db_client(user_id)

        query = """
            SELECT id, metadata
            FROM messages
            WHERE metadata->>'is_segment_boundary' = 'true'
                AND metadata->>'status' = 'collapsed'
                AND (metadata->>'memories_extracted' = 'false'
                     OR metadata->>'memories_extracted' IS NULL)
                AND COALESCE((metadata->>'extraction_content_failures')::int, 0) < %(max_attempts)s
                AND COALESCE(metadata->>'extraction_abandoned', 'false') <> 'true'
            ORDER BY created_at DESC
        """

        rows = db.execute_query(query, {'max_attempts': max_attempts})

        segments = []
        for row in rows:
            metadata = row.get('metadata', {})
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata) if metadata else {}

            segments.append({
                'message_id': str(row['id']),
                'segment_id': metadata.get('segment_id', str(row['id'])),
                'extraction_attempts': metadata.get('extraction_attempts', 0),
                'extraction_content_failures': metadata.get('extraction_content_failures', 0),
            })

        return segments

    def find_all_active_segments_admin(self) -> list[ActiveSegmentRow]:
        """
        Find all active segments across all users (admin query for timeout service).

        Also returns segments whose 'collapsing' claim has gone stale
        (older than COLLAPSE_CLAIM_STALE_MINUTES), so a crashed claim can be
        re-claimed instead of stranding the segment outside the sweep.

        Joined against `users` with `is_active = TRUE`: the sweep runs on
        the BYPASSRLS admin pool, so without the predicate it would keep
        collapsing segments belonging to deactivated accounts.
        Columns are qualified because the JOIN makes `id`/`user_id`/
        `created_at` ambiguous.

        Returns:
            List of dicts with segment data (id, continuum_id, user_id, metadata, created_at)
        """
        from utils.database_session_manager import get_shared_session_manager

        session_manager = get_shared_session_manager()

        with session_manager.get_admin_session() as session:
            rows = session.execute_query("""
                SELECT
                    messages.id,
                    messages.continuum_id,
                    messages.user_id,
                    messages.metadata,
                    messages.created_at
                FROM messages
                JOIN users ON users.id = messages.user_id
                WHERE messages.metadata->>'is_segment_boundary' = 'true'
                    AND (
                        messages.metadata->>'status' = 'active'
                        OR (
                            messages.metadata->>'status' = 'collapsing'
                            AND (messages.metadata->>'collapse_claimed_at')::timestamptz
                                < now() - make_interval(mins => %(stale_minutes)s)
                        )
                    )
                    AND users.is_active = TRUE
                ORDER BY messages.created_at ASC
            """, {'stale_minutes': COLLAPSE_CLAIM_STALE_MINUTES})

            # Normalize UUID objects to strings at boundary (database driver returns UUID objects)
            for row in rows:
                for key, value in row.items():
                    if isinstance(value, UUID):
                        row[key] = str(value)
            return rows

    def load_segment_messages(
        self,
        continuum_id: str | UUID,
        user_id: str,
        sentinel_time: datetime
    ) -> list[Message]:
        """
        Load all conversation messages from an active segment.

        Args:
            continuum_id: Continuum ID
            user_id: User ID
            sentinel_time: Creation time of segment sentinel

        Returns:
            List of messages in chronological order (excludes boundaries,
            system notifications, and keepsleeping heartbeat turns; breakout
            heartbeat turns are real conversation and stay in context)
        """
        db = self.get_user_db_client(user_id)

        query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND created_at >= %s
                AND COALESCE(metadata->>'is_segment_boundary', 'false') = 'false'
                AND COALESCE(metadata->>'system_notification', 'false') = 'false'
                AND NOT (
                    COALESCE(metadata->>'heartbeat', 'false') = 'true'
                    AND COALESCE(metadata->>'heartbeat_decision', 'keepsleeping') != 'breakout'
                )
                AND (
                    metadata->>'turn_id' IS NULL
                    OR NOT EXISTS (
                        -- turn_id is a per-turn UUID: equality alone identifies the
                        -- turn, and avoiding the continuum_id correlation keeps this
                        -- an index probe (idx_messages_keepsleeping_turn_id) instead
                        -- of a per-row re-scan of messages.
                        SELECT 1 FROM messages other
                        WHERE other.metadata->>'turn_id' = messages.metadata->>'turn_id'
                            AND other.metadata->>'heartbeat' = 'true'
                            AND COALESCE(other.metadata->>'heartbeat_decision', 'keepsleeping') != 'breakout'
                    )
                )
            ORDER BY created_at ASC
        """

        rows = db.execute_query(query, (str(continuum_id), sentinel_time))
        return self._parse_message_rows(rows)

    def load_messages_for_live_compaction_range(
        self,
        continuum_id: str | UUID,
        user_id: str,
        covered_start: datetime,
        covered_end: datetime,
    ) -> list[Message]:
        """
        Load DB-backed ordinary messages for live active-context compaction.

        This query intentionally excludes segment boundaries, system
        notifications, and any compaction synopsis scaffolding. It returns the
        raw persisted messages in chronological order.
        """
        db = self.get_user_db_client(user_id)

        query = """
            SELECT *
            FROM messages
            WHERE continuum_id = %s
              AND user_id = %s
              AND created_at > %s
              AND created_at <= %s
              AND COALESCE(metadata->>'is_segment_boundary', 'false') = 'false'
              AND COALESCE(metadata->>'system_notification', 'false') = 'false'
              AND COALESCE(metadata->>'is_compaction_synopsis', 'false') = 'false'
            ORDER BY created_at ASC, id ASC
        """

        rows = db.execute_query(
            query,
            (str(continuum_id), user_id, covered_start, covered_end),
        )
        return self._parse_message_rows(rows)

    def load_continuity_messages(
        self,
        continuum_id: str | UUID,
        user_id: str,
        turn_count: int
    ) -> list[Message]:
        """
        Load last N user/assistant message pairs from end of most recent collapsed segment.

        Called only during session cache loading after timeout, when all segments are collapsed.
        Provides conversational continuity by showing the tail end of the previous session.
        Excludes keepsleeping heartbeat turns (and their same-turn rows) exactly
        as load_segment_messages does, so they cannot displace genuine
        continuity pairs in the row budget; breakout heartbeats are real
        conversation and stay in context.

        Args:
            continuum_id: Continuum ID
            user_id: User ID
            turn_count: Number of user/assistant pairs to load

        Returns:
            Last N user/assistant message pairs in chronological order
        """
        db = self.get_user_db_client(user_id)

        # Get messages before the most recent collapsed segment's end time.
        # The stored boundary carries an explicit UTC offset; casting it to
        # timestamptz honors that offset. A plain ::timestamp cast strips it,
        # and Postgres then re-interprets the naive value in the session
        # timezone — shifting the cutoff by hours on non-UTC clusters.
        # Request 4x turn_count to ensure we have enough messages to find complete pairs
        query = """
            WITH boundary_time AS (
                SELECT (metadata->>'segment_end_time')::timestamptz as cutoff_time
                FROM messages
                WHERE continuum_id = %s
                    AND metadata->>'is_segment_boundary' = 'true'
                    AND metadata->>'status' = 'collapsed'
                ORDER BY created_at DESC
                LIMIT 1
            )
            SELECT m.* FROM messages m, boundary_time
            WHERE m.continuum_id = %s
                AND m.created_at < boundary_time.cutoff_time
                AND m.role IN ('user', 'assistant')
                AND COALESCE(m.metadata->>'is_segment_boundary', 'false') = 'false'
                AND COALESCE(m.metadata->>'system_notification', 'false') = 'false'
                AND COALESCE(m.metadata->>'has_tool_calls', 'false') != 'true'
                AND NOT (
                    COALESCE(m.metadata->>'heartbeat', 'false') = 'true'
                    AND COALESCE(m.metadata->>'heartbeat_decision', 'keepsleeping') != 'breakout'
                )
                AND (
                    m.metadata->>'turn_id' IS NULL
                    OR NOT EXISTS (
                        -- turn_id is a per-turn UUID: equality alone identifies the
                        -- turn, and avoiding the continuum_id correlation keeps this
                        -- an index probe (idx_messages_keepsleeping_turn_id) instead
                        -- of a per-row re-scan of messages.
                        SELECT 1 FROM messages other
                        WHERE other.metadata->>'turn_id' = m.metadata->>'turn_id'
                            AND other.metadata->>'heartbeat' = 'true'
                            AND COALESCE(other.metadata->>'heartbeat_decision', 'keepsleeping') != 'breakout'
                    )
                )
            ORDER BY m.created_at DESC
            LIMIT %s
        """

        continuum_id_str = str(continuum_id)
        rows = db.execute_query(query, (continuum_id_str, continuum_id_str, turn_count * 4))

        if not rows:
            return []

        # Parse messages (in reverse chronological order from query)
        all_messages = self._parse_message_rows(rows)

        # Work backwards to find last N assistant messages and their corresponding user messages
        pairs = []
        assistant_count = 0
        i = 0

        while i < len(all_messages) and assistant_count < turn_count:
            msg = all_messages[i]

            if msg.role == 'assistant':
                # Found an assistant message, now find its corresponding user message
                assistant_msg = msg
                user_msg = None

                # Look backwards for the user message
                for j in range(i + 1, len(all_messages)):
                    if all_messages[j].role == 'user':
                        user_msg = all_messages[j]
                        break

                # Only add if we found the user message
                if user_msg:
                    pairs.append((user_msg, assistant_msg))
                    assistant_count += 1

            i += 1

        # Reverse pairs to get chronological order (oldest first)
        pairs.reverse()

        # Flatten the pairs into a single list
        messages = []
        for user_msg, assistant_msg in pairs:
            messages.extend([user_msg, assistant_msg])

        return messages
