"""
Extraction orchestrator - high-level extraction workflows.

Two entry points:
- submit_segment_extraction(user_id, boundary_message_id): Self-contained segment
  extraction. Loads messages, extracts directly, marks boundary. Called by collapse handler
  and extract_unprocessed_segments.
- extract_unprocessed_segments(): Safety-net sweep for collapsed segments where
  extraction failed or was never attempted. Runs on a 6-hour schedule.

All complexity lives in submit_segment_extraction. Callers are trivial.
"""
import json
import logging
from typing import Dict, Any
from uuid import UUID

from cns.core.message import Message
from cns.infrastructure.continuum_repository import (
    ContinuumRepository,
    EXTRACTION_MAX_CONTENT_FAILURES,
)
from clients.llm.dialects.base import ProviderError
from clients.llm_provider import ContextOverflowError
from lt_memory.models import ProcessingChunk, MemoryContextSnapshot
from lt_memory.processing.extraction_engine import ExtractionEngine
from lt_memory.processing.execution_strategy import DirectExecutionStrategy
from lt_memory.processing.memory_processor import LLMResponseFormatError
from lt_memory.db_access import LTMemoryDB
from utils.user_context import set_current_user_id, get_current_user_id, clear_user_context

logger = logging.getLogger(__name__)


def _is_content_caused_extraction_failure(error: Exception) -> bool:
    """
    Classify an extraction failure as content-caused (consumes the budget).

    Only failures that are deterministic given the segment's own data count
    against the abandonment budget: a missing boundary row (RuntimeError) or
    a segment with no extractable payload (ValueError). Everything else is
    infra/LLM class and must NOT consume the budget — provider transport
    failures are raised as ProviderError (which subclasses RuntimeError, so
    it is explicitly excluded first) or ContextOverflowError, network/DB
    outages raise arbitrary library exception types, and degenerate model
    output raises LLMResponseFormatError (checked first: it subclasses
    ValueError). Pinned doctrine: retry counters never gate data on
    infrastructure failure.
    """
    if isinstance(error, LLMResponseFormatError):
        return False
    if isinstance(error, (ProviderError, ContextOverflowError)):
        return False
    return isinstance(error, (RuntimeError, ValueError))


class ExtractionOrchestrator:
    """
    High-level extraction workflow coordination.

    Single entry point: submit_segment_extraction owns the full lifecycle
    (load messages, build chunk, submit, mark boundary).

    Delegates to:
    - ExtractionEngine: Build payloads
    - DirectExecutionStrategy: Execute extraction through model_config=batch
    - ContinuumRepository: Load messages, find segments
    - LTMemoryDB: Safety valve checks, memory context loading
    """

    def __init__(
        self,
        extraction_engine: ExtractionEngine,
        execution_strategy: DirectExecutionStrategy,
        continuum_repo: 'ContinuumRepository',
        db: LTMemoryDB,
    ):
        self.extraction_engine = extraction_engine
        self.execution_strategy = execution_strategy
        self.continuum_repo = continuum_repo
        self.db = db

    def submit_segment_extraction(
        self,
        user_id: str,
        boundary_message_id: str,
    ) -> bool:
        """
        Self-contained segment extraction: load, submit, mark.

        Owns the full lifecycle for extracting memories from a single segment:
        1. Query boundary message to get segment_id and position
        2. Load messages between this boundary and the next
        3. Build single ProcessingChunk (no chunking - segments are natural units)
        4. Execute through the fixed background model route
        5. Mark memories_extracted=true on the boundary message

        Args:
            user_id: User ID
            boundary_message_id: UUID string of the segment boundary sentinel message
        Returns:
            True if extraction was submitted successfully

        Raises:
            RuntimeError: If boundary message not found or has no messages
        """
        db_client = self.continuum_repo.get_user_db_client(user_id)

        # Step 1: Query boundary row for segment_id, continuum_id, position
        boundary_row = db_client.execute_query("""
            SELECT id, continuum_id, created_at, metadata
            FROM messages
            WHERE id = %s
        """, (boundary_message_id,))

        if not boundary_row:
            raise RuntimeError(
                f"Boundary message {boundary_message_id} not found for user {user_id}"
            )

        row = boundary_row[0]
        continuum_id = row['continuum_id']
        boundary_time = row['created_at']
        metadata = self._parse_metadata(row.get('metadata', {}))
        segment_id = metadata.get('segment_id', boundary_message_id)
        segment_uuid = UUID(segment_id) if isinstance(segment_id, str) else segment_id

        # Step 2: Load messages after boundary, stop at next boundary.
        # Heartbeat filtering happens in the loop below, not in SQL, so the
        # all-heartbeat case (nothing extractable) can be distinguished from
        # the anomalous empty-segment case and marked as extracted instead of
        # being retried by the sweep forever.
        message_rows = db_client.execute_query("""
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND created_at > %s
                AND (metadata->>'system_notification' IS NULL
                     OR metadata->>'system_notification' = 'false')
            ORDER BY created_at
        """, (str(continuum_id), boundary_time))

        # Turn ids of keepsleeping heartbeat stimuli: every message of those
        # turns is dropped. Breakout turns (heartbeat_decision = 'breakout')
        # are real conversation and stay extractable.
        heartbeat_turn_ids = {
            parsed.get('turn_id')
            for row in message_rows
            for parsed in [self._parse_metadata(row.get('metadata', {}))]
            if parsed.get('heartbeat') == 'true'
            and parsed.get('heartbeat_decision', 'keepsleeping') != 'breakout'
        }
        heartbeat_turn_ids.discard(None)

        messages = []
        for msg_row in message_rows:
            msg_metadata = self._parse_metadata(msg_row.get('metadata', {}))

            # Stop at next segment boundary
            if msg_metadata.get('is_segment_boundary'):
                break

            if (
                msg_metadata.get('heartbeat') == 'true'
                and msg_metadata.get('heartbeat_decision', 'keepsleeping') != 'breakout'
            ) or (
                msg_metadata.get('turn_id')
                and msg_metadata.get('turn_id') in heartbeat_turn_ids
            ):
                continue

            messages.append(Message(
                id=msg_row['id'],
                content=msg_row['content'],
                role=msg_row['role'],
                created_at=msg_row['created_at'],
                metadata=msg_metadata
            ))

        if not message_rows:
            logger.warning(
                f"No messages found for segment {segment_id} "
                f"(boundary: {boundary_message_id})"
            )
            return False

        if not messages:
            # Every message in the segment belonged to keepsleeping heartbeat
            # turns: nothing to extract. Mark the boundary so the sweep does
            # not retry this segment forever.
            logger.info(
                f"Segment {segment_id} contains only keepsleeping heartbeat "
                "turns; marking extracted with no memories"
            )
            db_client.execute_query("""
                UPDATE messages
                SET metadata = jsonb_set(metadata, '{memories_extracted}', 'true')
                WHERE id = %s
            """, (boundary_message_id,))
            return True

        # Step 3: Build single ProcessingChunk (full segment, no chunking)
        chunk = ProcessingChunk.from_conversation_messages(
            messages,
            chunk_index=0,
            segment_id=segment_uuid
        )
        chunk.memory_context_snapshot = self._build_memory_context(messages, user_id)

        # Step 4: Execute directly through model_config=batch.
        extraction_id = self.execution_strategy.execute_extraction(user_id, [chunk])

        # Step 5: Mark boundary as extracted. Reached only when the LLM
        # returned a well-formed response (including a well-formed explicit
        # zero, []); an empty/degenerate response raises LLMResponseFormatError
        # in memory_processor and leaves the segment unextracted for retry.
        db_client.execute_query("""
            UPDATE messages
            SET metadata = jsonb_set(metadata, '{memories_extracted}', 'true')
            WHERE id = %s
        """, (boundary_message_id,))

        logger.info(
            f"Extracted segment {segment_id} "
            f"(execution: {extraction_id}, {len(messages)} messages)"
        )
        return True

    def extract_unprocessed_segments(self, user_id: str = None) -> Dict[str, Any]:
        """
        Safety-net sweep for collapsed segments where extraction failed.

        Finds all collapsed segments with memories_extracted != true and
        submits each via submit_segment_extraction. Per-segment error isolation
        ensures one bad segment doesn't block the rest.

        Failure accounting: only CONTENT-caused failures (deterministic given
        the segment's own data) consume the abandonment budget — a segment is
        marked 'extraction_abandoned' and excluded from future sweeps once its
        extraction_content_failures count reaches
        EXTRACTION_MAX_CONTENT_FAILURES. Infra/LLM outages (provider blips,
        degenerate responses) are not counted, so they can never permanently
        silence a segment.

        Args:
            user_id: Optional specific user. If None, processes all users.

        Returns:
            Extraction statistics
        """
        logger.info(
            f"Starting unprocessed segment extraction sweep"
            f"{f' for user {user_id}' if user_id else ''}"
        )

        users = (
            [{"id": user_id}] if user_id
            else self.db.get_users_with_memory_enabled()
        )
        results = {"segments_submitted": 0, "users_processed": 0, "errors": []}

        for user in users:
            uid = str(user["id"])
            try:
                # Find collapsed segments needing extraction
                failed_segments = self.continuum_repo.find_failed_extraction_segments(uid)
                if not failed_segments:
                    continue

                logger.info(f"Found {len(failed_segments)} unprocessed segments for user {uid}")
                set_current_user_id(uid)

                for segment in failed_segments:
                    # Total attempt counter (observability only — it does NOT
                    # gate the retry budget). Persists via jsonb_set.
                    attempts = segment.get('extraction_attempts', 0)
                    content_failures = segment.get('extraction_content_failures', 0)
                    db_client = self.continuum_repo.get_user_db_client(uid)
                    db_client.execute_returning("""
                        UPDATE messages
                        SET metadata = jsonb_set(metadata, '{extraction_attempts}', to_jsonb(%s))
                        WHERE id = %s
                            AND metadata->>'is_segment_boundary' = 'true'
                        RETURNING id
                    """, (attempts + 1, segment['message_id']))

                    try:
                        if self.submit_segment_extraction(uid, segment['message_id']):
                            results["segments_submitted"] += 1
                    except Exception as e:
                        logger.error(
                            f"Error extracting segment {segment.get('segment_id', '?')} "
                            f"for user {uid} (attempt {attempts + 1}): {e}",
                            exc_info=True
                        )
                        results["errors"].append(str(e))

                        # Retry budget: only content-caused failures may abandon
                        # a segment. Infra/LLM-class failures leave the budget
                        # untouched so the segment is retried on the next sweep.
                        if not _is_content_caused_extraction_failure(e):
                            logger.info(
                                f"Infra/LLM-class extraction failure for segment "
                                f"{segment.get('segment_id', '?')}; not counted toward "
                                f"the abandonment budget"
                            )
                            continue

                        new_content_failures = content_failures + 1
                        db_client.execute_returning("""
                            UPDATE messages
                            SET metadata = jsonb_set(metadata, '{extraction_content_failures}', to_jsonb(%s))
                            WHERE id = %s
                                AND metadata->>'is_segment_boundary' = 'true'
                            RETURNING id
                        """, (new_content_failures, segment['message_id']))

                        if new_content_failures >= EXTRACTION_MAX_CONTENT_FAILURES:
                            # Abandonment marker: budget-excluded segments stay
                            # visible to operators instead of silently vanishing
                            # from the sweep.
                            db_client.execute_returning("""
                                UPDATE messages
                                SET metadata = jsonb_set(metadata, '{extraction_abandoned}', 'true')
                                WHERE id = %s
                                    AND metadata->>'is_segment_boundary' = 'true'
                                RETURNING id
                            """, (segment['message_id'],))
                            logger.warning(
                                f"Segment {segment.get('segment_id', '?')} for user {uid} "
                                f"abandoned after {new_content_failures} content-caused "
                                f"extraction failures (extraction_abandoned=true; excluded "
                                f"from future extraction sweeps)"
                            )

                results["users_processed"] += 1

            except Exception as e:
                logger.error(f"Error processing user {uid}: {e}", exc_info=True)
                results["errors"].append(str(e))
            finally:
                try:
                    if get_current_user_id() == uid:
                        clear_user_context()
                except Exception:
                    pass

        logger.info(
            f"Unprocessed segment sweep complete: "
            f"{results['segments_submitted']} segments submitted"
        )
        return results

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _build_memory_context(
        self,
        messages: list[Message],
        user_id: str
    ) -> MemoryContextSnapshot:
        """
        Build memory context from referenced and pinned memories.

        Loads memory texts from database for all referenced memory IDs.
        Also collects pinned memory IDs (8-char) for importance boosting.

        Args:
            messages: Chunk messages
            user_id: User ID

        Returns:
            Memory context dict with:
            - memory_ids: Full UUIDs of referenced memories
            - referenced_memory_ids: Sorted full UUIDs (for extraction context)
            - memory_texts: Dict of {uuid: text}
            - pinned_short_ids: Deduplicated 8-char IDs for importance boost
        """
        referenced_ids = set()
        pinned_short_ids = set()

        for msg in messages:
            metadata = getattr(msg, "metadata", {}) or {}

            # Extract referenced memories (explicit LLM references)
            if isinstance(metadata.get("referenced_memories"), list):
                for ref in metadata["referenced_memories"]:
                    if isinstance(ref, str):
                        referenced_ids.add(ref)

            # Extract pinned memory IDs (8-char short IDs from retention)
            if isinstance(metadata.get("pinned_memory_ids"), list):
                for pin_id in metadata["pinned_memory_ids"]:
                    if isinstance(pin_id, str) and pin_id:
                        pinned_short_ids.add(pin_id.lower())

        # Load memory texts from database
        memory_texts = {}
        if referenced_ids:
            memory_uuids = [UUID(ref_id) for ref_id in referenced_ids]
            memories = self.db.get_memories_by_ids(memory_uuids, user_id=user_id)
            for mem in memories:
                memory_texts[str(mem.id)] = mem.text

            logger.debug(f"Loaded {len(memories)} referenced memories for context")

        if pinned_short_ids:
            logger.debug(f"Collected {len(pinned_short_ids)} unique pinned memory IDs")

        return {
            "memory_ids": list(referenced_ids),
            "referenced_memory_ids": sorted(referenced_ids),
            "memory_texts": memory_texts,
            "pinned_short_ids": sorted(pinned_short_ids),
        }

    @staticmethod
    def _parse_metadata(raw_metadata) -> Dict[str, Any]:
        """Parse message metadata from various formats."""
        if isinstance(raw_metadata, str):
            try:
                return json.loads(raw_metadata) if raw_metadata else {}
            except json.JSONDecodeError:
                return {}
        return dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
