"""
Segment collapse handler for processing timeout events.

Handles SessionTimeoutEvent by:
1. Finding the segment boundary sentinel
2. Loading segment messages
3. Generating summary with embedding (Continuity Engine)
4. Updating sentinel metadata
5. Triggering downstream processing (memory extraction, domain updates)
6. Extracting feedback signals (DIY reinforcement loop)
7. Running pattern synthesis if use-day threshold reached (every 7 use-days)
8. Portrait synthesis if use-day threshold reached (every 10 use-days)
9. Evaluating MIRA against the behavioral contract and refining Persona (parallel
   to 6-7 by D1: the user model describes the user, Persona prescribes to MIRA)
"""
from __future__ import annotations

import json
import logging
import threading
from contextvars import copy_context
from typing import List, Optional, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from datetime import datetime

    from lt_memory.factory import LTMemoryFactory
    from lt_memory.models import Memory
    from lt_memory.db_access import LTMemoryDB
    from cns.infrastructure.continuum_pool import ContinuumPool
    from tools.repo import ToolRepository

from cns.core.events import (
    SegmentTimeoutEvent, SegmentCollapsedEvent, ManifestUpdatedEvent, TurnCompletedEvent,
)
from cns.core.message import Message
from cns.services.segment_helpers import collapse_segment_sentinel
from cns.services.summary_generator import SummaryGenerator, SummaryResult, SummaryType
from cns.infrastructure.continuum_repository import (
    ContinuumRepository,
    COLLAPSE_CLAIM_STALE_MINUTES,
)
from clients.embeddings_provider import EmbeddingsProvider
from clients.valkey_client import get_valkey_client
from cns.integration.event_bus import EventBus
from utils.timezone_utils import utc_now, format_utc_iso, parse_time_string, ensure_utc, validate_timezone
from utils.user_context import (
    set_current_user_id, get_current_user_id, clear_user_context, get_user_preferences,
)

logger = logging.getLogger(__name__)

# Maximum collapse attempts before tombstoning a segment.
# Prevents infinite retry loops when a persistent failure (billing, DB schema,
# missing config) causes every attempt to fail after the expensive LLM call.
# The counter deliberately counts infrastructure failures too — the
# transient-vs-persistent distinction is intentionally not made (accepted
# decision 2026-09-20).
MAX_COLLAPSE_ATTEMPTS = 3

# Pending manual collapse. A manual collapse requested while the user's
# turn is in flight must not run mid-turn — turn messages commit only at turn
# end, and the collapse recheck's last_turn_at is stamped at message arrival,
# so the interleaving turn is invisible to it and would be orphaned from the
# digest. The actions API entry defers instead: it writes this single flag and
# returns; the completing turn's TurnCompletedEvent consumes it (see
# SegmentCollapseHandler._handle_turn_completed). No timers, no polling loops
# — the turn's own completion event is the trigger. The TTL only bounds a
# flag whose turn died without completing (crash, halt): a stale flag is
# harmless — the claim/not-found refusal in collapse_segment makes the late
# run a no-op, and the scheduled timeout path collapses the segment anyway.
PENDING_MANUAL_COLLAPSE_KEY = "pending_manual_collapse:{user_id}"
PENDING_MANUAL_COLLAPSE_TTL_SECONDS = 24 * 3600

# Pending manual-memory drain. The Valkey queue written by
# memory_tool.create_memory is the sole durable record of a user-confirmed
# memory, so an item counts as consumed only once a per-pending_id done
# marker is durably set — never via a blanket queue delete before the work
# (store_memories is NOT idempotent; the marker is what prevents a re-drain
# from double-storing, mirroring the keyed-idempotence shape of the
# extract_unprocessed_segments retry sweep).
PENDING_DONE_MARKER_TTL_SECONDS = 14 * 86400  # outlives the queue's refreshed TTL below
PENDING_QUEUE_RETRY_TTL_SECONDS = 7 * 86400   # keeps failing items retryable past the producer's 24h TTL
PENDING_ITEM_MAX_ATTEMPTS = 3                # dead-letter bound: a permanently-failing item must not wedge the queue


def _resolve_pending_memory_timezone(segment_id: str) -> str:
    """Timezone to read pending manual memory temporal fields in.

    `memory_tool.create_memory` queues `happens_at`/`expires_at` verbatim, so they
    arrive here as the model's naive local wall time and must be resolved against
    the user's timezone rather than the system default (UTC). Called once per
    collapse, never per field or per memory.

    Falls back to UTC when preferences cannot be read. This is a background
    durability path, not a request path: `get_user_preferences()` needs a user
    context plus Valkey and Postgres, and letting any of those raise would discard
    every pending memory in the segment. `validate_timezone` runs inside the guard
    for the same reason — a single malformed `users.timezone` row would otherwise
    make every parse in the batch fail and strip every temporal anchor.
    """
    try:
        return validate_timezone(get_user_preferences().timezone)
    except Exception:
        logger.warning(
            "User preferences unavailable while collapsing segment %s; "
            "reading pending manual memory times as UTC",
            segment_id,
            exc_info=True,
        )
        return "UTC"


def _parse_pending_temporal_field(
    value: str,
    field: str,
    pending_id: str,
    tz_name: str,
) -> Optional[datetime]:
    """Parse one temporal field of a pending manual memory, or return None.

    Returns a UTC-aware datetime -- the representation `lt_memory` and the
    `timestamptz` columns work in -- so the attributed instant never depends on a
    consumer knowing which zone the parse happened in.

    Returns None instead of raising: by collapse time the model is off the call
    stack, so an exception loses the memory outright rather than prompting a
    clarifying question, and one bad timestamp must not fail the whole segment.
    That trade-off is only defensible if the log carries what the silent drop
    lacked — the raw value, the timezone it was read against, and the parser's
    reason — so the anchor's loss is diagnosable and repairable by hand.
    """
    try:
        return ensure_utc(parse_time_string(value, tz_name=tz_name))
    except Exception as exc:
        logger.warning(
            "Unparseable %s on pending manual memory %s: value=%r, timezone=%s. "
            "Storing the memory without that temporal field -- text is preserved, but "
            "relevance scoring and decay will treat this record as untimed. "
            "Parser said: %s: %s",
            field,
            pending_id,
            value,
            tz_name,
            type(exc).__name__,
            exc,
        )
        return None


class SegmentCollapseHandler:
    """
    Handles segment collapse when timeout is reached.

    Subscribes to SessionTimeoutEvent and orchestrates the collapse pipeline:
    summary generation, embedding, sentinel update, downstream processing,
    and the DIY reinforcement loop (feedback extraction + pattern synthesis).
    """

    def __init__(
        self,
        continuum_repo: ContinuumRepository,
        summary_generator: SummaryGenerator,
        embeddings_provider: EmbeddingsProvider,
        event_bus: EventBus,
        continuum_pool: ContinuumPool,
        lt_memory_factory: LTMemoryFactory,
        tool_repo: 'ToolRepository',
        persona_enabled: bool,
    ):
        """
        Initialize collapse handler.

        Args:
            continuum_repo: Repository for loading/saving messages
            summary_generator: Service for generating segment summaries
            embeddings_provider: Provider for generating embeddings
            event_bus: Event bus for publishing events
            continuum_pool: Continuum pool for cache invalidation
            lt_memory_factory: LT_Memory factory for extraction
            tool_repo: Tool repository for spawning the MemoryCuratorAgent
            persona_enabled: Whether the Persona pass runs on collapse. Supplied by
                the factory from MIRA_PERSONA_ENABLED, so the switch decides
                construction here and the collapse path never re-reads config.
        """
        self.continuum_repo = continuum_repo
        self.summary_generator = summary_generator
        self.embeddings_provider = embeddings_provider
        self.event_bus = event_bus
        self.continuum_pool = continuum_pool
        self.lt_memory_factory = lt_memory_factory
        self.tool_repo = tool_repo

        # Late-register the integration-curator spawn callback on the
        # lt_memory factory. The factory was created before tool_repo existed,
        # so this is the seam: lt_memory storage calls on_memories_stored();
        # this handler (which owns tool_repo + event_bus) spawns the agent.
        # lt_memory never imports from agents/ — the callback is the boundary.
        try:
            self.lt_memory_factory.on_memories_stored = self._on_memories_stored
        except Exception:
            logger.warning("Could not register on_memories_stored callback", exc_info=True)

        # Initialize feedback loop components (lazy - may not be available yet)
        self._assessment_extractor = None
        self._feedback_repo = None
        self._feedback_tracker = None
        self._synthesizer = None
        self._feedback_loop_initialized = False

        # Persona is a second, parallel system. Its service is built lazily like
        # the feedback-loop components, but unlike them a disabled Persona is omitted
        # at construction by the factory rather than checked per collapse.
        self._persona_enabled = persona_enabled
        self._persona_service = None

        # Subscribe to timeout events
        self.event_bus.subscribe('SegmentTimeoutEvent', self.handle_timeout)
        logger.info("SegmentCollapseHandler subscribed to SegmentTimeoutEvent")

        # Consume a queued manual collapse at the turn's completion —
        # see _handle_turn_completed.
        self.event_bus.subscribe('TurnCompletedEvent', self._handle_turn_completed)
        logger.info("SegmentCollapseHandler subscribed to TurnCompletedEvent")

    def _init_feedback_loop(self) -> bool:
        """
        Lazy initialization of feedback loop components.

        Returns True if initialization successful, False otherwise.
        Retries on each call until successful — transient failures (missing
        prompt files, import issues) don't permanently kill the pipeline.
        """
        if self._feedback_loop_initialized:
            return True

        try:
            from cns.services.assessment_extractor import AssessmentExtractor
            from cns.infrastructure.feedback_repository import FeedbackRepository
            from cns.infrastructure.feedback_tracker import FeedbackTracker
            from cns.services.user_model_synthesizer import UserModelSynthesizer

            self._assessment_extractor = AssessmentExtractor()
            self._feedback_repo = FeedbackRepository()
            self._feedback_tracker = FeedbackTracker()
            self._synthesizer = UserModelSynthesizer(self._feedback_repo)

            self._feedback_loop_initialized = True
            logger.info("Feedback loop components initialized successfully")
            return True

        except Exception as e:
            logger.error(
                "USER MODEL PIPELINE INIT FAILED: %s — "
                "No assessment signals or synthesis will run until this is fixed. "
                "Will retry on next segment collapse.",
                e, exc_info=True
            )
            return False

    def _get_persona_service(self):
        """Build the Persona service on first use, like the feedback-loop components."""
        if self._persona_service is None:
            from cns.services.persona_service import PersonaService

            self._persona_service = PersonaService()
        return self._persona_service

    def handle_timeout(self, event: SegmentTimeoutEvent) -> None:
        """
        Event subscriber wrapper for segment timeout.

        Catches all exceptions so the event bus never sees failures.
        Segment remains active on failure and retries on the next timeout check.

        Args:
            event: SegmentTimeoutEvent with segment details
        """
        try:
            self.collapse_segment(event)
        except Exception as e:
            logger.error(
                f"COLLAPSE FAILURE: Segment {event.segment_id} collapse did not complete. "
                f"If the failure happened before the sentinel was saved, the segment "
                f"remains active and will retry on the next timeout check. A sentinel "
                f"already saved 'collapsed' is never re-matched; its downstream failures "
                f"(pending-memory drain, extraction, feedback/portrait passes) are logged "
                f"separately as a CRITICAL 'downstream processing failed' and flagged "
                f"'downstream_failed' on the sentinel for retry by the extraction sweep "
                f"and the next pending-memory drain. Operators should investigate if "
                f"this persists. Error: {e}",
                exc_info=True
            )

    def queue_manual_collapse(self, user_id: str, continuum_id: str, segment_id: str) -> None:
        """
        Queue a manual collapse behind the user's in-flight turn.

        Called by the actions API entry when the per-user turn lock is held:
        collapsing mid-turn orphans that turn from the digest (turn messages
        commit only at turn end, and the collapse recheck's last_turn_at is
        stamped at message arrival, so it cannot see the interleaving turn).
        The single flag written here is consumed by _handle_turn_completed at
        the next TurnCompletedEvent — no timers, no polling: the completing
        turn itself triggers the deferred collapse.

        Args:
            user_id: User whose segment to collapse
            continuum_id: Continuum UUID (string, captured at request time)
            segment_id: Segment UUID (string, captured at request time)
        """
        get_valkey_client().set(
            PENDING_MANUAL_COLLAPSE_KEY.format(user_id=user_id),
            json.dumps({
                "user_id": user_id,
                "continuum_id": continuum_id,
                "segment_id": segment_id,
            }),
            ex=PENDING_MANUAL_COLLAPSE_TTL_SECONDS,
        )
        logger.info(
            "Manual collapse for segment %s queued behind the in-flight turn; "
            "it will run when the turn completes",
            segment_id,
        )

    def _handle_turn_completed(self, event: TurnCompletedEvent) -> None:
        """
        TurnCompletedEvent subscriber: run a queued manual collapse, if any.

        Fires from the turn's post-commit callback, so the completing turn's
        messages are durably in the segment — collapsing here includes them,
        which is the whole point of the queue. The collapse itself runs on its
        own thread (canonical copy-context pattern, cns/services/tool_loop.py):
        this subscriber is synchronous inside the reply's commit, and the
        collapse's LLM summary must not stall the reply the user is waiting
        for. A user can hold only one turn at a time (the same lock the entry
        probes), so flag consumes cannot pile up; even a racing double
        consume is a refused no-op under the claim mutual exclusion.
        """
        try:
            valkey = get_valkey_client()
            key = PENDING_MANUAL_COLLAPSE_KEY.format(user_id=event.user_id)
            raw = valkey.get(key)
            if raw is None:
                return
            valkey.delete(key)
            payload = json.loads(raw)
        except Exception:
            logger.error(
                "Could not read the queued manual collapse for user %s; the "
                "scheduled timeout path still collapses the segment",
                event.user_id, exc_info=True,
            )
            return

        context = copy_context()
        threading.Thread(
            target=context.run,
            args=(self._run_queued_manual_collapse, payload),
            name=f"queued-collapse:{payload.get('segment_id')}",
            daemon=True,
        ).start()

    def _run_queued_manual_collapse(self, payload: dict) -> None:
        """Run one queued manual collapse; a failure logs and leaves the segment active."""
        event = SegmentTimeoutEvent.create(
            continuum_id=payload["continuum_id"],
            user_id=payload["user_id"],
            segment_id=payload["segment_id"],
            inactive_duration_minutes=0,  # Manual trigger (mirrors the actions API entry)
            local_hour=utc_now().hour,
        )
        try:
            self.collapse_segment(event)
        except Exception:
            logger.error(
                "Queued manual collapse for segment %s did not complete; the "
                "segment stays active and the scheduled timeout path retries",
                payload.get("segment_id"), exc_info=True,
            )

    def _claim_collapse(self, sentinel: Message, attempts: int) -> bool:
        """Atomically claim the collapse for this handler via mutual exclusion.

        The UPDATE only lands while the sentinel is still 'active'/'paused' —
        or while a previous 'collapsing' claim has gone stale (older than
        COLLAPSE_CLAIM_STALE_MINUTES), which is the crash-recovery path: a
        process death between claim and save must not strand the segment.
        On success the row flips to 'collapsing' with collapse_claimed_at
        stamped; every concurrent claimant (scheduled timeout vs manual
        collapse, or a racing tombstone) gets no rows and must abort.
        """
        db = self.continuum_repo.get_user_db_client(get_current_user_id())
        claimed = db.execute_returning("""
            UPDATE messages
            SET metadata = metadata || jsonb_build_object(
                'status', 'collapsing',
                'collapse_attempts', %s,
                'collapse_claimed_at', %s::text
            )
            WHERE id = %s
                AND metadata->>'is_segment_boundary' = 'true'
                AND (
                    metadata->>'status' IN ('active', 'paused')
                    OR (
                        metadata->>'status' = 'collapsing'
                        AND (metadata->>'collapse_claimed_at')::timestamptz
                            < now() - make_interval(mins => %s)
                    )
                )
            RETURNING id
        """, (
            attempts + 1,
            format_utc_iso(utc_now()),
            str(sentinel.id),
            COLLAPSE_CLAIM_STALE_MINUTES,
        ))
        return bool(claimed)

    def collapse_segment(
        self,
        event: SegmentTimeoutEvent,
    ) -> Message:
        """
        Collapse a segment: generate summary, update sentinel, trigger downstream.

        Called by handle_timeout (event subscriber, swallows exceptions) and by the
        actions API (needs exceptions to propagate and the collapsed sentinel returned).

        Args:
            event: SegmentTimeoutEvent with segment details

        Returns:
            Collapsed sentinel Message

        Raises:
            RuntimeError: On any collapse failure (sentinel not found, no messages,
                summary generation failed, etc.)
        """
        # Clear stale per-user cached fields (e.g. cumulative_activity_days)
        # before setting new identity — required when the timeout service
        # processes multiple users sequentially on the same thread.
        clear_user_context()
        set_current_user_id(event.user_id)

        logger.info(
            f"Processing timeout for segment {event.segment_id}, "
            f"continuum {event.continuum_id}, "
            f"inactive_duration={event.inactive_duration_minutes}min, "
            f"local_hour={event.local_hour}"
        )

        # Find the segment boundary sentinel
        sentinel = self._find_segment_sentinel(
            event.continuum_id,
            event.segment_id
        )

        if not sentinel:
            raise RuntimeError(
                f"Segment sentinel {event.segment_id} not found. "
                f"Data consistency violation - timeout event published for non-existent segment."
            )

        # Circuit breaker: stop retrying after MAX_COLLAPSE_ATTEMPTS
        attempts = sentinel.metadata.get('collapse_attempts', 0)
        if attempts >= MAX_COLLAPSE_ATTEMPTS:
            logger.critical(
                "Segment %s has failed %d collapse attempts — tombstoning to stop retry loop. "
                "Check previous COLLAPSE FAILURE logs for root cause.",
                event.segment_id, MAX_COLLAPSE_ATTEMPTS
            )
            # The tombstone save goes through the same mutual-exclusion claim as
            # the normal path: without it, this branch could overwrite a sentinel
            # a concurrent winner already collapsed (duplicate events, lost
            # summary). No claim means the segment is already collapsed or
            # actively being collapsed — return the current sentinel untouched.
            if not self._claim_collapse(sentinel, attempts):
                logger.warning(
                    "Tombstone for segment %s refused: collapse already claimed elsewhere",
                    event.segment_id
                )
                return sentinel
            sentinel.metadata['collapse_attempts'] = attempts + 1

            # Force-collapse with tombstone so segment exits timeout queue
            tombstone = collapse_segment_sentinel(
                sentinel,
                summary="[Segment collapse failed after maximum retry attempts]",
                precis="[Collapse Failed]",
                display_title="[Collapse Failed]",
                embedding=[0.0] * self.embeddings_provider.dimensions,
                inactive_duration_minutes=event.inactive_duration_minutes,
                processing_failed=True,
                tools_used=sentinel.metadata.get('tools_used', []),
                segment_end_time=utc_now(),
                complexity_score=0.0
            )
            user_id = get_current_user_id()
            self.continuum_repo.save_message(tombstone, event.continuum_id, user_id)
            self.continuum_pool.invalidate()

            # A tombstone is a real collapsed segment as far as every
            # subscriber is concerned — publish the same SegmentCollapsedEvent
            # the normal path publishes. Without it, poller threads keep
            # polling the dead segment and stateful trinkets carry stale
            # snapshots into the next segment.
            self.event_bus.publish(SegmentCollapsedEvent.create(
                continuum_id=event.continuum_id,
                segment_id=event.segment_id,
                summary=tombstone.content,
                tools_used=tombstone.metadata.get('tools_used', []),
            ))

            # A tombstoned segment never reaches the downstream stage, but its
            # pending manual memories are still the user's confirmed data and
            # must not be stranded on the queue to be deleted by TTL.
            # The drain is idempotent per item, so a best-effort call here is
            # safe; on failure the items stay queued for the next drain rather
            # than being destroyed.
            try:
                self._process_pending_manual_memories(user_id, event.segment_id)
            except Exception:
                logger.error(
                    "Pending manual memory drain failed for tombstoned segment %s; "
                    "items remain queued for retry on the next drain",
                    event.segment_id, exc_info=True,
                )

            # The manifest publish runs on the tombstone path too, mirroring
            # the normal path — without it the manifest TTL cache keeps showing
            # the segment active until the cache expires.
            self.event_bus.publish(ManifestUpdatedEvent.create(
                continuum_id=event.continuum_id,
                segment_count=self._count_user_segments()
            ))
            return tombstone

        # Load messages in segment (between this sentinel and next, or end of continuum).
        # Runs before the claim: a still-processing turn gets the cheap "not yet"
        # answer without taking the mutual-exclusion claim.
        messages = self._load_segment_messages(
            event.continuum_id,
            sentinel
        )

        if not messages:
            # An empty active segment is unfixable by retry, not a transient:
            # the sentinel is persisted at commit time together with its first
            # messages, so a sentinel with zero messages means a non-atomic
            # save lost them (e.g. sentinel committed, batch rolled back).
            # Flow this shape through the same attempt budget the circuit
            # breaker reads (the counter stamp _claim_collapse performs) so
            # after MAX_COLLAPSE_ATTEMPTS sweeps the breaker tombstones the
            # orphan instead of the sweep retrying it forever. Status stays
            # active/paused (no claim): if a message does land the segment
            # heals and collapses normally on a later pass, and a concurrent
            # claimant's own stamp is not double-counted.
            db = self.continuum_repo.get_user_db_client(get_current_user_id())
            db.execute_returning("""
                UPDATE messages
                SET metadata = jsonb_set(metadata, '{collapse_attempts}', to_jsonb(%s))
                WHERE id = %s
                    AND metadata->>'is_segment_boundary' = 'true'
                    AND metadata->>'status' IN ('active', 'paused')
                RETURNING id
            """, (attempts + 1, str(sentinel.id)))
            raise RuntimeError(
                f"Segment {event.segment_id} has no committed messages. A "
                f"sentinel persisted without its messages (non-atomic save) can "
                f"never be summarized on retry; collapse attempt {attempts + 1} of "
                f"{MAX_COLLAPSE_ATTEMPTS} was counted and the circuit breaker will "
                f"tombstone the segment if it stays empty."
            )

        # Claim the collapse via mutual exclusion (see _claim_collapse): the
        # UPDATE flips status to 'collapsing' and only lands while the sentinel
        # is active/paused or its previous claim has gone stale; a concurrent
        # claimant — manual collapse, scheduled timeout, or a racing tombstone —
        # gets no rows and must abort.
        if not self._claim_collapse(sentinel, attempts):
            raise RuntimeError("Segment no longer active; collapse already claimed")

        # Generate summary and embedding (raises on failure)
        result, embedding = self._generate_summary(
            messages,
            sentinel,
            event.continuum_id
        )

        # Evaluate MIRA behavior and refine Persona on the use-day cadence.
        # Runs BEFORE the sentinel save so its failures propagate to the
        # retry/circuit-breaker path: a persistent Persona breakage counts
        # toward MAX_COLLAPSE_ATTEMPTS and surfaces as the force-tombstone,
        # instead of stranding a collapsed segment with its persona signal
        # silently lost (the persona evaluator itself returned empty on the
        # 06:57 and 12:32 collapses). evaluate_segment is idempotent per
        # segment (segment_was_evaluated skip-guard), so a retry after a
        # successful pass re-runs nothing.
        # Runs beside the user-model loop below, not instead of it, and unlike
        # it failures propagate: see _process_persona's docstring.
        if self._persona_enabled:
            self._process_persona(
                messages=messages,
                segment_id=UUID(event.segment_id),
                continuum_id=UUID(event.continuum_id),
            )

        # Extract tools used from actual messages (not sentinel metadata)
        tools_used = self._extract_tools_from_messages(messages)

        # Set segment_end_time from last message (guaranteed to exist at this point)
        segment_end_time = messages[-1].created_at

        # Collapse sentinel (returns new Message with collapsed state)
        collapsed_sentinel = collapse_segment_sentinel(
            sentinel,
            summary=result.synopsis,
            precis=result.precis,
            display_title=result.display_title,
            embedding=embedding,
            inactive_duration_minutes=event.inactive_duration_minutes,
            processing_failed=False,  # Always False - failures raise instead of degrading
            tools_used=tools_used,
            segment_end_time=segment_end_time,
            complexity_score=result.complexity
        )

        user_id = get_current_user_id()

        # Re-read the sentinel: if a turn interleaved since entry (last_turn_at
        # moved) or the segment left active/paused (another collapse claimed it),
        # abort before overwriting — the segment stays active and retries on the
        # next timeout check; only this attempt's summary work is discarded.
        current = self.continuum_repo.find_segment_by_id(
            event.continuum_id, event.segment_id, user_id
        )
        if (
            current is None
            or current.metadata.get('status') not in ('active', 'paused', 'collapsing')
            or current.metadata.get('last_turn_at') != sentinel.metadata.get('last_turn_at')
        ):
            raise RuntimeError(
                f"Segment {event.segment_id} changed during collapse "
                f"(turn interleaved or status no longer active/paused); "
                f"aborting save. Segment remains active."
            )

        self.continuum_repo.save_message(
            collapsed_sentinel,
            event.continuum_id,
            user_id
        )

        # Invalidate Valkey cache to force reload with collapsed sentinel
        self.continuum_pool.invalidate()

        # Publish collapsed event
        self.event_bus.publish(SegmentCollapsedEvent.create(
            continuum_id=event.continuum_id,
            segment_id=event.segment_id,
            summary=result.synopsis,
            tools_used=tools_used
        ))

        # Downstream processing is deliberately separated from the collapse
        # itself: the sentinel is already saved 'collapsed', so a
        # downstream failure must not unwind as though the collapse had
        # failed. The pre-repair path skipped the pending-memory drain, the
        # feedback loop, portrait synthesis and the manifest publish, and
        # nothing ever re-matched a collapsed segment — so the queue was left
        # to be deleted by TTL. Here the failure is recorded on the sentinel
        # and the collapse still completes. The pending-memory drain is
        # idempotent per item (see _process_pending_manual_memories), so the
        # next collapse's rescue sweep re-drains stranded items without
        # double-storing; extraction retries stay with the 6-hour
        # extract_unprocessed_segments sweep — no second extraction run is
        # submitted here.
        try:
            self._trigger_downstream_processing(
                event.continuum_id,
                event.segment_id,
                collapsed_sentinel,
                messages,
                result.synopsis,
            )

            # DIY Reinforcement Loop: Extract feedback and run synthesis if due
            self._process_feedback_loop(
                messages=messages,
                segment_id=UUID(event.segment_id),
                continuum_id=UUID(event.continuum_id),
            )

            # Portrait synthesis if use-day threshold reached
            self._process_portrait_synthesis()
        except Exception:
            logger.critical(
                "Segment %s collapsed successfully, but downstream processing "
                "failed; the collapse stands. Pending manual memories are left "
                "queued (they retry on the next drain without double-storing), "
                "and the sentinel is flagged 'downstream_failed' for operators. "
                "Extraction itself is retried by the extract_unprocessed_segments "
                "sweep.",
                event.segment_id, exc_info=True,
            )
            collapsed_sentinel.metadata['downstream_failed'] = True
            collapsed_sentinel.metadata['downstream_failed_at'] = format_utc_iso(utc_now())
            try:
                self.continuum_repo.save_message(
                    collapsed_sentinel,
                    event.continuum_id,
                    user_id
                )
                self.continuum_pool.invalidate()
            except Exception:
                logger.critical(
                    "Could not record the downstream_failed flag on sentinel %s; "
                    "the collapse itself is durable — check the CRITICAL above "
                    "for what was skipped",
                    event.segment_id, exc_info=True,
                )
        finally:
            # The manifest publish runs on both paths — the pre-repair unwind
            # skipped it and left the manifest TTL cache stale.
            self.event_bus.publish(ManifestUpdatedEvent.create(
                continuum_id=event.continuum_id,
                segment_count=self._count_user_segments()
            ))

        if collapsed_sentinel.metadata.get('downstream_failed'):
            logger.info(
                "Collapsed segment %s with downstream failures flagged on the sentinel",
                event.segment_id
            )
        else:
            logger.info(f"Successfully collapsed segment {event.segment_id}")
        return collapsed_sentinel

    def _find_segment_sentinel(
        self,
        continuum_id: str,
        segment_id: str
    ) -> Optional[Message]:
        """
        Find segment boundary sentinel by segment_id.

        Requires: Active user context (set via set_current_user_id at handler entry)

        Args:
            continuum_id: Continuum UUID
            segment_id: Segment UUID from sentinel metadata

        Returns:
            Sentinel message or None if not found
        """
        user_id = get_current_user_id()
        return self.continuum_repo.find_segment_by_id(continuum_id, segment_id, user_id)

    def _load_segment_messages(
        self,
        continuum_id: str,
        sentinel: Message
    ) -> List[Message]:
        """
        Load messages belonging to this segment.

        Messages are from this sentinel to next sentinel (exclusive) or end of continuum,
        excluding session boundaries and summaries.

        NOTE: This method uses direct database access (encapsulation violation) to implement
        defensive "stop at next boundary" logic. While system constraints ensure only one
        active segment exists at a time (making this check theoretically unnecessary), the
        boundary check provides protection against:
        - Future race conditions in segment creation
        - Data inconsistencies from manual database operations
        - Changes to segment lifecycle management

        This defensive programming is accepted technical debt - the encapsulation violation
        is acknowledged but deemed acceptable for this single-use case with defensive value.

        Requires: Active user context (set via set_current_user_id at handler entry)

        Args:
            continuum_id: Continuum UUID
            sentinel: Segment boundary sentinel

        Returns:
            List of messages in segment
        """
        user_id = get_current_user_id()
        db = self.continuum_repo.get_user_db_client(user_id)

        # Load messages after sentinel timestamp, excluding boundaries/system
        # notifications and keepsleeping heartbeat turns (breakout heartbeat
        # turns are real conversation and are summarized like any other).
        query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND created_at > %s
                AND COALESCE(metadata->>'is_segment_boundary', 'false') != 'true'
                AND COALESCE(metadata->>'system_notification', 'false') != 'true'
                AND NOT (
                    COALESCE(metadata->>'heartbeat', 'false') = 'true'
                    AND COALESCE(metadata->>'heartbeat_decision', 'keepsleeping') != 'breakout'
                )
                AND (
                    metadata->>'turn_id' IS NULL
                    OR metadata->>'turn_id' NOT IN (
                        SELECT other.metadata->>'turn_id'
                        FROM messages other
                        WHERE other.continuum_id = messages.continuum_id
                            AND other.metadata->>'heartbeat' = 'true'
                            AND COALESCE(other.metadata->>'heartbeat_decision', 'keepsleeping') != 'breakout'
                            AND other.metadata->>'turn_id' IS NOT NULL
                    )
                )
            ORDER BY created_at ASC
        """

        rows = db.execute_query(query, (continuum_id, sentinel.created_at))

        # Stop at next segment boundary by filtering results
        segment_rows = []
        for row in rows:
            # Check for next segment boundary in remaining results
            metadata = row.get('metadata', {})
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata) if metadata else {}

            if metadata.get('is_segment_boundary'):
                break

            segment_rows.append(row)

        return self.continuum_repo._parse_message_rows(segment_rows)

    def _generate_summary(
        self,
        messages: List[Message],
        sentinel: Message,
        continuum_id: str
    ) -> tuple[SummaryResult, list[float]]:
        """
        Generate segment summary and embedding.

        Fetches recent collapsed segments for narrative continuity - the summarizer
        can then use connective phrases like "Building on Tuesday's auth work..."

        Args:
            messages: Messages in segment
            sentinel: Segment boundary sentinel (for tools_used)
            continuum_id: Continuum ID for fetching previous summaries

        Returns:
            Tuple of (SummaryResult, embedding)

        Raises:
            RuntimeError: If summary generation or embedding generation fails
        """
        tools_used = sentinel.metadata.get('tools_used', [])
        user_id = get_current_user_id()

        # Fetch recent collapsed segments for narrative continuity (sliding window)
        previous_summaries = self.continuum_repo.find_collapsed_segments(
            continuum_id=continuum_id,
            user_id=user_id,
            limit=5  # Last 5 segments for context
        )

        try:
            # Generate summary using SummaryGenerator with previous summaries for continuity
            result = self.summary_generator.generate_summary(
                messages=messages,
                summary_type=SummaryType.SEGMENT,
                tools_used=tools_used,
                previous_summaries=previous_summaries
            )

            # Generate embedding for segment search (required for semantic segment search)
            embedding = self.embeddings_provider.encode_deep(result.synopsis)

            # Convert ndarray to list for JSON serialization (storage boundary)
            embedding_list = embedding.tolist()

            return result, embedding_list

        except Exception as e:
            # Re-raise to fail the entire collapse operation
            # Segment will remain active and retry on next timeout check
            logger.exception(
                "Segment summary generation failed; segment %s will remain active and retry",
                sentinel.metadata.get('segment_id')
            )
            raise RuntimeError("Segment collapse failed: summary generation error") from e

    def _trigger_downstream_processing(
        self,
        continuum_id: str,
        segment_id: str,
        sentinel: Message,
        messages: List[Message],
        summary: str,
    ) -> None:
        """
        Trigger downstream processing after segment collapse.

        Submits segment to:
        1. Memory extraction (direct through model_config=batch) - skipped for tombstoned segments
        2. Domain knowledge updates (if enabled)

        Requires: Active user context (set via set_current_user_id at handler entry)

        Args:
            continuum_id: Continuum UUID
            segment_id: Segment UUID
            sentinel: Collapsed segment sentinel
            messages: Messages in segment
            summary: Generated summary text (checked for tombstone)

        Raises:
            RuntimeError: If memory extraction submission fails
        """
        user_id = get_current_user_id()

        # Skip memory extraction for tombstoned segments (LLM refused to summarize)
        if summary == "[Segment content not summarized]":
            logger.warning(f"Skipping memory extraction for tombstoned segment {segment_id}")
            return

        # Process pending manual memories (from memory_tool.create_memory)
        # FIRST and best-effort: the queue is the sole durable record of
        # user-confirmed memories, so an extraction submission
        # failure below must not strand it. The drain is idempotent per item,
        # so retrying it on the next collapse's rescue sweep is safe; a blow-up
        # here leaves the items queued rather than destroying them.
        try:
            self._process_pending_manual_memories(user_id, segment_id)
        except Exception:
            logger.exception(
                "Pending manual memory drain failed for segment %s; items remain "
                "queued for retry on the next drain",
                segment_id,
            )

        # Memory extraction (direct through the fixed background route)
        if messages:
            # submit_segment_extraction is self-contained: loads messages, extracts, marks boundary
            extracted = self.lt_memory_factory.extraction_orchestrator.submit_segment_extraction(
                user_id=user_id,
                boundary_message_id=str(sentinel.id),
            )

            if not extracted:
                raise RuntimeError(f"Failed to extract memories from segment {segment_id}")

            logger.info("Direct extraction completed for segment %s", segment_id)

        # Process pending manual memories (from memory_tool.create_memory)
        self._process_pending_manual_memories(user_id, segment_id)

        # Domain knowledge updates (if user has blocks enabled)
        # NOTE (2025-11-07): Considered implementing segment collapse flush to domain knowledge service.
        # Trade-off: Letta requires consistent batches of 10 messages for iterative learning,
        # but segment collapse would flush partial batches (< 10 messages). Need to resolve
        # whether to prioritize: (a) Letta batch consistency vs (b) data completeness on collapse.
        # Current behavior: Messages buffer until batch size (10) or explicit disable/delete.
        # Deferred for future architectural decision.

    def _process_pending_manual_memories(self, user_id: str, segment_id: str) -> None:
        """
        Process pending manual memories queued by memory_tool.create_memory().

        Drains the Valkey queues idempotently, one item at a time:
        - reading a queue is non-destructive; an item counts as consumed only
          after a per-pending_id done marker is durably set, because
          store_memories is NOT idempotent — a re-run would duplicate the
          memory, so the marker (not a queue delete) is the consumption record;
        - the done marker is CLAIMED atomically (SET NX) before the store
          two overlapping drains — this segment's drain and the rescue
          sweep over older segments' queues — can both pass a check-then-set
          gate and double-store; a failed claim means another drain owns the
          item and this drain skips it;
        - an item whose embedding/store fails stays in its queue for the next
          drain instead of being destroyed with the batch. Only content-caused
          failures (the item's own data rejected by validation or by the
          database row insert) consume the attempts budget — infrastructure/
          LLM outages do not: retry counters never gate data on
          infrastructure failure, and the queue holds data the user was told
          was saved;
        - a permanently-failing item is dead-lettered after
          PENDING_ITEM_MAX_ATTEMPTS content-caused failures so it cannot wedge
          the queue forever; its dead-letter marker carries a full copy of the
          item, so the loss stays repairable by hand.

        In addition to the collapsed segment's own queue and the presegment
        queue, this sweeps older `pending_memories:{user_id}:*` keys stranded
        by a previous collapse whose downstream stage failed after the
        sentinel was already saved 'collapsed' — those segments are
        never re-matched by a collapse claim, so without this sweep their
        queues would only ever be deleted by TTL. The done markers make every
        re-drain safe: a visited item is never stored twice. This mirrors the
        keyed-idempotence shape of the extract_unprocessed_segments retry
        sweep and does NOT submit any second extraction run — extraction
        retries remain that sweep's job.

        Args:
            user_id: User UUID
            segment_id: Segment UUID being collapsed
        """
        import json
        import psycopg.errors
        from pydantic import ValidationError
        from lt_memory.models import PendingManualMemory, ExtractedMemory, MemoryLink
        from lt_memory.db_access import LTMemoryDB
        from clients.embeddings_provider import get_embeddings_provider
        from utils.database_session_manager import get_shared_session_manager

        valkey = get_valkey_client()

        # The collapsed segment's queue, the presegment queue, then any older
        # stranded queue keys for this user (rescue sweep — see docstring).
        # Done/attempt markers use the `pending_memories_done:` and
        # `pending_memories_attempts:` prefixes so the scan pattern below only
        # ever matches queue keys.
        queue_keys = [
            f"pending_memories:{user_id}:{segment_id}",
            f"pending_memories:{user_id}:presegment",
        ]
        try:
            for key in valkey.scan_iter(match=f"pending_memories:{user_id}:*"):
                if key not in queue_keys:
                    queue_keys.append(key)
        except Exception:
            logger.warning(
                "Pending-memory rescue scan failed for user %s; draining only the "
                "current segment's queues this pass",
                user_id, exc_info=True,
            )

        # Phase 1 — classify without destroying anything. Items with a done
        # marker (or unparseable payloads, which can never succeed on any
        # retry) are consumed; the rest are stored below.
        consumed_raw: dict[str, set[str]] = {key: set() for key in queue_keys}
        work: list[tuple[str, str, PendingManualMemory]] = []  # (queue_key, raw_json, item)
        for queue_key in queue_keys:
            try:
                pending_json_list = valkey.lrange(queue_key, 0, -1)
            except Exception:
                logger.error(
                    "Could not read pending memory queue %s; leaving it untouched "
                    "for the next drain",
                    queue_key, exc_info=True,
                )
                continue
            for json_str in pending_json_list:
                try:
                    pending = PendingManualMemory.from_json(json_str)
                except Exception:
                    # Drop only this entry, never the whole queue: the surviving
                    # items still need their embedding/store pass.
                    consumed_raw[queue_key].add(json_str)
                    logger.warning(
                        "Dropping unparseable pending memory from %s: %.200s",
                        queue_key, json_str, exc_info=True,
                    )
                    continue
                done_value = valkey.get(f"pending_memories_done:{user_id}:{pending.pending_id}")
                if done_value == "claimed":
                    # Another drain holds the SET NX claim on this item
                    # Neither re-store it nor count it consumed —
                    # the claim holder may still release it on failure.
                    continue
                if done_value is not None:
                    # Already durably stored (or dead-lettered, with the
                    # item's content preserved in the marker) by an earlier
                    # drain; consume without re-storing.
                    consumed_raw[queue_key].add(json_str)
                    continue
                work.append((queue_key, json_str, pending))

        if not work and not any(consumed_raw.values()):
            logger.debug(f"No pending manual memories for segment {segment_id}")
            return

        logger.info(
            "Draining pending manual memories for segment %s: %d to store, %d already consumed",
            segment_id, len(work), sum(len(v) for v in consumed_raw.values()),
        )

        embeddings_provider = get_embeddings_provider()
        session_manager = get_shared_session_manager()
        db = LTMemoryDB(session_manager)

        # Resolved once for the whole batch, before the loop, so a preferences
        # outage degrades to UTC here instead of failing each memory below.
        memory_tz = _resolve_pending_memory_timezone(segment_id)

        # Collect stored manual memories so the integration curator can tend
        # them after the loop (preserving their user-specified attributes).
        stored_manual = []  # list[tuple[str, str]] of (full_uuid_str, text)

        # Phase 2 — store each pending item. One item's failure never touches
        # the others: the failing item stays in its queue with a bounded retry
        # counter that only content-caused failures consume, the surviving
        # items continue to their own durable store.
        for queue_key, json_str, mem in work:
            done_key = f"pending_memories_done:{user_id}:{mem.pending_id}"
            attempts_key = f"pending_memories_attempts:{user_id}:{mem.pending_id}"
            # Tracks whether this drain holds the SET NX claim, so the
            # failure paths below only ever release a marker they own.
            claimed = False
            try:
                # Claim the done marker ATOMICALLY BEFORE the store: SET NX
                # (set-if-not-exists, with TTL) — the same acquire shape the
                # compaction lock uses. The pre-repair gate was check-then-set
                # (exists(done) -> store_memories -> setex(done)), so two
                # overlapping drains could both pass the exists() check and
                # both store, and store_memories is NOT idempotent. With the
                # guard living in the mutation itself, only one drain can
                # flip the marker; claim failure means another drain owns
                # this item: skip.
                if not valkey.set(
                    done_key, "claimed", nx=True, ex=PENDING_DONE_MARKER_TTL_SECONDS
                ):
                    logger.debug(
                        "Pending manual memory %s is claimed by another drain; skipping",
                        mem.pending_id,
                    )
                    continue
                claimed = True

                # Document embedding (deep encoder)
                embedding = embeddings_provider.encode_deep([mem.text])[0].tolist()

                # Parse temporal fields in the user's timezone — these strings are
                # model-supplied local wall times, not UTC instants.
                parsed_happens_at = None
                parsed_expires_at = None
                if mem.happens_at:
                    parsed_happens_at = _parse_pending_temporal_field(
                        mem.happens_at, "happens_at", mem.pending_id, memory_tz
                    )
                if mem.expires_at:
                    parsed_expires_at = _parse_pending_temporal_field(
                        mem.expires_at, "expires_at", mem.pending_id, memory_tz
                    )

                # Create ExtractedMemory
                extracted = ExtractedMemory(
                    text=mem.text,
                    importance_score=mem.importance_score,
                    happens_at=parsed_happens_at,
                    expires_at=parsed_expires_at
                )

                # Store memory
                created_ids = db.store_memories([extracted], embeddings=[embedding])
                memory_id = created_ids[0]

                # Manual memories skip entity extraction (no LLM extraction context).
                # Entities get linked when segment extraction processes the segment.

                # Create supersedes links if provided
                for short_id in mem.supersedes_memory_ids:
                    target = self._find_memory_by_short_id(short_id, db)
                    if target:
                        link = MemoryLink(
                            source_id=memory_id,
                            target_id=target.id,
                            link_type="supersedes",
                            reasoning="Manual supersedes link",
                            created_at=utc_now()
                        )
                        db.create_links([link])
                        logger.debug(f"Created supersedes link: {memory_id} -> {target.id}")

                # Durable consumption record. This drain already holds the
                # claim (SET NX above), so overwriting the transient
                # "claimed" value with the terminal one is safe; the claim,
                # once granted, is what stops a concurrent drain from
                # double-storing the same item. The residual crash window
                # (DB commit succeeds, process dies around claim/overwrite)
                # is the same trade the extraction sweep accepts for its
                # boundary marker.
                valkey.setex(done_key, PENDING_DONE_MARKER_TTL_SECONDS, "1")
                valkey.delete(attempts_key)

                logger.info(
                    f"Processed manual memory {memory_id} "
                    f"(pending_id: {mem.pending_id})"
                )
                stored_manual.append((str(memory_id), mem.text))

            except (ValidationError, psycopg.errors.DataError) as exc:
                # Content-caused failure: the item's own data was
                # rejected — its fields failed the durable-representation
                # validation (ExtractedMemory), or Postgres rejected the row
                # itself (DataError). Deterministic on every retry, so this
                # — and only this — consumes the item's attempts budget;
                # infrastructure/LLM outages never do.
                attempts = valkey.increment_with_expiry(
                    attempts_key, PENDING_DONE_MARKER_TTL_SECONDS
                )
                if attempts >= PENDING_ITEM_MAX_ATTEMPTS:
                    # Dead-letter: mark done with a TERMINAL value (anything
                    # other than "claimed") that carries a full copy of the
                    # item's content, so the memory the user was told was
                    # saved stays diagnosable and repairable by hand like a
                    # dropped temporal anchor, and the queue can drain. This
                    # drain holds the claim, so overwriting it is safe.
                    valkey.setex(
                        done_key,
                        PENDING_DONE_MARKER_TTL_SECONDS,
                        json.dumps({
                            "status": "deadletter",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "attempts": PENDING_ITEM_MAX_ATTEMPTS,
                            "item": mem.model_dump(),
                        }),
                    )
                    valkey.delete(attempts_key)
                    logger.critical(
                        "Pending manual memory %s failed %d content-caused attempts; "
                        "dead-lettering it (content preserved in %s) so the queue "
                        "can drain. Memory text follows so the loss is repairable "
                        "by hand: %r",
                        mem.pending_id, PENDING_ITEM_MAX_ATTEMPTS, done_key, mem.text,
                    )
                else:
                    # Not yet dead-lettered: release the claim so a later
                    # drain retries the item. The item is NOT consumed.
                    try:
                        valkey.delete(done_key)
                    except Exception:
                        logger.warning(
                            "Could not release claim on %s; item %s stays skipped "
                            "until the claim TTL expires",
                            done_key, mem.pending_id, exc_info=True,
                        )
                    logger.error(
                        "Failed to process pending memory %s (content-caused, "
                        "attempt %d/%d); left in queue for the next drain",
                        mem.pending_id, attempts, PENDING_ITEM_MAX_ATTEMPTS,
                    )
            except Exception:
                # Infrastructure failure: the embedding encoder, the
                # database, Valkey — anything not attributable to this item's
                # content. It must NOT consume the attempts budget: an infra/
                # LLM outage burning all PENDING_ITEM_MAX_ATTEMPTS on one item
                # would dead-letter a memory the user was told was saved.
                # Pinned doctrine: retry counters never gate data on
                # infrastructure failure. The item stays queued; only the
                # claim is released so a later drain (once infrastructure
                # recovers) can retry it.
                if claimed:
                    try:
                        valkey.delete(done_key)
                    except Exception:
                        logger.warning(
                            "Could not release claim on %s; item %s stays skipped "
                            "until the claim TTL expires",
                            done_key, mem.pending_id, exc_info=True,
                        )
                logger.exception(
                    "Failed to process pending memory %s (infrastructure failure; "
                    "not counted against its %d-attempt budget); left in queue "
                    "for the next drain",
                    mem.pending_id, PENDING_ITEM_MAX_ATTEMPTS,
                )

        # Phase 3 — settle each queue. A queue is deleted only when every item
        # in it is durably consumed; otherwise its TTL is extended past the
        # producer's 24h so the failing items survive until a later drain.
        for queue_key in queue_keys:
            try:
                remaining = valkey.lrange(queue_key, 0, -1)
            except Exception:
                logger.error(
                    "Could not re-read pending memory queue %s for settlement; "
                    "leaving it untouched",
                    queue_key, exc_info=True,
                )
                continue
            if not remaining:
                continue
            for json_str in remaining:
                if json_str in consumed_raw[queue_key]:
                    continue
                try:
                    pending = PendingManualMemory.from_json(json_str)
                except Exception:
                    continue
                done_value = valkey.get(f"pending_memories_done:{user_id}:{pending.pending_id}")
                if done_value is None or done_value == "claimed":
                    # At least one item not durably consumed — still queued, or
                    # another drain's claim on it is still in flight: keep the
                    # queue alive for retry.
                    valkey.expire(queue_key, PENDING_QUEUE_RETRY_TTL_SECONDS)
                    break
            else:
                valkey.delete(queue_key)

        # Tend the manually-created memories via the integration curator too.
        # They were stored immediately with user-specified attributes (score,
        # happens_at, expires_at, supersedes) — preserved as-is; the curator
        # only decides link/merge/stand-alone on the neighborhood.
        if stored_manual:
            self._tend_manual_memories(user_id, segment_id, stored_manual)

    def _on_memories_stored(
        self,
        *,
        user_id: str,
        segment_id: Optional[str],
        memory_ids: list,
        memories: list,
        candidate_hints: dict,
    ) -> None:
        """Factory callback: spawn the integration curator for newly stored memories.

        Registered on lt_memory_factory.on_memories_stored in __init__. Called by
        store_and_tend_extraction (both execution paths) after memories land.
        """
        new_memories = [
            {"memory_id": str(mid), "text": getattr(mem, "text", "")}
            for mid, mem in zip(memory_ids, memories)
        ]
        self._spawn_integration_curator(
            user_id, segment_id, new_memories, candidate_hints, source="extraction"
        )

    def _tend_manual_memories(
        self,
        user_id: str,
        segment_id: str,
        stored_manual: list,
    ) -> None:
        """Build candidate hints for manual memories and spawn the curator.

        Manual memories carry user-specified attributes (score, happens_at,
        expires_at, supersedes) and skip entity extraction, so candidate hints
        come from discovery axes only (no extraction-time bonds).
        """
        candidate_hints: dict[str, list[dict]] = {}
        linking = self.lt_memory_factory.linking
        for memory_id_str, _text in stored_manual:
            try:
                hints = linking.find_candidate_hints(UUID(memory_id_str))
                if hints:
                    candidate_hints[memory_id_str] = hints
            except Exception:
                logger.warning("find_candidate_hints failed for manual memory %s", memory_id_str, exc_info=True)
        new_memories = [{"memory_id": mid, "text": text} for mid, text in stored_manual]
        self._spawn_integration_curator(
            user_id, segment_id, new_memories, candidate_hints, source="drain"
        )

    def _spawn_integration_curator(
        self,
        user_id: str,
        segment_id: Optional[str],
        new_memories: list[dict],
        candidate_hints: dict[str, list[dict]],
        *,
        source: str,
    ) -> None:
        """Spawn MemoryCuratorAgent integration mode (forage-style background thread).

        The agent tends each new memory: link / merge / stand-alone, using the
        pre-computed candidate hints (deterministic discovery + extraction
        bonds). Runs in a daemon thread with copied user context so the
        collapse chain doesn't block on curation.

        ``source`` tags which path spawned ("extraction" or "drain") so the
        two runs get distinct WorkItem identities and can coexist under the
        identity contract (item_id + UNIQUE(interface_name, thread_id)) —
        a dual-path collapse no longer races one shared UPSERT row.
        """
        from config import config

        if not config.memory_curator.enabled:
            logger.info(
                "memory_curator disabled; skipping integration curator spawn (%s)", source
            )
            return
        if not new_memories:
            return
        if self.tool_repo is None:
            logger.warning("tool_repo unavailable; skipping integration curator spawn")
            return

        from agents.implementations.memory_curator_agent import MemoryCuratorAgent
        from agents.sidebar import WorkItem
        from contextvars import copy_context
        import threading

        seg_id = str(segment_id) if segment_id else "unknown"
        work_item = WorkItem(
            item_id=f"integrate_{seg_id}_{user_id}:{source}",
            interface_name="memory_curator_integration",
            context={
                "mode": "integration",
                "segment_id": seg_id,
                "new_memories": new_memories,
                "candidate_hints": candidate_hints,
            },
        )

        try:
            agent = MemoryCuratorAgent(tool_repo=self.tool_repo)
            ctx = copy_context()
            thread = threading.Thread(
                target=ctx.run,
                args=(agent.run, work_item, self.event_bus),
                name=f"curator-integrate-{work_item.item_id[:24]}",
                daemon=True,
            )
            thread.start()
            logger.info(
                "Spawned MemoryCuratorAgent (integration) for segment %s, %d memories",
                seg_id, len(new_memories),
            )
        except Exception:
            # Curation is best-effort — never fail the collapse chain.
            logger.exception("Failed to spawn MemoryCuratorAgent (integration)")

    def _find_memory_by_short_id(self, short_id: str, db: LTMemoryDB) -> Memory | None:
        """
        Find a memory by short ID for supersedes linking.

        Args:
            short_id: Either "mem_XXXXXXXX" or raw "XXXXXXXX"
            db: LTMemoryDB instance

        Returns:
            Memory model or None if not found
        """
        from utils.tag_parser import parse_memory_id
        from lt_memory.models import Memory

        clean_id = parse_memory_id(short_id)
        if not clean_id or len(clean_id) < 8:
            return None

        user_id = get_current_user_id()
        with db.session_manager.get_session(user_id) as session:
            query = """
            SELECT * FROM memories
            WHERE REPLACE(id::text, '-', '') LIKE %(pattern)s
              AND is_archived = FALSE
            LIMIT 2
            """
            result = session.execute_query(query, {'pattern': f"{clean_id.lower()}%"})

            if len(result) > 1:
                raise ValueError(
                    f"Ambiguous short ID '{short_id}' — matches multiple memories; use the full UUID"
                )
            if result:
                return Memory(**result[0])
            return None

    def _extract_tools_from_messages(self, messages: List[Message]) -> List[str]:
        """
        Extract unique tools used by parsing message content for tool_call blocks.

        Args:
            messages: Messages in segment

        Returns:
            Sorted list of unique tool names
        """
        tools_used = set()

        for msg in messages:
            # Skip non-assistant messages (tools only in assistant responses)
            if msg.role != "assistant":
                continue

            # Check if content is structured (list of blocks)
            if isinstance(msg.content, list):
                for block in msg.content:
                    # Extract tool name from tool_call blocks
                    if isinstance(block, dict) and block.get('type') == 'tool_call':
                        tool_name = block.get('name')
                        if tool_name:
                            tools_used.add(tool_name)

        return sorted(list(tools_used))

    def _count_user_segments(self) -> int:
        """
        Count total segments for user (for ManifestUpdatedEvent).

        Requires: Active user context (set via set_current_user_id at handler entry)

        Returns:
            Number of segments for user

        Raises:
            RuntimeError: If database query fails
        """
        from utils.database_session_manager import get_shared_session_manager
        session_manager = get_shared_session_manager()

        user_id = get_current_user_id()
        with session_manager.get_session(user_id) as session:
            result = session.execute_single("""
                SELECT COUNT(*) as count
                FROM messages
                WHERE user_id = %s
                    AND metadata->>'is_segment_boundary' = 'true'
            """, (user_id,))

            return result['count'] if result else 0

    def _process_feedback_loop(
        self,
        messages: List[Message],
        segment_id: UUID,
        continuum_id: UUID,
    ) -> None:
        """
        Process the user model pipeline: assess behavior and synthesize if due.

        Evaluates conversation against the system prompt's behavioral contract,
        producing section-anchored signals. When synthesis threshold is reached,
        evolves the user model with critic validation.

        Args:
            messages: Messages in the collapsed segment
            segment_id: Segment UUID
            continuum_id: Continuum UUID
        """
        if not self._init_feedback_loop():
            return

        user_id = get_current_user_id()

        try:
            # Step 1: Get current user model (needed by assessor for calibration)
            current_model_xml = self._feedback_tracker.get_last_synthesis_output(user_id)

            # Step 2: Extract assessment signals from this segment
            signals = self._assessment_extractor.extract_signals(
                messages=messages,
                segment_id=segment_id,
                continuum_id=continuum_id,
                user_model_xml=current_model_xml
            )

            # Step 3: Persist signals
            if signals:
                self._feedback_repo.save_signals(signals)
                logger.info("Extracted %d assessment signals from segment %s", len(signals), segment_id)

            # Step 4: Synthesize user model if threshold reached (every 7 use-days)
            if self._feedback_tracker.should_synthesize(user_id):
                logger.info("Running user model synthesis for user %s", user_id)

                result = self._synthesizer.synthesize(
                    user_id=user_id,
                    current_model_xml=current_model_xml
                )

                # Store result
                self._feedback_tracker.mark_synthesized(
                    user_id,
                    result.raw_xml,
                    needs_checkin=len(result.checkin_topics) > 0
                )

                # Mark signals as synthesized
                unsynthesized = self._feedback_repo.get_unsynthesized_signals(user_id)
                signal_ids = [s['id'] for s in unsynthesized]
                if signal_ids:
                    self._feedback_repo.mark_signals_synthesized(user_id, signal_ids)

                # Invalidate LoRA trinket cache so next turn picks up new model
                self._invalidate_lora_trinket_cache(user_id)

                logger.info(
                    "User model synthesis complete: %d observations, %d checkin topics",
                    len(result.observations), len(result.checkin_topics)
                )

        except Exception as e:
            # Don't re-raise — segment is already collapsed and saved at this point.
            # But log at error so operators notice immediately.
            logger.error(
                "USER MODEL PIPELINE FAILED for segment %s: %s — "
                "Assessment signals and/or synthesis were lost for this segment.",
                segment_id, e, exc_info=True
            )

    def _process_persona(
        self,
        messages: List[Message],
        segment_id: UUID,
        continuum_id: UUID,
    ) -> None:
        """Store segment evidence and publish a validated Persona revision when due.

        Deliberately does not catch exceptions, unlike _process_feedback_loop above.
        That loop swallows failures because the user model is a best-effort overlay on
        an already-collapsed segment; a Persona failure is instead propagated to
        handle_timeout's caller and counted toward the MAX_COLLAPSE_ATTEMPTS tombstone,
        so a persistent breakage surfaces loudly rather than silently stopping the
        revision history from growing.
        """
        user_id = get_current_user_id()
        service = self._get_persona_service()
        signals = service.evaluate_segment(
            user_id,
            messages,
            segment_id=segment_id,
            continuum_id=continuum_id,
        )
        revision = service.refine_automatically_if_due(user_id)
        logger.info(
            "Persona processed segment %s: %d signals%s",
            segment_id,
            len(signals),
            f", published revision {revision.revision_number}" if revision else "",
        )

    def _process_portrait_synthesis(self) -> None:
        """
        Synthesize user portrait if use-day threshold reached.

        Runs after the feedback loop in the collapse chain. Portrait synthesis
        is optional content — failures are logged but do not fail the collapse.
        """
        from config import config
        from cns.services.portrait_service import should_synthesize_portrait, synthesize_and_store

        user_id = get_current_user_id()
        threshold = config.scheduled_jobs.portrait_synthesis_use_days

        try:
            if not should_synthesize_portrait(user_id, threshold):
                return

            logger.info("Running portrait synthesis for user %s", user_id)
            synthesize_and_store(user_id)

        except Exception as e:
            logger.error(
                "PORTRAIT SYNTHESIS FAILED for user %s: %s — "
                "Portrait will be stale until next qualifying collapse.",
                user_id, e, exc_info=True
            )

    def _invalidate_lora_trinket_cache(self, user_id: str) -> None:
        """Invalidate LoRA trinket cache after synthesis."""
        try:
            from working_memory.trinkets.base import TRINKET_KEY_PREFIX
            valkey = get_valkey_client()
            valkey.hdel_with_retry(f"{TRINKET_KEY_PREFIX}:{user_id}", "behavioral_directives")
        except Exception as e:
            logger.warning("Failed to invalidate LoRA trinket cache: %s", e, exc_info=True)


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_collapse_handler_instance: Optional[SegmentCollapseHandler] = None


def get_segment_collapse_handler() -> SegmentCollapseHandler:
    """
    Get singleton SegmentCollapseHandler instance.

    Returns:
        The initialized SegmentCollapseHandler

    Raises:
        RuntimeError: If handler not initialized (factory not run)
    """
    if _collapse_handler_instance is None:
        raise RuntimeError(
            "SegmentCollapseHandler not initialized. "
            "Ensure CNSIntegrationFactory has been initialized."
        )
    return _collapse_handler_instance


def initialize_segment_collapse_handler(handler: SegmentCollapseHandler) -> None:
    """
    Initialize the singleton handler instance (called by factory).

    Args:
        handler: The handler instance to store
    """
    global _collapse_handler_instance
    _collapse_handler_instance = handler
    logger.info("SegmentCollapseHandler singleton initialized")
