"""
Segment timeout detection service.

APScheduler job that runs every 5 minutes to find active segments
that have exceeded their inactivity threshold and publishes SegmentTimeoutEvent.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, TypedDict

from cns.core.events import SegmentTimeoutEvent
from cns.integration.event_bus import EventBus
from cns.infrastructure.continuum_repository import ActiveSegmentRow, get_continuum_repository
from utils.distributed_lock import UserRequestLock
from utils.database_session_manager import get_shared_session_manager
from utils.timezone_utils import utc_now, convert_from_utc, parse_utc_time_string
from config import config

logger = logging.getLogger(__name__)

# Per-user request lock, probed between timeout determination and
# publication. Same key the transports acquire (`user_lock:{user_id}`,
# pinned in UserRequestLock's constructor); probe-only use never acquires
# or renews, and TTL is irrelevant to `is_locked` (key existence). Built
# at module import, Valkey resolved lazily on first operation.
_user_request_lock = UserRequestLock()


class TimeoutCheckResult(TypedDict):
    """Statistics from a timeout check cycle."""
    segments_checked: int
    timeouts_published: int


class SegmentTimeoutService:
    """
    Detects and publishes timeout events for inactive segments.

    Runs as scheduled job every 5 minutes, checking all active segment
    sentinels against context-aware timeout thresholds.
    """

    def __init__(self, event_bus: EventBus, continuum_repository=None, session_manager=None):
        """
        Initialize timeout service.

        Args:
            event_bus: Event bus for publishing timeout events
            continuum_repository: Continuum repository (uses singleton if not provided)
            session_manager: Database session manager for user queries (uses shared if not provided)
        """
        self.event_bus = event_bus
        self.continuum_repository = continuum_repository or get_continuum_repository()
        self.session_manager = session_manager or get_shared_session_manager()
        self._user_timezone_cache = {}  # Cache timezones during check cycle

    def check_timeouts(self) -> TimeoutCheckResult:
        """
        Check all active segments for timeout and publish events.

        Returns:
            TimeoutCheckResult with segments_checked and timeouts_published

        Raises:
            Exception: If timeout detection fails (database errors, infrastructure failures).
                      Scheduled job will fail visibly and alert operators.
        """
        try:
            logger.debug("Starting segment timeout check")

            # Clear timezone cache for fresh data
            self._user_timezone_cache.clear()

            # Find all active segment sentinels (raises on database failure)
            active_segments = self._get_active_segments()

            if not active_segments:
                logger.debug("No active segments found")
                return {'segments_checked': 0, 'timeouts_published': 0}

            logger.debug(f"Checking {len(active_segments)} active segments")

            timeouts_published = 0
            current_time = utc_now()

            for segment in active_segments:
                # Check if segment has timed out
                if self._is_timed_out(segment, current_time):
                    # Held lock means a turn is in flight: turn persistence is
                    # atomic, so an uncommitted turn is invisible to the
                    # inactivity clock no matter how active it is. Skip this
                    # segment for this cycle: no event, no attempt, no claim;
                    # the next cycle re-evaluates, and the turn's final
                    # committed message re-anchors the clock at turn end.
                    # `is_locked` raises when Valkey is down; the raise
                    # propagates through this job's log-and-reraise. An
                    # unverifiable "no turn in flight" must never default
                    # to collapsing.
                    if _user_request_lock.is_locked(segment['user_id']):
                        logger.debug(
                            "Deferring timeout for segment %s of user %s: "
                            "turn in flight (user request lock held)",
                            segment['metadata'].get('segment_id'),
                            segment['user_id'],
                        )
                        continue

                    # Publish timeout event
                    self._publish_timeout_event(segment, current_time)
                    timeouts_published += 1

            logger.info(
                f"Timeout check complete: {len(active_segments)} segments checked, "
                f"{timeouts_published} timeouts published"
            )

            return {
                'segments_checked': len(active_segments),
                'timeouts_published': timeouts_published
            }

        except Exception as e:
            # Explicitly log full exception details before re-raising
            # (journalctl may suppress APScheduler's traceback)
            logger.error(
                f"Segment timeout check failed: {type(e).__name__}: {e}",
                exc_info=True
            )
            raise

    def _get_active_segments(self) -> list[ActiveSegmentRow]:
        """
        Query all active segment sentinels from database.

        Returns:
            List of dicts with segment data (id, continuum_id, user_id, metadata, created_at)

        Raises:
            RuntimeError: If database query fails
        """
        # Admin query - need to see all users' segments
        return self.continuum_repository.find_all_active_segments_admin()

    def _get_user_timezone(self, user_id: str) -> str:
        """
        Get timezone for specific user (with caching).

        Args:
            user_id: User UUID

        Returns:
            IANA timezone name, or system default if user has no timezone configured

        Raises:
            Exception: If database query fails (infrastructure issue)
        """
        # Check cache first
        if user_id in self._user_timezone_cache:
            return self._user_timezone_cache[user_id]

        # Query from database - raises on infrastructure failure
        with self.session_manager.get_admin_session() as session:
            row = session.execute_single(
                "SELECT timezone FROM users WHERE id = %s",
                (user_id,)
            )

            # User has configured timezone
            if row and row.get('timezone'):
                timezone = row['timezone']
                self._user_timezone_cache[user_id] = timezone
                return timezone

        # User has no timezone configured - system default is correct fallback
        default_tz = config.system.timezone
        self._user_timezone_cache[user_id] = default_tz
        logger.debug(f"User {user_id} has no timezone configured, using system default: {default_tz}")
        return default_tz

    def _is_timed_out(self, segment: ActiveSegmentRow, current_time: datetime) -> bool:
        """
        Check if segment has exceeded timeout threshold.

        Args:
            segment: Segment data dict
            current_time: Current UTC time

        Returns:
            True if segment has timed out
        """
        # A pending heartbeat wake is a liveness commitment: MIRA has been told
        # to sleep until heartbeat_wake_at and will produce activity when the
        # wake fires. Collapsing inside that window would compress the session
        # out from under the sleep. The grace window also covers the wake turn
        # itself: the stimulus commits when the turn completes, so there is a
        # stretch after wake_at where the sleep has ended but no fresh message
        # exists yet. A malformed or absent stamp changes nothing.
        wake_at_str = segment['metadata'].get('heartbeat_wake_at')
        if wake_at_str:
            try:
                wake_at = parse_utc_time_string(wake_at_str)
            except (ValueError, TypeError):
                wake_at = None
                logger.warning(
                    "Unparseable heartbeat_wake_at %r on segment %s; "
                    "ignoring the wake guard",
                    wake_at_str, segment['metadata'].get('segment_id'),
                )
            if wake_at is not None:
                guard_until = wake_at + timedelta(
                    seconds=config.heartbeat.wake_grace_seconds
                )
                if current_time < guard_until:
                    return False

        end_time = self._segment_activity_anchor(segment)

        # Calculate inactive duration
        inactive_duration = current_time - end_time
        inactive_minutes = inactive_duration.total_seconds() / 60

        # Time-of-day aware staleness threshold, evaluated in the segment
        # owner's local time. Optional per-window overrides fall back to the
        # base segment_timeout when unset.
        threshold = config.system.segment_timeout
        user_tz = self._get_user_timezone(segment['user_id'])
        local_hour = convert_from_utc(current_time, user_tz).hour
        if 6 <= local_hour <= 9:
            threshold = config.system.segment_timeout_morning or threshold
        elif 23 <= local_hour or local_hour <= 6:
            threshold = config.system.segment_timeout_late_night or threshold

        # Check if timeout exceeded
        timed_out = inactive_minutes >= threshold

        if timed_out:
            segment_id = segment['metadata'].get('segment_id')
            logger.debug(
                f"Segment {segment_id} timed out: "
                f"inactive_minutes={inactive_minutes:.1f}, threshold={threshold}"
            )

        return timed_out

    def _segment_activity_anchor(self, segment: ActiveSegmentRow) -> datetime:
        """
        Compute a segment's last-activity anchor: max of its last committed
        message, the sentinel's last_turn_at stamp, and (fallback) the
        sentinel's creation time.

        Single source for the timeout decision and the published event's
        inactive_duration_minutes: the two must agree by construction.
        """
        # Query for last message in segment, keyed on the segment's own
        # identity. Messages of OTHER segments — including a successor
        # segment's heartbeat ticks — are not this segment's activity and
        # must not feed its timeout decision (a stranded mid-collapse
        # sentinel whose recovery re-match computed end_time from a
        # successor's fresh messages never looked timed out, so the stale
        # claim was never re-claimed).
        end_time = self._get_last_message_time(
            segment['continuum_id'],
            segment['user_id'],
            segment['metadata'].get('segment_id')
        )

        if not end_time:
            # No messages found — use sentinel creation time as last activity.
            # Lets orphaned segments (empty, lost messages, race conditions)
            # reach the collapse handler's tombstone circuit breaker naturally.
            end_time = segment['created_at']

        # last_turn_at is the heartbeat liveness channel plus a bounded
        # crash grace, not the live-turn guard (the lock probe in
        # check_timeouts owns that). stamp_segment_liveness()
        # (cns/infrastructure/continuum_repository.py) stamps it at every
        # wake-turn dispatch, so a hung wake turn keeps this anchor fresh
        # for one threshold window even after its lock expires; the retry
        # path re-stamps under the _LIVENESS_STAMP_FAILURE_CAP = 3
        # anti-starvation bound (cns/services/heartbeat_service.py), so a
        # persistently failing heartbeat cannot starve this sweep forever.
        # A turn that died without committing (process death, failed
        # commit) leaves the arrival stamp (increment_segment_turn) as the
        # only fresher-than-committed signal once its lock expires; without
        # this read such a segment collapses on stale committed messages
        # alone. A malformed stamp is a per-segment fault and must not
        # abort the whole sweep cycle for every user; warn and treat as
        # absent, mirroring the tolerant heartbeat_wake_at parse in
        # _is_timed_out.
        last_turn_at_str = segment['metadata'].get('last_turn_at')
        if last_turn_at_str:
            try:
                last_turn_at = parse_utc_time_string(last_turn_at_str)
            except (ValueError, TypeError):
                logger.warning(
                    "Unparseable last_turn_at %r on segment %s; "
                    "ignoring the stamp",
                    last_turn_at_str, segment['metadata'].get('segment_id'),
                )
            else:
                if last_turn_at > end_time:
                    end_time = last_turn_at

        return end_time

    def _get_last_message_time(
        self,
        continuum_id: str,
        user_id: str,
        segment_id: str
    ) -> Optional[datetime]:
        """
        Query for timestamp of last message in segment, keyed on segment identity.

        Every turn message (assistant, tool result, user) carries the segment's
        id in metadata — membership is keyed on identity, not on a position in
        the continuum, so a successor segment's messages (including its
        heartbeat ticks) can never masquerade as this segment's activity.

        Args:
            continuum_id: Continuum UUID
            user_id: User UUID
            segment_id: Segment UUID whose activity is being measured

        Returns:
            Timestamp of last message, or None if no messages in segment

        Raises:
            RuntimeError: If database query fails
        """
        with self.session_manager.get_admin_session() as session:
            row = session.execute_single("""
                SELECT created_at FROM messages
                WHERE continuum_id = %s
                    AND user_id = %s
                    AND metadata->>'segment_id' = %s
                    AND (metadata->>'is_segment_boundary' IS NULL
                         OR metadata->>'is_segment_boundary' = 'false')
                    AND (metadata->>'system_notification' IS NULL
                         OR metadata->>'system_notification' = 'false')
                ORDER BY created_at DESC
                LIMIT 1
            """, (continuum_id, user_id, segment_id))

            # Database returns datetime objects directly (no normalization to strings)
            if row and row.get('created_at'):
                return row['created_at']
            return None

    def _publish_timeout_event(self, segment: ActiveSegmentRow, current_time: datetime) -> None:
        """
        Publish SegmentTimeoutEvent for timed-out segment.

        Args:
            segment: Segment data dict
            current_time: Current UTC time
        """
        metadata = segment['metadata']
        segment_id = metadata.get('segment_id')

        # Same anchor helper as the timeout decision, so the reported
        # inactive_duration_minutes agrees with the decision basis exactly.
        end_time = self._segment_activity_anchor(segment)

        # Calculate inactive duration
        inactive_duration = current_time - end_time
        inactive_minutes = int(inactive_duration.total_seconds() / 60)

        # Get local hour for event metadata (even though not used for threshold)
        user_id = segment['user_id']
        user_tz = self._get_user_timezone(user_id)
        local_time = convert_from_utc(current_time, user_tz)
        local_hour = local_time.hour

        # Publish event
        event = SegmentTimeoutEvent.create(
            continuum_id=segment['continuum_id'],
            user_id=user_id,
            segment_id=segment_id,
            inactive_duration_minutes=inactive_minutes,
            local_hour=local_hour
        )

        self.event_bus.publish(event)

        logger.info(
            f"Published SegmentTimeoutEvent for segment {segment_id}: "
            f"inactive={inactive_minutes}min, local_hour={local_hour}"
        )


# Singleton instance
_timeout_service = None


def get_timeout_service(event_bus: EventBus) -> SegmentTimeoutService:
    """Get or create singleton SegmentTimeoutService instance."""
    global _timeout_service
    if _timeout_service is None:
        _timeout_service = SegmentTimeoutService(event_bus)
        logger.info("SegmentTimeoutService singleton initialized")
    return _timeout_service


def register_timeout_job(scheduler_service, event_bus: EventBus) -> None:
    """
    Register segment timeout detection job with scheduler.

    Args:
        scheduler_service: System scheduler service
        event_bus: Event bus for publishing timeout events

    Raises:
        ImportError: If apscheduler is not installed
        RuntimeError: If scheduler service or event bus are not properly initialized
    """
    from apscheduler.triggers.interval import IntervalTrigger
    from utils.scheduled_task_monitor import ScheduledTaskMonitor

    timeout_service = get_timeout_service(event_bus)

    # Wrap with monitoring and timeout. The ceiling comes from
    # config.scheduled_jobs.job_timeout_seconds so operators can tune it per
    # deployment; keep it below the 5-minute interval to avoid overlapping runs.
    monitored_check_timeouts = ScheduledTaskMonitor.wrap_scheduled_job(
        job_id="segment_timeout_detection",
        func=timeout_service.check_timeouts,
        timeout_seconds=config.scheduled_jobs.job_timeout_seconds,
        kill_on_timeout=True
    )

    scheduler_service.register_job(
        job_id="segment_timeout_detection",
        func=monitored_check_timeouts,
        trigger=IntervalTrigger(minutes=5),
        component="cns",
        description="Check for timed-out active segments every 5 minutes"
    )

    logger.info("Successfully registered segment timeout detection job (5-minute interval)")
