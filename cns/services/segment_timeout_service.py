"""
Segment timeout detection service.

APScheduler job that runs every 5 minutes to find active segments
that have exceeded their inactivity threshold and publishes SegmentTimeoutEvent.
"""
import logging
from datetime import datetime
from typing import Optional, TypedDict

from cns.core.events import SegmentTimeoutEvent
from cns.integration.event_bus import EventBus
from cns.infrastructure.continuum_repository import ActiveSegmentRow, get_continuum_repository
from utils.database_session_manager import get_shared_session_manager
from utils.timezone_utils import utc_now, convert_from_utc, parse_utc_time_string
from config import config

logger = logging.getLogger(__name__)


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
        # Query for last message in segment (avoids persisting end_time on every turn)
        end_time = self._get_last_message_time(
            segment['continuum_id'],
            segment['user_id'],
            segment['created_at']  # Segment start = sentinel creation time
        )

        if not end_time:
            # No messages found — use sentinel creation time as last activity.
            # Lets orphaned segments (empty, lost messages, race conditions)
            # reach the collapse handler's tombstone circuit breaker naturally.
            end_time = segment['created_at']

        # Race condition guard: increment_segment_turn() stamps last_turn_at
        # atomically when a user message arrives, BEFORE the message is committed
        # via uow.commit(). Without this, a segment resumed after a long pause
        # appears timed-out because the last committed message is hours old.
        last_turn_at_str = segment['metadata'].get('last_turn_at')
        if last_turn_at_str:
            last_turn_at = parse_utc_time_string(last_turn_at_str)
            if last_turn_at > end_time:
                end_time = last_turn_at

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

    def _get_last_message_time(
        self,
        continuum_id: str,
        user_id: str,
        segment_start_time: datetime
    ) -> Optional[datetime]:
        """
        Query for timestamp of last message in segment.

        Args:
            continuum_id: Continuum UUID
            user_id: User UUID
            segment_start_time: When segment started (sentinel creation)

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
                    AND created_at > %s
                    AND (metadata->>'is_segment_boundary' IS NULL
                         OR metadata->>'is_segment_boundary' = 'false')
                    AND (metadata->>'system_notification' IS NULL
                         OR metadata->>'system_notification' = 'false')
                ORDER BY created_at DESC
                LIMIT 1
            """, (continuum_id, user_id, segment_start_time))

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

        # Query for last message time (same as timeout check)
        end_time = self._get_last_message_time(
            segment['continuum_id'],
            segment['user_id'],
            segment['created_at']
        )

        if not end_time:
            # No messages found — use sentinel creation time (same fallback as _is_timed_out)
            end_time = segment['created_at']

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
