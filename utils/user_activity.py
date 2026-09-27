"""
User activity tracking - engagement metrics and login tracking.

Handles user engagement data including:
- Activity day tracking (first message of day detection)
- Cumulative activity days (vacation-proof user engagement metric)
- Login timestamp tracking
- Granular activity logging for analytics

This module owns user engagement concerns separate from authentication.
"""
import logging
import pytz

from utils.timezone_utils import utc_now
from utils.user_context import get_current_user, update_current_user, get_user_preferences
from utils.database_session_manager import get_shared_session_manager

logger = logging.getLogger(__name__)


def increment_user_activity_day(user_id: str) -> int:
    """
    Increment user's cumulative activity day count if first message of their day.

    ⚠️  FIRST MESSAGE OF DAY HOOK POINT ⚠️
    This method detects the first message a user sends in their local day.
    Insert beginning-of-day actions here (morning summaries, daily notifications, etc).

    Uses user's local timezone to determine day boundaries, so "first message of day"
    is semantically correct regardless of where the user is or when they message.

    Rapidpath optimization: Context caching ensures only the first call per
    user-local day per session hits DB.

    Args:
        user_id: User ID to increment

    Returns:
        Updated cumulative_activity_days count

    Raises:
        ValueError: If user not found
    """
    # Get user's local date (not server date)
    user_tz = pytz.timezone(get_user_preferences().timezone)
    user_local_date = utc_now().astimezone(user_tz).date()

    # Rapidpath: already incremented for THIS user-local date in this context.
    # The cache carries the local date it was claimed for and is compared on
    # read, so a long-lived context crossing the user's local midnight
    # re-arms the fast path instead of skipping every subsequent day. The
    # date predicate in the UPDATE below stays the sole authority for the
    # claim; a stale cache only degrades to one DB round-trip.
    user_data = get_current_user()
    if user_data.get('_activity_day_incremented_today') == user_local_date:
        return user_data.get('cumulative_activity_days', 0)

    session_manager = get_shared_session_manager()
    with session_manager.get_session(user_id) as session:
        # Atomic first-of-day claim: the date predicate in the UPDATE itself
        # is the guard, so concurrent first-of-day callers increment exactly once.
        # last_login_at is updated in the same statement: opening a second
        # connection here would self-deadlock on this transaction's row lock.
        rowcount = session.execute_update("""
            UPDATE users
            SET cumulative_activity_days = cumulative_activity_days + 1,
                last_activity_date = %(activity_date)s,
                last_login_at = %(login_time)s
            WHERE id = %(user_id)s
              AND (last_activity_date IS NULL OR last_activity_date < %(activity_date)s)
        """, {
            'user_id': user_id,
            'activity_date': user_local_date,
            'login_time': utc_now()
        })

        current_user = session.execute_single("""
            SELECT cumulative_activity_days, last_activity_date
            FROM users
            WHERE id = %(user_id)s
        """, {'user_id': user_id})

        if not current_user:
            raise ValueError(f"User {user_id} not found for activity day increment")

        current_days = current_user.get('cumulative_activity_days', 0) or 0

        if rowcount == 0:
            # Already counted today - rapidpath for subsequent messages
            session.execute_update("""
                INSERT INTO user_activity_days (user_id, activity_date, first_message_at, message_count)
                VALUES (%(user_id)s, %(activity_date)s, %(timestamp)s, 1)
                ON CONFLICT (user_id, activity_date)
                DO UPDATE SET message_count = user_activity_days.message_count + 1
            """, {
                'user_id': user_id,
                'activity_date': user_local_date,
                'timestamp': utc_now()
            })

        else:
            # ========================================================================
            # 🌅 FIRST MESSAGE OF USER'S DAY - Insert daily actions here
            # ========================================================================

            logger.info(f"First message of day for user {user_id} (local date: {user_local_date})")

            # ========================================================================

            # Track in granular table
            session.execute_update("""
                INSERT INTO user_activity_days (user_id, activity_date, first_message_at, message_count)
                VALUES (%(user_id)s, %(activity_date)s, %(timestamp)s, 1)
                ON CONFLICT (user_id, activity_date)
                DO UPDATE SET message_count = user_activity_days.message_count + 1
            """, {
                'user_id': user_id,
                'activity_date': user_local_date,
                'timestamp': utc_now()
            })

    # Cache for rapidpath. Set only after the session exits: __exit__ commits
    # the transaction and re-raises on commit failure, so the in-request cache
    # can never claim an increment that did not persist.
    # Same write refreshes the sibling cache: user_context's
    # get_user_cumulative_activity_days serves 'cumulative_activity_days'
    # from this same shared context dict. update_current_user mutates that
    # dict in place, so every contextvar copy holding it observes both keys.
    update_current_user({
        'cumulative_activity_days': current_days,
        '_activity_day_incremented_today': user_local_date
    })

    logger.debug(f"User {user_id} activity day incremented to {current_days}")
    return current_days


def get_user_cumulative_activity_days(user_id: str) -> int:
    """
    Get user's cumulative activity days count.

    Args:
        user_id: User ID to query

    Returns:
        Cumulative activity days (0 if no activity)

    Raises:
        ValueError: If user not found
    """
    session_manager = get_shared_session_manager()
    with session_manager.get_session(user_id) as session:
        result = session.execute_single("""
            SELECT cumulative_activity_days
            FROM users
            WHERE id = %(user_id)s
        """, {'user_id': user_id})

        if not result:
            raise ValueError(f"User {user_id} not found")

        return result.get('cumulative_activity_days', 0) or 0
