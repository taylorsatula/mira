"""
Sentinel construction and display formatting for conversation segments.

A segment is a sentinel message in `messages` carrying
`metadata.is_segment_boundary = True`. `create_segment_boundary_sentinel` writes
the opening half of that metadata (status, `segment_id`, turn counters,
`last_turn_at`, `tools_used`) and `collapse_segment_sentinel` writes the
collapsed half (summary in content, `precis`, `display_title`, complexity,
embedding). This module is the vocabulary's writer: the timeout sweep, the
manifest query, the tier selection in `cns/core/segment_cache_loader.py`, and
the display formatters are its readers, so a new metadata key lands here and in
every reader in the same commit.

A sentinel's `created_at` is the first message's timestamp minus one
microsecond, so chronological ordering places the boundary before the messages
it brackets without relying on insertion order.
"""
import logging
from typing import List, Optional
from datetime import datetime, timedelta
from uuid import uuid4

from cns.core.message import Message
from utils.timezone_utils import utc_now, parse_utc_time_string

logger = logging.getLogger(__name__)


def create_segment_boundary_sentinel(
    first_message_time: datetime,
    continuum_id: str,
    segment_id: str | None = None,
) -> Message:
    """
    Create segment boundary sentinel message.

    Sentinel is created in 'active' status when the first message arrives.
    Summary and embedding are added during collapse.

    Args:
        first_message_time: Timestamp of first message in segment
        continuum_id: Continuum UUID (for reference)
        segment_id: Pre-allocated segment ID. When provided (via
            set_current_segment_id contextvar), the persisted sentinel uses
            the same ID tools saw this turn. When None, a fresh uuid4 is
            allocated.

    Returns:
        Message object with segment boundary metadata
    """
    allocated_id = segment_id or str(uuid4())
    metadata = {
        'is_segment_boundary': True,
        'status': 'active',
        'segment_id': allocated_id,
        'segment_start_time': first_message_time.isoformat(),
        'segment_end_time': first_message_time.isoformat(),
        'segment_turn_count': 1,  # Initialized to 1 since creation happens on first turn
        'last_turn_at': first_message_time.isoformat(),
        'tools_used': [],
        'memories_extracted': False,
        'domain_blocks_updated': False
    }

    # -1µs: created_at must strictly precede the first message so chronological
    # queries place the boundary marker before the messages it brackets, without
    # relying on insertion order.
    sentinel_time = first_message_time - timedelta(microseconds=1)

    sentinel = Message(
        content="[Segment in progress]",
        role="assistant",
        metadata=metadata,
        created_at=sentinel_time,
    )

    logger.info(
        f"Created segment boundary sentinel {metadata['segment_id']} "
        f"for continuum {continuum_id}"
    )

    return sentinel


def collapse_segment_sentinel(
    sentinel: Message,
    summary: str,
    precis: str,
    display_title: str,
    embedding: list[float],
    inactive_duration_minutes: int,
    processing_failed: bool = False,
    tools_used: list[str] = (),
    segment_end_time: Optional[datetime] = None,
    complexity_score: float = 2
) -> Message:
    """
    Collapse segment sentinel with summary and embedding.

    Returns new Message with collapsed state (Message is immutable).

    Args:
        sentinel: Segment boundary sentinel message
        summary: Generated telegraphic summary
        precis: 2-sentence compressed summary
        display_title: Short telegraphic title for manifest display
        embedding: Document embedding of the summary
        inactive_duration_minutes: Minutes of inactivity that triggered collapse
        processing_failed: True if summary generation failed and fallback was used
        tools_used: Tools used in segment (extracted from messages)
        segment_end_time: Timestamp of last message in segment
        complexity_score: Cognitive complexity score (0.5=trivial, 1=simple, 2=moderate, 3=complex)

    Returns:
        New Message with status='collapsed' and summary in content
    """
    collapsed_metadata = {
        **sentinel.metadata,
        'status': 'collapsed',
        'collapsed_at': utc_now().isoformat(),
        'inactive_duration_minutes': inactive_duration_minutes,
        'summary_generated_at': utc_now().isoformat(),
        'processing_failed': processing_failed,
        'precis': precis,
        'display_title': display_title,
        'complexity_score': complexity_score
    }

    if tools_used is not None:
        collapsed_metadata['tools_used'] = tools_used

    if segment_end_time is not None:
        collapsed_metadata['segment_end_time'] = segment_end_time.isoformat()

    # Store embedding in metadata for repository to extract and persist
    collapsed_metadata['segment_embedding_value'] = embedding
    collapsed_metadata['has_segment_embedding'] = True

    logger.info(
        f"Collapsed segment {sentinel.metadata['segment_id']}: "
        f"inactive_duration={inactive_duration_minutes}min, "
        f"tools_used={tools_used}, "
        f"summary='{summary[:50]}...'"
    )

    # Return new Message with clean summary - display formatting happens at display time
    return Message(
        id=sentinel.id,
        content=summary,
        role=sentinel.role,
        created_at=sentinel.created_at,
        metadata=collapsed_metadata
    )


def _format_segment_display(sentinel: Message, label: str, body: str) -> str:
    """Format a collapsed segment with label, timespan, and body content.

    Shared implementation for extended and precis display modes.

    Args:
        sentinel: Collapsed segment sentinel
        label: Display header label (e.g. "THIS IS AN EXTENDED SUMMARY OF")
        body: Content body (synopsis or precis text)

    Returns:
        Formatted display string

    Raises:
        KeyError: If required metadata (display_title, segment_start_time) is missing
    """
    from utils.timezone_utils import format_relative_time

    display_title = sentinel.metadata['display_title']
    start_time = parse_utc_time_string(sentinel.metadata['segment_start_time'])
    relative_time = format_relative_time(start_time)

    return f"{label}: {display_title}\nTIMESPAN: {relative_time}\n\n{body}"


def format_segment_for_display(sentinel: Message) -> str:
    """Format collapsed segment's full synopsis for display in working memory."""
    return _format_segment_display(sentinel, "THIS IS AN EXTENDED SUMMARY OF", sentinel.content)


def format_precis_for_display(sentinel: Message) -> str:
    """Format collapsed segment's 2-sentence precis for compact display in session cache Tier 2."""
    return _format_segment_display(sentinel, "PRECIS OF", sentinel.metadata['precis'])


def create_collapse_marker() -> Message:
    """
    Create collapse marker for segment display.

    This marker appears between collapsed segment summaries and recent messages,
    indicating older content has been compressed.

    Returns:
        Message with collapse_marker notification type
    """
    return Message(
        content='<mira:notification type="collapse_marker">Older messages and summaries available through search. Use continuum_tool to find specific information from past conversations.</mira:notification>',
        role="assistant",
        metadata={'system_notification': True, 'notification_type': 'collapse_marker'}
    )


def create_session_boundary_marker(segment_summaries: List[Message]) -> Message:
    """
    Create session boundary marker showing time gap between sessions.

    The boundary shows when the previous session ended (based on collapsed segments)
    and when the current session is beginning.

    Args:
        segment_summaries: Collapsed segment summaries from previous sessions

    Returns:
        Message with session_break notification type
    """
    from utils.timezone_utils import convert_from_utc, format_datetime
    from utils.user_context import get_user_preferences

    current_time = utc_now()

    if segment_summaries:
        last_segment = segment_summaries[-1]
        end_time_str = last_segment.metadata.get('segment_end_time')

        if end_time_str:
            last_session_end = parse_utc_time_string(end_time_str)
        else:
            last_session_end = last_segment.created_at
    else:
        user_tz = get_user_preferences().timezone

        current_time_user_tz = convert_from_utc(current_time, user_tz)
        current_time_str = format_datetime(current_time_user_tz, "time_only", include_timezone=True)

        return Message(
            content=f'<mira:notification type="session_break">This chat session began at {current_time_str}</mira:notification>',
            role="assistant",
            metadata={'system_notification': True, 'notification_type': 'session_break'}
        )

    user_tz = get_user_preferences().timezone

    last_time_user_tz = convert_from_utc(last_session_end, user_tz)
    current_time_user_tz = convert_from_utc(current_time, user_tz)

    last_time_str = format_datetime(last_time_user_tz, "time_only", include_timezone=True)
    current_time_str = format_datetime(current_time_user_tz, "time_only", include_timezone=True)

    gap = current_time - last_session_end
    gap_hours = int(gap.total_seconds() / 3600)
    gap_days = gap_hours // 24
    remaining_hours = gap_hours % 24

    if gap_days > 0:
        if remaining_hours > 0:
            gap_str = f"{gap_days} day{'s' if gap_days != 1 else ''} {remaining_hours} hour{'s' if remaining_hours != 1 else ''}"
        else:
            gap_str = f"{gap_days} day{'s' if gap_days != 1 else ''}"
    else:
        gap_str = f"{gap_hours} hour{'s' if gap_hours != 1 else ''}"

    boundary_content = (
        f'<mira:notification type="session_break">'
        f'Last chat session ended at {last_time_str}. '
        f'This chat session began at {current_time_str}. '
        f'Gap of {gap_str}.'
        f'</mira:notification>'
    )

    return Message(
        content=boundary_content,
        role="assistant",
        metadata={'system_notification': True, 'notification_type': 'session_break'}
    )
