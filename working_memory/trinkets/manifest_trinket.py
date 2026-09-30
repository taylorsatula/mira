"""Manifest trinket for displaying conversation segment manifest in system prompt."""
import logging
from collections import OrderedDict
from datetime import timedelta
from typing import Dict, Any, Optional

from cns.services.manifest_query_service import ManifestSegment
from .base import EventAwareTrinket
from utils.timezone_utils import utc_now, convert_from_utc, parse_utc_time_string, strftime_no_pad
from utils.user_context import get_current_user_id, get_user_preferences

logger = logging.getLogger(__name__)


class ManifestTrinket(EventAwareTrinket):
    """
    Displays conversation manifest in working memory.

    This trinket formats the segment-based conversation manifest
    into a structured XML section for the system prompt, showing recent
    conversation segments organized by time. Placed with stable orientation
    content rather than the HUD since it rarely changes mid-turn.
    """

    variable_name = "conversation_manifest"

    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate manifest content from segment boundaries.

        Returns:
            Formatted manifest section or empty string if no segments

        Raises:
            DatabaseError: If database query fails (infrastructure failure)
        """
        from cns.services.manifest_query_service import get_manifest_query_service

        user_id = get_current_user_id()

        # Get raw segment data from service (handles caching)
        segments = get_manifest_query_service().get_segments(user_id)

        if not segments:
            logger.debug("No segments available for manifest")
            return ""  # Legitimately empty - no conversation history yet

        # Format segments into XML
        manifest = self._format_manifest(segments)

        logger.debug(f"Generated manifest for user {user_id}")
        return manifest

    def _format_manifest(self, segments: list[ManifestSegment]) -> str:
        """
        Format segments as XML grouped by relative time.

        Args:
            segments: List of segment dictionaries

        Returns:
            Formatted XML manifest string
        """
        if not segments:
            return ""

        # Get user's timezone for local time display
        try:
            user_tz = get_user_preferences().timezone
        except Exception:
            user_tz = 'UTC'

        # Group segments by date
        grouped = self._group_segments_by_date(segments, user_tz)

        # Build XML structure
        lines = ["<conversation_manifest>"]

        for date_label, date_segments in grouped.items():
            lines.append(f'<date_group label="{date_label}">')

            # Format segments within date group
            for segment in date_segments:
                # Format time range
                time_range = self._format_time_range(
                    segment['start_time'],
                    segment['end_time'],
                    segment['status'],
                    user_tz
                )

                # Format segment as XML element
                display_title = segment['display_title']
                status = segment['status']
                lines.append(f'<segment time="{time_range}" status="{status}">{display_title}</segment>')

            lines.append("</date_group>")

        lines.append("</conversation_manifest>")
        return "\n".join(lines)

    def _group_segments_by_date(
        self,
        segments: list[ManifestSegment],
        timezone: str
    ) -> dict[str, list[ManifestSegment]]:
        """
        Group segments by relative date (Today, Yesterday, or date string).

        Args:
            segments: List of segment dictionaries
            timezone: User's timezone for date calculation

        Returns:
            Ordered dict mapping date labels to segment lists
        """
        now = utc_now()
        now_local = convert_from_utc(now, timezone)
        today = now_local.date()
        yesterday = today - timedelta(days=1)

        grouped = OrderedDict()

        for segment in reversed(segments):  # Process newest first for date grouping
            # Parse segment start time
            try:
                if segment['start_time']:
                    start_time_utc = parse_utc_time_string(segment['start_time'])
                    start_time_local = convert_from_utc(start_time_utc, timezone)
                    segment_date = start_time_local.date()
                else:
                    segment_date = today
            except Exception:
                segment_date = today

            # Determine date label
            if segment_date == today:
                date_label = "TODAY"
            elif segment_date == yesterday:
                date_label = "YESTERDAY"
            else:
                # Format as "JAN 18" or similar
                date_label = segment_date.strftime("%b %d").upper()

            # Add to grouped dict
            if date_label not in grouped:
                grouped[date_label] = []
            grouped[date_label].insert(0, segment)  # Insert at beginning for chronological order

        return grouped

    def _format_time_range(
        self,
        start_time_str: Optional[str],
        end_time_str: Optional[str],
        status: str,
        timezone: str
    ) -> str:
        """
        Format time range for segment display.

        Args:
            start_time_str: ISO format start time
            end_time_str: ISO format end time
            status: Segment status (active/collapsed)
            timezone: User's timezone

        Returns:
            Formatted time range like "[2:15PM - ACTIVE]" or "[9:00AM - 10:15AM]"
        """
        try:
            if not start_time_str:
                return "[Unknown]"

            start_time_utc = parse_utc_time_string(start_time_str)
            start_time_local = convert_from_utc(start_time_utc, timezone)

            if status == 'active':
                # Active segment - show start time and "ACTIVE"
                start_str = strftime_no_pad(start_time_local, "%-I:%M%p").upper()
                return f"[{start_str} - ACTIVE]"
            else:
                # Collapsed segment - show time range
                if end_time_str:
                    end_time_utc = parse_utc_time_string(end_time_str)
                    end_time_local = convert_from_utc(end_time_utc, timezone)

                    start_str = strftime_no_pad(start_time_local, "%-I:%M%p").upper()
                    end_str = strftime_no_pad(end_time_local, "%-I:%M%p").upper()
                    return f"[{start_str} - {end_str}]"
                else:
                    start_str = strftime_no_pad(start_time_local, "%-I:%M%p").upper()
                    return f"[{start_str}]"

        except Exception as e:
            logger.warning(f"Failed to format time range: {e}")
            return "[Unknown]"
