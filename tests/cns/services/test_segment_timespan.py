"""
Test that segment summaries include timespan information.

This test verifies the feature where segment summaries injected into the
context window include the time range (start and end times) of the segment.
"""
from datetime import datetime, timedelta

from cns.services.segment_helpers import (
    create_segment_boundary_sentinel,
    collapse_segment_sentinel
)
from utils.timezone_utils import utc_now


def test_collapsed_segment_includes_timespan():
    """CONTRACT: Collapsed segment summary includes timespan in formatted output."""
    # Create a segment with specific start time
    start_time = utc_now()
    sentinel = create_segment_boundary_sentinel(start_time, "cid")

    # Simulate segment ending 30 minutes later
    end_time = start_time + timedelta(minutes=30)

    # Collapse the segment with end time
    summary = "Discussion about Python testing"
    display_title = "Python testing discussion"
    collapsed = collapse_segment_sentinel(
        sentinel=sentinel,
        summary=summary,
        display_title=display_title,
        embedding=None,
        inactive_duration_minutes=60,
        segment_end_time=end_time
    )

    # Verify the formatted content includes timespan
    content = collapsed.content

    # Should contain the display title
    assert display_title in content, f"Expected '{display_title}' in content"

    # Should contain the summary
    assert summary in content, f"Expected summary in content"

    # Should contain timespan information
    assert "Timespan:" in content, "Expected 'Timespan:' label in content"
    assert " to " in content, "Expected ' to ' separator in timespan"

    # Verify the structure matches expected format
    assert content.startswith("This is an extended summary of:"), \
        "Expected content to start with standard prefix"


def test_collapsed_segment_handles_missing_timespan_gracefully():
    """CONTRACT: Collapsed segment works even without explicit timespan."""
    # Create segment without setting explicit end time
    start_time = utc_now()
    sentinel = create_segment_boundary_sentinel(start_time, "cid")

    # Collapse without providing segment_end_time
    summary = "Quick conversation"
    display_title = "Quick chat"
    collapsed = collapse_segment_sentinel(
        sentinel=sentinel,
        summary=summary,
        display_title=display_title,
        embedding=None,
        inactive_duration_minutes=60
        # segment_end_time not provided
    )

    # Should still contain basic information even if timespan isn't available
    content = collapsed.content
    assert display_title in content
    assert summary in content


def test_timespan_metadata_persists_in_collapsed_sentinel():
    """CONTRACT: Segment metadata retains start/end times after collapse."""
    start_time = utc_now()
    end_time = start_time + timedelta(hours=1)

    sentinel = create_segment_boundary_sentinel(start_time, "cid")

    collapsed = collapse_segment_sentinel(
        sentinel=sentinel,
        summary="Summary",
        display_title="Title",
        embedding=None,
        inactive_duration_minutes=60,
        segment_end_time=end_time
    )

    # Metadata should contain the timestamps
    assert 'segment_start_time' in collapsed.metadata
    assert 'segment_end_time' in collapsed.metadata

    # Timestamps should be ISO format strings
    start_iso = collapsed.metadata['segment_start_time']
    end_iso = collapsed.metadata['segment_end_time']

    # Should be parseable as ISO format
    parsed_start = datetime.fromisoformat(start_iso)
    parsed_end = datetime.fromisoformat(end_iso)

    # Should match our input times (within second precision)
    assert abs((parsed_start - start_time).total_seconds()) < 1
    assert abs((parsed_end - end_time).total_seconds()) < 1
