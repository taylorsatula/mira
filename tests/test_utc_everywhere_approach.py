"""
Tests for the UTC-everywhere approach integration.

These tests validate the end-to-end behavior of the UTC-everywhere datetime
handling approach, focusing on integration between different components.
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import json
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from utils.timezone_utils import (
    utc_now,
    ensure_utc,
    convert_to_timezone,
    convert_to_utc,
    format_datetime,
    parse_time_string
)
from utils.db_datetime_utils import (
    UTCDatetimeMixin,
    utc_datetime_column,
    serialize_model_datetime
)


# Create Base and engine for testing
TestBase = declarative_base()
test_engine = create_engine('sqlite:///:memory:')
TestSession = sessionmaker(bind=test_engine)


class TestEvent(UTCDatetimeMixin, TestBase):
    """Test event model for integration tests."""
    __tablename__ = 'test_events'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    event_time = Column('event_time', utc_datetime_column(nullable=False))
    end_time = Column('end_time', utc_datetime_column(nullable=True))
    
    def __repr__(self):
        return f"<TestEvent(id={self.id}, name={self.name}, event_time={self.event_time})>"


class TestUTCEverywhereIntegration(unittest.TestCase):
    """Integration tests for the UTC-everywhere approach."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test case."""
        # Create tables
        TestBase.metadata.create_all(test_engine)
    
    @classmethod
    def tearDownClass(cls):
        """Tear down test fixtures for the entire test case."""
        # Drop tables
        TestBase.metadata.drop_all(test_engine)

    def setUp(self):
        """Set up test fixtures."""
        # Create a session
        self.session = TestSession()
        
        # Clear any existing data
        self.session.query(TestEvent).delete()
        self.session.commit()
        
        # Set up sample data
        self.event_utc = TestEvent(
            id="event1",
            name="UTC Event",
            event_time=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2023, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
        )
        
        # Create an event with non-UTC timezone that should be converted to UTC
        ny_tz = timezone(timedelta(hours=-5))
        self.event_ny = TestEvent(
            id="event2",
            name="NY Event",
            event_time=datetime(2023, 1, 1, 7, 0, 0, tzinfo=ny_tz),  # 7am NY = 12pm UTC
            end_time=datetime(2023, 1, 1, 9, 0, 0, tzinfo=ny_tz)     # 9am NY = 2pm UTC
        )
        
        # Add to session
        self.session.add_all([self.event_utc, self.event_ny])
        self.session.commit()

    def tearDown(self):
        """Tear down test fixtures."""
        self.session.close()

    def test_utc_storage(self):
        """Test that all datetimes are stored in UTC."""
        # Verify both events were stored with UTC datetimes
        event1 = self.session.query(TestEvent).filter_by(id="event1").first()
        event2 = self.session.query(TestEvent).filter_by(id="event2").first()
        
        # Check event times
        self.assertEqual(event1.event_time.tzinfo, timezone.utc)
        self.assertEqual(event1.event_time.hour, 12)
        
        self.assertEqual(event2.event_time.tzinfo, timezone.utc)
        self.assertEqual(event2.event_time.hour, 12)  # 7am NY = 12pm UTC
        
        # Check end times
        self.assertEqual(event1.end_time.tzinfo, timezone.utc)
        self.assertEqual(event1.end_time.hour, 14)
        
        self.assertEqual(event2.end_time.tzinfo, timezone.utc)
        self.assertEqual(event2.end_time.hour, 14)   # 9am NY = 2pm UTC

    def test_local_time_display(self):
        """Test conversion to local time for display."""
        # Get events from database
        event1 = self.session.query(TestEvent).filter_by(id="event1").first()
        event2 = self.session.query(TestEvent).filter_by(id="event2").first()
        
        # Convert to LA timezone for display
        la_timezone = "America/Los_Angeles"
        
        event1_la_time = convert_to_timezone(event1.event_time, la_timezone)
        event2_la_time = convert_to_timezone(event2.event_time, la_timezone)
        
        # In January, LA is UTC-8, so 12pm UTC = 4am LA
        self.assertEqual(event1_la_time.hour, 4)
        self.assertEqual(event2_la_time.hour, 4)
        
        # Format for display
        event1_formatted = format_datetime(event1.event_time, "date_time", la_timezone)
        event2_formatted = format_datetime(event2.event_time, "date_time", la_timezone)
        
        self.assertEqual(event1_formatted, "2023-01-01 04:00:00")
        self.assertEqual(event2_formatted, "2023-01-01 04:00:00")

    def test_date_arithmetic(self):
        """Test date arithmetic with UTC datetimes."""
        # Get event from database
        event = self.session.query(TestEvent).filter_by(id="event1").first()
        
        # Add one day to event time
        new_time = event.event_time + timedelta(days=1)
        
        # Ensure timezone is preserved
        self.assertEqual(new_time.tzinfo, timezone.utc)
        self.assertEqual(new_time.day, 2)  # Jan 2
        self.assertEqual(new_time.hour, 12)
        
        # Update event
        event.event_time = new_time
        self.session.commit()
        
        # Reload and verify
        updated_event = self.session.query(TestEvent).filter_by(id="event1").first()
        self.assertEqual(updated_event.event_time.day, 2)
        self.assertEqual(updated_event.event_time.tzinfo, timezone.utc)

    def test_date_comparisons(self):
        """Test date comparisons with UTC datetimes."""
        # Get events from database
        events = self.session.query(TestEvent).all()
        
        # Test equal start times
        self.assertEqual(events[0].event_time, events[1].event_time)
        
        # Test equal end times
        self.assertEqual(events[0].end_time, events[1].end_time)
        
        # Add filter for events after a certain time
        test_time = datetime(2023, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        
        # Filter for events that end after test_time
        future_events = self.session.query(TestEvent).filter(
            TestEvent.end_time > test_time
        ).all()
        
        # Both events should match (both end at 14:00 UTC)
        self.assertEqual(len(future_events), 2)
        
        # Filter for events that start before test_time
        past_events = self.session.query(TestEvent).filter(
            TestEvent.event_time < test_time
        ).all()
        
        # Both events should match (both start at 12:00 UTC)
        self.assertEqual(len(past_events), 2)

    def test_parsing_local_time_strings(self):
        """Test parsing local time strings to UTC datetimes."""
        # Parse a NY time string
        ny_time_str = "2023-01-01T07:00:00-05:00"  # 7am NY = 12pm UTC
        utc_dt = parse_time_string(ny_time_str)
        
        # Verify converted to UTC
        self.assertEqual(utc_dt.tzinfo, timezone.utc)
        self.assertEqual(utc_dt.hour, 12)
        
        # Create a new event using the parsed time
        new_event = TestEvent(
            id="event3",
            name="Parsed Time Event",
            event_time=utc_dt,
            end_time=utc_dt + timedelta(hours=2)
        )
        
        # Add to session and commit
        self.session.add(new_event)
        self.session.commit()
        
        # Verify stored in UTC
        stored_event = self.session.query(TestEvent).filter_by(id="event3").first()
        self.assertEqual(stored_event.event_time.tzinfo, timezone.utc)
        self.assertEqual(stored_event.event_time.hour, 12)
        self.assertEqual(stored_event.end_time.hour, 14)

    def test_serialization_for_api(self):
        """Test serializing datetimes for API responses."""
        # Get event from database
        event = self.session.query(TestEvent).filter_by(id="event1").first()
        
        # Convert to dict
        event_dict = {
            "id": event.id,
            "name": event.name,
            "event_time": event.event_time,
            "end_time": event.end_time
        }
        
        # Serialize for different timezones
        ny_serialized = serialize_model_datetime(
            event_dict, 
            ["event_time", "end_time"], 
            "America/New_York"
        )
        
        la_serialized = serialize_model_datetime(
            event_dict, 
            ["event_time", "end_time"], 
            "America/Los_Angeles"
        )
        
        utc_serialized = serialize_model_datetime(
            event_dict, 
            ["event_time", "end_time"], 
            "UTC"
        )
        
        # NY is UTC-5, so 12pm UTC = 7am NY
        self.assertTrue("07:00:00" in ny_serialized["event_time"])
        # NY is UTC-5, so 2pm UTC = 9am NY
        self.assertTrue("09:00:00" in ny_serialized["end_time"])
        
        # LA is UTC-8, so 12pm UTC = 4am LA
        self.assertTrue("04:00:00" in la_serialized["event_time"])
        # LA is UTC-8, so 2pm UTC = 6am LA
        self.assertTrue("06:00:00" in la_serialized["end_time"])
        
        # UTC stays the same
        self.assertTrue("12:00:00" in utc_serialized["event_time"])
        self.assertTrue("14:00:00" in utc_serialized["end_time"])
        
        # Test JSON serialization
        try:
            json_str = json.dumps(ny_serialized)
            parsed = json.loads(json_str)
            self.assertEqual(parsed["event_time"], ny_serialized["event_time"])
        except Exception as e:
            self.fail(f"JSON serialization failed: {e}")


if __name__ == "__main__":
    unittest.main()