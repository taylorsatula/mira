"""
Tests for timezone utility functions.

This module provides tests for the timezone utility functions to ensure
proper timezone handling and conversions.
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytz

from utils.timezone_utils import (
    validate_timezone,
    get_default_timezone,
    convert_to_timezone,
    format_datetime,
    parse_time_string,
)
from errors import ToolError


class TestTimezoneUtils(unittest.TestCase):
    """Test cases for timezone utility functions."""

    def test_validate_timezone_valid_iana(self):
        """Test validating a valid IANA timezone name."""
        # Test with a standard IANA timezone name
        result = validate_timezone("America/New_York")
        self.assertEqual(result, "America/New_York")

    def test_validate_timezone_common_abbreviation(self):
        """Test validating a common timezone abbreviation."""
        # Test with a common abbreviation
        result = validate_timezone("CST")
        self.assertEqual(result, "America/Chicago")
        
        result = validate_timezone("EST")
        self.assertEqual(result, "America/New_York")

    def test_validate_timezone_invalid(self):
        """Test validating an invalid timezone name."""
        # Test with an invalid timezone name
        with self.assertRaises(ToolError):
            validate_timezone("Invalid_Zone")

    def test_validate_timezone_empty(self):
        """Test validating an empty timezone name."""
        # Should return default timezone
        with patch('utils.timezone_utils.get_default_timezone') as mock_get_default:
            mock_get_default.return_value = "America/Los_Angeles"
            result = validate_timezone("")
            self.assertEqual(result, "America/Los_Angeles")

    def test_get_default_timezone(self):
        """Test getting the default system timezone."""
        # Test with a valid IANA timezone in config
        with patch('utils.timezone_utils.config') as mock_config:
            mock_config.system.timezone = "America/Chicago"
            result = get_default_timezone()
            self.assertEqual(result, "America/Chicago")
            
        # Test with an abbreviation in config
        with patch('utils.timezone_utils.config') as mock_config:
            mock_config.system.timezone = "EST"
            result = get_default_timezone()
            self.assertEqual(result, "America/New_York")
            
        # Test with an invalid timezone in config
        with patch('utils.timezone_utils.config') as mock_config:
            mock_config.system.timezone = "Invalid_Zone"
            with patch('utils.timezone_utils.logger') as mock_logger:
                result = get_default_timezone()
                self.assertEqual(result, "UTC")
                mock_logger.warning.assert_called_once()

    def test_convert_to_timezone(self):
        """Test converting a datetime between timezones."""
        # Create a UTC datetime
        dt_utc = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Convert to America/New_York (EST, UTC-5)
        dt_est = convert_to_timezone(dt_utc, "America/New_York")
        
        # Check timezone
        self.assertEqual(dt_est.tzinfo.zone, "America/New_York")
        
        # Check hour (should be 07:00 in EST)
        self.assertEqual(dt_est.hour, 7)
        
        # Original should be unchanged
        self.assertEqual(dt_utc.hour, 12)
        
        # Test with a naive datetime
        dt_naive = datetime(2023, 1, 1, 12, 0, 0)
        dt_est = convert_to_timezone(dt_naive, "America/New_York", "UTC")
        
        # Check timezone
        self.assertEqual(dt_est.tzinfo.zone, "America/New_York")
        
        # Check hour (should be 07:00 in EST)
        self.assertEqual(dt_est.hour, 7)

    def test_format_datetime(self):
        """Test formatting a datetime in a specific timezone."""
        # Create a UTC datetime
        dt_utc = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Format in default format (standard time)
        result = format_datetime(dt_utc, "standard", "America/New_York")
        self.assertEqual(result, "07:00:00")
        
        # Format in short format
        result = format_datetime(dt_utc, "short", "America/New_York")
        self.assertEqual(result, "07:00")
        
        # Format in date_time format
        result = format_datetime(dt_utc, "date_time", "America/New_York")
        self.assertEqual(result, "2023-01-01 07:00:00")
        
        # Test including timezone name
        result = format_datetime(dt_utc, "short", "America/New_York", include_timezone=True)
        self.assertEqual(result, "07:00 America/New_York")

    def test_parse_time_string(self):
        """Test parsing a time string into a datetime object."""
        # Test parsing ISO format
        dt = parse_time_string("2023-01-01T12:00:00", "UTC")
        self.assertEqual(dt.year, 2023)
        self.assertEqual(dt.month, 1)
        self.assertEqual(dt.day, 1)
        self.assertEqual(dt.hour, 12)
        self.assertEqual(dt.minute, 0)
        self.assertEqual(dt.second, 0)
        self.assertEqual(dt.tzinfo.zone, "UTC")
        
        # Test parsing time-only format
        # Need a reference date for testing
        reference = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        dt = parse_time_string("14:30:00", "America/New_York", reference)
        self.assertEqual(dt.hour, 14)
        self.assertEqual(dt.minute, 30)
        self.assertEqual(dt.second, 0)
        self.assertEqual(dt.tzinfo.zone, "America/New_York")
        
        # Test invalid format
        with self.assertRaises(ToolError):
            parse_time_string("invalid", "UTC")


class TestTimezoneIntegration(unittest.TestCase):
    """Integration tests for timezone handling in automation models."""

    def setUp(self):
        """Set up test fixtures."""
        # Patch the database since we don't need actual db interactions
        self.db_patcher = patch('task_manager.automation.Database')
        self.mock_db = self.db_patcher.start()
        
        # Import the models here to avoid circular imports during test discovery
        from task_manager.automation import Automation, AutomationType, TaskFrequency
        
        # Create a sample automation with specified timezone
        self.automation = Automation(
            id="test_auto_1",
            name="Test Automation",
            type=AutomationType.SIMPLE_TASK,
            frequency=TaskFrequency.DAILY,
            scheduled_time=datetime(2023, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
            timezone="America/Los_Angeles"
        )

    def tearDown(self):
        """Tear down test fixtures."""
        self.db_patcher.stop()

    def test_automation_to_dict_timezone_conversion(self):
        """Test that Automation.to_dict() correctly converts timestamps to user timezone."""
        # Get the dict representation
        auto_dict = self.automation.to_dict()
        
        # The UTC time was 14:00, which is 06:00 in Los Angeles (PST/PDT is UTC-8/UTC-7)
        # Extract just the time part from the ISO string
        time_part = auto_dict["scheduled_time"].split("T")[1][:5]  # Get HH:MM
        
        # Verify the time was converted to Los Angeles time (should be 06:00)
        self.assertEqual(time_part, "06:00")

    def test_execution_to_dict_timezone_conversion(self):
        """Test that AutomationExecution.to_dict() correctly converts timestamps."""
        from task_manager.automation import AutomationExecution, ExecutionStatus, TriggerType
        
        # Create an execution with scheduled and start times in UTC
        execution = AutomationExecution(
            id="test_exec_1",
            automation_id=self.automation.id,
            automation=self.automation,  # Mock the relationship
            status=ExecutionStatus.COMPLETED,
            trigger_type=TriggerType.SCHEDULED,
            scheduled_time=datetime(2023, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
            started_at=datetime(2023, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2023, 1, 1, 14, 5, 0, tzinfo=timezone.utc)
        )
        
        # Get the dict representation
        exec_dict = execution.to_dict()
        
        # Extract start time
        start_time_part = exec_dict["started_at"].split("T")[1][:5]  # Get HH:MM
        
        # Verify the time was converted to Los Angeles time (should be 06:00)
        self.assertEqual(start_time_part, "06:00")


if __name__ == "__main__":
    unittest.main()