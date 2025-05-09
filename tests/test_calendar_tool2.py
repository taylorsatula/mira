"""Tests for the redesigned CalendarTool."""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import json
from datetime import datetime, timedelta

# Add the project root to Python path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from errors import ToolError, ErrorCode


class TestCalendarTool2(unittest.TestCase):
    """Test suite for the redesigned CalendarTool class."""

    def setUp(self):
        """Set up test fixtures and initialize the tool for each test."""
        # Patch the imports
        self.config_patcher = patch('tools.calendar_tool2.registry')
        self.mock_registry = self.config_patcher.start()
        
        # Configure mock registry
        self.mock_registry.get.return_value = MagicMock(
            enabled=True,
            default_date_range=7,
            calendars={
                'work': MagicMock(name='Work Calendar', url='https://work.example.com', type='ical'),
                'personal': MagicMock(name='Personal Calendar', url='https://personal.example.com', type='caldav',
                                    username='user@example.com', calendar_id='personal')
            },
            default_calendar_url='https://default.example.com/calendar.ics',
            cache_directory='data/calendar_tool',
            cache_duration=3600,
            max_events=100,
            default_event_duration=60
        )
        
        # Import the tool after patching the registry
        from tools.calendar_tool2 import CalendarTool
        
        # Create the tool instance
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            self.tool = CalendarTool()
            self.tool.logger = mock_logger

        # Sample valid inputs for testing
        self.valid_url = "https://caldav.example.com"
        self.valid_username = "user@example.com"
        self.valid_password = "password123"
        self.valid_calendar_id = "personal"
        self.valid_event_id = "event-12345"
        self.valid_summary = "Team Meeting"
        self.valid_start_time = "2025-04-15T10:00:00"
        self.valid_end_time = "2025-04-15T11:00:00"
        self.valid_description = "Weekly team sync"
        self.valid_location = "Conference Room B"
        self.valid_ical_url = "https://example.com/calendar.ics"
        
    def tearDown(self):
        """Clean up after each test."""
        self.config_patcher.stop()

    def test_initialization(self):
        """Test proper initialization of the CalendarTool."""
        self.tool.logger.info.assert_called_with("CalendarTool initialized")
    
    @patch('tools.calendar_tool2.ICalProvider')
    def test_run_list_all_events(self, mock_ical_provider_class):
        """Test the list_all_events action."""
        # Arrange
        mock_provider = MagicMock()
        mock_ical_provider_class.return_value = mock_provider
        
        mock_provider.get_events.return_value = {
            "events": [{"summary": "Test Event"}],
            "count": 1
        }
        
        # Act
        result = self.tool.run(
            action="list_all_events",
            start_date="2025-05-01",
            end_date="2025-05-07"
        )
        
        # Assert
        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(result["calendars"]), 1)
        self.assertEqual(result["start_date"], "2025-05-01")
        self.assertEqual(result["end_date"], "2025-05-07")
        self.assertIsInstance(result["total_events"], int)
    
    @patch('tools.calendar_tool2.ICalProvider')
    def test_run_read_ical_url(self, mock_ical_provider_class):
        """Test the read_ical_url action."""
        # Configure mock provider
        mock_provider = MagicMock()
        mock_ical_provider_class.return_value = mock_provider
        
        mock_provider.get_events.return_value = {
            "events": [{"summary": "Test Event"}],
            "count": 1
        }
        
        # Call the method
        result = self.tool.run(
            action="read_ical_url",
            ical_url=self.valid_ical_url,
            start_date="2025-05-01",
            end_date="2025-05-07"
        )
        
        # Verify the result structure
        self.assertTrue(result["success"])
        self.assertIn("events", result)
        self.assertEqual(result["count"], 1)
    
    @patch('tools.calendar_tool2.CalDAVProvider')
    def test_run_list_calendars(self, mock_caldav_provider_class):
        """Test the list_calendars action."""
        # Configure mock provider
        mock_provider = MagicMock()
        mock_caldav_provider_class.return_value = mock_provider
        
        mock_provider.list_calendars.return_value = {
            "calendars": [
                {"id": "personal", "display_name": "Personal Calendar"}
            ],
            "count": 1
        }
        
        # Call the method
        result = self.tool.run(
            action="list_calendars",
            url=self.valid_url,
            username=self.valid_username,
            password=self.valid_password
        )
        
        # Verify the result structure
        self.assertTrue(result["success"])
        self.assertIn("calendars", result)
        self.assertEqual(result["count"], 1)
    
    @patch('tools.calendar_tool2.CalDAVProvider')
    def test_run_list_events(self, mock_caldav_provider_class):
        """Test the list_events action."""
        # Configure mock provider
        mock_provider = MagicMock()
        mock_caldav_provider_class.return_value = mock_provider
        
        mock_provider.get_events.return_value = {
            "events": [{"summary": "Test Event"}],
            "count": 1
        }
        
        # Call the method
        result = self.tool.run(
            action="list_events",
            url=self.valid_url,
            username=self.valid_username,
            password=self.valid_password,
            calendar_id=self.valid_calendar_id,
            start_date="2025-05-01",
            end_date="2025-05-07"
        )
        
        # Verify the result structure
        self.assertTrue(result["success"])
        self.assertIn("events", result)
        self.assertEqual(result["count"], 1)
    
    @patch('tools.calendar_tool2.CalDAVProvider')
    def test_run_create_event(self, mock_caldav_provider_class):
        """Test the create_event action."""
        # Configure mock provider
        mock_provider = MagicMock()
        mock_caldav_provider_class.return_value = mock_provider
        
        mock_provider.create_event.return_value = {
            "event": {
                "event_id": self.valid_event_id,
                "summary": self.valid_summary
            }
        }
        
        # Call the method
        result = self.tool.run(
            action="create_event",
            url=self.valid_url,
            username=self.valid_username,
            password=self.valid_password,
            calendar_id=self.valid_calendar_id,
            summary=self.valid_summary,
            start_time=self.valid_start_time,
            end_time=self.valid_end_time,
            description=self.valid_description,
            location=self.valid_location
        )
        
        # Verify the result structure
        self.assertTrue(result["success"])
        self.assertIn("event", result)
        self.assertEqual(result["event"]["event_id"], self.valid_event_id)
    
    @patch('tools.calendar_tool2.CalDAVProvider')
    def test_run_delete_event(self, mock_caldav_provider_class):
        """Test the delete_event action."""
        # Configure mock provider
        mock_provider = MagicMock()
        mock_caldav_provider_class.return_value = mock_provider
        
        mock_provider.delete_event.return_value = {
            "event_id": self.valid_event_id,
            "deleted": True
        }
        
        # Call the method
        result = self.tool.run(
            action="delete_event",
            url=self.valid_url,
            username=self.valid_username,
            password=self.valid_password,
            calendar_id=self.valid_calendar_id,
            event_id=self.valid_event_id
        )
        
        # Verify the result structure
        self.assertTrue(result["success"])
        self.assertEqual(result["event_id"], self.valid_event_id)
    
    @patch('os.environ.get')
    def test_run_invalid_action(self, mock_environ_get):
        """Test run method with an invalid action."""
        # Set up the environment variables
        mock_environ_get.return_value = "password123"
        
        invalid_action = "invalid_action"
        
        # Set default_url and username to avoid validation errors
        self.mock_registry.get.return_value.default_url = "https://example.com"
        self.mock_registry.get.return_value.default_username = "testuser"
        
        with self.assertRaises(ToolError) as context:
            self.tool.run(action=invalid_action)
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        self.assertIn(f"Invalid action", str(context.exception))

    @patch('tools.calendar_tool2.ICalProvider')
    def test_run_list_all_events_specific_calendar(self, mock_ical_provider_class):
        """Test the list_all_events action with a specific calendar."""
        # Configure mock provider
        mock_provider = MagicMock()
        mock_ical_provider_class.return_value = mock_provider
        
        mock_provider.get_events.return_value = {
            "events": [{"summary": "Test Event"}],
            "count": 1
        }
        
        # Update the mock registry to have 'Work Calendar'
        self.mock_registry.get.return_value.calendars = {
            'work': MagicMock(name='Work Calendar', url='https://work.example.com', type='ical')
        }
        
        # Call the method
        result = self.tool.run(
            action="list_all_events",
            calendar_name="Work Calendar",
            start_date="2025-05-01",
            end_date="2025-05-07"
        )
        
        # Verify the result structure
        self.assertTrue(result["success"])
        self.assertEqual(len(result["calendars"]), 1)
        self.assertEqual(result["calendars"][0]["name"], "Work Calendar")

    @patch('os.environ.get')
    def test_invalid_parameters(self, mock_environ_get):
        """Test error handling with invalid parameters."""
        # Setup
        mock_environ_get.return_value = "password123"
        self.mock_registry.get.return_value.default_url = "https://example.com"
        self.mock_registry.get.return_value.default_username = "testuser"
        
        # Test missing required parameter for create_event
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                action="create_event", 
                url=self.valid_url,
                username=self.valid_username,
                password=self.valid_password,
                calendar_id=self.valid_calendar_id
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test invalid date format
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                action="list_all_events",
                start_date="05/01/2025"  # Non-ISO format
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)


if __name__ == '__main__':
    unittest.main()