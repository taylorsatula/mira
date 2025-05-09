import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta
import uuid
import caldav
import requests

from tools.calendar_tool import CalendarTool
from errors import ToolError, ErrorCode


class TestCalendarTool(unittest.TestCase):
    """Test suite for the CalendarTool class."""

    def setUp(self):
        """Set up test fixtures and initialize the tool for each test."""
        # Create a mocked tool instance
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

    def test_initialization(self):
        """Test proper initialization of the CalendarTool."""
        # Verify logger is initialized and used
        self.tool.logger.info.assert_called_once_with("CalendarTool initialized")

    def test_validate_common_params_valid(self):
        """Test validation of common parameters with valid inputs."""
        # This should not raise an exception
        self.tool._validate_common_params(
            self.valid_url, self.valid_username, self.valid_password
        )

    def test_validate_common_params_invalid(self):
        """Test validation of common parameters with invalid inputs."""
        invalid_inputs = [
            {"url": "", "username": self.valid_username, "password": self.valid_password},
            {"url": None, "username": self.valid_username, "password": self.valid_password},
            {"url": 123, "username": self.valid_username, "password": self.valid_password},
            {"url": self.valid_url, "username": "", "password": self.valid_password},
            {"url": self.valid_url, "username": None, "password": self.valid_password},
            {"url": self.valid_url, "username": 123, "password": self.valid_password},
            {"url": self.valid_url, "username": self.valid_username, "password": ""},
            {"url": self.valid_url, "username": self.valid_username, "password": None},
            {"url": self.valid_url, "username": self.valid_username, "password": 123},
        ]
        
        for inputs in invalid_inputs:
            with self.subTest(inputs=inputs):
                with self.assertRaises(ToolError) as context:
                    self.tool._validate_common_params(
                        inputs["url"], inputs["username"], inputs["password"]
                    )
                self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)

    def test_validate_calendar_id_valid(self):
        """Test validation of calendar ID with valid input."""
        # This should not raise an exception
        self.tool._validate_calendar_id(self.valid_calendar_id)

    def test_validate_calendar_id_invalid(self):
        """Test validation of calendar ID with invalid inputs."""
        invalid_inputs = ["", None, 123]
        
        for calendar_id in invalid_inputs:
            with self.subTest(calendar_id=calendar_id):
                with self.assertRaises(ToolError) as context:
                    self.tool._validate_calendar_id(calendar_id)
                self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)

    def test_validate_event_params_valid(self):
        """Test validation of event parameters with valid inputs."""
        # This should not raise an exception
        self.tool._validate_event_params(
            self.valid_summary, self.valid_start_time, self.valid_end_time
        )

    def test_validate_event_params_invalid(self):
        """Test validation of event parameters with invalid inputs."""
        invalid_inputs = [
            {"summary": "", "start_time": self.valid_start_time, "end_time": self.valid_end_time},
            {"summary": None, "start_time": self.valid_start_time, "end_time": self.valid_end_time},
            {"summary": 123, "start_time": self.valid_start_time, "end_time": self.valid_end_time},
            {"summary": self.valid_summary, "start_time": "", "end_time": self.valid_end_time},
            {"summary": self.valid_summary, "start_time": None, "end_time": self.valid_end_time},
            {"summary": self.valid_summary, "start_time": "invalid", "end_time": self.valid_end_time},
            {"summary": self.valid_summary, "start_time": self.valid_start_time, "end_time": ""},
            {"summary": self.valid_summary, "start_time": self.valid_start_time, "end_time": None},
            {"summary": self.valid_summary, "start_time": self.valid_start_time, "end_time": "invalid"},
            {"summary": self.valid_summary, "start_time": self.valid_end_time, "end_time": self.valid_start_time},
        ]
        
        for inputs in invalid_inputs:
            with self.subTest(inputs=inputs):
                with self.assertRaises(ToolError) as context:
                    self.tool._validate_event_params(
                        inputs["summary"], inputs["start_time"], inputs["end_time"]
                    )
                self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)

    def test_validate_event_id_valid(self):
        """Test validation of event ID with valid input."""
        # This should not raise an exception
        self.tool._validate_event_id(self.valid_event_id)

    def test_validate_event_id_invalid(self):
        """Test validation of event ID with invalid inputs."""
        invalid_inputs = ["", None, 123]
        
        for event_id in invalid_inputs:
            with self.subTest(event_id=event_id):
                with self.assertRaises(ToolError) as context:
                    self.tool._validate_event_id(event_id)
                self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)

    @patch('caldav.DAVClient')
    def test_get_client_success(self, mock_client_class):
        """Test successful client creation."""
        # Configure mock
        mock_client = mock_client_class.return_value
        mock_principal = MagicMock()
        mock_client.principal.return_value = mock_principal
        
        # Call method
        client = self.tool._get_client(
            self.valid_url, self.valid_username, self.valid_password
        )
        
        # Verify client was created with correct parameters
        mock_client_class.assert_called_once_with(
            url=self.valid_url,
            username=self.valid_username,
            password=self.valid_password
        )
        
        # Verify principal was called to test connection
        mock_client.principal.assert_called_once()
        
        # Verify client was returned
        self.assertEqual(client, mock_client)

    @patch('caldav.DAVClient')
    def test_get_client_auth_error(self, mock_client_class):
        """Test client creation with authentication error."""
        # Configure mock to raise auth error
        mock_client = mock_client_class.return_value
        mock_client.principal.side_effect = caldav.lib.error.AuthorizationError()
        
        # Call method and verify error
        with self.assertRaises(ToolError) as context:
            self.tool._get_client(
                self.valid_url, self.valid_username, self.valid_password
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        self.assertIn("Authentication failed", str(context.exception))

    @patch('caldav.DAVClient')
    def test_get_client_general_error(self, mock_client_class):
        """Test client creation with general error."""
        # Configure mock to raise general error
        mock_client = mock_client_class.return_value
        mock_client.principal.side_effect = Exception("Connection failed")
        
        # Call method and verify error
        with self.assertRaises(ToolError) as context:
            self.tool._get_client(
                self.valid_url, self.valid_username, self.valid_password
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_EXECUTION_ERROR)
        self.assertIn("Failed to connect", str(context.exception))

    @patch.object(CalendarTool, '_get_client')
    def test_list_calendars(self, mock_get_client):
        """Test listing calendars."""
        # Configure mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        mock_principal = MagicMock()
        mock_client.principal.return_value = mock_principal
        
        # Create mock calendars
        mock_calendar1 = MagicMock()
        mock_calendar1.name = "personal"
        mock_calendar1.url = "https://caldav.example.com/calendars/personal"
        mock_calendar1.get_properties.return_value = {"{DAV:}displayname": "Personal Calendar"}
        
        mock_calendar2 = MagicMock()
        mock_calendar2.name = "work"
        mock_calendar2.url = "https://caldav.example.com/calendars/work"
        mock_calendar2.get_properties.return_value = {"{DAV:}displayname": "Work Calendar"}
        
        mock_principal.calendars.return_value = [mock_calendar1, mock_calendar2]
        
        # Call method
        result = self.tool._list_calendars(
            self.valid_url, self.valid_username, self.valid_password
        )
        
        # Verify client was created
        mock_get_client.assert_called_once_with(
            self.valid_url, self.valid_username, self.valid_password
        )
        
        # Verify principal and calendars were called
        mock_client.principal.assert_called_once()
        mock_principal.calendars.assert_called_once()
        
        # Verify result structure
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["calendars"]), 2)
        
        # Verify calendar details
        self.assertEqual(result["calendars"][0]["id"], "personal")
        self.assertEqual(result["calendars"][0]["display_name"], "Personal Calendar")
        self.assertEqual(result["calendars"][0]["url"], "https://caldav.example.com/calendars/personal")
        
        self.assertEqual(result["calendars"][1]["id"], "work")
        self.assertEqual(result["calendars"][1]["display_name"], "Work Calendar")
        self.assertEqual(result["calendars"][1]["url"], "https://caldav.example.com/calendars/work")

    @patch.object(CalendarTool, '_get_client')
    def test_list_events(self, mock_get_client):
        """Test listing events."""
        # Configure mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        mock_principal = MagicMock()
        mock_client.principal.return_value = mock_principal
        
        # Create mock calendar
        mock_calendar = MagicMock()
        mock_calendar.name = self.valid_calendar_id
        mock_principal.calendars.return_value = [mock_calendar]
        
        # Create mock events
        mock_event1 = MagicMock()
        mock_event1.id = "event-1"
        mock_event1.data = "event data 1"
        
        mock_ical1 = MagicMock()
        mock_ical1.get.side_effect = lambda key, default=None: {
            "summary": "Team Meeting",
            "dtstart": MagicMock(dt=datetime.fromisoformat("2025-04-15T10:00:00")),
            "dtend": MagicMock(dt=datetime.fromisoformat("2025-04-15T11:00:00")),
            "description": "Weekly team sync",
            "location": "Conference Room B"
        }.get(key, default)
        
        mock_event1.icalendar_component = mock_ical1
        
        mock_event2 = MagicMock()
        mock_event2.id = "event-2"
        mock_event2.data = "event data 2"
        
        mock_ical2 = MagicMock()
        mock_ical2.get.side_effect = lambda key, default=None: {
            "summary": "Client Call",
            "dtstart": MagicMock(dt=datetime.fromisoformat("2025-04-16T14:00:00")),
            "dtend": MagicMock(dt=datetime.fromisoformat("2025-04-16T15:00:00")),
        }.get(key, default)
        
        mock_event2.icalendar_component = mock_ical2
        
        mock_calendar.date_search.return_value = [mock_event1, mock_event2]
        
        # Call method
        start_date = "2025-04-15"
        end_date = "2025-04-16"
        
        result = self.tool._list_events(
            self.valid_url, self.valid_username, self.valid_password,
            self.valid_calendar_id, start_date, end_date
        )
        
        # Verify client was created
        mock_get_client.assert_called_once_with(
            self.valid_url, self.valid_username, self.valid_password
        )
        
        # Verify principal and calendars were called
        mock_client.principal.assert_called_once()
        mock_principal.calendars.assert_called_once()
        
        # Verify date search was called
        mock_calendar.date_search.assert_called_once()
        call_args = mock_calendar.date_search.call_args[1]
        self.assertEqual(call_args["start"], datetime.fromisoformat(start_date))
        self.assertEqual(call_args["end"], datetime.fromisoformat(end_date))
        self.assertTrue(call_args["expand"])
        
        # Verify result structure
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["calendar_id"], self.valid_calendar_id)
        self.assertEqual(result["start_date"], start_date)
        self.assertEqual(result["end_date"], end_date)
        self.assertEqual(len(result["events"]), 2)
        
        # Verify event details
        self.assertEqual(result["events"][0]["event_id"], "event-1")
        self.assertEqual(result["events"][0]["summary"], "Team Meeting")
        self.assertEqual(result["events"][0]["start"], "2025-04-15T10:00:00")
        self.assertEqual(result["events"][0]["end"], "2025-04-15T11:00:00")
        self.assertEqual(result["events"][0]["description"], "Weekly team sync")
        self.assertEqual(result["events"][0]["location"], "Conference Room B")
        
        self.assertEqual(result["events"][1]["event_id"], "event-2")
        self.assertEqual(result["events"][1]["summary"], "Client Call")
        self.assertEqual(result["events"][1]["start"], "2025-04-16T14:00:00")
        self.assertEqual(result["events"][1]["end"], "2025-04-16T15:00:00")
        self.assertNotIn("description", result["events"][1])
        self.assertNotIn("location", result["events"][1])

    @patch.object(CalendarTool, '_get_client')
    @patch('uuid.uuid4')
    def test_create_event(self, mock_uuid, mock_get_client):
        """Test creating an event."""
        # Configure mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        mock_principal = MagicMock()
        mock_client.principal.return_value = mock_principal
        
        # Create mock calendar
        mock_calendar = MagicMock()
        mock_calendar.name = self.valid_calendar_id
        mock_principal.calendars.return_value = [mock_calendar]
        
        # Mock UUID generation
        event_id = "12345-67890"
        mock_uuid.return_value = event_id
        
        # Mock event creation
        mock_event = MagicMock()
        mock_event.id = event_id
        mock_calendar.save_event.return_value = mock_event
        
        # Call method
        result = self.tool._create_event(
            self.valid_url, self.valid_username, self.valid_password,
            self.valid_calendar_id, self.valid_summary, 
            self.valid_start_time, self.valid_end_time,
            self.valid_description, self.valid_location
        )
        
        # Verify client was created
        mock_get_client.assert_called_once_with(
            self.valid_url, self.valid_username, self.valid_password
        )
        
        # Verify principal and calendars were called
        mock_client.principal.assert_called_once()
        mock_principal.calendars.assert_called_once()
        
        # Verify save_event was called
        mock_calendar.save_event.assert_called_once()
        
        # Verify iCalendar data format
        ical_data = mock_calendar.save_event.call_args[0][0]
        self.assertIn(f"UID:{event_id}", ical_data)
        self.assertIn(f"SUMMARY:{self.valid_summary}", ical_data)
        self.assertIn(f"DESCRIPTION:{self.valid_description}", ical_data)
        self.assertIn(f"LOCATION:{self.valid_location}", ical_data)
        
        # Verify result structure
        self.assertTrue(result["success"])
        self.assertIn("message", result)
        self.assertEqual(result["event"]["event_id"], event_id)
        self.assertEqual(result["event"]["summary"], self.valid_summary)
        self.assertEqual(result["event"]["start"], self.valid_start_time)
        self.assertEqual(result["event"]["end"], self.valid_end_time)
        self.assertEqual(result["event"]["description"], self.valid_description)
        self.assertEqual(result["event"]["location"], self.valid_location)
        self.assertEqual(result["calendar_id"], self.valid_calendar_id)

    @patch.object(CalendarTool, '_get_client')
    def test_delete_event(self, mock_get_client):
        """Test deleting an event."""
        # Configure mocks
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        mock_principal = MagicMock()
        mock_client.principal.return_value = mock_principal
        
        # Create mock calendar
        mock_calendar = MagicMock()
        mock_calendar.name = self.valid_calendar_id
        mock_principal.calendars.return_value = [mock_calendar]
        
        # Create mock event
        mock_event = MagicMock()
        mock_event.id = self.valid_event_id
        mock_calendar.events.return_value = [mock_event]
        
        # Call method
        result = self.tool._delete_event(
            self.valid_url, self.valid_username, self.valid_password,
            self.valid_calendar_id, self.valid_event_id
        )
        
        # Verify client was created
        mock_get_client.assert_called_once_with(
            self.valid_url, self.valid_username, self.valid_password
        )
        
        # Verify principal and calendars were called
        mock_client.principal.assert_called_once()
        mock_principal.calendars.assert_called_once()
        
        # Verify event retrieval and deletion
        mock_calendar.events.assert_called_once()
        mock_event.delete.assert_called_once()
        
        # Verify result structure
        self.assertTrue(result["success"])
        self.assertIn("message", result)
        self.assertEqual(result["event_id"], self.valid_event_id)
        self.assertEqual(result["calendar_id"], self.valid_calendar_id)

    @patch.object(CalendarTool, '_list_calendars')
    @patch('os.environ.get')
    @patch('tools.calendar_tool.config')
    def test_run_list_calendars(self, mock_config, mock_environ_get, mock_list_calendars):
        """Test run method with list_calendars action."""
        # Configure mocks
        mock_config.calendar.default_url = "https://caldav.example.com"
        mock_config.calendar.default_username = "user@example.com"
        mock_config.calendar.default_calendar_id = "personal"
        
        expected_result = {"success": True, "calendars": []}
        mock_list_calendars.return_value = expected_result
        mock_environ_get.return_value = "password123"
        
        # Call method
        result = self.tool.run(action="list_calendars")
        
        # Verify method was called with correct parameters
        mock_list_calendars.assert_called_once_with(
            "https://caldav.example.com", "user@example.com", "password123"
        )
        
        # Verify result
        self.assertEqual(result, expected_result)

    @patch.object(CalendarTool, '_list_events')
    @patch('os.environ.get')
    @patch('tools.calendar_tool.config')
    def test_run_list_events(self, mock_config, mock_environ_get, mock_list_events):
        """Test run method with list_events action."""
        # Configure mocks
        mock_config.calendar.default_url = "https://caldav.example.com"
        mock_config.calendar.default_username = "user@example.com"
        mock_config.calendar.default_calendar_id = "personal"
        
        expected_result = {"success": True, "events": []}
        mock_list_events.return_value = expected_result
        mock_environ_get.return_value = "password123"
        
        # Call method
        result = self.tool.run(
            action="list_events",
            start_date="2025-04-15",
            end_date="2025-04-16"
        )
        
        # Verify method was called with correct parameters
        mock_list_events.assert_called_once_with(
            "https://caldav.example.com", "user@example.com", "password123",
            "personal", "2025-04-15", "2025-04-16"
        )
        
        # Verify result
        self.assertEqual(result, expected_result)

    @patch.object(CalendarTool, '_create_event')
    @patch('os.environ.get')
    @patch('tools.calendar_tool.config')
    def test_run_create_event(self, mock_config, mock_environ_get, mock_create_event):
        """Test run method with create_event action."""
        # Configure mocks
        mock_config.calendar.default_url = "https://caldav.example.com"
        mock_config.calendar.default_username = "user@example.com"
        mock_config.calendar.default_calendar_id = "personal"
        
        expected_result = {"success": True, "event": {}}
        mock_create_event.return_value = expected_result
        mock_environ_get.return_value = "password123"
        
        # Call method
        result = self.tool.run(
            action="create_event",
            summary=self.valid_summary,
            start_time=self.valid_start_time,
            end_time=self.valid_end_time,
            description=self.valid_description,
            location=self.valid_location
        )
        
        # Verify method was called with correct parameters
        mock_create_event.assert_called_once_with(
            "https://caldav.example.com", "user@example.com", "password123",
            "personal", self.valid_summary,
            self.valid_start_time, self.valid_end_time,
            self.valid_description, self.valid_location
        )
        
        # Verify result
        self.assertEqual(result, expected_result)

    @patch.object(CalendarTool, '_delete_event')
    @patch('os.environ.get')
    @patch('config.config.calendar.default_url', new=property(lambda self: "https://caldav.example.com"))
    @patch('config.config.calendar.default_username', new=property(lambda self: "user@example.com"))
    @patch('config.config.calendar.default_calendar_id', new=property(lambda self: "personal"))
    def test_run_delete_event(self, mock_environ_get, mock_delete_event):
        """Test run method with delete_event action."""
        # Configure mocks
        expected_result = {"success": True, "message": "Event deleted"}
        mock_delete_event.return_value = expected_result
        mock_environ_get.return_value = "password123"
        
        # Call method
        result = self.tool.run(
            action="delete_event",
            event_id=self.valid_event_id
        )
        
        # Verify method was called with correct parameters
        mock_delete_event.assert_called_once_with(
            "https://caldav.example.com", "user@example.com", "password123",
            "personal", self.valid_event_id
        )
        
        # Verify result
        self.assertEqual(result, expected_result)

    @patch('os.environ.get')
    @patch('config.config.calendar.default_url', new=property(lambda self: "https://caldav.example.com"))
    @patch('config.config.calendar.default_username', new=property(lambda self: "user@example.com"))
    def test_run_invalid_action(self, mock_environ_get):
        """Test run method with an invalid action."""
        invalid_action = "invalid_action"
        mock_environ_get.return_value = "password123"
        
        with self.assertRaises(ToolError) as context:
            self.tool.run(action=invalid_action)
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        self.assertIn(f"Invalid action: {invalid_action}", str(context.exception))


    @patch('time.time')
    @patch('os.path.exists')
    @patch('os.path.getmtime')
    @patch('config.config.calendar_tool.cache_duration', new=3600)
    def test_is_cache_valid(self, mock_getmtime, mock_exists, mock_time):
        """Test validation of cache file."""
        # Setup
        cache_path = "/tmp/cache/12345.json"
        
        # Test 1: Cache file doesn't exist
        mock_exists.return_value = False
        self.assertFalse(self.tool._is_cache_valid(cache_path))
        
        # Test 2: Cache file exists but is expired
        mock_exists.return_value = True
        mock_time.return_value = 10000
        mock_getmtime.return_value = 6000  # 4000 seconds old (> 3600s cache duration)
        self.assertFalse(self.tool._is_cache_valid(cache_path))
        
        # Test 3: Cache file exists and is valid
        mock_exists.return_value = True
        mock_time.return_value = 10000
        mock_getmtime.return_value = 9000  # 1000 seconds old (< 3600s cache duration)
        self.assertTrue(self.tool._is_cache_valid(cache_path))
    
    @patch('requests.get')
    @patch.object(CalendarTool, '_is_cache_valid')
    @patch.object(CalendarTool, '_get_cache_path')
    @patch('builtins.open', new_callable=mock_open)
    @patch('caldav.lib.error.vcal.VCalendar')
    def test_fetch_ical_url_fresh(self, mock_vcalendar_class, mock_file, mock_get_cache_path, 
                                 mock_is_cache_valid, mock_requests_get):
        """Test fetching fresh iCalendar data from URL."""
        # Setup mocks
        mock_get_cache_path.return_value = "/tmp/cache/12345.json"
        mock_is_cache_valid.return_value = False  # Force fresh fetch
        
        mock_response = MagicMock()
        mock_response.text = "BEGIN:VCALENDAR\nEND:VCALENDAR"
        mock_requests_get.return_value = mock_response
        
        # Mock VCalendar
        mock_vcalendar = MagicMock()
        mock_vcalendar_class.return_value = mock_vcalendar
        
        # Create mock event
        mock_event = MagicMock()
        mock_event.name = 'VEVENT'
        mock_event.get.side_effect = lambda key, default=None: {
            'uid': 'event-1',
            'summary': 'Test Event',
            'dtstart': MagicMock(value=datetime.fromisoformat("2025-05-15T10:00:00")),
            'dtend': MagicMock(value=datetime.fromisoformat("2025-05-15T11:00:00")),
            'description': 'Test Description',
            'location': 'Test Location'
        }.get(key, default)
        
        # Configure VCalendar to return our mock event
        mock_vcalendar.components.return_value = [mock_event]
        
        # Call the method
        url = "https://example.com/calendar.ics"
        start_date = "2025-05-15"
        end_date = "2025-05-16"
        result = self.tool._fetch_ical_url(url, start_date, end_date)
        
        # Verify requests.get was called
        mock_requests_get.assert_called_once_with(url, timeout=30)
        
        # Verify VCalendar was created with the response text
        mock_vcalendar_class.assert_called_once_with(mock_response.text)
        
        # Verify components were requested from VCalendar
        mock_vcalendar.components.assert_called_once()
        
        # Verify event properties were accessed
        mock_event.get.assert_any_call('uid', str(uuid.uuid4()))
        mock_event.get.assert_any_call('summary', 'No Title')
        mock_event.get.assert_any_call('dtstart')
        mock_event.get.assert_any_call('dtend')
        
        # Verify the result structure
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["calendar_id"], "external")
        self.assertEqual(result["url"], url)
        self.assertEqual(result["start_date"], start_date)
        self.assertEqual(result["end_date"], end_date)
        
        # Verify event details
        event = result["events"][0]
        self.assertEqual(event["event_id"], "event-1")
        self.assertEqual(event["summary"], "Test Event")
        self.assertEqual(event["start"], "2025-05-15T10:00:00")
        self.assertEqual(event["end"], "2025-05-15T11:00:00")
        self.assertEqual(event["description"], "Test Description")
        self.assertEqual(event["location"], "Test Location")
        
        # Verify cache was written
        mock_file().write.assert_called_once()
    
    @patch.object(CalendarTool, '_is_cache_valid')
    @patch.object(CalendarTool, '_get_cache_path')
    @patch('builtins.open', new_callable=mock_open)
    def test_fetch_ical_url_cached(self, mock_file, mock_get_cache_path, mock_is_cache_valid):
        """Test fetching cached iCalendar data."""
        # Setup mocks
        mock_get_cache_path.return_value = "/tmp/cache/12345.json"
        mock_is_cache_valid.return_value = True  # Use cache
        
        # Create mock cached data
        cached_data = {
            "success": True,
            "events": [
                {
                    "event_id": "event-1",
                    "summary": "Cached Event",
                    "start": "2025-05-15T10:00:00",
                    "end": "2025-05-15T11:00:00",
                    "description": "Cached Description",
                    "location": "Cached Location"
                }
            ],
            "count": 1,
            "calendar_id": "external",
            "url": "https://example.com/calendar.ics",
            "start_date": "2025-05-15",
            "end_date": "2025-05-16"
        }
        
        # Mock file read to return cached data
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(cached_data)
        
        # Call the method
        url = "https://example.com/calendar.ics"
        start_date = "2025-05-15"
        end_date = "2025-05-16"
        result = self.tool._fetch_ical_url(url, start_date, end_date)
        
        # Verify cache was checked and path was generated
        mock_get_cache_path.assert_called_once()
        mock_is_cache_valid.assert_called_once_with("/tmp/cache/12345.json")
        
        # Verify file was opened for reading
        mock_file.assert_called_once_with('/tmp/cache/12345.json', 'r')
        
        # Verify the result structure matches cached data
        self.assertEqual(result, cached_data)
    
    @patch.object(CalendarTool, '_fetch_ical_url')
    def test_run_read_ical_url(self, mock_fetch_ical_url):
        """Test run method with read_ical_url action."""
        # Configure mock
        expected_result = {"success": True, "events": []}
        mock_fetch_ical_url.return_value = expected_result
        
        # Call method
        url = "https://example.com/calendar.ics"
        start_date = "2025-05-15"
        end_date = "2025-05-16"
        
        result = self.tool.run(
            action="read_ical_url",
            ical_url=url,
            start_date=start_date,
            end_date=end_date
        )
        
        # Verify method was called with correct parameters
        mock_fetch_ical_url.assert_called_once_with(url, start_date, end_date)
        
        # Verify result
        self.assertEqual(result, expected_result)
    
    def test_run_read_ical_url_missing_url(self):
        """Test run method with read_ical_url action but missing URL."""        
        # Call method and verify error
        with self.assertRaises(ToolError) as context:
            self.tool.run(action="read_ical_url")
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        self.assertIn("iCalendar URL must be a non-empty string", str(context.exception))
        
    @patch.object(CalendarTool, '_fetch_all_calendars')
    def test_run_list_all_events(self, mock_fetch_all_calendars):
        """Test run method with list_all_events action."""
        # Configure mock
        expected_result = {
            "success": True,
            "calendars": [
                {
                    "name": "Work Calendar",
                    "events": [{"summary": "Meeting"}],
                    "count": 1
                }
            ],
            "total_events": 1
        }
        mock_fetch_all_calendars.return_value = expected_result
        
        # Call method
        calendar_name = "Work Calendar"
        start_date = "2025-05-01"
        end_date = "2025-05-07"
        
        result = self.tool.run(
            action="list_all_events",
            calendar_name=calendar_name,
            start_date=start_date,
            end_date=end_date
        )
        
        # Verify method was called with correct parameters
        mock_fetch_all_calendars.assert_called_once_with(calendar_name, start_date, end_date)
        
        # Verify result
        self.assertEqual(result, expected_result)
        
    @patch('config.config.calendar_tool.calendars', new={
        'work': MagicMock(name='Work Calendar', url='https://work.example.com', type='ical'),
        'personal': MagicMock(name='Personal Calendar', url='https://personal.example.com', type='ical')
    })
    @patch.object(CalendarTool, '_fetch_ical_url')
    def test_fetch_all_calendars(self, mock_fetch_ical_url):
        """Test fetching events from all configured calendars."""
        # Configure mock
        mock_fetch_ical_url.return_value = {
            "success": True,
            "events": [{"summary": "Meeting"}],
            "count": 1
        }
        
        # Call method
        result = self.tool._fetch_all_calendars(
            start_date="2025-05-01",
            end_date="2025-05-07"
        )
        
        # Verify method was called twice (once for each calendar)
        self.assertEqual(mock_fetch_ical_url.call_count, 2)
        
        # Verify result structure
        self.assertTrue(result["success"])
        self.assertEqual(len(result["calendars"]), 2)
        self.assertEqual(result["total_events"], 2)
        
    @patch('config.config.calendar_tool.calendars', new={
        'work': MagicMock(name='Work Calendar', url='https://work.example.com', type='ical'),
        'personal': MagicMock(name='Personal Calendar', url='https://personal.example.com', type='ical')
    })
    @patch.object(CalendarTool, '_fetch_ical_url')
    def test_fetch_all_calendars_with_name(self, mock_fetch_ical_url):
        """Test fetching events from a specific calendar by name."""
        # Configure mock
        mock_fetch_ical_url.return_value = {
            "success": True,
            "events": [{"summary": "Meeting"}],
            "count": 1
        }
        
        # Call method
        result = self.tool._fetch_all_calendars(
            calendar_name="Work Calendar",
            start_date="2025-05-01",
            end_date="2025-05-07"
        )
        
        # Verify method was called once (for the specified calendar)
        mock_fetch_ical_url.assert_called_once()
        
        # Verify result structure
        self.assertTrue(result["success"])
        self.assertEqual(len(result["calendars"]), 1)
        self.assertEqual(result["total_events"], 1)
        self.assertEqual(result["calendars"][0]["name"], "Work Calendar")


if __name__ == '__main__':
    unittest.main()