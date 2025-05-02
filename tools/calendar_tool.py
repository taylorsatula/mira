import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import caldav
import uuid

from pydantic import BaseModel, Field
from tools.repo import Tool
from errors import ErrorCode, error_context, ToolError
from config.registry import registry

# Define configuration class for CalendarTool
class CalendarToolConfig(BaseModel):
    """Configuration for the calendar_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    default_url: str = Field(
        default="https://caldav.example.com",
        description="Default CalDAV server URL"
    )
    default_username: str = Field(
        default="user@example.com",
        description="Default CalDAV username"
    )
    default_calendar_id: str = Field(
        default="personal",
        description="Default calendar ID to use when not specified"
    )
    timeout: int = Field(
        default=30,
        description="Timeout in seconds for CalDAV requests"
    )
    max_events: int = Field(
        default=100,
        description="Maximum number of events to return in a single request"
    )
    default_event_duration: int = Field(
        default=60,
        description="Default event duration in minutes if not specified"
    )

# Register with registry
registry.register("calendar_tool", CalendarToolConfig)


class CalendarTool(Tool):
    """
    CalDAV integration tool for calendar management.
    
    This tool provides functionality to interact with CalDAV servers,
    allowing users to view, create, update, and delete calendar events.
    It supports multiple calendars and provides a clean interface for
    managing scheduling.
    """

    name = "calendar_tool"
    description = """
    Manages personal calendar events via CalDAV protocol. This tool lets you interact with calendar servers
    to schedule, view, and manage personal calendar events. Use this tool ONLY when the user explicitly needs to interact 
    with their personal CalDAV calendar system including checking availability, creating personal appointments, or managing existing calendar events. This tool is NOT for business appointments or bookings.
    
    OPERATIONS:
    - list_calendars: Lists available calendars on the server
      Parameters:
        url (required): The CalDAV server URL
        username (required): Username for authentication
        password (required): Password for authentication
    
    - list_events: Lists events from a specific calendar within a date range
      Parameters:
        url (required): The CalDAV server URL
        username (required): Username for authentication
        password (required): Password for authentication
        calendar_id (required): The calendar ID to query
        start_date (optional, default=today): Start date in ISO format (YYYY-MM-DD)
        end_date (optional, default=7 days from start): End date in ISO format (YYYY-MM-DD)
    
    - create_event: Creates a new calendar event
      Parameters:
        url (required): The CalDAV server URL
        username (required): Username for authentication
        password (required): Password for authentication
        calendar_id (required): The calendar ID to add the event to
        summary (required): Event title/summary
        start_time (required): Start time in ISO format (YYYY-MM-DDTHH:MM:SS)
        end_time (required): End time in ISO format (YYYY-MM-DDTHH:MM:SS)
        description (optional): Event description
        location (optional): Event location
    
    - delete_event: Deletes a calendar event
      Parameters:
        url (required): The CalDAV server URL
        username (required): Username for authentication
        password (required): Password for authentication
        calendar_id (required): The calendar ID containing the event
        event_id (required): The unique ID of the event to delete
    
    RESPONSE FORMAT:
    - All operations return a dictionary with success status and relevant data
    - Errors include detailed information about what went wrong
    
    USAGE NOTES:
    - Use list_calendars first to discover available calendars and their IDs
    - Date ranges for list_events should be reasonable (e.g., 1-30 days)
    - All date and time parameters must be in ISO format
    - Authentication information is used only for the current operation and not stored
    
    LIMITATIONS:
    - Can only connect to CalDAV-compliant servers
    - Does not support recurring event creation (only displays them)
    - Limited to basic event properties (no attachments, attendees, or notifications)
    - Calendar permissions are determined by the provided credentials
    """
    
    usage_examples = [
        {
            "input": {
                "url": "https://caldav.example.com",
                "username": "user@example.com",
                "password": "password123",
                "calendar_id": "personal"
            },
            "output": {
                "success": True,
                "events": [
                    {
                        "summary": "Team Meeting",
                        "start": "2025-04-15T10:00:00",
                        "end": "2025-04-15T11:00:00",
                        "location": "Conference Room B",
                        "event_id": "12345-67890"
                    }
                ],
                "count": 1
            }
        }
    ]

    def __init__(self):
        """Initialize the calendar tool."""
        super().__init__()
        self.logger.info("CalendarTool initialized")

    def run(
        self,
        action: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        summary: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        event_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the calendar tool with the specified action.
        
        Args:
            action: The operation to perform (list_calendars, list_events, create_event, delete_event)
            url: CalDAV server URL
            username: Username for authentication
            password: Password for authentication
            calendar_id: The calendar ID to operate on (if applicable)
            start_date: Start date for listing events (YYYY-MM-DD)
            end_date: End date for listing events (YYYY-MM-DD)
            summary: Event title/summary for create_event
            start_time: Event start time for create_event (YYYY-MM-DDTHH:MM:SS)
            end_time: Event end time for create_event (YYYY-MM-DDTHH:MM:SS)
            description: Event description for create_event
            location: Event location for create_event
            event_id: Event ID for delete_event
            
        Returns:
            Dictionary containing the operation results
            
        Raises:
            ToolError: If parameters are invalid or the operation fails
        """
        self.logger.info(f"Running calendar tool with action: {action}")
        
        with error_context(
            component_name=self.name,
            operation=f"executing calendar operation '{action}'",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Import config when needed (avoids circular imports)
            from config import config
            
            # Get configuration values
            url = config.calendar_tool.default_url
            username = config.calendar_tool.default_username
            password = os.environ.get("CALDAV_ACCOUNT_PASSWORD")
            calendar_id = config.calendar_tool.default_calendar_id
            
            # Validate common parameters
            self._validate_common_params(url, username, password)
            
            # Route to appropriate action
            if action == "list_calendars":
                return self._list_calendars(url, username, password)
            elif action == "list_events":
                return self._list_events(url, username, password, calendar_id, start_date, end_date)
            elif action == "create_event":
                self._validate_event_params(summary, start_time, end_time)
                return self._create_event(
                    url, username, password, calendar_id, 
                    summary, start_time, end_time, description, location
                )
            elif action == "delete_event":
                self._validate_event_id(event_id)
                return self._delete_event(url, username, password, calendar_id, event_id)
            else:
                raise ToolError(
                    f"Invalid action: {action}. Must be one of: list_calendars, list_events, create_event, delete_event",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"provided_action": action}
                )
    
    def _validate_common_params(self, url: str, username: str, password: str) -> None:
        """
        Validate common parameters for all operations.
        
        Args:
            url: CalDAV server URL
            username: Username for authentication
            password: Password for authentication
            
        Raises:
            ToolError: If any parameter is invalid
        """
        if not url or not isinstance(url, str):
            raise ToolError(
                "URL must be a non-empty string",
                ErrorCode.TOOL_INVALID_INPUT,
                {"param": "url"}
            )
            
        if not username or not isinstance(username, str):
            raise ToolError(
                "Username must be a non-empty string",
                ErrorCode.TOOL_INVALID_INPUT,
                {"param": "username"}
            )
            
        if not password or not isinstance(password, str):
            raise ToolError(
                "Password must be a non-empty string",
                ErrorCode.TOOL_INVALID_INPUT,
                {"param": "password"}
            )
    
    def _validate_calendar_id(self, calendar_id: Optional[str]) -> None:
        """
        Validate calendar ID parameter.
        
        Args:
            calendar_id: The calendar ID to validate
            
        Raises:
            ToolError: If calendar_id is invalid
        """
        if not calendar_id or not isinstance(calendar_id, str):
            raise ToolError(
                "Calendar ID must be a non-empty string",
                ErrorCode.TOOL_INVALID_INPUT,
                {"param": "calendar_id"}
            )
    
    def _validate_event_params(self, summary: Optional[str], start_time: Optional[str], end_time: Optional[str]) -> None:
        """
        Validate event parameters.
        
        Args:
            summary: Event title/summary
            start_time: Event start time (YYYY-MM-DDTHH:MM:SS)
            end_time: Event end time (YYYY-MM-DDTHH:MM:SS)
            
        Raises:
            ToolError: If any parameter is invalid
        """
        if not summary or not isinstance(summary, str):
            raise ToolError(
                "Event summary must be a non-empty string",
                ErrorCode.TOOL_INVALID_INPUT,
                {"param": "summary"}
            )
            
        if not start_time or not isinstance(start_time, str):
            raise ToolError(
                "Start time must be a non-empty string in ISO format (YYYY-MM-DDTHH:MM:SS)",
                ErrorCode.TOOL_INVALID_INPUT,
                {"param": "start_time"}
            )
            
        if not end_time or not isinstance(end_time, str):
            raise ToolError(
                "End time must be a non-empty string in ISO format (YYYY-MM-DDTHH:MM:SS)",
                ErrorCode.TOOL_INVALID_INPUT,
                {"param": "end_time"}
            )
            
        # Validate ISO format
        try:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            
            if end_dt <= start_dt:
                raise ToolError(
                    "End time must be after start time",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"start_time": start_time, "end_time": end_time}
                )
        except ValueError:
            raise ToolError(
                "Invalid datetime format. Use ISO format: YYYY-MM-DDTHH:MM:SS",
                ErrorCode.TOOL_INVALID_INPUT,
                {"start_time": start_time, "end_time": end_time}
            )
    
    def _validate_event_id(self, event_id: Optional[str]) -> None:
        """
        Validate event ID parameter.
        
        Args:
            event_id: Event ID to validate
            
        Raises:
            ToolError: If event_id is invalid
        """
        if not event_id or not isinstance(event_id, str):
            raise ToolError(
                "Event ID must be a non-empty string",
                ErrorCode.TOOL_INVALID_INPUT,
                {"param": "event_id"}
            )
    
    def _get_client(self, url: str, username: str, password: str) -> caldav.DAVClient:
        """
        Create and return a CalDAV client.
        
        Args:
            url: CalDAV server URL
            username: Username for authentication
            password: Password for authentication
            
        Returns:
            A configured CalDAV client
            
        Raises:
            ToolError: If connection fails
        """
        try:
            client = caldav.DAVClient(
                url=url,
                username=username,
                password=password
            )
            # Test connection by getting principal
            client.principal()
            return client
        except caldav.lib.error.AuthorizationError:
            raise ToolError(
                "Authentication failed. Check your credentials.",
                ErrorCode.TOOL_INVALID_INPUT,
                {"url": url, "username": username}
            )
        except Exception as e:
            raise ToolError(
                f"Failed to connect to CalDAV server: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"url": url, "error": str(e)}
            )
    
    def _list_calendars(self, url: str, username: str, password: str) -> Dict[str, Any]:
        """
        List available calendars on the server.
        
        Args:
            url: CalDAV server URL
            username: Username for authentication
            password: Password for authentication
            
        Returns:
            Dictionary with list of calendars
        """
        client = self._get_client(url, username, password)
        
        try:
            principal = client.principal()
            calendars = principal.calendars()
            
            calendar_list = []
            for calendar in calendars:
                cal_info = {
                    "id": calendar.name,
                    "display_name": calendar.get_properties([caldav.elements.dav.DisplayName()])["{DAV:}displayname"] or calendar.name,
                    "url": calendar.url
                }
                calendar_list.append(cal_info)
            
            return {
                "success": True,
                "calendars": calendar_list,
                "count": len(calendar_list)
            }
        except Exception as e:
            raise ToolError(
                f"Failed to list calendars: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"error": str(e)}
            )
    
    def _list_events(
        self,
        url: str,
        username: str,
        password: str,
        calendar_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List events in a calendar within a date range.
        
        Args:
            url: CalDAV server URL
            username: Username for authentication
            password: Password for authentication
            calendar_id: ID of the calendar to query
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            
        Returns:
            Dictionary with list of events
        """
        client = self._get_client(url, username, password)
        
        # Set default date range if not provided
        if not start_date:
            start_date = datetime.now().date().isoformat()
        
        if not end_date:
            # Default to 7 days from start
            start_dt = datetime.fromisoformat(start_date)
            end_dt = start_dt + timedelta(days=7)
            end_date = end_dt.date().isoformat()
        
        try:
            # Parse dates
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            # Find the specific calendar
            principal = client.principal()
            calendars = principal.calendars()
            target_calendar = None
            
            for calendar in calendars:
                if calendar.name == calendar_id:
                    target_calendar = calendar
                    break
            
            if not target_calendar:
                raise ToolError(
                    f"Calendar with ID '{calendar_id}' not found",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"calendar_id": calendar_id}
                )
            
            # Get events in the date range
            events = target_calendar.date_search(
                start=start_dt,
                end=end_dt,
                expand=True  # Expand recurring events
            )
            
            event_list = []
            for event in events:
                event_data = event.data
                ical_data = event.icalendar_component
                
                # Extract event details from iCalendar component
                event_info = {
                    "event_id": event.id,
                    "summary": str(ical_data.get("summary", "No Title")),
                    "start": ical_data.get("dtstart").dt.isoformat(),
                    "end": ical_data.get("dtend").dt.isoformat()
                }
                
                # Add optional properties if available
                if ical_data.get("description"):
                    event_info["description"] = str(ical_data.get("description"))
                    
                if ical_data.get("location"):
                    event_info["location"] = str(ical_data.get("location"))
                
                event_list.append(event_info)
            
            return {
                "success": True,
                "events": event_list,
                "count": len(event_list),
                "calendar_id": calendar_id,
                "start_date": start_date,
                "end_date": end_date
            }
        except ValueError as e:
            raise ToolError(
                f"Invalid date format: {str(e)}. Use ISO format YYYY-MM-DD.",
                ErrorCode.TOOL_INVALID_INPUT,
                {"start_date": start_date, "end_date": end_date}
            )
        except Exception as e:
            raise ToolError(
                f"Failed to list events: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"error": str(e)}
            )
    
    def _create_event(
        self,
        url: str,
        username: str,
        password: str,
        calendar_id: str,
        summary: str,
        start_time: str,
        end_time: str,
        description: Optional[str] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new calendar event.
        
        Args:
            url: CalDAV server URL
            username: Username for authentication
            password: Password for authentication
            calendar_id: ID of the calendar to add the event to
            summary: Event title/summary
            start_time: Start time in ISO format (YYYY-MM-DDTHH:MM:SS)
            end_time: End time in ISO format (YYYY-MM-DDTHH:MM:SS)
            description: Optional event description
            location: Optional event location
            
        Returns:
            Dictionary with created event details
        """
        client = self._get_client(url, username, password)
        
        try:
            # Find the specific calendar
            principal = client.principal()
            calendars = principal.calendars()
            target_calendar = None
            
            for calendar in calendars:
                if calendar.name == calendar_id:
                    target_calendar = calendar
                    break
            
            if not target_calendar:
                raise ToolError(
                    f"Calendar with ID '{calendar_id}' not found",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"calendar_id": calendar_id}
                )
            
            # Parse times
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            
            # Create event
            event_id = str(uuid.uuid4())
            ical_data = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//botwithmemory//CalendarTool//EN
BEGIN:VEVENT
UID:{event_id}
DTSTAMP:{datetime.now().strftime("%Y%m%dT%H%M%SZ")}
DTSTART:{start_dt.strftime("%Y%m%dT%H%M%S")}
DTEND:{end_dt.strftime("%Y%m%dT%H%M%S")}
SUMMARY:{summary}"""
            
            if description:
                ical_data += f"\nDESCRIPTION:{description}"
                
            if location:
                ical_data += f"\nLOCATION:{location}"
                
            ical_data += """\nEND:VEVENT
END:VCALENDAR"""
            
            # Add event to calendar
            event = target_calendar.save_event(ical_data)
            
            return {
                "success": True,
                "message": "Event created successfully",
                "event": {
                    "event_id": event.id,
                    "summary": summary,
                    "start": start_time,
                    "end": end_time,
                    "description": description,
                    "location": location
                },
                "calendar_id": calendar_id
            }
        except ValueError as e:
            raise ToolError(
                f"Invalid datetime format: {str(e)}. Use ISO format YYYY-MM-DDTHH:MM:SS.",
                ErrorCode.TOOL_INVALID_INPUT,
                {"start_time": start_time, "end_time": end_time}
            )
        except Exception as e:
            raise ToolError(
                f"Failed to create event: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"error": str(e)}
            )
    
    def _delete_event(
        self,
        url: str,
        username: str,
        password: str,
        calendar_id: str,
        event_id: str
    ) -> Dict[str, Any]:
        """
        Delete a calendar event.
        
        Args:
            url: CalDAV server URL
            username: Username for authentication
            password: Password for authentication
            calendar_id: ID of the calendar containing the event
            event_id: ID of the event to delete
            
        Returns:
            Dictionary with deletion status
        """
        client = self._get_client(url, username, password)
        
        try:
            # Find the specific calendar
            principal = client.principal()
            calendars = principal.calendars()
            target_calendar = None
            
            for calendar in calendars:
                if calendar.name == calendar_id:
                    target_calendar = calendar
                    break
            
            if not target_calendar:
                raise ToolError(
                    f"Calendar with ID '{calendar_id}' not found",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"calendar_id": calendar_id}
                )
            
            # Find the specific event
            event = None
            events = target_calendar.events()
            
            for e in events:
                if e.id == event_id:
                    event = e
                    break
            
            if not event:
                raise ToolError(
                    f"Event with ID '{event_id}' not found in calendar '{calendar_id}'",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"event_id": event_id, "calendar_id": calendar_id}
                )
            
            # Delete the event
            event.delete()
            
            return {
                "success": True,
                "message": "Event deleted successfully",
                "event_id": event_id,
                "calendar_id": calendar_id
            }
        except Exception as e:
            raise ToolError(
                f"Failed to delete event: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"error": str(e)}
            )