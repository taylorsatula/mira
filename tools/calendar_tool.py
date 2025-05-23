"""
Calendar Tool for viewing and managing events across multiple calendar services.

This module provides a unified interface for accessing different types of calendars
(CalDAV, iCalendar URLs) and performing operations like reading events, creating
events, and managing calendars.
"""

import logging
import os
import json
import hashlib
import time
import uuid
import requests
from typing import Dict, List, Any, Optional, Union, Type
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

import caldav
from pydantic import BaseModel, Field
from tools.repo import Tool
from errors import ErrorCode, error_context, ToolError
from config.registry import registry
from utils.timezone_utils import (
    validate_timezone, get_default_timezone, convert_to_timezone,
    format_datetime, parse_time_string
)


# Define configuration classes
class CalendarEntry(BaseModel):
    """Configuration for a single calendar entry."""
    name: str = Field(description="User-friendly name for the calendar")
    url: str = Field(description="URL to the calendar (CalDAV server or iCalendar URL)")
    type: str = Field(description="Type of calendar: 'caldav' or 'ical'")
    username: Optional[str] = Field(None, description="Username for CalDAV authentication (if type is 'caldav')")
    calendar_id: Optional[str] = Field(None, description="Calendar ID for CalDAV server (if type is 'caldav')")


class CalendarToolConfig(BaseModel):
    """Configuration for the calendar_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    calendars: Dict[str, CalendarEntry] = Field(
        default={},
        description="Dictionary of calendar entries keyed by unique identifier"
    )
    default_calendar_url: str = Field(
        default=None,
        description="Default iCalendar URL to use (for simple configuration). MUST be set using DEFAULT_CALENDAR_URL environment variable."
    )
    default_url: str = Field(
        default="",
        description="Default CalDAV server URL"
    )
    default_username: str = Field(
        default="",
        description="Default CalDAV username"
    )
    default_calendar_id: str = Field(
        default="",
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
    default_date_range: int = Field(
        default=7,
        description="Default number of days to look ahead when listing events"
    )
    cache_directory: str = Field(
        default="data/tools/calendar_tool/cache",
        description="Directory to store cached calendar data"
    )
    cache_duration: int = Field(
        default=3600,
        description="Duration in seconds to cache iCalendar data (default: 1 hour)"
    )


# Register with registry
registry.register("calendar_tool", CalendarToolConfig)


class ValidationUtils:
    """Utility methods for validating calendar tool parameters."""
    
    @staticmethod
    def validate_date(date_str: Optional[str], param_name: str) -> Optional[datetime]:
        """
        Validate and convert a date string to a datetime object.
        
        Args:
            date_str: The date string in ISO format (YYYY-MM-DD)
            param_name: The parameter name for error messages
            
        Returns:
            A timezone-aware datetime object if valid, or None if date_str is None
            
        Raises:
            ToolError: If the date is invalid
        """
        if date_str is None:
            return None
            
        if not isinstance(date_str, str):
            raise ToolError(
                f"{param_name} must be a string in ISO format (YYYY-MM-DD)",
                ErrorCode.TOOL_INVALID_INPUT,
                {param_name: date_str}
            )
            
        try:
            # Parse date and convert to timezone-aware datetime
            dt = datetime.fromisoformat(date_str)
            # Set time to midnight in the default timezone
            dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            # Convert to timezone-aware datetime
            return convert_to_timezone(dt, get_default_timezone())
        except ValueError:
            raise ToolError(
                f"Invalid {param_name} format: '{date_str}'. Use ISO format (YYYY-MM-DD)",
                ErrorCode.TOOL_INVALID_INPUT,
                {param_name: date_str}
            )
    
    @staticmethod
    def validate_datetime(dt_str: Optional[str], param_name: str) -> Optional[datetime]:
        """
        Validate and convert a datetime string to a datetime object.
        
        Args:
            dt_str: The datetime string in ISO format (YYYY-MM-DDTHH:MM:SS)
            param_name: The parameter name for error messages
            
        Returns:
            A timezone-aware datetime object if valid, or None if dt_str is None
            
        Raises:
            ToolError: If the datetime is invalid
        """
        if dt_str is None:
            return None
            
        if not isinstance(dt_str, str):
            raise ToolError(
                f"{param_name} must be a string in ISO format (YYYY-MM-DDTHH:MM:SS)",
                ErrorCode.TOOL_INVALID_INPUT,
                {param_name: dt_str}
            )
            
        try:
            # Use our timezone utility to parse the datetime string
            # This will handle both naive and timezone-aware datetime strings
            return parse_time_string(dt_str)
        except ValueError:
            raise ToolError(
                f"Invalid {param_name} format: '{dt_str}'. Use ISO format (YYYY-MM-DDTHH:MM:SS)",
                ErrorCode.TOOL_INVALID_INPUT,
                {param_name: dt_str}
            )

    @staticmethod
    def require_non_empty_string(value: Any, param_name: str) -> str:
        """
        Validate that a parameter is a non-empty string.
        
        Args:
            value: The value to validate
            param_name: The parameter name for error messages
            
        Returns:
            The value if valid
            
        Raises:
            ToolError: If the value is not a non-empty string
        """
        if not value or not isinstance(value, str):
            raise ToolError(
                f"{param_name} must be a non-empty string",
                ErrorCode.TOOL_INVALID_INPUT,
                {param_name: value}
            )
        return value


class CalendarCache:
    """Handles caching of calendar data."""
    
    def __init__(self, cache_dir: str, cache_duration: int):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            cache_duration: Cache validity duration in seconds
        """
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        try:
            os.makedirs(cache_dir, exist_ok=True)
            logging.info(f"Created or verified calendar cache directory: {cache_dir}")
        except Exception as e:
            # Log the error but don't fail initialization
            logging.error(f"Failed to create cache directory '{cache_dir}': {e}")
            # Use a fallback directory in the current working directory if needed
            try:
                fallback_dir = os.path.join(os.getcwd(), "data", "calendar_cache")
                os.makedirs(fallback_dir, exist_ok=True)
                self.cache_dir = fallback_dir
                logging.warning(f"Using fallback cache directory: {fallback_dir}")
            except Exception as inner_e:
                logging.error(f"Also failed to create fallback cache directory: {inner_e}")
                # Continue without failing - operations will fail gracefully when needed
        
    def get_cache_path(self, key: str) -> str:
        """
        Get the cache file path for a given key.
        
        Args:
            key: The cache key (typically URL + date range)
            
        Returns:
            The path to the cache file
        """
        try:
            key_hash = hashlib.md5(key.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{key_hash}.json")
            # Verify that the directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            return cache_path
        except Exception as e:
            logging.error(f"Error getting cache path for key '{key}': {e}")
            # Return a fallback path in the system temp directory 
            # that won't fail but will cause cache misses
            import tempfile
            temp_dir = tempfile.gettempdir()
            return os.path.join(temp_dir, f"calendar_cache_{key_hash}.json")
        
    def is_valid(self, cache_path: str) -> bool:
        """
        Check if a cache file is valid and not expired.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not os.path.exists(cache_path):
            return False
            
        # Check if cache is expired
        cache_age = time.time() - os.path.getmtime(cache_path)
        return cache_age < self.cache_duration
        
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from cache if available and valid.
        
        Args:
            key: The cache key
            
        Returns:
            The cached data or None if not available
        """
        cache_path = self.get_cache_path(key)
        
        if not self.is_valid(cache_path):
            return None
            
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load cache for {key}: {str(e)}")
            return None
            
    def set(self, key: str, data: Dict[str, Any]) -> None:
        """
        Save data to cache.
        
        Args:
            key: The cache key
            data: The data to cache
        """
        try:
            cache_path = self.get_cache_path(key)
            
            # Ensure the parent directory exists
            parent_dir = os.path.dirname(cache_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Write the data to the cache file
            with open(cache_path, 'w') as f:
                json.dump(data, f)
                
            logging.debug(f"Successfully cached data for key '{key}'")
        except Exception as e:
            logging.warning(f"Failed to cache data for {key}: {str(e)}")
            # Continue without failing - this is non-critical functionality


class CalendarProvider(ABC):
    """Abstract base class for calendar providers."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the calendar provider.
        
        Args:
            logger: Logger instance for this provider
        """
        self.logger = logger
    
    @abstractmethod
    def get_events(self, start_date: Optional[str], end_date: Optional[str]) -> Dict[str, Any]:
        """
        Get events from the calendar within a date range.
        
        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            
        Returns:
            Dictionary with events data
        """
        pass


class ICalProvider(CalendarProvider):
    """Provider for iCalendar URL calendars (read-only)."""
    
    def __init__(self, url: str, cache: CalendarCache, logger: logging.Logger, 
                 max_events: int = 100, default_date_range: int = 7):
        """
        Initialize the iCalendar provider.
        
        Args:
            url: The iCalendar URL
            cache: Cache manager
            logger: Logger instance
            max_events: Maximum number of events to return
            default_date_range: Default number of days to look ahead
        """
        super().__init__(logger)
        self.url = url
        self.cache = cache
        self.max_events = max_events
        self.default_date_range = default_date_range
        
    def get_events(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get events from the iCalendar URL.
        
        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            
        Returns:
            Dictionary with events data
        """
        # Validate and normalize dates
        start_dt = ValidationUtils.validate_date(start_date, "start_date")
        if not start_dt:
            start_dt = datetime.now()
            start_date = start_dt.date().isoformat()
            
        end_dt = ValidationUtils.validate_date(end_date, "end_date")
        if not end_dt:
            # Use default date range
            end_dt = start_dt + timedelta(days=self.default_date_range)
            end_date = end_dt.date().isoformat()
        
        # Use cache if available
        cache_key = f"{self.url}_{start_date}_{end_date}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # Fetch and parse the iCalendar data
        try:
            response = requests.get(self.url, timeout=30)
            response.raise_for_status()
            
            # Parse the iCalendar content using icalendar library
            from icalendar import Calendar
            calendar = Calendar.from_ical(response.text)
            
            events = []
            event_count = 0
            
            # Extract events from the calendar
            for component in calendar.walk('VEVENT'):
                    dtstart = component.get('dtstart')
                    if not dtstart:
                        continue
                        
                    # Get event start time
                    event_start = dtstart.dt
                    
                    # Skip events outside our date range
                    if isinstance(event_start, datetime):
                        event_date = event_start.date()
                    else:
                        # Handle all-day events
                        event_date = event_start
                        
                    # Skip events outside our date range
                    if event_date < start_dt.date() or event_date > end_dt.date():
                        continue
                        
                    # Get event end time
                    dtend = component.get('dtend')
                    if dtend:
                        event_end = dtend.dt
                    else:
                        # Default to 1 hour for events without end time
                        if isinstance(event_start, datetime):
                            event_end = event_start + timedelta(hours=1)
                        else:
                            # All-day event
                            event_end = event_start + timedelta(days=1)
                    
                    # Create event info with timezone conversion
                    # Format datetime objects with proper timezone handling
                    start_str = ""
                    end_str = ""
                    
                    if isinstance(event_start, datetime):
                        # Get timezone from the parent tool if available
                        target_tz = None
                        if hasattr(self, 'parent_tool') and hasattr(self.parent_tool, 'get_tool_timezone'):
                            target_tz = self.parent_tool.get_tool_timezone()
                        else:
                            target_tz = get_default_timezone()
                            
                        # Convert to user timezone
                        start_aware = convert_to_timezone(event_start, target_tz)
                        start_str = start_aware.isoformat()
                    else:
                        # All-day event
                        start_str = str(event_start)
                        
                    if isinstance(event_end, datetime):
                        # Get timezone from the parent tool if available
                        target_tz = None
                        if hasattr(self, 'parent_tool') and hasattr(self.parent_tool, 'get_tool_timezone'):
                            target_tz = self.parent_tool.get_tool_timezone()
                        else:
                            target_tz = get_default_timezone()
                            
                        # Convert to user timezone
                        end_aware = convert_to_timezone(event_end, target_tz)
                        end_str = end_aware.isoformat()
                    else:
                        # All-day event
                        end_str = str(event_end)
                    
                    # Get UID value or generate one if not present
                    uid = component.get('uid')
                    uid_value = str(uid) if uid else str(uuid.uuid4())
                    
                    # Get summary value or use default
                    summary = component.get('summary')
                    summary_value = str(summary) if summary else 'No Title'
                    
                    event_info = {
                        "event_id": uid_value,
                        "summary": summary_value,
                        "start": start_str,
                        "end": end_str
                    }
                    
                    # Add optional properties if available
                    description = component.get('description')
                    if description:
                        event_info["description"] = str(description)
                        
                    location = component.get('location')
                    if location:
                        event_info["location"] = str(location)
                    
                    events.append(event_info)
                    event_count += 1
                    
                    # Limit the number of events to avoid overloading
                    if event_count >= self.max_events:
                        break
            
            result = {
                "events": events,
                "count": len(events),
                "calendar_id": "external",
                "url": self.url,
                "start_date": start_date,
                "end_date": end_date,
                "limited": event_count >= self.max_events
            }
            
            # Cache the result
            self.cache.set(cache_key, result)
            return result
            
        except requests.exceptions.RequestException as e:
            raise ToolError(
                f"Failed to fetch iCalendar URL: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"url": self.url, "error": str(e)}
            )
        except Exception as e:
            raise ToolError(
                f"Failed to parse iCalendar data: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"url": self.url, "error": str(e)}
            )


class CalDAVProvider(CalendarProvider):
    """Provider for CalDAV calendars (read-write)."""
    
    def __init__(self, url: str, username: str, password: str, logger: logging.Logger, 
                 default_date_range: int = 7):
        """
        Initialize the CalDAV provider.
        
        Args:
            url: The CalDAV server URL
            username: Username for authentication
            password: Password for authentication
            logger: Logger instance
            default_date_range: Default number of days to look ahead
        """
        super().__init__(logger)
        self.url = url
        self.username = username
        self.password = password
        self.default_date_range = default_date_range
        self._client = None
        
    def _get_client(self) -> caldav.DAVClient:
        """
        Get (or create) a CalDAV client.
        
        Returns:
            A connected CalDAV client
            
        Raises:
            ToolError: If connection fails
        """
        if self._client:
            return self._client
            
        try:
            client = caldav.DAVClient(
                url=self.url,
                username=self.username,
                password=self.password
            )
            # Test connection by getting principal
            client.principal()
            self._client = client
            return client
        except caldav.lib.error.AuthorizationError:
            raise ToolError(
                "Authentication failed. Check your credentials.",
                ErrorCode.TOOL_INVALID_INPUT,
                {"url": self.url, "username": self.username}
            )
        except Exception as e:
            raise ToolError(
                f"Failed to connect to CalDAV server: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"url": self.url, "error": str(e)}
            )
    
    def _get_calendar(self, calendar_id: str) -> caldav.Calendar:
        """
        Get a specific calendar by ID.
        
        Args:
            calendar_id: The calendar ID
            
        Returns:
            The calendar object
            
        Raises:
            ToolError: If calendar is not found
        """
        client = self._get_client()
        
        try:
            principal = client.principal()
            calendars = principal.calendars()
            
            for calendar in calendars:
                if calendar.name == calendar_id:
                    return calendar
                    
            raise ToolError(
                f"Calendar with ID '{calendar_id}' not found",
                ErrorCode.TOOL_INVALID_INPUT,
                {"calendar_id": calendar_id}
            )
        except Exception as e:
            if isinstance(e, ToolError):
                raise
            raise ToolError(
                f"Failed to access calendar: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"calendar_id": calendar_id, "error": str(e)}
            )
    
    def list_calendars(self) -> Dict[str, Any]:
        """
        List available calendars.
        
        Returns:
            Dictionary with list of calendars
        """
        client = self._get_client()
        
        try:
            principal = client.principal()
            caldav_calendars = principal.calendars()
            
            calendar_list = []
            for calendar in caldav_calendars:
                cal_info = {
                    "id": calendar.name,
                    "display_name": calendar.get_properties([caldav.elements.dav.DisplayName()])["{DAV:}displayname"] or calendar.name,
                    "url": calendar.url
                }
                calendar_list.append(cal_info)
            
            return {
                "calendars": calendar_list,
                "count": len(calendar_list)
            }
        except Exception as e:
            raise ToolError(
                f"Failed to list calendars: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"error": str(e)}
            )
    
    def get_events(self, calendar_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get events from a calendar.
        
        Args:
            calendar_id: The calendar ID
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            
        Returns:
            Dictionary with events data
        """
        # Validate and normalize dates
        start_dt = ValidationUtils.validate_date(start_date, "start_date")
        if not start_dt:
            start_dt = datetime.now()
            start_date = start_dt.date().isoformat()
            
        end_dt = ValidationUtils.validate_date(end_date, "end_date")
        if not end_dt:
            # Use default date range
            end_dt = start_dt + timedelta(days=self.default_date_range)
            end_date = end_dt.date().isoformat()
            
        calendar = self._get_calendar(calendar_id)
        
        try:
            # Get events in the date range
            events = calendar.date_search(
                start=start_dt,
                end=end_dt,
                expand=True  # Expand recurring events
            )
            
            event_list = []
            for event in events:
                ical_data = event.icalendar_component
                
                # Extract event details from iCalendar component with timezone handling
                start_dt = ical_data.get("dtstart").dt
                end_dt = ical_data.get("dtend").dt
                
                # Format datetime objects with proper timezone handling
                start_str = ""
                end_str = ""
                
                # Get timezone from the parent tool if available
                target_tz = None
                if hasattr(self, 'parent_tool') and hasattr(self.parent_tool, 'get_tool_timezone'):
                    target_tz = self.parent_tool.get_tool_timezone()
                else:
                    target_tz = get_default_timezone()
                
                if isinstance(start_dt, datetime):
                    # Convert to user timezone
                    start_aware = convert_to_timezone(start_dt, target_tz)
                    start_str = start_aware.isoformat()
                else:
                    # All-day event
                    start_str = str(start_dt)
                    
                if isinstance(end_dt, datetime):
                    # Convert to user timezone
                    end_aware = convert_to_timezone(end_dt, target_tz)
                    end_str = end_aware.isoformat()
                else:
                    # All-day event
                    end_str = str(end_dt)
                
                event_info = {
                    "event_id": event.id,
                    "summary": str(ical_data.get("summary", "No Title")),
                    "start": start_str,
                    "end": end_str
                }
                
                # Add optional properties if available
                if ical_data.get("description"):
                    event_info["description"] = str(ical_data.get("description"))
                    
                if ical_data.get("location"):
                    event_info["location"] = str(ical_data.get("location"))
                
                event_list.append(event_info)
            
            return {
                "events": event_list,
                "count": len(event_list),
                "calendar_id": calendar_id,
                "start_date": start_date,
                "end_date": end_date
            }
        except Exception as e:
            raise ToolError(
                f"Failed to get events: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"calendar_id": calendar_id, "error": str(e)}
            )
    
    def create_event(
        self,
        calendar_id: str,
        summary: str,
        start_time: str,
        end_time: str,
        description: Optional[str] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new event.
        
        Args:
            calendar_id: The calendar ID
            summary: Event title
            start_time: Event start time in ISO format
            end_time: Event end time in ISO format
            description: Optional event description
            location: Optional event location
            
        Returns:
            Dictionary with the created event
        """
        # Validate parameters
        ValidationUtils.require_non_empty_string(summary, "summary")
        start_dt = ValidationUtils.validate_datetime(start_time, "start_time")
        end_dt = ValidationUtils.validate_datetime(end_time, "end_time")
        
        if not start_dt or not end_dt:
            raise ToolError(
                "Start and end times are required",
                ErrorCode.TOOL_INVALID_INPUT,
                {"start_time": start_time, "end_time": end_time}
            )
            
        if end_dt <= start_dt:
            raise ToolError(
                "End time must be after start time",
                ErrorCode.TOOL_INVALID_INPUT,
                {"start_time": start_time, "end_time": end_time}
            )
        
        calendar = self._get_calendar(calendar_id)
        
        try:
            # Create event with proper timezone handling
            event_id = str(uuid.uuid4())
            
            # Ensure datetimes are in UTC for iCalendar format
            now_utc = convert_to_timezone(datetime.now(), "UTC")
            start_utc = convert_to_timezone(start_dt, "UTC")
            end_utc = convert_to_timezone(end_dt, "UTC")
            
            # Format in iCalendar format with Z suffix for UTC
            ical_data = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//MIRA//CalendarTool//EN
BEGIN:VEVENT
UID:{event_id}
DTSTAMP:{now_utc.strftime("%Y%m%dT%H%M%SZ")}
DTSTART:{start_utc.strftime("%Y%m%dT%H%M%SZ")}
DTEND:{end_utc.strftime("%Y%m%dT%H%M%SZ")}
SUMMARY:{summary}"""
            
            if description:
                ical_data += f"\nDESCRIPTION:{description}"
                
            if location:
                ical_data += f"\nLOCATION:{location}"
                
            ical_data += """\nEND:VEVENT
END:VCALENDAR"""
            
            # Add event to calendar
            event = calendar.save_event(ical_data)
            
            # Format start and end times in user's timezone for the response
            tool_timezone = self.get_tool_timezone()
            start_formatted = format_datetime(start_dt, "standard", tool_timezone)
            end_formatted = format_datetime(end_dt, "standard", tool_timezone)
            
            return {
                "event": {
                    "event_id": event.id,
                    "summary": summary,
                    "start": start_formatted,
                    "end": end_formatted,
                    "description": description,
                    "location": location,
                    "timezone": tool_timezone
                },
                "calendar_id": calendar_id
            }
        except Exception as e:
            raise ToolError(
                f"Failed to create event: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"calendar_id": calendar_id, "error": str(e)}
            )
    
    def delete_event(self, calendar_id: str, event_id: str) -> Dict[str, Any]:
        """
        Delete an event.
        
        Args:
            calendar_id: The calendar ID
            event_id: The event ID
            
        Returns:
            Dictionary with deletion status
        """
        # Validate parameters
        ValidationUtils.require_non_empty_string(event_id, "event_id")
        
        calendar = self._get_calendar(calendar_id)
        
        try:
            # Find the specific event
            event = None
            events = calendar.events()
            
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
                "event_id": event_id,
                "calendar_id": calendar_id,
                "deleted": True
            }
        except Exception as e:
            if isinstance(e, ToolError):
                raise
            raise ToolError(
                f"Failed to delete event: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"calendar_id": calendar_id, "event_id": event_id, "error": str(e)}
            )


class CalendarTool(Tool):
    """
    Calendar management tool for viewing and managing events.
    
    This tool provides functionality to view events across multiple calendars,
    create new events, and manage existing appointments. It supports calendars
    from various sources and provides a unified interface for accessing all
    calendar information.
    """
    
    def get_tool_timezone(self) -> str:
        """
        Get the timezone to use for this tool.
        
        Returns:
            IANA timezone name from system config
        """
        # Use the global system timezone from config
        return get_default_timezone()

    name = "calendar_tool"
    simple_description = """
    Manages calendar events for viewing appointments and scheduling. Use this tool when the user needs to check their calendar, 
    see upcoming events, or manage their personal appointments."""
    
    implementation_details = """
    OPERATIONS:
    - list_all_events: Lists events from all configured calendars within a date range
      Parameters:
        calendar_name (optional): Name of a specific calendar to query (if not provided, shows events from all configured calendars)
        start_date (optional, default=today): Start date in ISO format (YYYY-MM-DD)
        end_date (optional, default=7 days from start): End date in ISO format (YYYY-MM-DD)
    
    - list_calendars: Lists available calendars on the server
      Parameters:
        url (optional): The calendar server URL
        username (optional): Username for authentication
        password (optional): Password for authentication
    
    - list_events: Lists events from a specific calendar within a date range
      Parameters:
        url (optional): The calendar server URL
        username (optional): Username for authentication
        password (optional): Password for authentication
        calendar_id (required): The calendar ID to query
        start_date (optional, default=today): Start date in ISO format (YYYY-MM-DD)
        end_date (optional, default=7 days from start): End date in ISO format (YYYY-MM-DD)
    
    - create_event: Creates a new calendar event
      Parameters:
        url (optional): The calendar server URL
        username (optional): Username for authentication
        password (optional): Password for authentication
        calendar_id (required): The calendar ID to add the event to
        summary (required): Event title/summary
        start_time (required): Start time in ISO format (YYYY-MM-DDTHH:MM:SS)
        end_time (required): End time in ISO format (YYYY-MM-DDTHH:MM:SS)
        description (optional): Event description
        location (optional): Event location
    
    - delete_event: Deletes a calendar event
      Parameters:
        url (optional): The calendar server URL
        username (optional): Username for authentication
        password (optional): Password for authentication
        calendar_id (required): The calendar ID containing the event
        event_id (required): The unique ID of the event to delete
        
    - read_ical_url: Reads events from a calendar URL
      Parameters:
        ical_url (optional): URL to the calendar file
        start_date (optional, default=today): Start date in ISO format (YYYY-MM-DD)
        end_date (optional, default=7 days from start): End date in ISO format (YYYY-MM-DD)
    
    RESPONSE FORMAT:
    - All operations return a dictionary with success status and relevant data
    - Events include summary, start/end times, and optional location and description
    - Errors include detailed information about what went wrong
    
    USAGE NOTES:
    - Calendars can be configured in the tool settings with user-friendly names
    - Use list_all_events to see events from all configured calendars at once
    - For viewing a specific calendar by name, use the calendar_name parameter
    - Date ranges for listing events should be reasonable (e.g., 1-30 days)
    - All date and time parameters must be in ISO format
    - Calendar data is cached for improved performance
    
    LIMITATIONS:
    - Some calendars may be read-only depending on their configuration
    - Does not support recurring event creation (only displays them)
    - Limited to basic event properties (no attachments, attendees, or notifications)
    """
    
    description = simple_description + implementation_details
    
    def __init__(self):
        """Initialize the calendar tool."""
        super().__init__()
        self.logger.info("CalendarTool initialized")
        
        try:
            # Get configuration from config manager (not registry directly)
            from config import config
            tool_config = config.get_tool_config("calendar_tool")
            
            # Load default calendar URL from environment - required, no fallback
            default_calendar_url = os.environ.get("DEFAULT_CALENDAR_URL")
            if not default_calendar_url:
                raise ToolError(
                    "DEFAULT_CALENDAR_URL environment variable must be set",
                    ErrorCode.MISSING_ENV_VAR,
                    {"param": "DEFAULT_CALENDAR_URL"}
                )
            tool_config.default_calendar_url = default_calendar_url
            self.logger.info("Using calendar URL from environment variable")
            
            # Create cache directory if it doesn't exist
            self.logger.info(f"Using cache directory: {tool_config.cache_directory}")
            self.cache = CalendarCache(tool_config.cache_directory, tool_config.cache_duration)
            
            # Store the config for later use
            self.config = tool_config
        except Exception as e:
            self.logger.error(f"Error initializing CalendarCache: {e}")
            # Create a fallback cache in the data directory
            fallback_cache_dir = os.path.join("data", "tools", "calendar_tool", "cache")
            os.makedirs(fallback_cache_dir, exist_ok=True)
            self.logger.info(f"Using fallback cache directory: {fallback_cache_dir}")
            
            # Use default cache duration if config retrieval failed
            fallback_duration = 3600  # 1 hour in seconds
            self.cache = CalendarCache(fallback_cache_dir, fallback_duration)
            
            # Create a minimal config for fallback
            from pydantic import BaseModel, Field
            class MinimalConfig(BaseModel):
                cache_directory: str = Field(default=fallback_cache_dir)
                cache_duration: int = Field(default=fallback_duration)
                default_calendar_url: str = Field(default=None)
            
            # Initialize minimal config
            self.config = MinimalConfig()
            
            # Load default calendar URL from environment - required, no fallback
            default_calendar_url = os.environ.get("DEFAULT_CALENDAR_URL")
            if not default_calendar_url:
                raise ToolError(
                    "DEFAULT_CALENDAR_URL environment variable must be set",
                    ErrorCode.MISSING_ENV_VAR,
                    {"param": "DEFAULT_CALENDAR_URL"}
                )
            self.config.default_calendar_url = default_calendar_url
        
    def _create_caldav_provider(self, url: str, username: str, password: str) -> CalDAVProvider:
        """
        Create a CalDAV provider instance.
        
        Args:
            url: The CalDAV server URL
            username: Username for authentication
            password: Password for authentication
            
        Returns:
            A configured CalDAV provider
        """
        provider = CalDAVProvider(url, username, password, self.logger, 
                                 default_date_range=self.config.default_date_range)
        # Set parent tool reference for timezone access
        provider.parent_tool = self
        return provider
        
    def _create_ical_provider(self, url: str) -> ICalProvider:
        """
        Create an iCalendar provider instance.
        
        Args:
            url: The iCalendar URL
            
        Returns:
            A configured iCalendar provider
        """
        provider = ICalProvider(url, self.cache, self.logger, 
                               max_events=self.config.max_events,
                               default_date_range=self.config.default_date_range)
        # Set parent tool reference for timezone access
        provider.parent_tool = self
        return provider
        
    def _fetch_all_calendars(self, calendar_name: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch events from all configured calendars.
        
        Args:
            calendar_name: Optional name of a specific calendar to fetch
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)
            
        Returns:
            Dictionary with events from all calendars
        """
        # Set default date range if not provided
        if not start_date:
            start_date = datetime.now().date().isoformat()
        
        if not end_date:
            # Get the default date range from config
            days_ahead = self.config.default_date_range
            
            # Use the default date range
            start_dt = datetime.fromisoformat(start_date)
            end_dt = start_dt + timedelta(days=days_ahead)
            end_date = end_dt.date().isoformat()
        
        # Get a list of calendars to fetch
        calendars = dict(self.config.calendars)  # Make a copy
        
        # Use the default calendar URL from environment variable (must be set per our rule)
        # At this point, default_calendar_url must exist because we checked during initialization
        # If it doesn't exist due to some unexpected reason, this is an error
        default_url = self.config.default_calendar_url
        if not default_url:
            raise ToolError(
                "DEFAULT_CALENDAR_URL environment variable must be set",
                ErrorCode.MISSING_ENV_VAR,
                {"param": "DEFAULT_CALENDAR_URL"}
            )
        
        # Create a CalendarEntry for the default URL if it doesn't exist in calendars
        default_entry = CalendarEntry(
            name="Default Calendar",
            url=default_url,
            type="ical"
        )
        # Add to calendars if not already present
        if "default" not in calendars:
            calendars["default"] = default_entry
        
        if not calendars:
            # No calendars configured, return empty result
            return {
                "success": True,
                "calendars": [],
                "total_events": 0,
                "start_date": start_date,
                "end_date": end_date,
                "message": "No calendars configured in settings"
            }
        
        # Filter by name if specified
        if calendar_name:
            # Find the calendar with the specified name
            found_calendar = None
            for calendar_id, calendar in calendars.items():
                if calendar.name.lower() == calendar_name.lower():
                    found_calendar = (calendar_id, calendar)
                    break
            
            if not found_calendar:
                raise ToolError(
                    f"Calendar with name '{calendar_name}' not found in configuration",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"calendar_name": calendar_name}
                )
            
            # Use only the specified calendar
            calendars_to_fetch = {found_calendar[0]: found_calendar[1]}
        else:
            # Use all calendars
            calendars_to_fetch = calendars
        
        # Fetch events from each calendar
        calendar_results = []
        total_events = 0
        
        # Get password from environment variable
        password = os.environ.get("CALDAV_PASSWORD", "")
        
        for calendar_id, calendar in calendars_to_fetch.items():
            try:
                # Determine how to fetch events based on calendar type
                if calendar.type.lower() == 'ical':
                    # Fetch from iCalendar URL
                    provider = self._create_ical_provider(calendar.url)
                    result = provider.get_events(start_date, end_date)
                    
                    calendar_result = {
                        "name": calendar.name,
                        "events": result["events"],
                        "count": result["count"],
                        "type": "ical"
                    }
                elif calendar.type.lower() == 'caldav':
                    # Check for required fields
                    if not calendar.username:
                        raise ToolError(
                            f"Username required for CalDAV calendar '{calendar.name}'",
                            ErrorCode.TOOL_INVALID_INPUT,
                            {"calendar_name": calendar.name}
                        )
                    
                    if not calendar.calendar_id:
                        raise ToolError(
                            f"Calendar ID required for CalDAV calendar '{calendar.name}'",
                            ErrorCode.TOOL_INVALID_INPUT,
                            {"calendar_name": calendar.name}
                        )
                    
                    # Fetch from CalDAV server
                    provider = self._create_caldav_provider(calendar.url, calendar.username, password)
                    result = provider.get_events(calendar.calendar_id, start_date, end_date)
                    
                    calendar_result = {
                        "name": calendar.name,
                        "events": result["events"],
                        "count": result["count"],
                        "type": "caldav"
                    }
                else:
                    raise ToolError(
                        f"Invalid calendar type '{calendar.type}' for calendar '{calendar.name}'",
                        ErrorCode.TOOL_INVALID_INPUT,
                        {"calendar_name": calendar.name, "calendar_type": calendar.type}
                    )
                
                calendar_results.append(calendar_result)
                total_events += calendar_result["count"]
                
            except Exception as e:
                # Log the error but continue with other calendars
                self.logger.error(f"Error fetching calendar '{calendar.name}': {str(e)}")
                calendar_result = {
                    "name": calendar.name,
                    "events": [],
                    "count": 0,
                    "error": str(e),
                    "type": calendar.type
                }
                calendar_results.append(calendar_result)
        
        return {
            "success": True,
            "calendars": calendar_results,
            "total_events": total_events,
            "start_date": start_date,
            "end_date": end_date
        }

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
        event_id: Optional[str] = None,
        ical_url: Optional[str] = None,
        calendar_name: Optional[str] = None,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        calendar_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the calendar tool with the specified action.
        
        Args:
            action: The operation to perform (list_all_events, list_calendars, list_events, create_event, delete_event, read_ical_url)
            start_date: Start date for listing events (YYYY-MM-DD)
            end_date: End date for listing events (YYYY-MM-DD)
            summary: Event title/summary for create_event
            start_time: Event start time for create_event (YYYY-MM-DDTHH:MM:SS)
            end_time: Event end time for create_event (YYYY-MM-DDTHH:MM:SS)
            description: Event description for create_event
            location: Event location for create_event
            event_id: Event ID for delete_event
            ical_url: URL to an iCalendar (.ics) file for read_ical_url
            calendar_name: Name of a specific calendar for list_all_events
            url: CalDAV server URL for direct operations
            username: Username for CalDAV authentication
            password: Password for CalDAV authentication
            calendar_id: Calendar ID for CalDAV operations
            
        Returns:
            Dictionary containing the operation results
        """
        self.logger.info(f"Running calendar tool with action: {action}")
        
        with error_context(
            component_name=self.name,
            operation=f"executing calendar operation '{action}'",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            if action == "list_all_events":
                # Fetch events from all configured calendars
                return self._fetch_all_calendars(calendar_name, start_date, end_date)
                
            elif action == "read_ical_url":
                # If no ical_url provided, use the default calendar URL from environment
                if not ical_url:
                    # At this point, default_calendar_url must exist because we checked during initialization
                    # If it doesn't exist due to some unexpected reason, this is an error
                    if not self.config.default_calendar_url:
                        raise ToolError(
                            "DEFAULT_CALENDAR_URL environment variable must be set",
                            ErrorCode.MISSING_ENV_VAR,
                            {"param": "DEFAULT_CALENDAR_URL"}
                        )
                    ical_url = self.config.default_calendar_url
                
                ValidationUtils.require_non_empty_string(ical_url, "ical_url")
                
                # Create provider and fetch events
                provider = self._create_ical_provider(ical_url)
                result = provider.get_events(start_date, end_date)
                
                return {
                    "success": True,
                    **result
                }
                
            else:
                # For CalDAV operations, use provided parameters or defaults
                srv_url = url or self.config.default_url
                srv_username = username or self.config.default_username
                srv_password = password or os.environ.get("CALDAV_PASSWORD", "")
                srv_calendar_id = calendar_id or self.config.default_calendar_id
                
                # Validate common parameters for CalDAV operations
                ValidationUtils.require_non_empty_string(srv_url, "url")
                ValidationUtils.require_non_empty_string(srv_username, "username")
                ValidationUtils.require_non_empty_string(srv_password, "password")
                
                # Create provider
                provider = self._create_caldav_provider(srv_url, srv_username, srv_password)
                
                # Route to appropriate action
                if action == "list_calendars":
                    result = provider.list_calendars()
                    return {
                        "success": True,
                        **result
                    }
                    
                elif action == "list_events":
                    ValidationUtils.require_non_empty_string(srv_calendar_id, "calendar_id")
                    result = provider.get_events(srv_calendar_id, start_date, end_date)
                    return {
                        "success": True,
                        **result
                    }
                    
                elif action == "create_event":
                    ValidationUtils.require_non_empty_string(srv_calendar_id, "calendar_id")
                    ValidationUtils.require_non_empty_string(summary, "summary")
                    ValidationUtils.require_non_empty_string(start_time, "start_time")
                    ValidationUtils.require_non_empty_string(end_time, "end_time")
                    
                    result = provider.create_event(
                        srv_calendar_id, summary, start_time, end_time, description, location
                    )
                    return {
                        "success": True,
                        "message": "Event created successfully",
                        **result
                    }
                    
                elif action == "delete_event":
                    ValidationUtils.require_non_empty_string(srv_calendar_id, "calendar_id")
                    ValidationUtils.require_non_empty_string(event_id, "event_id")
                    
                    result = provider.delete_event(srv_calendar_id, event_id)
                    return {
                        "success": True,
                        "message": "Event deleted successfully",
                        **result
                    }
                    
                else:
                    raise ToolError(
                        f"Invalid action: {action}. Must be one of: list_all_events, list_calendars, list_events, create_event, delete_event, read_ical_url",
                        ErrorCode.TOOL_INVALID_INPUT,
                        {"provided_action": action}
                    )