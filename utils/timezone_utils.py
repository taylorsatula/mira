"""
Timezone utility functions for handling timezone conversions.

This module provides consistent timezone handling across the application,
with functions to validate timezone names, convert between timezones,
and handle time formatting.

Core principles:
- All datetimes are stored in UTC internally
- Conversion to local time happens only at display time
- All datetime objects should be timezone-aware
- Consistent formats are used for string representations
"""

import logging
from datetime import datetime, timezone, timedelta, UTC
from typing import Optional, Union, Any, Dict

import pytz
from zoneinfo import ZoneInfo, available_timezones
from dateutil import parser

from config import config
from errors import ToolError, ErrorCode

logger = logging.getLogger(__name__)

# Dictionary mapping common abbreviations to IANA timezone names
COMMON_TIMEZONE_ALIASES = {
    "EST": "America/New_York",
    "CST": "America/Chicago",
    "MST": "America/Denver",
    "PST": "America/Los_Angeles",
    "EDT": "America/New_York",
    "CDT": "America/Chicago",
    "MDT": "America/Denver",
    "PDT": "America/Los_Angeles",
    "GMT": "Etc/GMT",
    "UTC": "UTC",
}

# Time format patterns
TIME_FORMATS = {
    "standard": "%H:%M:%S",
    "short": "%H:%M",
    "date_time": "%Y-%m-%d %H:%M:%S",
    "date_time_short": "%Y-%m-%d %H:%M",
    "date": "%Y-%m-%d",
    "iso": "%Y-%m-%dT%H:%M:%S%z",
    "iso_ms": "%Y-%m-%dT%H:%M:%S.%f%z",
    "database": "%Y-%m-%d %H:%M:%S.%f"
}

# Singleton UTC timezone object for performance
UTC_TIMEZONE = timezone.utc


def validate_timezone(tz_name: str) -> str:
    """
    Validate and normalize timezone name.

    Args:
        tz_name: Timezone name or abbreviation to validate

    Returns:
        Normalized IANA timezone name

    Raises:
        ToolError: If the timezone is invalid
    """
    if not tz_name:
        return get_default_timezone()

    # Check if it's a common abbreviation
    if tz_name in COMMON_TIMEZONE_ALIASES:
        return COMMON_TIMEZONE_ALIASES[tz_name]

    # Check if it's a valid IANA timezone
    try:
        # Try both pytz and zoneinfo to be thorough
        if tz_name in available_timezones() or tz_name in pytz.all_timezones:
            return tz_name
    except (pytz.exceptions.UnknownTimeZoneError, LookupError):
        # Strict validation - if not in our alias list and not a valid IANA name, error
        raise ToolError(
            f"Invalid timezone: '{tz_name}'. Please use a valid IANA timezone name like "
            "'America/New_York' or 'Europe/London'.",
            ErrorCode.TOOL_INVALID_INPUT,
        )
    
    # If we get here, the timezone was not found
    raise ToolError(
        f"Invalid timezone: '{tz_name}'. Please use a valid IANA timezone name like "
        "'America/New_York' or 'Europe/London'.",
        ErrorCode.TOOL_INVALID_INPUT,
    )


def get_default_timezone() -> str:
    """
    Get the default system timezone.

    Returns:
        IANA timezone name
    """
    system_tz = config.system.timezone
    
    # If it's already a valid IANA timezone, return it
    try:
        ZoneInfo(system_tz)
        return system_tz
    except (KeyError, LookupError):
        pass
    
    try:
        pytz.timezone(system_tz)
        return system_tz
    except pytz.exceptions.UnknownTimeZoneError:
        # Try to convert from common abbreviation
        if system_tz in COMMON_TIMEZONE_ALIASES:
            return COMMON_TIMEZONE_ALIASES[system_tz]
        
        # Default to UTC if all else fails
        logger.warning(f"Invalid system timezone '{system_tz}', defaulting to UTC")
        return "UTC"


def get_timezone_instance(tz_name: Optional[str] = None) -> ZoneInfo:
    """
    Get a timezone instance from a timezone name using ZoneInfo.
    
    Prefer this over pytz for modern Python applications.

    Args:
        tz_name: Timezone name or abbreviation (defaults to system timezone)

    Returns:
        ZoneInfo timezone instance
    """
    normalized_tz = validate_timezone(tz_name or get_default_timezone())
    
    # Handle 'UTC' specially for performance
    if normalized_tz == "UTC":
        return UTC_TIMEZONE
    
    return ZoneInfo(normalized_tz)


def get_pytz_timezone_instance(tz_name: Optional[str] = None) -> pytz.BaseTzInfo:
    """
    Get a pytz timezone instance from a timezone name.
    
    This function exists for compatibility with pytz which is used by some libraries.

    Args:
        tz_name: Timezone name or abbreviation (defaults to system timezone)

    Returns:
        pytz timezone instance
    """
    normalized_tz = validate_timezone(tz_name or get_default_timezone())
    return pytz.timezone(normalized_tz)


def ensure_utc(dt: datetime) -> datetime:
    """
    Ensure a datetime object is UTC timezone-aware.
    
    - If naive, assume it's UTC and make it aware
    - If aware but not UTC, convert to UTC
    - If already UTC, return as is
    
    Args:
        dt: The datetime object to ensure is UTC timezone-aware
        
    Returns:
        UTC timezone-aware datetime
    """
    if dt.tzinfo is None:
        # Naive datetime, assume UTC
        return dt.replace(tzinfo=UTC_TIMEZONE)
    elif dt.tzinfo != UTC_TIMEZONE:
        # Aware but not UTC, convert
        return dt.astimezone(UTC_TIMEZONE)
    
    # Already UTC aware
    return dt


def utc_now() -> datetime:
    """
    Get the current UTC datetime with timezone info.
    
    This is the preferred way to get the current time in the application.
    
    Returns:
        Current datetime in UTC with timezone info
    """
    return datetime.now(UTC_TIMEZONE)


def convert_to_timezone(
    dt: datetime, 
    target_tz: Optional[str] = None,
    from_tz: Optional[str] = None
) -> datetime:
    """
    Convert a datetime object to the target timezone.

    Args:
        dt: The datetime object to convert
        target_tz: Target timezone name (defaults to system timezone)
        from_tz: Source timezone if dt is naive (defaults to UTC)

    Returns:
        Timezone-aware datetime in the target timezone
    """
    # Handle naive datetime objects
    if dt.tzinfo is None:
        from_timezone = get_timezone_instance(from_tz or "UTC")
        # Attach the timezone without adjusting the time
        dt = dt.replace(tzinfo=from_timezone)
    
    # Convert to target timezone
    target_timezone = get_timezone_instance(target_tz or get_default_timezone())
    return dt.astimezone(target_timezone)


def convert_to_utc(dt: datetime, from_tz: Optional[str] = None) -> datetime:
    """
    Convert a datetime object to UTC.
    
    This is a convenience function that makes the UTC conversion explicit.

    Args:
        dt: The datetime object to convert
        from_tz: Source timezone if dt is naive

    Returns:
        Timezone-aware datetime in UTC
    """
    return convert_to_timezone(dt, "UTC", from_tz)


def convert_from_utc(dt: datetime, to_tz: Optional[str] = None) -> datetime:
    """
    Convert a UTC datetime to a target timezone.
    
    This function assumes dt is already in UTC (will convert if not).

    Args:
        dt: The UTC datetime object to convert
        to_tz: Target timezone (defaults to system timezone)

    Returns:
        Timezone-aware datetime in the target timezone
    """
    # First ensure dt is in UTC
    dt = ensure_utc(dt)
    
    # Then convert to target timezone
    return convert_to_timezone(dt, to_tz or get_default_timezone())


def format_datetime(
    dt: datetime,
    format_type: str = "standard",
    tz_name: Optional[str] = None,
    include_timezone: bool = False
) -> str:
    """
    Format a datetime with the specified format in the target timezone.

    Args:
        dt: The datetime object to format
        format_type: The format type (standard, short, date_time, iso, etc.)
        tz_name: Target timezone name (defaults to system timezone)
        include_timezone: Whether to include the timezone name in the output

    Returns:
        Formatted datetime string
    """
    # Convert to target timezone
    tz_dt = convert_to_timezone(dt, tz_name)
    
    # Get format pattern
    format_pattern = TIME_FORMATS.get(format_type, TIME_FORMATS["standard"])
    
    # Format the datetime
    formatted = tz_dt.strftime(format_pattern)
    
    # Add timezone name if requested
    if include_timezone:
        tz_name = tz_name or get_default_timezone()
        formatted += f" {tz_name}"
    
    return formatted


def format_utc_for_storage(dt: datetime) -> str:
    """
    Format a datetime for database storage.
    
    Always stores as UTC without timezone info (timezone awareness
    is handled at the application level).

    Args:
        dt: The datetime object to format

    Returns:
        UTC datetime string in database format
    """
    utc_dt = ensure_utc(dt)
    return utc_dt.strftime(TIME_FORMATS["database"])


def format_utc_iso(dt: datetime, include_ms: bool = True) -> str:
    """
    Format a datetime as ISO 8601 in UTC.
    
    Args:
        dt: The datetime object to format
        include_ms: Whether to include milliseconds

    Returns:
        ISO 8601 formatted UTC datetime string
    """
    utc_dt = ensure_utc(dt)
    format_type = "iso_ms" if include_ms else "iso"
    return utc_dt.strftime(TIME_FORMATS[format_type])


def parse_time_string(
    time_str: str, 
    tz_name: Optional[str] = None,
    reference_date: Optional[datetime] = None
) -> datetime:
    """
    Parse a time string into a datetime object.

    Handles various formats including:
    - Full ISO format: "2023-04-01T14:30:00"
    - Date only: "2023-04-01"
    - Time only: "14:30:00" or "14:30" (uses reference_date for the date part)

    Args:
        time_str: The time string to parse
        tz_name: Timezone name to interpret the time in (defaults to system timezone)
        reference_date: Reference date to use for time-only strings (defaults to today)

    Returns:
        Timezone-aware datetime object

    Raises:
        ToolError: If the time string cannot be parsed
    """
    tz = get_timezone_instance(tz_name)
    
    # Get reference date in the target timezone
    if reference_date is None:
        reference_date = utc_now()
    
    reference_date = convert_to_timezone(reference_date, tz_name)
    
    # Try parsing ISO format
    try:
        dt = datetime.fromisoformat(time_str)
        # Add timezone if naive
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        return dt
    except ValueError:
        pass
    
    # Try different formats
    
    # Time-only format (HH:MM:SS or HH:MM)
    if ":" in time_str and "T" not in time_str:
        # Split into hours, minutes, seconds
        time_parts = time_str.split(":")
        try:
            hour = int(time_parts[0])
            minute = int(time_parts[1]) if len(time_parts) > 1 else 0
            second = int(time_parts[2]) if len(time_parts) > 2 else 0
            
            # Create datetime using reference date's year, month, day
            dt = reference_date.replace(
                hour=hour,
                minute=minute,
                second=second,
                microsecond=0
            )
            
            # If time has already passed today, use tomorrow
            if dt < reference_date:
                dt = dt + timedelta(days=1)
                
            return dt
        except (ValueError, IndexError):
            pass
    
    # Try dateutil parser as a fallback for natural language and various formats
    try:
        # Parse with dateutil - it can handle many formats
        dt = parser.parse(time_str)
        
        # Add timezone if naive
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
            
        return dt
    except (ValueError, parser.ParserError):
        pass
    
    # If we get here, the format is not recognized
    raise ToolError(
        f"Invalid time format: '{time_str}'. Please use ISO format (YYYY-MM-DDTHH:MM:SS) "
        "or time-only format (HH:MM:SS).",
        ErrorCode.TOOL_INVALID_INPUT,
    )


def parse_utc_time_string(time_str: str) -> datetime:
    """
    Parse a time string as UTC.
    
    Args:
        time_str: The time string to parse
    
    Returns:
        UTC timezone-aware datetime object
    """
    dt = parse_time_string(time_str, "UTC")
    return ensure_utc(dt)


def localize_datetime(dt: datetime, tz_name: Optional[str] = None) -> datetime:
    """
    Localize a naive datetime to a specific timezone without conversion.
    
    Unlike convert_to_timezone, this doesn't adjust the time value,
    it just attaches the timezone info. Use this when you know the 
    datetime is already in the target timezone but lacks tzinfo.

    Args:
        dt: The naive datetime object to localize
        tz_name: Timezone name (defaults to system timezone)
        
    Returns:
        Timezone-aware datetime in the specified timezone
    
    Raises:
        ValueError: If dt is already timezone-aware
    """
    if dt.tzinfo is not None:
        raise ValueError("Expected naive datetime, got timezone-aware datetime")
    
    tz = get_timezone_instance(tz_name or get_default_timezone())
    return dt.replace(tzinfo=tz)


def datetime_to_dict(dt: datetime, include_timezone: bool = True) -> Dict[str, Any]:
    """
    Convert a datetime object to a dictionary representation.
    
    Useful for JSON serialization and API responses.

    Args:
        dt: The datetime object to convert
        include_timezone: Whether to include timezone info
        
    Returns:
        Dictionary with datetime components
    """
    dt = ensure_utc(dt)
    
    result = {
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "hour": dt.hour,
        "minute": dt.minute,
        "second": dt.second,
        "microsecond": dt.microsecond,
        "iso": format_utc_iso(dt),
    }
    
    if include_timezone:
        result["timezone"] = "UTC"
    
    return result