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
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import pytz
from zoneinfo import ZoneInfo, available_timezones
from dateutil import parser

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
        ValueError: If the timezone is invalid
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
        logger.error(f"Timezone validation failed for '{tz_name}' - pytz lookup raised exception")
        raise ValueError(
            f"Invalid timezone: '{tz_name}'. Please use a valid IANA timezone name like "
            "'America/New_York' or 'Europe/London'."
        )
    
    # If we get here, the timezone was not found
    logger.error(f"Invalid timezone validation failed for '{tz_name}' - not found in IANA or pytz databases")
    raise ValueError(
        f"Invalid timezone: '{tz_name}'. Please use a valid IANA timezone name like "
        "'America/New_York' or 'Europe/London'."
    )


def get_default_timezone() -> str:
    """
    Get the system default timezone (UTC).

    Returns:
        IANA timezone name
    """
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
    
    # Handle 'UTC' specially — return ZoneInfo to match declared return type
    if normalized_tz == "UTC":
        return ZoneInfo("UTC")
    
    return ZoneInfo(normalized_tz)


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
    Format a datetime with the specified format.
    
    If tz_name is provided, converts to that timezone first.
    Otherwise, formats the datetime as-is in its current timezone.

    Args:
        dt: The datetime object to format
        format_type: The format type (standard, short, date_time, iso, etc.)
        tz_name: Target timezone name (if provided, converts to this timezone)
        include_timezone: Whether to include the timezone name in the output

    Returns:
        Formatted datetime string
    """
    # Only convert if a target timezone is explicitly specified
    if tz_name:
        tz_dt = convert_to_timezone(dt, tz_name)
    else:
        tz_dt = dt
    
    # Get format pattern
    format_pattern = TIME_FORMATS.get(format_type, TIME_FORMATS["standard"])
    
    # Format the datetime
    formatted = tz_dt.strftime(format_pattern)
    
    # Add timezone name if requested
    if include_timezone:
        # Get the timezone name from the datetime object or use the provided one
        if hasattr(tz_dt, 'tzinfo') and tz_dt.tzinfo:
            tz_name = str(tz_dt.tzinfo)
        else:
            tz_name = tz_name or get_default_timezone()
        formatted += f" {tz_name}"
    
    return formatted


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


def format_relative_time(dt: datetime, reference_time: Optional[datetime] = None) -> str:
    """
    Format a datetime as relative time (e.g., "5 hours ago", "2 days ago").

    Uses fine-grained granularity:
    - < 60 seconds: "just now"
    - < 60 minutes: "X minute(s) ago"
    - < 24 hours: "X hour(s) ago"
    - < 7 days: "X day(s) ago"
    - < 30 days: "X week(s) ago"
    - < 365 days: "X month(s) ago"
    - >= 365 days: "X year(s) ago"

    Args:
        dt: The datetime to format (will be converted to UTC if needed)
        reference_time: Optional reference time (defaults to current UTC time)

    Returns:
        Human-readable relative time string

    Raises:
        ValueError: If dt cannot be converted to UTC datetime
    """
    # Ensure both datetimes are UTC-aware
    dt_utc = ensure_utc(dt)
    ref_time = reference_time if reference_time else utc_now()
    ref_time_utc = ensure_utc(ref_time)

    # Calculate time delta
    delta = ref_time_utc - dt_utc
    total_seconds = delta.total_seconds()

    # Handle future times
    if total_seconds < 0:
        delta = dt_utc - ref_time_utc
        total_seconds = delta.total_seconds()
        future = True
    else:
        future = False

    # Calculate time units
    seconds = int(total_seconds)
    minutes = seconds // 60
    hours = minutes // 60
    days = delta.days
    weeks = days // 7
    months = days // 30  # Approximate
    years = days // 365  # Approximate

    # Format based on granularity
    if seconds < 60:
        return "just now"
    elif minutes < 60:
        unit = "minute" if minutes == 1 else "minutes"
        time_str = f"{minutes} {unit}"
    elif hours < 24:
        unit = "hour" if hours == 1 else "hours"
        time_str = f"{hours} {unit}"
    elif days < 7:
        unit = "day" if days == 1 else "days"
        time_str = f"{days} {unit}"
    elif days < 30:
        unit = "week" if weeks == 1 else "weeks"
        time_str = f"{weeks} {unit}"
    elif days < 365:
        unit = "month" if months == 1 else "months"
        time_str = f"{months} {unit}"
    else:
        unit = "year" if years == 1 else "years"
        time_str = f"{years} {unit}"

    # Add direction
    if future:
        return f"in {time_str}"
    else:
        return f"{time_str} ago"


_SMALL_NUMBERS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve",
}


def _number_word(n: int) -> str:
    """Word form for 1-12, digits for larger."""
    return _SMALL_NUMBERS.get(n, str(n))


def _fractional_duration(whole: int, frac: float, unit: str) -> str:
    """
    Format whole + fractional count as natural language.

    Thresholds:
      frac < 0.2  → "about {n} {unit}"
      0.2 ≤ frac < 0.4 → "{n} and change {unit}s"
      0.4 ≤ frac < 0.7 → "{n} and a half {unit}s"
      frac ≥ 0.7  → "about {n+1} {unit}"  (round up)
    """
    if frac >= 0.7:
        whole += 1
        frac = 0.0

    # Singular only for "about one/a {unit}", plural otherwise
    plural = whole != 1 or frac >= 0.2
    unit_str = unit + "s" if plural else unit

    if frac < 0.2:
        word = "a" if whole == 1 else _number_word(whole)
        return f"about {word} {unit_str}"
    elif frac < 0.4:
        return f"{_number_word(whole)} and change {unit_str}"
    else:
        return f"{_number_word(whole)} and a half {unit_str}"


def format_relationship_duration(created_at: datetime) -> str:
    """
    Format account age as conversational duration with creation date.

    Examples:
      "about three days (12-06-2025)"
      "two and a half months (12-06-2025)"
      "three and change years (12-06-2025)"
    """
    delta = utc_now() - ensure_utc(created_at)
    total_days = max(delta.days, 0)
    date_str = f"({created_at.strftime('%m-%d-%Y')})"

    if total_days < 1:
        return f"a short time {date_str}"
    elif total_days < 7:
        if total_days == 1:
            return f"about a day {date_str}"
        return f"about {_number_word(total_days)} days {date_str}"
    elif total_days < 30:
        weeks = total_days / 7.0
        return f"{_fractional_duration(int(weeks), weeks - int(weeks), 'week')} {date_str}"
    elif total_days < 365:
        months = total_days / 30.44
        return f"{_fractional_duration(int(months), months - int(months), 'month')} {date_str}"
    else:
        years = total_days / 365.25
        return f"{_fractional_duration(int(years), years - int(years), 'year')} {date_str}"


# Mirrors the strptime mask accepted by local_datetime_to_utc_iso: a full local
# wall time carrying no UTC designator and no offset. Seconds are optional because
# that is how the model-facing tool schemas advertise the format ('2025-01-02T09:00').
_EXACT_LOCAL_WALL_TIME = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[tT](?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?$"
)


def normalize_exact_local_wall_time(value: Optional[str]) -> Optional[str]:
    """Return an exact local wall time as YYYY-MM-DDTHH:MM:SS, or None.

    Model-facing tools gate on this before calling local_datetime_to_utc_iso, so that
    daylight-saving strictness applies only to a wall time the model supplied to name
    one specific moment. Relative phrases, date-only and time-only inputs, and strings
    already carrying an offset or 'Z' return None and keep their existing lenient path.
    """
    if not value:
        return None
    match = _EXACT_LOCAL_WALL_TIME.match(value.strip())
    if not match:
        return None
    return (
        f"{match.group('date')}T{match.group('hour')}:{match.group('minute')}"
        f":{match.group('second') or '00'}"
    )


def local_datetime_to_utc_iso(local_datetime: str, timezone_name: str) -> str:
    """Convert one exact local wall-clock datetime to a UTC ISO instant.

    Model-facing tools use local time so the model never performs daylight-saving
    arithmetic. Ambiguous and nonexistent wall times fail instead of silently
    selecting the wrong UTC instant.
    """
    try:
        naive = datetime.strptime(local_datetime, "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Local datetime must use exact ISO format YYYY-MM-DDTHH:MM:SS "
            "without Z or a timezone offset."
        ) from error

    local_timezone = get_timezone_instance(timezone_name)
    candidates = (
        naive.replace(tzinfo=local_timezone, fold=0),
        naive.replace(tzinfo=local_timezone, fold=1),
    )
    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate.astimezone(UTC_TIMEZONE)
        .astimezone(local_timezone)
        .replace(tzinfo=None)
        == naive
    ]
    if not valid_candidates:
        raise ValueError(
            f"Local datetime {local_datetime} does not exist in timezone "
            f"{timezone_name} because of a daylight-saving transition."
        )
    if (
        len(valid_candidates) == 2
        and valid_candidates[0].utcoffset() != valid_candidates[1].utcoffset()
    ):
        raise ValueError(
            f"Local datetime {local_datetime} occurs twice in timezone "
            f"{timezone_name} because of a daylight-saving transition. "
            "Choose an unambiguous appointment time."
        )

    utc_datetime = valid_candidates[0].astimezone(UTC_TIMEZONE)
    return utc_datetime.isoformat(timespec="seconds").replace("+00:00", "Z")


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
        ValueError: If the time string cannot be parsed
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
    logger.error(f"Time string parsing failed for '{time_str}' - no valid format match found")
    raise ValueError(
        f"Invalid time format: '{time_str}'. Please use ISO format (YYYY-MM-DDTHH:MM:SS) "
        "or time-only format (HH:MM:SS)."
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
