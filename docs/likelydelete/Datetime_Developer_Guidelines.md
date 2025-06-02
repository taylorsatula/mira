# Datetime Handling Guidelines for Developers

This document provides concise guidelines for handling dates and times in the application, following our UTC-everywhere approach.

## Quick Reference

| Task | DO | DON'T |
|------|----|----|
| Getting current time | `from utils.timezone_utils import utc_now; now = utc_now()` | ❌ `datetime.now()` or `datetime.utcnow()` |
| Creating datetime columns | `from utils.db_datetime_utils import utc_datetime_column; created_at = utc_datetime_column()` | ❌ `Column(DateTime)` |
| Storing datetimes | Store as UTC | ❌ Store as local time |
| Displaying datetimes | Convert from UTC to local time at display time | ❌ Store pre-converted |
| Comparing datetimes | Ensure both are in UTC first | ❌ Compare naive or mixed timezone objects |
| Date arithmetic | Use UTC datetimes | ❌ Do naive arithmetic that crosses DST boundaries |

## Core Principles

1. **UTC for Storage**: Always store datetimes in UTC
2. **Local for Display**: Convert to local time only when displaying to users
3. **Always Timezone-Aware**: Never use naive datetimes (without timezone info)
4. **Use the Utilities**: Always use the provided datetime utilities

## Creating New Models

When creating new database models with datetime fields:

```python
from utils.db_datetime_utils import UTCDatetimeMixin, utc_datetime_column

class MyModel(UTCDatetimeMixin, Base):
    __tablename__ = 'my_models'
    
    id = Column(String, primary_key=True)
    
    # Custom datetime fields
    event_time = utc_datetime_column(nullable=False)
    
    # created_at and updated_at are automatically added by the mixin
```

## Handling User Input

When accepting datetime input from users:

```python
from utils.timezone_utils import parse_time_string, convert_to_utc

# Parse with user's timezone
user_tz = "America/New_York"  # Get from user profile
local_dt = parse_time_string(input_str, user_tz)

# Convert to UTC for storage
utc_dt = convert_to_utc(local_dt)
```

## Displaying Datetimes to Users

When showing datetimes to users:

```python
from utils.timezone_utils import convert_from_utc, format_datetime

# Get user's timezone
user_tz = "America/New_York"  # Get from user profile

# Convert from UTC to user's timezone
local_dt = convert_from_utc(utc_dt, user_tz)

# Format for display
formatted = format_datetime(utc_dt, "date_time", user_tz)
```

## API Responses

For API responses with datetime fields:

```python
from utils.db_datetime_utils import serialize_model_datetime

# Convert model to dict
model_dict = model.to_dict()

# Serialize datetime fields for API response
serialized = serialize_model_datetime(
    model_dict,
    datetime_fields=["created_at", "updated_at", "event_time"],
    target_tz=user_timezone
)

# Return as JSON
return json.dumps(serialized)
```

## Database Queries

When filtering by dates in database queries:

```python
from utils.timezone_utils import ensure_utc, utc_now

# Get current time in UTC
now = utc_now()

# Filter for future items
future_items = session.query(Event).filter(
    Event.event_time > now
).all()

# Filter for items in a specific date range (user input converted to UTC)
start_date = ensure_utc(start_date_input)
end_date = ensure_utc(end_date_input)
items = session.query(Event).filter(
    Event.event_time >= start_date,
    Event.event_time <= end_date
).all()
```

## Common Patterns and Idioms

### Creating New Events

```python
from utils.timezone_utils import utc_now, convert_to_utc

# For events happening now
event = Event(
    event_time=utc_now(),
    # other fields...
)

# For events at a specific user local time
local_time = parse_time_string("2023-05-01T15:00:00", user_tz)
event = Event(
    event_time=convert_to_utc(local_time),
    # other fields...
)
```

### Date Arithmetic

```python
from utils.timezone_utils import utc_now
from datetime import timedelta

# Get tomorrow at the same time
tomorrow = utc_now() + timedelta(days=1)

# Calculate duration between two UTC times
duration = end_time - start_time  # both in UTC
hours = duration.total_seconds() / 3600
```

### Working with Date-Only Information

```python
from utils.timezone_utils import utc_now, convert_from_utc

# Get today's date in user's timezone
now = utc_now()
user_now = convert_from_utc(now, user_tz)
today = user_now.date()  # Extract date component

# Filter for items on a specific date
start_of_day = datetime.combine(target_date, time.min, tzinfo=user_tz_instance)
end_of_day = datetime.combine(target_date, time.max, tzinfo=user_tz_instance)
start_utc = convert_to_utc(start_of_day)
end_utc = convert_to_utc(end_of_day)

items = session.query(Event).filter(
    Event.event_time >= start_utc,
    Event.event_time <= end_utc
).all()
```

## Testing Datetime Logic

Use freezegun to test time-dependent code:

```python
from freezegun import freeze_time
from utils.timezone_utils import utc_now

# Freeze time to a specific UTC datetime
with freeze_time("2023-01-01 12:00:00", tz_offset=0):
    # Inside this block, utc_now() will return the frozen time
    assert utc_now().isoformat() == "2023-01-01T12:00:00+00:00"
    
    # Test your time-dependent code
    result = my_function_that_uses_current_time()
    assert result.day == 1
```

## Troubleshooting

### Timezone Issues

1. **Wrong Times Displayed**: Check if conversion is happening at both storage and display (double conversion)
2. **DST Transition Issues**: Ensure you're using timezone-aware objects and utilities
3. **Database Timestamps**: Verify that datetimes are stored as UTC in the database

### Common Mistakes

1. **Using `datetime.now()` without timezone**: Always use `utc_now()`
2. **Comparing naive and aware datetimes**: Use `ensure_utc()` on both sides
3. **Manual string parsing**: Use `parse_time_string()` instead of manual parsing
4. **Direct `strftime()`**: Use `format_datetime()` for consistent formatting

## Cheat Sheet

```python
# Import utilities
from utils.timezone_utils import (
    utc_now, ensure_utc, convert_to_utc, convert_from_utc,
    format_datetime, parse_time_string
)
from utils.db_datetime_utils import (
    UTCDatetimeMixin, utc_datetime_column, 
    serialize_model_datetime, deserialize_datetime_strings
)

# Current time in UTC
now = utc_now()

# Convert any datetime to UTC
utc_dt = ensure_utc(some_datetime)

# Parse time string to datetime
dt = parse_time_string("2023-01-01T12:00:00", "America/New_York")

# Format datetime for display
formatted = format_datetime(utc_dt, "date_time", "America/New_York")

# Convert between timezones
local_dt = convert_from_utc(utc_dt, "America/New_York")
utc_dt = convert_to_utc(local_dt)

# Serialize for API
serialized = serialize_model_datetime(model_dict, datetime_fields, user_tz)
```