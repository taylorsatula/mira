# UTC-Everywhere Datetime Handling Guide

This guide outlines best practices for handling date and time in the application, focusing on a "UTC-everywhere" approach. This implementation has been completed across the codebase to ensure consistent timezone handling.

## Core Principles

1. **Store in UTC**: All datetimes are stored in UTC in the database
2. **Display in Local Time**: Conversion to local time happens only at display time
3. **Always Timezone-Aware**: All datetime objects should have timezone information
4. **Consistent Formatting**: Use standard formats for string representations

## Database Models

### Creating DateTime Columns

Use the utilities from `utils.db_datetime_utils` to create standardized datetime columns:

```python
from utils.db_datetime_utils import utc_datetime_column, utc_created_at_column, utc_updated_at_column

class MyModel(Base):
    __tablename__ = 'my_models'
    
    id = Column(String, primary_key=True)
    
    # Standard timestamp fields (created_at/updated_at)
    created_at = utc_created_at_column()  # Defaults to current UTC time on creation
    updated_at = utc_updated_at_column()  # Updates on every save
    
    # Custom datetime fields
    scheduled_at = utc_datetime_column(nullable=False)  # Required field, no default
    completed_at = utc_datetime_column()  # Optional field (nullable=True)
    
    # With a specific default value
    expires_at = utc_datetime_column(default=lambda: utc_now() + timedelta(days=30))
```

### Using the UTCDatetimeMixin

For models with standard created_at/updated_at fields, use the UTCDatetimeMixin:

```python
from utils.db_datetime_utils import UTCDatetimeMixin
from sqlalchemy import Column, String

class MyModel(UTCDatetimeMixin, Base):
    __tablename__ = 'my_models'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    
    # created_at and updated_at are automatically added by the mixin
```

## Working with Datetime Objects

### Getting Current Time

Always use `utc_now()` instead of `datetime.now()` or `datetime.utcnow()`:

```python
from utils.timezone_utils import utc_now

# Get current time in UTC with timezone info
current_time = utc_now()
```

### Ensuring UTC

When working with datetime objects that might come from external sources, ensure they are UTC:

```python
from utils.timezone_utils import ensure_utc

# Convert a datetime to UTC (if needed)
some_datetime = ensure_utc(some_datetime)
```

### Converting Between Timezones

Use the timezone utilities for conversion:

```python
from utils.timezone_utils import convert_to_timezone, convert_to_utc, convert_from_utc

# Convert from any timezone to UTC
utc_time = convert_to_utc(local_time, from_tz="America/New_York")

# Convert from UTC to a specific timezone
local_time = convert_from_utc(utc_time, to_tz="America/New_York")

# General timezone conversion
other_tz_time = convert_to_timezone(original_time, target_tz="Europe/London")
```

## Serialization & Deserialization

### General JSON Serialization

The `serialization.py` module has been updated to ensure consistent UTC handling:

```python
from serialization import to_json, from_json

# Object with a datetime field
obj = {
    "id": "123",
    "name": "Sample",
    "created_at": utc_now()
}

# Serialize to JSON - ensures datetimes are UTC-formatted ISO strings
json_text = to_json(obj)

# Deserialize - optionally parse datetime strings back to UTC-aware datetime objects
data = from_json(json_text, parse_dates=True)
# data["created_at"] is now a UTC-aware datetime object
```

### Model Serialization

Use the provided utilities for consistent serialization:

```python
from utils.db_datetime_utils import serialize_model_datetime

# Example model dictionary with datetime fields
model_dict = {
    "id": "123",
    "name": "Sample",
    "created_at": datetime(2023, 1, 1, tzinfo=UTC)
}

# Serialize datetime fields for JSON responses (in user's timezone)
serialized = serialize_model_datetime(
    model_dict,
    datetime_fields=["created_at", "updated_at"],
    target_tz="America/New_York"
)
```

### Deserialization from API/JSON

When receiving datetime strings from APIs or JSON:

```python
from utils.db_datetime_utils import deserialize_datetime_strings

# Example data from API with string datetime
data = {
    "name": "Event",
    "scheduled_at": "2023-05-15T10:30:00+00:00"
}

# Convert string datetimes to UTC datetime objects
processed_data = deserialize_datetime_strings(
    data,
    datetime_fields=["scheduled_at", "expires_at"]
)

# Now processed_data["scheduled_at"] is a UTC datetime object
```

### Parsing Weather API Data

The `weather_tool.py` implements UTC handling for external API data:

```python
# Example: Processing timestamps from a third-party API
if 'T' in time_str and not ('+' in time_str or 'Z' in time_str):
    # If no timezone info, assume UTC
    time_str = f"{time_str}+00:00"

# Parse using our timezone utilities consistently
dt = parse_utc_time_string(time_str)
processed_times.append(dt.isoformat())
```

## Best Practices

### 1. Database Models

- Use `UTCDatetimeMixin` for all models with timestamp fields
- Always declare datetime columns using the utility functions
- Never store timezone info in the database (it's redundant with our UTC standard)

### 2. API Responses

- Convert UTC datetimes to the user's timezone before serializing
- Include timezone information in datetime string representations
- Use ISO 8601 format for datetime strings in APIs

### 3. User Input

- Always convert user input to UTC immediately
- Validate timezone names using `validate_timezone()`
- Parse datetime strings with `parse_utc_time_string()`

### 4. Date Comparisons

- Ensure all datetimes are UTC-aware before comparison
- Use `ensure_utc()` to standardize datetimes before comparisons
- For date-only comparisons, compare the `.date()` attributes after standardizing

### 5. Date Arithmetic

- Perform date arithmetic (adding/subtracting time) on UTC datetimes
- Be aware of DST transitions when using large time deltas
- Use Python's `timedelta` for simple arithmetic and `dateutil.relativedelta` for more complex operations

## Common Pitfalls

1. **Naive Datetimes**: Always ensure datetime objects have timezone information
2. **Local Time Storage**: Never store local time in the database - convert to UTC first
3. **Implicit Conversion**: Be explicit about timezone conversions
4. **String Parsing**: Always handle timezone information when parsing datetime strings
5. **DST Transitions**: Be careful with date arithmetic around DST transitions

## Testing

For testing time-sensitive code:

```python
from freezegun import freeze_time
from utils.timezone_utils import utc_now

# Freeze time to a specific UTC datetime
with freeze_time("2023-01-01 12:00:00", tz_offset=0):
    # Inside this block, utc_now() will return the frozen time
    assert utc_now().isoformat() == "2023-01-01T12:00:00+00:00"
```

## User Interface Considerations

### Conversation Time Information

The `conversation.py` module has been updated to display and track time consistently:

```python
# Get current time information using our utility for consistent UTC handling
now_utc = utc_now()
# Convert to user's configured timezone for display
user_tz = get_default_timezone()
now_local = convert_from_utc(now_utc, user_tz)
time_info = f"Current datetime: {format_datetime(now_local, 'date_time', include_timezone=True)} (UTC: {format_datetime(now_utc, 'date_time')})"
```

## Migration Guidelines

When migrating existing data:

1. Identify all datetime fields that need migration
2. Determine the original timezone of stored values
3. Convert all values to UTC with appropriate timezone info
4. Update database schemas to enforce UTC storage
5. Update application code to use the UTC-everywhere utilities

## Implementation Status

The UTC-everywhere approach has been implemented in:

1. **Core Utilities**: `/utils/timezone_utils.py` - Complete implementation
2. **Weather Tool**: `/tools/weather_tool.py` - Updated to use timezone utilities consistently
3. **Reminder Tool**: `/tools/reminder_tool.py` - Already using UTC-awareness properly
4. **Conversation Module**: `/conversation.py` - Updated with proper UTC handling
5. **Serialization Module**: `/serialization.py` - Updated to preserve UTC timezone information

## Further Reading

- [UTC is Enough for Everyone, Right?](https://zachholman.com/talk/utc-is-enough-for-everyone-right)
- [Storing UTC is not a Silver Bullet](https://codeblog.jonskeet.uk/2019/03/27/storing-utc-is-not-a-silver-bullet/)
- [Python datetime documentation](https://docs.python.org/3/library/datetime.html)
- [SQLAlchemy DateTime column documentation](https://docs.sqlalchemy.org/en/20/core/type_basics.html#sqlalchemy.types.DateTime)

## Conclusion

By implementing the UTC-everywhere approach consistently across our codebase, we ensure:

1. Reliable datetime operations across different timezones
2. Proper handling of Daylight Saving Time transitions
3. Consistent user experience regardless of location
4. Easier debugging and maintenance of datetime-related code
5. Better interoperability with external services and APIs