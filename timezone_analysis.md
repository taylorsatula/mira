# Datetime and Timezone Operations in Codebase

## Core Timezone Utilities

### `/utils/timezone_utils.py`
- Central module for timezone handling
- Provides validation and normalization of timezone names
- Handles timezone conversions for naive and aware datetime objects
- Formats datetimes with various output patterns
- Parses time strings with support for natural language

## Primary Files Using Datetime/Timezone Operations

### 1. `/task_manager/automation.py`
- Extensive use of timezone conversions for automation scheduling
- Converts datetimes when storing/retrieving from database
- Handles timezone-aware timestamps for scheduling, execution tracking
- Uses `utils.timezone_utils` for consistent timezone handling

### 2. `/tools/automation_tool.py`
- Creates and manages scheduled tasks
- Uses timezone utilities for message formatting and time display
- Integrates with timezone_utils for consistent date handling

### 3. `/tools/calendar_tool.py`
- Parses and validates ISO format dates
- Handles datetime conversions for calendar events
- Uses fromisoformat() for parsing date strings
- Would benefit from timezone_utils for consistent timezone handling
- Works with both all-day and timed events

### 4. `/tools/reminder_tool.py`
- Parses natural language dates using dateutil parser
- Converts between datetime formats
- Handles relative dates (tomorrow, in 3 weeks, etc.)
- Would benefit from timezone_utils for consistent timezone handling

### 5. `/task_manager/automation_engine.py`
- Schedules and executes automations at specific times
- Calculates next execution times based on frequency
- Works with timezone-aware datetimes

### 6. `/utils/automation_controller.py`
- Manages the execution of automated tasks
- Uses datetime for scheduling and tracking

## Other Files with Datetime Usage

### 7. `/tools/weather_tool.py`
- Likely handles date/time for forecasts and current conditions
- May parse timestamps from weather API responses

### 8. `/serialization.py`
- Handles serialization/deserialization of datetime objects
- Converts between string representations and datetime objects

### 9. `/db.py`
- May handle datetime conversion for database operations
- Stores timezone-aware timestamps

### 10. `/conversation.py`
- Uses datetime for tracking conversation history
- May need timezone awareness for accurate timestamps

## Recommendations for timezone_utils Integration

1. **Calendar Tool**: Replace direct datetime usage with timezone_utils functions for consistent timezone handling
2. **Reminder Tool**: Replace custom date parsing with timezone_utils.parse_time_string
3. **Weather Tool**: Use timezone_utils for displaying forecasts in user's local timezone
4. **Serialization**: Ensure datetime serialization/deserialization preserves timezone information
5. **Database Operations**: Standardize on UTC for storage with timezone_utils for display conversions
6. **Tests**: Update tests to use timezone_utils for date comparisons and generation

## Common Patterns to Refactor

1. **Replace direct fromisoformat()** with timezone_utils parsing functions
2. **Replace direct strftime()** calls with timezone_utils formatting functions
3. **Add timezone awareness** to naive datetime objects
4. **Standardize timezone display** across the application
5. **Use consistent timezone handling** for user-facing output