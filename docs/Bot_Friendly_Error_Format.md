# Bot-Friendly Error Format Guide

## Overview

This document describes how to implement structured JSON error responses that enable MIRA to programmatically understand errors, take appropriate recovery actions, and provide better user experiences during tool execution failures.

> **IMPORTANT**: This detailed error format should ONLY be implemented for components where MIRA needs to programmatically understand and respond to errors. Specifically:
>
> **Components that NEED structured error formats:**
> - Direct tool interfaces that MIRA calls
> - API endpoints that return errors to MIRA
> - Functions that process user input and are called by MIRA
>
> **Components that DON'T NEED structured error formats:**
> - Internal system components invisible to MIRA
> - Background processes
> - Utility functions not directly called by MIRA
> - System initialization and infrastructure code

## JSON Error Schema

All tool and system errors returned to MIRA should follow this standardized JSON structure:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "category": "CATEGORY",
    "source": "COMPONENT_NAME",
    "message": "Human readable description of what went wrong",
    "context": {
      "operation": "OPERATION_NAME",
      "parameters": {
        "param1": "value1",
        "param2": "value2"
      }
    },
    "recovery": {
      "is_retryable": true|false,
      "retry_strategy": {
        "suggested_delay": 1000,
        "max_retries": 3,
        "parameter_adjustments": {
          "param1": "new_value1"
        }
      },
      "alternatives": [
        {
          "description": "Try using X instead",
          "example": "Example of alternative approach"
        }
      ],
      "required_actions": [
        "Verify API key is valid",
        "Check if resource exists"
      ]
    },
    "debug": {
      "error_id": "UNIQUE_ID",
      "timestamp": "ISO_TIMESTAMP"
    }
  }
}
```

## Field Descriptions

### Core Error Fields

- **code**: A unique identifier for the error (string, required)
  - Format: `[DOMAIN]_[TYPE]_[SPECIFIC]` (e.g., `WEATHER_API_TIMEOUT`)
  - Should match an error code from our ErrorCode enum

- **category**: The error's functional domain (string, required)
  - Maps to our numeric ranges: "system" (1xx), "network" (2xx), etc.

- **source**: The component that generated the error (string, required)
  - Usually the tool name or system component

- **message**: Human-readable error description (string, required)
  - Should be clear, concise, and actionable
  - Write for both MIRA and end-users (may be shown to users)

### Context Fields

- **operation**: What operation was being attempted (string, required)
  - E.g., "get_weather", "create_reminder", "send_email"

- **parameters**: Non-sensitive parameters that led to the error (object, optional)
  - IMPORTANT: Sanitize sensitive values (passwords, tokens, etc.)

### Recovery Fields

- **is_retryable**: Whether the error can be retried (boolean, required)
  - Set to true for transient errors (network issues, rate limits)
  - Set to false for permanent errors (invalid input, missing resources)

- **retry_strategy**: How to retry if retryable (object, optional)
  - **suggested_delay**: Milliseconds to wait before retry (number, optional)
  - **max_retries**: Maximum number of retry attempts (number, optional)
  - **parameter_adjustments**: Suggested changes to parameters (object, optional)

- **alternatives**: Alternative approaches to try (array, optional)
  - **description**: Description of the alternative (string, required)
  - **example**: Example of the alternative (string, optional)

- **required_actions**: Actions needed before retry (array, optional)
  - List of strings describing required actions

### Debug Fields

- **error_id**: Unique identifier for this error instance (string, required)
  - Use UUID format for global uniqueness

- **timestamp**: When the error occurred (ISO 8601 string, required)
  - Always use UTC timezone

## Implementation

### 1. Extending the ToolError Class

Add a `to_bot_format` method to our ToolError class:

```python
def to_bot_format(self) -> dict:
    """Convert ToolError to bot-friendly format"""
    # Map error code to category
    category = self._map_code_to_category()
    
    # Determine retryability
    is_retryable = self._is_retryable()
    
    # Build the error response
    error_response = {
        "error": {
            "code": self.code.name,
            "category": category,
            "source": self.context.get("tool", "system"),
            "message": str(self),
            "context": {
                "operation": self.context.get("operation", "unknown"),
                "parameters": self._sanitize_params(self.context.get("params", {}))
            },
            "recovery": {
                "is_retryable": is_retryable,
                "required_actions": []
            },
            "debug": {
                "error_id": self.id,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
        }
    }
    
    # Add retry strategy for retryable errors
    if is_retryable:
        error_response["error"]["recovery"]["retry_strategy"] = self._get_retry_strategy()
    
    # Add alternatives if available
    alternatives = self._get_alternatives()
    if alternatives:
        error_response["error"]["recovery"]["alternatives"] = alternatives
    
    return error_response
```

### 2. Error Registry for Recovery Information

Create an error registry to store recovery information for each error code:

```python
ERROR_REGISTRY = {
    ErrorCode.NETWORK_ERROR: {
        "is_retryable": True,
        "retry_strategy": {
            "suggested_delay": 2000,
            "max_retries": 3
        },
        "required_actions": ["Check network connection"]
    },
    ErrorCode.RATE_LIMIT_ERROR: {
        "is_retryable": True,
        "retry_strategy": {
            "suggested_delay": 5000,
            "max_retries": 2
        },
        "required_actions": ["Wait before retrying"]
    },
    ErrorCode.INVALID_INPUT_ERROR: {
        "is_retryable": True,
        "required_actions": ["Correct the input parameters"]
    },
    # Add entries for all error codes
}
```

### 3. Tool Implementation Example

```python
def weather_tool(location, units="metric"):
    """Get weather for a location"""
    try:
        if not location:
            # Define recovery information for this specific error
            recovery_info = {
                "is_retryable": True,
                "alternatives": [
                    {
                        "description": "Try using a city name instead",
                        "example": "Use 'New York' instead of an empty string"
                    }
                ],
                "required_actions": ["Provide a non-empty location"]
            }
            
            raise standardize_tool_error(
                ValueError("Location cannot be empty"),
                "weather_tool",
                "get_weather",
                {"location": location, "units": units},
                recovery_info,
                ErrorCode.INVALID_INPUT_ERROR
            )
        
        # Regular tool implementation...
        
    except Exception as e:
        if not isinstance(e, ToolError):
            raise standardize_tool_error(
                e, "weather_tool", "get_weather", 
                {"location": location, "units": units}
            )
        raise
```

### 4. Enhanced Error Context Manager

Update the error_context manager to support bot-friendly errors:

```python
@contextmanager
def error_context(tool_name, operation, params=None):
    """Context manager for standardized error handling"""
    try:
        yield
    except Exception as e:
        if isinstance(e, ToolError):
            # Add context to existing ToolError
            e.context.update({
                "tool": tool_name,
                "operation": operation,
                "params": params
            })
            raise
        else:
            # Create new ToolError
            error_code = _determine_error_code(e)
            raise ToolError(
                f"{tool_name} {operation} failed: {e}",
                error_code,
                {
                    "tool": tool_name,
                    "operation": operation,
                    "params": params,
                    "original_error": e
                }
            )
```

### 5. Middleware for Bot Responses

Add middleware in the conversation handler to format errors for MIRA:

```python
def process_tool_result(result, for_bot=True):
    """Process tool result for response"""
    if isinstance(result, Exception):
        if isinstance(result, ToolError):
            # Format for bot consumption
            if for_bot:
                return json.dumps(result.to_bot_format())
            else:
                return str(result)
        else:
            # Wrap in ToolError
            error = ToolError(
                f"Tool execution failed: {result}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"original_error": result}
            )
            
            if for_bot:
                return json.dumps(error.to_bot_format())
            else:
                return str(error)
    
    # Return normal result
    return result
```

## Best Practices

### When to Apply These Practices

Apply these practices ONLY to components that MIRA directly interacts with:

- **DO** implement for tool methods that MIRA calls directly
- **DO** implement for API endpoints that MIRA sends requests to
- **DO** implement for conversation/message handling code
- **DON'T** implement for internal utility functions
- **DON'T** implement for background processes or initialization code
- **DON'T** implement for infrastructure components

### Creating Effective Bot-Friendly Errors

1. **Be Specific and Actionable**
   - Error messages should clearly explain what went wrong
   - Include specific guidance on how to fix the issue

2. **Provide Recovery Paths**
   - Always include at least one recovery option
   - For non-retryable errors, provide alternatives

3. **Use Consistent Error Codes**
   - Follow the domain-based categorization system
   - Use specific codes for distinct error conditions

4. **Include Relevant Context**
   - Add all non-sensitive parameters that led to the error
   - Include information about the operation state

5. **Keep Messages User-Friendly**
   - Write error messages that MIRA can relay to end-users
   - Avoid technical jargon when possible

### Security Considerations

1. **Sanitize Sensitive Information**
   - Never include passwords, tokens, or keys in error messages
   - Redact sensitive values in parameters

2. **Limit Technical Details**
   - Don't expose internal system details that could aid attacks
   - Keep stack traces and implementation details private

3. **Validate Error Messages**
   - Ensure error messages don't contain unintended information
   - Review error messages for potential information leakage

## Testing Bot-Friendly Errors

Create comprehensive tests for error handling:

1. **Unit Tests**
   - Verify each error code produces the expected JSON structure
   - Test sanitization of sensitive parameters
   - Ensure retryability is correctly determined

2. **Integration Tests**
   - Test end-to-end error flows from tools to MIRA
   - Verify error handling across system boundaries

3. **Manual Testing**
   - Review error messages for clarity and actionability
   - Verify MIRA can correctly interpret and act on errors

## Migration Strategy

1. **Update Error Framework**
   - Extend ToolError with bot-friendly formatting
   - Create the error registry with recovery information

2. **Implement in High-Impact Tools First**
   - Start with frequently used tools
   - Focus on tools with complex error conditions

3. **Add Testing and Validation**
   - Create tests for bot-friendly error formats
   - Validate error handling with MIRA

4. **Expand to All Tools**
   - Progressively update all tools
   - Standardize error patterns across the codebase

5. **Document and Monitor**
   - Document all error codes and recovery strategies
   - Monitor error patterns and MIRA's handling of errors

## Example Error Responses

Below are examples of bot-friendly error responses for tools that MIRA directly interacts with. Again, this level of detailed formatting is ONLY needed for components where MIRA directly receives and must programmatically handle the error.

### Network Error (from a Tool MIRA Called)
```json
{
  "error": {
    "code": "NETWORK_CONNECTION_ERROR",
    "category": "network",
    "source": "weather_tool",
    "message": "Failed to connect to weather service",
    "context": {
      "operation": "get_weather",
      "parameters": {
        "location": "New York",
        "units": "metric"
      }
    },
    "recovery": {
      "is_retryable": true,
      "retry_strategy": {
        "suggested_delay": 2000,
        "max_retries": 3
      },
      "required_actions": [
        "Check internet connection",
        "Verify the weather service is operational"
      ]
    },
    "debug": {
      "error_id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2023-04-25T12:34:56.789Z"
    }
  }
}
```

### Invalid Input Error (from a Tool MIRA Called)
```json
{
  "error": {
    "code": "INVALID_INPUT_ERROR",
    "category": "tool",
    "source": "reminder_tool",
    "message": "Cannot create reminder: date is in the past",
    "context": {
      "operation": "create_reminder",
      "parameters": {
        "title": "Doctor appointment",
        "date": "2023-01-01T10:00:00Z"
      }
    },
    "recovery": {
      "is_retryable": true,
      "parameter_adjustments": {
        "date": "future date required"
      },
      "alternatives": [
        {
          "description": "Create a note instead of a reminder for past events",
          "example": "Use notes_tool.create_note() for records of past events"
        }
      ],
      "required_actions": [
        "Provide a future date and time"
      ]
    },
    "debug": {
      "error_id": "550e8400-e29b-41d4-a716-446655440001",
      "timestamp": "2023-04-25T14:22:33.456Z"
    }
  }
}
```

By implementing this structured approach to error handling, we enable MIRA to:
1. Understand precisely what went wrong
2. Take appropriate recovery actions
3. Provide meaningful guidance to users
4. Learn from error patterns over time

This creates a more resilient system where errors become actionable information rather than opaque failures.

## Conclusion: Targeted Implementation

To summarize:

1. **DO implement** this format for:
   - Tool APIs directly called by MIRA
   - Functions that MIRA must retry or adjust parameters for
   - Endpoints that return results directly to MIRA

2. **DON'T implement** this format for:
   - Internal utility functions
   - Helper methods
   - Background processes
   - Infrastructure code
   - System components that MIRA never directly interacts with

The goal is to provide rich, structured error information where it's needed, while keeping the rest of the codebase simple and maintainable. Applying this approach selectively ensures we get the benefits of structured error handling without unnecessary complexity.