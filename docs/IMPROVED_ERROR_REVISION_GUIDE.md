# Improved Error Handling Guide

This document outlines a simplified, practical approach to error handling in the MIRA system. The goal is to make it easier for MIRA to receive useful information to help recover from recoverable issues and fail fast when hitting a true wall.

## Core Principles

1. **Simplicity over complexity**
   - Leverage Python's existing exception system
   - Avoid unnecessary abstraction layers
   - Focus on clear, descriptive errors

2. **Tool-specific error codes**
   - Each tool defines its own error codes 
   - Error codes communicate recoverability in their names
   - MIRA can parse meaning directly from error codes

3. **Single error boundary**
   - Each component has one error boundary
   - Avoid error wrapping chains
   - Preserve original error context

4. **Actionable error messages**
   - Tell MIRA how to fix the problem, not just what's wrong
   - Include specific details about invalid inputs
   - Provide clear guidance for recovery

## Implementation

### 1. Tool-Specific Error Codes

Each tool should define its own error codes as string constants:

```python
# In weather_tool.py
LOCATION_NOT_FOUND = "WEATHER_TOOL.LOCATION_NOT_FOUND"
LOCATION_TOO_VAGUE = "WEATHER_TOOL.LOCATION_TOO_VAGUE"
FORECAST_TOO_FAR_FUTURE = "WEATHER_TOOL.FORECAST_TOO_FAR_FUTURE"
API_KEY_INVALID = "WEATHER_TOOL.API_KEY_INVALID"

# In calendar_tool.py
EVENT_CONFLICT = "CALENDAR_TOOL.EVENT_CONFLICT"
EVENT_IN_PAST = "CALENDAR_TOOL.EVENT_IN_PAST"
CALENDAR_NOT_FOUND = "CALENDAR_TOOL.CALENDAR_NOT_FOUND"
ATTENDEE_INVALID = "CALENDAR_TOOL.ATTENDEE_INVALID"
```

Error code naming should follow these conventions:
- Use all uppercase with underscores
- Include tool name prefix to avoid conflicts
- Choose descriptive names that indicate what went wrong
- Use names that suggest whether an error is retryable

Examples of good error code names:
- `NETWORK_TOOL.TEMPORARY_FAILURE` (clearly retryable)
- `CALENDAR_TOOL.EVENT_IN_PAST` (clearly not retryable)
- `EMAIL_TOOL.RECIPIENT_INVALID` (fix input and retry)
- `WEATHER_TOOL.LOCATION_TOO_VAGUE` (provide more specific input)

### 2. Simple ToolError Class

The ToolError class should remain similar to the current implementation:

```python
class ToolError(AgentError):
    """Exception raised for tool-related errors."""

    def __init__(
        self,
        message: str,
        code: str,  # Changed from ErrorCode enum to string
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.details = details or {}  # Match the existing 'details' naming
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
```

Note that we're changing from using the `ErrorCode` enum to string-based error codes, but otherwise keeping the interface close to the existing implementation. No need for complex retryability flags or recovery strategies - the error code name will communicate this information.

### 3. Error Context Manager

Adapt the existing error_context manager to work with string-based error codes:

```python
@contextmanager
def error_context(
    component_name: str,
    operation: Optional[str] = None,
    error_class: Type[AgentError] = AgentError,
    working_memory=None,
    logger: Optional[logging.Logger] = None
):
    """
    Context manager for standardized error handling across the system.
    
    Provides consistent error handling, logging, and error wrapping
    for any component operation. Use with a 'with' statement to wrap code
    that may raise exceptions.
    
    Args:
        component_name: Name of the component (for error messages)
        operation: Description of the operation (for error messages)
        error_class: The AgentError subclass to use for wrapping
        working_memory: Optional working memory for error analysis
        logger: Logger to use (if None, creates a new one)
    """
    # Set up logger if not provided
    if logger is None:
        logger = logging.getLogger(f"error.{component_name}")
        
    try:
        yield
    except Exception as e:
        # Generate a UUID for tracking
        error_id = str(uuid.uuid4())
        
        # If it's already a ToolError, log and add to working memory if available
        if isinstance(e, ToolError):
            logger.error(f"[{error_id}] {component_name} error: {e}")
            
            # Add to working memory for analysis if available
            if working_memory:
                _add_error_to_working_memory(working_memory, e, error_id)
                
            raise
            
        # Generate error message
        error_msg = f"Error in {component_name}"
        if operation:
            error_msg += f" during {operation}"
            
        # Log and wrap other exceptions
        logger.error(f"[{error_id}] {error_msg}: {str(e)}")
        
        # Create a string-based error code
        error_code = f"{component_name.upper()}.UNEXPECTED_ERROR"
        
        # Create the error
        tool_error = error_class(
            f"{error_msg}: {str(e)}",
            error_code,
            {"original_error": str(e), "error_id": error_id}
        )
        
        # Add to working memory if available
        if working_memory and isinstance(tool_error, ToolError):
            _add_error_to_working_memory(working_memory, tool_error, error_id)
        
        raise tool_error
```

With a helper function to add errors to working memory:

```python
def _add_error_to_working_memory(working_memory, error, error_id):
    """Add an error to working memory for analysis."""
    error_info = {
        "error_id": error_id,
        "code": error.code,
        "message": error.message,
        "details": error.details,  # Use details instead of context
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    
    # Add to working memory
    working_memory.add(
        content="# Error Analysis Request\nAn error just occurred. Analyze this error based on the recent conversation and explain in ONE SENTENCE what the user was trying to do when it occurred. Include your analysis inside a <error_analysis error_id=\"" + error_id + "\">YOUR ANALYSIS HERE</error_analysis> tag within your thought process.\n\nError details:\n" + json.dumps(error_info, indent=2),
        category="error_for_analysis"
    )
```

### 4. Raising Tool-Specific Errors

When raising errors in tools, be specific and actionable:

```python
def get_weather(location, date):
    # Validate inputs
    if not location or location.strip() == "":
        raise ToolError(
            "Location cannot be empty. Please provide a valid location name or coordinates.",
            LOCATION_EMPTY,
            {"provided_location": location}
        )
    
    if location.lower() in ["somewhere", "there", "that place"]:
        raise ToolError(
            f"Location '{location}' is too vague. Please provide a city name, address, or coordinates.",
            LOCATION_TOO_VAGUE,
            {"provided_location": location}
        )
    
    # More implementation...
```

### 5. Handling Errors (for MIRA)

MIRA can handle errors by parsing the error code:

```python
try:
    result = weather_tool.get_weather(location, date)
except ToolError as e:
    error_code = e.code
    
    # Parse the error code to determine next steps
    if "TOO_VAGUE" in error_code:
        # Ask user for more specific input
        clarified_input = ask_for_clarification(e.message)
        # Retry with better input
        
    elif "NOT_FOUND" in error_code:
        # Suggest alternatives
        alternatives = suggest_alternatives(e.context.get("provided_location"))
        # Offer alternatives to user
        
    elif "TEMPORARY" in error_code or "TIMEOUT" in error_code:
        # Wait and retry
        wait_and_retry_later()
    
    else:
        # Handle unrecoverable errors
        apologize_to_user(e.message)
```

## Avoiding Error Duplication

To prevent error chains and duplication:

1. **Use error_context only at component boundaries**
   - Tool methods that are called directly by MIRA
   - API endpoints
   - Service entry points

2. **Let internal errors propagate normally**
   - Don't wrap errors in internal functions
   - Let Python's exception handling work as designed

3. **Don't catch and re-raise with the same information**
   - Only catch exceptions if you're adding meaningful context
   - If you're not changing the error, just let it propagate

## Example: Good Error Handling

```python
# In weather_tool.py
LOCATION_NOT_FOUND = "WEATHER_TOOL.LOCATION_NOT_FOUND"
API_REQUEST_FAILED = "WEATHER_TOOL.API_REQUEST_FAILED"

class WeatherTool(Tool):
    def run(self, location, date=None):
        """Get weather for a location."""
        with error_context("WeatherTool"):
            # Internal function raises ValueError normally
            validated_location = self._validate_location(location)
            
            try:
                # External API might fail
                api_response = self._call_weather_api(validated_location, date)
                return self._format_weather_response(api_response)
            except requests.RequestException as e:
                # Specific handling for network errors
                if "timeout" in str(e).lower():
                    raise ToolError(
                        f"Weather API request timed out. Please try again later.",
                        API_REQUEST_FAILED,
                        {"location": location, "error": str(e)}
                    )
                raise ToolError(
                    f"Failed to retrieve weather data: {e}",
                    API_REQUEST_FAILED,
                    {"location": location, "error": str(e)}
                )
    
    def _validate_location(self, location):
        """Internal function that raises normal exceptions."""
        if not location or not location.strip():
            raise ValueError("Location cannot be empty")
        
        # More validation...
        return location
```

## Error Logging

For logging, follow these simple guidelines:

1. **Log at the error boundary**
   - Log the full error details when it's first caught
   - Include stack traces for unexpected errors

2. **Log the error code and message**
   - Make error codes easily searchable in logs
   - Include enough context to understand what happened

3. **Don't log the same error multiple times**
   - Once an error is logged, pass it up without additional logging
   - Avoid log spam from the same error

4. **Use separate logfiles for system and tool errors**
   - Create a system error log for core system errors
   - Create a separate tool error log for tool-specific errors
   
### Separate Logfile Configuration

```python
# Configure separate loggers for system and tool errors
import logging

# System error logger
system_logger = logging.getLogger('system_errors')
system_handler = logging.FileHandler('persistent/logs/system_errors.log')
system_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
system_logger.addHandler(system_handler)

# Tool error logger
tool_logger = logging.getLogger('tool_errors')
tool_handler = logging.FileHandler('persistent/logs/tool_errors.log')
tool_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
tool_logger.addHandler(tool_handler)

# Usage in error_context
@contextmanager
def error_context(component_name, logger=None):
    """Context manager for standardized error handling at component boundaries."""
    try:
        yield
    except Exception as e:
        # If it's already a ToolError, log to tool error log
        if isinstance(e, ToolError):
            tool_logger.error(f"[{e.code}] {e.message}")
            raise
            
        # Otherwise, log to system error log
        system_logger.error(f"Error in {component_name}: {e}")
        
        # Create a generic error code for unexpected errors
        error_code = f"{component_name.upper()}.UNEXPECTED_ERROR"
        
        raise ToolError(
            f"Unexpected error in {component_name}: {str(e)}",
            error_code,
            {"original_error": str(e)}
        )
```

This separation allows you to:
- Monitor system errors separately from tool-specific errors
- Filter logs more effectively for debugging
- Develop better insights into error patterns by category

### Asynchronous Error Analysis with Working Memory

The implementation is already covered by the enhanced error_context manager above, which includes the working memory integration. To extract the analysis from MIRA's thought process, add the following to conversation.py:

# Extract the analysis from thought process in conversation.py
def _extract_error_analysis(self, response_text):
    """Extract error analysis from thought process."""
    import re
    
    # Look for error analysis tag with UUID in thought process
    pattern = r'<error_analysis error_id="([^"]+)">(.*?)</error_analysis>'
    match = re.search(pattern, response_text)
    
    if match:
        error_id = match.group(1)
        analysis = match.group(2).strip()
        
        # Log the analysis with the same error_id for correlation
        analysis_logger.info(f"[{error_id}] Analysis: {analysis}")
        
        # Remove from working memory to avoid clutter
        if self.working_memory:
            # Get items by category
            error_items = self.working_memory.get_items_by_category("error_for_analysis")
            for item in error_items:
                # Check if the item content contains our error_id
                if error_id in item.get("content", ""):
                    self.working_memory.remove(item.get("id"))
                    break
        
        return error_id, analysis
    
    return None, None

# Add this to the conversation.py's generate_response method:
def generate_response(self, user_input, ...):
    # ... existing code ...
    
    # Get the response from the LLM
    response = self.llm_bridge.generate_response(...)
    
    # Extract any error analysis from thought process
    if "<thought_process>" in str(response):
        error_id, analysis = self._extract_error_analysis(str(response))
    
    # ... rest of the function ...
```

This approach:

1. Generates a UUID when an error occurs and logs it immediately
2. Injects the error information into working memory with the UUID
3. Requests MIRA to analyze the error in one sentence
4. Extracts and logs the analysis alongside the original error ID
5. Cleans up the working memory afterward

Benefits:
- Errors are logged immediately, even if the application crashes later
- MIRA generates a human-readable explanation of what the user was trying to do
- The analysis happens in the background without blocking conversation flow
- Error logs and analysis logs can be correlated using the UUID
- Leverages MIRA's existing understanding of conversation context

## Summary

This simplified approach focuses on what matters most:

1. Clear, descriptive error codes that indicate what went wrong
2. Specific error messages that tell MIRA how to fix the problem
3. Clean error boundaries that prevent duplication
4. Tool-specific errors that provide domain-specific clarity

By following these guidelines, you'll create an error handling system that helps MIRA understand and recover from errors effectively without unnecessary complexity.