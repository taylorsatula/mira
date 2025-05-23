# Error Code Implementation Guide

This document provides guidance for implementing a robust, future-proof error handling system. It defines a structured approach to error categorization, handling, and management that supports the dynamic nature of our system.

> **IMPORTANT**: For guidance on implementing bot-friendly error formats that enable MIRA to programmatically understand and recover from errors, see [Bot-Friendly Error Format Guide](Bot_Friendly_Error_Format.md).

## Table of Contents
1. [Error System Design Philosophy](#error-system-design-philosophy)
2. [Error Classification Structure](#error-classification-structure)
3. [Error Handling Standards](#error-handling-standards)
4. [Error Categories](#error-categories)
5. [Error Context Enrichment](#error-context-enrichment)
6. [Error Recovery Strategies](#error-recovery-strategies)
7. [Implementation Plan](#implementation-plan)
8. [Centralized Error Logging](#centralized-error-logging)
9. [Maintenance and Evolution](#maintenance-and-evolution)
10. [Bot-Friendly Error Format](#bot-friendly-error-format)

## Error System Design Philosophy

A robust error handling system serves multiple purposes:

1. **Developer Experience**: Provides clear, actionable information about what went wrong
2. **Debugging Efficiency**: Enables rapid identification of error sources
3. **Error Recovery**: Facilitates graceful degradation and recovery paths
4. **Error Aggregation**: Supports meaningful error analytics and pattern detection
5. **User Experience**: Translates technical errors into user-friendly messages

The error system should be:
- **Hierarchical**: Organized in logical categories with inheritance relationships
- **Contextual**: Include relevant details about the error's context
- **Consistent**: Follow uniform patterns across the entire codebase
- **Extensible**: Support adding new error types without disrupting existing code
- **Granular**: Specific enough to differentiate between error conditions

## Error Classification Structure

All errors in the system should:

1. **Have a unique numeric code**: Facilitates logging, filtering, and cross-referencing
2. **Be grouped by functional domain**: Network, database, tool-specific, etc.
3. **Include detailed context**: Not just what happened, but relevant state information
4. **Follow inheritance patterns**: From general to specific error types
5. **Maintain error hierarchies**: Base errors and derived, more specific errors

### Numeric Code Ranges

Each functional domain is assigned a unique range of error codes:

- **System Errors** (1xx): Core framework errors
- **Network Errors** (2xx): API, DNS, SSL, and connectivity issues
- **Authentication/Authorization** (3xx): Permission and access control issues
- **Tool Errors** (4xx): Issues with tools, inputs, and execution
- **Data Errors** (5xx): Data validation, formatting, and state errors
- **Resource Errors** (6xx): Missing or unavailable resources
- **Database Errors** (7xx): Connection, query, and data integrity issues
- **Workflow Errors** (8xx): Workflow definition, state, and execution errors
- **Integration Errors** (9xx): Third-party integration issues

## Error Handling Standards

All error handling should follow these standards:

1. **Use the error_context manager**: Ensures consistent error wrapping
2. **Raise specific error types**: Never raise generic exceptions
3. **Include appropriate context**: Add relevant state information to help debugging
4. **Follow error propagation chains**: Wrap lower-level errors in domain-specific ones
5. **Support error recovery**: Include information that facilitates recovery
6. **Pass through original tool errors**: Errors from tools should be passed through without modification or prefixing

### Error Construction

Every error should include:

1. **Clear error message**: Human-readable description of the issue
2. **Error code**: Unique numeric identifier within its domain
3. **Context**: Dictionary of relevant state information for debugging
4. **Unique ID**: UUID for tracking the error instance across the system
5. **Recovery strategy**: Information embedded directly in the error object about:
   - Whether the error is retryable
   - Suggested parameter adjustments
   - Alternative approaches
   - Required actions before retry
6. **MIRA actionability**: Flag indicating if this error can be handled programmatically

### Error Propagation Control

To prevent duplicate error messages across multiple system layers:

1. **Single Source of Truth**: Errors should only be logged at their origination point
2. **Centralized Logging**: Use a central logging system for all error events
3. **Error ID Tracking**: Track errors through the system using their unique IDs
4. **Debug Mode Override**: Allow more verbose logging when in DEBUG mode
5. **Root Cause Preservation**: Always maintain the original error code and message when re-wrapping errors

## Error Categories

### System Errors (1xx)
Basic framework and configuration errors that prevent system operation.

### Network Errors (2xx)
Connectivity issues, API interactions, and protocol-specific errors.

### Authentication Errors (3xx)
Issues with authentication, authorization, and permissions.

### Tool Errors (4xx)
Tool-specific failures including invalid inputs, execution failures, and output processing.

### Data Errors (5xx)
Data validation, formatting, and content issues.

### Resource Errors (6xx)
Missing or unavailable resources needed for operation.

### Database Errors (7xx)
Database connection, query execution, and data integrity issues.

### Workflow Errors (8xx)
Workflow definition, state management, and execution problems.

### Integration Errors (9xx)
Third-party service integration issues not covered by network errors.

## Error Context Enrichment

Every error should include relevant contextual information:

1. **Input parameters**: The values that led to the error
2. **Expected values/formats**: What would have been valid
3. **System state**: Relevant information about the current state
4. **Resource identifiers**: IDs of involved resources (users, workflows, etc.)
5. **Timestamps**: When the error occurred
6. **Operation context**: What operation was being attempted

## Error Recovery Strategies

For each error category, define appropriate recovery strategies:

1. **Retry policies**: Which errors are retriable and with what backoff
2. **Fallback mechanisms**: Alternative paths when primary operations fail
3. **Graceful degradation**: How to provide partial functionality
4. **User feedback**: What information to surface to users
5. **Error logging**: What details to capture for debugging

## Implementation Plan

### Phase 1: Core Error System Enhancement

- Enhance `ToolError` class in `errors.py`:
  - Add unique IDs (UUID) for error instance tracking
  - Add MIRA actionability flags to indicate bot-handleable errors
  - Implement a `to_json()` method for structured error formatting
  - Add recovery strategy fields directly in the error objects
- Expand ErrorCode Enum in `errors.py` with more granular error categories
- Modify `error_context` manager in `errors.py` to preserve original error information
- Create a centralized error logging function in `utils/logger.py`
- Implement recovery strategy fields in error objects

#### Key Principle: Pass Through Original Tool Errors
The critical change is ensuring errors from tools are passed through the system without being modified or prefixed. The source tool name should be the only prefix in the error message seen by the bot, making it clear which component caused the error.

```python
# Example implementation of to_json() method for MIRA
def to_json(self) -> dict:
    """Convert error to MIRA-friendly JSON format with pure data approach"""
    # Extract parameter that failed validation (if applicable)
    validation = {}
    if hasattr(self, 'validation_info') and self.validation_info:
        validation = self.validation_info
    
    # Create constraints for parameters that need to be adjusted
    parameter_constraints = {}
    if hasattr(self, 'parameter_constraints') and self.parameter_constraints:
        parameter_constraints = self.parameter_constraints
    
    return {
        "error": {
            "type": {
                "code": self.code.name,
                "category": self._get_category_from_code()
            },
            "source": {
                "tool": self.context.get("tool", "unknown"),
                "operation": self.context.get("operation", "unknown")
            },
            "parameters": self.context.get("params", {}),
            "validation": validation,
            "recovery": {
                "is_retryable": getattr(self, "is_retryable", False),
                "parameter_constraints": parameter_constraints
            },
            "debug": {
                "id": self.id,
                "timestamp": self.timestamp.isoformat() + "Z"
            }
        }
    }
```

- Modify `error_context` manager in `errors.py` to avoid changing error messages
- Update the tool run method in `tools/repo.py` to add context and maintain error identity
- Update the conversation handler's error propagation to format errors appropriately for MIRA
- Apply consistent error handling pattern across all tools

```python
# Example implementation for standardizing tool errors
def standardize_tool_error(
    error, 
    tool_name, 
    operation, 
    params=None, 
    is_retryable=False,
    validation_info=None,
    parameter_constraints=None
):
    """
    Standardizes errors from tool operations with pure data approach
    
    Args:
        error: The original error
        tool_name: Name of the tool that raised the error
        operation: Operation that was being performed
        params: Operation parameters (sensitive values will be filtered)
        is_retryable: Whether the error can be retried
        validation_info: Information about parameter validation failure:
            - parameter: The parameter that failed validation
            - constraint: The constraint that was violated
            - value: The value that was provided
        parameter_constraints: Constraints that parameters must satisfy for retry:
            Dictionary mapping parameter names to constraint descriptions
    """
    import uuid
    from datetime import datetime
    
    # Create context with tool information
    context = {
        "tool": tool_name,
        "operation": operation
    }
    
    # Add sanitized parameters to context
    if params:
        context["params"] = {
            k: "***REDACTED***" if k.lower() in ["password", "token", "key", "secret"] 
            else v 
            for k, v in params.items()
        }
    
    # Default values
    if validation_info is None:
        validation_info = {}
    
    if parameter_constraints is None:
        parameter_constraints = {}
    
    if not isinstance(error, ToolError):
        # Create new ToolError with pure data approach
        return ToolError(
            message=f"{tool_name} {operation} failed: {error}",  # Still need a message for logging
            code=_determine_error_code(error),
            context=context,
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            is_retryable=is_retryable,
            validation_info=validation_info,
            parameter_constraints=parameter_constraints,
            original_error=error
        )
    else:
        # Update existing ToolError with additional context
        error.context.update(context)
        
        # Update fields for the pure data approach
        if not hasattr(error, 'validation_info') or not error.validation_info:
            error.validation_info = validation_info
            
        if not hasattr(error, 'parameter_constraints') or not error.parameter_constraints:
            error.parameter_constraints = parameter_constraints
            
        if not hasattr(error, 'is_retryable'):
            error.is_retryable = is_retryable
            
        # Ensure error has an ID
        if not hasattr(error, 'id') or not error.id:
            error.id = str(uuid.uuid4())
            
        # Ensure error has a timestamp
        if not hasattr(error, 'timestamp'):
            error.timestamp = datetime.utcnow()
        
        return error
```

# Example conversation.py implementation for error boundary with MIRA
```python
def _process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process tool calls from the LLM and format errors using pure data approach."""
    tool_results = {}
    
    for tool_call in tool_calls:
        tool_name = tool_call["tool_name"]
        tool_input = tool_call["input"]
        tool_id = tool_call["id"]
        
        with error_context(
            component_name="Conversation",
            operation=f"executing tool {tool_name}",
            error_class=ConversationError,
            logger=self.logger
        ):
            try:
                # Log tool call information
                self.logger.info(f"Tool call: {tool_name} with input: {json.dumps(tool_input, indent=2)}")
                
                # Invoke the tool
                result = self.tool_repo.invoke_tool(tool_name, tool_input)
                
                # Format successful result
                tool_results[tool_id] = {
                    "content": str(result),
                    "is_error": False
                }
                
            except Exception as e:
                # Always transform to structured JSON for all errors
                if isinstance(e, ToolError):
                    # Format error using pure data approach
                    tool_results[tool_id] = {
                        "content": json.dumps(e.to_json()),
                        "is_error": True
                    }
                else:
                    # Wrap generic exception in a ToolError with pure data
                    wrapped_error = standardize_tool_error(
                        error=e,
                        tool_name=tool_name,
                        operation="execute",
                        params=tool_input,
                        is_retryable=False  # Default to non-retryable for unknown errors
                    )
                    tool_results[tool_id] = {
                        "content": json.dumps(wrapped_error.to_json()),
                        "is_error": True
                    }
                
                # Always log errors
                self.logger.error(f"Tool execution error: {tool_name}: {e}")
                
    return tool_results
```

#### Asynchronous Error Handling Support
Provide corresponding async versions of error handling mechanisms for asynchronous code using the same error types:

```python
@asynccontextmanager
async def async_error_context(
    component_name: str,
    operation: Optional[str] = None,
    error_class: Type[AgentError] = AgentError,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    logger: Optional[logging.Logger] = None
):
    """Async version of error_context for use with async/await code."""
    if logger is None:
        logger = logging.getLogger(f"error.{component_name}")
        
    try:
        yield
    except Exception as e:
        # Same error handling logic as in the synchronous version
        if isinstance(e, AgentError):
            logger.error(f"{component_name} error: {e}")
            raise
            
        error_msg = f"Error in {component_name}"
        if operation:
            error_msg += f" during {operation}"
            
        logger.error(f"{error_msg}: {str(e)}")
        
        raise error_class(
            f"{error_msg}: {str(e)}",
            error_code,
            {"original_error": str(e)}
        )
```

- Create an `async_error_context` manager for use with async/await patterns
- Implement utilities for handling errors in background tasks and event loops
- Add retry patterns with exponential backoff for transient errors in async operations
- Ensure errors in async code include the same detailed context as synchronous errors

### Phase 2: Migration Path

Transitioning to the new error system should follow these steps:

1. **Define error hierarchy**: Establish the complete error type hierarchy
2. **Update core error system**: Implement base error classes and the error registry
3. **Create error categories**: Implement domain-specific error classes
4. **Implement error context manager**: Create reusable error handling mechanisms
5. **Add centralized logging**: Implement centralized logging with error ID tracking for deduplication
6. **Prioritize critical paths**: Start with high-impact, frequently used components
7. **Roll out incrementally**: Move through the codebase systematically
8. **Add error recovery mechanisms**: Implement retry policies and fallbacks
9. **Update error documentation**: Document all error codes and their meanings
10. **Implement error analytics**: Add logging and monitoring for error patterns

### Phase 3: Error Monitoring and Analysis

- Create an ErrorMetrics class in `utils/error_metrics.py`
- Integrate with centralized logging in `utils/logger.py`
- Create test cases for common error scenarios
- Test error propagation through multiple layers
- Verify no duplicate logging
- Check error context is preserved
- Ensure error IDs are useful for tracking
- Test recovery strategies for applicable error types
- Verify error metrics are properly incremented

### Implementation Priority Order
1. Core `ToolError` and `error_context` updates 
2. Expand error codes for more specificity
3. Centralized logging function
4. MIRA-friendly error formatting in conversation boundary
5. Asynchronous error handling support
6. Recovery strategies for common errors
7. Tool repository error handling
8. API and conversation layer integration

### Structured Error Format for MIRA

For errors at the boundary where MIRA directly interacts with tools, implement a structured format:

1. **Error Translation Boundary**: In conversation.py's `_process_tool_calls` method
2. **Structured JSON Format**: Convert ToolErrors to structured JSON for MIRA with:
   - Error code and category
   - Source component identification
   - Context information about the operation
   - Recovery guidance (retryable status, needed parameter adjustments)
   - Alternative approaches when available
3. **Actionability Classification**: Identify which errors MIRA can take action on:
   - Retryable network errors
   - Invalid input errors
   - Missing resource errors
   - Rate limiting and throttling
4. **Non-Actionable Errors**: Provide simplified format for system errors that MIRA cannot fix
5. **Conversation Handler Updates**: Modify error handling in conversation.py to use this structured format

### Testing Approach
- Unit tests for each updated component
- Integration tests for error flows
- Manual verification of logs for key error scenarios
- Stress testing with error injection

### Backward Compatibility Considerations
- Maintain existing error signatures
- Ensure tools expecting certain error formats continue working
- Preserve error handling behavior while eliminating duplicates

## Centralized Error Logging

A centralized logging system is essential for effective error management:

1. **Unified Log Storage**: Store all errors in a single location/system
2. **Structured Logging Format**: Use a consistent format (JSON recommended) with these fields:
   - Timestamp (in UTC)
   - Error code
   - Error message
   - Source component/module
   - Severity level
   - Context data
   - Stack trace (when appropriate)
   - User/session ID (for user-related errors)
3. **Log Rotation**: Implement automatic log rotation to manage file sizes
4. **Log Retention Policy**: Define how long logs are kept before archiving/deletion
5. **Log Access Controls**: Ensure sensitive information in logs is properly protected
6. **Search Capabilities**: Enable efficient searching through historical logs
7. **Error Correlation**: Support linking related errors via correlation IDs

### Log Level Guidelines

Use appropriate log levels for different error scenarios:

- **CRITICAL**: System-wide failures requiring immediate attention
- **ERROR**: Component failures that prevent specific operations
- **WARNING**: Non-fatal issues that might indicate future problems
- **INFO**: Normal operation information (not errors)
- **DEBUG**: Detailed diagnostic information (only in development/debugging)

### Historical Error Analysis

Implement tooling to analyze error patterns over time:

1. **Error frequency reporting**: Track most common error types
2. **Temporal analysis**: Identify time-based patterns in error occurrences
3. **Related error grouping**: Cluster related errors to identify systemic issues
4. **Error trend visualization**: Provide dashboards to monitor error rates
5. **Alert thresholds**: Set up notifications for unusual error patterns

## Maintenance and Evolution

The error system should evolve with the application:

1. **Regular review**: Periodically review error patterns to identify gaps
2. **Refine granularity**: Split too-general errors into more specific types
3. **Consolidate error patterns**: Group similar errors when appropriate
4. **Update recovery strategies**: Refine retry policies based on observed behavior
5. **Error analytics**: Use error data to guide system improvements
6. **Developer feedback**: Incorporate feedback on error usability

## Best Practices

- **Use error_context**: Always wrap operations in error_context managers
- **Provide context**: Include relevant operation parameters in error context
- **Don't re-wrap errors**: Allow ToolErrors to propagate without re-wrapping
- **Appropriate error codes**: Use specific error codes rather than generic ones
- **Recovery strategies**: Implement appropriate recovery for transient errors
- **Error boundaries**: Establish clear boundaries for error handling vs. propagation

## Expected Outcomes

- Cleaner, non-duplicated error logs
- Original tool-specific errors preserved for the bot
- Improved error traceability through unique IDs
- Better error context for debugging
- Standardized error handling across components
- Effective error recovery for transient issues
- More specific error codes for easier troubleshooting
- Insights into error patterns through metrics
- Clear guidelines for consistent error handling
- Errors that can be directly understood and processed by the bot

## Bot-Friendly Error Format

A critical aspect of our error system is enabling MIRA to programmatically understand, analyze, and recover from errors during tool execution. When a tool fails, MIRA will receive the error as a response and must be able to:

1. **Understand the exact nature of the failure**
2. **Determine if the error is retryable**
3. **Identify what adjustments are needed before retrying**
4. **Explain the issue clearly to users**
5. **Learn from error patterns to improve future interactions**

To support these requirements, we've developed a standardized bot-friendly error format that focuses on structured data rather than text messages, eliminating multiple sources of truth:

```json
{
  "error": {
    "type": {
      "code": "INVALID_INPUT_ERROR",
      "category": "tool"
    },
    "source": {
      "tool": "calendar_tool",
      "operation": "create_event"
    },
    "parameters": {
      "title": "Team meeting",
      "date": "2023-01-01T10:00:00Z"
    },
    "validation": {
      "parameter": "date",
      "constraint": "must_be_future_date",
      "value": "2023-01-01T10:00:00Z"
    },
    "recovery": {
      "is_retryable": true,
      "parameter_constraints": {
        "date": "future_date"
      }
    },
    "debug": {
      "id": "550e8400-e29b-41d4-a716-446655440001",
      "timestamp": "2023-04-25T14:22:33.456Z"
    }
  }
}
```

The key principles in this approach are:

1. **Pure Data, No Messages**: The error contains only data that describes what went wrong, without any human-language interpretation
2. **Single Source of Truth**: Each piece of information exists exactly once in the structure
3. **Constraint-Based**: Parameters have constraints rather than suggested values or messages about what went wrong
4. **MIRA Interprets**: MIRA is responsible for interpreting the error data and determining appropriate actions

This format includes only:
- **Error type**: Code and category to classify the error
- **Error source**: Which tool and operation produced the error
- **Parameters**: The exact parameters sent to the operation
- **Validation failure**: Which parameter failed validation and what constraint it violated
- **Recovery information**: Whether the error is retryable and what constraints need to be satisfied
- **Debug information**: Technical details for tracking and debugging

For detailed implementation guidance, standards, and examples, refer to the [Bot-Friendly Error Format Guide](Bot_Friendly_Error_Format.md).

By following these guidelines, you'll create an error handling system where MIRA can programmatically understand and respond to errors with complete consistency, without multiple sources of truth or interpretative information that could become outdated.