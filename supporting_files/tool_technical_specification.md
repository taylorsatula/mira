# LLM Tool Framework Technical Specification

## Overview

This document provides technical specifications for developing tools within our LLM foundation framework. It outlines the architecture, implementation patterns, and best practices based on the reference implementation (`WeatherTool`).

## Tool Architecture

### Core Components

1. **Base Class**: All tools inherit from the `Tool` base class
2. **Error Handling**: Tools use the `error_context` pattern and `ToolError` class
3. **Metadata**: Each tool includes descriptive metadata for discovery and documentation
4. **Interface**: Tools implement a standardized `run()` method as the main entry point

## Implementation Requirements

### Class Structure

Each tool must be implemented as a Python class that inherits from the `Tool` base class:

```python
from tools.repo import Tool

class YourTool(Tool):
    # Implementation here
```

### Required Metadata

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `name` | string | Unique identifier for the tool | `"weather_tool"` |
| `description` | string | Detailed description of the tool's functionality | See specification below |
| `usage_examples` | list | Examples of valid inputs and outputs | See specification below |

### Tool Description Format

The description should follow this structure:
- **Functionality overview**: What the tool does
- **Use cases**: When to use this tool
- **Parameters**: Description of input parameters
- **Limitations**: Any constraints or limitations
- **Performance characteristics**: Response time, resource usage

### Method Requirements

Each tool must implement the following methods:

1. **`__init__(self)`**: Initialize the tool with necessary configuration
2. **`run(self, **params)`**: Main entry point that processes inputs and returns results

## Error Handling

Tools must use the `error_context` pattern for robust error management:

```python
with error_context(
    component_name=self.name,
    operation="operation description",
    error_class=ToolError,
    error_code=ErrorCode.TOOL_INVALID_INPUT,
    logger=self.logger
):
    # Operation that might raise exceptions
```

### Error Codes

| Error Code | Usage |
|------------|-------|
| `TOOL_INVALID_INPUT` | For parameter validation failures |
| `TOOL_EXECUTION_ERROR` | For errors during tool execution |
| `TOOL_DEPENDENCY_ERROR` | For errors in external dependencies |

## Best Practices

### 1. Input Validation

Always validate all input parameters before processing:

```python
if not param_name or not isinstance(param_name, expected_type):
    raise ToolError(
        "Descriptive error message",
        ErrorCode.TOOL_INVALID_INPUT,
        {"provided_value": str(param_name)}
    )
```

### 2. Method Organization

- Public methods (`run()`) should focus on parameter validation and workflow
- Complex functionality should be moved to private helper methods (prefixed with `_`)
- Private methods should have single responsibilities

### 3. Logging

Use the inherited logger for operational visibility:

```python
self.logger.info("High-level operation information")
self.logger.debug("Detailed debug information")
```

### 4. Documentation

- Include comprehensive docstrings for all methods
- Document parameters, return values, and exceptions
- Add implementation notes for complex logic

## Response Format

Tools should return data as dictionaries with consistent structures:

```python
{
    "primary_result_field": value,
    "metadata_field1": value,
    "metadata_field2": value,
    "timestamp": "ISO-formatted timestamp"
}
```

## Tool Testing

New tools should include tests that verify:

1. Valid inputs produce expected outputs
2. Invalid inputs raise appropriate errors
3. Edge cases are handled correctly

## Example Implementation

The `WeatherTool` class demonstrates all these principles:

- Clear class and method documentation
- Proper error handling with specific error messages
- Separation of concerns with helper methods
- Comprehensive input validation
- Structured return values

## Deployment Process

1. Implement your tool following this specification
2. Include comprehensive tests
3. Submit for code review
4. Once approved, the tool will be automatically discovered and made available to the LLM

## Additional Resources

- `errors.py`: Contains error handling utilities
- `tools/repo.py`: Contains the base `Tool` class
- Existing tools in the codebase for reference implementations
