# Python Tool Testing Guide: Best Practices for BotWithMemory

This guide provides a comprehensive approach to writing effective pytest tests for tools in the BotWithMemory system. Following these guidelines will help you create reliable, maintainable tests that catch issues early.

## Table of Contents

1. [Testing Fundamentals](#testing-fundamentals)
2. [Setting Up Your Test Environment](#setting-up-your-test-environment)
3. [Test Structure for Tools](#test-structure-for-tools)
4. [Writing Effective Test Cases](#writing-effective-test-cases)
5. [Mocking Dependencies](#mocking-dependencies)
6. [Testing Error Handling](#testing-error-handling)
7. [Testing Edge Cases](#testing-edge-cases)
8. [Integration Testing](#integration-testing)
9. [Common Mistakes and How to Avoid Them](#common-mistakes-and-how-to-avoid-them)
10. [Troubleshooting Failed Tests](#troubleshooting-failed-tests)

## Testing Fundamentals

### Why Testing Is Important

- Tests verify your tool works as expected
- Tests catch regressions when code changes
- Tests document how your tool should behave
- Tests make refactoring safer

### Pytest Advantages

The BotWithMemory system uses pytest because it:

- Requires minimal boilerplate
- Provides clear error reporting
- Supports powerful fixtures
- Offers extensive plugin ecosystem

## Setting Up Your Test Environment

### Test File Location and Naming

Create your test files in the `tests/` directory following this naming convention:

```
tests/test_your_tool_name.py
```

Example: If your tool is in `tools/weather_tool.py`, your test file should be `tests/test_weather_tool.py`.

### Basic Test File Structure

Start with this template:

```python
"""Tests for the YourTool."""

import unittest
from unittest.mock import patch, MagicMock

from tools.your_tool import YourTool
from errors import ToolError, ErrorCode

class TestYourTool(unittest.TestCase):
    """Test cases for YourTool."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tool = YourTool()
        
    def tearDown(self):
        """Clean up after each test method."""
        # Any cleanup code needed
        pass
        
    # Test methods will go here
```

## Test Structure for Tools

### Test Categories to Include

For each tool, write tests in these categories:

1. **Initialization tests**: Verify the tool initializes correctly
2. **Input validation tests**: Check that invalid inputs are properly rejected
3. **Core functionality tests**: Ensure the tool's main functions work correctly
4. **Error handling tests**: Confirm errors are caught and handled properly
5. **Integration tests**: Verify the tool works with the broader system

### Example Test Plan

For a tool that processes data from an API:

```python
# Initialization tests
def test_init_creates_logger(self):
    """Test that the tool initializes with a proper logger."""

# Input validation tests
def test_run_with_empty_input(self):
    """Test that empty input raises appropriate error."""
    
def test_run_with_invalid_format(self):
    """Test that incorrectly formatted input raises error."""

# Core functionality tests
def test_run_with_valid_input(self):
    """Test that valid input produces expected output."""
    
def test_data_transformation(self):
    """Test that data is transformed correctly."""

# Error handling tests
def test_api_connection_failure(self):
    """Test that API connection failures are handled gracefully."""

# Integration tests
def test_tool_discoverable(self):
    """Test that tool is discoverable by the tool system."""
```

## Writing Effective Test Cases

### Test Method Naming

Use descriptive names that explain what you're testing:

```python
# Bad: Too vague
def test_run(self):
    
# Good: Clear what's being tested
def test_run_returns_properly_formatted_response(self):
```

### Arrange-Act-Assert Pattern

Structure your tests using the AAA pattern:

```python
def test_example(self):
    # Arrange: Set up the test conditions
    input_data = "test_value"
    expected_result = {"success": True, "output": "processed test_value"}
    
    # Act: Perform the action being tested
    result = self.tool.run(input_param=input_data)
    
    # Assert: Verify the results
    self.assertEqual(result["output"], expected_result["output"])
    self.assertTrue(result["success"])
```

### Independent Tests

Each test should be independent and not rely on the state from other tests:

```python
# Bad: Tests depend on each other
def test_part_one(self):
    self.shared_result = self.tool.run(input="test")
    self.assertIsNotNone(self.shared_result)
    
def test_part_two(self):
    # This depends on test_part_one running first
    self.assertEqual(self.shared_result["status"], "success")

# Good: Tests are independent
def test_result_exists(self):
    result = self.tool.run(input="test")
    self.assertIsNotNone(result)
    
def test_result_status(self):
    result = self.tool.run(input="test")
    self.assertEqual(result["status"], "success")
```

## Mocking Dependencies

### When to Use Mocks

Use mocks when your tool depends on:

- External APIs or services
- File system operations
- Database connections
- Other tools or system components

### Mocking External Requests

For tools that make HTTP requests:

```python
@patch('requests.get')
def test_api_request(self, mock_get):
    # Set up the mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"key": "value"}
    mock_get.return_value = mock_response
    
    # Run the tool
    result = self.tool.run(query="test")
    
    # Verify the request was made correctly
    mock_get.assert_called_once_with(
        "https://api.example.com/data?q=test",
        timeout=30
    )
    
    # Verify the result
    self.assertEqual(result["data"]["key"], "value")
```

### Mocking Tool Dependencies

For tools that depend on other tools:

```python
def test_tool_with_dependency(self):
    # Create a mock dependency tool
    mock_dependency = MagicMock()
    mock_dependency.run.return_value = {"result": "dependency_output"}
    
    # Create your tool with the mock dependency
    tool_repo = MagicMock()
    tool_repo.get_tool.return_value = mock_dependency
    
    # Initialize tool with mock dependency
    tool = YourTool(tool_repo=tool_repo)
    
    # Run your tool
    result = tool.run(input="test")
    
    # Verify dependency was called correctly
    mock_dependency.run.assert_called_once_with(param="expected_value")
    
    # Verify result
    self.assertEqual(result["processed"], "expected_processed_value")
```

### Mocking File Operations

For tools that read or write files:

```python
@patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='test data')
def test_file_reading(self, mock_file):
    result = self.tool.run(filename="test.txt")
    
    # Verify file was opened correctly
    mock_file.assert_called_once_with("test.txt", "r")
    
    # Verify result
    self.assertEqual(result["content"], "test data")
```

## Testing Error Handling

### Testing Exception Raising

To test that your tool raises appropriate exceptions:

```python
def test_invalid_input_raises_error(self):
    with self.assertRaises(ToolError) as context:
        self.tool.run(input="")
        
    # Verify the error code
    self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
    
    # Verify the error message
    self.assertIn("input cannot be empty", str(context.exception))
```

### Testing Error Context Pattern

For tools using the `error_context` pattern:

```python
@patch('errors.error_context')
def test_error_context_used(self, mock_error_context):
    # Set up mock for error_context to still work but be trackable
    mock_error_context.return_value.__enter__.return_value = None
    
    # Run the tool (assuming it uses error_context)
    self.tool.run(param="test")
    
    # Verify error_context was called with correct parameters
    mock_error_context.assert_called_once_with(
        component_name=self.tool.name,
        operation="processing data",
        error_class=ToolError,
        error_code=ErrorCode.TOOL_EXECUTION_ERROR,
        logger=self.tool.logger
    )
```

## Testing Edge Cases

Always test these edge cases:

### Boundary Values

```python
def test_minimum_value(self):
    """Test with the minimum allowed value."""
    result = self.tool.run(count=1)  # Minimum value
    self.assertTrue(result["success"])

def test_maximum_value(self):
    """Test with the maximum allowed value."""
    result = self.tool.run(count=100)  # Maximum value
    self.assertTrue(result["success"])
    
def test_above_maximum_value(self):
    """Test with a value above the maximum."""
    with self.assertRaises(ToolError):
        self.tool.run(count=101)  # Above maximum
```

### Special Characters and Formats

```python
def test_special_characters(self):
    """Test with input containing special characters."""
    result = self.tool.run(input="test!@#$%^&*()_+")
    self.assertTrue(result["success"])
    
def test_unicode_characters(self):
    """Test with input containing Unicode characters."""
    result = self.tool.run(input="こんにちは世界")  # "Hello world" in Japanese
    self.assertTrue(result["success"])
```

### Empty and None Values

```python
def test_empty_optional_param(self):
    """Test with empty optional parameter."""
    result = self.tool.run(required_param="test", optional_param="")
    self.assertTrue(result["success"])
    
def test_none_optional_param(self):
    """Test with None optional parameter."""
    result = self.tool.run(required_param="test", optional_param=None)
    self.assertTrue(result["success"])
```

## Integration Testing

### Testing Tool Discovery

Verify your tool is properly discoverable:

```python
def test_tool_discoverable(self):
    """Test that the tool is discoverable by the system."""
    from tools.repo import ToolRepository
    
    repo = ToolRepository()
    repo.discover_tools()
    
    # Check that our tool is registered
    self.assertIn("your_tool_name", repo.list_all_tools())
```

### Testing with Tool Repository

Test that your tool works with the tool repository:

```python
def test_tool_works_with_repository(self):
    """Test that the tool can be instantiated by the repository."""
    from tools.repo import ToolRepository
    
    repo = ToolRepository()
    tool = repo.get_tool("your_tool_name")
    
    # Check that the tool was found and instantiated
    self.assertIsNotNone(tool)
    self.assertEqual(tool.name, "your_tool_name")
    
    # Check that the tool runs correctly
    result = tool.run(param="test")
    self.assertTrue(result["success"])
```

## Common Mistakes and How to Avoid Them

### 1. Not Testing All Parameters

**Problem**: Testing only happy paths with specific parameter sets.

**Solution**: Test all parameters with various values, including edge cases.

```python
# Testing all parameters
def test_all_parameters_combinations(self):
    # Test with minimum values
    result1 = self.tool.run(param1="min", param2=1)
    self.assertTrue(result1["success"])
    
    # Test with maximum values
    result2 = self.tool.run(param1="max", param2=100)
    self.assertTrue(result2["success"])
    
    # Test with mixed values
    result3 = self.tool.run(param1="min", param2=100)
    self.assertTrue(result3["success"])
```

### 2. Brittle Mocks

**Problem**: Mocks that specify too many details, making tests fragile.

**Solution**: Mock only what's necessary, use less specific assertions.

```python
# Too brittle
@patch('requests.get')
def test_too_brittle(self, mock_get):
    result = self.tool.run(query="test")
    mock_get.assert_called_once_with(
        "https://api.example.com/data?q=test&exact=false&limit=10&page=1",
        headers={"Authorization": "Bearer token", "Content-Type": "application/json"},
        timeout=30
    )

# More robust
@patch('requests.get')
def test_more_robust(self, mock_get):
    result = self.tool.run(query="test")
    
    # Only assert the important parts
    args, kwargs = mock_get.call_args
    self.assertIn("https://api.example.com/data", args[0])
    self.assertIn("q=test", args[0])
    self.assertIn("timeout", kwargs)
```

### 3. Missing Error Tests

**Problem**: Not testing how the tool handles errors.

**Solution**: Explicitly test error conditions.

```python
@patch('requests.get')
def test_api_timeout(self, mock_get):
    # Simulate a timeout
    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
    
    # Check that the tool handles the error properly
    with self.assertRaises(ToolError) as context:
        self.tool.run(query="test")
        
    self.assertEqual(context.exception.code, ErrorCode.API_CONNECTION_ERROR)
    self.assertIn("timed out", str(context.exception))
```

### 4. Ignoring Return Values

**Problem**: Not thoroughly checking return values.

**Solution**: Verify all important fields in the response.

```python
# Incomplete checking
def test_incomplete(self):
    result = self.tool.run(input="test")
    self.assertTrue(result["success"])  # Only checks one field

# Thorough checking
def test_thorough(self):
    result = self.tool.run(input="test")
    self.assertTrue(result["success"])
    self.assertEqual(result["status"], "completed")
    self.assertIn("output", result)
    self.assertIsInstance(result["timestamp"], str)
    self.assertTrue(result["output"].startswith("Processed"))
```

## Troubleshooting Failed Tests

### Common Test Failures and Solutions

1. **"AssertionError: Expected call not found"**
   - Problem: Mock wasn't called as expected
   - Solution: Check if conditional logic is preventing the call or if the call arguments are different

2. **"TypeError: ... got an unexpected keyword argument ..."**
   - Problem: Mismatch between test parameters and actual method signature
   - Solution: Check the method signature and update test parameters

3. **"AttributeError: 'Mock' object has no attribute ..."**
   - Problem: Mock setup is incomplete
   - Solution: Ensure all necessary mock attributes and methods are configured

### Debugging Tips

1. **Use pytest's verbose mode**:
   ```
   python -m pytest test_your_tool.py -v
   ```

2. **Print debugging information**:
   ```python
   def test_with_debug(self):
       result = self.tool.run(input="test")
       print(f"DEBUG: Result = {result}")  # Will show in pytest output with -v
       self.assertTrue(result["success"])
   ```

3. **Use pytest's built-in debugger**:
   ```
   python -m pytest test_your_tool.py --pdb
   ```
   This drops you into a debugger when a test fails.

4. **Isolate failing tests**:
   ```
   python -m pytest test_your_tool.py::TestYourTool::test_specific_method -v
   ```

### Making Tests More Reliable

1. **Add more assertions**: Check intermediate values, not just the final result
2. **Improve test isolation**: Ensure tests don't depend on each other
3. **Use appropriate timeouts**: For operations that might take time
4. **Reset mocks between tests**: Use `setUp` to configure fresh mocks

---

## Quick Reference: Test Checklist

Before considering your tests complete, ensure you've tested:

- [ ] Basic functionality with valid inputs
- [ ] Input validation for all parameters
- [ ] Error handling for all potential error conditions
- [ ] Edge cases (min/max values, special characters, empty values)
- [ ] Integration with the tool system
- [ ] All public methods and important private methods
- [ ] Any dependencies are properly mocked

Following this guide will help you create robust, maintainable tests that catch issues early and document how your tool should behave.
