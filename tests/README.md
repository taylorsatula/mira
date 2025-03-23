# Tests for Bot With Memory

This directory contains tests for the AI agent system. The test suite includes unit tests, integration tests, and end-to-end tests for all major components of the system.

## Test Organization

- **Unit Tests**: Tests for individual components in isolation
- **Integration Tests**: Tests for interactions between components
- **Edge Cases**: Tests for boundary conditions and unusual inputs
- **Error Handling**: Tests for error conditions and recovery

## Running Tests

To run all tests:

```
pytest
```

To run tests for a specific module:

```
pytest tests/test_config.py
```

To run a specific test:

```
pytest tests/test_config.py::test_config_defaults
```

With verbose output:

```
pytest -v
```

## Test Structure

- **`conftest.py`**: Shared fixtures for all tests
- **`test_config.py`**: Tests for configuration management
- **`test_conversation.py`**: Tests for conversation management
- **`test_crud.py`**: Tests for file operations
- **`test_errors.py`**: Tests for error handling
- **`test_llm_bridge.py`**: Tests for LLM API communication
- **`test_main.py`**: Tests for main application flow
- **`test_stimuli.py`**: Tests for stimulus handling
- **`test_tools.py`**: Tests for tool system

## Test Coverage

The tests aim to cover:

1. Happy paths for all core functionality
2. Edge cases and boundary conditions
3. Error handling and recovery
4. Component interactions
5. End-to-end workflows

## Mock Strategy

The tests use mocks for:

- External API calls to Anthropic
- File system operations (using temporary directories)
- Time-dependent operations
- Tool execution that would have external effects

## Adding New Tests

When adding new features, please follow these guidelines:

1. Write unit tests for all new functions and methods
2. Add integration tests for component interactions
3. Update existing tests when modifying behavior
4. Test both happy paths and error conditions
5. Use fixtures from `conftest.py` where appropriate

## Test Plan

For a detailed breakdown of test areas and specific test cases, see `test_plan.md` in this directory.