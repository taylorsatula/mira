"""
Tests for the error handling module.

This module tests the error classes, error codes, and error
handling utility function.
"""
import pytest

from errors import (
    ErrorCode, AgentError, ConfigError, APIError,
    FileOperationError, ToolError, ConversationError,
    StimulusError, handle_error
)


def test_error_code_enum():
    """Test the ErrorCode enumeration."""
    # Check that error codes are categorized correctly
    # Configuration errors (1xx)
    assert ErrorCode.CONFIG_NOT_FOUND.value == 101
    assert ErrorCode.INVALID_CONFIG.value == 102
    
    # API errors (2xx)
    assert ErrorCode.API_CONNECTION_ERROR.value == 201
    assert ErrorCode.API_AUTHENTICATION_ERROR.value == 202
    
    # File operation errors (3xx)
    assert ErrorCode.FILE_NOT_FOUND.value == 301
    assert ErrorCode.FILE_PERMISSION_ERROR.value == 302
    
    # Tool errors (4xx)
    assert ErrorCode.TOOL_NOT_FOUND.value == 401
    assert ErrorCode.TOOL_EXECUTION_ERROR.value == 402
    
    # Conversation errors (5xx)
    assert ErrorCode.CONVERSATION_NOT_FOUND.value == 501
    assert ErrorCode.CONTEXT_OVERFLOW.value == 502
    
    # Stimulus errors (6xx)
    assert ErrorCode.STIMULUS_INVALID.value == 601
    
    # Uncategorized/system errors (9xx)
    assert ErrorCode.UNKNOWN_ERROR.value == 901


def test_base_agent_error():
    """Test the base AgentError class."""
    # Create an error
    error = AgentError("Test error message")
    
    # Check properties
    assert error.message == "Test error message"
    assert error.code == ErrorCode.UNKNOWN_ERROR
    assert error.details == {}
    
    # Check string representation
    assert str(error) == "[UNKNOWN_ERROR] Test error message"
    
    # With specific code and details
    error = AgentError(
        "Test error with code",
        ErrorCode.CONFIG_NOT_FOUND,
        {"file": "config.json"}
    )
    assert error.code == ErrorCode.CONFIG_NOT_FOUND
    assert error.details == {"file": "config.json"}
    assert str(error) == "[CONFIG_NOT_FOUND] Test error with code"


def test_specific_error_classes():
    """Test the specific error classes."""
    # ConfigError
    error = ConfigError("Configuration error")
    assert error.code == ErrorCode.INVALID_CONFIG
    assert isinstance(error, AgentError)
    
    # APIError
    error = APIError("API error")
    assert error.code == ErrorCode.API_RESPONSE_ERROR
    assert isinstance(error, AgentError)
    
    # FileOperationError
    error = FileOperationError("File error")
    assert error.code == ErrorCode.FILE_NOT_FOUND
    assert isinstance(error, AgentError)
    
    # ToolError
    error = ToolError("Tool error")
    assert error.code == ErrorCode.TOOL_EXECUTION_ERROR
    assert isinstance(error, AgentError)
    
    # ConversationError
    error = ConversationError("Conversation error")
    assert error.code == ErrorCode.CONVERSATION_NOT_FOUND
    assert isinstance(error, AgentError)
    
    # StimulusError
    error = StimulusError("Stimulus error")
    assert error.code == ErrorCode.STIMULUS_INVALID
    assert isinstance(error, AgentError)


def test_error_with_custom_code():
    """Test creating errors with custom error codes."""
    # ConfigError with custom code
    error = ConfigError(
        "Missing environment variable",
        ErrorCode.MISSING_ENV_VAR
    )
    assert error.code == ErrorCode.MISSING_ENV_VAR
    
    # APIError with custom code
    error = APIError(
        "Authentication failed",
        ErrorCode.API_AUTHENTICATION_ERROR
    )
    assert error.code == ErrorCode.API_AUTHENTICATION_ERROR


def test_error_handler_for_agent_errors():
    """Test the error handling utility function with AgentErrors."""
    # Handle an AgentError
    error = ConfigError(
        "Configuration file not found",
        ErrorCode.CONFIG_NOT_FOUND
    )
    result = handle_error(error)
    
    # Check result
    assert result == "Error: Configuration file not found"


def test_error_handler_for_standard_exceptions():
    """Test the error handling utility function with standard exceptions."""
    # Handle a standard exception
    error = ValueError("Invalid value")
    result = handle_error(error)
    
    # Check result
    assert result == "An unexpected error occurred: Invalid value"


def test_error_details():
    """Test error details preservation."""
    # Create an error with details
    details = {
        "file": "config.json",
        "line": 42,
        "context": "Loading configuration"
    }
    error = ConfigError(
        "Error in configuration file",
        ErrorCode.INVALID_CONFIG,
        details
    )
    
    # Check that details are preserved
    assert error.details == details
    
    # Check that details don't affect the string representation
    assert str(error) == "[INVALID_CONFIG] Error in configuration file"