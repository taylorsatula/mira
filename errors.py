"""
Custom exception hierarchy for the AI agent system.

This module defines standardized error codes, messages, and categorization
for different types of errors that can occur within the system.
"""
from contextlib import contextmanager
from enum import Enum
import logging
from typing import Optional, Dict, Any, Type, Callable


class ErrorCode(Enum):
    """
    Enumeration of error codes for standardized error handling.

    Error codes are grouped by category for easier identification:
    - 1xx: Configuration errors
    - 2xx: API errors
    - 3xx: File operation errors
    - 4xx: Tool errors
    - 5xx: Conversation errors
    - 6xx: Stimulus errors
    - 9xx: Uncategorized/system errors
    """
    # Configuration errors (1xx)
    CONFIG_NOT_FOUND = 101
    INVALID_CONFIG = 102
    MISSING_ENV_VAR = 103

    # API errors (2xx)
    API_CONNECTION_ERROR = 201
    API_AUTHENTICATION_ERROR = 202
    API_RATE_LIMIT_ERROR = 203
    API_RESPONSE_ERROR = 204
    API_TIMEOUT_ERROR = 205

    # File operation errors (3xx)
    FILE_NOT_FOUND = 301
    FILE_PERMISSION_ERROR = 302
    FILE_READ_ERROR = 303
    FILE_WRITE_ERROR = 304
    INVALID_JSON = 305

    # Tool errors (4xx)
    TOOL_NOT_FOUND = 401
    TOOL_EXECUTION_ERROR = 402
    TOOL_INVALID_INPUT = 403
    TOOL_INVALID_OUTPUT = 404
    TOOL_INITIALIZATION_ERROR = 405

    # Conversation errors (5xx)
    CONVERSATION_NOT_FOUND = 501
    CONTEXT_OVERFLOW = 502
    INVALID_INPUT = 503

    # Stimulus errors (6xx)
    STIMULUS_INVALID = 601
    STIMULUS_PROCESSING_ERROR = 602
    STIMULUS_HANDLER_NOT_FOUND = 603
    STIMULUS_TYPE_INVALID = 604

    # Uncategorized/system errors (9xx)
    UNKNOWN_ERROR = 901
    NOT_IMPLEMENTED = 902


class AgentError(Exception):
    """
    Base exception class for all agent-related errors.

    All other custom exceptions inherit from this class, allowing for
    standardized error handling throughout the system.
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new AgentError.

        Args:
            message: Human-readable error message
            code: Error code from the ErrorCode enum
            details: Additional error details for debugging or logging
        """
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.code.name}] {self.message}"


class ConfigError(AgentError):
    """Exception raised for configuration-related errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INVALID_CONFIG,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, details)


class APIError(AgentError):
    """Exception raised for API-related errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.API_RESPONSE_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, details)


class FileOperationError(AgentError):
    """Exception raised for file operation errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.FILE_NOT_FOUND,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, details)


class ToolError(AgentError):
    """Exception raised for tool-related errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.TOOL_EXECUTION_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, details)


class ConversationError(AgentError):
    """Exception raised for conversation-related errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CONVERSATION_NOT_FOUND,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, details)


class StimulusError(AgentError):
    """Exception raised for stimulus-related errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.STIMULUS_INVALID,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, details)


@contextmanager
def error_context(
    component_name: str,
    operation: str = None,
    error_class: Type[AgentError] = AgentError,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    logger: logging.Logger = None
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
        error_code: Error code to use for non-AgentError exceptions
        logger: Logger to use (if None, creates a new one)
        
    Yields:
        Control to the wrapped code block
        
    Raises:
        AgentError: With appropriate error information
    """
    # Set up logger if not provided
    if logger is None:
        logger = logging.getLogger(f"error.{component_name}")
        
    try:
        yield
    except Exception as e:
        # If it's already an AgentError, just log and re-raise
        if isinstance(e, AgentError):
            logger.error(f"{component_name} error: {e}")
            raise
            
        # Generate error message
        error_msg = f"Error in {component_name}"
        if operation:
            error_msg += f" during {operation}"
            
        # Log and wrap other exceptions
        logger.error(f"{error_msg}: {str(e)}")
        
        raise error_class(
            f"{error_msg}: {str(e)}",
            error_code,
            {"original_error": str(e)}
        )


def handle_error(error: Exception) -> str:
    """
    Utility function for standardized error handling.

    Converts exceptions to user-friendly error messages and performs any
    necessary logging or cleanup.

    Args:
        error: The exception to handle

    Returns:
        A user-friendly error message
    """
    if isinstance(error, AgentError):
        # Log the error with its code and details
        code_name = error.code.name if hasattr(error, 'code') else "UNKNOWN"
        # In a real system, you would log this properly
        print(f"ERROR [{code_name}]: {error.message}")

        # Return a user-friendly message
        return f"Error: {error.message}"
    else:
        # Handle unexpected exceptions
        print(f"UNEXPECTED ERROR: {str(error)}")
        return f"An unexpected error occurred: {str(error)}"
