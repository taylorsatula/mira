"""
Custom exception hierarchy for the AI agent system.

This module defines standardized error codes, messages, and categorization
for different types of errors that can occur within the system.
"""
from contextlib import contextmanager
from enum import Enum
import logging
from typing import Optional, Dict, Any, Type, Callable, Union, List
import json
import uuid
import datetime

# Import dedicated error loggers
from utils.error_logging import system_error_logger, tool_error_logger


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
    - 7xx: Network errors
    - 8xx: Data validation errors
    - 9xx: Uncategorized/system errors
    """
    # Configuration errors (1xx)
    CONFIG_NOT_FOUND = 101
    INVALID_CONFIG = 102
    MISSING_ENV_VAR = 103
    CONFIG_PARSE_ERROR = 104
    CONFIG_VALIDATION_ERROR = 105

    # API errors (2xx)
    API_CONNECTION_ERROR = 201
    API_AUTHENTICATION_ERROR = 202
    API_RATE_LIMIT_ERROR = 203
    API_RESPONSE_ERROR = 204
    API_TIMEOUT_ERROR = 205
    API_QUOTA_EXCEEDED = 206
    API_INVALID_REQUEST = 207
    API_SERVER_ERROR = 208
    API_SERVICE_UNAVAILABLE = 209

    # File operation errors (3xx)
    FILE_NOT_FOUND = 301
    FILE_PERMISSION_ERROR = 302
    FILE_READ_ERROR = 303
    FILE_WRITE_ERROR = 304
    INVALID_JSON = 305
    FILE_TOO_LARGE = 306
    FILE_LOCKED = 307
    DIRECTORY_NOT_FOUND = 308

    # Tool errors (4xx)
    TOOL_NOT_FOUND = 401
    TOOL_EXECUTION_ERROR = 402
    TOOL_INVALID_INPUT = 403
    TOOL_INVALID_OUTPUT = 404
    TOOL_INITIALIZATION_ERROR = 405
    TOOL_REGISTRATION_ERROR = 406
    TOOL_DUPLICATE_NAME = 407
    TOOL_NOT_ENABLED = 408
    TOOL_INVALID_PARAMETERS = 409
    TOOL_CIRCULAR_DEPENDENCY = 410
    TOOL_UNAVAILABLE = 411
    TOOL_AMBIGUOUS_INPUT = 412
    TOOL_DEPENDENCY_MISSING = 413
    TOOL_OPERATION_NOT_SUPPORTED = 414
    TOOL_RESOURCE_NOT_FOUND = 415
    TOOL_PERMISSION_DENIED = 416

    # Conversation errors (5xx)
    CONVERSATION_NOT_FOUND = 501
    CONTEXT_OVERFLOW = 502
    INVALID_INPUT = 503
    CONVERSATION_EXPIRED = 504
    CONVERSATION_LOCKED = 505

    # Stimulus errors (6xx)
    STIMULUS_INVALID = 601
    STIMULUS_PROCESSING_ERROR = 602
    STIMULUS_HANDLER_NOT_FOUND = 603
    STIMULUS_TYPE_INVALID = 604
    STIMULUS_TIMEOUT = 605

    # Network errors (7xx)
    NETWORK_UNREACHABLE = 701
    DNS_RESOLUTION_ERROR = 702
    SSL_CERTIFICATE_ERROR = 703
    PROXY_CONNECTION_ERROR = 704
    CONNECTION_REFUSED = 705
    CONNECTION_TIMEOUT = 706

    # Data validation errors (8xx)
    PARAMETER_MISSING = 801
    PARAMETER_TYPE_ERROR = 802
    PARAMETER_RANGE_ERROR = 803
    PARAMETER_FORMAT_ERROR = 804
    DATA_INTEGRITY_ERROR = 805
    SCHEMA_VALIDATION_ERROR = 806

    # Memory errors (85x)
    MEMORY_ERROR = 851
    MEMORY_BLOCK_NOT_FOUND = 852
    MEMORY_LIMIT_EXCEEDED = 853
    MEMORY_CORRUPTION = 854
    MEMORY_VERSION_CONFLICT = 855
    MEMORY_EMBEDDING_ERROR = 856
    MEMORY_SEARCH_ERROR = 857
    MEMORY_CONSOLIDATION_ERROR = 858

    # Uncategorized/system errors (9xx)
    UNKNOWN_ERROR = 901
    NOT_IMPLEMENTED = 902
    TEMPORARY_FAILURE = 903
    SYSTEM_OVERLOAD = 904
    MAINTENANCE_MODE = 905
    INITIALIZATION_FAILED = 906


class AgentError(Exception):
    """
    Base exception class for all agent-related errors.

    All other custom exceptions inherit from this class, allowing for
    standardized error handling throughout the system.
    """

    def __init__(
        self,
        message: str,
        code: Union[ErrorCode, str] = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new AgentError.

        Args:
            message: Human-readable error message
            code: Error code from the ErrorCode enum or string-based tool error code
            details: Additional error details for debugging or logging
        """
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        code_name = self.code.name if isinstance(self.code, ErrorCode) else self.code
        return f"[{code_name}] {self.message}"


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
    """
    Exception raised for tool-related errors with recovery guidance.
    
    Supports four recovery outcomes:
    1. retry_with: Try again with corrected parameters
    2. try_instead: Try an alternative approach
    3. needs_user_input: Need clarification from user
    4. permanent_failure: Unrecoverable, stop trying
    """

    def __init__(
        self,
        message: str,
        code: Union[ErrorCode, str] = ErrorCode.TOOL_EXECUTION_ERROR,
        details: Optional[Dict[str, Any]] = None,
        retry_with: Optional[Dict[str, Any]] = None,
        try_instead: Optional[str] = None,
        needs_user_input: bool = False,
        permanent_failure: bool = False
    ):
        """
        Initialize a ToolError with recovery guidance.
        
        Args:
            message: Clear error message explaining what went wrong
            code: Error code (ErrorCode enum or string for tool-specific codes)
            details: Additional error context for debugging
            retry_with: Corrected parameters to retry with
            try_instead: Alternative approach to try
            needs_user_input: Whether user clarification is needed
            permanent_failure: Whether this is unrecoverable
        """
        super().__init__(message, code, details)
        self.retry_with = retry_with
        self.try_instead = try_instead
        self.needs_user_input = needs_user_input
        self.permanent_failure = permanent_failure
        
        # Validate that only one recovery option is specified
        recovery_options = [
            retry_with is not None,
            try_instead is not None,
            needs_user_input,
            permanent_failure
        ]
        if sum(recovery_options) > 1:
            raise ValueError("Only one recovery option should be specified")
    
    def get_recovery_strategy(self) -> Dict[str, Any]:
        """Get the recovery strategy as a dictionary."""
        return {
            "retry_with": self.retry_with,
            "try_instead": self.try_instead,
            "needs_user_input": self.needs_user_input,
            "permanent_failure": self.permanent_failure
        }


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


def _add_error_to_working_memory(working_memory, error: Union[AgentError, ToolError], error_id: str):
    """
    Add an error to working memory for MIRA to analyze.
    
    Args:
        working_memory: The working memory instance
        error: The error that occurred
        error_id: Unique identifier for this error occurrence
    """
    error_info = {
        "error_id": error_id,
        "code": error.code.name if isinstance(error.code, ErrorCode) else error.code,
        "message": error.message,
        "details": error.details,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    
    # Add recovery strategy if it's a ToolError
    if isinstance(error, ToolError):
        error_info["recovery_strategy"] = error.get_recovery_strategy()
    
    # Add to working memory
    working_memory.add(
        content=(
            "# Error Analysis Request\n"
            "An error just occurred. Analyze this error based on the recent conversation "
            "and explain in ONE SENTENCE what the user was trying to do when it occurred. "
            f'Include your analysis inside a <error_analysis error_id="{error_id}">'
            "YOUR ANALYSIS HERE</error_analysis> tag within your thought process.\n\n"
            f"Error details:\n{json.dumps(error_info, indent=2)}"
        ),
        category="error_for_analysis"
    )


@contextmanager
def error_context(
    component_name: str,
    operation: Optional[str] = None,
    error_class: Type[AgentError] = AgentError,
    error_code: Union[ErrorCode, str] = ErrorCode.UNKNOWN_ERROR,
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
        error_code: Error code to use for non-AgentError exceptions
        working_memory: Optional working memory instance for error analysis
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
        # Generate a unique error ID for tracking
        error_id = str(uuid.uuid4())
        
        # If it's already an AgentError, log and add to working memory
        if isinstance(e, AgentError):
            # Log to appropriate error logger
            if isinstance(e, ToolError):
                tool_error_logger.error(f"[{error_id}] {component_name} - {e}")
            else:
                system_error_logger.error(f"[{error_id}] {component_name} - {e}")
            
            # Add to working memory for analysis if available
            if working_memory:
                _add_error_to_working_memory(working_memory, e, error_id)
                
            raise
            
        # Generate error message
        error_msg = f"Error in {component_name}"
        if operation:
            error_msg += f" during {operation}"
            
        # Log and wrap other exceptions - sanitize for sensitive data
        error_string = str(e)
        # Redact potentially sensitive information 
        if any(sensitive in error_string.lower() for sensitive in ["token", "bearer", "key", "auth", "password", "secret"]):
            error_string = "[REDACTED SENSITIVE INFORMATION]"
            
        # Create error with appropriate code
        if isinstance(error_code, str):
            # String-based error code (tool-specific)
            wrapped_error = error_class(
                f"{error_msg}: {error_string}",
                error_code,
                {"original_error": error_string, "error_id": error_id}
            )
        else:
            # ErrorCode enum
            wrapped_error = error_class(
                f"{error_msg}: {error_string}",
                error_code,
                {"original_error": error_string, "error_id": error_id}
            )
        
        # Log to appropriate error logger
        if error_class == ToolError or isinstance(wrapped_error, ToolError):
            tool_error_logger.error(f"[{error_id}] {error_msg}: {error_string}")
        else:
            system_error_logger.error(f"[{error_id}] {error_msg}: {error_string}")
        
        # Add to working memory if available
        if working_memory and isinstance(wrapped_error, (AgentError, ToolError)):
            _add_error_to_working_memory(working_memory, wrapped_error, error_id)
        
        raise wrapped_error


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
