"""
Base API infrastructure for MIRA endpoints.

Provides consistent response patterns, error handling, and middleware.
"""
import logging
from dataclasses import dataclass, replace
from typing import Any, NotRequired, TypedDict
from uuid import uuid4

from utils.timezone_utils import utc_now, format_utc_iso

logger = logging.getLogger(__name__)


class ErrorDetail(TypedDict):
    """Structured error information."""
    code: str
    message: str
    details: dict[str, Any]
    http_status: NotRequired[int]


class ResponseMeta(TypedDict, total=False):
    """Response metadata — all fields optional."""
    timestamp: str
    request_id: str
    http_status: int


@dataclass(frozen=True)
class SuccessResponse:
    """Successful API response."""
    data: dict[str, Any]
    meta: ResponseMeta

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"success": True, "data": self.data}
        if self.meta:
            result["meta"] = dict(self.meta)
        return result


@dataclass(frozen=True)
class ErrorResponse:
    """Error API response."""
    error: ErrorDetail
    meta: ResponseMeta

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"success": False, "error": dict(self.error)}
        if self.meta:
            result["meta"] = dict(self.meta)
        return result


APIResponse = SuccessResponse | ErrorResponse


class APIError(Exception):
    """Base API error with structured details."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details if details is not None else {}


class ValidationError(APIError):
    """Validation error for invalid input."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("VALIDATION_ERROR", message, details)


class NotFoundError(APIError):
    """Resource not found error."""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            "NOT_FOUND",
            f"{resource} not found: {identifier}",
            {"resource": resource, "identifier": identifier}
        )


class ServiceUnavailableError(APIError):
    """Service unavailable error."""

    def __init__(self, service: str, details: dict[str, Any] | None = None):
        super().__init__(
            "SERVICE_UNAVAILABLE",
            f"Service unavailable: {service}",
            details
        )


def create_success_response(
    data: dict[str, Any],
    meta: dict[str, Any] | None = None
) -> SuccessResponse:
    """Create a successful API response."""
    return SuccessResponse(data=data, meta=meta or ResponseMeta())


def create_error_response(
    error: APIError | Exception,
    request_id: str | None = None
) -> ErrorResponse:
    """Create an error API response."""
    if isinstance(error, APIError):
        error_detail: ErrorDetail = {
            "code": error.code,
            "message": error.message,
            "details": error.details
        }
    else:
        error_detail = {
            "code": "INTERNAL_ERROR",
            "message": str(error),
            "details": {}
        }

    meta: ResponseMeta = {
        "timestamp": format_utc_iso(utc_now())
    }

    if request_id:
        meta["request_id"] = request_id

    return ErrorResponse(error=error_detail, meta=meta)


def generate_request_id() -> str:
    """Generate unique request ID."""
    return f"req_{uuid4().hex[:12]}"


class BaseHandler:
    """Base handler for API endpoints."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_params(self, **params) -> dict[str, Any]:
        """Validate input parameters. Override in subclasses."""
        return params

    def handle_request(self, **params) -> APIResponse:
        """Handle API request with consistent error handling."""
        request_id = generate_request_id()

        try:
            validated_params = self.validate_params(**params)
            result = self.process_request(**validated_params)

            if isinstance(result, (SuccessResponse, ErrorResponse)):
                return result
            else:
                return create_success_response(result)

        except APIError as e:
            self.logger.warning(f"API error in {self.__class__.__name__}: {e.message}")
            return create_error_response(e, request_id)
        except Exception as e:
            self.logger.error(f"Unexpected error in {self.__class__.__name__}: {e}", exc_info=True)
            return create_error_response(e, request_id)

    def process_request(self, **params) -> dict[str, Any] | SuccessResponse | ErrorResponse:
        """Process the actual request. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement process_request")


def add_request_meta(response: APIResponse, **meta_data) -> APIResponse:
    """Add metadata to existing response."""
    new_meta = {**response.meta, **meta_data}
    return replace(response, meta=new_meta)
