"""
FastAPI endpoints for lean magic-link authentication.
"""

import logging
from datetime import datetime
from dataclasses import replace
from typing import Any, Optional
from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, field_validator

from .service import AuthService, get_auth_service
from .exceptions import AuthError
from .mode import auth_mode
from .types import UserProfile, SessionData, APITokenContext
from .webauthn_service import WebAuthnService
from utils.user_context import set_current_user_id, set_current_user_data
from utils.timezone_utils import format_utc_iso, utc_now, validate_timezone
from utils.profile_validation import validate_profile_name
from cns.api.base import (
    SuccessResponse,
    ErrorResponse,
    APIError,
    create_success_response,
    create_error_response,
    generate_request_id,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["auth"])

# Security scheme for bearer tokens (auto_error=False so we can handle missing auth properly)
security = HTTPBearer(auto_error=False)


UNSAFE_HTTP_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
CSRF_EXEMPT_PATH_SUFFIXES = {
    "/auth/csrf",
}



# Request/Response models
class SignupRequest(BaseModel):
    """User signup request."""
    email: EmailStr = Field(..., description="User email address")
    first_name: str = Field(..., min_length=1, max_length=100, description="User's first name")
    last_name: str = Field(..., min_length=1, max_length=100, description="User's last name")
    timezone: str = Field(..., min_length=1, max_length=100, description="User's timezone (e.g., America/New_York)")
    current_focus: str = Field(..., min_length=1, max_length=1000, description="User's current focus or goal")

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return validate_profile_name(value, info.field_name)

    @field_validator("timezone")
    @classmethod
    def validate_tz(cls, value: str) -> str:
        return validate_timezone(value)


class MagicLinkRequest(BaseModel):
    """Magic link request."""
    email: EmailStr = Field(..., description="User email address")


class MagicLinkVerifyRequest(BaseModel):
    """Magic link verification request."""
    token: str = Field(..., min_length=32, max_length=128, description="Magic link token")


# Auth-specific error types
class AuthenticationError(APIError):
    """Authentication-specific error."""

    def __init__(self, message: str, code: str = "AUTH_ERROR", details: dict[str, Any] | None = None):
        super().__init__(code, message, details)


class RateLimitError(APIError):
    """Rate limit error."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("RATE_LIMIT_EXCEEDED", message, details)


# Helper functions for auth responses
def create_auth_success_response(
    data: dict[str, Any],
    http_status: int = 200,
    request_id: str | None = None
) -> SuccessResponse:
    """Create auth success response with HTTP status in meta."""
    if request_id is None:
        request_id = generate_request_id()

    return create_success_response(
        data=data,
        meta={
            "request_id": request_id,
            "timestamp": format_utc_iso(utc_now()),
            "http_status": http_status
        }
    )


def create_auth_error_response(
    error: Exception,
    http_status: int,
    request_id: str | None = None
) -> ErrorResponse:
    """Create auth error response with HTTP status in both error and meta."""
    if request_id is None:
        request_id = generate_request_id()

    # Convert AuthError to appropriate APIError
    if isinstance(error, AuthError):
        if error.code == "rate_limit_exceeded":
            api_error: APIError | Exception = RateLimitError(error.message, error.details)
        else:
            api_error = AuthenticationError(error.message, error.code, error.details)
    else:
        api_error = error

    # Create the base error response (frozen)
    response = create_error_response(api_error, request_id)

    # Add HTTP status via replace (frozen dataclass)
    return replace(
        response,
        error={**response.error, "http_status": http_status},
        meta={**response.meta, "http_status": http_status},
    )


# Dependencies
def _requires_cookie_csrf(request: Request, token_source: Optional[str]) -> bool:
    """Return True when a cookie-authenticated write must provide a CSRF token."""
    path = request.url.path
    return (
        token_source == "cookie"
        and request.method in UNSAFE_HTTP_METHODS
        and not any(path.endswith(suffix) for suffix in CSRF_EXEMPT_PATH_SUFFIXES)
    )


def _validate_cookie_csrf(request: Request, session_token: str, auth_service: AuthService) -> None:
    """Validate the CSRF token for a cookie-authenticated write."""
    csrf_token = request.headers.get("x-csrf-token")
    if not csrf_token:
        raise AuthError("CSRF_REQUIRED", "Missing CSRF token for this request")
    if not auth_service.validate_csrf_token(session_token, csrf_token):
        raise AuthError("CSRF_INVALID", "Invalid CSRF token")


def _auth_http_exception(error: AuthError, status_code: int) -> HTTPException:
    """Build the standard auth HTTPException envelope outside get_current_user."""
    error_response = {
        "success": False,
        "error": {
            "code": error.code,
            "message": error.message,
            "http_status": status_code
        },
        "meta": {
            "timestamp": format_utc_iso(utc_now()),
            "http_status": status_code
        }
    }
    return HTTPException(status_code=status_code, detail=error_response)


# Dependency for getting current user from session token
#
# KNOWN, DELIBERATELY UNFIXED (census-20260930b, kata ticket hs02): this async
# dependency — and the async auth handlers below — run synchronous Valkey/Postgres
# IO directly on the event loop on every authenticated request. In the default
# single-worker process one infrastructure stall (Valkey/Postgres) freezes the
# whole server, streaming included, until the socket/pool bound. Not observed in
# production (Taylor, 2026-09-30); left as-is. If a whole-server freeze under an
# infra stall is ever investigated, reference kata ticket hs02 before redesigning —
# note the trap: converting to plain `def` runs the body in a worker thread whose
# ContextVar mutations (set_current_user_id below) do NOT propagate back to the
# request context, silently dropping user context.
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> SessionData | APITokenContext:
    """
    Get current authenticated user from session token or API token.

    Browser clients present a Valkey session cookie (optionally paired with
    a CSRF token for writes); server-to-server clients present an issued API
    token in the Authorization header. There is no static process-global key:
    under `single` mode the local account obtains its session through
    `GET /v0/auth/local/session` exactly as multi-mode accounts do.

    Returns SessionData for session-based auth, APITokenContext for API tokens.
    Raises HTTPException with standard error format on failure.
    """
    try:
        token: Optional[str] = None
        token_source: Optional[str] = None

        # Prefer Authorization header when provided
        if credentials and credentials.credentials:
            token = credentials.credentials
            token_source = "header"
        else:
            # Fall back to cookie-based session for browser requests
            token = request.cookies.get("session")
            if token:
                token_source = "cookie"

        if not token:
            raise AuthError("UNAUTHORIZED", "Authentication required")

        # For header-based auth, do NOT extend activity/TTL (good for API tokens)
        extend_activity = token_source != "header"

        if _requires_cookie_csrf(request, token_source):
            _validate_cookie_csrf(request, token, auth_service)

        # Try session validation first
        session_data: SessionData | APITokenContext | None = auth_service.validate_session(token, extend_activity)

        # If session validation fails and token came from header, try API token
        if not session_data and token_source == "header":
            api_token_data = auth_service.validate_api_token(token)
            if api_token_data:
                # Build typed context from API token
                session_data = APITokenContext(
                    user_id=api_token_data["user_id"],
                    token_type="api_token",
                    token_id=api_token_data["id"],
                    subject_kind=api_token_data["subject_kind"],
                    demo_expires_at=(
                        api_token_data["demo_expires_at"].isoformat()
                        if api_token_data["demo_expires_at"] is not None
                        else None
                    ),
                )

        if not session_data:
            raise AuthError("INVALID_TOKEN", "Invalid or expired token")

        # Set user context for downstream services
        set_current_user_id(session_data.user_id)
        set_current_user_data(session_data.model_dump())

        return session_data

    except AuthError as e:
        # Convert AuthError to HTTPException with our standard format
        status_code = 401
        if e.code in {"CSRF_REQUIRED", "CSRF_INVALID"}:
            status_code = 403
        error_response: dict[str, bool | dict[str, str | int]] = {
            "success": False,
            "error": {
                "code": e.code,
                "message": e.message,
                "http_status": status_code
            },
            "meta": {
                "timestamp": format_utc_iso(utc_now()),
                "http_status": status_code
            }
        }
        raise HTTPException(status_code=status_code, detail=error_response)


async def get_current_session_user(
    current_user: SessionData | APITokenContext = Depends(get_current_user)
) -> SessionData:
    """Require a browser/session token rather than an API token."""
    if isinstance(current_user, APITokenContext):
        raise _auth_http_exception(
            AuthError("SESSION_REQUIRED", "Session authentication required"),
            403
        )
    return current_user



# === API Token Endpoints ===
class APITokenCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Friendly name for the token")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Days until expiration, or null for no expiration")


class APITokenListItem(BaseModel):
    id: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None  # None = never expires


@router.post("/api-tokens")
async def create_api_token(
    request: APITokenCreateRequest,
    response: Response,
    current_user: SessionData = Depends(get_current_session_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Create a long-lived API token. Returns the token once - copy it immediately."""
    request_id = generate_request_id()
    try:
        token = auth_service.create_api_token(
            request.name,
            request.expires_in_days
        )
        api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
            data={
                "token": token,
                "note": "Store this token securely. It will be shown only once."
            },
            http_status=201,
            request_id=request_id
        )
        response.status_code = 201
        return api_response.to_dict()
    except AuthError as e:
        http_status = 400
        api_response = create_auth_error_response(e, http_status, request_id)
        response.status_code = http_status
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] API token creation error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.get("/api-tokens")
async def list_api_tokens(
    current_user: SessionData = Depends(get_current_session_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """List API tokens (metadata only)."""
    request_id = generate_request_id()
    try:
        items = auth_service.list_api_tokens()
        # Validate shape via Pydantic model for consistency
        validated: list[APITokenListItem] = [APITokenListItem(**i) for i in items]
        api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
            data={"tokens": [v.model_dump() for v in validated]},
            request_id=request_id
        )
        return api_response.to_dict()
    except AuthError as e:
        # Handle auth-specific errors with appropriate status codes
        http_status = 400
        api_response = create_auth_error_response(e, http_status, request_id)
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] API token list error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        return api_response.to_dict()


@router.delete("/api-tokens/{token_id}")
async def revoke_api_token(
    token_id: str,
    response: Response,
    current_user: SessionData = Depends(get_current_session_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Revoke an API token by its ID."""
    request_id = generate_request_id()
    try:
        success = auth_service.revoke_api_token(token_id)
        if success:
            api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
                data={"message": "Token revoked"},
                request_id=request_id
            )
            return api_response.to_dict()
        else:
            api_response = create_auth_error_response(
                AuthenticationError("Token not found", "NOT_FOUND"),
                http_status=404,
                request_id=request_id
            )
            response.status_code = 404
            return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] API token revoke error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


# Endpoints
@router.get("/local/session", include_in_schema=False)
def create_local_session(
    auth_service: AuthService = Depends(get_auth_service),
) -> RedirectResponse:
    """Bootstrap the single-mode local account and issue a browser session.

    The only identity surface under `single` mode: first visit auto-provisions
    `user@localhost` (adopting that row verbatim when an upgraded install
    already holds it), mints a session cookie, and redirects back to `/chat`.
    Refused elsewhere with the same 404 as `/signup`: one unauthenticated GET
    on a multi install would create a stray row alongside public signups.
    """
    if auth_mode() != "single":
        raise HTTPException(status_code=404, detail="Not found")

    _, session_token = auth_service.create_local_session()
    cookie_settings = auth_service.get_cookie_settings()
    response = RedirectResponse("/chat", status_code=303)
    response.set_cookie(
        key="session",
        value=session_token,
        max_age=cookie_settings.max_age,
        httponly=cookie_settings.httponly,
        secure=cookie_settings.secure,
        samesite=cookie_settings.samesite,
        path="/",
    )
    return response


@router.post("/signup")
def signup(
    request: SignupRequest,
    response: Response,
    http_request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Create a new user account.

    Refused in `single` mode: identity there is fixed to the local account,
    and a stray second row would split data across identities. The 404 (not
    403) matches `create_local_session`: installs without public signup must
    not advertise the surface to anonymous probes.
    """
    if auth_mode() == "single":
        raise HTTPException(status_code=404, detail="Not found")
    request_id = generate_request_id()
    try:
        # Get client info for security logging
        ip_address = http_request.client.host if http_request.client else ""
        user_agent = http_request.headers.get("user-agent", "")

        user_id = auth_service.create_user(
            request.email,
            request.first_name,
            request.last_name,
            request.timezone,
            request.current_focus,
            ip_address,
            user_agent
        )
        api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
            data={
                "user_id": user_id,
                "email": request.email,
                "message": "Account created successfully. Check your email for a magic link."
            },
            http_status=201,
            request_id=request_id
        )
        response.status_code = 201
        return api_response.to_dict()
    except AuthError as e:
        # A committed account with a failed link delivery is not a creation
        # failure: answer with a distinct delivery-failed outcome carrying a
        # resend cue, never the bare creation-error envelope.
        if e.code == "magic_link_delivery_failed":
            api_response = create_auth_success_response(
                data={
                    "email": request.email,
                    "message": (
                        "Account created, but the sign-in email could not be "
                        "delivered. Your account exists — request a magic "
                        "link to sign in."
                    )
                },
                http_status=201,
                request_id=request_id
            )
            response.status_code = 201
            return api_response.to_dict()
        # Determine HTTP status code based on error
        http_status = 400
        if e.code == "user_already_exists":
            http_status = 400

        api_response = create_auth_error_response(e, http_status, request_id)
        response.status_code = http_status
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] Signup error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.post("/magic-link")
def request_magic_link(
    request: MagicLinkRequest,
    req: Request,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Request a magic link for passwordless authentication.

    Refused in `single` mode with the same 404 as `/signup`: no account can
    exist beyond the local account, and no mailer is configured, so the
    lifecycle has nowhere to deliver.
    """
    if auth_mode() == "single":
        raise HTTPException(status_code=404, detail="Not found")
    request_id = generate_request_id()
    try:
        client_ip = req.client.host if req.client else "unknown"
        user_agent = req.headers.get("user-agent", "")[:500]

        auth_service.request_magic_link(
            request.email,
            client_ip,
            user_agent
        )

        # Generic success message that doesn't reveal if user exists
        api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
            data={"message": "If that email is registered, a magic link has been sent. Please check your inbox."},
            request_id=request_id
        )
        return api_response.to_dict()
    except AuthError as e:
        # Note: "user_not_found" is no longer raised (prevents user enumeration)
        if e.code == "rate_limit_exceeded":
            http_status = 429
        else:
            http_status = 400

        api_response = create_auth_error_response(e, http_status, request_id)
        response.status_code = http_status
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] Magic link request error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.post("/verify")
def verify_magic_link(
    request: MagicLinkVerifyRequest,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Verify magic link and create session.

    Refused in `single` mode with the same 404 as `/signup`: it is the
    completion half of the gated `/magic-link` flow; magic links do not
    exist outside `multi` mode.
    """
    if auth_mode() == "single":
        raise HTTPException(status_code=404, detail="Not found")
    request_id = generate_request_id()
    try:
        user_data, session_token = auth_service.verify_magic_link(
            request.token
        )

        # Create response with session cookie
        # Use model_dump(mode='json') to convert datetime to strings
        api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
            data={
                "user": user_data.model_dump(mode='json'),
                "session_token": session_token
            },
            request_id=request_id
        )

        # Set session cookie
        cookie_settings = auth_service.get_cookie_settings()
        response.set_cookie(
            key="session",
            value=session_token,
            max_age=cookie_settings.max_age,
            httponly=cookie_settings.httponly,
            secure=cookie_settings.secure,
            samesite=cookie_settings.samesite,
            path="/"  # Ensure cookie is available for all paths
        )

        return api_response.to_dict()
    except AuthError as e:
        if e.code == "expired_token":
            http_status = 410
        elif e.code == "invalid_token":
            http_status = 400
        else:
            http_status = 400

        api_response = create_auth_error_response(e, http_status, request_id)
        response.status_code = http_status
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] Magic link verification error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Logout current session."""
    request_id = generate_request_id()
    try:
        header_token = credentials.credentials if credentials and credentials.credentials else None
        cookie_token = request.cookies.get("session")

        if not header_token and not cookie_token:
            raise AuthError("UNAUTHORIZED", "Authentication required")

        success = False
        if header_token:
            success = auth_service.logout(header_token)

        if not success and cookie_token:
            _validate_cookie_csrf(request, cookie_token, auth_service)
            success = auth_service.logout(cookie_token)

        if success:
            api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
                data={"message": "Logged out successfully"},
                request_id=request_id
            )
            response.status_code = 200
        else:
            api_response = create_auth_error_response(
                AuthenticationError("Session not found or already expired", "SESSION_NOT_FOUND"),
                http_status=404,
                request_id=request_id
            )
            response.status_code = 404

        # Clear session cookie. Attributes come from the same CookieSettings
        # source as every set path (`get_cookie_settings`), so the deletion
        # cannot drift from the set path: under single-mode plain HTTP a
        # deletion carrying `Secure` is dropped by the browser (RFC 6265) and
        # the stale revoked cookie would survive logout.
        cookie_settings = auth_service.get_cookie_settings()
        response.delete_cookie(
            key="session",
            path="/",
            httponly=cookie_settings.httponly,
            secure=cookie_settings.secure,
            samesite=cookie_settings.samesite
        )
        return api_response.to_dict()

    except AuthError as e:
        if e.code == "UNAUTHORIZED":
            http_status = 401
        elif e.code in {"CSRF_REQUIRED", "CSRF_INVALID"}:
            http_status = 403
        else:
            http_status = 400
        api_response = create_auth_error_response(e, http_status, request_id)
        response.status_code = http_status
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] Logout error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.post("/logout-all")
async def logout_all_devices(
    response: Response,
    current_user: SessionData = Depends(get_current_session_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Logout all devices for a user."""
    request_id = generate_request_id()
    try:
        revoked_count = auth_service.logout_all_devices()

        api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
            data={
                "message": f"Logged out from {revoked_count} devices",
                "revoked_sessions": revoked_count
            },
            request_id=request_id
        )
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] Logout all error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.post("/logout-others")
async def logout_other_devices(
    request: Request,
    response: Response,
    current_user: SessionData = Depends(get_current_session_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """Revoke all of the user's browser sessions except this cookie session."""
    request_id = generate_request_id()
    try:
        session_token = request.cookies.get("session")
        if not session_token:
            raise AuthError("SESSION_REQUIRED", "Cookie session authentication required")
        revoked_count = auth_service.logout_other_devices(session_token)
        return create_auth_success_response(
            data={
                "message": f"Logged out from {revoked_count} other devices",
                "revoked_sessions": revoked_count,
            },
            request_id=request_id,
        ).to_dict()
    except AuthError as exc:
        status_code = 403 if exc.code == "SESSION_REQUIRED" else 401
        response.status_code = status_code
        return create_auth_error_response(exc, status_code, request_id).to_dict()
    except Exception as exc:
        logger.error("[%s] Logout other devices error: %s", request_id, exc, exc_info=True)
        response.status_code = 500
        return create_auth_error_response(exc, 500, request_id).to_dict()


@router.get("/session")
async def get_session(current_user: SessionData | APITokenContext = Depends(get_current_user)):
    """Get current session information."""
    request_id = generate_request_id()
    api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
        data=current_user.model_dump(mode="json"),
        request_id=request_id
    )
    return api_response.to_dict()


@router.post("/csrf")
async def get_csrf_token(
    request: Request,
    response: Response,
    current_user: SessionData = Depends(get_current_session_user),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Generate CSRF token for session."""
    request_id = generate_request_id()
    try:
        # Determine session token source (header takes precedence if present)
        session_token = None
        if credentials and getattr(credentials, 'credentials', None):
            session_token = credentials.credentials
        else:
            session_token = request.cookies.get("session")
        if not session_token:
            raise AuthError("UNAUTHORIZED", "Authentication required")

        # User is already validated by get_current_user; generate CSRF for this session
        csrf_token = auth_service.generate_csrf_token(session_token)
        api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
            data={"csrf_token": csrf_token},
            request_id=request_id
        )
        return api_response.to_dict()
    except AuthError as e:
        api_response = create_auth_error_response(e, 400, request_id)
        response.status_code = 400
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] CSRF token generation error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


# WebAuthn endpoints

# Request/Response models for WebAuthn
class WebAuthnLoginBeginRequest(BaseModel):
    """WebAuthn login begin request.

    Email is optional: omitting it starts a discoverable ceremony where the
    browser offers every passkey registered for this RP on the device.
    """
    email: Optional[EmailStr] = Field(default=None, description="User email address; omit for a discoverable login")


class WebAuthnLoginCompleteRequest(BaseModel):
    """WebAuthn login complete request.

    Exactly one of ``email`` (email-keyed ceremony) or ``challenge_id``
    (discoverable ceremony) must be provided.
    """
    email: Optional[EmailStr] = Field(default=None, description="User email address")
    challenge_id: Optional[str] = Field(default=None, description="Challenge ID returned by a discoverable begin")
    credential: dict[str, Any] = Field(..., description="WebAuthn credential response")


class WebAuthnRegisterCompleteRequest(BaseModel):
    """WebAuthn register complete request."""
    credential: dict[str, Any] = Field(..., description="WebAuthn credential response")


# Helper function for WebAuthn service
def get_webauthn_service() -> WebAuthnService:
    """Get WebAuthn service instance."""
    return WebAuthnService()


@router.post("/webauthn/register/begin")
async def webauthn_register_begin(
    response: Response,
    current_user: SessionData = Depends(get_current_session_user),
    webauthn_service: WebAuthnService = Depends(get_webauthn_service)
):
    """Begin WebAuthn registration for authenticated user."""
    request_id = generate_request_id()
    try:
        options = webauthn_service.generate_registration_options(
            current_user.email
        )

        api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
            data=options,
            request_id=request_id
        )
        return api_response.to_dict()
    except AuthError as e:
        api_response = create_auth_error_response(e, 400, request_id)
        response.status_code = 400
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] WebAuthn registration begin error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.post("/webauthn/register/complete")
async def webauthn_register_complete(
    request: WebAuthnRegisterCompleteRequest,
    response: Response,
    current_user: SessionData = Depends(get_current_session_user),
    webauthn_service: WebAuthnService = Depends(get_webauthn_service)
):
    """Complete WebAuthn registration."""
    request_id = generate_request_id()
    try:
        result = webauthn_service.verify_registration(
            request.credential
        )

        api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
            data={
                "message": "Biometric authentication enabled successfully",
                "credential_id": result["credential_id"]
            },
            request_id=request_id
        )
        return api_response.to_dict()
    except AuthError as e:
        # `internal_error` is a genuine server fault (5xx); every other code
        # from this route (invalid_credential, invalid_challenge, ...) is a
        # client-input problem and stays 400.
        http_status = 500 if e.code == "internal_error" else 400
        api_response = create_auth_error_response(e, http_status, request_id)
        response.status_code = http_status
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] WebAuthn registration complete error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.post("/webauthn/login/begin")
def webauthn_login_begin(
    request: WebAuthnLoginBeginRequest,
    http_request: Request,
    response: Response,
    webauthn_service: WebAuthnService = Depends(get_webauthn_service),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Begin WebAuthn authentication."""
    request_id = generate_request_id()
    try:
        client_ip = http_request.client.host if http_request.client else "unknown"
        # Discoverable logins disclose no email up front, so rate-limit them
        # on a per-IP identifier instead.
        rate_key = request.email or f"discoverable:{client_ip}"
        allowed, retry_after = auth_service.rate_limiter.is_allowed(rate_key, client_ip)
        if not allowed:
            raise AuthError(
                "rate_limit_exceeded",
                f"Too many requests. Try again in {retry_after} seconds.",
                {"retry_after": retry_after}
            )

        if request.email:
            options = webauthn_service.generate_authentication_options(
                request.email
            )
        else:
            options = webauthn_service.generate_discoverable_authentication_options()

        api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
            data=options,
            request_id=request_id
        )
        return api_response.to_dict()
    except AuthError as e:
        if e.code == "rate_limit_exceeded":
            http_status = 429
            api_response = create_auth_error_response(e, http_status, request_id)
            response.status_code = http_status
            return api_response.to_dict()
        if e.code in {"user_not_found", "no_credentials"}:
            http_status = 400
            api_response = create_auth_error_response(
                AuthenticationError("Biometric login is unavailable for this email", "WEBAUTHN_UNAVAILABLE"),
                http_status,
                request_id
            )
            response.status_code = http_status
            return api_response.to_dict()
        else:
            http_status = 400

        api_response = create_auth_error_response(e, http_status, request_id)
        response.status_code = http_status
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] WebAuthn login begin error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.post("/webauthn/login/complete")
def webauthn_login_complete(
    request: WebAuthnLoginCompleteRequest,
    http_request: Request,
    response: Response,
    webauthn_service: WebAuthnService = Depends(get_webauthn_service),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Complete WebAuthn authentication and create session."""
    request_id = generate_request_id()
    try:
        if request.email and request.challenge_id:
            raise AuthError("invalid_request", "Provide exactly one of email or challenge_id")

        client_ip = http_request.client.host if http_request.client else "unknown"
        rate_key = request.email or f"discoverable:{client_ip}"
        allowed, retry_after = auth_service.rate_limiter.is_allowed(rate_key, client_ip)
        if not allowed:
            raise AuthError(
                "rate_limit_exceeded",
                f"Too many requests. Try again in {retry_after} seconds.",
                {"retry_after": retry_after}
            )

        # Verify WebAuthn authentication
        if request.email:
            result = webauthn_service.verify_authentication(
                request.email,
                request.credential
            )
        elif request.challenge_id is not None:
            result = webauthn_service.verify_discoverable_authentication(
                request.challenge_id,
                request.credential
            )
        else:
            raise AuthError("invalid_request", "Provide exactly one of email or challenge_id")

        if not result["verified"]:
            raise AuthError("authentication_failed", "Authentication failed")

        # Get user data
        user = auth_service.db.get_user_by_id(result["user_id"])
        if not user:
            raise AuthError("user_not_found", "User not found")

        # Create session
        user_profile = UserProfile(
            id=str(user.id),
            email=user.email,
            is_active=user.is_active,
            created_at=user.created_at,
            subject_kind=user.subject_kind,
            demo_expires_at=user.demo_expires_at,
        )

        session_token = auth_service.session_manager.create_session(user)

        # Create response
        api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
            data={
                "user": user_profile.model_dump(mode='json'),
                "session_token": session_token,
                "message": "Authentication successful"
            },
            request_id=request_id
        )

        # Set session cookie
        cookie_settings = auth_service.get_cookie_settings()
        response.set_cookie(
            key="session",
            value=session_token,
            max_age=cookie_settings.max_age,
            httponly=cookie_settings.httponly,
            secure=cookie_settings.secure,
            samesite=cookie_settings.samesite,
            path="/"  # Ensure cookie is available for all paths
        )

        return api_response.to_dict()
    except AuthError as e:
        if e.code == "rate_limit_exceeded":
            http_status = 429
        elif e.code == "internal_error":
            # Genuine server fault, not a client-input problem
            http_status = 500
        else:
            http_status = 400
        if e.code in {"user_not_found", "unknown_credential", "invalid_challenge", "authentication_failed"}:
            api_response = create_auth_error_response(
                AuthenticationError("Biometric authentication failed", "WEBAUTHN_AUTHENTICATION_FAILED"),
                http_status,
                request_id
            )
        else:
            api_response = create_auth_error_response(e, http_status, request_id)
        response.status_code = http_status
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] WebAuthn login complete error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.delete("/webauthn/credential/{credential_id}")
async def webauthn_remove_credential(
    credential_id: str,
    response: Response,
    current_user: SessionData = Depends(get_current_session_user),
    webauthn_service: WebAuthnService = Depends(get_webauthn_service)
):
    """Remove a WebAuthn credential."""
    request_id = generate_request_id()
    try:
        success = webauthn_service.remove_credential(
            credential_id
        )

        if success:
            api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
                data={"message": "Credential removed successfully"},
                request_id=request_id
            )
            return api_response.to_dict()
        else:
            api_response = create_auth_error_response(
                AuthenticationError("Failed to remove credential", "REMOVAL_FAILED"),
                http_status=400,
                request_id=request_id
            )
            response.status_code = 400
            return api_response.to_dict()
    except AuthError as e:
        api_response = create_auth_error_response(e, 400, request_id)
        response.status_code = 400
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] WebAuthn credential removal error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.get("/webauthn/credentials")
async def webauthn_list_credentials(
    current_user: SessionData = Depends(get_current_session_user),
    webauthn_service: WebAuthnService = Depends(get_webauthn_service)
):
    """List user's WebAuthn credentials."""
    request_id = generate_request_id()
    try:
        credentials = webauthn_service.list_credentials()

        api_response: SuccessResponse | ErrorResponse = create_auth_success_response(
            data={"credentials": credentials},
            request_id=request_id
        )
        return api_response.to_dict()
    except Exception as e:
        logger.error(f"[{request_id}] WebAuthn list credentials error: {e}", exc_info=True)
        api_response = create_auth_error_response(e, 500, request_id)
        return api_response.to_dict()


# Dependency for HTML page authentication (checks cookies)
def _page_auth_failure() -> HTTPException:
    """Where an unauthenticated page request goes, by mode.

    mira-OSS ships no `/login/` page (the CRM web redesign is omitted).
    Under `single` the auth surface itself bootstraps the session:
    `/v0/auth/local/session` provisions the local account, mints a cookie,
    and bounces back to `/chat`. Under `multi` there is nothing to redirect
    to yet, so the page request fails with the standard 401 envelope and the
    operator's own sign-in surface is the entry point.
    """
    if auth_mode() == "single":
        return HTTPException(status_code=302, headers={"Location": "/v0/auth/local/session"})
    return _auth_http_exception(AuthError("UNAUTHORIZED", "Authentication required"), 401)


def get_current_user_for_pages(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
) -> SessionData:
    """
    Get current authenticated user for HTML page requests.

    Checks both Authorization header and cookies.
    For unauthenticated requests, redirects under `single` (local bootstrap)
    or fails 401 under `multi` — see `_page_auth_failure`.
    """
    # Try Authorization header first
    auth_header = request.headers.get("Authorization", "")
    token = None

    if auth_header.startswith("Bearer "):
        token = auth_header[7:]

    # Try cookie if no header token
    if not token:
        token = request.cookies.get("session")

    # No token found at all
    if not token:
        raise _page_auth_failure()
    # Validate the token
    try:
        session_data = auth_service.validate_session(token, True)
        if not session_data:
            raise _page_auth_failure()

        # Set user context for downstream services
        set_current_user_id(session_data.user_id)
        set_current_user_data(session_data.model_dump())

        return session_data

    except AuthError:
        # Invalid token - redirect/fail per mode
        raise _page_auth_failure()
