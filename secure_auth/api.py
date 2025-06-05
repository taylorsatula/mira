"""
FastAPI endpoints for secure authentication system.

Provides production-ready REST API endpoints for magic link authentication
with comprehensive security controls and audit logging.
"""

import os
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field

from secure_auth.models import User
from secure_auth.auth_service import get_auth_service
from secure_auth.session_service import get_session_service
from secure_auth.email_service import get_email_service
from secure_auth.audit_service import get_audit_service, SecurityEventType

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["secure-authentication"])

# Security scheme
security = HTTPBearer(auto_error=False)


# Request/Response models
class MagicLinkRequest(BaseModel):
    """Request model for magic link generation."""
    email: EmailStr = Field(..., description="User's email address")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    device_name: Optional[str] = Field(None, description="Human-readable device name")
    remember_device: bool = Field(default=False, description="Whether to remember this device")


class MagicLinkResponse(BaseModel):
    """Response model for magic link request."""
    success: bool
    message: str
    request_id: Optional[str] = None


class VerifyMagicLinkResponse(BaseModel):
    """Response model for magic link verification."""
    success: bool
    message: str
    user_email: Optional[str] = None
    session_token: Optional[str] = None
    requires_device_verification: bool = False


class UserInfoResponse(BaseModel):
    """Response model for user information."""
    id: str
    email: str
    tenant_id: str
    is_admin: bool
    email_verified: bool
    created_at: datetime
    last_login_at: Optional[datetime] = None


class LogoutResponse(BaseModel):
    """Response model for logout."""
    success: bool
    message: str


class CreateUserRequest(BaseModel):
    """Request model for user creation."""
    email: EmailStr = Field(..., description="User's email address")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    is_admin: bool = Field(default=False, description="Whether user should have admin privileges")


# Utility functions
def get_client_info(request: Request) -> dict:
    """Extract client information from request."""
    return {
        "ip_address": (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or
            request.headers.get("X-Real-IP") or
            (request.client.host if request.client else None)
        ),
        "user_agent": request.headers.get("User-Agent"),
        "device_fingerprint": None  # Could be enhanced with JavaScript fingerprinting
    }


# Dependency to get current user
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session_token: Optional[str] = Cookie(None, alias="secure_session")
) -> User:
    """
    Get the current authenticated user.
    
    Supports authentication via:
    1. Authorization header (Bearer token)
    2. Secure session cookie
    
    Args:
        request: FastAPI request object
        credentials: Optional bearer token from Authorization header
        session_token: Optional session token from cookie
        
    Returns:
        Current authenticated user
        
    Raises:
        HTTPException: If not authenticated or session invalid
    """
    # Extract session token
    token = None
    if credentials and credentials.credentials:
        token = credentials.credentials
    elif session_token:
        token = session_token
    
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get client information for context validation
    client_info = get_client_info(request)
    
    # Validate session
    session_service = get_session_service()
    session_result = session_service.get_session(
        session_token=token,
        ip_address=client_info["ip_address"],
        user_agent=client_info["user_agent"],
        device_fingerprint=client_info["device_fingerprint"]
    )
    
    if not session_result.success or not session_result.user:
        # Log failed authentication attempt
        audit_service = get_audit_service()
        audit_service.log_security_event(
            event_type=SecurityEventType.LOGIN_FAILURE,
            success=False,
            message="Invalid or expired session token",
            ip_address=client_info["ip_address"],
            user_agent=client_info["user_agent"],
            details={"failure_reason": session_result.error_code or "invalid_session"}
        )
        
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return session_result.user


# Optional dependency - returns None if not authenticated
async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session_token: Optional[str] = Cookie(None, alias="secure_session")
) -> Optional[User]:
    """
    Get the current user if authenticated, None otherwise.
    
    Args:
        request: FastAPI request object
        credentials: Optional bearer token
        session_token: Optional session cookie
        
    Returns:
        Current user or None
    """
    try:
        return await get_current_user(request, credentials, session_token)
    except HTTPException:
        return None


# Authentication endpoints
@router.post("/request-magic-link", response_model=MagicLinkResponse)
async def request_magic_link(
    request: Request,
    magic_link_data: MagicLinkRequest
) -> MagicLinkResponse:
    """
    Request a magic link for email authentication.
    
    This endpoint generates a secure magic link and sends it to the user's email.
    The link includes cryptographically secure tokens and expires after 10 minutes.
    
    Args:
        request: FastAPI request object
        magic_link_data: Magic link request data
        
    Returns:
        Magic link request response
    """
    client_info = get_client_info(request)
    
    # Get base URL for magic link construction
    base_url = str(request.base_url).rstrip('/')
    
    # Request magic link from auth service
    auth_service = get_auth_service()
    auth_result = auth_service.request_magic_link(
        email=magic_link_data.email,
        ip_address=client_info["ip_address"],
        user_agent=client_info["user_agent"],
        device_fingerprint=client_info["device_fingerprint"],
        base_url=base_url,
        tenant_id=magic_link_data.tenant_id
    )
    
    if not auth_result.success:
        if auth_result.error_code == "RATE_LIMITED":
            raise HTTPException(
                status_code=429,
                detail=auth_result.message,
                headers={"Retry-After": str(auth_result.retry_after_seconds or 60)}
            )
        elif auth_result.error_code == "VALIDATION_ERROR":
            raise HTTPException(status_code=400, detail=auth_result.message)
        else:
            raise HTTPException(status_code=500, detail=auth_result.message)
    
    # Send email if magic link was created successfully
    if auth_result.user and auth_result.magic_link_id:
        email_service = get_email_service()
        magic_link_url = f"{base_url}/auth/verify-magic-link/{auth_result.magic_link_id}"
        
        # Send magic link email (don't fail the request if email fails)
        try:
            email_service.send_magic_link(
                to_email=auth_result.user.email,
                magic_link=magic_link_url,
                expiry_minutes=10,
                user_id=str(auth_result.user.id),
                ip_address=client_info["ip_address"]
            )
        except Exception as e:
            logger.error(f"Failed to send magic link email: {e}")
    
    return MagicLinkResponse(
        success=True,
        message=auth_result.message,
        request_id=auth_result.magic_link_id
    )


@router.get("/verify-magic-link/{token}", response_model=VerifyMagicLinkResponse)
async def verify_magic_link(
    token: str,
    request: Request,
    response: Response,
    device_name: Optional[str] = None,
    remember_device: bool = False
) -> VerifyMagicLinkResponse:
    """
    Verify a magic link token and create an authenticated session.
    
    This endpoint validates the magic link token and creates a secure session
    with comprehensive context binding and device trust assessment.
    
    Args:
        token: Magic link token from URL
        request: FastAPI request object
        response: FastAPI response object
        device_name: Optional human-readable device name
        remember_device: Whether to extend session lifetime
        
    Returns:
        Magic link verification response
    """
    client_info = get_client_info(request)
    
    # Verify magic link
    auth_service = get_auth_service()
    auth_result = auth_service.verify_magic_link(
        token=token,
        ip_address=client_info["ip_address"],
        user_agent=client_info["user_agent"],
        device_fingerprint=client_info["device_fingerprint"]
    )
    
    if not auth_result.success:
        if auth_result.error_code == "RATE_LIMITED":
            raise HTTPException(
                status_code=429,
                detail=auth_result.message,
                headers={"Retry-After": str(auth_result.retry_after_seconds or 60)}
            )
        else:
            raise HTTPException(status_code=400, detail=auth_result.message)
    
    # Create session
    session_service = get_session_service()
    session_result = session_service.create_session(
        user=auth_result.user,
        ip_address=client_info["ip_address"],
        user_agent=client_info["user_agent"],
        device_fingerprint=client_info["device_fingerprint"],
        device_name=device_name or client_info["user_agent"],
        remember_device=remember_device,
        auth_method="magic_link"
    )
    
    if not session_result.success:
        raise HTTPException(status_code=500, detail=session_result.message)
    
    # Set secure session cookie
    is_development = os.environ.get("ENVIRONMENT") == "development"
    cookie_max_age = 2592000 if remember_device else 86400  # 30 days or 24 hours
    
    response.set_cookie(
        key="secure_session",
        value=session_result.session_token,
        httponly=True,
        secure=not is_development,
        samesite="strict",
        max_age=cookie_max_age,
        path="/"
    )
    
    # Send login notification email
    if auth_result.user:
        email_service = get_email_service()
        try:
            email_service.send_login_notification(
                to_email=auth_result.user.email,
                login_time=datetime.utcnow(),
                ip_address=client_info["ip_address"],
                device_info=device_name or client_info["user_agent"],
                location="Unknown",  # Could be enhanced with GeoIP
                user_id=str(auth_result.user.id)
            )
        except Exception as e:
            logger.error(f"Failed to send login notification: {e}")
    
    return VerifyMagicLinkResponse(
        success=True,
        message="Authentication successful",
        user_email=auth_result.user.email,
        session_token=session_result.session_token,
        requires_device_verification=session_result.requires_device_verification
    )


@router.get("/me", response_model=UserInfoResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
) -> UserInfoResponse:
    """
    Get information about the current authenticated user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information
    """
    return UserInfoResponse(
        id=str(current_user.id),
        email=current_user.email,
        tenant_id=current_user.tenant_id,
        is_admin=current_user.is_admin,
        email_verified=current_user.email_verified,
        created_at=current_user.created_at,
        last_login_at=current_user.last_login_at
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session_token: Optional[str] = Cookie(None, alias="secure_session")
) -> LogoutResponse:
    """
    Logout the current user and invalidate their session.
    
    Args:
        request: FastAPI request object
        response: FastAPI response object
        current_user: Current authenticated user
        credentials: Optional bearer token
        session_token: Optional session cookie
        
    Returns:
        Logout response
    """
    # Get session token
    token = None
    if credentials and credentials.credentials:
        token = credentials.credentials
    elif session_token:
        token = session_token
    
    if token:
        session_service = get_session_service()
        session_service.invalidate_session(token, reason="user_logout")
    
    # Clear session cookie
    response.delete_cookie("secure_session", path="/")
    
    # Log logout event
    client_info = get_client_info(request)
    audit_service = get_audit_service()
    audit_service.log_security_event(
        event_type=SecurityEventType.LOGOUT,
        success=True,
        message=f"User logged out: {current_user.email}",
        user_id=str(current_user.id),
        ip_address=client_info["ip_address"],
        user_agent=client_info["user_agent"]
    )
    
    return LogoutResponse(
        success=True,
        message="Logged out successfully"
    )


@router.post("/create-user", response_model=UserInfoResponse)
async def create_user(
    request: Request,
    user_data: CreateUserRequest,
    current_user: User = Depends(get_current_user)
) -> UserInfoResponse:
    """
    Create a new user account (admin only).
    
    Args:
        request: FastAPI request object
        user_data: User creation data
        current_user: Current authenticated user (must be admin)
        
    Returns:
        Created user information
    """
    # Check admin privileges
    if not current_user.is_admin:
        client_info = get_client_info(request)
        audit_service = get_audit_service()
        audit_service.log_security_event(
            event_type=SecurityEventType.PRIVILEGE_ESCALATION,
            success=False,
            message=f"Non-admin user attempted to create account: {current_user.email}",
            user_id=str(current_user.id),
            ip_address=client_info["ip_address"],
            user_agent=client_info["user_agent"],
            details={"attempted_action": "create_user", "target_email": user_data.email}
        )
        
        raise HTTPException(
            status_code=403,
            detail="Administrator privileges required"
        )
    
    # Create user
    auth_service = get_auth_service()
    auth_result = auth_service.create_user(
        email=user_data.email,
        tenant_id=user_data.tenant_id,
        is_admin=user_data.is_admin
    )
    
    if not auth_result.success:
        if auth_result.error_code == "USER_EXISTS":
            raise HTTPException(status_code=409, detail=auth_result.message)
        elif auth_result.error_code == "VALIDATION_ERROR":
            raise HTTPException(status_code=400, detail=auth_result.message)
        else:
            raise HTTPException(status_code=500, detail=auth_result.message)
    
    return UserInfoResponse(
        id=str(auth_result.user.id),
        email=auth_result.user.email,
        tenant_id=auth_result.user.tenant_id,
        is_admin=auth_result.user.is_admin,
        email_verified=auth_result.user.email_verified,
        created_at=auth_result.user.created_at,
        last_login_at=auth_result.user.last_login_at
    )


@router.get("/csrf-token")
async def get_csrf_token(request: Request) -> dict:
    """
    Get CSRF token for form submissions.
    
    The CSRF token is automatically generated and included in the response headers
    by the CSRF middleware. This endpoint provides a way to retrieve it explicitly.
    
    Args:
        request: FastAPI request object
        
    Returns:
        CSRF token information
    """
    csrf_token = request.headers.get("X-CSRF-Token", "will-be-set-by-middleware")
    
    return {
        "csrf_token": csrf_token,
        "usage": "Include this token in the X-CSRF-Token header for state-changing requests"
    }


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint for monitoring.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "service": "secure-authentication",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }