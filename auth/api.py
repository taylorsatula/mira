"""
FastAPI endpoints for authentication with JWT + Redis refresh tokens.
"""

from fastapi import APIRouter, Request, Response, HTTPException, Depends, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator, ValidationError
from typing import Optional, Dict, Any
import re
import html
import json

from .auth_service import auth_service, SessionLocal, engine
from .exceptions import AuthError
from .redis_client import get_redis
from utils.timezone_utils import utc_now

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()

class SignupRequest(BaseModel):
    email: EmailStr = Field(..., description="Valid email address")
    tenant_id: Optional[str] = Field(None, min_length=1, max_length=50, pattern=r'^[a-zA-Z0-9_-]+$', description="Alphanumeric tenant identifier")

class MagicLinkRequest(BaseModel):
    email: EmailStr = Field(..., description="Valid email address")
    tenant_id: Optional[str] = Field(None, min_length=1, max_length=50, pattern=r'^[a-zA-Z0-9_-]+$', description="Alphanumeric tenant identifier")

class MagicLinkVerify(BaseModel):
    token: str = Field(..., min_length=32, max_length=128, pattern=r'^[A-Za-z0-9_-]+$', description="Base64 URL-safe token")

class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=64, max_length=128, pattern=r'^[A-Za-z0-9_-]+$', description="Base64 URL-safe refresh token")

class WebAuthnCredential(BaseModel):
    credential: Dict[str, Any] = Field(..., description="WebAuthn credential object")
    challenge: str = Field(..., min_length=16, max_length=128, description="WebAuthn challenge string")
    
    @validator('credential')
    def validate_credential(cls, v):
        required_fields = ['id', 'rawId', 'response', 'type']
        if not all(field in v for field in required_fields):
            raise ValueError(f'Credential must contain fields: {required_fields}')
        if v.get('type') != 'public-key':
            raise ValueError('Credential type must be public-key')
        return v

class WebAuthnAuthFinish(BaseModel):
    email: EmailStr = Field(..., description="Valid email address")
    credential: Dict[str, Any] = Field(..., description="WebAuthn credential object")
    challenge: str = Field(..., min_length=16, max_length=128, description="WebAuthn challenge string")
    
    @validator('credential')
    def validate_credential(cls, v):
        required_fields = ['id', 'rawId', 'response', 'type']
        if not all(field in v for field in required_fields):
            raise ValueError(f'Credential must contain fields: {required_fields}')
        if v.get('type') != 'public-key':
            raise ValueError('Credential type must be public-key')
        return v

class WebAuthnStartResponse(BaseModel):
    options: Dict[str, Any] = Field(..., description="WebAuthn options object")
    challenge: str = Field(..., description="WebAuthn challenge string")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get current authenticated user from access token."""
    user = auth_service.get_user_from_access_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired access token")
    return user

@router.get("/health")
async def health_check():
    """Comprehensive health check for auth services."""
    health_status = {
        "status": "healthy",
        "timestamp": utc_now().isoformat(),
        "services": {}
    }
    
    # Check database connectivity
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis connectivity
    try:
        redis = await get_redis()
        await redis.redis.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Return appropriate status code
    status_code = 200 if health_status["status"] == "healthy" else 503
    return Response(
        content=json.dumps(health_status),
        status_code=status_code,
        media_type="application/json"
    )

@router.post("/signup")
async def signup(data: SignupRequest):
    """Create a new user account."""
    try:
        # Sanitize tenant_id if provided
        tenant_id = data.tenant_id.strip() if data.tenant_id else None
        user = await auth_service.signup_user(data.email, tenant_id)
        return {"message": "Account created successfully", "user": user}
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"error": "validation_failed", "details": e.errors()})
    except AuthError as e:
        raise HTTPException(
            status_code=400,
            detail=e.to_dict()
        )

@router.post("/request-magic-link")
async def request_magic_link(data: MagicLinkRequest, request: Request):
    """Request a magic link for passwordless authentication."""
    try:
        # Sanitize tenant_id if provided
        tenant_id = data.tenant_id.strip() if data.tenant_id else None
        user_agent = html.escape(request.headers.get("user-agent", ""))[:500]
        await auth_service.request_magic_link(
            data.email, 
            tenant_id, 
            request.client.host, 
            user_agent
        )
        return {"message": "Magic link sent to your email"}
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"error": "validation_failed", "details": e.errors()})
    except AuthError as e:
        status_code = 400
        if e.code == "rate_limit_exceeded":
            status_code = 429
        elif e.code == "user_not_found":
            status_code = 404
        raise HTTPException(
            status_code=status_code,
            detail=e.to_dict()
        )

@router.post("/verify-magic-link")
async def verify_magic_link(data: MagicLinkVerify, request: Request):
    """Verify magic link and create tokens."""
    try:
        # Sanitize user agent
        user_agent = html.escape(request.headers.get("user-agent", ""))[:500]  # Limit length
        user, access_token, refresh_token = await auth_service.verify_magic_link(
            data.token,
            request.client.host,
            user_agent
        )
        return {
            "user": user,
            "access_token": access_token,
            "refresh_token": refresh_token
        }
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"error": "validation_failed", "details": e.errors()})
    except AuthError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())

@router.get("/me")
async def get_me(user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user information."""
    return {"user": user}

@router.post("/refresh")
async def refresh_tokens(data: RefreshRequest, request: Request):
    """Refresh access token using refresh token."""
    try:
        # Sanitize user agent
        user_agent = html.escape(request.headers.get("user-agent", ""))[:500]  # Limit length
        result = await auth_service.refresh_tokens(
            data.refresh_token,
            request.client.host,
            user_agent
        )
        if not result:
            raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
        
        access_token, refresh_token = result
        return {
            "access_token": access_token,
            "refresh_token": refresh_token
        }
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"error": "validation_failed", "details": e.errors()})
    except Exception as e:
        raise HTTPException(status_code=401, detail="Token refresh failed")

@router.post("/logout")
async def logout(data: RefreshRequest):
    """Logout by revoking refresh token."""
    try:
        success = await auth_service.logout(data.refresh_token)
        if not success:
            raise HTTPException(status_code=400, detail="Logout failed")
        return {"message": "Logged out successfully"}
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"error": "validation_failed", "details": e.errors()})
    except AuthError as e:
        raise HTTPException(status_code=503, detail=e.to_dict())

@router.post("/logout-all")
async def logout_all_devices(user: Dict[str, Any] = Depends(get_current_user)):
    """Logout from all devices by revoking all refresh tokens."""
    try:
        count = await auth_service.logout_all_devices(user["id"])
        return {"message": f"Logged out from {count} devices"}
    except AuthError as e:
        raise HTTPException(status_code=503, detail=e.to_dict())

# WebAuthn endpoints
@router.post("/webauthn/register/start", response_model=WebAuthnStartResponse)
async def webauthn_register_start(user: Dict[str, Any] = Depends(get_current_user)):
    """Start WebAuthn registration."""
    try:
        options = auth_service.register_webauthn_start(user["id"])
        return WebAuthnStartResponse(
            options=options,
            challenge=options["challenge"]
        )
    except AuthError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())

@router.post("/webauthn/register/finish")
async def webauthn_register_finish(
    data: WebAuthnCredential,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Finish WebAuthn registration."""
    try:
        auth_service.register_webauthn_finish(
            user["id"],
            data.credential,
            data.challenge
        )
        return {"message": "WebAuthn credential registered successfully"}
    except AuthError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())

@router.post("/webauthn/authenticate/start", response_model=WebAuthnStartResponse)
async def webauthn_authenticate_start(data: MagicLinkRequest):
    """Start WebAuthn authentication."""
    try:
        options = auth_service.authenticate_webauthn_start(data.email)
        return WebAuthnStartResponse(
            options=options,
            challenge=options["challenge"]
        )
    except AuthError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())

@router.post("/webauthn/authenticate/finish")
async def webauthn_authenticate_finish(
    data: WebAuthnAuthFinish,
    request: Request
):
    """Finish WebAuthn authentication."""
    try:
        user, access_token, refresh_token = await auth_service.authenticate_webauthn_finish(
            data.email,
            data.credential,
            data.challenge,
            request.client.host,
            request.headers.get("user-agent", "")
        )
        return {
            "user": user,
            "access_token": access_token,
            "refresh_token": refresh_token
        }
    except AuthError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())