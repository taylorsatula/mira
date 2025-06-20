"""
CSRF protection middleware for FastAPI application.
Implements CSRF token validation for state-changing requests.
"""

import secrets
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class CSRFMiddleware(BaseHTTPMiddleware):
    """Middleware to protect against Cross-Site Request Forgery attacks."""
    
    def __init__(self, app, secret_key: Optional[str] = None):
        super().__init__(app)
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.csrf_token_header = "X-CSRF-Token"
        self.csrf_cookie_name = "csrf_token"
        # Methods that require CSRF protection
        self.protected_methods = {"POST", "PUT", "PATCH", "DELETE"}
        # Paths that are exempt from CSRF protection
        self.exempt_paths = {
            "/health",
            "/auth/login",
            "/auth/register", 
            "/auth/magic-link/send",
            "/auth/magic-link/verify",
            "/auth/webauthn/register/begin",
            "/auth/webauthn/register/finish",
            "/auth/webauthn/authenticate/begin",
            "/auth/webauthn/authenticate/finish"
        }
    
    def _generate_csrf_token(self) -> str:
        """Generate a new CSRF token."""
        return secrets.token_urlsafe(32)
    
    def _is_exempt(self, request: Request) -> bool:
        """Check if the request path is exempt from CSRF protection."""
        return request.url.path in self.exempt_paths
    
    def _validate_csrf_token(self, request: Request) -> bool:
        """Validate the CSRF token from header against cookie."""
        # Get token from header
        header_token = request.headers.get(self.csrf_token_header)
        if not header_token:
            return False
        
        # Get token from cookie
        cookie_token = request.cookies.get(self.csrf_cookie_name)
        if not cookie_token:
            return False
        
        # Tokens must match
        return secrets.compare_digest(header_token, cookie_token)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request with CSRF protection."""
        
        # Skip CSRF protection for exempt paths and safe methods
        if (request.method not in self.protected_methods or 
            self._is_exempt(request)):
            response = await call_next(request)
        else:
            # Validate CSRF token for protected requests
            if not self._validate_csrf_token(request):
                raise HTTPException(
                    status_code=403,
                    detail="CSRF token missing or invalid"
                )
            response = await call_next(request)
        
        # Always set/refresh CSRF token in cookie for the next request
        if not request.cookies.get(self.csrf_cookie_name):
            csrf_token = self._generate_csrf_token()
            response.set_cookie(
                key=self.csrf_cookie_name,
                value=csrf_token,
                httponly=True,
                secure=request.url.scheme == "https",
                samesite="strict",
                max_age=3600  # 1 hour
            )
        
        return response