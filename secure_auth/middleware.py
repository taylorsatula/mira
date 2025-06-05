"""
Security middleware for FastAPI authentication system.

Provides CSRF protection and comprehensive security headers
with enterprise-grade security controls.
"""

import os
import secrets
import logging
from typing import Callable, Optional

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from secure_auth.token_service import get_token_service
from secure_auth.audit_service import get_audit_service, SecurityEventType, RiskLevel

logger = logging.getLogger(__name__)


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CSRF protection using double-submit cookie pattern.
    
    Features:
    - Cryptographically secure token generation
    - Multiple token validation methods
    - Audit logging for CSRF violations
    - Configurable protection levels
    """
    
    def __init__(
        self,
        app,
        cookie_name: str = "secure_csrf_token",
        header_name: str = "X-CSRF-Token",
        form_field_name: str = "csrf_token",
        exempt_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.cookie_name = cookie_name
        self.header_name = header_name
        self.form_field_name = form_field_name
        self.safe_methods = {"GET", "HEAD", "OPTIONS", "TRACE"}
        self.exempt_paths = set(exempt_paths or [])
        
        self.token_service = get_token_service()
        self.audit_service = get_audit_service()
    
    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from CSRF protection."""
        return any(path.startswith(exempt) for exempt in self.exempt_paths)
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fall back to direct connection
        if request.client:
            return request.client.host
        
        return None
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with CSRF protection."""
        # Skip CSRF for safe methods and exempt paths
        if (request.method in self.safe_methods or 
            self._is_exempt_path(request.url.path)):
            response = await call_next(request)
            return self._add_csrf_cookie(response)
        
        # Get client information for audit logging
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent")
        
        # Get CSRF token from cookie
        csrf_cookie = request.cookies.get(self.cookie_name)
        
        if not csrf_cookie:
            # No CSRF cookie - this is suspicious for state-changing requests
            self.audit_service.log_security_event(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                success=False,
                message="CSRF protection triggered: Missing CSRF cookie",
                ip_address=client_ip,
                user_agent=user_agent,
                risk_level=RiskLevel.MEDIUM,
                details={
                    "violation_type": "missing_csrf_cookie",
                    "method": request.method,
                    "path": request.url.path
                }
            )
            
            raise HTTPException(
                status_code=403,
                detail="CSRF protection: Missing security token"
            )
        
        # Get submitted CSRF token
        submitted_token = None
        content_type = request.headers.get("Content-Type", "")
        
        if content_type.startswith("application/json"):
            # For JSON requests, expect token in header
            submitted_token = request.headers.get(self.header_name)
            if not submitted_token:
                self.audit_service.log_security_event(
                    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                    success=False,
                    message="CSRF protection triggered: Missing CSRF header for JSON request",
                    ip_address=client_ip,
                    user_agent=user_agent,
                    risk_level=RiskLevel.HIGH,
                    details={
                        "violation_type": "missing_csrf_header",
                        "method": request.method,
                        "path": request.url.path,
                        "content_type": content_type
                    }
                )
                
                raise HTTPException(
                    status_code=403,
                    detail="CSRF protection: Missing security header"
                )
                
        elif content_type.startswith("application/x-www-form-urlencoded"):
            # For form requests, expect token in form data
            try:
                form_data = await request.form()
                submitted_token = form_data.get(self.form_field_name)
                
                # Restore request body for downstream processing
                request._body = await request.body()
            except Exception as e:
                logger.warning(f"Failed to parse form data for CSRF check: {e}")
                
                self.audit_service.log_security_event(
                    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                    success=False,
                    message="CSRF protection triggered: Invalid form data",
                    ip_address=client_ip,
                    user_agent=user_agent,
                    risk_level=RiskLevel.MEDIUM,
                    details={
                        "violation_type": "invalid_form_data",
                        "method": request.method,
                        "path": request.url.path,
                        "error": str(e)
                    }
                )
                
                raise HTTPException(
                    status_code=400,
                    detail="Invalid request format"
                )
            
            if not submitted_token:
                self.audit_service.log_security_event(
                    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                    success=False,
                    message="CSRF protection triggered: Missing CSRF form field",
                    ip_address=client_ip,
                    user_agent=user_agent,
                    risk_level=RiskLevel.HIGH,
                    details={
                        "violation_type": "missing_csrf_form_field",
                        "method": request.method,
                        "path": request.url.path
                    }
                )
                
                raise HTTPException(
                    status_code=403,
                    detail="CSRF protection: Missing security token in form"
                )
        else:
            # For other content types, check header first, then form
            submitted_token = request.headers.get(self.header_name)
            
            if not submitted_token:
                self.audit_service.log_security_event(
                    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                    success=False,
                    message="CSRF protection triggered: No CSRF token provided",
                    ip_address=client_ip,
                    user_agent=user_agent,
                    risk_level=RiskLevel.HIGH,
                    details={
                        "violation_type": "no_csrf_token",
                        "method": request.method,
                        "path": request.url.path,
                        "content_type": content_type
                    }
                )
                
                raise HTTPException(
                    status_code=403,
                    detail="CSRF protection: Security token required"
                )
        
        # Verify CSRF token
        if not self.token_service.verify_csrf_token_pair(submitted_token, csrf_cookie):
            self.audit_service.log_security_event(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                success=False,
                message="CSRF protection triggered: Invalid CSRF token",
                ip_address=client_ip,
                user_agent=user_agent,
                risk_level=RiskLevel.HIGH,
                details={
                    "violation_type": "invalid_csrf_token",
                    "method": request.method,
                    "path": request.url.path
                }
            )
            
            raise HTTPException(
                status_code=403,
                detail="CSRF protection: Invalid security token"
            )
        
        # Process request
        response = await call_next(request)
        
        # Add/refresh CSRF cookie
        return self._add_csrf_cookie(response)
    
    def _add_csrf_cookie(self, response: Response) -> Response:
        """Add or refresh CSRF cookie."""
        # Generate new CSRF token pair
        token_value, token_hash = self.token_service.generate_csrf_token_pair()
        
        # Set cookie with security attributes
        is_development = os.environ.get("ENVIRONMENT") == "development"
        
        response.set_cookie(
            key=self.cookie_name,
            value=token_hash,  # Store hash in cookie
            httponly=False,    # JavaScript needs to read for AJAX requests
            secure=not is_development,  # HTTPS only in production
            samesite="strict", # Strict same-site policy
            max_age=86400,     # 24 hours
            path="/"
        )
        
        # Add token value to response header for JavaScript access
        response.headers[self.header_name] = token_value
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security headers middleware.
    
    Implements defense-in-depth security headers based on OWASP recommendations
    and industry best practices.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add comprehensive security headers to response."""
        response = await call_next(request)
        
        # Core security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        
        # Content Security Policy - strict by default
        csp_directives = [
            "default-src 'self'",
            "script-src 'self'",  # No unsafe-inline or unsafe-eval
            "style-src 'self' 'unsafe-inline'",  # Allow inline CSS for styling
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "media-src 'self'",
            "object-src 'none'",
            "frame-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
            "upgrade-insecure-requests"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # Permissions Policy (Feature Policy)
        permissions = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "payment=()",
            "usb=()",
            "magnetometer=()",
            "accelerometer=()",
            "gyroscope=()",
            "bluetooth=()",
            "ambient-light-sensor=()",
            "autoplay=()",
            "encrypted-media=()",
            "fullscreen=()",
            "picture-in-picture=()"
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions)
        
        # HSTS - Only in production with HTTPS
        environment = os.environ.get("ENVIRONMENT", "production")
        if environment != "development":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # Additional security headers
        response.headers["X-DNS-Prefetch-Control"] = "off"
        response.headers["X-Download-Options"] = "noopen"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        
        # Remove server identification headers
        response.headers.pop("Server", None)
        response.headers.pop("X-Powered-By", None)
        
        # Add security-focused cache control for sensitive endpoints
        if any(path in request.url.path for path in ["/auth/", "/api/", "/admin/"]):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Global rate limiting middleware for API protection.
    
    Provides basic rate limiting at the middleware level as a first line of defense.
    More sophisticated rate limiting is handled by the RateLimitService.
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        burst_limit: int = 20
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.audit_service = get_audit_service()
        
        # Simple in-memory rate limiting (in production, use Redis)
        self.request_counts = {}
        self.last_cleanup = None
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply global rate limiting."""
        client_id = self._get_client_identifier(request)
        current_time = int(time.time())
        
        # Simple rate limiting logic (replace with Redis in production)
        if client_id not in self.request_counts:
            self.request_counts[client_id] = {"count": 1, "window_start": current_time}
        else:
            client_data = self.request_counts[client_id]
            
            # Reset window if needed
            if current_time - client_data["window_start"] >= 60:  # 1 minute window
                client_data["count"] = 1
                client_data["window_start"] = current_time
            else:
                client_data["count"] += 1
                
                # Check rate limit
                if client_data["count"] > self.requests_per_minute:
                    # Log rate limit violation
                    self.audit_service.log_security_event(
                        event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                        success=False,
                        message=f"Global rate limit exceeded for {client_id}",
                        ip_address=client_id if client_id != "unknown" else None,
                        risk_level=RiskLevel.MEDIUM,
                        details={
                            "limit_type": "global_api",
                            "requests_per_minute": self.requests_per_minute,
                            "current_count": client_data["count"]
                        }
                    )
                    
                    raise HTTPException(
                        status_code=429,
                        detail="Too many requests",
                        headers={"Retry-After": "60"}
                    )
        
        return await call_next(request)