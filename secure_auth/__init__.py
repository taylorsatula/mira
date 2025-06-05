"""
Secure Authentication System

A production-ready authentication system with enterprise-grade security features
including magic link authentication, comprehensive audit logging, advanced rate
limiting, and multi-factor security controls.

Key Features:
- Magic link authentication with timing attack protection
- Advanced session management with context binding
- Comprehensive audit logging and security event tracking
- Sophisticated rate limiting with adaptive thresholds
- Device trust and recognition system
- CSRF protection and security headers
- Multi-tenant user isolation
- PostgreSQL database with optimized queries

Security Controls:
- Cryptographically secure token generation (512+ bits entropy)
- PBKDF2 key derivation with high iteration counts
- Session context binding to prevent hijacking
- Geographic and behavioral anomaly detection
- Rate limiting with exponential backoff
- Comprehensive audit trails for compliance
- Defense-in-depth security headers
- Input validation and sanitization

Usage:
    from secure_auth.api import router as auth_router
    from secure_auth.middleware import CSRFProtectionMiddleware, SecurityHeadersMiddleware
    
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(CSRFProtectionMiddleware)
    app.include_router(auth_router)

Environment Variables Required:
    # Database Configuration
    AUTH_DB_HOST - PostgreSQL host
    AUTH_DB_PORT - PostgreSQL port
    AUTH_DB_NAME - Database name
    AUTH_DB_USER - Database username
    AUTH_DB_PASSWORD - Database password
    
    # Cryptographic Keys
    AUTH_MASTER_KEY - 64-character hex string for token operations
    
    # Email Configuration
    SECURE_AUTH_SMTP_SERVER - SMTP server hostname
    SECURE_AUTH_SMTP_PORT - SMTP server port
    SECURE_AUTH_EMAIL_ADDRESS - From email address
    SECURE_AUTH_EMAIL_PASSWORD - Email password/app password
    
    # Environment
    ENVIRONMENT - "development" or "production" (affects cookie security)
"""

from .models import (
    User, UserSession, MagicLink, SecurityEvent, RateLimit, DeviceTrust,
    SecurityEventType, RiskLevel,
    create_auth_database, get_database_session
)

from .token_service import get_token_service, TokenType
from .auth_service import get_auth_service, AuthenticationResult
from .session_service import get_session_service, SessionResult
from .audit_service import get_audit_service
from .rate_limit_service import get_rate_limit_service, RateLimitAction
from .email_service import get_email_service

from .middleware import CSRFProtectionMiddleware, SecurityHeadersMiddleware, RateLimitMiddleware
from .api import router as auth_router

__version__ = "1.0.0"
__author__ = "Claude"
__description__ = "Enterprise-grade secure authentication system"

__all__ = [
    # Database models
    "User", "UserSession", "MagicLink", "SecurityEvent", "RateLimit", "DeviceTrust",
    "SecurityEventType", "RiskLevel",
    "create_auth_database", "get_database_session",
    
    # Services
    "get_token_service", "TokenType",
    "get_auth_service", "AuthenticationResult",
    "get_session_service", "SessionResult", 
    "get_audit_service",
    "get_rate_limit_service", "RateLimitAction",
    "get_email_service",
    
    # Middleware
    "CSRFProtectionMiddleware", "SecurityHeadersMiddleware", "RateLimitMiddleware",
    
    # API
    "auth_router",
    
    # Metadata
    "__version__", "__author__", "__description__"
]