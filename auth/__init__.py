"""
JWT + Redis refresh token authentication system.
"""

from typing import Dict, Any, Optional

from .auth_service import AuthService, auth_service
from .models import User, MagicLink, Session
from .exceptions import AuthError, AuthErrorCode
from .api import router as auth_router, get_current_user

def get_current_user_optional() -> Optional[Dict[str, Any]]:
    """Get current user without requiring authentication."""
    return None

def get_user_id() -> Optional[str]:
    """Get current user ID from context."""
    # This would be implemented with request context
    return None

__all__ = [
    'AuthService',
    'User',
    'MagicLink', 
    'Session',
    'AuthError',
    'AuthErrorCode',
    'auth_router',
    'get_current_user',
    'get_current_user_optional',
    'get_user_id',
    'auth_service'
]