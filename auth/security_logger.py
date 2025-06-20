"""
Structured logging for authentication security events.
"""

import logging
import json
from typing import Dict, Any, Optional
from utils.timezone_utils import utc_now


class SecurityLogger:
    """Structured security event logger."""
    
    def __init__(self):
        self.logger = logging.getLogger("auth.security")
        self.logger.setLevel(logging.INFO)
        
        # Create formatter for structured JSON logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add handler if not already added
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _log_security_event(
        self, 
        event_type: str, 
        email: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log a structured security event."""
        event = {
            "timestamp": utc_now().isoformat(),
            "event_type": event_type,
            "success": success,
            "email": email,
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent[:100] if user_agent else None,  # Truncate
            "details": details or {}
        }
        
        # Remove None values
        event = {k: v for k, v in event.items() if v is not None}
        
        if success:
            self.logger.info(json.dumps(event))
        else:
            self.logger.warning(json.dumps(event))
    
    def login_attempt(
        self, 
        email: str, 
        ip_address: str, 
        user_agent: str, 
        success: bool,
        method: str = "magic_link",
        details: Optional[Dict[str, Any]] = None
    ):
        """Log login attempt."""
        self._log_security_event(
            event_type="login_attempt",
            email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details={**(details or {}), "method": method}
        )
    
    def signup_attempt(
        self, 
        email: str, 
        ip_address: str, 
        user_agent: str, 
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log signup attempt."""
        self._log_security_event(
            event_type="signup_attempt",
            email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details
        )
    
    def magic_link_request(
        self, 
        email: str, 
        ip_address: str, 
        user_agent: str, 
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log magic link request."""
        self._log_security_event(
            event_type="magic_link_request",
            email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details
        )
    
    def token_refresh(
        self, 
        user_id: str, 
        ip_address: str, 
        user_agent: str, 
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log token refresh attempt."""
        self._log_security_event(
            event_type="token_refresh",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details
        )
    
    def logout_event(
        self, 
        user_id: str, 
        logout_type: str = "single",
        details: Optional[Dict[str, Any]] = None
    ):
        """Log logout event."""
        self._log_security_event(
            event_type="logout",
            user_id=user_id,
            success=True,
            details={**(details or {}), "logout_type": logout_type}
        )
    
    def rate_limit_exceeded(
        self, 
        email: str, 
        ip_address: str, 
        user_agent: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log rate limit violation."""
        self._log_security_event(
            event_type="rate_limit_exceeded",
            email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            details=details
        )
    
    def webauthn_event(
        self, 
        email: str, 
        ip_address: str, 
        user_agent: str, 
        action: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log WebAuthn event."""
        self._log_security_event(
            event_type="webauthn_event",
            email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details={**(details or {}), "action": action}
        )
    
    def security_violation(
        self, 
        violation_type: str,
        email: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security violation."""
        self._log_security_event(
            event_type="security_violation",
            email=email,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            details={**(details or {}), "violation_type": violation_type}
        )


# Global security logger instance
security_logger = SecurityLogger()