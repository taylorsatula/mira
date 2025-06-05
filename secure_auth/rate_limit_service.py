"""
Advanced rate limiting service with adaptive thresholds.

Provides sophisticated rate limiting with sliding windows,
adaptive thresholds, and abuse prevention mechanisms.
"""

import time
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import and_, or_, func

from secure_auth.models import RateLimit, User, get_database_session
from secure_auth.audit_service import get_audit_service

logger = logging.getLogger(__name__)


class RateLimitAction(str, Enum):
    """Types of actions that can be rate limited."""
    MAGIC_LINK_REQUEST = "magic_link_request"
    LOGIN_ATTEMPT = "login_attempt"
    PASSWORD_RESET = "password_reset"
    API_CALL = "api_call"
    SESSION_CREATE = "session_create"
    FAILED_LOGIN = "failed_login"
    ACCOUNT_LOOKUP = "account_lookup"


class RateLimitResult:
    """Result of a rate limit check."""
    
    def __init__(
        self,
        allowed: bool,
        current_count: int,
        limit_threshold: int,
        reset_time: datetime,
        retry_after_seconds: Optional[int] = None
    ):
        self.allowed = allowed
        self.current_count = current_count
        self.limit_threshold = limit_threshold
        self.reset_time = reset_time
        self.retry_after_seconds = retry_after_seconds


class RateLimitService:
    """
    Advanced rate limiting service with enterprise features.
    
    Features:
    - Sliding window rate limiting
    - Adaptive thresholds based on behavior
    - Multiple identifier types (email, IP, user)
    - Action-specific limits
    - Automatic escalation for repeated violations
    - Integration with audit logging
    """
    
    def __init__(self, db_session: Optional[DBSession] = None):
        """
        Initialize rate limit service.
        
        Args:
            db_session: Optional database session to use
        """
        self.db = db_session
        self.audit_service = get_audit_service()
        
        # Default rate limits (attempts per time window)
        self.default_limits = {
            RateLimitAction.MAGIC_LINK_REQUEST: {
                "email": {"limit": 3, "window_minutes": 20, "escalation_factor": 2},
                "ip": {"limit": 10, "window_minutes": 20, "escalation_factor": 1.5}
            },
            RateLimitAction.LOGIN_ATTEMPT: {
                "email": {"limit": 5, "window_minutes": 15, "escalation_factor": 2},
                "ip": {"limit": 20, "window_minutes": 15, "escalation_factor": 1.5},
                "user": {"limit": 8, "window_minutes": 30, "escalation_factor": 2}
            },
            RateLimitAction.FAILED_LOGIN: {
                "email": {"limit": 3, "window_minutes": 60, "escalation_factor": 3},
                "ip": {"limit": 10, "window_minutes": 60, "escalation_factor": 2},
                "user": {"limit": 5, "window_minutes": 60, "escalation_factor": 3}
            },
            RateLimitAction.PASSWORD_RESET: {
                "email": {"limit": 2, "window_minutes": 60, "escalation_factor": 2},
                "ip": {"limit": 5, "window_minutes": 60, "escalation_factor": 1.5}
            },
            RateLimitAction.API_CALL: {
                "user": {"limit": 1000, "window_minutes": 60, "escalation_factor": 1},
                "ip": {"limit": 2000, "window_minutes": 60, "escalation_factor": 1.2}
            },
            RateLimitAction.SESSION_CREATE: {
                "user": {"limit": 10, "window_minutes": 60, "escalation_factor": 2},
                "ip": {"limit": 50, "window_minutes": 60, "escalation_factor": 1.5}
            },
            RateLimitAction.ACCOUNT_LOOKUP: {
                "ip": {"limit": 100, "window_minutes": 60, "escalation_factor": 1.2}
            }
        }
    
    def _get_db(self) -> DBSession:
        """Get database session."""
        if self.db:
            return self.db
        return get_database_session()
    
    def _get_limit_config(
        self,
        action: RateLimitAction,
        identifier_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get rate limit configuration for action and identifier type.
        
        Args:
            action: Action being rate limited
            identifier_type: Type of identifier ("email", "ip", "user")
            
        Returns:
            Limit configuration or None if not configured
        """
        return self.default_limits.get(action, {}).get(identifier_type)
    
    def _calculate_adaptive_threshold(
        self,
        base_threshold: int,
        escalation_factor: float,
        violation_count: int
    ) -> int:
        """
        Calculate adaptive threshold based on violation history.
        
        Args:
            base_threshold: Base rate limit threshold
            escalation_factor: Factor to reduce limit by for each violation
            violation_count: Number of previous violations
            
        Returns:
            Adjusted threshold
        """
        if violation_count == 0:
            return base_threshold
        
        # Exponential backoff: threshold / (escalation_factor ^ violations)
        adjusted_threshold = int(base_threshold / (escalation_factor ** violation_count))
        
        # Ensure minimum threshold of 1
        return max(1, adjusted_threshold)
    
    def check_rate_limit(
        self,
        identifier_type: str,
        identifier_value: str,
        action: RateLimitAction,
        user_id: Optional[str] = None
    ) -> RateLimitResult:
        """
        Check if an action is within rate limits.
        
        Args:
            identifier_type: Type of identifier ("email", "ip", "user")
            identifier_value: Identifier value
            action: Action being performed
            user_id: Optional user ID for context
            
        Returns:
            Rate limit check result
        """
        db = self._get_db()
        
        try:
            config = self._get_limit_config(action, identifier_type)
            if not config:
                # No rate limit configured for this action/identifier combination
                return RateLimitResult(
                    allowed=True,
                    current_count=0,
                    limit_threshold=float('inf'),
                    reset_time=datetime.utcnow() + timedelta(hours=1)
                )
            
            base_limit = config["limit"]
            window_minutes = config["window_minutes"]
            escalation_factor = config["escalation_factor"]
            
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(minutes=window_minutes)
            
            # Find existing rate limit record for this window
            existing_limit = db.query(RateLimit).filter(
                and_(
                    RateLimit.identifier_type == identifier_type,
                    RateLimit.identifier_value == identifier_value,
                    RateLimit.action_type == action.value,
                    RateLimit.window_start >= window_start,
                    RateLimit.expires_at > current_time
                )
            ).first()
            
            if existing_limit:
                # Check violation history for adaptive threshold
                violation_count = self._get_violation_count(
                    identifier_type, identifier_value, action
                )
                
                # Calculate adaptive threshold
                current_threshold = self._calculate_adaptive_threshold(
                    base_limit, escalation_factor, violation_count
                )
                
                # Update threshold if changed
                if existing_limit.limit_threshold != current_threshold:
                    existing_limit.limit_threshold = current_threshold
                    existing_limit.escalation_factor = violation_count
                
                # Check if limit is exceeded
                is_exceeded = existing_limit.attempt_count >= current_threshold
                
                # Calculate reset time
                reset_time = existing_limit.window_start + timedelta(minutes=window_minutes)
                retry_after = int((reset_time - current_time).total_seconds()) if is_exceeded else None
                
                return RateLimitResult(
                    allowed=not is_exceeded,
                    current_count=existing_limit.attempt_count,
                    limit_threshold=current_threshold,
                    reset_time=reset_time,
                    retry_after_seconds=retry_after
                )
            else:
                # No existing limit record - this would be the first attempt
                return RateLimitResult(
                    allowed=True,
                    current_count=0,
                    limit_threshold=base_limit,
                    reset_time=current_time + timedelta(minutes=window_minutes)
                )
                
        finally:
            if not self.db:
                db.close()
    
    def record_attempt(
        self,
        identifier_type: str,
        identifier_value: str,
        action: RateLimitAction,
        user_id: Optional[str] = None,
        success: bool = True
    ) -> RateLimitResult:
        """
        Record an attempt and update rate limiting counters.
        
        Args:
            identifier_type: Type of identifier ("email", "ip", "user")
            identifier_value: Identifier value
            action: Action being performed
            user_id: Optional user ID for context
            success: Whether the attempt was successful
            
        Returns:
            Updated rate limit status
        """
        db = self._get_db()
        
        try:
            config = self._get_limit_config(action, identifier_type)
            if not config:
                # No rate limiting for this action/identifier
                return RateLimitResult(
                    allowed=True,
                    current_count=1,
                    limit_threshold=float('inf'),
                    reset_time=datetime.utcnow() + timedelta(hours=1)
                )
            
            base_limit = config["limit"]
            window_minutes = config["window_minutes"]
            escalation_factor = config["escalation_factor"]
            
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(minutes=window_minutes)
            
            # Find or create rate limit record
            existing_limit = db.query(RateLimit).filter(
                and_(
                    RateLimit.identifier_type == identifier_type,
                    RateLimit.identifier_value == identifier_value,
                    RateLimit.action_type == action.value,
                    RateLimit.window_start >= window_start,
                    RateLimit.expires_at > current_time
                )
            ).first()
            
            if existing_limit:
                # Update existing record
                existing_limit.attempt_count += 1
                existing_limit.last_violation_at = current_time if not success else existing_limit.last_violation_at
                
                # Get current threshold
                violation_count = self._get_violation_count(
                    identifier_type, identifier_value, action
                )
                current_threshold = self._calculate_adaptive_threshold(
                    base_limit, escalation_factor, violation_count
                )
                existing_limit.limit_threshold = current_threshold
                
                # Check if limit exceeded
                is_exceeded = existing_limit.attempt_count >= current_threshold
                
                # If limit exceeded, log violation and update escalation
                if is_exceeded and not existing_limit.is_exceeded:
                    existing_limit.reset_count += 1
                    self._log_rate_limit_violation(
                        identifier_type, identifier_value, action,
                        existing_limit.attempt_count, current_threshold, user_id
                    )
                
                db.commit()
                
                reset_time = existing_limit.window_start + timedelta(minutes=window_minutes)
                retry_after = int((reset_time - current_time).total_seconds()) if is_exceeded else None
                
                return RateLimitResult(
                    allowed=not is_exceeded,
                    current_count=existing_limit.attempt_count,
                    limit_threshold=current_threshold,
                    reset_time=reset_time,
                    retry_after_seconds=retry_after
                )
            else:
                # Create new rate limit record
                violation_count = self._get_violation_count(
                    identifier_type, identifier_value, action
                )
                current_threshold = self._calculate_adaptive_threshold(
                    base_limit, escalation_factor, violation_count
                )
                
                new_limit = RateLimit(
                    user_id=user_id,
                    identifier_type=identifier_type,
                    identifier_value=identifier_value,
                    action_type=action.value,
                    window_start=current_time,
                    window_duration_seconds=window_minutes * 60,
                    attempt_count=1,
                    limit_threshold=current_threshold,
                    base_threshold=base_limit,
                    escalation_factor=violation_count,
                    expires_at=current_time + timedelta(minutes=window_minutes),
                    last_violation_at=current_time if not success else None
                )
                
                db.add(new_limit)
                db.commit()
                
                return RateLimitResult(
                    allowed=True,
                    current_count=1,
                    limit_threshold=current_threshold,
                    reset_time=current_time + timedelta(minutes=window_minutes)
                )
                
        finally:
            if not self.db:
                db.close()
    
    def _get_violation_count(
        self,
        identifier_type: str,
        identifier_value: str,
        action: RateLimitAction,
        lookback_hours: int = 24
    ) -> int:
        """
        Get the number of rate limit violations in the lookback period.
        
        Args:
            identifier_type: Type of identifier
            identifier_value: Identifier value
            action: Action type
            lookback_hours: Hours to look back for violations
            
        Returns:
            Number of violations
        """
        db = self._get_db()
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
            
            violation_count = db.query(RateLimit).filter(
                and_(
                    RateLimit.identifier_type == identifier_type,
                    RateLimit.identifier_value == identifier_value,
                    RateLimit.action_type == action.value,
                    RateLimit.last_violation_at >= cutoff_time,
                    RateLimit.is_exceeded == True
                )
            ).count()
            
            return violation_count
            
        finally:
            if not self.db:
                db.close()
    
    def _log_rate_limit_violation(
        self,
        identifier_type: str,
        identifier_value: str,
        action: RateLimitAction,
        current_count: int,
        threshold: int,
        user_id: Optional[str] = None
    ):
        """Log a rate limit violation to audit system."""
        self.audit_service.log_rate_limit_violation(
            identifier_type=identifier_type,
            identifier_value=identifier_value,
            action_type=action.value,
            current_count=current_count,
            limit_threshold=threshold,
            user_id=user_id
        )
    
    def reset_rate_limit(
        self,
        identifier_type: str,
        identifier_value: str,
        action: RateLimitAction
    ) -> bool:
        """
        Manually reset rate limit for an identifier and action.
        
        Args:
            identifier_type: Type of identifier
            identifier_value: Identifier value
            action: Action type
            
        Returns:
            True if reset was successful
        """
        db = self._get_db()
        
        try:
            result = db.query(RateLimit).filter(
                and_(
                    RateLimit.identifier_type == identifier_type,
                    RateLimit.identifier_value == identifier_value,
                    RateLimit.action_type == action.value,
                    RateLimit.expires_at > datetime.utcnow()
                )
            ).delete()
            
            db.commit()
            
            if result > 0:
                logger.info(f"Reset rate limit for {identifier_type}:{identifier_value} on {action.value}")
                return True
            
            return False
            
        finally:
            if not self.db:
                db.close()
    
    def get_rate_limit_status(
        self,
        identifier_type: str,
        identifier_value: str,
        action: Optional[RateLimitAction] = None
    ) -> Dict[str, Any]:
        """
        Get current rate limit status for an identifier.
        
        Args:
            identifier_type: Type of identifier
            identifier_value: Identifier value
            action: Optional specific action to check
            
        Returns:
            Dictionary with rate limit status information
        """
        db = self._get_db()
        
        try:
            query = db.query(RateLimit).filter(
                and_(
                    RateLimit.identifier_type == identifier_type,
                    RateLimit.identifier_value == identifier_value,
                    RateLimit.expires_at > datetime.utcnow()
                )
            )
            
            if action:
                query = query.filter(RateLimit.action_type == action.value)
            
            active_limits = query.all()
            
            status = {
                "identifier": f"{identifier_type}:{identifier_value}",
                "active_limits": [],
                "total_violations": 0
            }
            
            for limit in active_limits:
                is_exceeded = limit.attempt_count >= limit.limit_threshold
                reset_time = limit.window_start + timedelta(seconds=limit.window_duration_seconds)
                
                limit_info = {
                    "action": limit.action_type,
                    "current_count": limit.attempt_count,
                    "limit_threshold": limit.limit_threshold,
                    "is_exceeded": is_exceeded,
                    "reset_time": reset_time.isoformat(),
                    "escalation_factor": limit.escalation_factor
                }
                
                status["active_limits"].append(limit_info)
                
                if is_exceeded:
                    status["total_violations"] += 1
            
            return status
            
        finally:
            if not self.db:
                db.close()
    
    def cleanup_expired_limits(self) -> int:
        """
        Clean up expired rate limit records.
        
        Returns:
            Number of records cleaned up
        """
        db = self._get_db()
        
        try:
            result = db.query(RateLimit).filter(
                RateLimit.expires_at <= datetime.utcnow()
            ).delete()
            
            db.commit()
            
            if result > 0:
                logger.info(f"Cleaned up {result} expired rate limit records")
            
            return result
            
        finally:
            if not self.db:
                db.close()


# Singleton instance
_rate_limit_service: Optional[RateLimitService] = None

def get_rate_limit_service() -> RateLimitService:
    """Get the rate limit service singleton."""
    global _rate_limit_service
    if _rate_limit_service is None:
        _rate_limit_service = RateLimitService()
    return _rate_limit_service