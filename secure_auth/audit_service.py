"""
Comprehensive audit logging service for security events.

Provides detailed security event tracking, compliance logging,
and forensic data collection for the authentication system.
"""

import uuid
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from contextlib import contextmanager

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import and_, or_, desc

from secure_auth.models import (
    SecurityEvent, SecurityEventType, RiskLevel, User, UserSession,
    get_database_session
)

logger = logging.getLogger(__name__)


class AuditService:
    """
    Enterprise-grade audit logging service.
    
    Features:
    - Comprehensive security event tracking
    - Risk-based event classification
    - Compliance-ready audit trails
    - Forensic data collection
    - Performance-optimized logging
    """
    
    def __init__(self, db_session: Optional[DBSession] = None):
        """
        Initialize audit service.
        
        Args:
            db_session: Optional database session to use
        """
        self.db = db_session
        self._request_correlation_id = None
        
    def _get_db(self) -> DBSession:
        """Get database session."""
        if self.db:
            return self.db
        return get_database_session()
    
    @contextmanager
    def correlation_context(self, correlation_id: str):
        """
        Context manager for request correlation tracking.
        
        Args:
            correlation_id: Unique identifier for correlated events
        """
        old_correlation_id = self._request_correlation_id
        self._request_correlation_id = correlation_id
        try:
            yield
        finally:
            self._request_correlation_id = old_correlation_id
    
    def log_security_event(
        self,
        event_type: SecurityEventType,
        success: bool,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        risk_level: RiskLevel = RiskLevel.LOW,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_fingerprint: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action_performed: Optional[str] = None,
        error_code: Optional[str] = None,
        duration_ms: Optional[int] = None,
        tenant_id: Optional[str] = None,
        compliance_flags: Optional[Dict[str, Any]] = None,
        forensic_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a comprehensive security event.
        
        Args:
            event_type: Type of security event
            success: Whether the event was successful
            message: Human-readable event description
            user_id: Associated user ID
            session_id: Associated session ID
            risk_level: Risk assessment level
            details: Additional event details
            ip_address: Source IP address
            user_agent: User agent string
            device_fingerprint: Device fingerprint
            resource_type: Type of resource accessed
            resource_id: ID of resource accessed
            action_performed: Action performed on resource
            error_code: Error code if failed
            duration_ms: Operation duration in milliseconds
            tenant_id: Tenant identifier for multi-tenancy
            compliance_flags: Compliance-related flags
            forensic_data: Additional forensic information
            
        Returns:
            Event ID for correlation
        """
        db = self._get_db()
        event_id = str(uuid.uuid4())
        
        try:
            # Sanitize input data
            if user_agent and len(user_agent) > 1000:
                user_agent = user_agent[:1000]
            
            if details and not isinstance(details, dict):
                details = {"raw_details": str(details)}
            
            # Create security event
            security_event = SecurityEvent(
                id=event_id,
                user_id=user_id,
                session_id=session_id,
                event_type=event_type,
                risk_level=risk_level,
                success=success,
                message=message,
                details=details or {},
                error_code=error_code,
                ip_address=ip_address,
                user_agent=user_agent,
                device_fingerprint=device_fingerprint,
                request_id=str(uuid.uuid4()),
                correlation_id=self._request_correlation_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action_performed=action_performed,
                duration_ms=duration_ms,
                tenant_id=tenant_id or "default",
                compliance_flags=compliance_flags or {},
                forensic_data=forensic_data
            )
            
            db.add(security_event)
            db.commit()
            
            # Log to application logger based on risk level
            log_message = f"Security Event [{event_type.value}]: {message}"
            if risk_level == RiskLevel.CRITICAL:
                logger.critical(log_message, extra={
                    "event_id": event_id,
                    "user_id": user_id,
                    "ip_address": ip_address
                })
            elif risk_level == RiskLevel.HIGH:
                logger.error(log_message, extra={
                    "event_id": event_id,
                    "user_id": user_id
                })
            elif risk_level == RiskLevel.MEDIUM:
                logger.warning(log_message, extra={"event_id": event_id})
            else:
                logger.info(log_message, extra={"event_id": event_id})
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
            if not self.db:
                db.close()
            return event_id
        finally:
            if not self.db:
                db.close()
    
    def log_authentication_attempt(
        self,
        email: str,
        success: bool,
        method: str = "magic_link",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        failure_reason: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Log an authentication attempt with appropriate risk assessment.
        
        Args:
            email: Email address used for authentication
            success: Whether authentication was successful
            method: Authentication method used
            ip_address: Source IP address
            user_agent: User agent string
            failure_reason: Reason for failure if unsuccessful
            user_id: User ID if known
            
        Returns:
            Event ID
        """
        event_type = SecurityEventType.LOGIN_SUCCESS if success else SecurityEventType.LOGIN_FAILURE
        risk_level = RiskLevel.LOW if success else RiskLevel.MEDIUM
        
        if not success and failure_reason:
            # Assess risk based on failure reason
            high_risk_reasons = ["account_locked", "suspicious_location", "rate_limited"]
            if any(reason in failure_reason.lower() for reason in high_risk_reasons):
                risk_level = RiskLevel.HIGH
        
        details = {
            "email": email,
            "method": method
        }
        
        if failure_reason:
            details["failure_reason"] = failure_reason
        
        message = f"Authentication {'successful' if success else 'failed'} for {email}"
        if failure_reason:
            message += f" - {failure_reason}"
        
        return self.log_security_event(
            event_type=event_type,
            success=success,
            message=message,
            user_id=user_id,
            risk_level=risk_level,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def log_magic_link_event(
        self,
        action: str,
        email: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log magic link related events.
        
        Args:
            action: Action performed ("requested", "verified", "expired", "reused")
            email: Email address
            success: Whether action was successful
            ip_address: Source IP address
            user_agent: User agent string
            user_id: User ID if known
            details: Additional details
            
        Returns:
            Event ID
        """
        event_type_map = {
            "requested": SecurityEventType.MAGIC_LINK_REQUESTED,
            "verified": SecurityEventType.MAGIC_LINK_VERIFIED,
            "expired": SecurityEventType.MAGIC_LINK_EXPIRED,
            "reused": SecurityEventType.MAGIC_LINK_REUSED
        }
        
        event_type = event_type_map.get(action, SecurityEventType.LOGIN_ATTEMPT)
        
        # Assess risk level
        risk_level = RiskLevel.LOW
        if action == "reused":
            risk_level = RiskLevel.HIGH
        elif action == "expired" and not success:
            risk_level = RiskLevel.MEDIUM
        
        event_details = {"email": email, "action": action}
        if details:
            event_details.update(details)
        
        message = f"Magic link {action} for {email}"
        
        return self.log_security_event(
            event_type=event_type,
            success=success,
            message=message,
            user_id=user_id,
            risk_level=risk_level,
            details=event_details,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def log_session_event(
        self,
        action: str,
        session_id: str,
        user_id: str,
        success: bool = True,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log session management events.
        
        Args:
            action: Action performed ("created", "refreshed", "expired", "invalidated")
            session_id: Session identifier
            user_id: User identifier
            success: Whether action was successful
            ip_address: Source IP address
            user_agent: User agent string
            details: Additional details
            
        Returns:
            Event ID
        """
        event_type_map = {
            "created": SecurityEventType.SESSION_CREATED,
            "expired": SecurityEventType.SESSION_EXPIRED,
            "invalidated": SecurityEventType.SESSION_INVALIDATED
        }
        
        event_type = event_type_map.get(action, SecurityEventType.SESSION_CREATED)
        risk_level = RiskLevel.LOW
        
        event_details = {"action": action, "session_id": session_id}
        if details:
            event_details.update(details)
        
        message = f"Session {action} for user {user_id}"
        
        return self.log_security_event(
            event_type=event_type,
            success=success,
            message=message,
            user_id=user_id,
            session_id=session_id,
            risk_level=risk_level,
            details=event_details,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def log_rate_limit_violation(
        self,
        identifier_type: str,
        identifier_value: str,
        action_type: str,
        current_count: int,
        limit_threshold: int,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Log rate limiting violations.
        
        Args:
            identifier_type: Type of identifier ("email", "ip", "user")
            identifier_value: Identifier value
            action_type: Type of action being rate limited
            current_count: Current attempt count
            limit_threshold: Rate limit threshold
            ip_address: Source IP address
            user_agent: User agent string
            user_id: User ID if known
            
        Returns:
            Event ID
        """
        details = {
            "identifier_type": identifier_type,
            "identifier_value": identifier_value,
            "action_type": action_type,
            "current_count": current_count,
            "limit_threshold": limit_threshold
        }
        
        # Assess risk based on how much the limit was exceeded
        excess_ratio = current_count / limit_threshold
        if excess_ratio > 3:
            risk_level = RiskLevel.HIGH
        elif excess_ratio > 2:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        message = f"Rate limit exceeded for {identifier_type}:{identifier_value} on {action_type}"
        
        return self.log_security_event(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            success=False,
            message=message,
            user_id=user_id,
            risk_level=risk_level,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def log_suspicious_activity(
        self,
        activity_type: str,
        description: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        risk_indicators: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log suspicious activity detection.
        
        Args:
            activity_type: Type of suspicious activity
            description: Description of the activity
            user_id: User ID if known
            session_id: Session ID if applicable
            ip_address: Source IP address
            user_agent: User agent string
            risk_indicators: Risk indicators detected
            
        Returns:
            Event ID
        """
        details = {
            "activity_type": activity_type,
            "risk_indicators": risk_indicators or {}
        }
        
        return self.log_security_event(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            success=False,
            message=f"Suspicious activity detected: {description}",
            user_id=user_id,
            session_id=session_id,
            risk_level=RiskLevel.HIGH,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def get_security_events(
        self,
        user_id: Optional[str] = None,
        event_types: Optional[List[SecurityEventType]] = None,
        risk_levels: Optional[List[RiskLevel]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        ip_address: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SecurityEvent]:
        """
        Query security events with filters.
        
        Args:
            user_id: Filter by user ID
            event_types: Filter by event types
            risk_levels: Filter by risk levels
            start_time: Filter events after this time
            end_time: Filter events before this time
            ip_address: Filter by IP address
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of matching security events
        """
        db = self._get_db()
        
        try:
            query = db.query(SecurityEvent)
            
            # Apply filters
            if user_id:
                query = query.filter(SecurityEvent.user_id == user_id)
            
            if event_types:
                query = query.filter(SecurityEvent.event_type.in_(event_types))
            
            if risk_levels:
                query = query.filter(SecurityEvent.risk_level.in_(risk_levels))
            
            if start_time:
                query = query.filter(SecurityEvent.timestamp >= start_time)
            
            if end_time:
                query = query.filter(SecurityEvent.timestamp <= end_time)
            
            if ip_address:
                query = query.filter(SecurityEvent.ip_address == ip_address)
            
            # Order by timestamp descending and apply pagination
            query = query.order_by(desc(SecurityEvent.timestamp))
            query = query.offset(offset).limit(limit)
            
            return query.all()
            
        finally:
            if not self.db:
                db.close()
    
    def get_user_risk_score(
        self,
        user_id: str,
        time_window_hours: int = 24
    ) -> int:
        """
        Calculate a risk score for a user based on recent security events.
        
        Args:
            user_id: User ID to assess
            time_window_hours: Time window for assessment
            
        Returns:
            Risk score (0-100)
        """
        db = self._get_db()
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            events = db.query(SecurityEvent).filter(
                and_(
                    SecurityEvent.user_id == user_id,
                    SecurityEvent.timestamp >= cutoff_time
                )
            ).all()
            
            risk_score = 0
            
            for event in events:
                # Base score by risk level
                if event.risk_level == RiskLevel.LOW:
                    event_score = 1
                elif event.risk_level == RiskLevel.MEDIUM:
                    event_score = 5
                elif event.risk_level == RiskLevel.HIGH:
                    event_score = 15
                elif event.risk_level == RiskLevel.CRITICAL:
                    event_score = 30
                else:
                    event_score = 1
                
                # Failed events contribute more to risk
                if not event.success:
                    event_score *= 2
                
                risk_score += event_score
            
            # Cap at 100
            return min(risk_score, 100)
            
        finally:
            if not self.db:
                db.close()
    
    def cleanup_old_events(self, retention_days: int = 90) -> int:
        """
        Clean up old security events based on retention policy.
        
        Args:
            retention_days: Number of days to retain events
            
        Returns:
            Number of events deleted
        """
        db = self._get_db()
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
            
            # Keep critical events longer
            result = db.query(SecurityEvent).filter(
                and_(
                    SecurityEvent.timestamp < cutoff_time,
                    SecurityEvent.risk_level != RiskLevel.CRITICAL
                )
            ).delete()
            
            db.commit()
            
            logger.info(f"Cleaned up {result} old security events (older than {retention_days} days)")
            return result
            
        finally:
            if not self.db:
                db.close()


# Singleton instance
_audit_service: Optional[AuditService] = None

def get_audit_service() -> AuditService:
    """Get the audit service singleton."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service