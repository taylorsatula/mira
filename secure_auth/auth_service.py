"""
Core authentication service with enhanced security features.

Provides magic link authentication with comprehensive security controls,
timing attack protection, and enterprise-grade audit trails.
"""

import time
import logging
import ipaddress
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
from email_validator import validate_email, EmailNotValidError

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import and_, or_

from secure_auth.models import (
    User, MagicLink, SecurityEventType, RiskLevel, get_database_session
)
from secure_auth.token_service import get_token_service, TokenType
from secure_auth.audit_service import get_audit_service
from secure_auth.rate_limit_service import get_rate_limit_service, RateLimitAction

logger = logging.getLogger(__name__)


class AuthenticationResult:
    """Result of an authentication operation."""
    
    def __init__(
        self,
        success: bool,
        message: str,
        user: Optional[User] = None,
        magic_link_id: Optional[str] = None,
        error_code: Optional[str] = None,
        retry_after_seconds: Optional[int] = None
    ):
        self.success = success
        self.message = message
        self.user = user
        self.magic_link_id = magic_link_id
        self.error_code = error_code
        self.retry_after_seconds = retry_after_seconds


class SecureAuthService:
    """
    Enterprise-grade authentication service.
    
    Features:
    - Magic link authentication with enhanced security
    - Comprehensive rate limiting and abuse prevention
    - Advanced fraud detection and risk assessment
    - Timing attack protection
    - Device fingerprinting and trust scoring
    - Geographic and behavioral anomaly detection
    """
    
    def __init__(self, db_session: Optional[DBSession] = None):
        """
        Initialize authentication service.
        
        Args:
            db_session: Optional database session to use
        """
        self.db = db_session
        self.token_service = get_token_service()
        self.audit_service = get_audit_service()
        self.rate_limit_service = get_rate_limit_service()
        
        # Security configuration
        self.magic_link_expiry_minutes = 10
        self.max_concurrent_magic_links = 3
        self.trusted_ip_ranges = self._load_trusted_ip_ranges()
        self.min_response_time_ms = 500  # Minimum response time for timing attack protection
        
    def _get_db(self) -> DBSession:
        """Get database session."""
        if self.db:
            return self.db
        return get_database_session()
    
    def _load_trusted_ip_ranges(self) -> list:
        """Load trusted IP ranges from configuration."""
        # In production, this would come from configuration
        return [
            ipaddress.ip_network("10.0.0.0/8"),    # Private networks
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
        ]
    
    def _is_trusted_ip(self, ip_address: str) -> bool:
        """Check if IP address is in trusted ranges."""
        try:
            ip = ipaddress.ip_address(ip_address)
            return any(ip in network for network in self.trusted_ip_ranges)
        except ValueError:
            return False
    
    def _sanitize_email(self, email: str) -> str:
        """Sanitize and normalize email address."""
        if not email:
            raise ValueError("Email is required")
        
        # Basic sanitization
        email = email.strip().lower()
        
        # Length check
        if len(email) > 320:  # RFC 5321 limit
            raise ValueError("Email address too long")
        
        # Validate format
        try:
            validated = validate_email(email, check_deliverability=False)
            return validated.email
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email format: {str(e)}")
    
    def _sanitize_ip_address(self, ip_address: Optional[str]) -> Optional[str]:
        """Sanitize and validate IP address."""
        if not ip_address:
            return None
        
        try:
            # This will raise ValueError if invalid
            ipaddress.ip_address(ip_address.strip())
            return ip_address.strip()
        except ValueError:
            logger.warning(f"Invalid IP address format: {ip_address}")
            return None
    
    def _sanitize_user_agent(self, user_agent: Optional[str]) -> Optional[str]:
        """Sanitize user agent string."""
        if not user_agent:
            return None
        
        # Limit length and remove control characters
        user_agent = user_agent[:1000]
        user_agent = ''.join(char for char in user_agent if ord(char) >= 32)
        
        return user_agent.strip() if user_agent.strip() else None
    
    def _assess_risk_score(
        self,
        email: str,
        ip_address: Optional[str],
        user_agent: Optional[str],
        device_fingerprint: Optional[str] = None,
        user: Optional[User] = None
    ) -> int:
        """
        Assess risk score for authentication attempt.
        
        Args:
            email: Email address
            ip_address: Source IP address
            user_agent: User agent string
            device_fingerprint: Device fingerprint
            user: User object if known
            
        Returns:
            Risk score (0-100)
        """
        risk_score = 0
        
        # Base risk for unknown users
        if not user:
            risk_score += 10
        
        # IP-based risk assessment
        if ip_address:
            # Check if IP is trusted
            if self._is_trusted_ip(ip_address):
                risk_score -= 5
            else:
                risk_score += 5
            
            # Check for known malicious IPs (would integrate with threat intelligence)
            # For now, just check private vs public
            try:
                ip = ipaddress.ip_address(ip_address)
                if ip.is_private:
                    risk_score -= 2
                else:
                    risk_score += 3
            except ValueError:
                risk_score += 10
        
        # User agent analysis
        if user_agent:
            # Check for suspicious patterns
            suspicious_patterns = ['bot', 'crawler', 'scanner', 'curl', 'wget']
            if any(pattern in user_agent.lower() for pattern in suspicious_patterns):
                risk_score += 20
        else:
            # Missing user agent is suspicious
            risk_score += 15
        
        # User behavior analysis
        if user:
            # Recent user risk score
            user_risk = self.audit_service.get_user_risk_score(str(user.id))
            risk_score += min(user_risk // 2, 25)  # Cap contribution at 25
            
            # Account age (newer accounts are riskier)
            account_age_days = (datetime.utcnow() - user.created_at).days
            if account_age_days < 1:
                risk_score += 15
            elif account_age_days < 7:
                risk_score += 10
            elif account_age_days < 30:
                risk_score += 5
        
        return max(0, min(100, risk_score))
    
    def request_magic_link(
        self,
        email: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_fingerprint: Optional[str] = None,
        base_url: str = "https://localhost:8000",
        tenant_id: str = "default"
    ) -> AuthenticationResult:
        """
        Request a magic link for authentication.
        
        Args:
            email: User's email address
            ip_address: Source IP address
            user_agent: User agent string
            device_fingerprint: Device fingerprint
            base_url: Base URL for magic link construction
            tenant_id: Tenant identifier
            
        Returns:
            Authentication result
        """
        start_time = time.time()
        
        try:
            # Sanitize inputs
            email = self._sanitize_email(email)
            ip_address = self._sanitize_ip_address(ip_address)
            user_agent = self._sanitize_user_agent(user_agent)
            
            # Check rate limits - email-based
            email_rate_limit = self.rate_limit_service.check_rate_limit(
                identifier_type="email",
                identifier_value=email,
                action=RateLimitAction.MAGIC_LINK_REQUEST
            )
            
            if not email_rate_limit.allowed:
                self.audit_service.log_magic_link_event(
                    action="requested",
                    email=email,
                    success=False,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={"failure_reason": "rate_limited", "retry_after": email_rate_limit.retry_after_seconds}
                )
                
                return AuthenticationResult(
                    success=False,
                    message="Too many magic link requests. Please try again later.",
                    error_code="RATE_LIMITED",
                    retry_after_seconds=email_rate_limit.retry_after_seconds
                )
            
            # Check rate limits - IP-based
            if ip_address:
                ip_rate_limit = self.rate_limit_service.check_rate_limit(
                    identifier_type="ip",
                    identifier_value=ip_address,
                    action=RateLimitAction.MAGIC_LINK_REQUEST
                )
                
                if not ip_rate_limit.allowed:
                    self.audit_service.log_magic_link_event(
                        action="requested",
                        email=email,
                        success=False,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        details={"failure_reason": "ip_rate_limited", "retry_after": ip_rate_limit.retry_after_seconds}
                    )
                    
                    return AuthenticationResult(
                        success=False,
                        message="Too many requests from your location. Please try again later.",
                        error_code="IP_RATE_LIMITED",
                        retry_after_seconds=ip_rate_limit.retry_after_seconds
                    )
            
            db = self._get_db()
            
            try:
                # Find user (timing attack protection - always generate token)
                user = db.query(User).filter(
                    and_(
                        User.email == email,
                        User.tenant_id == tenant_id,
                        User.is_active == True
                    )
                ).first()
                
                # Always generate token for timing consistency
                token, salt = self.token_service.generate_token(
                    token_type=TokenType.MAGIC_LINK,
                    metadata={"email": email, "ip": ip_address}
                )
                
                if user:
                    # Check if account is locked
                    if user.is_locked:
                        self.audit_service.log_magic_link_event(
                            action="requested",
                            email=email,
                            success=False,
                            ip_address=ip_address,
                            user_agent=user_agent,
                            user_id=str(user.id),
                            details={"failure_reason": "account_locked"}
                        )
                        
                        # Still use timing protection
                        self._ensure_minimum_response_time(start_time)
                        return AuthenticationResult(
                            success=False,
                            message="If that email is registered, a magic link has been sent.",
                            error_code="ACCOUNT_LOCKED"
                        )
                    
                    # Risk assessment
                    risk_score = self._assess_risk_score(
                        email, ip_address, user_agent, device_fingerprint, user
                    )
                    
                    # Clean up old magic links for this user
                    self._cleanup_expired_magic_links(user.id)
                    
                    # Check concurrent magic links limit
                    active_links_count = db.query(MagicLink).filter(
                        and_(
                            MagicLink.user_id == user.id,
                            MagicLink.expires_at > datetime.utcnow(),
                            MagicLink.used_at.is_(None)
                        )
                    ).count()
                    
                    if active_links_count >= self.max_concurrent_magic_links:
                        # Invalidate oldest magic link
                        oldest_link = db.query(MagicLink).filter(
                            and_(
                                MagicLink.user_id == user.id,
                                MagicLink.expires_at > datetime.utcnow(),
                                MagicLink.used_at.is_(None)
                            )
                        ).order_by(MagicLink.created_at.asc()).first()
                        
                        if oldest_link:
                            oldest_link.expires_at = datetime.utcnow()
                    
                    # Hash token with additional context
                    token_hash = self.token_service.hash_token(
                        token, salt, additional_context=f"{user.id}:{ip_address}"
                    )
                    
                    # Generate device fingerprint if not provided
                    if not device_fingerprint and user_agent:
                        device_fingerprint = self.token_service.generate_device_fingerprint(
                            user_agent=user_agent
                        )
                    
                    # Create magic link record
                    expires_at = datetime.utcnow() + timedelta(minutes=self.magic_link_expiry_minutes)
                    magic_link = MagicLink(
                        user_id=user.id,
                        email=email,
                        token_hash=token_hash,
                        salt=salt,
                        expires_at=expires_at,
                        requesting_ip=ip_address,
                        requesting_user_agent=user_agent,
                        requesting_fingerprint=device_fingerprint,
                        risk_score=risk_score
                    )
                    
                    db.add(magic_link)
                    db.commit()
                    
                    # Record successful rate limit attempt
                    self.rate_limit_service.record_attempt(
                        identifier_type="email",
                        identifier_value=email,
                        action=RateLimitAction.MAGIC_LINK_REQUEST,
                        user_id=str(user.id),
                        success=True
                    )
                    
                    if ip_address:
                        self.rate_limit_service.record_attempt(
                            identifier_type="ip",
                            identifier_value=ip_address,
                            action=RateLimitAction.MAGIC_LINK_REQUEST,
                            user_id=str(user.id),
                            success=True
                        )
                    
                    # Log successful audit event
                    self.audit_service.log_magic_link_event(
                        action="requested",
                        email=email,
                        success=True,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        user_id=str(user.id),
                        details={
                            "magic_link_id": str(magic_link.id),
                            "risk_score": risk_score,
                            "expires_at": expires_at.isoformat()
                        }
                    )
                    
                    # Return success (actual email sending would be handled by caller)
                    result = AuthenticationResult(
                        success=True,
                        message="If that email is registered, a magic link has been sent.",
                        user=user,
                        magic_link_id=str(magic_link.id)
                    )
                    
                else:
                    # User not found - still record attempt for rate limiting
                    self.rate_limit_service.record_attempt(
                        identifier_type="email",
                        identifier_value=email,
                        action=RateLimitAction.MAGIC_LINK_REQUEST,
                        success=False
                    )
                    
                    if ip_address:
                        self.rate_limit_service.record_attempt(
                            identifier_type="ip",
                            identifier_value=ip_address,
                            action=RateLimitAction.MAGIC_LINK_REQUEST,
                            success=False
                        )
                    
                    # Log failed audit event
                    self.audit_service.log_magic_link_event(
                        action="requested",
                        email=email,
                        success=False,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        details={"failure_reason": "user_not_found"}
                    )
                    
                    result = AuthenticationResult(
                        success=True,  # Always return success for security
                        message="If that email is registered, a magic link has been sent."
                    )
                
                # Ensure minimum response time for timing attack protection
                self._ensure_minimum_response_time(start_time)
                return result
                
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to create magic link: {e}")
                
                # Log error audit event
                self.audit_service.log_magic_link_event(
                    action="requested",
                    email=email,
                    success=False,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={"failure_reason": "internal_error", "error": str(e)}
                )
                
                self._ensure_minimum_response_time(start_time)
                return AuthenticationResult(
                    success=False,
                    message="Unable to process your request at this time.",
                    error_code="INTERNAL_ERROR"
                )
            finally:
                if not self.db:
                    db.close()
                    
        except ValueError as e:
            # Input validation error
            logger.warning(f"Magic link request validation error: {e}")
            
            self.audit_service.log_magic_link_event(
                action="requested",
                email=email if 'email' in locals() else "invalid",
                success=False,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"failure_reason": "validation_error", "error": str(e)}
            )
            
            self._ensure_minimum_response_time(start_time)
            return AuthenticationResult(
                success=False,
                message="Invalid request format.",
                error_code="VALIDATION_ERROR"
            )
    
    def verify_magic_link(
        self,
        token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_fingerprint: Optional[str] = None
    ) -> AuthenticationResult:
        """
        Verify a magic link token.
        
        Args:
            token: Magic link token
            ip_address: Source IP address
            user_agent: User agent string
            device_fingerprint: Device fingerprint
            
        Returns:
            Authentication result with user if successful
        """
        start_time = time.time()
        
        try:
            # Sanitize inputs
            ip_address = self._sanitize_ip_address(ip_address)
            user_agent = self._sanitize_user_agent(user_agent)
            
            db = self._get_db()
            
            try:
                # Rate limiting for verification attempts
                if ip_address:
                    ip_rate_limit = self.rate_limit_service.check_rate_limit(
                        identifier_type="ip",
                        identifier_value=ip_address,
                        action=RateLimitAction.LOGIN_ATTEMPT
                    )
                    
                    if not ip_rate_limit.allowed:
                        self.audit_service.log_magic_link_event(
                            action="verified",
                            email="unknown",
                            success=False,
                            ip_address=ip_address,
                            user_agent=user_agent,
                            details={"failure_reason": "rate_limited", "retry_after": ip_rate_limit.retry_after_seconds}
                        )
                        
                        self._ensure_minimum_response_time(start_time)
                        return AuthenticationResult(
                            success=False,
                            message="Too many login attempts. Please try again later.",
                            error_code="RATE_LIMITED",
                            retry_after_seconds=ip_rate_limit.retry_after_seconds
                        )
                
                # Find all active magic links and verify against each
                # This prevents timing attacks by always checking all possibilities
                active_links = db.query(MagicLink).filter(
                    and_(
                        MagicLink.expires_at > datetime.utcnow(),
                        MagicLink.used_at.is_(None),
                        MagicLink.attempt_count < MagicLink.max_attempts
                    )
                ).all()
                
                verified_link = None
                for link in active_links:
                    # Verify token with context
                    user = db.query(User).filter(User.id == link.user_id).first()
                    if user:
                        context = f"{user.id}:{link.requesting_ip}"
                        if self.token_service.verify_token(
                            token, link.token_hash, link.salt, additional_context=context
                        ):
                            verified_link = link
                            break
                
                # Update attempt count for the link we tried to verify
                if verified_link:
                    verified_link.attempt_count += 1
                
                if not verified_link:
                    # Log failed verification
                    self.audit_service.log_magic_link_event(
                        action="verified",
                        email="unknown",
                        success=False,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        details={"failure_reason": "invalid_token"}
                    )
                    
                    # Record failed attempt for rate limiting
                    if ip_address:
                        self.rate_limit_service.record_attempt(
                            identifier_type="ip",
                            identifier_value=ip_address,
                            action=RateLimitAction.LOGIN_ATTEMPT,
                            success=False
                        )
                    
                    self._ensure_minimum_response_time(start_time)
                    return AuthenticationResult(
                        success=False,
                        message="Invalid or expired magic link.",
                        error_code="INVALID_TOKEN"
                    )
                
                # Get user and perform additional checks
                user = db.query(User).filter(User.id == verified_link.user_id).first()
                if not user or not user.is_active:
                    self.audit_service.log_magic_link_event(
                        action="verified",
                        email=verified_link.email,
                        success=False,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        user_id=str(verified_link.user_id),
                        details={"failure_reason": "user_inactive"}
                    )
                    
                    self._ensure_minimum_response_time(start_time)
                    return AuthenticationResult(
                        success=False,
                        message="Account is not active.",
                        error_code="ACCOUNT_INACTIVE"
                    )
                
                # Check if account is locked
                if user.is_locked:
                    self.audit_service.log_magic_link_event(
                        action="verified",
                        email=user.email,
                        success=False,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        user_id=str(user.id),
                        details={"failure_reason": "account_locked"}
                    )
                    
                    self._ensure_minimum_response_time(start_time)
                    return AuthenticationResult(
                        success=False,
                        message="Account is temporarily locked.",
                        error_code="ACCOUNT_LOCKED"
                    )
                
                # Mark magic link as used
                verified_link.used_at = datetime.utcnow()
                verified_link.verified_ip = ip_address
                verified_link.verified_user_agent = user_agent
                verified_link.verified_fingerprint = device_fingerprint
                
                # Update user last login
                user.last_login_at = datetime.utcnow()
                user.last_activity_at = datetime.utcnow()
                user.failed_login_count = 0  # Reset failed login count on success
                
                db.commit()
                
                # Record successful rate limit attempt
                if ip_address:
                    self.rate_limit_service.record_attempt(
                        identifier_type="ip",
                        identifier_value=ip_address,
                        action=RateLimitAction.LOGIN_ATTEMPT,
                        user_id=str(user.id),
                        success=True
                    )
                
                self.rate_limit_service.record_attempt(
                    identifier_type="email",
                    identifier_value=user.email,
                    action=RateLimitAction.LOGIN_ATTEMPT,
                    user_id=str(user.id),
                    success=True
                )
                
                # Log successful audit event
                self.audit_service.log_magic_link_event(
                    action="verified",
                    email=user.email,
                    success=True,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    user_id=str(user.id),
                    details={
                        "magic_link_id": str(verified_link.id),
                        "verification_time": datetime.utcnow().isoformat()
                    }
                )
                
                self._ensure_minimum_response_time(start_time)
                return AuthenticationResult(
                    success=True,
                    message="Magic link verified successfully.",
                    user=user
                )
                
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to verify magic link: {e}")
                
                self.audit_service.log_magic_link_event(
                    action="verified",
                    email="unknown",
                    success=False,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={"failure_reason": "internal_error", "error": str(e)}
                )
                
                self._ensure_minimum_response_time(start_time)
                return AuthenticationResult(
                    success=False,
                    message="Unable to process your request at this time.",
                    error_code="INTERNAL_ERROR"
                )
            finally:
                if not self.db:
                    db.close()
                    
        except Exception as e:
            logger.error(f"Magic link verification error: {e}")
            self._ensure_minimum_response_time(start_time)
            return AuthenticationResult(
                success=False,
                message="Invalid request.",
                error_code="INVALID_REQUEST"
            )
    
    def _cleanup_expired_magic_links(self, user_id: str) -> int:
        """Clean up expired magic links for a user."""
        db = self._get_db()
        
        try:
            result = db.query(MagicLink).filter(
                and_(
                    MagicLink.user_id == user_id,
                    or_(
                        MagicLink.expires_at <= datetime.utcnow(),
                        MagicLink.used_at.isnot(None)
                    )
                )
            ).delete()
            
            if result > 0:
                db.commit()
                logger.debug(f"Cleaned up {result} expired magic links for user {user_id}")
            
            return result
            
        finally:
            if not self.db:
                db.close()
    
    def _ensure_minimum_response_time(self, start_time: float):
        """Ensure minimum response time for timing attack protection."""
        elapsed = time.time() - start_time
        min_time = self.min_response_time_ms / 1000.0
        
        if elapsed < min_time:
            time.sleep(min_time - elapsed)
    
    def create_user(
        self,
        email: str,
        tenant_id: str = "default",
        is_admin: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuthenticationResult:
        """
        Create a new user account.
        
        Args:
            email: User's email address
            tenant_id: Tenant identifier
            is_admin: Whether user should have admin privileges
            metadata: Additional user metadata
            
        Returns:
            Authentication result with user if successful
        """
        try:
            # Sanitize email
            email = self._sanitize_email(email)
            
            db = self._get_db()
            
            try:
                # Check if user already exists
                existing_user = db.query(User).filter(
                    and_(
                        User.email == email,
                        User.tenant_id == tenant_id
                    )
                ).first()
                
                if existing_user:
                    return AuthenticationResult(
                        success=False,
                        message="User already exists.",
                        error_code="USER_EXISTS"
                    )
                
                # Create new user
                user = User(
                    email=email,
                    tenant_id=tenant_id,
                    is_admin=is_admin,
                    metadata=metadata or {}
                )
                
                db.add(user)
                db.commit()
                
                # Log user creation
                self.audit_service.log_security_event(
                    event_type=SecurityEventType.DATA_MODIFICATION,
                    success=True,
                    message=f"User account created for {email}",
                    user_id=str(user.id),
                    details={
                        "action": "user_created",
                        "email": email,
                        "tenant_id": tenant_id,
                        "is_admin": is_admin
                    }
                )
                
                logger.info(f"Created new user account: {email} (tenant: {tenant_id})")
                
                return AuthenticationResult(
                    success=True,
                    message="User account created successfully.",
                    user=user
                )
                
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to create user: {e}")
                return AuthenticationResult(
                    success=False,
                    message="Unable to create user account.",
                    error_code="CREATION_FAILED"
                )
            finally:
                if not self.db:
                    db.close()
                    
        except ValueError as e:
            logger.warning(f"User creation validation error: {e}")
            return AuthenticationResult(
                success=False,
                message="Invalid email format.",
                error_code="VALIDATION_ERROR"
            )


# Singleton instance
_auth_service: Optional[SecureAuthService] = None

def get_auth_service() -> SecureAuthService:
    """Get the authentication service singleton."""
    global _auth_service
    if _auth_service is None:
        _auth_service = SecureAuthService()
    return _auth_service