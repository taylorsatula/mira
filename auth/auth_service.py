"""
Enhanced authentication service with JWT + Redis refresh tokens.
"""

from datetime import timedelta
from typing import Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError
import logging

from .models import User, MagicLink, Base
from .tokens import TokenService
from .email_service import EmailService
from .webauthn_service import WebAuthnService
from .rate_limiter import RateLimiter
from .exceptions import AuthError, AuthErrorCode
from .config import config
from .security_logger import security_logger
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)

# Database setup with connection pooling and error handling
try:
    engine = create_engine(
        config.DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
        pool_pre_ping=True  # Verify connections before use
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database connection established and tables created")
    
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise


class AuthService:
    """Enhanced authentication service with refresh tokens."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(
            max_requests=config.RATE_LIMIT_REQUESTS,
            window_seconds=config.RATE_LIMIT_WINDOW
        )
        self.token_service = TokenService()
        self.email_service = EmailService()
        self.webauthn_service = WebAuthnService()
    
    async def request_magic_link(self, email: str, tenant_id: Optional[str] = None, ip_address: str = "", user_agent: str = "") -> bool:
        """
        Request a magic link for email authentication.
        If user doesn't exist, raise error to redirect to signup.
        
        Args:
            email: User email
            tenant_id: Optional tenant ID for multi-tenant setups
            
        Returns:
            True if email sent successfully
            
        Raises:
            AuthError: If rate limited, email fails, or user not registered
        """
        # Check rate limit
        allowed, retry_after = await self.rate_limiter.is_allowed(email)
        if not allowed:
            security_logger.rate_limit_exceeded(email, ip_address, user_agent, {"retry_after": retry_after})
            raise AuthError(
                AuthErrorCode.RATE_LIMIT_EXCEEDED,
                f"Too many requests. Try again in {retry_after} seconds.",
                {"retry_after": retry_after}
            )
        
        try:
            db = SessionLocal()
        except (OperationalError, DisconnectionError) as e:
            logger.error(f"Database connection failed: {e}")
            raise AuthError(
                AuthErrorCode.SESSION_EXPIRED,
                "Service temporarily unavailable",
                {"reason": "database_unavailable"}
            )
        
        try:
            # Check if user exists - if not, redirect to signup
            user = db.query(User).filter_by(email=email).first()
            if not user:
                security_logger.magic_link_request(email, ip_address, user_agent, False, {"reason": "user_not_found"})
                raise AuthError(
                    AuthErrorCode.USER_NOT_FOUND,
                    "Email not registered. Please sign up first.",
                    {"redirect_to": "/signup", "email": email}
                )
            
            # Generate magic link
            token, token_hash = self.token_service.generate_magic_link_token()
            
            magic_link = MagicLink(
                user_id=user.id,
                email=email,
                token_hash=token_hash,
                expires_at=utc_now() + timedelta(seconds=config.MAGIC_LINK_EXPIRY)
            )
            db.add(magic_link)
            db.commit()
            
            # Send email
            try:
                self.email_service.send_magic_link(email, token)
                security_logger.magic_link_request(email, ip_address, user_agent, True)
            except Exception as e:
                security_logger.magic_link_request(email, ip_address, user_agent, False, {"reason": "email_send_failed"})
                raise AuthError(
                    AuthErrorCode.EMAIL_SEND_FAILED,
                    "Failed to send magic link email",
                    {"reason": str(e)}
                )
            
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in request_magic_link: {e}")
            if db:
                db.rollback()
            raise AuthError(
                AuthErrorCode.SESSION_EXPIRED,
                "Service temporarily unavailable",
                {"reason": "database_error"}
            )
        finally:
            if db:
                db.close()
    
    async def signup_user(self, email: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new user account.
        
        Args:
            email: User email
            tenant_id: Optional tenant ID for multi-tenant setups
            
        Returns:
            User data dict
            
        Raises:
            AuthError: If user already exists or other errors
        """
        db = SessionLocal()
        try:
            # Check if user already exists
            existing_user = db.query(User).filter_by(email=email).first()
            if existing_user:
                raise AuthError(
                    AuthErrorCode.USER_ALREADY_EXISTS,
                    "User with this email already exists",
                    {"redirect_to": "/login", "email": email}
                )
            
            # Create new user
            user = User(email=email, tenant_id=tenant_id)
            db.add(user)
            db.commit()
            
            return user.to_dict()
            
        finally:
            db.close()
    
    async def verify_magic_link(
        self, 
        token: str, 
        ip_address: str, 
        user_agent: str
    ) -> Tuple[Dict[str, Any], str, str]:
        """
        Verify magic link and create tokens.
        
        Args:
            token: Magic link token
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Tuple of (user_data, access_token, refresh_token)
            
        Raises:
            AuthError: If token is invalid
        """
        db = SessionLocal()
        try:
            token_hash = self.token_service.hash_token(token)
            
            # Find magic link
            magic_link = db.query(MagicLink).filter_by(token_hash=token_hash).first()
            if not magic_link:
                raise AuthError(
                    AuthErrorCode.INVALID_TOKEN,
                    "Invalid or expired magic link"
                )
            
            # Check if expired
            if magic_link.is_expired():
                raise AuthError(
                    AuthErrorCode.EXPIRED_TOKEN,
                    "Magic link has expired"
                )
            
            # Check if already used
            if magic_link.is_used():
                raise AuthError(
                    AuthErrorCode.INVALID_TOKEN,
                    "Magic link has already been used"
                )
            
            # Mark as used
            magic_link.used_at = utc_now()
            
            # Get user
            user = db.query(User).filter_by(id=magic_link.user_id).first()
            if not user or not user.is_active:
                raise AuthError(
                    AuthErrorCode.USER_NOT_FOUND,
                    "User not found or inactive"
                )
            
            # Update last login
            user.last_login_at = utc_now()
            db.commit()
            
            # Generate tokens
            access_token = self.token_service.generate_access_token(
                str(user.id),
                user.to_dict()
            )
            refresh_token = self.token_service.generate_refresh_token()
            device_fingerprint = self.token_service.generate_device_fingerprint(
                ip_address, user_agent
            )
            
            # Store refresh token in Redis
            try:
                await self.token_service.store_refresh_token(
                    refresh_token,
                    str(user.id),
                    device_fingerprint,
                    ip_address,
                    user_agent
                )
            except Exception as e:
                # If Redis fails, the authentication must fail
                raise AuthError(
                    AuthErrorCode.SESSION_EXPIRED,
                    "Authentication service temporarily unavailable",
                    {"reason": "session_storage_failed"}
                )
            
            return user.to_dict(), access_token, refresh_token
            
        finally:
            db.close()
    
    async def refresh_tokens(
        self, 
        refresh_token: str,
        ip_address: str,
        user_agent: str
    ) -> Optional[Tuple[str, str]]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Current refresh token
            ip_address: Client IP address  
            user_agent: Client user agent
            
        Returns:
            Tuple of (new_access_token, new_refresh_token) if successful
        """
        try:
            # Validate refresh token
            token_data = await self.token_service.validate_refresh_token(refresh_token)
            if not token_data:
                return None
            
            user_id = token_data["user_id"]
            device_fingerprint = self.token_service.generate_device_fingerprint(
                ip_address, user_agent
            )
            
            # Get user data from database
            db = SessionLocal()
            try:
                user = db.query(User).filter_by(id=user_id).first()
                if not user or not user.is_active:
                    return None
                
                user_data = user.to_dict()
            finally:
                db.close()
            
            # Generate new access token
            new_access_token = self.token_service.generate_access_token(user_id, user_data)
            
            if config.REFRESH_TOKEN_ROTATION:
                # Rotate refresh token
                result = await self.token_service.rotate_refresh_token(
                    refresh_token,
                    user_data,
                    device_fingerprint,
                    ip_address,
                    user_agent
                )
                return result
            else:
                # Just return new access token with same refresh token
                return new_access_token, refresh_token
                
        except Exception as e:
            logger.error(f"Error refreshing tokens: {e}")
            return None
    
    async def logout(self, refresh_token: str) -> bool:
        """
        Logout by revoking refresh token.
        
        Args:
            refresh_token: Refresh token to revoke
            
        Returns:
            True if logout successful
            
        Raises:
            AuthError: If Redis is unavailable
        """
        try:
            return await self.token_service.revoke_refresh_token(refresh_token)
        except Exception as e:
            raise AuthError(
                AuthErrorCode.SESSION_EXPIRED,
                "Logout service temporarily unavailable",
                {"reason": "session_revocation_failed"}
            )
    
    async def logout_all_devices(self, user_id: str) -> int:
        """
        Logout from all devices by revoking all refresh tokens.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of tokens revoked
            
        Raises:
            AuthError: If Redis is unavailable
        """
        try:
            return await self.token_service.revoke_all_user_tokens(user_id)
        except Exception as e:
            raise AuthError(
                AuthErrorCode.SESSION_EXPIRED,
                "Logout-all service temporarily unavailable",
                {"reason": "session_revocation_failed"}
            )
    
    def get_user_from_access_token(self, access_token: str) -> Optional[Dict[str, Any]]:
        """
        Get user data from access token.
        
        Args:
            access_token: JWT access token
            
        Returns:
            User data if token is valid, None otherwise
        """
        payload = self.token_service.verify_access_token(access_token)
        if not payload:
            return None
        
        # Get fresh user data from database
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(id=payload["user_id"]).first()
            if not user or not user.is_active:
                return None
            
            return user.to_dict()
        finally:
            db.close()
    
    # WebAuthn methods
    async def register_webauthn_start(self, user_id: str) -> Dict[str, Any]:
        """Start WebAuthn registration."""
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(id=user_id).first()
            if not user:
                raise AuthError(
                    AuthErrorCode.USER_NOT_FOUND,
                    "User not found"
                )
            
            return self.webauthn_service.generate_registration_options(
                str(user.id),
                user.email
            )
        finally:
            db.close()
    
    async def register_webauthn_finish(
        self, 
        user_id: str,
        credential: Dict[str, Any],
        challenge: str
    ) -> bool:
        """Finish WebAuthn registration."""
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(id=user_id).first()
            if not user:
                raise AuthError(
                    AuthErrorCode.USER_NOT_FOUND,
                    "User not found"
                )
            
            cred_data = self.webauthn_service.verify_registration(
                credential,
                challenge,
                str(user.id)
            )
            
            if not cred_data:
                raise AuthError(
                    AuthErrorCode.WEBAUTHN_FAILED,
                    "WebAuthn registration failed"
                )
            
            # Store credential
            if not user.webauthn_credentials:
                user.webauthn_credentials = {}
            
            user.webauthn_credentials[cred_data["credential_id"]] = cred_data
            db.commit()
            
            return True
            
        finally:
            db.close()
    
    async def authenticate_webauthn_start(self, email: str) -> Dict[str, Any]:
        """Start WebAuthn authentication."""
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(email=email).first()
            if not user:
                raise AuthError(
                    AuthErrorCode.USER_NOT_FOUND,
                    "Email not registered. Please sign up first.",
                    {"redirect_to": "/signup", "email": email}
                )
            
            if not user.webauthn_credentials:
                raise AuthError(
                    AuthErrorCode.USER_NOT_FOUND,
                    "No WebAuthn credentials found"
                )
            
            return self.webauthn_service.generate_authentication_options(
                user.webauthn_credentials
            )
        finally:
            db.close()
    
    async def authenticate_webauthn_finish(
        self,
        email: str,
        credential: Dict[str, Any],
        challenge: str,
        ip_address: str,
        user_agent: str
    ) -> Tuple[Dict[str, Any], str, str]:
        """Finish WebAuthn authentication."""
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(email=email).first()
            if not user:
                raise AuthError(
                    AuthErrorCode.USER_NOT_FOUND,
                    "User not found"
                )
            
            # Find matching credential
            cred_id = credential.get("id")
            if not cred_id or cred_id not in user.webauthn_credentials:
                raise AuthError(
                    AuthErrorCode.INVALID_CREDENTIALS,
                    "Invalid credential"
                )
            
            stored_cred = user.webauthn_credentials[cred_id]
            
            # Verify
            if not self.webauthn_service.verify_authentication(
                credential,
                challenge,
                stored_cred
            ):
                raise AuthError(
                    AuthErrorCode.WEBAUTHN_FAILED,
                    "WebAuthn authentication failed"
                )
            
            # Update user
            user.last_login_at = utc_now()
            db.commit()
            
            # Generate tokens
            access_token = self.token_service.generate_access_token(
                str(user.id),
                user.to_dict()
            )
            refresh_token = self.token_service.generate_refresh_token()
            device_fingerprint = self.token_service.generate_device_fingerprint(
                ip_address, user_agent
            )
            
            # Store refresh token in Redis
            try:
                await self.token_service.store_refresh_token(
                    refresh_token,
                    str(user.id),
                    device_fingerprint,
                    ip_address,
                    user_agent
                )
            except Exception as e:
                # If Redis fails, the authentication must fail
                raise AuthError(
                    AuthErrorCode.SESSION_EXPIRED,
                    "Authentication service temporarily unavailable",
                    {"reason": "session_storage_failed"}
                )
            
            return user.to_dict(), access_token, refresh_token
            
        finally:
            db.close()


# Global instance
auth_service = AuthService()