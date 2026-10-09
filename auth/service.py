"""
Core authentication service - orchestrates the magic-link flow.

This is the generic multi-user auth layer. Everything an account owed to a
system outside `auth` goes through the injected `AccountProvisioner` (see
`auth/provisioning.py`), and everything that puts text in an inbox goes
through the injected `MailSender` (see `auth/email_service.py`). Both default
to "resolved lazily / nothing to do", which is what lets `single` and `dev`
installs import, boot and serve with no mailer and no sidecar system.
"""

import secrets
import hashlib
import logging
import time
import random
from typing import Optional, Tuple
from datetime import timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from utils.timezone_utils import utc_now
from utils.user_context import get_current_user_id
from utils.profile_validation import validate_profile_name
from .config import config
from .database import AuthDatabase
from .email_service import MailSender, get_mail_sender
from .provisioning import AccountProvisioner, NullProvisioner
from .session import SessionManager
from .security_logger import security_logger
from .exceptions import AuthError
from .types import UserRecord, UserProfile, SessionData, CookieSettings

logger = logging.getLogger(__name__)

# Fixture identity for the auto-provisioned local account under `single`
# mode (`GET /v0/auth/local/session`). This must never adopt a real person's
# identity; `user@localhost` matches the row pre-multi-user installs seeded,
# so an upgraded install adopts its existing data instead of provisioning a
# second identity. The timezone and the account's first name are derived from
# this install's configured defaults (SystemConfig.timezone and
# SystemConfig.local_session_first_name — the latter deploy-collected via
# MIRA_LOCAL_SESSION_FIRST_NAME, defaulting to "Friend") rather than hardcoded
# here.
LOCAL_SESSION_EMAIL = "user@localhost"
LOCAL_SESSION_LAST_NAME: Optional[str] = None
LOCAL_SESSION_CURRENT_FOCUS = "Get oriented with MIRA"


def normalize_email(email: str) -> str:
    """Canonical email identity: the lowercased address.

    Applied at every seam where an email address enters the auth service,
    so the stored row, the WHERE-email lookup, and the rate-limit key all
    key on one form — a case-variant address is one mailbox, never two
    accounts.
    """
    return email.lower()


class AuthService:
    """Lean magic-link authentication service."""

    def __init__(
        self,
        provisioner: AccountProvisioner = NullProvisioner(),
        mailer: Optional[MailSender] = None,
    ):
        self.db = AuthDatabase()
        self.session_manager = SessionManager()
        self.provisioner = provisioner
        self.security_logger = security_logger

        # None means "resolve at send time", so a single-mode install never
        # reads mail configuration it does not need, and an operator can
        # inject any structural MailSender without touching this module.
        self.mailer = mailer

        # Initialize required services immediately
        from .rate_limiter import RateLimiter
        self.rate_limiter = RateLimiter()

        # Configurable timeouts (for testing)
        self.SESSION_IDLE_TIMEOUT = config.SESSION_IDLE_TIMEOUT
        self.SESSION_MAX_LIFETIME = config.SESSION_MAX_LIFETIME

    def _generate_secure_token(self) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(32)

    def _hash_token(self, token: str) -> str:
        """Hash token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    def create_user(
        self,
        email: str,
        first_name: str,
        last_name: str,
        timezone: str,
        current_focus: str,
        ip_address: str = "",
        user_agent: str = ""
    ) -> str:
        """
        Create a new user and send magic link.

        Args:
            email: User's email address
            first_name: User's first name
            last_name: User's last name
            timezone: User's timezone (e.g., America/New_York)
            current_focus: User's current focus or goal
            ip_address: Client IP address for security logging
            user_agent: Client user agent for security logging

        Returns:
            User ID (UUID as string)
        """
        # One canonical form for identity: validation, the rate-limit key,
        # the stored row, and the WHERE-email lookup all see lowercased.
        email = normalize_email(email)

        # Public account creation requires a routable deliverable address;
        # the single-mode local bootstrap bypasses this method deliberately.
        import re

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise AuthError("invalid_email", "Invalid email format")

        try:
            first_name = validate_profile_name(first_name, "first_name")
            last_name = validate_profile_name(last_name, "last_name")
        except ValueError as e:
            raise AuthError("invalid_profile_name", str(e))
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise AuthError("invalid_timezone", "timezone must be a valid IANA timezone") from error

        # Check rate limit to prevent signup spam
        allowed, retry_after = self.rate_limiter.is_allowed(email, ip_address or None)
        if not allowed:
            self.security_logger.log_event(
                "auth.signup_rate_limit_exceeded",
                success=False,
                email=email,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"retry_after": retry_after}
            )
            raise AuthError(
                "rate_limit_exceeded",
                f"Too many signup attempts. Try again in {retry_after} seconds.",
                {"retry_after": retry_after}
            )

        user_id = self.db.create_user(
            email,
            first_name,
            last_name,
            timezone,
            current_focus,
            subject_kind="member",
        )
        try:
            self._initialize_account(
                user_id=user_id,
                first_name=first_name,
                timezone=timezone,
                current_focus=current_focus,
            )
        except Exception:
            try:
                cleanup_complete = self.provisioner.delete(user_id)
            except Exception:
                # A failed commit propagates out of local_teardown:
                # the account row was NOT deleted. Log the pending
                # cleanup with the full traceback, then re-raise — never
                # swallowed, and no destructive teardown step ran.
                logger.error(
                    "Account cleanup for %s failed after a provisioning "
                    "error; account is pending garbage collection",
                    user_id,
                    exc_info=True,
                )
                raise
            if not cleanup_complete:
                logger.error(
                    "Account provisioning failed and cleanup is pending for %s",
                    user_id,
                )
            raise

        self.security_logger.log_event(
            "auth.user_created",
            success=True,
            user_id=user_id,
            email=email
        )

        # Send magic link to new user. The account row has already committed:
        # a delivery failure is not a creation failure, and the row is never
        # torn down for it. Surface the distinct delivery-failed outcome with
        # a resend cue — the user already owns the account and can sign in
        # by requesting a magic link. Rate-limit refusals still propagate
        # unchanged.
        try:
            self.request_magic_link(email, ip_address, user_agent)
        except AuthError as error:
            if error.code != "service_error":
                raise
            self.security_logger.log_event(
                "auth.magic_link_delivery_failed",
                success=False,
                user_id=user_id,
                email=email,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"reason": "signup_link_send_failed"},
            )
            raise AuthError(
                "magic_link_delivery_failed",
                "Your account was created, but the sign-in email could not "
                "be delivered. Your account exists — request a magic link "
                "to sign in.",
                {},
            ) from error

        return user_id

    def create_local_session(self) -> Tuple[UserProfile, str]:
        """Create or reuse the single-mode local account and issue a browser session."""
        from config.config_manager import config as system_config

        timezone = system_config.system.timezone
        first_name = system_config.system.local_session_first_name
        user = self.db.get_user_by_email(LOCAL_SESSION_EMAIL)
        if user is None:
            user_id = self.db.create_user(
                email=LOCAL_SESSION_EMAIL,
                first_name=first_name,
                last_name=LOCAL_SESSION_LAST_NAME,
                timezone=timezone,
                current_focus=LOCAL_SESSION_CURRENT_FOCUS,
                subject_kind="member",
            )
            self._initialize_account(
                user_id=user_id,
                first_name=first_name,
                timezone=timezone,
                current_focus=LOCAL_SESSION_CURRENT_FOCUS,
            )
            user = self.db.get_user_by_id(user_id)
            if user is None:
                raise RuntimeError("Local account disappeared after initialization")
        else:
            self.provisioner.ensure(str(user.id), user.timezone)

        if not user.is_active:
            raise RuntimeError("Local account is inactive")

        self.db.update_user_login(str(user.id))
        session_token = self.session_manager.create_session(
            user,
            idle_timeout=self.SESSION_IDLE_TIMEOUT,
            max_lifetime=self.SESSION_MAX_LIFETIME,
        )
        self.security_logger.log_event(
            "auth.local_session",
            success=True,
            user_id=str(user.id),
            email=user.email,
        )
        return (
            UserProfile(
                id=str(user.id),
                email=user.email,
                is_active=user.is_active,
                created_at=user.created_at,
                subject_kind=user.subject_kind,
                demo_expires_at=user.demo_expires_at,
            ),
            session_token,
        )

    def _initialize_account(
        self,
        *,
        user_id: str,
        first_name: str,
        timezone: str,
        current_focus: str,
    ) -> None:
        """Create every required resource for an account.

        `initialize_mira_account` is MIRA's own work (continuum + welcome
        content); feedback-tracking state comes from `seed_lora_postgres`,
        which runs here so every provisioning path — signup, local-session
        bootstrap — gets it. Anything a sidecar system owes the account goes
        through the provisioner — in the OSS default, nothing.
        """
        self.db.initialize_mira_account(user_id, first_name, current_focus)

        from auth.seed_lora import seed_lora_postgres
        seed_lora_postgres(user_id)

        self.provisioner.provision(user_id, timezone)

    def get_user(self) -> UserRecord:
        """Get authenticated user. Raises if user not found (data corruption)."""
        user_id = get_current_user_id()
        user = self.db.get_user_by_id(user_id)
        if not user:
            raise AuthError("user_not_found", f"Authenticated user {user_id} not found in database")
        return user

    def _resolve_mailer(self) -> MailSender:
        """Return the mail transport for a send, or raise.

        Fails loud at the point of the send — never at import and never at
        startup — so `single` and `dev` installs serve traffic with no mail
        configuration of any kind.
        """
        if self.mailer is not None:
            return self.mailer
        mailer = get_mail_sender()
        if mailer is None:
            raise AuthError(
                "email_not_configured",
                "Email transport is not configured. Set smtp_host (and "
                "smtp_from) in Vault secret/mira/services — the only runtime "
                "SMTP source; post-install, `vault kv patch "
                "secret/mira/services smtp_host=... smtp_from=...` — to "
                "enable magic-link delivery.",
            )
        return mailer

    def _send_magic_link_email(self, mailer: MailSender, email: str, token: str) -> None:
        """Compose and deliver one magic-link email.

        The link target — `<app_url>/verify-magic-link` — is the path the
        sign-in UI is expected to occupy; the API contract that consumes the
        token is `POST /v0/auth/verify`. The token appears only in this
        composed message, never in a log line.
        """
        minutes = max(1, config.MAGIC_LINK_EXPIRY // 60)
        # APP_URL is canonical origin form (no trailing slash) by the time it
        # reaches here — normalized once at the auth config boundary.
        link = f"{config.APP_URL}/verify-magic-link?token={token}"
        subject = "Your MIRA sign-in link"
        body = (
            "Use the link below to sign in to MIRA. "
            f"It expires in {minutes} minutes and can be used once.\n\n"
            f"{link}\n\n"
            "If you did not request this, you can ignore this email — "
            "nothing happens unless the link is opened."
        )
        mailer.send(email, subject, body)

    def request_magic_link(
        self,
        email: str,
        ip_address: str = "",
        user_agent: str = ""
    ) -> bool:
        """Request a magic link for authentication."""
        # Same canonical form as signup: the rate-limit key and the
        # WHERE-email lookup see the lowercased address.
        email = normalize_email(email)
        # Check rate limit (both email and IP-based)
        allowed, retry_after = self.rate_limiter.is_allowed(email, ip_address or None)
        if not allowed:
            self.security_logger.log_event(
                "auth.rate_limit_exceeded",
                success=False,
                email=email,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"retry_after": retry_after}
            )
            raise AuthError(
                "rate_limit_exceeded",
                f"Too many requests. Try again in {retry_after} seconds.",
                {"retry_after": retry_after}
            )

        # Get user (but don't reveal whether they exist)
        user = self.db.get_user_by_email(email)
        if not user:
            # SECURITY FIX: Log internally but return success to prevent user enumeration
            self.security_logger.log_event(
                "auth.magic_link_request",
                success=False,
                email=email,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"reason": "user_not_found"}
            )
            # Add randomized delay to prevent timing attacks (simulate email send time)
            # Uses normal distribution: mean=0.5s, 85% within ±0.1s, total spread 0.3s
            delay = max(0.35, min(0.65, random.gauss(0.5, 0.07)))
            time.sleep(delay)
            # Return success - don't reveal that user doesn't exist
            return True

        if user.subject_kind == "demo":
            self.security_logger.log_event(
                "auth.magic_link_request",
                success=False,
                user_id=str(user.id),
                email=email,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"reason": "demo_account"},
            )
            delay = max(0.35, min(0.65, random.gauss(0.5, 0.07)))
            time.sleep(delay)
            return True

        # User exists - proceed with magic link generation
        token = self._generate_secure_token()
        token_hash = self._hash_token(token)
        expires_at = utc_now() + timedelta(seconds=config.MAGIC_LINK_EXPIRY)

        # Try to send email first (before storing in DB)
        # This ensures we don't store magic links if email fails
        try:
            mailer = self._resolve_mailer()
        except AuthError:
            self.security_logger.log_event(
                "auth.magic_link_request",
                success=False,
                user_id=str(user.id),
                email=email,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"reason": "email_not_configured"}
            )
            # Same generic envelope as a failed send: whether the deployment
            # has a mailer is server configuration, not the caller's business.
            raise AuthError(
                "service_error",
                "Unable to send magic link. Please try again.",
                {}
            )
        try:
            self._send_magic_link_email(mailer, email, token)
        except Exception:
            # Log failure but don't expose error details
            self.security_logger.log_event(
                "auth.magic_link_request",
                success=False,
                user_id=str(user.id),
                email=email,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"reason": "email_send_failed"}
            )
            # Generic error message (don't reveal email-specific failure)
            raise AuthError(
                "service_error",
                "Unable to send magic link. Please try again.",
                {}
            )

        # Store magic link only if email was sent successfully
        self.db.create_magic_link(
            user_id=str(user.id),
            email=email,
            token_hash=token_hash,
            expires_at=expires_at
        )

        self.security_logger.log_event(
            "auth.magic_link_request",
            success=True,
            user_id=str(user.id),
            email=email,
            ip_address=ip_address,
            user_agent=user_agent
        )
        return True

    def verify_magic_link(self, token: str) -> Tuple[UserProfile, str]:
        """Verify magic link and create session."""
        token_hash = self._hash_token(token)

        magic_link = self.db.consume_magic_link(token_hash)
        if not magic_link:
            existing_link = self.db.get_magic_link_by_token(token_hash)
            if existing_link and existing_link.expires_at < utc_now():
                self.security_logger.log_event(
                    "auth.magic_link_verify",
                    success=False,
                    user_id=existing_link.user_id,
                    email=existing_link.email,
                    details={"reason": "expired"}
                )
                raise AuthError("expired_token", "Magic link has expired")
            if existing_link and existing_link.used_at is not None:
                self.security_logger.log_event(
                    "auth.magic_link_verify",
                    success=False,
                    user_id=existing_link.user_id,
                    email=existing_link.email,
                    details={"reason": "already_used"}
                )
                raise AuthError("invalid_token", "Magic link has already been used")
            self.security_logger.log_event(
                "auth.magic_link_verify",
                success=False,
                details={"reason": "invalid_token"}
            )
            raise AuthError("invalid_token", "Invalid or expired magic link")

        # Get user
        user = self.db.get_user_by_id(magic_link.user_id)
        if not user or not user.is_active:
            raise AuthError("user_not_found", "User not found or inactive")

        # Update last login; 0 rows means the account was deleted concurrently
        if not self.db.update_user_login(str(user.id)):
            raise AuthError("user_not_found", "User not found or inactive")

        # Create session
        session_token = self.session_manager.create_session(
            user,
            idle_timeout=self.SESSION_IDLE_TIMEOUT,
            max_lifetime=self.SESSION_MAX_LIFETIME
        )

        self.security_logger.log_event(
            "auth.magic_link_verify",
            success=True,
            user_id=str(user.id),
            email=user.email
        )

        # Return user profile and session token
        user_profile = UserProfile(
            id=str(user.id),
            email=user.email,
            is_active=user.is_active,
            created_at=user.created_at,
            subject_kind=user.subject_kind,
            demo_expires_at=user.demo_expires_at,
        )

        return user_profile, session_token

    # === API Token Support (PostgreSQL-backed, hashed storage) ===
    def create_api_token(self, name: str, expires_in_days: Optional[int] = None) -> str:
        """Create a long-lived API token for programmatic access.

        Token is hashed before storage - the raw token is returned once and
        cannot be retrieved again. User must copy it immediately.

        Args:
            name: Friendly name for the token
            expires_in_days: Days until expiration (1-365), or None for no expiration

        Returns:
            Raw token string (shown once, never stored)
        """
        user_id = get_current_user_id()
        user = self.db.get_user_by_id(user_id)
        if not user or not user.is_active:
            raise AuthError("user_not_found", "User not found or inactive")

        # Enforce per-user token cap
        from config import config
        token_cap = config.auth.max_api_tokens_per_user
        existing_count = self.db.count_user_api_tokens(user_id)
        if existing_count >= token_cap:
            raise AuthError("too_many_tokens", "Token limit reached", {"limit": token_cap})

        # Generate cryptographically secure token
        raw_token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(raw_token)

        # Calculate expiration
        expires_at = None
        if expires_in_days is not None:
            expires_at = utc_now() + timedelta(days=max(1, min(365, expires_in_days)))

        # Store hashed token in PostgreSQL
        token_id = self.db.create_api_token(
            user_id=user_id,
            token_hash=token_hash,
            name=name or "API Token",
            expires_at=expires_at
        )

        self.security_logger.log_event(
            "auth.api_token_created",
            success=True,
            user_id=user_id,
            email=user.email,
            details={"token_id": token_id, "name": name, "expires_in_days": expires_in_days}
        )

        return raw_token

    def validate_api_token(self, raw_token: str) -> Optional[dict]:
        """Validate an API token and return user info if valid.

        Args:
            raw_token: The raw token string from Authorization header

        Returns:
            Dict with user_id, token_id, name if valid, None otherwise
        """
        token_hash = self._hash_token(raw_token)
        token_data = self.db.get_api_token_by_hash(token_hash)
        if not token_data:
            return None
        return token_data

    def list_api_tokens(self) -> list[dict]:
        """List all active API tokens for current user (metadata only)."""
        user_id = get_current_user_id()
        return self.db.list_api_tokens(user_id)

    def revoke_api_token(self, token_id: str) -> bool:
        """Revoke an API token by its ID."""
        user_id = get_current_user_id()
        result = self.db.revoke_api_token(user_id, token_id)

        if result:
            self.security_logger.log_event(
                "auth.api_token_revoked",
                success=True,
                user_id=user_id,
                details={"token_id": token_id}
            )
        return result

    def create_session(self, user_data: UserRecord) -> str:
        """Create a new session."""
        return self.session_manager.create_session(
            user_data,
            idle_timeout=self.SESSION_IDLE_TIMEOUT,
            max_lifetime=self.SESSION_MAX_LIFETIME
        )

    def validate_session(
        self,
        session_token: str,
        extend_activity: bool = False
    ) -> Optional[SessionData]:
        """Validate a session token."""
        return self.session_manager.validate_session(session_token, extend_activity)

    def logout(self, session_token: str) -> bool:
        """Logout by revoking session."""
        session_data = self.session_manager.validate_session(session_token)
        if session_data:
            user_id = session_data.user_id
            email = session_data.email

            result = self.session_manager.revoke_session(session_token)

            self.security_logger.log_event(
                "auth.logout",
                success=result,
                user_id=user_id,
                email=email
            )

            return result
        return False

    def logout_all_devices(self) -> int:
        """Logout all devices by revoking all user sessions."""
        user_id = get_current_user_id()
        user = self.db.get_user_by_id(user_id)
        if not user:
            raise AuthError("user_not_found", f"Authenticated user {user_id} not found in database")
        email = user.email

        revoked_count = self.session_manager.revoke_all_user_sessions()

        self.security_logger.log_event(
            "auth.logout_all_devices",
            success=True,
            user_id=user_id,
            email=email,
            details={"revoked_sessions": revoked_count}
        )

        return revoked_count

    def logout_other_devices(self, current_session_token: str) -> int:
        """Revoke every browser session except the cookie session making the request."""
        user_id = get_current_user_id()
        user = self.db.get_user_by_id(user_id)
        if not user:
            raise AuthError("user_not_found", f"Authenticated user {user_id} not found in database")

        current_session = self.session_manager.validate_session(current_session_token)
        if not current_session or current_session.user_id != user_id:
            raise AuthError("INVALID_TOKEN", "Current browser session is invalid")

        revoked_count = self.session_manager.revoke_user_sessions_except(
            user_id,
            current_session_token,
        )
        self.security_logger.log_event(
            "auth.logout_other_devices",
            success=True,
            user_id=user_id,
            email=user.email,
            details={"revoked_sessions": revoked_count},
        )
        return revoked_count

    def cleanup_expired_tokens(self) -> int:
        """Clean up expired magic links."""
        deleted_count = self.db.cleanup_expired_magic_links()
        logger.info(f"Cleaned up {deleted_count} expired magic links")
        return deleted_count

    def get_cookie_settings(self) -> CookieSettings:
        """Get secure cookie settings."""
        # Lax (not Strict) so cross-site top-level GET navigations — magic-link
        # clicks from an email client — carry the session. Browser writes are
        # guarded by explicit CSRF tokens, not SameSite. The Secure flag follows
        # the identity model: `single` mode serves plain HTTP on localhost where
        # browsers reject Secure cookies; `multi` installs sit behind real TLS.
        from .mode import auth_mode

        return CookieSettings(
            samesite="lax",
            httponly=True,
            secure=(auth_mode() == "multi"),
            max_age=config.SESSION_MAX_LIFETIME
        )

    def generate_csrf_token(self, session_token: str) -> str:
        """Generate CSRF token for session."""
        return self.session_manager.generate_csrf_token(session_token)

    def validate_csrf_token(self, session_token: str, csrf_token: str) -> bool:
        """Validate CSRF token."""
        return self.session_manager.validate_csrf_token(session_token, csrf_token)

    def register_cleanup_jobs(self, scheduler_service) -> None:
        """Register scheduled cleanup jobs with APScheduler."""
        from apscheduler.triggers.cron import CronTrigger
        from auth.account_gc import register_account_gc_job

        # Register magic link cleanup (hourly)
        scheduler_service.register_job(
            job_id="auth_cleanup",
            func=self.cleanup_expired_tokens,
            trigger=CronTrigger.from_crontab("15 * * * *"),  # Run at :15 every hour
            component="auth",
            description="Clean up expired magic links from database"
        )

        logger.info("Auth cleanup job registered (hourly at :15)")

        # Register account garbage collection (daily at 3am)
        register_account_gc_job(scheduler_service)


_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Return the process-wide AuthService, constructing it on first call.

    There is deliberately no module-level `auth_service = AuthService()`.
    Constructing the service opens no connection, but it is still work that
    must not be a precondition for *importing* this module: a module-level
    singleton makes every importer — the scheduler registry among them —
    depend on construction succeeding, in every mode, at import time.
    """
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
