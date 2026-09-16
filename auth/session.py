"""
Session management using Valkey.
"""

import secrets
import hashlib
import hmac
import logging
from typing import Optional
from datetime import timedelta
from utils.timezone_utils import utc_now, parse_utc_time_string
from utils.user_context import get_current_user_id
from clients.valkey_client import get_valkey
from .config import config
from .types import UserRecord, SessionData

logger = logging.getLogger(__name__)


class SessionManager:
    """Valkey-based session management."""

    def __init__(self):
        self.key_prefix = "session:"
        self.csrf_prefix = "csrf:"

    def _generate_session_token(self) -> str:
        """Generate a secure session token."""
        return secrets.token_urlsafe(32)

    def _token_digest(self, session_token: str) -> str:
        """Hash session tokens before using them in Valkey keys."""
        return hashlib.sha256(session_token.encode()).hexdigest()

    def _session_key(self, session_token: str) -> str:
        """Build the Valkey session key for a raw session token."""
        return f"{self.key_prefix}{self._token_digest(session_token)}"

    def _csrf_key(self, session_token: str) -> str:
        """Build the Valkey CSRF key for a raw session token."""
        return f"{self.csrf_prefix}{self._token_digest(session_token)}"

    def create_session(
        self,
        user_data: UserRecord,
        idle_timeout: Optional[int] = None,
        max_lifetime: Optional[int] = None,
        extra: Optional[dict] = None
    ) -> str:
        """Create a new session with configurable timeouts."""
        try:
            valkey = get_valkey()
            session_token = self._generate_session_token()

            # Extract user_id from UserRecord (Pydantic model)
            user_id = str(user_data.id)

            # Use provided timeouts or defaults
            idle = idle_timeout if idle_timeout is not None else config.SESSION_IDLE_TIMEOUT
            max_life = max_lifetime if max_lifetime is not None else config.SESSION_MAX_LIFETIME

            session_data = {
                "user_id": user_id,
                "email": user_data.email,
                "first_name": user_data.first_name,
                "last_name": user_data.last_name,
                "timezone": user_data.timezone,
                "subject_kind": user_data.subject_kind,
                "demo_expires_at": (
                    user_data.demo_expires_at.isoformat()
                    if user_data.demo_expires_at is not None
                    else None
                ),
                "created_at": utc_now().isoformat(),
                "last_activity": utc_now().isoformat(),
                "max_expiry": (utc_now() + timedelta(seconds=max_life)).isoformat()
            }

            # Optional additional metadata (e.g., API token markers)
            if extra and isinstance(extra, dict):
                for k, v in extra.items():
                    # Avoid overwriting core fields unless explicit
                    session_data[k] = v

            key = self._session_key(session_token)
            valkey.json_set_with_expiry(
                key,
                "$",
                session_data,
                idle
            )

            return session_token

        except Exception as e:
            logger.error(f"Failed to create session: {e}", exc_info=True)
            raise RuntimeError("Session creation failed")

    def validate_session(
        self,
        session_token: str,
        extend_activity: bool = False
    ) -> Optional[SessionData]:
        """
        Validate and optionally extend session.

        Returns None if session doesn't exist or is expired (legitimate cases).
        Raises exception if infrastructure fails.
        """
        valkey = get_valkey()
        key = self._session_key(session_token)

        session_data_list = valkey.json_get(key, "$")
        if not session_data_list or len(session_data_list) == 0:
            return None

        session_data = session_data_list[0]

        demo_expires_at = session_data.get("demo_expires_at")
        if session_data.get("subject_kind") == "demo":
            if not demo_expires_at or parse_utc_time_string(demo_expires_at) <= utc_now():
                valkey.delete(key)
                valkey.delete(self._csrf_key(session_token))
                return None

        # Check max lifetime
        max_expiry = session_data.get("max_expiry")
        if max_expiry:
            if parse_utc_time_string(max_expiry) < utc_now():
                # Session exceeded max lifetime
                valkey.delete(key)
                valkey.delete(self._csrf_key(session_token))
                return None

        if extend_activity:
            # Update last activity and extend TTL
            session_data["last_activity"] = utc_now().isoformat()
            valkey.json_set_with_expiry(
                key,
                "$",
                session_data,
                config.SESSION_IDLE_TIMEOUT
            )
            valkey.expire(self._csrf_key(session_token), config.SESSION_IDLE_TIMEOUT)
        return SessionData(**session_data)

    def revoke_session(self, session_token: str) -> bool:
        """Revoke a session."""
        try:
            valkey = get_valkey()
            key = self._session_key(session_token)
            deleted = valkey.delete(key)

            # Also delete associated CSRF token if any
            csrf_key = self._csrf_key(session_token)
            valkey.delete(csrf_key)

            return deleted > 0

        except Exception as e:
            logger.error(f"Failed to revoke session: {e}", exc_info=True)
            raise RuntimeError("Session revocation failed")

    def revoke_all_user_sessions(self) -> int:
        """Revoke all sessions for a user."""
        user_id = get_current_user_id()
        return self.revoke_user_sessions(user_id)

    def revoke_user_sessions(self, user_id: str) -> int:
        """Revoke every session for an explicitly selected user."""
        valkey = get_valkey()
        cursor = 0
        revoked_count = 0

        try:
            while True:
                # Scan for session keys
                cursor, keys = valkey.scan(
                    cursor,
                    match=f"{self.key_prefix}*",
                    count=100
                )

                for key in keys:
                    # Check if session belongs to user
                    session_data_list = valkey.json_get(key, "$")
                    if (session_data_list and len(session_data_list) > 0 and
                        session_data_list[0].get("user_id") == user_id):

                        if valkey.delete(key):
                            revoked_count += 1

                            # Also delete CSRF token
                            session_digest = key.replace(self.key_prefix, "", 1)
                            csrf_key = f"{self.csrf_prefix}{session_digest}"
                            valkey.delete(csrf_key)

                if cursor == 0:
                    break

            return revoked_count

        except Exception as e:
            logger.error(
                f"Failed to revoke all user sessions after revoking {revoked_count} sessions. "
                f"Error: {e}. Consider running logout-all again for user {user_id}.",
                exc_info=True
            )
            raise

    def revoke_user_sessions_except(self, user_id: str, current_session_token: str) -> int:
        """Revoke a user's browser sessions while preserving the current session."""
        valkey = get_valkey()
        current_key = self._session_key(current_session_token)
        cursor = 0
        revoked_count = 0

        try:
            while True:
                cursor, keys = valkey.scan(
                    cursor,
                    match=f"{self.key_prefix}*",
                    count=100,
                )
                for key in keys:
                    if key == current_key:
                        continue
                    session_data_list = valkey.json_get(key, "$")
                    if not session_data_list or session_data_list[0].get("user_id") != user_id:
                        continue
                    if valkey.delete(key):
                        revoked_count += 1
                        session_digest = key.replace(self.key_prefix, "", 1)
                        valkey.delete(f"{self.csrf_prefix}{session_digest}")

                if cursor == 0:
                    break
            return revoked_count
        except Exception as exc:
            logger.error(
                "Failed to revoke other sessions after revoking %s sessions for user %s: %s",
                revoked_count,
                user_id,
                exc,
                exc_info=True,
            )
            raise

    def generate_csrf_token(self, session_token: str) -> str:
        """Generate CSRF token for session."""
        try:
            valkey = get_valkey()
            csrf_token = secrets.token_urlsafe(32)

            key = self._csrf_key(session_token)
            valkey.json_set_with_expiry(
                key,
                "$",
                {"token": csrf_token, "created_at": utc_now().isoformat()},
                config.SESSION_IDLE_TIMEOUT
            )

            return csrf_token

        except Exception as e:
            logger.error(f"Failed to generate CSRF token: {e}", exc_info=True)
            raise RuntimeError("CSRF token generation failed")

    def validate_csrf_token(self, session_token: str, csrf_token: str) -> bool:
        """
        Validate CSRF token.

        Returns False if token doesn't exist or doesn't match (legitimate cases).
        Raises exception if infrastructure fails.
        """
        valkey = get_valkey()
        key = self._csrf_key(session_token)

        token_data_list = valkey.json_get(key, "$")
        if not token_data_list or len(token_data_list) == 0:
            return False

        stored_token = token_data_list[0].get("token")
        return bool(stored_token) and hmac.compare_digest(stored_token, csrf_token)
