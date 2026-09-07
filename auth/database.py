"""
Database operations for the lean auth system.

Session discipline is the security contract of this module (plan §6.3.2):
pre-authentication reads — user lookup, magic-link CRUD, API-token hash
validation — run on the BYPASSRLS admin session because they happen before
any user context exists, and post-authentication per-user token operations
run on `get_session(user_id)` so RLS on `api_tokens` enforces ownership.
With RLS enabled on `users`, `magic_links` and `api_tokens`, using the wrong
side of this split does not error — the fail-closed `NULLIF` policy predicate
yields silent zero-row results.
"""

import logging
from typing import Optional
from datetime import datetime
from utils.timezone_utils import utc_now
from utils.database_session_manager import get_shared_session_manager
from .types import MagicLinkRecord, SubjectKind, UserRecord

logger = logging.getLogger(__name__)


class AuthDatabase:
    """Minimal database operations for auth."""

    def __init__(self):
        self.session_manager = get_shared_session_manager()

    def create_user(
        self,
        email: str,
        first_name: str,
        last_name: str,
        timezone: str,
        current_focus: str,
        subject_kind: SubjectKind = "member",
        demo_start_at: Optional[datetime] = None,
        demo_expires_at: Optional[datetime] = None,
    ) -> str:
        """
        Create a new user.

        Args:
            email: User's email address
            first_name: User's first name
            last_name: User's last name
            timezone: User's timezone (e.g., America/New_York)
            current_focus: User's current focus or goal

        Returns:
            User ID (UUID as string)

        Raises:
            AuthError: If email is invalid or user already exists
        """
        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            from .exceptions import AuthError
            raise AuthError("invalid_email", "Invalid email format")

        # Check if user exists
        existing = self.get_user_by_email(email)
        if existing:
            from .exceptions import AuthError
            raise AuthError("user_already_exists", "User with this email already exists")

        # Create user and initial memories in a single transaction
        # This ensures atomic rollback if memory creation fails
        with self.session_manager.get_admin_session() as session:
            result = session.execute_single("""
                INSERT INTO users (
                    email, first_name, last_name, timezone, is_active, created_at,
                    subject_kind, demo_start_at, demo_expires_at
                )
                VALUES (
                    %(email)s, %(first_name)s, %(last_name)s, %(timezone)s, TRUE,
                    %(created_at)s, %(subject_kind)s, %(demo_start_at)s,
                    %(demo_expires_at)s
                )
                RETURNING id
            """, {
                'email': email,
                'first_name': first_name,
                'last_name': last_name,
                'timezone': timezone,
                'created_at': utc_now(),
                'subject_kind': subject_kind,
                'demo_start_at': demo_start_at,
                'demo_expires_at': demo_expires_at,
            })
            user_id = str(result['id'])

            # Create initial memories in same transaction
            memories = [
                {'text': f"The user's name is {first_name}.", 'importance': 0.9},
                {'text': f"The user is in {timezone} timezone.", 'importance': 0.8},
                {'text': f"The user's current focus is: {current_focus}", 'importance': 0.8},
                {'text': f"The user's email address is {email}.", 'importance': 0.7}
            ]
            for memory in memories:
                session.execute_update("""
                    INSERT INTO memories (user_id, text, importance_score)
                    VALUES (%(user_id)s, %(text)s, %(importance)s)
                """, {
                    'user_id': user_id,
                    'text': memory['text'],
                    'importance': memory['importance']
                })
            logger.info(f"Created user with {len(memories)} initial memories")

        return user_id

    def initialize_mira_account(
        self,
        user_id: str,
        first_name: str,
        current_focus: str,
    ) -> None:
        """Create the required Continuum and orientation content for a user."""
        from cns.infrastructure.continuum_repository import get_continuum_repository
        continuum_repo = get_continuum_repository()
        continuum = continuum_repo.get_continuum(user_id)
        if not continuum:
            continuum = continuum_repo.create_continuum(user_id)
            logger.debug(f"Continuum {continuum.id} created")

        self._prepopulate_welcome_content(user_id, first_name, current_focus)

    # Columns matching UserRecord fields (id cast to text for Pydantic str type)
    _USER_RECORD_COLUMNS = """
        id::text, email, first_name, last_name, is_active, created_at, last_login_at,
        webauthn_credentials, memory_manipulation_enabled,
        daily_manipulation_last_run, timezone, subject_kind, demo_expires_at
    """

    def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        """Get user by ID."""
        with self.session_manager.get_admin_session() as session:
            row = session.execute_single(
                f"SELECT {self._USER_RECORD_COLUMNS} FROM users WHERE id = %(user_id)s",
                {'user_id': user_id}
            )
            if row is None:
                return None
            return UserRecord(**row)

    def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        """Get user by email."""
        with self.session_manager.get_admin_session() as session:
            row = session.execute_single(
                f"SELECT {self._USER_RECORD_COLUMNS} FROM users WHERE email = %(email)s",
                {'email': email}
            )
            if row is None:
                return None
            return UserRecord(**row)

    def update_user_login(self, user_id: str) -> bool:
        """Update user's last login timestamp."""
        with self.session_manager.get_admin_session() as session:
            rows_updated = session.execute_update("""
                UPDATE users
                SET last_login_at = %(login_time)s
                WHERE id = %(user_id)s
            """, {
                'user_id': user_id,
                'login_time': utc_now()
            })
            return rows_updated > 0

    def create_magic_link(
        self,
        user_id: str,
        email: str,
        token_hash: str,
        expires_at: datetime
    ) -> str:
        """Create a magic link."""
        with self.session_manager.get_admin_session() as session:
            result = session.execute_single("""
                INSERT INTO magic_links (user_id, email, token_hash, expires_at, created_at)
                VALUES (%(user_id)s, %(email)s, %(token_hash)s, %(expires_at)s, %(created_at)s)
                RETURNING id
            """, {
                'user_id': user_id,
                'email': email,
                'token_hash': token_hash,
                'expires_at': expires_at,
                'created_at': utc_now()
            })
            return str(result['id'])

    def get_magic_link_by_token(self, token_hash: str) -> Optional[MagicLinkRecord]:
        """Get magic link by token hash."""
        with self.session_manager.get_admin_session() as session:
            row = session.execute_single(
                """SELECT id::text, user_id::text, email, token_hash,
                          expires_at, used_at, created_at
                   FROM magic_links WHERE token_hash = %(token_hash)s""",
                {'token_hash': token_hash}
            )
            if row is None:
                return None
            return MagicLinkRecord(**row)

    def consume_magic_link(self, token_hash: str) -> Optional[MagicLinkRecord]:
        """Atomically consume a valid, unused magic link."""
        now = utc_now()
        with self.session_manager.get_admin_session() as session:
            row = session.execute_single("""
                UPDATE magic_links
                SET used_at = %(used_at)s
                WHERE token_hash = %(token_hash)s
                  AND used_at IS NULL
                  AND expires_at > %(used_at)s
                RETURNING id::text, user_id::text, email, token_hash,
                          expires_at, used_at, created_at
            """, {
                'token_hash': token_hash,
                'used_at': now
            })
            if row is None:
                return None
            return MagicLinkRecord(**row)

    def cleanup_expired_magic_links(self) -> int:
        """Delete expired magic links."""
        with self.session_manager.get_admin_session() as session:
            rows_deleted = session.execute_update("""
                DELETE FROM magic_links
                WHERE expires_at < %(now)s
            """, {
                'now': utc_now()
            })
            logger.info(f"Cleanup: Removed {rows_deleted} expired magic links")
            return rows_deleted

    def _prepopulate_welcome_content(
        self,
        user_id: str,
        first_name: str,
        current_focus: str
    ) -> None:
        """
        Prepopulate welcome messages and domaindoc for new user.

        Called after user + memories are created atomically in create_user().
        Failures here are logged but don't prevent account creation.

        Args:
            user_id: UUID of the user
            first_name: User's first name
            current_focus: User's current focus
        """
        from cns.infrastructure.continuum_repository import get_continuum_repository
        from cns.infrastructure.continuum_pool import get_continuum_pool
        from uuid import uuid4

        # Get continuum and create unit of work
        continuum_repo = get_continuum_repository()
        continuum = continuum_repo.get_continuum(user_id)

        if not continuum:
            raise RuntimeError(f"Continuum not found for user {user_id}. Must create continuum before prepopulation.")

        pool = get_continuum_pool()

        # Create messages using native methods with delays for sequential timestamps
        from cns.core.message import Message
        import time
        from utils.user_context import set_current_user_id

        # Set user context for the prepopulation process
        # This is needed because save_messages_batch tries to increment activity day
        set_current_user_id(user_id)

        # Message 1: Beginning marker
        msg1 = Message(
            content=".. this is the beginning of the conversation. there are no messages older than this one ..",
            role="user",
            metadata={'system_generated': True, 'system_notification': True}
        )
        continuum._message_cache.append(msg1)
        # Create new UnitOfWork for each message to avoid duplicate key errors
        uow = pool.begin_work(continuum)
        uow.add_messages(msg1)
        uow.commit()
        time.sleep(0.1)  # 100ms delay

        # Message 2: Active segment sentinel
        segment_metadata = {
            'is_segment_boundary': True,
            'status': 'active',
            'segment_id': str(uuid4()),
            'segment_start_time': utc_now().isoformat(),
            'segment_end_time': utc_now().isoformat(),
            'segment_turn_count': 1,
            'tools_used': [],
            'memories_extracted': False,
            'domain_blocks_updated': False
        }
        msg2 = Message(
            content="[Segment in progress]",
            role="assistant",
            metadata=segment_metadata
        )
        continuum._message_cache.append(msg2)
        uow = pool.begin_work(continuum)  # New UnitOfWork for msg2
        uow.add_messages(msg2)
        uow.commit()
        time.sleep(0.1)  # 100ms delay

        # Message 3: System context message
        system_context = (
            f"MIRA THIS IS A SYSTEM MESSAGE TO HELP YOU ORIENT YOURSELF AND LEARN MORE ABOUT THE USER: "
            f"The user is named {first_name} and they said during the initial intake form that their current focus is: {current_focus}. "
            f"During this initial period, focus on being helpful and naturally learning about the user through "
            f"conversation. Ask clarifying questions only when necessary to provide good assistance, "
            f"not to gather information proactively."
        )
        msg3 = Message(
            content=system_context,
            role="user",
            metadata={'system_generated': True, 'system_notification': True}
        )
        continuum._message_cache.append(msg3)
        uow = pool.begin_work(continuum)  # New UnitOfWork for msg3
        uow.add_messages(msg3)
        uow.commit()
        time.sleep(0.1)  # 100ms delay

        # Message 4: User introduction
        msg4 = Message(
            content=f"Hi, my name is {first_name}.",
            role="user",
            metadata={'system_generated': True}
        )
        continuum._message_cache.append(msg4)
        uow = pool.begin_work(continuum)  # New UnitOfWork for msg4
        uow.add_messages(msg4)
        uow.commit()
        time.sleep(0.1)  # 100ms delay

        # Message 5: MIRA introduction
        mira_intro = f"""Hi {first_name}, nice to meet you. My name is MIRA and I'm a stateful AI assistant. That means that unlike AIs like ChatGPT or Claude, you and I will have one continuous conversation thread for as long as you have an account. Just log back into your MIRA instance and I'll be here ready to help. I extract and save facts & context automatically just like a person would. If you need to reference information from past sessions you can simply ask me about it and I'll be able to search our conversation history to bring myself up to speed. I look forward to working with you and I hope that you find value in our chats.

So, now that that's out of the way: What do you want to chat about first? I can help you with a work project, we can brainstorm an idea, or just chitchat for a bit."""

        msg5 = Message(
            content=mira_intro,
            role="assistant",
            metadata={'system_generated': True}
        )
        continuum._message_cache.append(msg5)
        uow = pool.begin_work(continuum)  # New UnitOfWork for msg5
        uow.add_messages(msg5)
        uow.commit()
        # No delay needed after last message

        # Clear user context after prepopulation
        from utils.user_context import clear_user_context
        clear_user_context()

        logger.debug("Prepopulated welcome content: 5 messages")

    def update_webauthn_credentials(self, user_id: str, credentials: dict) -> bool:
        """Update user's WebAuthn credentials.

        Args:
            user_id: User ID to update
            credentials: Dictionary of WebAuthn credentials

        Returns:
            True if update was successful
        """
        import json
        with self.session_manager.get_admin_session() as session:
            rows_updated = session.execute_update("""
                UPDATE users
                SET webauthn_credentials = %(credentials)s::jsonb
                WHERE id = %(user_id)s
            """, {
                'user_id': user_id,
                'credentials': json.dumps(credentials)
            })
            return rows_updated > 0

    # =========================================================================
    # API Token Operations (persistent, hashed storage)
    # =========================================================================

    def create_api_token(
        self,
        user_id: str,
        token_hash: str,
        name: str,
        expires_at: Optional[datetime] = None
    ) -> str:
        """
        Create a new API token record.

        Args:
            user_id: User who owns the token
            token_hash: SHA256 hash of the raw token (raw token never stored)
            name: Friendly name for the token
            expires_at: Optional expiration datetime (None = never expires)

        Returns:
            Token ID (UUID as string)

        Raises:
            AuthError: If token name already exists for this user
        """
        import psycopg
        from .exceptions import AuthError

        with self.session_manager.get_session(user_id) as session:
            try:
                result = session.execute_single("""
                    INSERT INTO api_tokens (user_id, token_hash, name, expires_at, created_at)
                    VALUES (%(user_id)s, %(token_hash)s, %(name)s, %(expires_at)s, %(created_at)s)
                    RETURNING id
                """, {
                    'user_id': user_id,
                    'token_hash': token_hash,
                    'name': name[:100] if name else 'API Token',
                    'expires_at': expires_at,
                    'created_at': utc_now()
                })
                return str(result['id'])
            except psycopg.errors.UniqueViolation:
                raise AuthError("duplicate_token_name", f"A token named '{name}' already exists")

    def get_api_token_by_hash(self, token_hash: str) -> Optional[dict]:
        """
        Get API token by its hash (for validation during API requests).

        Returns None if token not found, expired, or revoked.
        Uses admin session since this is called before user context is established.
        """
        with self.session_manager.get_admin_session() as session:
            return session.execute_single("""
                SELECT
                    api_tokens.id::text,
                    api_tokens.user_id::text,
                    api_tokens.name,
                    api_tokens.created_at,
                    api_tokens.expires_at,
                    users.subject_kind,
                    users.demo_expires_at
                FROM api_tokens
                JOIN users ON users.id = api_tokens.user_id
                WHERE api_tokens.token_hash = %(token_hash)s
                  AND api_tokens.revoked_at IS NULL
                  AND (api_tokens.expires_at IS NULL OR api_tokens.expires_at > %(now)s)
                  AND users.is_active = TRUE
                  AND (
                      users.subject_kind = 'member'
                      OR users.demo_expires_at > %(now)s
                  )
            """, {
                'token_hash': token_hash,
                'now': utc_now()
            })

    def list_api_tokens(self, user_id: str) -> list[dict]:
        """
        List all active API tokens for a user (metadata only, no hashes).

        Returns list of token records with id, name, created_at, expires_at.
        """
        with self.session_manager.get_session(user_id) as session:
            return session.execute_query("""
                SELECT id::text, name, created_at, expires_at
                FROM api_tokens
                WHERE revoked_at IS NULL
                  AND (expires_at IS NULL OR expires_at > %(now)s)
                ORDER BY created_at DESC
            """, {'now': utc_now()})

    def revoke_api_token(self, user_id: str, token_id: str) -> bool:
        """
        Revoke an API token (soft delete for audit trail).

        Returns True if token was revoked, False if not found.
        """
        with self.session_manager.get_session(user_id) as session:
            rows_updated = session.execute_update("""
                UPDATE api_tokens
                SET revoked_at = %(revoked_at)s
                WHERE id = %(token_id)s
                  AND revoked_at IS NULL
            """, {
                'token_id': token_id,
                'revoked_at': utc_now()
            })
            return rows_updated > 0

    def count_user_api_tokens(self, user_id: str) -> int:
        """Count active API tokens for a user (for rate limiting token creation)."""
        with self.session_manager.get_session(user_id) as session:
            result = session.execute_single("""
                SELECT COUNT(*) as count
                FROM api_tokens
                WHERE revoked_at IS NULL
                  AND (expires_at IS NULL OR expires_at > %(now)s)
            """, {'now': utc_now()})
            return result['count'] if result else 0
