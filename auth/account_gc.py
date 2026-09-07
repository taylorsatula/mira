"""
Account garbage collection service.

Removes unactivated accounts (users who signed up but never logged in)
after 24 hours to prevent bot spam and database bloat, and removes
expired demo subjects. This is what makes open multi-user signup safe to
leave on: an abandoned signup is transient state, not permanent rows.

Deletion goes through the injected `AccountProvisioner`, whose
`NullProvisioner.delete` reaches `local_teardown` — revoke sessions,
clear the per-user SQLite manager cache, remove the tool data directory,
and delete the row on an admin session.
"""
import logging
import time
from typing import Any, Dict, Optional

from auth.provisioning import AccountProvisioner, NullProvisioner
from utils.database_session_manager import get_shared_session_manager

logger = logging.getLogger(__name__)


class AccountGarbageCollectionService:
    """
    Garbage collection service for unactivated user accounts.

    Removes accounts that were created but never logged in (last_login_at IS NULL)
    after 24 hours. This prevents bot signups and abandoned accounts from
    accumulating in the database.

    The member branch is live in every deployment. The demo branch is inert
    while no code path can mint a `subject_kind='demo'` row (decision D12);
    it exists so a hand-created or future demo row is still collected at its
    expiry rather than leaking.
    """

    def __init__(
        self,
        session_manager=None,
        provisioner: Optional[AccountProvisioner] = None,
    ):
        """
        Initialize account GC service.

        Args:
            session_manager: Database session manager (uses shared if not provided)
            provisioner: Account teardown implementation (uses the OSS null
                provisioner — i.e. `local_teardown` — if not provided)
        """
        self.session_manager = session_manager or get_shared_session_manager()
        self.provisioner: AccountProvisioner = provisioner or NullProvisioner()

    def cleanup_unactivated_accounts(self) -> Dict[str, Any]:
        """
        Delete accounts that never logged in after 24 hours.

        Queries for users where:
        - last_login_at IS NULL (never verified magic link)
        - created_at < NOW() - INTERVAL '24 hours'

        Deletion cascades to all related data:
        - continuums, messages, magic_links, user_activity_days,
        - domain_knowledge_blocks, memories, entities

        The scan runs on the admin session (BYPASSRLS): it is a cross-user
        sweep with no request identity to scope it by.

        Returns:
            Dict with statistics (accounts_deleted, deleted_emails)

        Raises:
            RuntimeError: If database operation fails (infrastructure issue)
        """
        logger.debug("Starting account garbage collection scan")
        start_time = time.time()

        try:
            with self.session_manager.get_admin_session() as session:
                accounts_to_delete = session.execute_query("""
                    SELECT users.id::text, users.email, users.created_at
                    FROM users
                    WHERE (
                        users.subject_kind = 'member'
                        AND users.last_login_at IS NULL
                        AND users.created_at < NOW() - INTERVAL '24 hours'
                    )
                    OR (
                        users.subject_kind = 'demo'
                        AND users.demo_expires_at <= NOW()
                    )
                """)

                if not accounts_to_delete:
                    return {
                        'accounts_deleted': 0,
                        'deleted_emails': []
                    }

            deleted_emails: list[str] = []
            pending_emails: list[str] = []
            for account in accounts_to_delete:
                if self.provisioner.delete(account["id"]):
                    deleted_emails.append(account["email"])
                else:
                    pending_emails.append(account["email"])

            duration = time.time() - start_time
            logger.info(
                "Account GC: deleted %d accounts; %d pending retry in %.1fs",
                len(deleted_emails),
                len(pending_emails),
                duration,
            )
            logger.debug("Deleted account emails: %s", deleted_emails)

            return {
                "accounts_deleted": len(deleted_emails),
                "deleted_emails": deleted_emails,
                "cleanup_pending_emails": pending_emails,
            }

        except Exception as e:
            # Log full exception before re-raising
            logger.error(
                f"Account GC failed: {type(e).__name__}: {e}",
                exc_info=True
            )
            raise


# Singleton instance
_account_gc_service = None


def get_account_gc_service() -> AccountGarbageCollectionService:
    """Get or create singleton AccountGarbageCollectionService instance."""
    global _account_gc_service
    if _account_gc_service is None:
        _account_gc_service = AccountGarbageCollectionService()
        logger.debug("AccountGarbageCollectionService singleton initialized")
    return _account_gc_service


def register_account_gc_job(scheduler_service) -> None:
    """
    Register account garbage collection job with scheduler.

    Runs daily at 3:00 AM UTC to delete accounts that signed up
    but never logged in after 24 hours.

    Args:
        scheduler_service: System scheduler service

    Raises:
        ImportError: If apscheduler is not installed
        RuntimeError: If scheduler service is not properly initialized
    """
    from apscheduler.triggers.cron import CronTrigger

    account_gc_service = get_account_gc_service()

    scheduler_service.register_job(
        job_id="account_garbage_collection",
        func=account_gc_service.cleanup_unactivated_accounts,
        trigger=CronTrigger.from_crontab("0 3 * * *"),  # Run at 3:00 AM UTC daily
        component="auth",
        description="Delete expired demos and unactivated accounts"
    )

    logger.info("Account garbage collection job registered (daily at 3:00 AM UTC)")
