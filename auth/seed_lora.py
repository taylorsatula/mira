"""
Seed feedback tracking for a newly provisioned account.

Initializes the `feedback_synthesis_tracking` row that the user-model and
assessment pipeline later populates with observations; until that pipeline runs,
the row is what makes the account known to it at all. The sole caller is
`AuthService._initialize_account`, so every provisioning path — public signup and
the single-mode local bootstrap alike — seeds it.

The name is inherited rather than descriptive: what this call creates is the
generic `feedback_synthesis_tracking` row, and the `lora` spelling lives on in
other surfaces too (`rg 'lora'` also hits the browser settings' `lora` tab, which
is unrelated), so a rename gets half applied.

A failure raises `RuntimeError` rather than returning. Account creation fails
loudly instead of leaving behind an account the user model has no row for.
"""

import logging

from cns.infrastructure.feedback_tracker import FeedbackTracker

logger = logging.getLogger(__name__)


def seed_lora_postgres(user_id: str) -> None:
    """
    Initialize feedback tracking in PostgreSQL for a new user.

    Creates a feedback_synthesis_tracking row with defaults.
    The user model will be populated by the assessment/synthesis pipeline
    after sufficient use-days.

    Args:
        user_id: UUID of the user (as string)

    Raises:
        RuntimeError: If database initialization fails
    """
    try:
        tracker = FeedbackTracker()
        tracker.initialize_user(user_id)
        logger.debug("Initialized feedback tracking")

    except Exception as e:
        logger.error("Failed to initialize feedback tracking: %s", e, exc_info=True)
        raise RuntimeError(f"Feedback tracking initialization failed: {e}")
