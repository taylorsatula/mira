"""
Assessment signals — the behavioral evidence the user-model pipeline
synthesizes from — persisted to `feedback_signals`.

Signals arrive as `AssessmentSignal` objects from
`cns.services.assessment_extractor`, the one upward import in this package.
That type carries no user field, so `user_id` comes from the ambient context
variable: a list produced for one user is written under whoever is current.

`get_unsynthesized_signals()` returns only rows with `synthesized = FALSE`, so
`mark_signals_synthesized()` is what removes evidence from the next synthesis
run's input. Marking before the synthesis output is durably stored drops that
evidence — nothing brings the rows back into the unsynthesized set.
"""
from __future__ import annotations

import logging
from typing import TypedDict
from uuid import UUID

from cns.services.assessment_extractor import AssessmentSignal
from utils.database_session_manager import get_shared_session_manager
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)


class FeedbackSignalRow(TypedDict):
    """Row shape returned by get_unsynthesized_signals()."""
    id: UUID
    signal_type: str
    section_id: str
    strength: str
    evidence: str
    extracted_at: str
    segment_id: str
    continuum_id: str


class FeedbackRepository:
    """
    Repository for assessment signal persistence.

    Stores signals in the feedback_signals table for later user model synthesis.
    """

    def save_signals(self, signals: list[AssessmentSignal]) -> int:
        """
        Save assessment signals to database.

        Uses the current user context for user_id.

        Args:
            signals: List of AssessmentSignal objects to persist

        Returns:
            Number of signals successfully saved
        """
        if not signals:
            return 0

        user_id = get_current_user_id()
        session_manager = get_shared_session_manager()

        saved_count = 0

        with session_manager.get_session(user_id) as session:
            for signal in signals:
                session.execute_single("""
                    INSERT INTO feedback_signals (
                        user_id, segment_id, continuum_id, signal_type,
                        section_id, strength, evidence, extracted_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    str(signal.segment_id),
                    str(signal.continuum_id),
                    signal.signal_type,
                    signal.section_id,
                    signal.strength,
                    signal.evidence,
                    signal.extracted_at
                ))
                saved_count += 1

        logger.info("Saved %d/%d assessment signals for user %s", saved_count, len(signals), user_id)
        return saved_count

    def get_unsynthesized_signals(self, user_id: str) -> list[FeedbackSignalRow]:
        """
        Get all unsynthesized assessment signals for a user.

        Args:
            user_id: User ID

        Returns:
            List of signal dicts with id, signal_type, section_id, strength,
            evidence, extracted_at, segment_id, continuum_id
        """
        session_manager = get_shared_session_manager()

        with session_manager.get_session(user_id) as session:
            rows = session.execute_query("""
                SELECT id, signal_type, section_id, strength, evidence,
                       extracted_at, segment_id, continuum_id
                FROM feedback_signals
                WHERE user_id = %s AND synthesized = FALSE
                ORDER BY extracted_at ASC
            """, (user_id,))

            return [dict(row) for row in rows]

    def mark_signals_synthesized(self, user_id: str, signal_ids: list[UUID]) -> int:
        """
        Mark assessment signals as synthesized after user model synthesis.

        Args:
            user_id: User ID
            signal_ids: List of signal UUIDs to mark

        Returns:
            Number of signals marked
        """
        if not signal_ids:
            return 0

        session_manager = get_shared_session_manager()

        with session_manager.get_session(user_id) as session:
            id_strings = [str(sid) for sid in signal_ids]
            placeholders = ','.join(['%s'] * len(id_strings))

            session.execute_single(f"""
                UPDATE feedback_signals
                SET synthesized = TRUE
                WHERE user_id = %s AND id IN ({placeholders})
            """, (user_id, *id_strings))

            return len(signal_ids)
