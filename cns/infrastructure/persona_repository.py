"""PostgreSQL repository for immutable Persona revisions and evidence."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from utils.database_session_manager import get_shared_session_manager
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)

PersonaRevisionSource = Literal["baseline", "automatic", "user", "rollback"]
PersonaOutcome = Literal["alignment", "misalignment", "contextual_pass"]
PersonaStrength = Literal["strong", "moderate", "mild"]


class PersonaSignal(BaseModel):
    """One evidence item about MIRA's behavior in a collapsed segment."""

    user_id: UUID = Field(description="User whose MIRA behavior was evaluated")
    continuum_id: UUID = Field(description="Continuum containing the evaluated segment")
    segment_id: UUID = Field(description="Collapsed segment identifier")
    behavioral_section: str = Field(min_length=1, description="Behavioral-contract section")
    outcome: PersonaOutcome = Field(description="Observed contract outcome")
    strength: PersonaStrength = Field(description="Evidence strength")
    evidence: str = Field(min_length=1, description="Specific observed behavior")


class PersonaRevision(BaseModel):
    """Immutable Persona revision returned by the repository."""

    id: UUID
    user_id: UUID
    revision_number: int
    directives: str
    source: PersonaRevisionSource
    parent_revision_id: UUID | None
    evidence_ids: list[UUID]
    created_at: datetime


class PersonaRepository:
    """Own all SQL for Persona revisions, state, and evidence rows.

    Persona evidence lives in ``persona_signals``, not ``feedback_signals``:
    the user-model table shares no column with it beyond the keys (D1 keeps both
    subsystems), and the merge of the two column sets would be a corrupt hybrid.
    """

    def __init__(self) -> None:
        self._sessions = get_shared_session_manager()

    def get_current_revision(self, user_id: str) -> PersonaRevision:
        with self._sessions.get_session(user_id) as session:
            row = session.execute_single(
                """
                SELECT revision.*
                FROM persona_state state
                JOIN persona_revisions revision ON revision.id = state.current_revision_id
                WHERE state.user_id = %s
                """,
                (user_id,),
            )
        if not row:
            raise RuntimeError(f"Persona baseline missing for user {user_id}")
        return PersonaRevision(**row)

    def get_revision(self, user_id: str, revision_id: UUID) -> PersonaRevision:
        with self._sessions.get_session(user_id) as session:
            row = session.execute_single(
                "SELECT * FROM persona_revisions WHERE user_id = %s AND id = %s",
                (user_id, revision_id),
            )
        if not row:
            raise ValueError(f"Persona revision {revision_id} not found")
        return PersonaRevision(**row)

    def list_revisions(self, user_id: str) -> list[PersonaRevision]:
        with self._sessions.get_session(user_id) as session:
            rows = session.execute_query(
                """
                SELECT * FROM persona_revisions
                WHERE user_id = %s
                ORDER BY revision_number DESC
                """,
                (user_id,),
            )
        return [PersonaRevision(**row) for row in rows]

    def save_signals(self, user_id: str, signals: list[PersonaSignal]) -> int:
        if not signals:
            return 0
        with self._sessions.get_session(user_id) as session:
            with session.transaction():
                for signal in signals:
                    if str(signal.user_id) != user_id:
                        raise ValueError("Persona signal user_id does not match repository scope")
                    session.execute_single(
                        """
                        INSERT INTO persona_signals (
                            user_id, continuum_id, segment_id, behavioral_section,
                            outcome, strength, evidence, evaluated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            signal.continuum_id,
                            signal.segment_id,
                            signal.behavioral_section,
                            signal.outcome,
                            signal.strength,
                            signal.evidence,
                            utc_now(),
                        ),
                    )
                session.execute_single(
                    """
                    UPDATE persona_state
                    SET latest_evaluated_segment_id = %s
                    WHERE user_id = %s
                    """,
                    (signals[0].segment_id, user_id),
                )
        return len(signals)

    def segment_was_evaluated(self, user_id: str, segment_id: UUID) -> bool:
        with self._sessions.get_session(user_id) as session:
            row = session.execute_single(
                """
                SELECT latest_evaluated_segment_id = %s AS evaluated
                FROM persona_state WHERE user_id = %s
                """,
                (segment_id, user_id),
            )
        return bool(row and row["evaluated"])

    def mark_segment_evaluated(self, user_id: str, segment_id: UUID) -> None:
        with self._sessions.get_session(user_id) as session:
            session.execute_single(
                """
                UPDATE persona_state
                SET latest_evaluated_segment_id = %s
                WHERE user_id = %s
                """,
                (segment_id, user_id),
            )

    def get_unconsumed_signals(self, user_id: str) -> list[dict]:
        with self._sessions.get_session(user_id) as session:
            return session.execute_query(
                """
                SELECT id, behavioral_section, outcome, strength, evidence, evaluated_at,
                       segment_id, continuum_id
                FROM persona_signals
                WHERE user_id = %s AND consumed_by_revision_id IS NULL
                ORDER BY evaluated_at, id
                """,
                (user_id,),
            )

    def refinement_due(self, user_id: str, activity_day: int, interval: int = 7) -> bool:
        if activity_day <= 0 or activity_day % interval != 0:
            return False
        with self._sessions.get_session(user_id) as session:
            row = session.execute_single(
                """
                SELECT refinement_checkpoint_activity_day
                FROM persona_state WHERE user_id = %s
                """,
                (user_id,),
            )
        if not row:
            raise RuntimeError(f"Persona state missing for user {user_id}")
        return activity_day > row["refinement_checkpoint_activity_day"]

    def append_revision(
        self,
        user_id: str,
        directives: str,
        source: PersonaRevisionSource,
        *,
        evidence_ids: list[UUID] | None = None,
        expected_parent_revision_id: UUID | None = None,
        activity_day_checkpoint: int | None = None,
        audit_metadata: dict | None = None,
    ) -> PersonaRevision:
        evidence_ids = list(evidence_ids or [])
        revision_id = uuid4()

        with self._sessions.get_session(user_id) as session:
            with session.transaction():
                state = session.execute_single(
                    """
                    SELECT current_revision_id
                    FROM persona_state
                    WHERE user_id = %s
                    FOR UPDATE
                    """,
                    (user_id,),
                )
                if not state:
                    raise RuntimeError(f"Persona state missing for user {user_id}")
                current_revision_id = state["current_revision_id"]
                if (
                    expected_parent_revision_id is not None
                    and current_revision_id != expected_parent_revision_id
                ):
                    raise ValueError(
                        "Persona changed after this preview was created; generate a new preview"
                    )

                next_number = session.execute_single(
                    """
                    SELECT COALESCE(MAX(revision_number), 0) + 1 AS revision_number
                    FROM persona_revisions WHERE user_id = %s
                    """,
                    (user_id,),
                )["revision_number"]

                # mira-OSS carries no audit_events journal -- the greenfield schema
                # omits it deliberately and this codebase logs diagnostics instead. The
                # revision row is itself append-only history, so the actor attribution
                # that a journal row would have carried is logged here instead.
                logger.info(
                    "PERSONA REVISION action=persona.%s actor=%s subject=%s revision=%s "
                    "parent=%s evidence=%d details=%s",
                    source,
                    "user" if source in {"user", "rollback"} else "system",
                    user_id,
                    revision_id,
                    current_revision_id,
                    len(evidence_ids),
                    json.dumps(audit_metadata or {}),
                )

                row = session.execute_single(
                    """
                    INSERT INTO persona_revisions (
                        id, user_id, revision_number, directives, source,
                        parent_revision_id, evidence_ids
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        revision_id,
                        user_id,
                        next_number,
                        directives,
                        source,
                        current_revision_id,
                        evidence_ids,
                    ),
                )

                if activity_day_checkpoint is None:
                    session.execute_single(
                        "UPDATE persona_state SET current_revision_id = %s WHERE user_id = %s",
                        (revision_id, user_id),
                    )
                else:
                    session.execute_single(
                        """
                        UPDATE persona_state
                        SET current_revision_id = %s,
                            refinement_checkpoint_activity_day = %s
                        WHERE user_id = %s
                        """,
                        (revision_id, activity_day_checkpoint, user_id),
                    )

                if evidence_ids:
                    session.execute_single(
                        """
                        UPDATE persona_signals
                        SET consumed_by_revision_id = %s
                        WHERE user_id = %s AND id = ANY(%s)
                          AND consumed_by_revision_id IS NULL
                        """,
                        (revision_id, user_id, evidence_ids),
                    )

        return PersonaRevision(**row)

    def mark_refinement_attempt(
        self,
        user_id: str,
        activity_day: int,
        *,
        outcome: Literal["no_evidence", "validation_failed"],
        details: dict | None = None,
    ) -> None:
        with self._sessions.get_session(user_id) as session:
            with session.transaction():
                session.execute_single(
                    """
                    UPDATE persona_state
                    SET refinement_checkpoint_activity_day = %s
                    WHERE user_id = %s
                    """,
                    (activity_day, user_id),
                )
        logger.info(
            "PERSONA REFINEMENT outcome=persona.%s subject=%s activity_day=%s details=%s",
            outcome,
            user_id,
            activity_day,
            json.dumps(details or {}),
        )
