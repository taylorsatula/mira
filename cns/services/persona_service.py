"""Persona evaluation, automatic evolution, and manual revision workflows."""

from __future__ import annotations

import logging
import re
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from clients.llm_provider import LLMProvider
from cns.core.message import Message, preprocess_content_blocks
from cns.infrastructure.persona_repository import (
    PersonaRepository,
    PersonaRevision,
    PersonaSignal,
)
from config import config
from config.prompts import load_prompt
from utils.user_activity import get_user_cumulative_activity_days

logger = logging.getLogger(__name__)

PERSONA_REFINEMENT_USE_DAYS = 7
PERSONA_VALIDATION_ATTEMPTS = 3
PERSONA_PREVIEW_TTL_SECONDS = 600
PERSONA_PREVIEW_PREFIX = "persona_preview"


class PersonaPreview(BaseModel):
    """Server-side manual revision preview stored in Valkey."""

    directives: str = Field(description="Validated proposed Persona directives")
    parent_revision_id: UUID = Field(description="Current revision when preview was created")


class PersonaValidation(BaseModel):
    """Parsed critic result for a candidate Persona revision."""

    passed: bool
    feedback: str = ""


class PersonaParseError(ValueError):
    """Raised when a model response violates the Persona XML contract."""


class PersonaService:
    """Evolve and edit MIRA's behavioral directives for one user."""

    def __init__(
        self,
        repository: PersonaRepository | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.repository = repository or PersonaRepository()
        self.llm = llm_provider or LLMProvider()
        self._evaluation_system = load_prompt("persona_evaluation_system.txt")
        self._evaluation_user = load_prompt("persona_evaluation_user.txt")
        self._refinement_system = load_prompt("persona_refinement_system.txt")
        self._refinement_user = load_prompt("persona_refinement_user.txt")
        self._manual_user = load_prompt("persona_manual_refinement_user.txt")
        self._critic_system = load_prompt("persona_critic_system.txt")
        self._critic_user = load_prompt("persona_critic_user.txt")

    def get_current(self, user_id: str) -> PersonaRevision:
        return self.repository.get_current_revision(user_id)

    def get_history(self, user_id: str) -> list[PersonaRevision]:
        return self.repository.list_revisions(user_id)

    def evaluate_segment(
        self,
        user_id: str,
        messages: list[Message],
        segment_id: UUID,
        continuum_id: UUID,
    ) -> list[PersonaSignal]:
        """Evaluate MIRA's behavior once for a collapsed segment."""
        if self.repository.segment_was_evaluated(user_id, segment_id):
            return []

        current = self.repository.get_current_revision(user_id)
        user_prompt = self._evaluation_user.format(
            behavioral_contract=config.system_prompt,
            current_persona=current.directives or "(no additional directives)",
            conversation=self._format_messages(messages),
        )
        response = self.llm.generate_response(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=self._evaluation_system,
            model_config="primary",
        )
        signals = self._parse_evaluation(
            self.llm.extract_text_content(response),
            user_id=user_id,
            segment_id=segment_id,
            continuum_id=continuum_id,
        )
        if signals:
            self.repository.save_signals(user_id, signals)
        else:
            self.repository.mark_segment_evaluated(user_id, segment_id)
        return signals

    def refine_automatically_if_due(self, user_id: str) -> PersonaRevision | None:
        """Publish one validated automatic revision on the seven-use-day boundary."""
        activity_day = get_user_cumulative_activity_days(user_id)
        if not self.repository.refinement_due(
            user_id,
            activity_day,
            PERSONA_REFINEMENT_USE_DAYS,
        ):
            return None

        evidence = self.repository.get_unconsumed_signals(user_id)
        if not evidence:
            self.repository.mark_refinement_attempt(
                user_id,
                activity_day,
                outcome="no_evidence",
            )
            return None

        current = self.repository.get_current_revision(user_id)
        candidate = ""
        critic_feedback = "(none)"
        last_validation = PersonaValidation(passed=False, feedback="No candidate generated")

        for _attempt in range(PERSONA_VALIDATION_ATTEMPTS):
            try:
                candidate = self._generate_candidate(
                    current.directives,
                    evidence,
                    critic_feedback=critic_feedback,
                )
            except PersonaParseError as error:
                last_validation = PersonaValidation(passed=False, feedback=str(error))
                critic_feedback = str(error)
                continue
            last_validation = self._validate_candidate(candidate)
            if last_validation.passed:
                revision = self.repository.append_revision(
                    user_id,
                    candidate,
                    "automatic",
                    evidence_ids=[UUID(str(row["id"])) for row in evidence],
                    expected_parent_revision_id=current.id,
                    activity_day_checkpoint=activity_day,
                    audit_metadata={"evidence_count": len(evidence)},
                )
                self._invalidate_cache(user_id)
                return revision
            critic_feedback = last_validation.feedback

        self.repository.mark_refinement_attempt(
            user_id,
            activity_day,
            outcome="validation_failed",
            details={"feedback": last_validation.feedback[:1000]},
        )
        logger.error(
            "Persona automatic refinement failed validation for user %s: %s",
            user_id,
            last_validation.feedback,
        )
        return None

    def create_preview(self, user_id: str, instructions: str) -> dict[str, str]:
        """Generate and store a validated manual Persona revision preview."""
        instructions = instructions.strip()
        if not instructions:
            raise ValueError("Persona revision instructions cannot be empty")

        current = self.repository.get_current_revision(user_id)
        candidate = ""
        critic_feedback = "(none)"
        validation = PersonaValidation(passed=False, feedback="No candidate generated")
        for _attempt in range(PERSONA_VALIDATION_ATTEMPTS):
            prompt = self._manual_user.format(
                current_persona=current.directives or "(no additional directives)",
                user_instructions=instructions,
                critic_feedback=critic_feedback,
            )
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=self._refinement_system,
                model_config="primary",
            )
            try:
                candidate = self._parse_persona(self.llm.extract_text_content(response))
            except PersonaParseError as error:
                validation = PersonaValidation(passed=False, feedback=str(error))
                critic_feedback = str(error)
                continue
            validation = self._validate_candidate(candidate)
            if validation.passed:
                break
            critic_feedback = validation.feedback
        else:
            raise ValueError(
                "Persona revision failed validation after three attempts: "
                f"{validation.feedback}"
            )

        preview_id = str(uuid4())
        preview = PersonaPreview(
            directives=candidate,
            parent_revision_id=current.id,
        )
        from clients.valkey_client import get_valkey_client

        get_valkey_client().setex(
            f"{PERSONA_PREVIEW_PREFIX}:{user_id}:{preview_id}",
            PERSONA_PREVIEW_TTL_SECONDS,
            preview.model_dump_json(),
        )
        return {"preview_id": preview_id, "proposed": candidate}

    def accept_preview(self, user_id: str, preview_id: str) -> PersonaRevision:
        preview = self._pop_preview(user_id, preview_id)
        revision = self.repository.append_revision(
            user_id,
            preview.directives,
            "user",
            expected_parent_revision_id=preview.parent_revision_id,
        )
        self._invalidate_cache(user_id)
        return revision

    def decline_preview(self, user_id: str, preview_id: str) -> None:
        self._pop_preview(user_id, preview_id)

    def rollback(self, user_id: str, revision_id: UUID) -> PersonaRevision:
        target = self.repository.get_revision(user_id, revision_id)
        current = self.repository.get_current_revision(user_id)
        revision = self.repository.append_revision(
            user_id,
            target.directives,
            "rollback",
            expected_parent_revision_id=current.id,
            audit_metadata={"restored_revision_id": str(target.id)},
        )
        self._invalidate_cache(user_id)
        return revision

    def _generate_candidate(
        self,
        current_directives: str,
        evidence: list[dict],
        *,
        critic_feedback: str,
    ) -> str:
        evidence_text = "\n".join(
            f"- [{row['behavioral_section']}; {row['outcome']}; {row['strength']}] {row['evidence']}"
            for row in evidence
        )
        prompt = self._refinement_user.format(
            current_persona=current_directives or "(no additional directives)",
            evidence=evidence_text,
            critic_feedback=critic_feedback,
        )
        response = self.llm.generate_response(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=self._refinement_system,
            model_config="primary",
        )
        return self._parse_persona(self.llm.extract_text_content(response))

    def _validate_candidate(self, candidate: str) -> PersonaValidation:
        if not candidate.strip():
            return PersonaValidation(passed=False, feedback="Persona directives are empty")
        response = self.llm.generate_response(
            messages=[
                {
                    "role": "user",
                    "content": self._critic_user.format(candidate_persona=candidate),
                }
            ],
            system_prompt=self._critic_system,
            model_config="primary",
        )
        output = self.llm.extract_text_content(response)
        status = re.search(r'<mira:persona_review\s+status="(pass|fail)"', output)
        if not status:
            return PersonaValidation(
                passed=False,
                feedback="Critic response omitted mira:persona_review status",
            )
        if status.group(1) == "pass":
            return PersonaValidation(passed=True)
        issues = re.findall(r"<mira:issue>(.*?)</mira:issue>", output, re.DOTALL)
        return PersonaValidation(
            passed=False,
            feedback="\n".join(issue.strip() for issue in issues) or "Critic rejected candidate",
        )

    def _pop_preview(self, user_id: str, preview_id: str) -> PersonaPreview:
        if not preview_id.strip():
            raise ValueError("preview_id is required")
        from clients.valkey_client import get_valkey_client

        valkey = get_valkey_client()
        key = f"{PERSONA_PREVIEW_PREFIX}:{user_id}:{preview_id}"
        raw = valkey.get(key)
        if raw is None:
            raise ValueError("Persona preview not found; it may have expired")
        valkey.delete(key)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return PersonaPreview.model_validate_json(raw)

    @staticmethod
    def _parse_persona(output: str) -> str:
        match = re.search(r"<mira:persona>(.*?)</mira:persona>", output, re.DOTALL)
        if not match:
            raise PersonaParseError("Persona model response omitted <mira:persona>")
        return match.group(1).strip()

    @staticmethod
    def _parse_evaluation(
        output: str,
        *,
        user_id: str,
        segment_id: UUID,
        continuum_id: UUID,
    ) -> list[PersonaSignal]:
        if re.fullmatch(r"\s*<mira:persona_evaluation\s*/>\s*", output):
            return []
        if not re.search(
            r"<mira:persona_evaluation>.*</mira:persona_evaluation>",
            output,
            re.DOTALL,
        ):
            raise PersonaParseError(
                "Persona evaluator response omitted <mira:persona_evaluation>"
            )

        signals: list[PersonaSignal] = []
        pattern = re.compile(
            r'<mira:signal\s+section="([^"]+)"\s+'
            r'outcome="(alignment|misalignment|contextual_pass)"\s+'
            r'strength="(strong|moderate|mild)">\s*'
            r'<evidence>(.*?)</evidence>\s*</mira:signal>',
            re.DOTALL,
        )
        for section, outcome, strength, evidence in pattern.findall(output):
            signals.append(
                PersonaSignal(
                    user_id=UUID(user_id),
                    continuum_id=continuum_id,
                    segment_id=segment_id,
                    behavioral_section=section.strip(),
                    outcome=outcome,
                    strength=strength,
                    evidence=evidence.strip(),
                )
            )
        if "<mira:signal" in output and not signals:
            raise PersonaParseError("Persona evaluator returned malformed signal XML")
        return signals

    @staticmethod
    def _format_messages(messages: list[Message]) -> str:
        rendered: list[str] = []
        for message in messages:
            if message.metadata.get("system_notification"):
                continue
            content = preprocess_content_blocks(message.content)
            text = " ".join(content.text_parts)
            if content.image_count:
                text = f"[{content.image_count} image(s)] {text}".strip()
            if text:
                rendered.append(f"<{message.role}>{text}</{message.role}>")
        return "\n".join(rendered)

    @staticmethod
    def _invalidate_cache(user_id: str) -> None:
        from clients.valkey_client import get_valkey_client
        from working_memory.trinkets.base import TRINKET_KEY_PREFIX

        # Field name is Persona's own slot. Upstream invalidates
        # "behavioral_directives" because crm_mira deleted the user model and took over
        # its slot; here that field belongs to LoraTrinket, and clearing it would drop
        # the user model's cached section while leaving stale Persona directives.
        get_valkey_client().hdel_with_retry(
            f"{TRINKET_KEY_PREFIX}:{user_id}",
            "persona_directives",
        )
