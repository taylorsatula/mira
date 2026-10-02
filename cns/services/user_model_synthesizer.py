"""
User model synthesizer for the user model pipeline.

Synthesizes assessment signals into a descriptive user model: observations about the
user anchored to system prompt sections. Includes a critic validation loop that
catches observation laundering, personality labels, and internal contradictions.
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Literal, Optional

from cns.infrastructure.feedback_repository import FeedbackRepository, FeedbackSignalRow
from cns.infrastructure.feedback_tracker import FeedbackTracker
from cns.services.system_prompt_parser import (
    format_section_list,
    get_assessable_section_ids,
    get_assessable_sections,
    validate_section_anchors,
)
from clients.llm_provider import LLMProvider, get_llm_provider
from config import config

logger = logging.getLogger(__name__)

CRITIC_MAX_ATTEMPTS = 3


class CriticExhaustedError(Exception):
    """Synthesis ended with no candidate accepted by the critic.

    Raised so callers can distinguish exhaustion from success and skip
    their destructive post-synthesis steps (marking signals synthesized,
    consuming check-in feedback) — inputs stay intact for a later retry.
    """


@dataclass
class UserObservation:
    """A parsed observation from the user model."""
    section_id: str
    observation: str
    confidence: Literal['high', 'moderate', 'low']
    changelog: str


@dataclass
class CheckinTopic:
    """A check-in topic for behavioral debrief."""
    section_id: str
    reason: str


@dataclass
class SynthesisResult:
    """Complete result from user model synthesis."""
    observations: List[UserObservation]
    checkin_topics: List[CheckinTopic]
    raw_xml: str


@dataclass
class CriticResult:
    """Result of critic validation on a candidate user model."""
    passed: bool
    feedback: str


class UserModelSynthesizer:
    """
    Synthesizes assessment signals into a user model with critic validation.

    Pipeline:
    1. Fetch unsynthesized signals
    2. Call synthesis LLM to produce candidate user model
    3. Run critic validation (Haiku) to check for quality issues
    4. If critic fails: rerun synthesis with feedback (up to 3 attempts)
    5. Return final user model
    """

    def __init__(
        self,
        feedback_repo: FeedbackRepository,
        llm_provider: Optional[LLMProvider] = None
    ):
        self.feedback_repo = feedback_repo
        self.llm_provider = llm_provider or get_llm_provider()
        self._load_prompts()

        # Pre-compute section list for critic and synthesis prompt context
        raw_prompt = config.system_prompt
        sections = get_assessable_sections(raw_prompt)
        self._section_list = format_section_list(sections)

        logger.info("UserModelSynthesizer initialized")

    def _load_prompts(self) -> None:
        """Load synthesis and critic prompts.

        The synthesis system prompt is a template: its changelog examples'
        section names ({example_section_a}/{example_section_b}) are filled
        from the assessable section vocabulary — the same source
        validate_section_anchors() enforces — so the examples the model
        imitates can never drift from the taxonomy the code checks.
        """
        from config.prompts.loader import load_prompt
        section_ids = get_assessable_section_ids(config.system_prompt)
        if len(section_ids) < 2:
            raise ValueError(
                f"System prompt defines {len(section_ids)} assessable section(s); "
                "the synthesis prompt examples need at least 2"
            )
        self._synthesis_system_prompt = load_prompt("user_model_synthesis_system.txt").format(
            example_section_a=section_ids[0],
            example_section_b=section_ids[1],
        )
        self._synthesis_user_template = load_prompt("user_model_synthesis_user.txt")
        self._critic_system_prompt = load_prompt("user_model_critic_system.txt")
        self._critic_user_template = load_prompt("user_model_critic_user.txt")

    def synthesize(self, user_id: str, current_model_xml: Optional[str] = None) -> SynthesisResult:
        """
        Synthesize a user model from accumulated assessment signals.

        Args:
            user_id: User whose signals to synthesize
            current_model_xml: Current user model XML (for evolutionary continuity)

        Returns:
            SynthesisResult with observations, checkin topics, and raw XML

        Raises:
            CriticExhaustedError: The critic accepted no candidate after
                CRITIC_MAX_ATTEMPTS — inputs (signals, check-in feedback) are
                preserved for retry; nothing is published.
            Exception: On other synthesis failure (caller handles)
        """
        signals = self.feedback_repo.get_unsynthesized_signals(user_id)

        if not signals and not current_model_xml:
            logger.info("No signals or existing model for user %s", user_id)
            return SynthesisResult(observations=[], checkin_topics=[], raw_xml="")

        # Peek at any pending check-in feedback without consuming it.
        # The destructive consume (get_and_clear_checkin_response) is deferred
        # until after the synthesis LLM call, the critic validation loop, and
        # XML parsing have all succeeded — any raise on that path leaves the
        # user's check-in response intact instead of permanently losing it.
        tracker = FeedbackTracker()
        checkin_feedback = tracker.get_checkin_response(user_id)

        # Format signals grouped by section
        signals_text = self._format_signals_by_section(signals) if signals else "No new signals."
        current_model_text = current_model_xml if current_model_xml else "No existing user model (first synthesis)."

        # Initial synthesis
        candidate_xml = self._run_synthesis(signals_text, current_model_text, checkin_feedback)

        # Critic validation loop
        for attempt in range(CRITIC_MAX_ATTEMPTS):
            # Deterministic section-anchor validation runs BEFORE the LLM
            # critic: an invalid anchor is a code-checked failure, never left
            # for the critic to catch. On failure it feeds the same retry loop
            # (feedback names the valid set); exhaustion raises via the
            # circuit breaker below, so no invalid anchor is ever published.
            anchor_result = self._validate_section_anchors(candidate_xml)
            if anchor_result.passed:
                critic = self._validate_with_critic(candidate_xml)
            else:
                critic = anchor_result

            if critic.passed:
                logger.info("User model passed critic validation (attempt %d)", attempt + 1)
                break

            logger.warning("Critic rejected user model (attempt %d): %s", attempt + 1, critic.feedback[:200])

            if attempt < CRITIC_MAX_ATTEMPTS - 1:
                candidate_xml = self._rerun_synthesis_with_feedback(
                    critic.feedback, signals_text, current_model_text, checkin_feedback
                )
        else:
            # Circuit breaker: the critic never accepted any candidate, so this
            # is a distinct failure status, not a success. Raising follows the
            # method's existing failure contract ("Raises: Exception — caller
            # handles"): the caller's except path skips mark_synthesized and
            # mark_signals_synthesized, and the destructive check-in consume
            # below is never reached (consume-after-acceptance). Signals
            # and check-in feedback are preserved for the next synthesis run;
            # no known-suspect candidate or stale model is published as new.
            logger.warning(
                "Critic validation exhausted %d attempts for user %s, aborting synthesis",
                CRITIC_MAX_ATTEMPTS, user_id
            )
            raise CriticExhaustedError(
                f"Critic validation exhausted {CRITIC_MAX_ATTEMPTS} attempts; "
                "no candidate accepted. Inputs preserved for retry."
            )

        result = self._parse_user_model_xml(candidate_xml)

        # Synthesis fully succeeded (LLM call, critic loop, and parsing all
        # complete): now atomically consume the check-in feedback that was
        # incorporated. Past this point no raise on the synthesis path can
        # orphan the data, and the single-consume UPDATE ... RETURNING + NULL
        # semantics are preserved.
        tracker.get_and_clear_checkin_response(user_id)

        logger.info(
            "Synthesized user model: %d observations, %d checkin topics",
            len(result.observations), len(result.checkin_topics)
        )
        return result

    def _run_synthesis(
        self,
        signals_text: str,
        current_model_text: str,
        checkin_feedback: str | None = None
    ) -> str:
        """Run the synthesis LLM call and return raw XML output."""
        user_prompt = self._synthesis_user_template.format(
            section_id_list=self._section_list,
            current_user_model=current_model_text,
            assessment_signals=signals_text
        )

        if checkin_feedback:
            user_prompt += f"\n\n## User Check-in Feedback\n{checkin_feedback}"

        llm_messages = [
            {"role": "system", "content": self._synthesis_system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm_provider.generate_response(
            messages=llm_messages,
            model_config="batch",
        )

        return self.llm_provider.extract_text_content(response)

    def _validate_section_anchors(self, candidate_xml: str) -> CriticResult:
        """
        Deterministic section-anchor validation, run before the LLM critic.

        Wraps system_prompt_parser.validate_section_anchors into the loop's
        CriticResult shape: an invalid anchor fails validation with feedback
        naming the valid set, the synthesis retry loop re-runs with that
        feedback, and exhaustion raises CriticExhaustedError like any other
        validation failure. The check itself is code, not the critic.
        """
        try:
            validate_section_anchors(candidate_xml, config.system_prompt)
        except ValueError as e:
            logger.warning("Section anchor validation failed: %s", e)
            return CriticResult(
                passed=False,
                feedback=(
                    f"{e} Re-anchor every observation and check-in topic to a "
                    "valid section."
                )
            )
        return CriticResult(passed=True, feedback="")

    def _validate_with_critic(self, candidate_xml: str) -> CriticResult:
        """
        Run critic validation on a candidate user model.

        Returns:
            CriticResult with passed=True and empty feedback on success,
            or passed=False with actionable revision instructions.
        """
        user_prompt = self._critic_user_template.format(
            section_id_list=self._section_list,
            candidate_user_model=candidate_xml
        )

        llm_messages = [
            {"role": "system", "content": self._critic_system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm_provider.generate_response(
            messages=llm_messages,
            model_config="batch",
        )

        raw_output = self.llm_provider.extract_text_content(response)

        # Parse critic result
        status_match = re.search(r'<mira:critic_review\s+status="(\w+)"', raw_output)
        if not status_match:
            # Fail closed: an unparseable critic verdict is a validation failure,
            # never a pass — the loop retries with the feedback below and the
            # exhaustion branch raises. The critic is the only quality gate
            # before auto-publish; unreadable output must not auto-publish.
            logger.warning("Could not parse critic output, failing validation")
            return CriticResult(
                passed=False,
                feedback=(
                    "The quality critic returned an unparseable response (no "
                    "verdict could be extracted). Regenerate the user model, "
                    "ensuring observations are section-anchored, evidence-grounded, "
                    "free of personality labels, and internally consistent."
                )
            )

        status = status_match.group(1)

        if status == "pass":
            return CriticResult(passed=True, feedback="")

        # Extract issue descriptions as feedback
        issues = []
        issue_pattern = r'<mira:issue\s+type="([^"]+)"\s+section="([^"]+)">(.*?)</mira:issue>'
        for match in re.finditer(issue_pattern, raw_output, re.DOTALL):
            issue_type = match.group(1)
            section = match.group(2)
            detail = match.group(3).strip()
            issues.append(f"[{issue_type} in {section}] {detail}")

        return CriticResult(passed=False, feedback="\n".join(issues))

    def _rerun_synthesis_with_feedback(
        self,
        feedback: str,
        signals_text: str,
        current_model_text: str,
        checkin_feedback: str | None = None
    ) -> str:
        """Rerun synthesis with critic feedback appended to the prompt."""
        user_prompt = self._synthesis_user_template.format(
            section_id_list=self._section_list,
            current_user_model=current_model_text,
            assessment_signals=signals_text
        )

        if checkin_feedback:
            user_prompt += f"\n\n## User Check-in Feedback\n{checkin_feedback}"

        user_prompt += f"\n\n## Quality Critic Feedback\nThe quality critic flagged these issues in the previous attempt. Revise the user model to address them:\n\n{feedback}"

        llm_messages = [
            {"role": "system", "content": self._synthesis_system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm_provider.generate_response(
            messages=llm_messages,
            model_config="batch",
        )

        return self.llm_provider.extract_text_content(response)

    def _format_signals_by_section(self, signals: list[FeedbackSignalRow]) -> str:
        """Group and format signals by section_id for the synthesis prompt."""
        by_section: dict[str, list[FeedbackSignalRow]] = {}

        for signal in signals:
            section = signal['section_id']
            if section not in by_section:
                by_section[section] = []
            by_section[section].append(signal)

        parts = []
        for section_id, section_signals in sorted(by_section.items()):
            parts.append(f"### {section_id}")
            for s in section_signals:
                signal_type = s['signal_type']
                strength = s['strength']
                evidence = s['evidence']
                parts.append(f"- **{signal_type}** ({strength}): {evidence}")
            parts.append("")

        return "\n".join(parts) if parts else "No new signals."

    def _parse_user_model_xml(self, xml_output: str) -> SynthesisResult:
        """
        Parse a user model XML document into structured data.

        Expected format:
        <mira:user_model>
            <mira:observation section="..." confidence="...">
                Observation text.
                <changelog>What changed.</changelog>
            </mira:observation>
            <mira:checkin>
                <mira:topic section="..." reason="..."/>
            </mira:checkin>
        </mira:user_model>
        """
        observations = []
        checkin_topics = []

        # Parse observations
        obs_pattern = r'<mira:observation\s+section="([^"]+)"\s+confidence="([^"]+)">(.*?)</mira:observation>'

        for match in re.finditer(obs_pattern, xml_output, re.DOTALL):
            section_id = match.group(1).strip()
            confidence = match.group(2).strip()
            body = match.group(3)

            # Extract changelog
            changelog_match = re.search(r'<changelog>(.*?)</changelog>', body, re.DOTALL)
            changelog = changelog_match.group(1).strip() if changelog_match else ""

            # Observation text is everything except the changelog
            obs_text = re.sub(r'<changelog>.*?</changelog>', '', body, flags=re.DOTALL).strip()

            if confidence not in ('high', 'moderate', 'low'):
                logger.warning("Unknown confidence level: %s", confidence)
                confidence = 'moderate'

            observations.append(UserObservation(
                section_id=section_id,
                observation=obs_text,
                confidence=confidence,
                changelog=changelog
            ))

        # Parse check-in topics
        topic_pattern = r'<mira:topic\s+section="([^"]+)"\s+reason="([^"]+)"\s*(?:/>|>\s*</mira:topic>)'

        for match in re.finditer(topic_pattern, xml_output, re.DOTALL):
            checkin_topics.append(CheckinTopic(
                section_id=match.group(1).strip(),
                reason=match.group(2).strip()
            ))

        return SynthesisResult(
            observations=observations,
            checkin_topics=checkin_topics,
            raw_xml=xml_output
        )
