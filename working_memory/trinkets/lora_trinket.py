"""
LoRA trinket for injecting user model observations into the system prompt.

Reads the user model from feedback_synthesis_tracking.last_synthesis_output
and renders it as contextual knowledge about the user. The user model is
descriptive (observations about the user), not prescriptive (instructions to Mira).

When needs_checkin is true, also renders check-in guidance for behavioral debrief.
"""
import logging
import re
from typing import Dict, Any, TypedDict

from cns.infrastructure.feedback_tracker import FeedbackTracker
from utils.user_context import get_current_user_id, get_user_preferences
from .base import EventAwareTrinket


class _Observation(TypedDict):
    section_id: str
    confidence: str
    text: str


class _CheckinTopic(TypedDict):
    section_id: str
    reason: str

logger = logging.getLogger(__name__)


class LoraTrinket(EventAwareTrinket):
    """
    Injects user model observations from the user model pipeline.

    Reads from feedback_synthesis_tracking and renders observations as
    contextual knowledge in the system prompt.
    """

    variable_name = "behavioral_directives"
    cache_policy = True

    def __init__(self, event_bus, working_memory):
        self._feedback_tracker = FeedbackTracker()
        super().__init__(event_bus, working_memory)

    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate user model content for system prompt injection.

        Returns:
            Formatted user model XML, or empty string if no model exists
        """
        user_id = get_current_user_id()

        lora_content = self._feedback_tracker.get_lora_content(user_id)
        synthesis_xml = lora_content['synthesis_xml']
        needs_checkin = lora_content['needs_checkin']

        if not synthesis_xml:
            logger.debug("No user model for user %s", user_id)
            return ""

        observations, checkin_topics = self._parse_user_model_xml(synthesis_xml)

        if not observations:
            return ""

        parts = [self._format_user_model(observations)]

        if needs_checkin and checkin_topics:
            parts.append(self._format_checkin_guidance(checkin_topics))

        return "\n\n".join(parts)

    def _parse_user_model_xml(self, xml_output: str) -> tuple[list[_Observation], list[_CheckinTopic]]:
        """
        Parse user model XML into observations and check-in topics.

        Returns:
            Tuple of (observations list, checkin topics list)
        """
        observations = []
        checkin_topics = []

        # Parse observations
        obs_pattern = r'<mira:observation\s+section="([^"]+)"\s+confidence="([^"]+)">(.*?)</mira:observation>'

        for match in re.finditer(obs_pattern, xml_output, re.DOTALL):
            section_id = match.group(1).strip()
            confidence = match.group(2).strip()
            body = match.group(3)

            # Strip changelog from display text
            obs_text = re.sub(r'<changelog>.*?</changelog>', '', body, flags=re.DOTALL).strip()

            observations.append({
                'section_id': section_id,
                'confidence': confidence,
                'text': obs_text
            })

        # Parse check-in topics
        topic_pattern = r'<mira:topic\s+section="([^"]+)"\s+reason="([^"]+)"\s*(?:/>|>\s*</mira:topic>)'

        for match in re.finditer(topic_pattern, xml_output, re.DOTALL):
            checkin_topics.append({
                'section_id': match.group(1).strip(),
                'reason': match.group(2).strip()
            })

        return observations, checkin_topics

    def _format_user_model(self, observations: list[_Observation]) -> str:
        """Format observations as user model XML for system prompt injection."""
        prefs = get_user_preferences()
        first_name = (prefs.first_name or '').strip() or "this user"

        parts = ["<user_model>",
                 f"What you've learned about {first_name} through observation:",
                 ""]

        for obs in observations:
            parts.append(
                f'<observation section="{obs["section_id"]}">'
                f'{obs["text"]}'
                f'</observation>'
            )

        parts.append("</user_model>")
        return "\n".join(parts)

    def _format_checkin_guidance(self, topics: list[_CheckinTopic]) -> str:
        """Format check-in topics as behavioral debrief guidance with delivery nudge."""
        parts = [
            "<behavioral_checkin>",
            "<instruction>You have unresolved behavioral check-ins. Bring these up during this "
            "conversation. Find a natural moment, but do not let the session end without raising them. "
            "The user cannot see these topics unless you voice them.</instruction>",
            "",
            "Be direct: explain what you've been noticing and ask whether your read is accurate. "
            "This is a collaborative debrief, not a preference survey. Don't ask what they want. "
            "Ask if you're reading the room right.",
            ""
        ]

        for topic in topics:
            parts.append(
                f'- Regarding {topic["section_id"]}: {topic["reason"]}'
            )

        parts.append("")
        parts.append(
            "<instruction>After the user responds to the check-in, close the feedback loop:\n"
            "1. Echo their feedback verbatim as a markdown blockquote (> prefix) — this is what the "
            "user sees.\n"
            "2. Repeat the identical content in a <mira:checkin_response> tag (invisible to user, "
            "parsed by backend). Blockquote and tag must match exactly.\n"
            "3. Ask the user to confirm wording accuracy. If they correct you, reissue both blockquote "
            "and tag with the revised version. Latest output wins.\n"
            "4. Consolidate all check-in feedback into one blockquote + tag pair, even across multiple "
            "topics.\n\n"
            "Example output after user responds:\n"
            "```\n"
            "Here's what I'm recording:\n\n"
            "> User prefers blunt technical feedback without diplomatic hedging.\n"
            "> When I soften criticism, they find it condescending rather than considerate.\n\n"
            "<mira:checkin_response>User prefers blunt technical feedback without diplomatic hedging. "
            "When I soften criticism, they find it condescending rather than considerate."
            "</mira:checkin_response>\n\n"
            "Is this wording accurate?\n"
            "```</instruction>"
        )

        parts.append("</behavioral_checkin>")
        return "\n".join(parts)
