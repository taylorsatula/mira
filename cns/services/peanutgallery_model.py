"""
Peanut Gallery observer model: one fast-route call over a bounded delta window
plus a compact evidence ledger, maintaining the standing session record.

`evaluate` narrows the continuum to the recent delta window (the work since
the previous pass), builds the ledger (latest user request, latest assistant
response, assistant commitments extracted by phrase pattern, and tool
activity), and asks the observer to extend the standing record. Steering is a
rare, secondary output.

Parsing is the entire contract, and it fails toward silence: an unparseable
response, a missing session log, or an unparseable steering block degrade to
an empty result rather than raising. A malformed emission therefore costs one
observation, never a turn.

Everything spliced into the prompt from conversation content is XML-escaped at
the interpolation point, because that text is user-authored and would otherwise
be able to close a container and speak as the prompt.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from config.prompts.loader import load_prompt
from cns.core.message import Message, preprocess_content_blocks
from clients.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


def _escape(value: str) -> str:
    """Escape XML-like delimiters so external text cannot break prompt containers."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


@dataclass
class PeanutGalleryResult:
    """Result from the Peanut Gallery observer evaluation.

    session_log: new one-sentence beats to append to the standing record.
    steering:    at most one rare directive | None.
    steering_critical: whether the steering authorizes the fourth wall.
    """
    session_log: List[str] = field(default_factory=list)
    steering: Optional[str] = None
    steering_critical: bool = False


class PeanutGalleryModel:
    """
    Single-stage LLM model for the session's standing record.

    Each pass:
    - slices the continuum to the delta window (recent conversational messages)
    - builds a compact ledger of the delta (commitments, tool activity, latest
      user request and assistant response)
    - asks the observer to append 1-3 beats and (rarely) one steering directive
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        analysis_interval_turns: int
    ):
        self.llm_provider = llm_provider
        # The delta window is the last `analysis_interval_turns` turns, keyed on
        # the turn_id stamped on user and assistant messages: an agentic turn
        # emits several conversational messages, so a message count would drop
        # whole turns from the window. The message-count fallback covers
        # histories with no turn_id stamps.
        self._delta_window_turns = analysis_interval_turns
        self._delta_window = 2 * analysis_interval_turns

        self._system_prompt = load_prompt("peanutgallery_system.txt")
        self._user_template = load_prompt("peanutgallery_user.txt")

        logger.info(
            "PeanutGalleryModel initialized (delta_window_turns=%d)",
            self._delta_window_turns,
        )

    def evaluate(
        self,
        messages: List[Message],
        standing_log: Optional[List[str]] = None,
    ) -> PeanutGalleryResult:
        """
        Evaluate the delta window for a record extension and (rarely) steering.

        Args:
            messages: Messages from the active continuum snapshot.
            standing_log: Current standing record beats (empty on first pass).

        Returns:
            PeanutGalleryResult with session_log beats and optional steering
        """
        delta_messages = self._delta_slice(messages)
        if not delta_messages:
            return PeanutGalleryResult()

        conversational_messages = self._extract_conversational_messages(delta_messages)
        if len(conversational_messages) < 3:
            logger.debug("Not enough conversational messages (%d) for observer evaluation", len(conversational_messages))
            return PeanutGalleryResult()

        execution_trace = self._build_delta_ledger(delta_messages)
        return self._run_observer(conversational_messages, execution_trace, standing_log or [])

    def _delta_slice(self, messages: List[Message]) -> list[Message]:
        """
        Scope raw messages to the delta window the observer reviews.

        Keeps the conversational messages from the last `self._delta_window_turns`
        turns (keyed on the turn_id stamped on user and assistant messages), then
        returns the raw slice from the earliest kept message so tool and assistant
        noise between kept messages is preserved for the ledger. Falls back to a
        message-count window when the history carries no turn_id stamps.
        """
        conversational = self._extract_conversational_messages(messages)
        if not conversational:
            return []

        turn_ids: list[str] = []
        for message in conversational:
            turn_id = (message.metadata or {}).get("turn_id")
            if turn_id and turn_id not in turn_ids:
                turn_ids.append(turn_id)

        if len(turn_ids) >= self._delta_window_turns:
            keep = set(turn_ids[-self._delta_window_turns:])
            recent = [
                message for message in conversational
                if (message.metadata or {}).get("turn_id") in keep
            ]
        else:
            recent = conversational[-self._delta_window:]
        if not recent:
            return []

        first_recent_id = recent[0].id
        for idx, message in enumerate(messages):
            if message.id == first_recent_id:
                return messages[idx:]

        logger.warning("Failed to align Peanut Gallery delta window; using filtered messages only")
        return recent

    def _run_observer(
        self,
        messages: List[Message],
        execution_trace: str,
        standing_log: List[str],
    ) -> PeanutGalleryResult:
        """
        Run observer to extend the record and decide on steering.

        Args:
            messages: Delta window conversational messages (already scoped)
            execution_trace: Compact ledger of the delta window
            standing_log: Current standing record beats

        Returns:
            PeanutGalleryResult with session_log beats and optional steering
        """
        formatted_messages = self._format_with_ids(messages)
        standing_record = "\n".join(
            f"- {_escape(line)}" for line in standing_log
        ) if standing_log else "(empty — first pass)"

        # Single pass: replacement text is never rescanned, so a beat that
        # contains a placeholder token cannot splice another section in.
        substitutions = {
            "standing_record": standing_record,
            "formatted_messages": formatted_messages,
            "delta_ledger": execution_trace,
        }
        user_prompt = re.sub(
            r"\{(standing_record|formatted_messages|delta_ledger)\}",
            lambda match: substitutions[match.group(1)],
            self._user_template,
        )

        response = self.llm_provider.generate_response(
            messages=[{"role": "user", "content": user_prompt}],
            model_config="fast",
            system_prompt=self._system_prompt
        )

        response_text = self.llm_provider.extract_text_content(response).strip()

        if not response_text:
            logger.debug("Observer returned empty response")
            return PeanutGalleryResult()

        return self._parse_observer_response(response_text)

    def _parse_observer_response(self, response_text: str) -> PeanutGalleryResult:
        """
        Parse observer response into PeanutGalleryResult.

        Expected shape:
        <mira:session_log> 1-3 lines, one sentence each </mira:session_log>
        <mira:steering critical="true"> one directive sentence </mira:steering>

        Both tags are optional. Silence degrades to an empty result.
        """
        session_log = self._parse_session_log(response_text)
        steering, steering_critical = self._parse_steering(response_text)
        return PeanutGalleryResult(
            session_log=session_log,
            steering=steering,
            steering_critical=steering_critical,
        )

    def _parse_session_log(self, response_text: str) -> List[str]:
        """
        Extract and normalize the standing-record beats.

        Bullets are stripped, blanks dropped, over 300 chars truncated, exact
        duplicates removed, and capped at three beats per pass.
        """
        match = re.search(
            r'<mira:session_log\b[^>]*>(.*?)</mira:session_log>',
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        if not match:
            return []

        lines: List[str] = []
        for raw_line in match.group(1).splitlines():
            line = re.sub(r'^(\s*[-*•]+\s*)+', '', raw_line).strip()
            if not line:
                continue
            if len(line) > 300:
                line = line[:300].strip() + "..."
            if line not in lines:
                lines.append(line)
        return lines[:3]

    def _parse_steering(self, response_text: str) -> tuple[Optional[str], bool]:
        """
        Extract a steering directive, if present.

        Returns (text, critical). The fourth-wall path is taken only for
        critical="true" (case-insensitive), the one form the observer prompt
        documents. A steering tag quoted inside the session_log block is data,
        not a directive, and is ignored.
        """
        without_log = re.sub(
            r'<mira:session_log\b[^>]*>.*?</mira:session_log>',
            '',
            response_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # An unterminated session_log swallows the rest of the response;
        # nothing after it is a directive.
        without_log = re.sub(
            r'<mira:session_log\b[^>]*>.*$',
            '',
            without_log,
            flags=re.DOTALL | re.IGNORECASE,
        )
        match = re.search(
            r'<mira:steering\b([^>]*)>(.*?)</mira:steering>',
            without_log,
            re.DOTALL | re.IGNORECASE
        )
        if not match:
            return None, False

        attributes = match.group(1)
        text = match.group(2).strip()
        if not text:
            return None, False
        if len(text) > 600:
            text = text[:600].strip() + "..."

        critical = False
        critical_match = re.search(r'\bcritical="([^"]*)"', attributes, re.IGNORECASE)
        if critical_match and critical_match.group(1).lower() == "true":
            critical = True

        return text, critical

    def _extract_conversational_messages(self, messages: List[Message]) -> list[Message]:
        """Extract human-visible user/assistant messages for observer review."""
        return [message for message in messages if self._is_conversational_message(message)]

    def _is_conversational_message(self, message: Message) -> bool:
        """Return True when a message contributes to the visible conversation flow."""
        metadata = message.metadata or {}
        if message.role not in ('user', 'assistant'):
            return False

        if metadata.get('is_segment_boundary') or metadata.get('system_notification'):
            return False

        if isinstance(message.content, str):
            return bool(message.content.strip())

        has_text_content = any(
            (isinstance(block, str) and block.strip()) or
            (
                isinstance(block, dict) and
                block.get("type") == "text" and
                bool(str(block.get("text", "")).strip())
            )
            for block in message.content
        )
        preprocessed = preprocess_content_blocks(message.content)
        has_media_content = preprocessed.image_count > 0
        return has_text_content or has_media_content

    def _build_delta_ledger(self, messages: List[Message]) -> str:
        """Build a compact evidence ledger for the delta window.

        Same shape as the old full-window trace, but scoped to the delta
        window only — the new work since the last pass.
        """
        conversational_messages = self._extract_conversational_messages(messages)
        assistant_commitments: list[str] = []
        tool_activity: list[str] = []
        latest_user_request = "none"
        latest_assistant_response = "none"

        for msg in reversed(conversational_messages):
            rendered = self._render_message_content(msg)
            if latest_user_request == "none" and msg.role == "user" and rendered:
                latest_user_request = rendered
            if latest_assistant_response == "none" and msg.role == "assistant" and rendered:
                latest_assistant_response = rendered
            if latest_user_request != "none" and latest_assistant_response != "none":
                break

        for msg in messages:
            metadata = msg.metadata or {}
            if metadata.get('is_segment_boundary') or metadata.get('system_notification'):
                continue

            if msg.role == "assistant":
                if self._is_conversational_message(msg):
                    assistant_commitments.extend(
                        self._extract_commitments(self._render_message_content(msg))
                    )
                tool_activity.extend(self._extract_tool_uses(msg))
            elif msg.role == "tool":
                tool_call_id = msg.tool_call_id or "unknown"
                result_text = self._render_message_content(msg)
                if len(result_text) > 180:
                    result_text = result_text[:180] + "..."
                if not result_text:
                    tool_activity.append(
                        f"tool result received for call {tool_call_id} (empty content)"
                    )
                else:
                    tool_activity.append(
                        f"tool result received for call {tool_call_id}: {result_text}"
                    )

        if not assistant_commitments:
            assistant_commitments.append("none")
        if not tool_activity:
            tool_activity.append("none observed")

        trace_parts = [
            "<delta_ledger>",
            f"<latest_user_request>{_escape(latest_user_request)}</latest_user_request>",
            f"<latest_assistant_response>{_escape(latest_assistant_response)}</latest_assistant_response>",
            "<assistant_commitments>",
        ]
        trace_parts.extend(
            f"- {_escape(commitment)}" for commitment in assistant_commitments[:5]
        )
        trace_parts.append("</assistant_commitments>")
        trace_parts.append("<tool_activity>")
        trace_parts.extend(
            f"- {_escape(activity)}" for activity in tool_activity[:8]
        )
        trace_parts.append("</tool_activity>")
        trace_parts.append("</delta_ledger>")
        return "\n".join(trace_parts)

    def _extract_commitments(self, rendered_message: str) -> list[str]:
        """Extract concrete assistant commitments from a rendered message."""
        if not rendered_message:
            return []

        commitment_patterns = [
            r"\bI'll\s+([^.!?]+)",
            r"\bI will\s+([^.!?]+)",
            r"\bLet me\s+([^.!?]+)",
            r"\bI(?:'m| am)\s+going to\s+([^.!?]+)",
        ]

        commitments: list[str] = []
        for pattern in commitment_patterns:
            for match in re.finditer(pattern, rendered_message, re.IGNORECASE):
                commitment = match.group(0).strip()
                if commitment not in commitments:
                    commitments.append(commitment)
        return commitments

    def _extract_tool_uses(self, message: Message) -> list[str]:
        """Extract tool-use summaries from assistant content blocks and metadata."""
        tool_activity: list[str] = []
        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "tool_call":
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input")
                    if tool_input:
                        input_text = json.dumps(tool_input, sort_keys=True)
                        if len(input_text) > 180:
                            input_text = input_text[:180] + "..."
                        tool_activity.append(f"assistant invoked tool {tool_name} with input {input_text}")
                    else:
                        tool_activity.append(f"assistant invoked tool {tool_name}")

        tools_used = message.metadata.get("tools_used", [])
        for tool_name in tools_used:
            summary = f"assistant used tool {tool_name}"
            if summary not in tool_activity:
                tool_activity.append(summary)

        return tool_activity

    def _render_message_content(self, message: Message) -> str:
        """Render message content into compact plain text for observer context."""
        content = message.content
        if isinstance(content, list):
            preprocessed = preprocess_content_blocks(content)
            rendered = " ".join(preprocessed.text_parts)
            if preprocessed.image_count > 0:
                rendered = f"[{preprocessed.image_count} image(s) shared] {rendered}".strip()
        else:
            rendered = content

        if len(rendered) > 500:
            return rendered[:500] + "... [truncated]"
        return rendered

    def _format_with_ids(self, messages: List[Message]) -> str:
        """
        Format messages with 8-char IDs for observer evaluation.

        Format: [ID:xxxxxxxx] role: content
        """
        lines = []
        for msg in messages:
            msg_id = str(msg.id)[:8]
            content = self._render_message_content(msg)
            lines.append(f"[ID:{msg_id}] {msg.role}: {_escape(content)}")

        return "\n".join(lines)
