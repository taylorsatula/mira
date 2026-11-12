"""
Subcortical layer: retrieval-optimized query expansion, retention, and entities.

The expansion replaces the original query for retrieval. The model returns one
`record_subcortical_analysis` tool call, whose schema comes from
`SubcorticalAnalysis`; no tool is executed. Anything else — prose instead of a
call, two calls, refused arguments — raises `SubcorticalAnalysisError`, which
the orchestrator answers by degrading the turn to its retained memories. A
fabricated retention ID is unrepresentable rather than policed: `_analysis_tool`
narrows `pinned_memory_ids` to the IDs this call supplied, and the dialect
validates the arguments against that schema before returning the call.

`_build_user_prompt` supplies both generate() and the speculative warm_cache()
request, so their shared prefix stays byte-identical. `pinned_memory_ids` in
`SubcorticalResult` is the model's own retention picks; the union with memory
IDs cited in conversation turns is performed by the orchestrator, in
`recent_turns_plain`, so that union has exactly one home across the normal and
best-effort paths. Primary thinking effort is configured by the primary route,
not this layer.
"""
import contextvars
import html
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Iterator, List, Set, TypedDict, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from clients.llm.tool_messages import append_tool_result_messages
from clients.llm.types import Result, ToolCall, ToolDefinition, ToolResult

from cns.core.continuum import Continuum
from cns.core.message import preprocess_content_blocks

if TYPE_CHECKING:
    from clients.llm_provider import LLMProvider
    from cns.core.message import Message
from utils.tag_parser import format_memory_id, TagParser
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)

# All <mira:*> internal tags (paired and self-closing) — stripped before
# subcortical processing to avoid rare-token attention sinks.
_MIRA_TAG_PATTERN = re.compile(r'<mira:[^>]*>.*?</mira:[^>]*>|<mira:[^/]*/\s*>', re.DOTALL)


class SubcorticalAnalysisError(RuntimeError):
    """The fast route returned no usable analysis call for this turn.

    A shape failure only: the caller degrades the turn to the memories already
    in context. Provider, network, and context-overflow errors are not this
    type and propagate as the infrastructure failures they are.
    """


class SurfacedMemory(TypedDict):
    """Memory previously surfaced in conversation, evaluated for retention."""
    id: str  # Full UUID string
    text: str
    importance_score: float  # 0.0-1.0


class SubcorticalAnalysis(BaseModel):
    """Model-facing analysis arguments; also generates their tool schema."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)

    query_expansion: str = Field(
        min_length=1,
        description=(
            "Current message rewritten as 2-4 concrete search phrases, separated by "
            "semicolons. Resolve pronouns using recent history. Do not answer the message."
        ),
    )
    entities: list[Annotated[str, Field(min_length=1)]] = Field(
        description=(
            "Proper names in query_expansion: people, organizations, products, events, "
            "or places. Not general nouns. Empty array when none."
        ),
    )
    pinned_memory_ids: list[Annotated[str, Field(pattern=r"^mem_[a-fA-F0-9]{8}$")]] = Field(
        description=(
            "IDs of supplied passages about query_expansion's topic. Copy exact mem_ "
            "IDs. Empty array when none match."
        ),
    )


SUBCORTICAL_TOOL_NAME = "record_subcortical_analysis"
SUBCORTICAL_TOOL_DESCRIPTION = (
    "Return search phrases, their proper names, and matching stored passage IDs. "
    "Data only; no action is executed."
)


@dataclass
class SubcorticalResult:
    """Retrieval inputs, with retained memory IDs normalized to 8-char hex."""
    query_expansion: str
    pinned_memory_ids: Set[str] = field(default_factory=set)
    entities: List[str] = field(default_factory=list)

# Number of user/assistant pairs to include as context (3 pairs = 6 messages).
# Reduced from 6 based on MI analysis: conv_turns token mass was 9.5x current_message,
# driving 2.34x context bleed ratio. At 3 pairs the ratio drops to ~1.1x.
CONTEXT_PAIRS = 3


def _is_segment_summary(message: 'Message') -> bool:
    """True when a message is a collapsed segment's sentinel summary, not a turn."""
    metadata = getattr(message, 'metadata', {}) or {}
    return (
        metadata.get('is_segment_boundary', False) and
        metadata.get('status') == 'collapsed'
    )


def iter_recent_pairs(
    continuum: Continuum,
    max_pairs: int,
) -> Iterator[tuple['Message', 'Message']]:
    """Yield the last up-to-max_pairs user/assistant turns, oldest first.

    The one home for the recent-turn walk: collapsed segment summaries are
    never a pair's assistant half, an assistant message without a preceding
    user message ends the walk, and the yielded order is chronological. Both
    the model-facing prompt builder and the best-effort retrieval composer
    consume this, so the summary-exclusion rule cannot fork between them.
    """
    pairs_found = 0
    discovered: list[tuple['Message', 'Message']] = []
    i = len(continuum.messages) - 1
    while i >= 0 and pairs_found < max_pairs:
        while i >= 0:
            msg = continuum.messages[i]
            if msg.role == "assistant" and not _is_segment_summary(msg):
                break
            i -= 1
        if i < 0:
            break
        assistant_msg = continuum.messages[i]
        i -= 1

        while i >= 0 and continuum.messages[i].role != "user":
            i -= 1
        if i < 0:
            break
        user_msg = continuum.messages[i]
        i -= 1

        pairs_found += 1
        discovered.append((user_msg, assistant_msg))
    yield from reversed(discovered)


def recent_turns_plain(
    continuum: Continuum,
    max_pairs: int = CONTEXT_PAIRS,
) -> tuple[str, set[str]]:
    """Render the recent turns as plain text plus the mem_ IDs they cite.

    Both surfacing callers consume only the cited-ID harvest; the plain text is
    not the retrieval query (lite retrieves on the current message). The mem_
    harvest runs over raw assistant content BEFORE the <mira:*> strip, because a
    cited ID lives inside the very tags the strip removes; the harvest is the
    single home for the conversation-cited retention union, so it must not miss
    them.
    """
    lines: list[str] = []
    cited_ids: set[str] = set()
    for user_msg, assistant_msg in iter_recent_pairs(continuum, max_pairs):
        user_content = _extract_text_content(user_msg.content)[:2000]
        assistant_raw = _extract_text_content(assistant_msg.content)
        cited_ids.update(
            memory_id.lower()
            for memory_id in TagParser.MEMORY_ID_PATTERN.findall(assistant_raw)
        )
        lines.append(f"User: {user_content}")
        lines.append(
            f"Assistant: {_MIRA_TAG_PATTERN.sub('', assistant_raw)[:2000]}"
        )
    return "\n".join(lines), cited_ids


class SubcorticalLayer:
    """
    Subcortical processing layer for retrieval-optimized query expansion.

    Uses the fixed fast model route to expand fragmentary queries into detailed
    specifics that match stored memory vocabulary for better embedding similarity.
    """

    def __init__(
        self,
        llm_provider: 'LLMProvider',
        prefill_warmup_enabled: bool = False,
    ):
        """
        Initialize subcortical layer.

        Args:
            llm_provider: LLM provider for subcortical processing calls
            prefill_warmup_enabled: When True, warm_cache() fires a background
                max_tokens=1 request after each turn so vLLM's prefix cache is
                populated before the next user message arrives. Wastes billed
                tokens on cloud providers; only enable for local vLLM.

        Raises:
            FileNotFoundError: If prompt files not found
            ValueError: If API key not found in Vault
        """
        self.llm_provider = llm_provider
        self.prefill_warmup_enabled = prefill_warmup_enabled

        from config.prompts.loader import load_prompt
        self.system_prompt = load_prompt("subcortical_system.txt")
        self.user_prompt_template = load_prompt("subcortical_user.txt")
        self._demonstration = self._build_demonstration()

        logger.info("SubcorticalLayer initialized")

    def generate(
        self,
        continuum: Continuum,
        current_user_message: str,
        previous_memories: list[SurfacedMemory] | None = None
    ) -> SubcorticalResult:
        """
        Generate retrieval-optimized query expansion, evaluate retention, and extract entities.

        Expands fragmentary queries into detailed specifics:
        - Resolves "that", "it", "the one" to concrete references
        - Expands implicit context to explicit names, places, dates
        - Outputs vocabulary that matches stored memories

        Also evaluates which previously surfaced memories should remain in context,
        and extracts named entities for hub-based memory discovery.

        Args:
            continuum: Current continuum with message history
            current_user_message: User message to expand
            previous_memories: Memories from previous turn to evaluate for retention

        Returns:
            SubcorticalResult with query_expansion, pinned_memory_ids, and entities.

        Raises:
            SubcorticalAnalysisError: When the response is not exactly one valid
                record_subcortical_analysis call — prose in its place, two calls,
                another tool's name, or arguments the per-call schema refused. The
                orchestrator degrades that turn to its retained memories. Provider
                and network failures are not this type and propagate.
        """
        user_message, _ = self._build_user_prompt(
            continuum, current_user_message, previous_memories
        )

        logger.debug(f"Generating query expansion for: {current_user_message[:100]}...")
        if previous_memories:
            logger.debug(f"Evaluating retention for {len(previous_memories)} memories")

        # temperature is load-bearing: config.api.temperature defaults to 1.0 and overrides
        # the GGUF metadata's 0.1. effort="none" reaches llama-server as a reasoning_effort
        # it ignores — thinking-off there is the service unit's chat-template kwarg.
        response = self.llm_provider.generate_response(
            messages=self._analysis_messages(user_message),
            tools=[self._analysis_tool(previous_memories)],
            model_config="fast",
            system_prompt=self.system_prompt,
            effort="none",
            temperature=0.1,
        )

        result = self._parse_response(response)

        logger.info(f"Generated query expansion: {result.query_expansion[:150]}...")
        if previous_memories:
            logger.info(
                f"Retention: {len(result.pinned_memory_ids)}/{len(previous_memories)} memories pinned"
            )
        if result.entities:
            logger.info(f"Extracted {len(result.entities)} entities for hub discovery")

        # Persist full prompt/response for fine-tuning curation
        self._save_output(
            prompt=user_message,
            raw_response=json.dumps(response.tool_calls[0].to_message_block(), ensure_ascii=False),
        )

        return result

    def warm_cache(
        self,
        continuum: Continuum,
        previous_memories: list[SurfacedMemory] | None = None,
    ) -> None:
        """
        Speculatively warm the subcortical KV cache for the next turn.

        Fires a fire-and-forget max_tokens=1 request with the prefix that's
        already known at end-of-turn (system prompt + conversation_turns +
        previous_memories) so vLLM's prefix cache populates those blocks while
        the user is reading and typing. The next real subcortical call only
        prefills the new user-message tokens, removing prefill from the
        user-perceived critical path.

        No-op when prefill_warmup_enabled is False. Failures are logged at
        warning and swallowed — warmup is an optimization, not required
        infrastructure, so it must not break the response path.

        Only worthwhile against a vLLM (or other prefix-cache-aware) backend.
        On cloud providers without prefix caching, the call burns a billed
        token per turn for no benefit.
        """
        if not self.prefill_warmup_enabled:
            return
        ctx = contextvars.copy_context()
        threading.Thread(
            target=ctx.run,
            args=(self._do_warmup, continuum, previous_memories),
            daemon=True,
        ).start()

    def _do_warmup(
        self,
        continuum: Continuum,
        previous_memories: list[SurfacedMemory] | None,
    ) -> None:
        try:
            user_message, _ = self._build_user_prompt(
                continuum, "", previous_memories
            )
            self.llm_provider.generate_response(
                messages=self._analysis_messages(user_message),
                tools=[self._analysis_tool(previous_memories)],
                model_config="fast",
                system_prompt=self.system_prompt,
                effort="none",
                temperature=0.1,
                max_tokens=1,
            )
            logger.debug("Subcortical KV cache warmed")
        except Exception as e:
            logger.warning("Subcortical prefill warmup failed: %s", e)

    @staticmethod
    def _analysis_tool(previous_memories: list[SurfacedMemory] | None) -> ToolDefinition:
        """Constrain retention output to IDs supplied on this call.

        The enum is the whole gate: the dialect validates the model's arguments
        against this schema (openai_chat_base._coerce_tool_input) and reports a
        violation as ToolCall.invalid_reason, so a fabricated ID never reaches
        _parse_response as data.
        """
        schema = SubcorticalAnalysis.model_json_schema()
        pinned_schema = schema["properties"]["pinned_memory_ids"]
        memory_ids = sorted({format_memory_id(memory["id"]) for memory in previous_memories or []})
        if memory_ids:
            pinned_schema["items"]["enum"] = memory_ids
        else:
            pinned_schema["maxItems"] = 0
        return ToolDefinition(
            name=SUBCORTICAL_TOOL_NAME,
            description=SUBCORTICAL_TOOL_DESCRIPTION,
            input_schema=schema,
        )

    def _build_demonstration(self) -> list[dict[str, object]]:
        """Build the two worked examples that make a small model call the tool.

        Prose instructions alone did not produce a call from a 1.7B model; a
        completed call in history did. The two examples are a contrast pair: same
        recent turns, a different current turn, and a different passage on offer.
        The first resolves a pronoun against the recent turns and pins the matching
        passage out of two; the second offers that passage alone, asks about
        something else entirely, and pins nothing — the current turn picks the
        topic, and an off-topic passage is not pinned.

        Every noun is invented, and that is load-bearing: real case vocabulary
        leaked the example's details into unrelated answers. The example IDs
        cannot be emitted by a real call, because a real call's schema enumerates
        only the IDs that call supplied.
        """
        crate_id = "0f0f0f0f-0000-4000-8000-000000000001"
        crate = {"id": crate_id, "text": "Velmari crate arrived in Ostrek.", "importance_score": 0.5}
        weather = {"id": "0e0e0e0e-0000-4000-8000-000000000002", "text": "Unrelated note about the weather.", "importance_score": 0.5}
        recent_turns = (
            '<turn speaker="user" time="00:00">Velmari sent the crate to Ostrek.</turn>\n'
            '<turn speaker="assistant" time="00:00">Noted.</turn>'
        )
        examples = (
            ([crate, weather], "When did she get it there?", SubcorticalAnalysis(
                query_expansion="Velmari crate arrival in Ostrek; Velmari shipment date",
                entities=["Velmari", "Ostrek"],
                pinned_memory_ids=[format_memory_id(crate_id)],
            )),
            ([crate], "When does the Pelliston ferry sail?", SubcorticalAnalysis(
                query_expansion="Pelliston ferry departure schedule; Pelliston ferry times",
                entities=["Pelliston"],
                pinned_memory_ids=[],
            )),
        )
        messages: list[dict[str, object]] = []
        for position, (memories, current, example) in enumerate(examples):
            call = ToolCall(
                id=f"subcortical_example_{position}",
                tool_name=SUBCORTICAL_TOOL_NAME,
                input=example.model_dump(),
            )
            messages = append_tool_result_messages(
                [
                    *messages,
                    {"role": "user", "content": self._fill_user_template(
                        recent_turns, self._format_previous_memories(memories), current
                    )},
                ],
                Result(tool_calls=(call,), stop_reason="tool_use"),
                (ToolResult(tool_call_id=call.id, content="Recorded."),),
            )
        return messages

    def _analysis_messages(self, user_message: str) -> list[dict[str, object]]:
        """Prepend the worked examples so the real request is the third turn."""
        return [*self._demonstration, {"role": "user", "content": user_message}]

    def _build_user_prompt(
        self,
        continuum: Continuum,
        current_user_message: str,
        previous_memories: list[SurfacedMemory] | None,
    ) -> tuple[str, str]:
        """
        Build the subcortical user prompt; returns (prompt, conversation_turns).

        Single source of truth shared by generate() and warm_cache(). Producing
        byte-exact-same tokens here for warmup and real calls is what makes the
        vLLM prefix cache hit — any divergence breaks the optimization silently.
        """
        # Under memory pressure, narrow the window so retention decisions scope to the recent exchange.
        from lt_memory.proactive import MAX_PINNED_MEMORIES
        under_pressure = (
            previous_memories is not None
            and len(previous_memories) >= MAX_PINNED_MEMORIES
        )
        pairs = CONTEXT_PAIRS - 1 if under_pressure else CONTEXT_PAIRS

        conversation_turns = self._format_recent_turns(
            continuum,
            current_user_message,
            max_pairs=pairs,
        )

        # Format previous memories for prompt (unfiltered - LLM needs full context)
        # When under pressure, _format_previous_memories injects a pruning alert
        memories_block = self._format_previous_memories(previous_memories)

        user_prompt = self._fill_user_template(
            conversation_turns, memories_block, current_user_message
        )
        return user_prompt, conversation_turns

    def _fill_user_template(
        self,
        conversation_turns: str,
        memories_block: str,
        user_message: str,
    ) -> str:
        """Fill subcortical_user.txt for the real prompt and the demonstration alike."""
        # Collapse newlines — template is single-line by design, so injected
        # content must not reintroduce them.
        def _collapse(text: str) -> str:
            return " ".join(text.split())

        return self.user_prompt_template.replace(
            "{conversation_turns}",
            _collapse(conversation_turns)
        ).replace(
            "{user_message}",
            _collapse(html.escape(user_message, quote=False))
        ).replace(
            "{previous_memories}",
            _collapse(memories_block)
        )

    def _save_output(
        self,
        prompt: str,
        raw_response: str,
    ) -> None:
        """
        Persist raw prompt/response pairs for fine-tuning curation.

        Saves to data/users/{user_id}/subcortical_debuglogs/{YYYY-MM-DD}.jsonl
        with '------------------' delimiters between invocations.
        """
        try:
            from utils.user_context import get_current_user_id
            user_id = get_current_user_id()
            if not user_id:
                logger.debug("No user context, skipping subcortical output persistence")
                return

            output_dir = Path("data/users") / str(user_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            debug_dir = output_dir / "subcortical_debuglogs"
            debug_dir.mkdir(parents=True, exist_ok=True)
            date = utc_now().strftime("%Y-%m-%d")
            debug_file = debug_dir / f"{date}.jsonl"
            file_has_content = debug_file.exists() and debug_file.stat().st_size > 0

            debug_record = {
                "timestamp": utc_now().isoformat(),
                "prompt": prompt,
                "response": raw_response,
            }

            with open(debug_file, "a") as f:
                if file_has_content:
                    f.write("\n------------------\n")
                f.write(json.dumps(debug_record))

            logger.debug(f"Saved subcortical debug log to {debug_file}")

        except Exception as e:
            # Don't let persistence failures break the subcortical pipeline
            logger.warning(f"Failed to save subcortical output: {e}")

    @staticmethod
    def _importance_to_dots(importance_score: float) -> str:
        """
        Convert importance score (0.0-1.0) to 5-dot visual indicator.

        Scale:
            ●●●●● = 0.8-1.0 (high importance)
            ●●●●○ = 0.6-0.8
            ●●●○○ = 0.4-0.6
            ●●○○○ = 0.2-0.4
            ●○○○○ = 0.0-0.2 (low importance)
        """
        score = max(0.0, min(1.0, importance_score))
        filled = int(score * 5) + (1 if score > 0 else 0)  # At least 1 dot if score > 0
        filled = min(5, max(1, filled)) if score > 0 else 1
        return "●" * filled + "○" * (5 - filled)

    def _format_previous_memories(
        self,
        memories: list[SurfacedMemory] | None
    ) -> str:
        """
        Format previous memories for the prompt with 8-char IDs and importance indicators.

        Matches system prompt input format: mem_a1B2c3D4 [●●●○○] - Passage text

        Uses a two-tier graduated pressure system for the curated memory window:
        - Warning at MAX_PINNED_MEMORIES (6): budget is tightening; be selective
        - Critical at MAX_SURFACED_MEMORIES (8): cache is full; force-truncation imminent

        Args:
            memories: Surfaced memories to format for retention evaluation

        Returns:
            Formatted passage lines or empty string if no memories
        """
        if not memories:
            return ""

        lines = []

        # MAX_PINNED_MEMORIES is both the warning trigger and a hard cap for retained previous memories.
        from lt_memory.proactive import MAX_PINNED_MEMORIES, MAX_SURFACED_MEMORIES
        max_pinned = MAX_PINNED_MEMORIES
        critical_threshold = MAX_SURFACED_MEMORIES
        warning_threshold = MAX_PINNED_MEMORIES
        count = len(memories)

        if count >= critical_threshold:
            # Retained pins beyond max_pinned will be truncated.
            prune_count = max(1, count - max_pinned)
            lines.append(
                f'<mira:system_alert>\U0001f6a8 SURFACED MEMORY CACHE FULL ({count}/{critical_threshold}). '
                f'At most {max_pinned} previous memories may be retained; the system WILL force-drop '
                f'lowest-importance retained memories beyond that cap after this evaluation. '
                f'Aggressively prune \u2014 keep only memories that provide necessary context for '
                f'the active conversation topic. Drop tangential, redundant, or background '
                f'memories that aren\u2019t actively informing the discussion. You MUST remove at '
                f'least {prune_count} to avoid forced truncation and preserve fresh discovery slots. '
                f'\U0001f6a8</mira:system_alert>'
            )
        elif count >= warning_threshold:
            lines.append(
                f'<mira:system_alert>\u26a0\ufe0f {count} surfaced memories are available for retention. '
                f'Only {max_pinned} previous memories can be retained, and retaining too many reduces '
                f'the budget for discovering fresh relevant memories. Be selective \u2014 retain memories '
                f'that support the active topic, but let go of memories from topics the conversation '
                f'has moved past.</mira:system_alert>'
            )

        # Truncate passage text to reduce token mass competing with current_message.
        # MI analysis (round 2) showed 15 words preserves enough context for relevance
        # judgment while freeing ~60% of passage token budget. Per-token attention for
        # current_message gained +18.1% with this truncation.
        max_passage_words = 15

        for memory in memories:
            text = memory.get('text', '')
            memory_id = memory.get('id', '')
            importance = memory.get('importance_score', 0.5)
            formatted_id = format_memory_id(memory_id)
            dots = SubcorticalLayer._importance_to_dots(importance)

            # Truncate to first N words — ID + topic is sufficient for relevance filtering
            words = text.split()
            if len(words) > max_passage_words:
                text = " ".join(words[:max_passage_words]) + "..."

            if text and formatted_id:
                lines.append(f"{formatted_id} [{dots}] - {html.escape(text, quote=False)}")
            elif text:
                lines.append(f"[{dots}] - {html.escape(text, quote=False)}")

        return "\n".join(lines)

    def _parse_response(self, response: Result) -> SubcorticalResult:
        """Take the one analysis call, or name the shape that arrived instead.

        Raises:
            SubcorticalAnalysisError: On any shape other than exactly one valid
                record_subcortical_analysis call.
        """
        if response.stop_reason != "tool_use":
            raise SubcorticalAnalysisError(
                f"no completed analysis call (stop_reason={response.stop_reason})"
            )
        if len(response.tool_calls) != 1:
            raise SubcorticalAnalysisError(
                f"expected one analysis call, got {len(response.tool_calls)}"
            )
        call = response.tool_calls[0]
        if call.tool_name != SUBCORTICAL_TOOL_NAME:
            raise SubcorticalAnalysisError(f"unexpected tool called: {call.tool_name}")
        if call.invalid_reason is not None:
            raise SubcorticalAnalysisError(f"arguments rejected: {call.invalid_reason}")
        try:
            analysis = SubcorticalAnalysis.model_validate(dict(call.input))
        except ValidationError as error:
            raise SubcorticalAnalysisError(f"arguments failed validation: {error}") from error
        return SubcorticalResult(
            query_expansion=analysis.query_expansion,
            pinned_memory_ids={
                memory_id.removeprefix("mem_") for memory_id in analysis.pinned_memory_ids
            },
            entities=analysis.entities,
        )

    def _format_recent_turns(
        self,
        continuum: Continuum,
        current_user_message: str,
        max_pairs: int = CONTEXT_PAIRS
    ) -> str:
        """
        Format recent conversation turns for context.

        Skips collapsed segment summaries to only include actual conversation pairs.
        Uses XML <turn> elements with timestamps for structured representation.
        Does NOT include the current user message (template handles that separately).

        Args:
            continuum: Continuum with message cache
            current_user_message: Current user message (unused, kept for signature compat)
            max_pairs: Maximum conversation pairs to include (default: CONTEXT_PAIRS).
                       Reduced under memory pressure to narrow scope for retention decisions.

        Returns:
            Formatted string with last N pairs as timestamped <turn> elements
        """
        lines = []

        # Resolve the user's timezone from the user-context contextvar — the same
        # source message_formatter's format_messages_for_api reads — so
        # the model-facing <turn time=...> attributes render the user's local
        # wall clock instead of raw UTC. warm_cache() copies contextvars before
        # spawning its background thread, so this resolves on that path too.
        from utils.timezone_utils import convert_from_utc
        from utils.user_context import get_user_preferences
        user_tz = get_user_preferences().timezone

        for user_msg, assistant_msg in iter_recent_pairs(continuum, max_pairs):
            user_time = convert_from_utc(user_msg.created_at, user_tz).strftime("%H:%M")
            assistant_time = convert_from_utc(assistant_msg.created_at, user_tz).strftime("%H:%M")

            # Prepend pair (we're walking backwards), truncating long messages.
            user_content = html.escape(
                _extract_text_content(user_msg.content)[:2000], quote=False
            )
            # Strip all <mira:*> internal tags (emotion emojis, memory refs,
            # etc.) — rare-token attention sinks that waste budget without
            # contributing to entity/expansion/passage tasks. Strip BEFORE
            # escaping so legitimate <mira:memory …/> tags in assistant content
            # still match the pattern; then escape whatever remains. Structured
            # content (tool calls, reasoning blocks, lists) is rendered to
            # natural labeled text first — never str() of a raw structure.
            assistant_content = html.escape(
                _MIRA_TAG_PATTERN.sub(
                    '', _extract_text_content(assistant_msg.content)
                )[:2000],
                quote=False
            )
            lines.append(f"<turn speaker=\"user\" time=\"{user_time}\">{user_content}</turn>")
            lines.append(f"<turn speaker=\"assistant\" time=\"{assistant_time}\">{assistant_content}</turn>")

        return "\n".join(lines)

def _extract_text_content(content: str | list[dict[str, object]]) -> str:
    """Extract natural text from potentially structured content.

    Delegates to cns.core.message.preprocess_content_blocks (the repo's
    canonical structured-content preprocessor, mirrored from
    summary_generator/persona_service): text and reasoning blocks render
    as text, tool_call blocks as a labeled marker, media blocks are
    stripped. Never returns str() of a raw structure.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        preprocessed = preprocess_content_blocks(content)
        if preprocessed.text_parts:
            return ' '.join(preprocessed.text_parts)
        if preprocessed.image_count:
            return '[image content]'
        return '[non-text content]'

    return '[non-text content]'
