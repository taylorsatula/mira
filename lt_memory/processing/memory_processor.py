"""
Memory processor - parse and validate LLM extraction responses.

Consolidates all response processing logic into a single, focused module:
- JSON parsing with repair fallback
- Structure validation
- UUID remapping (short → full)
- Memory field validation and sanitization
- Duplicate detection (fuzzy + vector)
- Index remapping for filtered memories

This module is pure data processing with no side effects.
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, TypedDict, Union
from uuid import UUID

from rapidfuzz import fuzz
from json_repair import repair_json

from lt_memory.models import ExtractedMemory, ExtractionResult, MemoryContext
from lt_memory.vector_ops import VectorOps
from utils.tag_parser import parse_memory_id
from utils.timezone_utils import ensure_utc, parse_time_string, validate_timezone
from utils.user_context import get_user_preferences

logger = logging.getLogger(__name__)

# Extraction deduplication tuning
DEDUP_SIMILARITY_THRESHOLD = 0.92  # Cosine similarity for duplicate detection
DEFAULT_IMPORTANCE_SCORE = 0.5     # Default importance for newly extracted memories


def sanitize_entity_list(entities: Any) -> List[Dict[str, str]]:
    """Filter a raw LLM-emitted ``entities`` value to well-formed link candidates.

    The single shape contract for every entity source entering the memory
    pipeline: extraction responses (``_validate_extracted_memory`` here) and
    the pending-drain entity pass
    (``cns/services/segment_collapse_handler.py``). Accepts the untrusted raw
    value and projects each kept entry to exactly ``{"name": str, "type": str}``
    — extra keys emitted by the model are dropped, so a noisy entry can never
    fail Pydantic's ``Dict[str, str]`` validation downstream (which would burn
    a drain item's content-caused attempts budget on LLM noise); a malformed
    entry degrades to "no entity" instead of poisoning the entity table.
    Absent/None is a legitimate "no entities" answer and returns ``[]``
    silently.
    """
    if not isinstance(entities, list):
        if entities is not None:
            logger.warning(f"Fixing entities: converting {type(entities)} to empty list")
        return []
    valid_entities: List[Dict[str, str]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        if 'name' not in entity or 'type' not in entity:
            continue
        if not isinstance(entity['name'], str) or not isinstance(entity['type'], str):
            continue
        # Normalize entity name (strip whitespace)
        entity['name'] = entity['name'].strip()
        if len(entity['name']) < 2:
            continue
        valid_entities.append({"name": entity["name"], "type": entity["type"]})
    return valid_entities


class LLMResponseFormatError(ValueError):
    """The extraction LLM's response is empty or otherwise unusable.

    This is an infra/LLM-class failure, NOT a content failure, and it must NOT
    be treated as a successful zero-memory extraction: returning an empty
    list for a degenerate response would let the caller mark the segment
    extracted and silently end its memory extraction forever. Raising keeps
    the segment unextracted so the sweep retries it, and callers classify
    this type to keep it off the segment's content-failure retry budget
    (see ExtractionOrchestrator.extract_unprocessed_segments).
    """


class DuplicateCheckResult(NamedTuple):
    """Result of duplicate memory check."""
    is_duplicate: bool
    similarity: float | None
    duplicate_id: str | None


def _resolve_extraction_timezone() -> str:
    """Timezone to read model-supplied temporal fields in.

    The extraction prompt asks the model for wall times as stated in the
    conversation (the user's local time), so a naive ISO string must be resolved
    against the user's timezone — never against the system default. Pydantic's own
    coercion of a naive string yields a naive datetime whose stored instant is then
    decided by the Postgres server/session ``TimeZone`` setting, a configuration
    value the application never sets (verified four-hour drift on a live server).

    Called once per response, before the memory loop, and falls back to UTC when
    preferences are unreachable: extraction is a background durability path, not a
    request path — raising here would discard every memory in the segment. Runs
    ``validate_timezone`` inside the guard for the same reason; a single malformed
    ``users.timezone`` row must not strip every temporal anchor in the batch.
    """
    try:
        return validate_timezone(get_user_preferences().timezone)
    except Exception:
        logger.warning(
            "User preferences unavailable during extraction; "
            "reading model-supplied memory times as UTC",
            exc_info=True,
        )
        return "UTC"


def _parse_model_temporal_field(
    value: Union[str, datetime],
    field: str,
    memory_text: str,
    tz_name: str,
) -> Optional[datetime]:
    """Parse one model-supplied temporal field into an aware UTC datetime, or None.

    Returns None instead of raising: by extraction time the model is off the call
    stack, so an exception loses the memory outright rather than prompting a
    clarifying question — a ValidationError here aborts every memory constructed so
    far in this response (no per-memory catch exists upstream). A value that will
    not parse logs the raw string, the timezone it was read against, and the
    parser's own reason, so the lost anchor stays diagnosable, and the memory is
    still stored without that field. Text is never dropped.

    Deliberately lenient about daylight-saving ambiguity and nonexistent times
    (``parse_time_string`` resolves both against the user's zone): strictness
    belongs to the model-facing scheduling tools, where the caller can be asked to
    choose. Nobody can be asked here.
    """
    if isinstance(value, datetime):
        # Defensive: an already-parsed value bypasses string parsing but still
        # must never reach the column naive.
        return ensure_utc(value)
    try:
        return ensure_utc(parse_time_string(value, tz_name=tz_name))
    except Exception as exc:
        logger.warning(
            "Unparseable %s on extracted memory %.60r: value=%r, timezone=%s. "
            "Storing the memory without that temporal field -- text is preserved, "
            "but temporal scoring and decay treat this record as untimed. "
            "Parser said: %s: %s",
            field,
            memory_text,
            value,
            tz_name,
            type(exc).__name__,
            exc,
        )
        return None


class RawMemoryDict(TypedDict, total=False):
    """Raw memory dict from LLM extraction response before validation."""
    text: str
    importance_score: float
    expires_at: str | None
    happens_at: str | None
    related_memory_ids: list[dict[str, str]]
    entities: list[dict[str, str]]


class MemoryProcessor:
    """
    Process LLM extraction responses into validated ExtractedMemory objects.

    Single Responsibility: Transform raw LLM text → validated memories

    No side effects - pure data processing that can be tested independently.
    """

    def __init__(self, vector_ops: VectorOps):
        self.vector_ops = vector_ops

    def process_extraction_response(
        self,
        response_text: str,
        short_to_uuid: Dict[str, str],
        memory_context: MemoryContext,
        segment_id: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Process a direct extraction result from the LLM response.

        Complete pipeline: Parse → Remap IDs → Validate → Deduplicate

        Args:
            response_text: LLM response text (JSON format)
            short_to_uuid: Mapping from shortened IDs to full UUIDs
            memory_context: Memory context used during extraction (for deduplication)
            segment_id: UUID string of the segment being extracted — threaded
                into every built ExtractedMemory as source_segment_id so
                db_access.store_memories persists the extraction provenance
                anchor. None only when the chunk genuinely has no segment.

        Returns:
            ExtractionResult containing validated memories

        Raises:
            LLMResponseFormatError: If the response is empty/degenerate (a
                ValueError subclass) — the segment must NOT be marked extracted
            ValueError: If response parsing fails catastrophically
        """
        # Step 1: Parse JSON response
        memories_data = self._parse_extraction_response(response_text)

        # Step 2: Remap shortened UUIDs to full UUIDs
        memories_data = self._remap_short_ids_to_full(memories_data, short_to_uuid)

        # Step 3: Validate and deduplicate memories
        # Timezone resolved once for the whole response, before the loop, so a
        # preferences outage degrades to UTC here instead of failing each memory.
        extraction_tz = _resolve_extraction_timezone()

        # Track index mapping: original LLM response index → filtered list index
        extracted_memories = []
        original_to_filtered_idx = {}

        for original_idx, memory_dict in enumerate(memories_data):
            # Validate structure
            if not self._validate_extracted_memory(memory_dict):
                continue

            # Check for duplicates
            is_duplicate, similarity, duplicate_id = self._is_duplicate_memory(
                memory_dict,
                memory_context
            )

            if is_duplicate:
                logger.info(
                    f"Skipping duplicate memory: similarity {similarity:.3f} "
                    f"with existing memory {duplicate_id}"
                )
                continue

            # Create ExtractedMemory object.
            # Temporal fields are parsed HERE, not by Pydantic: coercion of a naive
            # model string yields a naive datetime, which psycopg then binds under
            # whatever TimeZone Postgres happens to have configured. Parsing through
            # the user's timezone keeps the stored instant the user meant, and the
            # tolerate-and-log shape keeps one bad timestamp from losing the batch.
            happens_at_raw = memory_dict.get("happens_at")
            expires_at_raw = memory_dict.get("expires_at")
            extracted_memory = ExtractedMemory(
                text=memory_dict["text"],
                importance_score=DEFAULT_IMPORTANCE_SCORE,
                expires_at=(
                    _parse_model_temporal_field(
                        expires_at_raw, "expires_at", memory_dict["text"], extraction_tz
                    )
                    if expires_at_raw else None
                ),
                happens_at=(
                    _parse_model_temporal_field(
                        happens_at_raw, "happens_at", memory_dict["text"], extraction_tz
                    )
                    if happens_at_raw else None
                ),
                related_memory_ids=memory_dict.get("related_memory_ids", []),
                entities=memory_dict.get("entities", []),
                source_segment_id=UUID(segment_id) if segment_id else None,
            )

            # Track mapping from original index to filtered index
            filtered_idx = len(extracted_memories)
            original_to_filtered_idx[original_idx] = filtered_idx
            extracted_memories.append(extracted_memory)

        logger.info(
            f"Processed extraction response: {len(extracted_memories)} memories"
        )

        return ExtractionResult(
            memories=extracted_memories
        )

    def _parse_extraction_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse JSON extraction response from LLM.

        Handles multiple formats with repair fallback:
        - List of memory dicts (standard)
        - Single memory dict → wrapped in list
        - {"memories": [...]} wrapper → extracted
        - Empty/whitespace responses → LLMResponseFormatError (a degenerate
          response is a FAILURE, not a zero-memory success; a genuine zero is
          a well-formed [] or {"memories": []})

        Args:
            response_text: LLM response text (JSON format)

        Returns:
            List of memory dictionaries

        Raises:
            LLMResponseFormatError: If the response is empty/degenerate or not
                repairable into the memory list schema
        """
        response_text = response_text.strip()

        # Empty/degenerate responses are a FAILURE, not a zero-memory success:
        # returning [] here would let the caller mark the segment extracted and
        # silently end its memory extraction forever. A genuine zero-memory
        # result is a well-formed [] / {"memories": []} response. Raise so the
        # segment stays unextracted and is retried by the sweep — as an
        # infra/LLM-class failure, it does not consume the content budget.
        if not response_text:
            logger.warning(
                "LLM returned empty response - should return valid JSON like [] or {\"memories\": []}. "
                "May indicate API issue or prompt problem."
            )
            raise LLMResponseFormatError(
                "LLM extraction response was empty; a well-formed zero-memory "
                "response (e.g. []) is required to mark a segment extracted"
            )

        # Try parsing as-is first (handles compliant responses)
        try:
            parsed = json.loads(response_text)

            # Handle different response formats
            if isinstance(parsed, list):
                if not self._validate_memory_list_structure(parsed):
                    raise LLMResponseFormatError("Parsed list contains invalid memory structures")
                return parsed
            elif isinstance(parsed, dict):
                # Single memory object or {"memories": [...]} wrapper
                if "memories" in parsed:
                    memories = parsed["memories"]
                    if isinstance(memories, list):
                        if not self._validate_memory_list_structure(memories):
                            raise LLMResponseFormatError("Parsed 'memories' field contains invalid structures")
                        return memories
                    elif memories:
                        # A non-list, non-empty value (e.g. {"memories": "none"})
                        # is not a genuine zero-memory response; returning it
                        # unvalidated lets the per-memory validator silently
                        # drop it into a zero-memory success.
                        raise LLMResponseFormatError(
                            "Parsed 'memories' field is a non-list, non-empty "
                            "value; a genuine zero-memory response is "
                            "{\"memories\": []}"
                        )
                    else:
                        # Genuine empty (null, "", 0, {})
                        return []
                else:
                    # Single memory object — must be a well-formed memory dict,
                    # otherwise it is silently dropped into a zero-memory
                    # success and the segment is marked extracted forever.
                    if not self._validate_memory_list_structure([parsed]):
                        raise LLMResponseFormatError(
                            "Parsed object is not a well-formed memory dict "
                            "(missing or unusable 'text' field)"
                        )
                    return [parsed]
            else:
                raise LLMResponseFormatError(
                    f"Invalid extraction response format: expected list or dict, got {type(parsed).__name__}"
                )

        except json.JSONDecodeError as e:
            # Whitespace-only output is the same degenerate-response failure as
            # an empty one: it must not be recorded as a zero-memory success.
            if not response_text or response_text.isspace():
                logger.debug("Response contains only whitespace - degenerate response")
                raise LLMResponseFormatError(
                    "LLM extraction response contained only whitespace"
                )

            # Log the actual error with the first part of the response for debugging
            logger.warning(f"JSON parsing failed: {e}")
            logger.debug(f"Response text (first 200 chars): {response_text[:200]!r}")

            # Try json_repair (required dependency, imported at module level)
            try:
                repaired = repair_json(response_text)

                # Check if repair actually changed something
                if repaired == response_text:
                    logger.error(
                        f"json_repair could not repair response (returned unchanged). "
                        f"Response is not valid JSON. First 200 chars: {response_text[:200]!r}"
                    )
                    raise LLMResponseFormatError(
                        "LLM response is not valid JSON and json_repair could not fix it. "
                        "Indicates LLM output format issue."
                    )

                parsed = json.loads(repaired)
                logger.debug("Successfully repaired malformed JSON")

                # Handle repaired response formats
                if isinstance(parsed, list):
                    if not self._validate_memory_list_structure(parsed):
                        logger.debug(f"Repaired JSON has invalid structure: {parsed}")
                        raise LLMResponseFormatError("Repaired JSON does not match memory list schema")
                    return parsed
                elif isinstance(parsed, dict):
                    if "memories" in parsed:
                        memories = parsed["memories"]
                        if isinstance(memories, list):
                            if not self._validate_memory_list_structure(memories):
                                logger.debug(f"Repaired JSON 'memories' field invalid: {memories}")
                                raise LLMResponseFormatError("Repaired JSON memories field does not match schema")
                            return memories
                        elif memories:
                            # Non-list, non-empty value: not a genuine empty —
                            # same degenerate-shape failure as the primary parse.
                            raise LLMResponseFormatError(
                                "Repaired JSON 'memories' field is a non-list, "
                                "non-empty value; a genuine zero-memory "
                                "response is {\"memories\": []}"
                            )
                        else:
                            # Genuine empty (null, "", 0, {})
                            return []
                    else:
                        # Single memory object — must be well-formed
                        if not self._validate_memory_list_structure([parsed]):
                            logger.debug(f"Repaired JSON object not a memory dict: {parsed}")
                            raise LLMResponseFormatError(
                                "Repaired JSON object is not a well-formed "
                                "memory dict (missing or unusable 'text' field)"
                            )
                        return [parsed]
                else:
                    raise LLMResponseFormatError(
                        f"Invalid extraction response format after repair: expected list or dict, got {type(parsed).__name__}"
                    )

            except json.JSONDecodeError as e:
                # Even after repair, still not valid JSON - this is an error
                logger.error(
                    f"Repaired response still not valid JSON: {e}. "
                    f"Repaired text (first 200 chars): {repaired[:200]!r}"
                )
                raise LLMResponseFormatError(
                    f"LLM response invalid even after json_repair attempt: {e}"
                ) from e
            except Exception as repair_error:
                # Unexpected error during repair - propagate it
                logger.error(f"Unexpected error during JSON repair: {repair_error}")
                raise LLMResponseFormatError(
                    f"JSON repair failed with unexpected error: {repair_error}"
                ) from repair_error

    def _validate_memory_list_structure(self, parsed: object) -> bool:
        """
        Validate that parsed result is a list of memory dicts.

        Ensures the parsed JSON matches the expected schema: a list where each
        element is a dictionary containing at minimum a "text" field.

        Args:
            parsed: Result from json.loads()

        Returns:
            True if valid structure, False otherwise
        """
        if not isinstance(parsed, list):
            logger.warning(f"Parse result is not a list: {type(parsed).__name__}")
            return False

        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                logger.warning(
                    f"Memory list item {idx} is not a dict: {type(item).__name__} = {item}"
                )
                return False

            if "text" not in item:
                logger.warning(f"Memory list item {idx} missing required 'text' field")
                return False

            text = item["text"]
            if not isinstance(text, str) or not text.strip():
                # An empty/whitespace/non-string text is unusable: it would be
                # silently dropped by the per-memory validator and the response
                # would degrade into a zero-memory success. Reject it here so
                # the degenerate response raises instead.
                logger.warning(
                    f"Memory list item {idx} has an unusable 'text' field: {text!r}"
                )
                return False

        return True

    def _remap_short_ids_to_full(
        self,
        memories_data: List[RawMemoryDict],
        short_to_full: Dict[str, str]
    ) -> List[RawMemoryDict]:
        """
        Remap shortened UUID identifiers back to full UUIDs.

        LLM uses shortened IDs in response; this converts them back.

        Args:
            memories_data: List of memory dicts with shortened IDs
            short_to_full: Mapping from short ID to full UUID

        Returns:
            Updated memory dicts with full UUIDs
        """
        # Normalize the vocabulary once per response, the way every sanctioned
        # short-ID resolver does (utils/tag_parser.py match_memory_id,
        # lt_memory/db_access.py, orchestrator): fold case first, then strip
        # the optional mem_ prefix. Keys here are format_memory_id outputs
        # ("mem_<8 hex>"), so they reduce to bare lowercase hex — and an id the
        # model echoes with drifted casing (MEM_/MeM_) or without the mem_
        # prefix resolves against that lowercased vocabulary instead of being
        # dropped. A miss still falls through to the drop-and-warn path below
        # unchanged.
        lowered_map = {
            parse_memory_id(key.lower()): full
            for key, full in short_to_full.items()
        }
        for memory_dict in memories_data:
            # Remap related_memory_ids: remap id field inside dicts, drop unresolved
            if "related_memory_ids" in memory_dict:
                related_ids = memory_dict["related_memory_ids"]
                if isinstance(related_ids, list):
                    valid_refs = []
                    for ref in related_ids:
                        if isinstance(ref, dict) and "id" in ref:
                            ref_id = ref["id"]
                            if isinstance(ref_id, str):
                                ref["id"] = lowered_map.get(
                                    parse_memory_id(ref_id.lower()), ref_id
                                )
                            try:
                                UUID(ref["id"])
                                valid_refs.append(ref)
                            except ValueError:
                                logger.warning(f"Dropping unresolved related_memory_id after remap: {ref['id']}")
                        else:
                            valid_refs.append(ref)
                    memory_dict["related_memory_ids"] = valid_refs

        return memories_data

    def _validate_extracted_memory(self, memory_dict: Dict[str, Any]) -> bool:
        """
        Validate and sanitize extracted memory structure.

        Rejects invalid memories, applies intelligent fallbacks for recoverable issues.
        Malformed related_memory_ids entries are dropped (with a WARNING), not
        rejected — the segment is already collapsed, so a rejected memory is
        lost permanently, and one bad optional link must not destroy it.
        Modifies memory_dict in-place with fixes.

        Args:
            memory_dict: Memory dictionary from LLM (modified in-place)

        Returns:
            True if valid (possibly after fixes), False if unrecoverable
        """
        # REJECT: Not a dictionary (defensive type guard)
        if not isinstance(memory_dict, dict):
            logger.warning(f"Rejecting memory: expected dict, got {type(memory_dict).__name__}")
            return False

        # REJECT: Missing or invalid text (unrecoverable)
        if not memory_dict.get("text"):
            logger.warning("Rejecting memory: no text field")
            return False

        if not isinstance(memory_dict["text"], str):
            logger.warning(f"Rejecting memory: text is {type(memory_dict['text'])}, not string")
            return False

        # REJECT: Text too short (unrecoverable)
        text = memory_dict["text"].strip()
        if len(text) < 10:
            logger.warning(f"Rejecting memory: text too short ({len(text)} chars): {text}")
            return False

        # FIX (drop-malformed): related_memory_ids are optional links; the
        # segment is already collapsed, so rejecting the memory here loses it
        # permanently. Drop the malformed entry with a WARNING and keep the
        # rest of the memory — same tolerate-and-filter shape as the
        # entities block below.
        if "related_memory_ids" in memory_dict:
            refs = memory_dict["related_memory_ids"]
            if not isinstance(refs, list):
                logger.warning(
                    f"Dropping malformed related_memory_ids ({type(refs).__name__}, not a list); "
                    f"keeping the memory itself"
                )
                memory_dict["related_memory_ids"] = []
            else:
                valid_refs = []
                for ref in refs:
                    if isinstance(ref, dict) and "id" in ref and "bond" in ref:
                        valid_refs.append(ref)
                    else:
                        logger.warning(
                            f"Dropping malformed related_memory_ids entry {ref!r}. "
                            f"Expected {{'id': str, 'bond': str}}; keeping the rest of the memory"
                        )
                memory_dict["related_memory_ids"] = valid_refs

        # FIX: Validate numeric fields with fallbacks
        if "importance_score" in memory_dict:
            importance = memory_dict["importance_score"]
            if not isinstance(importance, (int, float)) or not (0.0 <= importance <= 1.0):
                logger.warning(f"Fixing invalid importance_score {importance} -> None (will use default)")
                memory_dict.pop("importance_score", None)

        # Validate and filter entities list — one shared shape contract for
        # every entity source (extraction responses and the pending drain's
        # entity pass use the same sanitizer).
        memory_dict["entities"] = sanitize_entity_list(memory_dict.get("entities"))

        return True

    def _is_duplicate_memory(
        self,
        memory_dict: Dict[str, Any],
        memory_context: MemoryContext
    ) -> DuplicateCheckResult:
        """
        Check if extracted memory is duplicate of existing memory.

        Uses three-stage checking:
        1. Fuzzy text matching against memory context (fast, catches variations)
        2. Vector similarity search (slower but semantic)
        3. Garbage collection handles deep similarity checks later

        Args:
            memory_dict: Extracted memory dictionary
            memory_context: Memory context with existing IDs and texts

        Returns:
            Tuple of (is_duplicate, similarity_score, duplicate_id)
        """
        memory_text = memory_dict.get("text", "").strip()
        if not memory_text:
            return DuplicateCheckResult(False, None, None)

        # Stage 1: Fuzzy text matching (wider net than exact, cheaper than vector)
        if memory_context:
            context_texts = memory_context.get("memory_texts", [])

            # Dict format: {uuid: text} — produced by extraction_engine.py
            if isinstance(context_texts, dict):
                for memory_id, existing_text in context_texts.items():
                    if not existing_text:
                        continue

                    # Use rapidfuzz for fuzzy matching
                    similarity = fuzz.ratio(existing_text.strip(), memory_text) / 100.0
                    if similarity >= 0.95:  # High fuzzy threshold for near-duplicates
                        logger.debug(
                            f"Fuzzy match found: similarity {similarity:.3f} "
                            f"with existing memory {memory_id}"
                        )
                        return DuplicateCheckResult(True, similarity, memory_id)

        # Stage 2: Vector similarity search
        similar_memories = self.vector_ops.find_similar_for_dedup(
            query_text=memory_text,
            limit=5,
            similarity_threshold=DEDUP_SIMILARITY_THRESHOLD,
            min_importance=0.001  # Filter cold storage (0.0) memories
        )

        if not similar_memories:
            return DuplicateCheckResult(False, None, None)

        # Extract best match score and ID
        best_score = None
        best_memory_id = None

        for memory in similar_memories:
            score = memory.similarity_score
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_memory_id = memory.id

        # Fallback if no scores extracted
        if best_score is None:
            best_score = DEDUP_SIMILARITY_THRESHOLD
            if similar_memories:
                best_memory_id = similar_memories[0].id

        return DuplicateCheckResult(True, best_score, best_memory_id)
