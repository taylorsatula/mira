"""
LoRA (user model) refinement service.

Generates a revised user model based on user instructions and stores it as a
preview in Valkey. The user then accepts or declines the preview. Critic
validation runs before presenting the preview, ensuring the refined model
meets the same quality standards as synthesized models.

Unlike portrait refinement (plain prose), user model refinement must produce
valid XML and pass the critic's quality checks before the preview is shown.
"""
import logging
import re
from typing import Optional
from uuid import uuid4

from clients.llm_provider import LLMProvider, get_llm_provider

logger = logging.getLogger(__name__)

# TTL for LoRA refinement previews stored in Valkey. Rationale and the
# accept/decline expiry semantics live with the portrait constant
# (cns/services/portrait_service.py PREVIEW_TTL_SECONDS), the owning copy.
PREVIEW_TTL_SECONDS = 600

_LORA_PREVIEW_PREFIX = "lora_preview"

# Module-level state — loaded once per process
_refinement_system_prompt: Optional[str] = None
_critic_system_prompt: Optional[str] = None
_critic_user_template: Optional[str] = None
_llm_provider: Optional[LLMProvider] = None

# Section list for critic context (populated on first use)
_section_list: Optional[str] = None

CRITIC_MAX_ATTEMPTS = 3


def _load_prompts() -> None:
    """Load refinement and critic prompt templates (lazy, once per process).

    The refinement system prompt is a template: its example's section anchors
    ({example_section_a}/{example_section_b}) are filled from the assessable
    section vocabulary — the same source validate_section_anchors() enforces —
    so the example the model imitates can never drift from the taxonomy the
    code checks (Generated, Not Transcribed).
    """
    global _refinement_system_prompt, _critic_system_prompt, _critic_user_template

    if _refinement_system_prompt is not None:
        return

    from config.prompts.loader import load_prompt
    from cns.services.system_prompt_parser import get_assessable_section_ids
    from config import config
    section_ids = get_assessable_section_ids(config.system_prompt)
    if len(section_ids) < 2:
        raise ValueError(
            f"System prompt defines {len(section_ids)} assessable section(s); the "
            "refinement prompt example needs at least 2"
        )
    _refinement_system_prompt = load_prompt("lora_refinement_system.txt").format(
        example_section_a=section_ids[0],
        example_section_b=section_ids[1],
    )
    _critic_system_prompt = load_prompt("user_model_critic_system.txt")
    _critic_user_template = load_prompt("user_model_critic_user.txt")


def _get_section_list() -> str:
    """Pre-compute section list for critic context."""
    global _section_list
    if _section_list is not None:
        return _section_list

    from cns.services.system_prompt_parser import format_section_list, get_assessable_sections
    from config import config
    sections = get_assessable_sections(config.system_prompt)
    _section_list = format_section_list(sections)
    return _section_list


def _get_llm_provider() -> LLMProvider:
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = get_llm_provider()
    return _llm_provider


def read_lora(user_id: str) -> str:
    """
    Read the stored user model XML for a user.

    Returns "" when no model exists.
    """
    from cns.infrastructure.feedback_tracker import FeedbackTracker
    tracker = FeedbackTracker()
    content = tracker.get_lora_content(user_id)
    return content.get("synthesis_xml") or ""


def refine_lora(user_id: str, instructions: str) -> dict[str, str]:
    """
    Generate a revised user model based on user instructions and store as a preview.

    The existing model is combined with the user's refinement instructions
    and sent to the LLM. The result is validated by the critic before being
    stored as a preview. If the critic rejects the output, refinement is
    re-attempted with critic feedback (up to CRITIC_MAX_ATTEMPTS times).

    Returns dict with 'preview_id' and 'proposed' (the revised user model XML).
    Raises ValueError if no existing model or if refinement produces no output.

    Args:
        user_id: UUID string for the user (contextvar must already be set by caller)
        instructions: The user's free-text instructions for refining the user model
    """
    _load_prompts()

    current = read_lora(user_id)
    if not current:
        raise ValueError(
            "No user model exists to refine — user model synthesis runs "
            "automatically based on conversation feedback."
        )

    if not instructions or not instructions.strip():
        raise ValueError("Refinement instructions cannot be empty.")

    # Run refinement with critic validation loop
    candidate_xml = _run_refinement(current, instructions)

    # Critic validation loop (mirrors UserModelSynthesizer.synthesize)
    for attempt in range(CRITIC_MAX_ATTEMPTS):
        critic = _validate_with_critic(candidate_xml)

        if critic["passed"]:
            logger.info(
                "LoRA refinement passed critic (attempt %d) for user %s",
                attempt + 1, user_id
            )
            break

        logger.warning(
            "Critic rejected LoRA refinement (attempt %d) for user %s: %s",
            attempt + 1, user_id, critic["feedback"][:200]
        )

        if attempt < CRITIC_MAX_ATTEMPTS - 1:
            candidate_xml = _rerun_refinement_with_feedback(
                current, instructions, critic["feedback"]
            )
    else:
        # Circuit breaker: return the candidate anyway, but log a warning.
        # Unlike synthesis (which falls back to the previous model),
        # refinement is user-initiated — the user should see what was produced
        # and decide whether to accept it.
        logger.warning(
            "LoRA refinement exhausted %d critic attempts for user %s, "
            "presenting last candidate for user review",
            CRITIC_MAX_ATTEMPTS, user_id
        )

    if not candidate_xml or not candidate_xml.strip():
        raise ValueError("LoRA refinement produced no output — try different instructions.")

    # Deterministic section-anchor validation, before the preview is stored:
    # an observation anchored to a section the system prompt does not define
    # (e.g. a hallucinated anchor like "context") fails loud here with the
    # valid set named — never stored as a preview, never left for the LLM
    # critic or the user to catch. This also gates the critic-exhaustion
    # circuit breaker above: an invalid anchor is not presentable.
    from cns.services.system_prompt_parser import validate_section_anchors
    from config import config
    validate_section_anchors(candidate_xml, config.system_prompt)

    # Store preview in Valkey with TTL
    preview_id = str(uuid4())
    valkey_key = f"{_LORA_PREVIEW_PREFIX}:{user_id}:{preview_id}"

    from clients.valkey_client import get_valkey_client
    valkey = get_valkey_client()
    valkey.setex(valkey_key, PREVIEW_TTL_SECONDS, candidate_xml)

    logger.info(
        "LoRA preview created for user %s: preview_id=%s, %d chars",
        user_id, preview_id, len(candidate_xml)
    )

    return {"preview_id": preview_id, "proposed": candidate_xml}


def accept_lora(user_id: str, preview_id: str) -> None:
    """
    Accept a LoRA preview: fetch from Valkey and persist to feedback tracking.

    Raises ValueError if the preview_id is expired, invalid, or does not belong
    to this user.

    Args:
        user_id: UUID string for the user
        preview_id: Opaque ID returned by refine_lora()
    """
    proposed = _pop_preview(user_id, preview_id)

    from cns.infrastructure.feedback_tracker import FeedbackTracker
    tracker = FeedbackTracker()
    tracker.set_synthesis_output(user_id, proposed)
    _invalidate_lora_cache(user_id)

    logger.info(
        "LoRA preview accepted for user %s: preview_id=%s, %d chars",
        user_id, preview_id, len(proposed)
    )


def decline_lora(user_id: str, preview_id: str) -> None:
    """
    Decline a LoRA preview: delete from Valkey without persisting.

    No-op if the preview has already expired or been consumed.

    Args:
        user_id: UUID string for the user
        preview_id: Opaque ID returned by refine_lora()
    """
    if _pop_preview(user_id, preview_id, required=False) is None:
        logger.info(
            "LoRA preview already expired or consumed for user %s: preview_id=%s",
            user_id, preview_id
        )
        return

    logger.info(
        "LoRA preview declined for user %s: preview_id=%s",
        user_id, preview_id
    )


def _pop_preview(user_id: str, preview_id: str, required: bool = True) -> Optional[str]:
    """
    Fetch and delete a LoRA preview from Valkey (single-consume).

    Mirrors portrait_service._pop_preview exactly, with a different prefix.
    Returns None when the key is missing and required is False.
    """
    if not preview_id or not preview_id.strip():
        raise ValueError("preview_id is required.")

    valkey_key = f"{_LORA_PREVIEW_PREFIX}:{user_id}:{preview_id}"

    from clients.valkey_client import get_valkey_client
    valkey = get_valkey_client()

    proposed = valkey.getdel(valkey_key)
    if proposed is None:
        if not required:
            return None
        raise ValueError(
            "User model preview not found — it may have expired (10-minute lifetime). "
            "Please generate a new preview."
        )

    # Valkey returns bytes
    if isinstance(proposed, bytes):
        proposed = proposed.decode("utf-8")

    return proposed


def _run_refinement(current_xml: str, instructions: str) -> str:
    """Call the LLM to refine the user model."""
    assert _refinement_system_prompt is not None

    user_message = (
        f"## Valid Section Anchors\n"
        f"Every observation and check-in topic must be anchored to one of these "
        f"sections:\n{_get_section_list()}\n\n"
        f"## Existing User Model\n{current_xml}\n\n"
        f"## Instructions\n{instructions.strip()}"
    )

    llm = _get_llm_provider()
    response = llm.generate_response(
        messages=[{"role": "user", "content": user_message}],
        system_prompt=_refinement_system_prompt,
        model_config="batch",
    )
    return llm.extract_text_content(response).strip()


def _rerun_refinement_with_feedback(
    current_xml: str, instructions: str, critic_feedback: str
) -> str:
    """Rerun refinement with critic feedback appended."""
    assert _refinement_system_prompt is not None

    user_message = (
        f"## Valid Section Anchors\n"
        f"Every observation and check-in topic must be anchored to one of these "
        f"sections:\n{_get_section_list()}\n\n"
        f"## Existing User Model\n{current_xml}\n\n"
        f"## Instructions\n{instructions.strip()}\n\n"
        f"## Quality Critic Feedback\n"
        f"The quality critic flagged these issues in the previous attempt. "
        f"Revise the user model to address them:\n\n{critic_feedback}"
    )

    llm = _get_llm_provider()
    response = llm.generate_response(
        messages=[{"role": "user", "content": user_message}],
        system_prompt=_refinement_system_prompt,
        model_config="batch",
    )
    return llm.extract_text_content(response).strip()


def _validate_with_critic(candidate_xml: str) -> dict:
    """
    Run critic validation on a candidate user model.

    Mirrors UserModelSynthesizer._validate_with_critic.
    """
    assert _critic_system_prompt is not None
    assert _critic_user_template is not None

    section_list = _get_section_list()
    user_prompt = _critic_user_template.format(
        section_id_list=section_list,
        candidate_user_model=candidate_xml
    )

    llm_messages = [
        {"role": "system", "content": _critic_system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    llm = _get_llm_provider()
    response = llm.generate_response(
        messages=llm_messages,
        model_config="batch",
    )

    raw_output = llm.extract_text_content(response)

    status_match = re.search(r'<mira:critic_review\s+status="(\w+)"', raw_output)
    if not status_match:
        # Fail closed: an unparseable critic verdict is a validation failure,
        # never a pass (mirrors UserModelSynthesizer._validate_with_critic) —
        # the loop retries with the feedback below and, if exhaustion hits,
        # the last candidate is presented for explicit human review only.
        logger.warning("Could not parse critic output, failing validation")
        return {
            "passed": False,
            "feedback": (
                "The quality critic returned an unparseable response (no "
                "verdict could be extracted). Regenerate the user model, "
                "ensuring observations are section-anchored, evidence-grounded, "
                "free of personality labels, and internally consistent."
            )
        }

    status = status_match.group(1)
    if status == "pass":
        return {"passed": True, "feedback": ""}

    issues = []
    issue_pattern = r'<mira:issue\s+type="([^"]+)"\s+section="([^"]+)">(.*?)</mira:issue>'
    for match in re.finditer(issue_pattern, raw_output, re.DOTALL):
        issue_type = match.group(1)
        section = match.group(2)
        detail = match.group(3).strip()
        issues.append(f"[{issue_type} in {section}] {detail}")

    return {"passed": False, "feedback": "\n".join(issues)}


def _invalidate_lora_cache(user_id: str) -> None:
    """Invalidate the LoraTrinket's Valkey cache after accepting a refined model."""
    from cns.services.orchestrator import get_orchestrator

    get_orchestrator().working_memory.invalidate_trinket("LoraTrinket", user_id)
