"""
System prompt section parser for the user model pipeline.

Extracts section IDs and content from the system prompt XML structure.
Used by both assessment extraction and user model synthesis to establish
the canonical vocabulary of section IDs that flows end-to-end.
"""
import re
from dataclasses import dataclass
from typing import List

# Sections excluded from assessment (contain runtime data, not behavioral contract)
_BLOCKLISTED_SECTIONS = frozenset({"user", "environment"})


@dataclass(frozen=True)
class SystemPromptSection:
    """A parsed section from the system prompt."""
    section_id: str
    content: str


def _parse_system_prompt_sections(raw_prompt: str) -> List[SystemPromptSection]:
    """
    Extract top-level sections from the system prompt XML.

    Only returns sections that are direct children of the <mira:system_prompt>
    wrapper, not nested tags (e.g., <memory_refs> inside <output_directives>
    is excluded because it's nested).

    Args:
        raw_prompt: Raw system prompt text

    Returns:
        List of SystemPromptSection with section_id and content
    """
    # Match all tag pairs with their spans
    pattern = r'<(\w+)>(.*?)</\1>'
    all_matches = []

    for match in re.finditer(pattern, raw_prompt, re.DOTALL):
        section_id = match.group(1)
        if ':' in section_id:
            continue
        all_matches.append((section_id, match.group(2).strip(), match.start(), match.end()))

    # Filter to top-level only: exclude any match whose span falls inside another match
    sections = []
    for section_id, content, start, end in all_matches:
        is_nested = any(
            other_start < start and end < other_end
            for other_id, _, other_start, other_end in all_matches
            if other_id != section_id
        )
        if not is_nested:
            sections.append(SystemPromptSection(section_id=section_id, content=content))

    return sections


def _filter_sections(
    sections: List[SystemPromptSection],
    blocklist: frozenset = _BLOCKLISTED_SECTIONS
) -> List[SystemPromptSection]:
    """Remove blocklisted sections from a section list."""
    return [s for s in sections if s.section_id not in blocklist]


def anonymize_prompt(raw_prompt: str) -> str:
    """
    Remove user-identifying information from the system prompt.

    Used when passing the prompt to the assessor so it can evaluate behavior
    against the contract without user-specific details.

    Replacements:
        - {first_name} template var -> "The User"
        - {relative time since account creation} -> "a while"
        - <user>...</user> block -> removed entirely
    """
    result = raw_prompt

    # Replace template variables
    result = result.replace("{first_name}", "The User")
    result = result.replace("{relative time since account creation}", "a while")

    # Remove blocklisted sections entirely
    result = re.sub(r'<user>.*?</user>', '', result, flags=re.DOTALL)
    result = re.sub(r'<environment>.*?</environment>', '', result, flags=re.DOTALL)

    return result.strip()


def format_section_list(sections: List[SystemPromptSection]) -> str:
    """Format section IDs as a bullet list for prompt injection."""
    return "\n".join(f"- {s.section_id}" for s in sections)


def get_assessable_sections(raw_prompt: str) -> List[SystemPromptSection]:
    """
    Parse the system prompt and return only assessable (non-blocklisted) sections.

    Convenience function combining parse + filter.
    """
    return _filter_sections(_parse_system_prompt_sections(raw_prompt))


def get_assessable_section_ids(raw_prompt: str) -> List[str]:
    """
    The canonical assessable section vocabulary, in system-prompt order.

    Single source for every consumer of the section taxonomy: prompt
    examples that name sections (generated, never hand-written) and
    validate_section_anchors() both derive from this list, so the vocabulary
    cannot drift between what the LLM is shown and what the code enforces.
    """
    return [s.section_id for s in get_assessable_sections(raw_prompt)]


def validate_section_anchors(user_model_xml: str, raw_prompt: str) -> None:
    """
    Deterministically validate every section anchor in a user model XML.

    The single sanctioned check for the section-anchor hazard: LLM-emitted
    <mira:observation>/<mira:topic> anchors must name an assessable system
    prompt section. Attribute regions are matched order-independently (same
    approach as LoraTrinket), so attribute-order drift cannot bypass the
    check. Callers must run this before a user model is stored or previewed —
    an invalid anchor is never left for the LLM critic to catch.

    Args:
        user_model_xml: Candidate user model XML (<mira:user_model> document)
        raw_prompt: The system prompt whose sections define the valid anchors

    Raises:
        ValueError: naming every invalid anchor and the full valid set
    """
    valid = get_assessable_section_ids(raw_prompt)
    valid_set = set(valid)

    invalid: List[str] = []
    for tag in ("mira:observation", "mira:topic"):
        for match in re.finditer(rf"<{tag}\b([^>]*)>", user_model_xml):
            section_match = re.search(r'\bsection="([^"]*)"', match.group(1))
            if section_match is None:
                # A tag with no section attribute at all is a malformed-model
                # problem, owned by the parsers; this check owns only anchors
                # that exist and are wrong.
                continue
            section = section_match.group(1).strip()
            if section not in valid_set and section not in invalid:
                invalid.append(section)

    if invalid:
        raise ValueError(
            f"Invalid section anchor(s): {', '.join(invalid)}. Every "
            "<mira:observation> and <mira:topic> section attribute must be "
            f"one of the assessable system prompt sections: {', '.join(valid)}."
        )
