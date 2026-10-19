"""Validation helpers for user-supplied profile display names.

``validate_profile_name`` is the single gate for a name a user types (signup and
profile update both call it): it collapses internal whitespace, rejects empty,
over-100-character, control-character, and angle-bracket input, and returns the
normalized string for the caller to store. Angle brackets are refused rather
than escaped because these names are interpolated into prompt text, where an
escaped bracket is still a forged tag.

A ``ValueError`` names the field and the violated rule, so the API layer can
return the message to the client unchanged.
"""

import re


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_HTML_SPECIALS = re.compile(r"[<>]")
_WHITESPACE = re.compile(r"\s+")


def validate_profile_name(value: str, field_name: str) -> str:
    """Normalize and validate a profile name field."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    normalized = _WHITESPACE.sub(" ", value).strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    if len(normalized) > 100:
        raise ValueError(f"{field_name} must be 100 characters or fewer")
    if _CONTROL_CHARS.search(normalized):
        raise ValueError(f"{field_name} cannot contain control characters")
    if _HTML_SPECIALS.search(normalized):
        raise ValueError(f"{field_name} cannot contain angle brackets")

    return normalized
