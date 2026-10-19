"""
The one sanctioned way to read a prompt template out of `config/prompts/`.

`load_prompt()` resolves its argument against this directory and never
against the caller's cwd, refuses a name that does not end in `.txt` or that
resolves outside the directory, reads UTF-8, and strips. Subdirectory names
are ordinary arguments (`"agents/forage_system.txt"`).

Two consequences a caller has to plan for. A required template that is
missing, or empty after stripping, raises `FileNotFoundError` ("prompts are
system configuration, not optional features") — there is no empty-string
fallback to guard against, so a consumer that catches it would be masking a
broken deployment. And because the strip happens before the emptiness check,
a whitespace-only template is indistinguishable from an absent one: a
template cannot be switched off by emptying it.

`required=False` returns `""` for an absent file and is reserved for a genuine
addendum whose absence degrades a feature rather than the boot. It also
returns `""` for a file that exists and is empty, so the empty-versus-missing
distinction survives only on the required path.

Usage:
    from config.prompts import load_prompt

    system_prompt = load_prompt("subcortical_system.txt")
    agent_prompt = load_prompt("agents/forage_system.txt")
    instructions = load_prompt("thinking_block_instructions.txt", required=False)
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.resolve()
_ALLOWED_SUFFIX = ".txt"


def load_prompt(filename: str, *, required: bool = True) -> str:
    """Load a prompt file from config/prompts/.

    Args:
        filename: Relative path within config/prompts/ (e.g. "subcortical_system.txt",
                  "agents/forage_system.txt"). Must end in .txt.
        required: If True (default), raises FileNotFoundError when the file is missing
                  or empty after stripping. If False, returns an empty string when the
                  file is missing — use for optional addendum prompts that may not exist
                  for all deployments.

    Returns:
        The file contents as a string, stripped of leading/trailing whitespace.

    Raises:
        ValueError: If filename attempts path traversal or has a disallowed extension.
        FileNotFoundError: When required=True and the file does not exist or is empty.
    """
    if not filename.endswith(_ALLOWED_SUFFIX):
        raise ValueError(
            f"Prompt filename must end in {_ALLOWED_SUFFIX!r}, "
            f"got {filename!r}"
        )

    resolved = (_PROMPTS_DIR / filename).resolve()
    if not resolved.is_relative_to(_PROMPTS_DIR):
        raise ValueError(
            f"Prompt filename {filename!r} resolves outside config/prompts/ "
            f"(resolved to {resolved})"
        )

    if not resolved.exists():
        if required:
            raise FileNotFoundError(
                f"Required prompt file not found: {resolved}. "
                f"Prompts are system configuration, not optional features."
            )
        return ""

    content = resolved.read_text(encoding="utf-8").strip()

    if not content:
        if required:
            raise FileNotFoundError(
                f"Required prompt file is empty: {resolved}. "
                f"Prompt files must contain non-whitespace content."
            )
        return ""

    logger.debug("Loaded prompt: %s (%d chars)", filename, len(content))
    return content
