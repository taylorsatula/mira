"""Explicit development-only authentication controls."""

import os


def development_mode_enabled() -> bool:
    """Return whether the process was intentionally started in development mode."""
    return os.getenv("MIRA_DEV", "false").lower() in {"1", "true", "yes"}
