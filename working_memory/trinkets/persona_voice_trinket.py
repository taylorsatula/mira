"""Renders the install-chosen persona voice into its own prompt slot.

The voice is a repo constant (`config/persona_voices.py`) selected by
`SystemConfig.persona_voice`, not a database row. That is deliberate: the character
is frozen by construction, because nothing that rewrites the evidence-derived
`persona_directives` column can reach a constant in this file. The evidence lane
keeps learning how to behave with one person; this slot holds who the agent is.

It renders as a passive block. Nothing here addresses the agent as an instruction
to follow, because a persona is a disposition it reasons from, and the six
sentences are already written in the indicative for exactly that reason.

An unknown name raises rather than rendering nothing: a preset the operator chose
that does not exist is a broken install, and a silent empty slot would present it
as a characterless Mira instead.
"""

from __future__ import annotations

from typing import Any

from config import config
from config.persona_voices import prose
from working_memory.trinkets.base import EventAwareTrinket


class PersonaVoiceTrinket(EventAwareTrinket):
    """Render the selected preset's six sentences into the ``persona_voice`` slot."""

    variable_name = "persona_voice"
    cache_policy = True

    def generate_content(self, context: dict[str, Any]) -> str:
        text = prose(config.system.persona_voice).strip()
        if not text:
            return ""
        return f"<persona_voice>\n{text}\n</persona_voice>"
