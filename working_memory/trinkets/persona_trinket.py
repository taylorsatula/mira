"""Inject the current Persona behavioral directives into MIRA's system prompt."""

from __future__ import annotations

from typing import Any

from cns.infrastructure.persona_repository import PersonaRepository
from utils.user_context import get_current_user_id
from working_memory.trinkets.base import EventAwareTrinket


class PersonaTrinket(EventAwareTrinket):
    """Render only the current immutable Persona revision's directives.

    Occupies its own prompt slot, ``persona_directives``. Upstream reuses the
    user model's ``behavioral_directives`` slot because crm_mira deleted the user
    model; mira-OSS keeps both subsystems (D1), and the two trinkets share one
    Valkey hash keyed by ``variable_name``, so they must not share a name.
    """

    variable_name = "persona_directives"
    cache_policy = True

    def __init__(self, event_bus, working_memory) -> None:
        self._repository = PersonaRepository()
        super().__init__(event_bus, working_memory)

    def generate_content(self, context: dict[str, Any]) -> str:
        revision = self._repository.get_current_revision(get_current_user_id())
        directives = revision.directives.strip()
        if not directives:
            return ""
        return f"<persona_directives>\n{directives}\n</persona_directives>"
