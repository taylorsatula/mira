"""
Skills catalog trinket — lists the user's available skills in the system prompt.

Renders the merged catalog (utils.skill_files.catalog_for): the boot-cached
global skills (working_memory/skills/) plus the user's own (per-user skills
dir), name + description only; bodies are loaded on demand via the
invoke_skill_tool tool. Global skills are stable for process life; per-user
skills are re-scanned per render — with the deliberate empty-directory
negative cache owned by utils.skill_files.user_catalog.
"""

import html
import logging
from typing import Dict, Any

from working_memory.trinkets.base import EventAwareTrinket
from utils.skill_files import catalog_for
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)


class SkillsCatalogTrinket(EventAwareTrinket):
    """Renders the user's skill catalog into the cached system prompt."""

    variable_name = "skills_catalog"
    cache_policy = True

    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Render the merged (user + boot-cached global) skill catalog.

        A per-user skill edit mid-session busts the cached prefix once —
        accepted; skills rarely change while a conversation is running.
        """
        skills = catalog_for(get_current_user_id())
        if not skills:
            return ""

        lines = ["<skills_catalog>"]
        for skill in skills:
            lines.append(
                f'  <skill name="{html.escape(skill.name, quote=True)}" '
                f'description="{html.escape(skill.description, quote=True)}"/>'
            )
        lines.append("</skills_catalog>")
        # Model-facing instruction — literal wording, per the LLM-caller
        # interface standard: name the tool, the exact-name requirement, and
        # what the model receives back.
        lines.append(
            "When a task matches a skill listed above, call the invoke_skill_tool tool "
            "with that skill's exact name. It returns the skill's full instructions, "
            "which also remain in your system prompt for the rest of this segment — "
            "follow them."
        )
        return "\n".join(lines)
