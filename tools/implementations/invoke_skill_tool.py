"""
Invoke skill tool — loads an Agent Skill's instructions and activates them.

Returns the skill's markdown body to the model for this turn AND publishes an
activation to ActiveSkillsTrinket so the body persists in the system prompt
for the rest of the segment instead of scrolling away in tool-result history.
"""

import logging
from typing import Dict, Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from utils.skill_files import SkillNotFoundError, load_skill
from utils.user_context import get_current_user_id

if TYPE_CHECKING:
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class InvokeSkillToolConfig(BaseModel):
    """Configuration for the invoke_skill_tool tool."""
    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled by default"
    )


registry.register("invoke_skill_tool", InvokeSkillToolConfig)

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class InvokeSkillTool(Tool):
    """
    Loads a skill's full instructions by exact name.

    The available skills and their names are listed in the skills_catalog
    section of the system prompt — the user's own skills plus the boot-cached
    global catalog, user names winning on collision. Name resolution goes
    through scan results, never caller-built paths — a traversal-shaped
    argument matches nothing.
    """

    name = "invoke_skill_tool"
    parallel_safe = True

    simple_description = (
        "Load an Agent Skill's instructions for the current task; "
        "the skill then stays active in the system prompt."
    )

    tool_schema = {
        "name": "invoke_skill_tool",
        "description": (
            "Load a skill's full instructions. The skills_catalog section of your "
            "system prompt lists every available skill with a name and description. "
            "Pass the exact name of the skill whose instructions you need."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": (
                        "Exact name of the skill, matching a name attribute in the "
                        "skills_catalog section of your system prompt. Literal string — "
                        "not a pattern or guess."
                    ),
                },
            },
            "required": ["skill_name"],
            "additionalProperties": False,
        },
    }

    def __init__(self, working_memory: 'WorkingMemory'):
        # Required (no default) so the repository's DI injects it — a defaulted
        # Optional param is skipped and every publish would be a silent no-op.
        super().__init__()
        self.working_memory = working_memory

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, skill_name: str, **kwargs) -> Dict[str, Any]:
        """
        Load a skill body and activate it in the system prompt.

        Args:
            skill_name: Exact skill name from the catalog.

        Returns:
            Dict with success status, the skill name, and its full body —
            or success False with available names when nothing matches.

        Raises:
            ValueError: If skill_name is empty.
        """
        if not skill_name or not skill_name.strip():
            raise ValueError("skill_name is required and must not be empty")

        user_id = get_current_user_id()
        try:
            record, body = load_skill(user_id, skill_name.strip())
        except SkillNotFoundError as e:
            logger.info("Skill not found: %s (user=%s)", skill_name, user_id)
            return {
                "success": False,
                "message": (
                    f"{e} Re-check the names in the skills_catalog section of your "
                    "system prompt and pass the exact name."
                ),
            }

        self.working_memory.publish_trinket_update(
            target_trinket="ActiveSkillsTrinket",
            context={
                "action": "activate",
                "name": record.name,
                "body": body,
            },
        )

        logger.info("Skill loaded and activated: %s (user=%s)", record.name, user_id)
        return {
            "success": True,
            "skill": record.name,
            "body": body,
            "message": (
                f"Skill '{record.name}' loaded. Its instructions follow and also "
                "remain in your system prompt for the rest of this segment — follow them."
            ),
        }

    # ------------------------------------------------------------------
    # Usage examples
    # ------------------------------------------------------------------

    usage_examples = [
        {
            "input": {"skill_name": "sitrep"},
            "output": {
                "success": True,
                "skill": "sitrep",
                "body": "<full markdown body of the sitrep SKILL.md>",
                "message": "Skill 'sitrep' loaded. Its instructions follow…",
            },
        },
        {
            "input": {"skill_name": "no-such-skill"},
            "output": {
                "success": False,
                "message": "No skill named 'no-such-skill'. Available skills: sitrep. …",
            },
        },
    ]
