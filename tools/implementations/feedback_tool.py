"""
Feedback tool for capturing user friction, feature requests, and confusion.

Mira invokes this proactively when the user expresses frustration, asks about
missing features, seems confused, or provides any kind of feedback — positive
or negative. This gives the developer visibility into real user pain points
without requiring the user to send an email or file a ticket.
"""

import logging
from typing import Dict, Any

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from clients.postgres_client import PostgresClient
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class FeedbackToolConfig(BaseModel):
    """Configuration for the feedback_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")

registry.register("feedback_tool", FeedbackToolConfig)

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

_CATEGORIES = ("feature_request", "bug_report", "confusion", "praise", "other")


class FeedbackTool(Tool):
    """
    Captures user feedback and friction signals in PostgreSQL.

    Invoked proactively whenever the user:
      - Gives feedback of any kind (positive or negative)
      - Asks about or tries to use a feature that doesn't exist
      - Seems confused, frustrated, or is having trouble
      - Reports something broken

    These signals flow back to the developer so real-world pain points
    can be addressed — turning Mira into a self-evolving system.
    """

    name = "feedback_tool"
    parallel_safe = True

    simple_description = (
        "Captures user feedback, friction, and feature requests for the developer."
    )

    tool_schema = {
        "name": "feedback_tool",
        "description": (
            "Record user feedback — feature requests, bug reports, confusion, praise — "
            "and send it to the developer. Invoke proactively when you notice these "
            "signals, don't wait for the user to ask."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": list(_CATEGORIES),
                    "description": (
                        "Type of feedback: feature_request (missing capability), "
                        "bug_report (something broken), confusion (user is lost or "
                        "frustrated), praise (positive feedback), other (anything else)"
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Concise summary of the feedback or friction point. "
                        "Include enough detail for the developer to understand and act on it."
                    ),
                },
            },
            "required": ["category", "description"],
            "additionalProperties": False,
        },
    }

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, category: str, description: str, **kwargs) -> Dict[str, Any]:
        """
        Record a feedback entry in PostgreSQL.

        Args:
            category: One of feature_request, bug_report, confusion, praise, other.
            description: Concise summary of the feedback or friction point.

        Returns:
            Dict with success status and confirmation message.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if not category or category not in _CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of: {', '.join(_CATEGORIES)}"
            )
        if not description or not description.strip():
            raise ValueError("Description is required and must not be empty")

        user_id = get_current_user_id()
        db = PostgresClient("mira_service", user_id=user_id)

        try:
            db.execute_insert(
                """
                INSERT INTO user_feedback (user_id, category, description)
                VALUES (%(user_id)s, %(category)s, %(description)s)
                """,
                {
                    "user_id": user_id,
                    "category": category,
                    "description": description.strip(),
                },
            )
            self.logger.info(
                "Feedback recorded — user=%s category=%s",
                user_id,
                category,
            )
            return {
                "success": True,
                "message": (
                    "Feedback recorded — thanks! This helps improve Mira."
                ),
            }
        except Exception as e:
            self.logger.error("Failed to record feedback: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to record feedback: {e}") from e

    # ------------------------------------------------------------------
    # Usage examples
    # ------------------------------------------------------------------

    usage_examples = [
        {
            "input": {
                "category": "feature_request",
                "description": "User wants to export their reminders as a CSV file",
            },
            "output": {
                "success": True,
                "message": "Feedback recorded — thanks! This helps improve Mira.",
            },
        },
        {
            "input": {
                "category": "confusion",
                "description": (
                    "User is confused about how to attach a file to a reminder — "
                    "can't find where the attachment option is"
                ),
            },
            "output": {
                "success": True,
                "message": "Feedback recorded — thanks! This helps improve Mira.",
            },
        },
        {
            "input": {
                "category": "bug_report",
                "description": (
                    "User reports that weather_tool returns stale data — "
                    "shows sunny when it's clearly raining outside"
                ),
            },
            "output": {
                "success": True,
                "message": "Feedback recorded — thanks! This helps improve Mira.",
            },
        },
    ]
