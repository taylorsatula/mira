"""
Trigger Rules API -- CRUD for per-user sidebar trigger filter rules.

Settings-tier configuration (not conversational). Each rule specifies
which trigger it belongs to (trigger_id), where to look (scope), what
field to match (field), and a regex pattern. The trigger reads its own
rules by filtering on trigger_id.

Endpoints:
  GET    /triggers/rules              — list all rules (optional ?trigger_id= filter)
  POST   /triggers/rules              — create a rule
  PUT    /triggers/rules/{rule_id}    — update a rule
  DELETE /triggers/rules/{rule_id}    — delete a rule
"""
import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from auth.api import get_current_user
from auth.types import SessionData, APITokenContext
from cns.api.base import (
    ValidationError,
    NotFoundError,
    create_success_response,
    create_error_response,
    generate_request_id,
)
from utils.timezone_utils import format_utc_iso, utc_now
from utils.user_context import set_current_user_id
from utils.userdata_manager import get_user_data_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Known trigger types and their valid fields
_TRIGGER_FIELDS: dict[str, set[str]] = {
    "imap_email": {"from", "subject", "body"},
}


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------

class CreateRuleRequest(BaseModel):
    trigger_id: str = Field(description="Trigger this rule belongs to (e.g. 'imap_email')")
    scope: str = Field(default="INBOX", description="Where to look (e.g. IMAP folder name)")
    field: str = Field(description="Which part to match: 'from', 'subject', or 'body'")
    pattern: str = Field(description="Regex pattern (matched case-insensitively)")
    prompt: str | None = Field(default=None, description="Agent system prompt for items matching this rule")


class UpdateRuleRequest(BaseModel):
    scope: str | None = Field(default=None, description="New scope value")
    field: str | None = Field(default=None, description="New field value")
    pattern: str | None = Field(default=None, description="New regex pattern")
    prompt: str | None = Field(default=None, description="Agent system prompt for items matching this rule")
    enabled: bool | None = Field(default=None, description="Enable or disable the rule")


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------

def _validate_trigger_field(trigger_id: str, field: str) -> None:
    valid_fields = _TRIGGER_FIELDS.get(trigger_id)
    if valid_fields is None:
        known = ", ".join(sorted(_TRIGGER_FIELDS.keys()))
        raise ValidationError(
            f"Unknown trigger_id '{trigger_id}'",
            {"known_triggers": known},
        )
    if field not in valid_fields:
        raise ValidationError(
            f"Invalid field '{field}' for trigger '{trigger_id}'",
            {"valid_fields": sorted(valid_fields)},
        )


def _validate_pattern(pattern: str) -> None:
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValidationError(f"Invalid regex pattern: {e}")


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.get("/triggers/rules")
def list_rules(
    response: Response,
    trigger_id: str | None = None,
    current_user: SessionData | APITokenContext = Depends(get_current_user),
) -> dict[str, Any]:
    """List trigger rules, optionally filtered by trigger_id."""
    set_current_user_id(current_user.user_id)
    request_id = generate_request_id()

    try:
        db = get_user_data_manager(current_user.user_id)

        if trigger_id:
            rows = db.select(
                "trigger_rules",
                where="trigger_id = :trigger_id",
                params={"trigger_id": trigger_id},
                order_by="scope, id",
            )
        else:
            rows = db.select(
                "trigger_rules",
                order_by="trigger_id, scope, id",
            )

        rules = [
            {
                "id": r["id"],
                "trigger_id": r["trigger_id"],
                "scope": r["scope"],
                "field": r["field"],
                "pattern": r["pattern"],
                "prompt": r.get("prompt"),
                "enabled": bool(r["enabled"]),
            }
            for r in rows
        ]

        return create_success_response(
            data={"rules": rules, "count": len(rules)},
            meta={"request_id": request_id, "timestamp": format_utc_iso(utc_now())},
        ).to_dict()

    except Exception as e:
        logger.error(f"Error listing trigger rules: {e}", exc_info=True)
        api_response = create_error_response(e, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.post("/triggers/rules")
def create_rule(
    body: CreateRuleRequest,
    response: Response,
    current_user: SessionData | APITokenContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a new trigger rule."""
    set_current_user_id(current_user.user_id)
    request_id = generate_request_id()

    try:
        _validate_trigger_field(body.trigger_id, body.field)
        _validate_pattern(body.pattern)

        db = get_user_data_manager(current_user.user_id)
        insert_data: dict[str, Any] = {
            "trigger_id": body.trigger_id,
            "scope": body.scope,
            "field": body.field,
            "pattern": body.pattern,
            "enabled": 1,
        }
        if body.prompt is not None:
            insert_data["prompt"] = body.prompt
        row_id = db.insert("trigger_rules", insert_data)

        return create_success_response(
            data={
                "rule": {
                    "id": int(row_id),
                    "trigger_id": body.trigger_id,
                    "scope": body.scope,
                    "field": body.field,
                    "pattern": body.pattern,
                    "prompt": body.prompt,
                    "enabled": True,
                },
                "message": "Rule created",
            },
            meta={"request_id": request_id, "timestamp": format_utc_iso(utc_now())},
        ).to_dict()

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error creating trigger rule: {e}", exc_info=True)
        api_response = create_error_response(e, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.put("/triggers/rules/{rule_id}")
def update_rule(
    rule_id: int,
    body: UpdateRuleRequest,
    response: Response,
    current_user: SessionData | APITokenContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Update an existing trigger rule."""
    set_current_user_id(current_user.user_id)
    request_id = generate_request_id()

    try:
        db = get_user_data_manager(current_user.user_id)

        existing = db.select(
            "trigger_rules",
            where="id = :id",
            params={"id": rule_id},
        )
        if not existing:
            raise NotFoundError("trigger_rule", str(rule_id))

        updates: dict[str, Any] = {}

        if body.field is not None:
            _validate_trigger_field(existing[0]["trigger_id"], body.field)
            updates["field"] = body.field

        if body.pattern is not None:
            _validate_pattern(body.pattern)
            updates["pattern"] = body.pattern

        if body.scope is not None:
            updates["scope"] = body.scope

        if "prompt" in body.model_fields_set:
            updates["prompt"] = body.prompt

        if body.enabled is not None:
            updates["enabled"] = 1 if body.enabled else 0

        if not updates:
            raise ValidationError("No fields to update")

        db.update(
            "trigger_rules",
            data=updates,
            where="id = :rule_id",
            params={"rule_id": rule_id},
        )

        return create_success_response(
            data={"message": "Rule updated", "rule_id": rule_id},
            meta={"request_id": request_id, "timestamp": format_utc_iso(utc_now())},
        ).to_dict()

    except (ValidationError, NotFoundError):
        raise
    except Exception as e:
        logger.error(f"Error updating trigger rule: {e}", exc_info=True)
        api_response = create_error_response(e, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.delete("/triggers/rules/{rule_id}")
def delete_rule(
    rule_id: int,
    response: Response,
    current_user: SessionData | APITokenContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Delete a trigger rule."""
    set_current_user_id(current_user.user_id)
    request_id = generate_request_id()

    try:
        db = get_user_data_manager(current_user.user_id)
        deleted = db.delete(
            "trigger_rules",
            where="id = :id",
            params={"id": rule_id},
        )

        if deleted == 0:
            raise NotFoundError("trigger_rule", str(rule_id))

        return create_success_response(
            data={"message": "Rule deleted", "rule_id": rule_id},
            meta={"request_id": request_id, "timestamp": format_utc_iso(utc_now())},
        ).to_dict()

    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error deleting trigger rule: {e}", exc_info=True)
        api_response = create_error_response(e, request_id)
        response.status_code = 500
        return api_response.to_dict()


@router.get("/triggers/conflicts")
def list_conflicts(
    response: Response,
    current_user: SessionData | APITokenContext = Depends(get_current_user),
) -> dict[str, Any]:
    """List recent rule conflicts from sidebar_activity."""
    set_current_user_id(current_user.user_id)
    request_id = generate_request_id()

    try:
        db = get_user_data_manager(current_user.user_id)

        from agents.base import ensure_activity_schema
        ensure_activity_schema(db)

        rows = db.execute(
            "SELECT thread_id, summary, updated_at FROM sidebar_activity "
            "WHERE status = 'conflict' ORDER BY updated_at DESC LIMIT 20",
        )

        conflicts = [
            {
                "item_id": r["thread_id"],
                "summary": r["summary"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

        return create_success_response(
            data={"conflicts": conflicts, "count": len(conflicts)},
            meta={"request_id": request_id, "timestamp": format_utc_iso(utc_now())},
        ).to_dict()

    except Exception as e:
        logger.error(f"Error listing trigger conflicts: {e}", exc_info=True)
        api_response = create_error_response(e, request_id)
        response.status_code = 500
        return api_response.to_dict()
