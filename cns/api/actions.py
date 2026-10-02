"""
Actions API endpoint - domain-routed state mutations.

Executes state-changing operations through domain-specific handlers that
call tools and services directly, just as MIRA does during continuums.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict
from enum import Enum
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, field_validator

from config import config

from utils.user_context import get_current_user_id, set_current_user_id, invalidate_user_preferences_cache
from auth.api import get_current_user
from auth.types import SessionData, APITokenContext
from .base import (
    APIError,
    BaseHandler,
    PropagatingHandler,
    ValidationError,
    NotFoundError,
    generate_request_id,
)
from utils.timezone_utils import utc_now, format_utc_iso
from clients.valkey_client import get_valkey_client
from working_memory.trinkets.base import TRINKET_KEY_PREFIX
from utils.userdata_manager import UserDataManager

logger = logging.getLogger(__name__)

router = APIRouter()



class DomainType(str, Enum):
    """Supported action domains."""
    REMINDER = "reminder"
    MEMORY = "memory"
    USER = "user"
    CONTACTS = "contacts"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CONTINUUM = "continuum"
    LORA = "lora"
    PERSONA = "persona"
    FEEDBACK = "feedback"
    PORTRAIT = "portrait"
    SKILLS = "skills"


class ActionRequest(BaseModel):
    """Action request schema."""
    domain: DomainType = Field(..., description="Domain for the action")
    action: str = Field(..., description="Action to perform")
    data: dict[str, Any] = Field(default_factory=dict, description="Action-specific data")
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        if not v.strip():
            raise ValueError("Action cannot be empty")
        return v.strip()


class ActionSchema(TypedDict, total=False):
    """Schema for action validation."""
    required: list[str]
    optional: list[str]
    types: dict[str, type | tuple[type, ...] | str]


class BaseDomainHandler(BaseHandler):
    """Base handler for domain-specific actions."""

    # Define available actions and their required/optional fields
    ACTIONS: dict[str, ActionSchema] = {}
    
    def __init__(self):
        super().__init__()  # Initialize BaseHandler (logger, thread pool)
        from utils.user_context import get_current_user_id
        self.user_id = get_current_user_id()
    
    def validate_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Validate action and its data against schema."""
        if action not in self.ACTIONS:
            available_actions = list(self.ACTIONS.keys())
            raise ValidationError(
                f"Unknown action '{action}' for {self.__class__.__name__}. "
                f"Available actions: {', '.join(available_actions)}"
            )
        
        # Get schema for this action
        schema = self.ACTIONS[action]
        required_fields = schema.get('required', [])
        optional_fields = schema.get('optional', [])
        all_fields = required_fields + optional_fields
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError(
                f"Missing required fields for action '{action}': {', '.join(missing_fields)}"
            )
        
        # Check for unknown fields
        unknown_fields = [field for field in data.keys() if field not in all_fields]
        if unknown_fields:
            raise ValidationError(
                f"Unknown fields for action '{action}': {', '.join(unknown_fields)}. "
                f"Valid fields: {', '.join(all_fields)}"
            )
        
        # Validate field types
        for field, value in data.items():
            if field in schema.get('types', {}):
                expected_type = schema['types'][field]
                
                # Special handling for UUID type
                if expected_type == 'uuid':
                    from uuid import UUID
                    try:
                        UUID(value)
                    except (ValueError, TypeError):
                        raise ValidationError(f"Field '{field}' must be a valid UUID")
                elif not isinstance(value, expected_type):
                    raise ValidationError(
                        f"Field '{field}' has invalid type, got {type(value).__name__}"
                    )
        
        return data
    
    def execute_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Execute the action. Override in subclasses."""
        raise NotImplementedError(f"Action '{action}' not implemented")


def _project_reminder(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Project a reminder tool dict onto documented API field names.

    The `encrypted__` prefix is a UserDataManager storage-column detail (it
    drives transparent Fernet encryption) and never crosses the API boundary:
    responses expose the documented fields with the prefix stripped and
    contact fields renamed to match the create/update action schema
    (contact_name, contact_email, contact_phone). Projection is an explicit
    per-field allowlist, so a new storage column cannot leak into responses
    by accident.
    """
    if raw is None:
        return None
    projected: dict[str, Any] = {
        "id": raw.get("id"),
        "title": raw.get("encrypted__title"),
        "description": raw.get("encrypted__description"),
        "additional_notes": raw.get("encrypted__additional_notes"),
        "resolution_note": raw.get("encrypted__resolution_note"),
        "reminder_date": raw.get("reminder_date"),
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "completed": raw.get("completed"),
        "completed_at": raw.get("completed_at"),
        "category": raw.get("category", "user"),
    }
    if raw.get("contact_uuid"):
        projected["contact_uuid"] = raw["contact_uuid"]
    for storage_key, api_key in (
        ("contact_encrypted__name", "contact_name"),
        ("contact_encrypted__email", "contact_email"),
        ("contact_encrypted__phone", "contact_phone"),
    ):
        if storage_key in raw:
            projected[api_key] = raw[storage_key]
    return projected


def _project_contact(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Project a contact dict (tool-formatted or raw row) onto documented API field names.

    Same storage-boundary rule as _project_reminder: the `encrypted__` prefix
    and the row's `id` storage key are projected to the documented API names
    (`name`, `email`, …, `uuid`).
    """
    if raw is None:
        return None
    projected: dict[str, Any] = {
        "uuid": raw.get("uuid", raw.get("id")),
        "name": raw.get("encrypted__name"),
        "email": raw.get("encrypted__email"),
        "phone": raw.get("encrypted__phone"),
        "street": raw.get("encrypted__street"),
        "city": raw.get("encrypted__city"),
        "state": raw.get("encrypted__state"),
        "zip": raw.get("encrypted__zip"),
        "pager_address": raw.get("encrypted__pager_address"),
    }
    if "created_at" in raw:
        projected["created_at"] = raw["created_at"]
    if "updated_at" in raw:
        projected["updated_at"] = raw["updated_at"]
    return projected


class ReminderDomainHandler(BaseDomainHandler):
    """Handler for reminder domain actions."""
    
    ACTIONS = {
        "complete": {
            "required": ["id"],
            "optional": ["resolution_note"],
            "types": {"id": str, "resolution_note": str}
        },
        "bulk_complete": {
            "required": ["ids"],
            "optional": ["resolution_note"],
            "types": {"ids": list, "resolution_note": str}
        },
        "create": {
            "required": ["title", "date"],
            "optional": ["description", "contact_name", "additional_notes"],
            "types": {
                "title": str,
                "date": str,
                "description": str,
                "contact_name": str,
                "additional_notes": str
            }
        },
        "update": {
            "required": ["id"],
            "optional": ["title", "date", "description", "contact_name", "additional_notes"],
            "types": {
                "id": str,
                "title": str,
                "date": str,
                "description": str,
                "contact_name": str,
                "additional_notes": str
            }
        },
        "delete": {
            "required": ["id"],
            "optional": [],
            "types": {"id": str}
        }
    }
    
    def execute_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Execute reminder actions using ReminderTool.

        Not-found is the tool's documented raise channel: the tool raises
        ReminderNotFoundError (a ValueError subclass) and this handler maps it
        to the standard 404 missing-id envelope, so an absent reminder can
        never be reported as completed/updated/deleted. bulk_complete
        aggregates each item's outcome honestly instead of assuming success.
        """
        from tools.implementations.reminder_tool import ReminderTool, ReminderNotFoundError
        reminder_tool = ReminderTool()
        
        try:
            if action == "complete":
                # Mark single reminder as completed
                run_kwargs: dict[str, Any] = {
                    "operation": "mark_completed",
                    "reminder_id": data["id"]
                }
                if data.get("resolution_note"):
                    run_kwargs["resolution_note"] = data["resolution_note"]
                result = reminder_tool.run(**run_kwargs)

                return {
                    "completed": True,
                    "reminder": _project_reminder(result.get("reminder")),
                    "message": result.get("message", "Reminder marked as completed")
                }

            elif action == "bulk_complete":
                # Mark multiple reminders as completed
                reminder_ids = data["ids"]
                resolution_note = data.get("resolution_note")

                # Validate that ids is a non-empty list of strings
                if not reminder_ids:
                    raise ValidationError("At least one reminder ID is required")

                if not all(isinstance(id, str) for id in reminder_ids):
                    raise ValidationError("All reminder IDs must be strings")

                completed = []
                failed = []

                for reminder_id in reminder_ids:
                    try:
                        run_kwargs = {
                            "operation": "mark_completed",
                            "reminder_id": reminder_id
                        }
                        if resolution_note:
                            run_kwargs["resolution_note"] = resolution_note
                        result = reminder_tool.run(**run_kwargs)
                        # Strict access: the tool's contract returns the
                        # completed reminder; a contract drift must surface as
                        # an error, not a titleless success entry.
                        completed.append({
                            "id": reminder_id,
                            "title": _project_reminder(result["reminder"])["title"]
                        })
                    except ValueError as e:
                        # Per-item failure (not-found or otherwise) is an
                        # honest failed-list entry with its reason — the rest
                        # of the batch still reports what actually completed.
                        failed.append({
                            "id": reminder_id,
                            "error": str(e)
                        })
                
                return {
                    "completed_count": len(completed),
                    "failed_count": len(failed),
                    "completed": completed,
                    "failed": failed,
                    "message": f"Completed {len(completed)} of {len(reminder_ids)} reminders"
                }

            elif action == "create":
                # Create new reminder with all provided fields
                result = reminder_tool.run(
                    operation="add_reminder",
                    **data  # Pass all validated data
                )
                return {
                    "created": True,
                    "duplicate": result.get("duplicate_detected", False),
                    "reminder": _project_reminder(result.get("reminder")),
                    "contact_found": result.get("contact_found", False),
                    "contact_info": _project_contact(result.get("contact_info")),
                    "message": result.get("message", "Reminder created")
                }
            
            elif action == "update":
                # Update existing reminder
                reminder_id = data["id"]
                # Only pass fields that were actually provided
                update_fields = {k: v for k, v in data.items() if k != "id" and v is not None}
                
                if not update_fields:
                    raise ValidationError("At least one field to update must be provided")
                
                result = reminder_tool.run(
                    operation="update_reminder",
                    reminder_id=reminder_id,
                    **update_fields
                )
                return {
                    "updated": True,
                    "reminder": _project_reminder(result.get("reminder")),
                    "updated_fields": result.get("updated_fields", []),
                    "message": result.get("message", "Reminder updated")
                }
            
            elif action == "delete":
                # Delete reminder
                result = reminder_tool.run(
                    operation="delete_reminder",
                    reminder_id=data["id"]
                )
                return {
                    "deleted": True,
                    "id": data["id"],
                    "message": result.get("message", "Reminder deleted")
                }
            
            else:
                raise ValidationError(f"Unknown action: {action}")

        except ReminderNotFoundError as e:
            # The tool's documented not-found channel: single-id actions map to
            # the standard 404 missing-id envelope instead of 200 false-success.
            raise NotFoundError("reminder", data.get("id", "")) from e
        except ValueError as e:
            # Tool raises ValueError for business logic errors
            raise ValidationError(str(e)) from e


class MemoryDomainHandler(BaseDomainHandler):
    """Handler for memory domain actions."""
    
    ACTIONS = {
        "create": {
            "required": ["content"],
            "optional": ["importance"],
            "types": {
                "content": str,
                "importance": (int, float)
            }
        },
        "delete": {
            "required": ["id"],
            "optional": [],
            "types": {"id": "uuid"}
        },
        "bulk_delete": {
            "required": ["ids"],
            "optional": [],
            "types": {"ids": list}
        }
    }
    
    def execute_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Execute memory actions using LTMemoryDB."""
        from lt_memory.db_access import LTMemoryDB
        from utils.database_session_manager import get_shared_session_manager

        # Set user context for LTMemoryDB operations
        set_current_user_id(self.user_id)

        # Use shared session manager to prevent connection pool exhaustion
        session_manager = get_shared_session_manager()
        lt_db = LTMemoryDB(session_manager)
        
        if action == "create":
            # Manual memory creation
            content = data["content"]
            importance = data.get("importance", 0.5)
            
            # Validate importance score
            if not 0 <= importance <= 1:
                raise ValidationError("Importance score must be between 0 and 1")
            
            # Generate embedding for the memory
            from clients.embeddings_provider import get_embeddings_provider
            from lt_memory.models import ExtractedMemory
            embeddings_provider = get_embeddings_provider()  # Use singleton
            # Document encoding for memory storage
            embedding = embeddings_provider.encode_deep(content)

            # Convert ndarray to list for database storage (serialization boundary)
            embedding_list = embedding.tolist()

            # Create ExtractedMemory object (embedding passed separately to store_memories)
            memory = ExtractedMemory(
                text=content,
                importance_score=importance
            )

            memory_ids = lt_db.store_memories([memory], embeddings=[embedding_list])
            
            if not memory_ids:
                raise ValidationError("Failed to create memory")
            
            # Get the created memory to return
            created_memory = lt_db.get_memory(memory_ids[0])

            return jsonable_encoder({
                "created": True,
                "memory": {
                    "id": created_memory.id,
                    "text": created_memory.text,
                    "importance_score": created_memory.importance_score,
                    "created_at": created_memory.created_at
                },
                "message": "Memory created successfully"
            })
        
        elif action == "delete":
            # Delete single memory
            memory_id = data["id"]
            
            # Verify memory exists
            memory = lt_db.get_memory(UUID(memory_id))
            if not memory:
                raise NotFoundError("memory", memory_id)

            # Archive the memory (soft delete)
            lt_db.archive_memory(UUID(memory_id))

            return {
                "deleted": True,
                "id": memory_id,
                "message": "Memory deleted successfully"
            }
        
        elif action == "bulk_delete":
            # Delete multiple memories
            memory_ids = data["ids"]
            
            # Validate that ids is a non-empty list of strings
            if not memory_ids:
                raise ValidationError("At least one memory ID is required")
            
            if not all(isinstance(id, str) for id in memory_ids):
                raise ValidationError("All memory IDs must be strings")

            try:
                parsed_ids = [UUID(mid) for mid in memory_ids]
            except ValueError:
                raise ValidationError("All memory IDs must be valid UUIDs")

            missing = [
                str(mid) for mid in parsed_ids
                if not lt_db.get_memory(mid)
            ]
            if missing:
                raise NotFoundError("memories", ", ".join(missing))

            # Archive memories (soft delete)
            for mid in parsed_ids:
                lt_db.archive_memory(mid)

            return {
                "deleted_count": len(memory_ids),
                "requested_count": len(memory_ids),
                "ids": memory_ids,
                "message": f"Deleted {len(memory_ids)} of {len(memory_ids)} memories"
            }
        
        else:
            raise ValidationError(f"Unknown action: {action}")


class ContactsDomainHandler(BaseDomainHandler):
    """Handler for contacts domain actions."""
    
    ACTIONS = {
        "create": {
            "required": ["name"],
            "optional": ["email", "phone"],
            "types": {
                "name": str,
                "email": str,
                "phone": str
            }
        },
        "get": {
            "required": ["identifier"],
            "optional": [],
            "types": {"identifier": str}
        },
        "list": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "update": {
            "required": ["identifier"],
            "optional": ["name", "email", "phone"],
            "types": {
                "identifier": str,
                "name": str,
                "email": str,
                "phone": str
            }
        },
        "delete": {
            "required": ["identifier"],
            "optional": [],
            "types": {"identifier": str}
        }
    }
    
    def execute_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Execute contacts actions using ContactsTool.

        Not-found is the tool's raise channel: ContactNotFoundError (a
        ValueError subclass) maps to the standard 404 missing-id envelope, so
        an absent contact can never be reported as found/updated/deleted. The
        tool's designed soft paths — ambiguous matches and partial-match
        confirmations — are surfaced truthfully as ambiguous/needs_confirmation
        payloads with deleted/updated: False, never labeled completed
        operations.
        """
        from tools.implementations.contacts_tool import ContactsTool, ContactNotFoundError
        contacts_tool = ContactsTool()
        
        try:
            if action == "create":
                # Create new contact
                result = contacts_tool.run(
                    operation="add_contact",
                    **data  # Pass all validated data
                )
                return {
                    "created": True,
                    "duplicate": result.get("duplicate", False),
                    "contact": _project_contact(result.get("contact")),
                    "message": result.get("message", "Contact created")
                }
            
            elif action == "get":
                # Get contact by UUID or name
                result = contacts_tool.run(
                    operation="get_contact",
                    identifier=data["identifier"]
                )
                if result.get("ambiguous"):
                    return {
                        "found": False,
                        "ambiguous": True,
                        "matches": [_project_contact(m) for m in result["matches"]],
                        "message": result.get("message", "Multiple contacts match")
                    }
                return {
                    "found": True,
                    "contact": _project_contact(result["contact"]),
                    "message": result.get("message", "Contact found")
                }
            
            elif action == "list":
                # List all contacts
                result = contacts_tool.run(
                    operation="list_contacts"
                )
                return {
                    "contacts": [_project_contact(c) for c in result["contacts"]],
                    "count": len(result["contacts"]),
                    "message": result.get("message", "Contacts retrieved")
                }
            
            elif action == "update":
                # Update existing contact
                identifier = data["identifier"]
                # Only pass fields that were actually provided
                update_fields = {k: v for k, v in data.items() if k != "identifier" and v is not None}
                
                if not update_fields:
                    raise ValidationError("At least one field to update must be provided")
                
                result = contacts_tool.run(
                    operation="update_contact",
                    identifier=identifier,
                    **update_fields
                )
                if result.get("ambiguous"):
                    return {
                        "updated": False,
                        "ambiguous": True,
                        "matches": [_project_contact(m) for m in result["matches"]],
                        "message": result.get("message", "Multiple contacts match")
                    }
                if result.get("needs_confirmation"):
                    return {
                        "updated": False,
                        "needs_confirmation": True,
                        "candidate": _project_contact(result["candidate"]),
                        "message": result.get("message", "Partial match needs confirmation")
                    }
                return {
                    "updated": True,
                    "contact": _project_contact(result["contact"]),
                    "message": result.get("message", "Contact updated")
                }
            
            elif action == "delete":
                # Delete contact
                result = contacts_tool.run(
                    operation="delete_contact",
                    identifier=data["identifier"]
                )
                if result.get("ambiguous"):
                    return {
                        "deleted": False,
                        "ambiguous": True,
                        "matches": [_project_contact(m) for m in result["matches"]],
                        "message": result.get("message", "Multiple contacts match")
                    }
                if result.get("needs_confirmation"):
                    return {
                        "deleted": False,
                        "needs_confirmation": True,
                        "candidate": _project_contact(result["candidate"]),
                        "message": result.get("message", "Partial match needs confirmation")
                    }
                return {
                    "deleted": True,
                    "deleted_contact": _project_contact(result["deleted_contact"]),
                    "message": result.get("message", "Contact deleted")
                }
            
            else:
                raise ValidationError(f"Unknown action: {action}")
                
        except ContactNotFoundError as e:
            # The tool's documented not-found channel: the standard 404
            # missing-id envelope instead of found:true/400 false shapes.
            raise NotFoundError("contact", data.get("identifier", "")) from e
        except ValueError as e:
            # Tool raises ValueError for business logic errors
            raise ValidationError(str(e)) from e


class UserDomainHandler(BaseDomainHandler):
    """Handler for user preference/settings actions."""

    ACTIONS = {
        "get_profile": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "update_profile": {
            "required": [],
            "optional": ["first_name", "last_name", "timezone", "temperature_unit"],
            "types": {
                "first_name": str,
                "last_name": str,
                "timezone": str,
                "temperature_unit": str
            }
        },
        "store_calendar_config": {
            "required": ["calendar_url"],
            "optional": [],
            "types": {
                "calendar_url": str
            }
        },
        "get_calendar_config": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "store_http_credential": {
            "required": ["name", "value", "allowed_domains"],
            "optional": [],
            "types": {
                "name": str,
                "value": str,
                "allowed_domains": list
            }
        },
        "list_http_credentials": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "delete_http_credential": {
            "required": ["name"],
            "optional": [],
            "types": {
                "name": str
            }
        },
        "set_effort_override": {
            "required": ["effort"],
            "optional": [],
            "types": {
                "effort": str
            }
        },
        "get_effort_override": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "clear_effort_override": {
            "required": [],
            "optional": [],
            "types": {}
        }
    }
    
    def execute_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Execute user preference actions."""
        if action == "get_profile":
            from utils.user_context import get_user_preferences

            prefs = get_user_preferences()
            return {
                "success": True,
                "profile": {
                    "first_name": prefs.first_name,
                    "last_name": prefs.last_name,
                    "timezone": prefs.timezone,
                    "temperature_unit": prefs.temperature_unit
                }
            }

        elif action == "update_profile":
            from utils.database_session_manager import get_shared_session_manager
            from utils.profile_validation import validate_profile_name
            # get_valkey_client stays module-level (imported at :35) — a local
            # re-import here shadows it for the whole execute_action scope and
            # UnboundLocalErrors the effort-override branches below.

            # At least one field must be provided
            if not any(k in data for k in ["first_name", "last_name", "timezone", "temperature_unit"]):
                raise ValidationError("At least one field (first_name, last_name, timezone, temperature_unit) must be provided")

            # Validate timezone if provided
            if "timezone" in data:
                from utils.timezone_utils import validate_timezone
                try:
                    validate_timezone(data["timezone"])
                except Exception:
                    raise ValidationError(f"Invalid timezone: {data['timezone']}")

            # Validate temperature_unit if provided
            if "temperature_unit" in data:
                valid_units = ("fahrenheit", "celsius")
                if data["temperature_unit"] not in valid_units:
                    raise ValidationError(
                        f"Invalid temperature unit '{data['temperature_unit']}'. "
                        f"Valid units: {', '.join(valid_units)}"
                    )

            # Build UPDATE query dynamically based on provided fields
            update_fields = []
            params = {"user_id": self.user_id}

            if "first_name" in data:
                update_fields.append("first_name = %(first_name)s")
                try:
                    params["first_name"] = validate_profile_name(data["first_name"], "first_name")
                except ValueError as e:
                    raise ValidationError(str(e))

            if "last_name" in data:
                update_fields.append("last_name = %(last_name)s")
                try:
                    params["last_name"] = validate_profile_name(data["last_name"], "last_name")
                except ValueError as e:
                    raise ValidationError(str(e))

            if "timezone" in data:
                update_fields.append("timezone = %(timezone)s")
                params["timezone"] = validate_timezone(data["timezone"])

            if "temperature_unit" in data:
                update_fields.append("temperature_unit = %(temperature_unit)s")
                params["temperature_unit"] = data["temperature_unit"]

            # Execute update
            session_manager = get_shared_session_manager()
            with session_manager.get_session(self.user_id) as db:
                db.execute_update(
                    f"UPDATE users SET {', '.join(update_fields)} WHERE id = %(user_id)s",
                    params
                )

            # Invalidate user preferences cache so changes take effect immediately
            invalidate_user_preferences_cache(self.user_id)

            return {
                "success": True,
                "updated_fields": list(data.keys()),
                "message": "Profile updated successfully"
            }

        elif action == "store_calendar_config":
            from utils.user_credentials import UserCredentialService

            calendar_url = data["calendar_url"]

            # Store calendar URL in user credentials
            credential_service = UserCredentialService()
            credential_service.store_credential(
                credential_type="calendar_url",
                service_name="calendar",
                credential_value=calendar_url
            )

            return {
                "success": True,
                "message": "Calendar URL stored successfully"
            }
        
        elif action == "get_calendar_config":
            from utils.user_credentials import UserCredentialService

            credential_service = UserCredentialService()
            calendar_url = credential_service.get_credential(
                credential_type="calendar_url",
                service_name="calendar"
            )

            return {
                "success": True,
                "calendar_url": calendar_url if calendar_url else None,
                "message": "Calendar configuration retrieved" if calendar_url else "No calendar URL configured"
            }

        elif action == "store_http_credential":
            from utils.user_credentials import UserCredentialService
            from utils.url_safety import validate_allowed_domains

            name = data["name"]
            value = data["value"]
            try:
                allowed_domains = validate_allowed_domains(data["allowed_domains"])
            except ValueError as e:
                raise ValidationError(str(e))

            # Validate name format (alphanumeric + underscore/hyphen)
            if not name or not name.replace("_", "").replace("-", "").isalnum():
                raise ValidationError("Credential name must be alphanumeric (underscores/hyphens allowed)")

            credential_service = UserCredentialService()
            credential_service.store_credential(
                credential_type="api_key",
                service_name=name,
                credential_value=value,
                metadata={"allowed_domains": allowed_domains}
            )

            return {
                "success": True,
                "name": name,
                "allowed_domains": allowed_domains,
                "message": f"Credential '{name}' stored successfully"
            }

        elif action == "list_http_credentials":
            from utils.user_credentials import UserCredentialService

            credential_service = UserCredentialService()
            all_creds = credential_service.list_user_credentials()

            # Extract api_key credentials - structure is {"api_key": {"name": {...}}}
            http_creds_dict = all_creds.get("api_key", {})
            http_creds = [
                {
                    "name": service_name,
                    "created_at": cred_data.get("created_at"),
                    "allowed_domains": cred_data.get("metadata", {}).get("allowed_domains", [])
                }
                for service_name, cred_data in http_creds_dict.items()
            ]

            return {
                "success": True,
                "credentials": http_creds,
                "count": len(http_creds)
            }

        elif action == "delete_http_credential":
            from utils.user_credentials import UserCredentialService

            name = data["name"]
            credential_service = UserCredentialService()
            deleted = credential_service.delete_credential(
                credential_type="api_key",
                service_name=name
            )

            if not deleted:
                raise NotFoundError("credential", name)

            return {
                "success": True,
                "name": name,
                "message": f"Credential '{name}' deleted"
            }

        elif action == "set_effort_override":
            from clients.llm.types import EFFORT_LEVELS

            effort_value = data["effort"]
            if effort_value not in EFFORT_LEVELS:
                raise ValidationError(
                    f"Invalid effort level '{effort_value}'. Valid levels: {', '.join(sorted(EFFORT_LEVELS))}"
                )

            valkey = get_valkey_client()
            key = f"effort_override:{self.user_id}"
            valkey.setex(key, 3600, effort_value)

            return {
                "success": True,
                "effort": effort_value,
                "ttl_seconds": 3600,
                "message": f"Effort override set to '{effort_value}' (expires in 1 hour)"
            }

        elif action == "get_effort_override":
            valkey = get_valkey_client()
            key = f"effort_override:{self.user_id}"
            current = valkey.get(key)
            ttl = valkey.ttl(key) if current is not None else None

            if current is None:
                return {
                    "active": False,
                    "effort": None,
                    "ttl_seconds": None,
                    "message": "No effort override active"
                }

            return {
                "active": True,
                "effort": current.decode() if isinstance(current, bytes) else current,
                "ttl_seconds": ttl,
                "message": f"Effort override active: '{current.decode() if isinstance(current, bytes) else current}'"
            }

        elif action == "clear_effort_override":
            valkey = get_valkey_client()
            key = f"effort_override:{self.user_id}"
            deleted = valkey.delete(key)

            return {
                "success": True,
                "cleared": bool(deleted),
                "message": "Effort override cleared"
            }

        else:
            raise ValidationError(f"Unknown action: {action}")


class DomainKnowledgeDomainHandler(BaseDomainHandler):
    """Handler for domaindoc actions with SQLite-based section storage."""

    ACTIONS = {
        "create": {
            "required": ["label", "description"],
            "optional": [],
            "types": {"label": str, "description": str}
        },
        "enable": {
            "required": ["label"],
            "optional": [],
            "types": {"label": str}
        },
        "disable": {
            "required": ["label"],
            "optional": [],
            "types": {"label": str}
        },
        "delete": {
            "required": ["label"],
            "optional": [],
            "types": {"label": str}
        },
        "archive": {
            "required": ["label"],
            "optional": [],
            "types": {"label": str}
        },
        "unarchive": {
            "required": ["label"],
            "optional": [],
            "types": {"label": str}
        },
        "list": {
            "required": [],
            "optional": ["archived"],
            "types": {"archived": bool}
        },
        "get": {
            "required": ["label"],
            "optional": [],
            "types": {"label": str}
        },
        "modify_metadata": {
            "required": ["label"],
            "optional": ["new_label", "description"],
            "types": {"label": str, "new_label": str, "description": str}
        },
        "list_sections": {
            "required": ["label"],
            "optional": ["parent"],
            "types": {"label": str, "parent": str}
        },
        "get_section": {
            "required": ["label", "section"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "parent": str}
        },
        "update_section": {
            "required": ["label", "section", "content"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "content": str, "parent": str}
        },
        "create_section": {
            "required": ["label", "section", "content"],
            "optional": ["after", "parent"],
            "types": {"label": str, "section": str, "content": str, "after": str, "parent": str}
        },
        "rename_section": {
            "required": ["label", "section", "new_name"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "new_name": str, "parent": str}
        },
        "delete_section": {
            "required": ["label", "section"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "parent": str}
        },
        "reorder_sections": {
            "required": ["label", "order"],
            "optional": ["parent"],
            "types": {"label": str, "order": list, "parent": str}
        },
        "expand_section": {
            "required": ["label", "section"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "parent": str}
        },
        "collapse_section": {
            "required": ["label", "section"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "parent": str}
        },
        "get_section_history": {
            "required": ["label", "section"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "parent": str}
        },
        "rollback_section": {
            "required": ["label", "section", "version_num"],
            "optional": ["parent"],
            "types": {"label": str, "section": str, "version_num": int, "parent": str}
        },
        "share": {
            "required": ["label", "email"],
            "optional": [],
            "types": {"label": str, "email": str}
        },
        "unshare": {
            "required": ["label", "email"],
            "optional": [],
            "types": {"label": str, "email": str}
        },
        "list_shares": {
            "required": ["label"],
            "optional": [],
            "types": {"label": str}
        },
        "accept_share": {
            "required": ["share_id"],
            "optional": [],
            "types": {"share_id": str}
        },
        "reject_share": {
            "required": ["share_id"],
            "optional": [],
            "types": {"share_id": str}
        },
        "list_pending_shares": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "list_shared_with_me": {
            "required": [],
            "optional": [],
            "types": {}
        }
    }

    def _get_db(self):
        """Get UserDataManager for current user."""
        from utils.userdata_manager import get_user_data_manager
        return get_user_data_manager(self.user_id)

    def _get_pg(self):
        from clients.postgres_client import PostgresClient
        return PostgresClient("mira_service", user_id=str(self.user_id))

    def _validate_label(self, label: str) -> None:
        """Validate domaindoc label."""
        if not label:
            raise ValidationError("Label cannot be empty")
        if not label.replace("_", "").isalnum():
            raise ValidationError(f"Invalid label '{label}'. Use only letters, numbers, and underscores.")
        from utils.domaindoc_shares import SHARED_SUFFIX
        if label.endswith(SHARED_SUFFIX):
            raise ValidationError(f"Labels cannot end with '{SHARED_SUFFIX}' — this suffix is reserved for shared documents")

    def _get_domaindoc(self, db: UserDataManager, label: str) -> dict[str, Any]:
        """Get domaindoc by label, raising ValidationError if not found."""
        results = db.select("domaindocs", "label = :label", {"label": label})
        if not results:
            raise ValidationError(f"Domaindoc '{label}' not found")
        return results[0]

    def _resolve_section_by_header(self, db: UserDataManager, domaindoc_id: int, header: str) -> dict[str, Any]:
        """Resolve a section by header at any nesting depth.

        Parent targeting (parent="X") must find X whether X is a top-level
        section, a subsection, or a sub-subsection — resolving only top-level
        parents made depth-2 sections impossible to create or address. When a
        header matches sections at several depths the shallowest wins (the
        match the old top-level-only lookup would have returned); a header
        matching several sections at the same depth is ambiguous and rejected
        rather than silently targeting one of them.
        """
        rows = db.fetchall(
            "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND header = :header ORDER BY id",
            {"doc_id": domaindoc_id, "header": header}
        )
        if not rows:
            raise ValidationError(f"Section '{header}' not found")
        if len(rows) > 1:
            with_depth = [(self._section_depth(db, row), row) for row in rows]
            min_depth = min(depth for depth, _ in with_depth)
            shallowest = [row for depth, row in with_depth if depth == min_depth]
            if len(shallowest) > 1:
                raise ValidationError(
                    f"Section header '{header}' is ambiguous — it names sections under multiple "
                    "parents at the same level. A parent target requires a unique header."
                )
            return db._decrypt_dict(shallowest[0])
        return db._decrypt_dict(rows[0])

    def _section_depth(self, db: UserDataManager, section: dict[str, Any]) -> int:
        """Nesting depth of a section: 0 = top-level, 1 = subsection, 2 = sub-subsection."""
        depth = 0
        parent_id = section.get("parent_section_id")
        seen = {section["id"]}
        while parent_id is not None:
            if parent_id in seen:
                raise ValidationError("Corrupt domaindoc section tree: parent cycle detected")
            seen.add(parent_id)
            parent = db.fetchone(
                "SELECT id, parent_section_id FROM domaindoc_sections WHERE id = :id",
                {"id": parent_id}
            )
            if not parent:
                raise ValidationError("Corrupt domaindoc section tree: parent section missing")
            depth += 1
            parent_id = parent.get("parent_section_id")
        return depth

    def _get_section(self, db: UserDataManager, domaindoc_id: int, header: str, parent_header: str | None = None) -> dict[str, Any]:
        """Get section by header, optionally under a parent.

        The parent resolves at any nesting depth (a subsection can itself be
        targeted as a parent), so sub-subsections are addressable.
        """
        if parent_header:
            parent = self._resolve_section_by_header(db, domaindoc_id, parent_header)
            results = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND header = :header AND parent_section_id = :parent_id",
                {"doc_id": domaindoc_id, "header": header, "parent_id": parent["id"]}
            )
            if not results:
                raise ValidationError(f"Subsection '{header}' not found under '{parent_header}'")
        else:
            results = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND header = :header AND parent_section_id IS NULL",
                {"doc_id": domaindoc_id, "header": header}
            )
            if not results:
                raise ValidationError(f"Section '{header}' not found")
        return db._decrypt_dict(results[0])

    def _get_all_sections(self, db: UserDataManager, domaindoc_id: int, parent_id: int | None = None) -> list[dict[str, Any]]:
        """Get sections ordered by sort_order, optionally filtered by parent."""
        if parent_id is not None:
            results = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND parent_section_id = :parent_id ORDER BY sort_order",
                {"doc_id": domaindoc_id, "parent_id": parent_id}
            )
        else:
            results = db.fetchall(
                "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id AND parent_section_id IS NULL ORDER BY sort_order",
                {"doc_id": domaindoc_id}
            )
        return [db._decrypt_dict(row) for row in results]

    def _get_subsections(self, db: UserDataManager, parent_id: int) -> list[dict[str, Any]]:
        """Get all subsections of a parent section."""
        results = db.fetchall(
            "SELECT * FROM domaindoc_sections WHERE parent_section_id = :parent_id ORDER BY sort_order",
            {"parent_id": parent_id}
        )
        return [db._decrypt_dict(row) for row in results]

    def _check_duplicate_section(self, db: UserDataManager, domaindoc_id: int, header: str, parent_section_id: int | None, exclude_section_id: int | None = None) -> None:
        """Raise ValidationError if a section with this header already exists at the same level."""
        query = """
            SELECT id FROM domaindoc_sections
            WHERE domaindoc_id = :doc_id
            AND header = :header
            AND parent_section_id IS NOT DISTINCT FROM :parent_id
        """
        params = {"doc_id": domaindoc_id, "header": header, "parent_id": parent_section_id}

        if exclude_section_id:
            query += " AND id != :exclude_id"
            params["exclude_id"] = exclude_section_id

        existing = db.fetchone(query, params)
        if existing:
            level = "subsection" if parent_section_id else "section"
            raise ValidationError(f"A {level} named '{header}' already exists at this level")

    def _record_version(self, db: UserDataManager, domaindoc_id: int, operation: str, diff_data: dict[str, str | int | None], section_id: int) -> int:
        """Record a version entry for a section operation.

        Version numbers are contiguous per section: get_section_history and
        rollback address one section's versions, so numbering restarts at 1 for
        each section rather than running per-document. Every payload carries
        the section's content as of this version ("content") so any version is
        a restorable rollback target.
        """
        import json
        result = db.fetchone(
            "SELECT MAX(version_num) as max_ver FROM domaindoc_versions WHERE domaindoc_id = :doc_id AND section_id = :section_id",
            {"doc_id": domaindoc_id, "section_id": section_id}
        )
        version_num = (result.get("max_ver") or 0) + 1
        now = format_utc_iso(utc_now())

        db.insert("domaindoc_versions", {
            "domaindoc_id": domaindoc_id,
            "section_id": section_id,
            "version_num": version_num,
            "operation": operation,
            "encrypted__diff_data": json.dumps(diff_data),
            "created_at": now
        })
        return version_num

    def _invalidate_trinket_cache(self) -> None:
        """Invalidate domaindoc trinket cache after state changes."""
        user_id = get_current_user_id()
        hash_key = f"{TRINKET_KEY_PREFIX}:{user_id}"
        valkey = get_valkey_client()
        valkey.hdel_with_retry(hash_key, "domaindoc")

    # Read-only actions that don't require cache invalidation
    _READ_ONLY_ACTIONS = {"list", "get", "list_sections", "get_section", "get_section_history", "list_shares", "list_pending_shares", "list_shared_with_me"}

    def execute_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Execute domaindoc actions using SQLite storage."""
        db = self._get_db()

        # Actions that operate on the owner's db for shared docs
        _shared_edit_actions = {
            "update_section", "create_section", "rename_section", "delete_section",
            "reorder_sections", "expand_section", "collapse_section",
            "get_section", "list_sections", "get_section_history", "rollback_section"
        }

        # Resolve once, reuse for both db routing and post-action cache invalidation
        resolved = None
        if action in _shared_edit_actions and data.get("label"):
            from utils.domaindoc_shares import (
                is_shared_label,
                owner_label_from_shared,
                resolve_domaindoc,
            )
            label = data["label"]
            try:
                resolved = resolve_domaindoc(self.user_id, label)
                if resolved.is_shared:
                    db = resolved.db
                    # Downstream methods use data["label"] for DB lookups —
                    # must be the owner's original label, not the suffixed one
                    data["label"] = owner_label_from_shared(label)
            except (ValidationError, ValueError) as e:
                # The resolver is the routing authority for `_shared` labels:
                # when it cannot resolve one, the share is missing, revoked,
                # or unavailable — a not-found, never a label-syntax problem.
                # Left to fall through, _validate_label's reserved-suffix
                # rejection would misattribute the denial to label syntax.
                # Malformed labels (empty / illegal characters) still fall
                # through to the action's own validation, which reports the
                # syntax error exactly as before.
                if is_shared_label(label) and label and label.replace("_", "").isalnum():
                    raise _not_found_envelope(
                        ValueError("Shared document not found or access revoked"),
                        label=label,
                    ) from e

        if action == "create":
            result = self._action_create(db, data)
        elif action == "enable":
            result = self._action_enable(db, data)
        elif action == "disable":
            result = self._action_disable(db, data)
        elif action == "delete":
            result = self._action_delete(db, data)
        elif action == "archive":
            result = self._action_archive(db, data)
        elif action == "unarchive":
            result = self._action_unarchive(db, data)
        elif action == "list":
            result = self._action_list(db, data)
        elif action == "get":
            result = self._action_get(db, data)
        elif action == "modify_metadata":
            result = self._action_modify_metadata(db, data)
        elif action == "list_sections":
            result = self._action_list_sections(db, data)
        elif action == "get_section":
            result = self._action_get_section(db, data)
        elif action == "update_section":
            result = self._action_update_section(db, data)
        elif action == "create_section":
            result = self._action_create_section(db, data)
        elif action == "rename_section":
            result = self._action_rename_section(db, data)
        elif action == "delete_section":
            result = self._action_delete_section(db, data)
        elif action == "reorder_sections":
            result = self._action_reorder_sections(db, data)
        elif action == "expand_section":
            result = self._action_expand_section(db, data)
        elif action == "collapse_section":
            result = self._action_collapse_section(db, data)
        elif action == "get_section_history":
            result = self._action_get_section_history(db, data)
        elif action == "rollback_section":
            result = self._action_rollback_section(db, data)
        elif action == "share":
            result = self._action_share(data)
        elif action == "unshare":
            result = self._action_unshare(data)
        elif action == "list_shares":
            result = self._action_list_shares(data)
        elif action == "accept_share":
            result = self._action_accept_share(data)
        elif action == "reject_share":
            result = self._action_reject_share(data)
        elif action == "list_pending_shares":
            result = self._action_list_pending_shares(data)
        elif action == "list_shared_with_me":
            result = self._action_list_shared_with_me(data)
        else:
            raise ValidationError(f"Unknown action: {action}")

        # Invalidate trinket cache after write operations
        if action not in self._READ_ONLY_ACTIONS:
            self._invalidate_trinket_cache()
            if resolved and resolved.is_shared and resolved.owner_user_id:
                from utils.domaindoc_shares import invalidate_domaindoc_cache
                invalidate_domaindoc_cache(resolved.owner_user_id)

        return result

    def _action_create(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new domaindoc."""
        label = data["label"]
        description = data["description"]

        self._validate_label(label)
        if not description or len(description) > 1000:
            raise ValidationError("Description must be 1-1000 characters")

        existing = db.select("domaindocs", "label = :label", {"label": label})
        if existing:
            raise ValidationError(f"Domaindoc '{label}' already exists")

        expanded_description = self._expand_description(label, description)
        now = format_utc_iso(utc_now())

        domaindoc_id = db.insert("domaindocs", {
            "label": label,
            "encrypted__description": expanded_description,
            "enabled": False,
            "created_at": now,
            "updated_at": now
        })

        section_id = db.insert("domaindoc_sections", {
            "domaindoc_id": int(domaindoc_id),
            "header": "OVERVIEW",
            "encrypted__content": "",
            "sort_order": 0,
            "collapsed": False,
            "created_at": now,
            "updated_at": now
        })

        # Every section's history starts at its birth: version 1 captures the
        # initial section's as-created content.
        self._record_version(db, int(domaindoc_id), "create", {
            "section": "OVERVIEW",
            "content": "",
            "parent": None
        }, int(section_id))

        return {
            "created": True,
            "label": label,
            "description": expanded_description,
            "message": f"Domaindoc '{label}' created. Use enable action to activate it."
        }

    def _action_enable(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Enable a domaindoc. Collapses all unpinned sections."""
        label = data["label"]
        from utils.domaindoc_shares import is_shared_label
        if is_shared_label(label):
            raise ValidationError("Cannot enable a shared domaindoc — only the owner can manage document lifecycle")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        if doc.get("archived", False):
            raise ValidationError(f"Cannot enable archived domaindoc '{label}'. Unarchive it first.")

        now = format_utc_iso(utc_now())

        db.execute(
            "UPDATE domaindocs SET enabled = TRUE, updated_at = :now WHERE id = :id",
            {"now": now, "id": doc["id"]}
        )
        db.execute(
            "UPDATE domaindoc_sections SET collapsed = TRUE, updated_at = :now WHERE domaindoc_id = :doc_id AND pinned = FALSE",
            {"now": now, "doc_id": doc["id"]}
        )

        return {"enabled": True, "label": label, "message": f"Domaindoc '{label}' enabled"}

    def _action_disable(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Disable a domaindoc."""
        label = data["label"]
        from utils.domaindoc_shares import is_shared_label
        if is_shared_label(label):
            raise ValidationError("Cannot disable a shared domaindoc — only the owner can manage document lifecycle")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        now = format_utc_iso(utc_now())

        db.execute(
            "UPDATE domaindocs SET enabled = FALSE, updated_at = :now WHERE id = :id",
            {"now": now, "id": doc["id"]}
        )

        return {"disabled": True, "label": label, "message": f"Domaindoc '{label}' disabled"}

    def _action_delete(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Delete a domaindoc and all its sections."""
        label = data["label"]
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        # Shares are keyed on the mutable label with no FK bridging the
        # SQLite↔Postgres boundary, so accepted shares must be revoked before
        # deleting the doc — otherwise delete/recreate of the same label
        # silently re-grants prior collaborators access to the new document.
        from utils.domaindoc_shares import invalidate_domaindoc_cache
        pg = self._get_pg()
        collaborators = pg.execute_query(
            "SELECT collaborator_user_id FROM domaindoc_shares "
            "WHERE owner_user_id = %(uid)s AND domaindoc_label = %(label)s AND status = 'accepted'",
            {"uid": self.user_id, "label": label}
        )
        if collaborators:
            pg.execute_update(
                "UPDATE domaindoc_shares SET status = 'revoked' "
                "WHERE owner_user_id = %(uid)s AND domaindoc_label = %(label)s AND status = 'accepted'",
                {"uid": self.user_id, "label": label}
            )
            for row in collaborators:
                invalidate_domaindoc_cache(row["collaborator_user_id"])

        db.execute("DELETE FROM domaindocs WHERE id = :id", {"id": doc["id"]})

        return {"deleted": True, "label": label, "message": f"Domaindoc '{label}' deleted"}

    def _action_archive(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Archive a domaindoc. Sets archived=TRUE and enabled=FALSE atomically."""
        label = data["label"]
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        if doc.get("archived", False):
            return {"archived": True, "label": label, "message": f"Domaindoc '{label}' is already archived"}

        now = format_utc_iso(utc_now())
        db.execute(
            "UPDATE domaindocs SET archived = TRUE, enabled = FALSE, updated_at = :now WHERE id = :id",
            {"now": now, "id": doc["id"]}
        )

        return {"archived": True, "label": label, "message": f"Domaindoc '{label}' archived"}

    def _action_unarchive(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Unarchive a domaindoc. Sets archived=FALSE only (does NOT re-enable)."""
        label = data["label"]
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        if not doc.get("archived", False):
            return {"archived": False, "label": label, "message": f"Domaindoc '{label}' is not archived"}

        now = format_utc_iso(utc_now())
        db.execute(
            "UPDATE domaindocs SET archived = FALSE, updated_at = :now WHERE id = :id",
            {"now": now, "id": doc["id"]}
        )

        return {"archived": False, "label": label, "message": f"Domaindoc '{label}' unarchived (still disabled — enable separately)"}

    def _action_list(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """List domaindocs. Excludes archived by default; pass archived=True to see archived only."""
        show_archived = data.get("archived", False)

        if show_archived:
            results = db.fetchall("SELECT * FROM domaindocs WHERE archived = TRUE ORDER BY label")
        else:
            results = db.fetchall("SELECT * FROM domaindocs WHERE archived = FALSE ORDER BY label")

        domaindocs = [db._decrypt_dict(row) for row in results]

        return {
            "domaindocs": [
                {
                    "label": d["label"],
                    "description": d.get("encrypted__description", ""),
                    "enabled": d.get("enabled", False),
                    "archived": d.get("archived", False),
                    "created_at": d.get("created_at"),
                    "updated_at": d.get("updated_at")
                }
                for d in domaindocs
            ],
            "count": len(domaindocs),
            "message": f"Found {len(domaindocs)} domaindocs"
        }

    def _action_get(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Get a domaindoc with all its sections (including subsections)."""
        label = data["label"]
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        # Get ALL sections for this domaindoc (both top-level and subsections)
        all_results = db.fetchall(
            "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id ORDER BY parent_section_id NULLS FIRST, sort_order",
            {"doc_id": doc["id"]}
        )
        all_sections = [db._decrypt_dict(row) for row in all_results]

        return {
            "label": label,
            "description": doc.get("encrypted__description", ""),
            "enabled": doc.get("enabled", False),
            "archived": doc.get("archived", False),
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
            "sections": [
                {
                    "id": s["id"],
                    "header": s["header"],
                    "content": s.get("encrypted__content", ""),
                    "collapsed": s.get("collapsed", False),
                    "pinned": s.get("pinned", False),
                    "sort_order": s.get("sort_order", 0),
                    "parent_section_id": s.get("parent_section_id"),
                    "updated_at": s.get("updated_at"),
                    "created_at": s.get("created_at")
                }
                for s in all_sections
            ],
            "message": f"Retrieved domaindoc '{label}'"
        }

    def _action_modify_metadata(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Modify domaindoc label or description."""
        label = data["label"]
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        new_label = data.get("new_label")
        new_description = data.get("description")

        if not new_label and not new_description:
            raise ValidationError("At least one of 'new_label' or 'description' must be provided")

        updates = {"updated_at": format_utc_iso(utc_now())}

        if new_label:
            self._validate_label(new_label)
            # Check for collision
            existing = db.select("domaindocs", "label = :label AND id != :id", {"label": new_label, "id": doc["id"]})
            if existing:
                raise ValidationError(f"Domaindoc '{new_label}' already exists")
            updates["label"] = new_label

        if new_description:
            if len(new_description) > 2000:
                raise ValidationError("Description must be under 2000 characters")
            updates["encrypted__description"] = new_description

        db.update("domaindocs", updates, "id = :id", {"id": doc["id"]})

        return {
            "updated": True,
            "label": new_label or label,
            "description": new_description or doc.get("encrypted__description"),
            "message": "Domaindoc metadata updated"
        }

    def _action_list_sections(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """List sections for a domaindoc, optionally under a parent."""
        label = data["label"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        if parent_header:
            parent = self._resolve_section_by_header(db, doc["id"], parent_header)
            sections = self._get_all_sections(db, doc["id"], parent_id=parent["id"])
        else:
            sections = self._get_all_sections(db, doc["id"])

        return {
            "label": label,
            "parent": parent_header,
            "sections": [
                {
                    "header": s["header"],
                    "summary": s.get("encrypted__summary"),
                    "collapsed": s.get("collapsed", False),
                    "pinned": s.get("pinned", False),
                    "sort_order": s.get("sort_order", 0),
                    "char_count": len(s.get("encrypted__content", "")),
                    "has_children": len(self._get_subsections(db, s["id"])) > 0,
                    "child_count": len(self._get_subsections(db, s["id"]))
                }
                for s in sections
            ]
        }

    def _action_get_section(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Get a single section's content."""
        label = data["label"]
        header = data["section"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)
        subsections = self._get_subsections(db, section["id"])

        return {
            "label": label,
            "section": header,
            "parent": parent_header,
            "content": section.get("encrypted__content", ""),
            "summary": section.get("encrypted__summary"),
            "collapsed": section.get("collapsed", False),
            "pinned": section.get("pinned", False),
            "sort_order": section.get("sort_order", 0),
            "has_children": len(subsections) > 0,
            "child_count": len(subsections)
        }

    def _action_update_section(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Update a section's content (full replacement via UI)."""
        label = data["label"]
        header = data["section"]
        content = data["content"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)
        now = format_utc_iso(utc_now())

        db.update(
            "domaindoc_sections",
            {"encrypted__content": content, "updated_at": now},
            "id = :id",
            {"id": section["id"]}
        )

        # Record version for UI edits — "content" is the section content as of
        # this version, the restore target for rollback_section.
        self._record_version(db, doc["id"], "ui_replace", {
            "section": header,
            "content": content,
            "parent": parent_header
        }, section["id"])

        # Update section summary
        from cns.services.domaindoc_summary_service import update_section_summary
        update_section_summary(db, section["id"], header, content)

        return {"updated": True, "label": label, "section": header, "parent": parent_header, "char_count": len(content)}

    def _action_create_section(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new section or subsection."""
        label = data["label"]
        header = data["section"]
        content = data["content"]
        after = data.get("after")
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        now = format_utc_iso(utc_now())

        parent_id = None
        if parent_header:
            # Creating a nested section — resolve the parent at any depth and
            # enforce the nesting bound: section (depth 0) → subsection (depth
            # 1) → sub-subsection (depth 2); children of a sub-subsection
            # (depth 3) are rejected. The trinket renders TAG_NAMES[2] but not
            # depth 3.
            parent = self._resolve_section_by_header(db, doc["id"], parent_header)
            if self._section_depth(db, parent) >= 2:
                raise ValidationError("Maximum nesting depth is 2. Cannot add children to a sub-subsection.")
            parent_id = parent["id"]

        # Get sections at the target level (top-level or within parent)
        sections = self._get_all_sections(db, doc["id"], parent_id)

        # Check for duplicate section name at this level
        self._check_duplicate_section(db, doc["id"], header, parent_id)

        if after:
            after_sec = self._get_section(db, doc["id"], after, parent_header)
            new_order = after_sec["sort_order"] + 1
            for sec in sections:
                if sec["sort_order"] >= new_order:
                    db.execute(
                        "UPDATE domaindoc_sections SET sort_order = sort_order + 1 WHERE id = :id",
                        {"id": sec["id"]}
                    )
        else:
            new_order = max((s["sort_order"] for s in sections), default=-1) + 1

        section_id = db.insert("domaindoc_sections", {
            "domaindoc_id": doc["id"],
            "parent_section_id": parent_id,
            "header": header,
            "encrypted__content": content,
            "sort_order": new_order,
            "collapsed": True,  # New sections start collapsed
            "created_at": now,
            "updated_at": now
        })

        # Record version — "content" is the section content as of this version
        self._record_version(db, doc["id"], "ui_create_section", {
            "header": header,
            "content": content,
            "after": after,
            "sort_order": new_order,
            "parent": parent_header
        }, int(section_id))

        # Generate section summary (if content is provided)
        if content:
            from cns.services.domaindoc_summary_service import update_section_summary
            update_section_summary(db, int(section_id), header, content)

        return {"created": True, "label": label, "section": header, "parent": parent_header, "sort_order": new_order}

    def _action_rename_section(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Rename a section or subsection header."""
        label = data["label"]
        header = data["section"]
        new_name = data["new_name"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)

        # Check for duplicate (exclude current section to allow case changes)
        parent_id = section.get("parent_section_id")
        self._check_duplicate_section(db, doc["id"], new_name, parent_id, exclude_section_id=section["id"])

        now = format_utc_iso(utc_now())

        db.execute(
            "UPDATE domaindoc_sections SET header = :new_name, updated_at = :now WHERE id = :id",
            {"new_name": new_name, "now": now, "id": section["id"]}
        )

        # Record version — content is unchanged by a rename; record the as-of
        # content so the rename version is a restorable rollback target.
        self._record_version(db, doc["id"], "ui_rename_section", {
            "old_name": header,
            "new_name": new_name,
            "content": section.get("encrypted__content", ""),
            "parent": parent_header
        }, section["id"])

        return {"renamed": True, "label": label, "from": header, "to": new_name, "parent": parent_header}

    def _action_delete_section(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Delete a section or subsection."""
        label = data["label"]
        header = data["section"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)

        # Can't delete pinned sections
        if section.get("pinned"):
            raise ValidationError(f"Cannot delete pinned section '{header}'. Unpin it first.")

        # For top-level sections, check if section AND all subsections are expanded
        deleted_children = []
        if section.get("parent_section_id") is None:
            subsections = self._get_subsections(db, section["id"])
            if subsections:
                # Check section itself is expanded
                if section.get("collapsed"):
                    raise ValidationError(f"Please expand section '{header}' before deleting to review its contents")
                # Check all subsections are expanded
                collapsed_subs = [s["header"] for s in subsections if s.get("collapsed")]
                if collapsed_subs:
                    raise ValidationError(
                        f"Please expand all subsections of '{header}' before deleting to confirm you've reviewed their contents: {collapsed_subs}"
                    )
                deleted_children = [s["header"] for s in subsections]

        # Record version BEFORE delete (destructive - store content for undo)
        self._record_version(db, doc["id"], "ui_delete_section", {
            "header": header,
            "deleted_content": section.get("encrypted__content", ""),
            "sort_order": section["sort_order"],
            "parent": parent_header,
            "deleted_children": deleted_children
        }, section["id"])

        # Delete cascades to children via FK ON DELETE CASCADE
        db.execute("DELETE FROM domaindoc_sections WHERE id = :id", {"id": section["id"]})

        # Renumber remaining sections at the same level
        parent_id = section.get("parent_section_id")
        sections = self._get_all_sections(db, doc["id"], parent_id)
        for i, s in enumerate(sections):
            if s["sort_order"] != i:
                db.execute(
                    "UPDATE domaindoc_sections SET sort_order = :order WHERE id = :id",
                    {"order": i, "id": s["id"]}
                )

        result = {"deleted": True, "label": label, "section": header, "parent": parent_header}
        if deleted_children:
            result["deleted_children"] = deleted_children
        return result

    def _action_reorder_sections(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Reorder sections at a level (top-level or within a parent)."""
        label = data["label"]
        order = data["order"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)

        parent_id = None
        if parent_header:
            parent = self._resolve_section_by_header(db, doc["id"], parent_header)
            parent_id = parent["id"]

        sections = self._get_all_sections(db, doc["id"], parent_id)

        if len(order) != len(set(order)):
            raise ValidationError(f"Order must list each section exactly once — found duplicates: {order}")

        existing = {s["header"] for s in sections}
        provided = set(order)

        if existing != provided:
            level_desc = f"subsections of '{parent_header}'" if parent_header else "top-level sections"
            raise ValidationError(f"Order must contain exactly all {level_desc}: {list(existing)}")

        now = format_utc_iso(utc_now())
        for new_order, header in enumerate(order):
            sec = next(s for s in sections if s["header"] == header)
            db.execute(
                "UPDATE domaindoc_sections SET sort_order = :order, updated_at = :now WHERE id = :id",
                {"order": new_order, "now": now, "id": sec["id"]}
            )

        return {"reordered": True, "label": label, "order": order, "parent": parent_header}

    def _action_expand_section(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Expand a section or subsection."""
        label = data["label"]
        header = data["section"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)
        now = format_utc_iso(utc_now())

        db.execute(
            "UPDATE domaindoc_sections SET collapsed = FALSE, updated_at = :now WHERE id = :id",
            {"now": now, "id": section["id"]}
        )

        return {"expanded": True, "label": label, "section": header, "parent": parent_header}

    def _action_collapse_section(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Collapse a section or subsection."""
        label = data["label"]
        header = data["section"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)

        # Can't collapse pinned sections — same guard shape as delete_section
        # (pinned sections are always expanded; a silent no-op would let the
        # caller believe the collapse took effect)
        if section.get("pinned"):
            raise ValidationError(
                f"Cannot collapse pinned section '{header}'. Pinned sections are always expanded — unpin it first."
            )

        now = format_utc_iso(utc_now())
        db.execute(
            "UPDATE domaindoc_sections SET collapsed = TRUE, updated_at = :now WHERE id = :id",
            {"now": now, "id": section["id"]}
        )

        return {"collapsed": True, "label": label, "section": header, "parent": parent_header}

    def _action_get_section_history(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Get version history for a section or subsection."""
        import json
        label = data["label"]
        header = data["section"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)

        results = db.fetchall(
            "SELECT * FROM domaindoc_versions WHERE domaindoc_id = :doc_id AND section_id = :sec_id ORDER BY version_num DESC",
            {"doc_id": doc["id"], "sec_id": section["id"]}
        )
        versions = [db._decrypt_dict(row) for row in results]

        return {
            "label": label,
            "section": header,
            "parent": parent_header,
            "versions": [
                {
                    "version_num": v["version_num"],
                    "operation": v["operation"],
                    "diff_data": json.loads(v.get("encrypted__diff_data", "{}")) if v.get("encrypted__diff_data") else {},
                    "created_at": v.get("created_at")
                }
                for v in versions
            ]
        }

    def _action_rollback_section(self, db: UserDataManager, data: dict[str, Any]) -> dict[str, Any]:
        """Rollback a section or subsection to the content as of a given version.

        Version N's diff_data carries the section content as of version N
        ("content"); the rollback restores that content and appends a new
        version carrying the same content, so a rollback is itself a
        restorable rollback target.
        """
        import json
        label = data["label"]
        header = data["section"]
        version_num = data["version_num"]
        parent_header = data.get("parent")
        self._validate_label(label)
        doc = self._get_domaindoc(db, label)
        section = self._get_section(db, doc["id"], header, parent_header)

        # Find the target version
        result = db.fetchone(
            "SELECT * FROM domaindoc_versions WHERE domaindoc_id = :doc_id AND section_id = :sec_id AND version_num = :ver",
            {"doc_id": doc["id"], "sec_id": section["id"], "ver": version_num}
        )
        if not result:
            raise ValidationError(f"Version {version_num} not found for section '{header}'")

        version = db._decrypt_dict(result)
        diff_data = json.loads(version.get("encrypted__diff_data", "{}"))

        if "content" not in diff_data:
            raise ValidationError(f"Version {version_num} does not contain restorable content")

        # Restore the content as of version N
        restored_content = diff_data["content"]
        now = format_utc_iso(utc_now())

        db.update(
            "domaindoc_sections",
            {"encrypted__content": restored_content, "updated_at": now},
            "id = :id",
            {"id": section["id"]}
        )

        # Record the rollback as a new version carrying the restored content,
        # so it is itself restorable
        new_version = self._record_version(db, doc["id"], "rollback", {
            "rolled_back_to": version_num,
            "content": restored_content,
            "parent": parent_header
        }, section["id"])

        return {
            "rolled_back": True,
            "label": label,
            "section": header,
            "parent": parent_header,
            "to_version": version_num,
            "new_version": new_version,
            "restored_chars": len(restored_content)
        }

    def _action_share(self, data: dict[str, Any]) -> dict[str, Any]:
        """Share a domaindoc with another user by email."""
        label = data["label"]
        email = data["email"].strip().lower()
        self._validate_label(label)

        db = self._get_db()
        # Existence check only — _get_domaindoc raises ValidationError if the doc is missing
        _ = self._get_domaindoc(db, label)

        pg = self._get_pg()
        # RLS on users is unconditional, so reading the table directly by email returns
        # zero rows for anyone but the caller and makes sharing impossible. These two
        # SECURITY DEFINER functions expose exactly id/email/first_name for active
        # users, which is all this lookup needs.
        target_user = pg.execute_single(
            "SELECT id, first_name, email FROM resolve_active_user_identity(%(email)s)",
            {"email": email}
        )
        if not target_user:
            raise ValidationError(f"No active user found with email '{email}'")

        target_user_id = target_user["id"]
        if str(target_user_id) == str(self.user_id):
            raise ValidationError("Cannot share a domaindoc with yourself")

        existing = pg.execute_single(
            "SELECT id, status FROM domaindoc_shares WHERE owner_user_id = %(uid)s AND domaindoc_label = %(label)s AND collaborator_user_id = %(cid)s",
            {"uid": self.user_id, "label": label, "cid": target_user_id}
        )
        if existing:
            if existing["status"] == "accepted":
                raise ValidationError(f"Already sharing '{label}' with {email}")
            elif existing["status"] == "pending":
                raise ValidationError(f"Already invited {email} to '{label}'")
            elif existing["status"] in ("revoked", "rejected"):
                # SET deliberately scoped to the granted columns — schema grants mira_dbuser
                # only UPDATE (status, accepted_at) on domaindoc_shares; writing invited_at
                # would raise 42501 and surface as a spurious 500 on re-invite.
                pg.execute_update(
                    "UPDATE domaindoc_shares SET status = 'pending', accepted_at = NULL WHERE id = %(sid)s",
                    {"sid": existing["id"]}
                )
                return {"shared": True, "label": label, "email": email, "status": "re-invited"}

        pg.execute_insert(
            "INSERT INTO domaindoc_shares (owner_user_id, domaindoc_label, collaborator_user_id, status) VALUES (%(uid)s, %(label)s, %(cid)s, 'pending')",
            {"uid": self.user_id, "label": label, "cid": target_user_id}
        )

        return {"shared": True, "label": label, "email": email, "status": "pending"}

    def _action_unshare(self, data: dict[str, Any]) -> dict[str, Any]:
        """Revoke sharing of a domaindoc with a user."""
        label = data["label"]
        email = data["email"].strip().lower()
        self._validate_label(label)

        pg = self._get_pg()
        target_user = pg.execute_single(
            "SELECT id FROM resolve_active_user_identity(%(email)s)",
            {"email": email}
        )
        if not target_user:
            raise ValidationError(f"No active user found with email '{email}'")

        share = pg.execute_single(
            "SELECT id, status FROM domaindoc_shares WHERE owner_user_id = %(uid)s AND domaindoc_label = %(label)s AND collaborator_user_id = %(cid)s",
            {"uid": self.user_id, "label": label, "cid": target_user["id"]}
        )
        if not share:
            raise ValidationError(f"Not sharing '{label}' with {email}")

        if share["status"] == "accepted":
            pg.execute_update(
                "UPDATE domaindoc_shares SET status = 'revoked' WHERE id = %(sid)s",
                {"sid": share["id"]}
            )
        else:
            # Pending/rejected shares can be fully deleted by the owner
            pg.execute_update(
                "DELETE FROM domaindoc_shares WHERE id = %(sid)s",
                {"sid": share["id"]}
            )

        from utils.domaindoc_shares import invalidate_domaindoc_cache
        invalidate_domaindoc_cache(target_user["id"])

        return {"unshared": True, "label": label, "email": email}

    def _action_list_shares(self, data: dict[str, Any]) -> dict[str, Any]:
        """List all shares for a domaindoc owned by current user."""
        label = data["label"]
        self._validate_label(label)

        pg = self._get_pg()
        shares = pg.execute_query(
            "SELECT ds.id, ds.domaindoc_label, ds.status, ds.invited_at, ds.accepted_at, "
            "ai.email, ai.first_name FROM domaindoc_shares ds "
            "JOIN LATERAL active_user_identity(ds.collaborator_user_id) ai ON TRUE "
            "WHERE ds.owner_user_id = %(uid)s AND ds.domaindoc_label = %(label)s AND ds.status != 'rejected'",
            {"uid": self.user_id, "label": label}
        )

        return {
            "label": label,
            "shares": [
                {
                    "id": str(s["id"]),
                    "email": s["email"],
                    "first_name": s["first_name"],
                    "status": s["status"],
                    "invited_at": s["invited_at"].isoformat() if s.get("invited_at") else None,
                    "accepted_at": s["accepted_at"].isoformat() if s.get("accepted_at") else None,
                }
                for s in shares
            ]
        }

    def _action_accept_share(self, data: dict[str, Any]) -> dict[str, Any]:
        """Accept a pending share invitation."""
        share_id = data["share_id"]

        pg = self._get_pg()
        share = pg.execute_single(
            "SELECT id, domaindoc_label, status FROM domaindoc_shares WHERE id = %(sid)s AND collaborator_user_id = %(uid)s",
            {"sid": share_id, "uid": self.user_id}
        )
        if not share:
            raise ValidationError("Share invitation not found")
        if share["status"] != "pending":
            raise ValidationError(f"Share is not pending (current status: {share['status']})")

        # Check for label collision: the shared doc will appear as label_shared
        from utils.domaindoc_shares import SHARED_SUFFIX
        suffixed_label = f"{share['domaindoc_label']}{SHARED_SUFFIX}"
        db = self._get_db()
        collision = db.select("domaindocs", "label = :label", {"label": suffixed_label})
        if collision:
            raise ValidationError(
                f"Cannot accept — you already have a domaindoc named '{suffixed_label}'. "
                f"Rename or archive yours first."
            )

        pg.execute_update(
            "UPDATE domaindoc_shares SET status = 'accepted', accepted_at = NOW() WHERE id = %(sid)s",
            {"sid": share_id}
        )

        self._invalidate_trinket_cache()

        return {"accepted": True, "label": suffixed_label}

    def _action_reject_share(self, data: dict[str, Any]) -> dict[str, Any]:
        """Reject a pending share invitation."""
        share_id = data["share_id"]

        pg = self._get_pg()
        share = pg.execute_single(
            "SELECT id, domaindoc_label, status FROM domaindoc_shares WHERE id = %(sid)s AND collaborator_user_id = %(uid)s",
            {"sid": share_id, "uid": self.user_id}
        )
        if not share:
            raise ValidationError("Share invitation not found")
        if share["status"] != "pending":
            raise ValidationError(f"Share is not pending (current status: {share['status']})")

        pg.execute_update(
            "UPDATE domaindoc_shares SET status = 'rejected' WHERE id = %(sid)s",
            {"sid": share_id}
        )

        return {"rejected": True, "label": share["domaindoc_label"]}

    def _action_list_pending_shares(self, data: dict[str, Any]) -> dict[str, Any]:
        """List pending share invitations for current user."""
        pg = self._get_pg()
        shares = pg.execute_query(
            "SELECT ds.id, ds.domaindoc_label, ds.invited_at, "
            "ai.email, ai.first_name FROM domaindoc_shares ds "
            "JOIN LATERAL active_user_identity(ds.owner_user_id) ai ON TRUE "
            "WHERE ds.collaborator_user_id = %(uid)s AND ds.status = 'pending'",
            {"uid": self.user_id}
        )

        return {
            "pending_shares": [
                {
                    "id": str(s["id"]),
                    "label": s["domaindoc_label"],
                    "from_email": s["email"],
                    "from_name": s["first_name"],
                    "invited_at": s["invited_at"].isoformat() if s.get("invited_at") else None,
                }
                for s in shares
            ]
        }

    def _action_list_shared_with_me(self, data: dict[str, Any]) -> dict[str, Any]:
        """List all domaindocs shared with current user (accepted only)."""
        pg = self._get_pg()
        shares = pg.execute_query(
            "SELECT ds.id, ds.domaindoc_label, ds.accepted_at, "
            "ai.email, ai.first_name FROM domaindoc_shares ds "
            "JOIN LATERAL active_user_identity(ds.owner_user_id) ai ON TRUE "
            "WHERE ds.collaborator_user_id = %(uid)s AND ds.status = 'accepted'",
            {"uid": self.user_id}
        )

        return {
            "shared_with_me": [
                {
                    "id": str(s["id"]),
                    "label": s["domaindoc_label"],
                    "from_email": s["email"],
                    "from_name": s["first_name"],
                    "accepted_at": s["accepted_at"].isoformat() if s.get("accepted_at") else None,
                }
                for s in shares
            ]
        }


    def _expand_description(self, label: str, description: str) -> str:
        """Use LLM to expand a brief description into comprehensive guidance."""
        from clients.llm_provider import get_llm_provider

        logger.debug(
            "_expand_description called with label length=%s and description length=%s",
            len(label),
            len(description)
        )

        try:
            llm = get_llm_provider()

            prompt = f"""You are helping expand a brief description into comprehensive guidance for a knowledge document.

The user wants to create a domain knowledge document called "{label}" with this description:
"{description}"

Expand this into a well-formed descriptor (1-3 sentences) that clearly explains:
1. What topics this document covers
2. What specific information should be recorded
3. The level of detail expected

Write ONLY the expanded description, nothing else. Be specific and actionable.
Example input: "plants, bugs, where I buy stuff"
Example output: "Backyard garden management: current plantings with locations and planting dates, pest and disease observations with treatments applied, soil amendments and fertilizer schedules, preferred suppliers and product recommendations, and seasonal lessons learned."
"""

            logger.debug("Calling LLM generate_response...")
            response = llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model_config="fast",
            )
            logger.debug(f"LLM response received: stop_reason={response.stop_reason}")

            expanded = llm.extract_text_content(response).strip()
            logger.debug("Extracted text length=%s", len(expanded))

            if expanded and len(expanded) > 20:
                logger.info(f"LLM expansion successful, length={len(expanded)}")
                return expanded
            else:
                logger.warning("LLM expansion too short (len=%s), using original description", len(expanded))
                return description

        except Exception:
            logger.warning("Failed to expand description via LLM", exc_info=True)
            return description


class ContinuumDomainHandler(BaseDomainHandler):
    """Handler for continuum segment lifecycle actions."""

    ACTIONS = {
        "collapse_segment": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "pause_session": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "resume_session": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "get_segment_status": {
            "required": [],
            "optional": [],
            "types": {}
        }
    }

    def _latest_boundary_is_collapsing(self, continuum_repo, continuum_id) -> bool:
        """Whether the newest segment boundary sentinel is mid-collapse-claim.

        'collapsing' is a transient mutual-exclusion state set by the collapse
        handler's claim; it is invisible to find_active_segment, so callers
        that found no active sentinel use this to distinguish "collapse in
        progress" from "no segment exists".
        """
        db = continuum_repo.get_user_db_client(self.user_id)
        rows = db.execute_query("""
            SELECT metadata->>'status' AS status
            FROM messages
            WHERE continuum_id = %s
                AND metadata->>'is_segment_boundary' = 'true'
            ORDER BY created_at DESC
            LIMIT 1
        """, (str(continuum_id),))
        return bool(rows) and rows[0].get('status') == 'collapsing'

    def execute_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Execute segment lifecycle actions."""
        if action == "collapse_segment":
            from cns.infrastructure.continuum_pool import get_continuum_pool
            from cns.infrastructure.continuum_repository import get_continuum_repository
            from cns.core.events import SegmentTimeoutEvent
            from cns.services.segment_collapse_handler import get_segment_collapse_handler

            # Get user's continuum and active segment
            continuum_pool = get_continuum_pool()
            continuum = continuum_pool.get_or_create()

            continuum_repo = get_continuum_repository()
            sentinel = continuum_repo.find_active_segment(continuum.id, self.user_id)

            if not sentinel:
                # A segment mid-claim ('collapsing') is invisible to
                # find_active_segment; distinguish it from a genuinely absent
                # segment so the user gets a truthful error, not "not found".
                if self._latest_boundary_is_collapsing(continuum_repo, continuum.id):
                    raise ValidationError(
                        "A segment collapse is already in progress; try again in a moment."
                    )
                raise NotFoundError("segment", "active")

            segment_id = sentinel.metadata.get("segment_id")

            # A live turn holds the per-user request lock (module-level in
            # chat.py, renewed in the background for the turn's lifetime — the WS
            # transport probes the same Valkey key). Collapsing under it orphans
            # the in-flight turn from the digest: turn messages commit only at
            # turn end, and the collapse recheck's last_turn_at is stamped at
            # message arrival, so the interleaving turn is invisible to it.
            # Defer instead — queue behind the lock and let the turn's completion
            # path run the collapse (SegmentCollapseHandler._handle_turn_completed).
            from cns.api.chat import _user_request_lock
            if _user_request_lock.is_locked(self.user_id):
                get_segment_collapse_handler().queue_manual_collapse(
                    self.user_id, str(continuum.id), segment_id
                )
                return {
                    "collapsed": False,
                    "segment_id": segment_id,
                    "message": "Your conversation is still being processed — "
                               "the segment will collapse as soon as the current response finishes."
                }

            # Create event and invoke the collapse handler
            event = SegmentTimeoutEvent.create(
                continuum_id=str(continuum.id),
                user_id=self.user_id,
                segment_id=segment_id,
                inactive_duration_minutes=0,  # Manual trigger
                local_hour=utc_now().hour
            )

            handler = get_segment_collapse_handler()
            try:
                collapsed_sentinel = handler.collapse_segment(event)
            except RuntimeError as e:
                if "no committed messages" in str(e):
                    return {
                        "collapsed": False,
                        "segment_id": segment_id,
                        "message": "Your conversation is still being processed — "
                                   "it will wrap up automatically once the response finishes."
                    }
                raise

            return {
                "collapsed": True,
                "segment_id": segment_id,
                "summary": collapsed_sentinel.content,
                "title": collapsed_sentinel.metadata.get("display_title"),
                "message": "Segment collapsed successfully"
            }

        elif action == "pause_session":
            from cns.infrastructure.continuum_pool import get_continuum_pool
            from cns.infrastructure.continuum_repository import get_continuum_repository

            continuum_pool = get_continuum_pool()
            continuum = continuum_pool.get_or_create()
            continuum_repo = get_continuum_repository()

            # find_active_segment returns active OR paused — check current state
            sentinel = continuum_repo.find_active_segment(continuum.id, self.user_id)
            if not sentinel:
                # Same truthful-error rule as collapse_segment: a mid-claim
                # segment cannot be paused, but it exists.
                if self._latest_boundary_is_collapsing(continuum_repo, continuum.id):
                    raise ValidationError(
                        "A segment collapse is already in progress; try again in a moment."
                    )
                raise NotFoundError("segment", "active")

            if sentinel.metadata.get("status") == "paused":
                raise ValidationError("Session is already paused")

            segment_id = sentinel.metadata.get("segment_id")

            success = continuum_repo.pause_segment(continuum.id, self.user_id)
            if not success:
                raise ValidationError("Failed to pause — no active segment found")

            return {
                "paused": True,
                "segment_id": segment_id,
                "message": "Session paused. It will resume automatically when you send your next message."
            }

        elif action == "get_segment_status":
            from cns.infrastructure.continuum_pool import get_continuum_pool
            from cns.infrastructure.continuum_repository import get_continuum_repository
            from datetime import timedelta
            from config import config

            continuum_pool = get_continuum_pool()
            continuum = continuum_pool.get_or_create()

            continuum_repo = get_continuum_repository()
            sentinel = continuum_repo.find_active_segment(continuum.id, self.user_id)

            if not sentinel:
                return {
                    "has_active_segment": False,
                    "segment_id": None,
                    "collapse_at": None,
                    "is_paused": False
                }

            segment_id = sentinel.metadata.get("segment_id")
            status = sentinel.metadata.get("status")
            is_paused = status == "paused"

            # Paused segments have no collapse_at — timeout is suspended
            if is_paused:
                return {
                    "has_active_segment": True,
                    "segment_id": segment_id,
                    "last_activity": sentinel.metadata.get("paused_at"),
                    "collapse_at": None,
                    "timeout_minutes": config.system.segment_timeout,
                    "is_paused": True
                }

            # Active segment — calculate collapse time from last message
            segment_messages = continuum_repo.load_segment_messages(
                continuum.id, self.user_id, sentinel.created_at
            )
            if segment_messages:
                last_activity = segment_messages[-1].created_at
            else:
                last_activity = sentinel.created_at

            timeout_minutes = config.system.segment_timeout
            collapse_at = last_activity + timedelta(minutes=timeout_minutes)

            return {
                "has_active_segment": True,
                "segment_id": segment_id,
                "last_activity": format_utc_iso(last_activity),
                "collapse_at": format_utc_iso(collapse_at),
                "timeout_minutes": timeout_minutes,
                "is_paused": False
            }

        elif action == "resume_session":
            from cns.infrastructure.continuum_pool import get_continuum_pool
            from cns.infrastructure.continuum_repository import get_continuum_repository

            continuum_pool = get_continuum_pool()
            continuum = continuum_pool.get_or_create()
            continuum_repo = get_continuum_repository()

            sentinel = continuum_repo.find_active_segment(continuum.id, self.user_id)
            if not sentinel:
                raise NotFoundError("segment", "active or paused")

            if sentinel.metadata.get("status") != "paused":
                raise ValidationError("Session is already active — nothing to resume")

            segment_id = sentinel.metadata.get("segment_id")
            success = continuum_repo.unpause_segment(continuum.id, self.user_id)
            if not success:
                raise ValidationError("Failed to unpause — segment may have been modified concurrently")

            return {
                "resumed": True,
                "segment_id": segment_id,
                "message": "Session resumed"
            }

        else:
            raise ValidationError(f"Unknown action: {action}")

def _not_found_envelope(e: ValueError, **details: Any) -> APIError:
    """404-class envelope preserving a service's not-found ValueError message.

    The preview services (lora/persona/portrait) and the persona repository
    signal blank, consumed, and expired preview ids — and missing revision
    ids — with plain ValueError carrying actionable text. These are
    missing-resource outcomes, not internal faults: this maps them onto the
    NOT_FOUND code (HTTP 404 via main.py's APIError handler) with the
    service's message intact, so they never fall through to the 500
    catch-all.
    """
    return APIError("NOT_FOUND", str(e), details)


class LoraDomainHandler(BaseDomainHandler):
    """Handler for user model actions.

    Provides a preview-before-save refinement workflow (mirrors portrait):
    the user requests a refinement with free-text instructions, receives a
    proposed user model to review, then accepts or declines it. The refined
    model XML never passes through the client on save — the preview is stored
    server-side in Valkey with a 10-minute TTL and identified by an opaque
    preview_id. Critic validation runs before presenting the preview.
    """

    ACTIONS = {
        "get": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "update": {
            "required": ["xml"],
            "optional": [],
            "types": {
                "xml": str
            }
        },
        "reset": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "refine": {
            "required": ["instructions"],
            "optional": [],
            "types": {"instructions": str},
        },
        "accept": {
            "required": ["preview_id"],
            "optional": [],
            "types": {"preview_id": str},
        },
        "decline": {
            "required": ["preview_id"],
            "optional": [],
            "types": {"preview_id": str},
        },
    }

    def execute_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Execute LoRA actions."""
        from cns.infrastructure.feedback_tracker import FeedbackTracker

        tracker = FeedbackTracker()

        if action == "get":
            lora_content = tracker.get_lora_content(self.user_id)
            tracking_status = tracker.get_tracking_status(self.user_id)

            return {
                "success": True,
                "has_synthesis": lora_content['synthesis_xml'] is not None,
                "xml": lora_content['synthesis_xml'],
                "needs_checkin": lora_content['needs_checkin'],
                "tracking": {
                    "use_days_since_synthesis": tracking_status.get('use_days_since_synthesis', 0),
                    "last_synthesis_at": format_utc_iso(tracking_status['last_synthesis_at']) if tracking_status.get('last_synthesis_at') else None
                }
            }

        elif action == "update":
            xml = data.get("xml")

            if not xml or not xml.strip():
                raise ValidationError("XML content cannot be empty")
            if "<mira:user_model>" not in xml:
                raise ValidationError("Invalid user model XML - must contain <mira:user_model> element")

            # The store path validates anchors just like the refine path:
            # every stored user model must name assessable sections, so a
            # manual edit cannot bypass the check the synthesizer's output
            # already passed. ValueError naming the valid set → 400.
            from cns.services.system_prompt_parser import validate_section_anchors
            try:
                validate_section_anchors(xml, config.system_prompt)
            except ValueError as e:
                raise ValidationError(str(e)) from e

            tracker.set_synthesis_output(self.user_id, xml)
            self._invalidate_lora_cache()

            return {
                "success": True,
                "updated": True,
                "message": "Updated user model"
            }

        elif action == "reset":
            tracker.reset_synthesis(self.user_id)
            self._invalidate_lora_cache()

            return {
                "success": True,
                "reset": True,
                "message": (
                    "User model reset — model content and synthesis tracking "
                    "restored to fresh-install baseline"
                ),
            }

        elif action == "refine":
            from cns.services.lora_service import refine_lora
            instructions = data["instructions"]
            try:
                result = refine_lora(self.user_id, instructions)
            except ValueError as e:
                # Precondition failures (no model to refine, empty
                # instructions, no output, invalid section anchors naming
                # the valid set) are caller-facing 400s; the service's message
                # tells the user what to do. Infrastructure failures are not
                # ValueErrors and keep propagating.
                raise ValidationError(str(e)) from e
            return {
                "success": True,
                "preview_id": result["preview_id"],
                "proposed": result["proposed"],
            }

        elif action == "accept":
            from cns.services.lora_service import accept_lora
            preview_id = data["preview_id"]
            try:
                accept_lora(self.user_id, preview_id)
            except ValueError as e:
                # blank/consumed/expired preview_id is the only ValueError source here; maps to not-found.
                raise _not_found_envelope(e, preview_id=preview_id) from e
            return {
                "success": True,
                "accepted": True,
                "message": "User model updated",
            }

        elif action == "decline":
            from cns.services.lora_service import decline_lora
            preview_id = data["preview_id"]
            try:
                decline_lora(self.user_id, preview_id)
            except ValueError as e:
                # only a blank preview_id raises (consumed/expired no-ops); maps to not-found.
                raise _not_found_envelope(e, preview_id=preview_id) from e
            return {
                "success": True,
                "declined": True,
                "message": "User model preview discarded",
            }

        else:
            raise ValidationError(f"Unknown action: {action}")

    def _invalidate_lora_cache(self) -> None:
        """Invalidate LoRA trinket cache after state changes."""
        hash_key = f"{TRINKET_KEY_PREFIX}:{self.user_id}"
        valkey = get_valkey_client()
        valkey.hdel_with_retry(hash_key, "behavioral_directives")


class PersonaDomainHandler(BaseDomainHandler):
    """Handler for immutable Persona revision workflows.

    Persona evaluates MIRA against the behavioral contract and stores prescriptive
    directives. It is a second system beside the user model, not a replacement for it,
    so it gets its own domain and its own action names: get, refine, accept,
    decline, update and reset keep serving the settings page's user-model panel through
    LoraDomainHandler, and reusing those names here would read as one feature
    duplicated rather than two features that coexist.

    Directives never round-trip through the client on save: `propose` stores the
    validated candidate server-side in Valkey and returns an opaque preview_id, which
    `approve` or `discard` consumes. `rollback` appends a new revision copying a
    historical one, so the history stays append-only and the expected parent on every
    write makes a concurrent edit fail loudly instead of losing one.
    """

    ACTIONS = {
        "current": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "history": {
            "required": [],
            "optional": [],
            "types": {}
        },
        "propose": {
            "required": ["instructions"],
            "optional": [],
            "types": {"instructions": str},
        },
        "approve": {
            "required": ["preview_id"],
            "optional": [],
            "types": {"preview_id": str},
        },
        "discard": {
            "required": ["preview_id"],
            "optional": [],
            "types": {"preview_id": str},
        },
        "rollback": {
            "required": ["revision_id"],
            "optional": [],
            "types": {"revision_id": "uuid"},
        },
    }

    def execute_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        from cns.services.persona_service import PersonaService

        service = PersonaService()

        if action == "current":
            revision = service.get_current(self.user_id)
            return {
                "success": True,
                "revision": self._serialize_revision(revision),
                "has_directives": bool(revision.directives.strip()),
            }

        elif action == "history":
            return {
                "success": True,
                "revisions": [
                    self._serialize_revision(revision)
                    for revision in service.get_history(self.user_id)
                ],
            }

        elif action == "propose":
            try:
                result = service.create_preview(self.user_id, data["instructions"])
            except ValueError as e:
                # Caller-facing failures (empty instructions, validation
                # exhausted) are 400s preserving the service's feedback.
                raise ValidationError(str(e)) from e
            return {
                "success": True,
                "preview_id": result["preview_id"],
                "proposed": result["proposed"],
            }

        elif action == "approve":
            try:
                revision = service.accept_preview(self.user_id, data["preview_id"])
            except ValueError as e:
                # blank/consumed/expired preview_id is the only ValueError source here; maps to not-found.
                raise _not_found_envelope(e, preview_id=data["preview_id"]) from e
            return {
                "success": True,
                "approved": True,
                "revision": self._serialize_revision(revision),
                "message": "Persona updated",
            }

        elif action == "discard":
            try:
                service.decline_preview(self.user_id, data["preview_id"])
            except ValueError as e:
                # only a blank preview_id raises (consumed/expired no-ops); maps to not-found.
                raise _not_found_envelope(e, preview_id=data["preview_id"]) from e
            return {
                "success": True,
                "discarded": True,
                "message": "Persona preview discarded",
            }

        elif action == "rollback":
            revision_id = data["revision_id"]
            try:
                revision = service.rollback(self.user_id, UUID(revision_id))
            except ValueError as e:
                # The repository raises ValueError only for a missing
                # revision id — a 404, not an internal fault.
                raise _not_found_envelope(e, revision_id=str(revision_id)) from e
            return {
                "success": True,
                "revision": self._serialize_revision(revision),
                "message": "Persona rollback revision created",
            }

        else:
            raise ValidationError(f"Unknown action: {action}")

    @staticmethod
    def _serialize_revision(revision) -> dict[str, Any]:
        return {
            "id": str(revision.id),
            "revision_number": revision.revision_number,
            "directives": revision.directives,
            "source": revision.source,
            "parent_revision_id": (
                str(revision.parent_revision_id) if revision.parent_revision_id else None
            ),
            "created_at": format_utc_iso(revision.created_at),
        }


_REPULSION_REWRITER_EXECUTOR = ThreadPoolExecutor(
    max_workers=config.worker_pools.repulsion_rewriter_workers,
    thread_name_prefix="repulsion_rewriter",
)

_REWRITER_SYSTEM_PROMPT_FILE = "repulsion_rewriter_system.txt"
_REWRITER_USER_PROMPT_FILE = "repulsion_rewriter_user.txt"
# D6: the repulsion rewriter runs with effort='high' on the batch route
# (main chat is the only primary consumer; batch keeps rewrites off the
# single local llama-server slot so they cannot evict the chat KV cache).
_REWRITER_MODEL_CONFIG = "batch"


class FeedbackDomainHandler(BaseDomainHandler):
    """Handler for direct user feedback capture."""

    ACTIONS = {
        "capture_repulsion": {
            "required": ["response_text", "preceding_user_message"],
            "optional": ["reason", "matched_tells"],
            "types": {
                "reason": str,
                "response_text": str,
                "preceding_user_message": str,
                "matched_tells": list,
            },
        }
    }

    def execute_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        if action != "capture_repulsion":
            raise ValidationError(f"Unknown action: {action}")

        if not self.user_id:
            raise ValidationError("No user context available for feedback capture")

        import contextvars
        import uuid
        from pathlib import Path

        record_id = str(uuid.uuid4())
        reason = data.get("reason", "").strip()
        response_text = data["response_text"]
        preceding_user_message = data["preceding_user_message"]
        matched_tells = data.get("matched_tells", [])

        click_record = {
            "id": record_id,
            "timestamp": utc_now().isoformat(),
            "kind": "click",
            "reason": reason,
            "response_text": response_text,
            "preceding_user_message": preceding_user_message,
            "matched_tells": matched_tells,
        }

        output_dir = Path("data/users") / str(self.user_id) / "repulsion_feedback"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{utc_now().strftime('%Y-%m-%d')}.jsonl"

        self._append_record(output_file, click_record)

        logger.info(
            f"Captured repulsion feedback for user {self.user_id} "
            f"(id={record_id}, tells={matched_tells}, file={output_file})"
        )

        ctx = contextvars.copy_context()
        _REPULSION_REWRITER_EXECUTOR.submit(
            ctx.run,
            self._run_rewriter,
            record_id,
            response_text,
            preceding_user_message,
            matched_tells,
            output_file,
        )

        return {"captured": True, "id": record_id}

    @staticmethod
    def _append_record(output_file: "Path", record: dict[str, Any]) -> None:
        import json

        file_has_content = output_file.exists() and output_file.stat().st_size > 0
        with open(output_file, "a", encoding="utf-8") as f:
            if file_has_content:
                f.write("\n------------------\n")
            f.write(json.dumps(record))

    @staticmethod
    def _run_rewriter(
        record_id: str,
        response_text: str,
        preceding_user_message: str,
        matched_tells: list[str],
        output_file: "Path",
    ) -> None:
        try:
            from clients.llm_provider import get_llm_provider
            from config.prompts.loader import load_prompt
            from utils.user_context import get_model_config

            system_prompt = load_prompt(_REWRITER_SYSTEM_PROMPT_FILE)
            user_template = load_prompt(_REWRITER_USER_PROMPT_FILE)

            user_prompt = user_template.format(
                user_message=preceding_user_message,
                ai_response=response_text,
                matched_tells=", ".join(matched_tells) if matched_tells else "(none)",
            )

            llm = get_llm_provider()
            response = llm.generate_response(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                model_config=_REWRITER_MODEL_CONFIG,
                effort="high",
            )
            chosen_text = llm.extract_text_content(response).strip()

            if not chosen_text:
                logger.warning(
                    f"Repulsion rewriter returned empty output for record {record_id}"
                )
                return

            rewrite_record = {
                "id": record_id,
                "timestamp": utc_now().isoformat(),
                "kind": "rewrite_v1",
                "rewriter_model": get_model_config(_REWRITER_MODEL_CONFIG).model,
                "chosen": chosen_text,
            }
            FeedbackDomainHandler._append_record(output_file, rewrite_record)

            logger.info(
                f"Repulsion rewriter completed for record {record_id} "
                f"(output_chars={len(chosen_text)})"
            )
        except Exception as e:
            logger.warning(
                f"Repulsion rewriter failed for record {record_id} "
                f"(non-critical): {e}"
            )


class PortraitDomainHandler(BaseDomainHandler):
    """Handler for user portrait refinement actions.
    Provides a preview-before-save workflow: the user requests a refinement
    with free-text instructions, receives a proposed portrait to review, then
    accepts or declines it. The portrait text never passes through the client
    on save — the preview is stored server-side in Valkey with a 10-minute TTL
    and identified by an opaque preview_id.
    """
    ACTIONS = {
        "get": {
            "required": [],
            "optional": [],
            "types": {},
        },
        "refine": {
            "required": ["instructions"],
            "optional": [],
            "types": {"instructions": str},
        },
        "accept": {
            "required": ["preview_id"],
            "optional": [],
            "types": {"preview_id": str},
        },
        "decline": {
            "required": ["preview_id"],
            "optional": [],
            "types": {"preview_id": str},
        },
    }

    def execute_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Execute portrait actions."""
        from cns.services.portrait_service import (
            read_portrait,
            refine_portrait,
            accept_portrait,
            decline_portrait,
        )

        if action == "get":
            portrait = read_portrait(self.user_id)
            return {
                "success": True,
                "portrait": portrait,
                "has_portrait": bool(portrait),
            }

        elif action == "refine":
            instructions = data["instructions"]
            try:
                result = refine_portrait(self.user_id, instructions)
            except ValueError as e:
                # Precondition failures (no portrait to refine, empty
                # instructions, no output) are caller-facing 400s; the
                # service's message tells the user what to do.
                raise ValidationError(str(e)) from e
            return {
                "success": True,
                "preview_id": result["preview_id"],
                "proposed": result["proposed"],
            }

        elif action == "accept":
            preview_id = data["preview_id"]
            try:
                accept_portrait(self.user_id, preview_id)
            except ValueError as e:
                # blank/consumed/expired preview_id is the only ValueError source here; maps to not-found.
                raise _not_found_envelope(e, preview_id=preview_id) from e
            return {
                "success": True,
                "accepted": True,
                "message": "Portrait updated",
            }

        elif action == "decline":
            preview_id = data["preview_id"]
            try:
                decline_portrait(self.user_id, preview_id)
            except ValueError as e:
                # only a blank preview_id raises (consumed/expired no-ops); maps to not-found.
                raise _not_found_envelope(e, preview_id=preview_id) from e
            return {
                "success": True,
                "declined": True,
                "message": "Portrait preview discarded",
            }

        else:
            raise ValidationError(f"Unknown action: {action}")


class SkillsDomainHandler(BaseDomainHandler):
    """Handler for per-user skill management (create/delete).

    Skills are files under the user's skills directory (write/delete owned by
    utils/skill_files.py). Global skills (working_memory/skills/) are
    read-only repo content — no action here may touch them, which the delete
    path enforces with a distinct message. Reads live in the data endpoint
    (type=skills); only mutations live here.
    """

    ACTIONS = {
        "create": {
            "required": ["name", "description", "body"],
            "optional": [],
            "types": {"name": str, "description": str, "body": str}
        },
        "delete": {
            "required": ["name"],
            "optional": [],
            "types": {"name": str}
        }
    }

    def _invalidate_trinket_cache(self) -> None:
        """Drop the stale skills_catalog field so API reads between composes
        don't serve pre-write content (the catalog re-renders every compose
        regardless). Same idiom as DomainKnowledgeDomainHandler."""
        hash_key = f"{TRINKET_KEY_PREFIX}:{get_current_user_id()}"
        valkey = get_valkey_client()
        valkey.hdel_with_retry(hash_key, "skills_catalog")

    def execute_action(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        """Execute skill create/delete through the sanctioned file path."""
        from utils import skill_files

        if action == "create":
            try:
                record = skill_files.write_user_skill(
                    self.user_id, data["name"], data["description"], data["body"]
                )
            except ValueError as e:
                raise ValidationError(str(e))
            self._invalidate_trinket_cache()
            return {
                "created": True,
                "name": record.name,
                "message": f"Skill '{record.name}' created"
            }

        elif action == "delete":
            try:
                skill_files.delete_user_skill(self.user_id, data["name"])
            except skill_files.GlobalSkillReadOnlyError as e:
                # Exists but is repo content — a client misunderstanding, not a
                # missing resource, so 400 with the reason, not a bare 404.
                raise ValidationError(str(e))
            except skill_files.SkillNotFoundError:
                raise NotFoundError("skill", data["name"])
            self._invalidate_trinket_cache()
            return {
                "deleted": True,
                "name": data["name"],
                "message": f"Skill '{data['name']}' deleted"
            }

        else:
            raise ValidationError(f"Unknown action: {action}")


class ActionsEndpoint(PropagatingHandler):
    """Main actions endpoint handler with domain-based routing."""

    def __init__(self):
        super().__init__()
        self.domain_handlers = {
            DomainType.REMINDER: ReminderDomainHandler,
            DomainType.MEMORY: MemoryDomainHandler,
            DomainType.USER: UserDomainHandler,
            DomainType.CONTACTS: ContactsDomainHandler,
            DomainType.DOMAIN_KNOWLEDGE: DomainKnowledgeDomainHandler,
            DomainType.CONTINUUM: ContinuumDomainHandler,
            DomainType.LORA: LoraDomainHandler,
            DomainType.FEEDBACK: FeedbackDomainHandler,
            DomainType.PORTRAIT: PortraitDomainHandler,
            DomainType.SKILLS: SkillsDomainHandler,
        }
        # MIRA_PERSONA_ENABLED=0 omits the domain at construction instead of branching
        # inside the handler, so a disabled install rejects `persona/*` as an unknown
        # domain rather than writing revisions nothing ever injects.
        from config import config

        if config.system.persona_enabled:
            self.domain_handlers[DomainType.PERSONA] = PersonaDomainHandler
    
    def process_request(self, **params) -> dict[str, Any]:
        """Route request to appropriate domain handler."""
        current_user = params['current_user']
        user_id = current_user.user_id
        request_data = params['request_data']
        
        # Set user context for any functions that need it
        from utils.user_context import set_current_user_id
        set_current_user_id(user_id)
        
        domain = request_data.domain
        action = request_data.action
        data = request_data.data
        
        # Get domain handler
        handler_class = self.domain_handlers.get(domain)
        if not handler_class:
            raise ValidationError(f"Unknown domain: {domain}")

        # Create handler instance
        handler = handler_class()
        
        # Validate action and data
        validated_data = handler.validate_action(action, data)
        
        # Execute action
        result = handler.execute_action(action, validated_data)
        
        # Add metadata
        result["meta"] = {
            "domain": domain.value,
            "action": action,
            "timestamp": format_utc_iso(utc_now())
        }
        
        return result


def get_actions_handler() -> ActionsEndpoint:
    """Get actions endpoint handler instance."""
    return ActionsEndpoint()


@router.post("/actions")
def actions_endpoint(
    request_data: ActionRequest,
    current_user: SessionData | APITokenContext = Depends(get_current_user)
):
    """Execute state-changing operations through domain-routed actions.

    Deliberately sync (not async def) so Starlette runs it in a threadpool
    instead of blocking the event loop during blocking tool/DB work.

    Handler errors propagate to main.py's global exception handlers, which
    assign the HTTP status (APIError codes -> 400/401/403/404/429/500/503;
    anything else -> 500) and build the standard error body.
    """
    handler = get_actions_handler()
    response = handler.handle_request(request_data=request_data, current_user=current_user)
    return response.to_dict()


# =============================================================================
# TOOL QUERY ENDPOINT
# =============================================================================

# Whitelist of tools that can be queried directly via API
QUERYABLE_TOOLS = {"reminder_tool", "contacts_tool"}

# Required kwargs per (tool, operation), validated by query_tool before
# dispatch so a missing argument is a 400 naming the argument instead of a
# TypeError from the tool method surfacing as a 500. The tools' JSON schemas
# mark only "operation" required, and the per-operation requirements live in
# method signatures that run() keeps private, so the endpoint states them
# here. Operations absent from this table either take no required arguments
# (list_contacts) or validate their own optionality with ValueError
# (snooze_reminder).
TOOL_QUERY_REQUIRED_KWARGS: dict[tuple[str, str], tuple[str, ...]] = {
    ("reminder_tool", "add_reminder"): ("title", "date"),
    ("reminder_tool", "get_reminders"): ("date_filter",),
    ("reminder_tool", "mark_completed"): ("reminder_id",),
    ("reminder_tool", "update_reminder"): ("reminder_id",),
    ("reminder_tool", "delete_reminder"): ("reminder_id",),
    ("reminder_tool", "batch"): ("batch_action", "reminder_ids"),
    ("contacts_tool", "add_contact"): ("name",),
    ("contacts_tool", "get_contact"): ("identifier",),
    ("contacts_tool", "delete_contact"): ("identifier",),
    ("contacts_tool", "update_contact"): ("identifier",),
}


def _get_tool_instance(tool_name: str):
    """Import and instantiate a tool by name."""
    if tool_name == "reminder_tool":
        from tools.implementations.reminder_tool import ReminderTool
        return ReminderTool()
    elif tool_name == "contacts_tool":
        from tools.implementations.contacts_tool import ContactsTool
        return ContactsTool()
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


@router.get("/tools/{tool_name}/query")
def query_tool(
    tool_name: str,
    operation: str = Query(..., description="Tool operation to execute"),
    date_filter: str | None = Query(None, description="Date filter type (for reminder_tool)"),
    category: str | None = Query(None, description="Category filter (for reminder_tool)"),
    current_user: SessionData | APITokenContext = Depends(get_current_user)
):
    """
    Query a tool directly for read-only operations.

    This endpoint allows the UI to query tool data without going through
    the LLM or trinket system. Useful for polling current state.

    Only whitelisted tools can be queried.

    Deliberately sync (not async def) so Starlette runs it in a threadpool
    instead of blocking the event loop during the blocking tool DB queries.
    """
    # Set user context
    set_current_user_id(current_user.user_id)

    if tool_name not in QUERYABLE_TOOLS:
        return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "error": {
                    "code": "FORBIDDEN",
                    "message": f"Tool '{tool_name}' is not queryable via API"
                }
            }
        )

    try:
        tool = _get_tool_instance(tool_name)

        # Build kwargs from query params (only include non-None values)
        kwargs = {"operation": operation}
        if date_filter is not None:
            kwargs["date_filter"] = date_filter
        if category is not None:
            kwargs["category"] = category

        # A missing required argument is a caller input error, not an internal
        # fault — raise ValueError so the 400 branch below returns the standard
        # VALIDATION_ERROR envelope naming the missing argument.
        required = TOOL_QUERY_REQUIRED_KWARGS.get((tool_name, operation), ())
        missing = [name for name in required if name not in kwargs]
        if missing:
            raise ValueError(
                f"Operation '{operation}' on {tool_name} requires query parameter(s): {', '.join(missing)}"
            )

        result = tool.run(**kwargs)

        # Storage-column names (the encrypted__ prefix) never cross the API
        # boundary: the reminder/contact list payloads are projected to the
        # documented API field names, same as the actions handlers.
        if "reminders" in result:
            result["reminders"] = [_project_reminder(r) for r in result["reminders"]]
        if "contacts" in result:
            result["contacts"] = [_project_contact(c) for c in result["contacts"]]

        return {
            "success": True,
            "data": result,
            "meta": {
                "tool": tool_name,
                "operation": operation,
                "timestamp": format_utc_iso(utc_now())
            }
        }

    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(e)
                }
            }
        )
    except Exception:
        request_id = generate_request_id()
        logger.exception("Tool query error for %s (request_id: %s)", tool_name, request_id)
        # Fixed message plus request id, mirroring main.py's
        # general_exception_handler; the real exception stays in the log only.
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "details": {"request_id": request_id}
                }
            }
        )
