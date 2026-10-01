"""Per-user tool configuration storage with server-side secret fields."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from tools.registry import registry
from utils.user_credentials import UserCredentialService
from utils.userdata_manager import UserDataManager, get_user_data_manager
from utils.timezone_utils import format_utc_iso, utc_now


SECRET_REDACTION_SENTINEL = "__MIRA_SECRET_CONFIGURED__"
SECRET_CREDENTIAL_TYPE = "tool_config_secret"
TOOL_CONFIG_CREDENTIAL_TYPE = "tool_config"


@dataclass(frozen=True)
class ToolConfigPreparation:
    """Validated input config plus requested secret-field mutations."""

    config: dict[str, Any]
    secret_updates: dict[str, str | None]


def get_secret_field_names(config_class: type[BaseModel]) -> set[str]:
    """Return config fields marked as secret via json_schema_extra.secret."""
    secret_fields: set[str] = set()
    for name, field in config_class.model_fields.items():
        extra = field.json_schema_extra or {}
        if extra.get("secret") is True:
            secret_fields.add(name)
    return secret_fields


def get_tool_secret_service_name(tool_name: str, field_name: str) -> str:
    """Build the credential service name for a tool secret field."""
    return f"{tool_name}:{field_name}"


def load_user_tool_config(tool_name: str, hydrate_secrets: bool = False) -> dict[str, Any] | None:
    """Load a user's saved tool config, optionally hydrating secret fields."""
    credential_service = UserCredentialService()
    config_json = credential_service.get_credential(
        credential_type=TOOL_CONFIG_CREDENTIAL_TYPE,
        service_name=tool_name,
    )
    if not config_json:
        config: dict[str, Any] | None = None
    else:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupt tool config for {tool_name}: {e}") from e

    if not hydrate_secrets:
        return config

    config_class = registry.get(tool_name)
    if config_class is None:
        return config

    hydrated = dict(config or {})
    for field_name in get_secret_field_names(config_class):
        secret = credential_service.get_credential(
            credential_type=SECRET_CREDENTIAL_TYPE,
            service_name=get_tool_secret_service_name(tool_name, field_name),
        )
        if secret is not None:
            hydrated[field_name] = secret
    return hydrated


def save_user_tool_config(tool_name: str, config: dict[str, Any]) -> None:
    """Save a user's non-secret tool config JSON."""
    credential_service = UserCredentialService()
    credential_service.store_credential(
        credential_type=TOOL_CONFIG_CREDENTIAL_TYPE,
        service_name=tool_name,
        credential_value=json.dumps(config),
    )


def delete_user_tool_config(tool_name: str) -> bool:
    """Delete a user's saved non-secret config and any secret fields."""
    credential_service = UserCredentialService()
    deleted = credential_service.delete_credential(
        credential_type=TOOL_CONFIG_CREDENTIAL_TYPE,
        service_name=tool_name,
    )

    config_class = registry.get(tool_name)
    if config_class is not None:
        for field_name in get_secret_field_names(config_class):
            secret_deleted = credential_service.delete_credential(
                credential_type=SECRET_CREDENTIAL_TYPE,
                service_name=get_tool_secret_service_name(tool_name, field_name),
            )
            deleted = deleted or secret_deleted
    return deleted


def has_tool_secret(tool_name: str, field_name: str) -> bool:
    """Return true when a secret value is stored for a tool config field."""
    credential_service = UserCredentialService()
    secret = credential_service.get_credential(
        credential_type=SECRET_CREDENTIAL_TYPE,
        service_name=get_tool_secret_service_name(tool_name, field_name),
    )
    return secret is not None


def redact_tool_config(tool_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Redact secret fields before returning config data to any client."""
    config_class = registry.get(tool_name)
    if config_class is None:
        return dict(config)

    redacted = dict(config)
    for field_name in get_secret_field_names(config_class):
        redacted[field_name] = (
            SECRET_REDACTION_SENTINEL if has_tool_secret(tool_name, field_name) else ""
        )
    return redacted


def prepare_tool_config_for_validation(
    tool_name: str,
    incoming_config: dict[str, Any],
) -> ToolConfigPreparation:
    """Hydrate incoming config and return requested secret field mutations."""
    config_class = registry.get(tool_name)
    if config_class is None:
        return ToolConfigPreparation(config=dict(incoming_config), secret_updates={})

    secret_fields = get_secret_field_names(config_class)
    hydrated_existing = load_user_tool_config(tool_name, hydrate_secrets=True) or {}
    prepared = dict(incoming_config)
    secret_updates: dict[str, str | None] = {}

    for field_name in secret_fields:
        # Omitted secret field = sentinel semantics: preserve existing, write nothing
        incoming_value = incoming_config.get(field_name, SECRET_REDACTION_SENTINEL)
        existing_value = hydrated_existing.get(field_name)

        if incoming_value == SECRET_REDACTION_SENTINEL:
            if existing_value is not None:
                prepared[field_name] = existing_value
            else:
                # Sentinel echoed for a field with no stored value: treat as unset
                # (same as omission) instead of passing the literal sentinel through.
                prepared[field_name] = ""
            secret_updates[field_name] = None
        elif incoming_value == "":
            prepared[field_name] = ""
            secret_updates[field_name] = ""
        else:
            prepared[field_name] = incoming_value
            secret_updates[field_name] = incoming_value

    return ToolConfigPreparation(config=prepared, secret_updates=secret_updates)


def persist_secret_updates(tool_name: str, secret_updates: dict[str, str | None]) -> None:
    """Apply secret field mutations produced by prepare_tool_config_for_validation."""
    credential_service = UserCredentialService()
    for field_name, value in secret_updates.items():
        service_name = get_tool_secret_service_name(tool_name, field_name)
        if value is None:
            continue
        if value == "":
            credential_service.delete_credential(
                credential_type=SECRET_CREDENTIAL_TYPE,
                service_name=service_name,
            )
            continue
        credential_service.store_credential(
            credential_type=SECRET_CREDENTIAL_TYPE,
            service_name=service_name,
            credential_value=value,
        )


def _upsert_credential_row(
    dm: UserDataManager,
    credential_type: str,
    service_name: str,
    credential_value: str,
) -> None:
    """Upsert one credentials row WITHOUT committing.

    Mirrors UserCredentialService.store_credential's row shape, but leaves the
    commit to the caller so several writes can share a single transaction.
    """
    now = format_utc_iso(utc_now())
    row = {
        "credential_type": credential_type,
        "service_name": service_name,
        "encrypted__credential_value": credential_value,
        "metadata": "{}",
        "updated_at": now,
    }
    encrypted = dm._encrypt_dict(row)
    cursor = dm.connection.cursor()
    existing = cursor.execute(
        "SELECT id FROM credentials "
        "WHERE credential_type = :ctype AND service_name = :service",
        {"ctype": credential_type, "service": service_name},
    ).fetchone()
    if existing is not None:
        set_clause = ", ".join(f"{col} = :{col}" for col in encrypted)
        cursor.execute(
            f"UPDATE credentials SET {set_clause} "
            "WHERE credential_type = :ctype AND service_name = :service",
            {**encrypted, "ctype": credential_type, "service": service_name},
        )
    else:
        encrypted["id"] = str(uuid.uuid4())
        encrypted["created_at"] = now
        columns = ", ".join(encrypted)
        placeholders = ", ".join(f":{col}" for col in encrypted)
        cursor.execute(
            f"INSERT INTO credentials ({columns}) VALUES ({placeholders})", encrypted
        )


def save_tool_config_update(
    tool_name: str,
    secret_updates: dict[str, str | None],
    public_config: dict[str, Any],
) -> None:
    """Apply secret-field updates and the public tool config in ONE transaction.

    store_credential() commits on every call, so routing the secret rows and
    the public config through it persists them as separately-committed writes:
    a failure of the config write would strand the already-committed secret
    change. This path performs every write on the per-user connection inside
    a single SQLite transaction and commits once — a failure of the public
    -config write rolls the secret writes back.

    Sentinel semantics are those of persist_secret_updates: a None value
    writes nothing (preserve existing), "" deletes the stored secret, and any
    other value stores it.
    """
    credential_service = UserCredentialService()
    dm = get_user_data_manager(credential_service.user_id)
    dm._ensure_credentials_table()
    conn = dm.connection
    # Close any implicit read transaction left open by earlier statements so
    # BEGIN IMMEDIATE cannot nest inside it.
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        for field_name, value in secret_updates.items():
            if value is None:
                # Sentinel: field omitted/echoed — preserve existing, write nothing
                continue
            service_name = get_tool_secret_service_name(tool_name, field_name)
            if value == "":
                conn.execute(
                    "DELETE FROM credentials "
                    "WHERE credential_type = :ctype AND service_name = :service",
                    {"ctype": SECRET_CREDENTIAL_TYPE, "service": service_name},
                )
            else:
                _upsert_credential_row(dm, SECRET_CREDENTIAL_TYPE, service_name, value)
        _upsert_credential_row(
            dm,
            TOOL_CONFIG_CREDENTIAL_TYPE,
            tool_name,
            json.dumps(public_config),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def strip_secret_fields(tool_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Remove secret fields from config before storing the public config JSON."""
    config_class = registry.get(tool_name)
    if config_class is None:
        return dict(config)

    secret_fields = get_secret_field_names(config_class)
    return {key: value for key, value in config.items() if key not in secret_fields}
