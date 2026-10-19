"""
Vault KV-v2 secret reads over AppRole authentication.

The AppRole bootstrap credentials (`VAULT_ADDR`, `VAULT_ROLE_ID`,
`VAULT_SECRET_ID`) come from the process environment; every secret value comes
from Vault. Each field read is cached in `_secret_cache` for the process
lifetime and nothing invalidates it, so a secret rotated in Vault does not
reach a running process — code that re-reads to "refresh" serves the stale
value until restart. `preload_secrets()` fills the three groups at startup
while the AppRole token is fresh and raises on any failed group, which turns a
missing secret into one startup error instead of scattered runtime failures.

`check_required_secrets()` reports manifest fields absent from Vault for the
boot gate; it raises rather than reporting a gap when Vault itself is
unreachable or unauthenticated, so an outage is never mistaken for a missing
secret. `get_auth_secret` (the `mira/auth` family) has no caller in the tree.
"""

import json
import os
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, Dict, TypedDict
import hvac
from hvac.exceptions import VaultError, InvalidPath, Unauthorized, Forbidden

logger = logging.getLogger(__name__)

_vault_client_instance: Optional['VaultClient'] = None
_secret_cache: Dict[str, str] = {}
_vertex_credentials = None  # google.oauth2.service_account.Credentials singleton


def _ensure_vault_client() -> 'VaultClient':
    global _vault_client_instance
    if _vault_client_instance is None:
        _vault_client_instance = VaultClient()
    return _vault_client_instance


class VaultClient:
    """Production client with AppRole auth, env-based config, fail-fast validation, and KV v2 API."""

    def __init__(self, vault_addr: Optional[str] = None,
                 vault_namespace: Optional[str] = None):
        """Initializes with env vars, validates config, authenticates via AppRole, fails fast on errors."""
        try:

            self.vault_addr = vault_addr or os.getenv('VAULT_ADDR')
            self.vault_namespace = vault_namespace or os.getenv('VAULT_NAMESPACE')
            
            self.vault_role_id = os.getenv('VAULT_ROLE_ID')
            self.vault_secret_id = os.getenv('VAULT_SECRET_ID')
            
            if not self.vault_addr:
                raise ValueError("VAULT_ADDR environment variable is required")
            
            if not self.vault_role_id or not self.vault_secret_id:
                raise ValueError("VAULT_ROLE_ID and VAULT_SECRET_ID environment variables are required")
            
            client_kwargs = {'url': self.vault_addr}
            
            if self.vault_namespace:
                client_kwargs['namespace'] = self.vault_namespace
            
            self.client = hvac.Client(**client_kwargs)
            self._authenticate_approle()
            
            if not self.client.is_authenticated():
                raise PermissionError("Vault authentication failed")

            logger.toast(f"Vault client initialized: {self.vault_addr}")

        except Exception as e:
            logger.error(f"Vault client initialization failed: {e}", exc_info=True)
            raise
    
    def _authenticate_approle(self):
        try:
            auth_response = self.client.auth.approle.login(
                role_id=self.vault_role_id,
                secret_id=self.vault_secret_id
            )
            
            self.client.token = auth_response['auth']['client_token']
            logger.toast("AppRole authentication successful")

        except Exception as e:
            logger.error(f"AppRole authentication failed: {e}", exc_info=True)
            raise PermissionError(f"AppRole authentication failed: {str(e)}")
    
    def _read_secret_version(self, path: str) -> dict:
        """Read a secret, re-authenticating once when the AppRole token expires."""
        try:
            return self.client.secrets.kv.v2.read_secret_version(
                path=path, raise_on_deleted_version=True
            )
        except (Unauthorized, Forbidden):
            logger.warning("Vault token expired or was revoked; re-authenticating AppRole")
            self._authenticate_approle()
            return self.client.secrets.kv.v2.read_secret_version(
                path=path, raise_on_deleted_version=True
            )

    def get_secret(self, path: str, field: str) -> str:
        """Retrieves single field from KV v2 API with structured error handling."""
        try:
            response = self._read_secret_version(path)
            secret_data = response['data']['data']

            if field not in secret_data:
                available_fields = list(secret_data.keys())
                raise KeyError(f"Field '{field}' not found in secret '{path}'. Available: {', '.join(available_fields)}")

            return secret_data[field]

        except InvalidPath:
            logger.error(f"Secret path not found: {path}", exc_info=True)
            raise FileNotFoundError(f"Secret path '{path}' not found in Vault")

        except (Unauthorized, Forbidden) as e:
            logger.error(f"Access denied to secret {path}/{field}: {e}", exc_info=True)
            raise PermissionError(f"Access denied to secret '{path}': {str(e)}")

        except VaultError as e:
            logger.error(f"Vault API error for {path}/{field}: {e}", exc_info=True)
            raise RuntimeError(f"Vault API error retrieving '{path}/{field}': {str(e)}")

def get_database_url(service: str, admin: bool = False) -> str:
    """
    Get database URL from Vault for mira_service.

    Args:
        service: Database service name (only 'mira_service' supported)
        admin: If True, returns admin connection string (mira_admin role with BYPASSRLS)

    Returns:
        PostgreSQL connection URL
    """
    if service != 'mira_service':
        raise ValueError(f"Unknown database service: '{service}'. Only 'mira_service' is supported.")

    field = 'admin_url' if admin else 'service_url'
    cache_key = f"mira/database/{field}"

    if cache_key in _secret_cache:
        return _secret_cache[cache_key]

    vault_client = _ensure_vault_client()
    value = vault_client.get_secret('mira/database', field)
    _secret_cache[cache_key] = value
    return value


def _mint_vertex_token() -> str:
    """Mint or refresh a Vertex AI access token from SA credentials in Vault."""
    global _vertex_credentials
    if _vertex_credentials is None:
        from google.oauth2 import service_account
        cache_key = 'mira/api_keys/google_vertex_sa'
        sa_json = _secret_cache.get(cache_key) or _ensure_vault_client().get_secret('mira/api_keys', 'google_vertex_sa')
        info = json.loads(sa_json)
        _vertex_credentials = service_account.Credentials.from_service_account_info(
            info, scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        logger.info("Vertex AI service account credentials initialized")
    if not _vertex_credentials.valid:
        import google.auth.transport.requests
        _vertex_credentials.refresh(google.auth.transport.requests.Request())
    return _vertex_credentials.token


def get_api_key(key_name: str) -> str:
    if key_name == 'google_vertex':
        return _mint_vertex_token()

    cache_key = f"mira/api_keys/{key_name}"

    if cache_key in _secret_cache:
        return _secret_cache[cache_key]

    vault_client = _ensure_vault_client()
    value = vault_client.get_secret('mira/api_keys', key_name)
    _secret_cache[cache_key] = value
    return value


def get_auth_secret(secret_name: str) -> str:
    cache_key = f"mira/auth/{secret_name}"
    
    if cache_key in _secret_cache:
        return _secret_cache[cache_key]
    
    vault_client = _ensure_vault_client()
    value = vault_client.get_secret('mira/auth', secret_name)
    _secret_cache[cache_key] = value
    return value


def get_service_config(field: str) -> str:
    cache_key = f"mira/services/{field}"

    if cache_key in _secret_cache:
        return _secret_cache[cache_key]

    vault_client = _ensure_vault_client()
    value = vault_client.get_secret('mira/services', field)
    _secret_cache[cache_key] = value
    return value


def preload_secrets() -> None:
    """
    Load all secrets into memory cache at startup.

    This prevents token expiration issues by caching everything
    while the AppRole token is still valid. Raises on any failure —
    missing secrets at startup means scattered failures at runtime.
    """
    vault_client = _ensure_vault_client()

    secret_groups = [
        ('mira/api_keys', 'API keys'),
        ('mira/database', 'database secrets'),
        ('mira/services', 'service config secrets'),
    ]

    successes = []
    failures = []

    for path, label in secret_groups:
        try:
            response = vault_client._read_secret_version(path)
            secrets = response['data']['data']
            for field, value in secrets.items():
                _secret_cache[f"{path}/{field}"] = value
            successes.append(f"{len(secrets)} {label}")
        except Exception as e:
            failures.append(f"{label}: {e}")

    if successes:
        logger.info(f"Preloaded secrets: {', '.join(successes)}")

    if failures:
        raise RuntimeError(
            f"Failed to preload {len(failures)} secret group(s): {'; '.join(failures)}"
        )


# Fields every install writes and the runtime cannot start without. Writers:
# deploy/postgresql.sh Step 14 and deploy/docker/scripts/init-mira.sh:init_vault.
# mira/api_keys names only subcortical_key: the chat-tier key's field depends
# on the provider mode (provider_key or anthropic_key), and a container writes
# neither — requiring one would park every install of the other kind.
REQUIRED_SECRET_MANIFEST: Dict[str, tuple[str, ...]] = {
    "mira/database": ("admin_url", "service_url", "username", "password"),
    "mira/services": ("valkey_url", "app_url", "userdata_encryption_key"),
    "mira/api_keys": ("subcortical_key",),
}


@dataclass(frozen=True)
class SecretGap:
    """A manifest field absent from Vault; path_exists says whether its secret exists at all."""

    path: str
    field: str
    path_exists: bool


def check_required_secrets() -> list[SecretGap]:
    """
    Every REQUIRED_SECRET_MANIFEST field absent from Vault ([] when none is).

    Presence is the contract; values are validated by their readers.

    Raises:
        PermissionError, RuntimeError, ValueError: Vault unreachable, sealed,
            unauthenticated, or misconfigured — an infrastructure failure is
            never reported as a gap.
    """
    vault_client = _ensure_vault_client()
    gaps: list[SecretGap] = []
    for path, fields in REQUIRED_SECRET_MANIFEST.items():
        try:
            data = vault_client._read_secret_version(path)["data"]["data"]
            exists = True
        except InvalidPath:
            data = {}
            exists = False
        except (Unauthorized, Forbidden) as e:
            raise PermissionError(f"Access denied to secret '{path}': {e}") from e
        except VaultError as e:
            raise RuntimeError(f"Vault API error reading '{path}': {e}") from e
        gaps.extend(SecretGap(path, field, exists) for field in fields if field not in data)
    return gaps


def format_secret_gap(gaps: Iterable[SecretGap]) -> str:
    """The exact command(s) that fill the given gaps, "; "-joined.

    A gap in a secret that exists is one `vault kv patch` per field
    (patch merges, so the secret's other fields survive). A secret
    missing altogether is a single `vault kv put` carrying every missing
    field of that path — put replaces the whole secret, so one put per
    field would erase the fixes before it.
    """
    by_path: dict[str, list[SecretGap]] = {}
    for gap in gaps:
        by_path.setdefault(gap.path, []).append(gap)
    commands: list[str] = []
    for path, path_gaps in by_path.items():
        if all(gap.path_exists for gap in path_gaps):
            commands.extend(
                f"vault kv patch secret/{path} {gap.field}=<value>"
                for gap in path_gaps
            )
        else:
            fields = " ".join(f"{gap.field}=<value>" for gap in path_gaps)
            commands.append(f"vault kv put secret/{path} {fields}")
    return "; ".join(commands)


class VaultHealthCheck(TypedDict, total=False):
    """Health check result from test_vault_connection."""
    status: str
    vault_addr: str
    namespace: Optional[str]
    authenticated: bool
    message: str


def test_vault_connection() -> VaultHealthCheck:
    try:
        vault_client = _ensure_vault_client()
        vault_client.get_secret('mira/database', 'username')

        logger.info("Vault connection test successful")
        return {
            "status": "success",
            "vault_addr": vault_client.vault_addr,
            "namespace": vault_client.vault_namespace,
            "authenticated": True,
            "message": "Vault connection successful"
        }

    except Exception as e:
        logger.error(f"Vault connection test failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "authenticated": False
        }