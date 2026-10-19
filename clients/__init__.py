"""
Client modules for external service integrations.

Importing this package pulls in the embeddings, Postgres, SQLite, Valkey, and
Vault modules and opens nothing: every connection is made by a getter or a
constructor at first use, so an offline battery can import the package with no
live infrastructure. `LLMProvider` is deferred through `__getattr__` because
importing `llm_provider` installs a SIGUSR1 handler, and `signal.signal()`
raises off the main thread.
"""

from .embeddings_provider import get_embeddings_provider, EmbeddingsProvider
from .postgres_client import PostgresClient
from .sqlite_client import SQLiteClient
from .valkey_client import ValkeyClient, get_valkey, get_valkey_client
from .vault_client import get_auth_secret, get_database_url, get_service_config


def __getattr__(name):
    if name == "LLMProvider":
        from .llm_provider import LLMProvider
        return LLMProvider
    raise AttributeError(name)

__all__ = [
    'EmbeddingsProvider',
    'get_embeddings_provider',
    'LLMProvider',
    'PostgresClient',
    'SQLiteClient',
    'ValkeyClient',
    'get_valkey',
    'get_valkey_client',
    'get_auth_secret',
    'get_database_url',
    'get_service_config'
]
