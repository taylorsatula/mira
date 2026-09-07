"""
Infrastructure availability probe and integration-test classification (O-22).

The recovered suite mixes pure unit tests with tests that transitively
construct a Vault client, PostgreSQL/Valkey connections or the full
FastAPI application.  ``tests/conftest.py`` consults this module to:

* decide whether the session runs in **integration mode**
  (``--integration`` passed, or all of ``VAULT_ADDR`` / ``VAULT_ROLE_ID`` /
  ``VAULT_SECRET_ID`` present in the environment),
* tag every infrastructure-dependent test with ``pytest.mark.integration``
  and — when not in integration mode — report it as SKIPPED instead of
  ERROR at setup, and
* let the autouse reset/cleanup fixtures
  (``reset_test_environment``, ``reset_global_state``,
  ``cleanup_test_data``, ``clean_test_valkey``) stay inert no-ops outside
  integration mode.

The availability probe is deliberately an environment-variable presence
check, never a network probe: it must not throw and must not add connect
timeouts to every test.  A machine with the Vault AppRole variables
exported is assumed to have (or want) live infrastructure; if it does
not, integration mode fails loudly — which is the honest signal for that
configuration.

Classification rules, in order (a test is integration-style if any hits):

1. it carries an explicit ``@pytest.mark.integration``;
2. its file is listed in :data:`FILE_SKIP_REASONS` — evidence: the module
   constructs live infrastructure (or imports the stack that does) directly
   in test/helper bodies, which no fixture-based rule can see;
3. its transitive fixture closure requests a name in
   :data:`INFRA_FIXTURES` — evidence: those fixtures construct
   ``PostgresClient``, ``LTMemorySession``-backed managers, a Vault or
   Valkey client, ``AuthDatabase``, or ``create_app()``.

Autouse cleanup fixtures are *not* in :data:`INFRA_FIXTURES`: they appear
in every item's closure and are gated by integration mode instead
(see :func:`integration_mode`).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pytest

# Environment variables whose presence (non-empty) is treated as "live
# infrastructure is configured".  Same trio that clients/vault_client.py
# requires before it will even attempt an AppRole login.
INTEGRATION_ENV_VARS = ("VAULT_ADDR", "VAULT_ROLE_ID", "VAULT_SECRET_ID")

SKIP_REASON = (
    "integration test: requires live Vault/Postgres/Valkey; "
    "run with `pytest --integration` or set VAULT_ADDR/VAULT_ROLE_ID/VAULT_SECRET_ID"
)

_TESTS_DIR = Path(__file__).resolve().parent.parent

# Fixtures (defined in tests/conftest.py, tests/fixtures/{core,auth,lt_memory}
# ) that construct live infrastructure when instantiated.  Requesting one of
# these — directly or transitively — makes a test integration-style.
INFRA_FIXTURES = frozenset(
    {
        # tests/fixtures/core.py — Postgres / Vault / Valkey / app
        "test_db",
        "test_memory_db",
        "conversation_repository",
        "continuum_repo",
        "vault_client",
        "valkey_client",
        "authenticated_user",
        "second_authenticated_user",
        "realistic_conversation",
        "chat_handler",
        "test_client",
        "authenticated_client",
        "async_client",
        "authenticated_async_client",
        # tests/fixtures/auth.py — AuthDatabase / sessions
        "magic_link_token",
        "expired_magic_link_token",
        "session_token",
        "auth_service",
        # tests/fixtures/isolation.py — Valkey lock clearing
        "clean_user_locks",
        # tests/lt_memory/conftest.py — real LT_Memory stack and embeddings service
        "test_user",
        "lt_memory_session_manager",
        "lt_memory_db",
        "vector_ops",
        "embeddings_provider",
    }
)

# Whole files whose tests cannot run against the current tree without more
# than pure-Python inputs, but which no fixture-closure rule can see because
# they construct live infrastructure (or import the stack that does) directly
# in test/helper bodies.  Each entry carries the skip reason shown when the
# file is not run; every one is justified below and was verified by running
# the file without live infrastructure and confirming the failure is
# infrastructure/absent-stack (Vault ValueError, pool creation failure,
# provider construction), not a behavioural assertion.
# A --integration run executes these files anyway, so the entries hide
# nothing from a run that has the stack.
FILE_SKIP_REASONS = {
    # Per-test bodies construct UserDataManager pools, which resolve
    # credentials through Vault-backed session managers (15/15 tests raise
    # "VAULT_ADDR environment variable is required").
    "tests/utils/test_userdata_manager_connection.py":
        "requires live Vault/Postgres (UserDataManager pools resolve via Vault)",
    # Fixtures construct the real HybridEmbeddingsProvider, whose service
    # config resolves through Vault.  Note: 19 of these tests additionally
    # fail on a genuine 1.x-to-HEAD signature drift (the fixtures pass
    # enable_reranker, which clients/hybrid_embeddings_provider.py dropped);
    # that signal surfaces under --integration, not as a skip here.
    "tests/clients/test_hybrid_embeddings_provider.py":
        "requires live Vault + embedding service (real HybridEmbeddingsProvider construction)",
    # The 28 items that request no infra fixture still fail at setup: the
    # continuum search fixtures build the embeddings stack, which imports
    # sentence-transformers/CrossEncoder and resolves the provider URL via
    # Vault.  The rest of the file was already classified through
    # authenticated_user/test_db.
    "tests/tools/implementations/test_continuum_tool.py":
        "requires live continuum/embeddings stack (search fixtures construct the provider)",
}


@lru_cache(maxsize=1)
def infra_env_configured() -> bool:
    """True when the Vault AppRole environment is fully populated.

    Pure environment check: fast, non-throwing, no network I/O.
    """
    return all(os.environ.get(name, "").strip() for name in INTEGRATION_ENV_VARS)


def integration_mode(config: pytest.Config) -> bool:
    """True when this session should exercise live infrastructure.

    Driven by ``--integration`` or, as a convenience for developers with a
    live environment already configured, by the Vault env probe.
    """
    if config.getoption("integration", default=False):
        return True
    return infra_env_configured()


def _relative_test_path(item: pytest.Item) -> str:
    """Repo-root-relative POSIX path of the file an item was collected from."""
    resolved = Path(str(item.fspath)).resolve()
    try:
        return resolved.relative_to(_TESTS_DIR.parent).as_posix()
    except ValueError:
        return resolved.as_posix()


def needs_infra(item: pytest.Item) -> bool:
    """Classify a collected item as integration-style."""
    if item.get_closest_marker("integration") is not None:
        return True
    if _relative_test_path(item) in FILE_SKIP_REASONS:
        return True
    return bool(INFRA_FIXTURES & set(getattr(item, "fixturenames", ())))


def skip_reason_for(item: pytest.Item) -> str:
    """Human-auditable reason for why an item cannot run without infrastructure."""
    reason = FILE_SKIP_REASONS.get(_relative_test_path(item))
    if reason:
        return f"integration test: {reason}; run with `pytest --integration`"
    return SKIP_REASON


def mark_and_gate_integration_items(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tag integration-style items; skip them when infrastructure mode is off.

    Called from ``pytest_collection_modifyitems`` in tests/conftest.py.
    Skipping (not erroring) is the contract: a skipped test is an honest
    "cannot run here" signal, while a setup ERROR is indistinguishable from
    a broken test.
    """
    for item in items:
        if not needs_infra(item):
            continue
        item.add_marker(pytest.mark.integration)
        if not integration_mode(config):
            item.add_marker(pytest.mark.skip(reason=skip_reason_for(item)))
