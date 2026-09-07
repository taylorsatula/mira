"""
Global pytest configuration and fixtures for MIRA tests.

Harness split (O-22): the recovered suite mixes pure unit tests with
tests that require live Vault/Postgres/Valkey.  By default only the unit
tests run; infrastructure-dependent tests are tagged ``integration`` and
reported SKIPPED (never ERROR at setup).  Run them with::

    python3 -m pytest tests/ --integration

or export VAULT_ADDR / VAULT_ROLE_ID / VAULT_SECRET_ID, which is detected
automatically.  Classification logic and the availability probe live in
``tests.fixtures.infra``.
"""
import pytest

from tests.fixtures import infra
from tests.fixtures.infra import integration_mode
from tests.fixtures.reset import full_reset

# Import fixtures to make them available globally
from tests.fixtures.isolation import *
from tests.fixtures.auth import *
from tests.fixtures.core import *


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires live DB + Vault + Valkey)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: marks test as requiring live DB + Vault"
    )
    # tests/fixtures/sqlite_test_db.py contract used by tool schema tests;
    # registered here so the suite no longer warns about an unknown marker.
    config.addinivalue_line(
        "markers", "schema_files(files): SQL schema files to load into the SQLite test DB"
    )
    # 18 recovered files fail at *collection* because their subject code
    # arrives in later work packages or was deleted (tests/TRIAGE.md §A/§B).
    # That is the expected WP0 end state; it must not abort the 671 tests
    # that do collect, so a default run continues past collection errors.
    # The errors are still reported and still exit non-zero.
    if not config.option.continue_on_collection_errors:
        config.option.continue_on_collection_errors = True


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    infra.mark_and_gate_integration_items(config, items)


@pytest.fixture(autouse=True, scope="function")
def reset_test_environment(request):
    """
    Reset the test environment before and after each test.

    Ensures each test starts with completely fresh state by clearing
    all connection pools, singletons, and caches.

    Inert outside integration mode: ``full_reset()`` transitively builds a
    Vault client and database pools, which errors for every test when the
    infrastructure is absent.  See ``tests.fixtures.infra``.
    """
    if not integration_mode(request.config):
        yield
        return

    # Reset before test
    full_reset()

    yield

    # Reset after test
    full_reset()


@pytest.fixture(scope="session", autouse=True)
def event_loop_policy():
    """
    Set event loop policy for the entire test session.

    Ensures consistent event loop behavior across all tests.
    """
    import asyncio
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
