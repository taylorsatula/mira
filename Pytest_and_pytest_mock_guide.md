# Pytest and pytest-mock Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [pytest Basics](#pytest-basics)
3. [pytest-mock Plugin](#pytest-mock-plugin)
4. [Fixtures Deep Dive](#fixtures-deep-dive)
5. [Parametrization](#parametrization)
6. [Async Testing with pytest-asyncio](#async-testing-with-pytest-asyncio)
7. [Built-in Fixtures](#built-in-fixtures)
8. [Test Organization and Best Practices](#test-organization-and-best-practices)
9. [Common Patterns and Advanced Usage](#common-patterns-and-advanced-usage)
10. [Migration from unittest](#migration-from-unittest)
11. [Quick Reference](#quick-reference)

## Introduction

This guide provides comprehensive coverage of pytest and pytest-mock, based on 2024 best practices. It complements the unittest guide and shows how pytest offers a more modern, flexible approach to Python testing.

### Why pytest?
- **Concise Syntax**: Tests are plain Python functions, minimal boilerplate
- **Powerful Fixtures**: Dependency injection for test setup and teardown
- **Rich Plugin Ecosystem**: Hundreds of plugins available
- **Better Assertion Messages**: Introspective assertion failures
- **Parametrization**: Built-in support for test parameterization
- **Parallel Execution**: Native support via pytest-xdist

### When to Use pytest vs unittest
- **Use pytest** for: Modern projects, flexibility, advanced features, minimal boilerplate
- **Use unittest** for: Legacy compatibility, standard library preference, structured OOP approach

## pytest Basics

### Simple Test Structure

```python
# test_calculator.py
def add(x, y):
    return x + y

def test_add():
    """Test addition functionality."""
    result = add(2, 3)
    assert result == 5

def test_add_negative():
    """Test addition with negative numbers."""
    result = add(-1, 1)
    assert result == 0

def test_add_zero():
    """Test addition with zero."""
    result = add(5, 0)
    assert result == 5
```

### Running Tests

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Run specific test file
pytest test_calculator.py

# Run specific test function
pytest test_calculator.py::test_add

# Run tests matching pattern
pytest -k "add"

# Stop after first failure
pytest -x

# Show local variables in tracebacks
pytest -l

# Run with coverage
pytest --cov=myapp

# Parallel execution (requires pytest-xdist)
pytest -n auto
```

### Test Discovery

pytest discovers tests by:
- Files: `test_*.py` or `*_test.py`
- Functions: Functions starting with `test_`
- Classes: Classes starting with `Test` (without `__init__` method)
- Methods: Methods starting with `test_` in `Test` classes

### Assertion Introspection

```python
def test_assertion_introspection():
    a = [1, 2, 3]
    b = [1, 2, 4]
    
    # pytest provides detailed failure information
    assert a == b
    # AssertionError: assert [1, 2, 3] == [1, 2, 4]
    # At index 2 diff: 3 != 4
```

### Exception Testing

```python
import pytest

def divide(x, y):
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y

def test_divide_by_zero():
    """Test exception handling."""
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)

def test_divide_by_zero_captures_exception():
    """Test exception handling with exception capture."""
    with pytest.raises(ValueError) as exc_info:
        divide(10, 0)
    
    assert "Cannot divide by zero" in str(exc_info.value)
    assert exc_info.type is ValueError
```

## pytest-mock Plugin

### Installation and Setup

```bash
pip install pytest-mock
```

The pytest-mock plugin provides a `mocker` fixture that wraps `unittest.mock` and provides automatic cleanup.

### Basic Mocking with mocker

```python
import requests

def fetch_user_data(user_id):
    """Fetch user data from external API."""
    response = requests.get(f"https://api.example.com/users/{user_id}")
    return response.json()

def test_fetch_user_data(mocker):
    """Test API call with mocked response."""
    # Mock the requests.get method
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"id": 123, "name": "John Doe"}
    mock_response.status_code = 200
    
    mocker.patch('requests.get', return_value=mock_response)
    
    # Test the function
    result = fetch_user_data(123)
    
    # Assertions
    assert result["id"] == 123
    assert result["name"] == "John Doe"
    
    # Verify the mock was called correctly
    requests.get.assert_called_once_with("https://api.example.com/users/123")
```

### mocker vs unittest.mock

```python
# Using unittest.mock (manual cleanup required)
from unittest.mock import patch

def test_with_unittest_mock():
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {"data": "test"}
        # Test code here
        # Cleanup happens at end of context manager

# Using pytest-mock (automatic cleanup)
def test_with_mocker(mocker):
    mock_get = mocker.patch('requests.get')
    mock_get.return_value.json.return_value = {"data": "test"}
    # Test code here
    # Cleanup happens automatically after test
```

### mocker Methods

```python
def test_mocker_methods(mocker):
    # Patch a function/method
    mock_func = mocker.patch('module.function')
    
    # Patch object method
    mock_method = mocker.patch.object(MyClass, 'method')
    
    # Create mock objects
    mock_obj = mocker.Mock()
    magic_mock = mocker.MagicMock()
    
    # Create spy (partial mock)
    spy = mocker.spy(MyClass, 'method')
    
    # Create property mock
    prop_mock = mocker.PropertyMock(return_value="mocked_value")
    
    # Mock multiple things
    mocker.patch.multiple(
        'module',
        function1=mocker.Mock(return_value="mock1"),
        function2=mocker.Mock(return_value="mock2")
    )
```

### Side Effects and Return Values

```python
def test_side_effects(mocker):
    mock_func = mocker.Mock()
    
    # Simple return value
    mock_func.return_value = "fixed_result"
    
    # Side effect with function
    def custom_side_effect(arg):
        if arg == "error":
            raise ValueError("Custom error")
        return f"processed_{arg}"
    
    mock_func.side_effect = custom_side_effect
    
    # Side effect with list (sequential returns)
    mock_func.side_effect = [1, 2, 3, StopIteration]
    
    # Side effect with exception
    mock_func.side_effect = ValueError("Something went wrong")
```

## Fixtures Deep Dive

### Basic Fixtures

```python
import pytest

@pytest.fixture
def user_data():
    """Provide test user data."""
    return {
        "id": 123,
        "name": "John Doe",
        "email": "john@example.com"
    }

def test_user_creation(user_data):
    """Test using fixture."""
    user = create_user(user_data)
    assert user.name == "John Doe"
    assert user.email == "john@example.com"
```

### Fixture Scopes

```python
# Function scope (default) - runs for each test function
@pytest.fixture(scope="function")
def function_fixture():
    return "function_data"

# Class scope - runs once per test class
@pytest.fixture(scope="class")
def class_fixture():
    return "class_data"

# Module scope - runs once per test module
@pytest.fixture(scope="module")
def module_fixture():
    return "module_data"

# Package scope - runs once per test package
@pytest.fixture(scope="package")
def package_fixture():
    return "package_data"

# Session scope - runs once per test session
@pytest.fixture(scope="session")
def session_fixture():
    return "session_data"
```

### Fixture with Setup and Teardown

```python
@pytest.fixture
def database_connection():
    """Provide database connection with cleanup."""
    # Setup
    connection = create_database_connection()
    connection.begin_transaction()
    
    yield connection  # This is what gets injected into tests
    
    # Teardown
    connection.rollback()
    connection.close()

def test_database_operation(database_connection):
    """Test using database fixture."""
    result = database_connection.execute("SELECT 1")
    assert result is not None
```

### Fixture Dependencies

```python
@pytest.fixture
def database():
    """Database fixture."""
    return create_test_database()

@pytest.fixture
def user_repository(database):
    """User repository that depends on database fixture."""
    return UserRepository(database)

@pytest.fixture
def user_service(user_repository):
    """User service that depends on user repository."""
    return UserService(user_repository)

def test_user_creation(user_service):
    """Test using composed fixtures."""
    user = user_service.create_user("test@example.com", "Test User")
    assert user.email == "test@example.com"
```

### Autouse Fixtures

```python
@pytest.fixture(autouse=True)
def setup_logging():
    """Automatically run for every test without explicit request."""
    # Setup logging configuration
    configure_test_logging()
    yield
    # Cleanup logging
    reset_logging()

@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Setup test environment once per session."""
    # Initialize test environment
    initialize_test_env()
    yield
    # Cleanup test environment
    cleanup_test_env()
```

### conftest.py

```python
# conftest.py - shared fixtures across multiple test files
import pytest
from myapp import create_app, get_database

@pytest.fixture(scope="session")
def app():
    """Create application instance."""
    app = create_app(testing=True)
    yield app

@pytest.fixture(scope="session")
def client(app):
    """Create test client."""
    return app.test_client()

@pytest.fixture(scope="function")
def db_session():
    """Create database session with rollback."""
    session = get_database().session
    session.begin()
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def authenticated_user(client):
    """Create authenticated user for tests."""
    user_data = {"email": "test@example.com", "password": "password"}
    response = client.post("/auth/login", json=user_data)
    return response.json()
```

## Parametrization

### Basic Parametrization

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
    (5, 25),
])
def test_square(input, expected):
    """Test square function with multiple inputs."""
    assert square(input) == expected

@pytest.mark.parametrize("x,y,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (10, -5, 5),
])
def test_addition(x, y, expected):
    """Test addition with multiple input combinations."""
    assert add(x, y) == expected
```

### Parametrization with IDs

```python
@pytest.mark.parametrize(
    "test_input,expected",
    [
        pytest.param(2, 4, id="positive"),
        pytest.param(0, 0, id="zero"),
        pytest.param(-3, 9, id="negative"),
    ]
)
def test_square_with_ids(test_input, expected):
    assert square(test_input) == expected

# Custom ID function
def id_func(param):
    if isinstance(param, dict):
        return f"user_{param.get('id', 'unknown')}"
    return str(param)

@pytest.mark.parametrize(
    "user_data",
    [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"},
    ],
    ids=id_func
)
def test_user_processing(user_data):
    result = process_user(user_data)
    assert result["processed"] is True
```

### Stacking Parametrization

```python
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [3, 4])
def test_cartesian_product(x, y):
    """Test all combinations of x and y."""
    # This creates 4 tests: (1,3), (1,4), (2,3), (2,4)
    result = multiply(x, y)
    assert isinstance(result, int)
```

### Indirect Parametrization

```python
@pytest.fixture
def database_type(request):
    """Fixture that receives parameter indirectly."""
    db_type = request.param
    if db_type == "sqlite":
        return create_sqlite_database()
    elif db_type == "postgres":
        return create_postgres_database()
    else:
        raise ValueError(f"Unknown database type: {db_type}")

@pytest.mark.parametrize(
    "database_type",
    ["sqlite", "postgres"],
    indirect=True  # Pass parameter through fixture
)
def test_database_operations(database_type):
    """Test with different database backends."""
    result = database_type.query("SELECT 1")
    assert result is not None
```

### Dynamic Parametrization

```python
def pytest_generate_tests(metafunc):
    """Generate tests dynamically."""
    if "user_role" in metafunc.fixturenames:
        # Dynamically generate test parameters
        roles = get_available_roles()  # Could read from config/database
        metafunc.parametrize("user_role", roles)

def test_user_permissions(user_role):
    """Test user permissions for different roles."""
    permissions = get_permissions_for_role(user_role)
    assert len(permissions) > 0
```

## Async Testing with pytest-asyncio

### Installation

```bash
pip install pytest-asyncio
```

### Basic Async Testing

```python
import pytest
import asyncio
import aiohttp

# Mark module as async
pytestmark = pytest.mark.asyncio

async def fetch_data(url):
    """Async function to fetch data."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

@pytest.mark.asyncio
async def test_fetch_data():
    """Test async function."""
    # Mock or use real endpoint
    result = await fetch_data("https://api.example.com/data")
    assert "data" in result

@pytest.mark.asyncio
async def test_fetch_data_with_mock(mocker):
    """Test async function with mock."""
    mock_response = mocker.AsyncMock()
    mock_response.json.return_value = {"data": "test_value"}
    
    mock_session = mocker.AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response
    
    mocker.patch('aiohttp.ClientSession', return_value=mock_session)
    
    result = await fetch_data("https://api.example.com/data")
    assert result["data"] == "test_value"
```

### Async Fixtures

```python
@pytest.fixture
async def async_client():
    """Async fixture providing HTTP client."""
    async with aiohttp.ClientSession() as session:
        yield session

@pytest.fixture(scope="session")
async def async_database():
    """Session-scoped async fixture."""
    db = await create_async_database()
    yield db
    await db.close()

@pytest.mark.asyncio
async def test_with_async_fixture(async_client):
    """Test using async fixture."""
    async with async_client.get("https://api.example.com") as response:
        data = await response.json()
        assert response.status == 200
```

### Async Parametrization

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("url,expected_status", [
    ("https://api.example.com/users", 200),
    ("https://api.example.com/posts", 200),
    ("https://api.example.com/invalid", 404),
])
async def test_api_endpoints(async_client, url, expected_status):
    """Test multiple API endpoints."""
    async with async_client.get(url) as response:
        assert response.status == expected_status
```

### Testing Async Context Managers

```python
class AsyncResource:
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        # Connection logic
        pass
    
    async def disconnect(self):
        # Disconnection logic
        pass

@pytest.mark.asyncio
async def test_async_context_manager(mocker):
    """Test async context manager."""
    resource = AsyncResource()
    
    # Mock the methods
    mocker.patch.object(resource, 'connect', new_callable=mocker.AsyncMock)
    mocker.patch.object(resource, 'disconnect', new_callable=mocker.AsyncMock)
    
    async with resource:
        # Resource should be connected
        resource.connect.assert_called_once()
    
    # Resource should be disconnected after exiting context
    resource.disconnect.assert_called_once()
```

## Built-in Fixtures

### tmp_path and tmp_path_factory

```python
def test_file_operations(tmp_path):
    """Test file operations with temporary directory."""
    # tmp_path is a pathlib.Path object
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, World!")
    
    content = test_file.read_text()
    assert content == "Hello, World!"
    
    # Directory is automatically cleaned up

def test_shared_tmp_dir(tmp_path_factory):
    """Test with shared temporary directory."""
    shared_dir = tmp_path_factory.mktemp("shared")
    config_file = shared_dir / "config.json"
    config_file.write_text('{"setting": "value"}')
    
    # This directory can be shared across multiple tests
    assert config_file.exists()
```

### capfd and capsys

```python
def noisy_function():
    print("stdout message")
    print("stderr message", file=sys.stderr)
    return "result"

def test_output_capture(capsys):
    """Test stdout/stderr capture."""
    result = noisy_function()
    captured = capsys.readouterr()
    
    assert captured.out == "stdout message\n"
    assert captured.err == "stderr message\n"
    assert result == "result"

def test_output_capture_binary(capfd):
    """Test file descriptor capture (includes subprocess output)."""
    import subprocess
    subprocess.run(["echo", "hello"])
    
    captured = capfd.readouterr()
    assert "hello" in captured.out
```

### caplog

```python
import logging

def function_that_logs():
    logger = logging.getLogger(__name__)
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")

def test_logging(caplog):
    """Test log capture."""
    with caplog.at_level(logging.INFO):
        function_that_logs()
    
    # Check log messages
    assert "info message" in caplog.text
    assert "warning" in caplog.text
    assert "error" in caplog.text
    
    # Check log records
    assert len(caplog.records) == 3
    assert caplog.records[0].levelname == "INFO"
    assert caplog.records[1].levelname == "WARNING"
    assert caplog.records[2].levelname == "ERROR"
    
    # Check specific messages
    assert "This is an info message" in caplog.messages
```

### monkeypatch

```python
import os

def test_environment_variables(monkeypatch):
    """Test environment variable mocking."""
    # Set environment variable
    monkeypatch.setenv("API_KEY", "test_key")
    assert os.environ["API_KEY"] == "test_key"
    
    # Delete environment variable
    monkeypatch.delenv("PATH", raising=False)
    
    # Mock attribute
    monkeypatch.setattr("sys.platform", "test_platform")
    import sys
    assert sys.platform == "test_platform"

def test_system_path(monkeypatch):
    """Test sys.path modification."""
    import sys
    original_path = sys.path.copy()
    
    # Add to sys.path
    monkeypatch.syspath_prepend("/custom/path")
    assert "/custom/path" in sys.path
    
    # sys.path is automatically restored after test
```

## Test Organization and Best Practices

### Directory Structure

```
project/
├── src/
│   └── myapp/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── user.py
│       ├── services/
│       │   ├── __init__.py
│       │   └── user_service.py
│       └── controllers/
│           ├── __init__.py
│           └── user_controller.py
├── tests/
│   ├── conftest.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── test_user.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── test_user_service.py
│   └── controllers/
│       ├── __init__.py
│       └── test_user_controller.py
└── pytest.ini
```

### pytest.ini Configuration

```ini
[tool:pytest]
# Test paths
testpaths = tests

# Test discovery patterns
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

# Minimum version
minversion = 6.0

# Add options
addopts = 
    -ra
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80

# Markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    smoke: marks tests as smoke tests

# Ignore certain warnings
filterwarnings =
    ignore::UserWarning
    ignore::DeprecationWarning
```

### Test Organization Patterns

```python
# tests/conftest.py
import pytest
from myapp import create_app
from myapp.database import init_db

@pytest.fixture(scope="session")
def app():
    """Create application for testing."""
    app = create_app(testing=True)
    with app.app_context():
        init_db()
        yield app

@pytest.fixture(scope="function")
def client(app):
    """Create test client."""
    return app.test_client()

# tests/models/test_user.py
class TestUserModel:
    """Test User model functionality."""
    
    def test_user_creation(self):
        """Test user creation."""
        user = User(email="test@example.com", name="Test User")
        assert user.email == "test@example.com"
        assert user.name == "Test User"
    
    def test_user_validation(self):
        """Test user validation."""
        with pytest.raises(ValueError):
            User(email="invalid-email", name="Test User")

# tests/services/test_user_service.py
class TestUserService:
    """Test UserService functionality."""
    
    @pytest.fixture
    def user_service(self, mocker):
        """Create UserService with mocked dependencies."""
        mock_repository = mocker.Mock()
        return UserService(mock_repository)
    
    def test_create_user_success(self, user_service, mocker):
        """Test successful user creation."""
        user_data = {"email": "test@example.com", "name": "Test User"}
        expected_user = User(**user_data)
        
        user_service.repository.save.return_value = expected_user
        
        result = user_service.create_user(user_data)
        
        assert result == expected_user
        user_service.repository.save.assert_called_once()
```

### Test Naming Conventions

```python
# Good test names are descriptive and follow patterns

def test_user_creation_with_valid_data_should_return_user():
    """Test that user creation with valid data returns User object."""
    pass

def test_user_creation_with_invalid_email_should_raise_validation_error():
    """Test that invalid email raises ValidationError."""
    pass

def test_user_service_create_user_calls_repository_save():
    """Test that UserService.create_user calls repository.save."""
    pass

# Use classes to group related tests
class TestUserAuthentication:
    """Test user authentication functionality."""
    
    def test_login_with_valid_credentials_should_return_token(self):
        pass
    
    def test_login_with_invalid_credentials_should_raise_error(self):
        pass
    
    def test_logout_should_invalidate_token(self):
        pass
```

### Markers and Test Selection

```python
import pytest

@pytest.mark.slow
def test_large_dataset_processing():
    """Test that takes a long time to run."""
    pass

@pytest.mark.integration
def test_database_integration():
    """Test that requires database."""
    pass

@pytest.mark.parametrize("browser", ["chrome", "firefox"])
@pytest.mark.smoke
def test_login_flow(browser):
    """Smoke test for login flow."""
    pass

# Custom markers
@pytest.mark.external_api
def test_external_api_call():
    """Test that calls external API."""
    pass

# Running specific markers
# pytest -m "not slow"  # Skip slow tests
# pytest -m "smoke"     # Run only smoke tests
# pytest -m "integration and not slow"  # Run integration but not slow tests
```

## Common Patterns and Advanced Usage

### Factory Fixtures

```python
@pytest.fixture
def user_factory():
    """Factory for creating test users."""
    def _create_user(**kwargs):
        defaults = {
            "email": "test@example.com",
            "name": "Test User",
            "age": 25,
            "active": True
        }
        defaults.update(kwargs)
        return User(**defaults)
    return _create_user

def test_user_with_factory(user_factory):
    """Test using user factory."""
    # Create default user
    user1 = user_factory()
    assert user1.email == "test@example.com"
    
    # Create user with custom data
    user2 = user_factory(email="custom@example.com", age=30)
    assert user2.email == "custom@example.com"
    assert user2.age == 30
```

### Dependency Injection Pattern

```python
@pytest.fixture
def mock_database(mocker):
    """Mock database."""
    return mocker.Mock(spec=Database)

@pytest.fixture
def mock_email_service(mocker):
    """Mock email service."""
    return mocker.Mock(spec=EmailService)

@pytest.fixture
def user_service(mock_database, mock_email_service):
    """UserService with injected dependencies."""
    return UserService(
        database=mock_database,
        email_service=mock_email_service
    )

def test_user_registration(user_service, mock_database, mock_email_service):
    """Test user registration with mocked dependencies."""
    user_data = {"email": "test@example.com", "name": "Test User"}
    
    # Configure mocks
    mock_database.save.return_value = User(**user_data)
    mock_email_service.send_welcome_email.return_value = True
    
    # Test the service
    result = user_service.register_user(user_data)
    
    # Verify behavior
    assert result.email == "test@example.com"
    mock_database.save.assert_called_once()
    mock_email_service.send_welcome_email.assert_called_once_with(result)
```

### Pytest Plugins and Hooks

```python
# conftest.py - Custom plugin hooks

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "external: mark test as requiring external resources"
    )

def pytest_collection_modifyitems(config, items):
    """Modify collected test items."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(request):
    """Setup test environment."""
    # Setup code
    yield
    # Teardown code
```

### Error Handling and Debugging

```python
def test_with_custom_assertion_message():
    """Test with custom assertion messages."""
    user = create_user("test@example.com")
    
    assert user.is_active, f"User {user.email} should be active"
    assert user.email == "test@example.com", \
        f"Expected email 'test@example.com', got '{user.email}'"

def test_with_debugging_info(capsys):
    """Test that shows debugging techniques."""
    # Use print for debugging (captured by capsys)
    print(f"Testing user creation")
    
    user = create_user("test@example.com")
    print(f"Created user: {user}")
    
    # Assert with debugging
    assert user.email == "test@example.com"
    
    # Check captured output if needed
    captured = capsys.readouterr()
    assert "Testing user creation" in captured.out

@pytest.mark.xfail(reason="Known issue with external service")
def test_external_service_integration():
    """Test that is expected to fail."""
    # This test is marked as expected to fail
    result = call_external_service()
    assert result.success

@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    """Test for feature not yet implemented."""
    pass
```

## Migration from unittest

### Converting unittest to pytest

```python
# Before (unittest)
import unittest
from unittest.mock import patch, Mock

class TestUserService(unittest.TestCase):
    
    def setUp(self):
        self.user_service = UserService()
    
    def tearDown(self):
        # Cleanup code
        pass
    
    @patch('myapp.services.database')
    def test_create_user(self, mock_database):
        mock_database.save.return_value = Mock(id=1)
        
        result = self.user_service.create_user("test@example.com")
        
        self.assertEqual(result.id, 1)
        mock_database.save.assert_called_once()

# After (pytest)
import pytest

class TestUserService:
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.user_service = UserService()
        yield
        # Cleanup code (if needed)
    
    def test_create_user(self, mocker):
        mock_database = mocker.patch('myapp.services.database')
        mock_database.save.return_value = mocker.Mock(id=1)
        
        result = self.user_service.create_user("test@example.com")
        
        assert result.id == 1
        mock_database.save.assert_called_once()

# Even better (pytest style)
@pytest.fixture
def user_service():
    return UserService()

def test_create_user(user_service, mocker):
    mock_database = mocker.patch('myapp.services.database')
    mock_database.save.return_value = mocker.Mock(id=1)
    
    result = user_service.create_user("test@example.com")
    
    assert result.id == 1
    mock_database.save.assert_called_once()
```

### Assertion Migration

```python
# unittest assertions -> pytest assertions

# self.assertEqual(a, b) -> assert a == b
# self.assertNotEqual(a, b) -> assert a != b
# self.assertTrue(x) -> assert x
# self.assertFalse(x) -> assert not x
# self.assertIs(a, b) -> assert a is b
# self.assertIsNot(a, b) -> assert a is not b
# self.assertIsNone(x) -> assert x is None
# self.assertIsNotNone(x) -> assert x is not None
# self.assertIn(a, b) -> assert a in b
# self.assertNotIn(a, b) -> assert a not in b
# self.assertIsInstance(a, b) -> assert isinstance(a, b)
# self.assertRaises(Exception) -> pytest.raises(Exception)

# Example conversion
def test_assertions():
    user = User("test@example.com")
    
    # unittest style
    # self.assertEqual(user.email, "test@example.com")
    # self.assertTrue(user.is_valid())
    # self.assertIsInstance(user, User)
    # with self.assertRaises(ValueError):
    #     User("invalid-email")
    
    # pytest style
    assert user.email == "test@example.com"
    assert user.is_valid()
    assert isinstance(user, User)
    with pytest.raises(ValueError):
        User("invalid-email")
```

## Quick Reference

### pytest Command Line

```bash
# Basic usage
pytest                          # Run all tests
pytest test_file.py            # Run specific file
pytest test_file.py::test_func # Run specific test
pytest -k "test_user"          # Run tests matching pattern
pytest -m "not slow"           # Run tests not marked as slow
pytest -x                      # Stop after first failure
pytest --maxfail=2             # Stop after 2 failures
pytest -v                      # Verbose output
pytest -s                      # Don't capture output
pytest --lf                    # Run last failed tests
pytest --ff                    # Run failures first
pytest --collect-only          # Show what tests would be collected

# Coverage
pytest --cov=myapp             # Run with coverage
pytest --cov=myapp --cov-report=html  # HTML coverage report

# Parallel execution
pytest -n auto                 # Auto-detect CPU count
pytest -n 4                    # Use 4 processes

# Debugging
pytest --pdb                   # Drop into debugger on failure
pytest --pdbcls=IPython.terminal.debugger:Pdb  # Use IPython debugger
```

### Fixture Quick Reference

```python
# Fixture scopes
@pytest.fixture(scope="function")  # Default, runs per test
@pytest.fixture(scope="class")     # Runs per test class
@pytest.fixture(scope="module")    # Runs per test module
@pytest.fixture(scope="package")   # Runs per test package
@pytest.fixture(scope="session")   # Runs once per session

# Fixture features
@pytest.fixture(autouse=True)      # Automatically used
@pytest.fixture(params=[1, 2, 3])  # Parametrized fixture

# Built-in fixtures
def test_example(tmp_path, capsys, caplog, mocker, monkeypatch):
    pass
```

### Parametrization Quick Reference

```python
# Basic parametrization
@pytest.mark.parametrize("input,expected", [(1, 2), (2, 4), (3, 6)])
def test_double(input, expected):
    assert double(input) == expected

# With IDs
@pytest.mark.parametrize(
    "input,expected",
    [
        pytest.param(1, 2, id="positive"),
        pytest.param(0, 0, id="zero"),
        pytest.param(-1, -2, id="negative"),
    ]
)

# Multiple parameters
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [3, 4])  # Creates cartesian product

# Indirect parametrization
@pytest.mark.parametrize("fixture_name", [value1, value2], indirect=True)
```

### Mocking Quick Reference

```python
# mocker methods
mocker.Mock()                    # Create mock
mocker.MagicMock()              # Create magic mock
mocker.patch('module.function') # Patch function
mocker.patch.object(obj, 'method')  # Patch object method
mocker.spy(obj, 'method')       # Create spy
mocker.PropertyMock()           # Mock property

# Mock configuration
mock.return_value = "result"
mock.side_effect = Exception("error")
mock.side_effect = [1, 2, 3]

# Mock assertions
mock.assert_called()
mock.assert_called_once()
mock.assert_called_with(*args, **kwargs)
mock.assert_called_once_with(*args, **kwargs)
mock.assert_not_called()
```

### Markers Quick Reference

```python
# Built-in markers
@pytest.mark.skip(reason="Not implemented")
@pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
@pytest.mark.xfail(reason="Known issue")
@pytest.mark.parametrize("arg", [1, 2, 3])

# Custom markers (define in pytest.ini)
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.smoke

# Running with markers
# pytest -m "slow"              # Run slow tests only
# pytest -m "not slow"          # Skip slow tests
# pytest -m "slow and integration"  # Multiple conditions
```

This comprehensive guide should help you write efficient, maintainable tests with pytest and pytest-mock. Remember: pytest's strength lies in its simplicity and flexibility, allowing you to write clear, readable tests with minimal boilerplate.