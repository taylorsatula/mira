# pytest Distilled Reference Guide

## 🎯 CRITICAL TESTING PRINCIPLE

**Write tests that catch real bugs and vulnerabilities, not tests that merely achieve coverage metrics.**

Every test should verify actual behavior that matters for correctness, security, or reliability. A test that only exercises code without meaningful assertions is worse than no test - it provides false confidence.

Good tests answer: "What could go wrong here, and would my test catch it?"

---

## pytest-mock: Key Differences from unittest.mock

```python
# pytest-mock provides AUTOMATIC cleanup
def test_with_mocker(mocker):
    mock_get = mocker.patch('requests.get')
    mock_get.return_value.json.return_value = {"data": "test"}
    # Cleanup happens automatically after test

# unittest.mock requires manual cleanup
def test_with_unittest_mock():
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {"data": "test"}
        # Cleanup happens at end of context manager only
```

### mocker Advanced Methods

```python
def test_advanced_mocking(mocker):
    # Create spy (partial mock - real method still called)
    spy = mocker.spy(MyClass, 'method')
    
    # Mock property
    prop_mock = mocker.PropertyMock(return_value="mocked_value")
    type(my_object).my_property = prop_mock
    
    # Mock multiple things at once
    mocker.patch.multiple(
        'module',
        function1=mocker.Mock(return_value="mock1"),
        function2=mocker.Mock(return_value="mock2")
    )
    
    # Async mock
    async_mock = mocker.AsyncMock()
    async_mock.return_value = {"data": "async_result"}
```

---

## Advanced Fixture Patterns

### Fixture Scopes (Beyond Function)

```python
@pytest.fixture(scope="package")    # Runs once per test package
def package_fixture():
    return expensive_setup()

@pytest.fixture(scope="session")    # Runs once per test session
def session_fixture():
    return global_setup()
```

### Factory Fixtures

```python
@pytest.fixture
def user_factory():
    """Factory for creating test users with customization."""
    def _create_user(**kwargs):
        defaults = {"email": "test@example.com", "name": "Test User", "active": True}
        defaults.update(kwargs)
        return User(**defaults)
    return _create_user

def test_user_variations(user_factory):
    admin = user_factory(role="admin", email="admin@example.com")
    inactive = user_factory(active=False)
```

### Dependency Injection with Fixtures

```python
@pytest.fixture
def mock_database(mocker):
    return mocker.Mock(spec=Database)

@pytest.fixture
def mock_email_service(mocker):
    return mocker.Mock(spec=EmailService)

@pytest.fixture
def user_service(mock_database, mock_email_service):
    """Service with all dependencies injected and mocked."""
    return UserService(database=mock_database, email_service=mock_email_service)

def test_user_registration(user_service, mock_database, mock_email_service):
    # All dependencies are pre-mocked and injectable
    result = user_service.register_user({"email": "test@example.com"})
    mock_database.save.assert_called_once()
    mock_email_service.send_welcome_email.assert_called_once()
```

---

## Advanced Parametrization

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

@pytest.mark.parametrize(
    "database_type", 
    ["sqlite", "postgres"], 
    indirect=True  # Pass through fixture instead of direct injection
)
def test_database_operations(database_type):
    result = database_type.query("SELECT 1")
    assert result is not None
```

### Dynamic Test Generation

```python
def pytest_generate_tests(metafunc):
    """Generate tests dynamically based on available data."""
    if "user_role" in metafunc.fixturenames:
        # Could read from config, database, etc.
        roles = get_available_roles_from_config()
        metafunc.parametrize("user_role", roles)

def test_user_permissions(user_role):
    """This test runs for each role found in config."""
    permissions = get_permissions_for_role(user_role)
    assert len(permissions) > 0
```

---

## Async Testing Essentials

```python
pytestmark = pytest.mark.asyncio  # Mark entire module as async

@pytest.fixture
async def async_client():
    """Async fixture with proper cleanup."""
    async with aiohttp.ClientSession() as session:
        yield session

@pytest.mark.asyncio
async def test_async_with_mock(mocker):
    """Async testing with mocks."""
    mock_response = mocker.AsyncMock()
    mock_response.json.return_value = {"data": "test_value"}
    
    # Mock async context manager
    mock_session = mocker.AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response
    
    mocker.patch('aiohttp.ClientSession', return_value=mock_session)
    
    result = await fetch_data("https://api.example.com")
    assert result["data"] == "test_value"
```

---

## Lesser-Known Built-in Fixtures

### tmp_path_factory

```python
def test_shared_tmp_dir(tmp_path_factory):
    """Create shared temp directories across tests."""
    shared_dir = tmp_path_factory.mktemp("shared_data")
    config_file = shared_dir / "config.json"
    config_file.write_text('{"setting": "value"}')
    # This directory persists across multiple tests in same session
```

### caplog with Levels

```python
def test_specific_log_levels(caplog):
    """Test log capture with specific levels."""
    with caplog.at_level(logging.WARNING):  # Only capture WARNING and above
        function_that_logs()
    
    # Check specific log records
    assert len([r for r in caplog.records if r.levelname == "ERROR"]) == 1
    assert "critical error" in caplog.text
```

---

## pytest Configuration Hooks

### Custom Test Collection

```python
# conftest.py
def pytest_collection_modifyitems(config, items):
    """Modify tests during collection."""
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

def pytest_addoption(parser):
    """Add custom CLI options."""
    parser.addoption("--runslow", action="store_true", help="run slow tests")
```

### Environment Setup

```python
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Automatically setup/teardown test environment."""
    # Setup
    configure_test_database()
    seed_test_data()
    yield
    # Teardown
    cleanup_test_database()
```

---

## Advanced Assertion Patterns

### Custom Failure Messages

```python
def test_with_debugging_context():
    """Provide context in assertion failures."""
    user = create_user("test@example.com")
    
    assert user.is_active, f"User {user.email} should be active but got {user.status}"
    assert user.email == "test@example.com", \
        f"Expected 'test@example.com', got '{user.email}'"
```

### Expected Failures

```python
@pytest.mark.xfail(reason="Known issue with external service")
def test_external_integration():
    """Test marked as expected to fail."""
    result = call_unreliable_external_service()
    assert result.success  # Won't fail the test suite when it fails

@pytest.mark.xfail(strict=True)  # Must fail, or test suite fails
def test_should_definitely_fail():
    pass
```

---

## Quick Reference

### Command Line Power Users

```bash
pytest --lf                    # Run last failed tests only
pytest --ff                    # Run failures first, then rest
pytest --maxfail=3             # Stop after 3 failures
pytest -x --pdb               # Drop into debugger on first failure
pytest --collect-only          # Show what tests would run (dry run)
```

### Marker Combinations

```bash
pytest -m "slow and not integration"     # Complex marker logic
pytest -m "smoke or critical"            # Run smoke OR critical tests
```

### Coverage with Failure Threshold

```bash
pytest --cov=src --cov-fail-under=80    # Fail if coverage below 80%
```