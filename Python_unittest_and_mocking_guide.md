# Python unittest and Mocking Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [unittest Basics](#unittest-basics)
3. [unittest.mock Fundamentals](#unittestmock-fundamentals)
4. [Advanced Mocking Patterns](#advanced-mocking-patterns)
5. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
6. [Real-World Testing Scenarios](#real-world-testing-scenarios)
7. [Best Practices](#best-practices)
8. [Quick Reference](#quick-reference)

## Introduction

This guide provides comprehensive coverage of Python's unittest framework and mocking capabilities, based on 2024 best practices. It's designed to help you write robust, maintainable tests that avoid common pitfalls.

### Why unittest?
- **Built-in**: Part of Python's standard library, no additional dependencies
- **Mature**: Battle-tested framework with consistent behavior
- **OOP-based**: Familiar structure for developers coming from JUnit/xUnit frameworks
- **Comprehensive**: Built-in assertion methods and test discovery

### When to Use unittest vs Alternatives
- **Use unittest** for: Standard library preference, OOP-style tests, legacy codebases
- **Consider pytest** for: More concise syntax, better fixtures, parametrized tests

## unittest Basics

### Test Structure

```python
import unittest

class TestCalculator(unittest.TestCase):
    
    def setUp(self):
        """Called before each test method"""
        self.calculator = Calculator()
    
    def tearDown(self):
        """Called after each test method"""
        # Cleanup code here
        pass
    
    def test_addition(self):
        """Test addition functionality"""
        result = self.calculator.add(2, 3)
        self.assertEqual(result, 5)
    
    def test_division_by_zero(self):
        """Test exception handling"""
        with self.assertRaises(ZeroDivisionError):
            self.calculator.divide(10, 0)

if __name__ == '__main__':
    unittest.main()
```

### Essential Assertion Methods

```python
# Equality assertions
self.assertEqual(a, b)           # a == b
self.assertNotEqual(a, b)        # a != b
self.assertIs(a, b)              # a is b
self.assertIsNot(a, b)           # a is not b

# Truth assertions
self.assertTrue(x)               # bool(x) is True
self.assertFalse(x)              # bool(x) is False

# Membership assertions
self.assertIn(a, b)              # a in b
self.assertNotIn(a, b)           # a not in b

# Exception assertions
self.assertRaises(ValueError, func, *args)
with self.assertRaises(ValueError):
    func(*args)

# Approximate equality (for floats)
self.assertAlmostEqual(a, b, places=7)

# Container assertions
self.assertListEqual(list1, list2)
self.assertDictEqual(dict1, dict2)
self.assertSetEqual(set1, set2)

# String assertions
self.assertRegex(text, regex)
self.assertMultiLineEqual(first, second)
```

### Test Discovery and Running

```bash
# Run all tests in current directory
python -m unittest discover

# Run specific test module
python -m unittest test_module

# Run specific test class
python -m unittest test_module.TestClass

# Run specific test method
python -m unittest test_module.TestClass.test_method

# Verbose output
python -m unittest -v

# Stop on first failure
python -m unittest -f
```

## unittest.mock Fundamentals

### Basic Mock Objects

```python
from unittest.mock import Mock, MagicMock

# Basic Mock
mock = Mock()
mock.method.return_value = 42
result = mock.method()  # Returns 42

# MagicMock (includes magic methods)
magic_mock = MagicMock()
len(magic_mock)  # Works because __len__ is implemented
str(magic_mock)  # Works because __str__ is implemented

# Configure mock behavior
mock.configure_mock(return_value=10, side_effect=ValueError)
```

### Mock vs MagicMock

| Feature | Mock | MagicMock |
|---------|------|-----------|
| Basic methods | ✓ | ✓ |
| Magic methods (`__len__`, `__str__`, etc.) | ✗ | ✓ |
| Performance | Faster | Slightly slower |
| Use case | Simple mocking | Complex object mocking |

### Setting Return Values and Side Effects

```python
# Return value
mock.method.return_value = "fixed_result"
mock.method()  # Returns "fixed_result"

# Side effect with function
def custom_side_effect(*args, **kwargs):
    if args[0] == 'error':
        raise ValueError("Invalid input")
    return f"Processed: {args[0]}"

mock.method.side_effect = custom_side_effect

# Side effect with list (sequential returns)
mock.method.side_effect = [1, 2, 3, StopIteration]
mock.method()  # Returns 1
mock.method()  # Returns 2
mock.method()  # Returns 3
mock.method()  # Raises StopIteration

# Side effect with exception
mock.method.side_effect = ValueError("Something went wrong")
mock.method()  # Raises ValueError
```

### The patch() Decorator and Context Manager

#### As Decorator

```python
from unittest.mock import patch

class TestEmailService(unittest.TestCase):
    
    @patch('requests.post')
    def test_send_email_success(self, mock_post):
        mock_post.return_value.status_code = 200
        
        service = EmailService()
        result = service.send_email("test@example.com", "Hello")
        
        self.assertTrue(result)
        mock_post.assert_called_once()

    @patch('smtplib.SMTP')
    @patch('email.mime.text.MIMEText')
    def test_multiple_patches(self, mock_mime, mock_smtp):
        # Multiple patches are applied bottom-up
        # mock_smtp is the first parameter, mock_mime is the second
        pass
```

#### As Context Manager

```python
def test_send_email_with_context_manager(self):
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        
        service = EmailService()
        result = service.send_email("test@example.com", "Hello")
        
        self.assertTrue(result)
        mock_post.assert_called_once()
```

#### patch.object for Specific Objects

```python
class TestUserService(unittest.TestCase):
    
    def test_create_user(self):
        with patch.object(UserService, 'validate_email', return_value=True):
            service = UserService()
            result = service.create_user("invalid-email")
            # validate_email is mocked to return True
```

### Where to Patch: Critical Concept

**Rule: Patch where the object is used, not where it's defined.**

```python
# mymodule.py
from datetime import datetime

def get_current_time():
    return datetime.now()

# Wrong - patches datetime in datetime module
@patch('datetime.datetime')
def test_wrong(self, mock_datetime):
    pass

# Correct - patches datetime in mymodule
@patch('mymodule.datetime')
def test_correct(self, mock_datetime):
    mock_datetime.now.return_value = datetime(2024, 1, 1)
    result = get_current_time()
    # Now get_current_time() uses the mocked datetime
```

## Advanced Mocking Patterns

### AsyncMock for Async Functions

```python
import unittest
from unittest.mock import AsyncMock, patch

class TestAsyncService(unittest.IsolatedAsyncioTestCase):
    
    async def test_async_operation(self):
        # Basic AsyncMock usage
        mock_async_func = AsyncMock(return_value="async_result")
        result = await mock_async_func()
        self.assertEqual(result, "async_result")
        
        # Verify it was awaited
        mock_async_func.assert_awaited_once()
    
    @patch('aiohttp.ClientSession.get')
    async def test_async_http_call(self, mock_get):
        # Mock async context manager
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"data": "test"})
        mock_get.return_value.__aenter__.return_value = mock_response
        
        service = AsyncHttpService()
        result = await service.fetch_data("http://example.com")
        
        self.assertEqual(result["data"], "test")
```

### Spec and Autospec

```python
# Using spec to match interface
class RealService:
    def method_a(self, param1):
        pass
    
    def method_b(self, param1, param2):
        pass

# Mock with spec
mock_service = Mock(spec=RealService)
mock_service.method_a("param")  # OK
mock_service.method_c("param")  # Raises AttributeError

# Autospec for automatic signature checking
with patch('mymodule.RealService', autospec=True) as mock_service:
    mock_service.return_value.method_a.return_value = "result"
    
    service = mymodule.RealService()
    # This will work
    result = service.method_a("valid_param")
    
    # This will raise TypeError due to wrong signature
    service.method_a("param1", "param2")
```

### Mock Property and Attributes

```python
class TestPropertyMocking(unittest.TestCase):
    
    def test_property_mock(self):
        mock_obj = Mock()
        
        # Mock property with PropertyMock
        with patch.object(type(mock_obj), 'my_property', 
                         new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = "mocked_value"
            result = mock_obj.my_property
            self.assertEqual(result, "mocked_value")
    
    def test_attribute_mock(self):
        with patch.object(MyClass, 'class_attribute', 'new_value'):
            obj = MyClass()
            self.assertEqual(obj.class_attribute, 'new_value')
```

### Mocking File Operations

```python
from unittest.mock import mock_open

class TestFileOperations(unittest.TestCase):
    
    @patch("builtins.open", new_callable=mock_open, read_data="file content")
    def test_read_file(self, mock_file):
        result = read_file_function("dummy_path")
        
        mock_file.assert_called_once_with("dummy_path", "r")
        self.assertEqual(result, "file content")
    
    def test_write_file(self):
        with patch("builtins.open", mock_open()) as mock_file:
            write_file_function("dummy_path", "test content")
            
            mock_file.assert_called_once_with("dummy_path", "w")
            mock_file().write.assert_called_once_with("test content")
```

### Mocking Multiple Files

```python
def test_multiple_files(self):
    files = {
        "file1.txt": "content1",
        "file2.txt": "content2",
        "default.txt": "default_content"
    }
    
    def open_side_effect(filename, mode='r'):
        content = files.get(filename, files["default.txt"])
        return mock_open(read_data=content).return_value
    
    with patch("builtins.open", side_effect=open_side_effect):
        # Test reading different files
        content1 = read_file("file1.txt")
        content2 = read_file("file2.txt")
        content_default = read_file("unknown.txt")
        
        self.assertEqual(content1, "content1")
        self.assertEqual(content2, "content2")
        self.assertEqual(content_default, "default_content")
```

### Mocking Datetime and Time

```python
from datetime import datetime, date
from unittest.mock import patch

class TestDatetime(unittest.TestCase):
    
    def test_mock_datetime_now(self):
        with patch('mymodule.datetime') as mock_datetime:
            # Mock datetime.now() while preserving constructor
            mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            result = get_current_timestamp()  # Function using datetime.now()
            expected = datetime(2024, 1, 1, 12, 0, 0)
            self.assertEqual(result, expected)
    
    def test_mock_date_today(self):
        with patch('mymodule.date') as mock_date:
            mock_date.today.return_value = date(2024, 1, 1)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            
            result = get_today_date()
            self.assertEqual(result, date(2024, 1, 1))
```

### Mocking Environment Variables

```python
import os
from unittest.mock import patch

class TestEnvironmentVariables(unittest.TestCase):
    
    @patch.dict(os.environ, {'API_KEY': 'test_key', 'DEBUG': 'True'})
    def test_with_env_vars(self):
        # Function that reads environment variables
        result = get_api_configuration()
        
        self.assertEqual(result['api_key'], 'test_key')
        self.assertTrue(result['debug'])
    
    def test_missing_env_var(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(KeyError):
                get_required_env_var('MISSING_VAR')
```

## Common Pitfalls and Solutions

### 1. Assertion Method Confusion

```python
# PITFALL: Using wrong assertion method
mock.method()
mock.method()

# WRONG: Only checks the last call
mock.assert_called_with()

# CORRECT: Check if called exactly once
mock.assert_called_once()

# CORRECT: Check all calls
expected_calls = [call(), call()]
mock.assert_has_calls(expected_calls)

# CORRECT: Check call count and calls
self.assertEqual(mock.call_count, 2)
```

### 2. Mutable Arguments Problem

```python
# PITFALL: Mutable arguments are stored by reference
def test_mutable_args_wrong(self):
    mock = Mock()
    data = {'key': 'value'}
    
    mock.method(data)
    data['key'] = 'modified'  # This modifies the stored call args!
    
    # This assertion might fail unexpectedly
    mock.assert_called_with({'key': 'value'})

# SOLUTION: Use copy or check immediately
def test_mutable_args_correct(self):
    mock = Mock()
    data = {'key': 'value'}
    
    mock.method(data)
    # Check immediately before modification
    mock.assert_called_with({'key': 'value'})
```

### 3. Patch Location Errors

```python
# mymodule.py
from requests import get

def fetch_data():
    return get('http://api.example.com')

# WRONG: Patches requests module globally
@patch('requests.get')
def test_wrong_patch(self, mock_get):
    pass

# CORRECT: Patches where it's imported
@patch('mymodule.get')
def test_correct_patch(self, mock_get):
    pass
```

### 4. Autospec Misuse

```python
# PITFALL: Not using autospec allows invalid calls
with patch('mymodule.SomeClass') as mock_class:
    mock_instance = mock_class.return_value
    # This might pass even if real_method doesn't exist
    mock_instance.nonexistent_method.return_value = "result"

# SOLUTION: Use autospec
with patch('mymodule.SomeClass', autospec=True) as mock_class:
    mock_instance = mock_class.return_value
    # This will raise AttributeError if method doesn't exist
    mock_instance.real_method.return_value = "result"
```

### 5. AsyncMock vs Mock Confusion

```python
# PITFALL: Using regular Mock for async functions
async def test_async_wrong(self):
    with patch('mymodule.async_function') as mock_func:
        mock_func.return_value = "result"
        # This won't work as expected - returns a Mock, not a coroutine
        result = await mymodule.async_function()

# SOLUTION: Use AsyncMock
async def test_async_correct(self):
    with patch('mymodule.async_function', new_callable=AsyncMock) as mock_func:
        mock_func.return_value = "result"
        result = await mymodule.async_function()
        self.assertEqual(result, "result")
```

### 6. Side Effect vs Return Value

```python
# PITFALL: Confusing side_effect and return_value
mock = Mock()
mock.return_value = "value"
mock.side_effect = ["different_value"]  # This overrides return_value!

result = mock()  # Returns "different_value", not "value"

# SOLUTION: Understand precedence
# side_effect takes precedence over return_value
# Use only one of them for clarity
```

## Real-World Testing Scenarios

### Database Testing

```python
import unittest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class TestUserRepository(unittest.TestCase):
    
    def setUp(self):
        # Use in-memory SQLite for fast tests
        self.engine = create_engine('sqlite:///:memory:')
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        
        # Create tables
        Base.metadata.create_all(self.engine)
    
    def tearDown(self):
        self.session.close()
    
    def test_create_user_integration(self):
        """Integration test with real database"""
        repo = UserRepository(self.session)
        user = repo.create_user("test@example.com", "John Doe")
        
        self.assertIsNotNone(user.id)
        self.assertEqual(user.email, "test@example.com")
    
    @patch('myapp.models.User')
    def test_create_user_mocked(self, mock_user_class):
        """Unit test with mocked database"""
        mock_user_instance = MagicMock()
        mock_user_class.return_value = mock_user_instance
        
        mock_session = MagicMock()
        repo = UserRepository(mock_session)
        
        result = repo.create_user("test@example.com", "John Doe")
        
        mock_session.add.assert_called_once_with(mock_user_instance)
        mock_session.commit.assert_called_once()
        self.assertEqual(result, mock_user_instance)
```

### API Testing

```python
class TestAPIClient(unittest.TestCase):
    
    @patch('requests.Session.request')
    def test_api_call_success(self, mock_request):
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "success"}
        mock_request.return_value = mock_response
        
        client = APIClient("http://api.example.com")
        result = client.get_data("endpoint")
        
        self.assertEqual(result["data"], "success")
        mock_request.assert_called_once_with(
            "GET", 
            "http://api.example.com/endpoint",
            headers={'Content-Type': 'application/json'}
        )
    
    @patch('requests.Session.request')
    def test_api_call_failure(self, mock_request):
        # Mock API error
        mock_request.side_effect = requests.RequestException("Network error")
        
        client = APIClient("http://api.example.com")
        
        with self.assertRaises(APIException):
            client.get_data("endpoint")
    
    @patch('requests.Session.request')
    def test_api_call_timeout(self, mock_request):
        # Mock timeout
        mock_request.side_effect = requests.Timeout()
        
        client = APIClient("http://api.example.com")
        
        with self.assertRaises(APITimeoutException):
            client.get_data("endpoint")
```

### Complex Service Testing

```python
class TestEmailService(unittest.TestCase):
    
    def setUp(self):
        self.email_service = EmailService()
    
    @patch('smtplib.SMTP')
    @patch('myapp.email_service.MIMEText')
    @patch('myapp.email_service.datetime')
    def test_send_email_comprehensive(self, mock_datetime, mock_mime, mock_smtp):
        # Setup mocks
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        mock_message = MagicMock()
        mock_mime.return_value = mock_message
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Test the service
        result = self.email_service.send_email(
            to="recipient@example.com",
            subject="Test Subject",
            body="Test Body"
        )
        
        # Verify behavior
        self.assertTrue(result)
        
        # Verify SMTP connection
        mock_smtp.assert_called_once_with('smtp.gmail.com', 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        
        # Verify message construction
        mock_mime.assert_called_once_with("Test Body")
        self.assertEqual(mock_message['Subject'], "Test Subject")
        self.assertEqual(mock_message['From'], self.email_service.from_email)
        self.assertEqual(mock_message['To'], "recipient@example.com")
        
        # Verify email sending
        mock_server.send_message.assert_called_once_with(mock_message)
```

### Configuration and Settings Testing

```python
class TestConfigurationManager(unittest.TestCase):
    
    @patch.dict(os.environ, {
        'DATABASE_URL': 'postgresql://test:test@localhost/test',
        'DEBUG': 'True',
        'SECRET_KEY': 'test-secret-key'
    })
    def test_load_from_environment(self):
        config = ConfigurationManager()
        config.load_from_environment()
        
        self.assertEqual(config.database_url, 'postgresql://test:test@localhost/test')
        self.assertTrue(config.debug)
        self.assertEqual(config.secret_key, 'test-secret-key')
    
    @patch('builtins.open', mock_open(read_data='''
        {
            "database_url": "sqlite:///test.db",
            "debug": false,
            "secret_key": "file-secret"
        }
    '''))
    def test_load_from_file(self):
        config = ConfigurationManager()
        config.load_from_file('config.json')
        
        self.assertEqual(config.database_url, 'sqlite:///test.db')
        self.assertFalse(config.debug)
        self.assertEqual(config.secret_key, 'file-secret')
```

## Best Practices

### 1. Test Organization

```python
class TestUserService(unittest.TestCase):
    """Test the UserService class functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.user_service = UserService()
        self.valid_user_data = {
            'email': 'test@example.com',
            'name': 'Test User',
            'age': 25
        }
    
    def test_create_user_with_valid_data_should_return_user_object(self):
        """Test that creating a user with valid data returns a User object."""
        # Given: Valid user data (in setUp)
        
        # When: Creating a user
        user = self.user_service.create_user(**self.valid_user_data)
        
        # Then: Should return a User object with correct attributes
        self.assertIsInstance(user, User)
        self.assertEqual(user.email, self.valid_user_data['email'])
        self.assertEqual(user.name, self.valid_user_data['name'])
        self.assertEqual(user.age, self.valid_user_data['age'])
```

### 2. Mock Management

```python
class TestWithProperMockManagement(unittest.TestCase):
    
    def setUp(self):
        # Create reusable mocks
        self.mock_database = Mock(spec=Database)
        self.mock_email_service = Mock(spec=EmailService)
        
        # Configure common behavior
        self.mock_database.save.return_value = True
        self.mock_email_service.send.return_value = True
    
    def test_user_registration_success(self):
        with patch('myapp.services.Database', return_value=self.mock_database), \
             patch('myapp.services.EmailService', return_value=self.mock_email_service):
            
            service = UserRegistrationService()
            result = service.register_user("test@example.com", "password")
            
            self.assertTrue(result)
            # Verify interactions
            self.mock_database.save.assert_called_once()
            self.mock_email_service.send.assert_called_once()
```

### 3. Assertion Best Practices

```python
class TestAssertionBestPractices(unittest.TestCase):
    
    def test_with_descriptive_messages(self):
        user = User("test@example.com")
        
        # Good: Descriptive assertion messages
        self.assertTrue(
            user.is_valid(), 
            f"User with email {user.email} should be valid"
        )
        
        # Good: Specific assertions
        self.assertEqual(
            user.email, 
            "test@example.com",
            "User email should match the provided email"
        )
    
    def test_multiple_related_assertions(self):
        user_data = create_test_user()
        
        # Use subTest for related assertions
        with self.subTest("Check user attributes"):
            self.assertIsNotNone(user_data.id)
            self.assertEqual(user_data.status, "active")
            self.assertIsInstance(user_data.created_at, datetime)
        
        with self.subTest("Check user methods"):
            self.assertTrue(user_data.is_active())
            self.assertFalse(user_data.is_deleted())
```

### 4. Error Testing

```python
class TestErrorHandling(unittest.TestCase):
    
    def test_specific_exception_with_message(self):
        service = ValidationService()
        
        with self.assertRaises(ValidationError) as context:
            service.validate_email("invalid-email")
        
        self.assertIn("Invalid email format", str(context.exception))
    
    def test_exception_chain(self):
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.ConnectionError("Network failure")
            
            service = ExternalAPIService()
            
            with self.assertRaises(ServiceUnavailableError) as context:
                service.fetch_data()
            
            # Verify the original exception is preserved
            self.assertIsInstance(context.exception.__cause__, requests.ConnectionError)
```

### 5. Performance Testing

```python
import time
from unittest.mock import patch

class TestPerformance(unittest.TestCase):
    
    def test_operation_completes_within_time_limit(self):
        """Test that operation completes within acceptable time."""
        start_time = time.time()
        
        service = DataProcessingService()
        service.process_large_dataset(test_data)
        
        execution_time = time.time() - start_time
        self.assertLess(execution_time, 5.0, "Operation took too long")
    
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_retry_mechanism(self, mock_sleep):
        """Test retry mechanism without actual delays."""
        with patch('requests.get') as mock_get:
            # Fail twice, then succeed
            mock_get.side_effect = [
                requests.RequestException("First failure"),
                requests.RequestException("Second failure"),
                Mock(status_code=200, json=lambda: {"data": "success"})
            ]
            
            service = RetryableService()
            result = service.fetch_with_retry()
            
            self.assertEqual(result["data"], "success")
            self.assertEqual(mock_get.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)  # Two retries
```

### 6. Integration Test Patterns

```python
class TestIntegrationPatterns(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up resources that are expensive to create."""
        cls.test_database = create_test_database()
        cls.test_server = start_test_server()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up expensive resources."""
        cls.test_database.drop_all()
        cls.test_server.stop()
    
    def setUp(self):
        """Set up test data for each test."""
        self.test_database.clear_all_tables()
        self.seed_test_data()
    
    def test_full_user_workflow(self):
        """Test complete user workflow from registration to deletion."""
        # Registration
        user_data = self.create_test_user()
        user_id = self.api_client.register_user(user_data)
        
        # Verification
        user = self.api_client.get_user(user_id)
        self.assertEqual(user['email'], user_data['email'])
        
        # Update
        updated_data = {'name': 'Updated Name'}
        self.api_client.update_user(user_id, updated_data)
        
        # Verification
        updated_user = self.api_client.get_user(user_id)
        self.assertEqual(updated_user['name'], 'Updated Name')
        
        # Deletion
        self.api_client.delete_user(user_id)
        
        # Verification
        with self.assertRaises(UserNotFoundError):
            self.api_client.get_user(user_id)
```

## Quick Reference

### unittest Cheat Sheet

```python
# Test class structure
class TestMyClass(unittest.TestCase):
    def setUp(self): pass           # Before each test
    def tearDown(self): pass        # After each test
    def test_something(self): pass  # Test method (must start with 'test_')

# Common assertions
self.assertEqual(a, b)        # a == b
self.assertNotEqual(a, b)     # a != b
self.assertTrue(x)            # bool(x) is True
self.assertFalse(x)           # bool(x) is False
self.assertIs(a, b)           # a is b
self.assertIsNot(a, b)        # a is not b
self.assertIsNone(x)          # x is None
self.assertIsNotNone(x)       # x is not None
self.assertIn(a, b)           # a in b
self.assertNotIn(a, b)        # a not in b
self.assertIsInstance(a, b)   # isinstance(a, b)
self.assertNotIsInstance(a, b) # not isinstance(a, b)
self.assertAlmostEqual(a, b)  # round(a-b, 7) == 0
self.assertGreater(a, b)      # a > b
self.assertLess(a, b)         # a < b
self.assertRegex(s, r)        # r.search(s)
self.assertRaises(exc, fun, *args, **kwds)
```

### Mock Cheat Sheet

```python
# Mock creation
from unittest.mock import Mock, MagicMock, AsyncMock, patch

mock = Mock()                 # Basic mock
magic = MagicMock()          # Mock with magic methods
async_mock = AsyncMock()     # Mock for async functions

# Mock configuration
mock.return_value = 'result'
mock.side_effect = Exception('error')
mock.side_effect = [1, 2, 3]  # Sequential returns

# Patch patterns
@patch('module.Class')               # Decorator
with patch('module.function'):       # Context manager
patch.object(obj, 'method')          # Specific object method
patch.dict('module.dict', values)    # Dictionary patching

# Mock assertions
mock.assert_called()                 # Called at least once
mock.assert_called_once()           # Called exactly once
mock.assert_called_with(*args)      # Last call had these args
mock.assert_called_once_with(*args) # Called once with these args
mock.assert_any_call(*args)         # Any call had these args
mock.assert_has_calls([call1, call2]) # Has these calls in order
mock.assert_not_called()            # Never called

# Mock inspection
mock.called                         # True if called
mock.call_count                     # Number of calls
mock.call_args                      # Last call arguments
mock.call_args_list                 # All call arguments
```

### Common Patterns Quick Reference

```python
# Mock file operations
with patch("builtins.open", mock_open(read_data="data")):
    # Test file reading

# Mock datetime
with patch('module.datetime') as mock_dt:
    mock_dt.now.return_value = datetime(2024, 1, 1)
    mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

# Mock environment variables
with patch.dict(os.environ, {'VAR': 'value'}):
    # Test code using environment variables

# Mock external APIs
with patch('requests.get') as mock_get:
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"data": "test"}

# Test exceptions
with self.assertRaises(ValueError):
    function_that_should_raise()

# Async testing
class TestAsync(unittest.IsolatedAsyncioTestCase):
    async def test_async_function(self):
        result = await async_function()
        self.assertEqual(result, expected)
```

This comprehensive guide should help you write robust, maintainable tests while avoiding common pitfalls. Remember: good tests are clear, focused, and test one thing at a time. Mock external dependencies, but don't over-mock your own code.