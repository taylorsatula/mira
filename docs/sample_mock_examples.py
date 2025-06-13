#!/usr/bin/env python3
"""
Comprehensive Mock Examples for Python unittest
===============================================

This file demonstrates practical mocking patterns and best practices
for testing Python applications using unittest.mock.

Each example class shows different mocking scenarios you'll encounter
in real-world testing situations.
"""

import os
import json
import asyncio
import unittest
from datetime import datetime, date
from unittest.mock import (
    Mock, MagicMock, AsyncMock, patch, mock_open, PropertyMock,
    call, ANY, create_autospec
)
from unittest import IsolatedAsyncioTestCase


# ==============================================================================
# SAMPLE CLASSES TO TEST (These would normally be in separate modules)
# ==============================================================================

class DatabaseConnection:
    """Sample database connection class."""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connected = False
    
    def connect(self):
        """Connect to database."""
        # In real code, this would establish a connection
        self.connected = True
        return True
    
    def execute_query(self, query, params=None):
        """Execute a database query."""
        if not self.connected:
            raise ConnectionError("Not connected to database")
        # In real code, this would execute the query
        return [{"id": 1, "name": "Test User"}]
    
    def close(self):
        """Close database connection."""
        self.connected = False


class EmailService:
    """Sample email service class."""
    
    def __init__(self, smtp_server="smtp.gmail.com", port=587):
        self.smtp_server = smtp_server
        self.port = port
    
    def send_email(self, to_address, subject, body):
        """Send an email."""
        # In real code, this would send an email via SMTP
        return {"status": "sent", "message_id": "12345"}


class FileProcessor:
    """Sample file processing class."""
    
    def read_config(self, filename):
        """Read configuration from file."""
        with open(filename, 'r') as f:
            return json.load(f)
    
    def write_log(self, filename, message):
        """Write log message to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")


class UserService:
    """Sample user service class with dependencies."""
    
    def __init__(self, db_connection, email_service):
        self.db = db_connection
        self.email_service = email_service
    
    def create_user(self, email, name):
        """Create a new user."""
        # Validate email
        if '@' not in email:
            raise ValueError("Invalid email address")
        
        # Save to database
        user_data = {"email": email, "name": name, "created_at": datetime.now()}
        result = self.db.execute_query(
            "INSERT INTO users (email, name, created_at) VALUES (?, ?, ?)",
            (email, name, user_data["created_at"])
        )
        
        # Send welcome email
        self.email_service.send_email(
            email,
            "Welcome!",
            f"Hello {name}, welcome to our service!"
        )
        
        return {"id": result[0]["id"], **user_data}


class AsyncAPIClient:
    """Sample async API client."""
    
    async def fetch_user_data(self, user_id):
        """Fetch user data from external API."""
        # In real code, this would make an HTTP request
        await asyncio.sleep(0.1)  # Simulate network delay
        return {"id": user_id, "name": "API User", "status": "active"}
    
    async def update_user(self, user_id, data):
        """Update user data via API."""
        await asyncio.sleep(0.1)
        return {"id": user_id, "updated": True, **data}


class ConfigManager:
    """Sample configuration manager."""
    
    def __init__(self):
        self.config = {}
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        self.config = {
            'database_url': os.environ['DATABASE_URL'],
            'api_key': os.environ['API_KEY'],
            'debug': os.environ.get('DEBUG', 'False').lower() == 'true'
        }
    
    @property
    def database_url(self):
        return self.config.get('database_url')
    
    @property
    def is_debug(self):
        return self.config.get('debug', False)


# ==============================================================================
# BASIC MOCKING EXAMPLES
# ==============================================================================

class TestBasicMocking(unittest.TestCase):
    """Examples of basic mocking patterns."""
    
    def test_mock_vs_magicmock(self):
        """Demonstrate difference between Mock and MagicMock."""
        # Regular Mock
        regular_mock = Mock()
        regular_mock.method.return_value = "result"
        
        self.assertEqual(regular_mock.method(), "result")
        
        # MagicMock includes magic methods
        magic_mock = MagicMock()
        magic_mock.__len__.return_value = 5
        magic_mock.__str__.return_value = "magic"
        
        self.assertEqual(len(magic_mock), 5)
        self.assertEqual(str(magic_mock), "magic")
    
    def test_return_value_and_side_effect(self):
        """Demonstrate return_value vs side_effect."""
        mock = Mock()
        
        # Simple return value
        mock.simple_method.return_value = "fixed_result"
        self.assertEqual(mock.simple_method(), "fixed_result")
        
        # Side effect with function
        def custom_side_effect(arg):
            if arg == "error":
                raise ValueError("Custom error")
            return f"processed_{arg}"
        
        mock.dynamic_method.side_effect = custom_side_effect
        self.assertEqual(mock.dynamic_method("test"), "processed_test")
        
        with self.assertRaises(ValueError):
            mock.dynamic_method("error")
        
        # Side effect with list (sequential returns)
        mock.sequence_method.side_effect = [1, 2, 3]
        self.assertEqual(mock.sequence_method(), 1)
        self.assertEqual(mock.sequence_method(), 2)
        self.assertEqual(mock.sequence_method(), 3)
    
    def test_mock_assertions(self):
        """Demonstrate various mock assertion methods."""
        mock = Mock()
        
        # Call the mock
        mock.method("arg1", "arg2", keyword="value")
        mock.method("arg3", "arg4")
        
        # Basic assertions
        self.assertTrue(mock.method.called)
        self.assertEqual(mock.method.call_count, 2)
        
        # Check last call
        mock.method.assert_called_with("arg3", "arg4")
        
        # Check any call
        mock.method.assert_any_call("arg1", "arg2", keyword="value")
        
        # Check all calls
        expected_calls = [
            call("arg1", "arg2", keyword="value"),
            call("arg3", "arg4")
        ]
        mock.method.assert_has_calls(expected_calls)


# ==============================================================================
# PATCH DECORATOR AND CONTEXT MANAGER EXAMPLES
# ==============================================================================

class TestPatchPatterns(unittest.TestCase):
    """Examples of patch decorator and context manager usage."""
    
    @patch('__main__.DatabaseConnection')
    def test_patch_as_decorator(self, mock_db_class):
        """Example of using patch as a decorator."""
        # Configure the mock
        mock_db_instance = Mock()
        mock_db_instance.connect.return_value = True
        mock_db_instance.execute_query.return_value = [{"id": 1}]
        mock_db_class.return_value = mock_db_instance
        
        # Test the code
        db = DatabaseConnection("test_connection")
        self.assertTrue(db.connect())
        result = db.execute_query("SELECT * FROM users")
        
        # Verify the mock was used correctly
        mock_db_class.assert_called_once_with("test_connection")
        mock_db_instance.connect.assert_called_once()
        self.assertEqual(result, [{"id": 1}])
    
    def test_patch_as_context_manager(self):
        """Example of using patch as a context manager."""
        with patch('__main__.EmailService') as mock_email_class:
            # Configure the mock
            mock_email_instance = Mock()
            mock_email_instance.send_email.return_value = {"status": "sent"}
            mock_email_class.return_value = mock_email_instance
            
            # Test the code
            email_service = EmailService()
            result = email_service.send_email("test@example.com", "Subject", "Body")
            
            # Verify
            self.assertEqual(result["status"], "sent")
            mock_email_instance.send_email.assert_called_once_with(
                "test@example.com", "Subject", "Body"
            )
    
    @patch.object(FileProcessor, 'read_config')
    def test_patch_object(self, mock_read_config):
        """Example of patching a specific object method."""
        # Configure the mock
        mock_read_config.return_value = {"setting": "value"}
        
        # Test the code
        processor = FileProcessor()
        config = processor.read_config("config.json")
        
        # Verify
        self.assertEqual(config["setting"], "value")
        mock_read_config.assert_called_once_with("config.json")
    
    @patch('__main__.EmailService')
    @patch('__main__.DatabaseConnection')
    def test_multiple_patches(self, mock_db_class, mock_email_class):
        """Example of multiple patches (note: order is reversed)."""
        # Configure mocks
        mock_db = Mock()
        mock_db.execute_query.return_value = [{"id": 1}]
        mock_db_class.return_value = mock_db
        
        mock_email = Mock()
        mock_email.send_email.return_value = {"status": "sent"}
        mock_email_class.return_value = mock_email
        
        # Test the service
        user_service = UserService(
            DatabaseConnection("test_db"),
            EmailService()
        )
        result = user_service.create_user("test@example.com", "Test User")
        
        # Verify
        self.assertIsNotNone(result["id"])
        mock_db.execute_query.assert_called_once()
        mock_email.send_email.assert_called_once()


# ==============================================================================
# FILE OPERATION MOCKING EXAMPLES
# ==============================================================================

class TestFileMocking(unittest.TestCase):
    """Examples of mocking file operations."""
    
    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_mock_file_read(self, mock_file):
        """Example of mocking file reading."""
        processor = FileProcessor()
        config = processor.read_config("config.json")
        
        # Verify file was opened correctly
        mock_file.assert_called_once_with("config.json", "r")
        
        # Verify content was parsed correctly
        self.assertEqual(config["key"], "value")
    
    @patch("builtins.open", new_callable=mock_open)
    @patch('__main__.datetime')
    def test_mock_file_write(self, mock_datetime, mock_file):
        """Example of mocking file writing with datetime."""
        # Mock datetime
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.strftime = datetime.strftime  # Keep strftime working
        
        processor = FileProcessor()
        processor.write_log("app.log", "Test message")
        
        # Verify file operations
        mock_file.assert_called_once_with("app.log", "a")
        mock_file().write.assert_called_once_with("[2024-01-01 12:00:00] Test message\n")
    
    def test_mock_multiple_files(self):
        """Example of mocking multiple different files."""
        file_contents = {
            "config.json": '{"env": "test"}',
            "users.json": '{"users": ["alice", "bob"]}',
            "default": '{"error": "file not found"}'
        }
        
        def open_side_effect(filename, mode='r'):
            content = file_contents.get(filename, file_contents["default"])
            return mock_open(read_data=content).return_value
        
        with patch("builtins.open", side_effect=open_side_effect):
            processor = FileProcessor()
            
            # Test different files
            config = processor.read_config("config.json")
            users = processor.read_config("users.json")
            unknown = processor.read_config("unknown.json")
            
            self.assertEqual(config["env"], "test")
            self.assertEqual(users["users"], ["alice", "bob"])
            self.assertEqual(unknown["error"], "file not found")


# ==============================================================================
# DATETIME AND ENVIRONMENT MOCKING EXAMPLES
# ==============================================================================

class TestDatetimeAndEnvironment(unittest.TestCase):
    """Examples of mocking datetime and environment variables."""
    
    @patch('__main__.datetime')
    def test_mock_datetime_now(self, mock_datetime):
        """Example of mocking datetime.now()."""
        # Configure mock to return fixed datetime for now()
        # but still allow datetime constructor to work
        fixed_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = fixed_time
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        processor = FileProcessor()
        processor.write_log = Mock()  # Mock the write method to avoid file operations
        
        # The datetime.now() call inside write_log will use our mock
        processor.write_log("app.log", "Test message")
        
        # Verify the mock was called (this is a simplified example)
        processor.write_log.assert_called_once()
    
    @patch.dict(os.environ, {
        'DATABASE_URL': 'postgresql://test:test@localhost/test',
        'API_KEY': 'test-api-key',
        'DEBUG': 'True'
    })
    def test_mock_environment_variables(self):
        """Example of mocking environment variables."""
        config_manager = ConfigManager()
        config_manager.load_from_env()
        
        # Verify environment variables were read correctly
        self.assertEqual(config_manager.database_url, 'postgresql://test:test@localhost/test')
        self.assertTrue(config_manager.is_debug)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_environment_variables(self):
        """Example of testing missing environment variables."""
        config_manager = ConfigManager()
        
        with self.assertRaises(KeyError):
            config_manager.load_from_env()


# ==============================================================================
# ASYNC MOCKING EXAMPLES
# ==============================================================================

class TestAsyncMocking(IsolatedAsyncioTestCase):
    """Examples of mocking async functions and methods."""
    
    async def test_basic_async_mock(self):
        """Basic AsyncMock usage."""
        mock_async_func = AsyncMock(return_value="async_result")
        
        result = await mock_async_func("arg1", "arg2")
        
        self.assertEqual(result, "async_result")
        mock_async_func.assert_awaited_once_with("arg1", "arg2")
    
    async def test_async_mock_side_effect(self):
        """AsyncMock with side_effect."""
        async def async_side_effect(user_id):
            if user_id == "error":
                raise ValueError("User not found")
            return {"id": user_id, "name": f"User {user_id}"}
        
        mock_async_func = AsyncMock(side_effect=async_side_effect)
        
        # Test successful case
        result = await mock_async_func("123")
        self.assertEqual(result["id"], "123")
        
        # Test error case
        with self.assertRaises(ValueError):
            await mock_async_func("error")
    
    @patch('__main__.AsyncAPIClient.fetch_user_data')
    async def test_patch_async_method(self, mock_fetch):
        """Example of patching async methods."""
        # Configure async mock
        mock_fetch.return_value = {"id": "123", "name": "Mocked User"}
        
        # Test the code
        client = AsyncAPIClient()
        result = await client.fetch_user_data("123")
        
        # Verify
        self.assertEqual(result["name"], "Mocked User")
        mock_fetch.assert_awaited_once_with("123")
    
    async def test_async_context_manager_mock(self):
        """Example of mocking async context managers."""
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = "context_value"
        mock_cm.__aexit__.return_value = None
        
        async with mock_cm as value:
            self.assertEqual(value, "context_value")
        
        mock_cm.__aenter__.assert_awaited_once()
        mock_cm.__aexit__.assert_awaited_once()


# ==============================================================================
# SPEC AND AUTOSPEC EXAMPLES
# ==============================================================================

class TestSpecAndAutospec(unittest.TestCase):
    """Examples of using spec and autospec for safer mocking."""
    
    def test_mock_with_spec(self):
        """Example of using spec to match interface."""
        # Mock with spec ensures only real methods can be called
        mock_db = Mock(spec=DatabaseConnection)
        
        # This works - connect is a real method
        mock_db.connect.return_value = True
        self.assertTrue(mock_db.connect())
        
        # This would raise AttributeError - nonexistent_method doesn't exist
        with self.assertRaises(AttributeError):
            mock_db.nonexistent_method()
    
    def test_create_autospec(self):
        """Example of using create_autospec for signature checking."""
        # Create autospec mock
        mock_db = create_autospec(DatabaseConnection)
        
        # Configure mock
        mock_db.return_value.connect.return_value = True
        mock_db.return_value.execute_query.return_value = [{"id": 1}]
        
        # Use the mock
        db_instance = mock_db("connection_string")
        self.assertTrue(db_instance.connect())
        
        # This would raise TypeError - wrong number of arguments
        with self.assertRaises(TypeError):
            db_instance.execute_query()  # Missing required argument
    
    @patch('__main__.DatabaseConnection', autospec=True)
    def test_autospec_with_patch(self, mock_db_class):
        """Example of using autospec with patch."""
        # Configure the autospec mock
        mock_db_instance = mock_db_class.return_value
        mock_db_instance.connect.return_value = True
        
        # This works with correct signature
        db = DatabaseConnection("test_connection")
        result = db.connect()
        
        # Verify
        self.assertTrue(result)
        mock_db_class.assert_called_once_with("test_connection")


# ==============================================================================
# PROPERTY AND ATTRIBUTE MOCKING
# ==============================================================================

class TestPropertyMocking(unittest.TestCase):
    """Examples of mocking properties and attributes."""
    
    def test_mock_property(self):
        """Example of mocking a property."""
        with patch.object(type(ConfigManager), 'database_url', 
                         new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = "mocked://database/url"
            
            config = ConfigManager()
            result = config.database_url
            
            self.assertEqual(result, "mocked://database/url")
            mock_prop.assert_called_once()
    
    def test_mock_class_attribute(self):
        """Example of mocking class attributes."""
        original_value = DatabaseConnection.connected if hasattr(DatabaseConnection, 'connected') else None
        
        with patch.object(DatabaseConnection, 'connected', True):
            db = DatabaseConnection("test")
            # In this example, we're just showing the pattern
            # The actual behavior would depend on how the class uses this attribute
            pass


# ==============================================================================
# ERROR HANDLING AND EDGE CASES
# ==============================================================================

class TestErrorHandlingMocking(unittest.TestCase):
    """Examples of testing error conditions and edge cases."""
    
    @patch('__main__.DatabaseConnection')
    def test_database_connection_error(self, mock_db_class):
        """Test handling of database connection errors."""
        mock_db_instance = Mock()
        mock_db_instance.connect.side_effect = ConnectionError("Database unavailable")
        mock_db_class.return_value = mock_db_instance
        
        with self.assertRaises(ConnectionError) as context:
            db = DatabaseConnection("test")
            db.connect()
        
        self.assertIn("Database unavailable", str(context.exception))
    
    @patch('__main__.EmailService')
    def test_email_service_retry_logic(self, mock_email_class):
        """Test retry logic with mock side effects."""
        mock_email = Mock()
        # Fail twice, then succeed
        mock_email.send_email.side_effect = [
            Exception("Network error"),
            Exception("Temporary failure"), 
            {"status": "sent", "message_id": "12345"}
        ]
        mock_email_class.return_value = mock_email
        
        # This would be implemented in a real retry service
        email_service = EmailService()
        
        # Simulate retry logic
        for attempt in range(3):
            try:
                result = email_service.send_email("test@example.com", "Subject", "Body")
                break
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                continue
        
        self.assertEqual(result["status"], "sent")
        self.assertEqual(mock_email.send_email.call_count, 3)
    
    def test_mock_call_args_validation(self):
        """Example of validating mock call arguments."""
        mock = Mock()
        
        # Call with different argument styles
        mock.method("pos1", "pos2", keyword="value")
        mock.method(arg1="pos1", arg2="pos2", keyword="value")
        
        # Verify using ANY for flexible matching
        mock.method.assert_any_call("pos1", "pos2", keyword="value")
        mock.method.assert_any_call(arg1=ANY, arg2=ANY, keyword="value")
        
        # Check all calls
        calls = mock.method.call_args_list
        self.assertEqual(len(calls), 2)


# ==============================================================================
# INTEGRATION TESTING WITH MOCKS
# ==============================================================================

class TestIntegrationWithMocks(unittest.TestCase):
    """Examples of integration testing using selective mocking."""
    
    @patch('__main__.EmailService')
    @patch('__main__.DatabaseConnection')
    def test_user_service_integration(self, mock_db_class, mock_email_class):
        """Integration test of UserService with mocked dependencies."""
        # Setup database mock
        mock_db = Mock()
        mock_db.execute_query.return_value = [{"id": 1}]
        mock_db_class.return_value = mock_db
        
        # Setup email mock
        mock_email = Mock()
        mock_email.send_email.return_value = {"status": "sent"}
        mock_email_class.return_value = mock_email
        
        # Test the integrated service
        user_service = UserService(
            DatabaseConnection("test_db"),
            EmailService("smtp.test.com")
        )
        
        result = user_service.create_user("user@example.com", "Test User")
        
        # Verify the integration
        self.assertEqual(result["email"], "user@example.com")
        self.assertEqual(result["name"], "Test User")
        self.assertIsNotNone(result["created_at"])
        
        # Verify database interaction
        mock_db.execute_query.assert_called_once()
        db_call_args = mock_db.execute_query.call_args
        self.assertIn("INSERT INTO users", db_call_args[0][0])
        
        # Verify email interaction
        mock_email.send_email.assert_called_once_with(
            "user@example.com",
            "Welcome!",
            "Hello Test User, welcome to our service!"
        )
    
    def test_user_service_validation_error(self):
        """Test validation without mocking (unit test of business logic)."""
        # Create real dependencies for this focused test
        mock_db = Mock()
        mock_email = Mock()
        
        user_service = UserService(mock_db, mock_email)
        
        # Test validation logic
        with self.assertRaises(ValueError) as context:
            user_service.create_user("invalid-email", "Test User")
        
        self.assertIn("Invalid email address", str(context.exception))
        
        # Verify no database or email calls were made
        mock_db.execute_query.assert_not_called()
        mock_email.send_email.assert_not_called()


# ==============================================================================
# HELPER FUNCTIONS FOR RUNNING EXAMPLES
# ==============================================================================

def run_specific_example(test_class_name):
    """Run a specific test class."""
    suite = unittest.TestLoader().loadTestsFromName(f'__main__.{test_class_name}')
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def run_all_examples():
    """Run all example test classes."""
    # Discover and run all tests in this module
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__('__main__'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    print("Python unittest and Mock Examples")
    print("=" * 50)
    print("\nAvailable example classes:")
    print("- TestBasicMocking: Basic mock usage patterns")
    print("- TestPatchPatterns: Patch decorator and context manager examples")
    print("- TestFileMocking: File operation mocking")
    print("- TestDatetimeAndEnvironment: Datetime and environment variable mocking")
    print("- TestAsyncMocking: Async function mocking")
    print("- TestSpecAndAutospec: Spec and autospec examples")
    print("- TestPropertyMocking: Property and attribute mocking")
    print("- TestErrorHandlingMocking: Error condition testing")
    print("- TestIntegrationWithMocks: Integration testing with mocks")
    print("\nRun all examples with: python sample_mock_examples.py")
    print("Run specific class with: python -m unittest sample_mock_examples.TestBasicMocking")
    
    # Run all tests
    unittest.main(verbosity=2)