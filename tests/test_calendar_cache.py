"""Tests for CalendarCache from calendar_tool2.py."""

import unittest
import sys
import os
from unittest.mock import patch, mock_open, MagicMock
import json
import time

# Add the project root to Python path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.calendar_tool2 import CalendarCache


class TestCalendarCache(unittest.TestCase):
    """Test suite for CalendarCache class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache_dir = "/tmp/calendar_cache"
        self.cache_duration = 3600  # 1 hour
        self.cache = CalendarCache(self.cache_dir, self.cache_duration)
        self.test_key = "test_key_123"
        self.test_data = {"key": "value", "count": 42}
        
    @patch('os.makedirs')
    def test_initialization(self, mock_makedirs):
        """Test that CalendarCache initializes correctly."""
        # Arrange & Act
        cache = CalendarCache("/tmp/test_cache", 1800)
        
        # Assert
        mock_makedirs.assert_called_once_with("/tmp/test_cache", exist_ok=True)
        self.assertEqual(cache.cache_dir, "/tmp/test_cache")
        self.assertEqual(cache.cache_duration, 1800)
    
    def test_get_cache_path(self):
        """Test that get_cache_path returns the correct path."""
        # Arrange
        key = "test-calendar-url_2025-05-01_2025-05-07"
        
        # Act
        result = self.cache.get_cache_path(key)
        
        # Assert
        self.assertTrue(result.startswith("/tmp/calendar_cache/"))
        self.assertTrue(result.endswith(".json"))
        # The actual hash value might change depending on the hashing implementation
        # so we'll just check the format is correct
    
    @patch('os.path.exists')
    @patch('os.path.getmtime')
    @patch('time.time')
    def test_is_valid_with_valid_cache(self, mock_time, mock_getmtime, mock_exists):
        """Test is_valid returns True for a valid cache file."""
        # Arrange
        cache_path = "/tmp/calendar_cache/test.json"
        mock_exists.return_value = True
        mock_time.return_value = 10000
        mock_getmtime.return_value = 9000  # File is 1000 seconds old (< 3600s cache duration)
        
        # Act
        result = self.cache.is_valid(cache_path)
        
        # Assert
        self.assertTrue(result)
        mock_exists.assert_called_once_with(cache_path)
        mock_getmtime.assert_called_once_with(cache_path)
    
    @patch('os.path.exists')
    @patch('os.path.getmtime')
    @patch('time.time')
    def test_is_valid_with_expired_cache(self, mock_time, mock_getmtime, mock_exists):
        """Test is_valid returns False for an expired cache file."""
        # Arrange
        cache_path = "/tmp/calendar_cache/test.json"
        mock_exists.return_value = True
        mock_time.return_value = 10000
        mock_getmtime.return_value = 6000  # File is 4000 seconds old (> 3600s cache duration)
        
        # Act
        result = self.cache.is_valid(cache_path)
        
        # Assert
        self.assertFalse(result)
        mock_exists.assert_called_once_with(cache_path)
        mock_getmtime.assert_called_once_with(cache_path)
    
    @patch('os.path.exists')
    def test_is_valid_with_nonexistent_file(self, mock_exists):
        """Test is_valid returns False for a non-existent file."""
        # Arrange
        cache_path = "/tmp/calendar_cache/test.json"
        mock_exists.return_value = False
        
        # Act
        result = self.cache.is_valid(cache_path)
        
        # Assert
        self.assertFalse(result)
        mock_exists.assert_called_once_with(cache_path)
    
    @patch.object(CalendarCache, 'get_cache_path')
    @patch.object(CalendarCache, 'is_valid')
    @patch('builtins.open', new_callable=mock_open, read_data='{"key": "value", "count": 42}')
    def test_get_with_valid_cache(self, mock_file, mock_is_valid, mock_get_cache_path):
        """Test get retrieves data from a valid cache file."""
        # Arrange
        mock_get_cache_path.return_value = "/tmp/calendar_cache/test.json"
        mock_is_valid.return_value = True
        
        # Act
        result = self.cache.get(self.test_key)
        
        # Assert
        self.assertEqual(result, self.test_data)
        mock_get_cache_path.assert_called_once_with(self.test_key)
        mock_is_valid.assert_called_once_with("/tmp/calendar_cache/test.json")
        mock_file.assert_called_once_with("/tmp/calendar_cache/test.json", "r")
    
    @patch.object(CalendarCache, 'get_cache_path')
    @patch.object(CalendarCache, 'is_valid')
    def test_get_with_invalid_cache(self, mock_is_valid, mock_get_cache_path):
        """Test get returns None for an invalid cache."""
        # Arrange
        mock_get_cache_path.return_value = "/tmp/calendar_cache/test.json"
        mock_is_valid.return_value = False
        
        # Act
        result = self.cache.get(self.test_key)
        
        # Assert
        self.assertIsNone(result)
        mock_get_cache_path.assert_called_once_with(self.test_key)
        mock_is_valid.assert_called_once_with("/tmp/calendar_cache/test.json")
    
    @patch.object(CalendarCache, 'get_cache_path')
    @patch.object(CalendarCache, 'is_valid')
    @patch('builtins.open')
    def test_get_with_file_error(self, mock_file, mock_is_valid, mock_get_cache_path):
        """Test get handles file reading errors gracefully."""
        # Arrange
        mock_get_cache_path.return_value = "/tmp/calendar_cache/test.json"
        mock_is_valid.return_value = True
        mock_file.side_effect = IOError("File read error")
        
        # Act
        result = self.cache.get(self.test_key)
        
        # Assert
        self.assertIsNone(result)
        mock_get_cache_path.assert_called_once_with(self.test_key)
        mock_is_valid.assert_called_once_with("/tmp/calendar_cache/test.json")
    
    @patch.object(CalendarCache, 'get_cache_path')
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_set(self, mock_file, mock_json_dump, mock_get_cache_path):
        """Test set writes data to cache."""
        # Arrange
        mock_get_cache_path.return_value = "/tmp/calendar_cache/test.json"
        
        # Act
        self.cache.set(self.test_key, self.test_data)
        
        # Assert
        mock_get_cache_path.assert_called_once_with(self.test_key)
        mock_file.assert_called_once_with("/tmp/calendar_cache/test.json", "w")
        mock_json_dump.assert_called_once()
        # Check that json.dump was called with our data
        args, kwargs = mock_json_dump.call_args
        self.assertEqual(args[0], self.test_data)  # First arg is the data
    
    @patch.object(CalendarCache, 'get_cache_path')
    @patch('builtins.open')
    @patch('logging.warning')
    def test_set_with_file_error(self, mock_warning, mock_file, mock_get_cache_path):
        """Test set handles file writing errors gracefully."""
        # Arrange
        mock_get_cache_path.return_value = "/tmp/calendar_cache/test.json"
        mock_file.side_effect = IOError("File write error")
        
        # Act
        self.cache.set(self.test_key, self.test_data)
        
        # Assert
        mock_get_cache_path.assert_called_once_with(self.test_key)
        mock_file.assert_called_once_with("/tmp/calendar_cache/test.json", "w")
        # Verify a warning was logged
        mock_warning.assert_called_once()
        self.assertIn("Failed to cache data", mock_warning.call_args[0][0])


if __name__ == '__main__':
    unittest.main()