"""Tests for ValidationUtils from calendar_tool2.py."""

import unittest
import sys
import os
from unittest.mock import MagicMock
from datetime import datetime

# Add the project root to Python path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from errors import ToolError, ErrorCode
from tools.calendar_tool2 import ValidationUtils


class TestValidationUtils(unittest.TestCase):
    """Test suite for ValidationUtils class."""
    
    def test_validate_date_with_valid_input(self):
        """Test validate_date with valid ISO format date."""
        # Arrange
        date_str = "2025-05-01"
        param_name = "test_date"
        
        # Act
        result = ValidationUtils.validate_date(date_str, param_name)
        
        # Assert
        self.assertIsInstance(result, datetime)
        self.assertEqual(result.year, 2025)
        self.assertEqual(result.month, 5)
        self.assertEqual(result.day, 1)
    
    def test_validate_date_with_none(self):
        """Test validate_date with None input."""
        # Arrange
        date_str = None
        param_name = "test_date"
        
        # Act
        result = ValidationUtils.validate_date(date_str, param_name)
        
        # Assert
        self.assertIsNone(result)
    
    def test_validate_date_with_invalid_type(self):
        """Test validate_date with non-string input."""
        # Arrange
        date_str = 12345  # Not a string
        param_name = "test_date"
        
        # Act & Assert
        with self.assertRaises(ToolError) as context:
            ValidationUtils.validate_date(date_str, param_name)
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        self.assertIn("must be a string", str(context.exception))
    
    def test_validate_date_with_invalid_format(self):
        """Test validate_date with invalid date format."""
        # Arrange
        date_str = "05/01/2025"  # Not ISO format
        param_name = "test_date"
        
        # Act & Assert
        with self.assertRaises(ToolError) as context:
            ValidationUtils.validate_date(date_str, param_name)
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        self.assertIn("Invalid test_date format", str(context.exception))
    
    def test_validate_datetime_with_valid_input(self):
        """Test validate_datetime with valid ISO format datetime."""
        # Arrange
        dt_str = "2025-05-01T10:30:00"
        param_name = "test_datetime"
        
        # Act
        result = ValidationUtils.validate_datetime(dt_str, param_name)
        
        # Assert
        self.assertIsInstance(result, datetime)
        self.assertEqual(result.year, 2025)
        self.assertEqual(result.month, 5)
        self.assertEqual(result.day, 1)
        self.assertEqual(result.hour, 10)
        self.assertEqual(result.minute, 30)
        self.assertEqual(result.second, 0)
    
    def test_validate_datetime_with_none(self):
        """Test validate_datetime with None input."""
        # Arrange
        dt_str = None
        param_name = "test_datetime"
        
        # Act
        result = ValidationUtils.validate_datetime(dt_str, param_name)
        
        # Assert
        self.assertIsNone(result)
    
    def test_validate_datetime_with_invalid_type(self):
        """Test validate_datetime with non-string input."""
        # Arrange
        dt_str = 12345  # Not a string
        param_name = "test_datetime"
        
        # Act & Assert
        with self.assertRaises(ToolError) as context:
            ValidationUtils.validate_datetime(dt_str, param_name)
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        self.assertIn("must be a string", str(context.exception))
    
    def test_validate_datetime_with_invalid_format(self):
        """Test validate_datetime with invalid datetime format."""
        # Arrange
        dt_str = "05/01/2025 10:30:00"  # Not ISO format
        param_name = "test_datetime"
        
        # Act & Assert
        with self.assertRaises(ToolError) as context:
            ValidationUtils.validate_datetime(dt_str, param_name)
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        self.assertIn("Invalid test_datetime format", str(context.exception))
    
    def test_require_non_empty_string_with_valid_input(self):
        """Test require_non_empty_string with valid input."""
        # Arrange
        value = "test value"
        param_name = "test_param"
        
        # Act
        result = ValidationUtils.require_non_empty_string(value, param_name)
        
        # Assert
        self.assertEqual(result, value)
    
    def test_require_non_empty_string_with_empty_string(self):
        """Test require_non_empty_string with empty string."""
        # Arrange
        value = ""
        param_name = "test_param"
        
        # Act & Assert
        with self.assertRaises(ToolError) as context:
            ValidationUtils.require_non_empty_string(value, param_name)
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        self.assertIn("must be a non-empty string", str(context.exception))
    
    def test_require_non_empty_string_with_none(self):
        """Test require_non_empty_string with None."""
        # Arrange
        value = None
        param_name = "test_param"
        
        # Act & Assert
        with self.assertRaises(ToolError) as context:
            ValidationUtils.require_non_empty_string(value, param_name)
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        self.assertIn("must be a non-empty string", str(context.exception))
    
    def test_require_non_empty_string_with_non_string(self):
        """Test require_non_empty_string with non-string input."""
        # Arrange
        value = 12345  # Not a string
        param_name = "test_param"
        
        # Act & Assert
        with self.assertRaises(ToolError) as context:
            ValidationUtils.require_non_empty_string(value, param_name)
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        self.assertIn("must be a non-empty string", str(context.exception))


if __name__ == '__main__':
    unittest.main()