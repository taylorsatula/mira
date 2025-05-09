"""
Tests for database datetime utilities.

These tests validate the behavior of the database-specific datetime utility functions,
ensuring proper UTC storage, serialization/deserialization, and timezone handling.
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import json
from sqlalchemy import Column, String

from utils.db_datetime_utils import (
    utc_datetime_column,
    utc_created_at_column,
    utc_updated_at_column,
    convert_datetime_to_utc_for_storage,
    format_db_datetime_as_iso,
    format_db_datetime,
    serialize_model_datetime,
    deserialize_datetime_strings,
    UTCDatetimeMixin
)
from utils.timezone_utils import utc_now, ensure_utc
from sqlalchemy.ext.declarative import declarative_base


# Create a Base for testing
TestBase = declarative_base()


class TestModel(UTCDatetimeMixin, TestBase):
    """Test model with UTCDatetimeMixin."""
    __tablename__ = 'test_models'
    
    id = Column(String, primary_key=True)
    name = Column(String)


class TestDbDatetimeUtils(unittest.TestCase):
    """Test cases for database datetime utilities."""

    def test_utc_datetime_column(self):
        """Test creating UTC datetime column."""
        # Test with defaults
        column = utc_datetime_column()
        self.assertTrue(column.nullable)
        self.assertIsNone(column.default)
        self.assertIsNone(column.onupdate)
        
        # Test with nullable=False
        column = utc_datetime_column(nullable=False)
        self.assertFalse(column.nullable)
        
        # Test with default=True (should use utc_now)
        column = utc_datetime_column(default=True)
        self.assertIsNotNone(column.default)
        
        # Test with custom default
        test_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        column = utc_datetime_column(default=test_dt)
        self.assertEqual(column.default.arg, test_dt)
        
        # Test with onupdate
        column = utc_datetime_column(onupdate=True)
        self.assertIsNotNone(column.onupdate)

    def test_utc_created_at_column(self):
        """Test creating UTC created_at column."""
        column = utc_created_at_column()
        self.assertFalse(column.nullable)
        self.assertIsNotNone(column.default)
        self.assertIsNone(column.onupdate)

    def test_utc_updated_at_column(self):
        """Test creating UTC updated_at column."""
        column = utc_updated_at_column()
        self.assertFalse(column.nullable)
        self.assertIsNotNone(column.default)
        self.assertIsNotNone(column.onupdate)

    def test_convert_datetime_to_utc_for_storage(self):
        """Test converting datetime to UTC for storage."""
        # Test with UTC datetime
        utc_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = convert_datetime_to_utc_for_storage(utc_dt)
        self.assertEqual(result, utc_dt)
        
        # Test with non-UTC datetime
        ny_dt = datetime(2023, 1, 1, 7, 0, 0, tzinfo=timezone.utc).astimezone(timezone(-timedelta(hours=5)))
        result = convert_datetime_to_utc_for_storage(ny_dt)
        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.hour, 12)  # 7am NY = 12pm UTC
        
        # Test with naive datetime
        naive_dt = datetime(2023, 1, 1, 12, 0, 0)
        result = convert_datetime_to_utc_for_storage(naive_dt)
        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.hour, 12)
        
        # Test with non-datetime
        self.assertEqual(convert_datetime_to_utc_for_storage("not a datetime"), "not a datetime")
        self.assertEqual(convert_datetime_to_utc_for_storage(None), None)

    def test_format_db_datetime_as_iso(self):
        """Test formatting database datetime as ISO string."""
        # Test with UTC datetime
        utc_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = format_db_datetime_as_iso(utc_dt)
        self.assertEqual(result, "2023-01-01T12:00:00+00:00")
        
        # Test with target timezone
        result = format_db_datetime_as_iso(utc_dt, "America/New_York")
        self.assertEqual(result, "2023-01-01T07:00:00-05:00")
        
        # Test with None
        self.assertIsNone(format_db_datetime_as_iso(None))

    def test_format_db_datetime(self):
        """Test formatting database datetime with different formats."""
        # Test with UTC datetime
        utc_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Default format (date_time)
        result = format_db_datetime(utc_dt)
        self.assertEqual(result, "2023-01-01 12:00:00")
        
        # With specific format
        result = format_db_datetime(utc_dt, "short")
        self.assertEqual(result, "12:00")
        
        # With target timezone
        result = format_db_datetime(utc_dt, "short", "America/New_York")
        self.assertEqual(result, "07:00")
        
        # Test with None
        self.assertIsNone(format_db_datetime(None))

    def test_serialize_model_datetime(self):
        """Test serializing model datetime fields."""
        # Create a dictionary with datetime fields
        model_dict = {
            "id": "test1",
            "name": "Test Model",
            "created_at": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "updated_at": datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            "non_datetime": "string value"
        }
        
        # Serialize with default timezone
        result = serialize_model_datetime(model_dict, ["created_at", "updated_at"])
        
        # Check datetime fields were converted to ISO strings
        self.assertEqual(result["created_at"], "2023-01-01T12:00:00+00:00")
        self.assertEqual(result["updated_at"], "2023-01-02T12:00:00+00:00")
        self.assertEqual(result["non_datetime"], "string value")
        
        # Serialize with target timezone
        result = serialize_model_datetime(model_dict, ["created_at", "updated_at"], "America/New_York")
        
        # Check datetime fields were converted to target timezone ISO strings
        self.assertEqual(result["created_at"], "2023-01-01T07:00:00-05:00")
        self.assertEqual(result["updated_at"], "2023-01-02T07:00:00-05:00")

    def test_deserialize_datetime_strings(self):
        """Test deserializing datetime string fields to UTC datetime objects."""
        # Create a dictionary with datetime string fields
        data = {
            "id": "test1",
            "name": "Test Model",
            "created_at": "2023-01-01T12:00:00+00:00",  # UTC
            "updated_at": "2023-01-02T07:00:00-05:00",  # NY time (equivalent to UTC noon)
            "non_datetime": "string value"
        }
        
        # Deserialize
        result = deserialize_datetime_strings(data, ["created_at", "updated_at"])
        
        # Check datetime strings were converted to UTC datetime objects
        self.assertIsInstance(result["created_at"], datetime)
        self.assertIsInstance(result["updated_at"], datetime)
        self.assertEqual(result["created_at"].tzinfo, timezone.utc)
        self.assertEqual(result["updated_at"].tzinfo, timezone.utc)
        self.assertEqual(result["created_at"].hour, 12)
        self.assertEqual(result["updated_at"].hour, 12)  # NY 7am = UTC noon
        self.assertEqual(result["non_datetime"], "string value")
        
        # Test with invalid datetime string
        data_with_error = {
            "created_at": "invalid datetime string"
        }
        
        # Should not raise an error, but log a warning
        with patch('utils.db_datetime_utils.logger') as mock_logger:
            result = deserialize_datetime_strings(data_with_error, ["created_at"])
            mock_logger.warning.assert_called_once()
            self.assertEqual(result["created_at"], "invalid datetime string")


class TestUTCDatetimeMixin(unittest.TestCase):
    """Test UTCDatetimeMixin behavior."""

    def test_init_converts_datetime_to_utc(self):
        """Test initialization converts datetime values to UTC."""
        # Create a non-UTC datetime
        ny_dt = datetime(2023, 1, 1, 7, 0, 0, tzinfo=timezone.utc).astimezone(timezone(-timedelta(hours=5)))
        
        # Initialize model with non-UTC datetime
        model = TestModel(id="test1", name="Test", created_at=ny_dt)
        
        # Check that created_at was converted to UTC
        self.assertEqual(model.created_at.tzinfo, timezone.utc)
        self.assertEqual(model.created_at.hour, 12)  # 7am NY = 12pm UTC

    def test_to_dict_formats_datetimes(self):
        """Test to_dict() formats datetime fields as ISO strings."""
        # Create mock mapper for testing
        mock_column1 = MagicMock()
        mock_column1.key = "id"
        
        mock_column2 = MagicMock()
        mock_column2.key = "created_at"
        
        mock_mapper = MagicMock()
        mock_mapper.columns = [mock_column1, mock_column2]
        
        # Create a model instance
        model = TestModel(id="test1", name="Test")
        model.created_at = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Mock __mapper__ property
        with patch.object(TestModel, '__mapper__', mock_mapper):
            # Get dict representation
            result = model.to_dict()
            
            # Check that created_at was formatted as ISO string
            self.assertEqual(result["id"], "test1")
            self.assertEqual(result["created_at"], "2023-01-01T12:00:00+00:00")

    def test_from_dict_converts_datetime_strings(self):
        """Test from_dict() converts datetime strings to UTC datetime objects."""
        # Mock mapper for testing
        mock_column1 = MagicMock()
        mock_column1.key = "id"
        
        mock_column2 = MagicMock()
        mock_column2.key = "created_at"
        mock_column2.type = MagicMock(__class__.__name__="DateTime")
        
        mock_mapper = MagicMock()
        mock_mapper.columns = [mock_column1, mock_column2]
        
        # Data with datetime string
        data = {
            "id": "test1",
            "name": "Test",
            "created_at": "2023-01-01T07:00:00-05:00"  # NY time
        }
        
        # Mock __mapper__ property
        with patch.object(TestModel, '__mapper__', mock_mapper):
            # Create model from dict
            model = TestModel.from_dict(data)
            
            # Check that created_at was converted to UTC datetime
            self.assertEqual(model.id, "test1")
            self.assertEqual(model.name, "Test")
            self.assertIsInstance(model.created_at, datetime)
            self.assertEqual(model.created_at.tzinfo, timezone.utc)
            self.assertEqual(model.created_at.hour, 12)  # 7am NY = 12pm UTC


if __name__ == "__main__":
    unittest.main()