"""
Tests for the serialization module.
"""

import json
import unittest
import datetime
import uuid
from typing import Dict, Any

from serialization import to_json, from_json


class TestSerialization(unittest.TestCase):
    """Test cases for the serialization module."""

    def test_to_json_basic_types(self):
        """Test to_json with basic Python types."""
        # Test with a dictionary of basic types
        data = {
            "string": "test",
            "int": 123,
            "float": 123.45,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2}
        }
        
        # Convert to JSON and back to ensure it works
        json_str = to_json(data)
        decoded = json.loads(json_str)
        
        self.assertEqual(data, decoded)

    def test_to_json_datetime(self):
        """Test to_json with datetime objects."""
        now = datetime.datetime.now()
        today = datetime.date.today()
        
        data = {
            "datetime": now,
            "date": today
        }
        
        json_str = to_json(data)
        decoded = json.loads(json_str)
        
        # Datetime and date objects should be serialized as ISO format strings
        self.assertEqual(decoded["datetime"], now.isoformat())
        self.assertEqual(decoded["date"], today.isoformat())

    def test_to_json_uuid(self):
        """Test to_json with UUID objects."""
        id_obj = uuid.uuid4()
        
        data = {
            "id": id_obj
        }
        
        json_str = to_json(data)
        decoded = json.loads(json_str)
        
        # UUID objects should be serialized as strings
        self.assertEqual(decoded["id"], str(id_obj))

    def test_to_json_custom_object(self):
        """Test to_json with a custom object that has a to_dict method."""
        class CustomObject:
            def __init__(self, name, value):
                self.name = name
                self.value = value
                
            def to_dict(self) -> Dict[str, Any]:
                return {
                    "name": self.name,
                    "value": self.value
                }
        
        obj = CustomObject("test", 123)
        
        json_str = to_json(obj)
        decoded = json.loads(json_str)
        
        self.assertEqual(decoded, {"name": "test", "value": 123})

    def test_from_json_basic(self):
        """Test from_json with basic JSON string."""
        json_str = '{"name": "test", "value": 123}'
        
        decoded = from_json(json_str)
        
        self.assertEqual(decoded, {"name": "test", "value": 123})

    def test_from_json_with_class(self):
        """Test from_json with a class that has a from_dict method."""
        class CustomObject:
            def __init__(self, name="", value=0):
                self.name = name
                self.value = value
                
            @classmethod
            def from_dict(cls, data):
                obj = cls()
                obj.name = data.get("name", "")
                obj.value = data.get("value", 0)
                return obj
                
            def to_dict(self):
                return {
                    "name": self.name,
                    "value": self.value
                }
        
        json_str = '{"name": "test", "value": 123}'
        
        # Test deserialization with the class
        obj = from_json(json_str, CustomObject)
        
        self.assertIsInstance(obj, CustomObject)
        self.assertEqual(obj.name, "test")
        self.assertEqual(obj.value, 123)
        
        # Test round-trip serialization and deserialization
        round_trip = from_json(to_json(obj), CustomObject)
        
        self.assertIsInstance(round_trip, CustomObject)
        self.assertEqual(round_trip.name, "test")
        self.assertEqual(round_trip.value, 123)


if __name__ == "__main__":
    unittest.main()