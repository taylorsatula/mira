"""
Unit tests for the Maps Tool.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from tools.maps_tool import GoogleMapsTool
from errors import ToolError, ErrorCode


class TestGoogleMapsTool(unittest.TestCase):
    """Test cases for the Maps Tool."""

    def setUp(self):
        """Set up test environment."""
        # Create a mock API key in the environment
        os.environ["GOOGLE_MAPS_API_KEY"] = "test_api_key"
        self.tool = GoogleMapsTool()

    @patch('tools.maps_tool.GoogleMapsTool.client')
    def test_geocode_operation(self, mock_client):
        """Test geocode operation."""
        # Mock response from Maps API
        mock_client.geocode.return_value = [
            {
                "formatted_address": "Eiffel Tower, Paris, France",
                "place_id": "test_place_id",
                "geometry": {
                    "location": {
                        "lat": 48.8584,
                        "lng": 2.2945
                    }
                },
                "types": ["tourist_attraction", "point_of_interest"]
            }
        ]

        # Run the geocode operation
        result = self.tool.run(operation="geocode", query="Eiffel Tower")

        # Verify the mock was called correctly
        mock_client.geocode.assert_called_once_with("Eiffel Tower")

        # Verify the result structure
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["formatted_address"], "Eiffel Tower, Paris, France")
        self.assertEqual(result["results"][0]["place_id"], "test_place_id")
        self.assertEqual(result["results"][0]["location"]["lat"], 48.8584)
        self.assertEqual(result["results"][0]["location"]["lng"], 2.2945)

    @patch('tools.maps_tool.GoogleMapsTool.client')
    def test_reverse_geocode_operation(self, mock_client):
        """Test reverse geocode operation."""
        # Mock response from Maps API
        mock_client.reverse_geocode.return_value = [
            {
                "formatted_address": "Eiffel Tower, Champ de Mars, Paris, France",
                "place_id": "test_place_id",
                "types": ["tourist_attraction", "point_of_interest"]
            }
        ]

        # Run the reverse geocode operation
        result = self.tool.run(operation="reverse_geocode", lat=48.8584, lng=2.2945)

        # Verify the mock was called correctly
        mock_client.reverse_geocode.assert_called_once_with((48.8584, 2.2945))

        # Verify the result structure
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["formatted_address"], "Eiffel Tower, Champ de Mars, Paris, France")
        self.assertEqual(result["results"][0]["place_id"], "test_place_id")

    @patch('tools.maps_tool.GoogleMapsTool.client')
    def test_places_nearby_operation(self, mock_client):
        """Test places nearby operation."""
        # Mock response from Maps API
        mock_client.places_nearby.return_value = {
            "results": [
                {
                    "name": "Test Restaurant",
                    "place_id": "test_restaurant_id",
                    "vicinity": "Near Eiffel Tower",
                    "types": ["restaurant", "food"],
                    "rating": 4.5,
                    "geometry": {
                        "location": {
                            "lat": 48.8580,
                            "lng": 2.2940
                        }
                    }
                }
            ]
        }

        # Run the places nearby operation
        result = self.tool.run(
            operation="places_nearby",
            lat=48.8584,
            lng=2.2945,
            radius=500,
            type="restaurant"
        )

        # Verify the mock was called correctly
        mock_client.places_nearby.assert_called_once_with(
            location=(48.8584, 2.2945),
            radius=500,
            type="restaurant"
        )

        # Verify the result structure
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["name"], "Test Restaurant")
        self.assertEqual(result["results"][0]["vicinity"], "Near Eiffel Tower")
        self.assertEqual(result["results"][0]["rating"], 4.5)

    def test_missing_parameters(self):
        """Test error handling for missing parameters."""
        # Test geocode without query
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation="geocode")
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test reverse_geocode without coordinates
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation="reverse_geocode")
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test places_nearby without coordinates
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation="places_nearby", radius=500)
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)

    def test_invalid_operation(self):
        """Test error handling for invalid operation."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation="invalid_operation")
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)


if __name__ == "__main__":
    unittest.main()