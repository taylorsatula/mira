"""
Unit tests for the Ring Tool.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json

# Adjust imports to handle running the test directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.ring_tool import RingTool, RingApiClient
from errors import ToolError, ErrorCode


class TestRingTool(unittest.TestCase):
    """Test cases for the Ring Tool."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock token file to avoid interactive authentication
        self.data_dir = os.path.join("data", "tools", "ring_tool")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.token_path = os.path.join(self.data_dir, "token.json")
        self.token_data = {"refresh_token": "test_refresh_token"}
        
        # Write token data to file if it doesn't exist
        if not os.path.exists(self.token_path):
            with open(self.token_path, "w") as f:
                json.dump(self.token_data, f)
        
        # Create the tool instance
        self.tool = RingTool()
        
        # Mock the API client to avoid actual API calls
        self.mock_api_client = MagicMock(spec=RingApiClient)
        self.tool.api_client = self.mock_api_client
    
    def test_list_locations(self):
        """Test listing Ring locations."""
        # Mock locations response
        mock_locations = [
            {
                "location_id": "123456",
                "name": "Home",
                "address": {"address1": "123 Main St"},
                "timezone": "America/New_York"
            },
            {
                "location_id": "789012",
                "name": "Office",
                "address": {"address1": "456 Business Ave"},
                "timezone": "America/Los_Angeles"
            }
        ]
        
        self.mock_api_client.get_locations.return_value = mock_locations
        
        # Call the operation
        result = self.tool.run(operation="list_locations")
        
        # Verify the result
        self.assertIn("locations", result)
        self.assertEqual(len(result["locations"]), 2)
        self.assertEqual(result["locations"][0]["id"], "123456")
        self.assertEqual(result["locations"][0]["name"], "Home")
        self.assertEqual(result["locations"][1]["id"], "789012")
        self.assertEqual(result["locations"][1]["name"], "Office")
    
    def test_list_devices(self):
        """Test listing devices for a location."""
        # Mock devices response
        mock_devices = {
            "doorbots": [
                {
                    "id": 12345,
                    "description": "Front Door",
                    "firmware_version": "1.2.3",
                    "battery_life": 80,
                    "location_id": "123456"
                }
            ],
            "stickup_cams": [
                {
                    "id": 67890,
                    "description": "Backyard",
                    "firmware_version": "2.0.0",
                    "battery_life": 75,
                    "location_id": "123456"
                }
            ],
            "base_stations": [
                {
                    "id": 54321,
                    "description": "Alarm Base",
                    "firmware_version": "3.1.0",
                    "kind": "security-panel",
                    "location_id": "123456"
                }
            ],
            "chimes": [],
            "beams_bridges": [],
            "other": []
        }
        
        self.mock_api_client.get_location_devices.return_value = mock_devices
        
        # Call the operation
        result = self.tool.run(operation="list_devices", location_id="123456")
        
        # Verify the result
        self.assertIn("devices", result)
        self.assertEqual(len(result["devices"]), 3)
        
        # Check that devices are correctly parsed
        device_types = [device["type"] for device in result["devices"]]
        self.assertIn("doorbell", device_types)
        self.assertIn("camera", device_types)
        self.assertIn("security_panel", device_types)
    
    def test_get_alarm_mode(self):
        """Test getting the alarm mode for a location."""
        # Mock locations and alarm mode responses
        mock_locations = [
            {
                "location_id": "123456",
                "name": "Home"
            }
        ]
        
        mock_alarm_mode = {
            "mode": "some"  # 'home' mode in Ring API
        }
        
        self.mock_api_client.get_locations.return_value = mock_locations
        self.mock_api_client.get_alarm_mode.return_value = mock_alarm_mode
        
        # Call the operation
        result = self.tool.run(operation="get_alarm_mode", location_id="123456")
        
        # Verify the result
        self.assertEqual(result["location_name"], "Home")
        self.assertEqual(result["mode"], "home")
    
    def test_set_alarm_mode(self):
        """Test setting the alarm mode for a location."""
        # Mock responses for set_alarm_mode
        mock_locations = [
            {
                "location_id": "123456",
                "name": "Home"
            }
        ]
        
        mock_current_mode = {
            "mode": "none"  # 'disarmed' in friendly terms
        }
        
        mock_set_mode_response = {"success": True}
        
        self.mock_api_client.get_locations.return_value = mock_locations
        self.mock_api_client.get_alarm_mode.return_value = mock_current_mode
        self.mock_api_client.set_alarm_mode.return_value = mock_set_mode_response
        self.mock_api_client.verify_alarm_mode.return_value = True
        
        # Call the operation
        result = self.tool.run(
            operation="set_alarm_mode", 
            location_id="123456", 
            mode="home"
        )
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["location"], "Home")
        self.assertEqual(result["previous_mode"], "disarmed")
        self.assertEqual(result["current_mode"], "home")
        self.assertIn("armed in home mode", result["message"])
    
    def test_invalid_operation(self):
        """Test that an error is raised for invalid operations."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation="invalid_operation")
            
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
    
    def test_missing_parameters(self):
        """Test that an error is raised when required parameters are missing."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation="list_devices")  # missing location_id
            
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation="set_alarm_mode", location_id="123456")  # missing mode
            
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)


if __name__ == "__main__":
    unittest.main()