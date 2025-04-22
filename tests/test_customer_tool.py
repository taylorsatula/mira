"""
Tests for the Customer tool with location-based features.
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock, PropertyMock

from tools.customer_tool import CustomerTool
from errors import ToolError, ErrorCode


@pytest.fixture
def mock_customer_directory():
    """Return a mock customer directory with location data."""
    return {
        "customers": {
            "CUST1": {
                "id": "CUST1",
                "given_name": "Rob",
                "family_name": "Grosnick",
                "email_address": "rlgrosnick@bellsouth.net",
                "address": {
                    "address_line_1": "8 William Way Pl SE",
                    "locality": "Gurley",
                    "administrative_district_level_1": "Alabama",
                    "postal_code": "35748",
                    "country": "US"
                },
                "geocoding_data": {
                    "coordinates": {
                        "lat": 34.7075,
                        "lng": -86.4125
                    },
                    "geocoded_at": 1609459200
                }
            }
        },
        "last_updated": 1609459200
    }


class TestCustomerTool:
    """Test suite for the Customer tool."""

    def test_find_closest_customers(self, mock_customer_directory):
        """Test finding customers by location."""
        with patch('tools.customer_tool.CustomerTool._load_customer_directory', 
                  return_value=mock_customer_directory):
            # Create patched maps_tool that returns a predictable distance
            mock_maps_run = MagicMock(return_value={"distance_meters": 903.0})
            mock_maps_tool = MagicMock()
            mock_maps_tool.run = mock_maps_run
            
            with patch('tools.maps_tool.MapsTool', return_value=mock_maps_tool):
                tool = CustomerTool()
                
                # Call the find_closest_customers method
                result = tool.run(
                    operation="find_closest_customers",
                    kwargs=json.dumps({
                        "lat": 34.703756177648486, 
                        "lng": -86.41752251022103,
                        "limit": 5
                    })
                )
                
                # Verify the results
                assert "customers" in result
                assert len(result["customers"]) == 1
                assert result["customers"][0]["id"] == "CUST1"
                assert result["customers"][0]["given_name"] == "Rob"
                assert result["customers"][0]["distance_meters"] == 903.0