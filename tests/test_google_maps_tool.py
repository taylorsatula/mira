"""
Tests for the Maps tool.
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock

from tools.maps_tool import MapsTool
from errors import ToolError, ErrorCode


class TestMapsTool:
    """Test suite for the Maps tool."""

    def test_calculate_distance(self):
        """Test the calculation of distance between two points."""
        # No need to mock anything as the _haversine_distance method uses math only
        tool = MapsTool()
        result = tool.run(
            operation="calculate_distance",
            lat1=34.7304944,
            lng1=-86.5860382,
            lat2=34.7211561,
            lng2=-86.5807587
        )
        
        assert "distance_meters" in result
        assert "distance_kilometers" in result
        assert "distance_miles" in result
        
        # The distance should be around 1145 meters
        assert 1100 < result["distance_meters"] < 1200