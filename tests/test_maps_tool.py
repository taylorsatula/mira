"""
Comprehensive test suite for maps_tool.py

Tests real-world scenarios, failure modes, and edge cases that could occur in production.
Follows pytest best practices with proper mocking and realistic test conditions.
"""

import pytest
import json
import math
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from tools.maps_tool import MapsTool, MapsToolConfig
from errors import ToolError, ErrorCode


class TestMapsToolConfiguration:
    """Test MapsTool configuration and initialization."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = MapsToolConfig()
        assert config.enabled is True
        assert config.timeout == 60
        assert config.max_retries == 3
        assert config.backoff_factor == 2.0
        assert config.cache_timeout == 86400
    
    def test_config_customization(self):
        """Test custom configuration values."""
        config = MapsToolConfig(
            enabled=False,
            timeout=30,
            max_retries=5,
            backoff_factor=1.5,
            cache_timeout=3600
        )
        assert config.enabled is False
        assert config.timeout == 30
        assert config.max_retries == 5
        assert config.backoff_factor == 1.5
        assert config.cache_timeout == 3600


class TestMapsToolInitialization:
    """Test MapsTool initialization and client setup."""
    
    def test_tool_initialization(self):
        """Test basic tool initialization."""
        tool = MapsTool()
        assert tool.name == "maps_tool"
        assert "comprehensive location intelligence" in tool.simple_description.lower()
        assert tool._client is None  # Client should be lazy-loaded
    
    def test_client_property_lazy_loading_success(self, mocker):
        """Test successful lazy loading of Google Maps client."""
        # Mock the Google Maps client
        mock_client_class = mocker.patch('tools.maps_tool.Client')
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock config to provide API key
        mock_config = Mock()
        mock_config.google_maps_api_key = "test_api_key"
        mocker.patch('tools.maps_tool.config', mock_config)
        
        tool = MapsTool()
        client = tool.client
        
        # Verify client was created correctly
        assert client == mock_client_instance
        mock_client_class.assert_called_once_with(key="test_api_key")
        
        # Verify client is cached (second access doesn't recreate)
        client2 = tool.client
        assert client2 == mock_client_instance
        assert mock_client_class.call_count == 1
    
    def test_client_property_missing_api_key(self, mocker):
        """Test client initialization fails when API key is missing."""
        # Mock config with no API key
        mock_config = Mock()
        mock_config.google_maps_api_key = None
        mocker.patch('tools.maps_tool.config', mock_config)
        
        tool = MapsTool()
        
        with pytest.raises(ToolError) as exc_info:
            _ = tool.client
        
        assert exc_info.value.error_code == ErrorCode.TOOL_EXECUTION_ERROR
        assert "Google Maps API key not found" in str(exc_info.value)
    
    def test_client_property_missing_googlemaps_library(self, mocker):
        """Test client initialization fails when googlemaps library is not installed."""
        # Mock ImportError when importing Client
        mocker.patch('tools.maps_tool.Client', side_effect=ImportError("No module named 'googlemaps'"))
        
        tool = MapsTool()
        
        with pytest.raises(ToolError) as exc_info:
            _ = tool.client
        
        assert exc_info.value.error_code == ErrorCode.TOOL_INITIALIZATION_ERROR
        assert "googlemaps library not installed" in str(exc_info.value)
    
    def test_client_property_initialization_error(self, mocker):
        """Test client initialization fails due to other errors."""
        # Mock config with API key
        mock_config = Mock()
        mock_config.google_maps_api_key = "test_api_key"
        mocker.patch('tools.maps_tool.config', mock_config)
        
        # Mock Client to raise an exception
        mock_client_class = mocker.patch('tools.maps_tool.Client')
        mock_client_class.side_effect = ValueError("Invalid API key format")
        
        tool = MapsTool()
        
        with pytest.raises(ToolError) as exc_info:
            _ = tool.client
        
        assert exc_info.value.error_code == ErrorCode.TOOL_EXECUTION_ERROR
        assert "Failed to initialize Google Maps client" in str(exc_info.value)


class TestMapsToolGeocoding:
    """Test geocoding operations."""
    
    @pytest.fixture
    def mock_maps_tool(self, mocker):
        """Create MapsTool with mocked client."""
        tool = MapsTool()
        mock_client = Mock()
        tool._client = mock_client
        return tool, mock_client
    
    def test_geocode_success(self, mock_maps_tool):
        """Test successful geocoding operation."""
        tool, mock_client = mock_maps_tool
        
        # Mock successful API response
        mock_response = [
            {
                "formatted_address": "Eiffel Tower, 5 Avenue Anatole France, 75007 Paris, France",
                "place_id": "ChIJLU7jZClu5kcR4PcOOO6p3I0",
                "geometry": {
                    "location": {"lat": 48.8583701, "lng": 2.2944813}
                },
                "types": ["tourist_attraction", "point_of_interest", "establishment"]
            }
        ]
        mock_client.geocode.return_value = mock_response
        
        result = tool.run(operation="geocode", query="Eiffel Tower, Paris")
        
        # Verify API was called correctly
        mock_client.geocode.assert_called_once_with(address="Eiffel Tower, Paris")
        
        # Verify response structure
        assert "results" in result
        assert len(result["results"]) == 1
        
        location = result["results"][0]
        assert location["formatted_address"] == "Eiffel Tower, 5 Avenue Anatole France, 75007 Paris, France"
        assert location["place_id"] == "ChIJLU7jZClu5kcR4PcOOO6p3I0"
        assert location["location"]["lat"] == 48.8583701
        assert location["location"]["lng"] == 2.2944813
        assert "tourist_attraction" in location["types"]
    
    def test_geocode_multiple_results(self, mock_maps_tool):
        """Test geocoding with multiple matching results."""
        tool, mock_client = mock_maps_tool
        
        # Mock response with multiple results
        mock_response = [
            {
                "formatted_address": "Paris, France",
                "place_id": "ChIJD7fiBh9u5kcRYJSMaMOCCwQ",
                "geometry": {"location": {"lat": 48.856614, "lng": 2.3522219}},
                "types": ["locality", "political"]
            },
            {
                "formatted_address": "Paris, TX, USA",
                "place_id": "ChIJmysnFgZYSoYRSfPTL2YJuck",
                "geometry": {"location": {"lat": 33.6609389, "lng": -95.555513}},
                "types": ["locality", "political"]
            }
        ]
        mock_client.geocode.return_value = mock_response
        
        result = tool.run(operation="geocode", query="Paris")
        
        assert len(result["results"]) == 2
        assert result["results"][0]["formatted_address"] == "Paris, France"
        assert result["results"][1]["formatted_address"] == "Paris, TX, USA"
    
    def test_geocode_no_results(self, mock_maps_tool):
        """Test geocoding with no matching results."""
        tool, mock_client = mock_maps_tool
        
        mock_client.geocode.return_value = []
        
        result = tool.run(operation="geocode", query="nonexistent place 12345")
        
        assert result["results"] == []
    
    def test_geocode_missing_query(self, mock_maps_tool):
        """Test geocoding fails when query parameter is missing."""
        tool, mock_client = mock_maps_tool
        
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="geocode")
        
        assert exc_info.value.error_code == ErrorCode.TOOL_INVALID_INPUT
        assert "query parameter is required" in str(exc_info.value)
    
    def test_geocode_api_error(self, mock_maps_tool):
        """Test geocoding handles API errors gracefully."""
        tool, mock_client = mock_maps_tool
        
        # Mock API error
        mock_client.geocode.side_effect = Exception("API quota exceeded")
        
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="geocode", query="test location")
        
        assert exc_info.value.error_code == ErrorCode.TOOL_EXECUTION_ERROR
        assert "Failed to geocode query" in str(exc_info.value)
        assert "API quota exceeded" in str(exc_info.value)


class TestMapsToolReverseGeocoding:
    """Test reverse geocoding operations."""
    
    @pytest.fixture
    def mock_maps_tool(self, mocker):
        """Create MapsTool with mocked client."""
        tool = MapsTool()
        mock_client = Mock()
        tool._client = mock_client
        return tool, mock_client
    
    def test_reverse_geocode_success(self, mock_maps_tool):
        """Test successful reverse geocoding operation."""
        tool, mock_client = mock_maps_tool
        
        mock_response = [
            {
                "formatted_address": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
                "place_id": "ChIJLU7jZClu5kcR4PcOOO6p3I0",
                "types": ["tourist_attraction", "point_of_interest", "establishment"]
            }
        ]
        mock_client.reverse_geocode.return_value = mock_response
        
        result = tool.run(operation="reverse_geocode", lat=48.8583701, lng=2.2944813)
        
        mock_client.reverse_geocode.assert_called_once_with((48.8583701, 2.2944813))
        
        assert "results" in result
        assert len(result["results"]) == 1
        location = result["results"][0]
        assert "Champ de Mars" in location["formatted_address"]
        assert location["place_id"] == "ChIJLU7jZClu5kcR4PcOOO6p3I0"
    
    def test_reverse_geocode_missing_coordinates(self, mock_maps_tool):
        """Test reverse geocoding fails when coordinates are missing."""
        tool, mock_client = mock_maps_tool
        
        # Test missing latitude
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="reverse_geocode", lng=2.2944813)
        
        assert exc_info.value.error_code == ErrorCode.TOOL_INVALID_INPUT
        assert "lat and lng parameters are required" in str(exc_info.value)
        
        # Test missing longitude
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="reverse_geocode", lat=48.8583701)
        
        assert exc_info.value.error_code == ErrorCode.TOOL_INVALID_INPUT
        assert "lat and lng parameters are required" in str(exc_info.value)
    
    def test_reverse_geocode_api_error(self, mock_maps_tool):
        """Test reverse geocoding handles API errors."""
        tool, mock_client = mock_maps_tool
        
        mock_client.reverse_geocode.side_effect = Exception("Invalid coordinates")
        
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="reverse_geocode", lat=48.8583701, lng=2.2944813)
        
        assert exc_info.value.error_code == ErrorCode.TOOL_EXECUTION_ERROR
        assert "Failed to reverse geocode coordinates" in str(exc_info.value)


class TestMapsToolPlaceDetails:
    """Test place details operations."""
    
    @pytest.fixture
    def mock_maps_tool(self, mocker):
        """Create MapsTool with mocked client."""
        tool = MapsTool()
        mock_client = Mock()
        tool._client = mock_client
        return tool, mock_client
    
    def test_place_details_success(self, mock_maps_tool):
        """Test successful place details retrieval."""
        tool, mock_client = mock_maps_tool
        
        mock_response = {
            "result": {
                "name": "Eiffel Tower",
                "formatted_address": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
                "formatted_phone_number": "+33 892 70 12 39",
                "international_phone_number": "+33 8 92 70 12 39",
                "website": "https://www.toureiffel.paris/en",
                "url": "https://maps.google.com/?cid=14430729982329876892",
                "rating": 4.6,
                "types": ["tourist_attraction", "point_of_interest", "establishment"],
                "geometry": {
                    "location": {"lat": 48.8583701, "lng": 2.2944813}
                },
                "opening_hours": {
                    "open_now": True,
                    "periods": [
                        {"open": {"day": 0, "time": "0930"}, "close": {"day": 0, "time": "2245"}}
                    ]
                }
            }
        }
        mock_client.place.return_value = mock_response
        
        result = tool.run(operation="place_details", place_id="ChIJLU7jZClu5kcR4PcOOO6p3I0")
        
        mock_client.place.assert_called_once_with(place_id="ChIJLU7jZClu5kcR4PcOOO6p3I0")
        
        assert result["name"] == "Eiffel Tower"
        assert result["formatted_address"] == "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France"
        assert result["formatted_phone_number"] == "+33 892 70 12 39"
        assert result["website"] == "https://www.toureiffel.paris/en"
        assert result["rating"] == 4.6
        assert result["location"]["lat"] == 48.8583701
        assert result["opening_hours"]["open_now"] is True
    
    def test_place_details_minimal_data(self, mock_maps_tool):
        """Test place details with minimal available data."""
        tool, mock_client = mock_maps_tool
        
        mock_response = {
            "result": {
                "name": "Small Local Business",
                "types": ["establishment"]
            }
        }
        mock_client.place.return_value = mock_response
        
        result = tool.run(operation="place_details", place_id="test_place_id")
        
        assert result["name"] == "Small Local Business"
        assert result["formatted_address"] == ""
        assert result["formatted_phone_number"] == ""
        assert result["website"] == ""
        assert result["rating"] == 0
        assert "location" not in result
        assert "opening_hours" not in result
    
    def test_place_details_missing_place_id(self, mock_maps_tool):
        """Test place details fails when place_id is missing."""
        tool, mock_client = mock_maps_tool
        
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="place_details")
        
        assert exc_info.value.error_code == ErrorCode.TOOL_INVALID_INPUT
        assert "place_id parameter is required" in str(exc_info.value)
    
    def test_place_details_no_result(self, mock_maps_tool):
        """Test place details when no result is found."""
        tool, mock_client = mock_maps_tool
        
        mock_response = {"status": "NOT_FOUND"}
        mock_client.place.return_value = mock_response
        
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="place_details", place_id="invalid_place_id")
        
        assert exc_info.value.error_code == ErrorCode.TOOL_EXECUTION_ERROR
        assert "No details found for place ID" in str(exc_info.value)


class TestMapsToolPlacesNearby:
    """Test places nearby search operations."""
    
    @pytest.fixture
    def mock_maps_tool(self, mocker):
        """Create MapsTool with mocked client."""
        tool = MapsTool()
        mock_client = Mock()
        tool._client = mock_client
        return tool, mock_client
    
    def test_places_nearby_success(self, mock_maps_tool):
        """Test successful nearby places search."""
        tool, mock_client = mock_maps_tool
        
        mock_response = {
            "results": [
                {
                    "name": "Le Jules Verne",
                    "place_id": "ChIJe8hLsClw5kcRCA4gYfL5Pxc",
                    "vicinity": "Avenue Gustave Eiffel, Paris",
                    "types": ["restaurant", "point_of_interest", "establishment"],
                    "rating": 4.1,
                    "geometry": {"location": {"lat": 48.8583, "lng": 2.2945}},
                    "opening_hours": {"open_now": True}
                }
            ]
        }
        mock_client.places_nearby.return_value = mock_response
        
        result = tool.run(
            operation="places_nearby",
            lat=48.8583701,
            lng=2.2944813,
            radius=1000,
            type="restaurant"
        )
        
        mock_client.places_nearby.assert_called_once_with(
            location=(48.8583701, 2.2944813),
            radius=1000,
            type="restaurant"
        )
        
        assert "results" in result
        assert len(result["results"]) == 1
        place = result["results"][0]
        assert place["name"] == "Le Jules Verne"
        assert place["rating"] == 4.1
        assert place["open_now"] is True
    
    def test_places_nearby_with_all_parameters(self, mock_maps_tool):
        """Test nearby places search with all optional parameters."""
        tool, mock_client = mock_maps_tool
        
        mock_response = {"results": []}
        mock_client.places_nearby.return_value = mock_response
        
        tool.run(
            operation="places_nearby",
            lat=48.8583701,
            lng=2.2944813,
            radius=500,
            type="restaurant",
            keyword="french cuisine",
            open_now=True,
            language="en"
        )
        
        mock_client.places_nearby.assert_called_once_with(
            location=(48.8583701, 2.2944813),
            radius=500,
            type="restaurant",
            keyword="french cuisine",
            open_now=True,
            language="en"
        )
    
    def test_places_nearby_missing_coordinates(self, mock_maps_tool):
        """Test nearby places search fails when coordinates are missing."""
        tool, mock_client = mock_maps_tool
        
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="places_nearby", lng=2.2944813)
        
        assert exc_info.value.error_code == ErrorCode.TOOL_INVALID_INPUT
        assert "lat and lng parameters are required" in str(exc_info.value)
    
    def test_places_nearby_no_results(self, mock_maps_tool):
        """Test nearby places search with no results."""
        tool, mock_client = mock_maps_tool
        
        mock_response = {"results": []}
        mock_client.places_nearby.return_value = mock_response
        
        result = tool.run(operation="places_nearby", lat=0.0, lng=0.0)
        
        assert result["results"] == []


class TestMapsToolFindPlace:
    """Test find place operations."""
    
    @pytest.fixture
    def mock_maps_tool(self, mocker):
        """Create MapsTool with mocked client."""
        tool = MapsTool()
        mock_client = Mock()
        tool._client = mock_client
        return tool, mock_client
    
    def test_find_place_success(self, mock_maps_tool):
        """Test successful find place operation."""
        tool, mock_client = mock_maps_tool
        
        mock_response = {
            "candidates": [
                {
                    "name": "Huntsville Hospital",
                    "place_id": "ChIJJZZlLNdrYogR44PRuZ4EXU8",
                    "formatted_address": "101 Sivley Rd SW, Huntsville, AL 35801, USA",
                    "geometry": {"location": {"lat": 34.7211561, "lng": -86.5807587}},
                    "types": ["hospital", "health"],
                    "rating": 4.2
                }
            ]
        }
        mock_client.find_place.return_value = mock_response
        
        result = tool.run(operation="find_place", query="Huntsville Hospital")
        
        mock_client.find_place.assert_called_once_with(
            input="Huntsville Hospital",
            input_type="textquery",
            fields=['place_id', 'name', 'formatted_address', 'geometry', 'types', 'business_status', 'rating']
        )
        
        assert "results" in result
        assert len(result["results"]) == 1
        place = result["results"][0]
        assert place["name"] == "Huntsville Hospital"
        assert place["formatted_address"] == "101 Sivley Rd SW, Huntsville, AL 35801, USA"
        assert place["rating"] == 4.2
    
    def test_find_place_custom_fields(self, mock_maps_tool):
        """Test find place with custom fields."""
        tool, mock_client = mock_maps_tool
        
        mock_response = {"candidates": []}
        mock_client.find_place.return_value = mock_response
        
        custom_fields = ["place_id", "name", "rating"]
        tool.run(operation="find_place", query="test place", fields=custom_fields)
        
        mock_client.find_place.assert_called_once_with(
            input="test place",
            input_type="textquery",
            fields=custom_fields
        )
    
    def test_find_place_no_candidates(self, mock_maps_tool):
        """Test find place with no candidates found."""
        tool, mock_client = mock_maps_tool
        
        mock_response = {"candidates": []}
        mock_client.find_place.return_value = mock_response
        
        result = tool.run(operation="find_place", query="nonexistent place")
        
        assert result["results"] == []
    
    def test_find_place_missing_query(self, mock_maps_tool):
        """Test find place fails when query is missing."""
        tool, mock_client = mock_maps_tool
        
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="find_place")
        
        assert exc_info.value.error_code == ErrorCode.TOOL_INVALID_INPUT
        assert "query parameter is required" in str(exc_info.value)


class TestMapsToolDistanceCalculation:
    """Test distance calculation functionality."""
    
    @pytest.fixture
    def mock_maps_tool(self, mocker):
        """Create MapsTool with mocked client."""
        tool = MapsTool()
        mock_client = Mock()
        tool._client = mock_client
        return tool, mock_client
    
    def test_calculate_distance_success(self, mock_maps_tool):
        """Test successful distance calculation."""
        tool, mock_client = mock_maps_tool
        
        # Huntsville Hospital to Research Park Boulevard (approximately 1.06 km)
        result = tool.run(
            operation="calculate_distance",
            lat1=34.7211561,
            lng1=-86.5807587,
            lat2=34.7304944,
            lng2=-86.5860382
        )
        
        # Verify distance is reasonable (approximately 1058 meters)
        assert "distance_meters" in result
        assert "distance_kilometers" in result
        assert "distance_miles" in result
        
        distance_m = result["distance_meters"]
        distance_km = result["distance_kilometers"]
        distance_mi = result["distance_miles"]
        
        # Check that distance is approximately correct (within 10% tolerance)
        expected_distance = 1058  # meters
        assert abs(distance_m - expected_distance) < expected_distance * 0.1
        
        # Check unit conversions
        assert abs(distance_km - distance_m / 1000) < 0.001
        assert abs(distance_mi - distance_m / 1609.34) < 0.001
    
    def test_calculate_distance_zero_distance(self, mock_maps_tool):
        """Test distance calculation between identical points."""
        tool, mock_client = mock_maps_tool
        
        result = tool.run(
            operation="calculate_distance",
            lat1=48.8583701,
            lng1=2.2944813,
            lat2=48.8583701,
            lng2=2.2944813
        )
        
        assert result["distance_meters"] == 0.0
        assert result["distance_kilometers"] == 0.0
        assert result["distance_miles"] == 0.0
    
    def test_calculate_distance_large_distance(self, mock_maps_tool):
        """Test distance calculation across large distances."""
        tool, mock_client = mock_maps_tool
        
        # Paris to New York (approximately 5837 km)
        result = tool.run(
            operation="calculate_distance",
            lat1=48.8566,  # Paris
            lng1=2.3522,
            lat2=40.7128,  # New York
            lng2=-74.0060
        )
        
        distance_km = result["distance_kilometers"]
        # Should be approximately 5837 km (within 5% tolerance)
        expected_distance = 5837
        assert abs(distance_km - expected_distance) < expected_distance * 0.05
    
    def test_calculate_distance_missing_coordinates(self, mock_maps_tool):
        """Test distance calculation fails when coordinates are missing."""
        tool, mock_client = mock_maps_tool
        
        # Test missing lat1
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="calculate_distance", lng1=-86.5807587, lat2=34.7304944, lng2=-86.5860382)
        
        assert exc_info.value.error_code == ErrorCode.TOOL_INVALID_INPUT
        assert "lat1, lng1, lat2, lng2 parameters are required" in str(exc_info.value)
    
    def test_calculate_distance_invalid_coordinates(self, mock_maps_tool):
        """Test distance calculation with invalid coordinate values."""
        tool, mock_client = mock_maps_tool
        
        with pytest.raises(ToolError) as exc_info:
            tool.run(
                operation="calculate_distance",
                lat1="invalid",
                lng1=-86.5807587,
                lat2=34.7304944,
                lng2=-86.5860382
            )
        
        assert exc_info.value.error_code == ErrorCode.TOOL_INVALID_INPUT
        assert "Coordinates must be valid numbers" in str(exc_info.value)


class TestMapsToolErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def mock_maps_tool(self, mocker):
        """Create MapsTool with mocked client."""
        tool = MapsTool()
        mock_client = Mock()
        tool._client = mock_client
        return tool, mock_client
    
    def test_invalid_operation(self, mock_maps_tool):
        """Test handling of invalid operation."""
        tool, mock_client = mock_maps_tool
        
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="invalid_operation")
        
        assert exc_info.value.error_code == ErrorCode.TOOL_INVALID_INPUT
        assert "Unknown operation: invalid_operation" in str(exc_info.value)
        assert "Valid operations are:" in str(exc_info.value)
    
    def test_operation_requires_location_access(self, mock_maps_tool):
        """Test operations that require location when coordinates are None."""
        tool, mock_client = mock_maps_tool
        
        # Test places_nearby without coordinates
        result = tool.run(operation="places_nearby")
        
        assert "error" in result
        assert "requires_location" in result
        assert result["requires_location"] is True
        assert "location access" in result["error"]
    
    def test_api_network_timeout_simulation(self, mock_maps_tool):
        """Test handling of network timeouts during API calls."""
        tool, mock_client = mock_maps_tool
        
        # Simulate timeout error
        import socket
        mock_client.geocode.side_effect = socket.timeout("Request timed out")
        
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="geocode", query="test location")
        
        assert exc_info.value.error_code == ErrorCode.TOOL_EXECUTION_ERROR
        assert "Failed to geocode query" in str(exc_info.value)
    
    def test_api_rate_limit_error(self, mock_maps_tool):
        """Test handling of API rate limit errors."""
        tool, mock_client = mock_maps_tool
        
        # Simulate rate limit error
        mock_client.places_nearby.side_effect = Exception("You have exceeded your rate-limit")
        
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="places_nearby", lat=48.8583701, lng=2.2944813)
        
        assert exc_info.value.error_code == ErrorCode.TOOL_EXECUTION_ERROR
        assert "rate-limit" in str(exc_info.value)
    
    def test_api_quota_exceeded_error(self, mock_maps_tool):
        """Test handling of API quota exceeded errors."""
        tool, mock_client = mock_maps_tool
        
        # Simulate quota exceeded error
        mock_client.reverse_geocode.side_effect = Exception("You have exceeded your daily request quota")
        
        with pytest.raises(ToolError) as exc_info:
            tool.run(operation="reverse_geocode", lat=48.8583701, lng=2.2944813)
        
        assert exc_info.value.error_code == ErrorCode.TOOL_EXECUTION_ERROR
        assert "quota" in str(exc_info.value)


class TestMapsToolIntegrationScenarios:
    """Test realistic integration scenarios that could occur in production."""
    
    @pytest.fixture
    def mock_maps_tool(self, mocker):
        """Create MapsTool with mocked client."""
        tool = MapsTool()
        mock_client = Mock()
        tool._client = mock_client
        return tool, mock_client
    
    def test_workflow_geocode_then_nearby_places(self, mock_maps_tool):
        """Test realistic workflow: geocode location then find nearby places."""
        tool, mock_client = mock_maps_tool
        
        # Step 1: Geocode "downtown seattle"
        geocode_response = [
            {
                "formatted_address": "Downtown, Seattle, WA, USA",
                "place_id": "ChIJ-bfVTh8VkFQRDZLQnmioK9s",
                "geometry": {"location": {"lat": 47.6062, "lng": -122.3321}},
                "types": ["sublocality", "political"]
            }
        ]
        mock_client.geocode.return_value = geocode_response
        
        geocode_result = tool.run(operation="geocode", query="downtown seattle")
        
        # Extract coordinates from geocode result
        location = geocode_result["results"][0]["location"]
        lat, lng = location["lat"], location["lng"]
        
        # Step 2: Find nearby restaurants
        nearby_response = {
            "results": [
                {
                    "name": "The Pink Door",
                    "place_id": "ChIJ8x8q8h8VkFQR8KJb2IuDMUU",
                    "vicinity": "1919 Post Alley, Seattle",
                    "types": ["restaurant", "establishment"],
                    "rating": 4.3,
                    "geometry": {"location": {"lat": 47.6088, "lng": -122.3408}}
                }
            ]
        }
        mock_client.places_nearby.return_value = nearby_response
        
        nearby_result = tool.run(
            operation="places_nearby",
            lat=lat,
            lng=lng,
            radius=1000,
            type="restaurant"
        )
        
        # Verify the workflow worked
        assert len(nearby_result["results"]) == 1
        assert nearby_result["results"][0]["name"] == "The Pink Door"
        
        # Verify API calls were made correctly
        mock_client.geocode.assert_called_once_with(address="downtown seattle")
        mock_client.places_nearby.assert_called_once_with(
            location=(47.6062, -122.3321),
            radius=1000,
            type="restaurant"
        )
    
    def test_concurrent_api_calls_simulation(self, mock_maps_tool):
        """Test that tool handles multiple rapid calls without state corruption."""
        tool, mock_client = mock_maps_tool
        
        # Simulate multiple geocoding calls in quick succession
        mock_responses = [
            [{"formatted_address": f"Location {i}", "place_id": f"place_{i}", 
              "geometry": {"location": {"lat": 40.0 + i, "lng": -74.0 + i}}, "types": []}]
            for i in range(5)
        ]
        
        mock_client.geocode.side_effect = mock_responses
        
        # Make multiple calls
        results = []
        for i in range(5):
            result = tool.run(operation="geocode", query=f"location {i}")
            results.append(result)
        
        # Verify each call got the correct response
        for i, result in enumerate(results):
            assert len(result["results"]) == 1
            assert result["results"][0]["formatted_address"] == f"Location {i}"
            assert result["results"][0]["place_id"] == f"place_{i}"
        
        # Verify all calls were made
        assert mock_client.geocode.call_count == 5
    
    def test_malformed_api_response_handling(self, mock_maps_tool):
        """Test handling of malformed or unexpected API responses."""
        tool, mock_client = mock_maps_tool
        
        # Test incomplete geocode response
        malformed_response = [
            {
                "formatted_address": "Some Place",
                # Missing place_id, geometry, types
            }
        ]
        mock_client.geocode.return_value = malformed_response
        
        result = tool.run(operation="geocode", query="test")
        
        # Should handle missing fields gracefully
        assert len(result["results"]) == 1
        location = result["results"][0]
        assert location["formatted_address"] == "Some Place"
        assert location["place_id"] == ""
        assert location["location"] == {}
        assert location["types"] == []
    
    def test_edge_case_coordinate_values(self, mock_maps_tool):
        """Test handling of edge case coordinate values."""
        tool, mock_client = mock_maps_tool
        
        edge_cases = [
            # North Pole
            (90.0, 0.0),
            # South Pole
            (-90.0, 0.0),
            # International Date Line
            (0.0, 180.0),
            (0.0, -180.0),
            # Equator/Prime Meridian
            (0.0, 0.0)
        ]
        
        for lat, lng in edge_cases:
            mock_client.reverse_geocode.return_value = [
                {"formatted_address": f"Location at {lat}, {lng}", "place_id": "test", "types": []}
            ]
            
            result = tool.run(operation="reverse_geocode", lat=lat, lng=lng)
            
            assert len(result["results"]) == 1
            mock_client.reverse_geocode.assert_called_with((lat, lng))


class TestMapsToolPerformanceConsiderations:
    """Test performance-related aspects and potential bottlenecks."""
    
    @pytest.fixture
    def mock_maps_tool(self, mocker):
        """Create MapsTool with mocked client."""
        tool = MapsTool()
        mock_client = Mock()
        tool._client = mock_client
        return tool, mock_client
    
    def test_large_places_nearby_response(self, mock_maps_tool):
        """Test handling of large API responses from places nearby."""
        tool, mock_client = mock_maps_tool
        
        # Create a large response with 20 places (Google's typical max)
        large_response = {
            "results": [
                {
                    "name": f"Place {i}",
                    "place_id": f"place_id_{i}",
                    "vicinity": f"Address {i}",
                    "types": ["establishment"],
                    "rating": 4.0 + (i % 10) * 0.1,
                    "geometry": {"location": {"lat": 40.0 + i * 0.001, "lng": -74.0 + i * 0.001}}
                }
                for i in range(20)
            ]
        }
        mock_client.places_nearby.return_value = large_response
        
        result = tool.run(operation="places_nearby", lat=40.0, lng=-74.0)
        
        # Verify all results are processed correctly
        assert len(result["results"]) == 20
        for i, place in enumerate(result["results"]):
            assert place["name"] == f"Place {i}"
            assert place["rating"] == 4.0 + (i % 10) * 0.1
    
    def test_memory_usage_with_large_addresses(self, mock_maps_tool):
        """Test handling of very long address strings."""
        tool, mock_client = mock_maps_tool
        
        # Create response with very long address
        long_address = "A" * 1000 + " Street, " + "B" * 500 + " City, " + "C" * 200 + " State"
        mock_response = [
            {
                "formatted_address": long_address,
                "place_id": "long_address_place",
                "geometry": {"location": {"lat": 40.0, "lng": -74.0}},
                "types": ["street_address"]
            }
        ]
        mock_client.geocode.return_value = mock_response
        
        result = tool.run(operation="geocode", query="test")
        
        assert result["results"][0]["formatted_address"] == long_address
        assert len(result["results"][0]["formatted_address"]) > 1500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])