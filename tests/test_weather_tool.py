"""
Unit tests for the weather_tool.py module.

This module contains tests for the WeatherTool, including validation,
API interaction, cache functionality, and heat stress calculations.
"""

import os
import json
import time
import hashlib
import pytest
import requests
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import the tool and related modules
from tools.weather_tool import WeatherTool, ValidationUtils, WeatherCache
from errors import ToolError


# -------------------- FIXTURES --------------------

@pytest.fixture
def weather_tool():
    """Fixture that provides a WeatherTool instance."""
    return WeatherTool()


@pytest.fixture
def sample_weather_data():
    """Fixture that provides sample weather data for testing."""
    return {
        "latitude": 34.7304,
        "longitude": -86.5859,
        "elevation": 197.0,
        "timezone": "America/Chicago",
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "wet_bulb_temperature_2m": "°C",
            "shortwave_radiation": "W/m²",
            "wind_speed_10m": "km/h"
        },
        "hourly": {
            "time": ["2025-05-06T12:00", "2025-05-06T13:00"],
            "temperature_2m": [20.6, 21.4],
            "wet_bulb_temperature_2m": [14.9, 15.0],
            "shortwave_radiation": [955.0, 997.5],
            "wind_speed_10m": [4.0, 2.2]
        },
        "daily_units": {
            "time": "iso8601",
            "temperature_2m_max": "°C",
            "temperature_2m_min": "°C"
        },
        "daily": {
            "time": ["2025-05-06"],
            "temperature_2m_max": [22.3],
            "temperature_2m_min": [11.3]
        }
    }


# -------------------- VALIDATION TESTS --------------------

def test_validate_coordinates():
    """Test the validation of latitude and longitude coordinates."""
    # Valid coordinates
    assert ValidationUtils.validate_coordinates(34.7304, -86.5859) == (34.7304, -86.5859)
    assert ValidationUtils.validate_coordinates("34.7304", "-86.5859") == (34.7304, -86.5859)
    
    # Invalid latitude
    with pytest.raises(ToolError) as excinfo:
        ValidationUtils.validate_coordinates(91, -86.5859)
    assert "Latitude must be between -90 and 90" in str(excinfo.value)
    
    # Invalid longitude
    with pytest.raises(ToolError) as excinfo:
        ValidationUtils.validate_coordinates(34.7304, 181)
    assert "Longitude must be between -180 and 180" in str(excinfo.value)
    
    # Missing latitude
    with pytest.raises(ToolError) as excinfo:
        ValidationUtils.validate_coordinates(None, -86.5859)
    assert "Latitude is required" in str(excinfo.value)
    
    # Non-numeric values
    with pytest.raises(ToolError) as excinfo:
        ValidationUtils.validate_coordinates("invalid", -86.5859)
    assert "Latitude must be a valid number" in str(excinfo.value)


def test_validate_date():
    """Test the validation of date strings."""
    # Valid date
    today = datetime.now().strftime("%Y-%m-%d")
    assert ValidationUtils.validate_date(today) == today
    
    # None value (default)
    assert ValidationUtils.validate_date(None) is None
    
    # Invalid format
    with pytest.raises(ToolError) as excinfo:
        ValidationUtils.validate_date("05/06/2025")
    assert "Invalid date format" in str(excinfo.value)
    
    # Date too far in future
    future_date = (datetime.now().replace(year=datetime.now().year + 2)).strftime("%Y-%m-%d")
    with pytest.raises(ToolError) as excinfo:
        ValidationUtils.validate_date(future_date)
    assert "cannot be more than" in str(excinfo.value)
    
    # Non-string value
    with pytest.raises(ToolError) as excinfo:
        ValidationUtils.validate_date(12345)
    assert "Date must be a string" in str(excinfo.value)


def test_validate_parameters():
    """Test the validation of parameter lists."""
    # None value (all parameters)
    assert ValidationUtils.validate_parameters(None) is None
    
    # Empty string (all parameters)
    assert ValidationUtils.validate_parameters("") is None
    
    # String with comma-separated values
    assert ValidationUtils.validate_parameters("temperature_2m,precipitation") == ["temperature_2m", "precipitation"]
    
    # List of parameters
    assert ValidationUtils.validate_parameters(["temperature_2m", "precipitation"]) == ["temperature_2m", "precipitation"]
    
    # Empty list (all parameters)
    assert ValidationUtils.validate_parameters([]) is None
    
    # Invalid type
    with pytest.raises(ToolError) as excinfo:
        ValidationUtils.validate_parameters(12345)
    assert "Parameters must be a list or comma-separated string" in str(excinfo.value)
    
    # List with non-string value
    with pytest.raises(ToolError) as excinfo:
        ValidationUtils.validate_parameters(["temperature_2m", 12345])
    assert "Parameter at index 1 must be a string" in str(excinfo.value)


def test_validate_forecast_type():
    """Test the validation of forecast type."""
    # Default value
    assert ValidationUtils.validate_forecast_type(None) == "hourly"
    
    # Valid values
    assert ValidationUtils.validate_forecast_type("hourly") == "hourly"
    assert ValidationUtils.validate_forecast_type("daily") == "daily"
    assert ValidationUtils.validate_forecast_type("HOURLY") == "hourly"
    
    # Invalid value
    with pytest.raises(ToolError) as excinfo:
        ValidationUtils.validate_forecast_type("weekly")
    assert "Invalid forecast type" in str(excinfo.value)
    
    # Non-string value
    with pytest.raises(ToolError) as excinfo:
        ValidationUtils.validate_forecast_type(12345)
    assert "Forecast type must be a string" in str(excinfo.value)


# -------------------- CACHE TESTS --------------------

@pytest.fixture
def test_cache_dir(tmp_path):
    """Fixture that provides a temporary directory for cache testing."""
    cache_dir = tmp_path / "weather_cache"
    cache_dir.mkdir()
    return str(cache_dir)


def test_weather_cache_init(test_cache_dir):
    """Test WeatherCache initialization."""
    cache = WeatherCache(test_cache_dir, 3600)
    assert cache.cache_dir == test_cache_dir
    assert cache.cache_duration == 3600
    assert os.path.exists(test_cache_dir)


def test_weather_cache_get_cache_path(test_cache_dir):
    """Test cache path generation."""
    cache = WeatherCache(test_cache_dir, 3600)
    key = "test_key"
    cache_path = cache.get_cache_path(key)
    
    # Verify path contains hash of key
    key_hash = hashlib.md5(key.encode()).hexdigest()
    assert key_hash in cache_path
    assert cache_path.endswith(".json")
    assert test_cache_dir in cache_path


def test_weather_cache_set_get(test_cache_dir):
    """Test setting and getting data from cache."""
    cache = WeatherCache(test_cache_dir, 3600)
    key = "test_key"
    data = {"test": "data"}
    
    # Set data in cache
    cache.set(key, data)
    
    # Verify file exists
    cache_path = cache.get_cache_path(key)
    assert os.path.exists(cache_path)
    
    # Get data from cache
    cached_data = cache.get(key)
    assert cached_data == data


def test_weather_cache_expiration(test_cache_dir):
    """Test cache expiration."""
    # Create cache with short duration
    cache = WeatherCache(test_cache_dir, 1)  # 1 second
    key = "test_key"
    data = {"test": "data"}
    
    # Set data in cache
    cache.set(key, data)
    
    # Verify it's initially valid
    assert cache.get(key) == data
    
    # Wait for cache to expire
    time.sleep(2)
    
    # Verify it's no longer valid
    assert cache.get(key) is None


# -------------------- WBGT CALCULATION TESTS --------------------

def test_calculate_wbgt(weather_tool):
    """Test WBGT calculation formula."""
    # Test with typical values
    wbgt = weather_tool._calculate_wbgt(
        wet_bulb_temperature=20.0,
        temperature=30.0,
        shortwave_radiation=800.0,
        wind_speed=10.0
    )
    
    # Manual calculation
    expected = (0.7 * 20.0) + (0.3 * 30.0) + (0.14 * (800.0 / 1000)) - (0.016 * 10.0)
    assert wbgt == pytest.approx(expected)
    
    # Test with zero values
    wbgt = weather_tool._calculate_wbgt(0, 0, 0, 0)
    assert wbgt == 0.0
    
    # Test with extreme values
    wbgt = weather_tool._calculate_wbgt(35.0, 45.0, 1200.0, 2.0)
    expected = (0.7 * 35.0) + (0.3 * 45.0) + (0.14 * (1200.0 / 1000)) - (0.016 * 2.0)
    assert wbgt == pytest.approx(expected)


def test_get_heat_stress_risk_level(weather_tool):
    """Test heat stress risk level determination based on WBGT values."""
    # Test each risk level threshold
    assert weather_tool._get_heat_stress_risk_level(20.0) == "Low"
    assert weather_tool._get_heat_stress_risk_level(26.0) == "Moderate"
    assert weather_tool._get_heat_stress_risk_level(28.5) == "High"
    assert weather_tool._get_heat_stress_risk_level(31.0) == "Very High"
    assert weather_tool._get_heat_stress_risk_level(35.0) == "Extreme"


# -------------------- API INTERACTION TESTS --------------------

@patch('tools.weather_tool.requests.get')
def test_get_weather_data_api_call(mock_get, weather_tool, sample_weather_data):
    """Test API call in _get_weather_data method."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.json.return_value = sample_weather_data
    mock_get.return_value = mock_response
    
    # Call the method
    with patch('tools.weather_tool.WeatherCache.get', return_value=None):  # Bypass cache
        with patch('tools.weather_tool.WeatherCache.set'):  # Mock cache set
            result = weather_tool._get_weather_data(34.7304, -86.5859, "hourly")
    
    # Verify API call parameters
    mock_get.assert_called_once()
    call_args = mock_get.call_args[0][1]
    assert "latitude" in call_args
    assert "longitude" in call_args
    assert "hourly" in call_args
    
    # Verify result
    assert result == sample_weather_data


@patch('tools.weather_tool.requests.get')
def test_get_weather_data_api_error(mock_get, weather_tool):
    """Test error handling for API call failures."""
    # Setup mock to raise exception
    mock_get.side_effect = requests.RequestException("API error")
    
    # Call the method and expect ToolError
    with patch('tools.weather_tool.WeatherCache.get', return_value=None):  # Bypass cache
        with pytest.raises(ToolError) as excinfo:
            weather_tool._get_weather_data(34.7304, -86.5859, "hourly")
    
    # Verify error message
    assert "Failed to fetch weather data" in str(excinfo.value)
    assert excinfo.value.code.name == "API_CONNECTION_ERROR"


# -------------------- MAIN OPERATION TESTS --------------------

@patch('tools.weather_tool.WeatherTool._get_weather_data')
def test_get_forecast(mock_get_weather, weather_tool, sample_weather_data):
    """Test get_forecast operation."""
    # Setup mock to return sample data
    mock_get_weather.return_value = sample_weather_data
    
    # Call the method
    result = weather_tool._get_forecast(34.7304, -86.5859, "hourly", None, None)
    
    # Verify weather data was requested correctly
    mock_get_weather.assert_called_once_with(34.7304, -86.5859, "hourly", None, None)
    
    # Verify result format
    assert "location" in result
    assert "forecast_type" in result
    assert "forecast" in result
    assert result["forecast_type"] == "hourly"
    assert "hourly" in result["forecast"]
    assert "hourly_units" in result["forecast"]


@patch('tools.weather_tool.WeatherTool._get_weather_data')
@patch('tools.weather_tool.WeatherTool._calculate_wbgt')
@patch('tools.weather_tool.WeatherTool._get_heat_stress_risk_level')
def test_get_heat_stress(mock_risk_level, mock_calculate_wbgt, mock_get_weather, weather_tool, sample_weather_data):
    """Test get_heat_stress operation."""
    # Setup mocks
    mock_get_weather.return_value = sample_weather_data
    mock_calculate_wbgt.return_value = 25.0
    mock_risk_level.return_value = "Moderate"
    
    # Call the method
    result = weather_tool._get_heat_stress(34.7304, -86.5859, "hourly", None, None)
    
    # Verify weather data was requested with required parameters
    mock_get_weather.assert_called_once()
    args = mock_get_weather.call_args[0]
    assert args[0] == 34.7304
    assert args[1] == -86.5859
    assert args[2] == "hourly"
    
    # Verify result format
    assert "location" in result
    assert "forecast_type" in result
    assert "forecast" in result
    assert result["forecast_type"] == "hourly"
    assert "hourly" in result["forecast"]
    assert "hourly_units" in result["forecast"]


@patch('tools.weather_tool.WeatherTool._get_forecast')
def test_run_get_forecast(mock_get_forecast, weather_tool):
    """Test the main run method with get_forecast operation."""
    # Setup mock to return sample result
    mock_result = {"test": "result"}
    mock_get_forecast.return_value = mock_result
    
    # Call the run method
    result = weather_tool.run(
        operation="get_forecast",
        latitude=34.7304,
        longitude=-86.5859,
        forecast_type="hourly",
        date="2025-05-06",
        parameters="temperature_2m,precipitation"
    )
    
    # Verify parameters were validated and passed correctly
    mock_get_forecast.assert_called_once()
    args = mock_get_forecast.call_args[0]
    assert args[0] == 34.7304
    assert args[1] == -86.5859
    assert args[2] == "hourly"
    assert args[3] == "2025-05-06"
    assert "temperature_2m" in args[4]
    assert "precipitation" in args[4]
    
    # Verify result format
    assert result["success"] is True
    assert "test" in result
    assert result["test"] == "result"


@patch('tools.weather_tool.WeatherTool._get_heat_stress')
def test_run_get_heat_stress(mock_get_heat_stress, weather_tool):
    """Test the main run method with get_heat_stress operation."""
    # Setup mock to return sample result
    mock_result = {"test": "result"}
    mock_get_heat_stress.return_value = mock_result
    
    # Call the run method
    result = weather_tool.run(
        operation="get_heat_stress",
        latitude=34.7304,
        longitude=-86.5859
    )
    
    # Verify _get_heat_stress was called with default parameters
    mock_get_heat_stress.assert_called_once()
    args = mock_get_heat_stress.call_args[0]
    assert args[0] == 34.7304
    assert args[1] == -86.5859
    assert args[2] == "hourly"  # Default
    
    # Verify result format
    assert result["success"] is True
    assert "test" in result
    assert result["test"] == "result"


def test_run_invalid_operation(weather_tool):
    """Test the run method with an invalid operation."""
    with pytest.raises(ToolError) as excinfo:
        weather_tool.run(
            operation="invalid_operation",
            latitude=34.7304,
            longitude=-86.5859
        )
    
    assert "Invalid operation" in str(excinfo.value)
    assert excinfo.value.code.name == "TOOL_INVALID_INPUT"