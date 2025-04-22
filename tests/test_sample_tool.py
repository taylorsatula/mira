import unittest
from unittest.mock import patch, MagicMock
import random
from datetime import datetime

from tools.sample_tool import WeatherTool
from errors import ToolError, ErrorCode


class TestWeatherTool(unittest.TestCase):
    """
    Comprehensive test suite for the WeatherTool class.
    
    These tests demonstrate best practices for testing tool implementations:
    1. Proper use of test fixtures and setup
    2. Mocking external dependencies and randomness
    3. Testing both success and error paths
    4. Validating input handling and validation
    5. Verifying output formats and content
    6. Testing error conditions and messages
    7. Testing helper methods and private functions
    """

    def setUp(self):
        """Set up test fixtures and initialize the tool for each test."""
        # Create a logger mock to avoid actual logging during tests
        with patch('tools.repo.Tool.__init__') as mock_init:
            self.tool = WeatherTool()
            # Set up a mock logger
            self.tool.logger = MagicMock()

        # Sample valid inputs for testing
        self.valid_location = "New York"
        self.valid_units_celsius = "celsius"
        self.valid_units_fahrenheit = "fahrenheit"
        
        # Define expected structure for weather data responses
        self.expected_weather_fields = [
            "location", "temperature", "units", "conditions", 
            "humidity", "wind_speed", "timestamp"
        ]
        self.expected_forecast_fields = [
            "day", "day_name", "temperature", "conditions", 
            "humidity", "precipitation_chance", "wind_speed"
        ]

    def test_initialization(self):
        """Test proper initialization of the WeatherTool."""
        # Verify weather conditions list is properly initialized
        self.assertIsInstance(self.tool.weather_conditions, list)
        self.assertTrue(len(self.tool.weather_conditions) > 0)
        
        # Verify logger is initialized and used
        self.tool.logger.info.assert_called_once()

    @patch('random.randint')
    @patch('random.choice')
    def test_run_with_celsius(self, mock_choice, mock_randint):
        """Test the run method with celsius units."""
        # Set up mocks
        mock_randint.side_effect = [25, 60, 10]  # temp, humidity, wind
        mock_choice.return_value = "Sunny"
        
        # Mock timestamp to make test deterministic
        with patch.object(self.tool, '_get_current_timestamp') as mock_timestamp:
            mock_timestamp.return_value = "2023-04-12T12:00:00Z"
            
            # Run the tool
            result = self.tool.run(self.valid_location, self.valid_units_celsius)
            
            # Verify result structure
            for field in self.expected_weather_fields:
                self.assertIn(field, result)
                
            # Verify specific values
            self.assertEqual(result["location"], self.valid_location)
            self.assertEqual(result["temperature"], 25.0)
            self.assertEqual(result["units"], self.valid_units_celsius)
            self.assertEqual(result["conditions"], "Sunny")
            self.assertEqual(result["humidity"], 60)
            self.assertEqual(result["wind_speed"], 10)
            self.assertEqual(result["timestamp"], "2023-04-12T12:00:00Z")
            
            # Verify logger was called
            self.tool.logger.info.assert_called_with(
                f"Fetching weather for {self.valid_location} in {self.valid_units_celsius}"
            )

    @patch('random.randint')
    @patch('random.choice')
    def test_run_with_fahrenheit(self, mock_choice, mock_randint):
        """Test the run method with fahrenheit units."""
        # Set up mocks
        mock_randint.side_effect = [25, 60, 10]  # temp, humidity, wind
        mock_choice.return_value = "Cloudy"
        
        # Mock timestamp
        with patch.object(self.tool, '_get_current_timestamp') as mock_timestamp:
            mock_timestamp.return_value = "2023-04-12T12:00:00Z"
            
            # Run the tool
            result = self.tool.run(self.valid_location, self.valid_units_fahrenheit)
            
            # Verify result structure
            for field in self.expected_weather_fields:
                self.assertIn(field, result)
                
            # Verify specific values
            self.assertEqual(result["location"], self.valid_location)
            # 25°C converted to fahrenheit: 25 * 9/5 + 32 = 77.0
            self.assertEqual(result["temperature"], 77.0)
            self.assertEqual(result["units"], self.valid_units_fahrenheit)
            self.assertEqual(result["conditions"], "Cloudy")
            self.assertEqual(result["humidity"], 60)
            self.assertEqual(result["wind_speed"], 10)

    @patch('random.randint')
    @patch('random.choice')
    @patch('random.uniform')
    @patch('random.random')
    def test_run_with_forecast(self, mock_random, mock_uniform, mock_choice, mock_randint):
        """Test the run method with forecast included."""
        # Set up mocks
        mock_randint.side_effect = [25, 60, 10, 55, 20, 65, 15, 70, 5, 75, 25, 80, 0]
        mock_choice.side_effect = ["Sunny", "Partly Cloudy", "Sunny", "Partly Cloudy", "Rainy"]
        mock_uniform.side_effect = [-2, 1, 0.5, -1, 2]
        mock_random.return_value = 0.8  # Above 0.7 to change conditions
        
        # Mock helper methods
        with patch.object(self.tool, '_get_current_timestamp') as mock_timestamp:
            with patch.object(self.tool, '_get_day_name') as mock_day_name:
                mock_timestamp.return_value = "2023-04-12T12:00:00Z"
                mock_day_name.side_effect = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                
                # Run the tool with forecast
                result = self.tool.run(
                    self.valid_location, 
                    self.valid_units_celsius, 
                    include_forecast=True
                )
                
                # Verify forecast is included
                self.assertIn("forecast", result)
                self.assertEqual(len(result["forecast"]), 5)
                
                # Verify forecast structure for each day
                for day_forecast in result["forecast"]:
                    for field in self.expected_forecast_fields:
                        self.assertIn(field, day_forecast)
                
                # Verify specific values for the first day
                first_day = result["forecast"][0]
                self.assertEqual(first_day["day"], 1)
                self.assertEqual(first_day["day_name"], "Monday")
                self.assertEqual(first_day["temperature"], 23.0)  # 25 - 2
                self.assertEqual(first_day["conditions"], "Partly Cloudy")
                
                # Verify the logger was called for forecast
                self.tool.logger.debug.assert_called()

    def test_invalid_location(self):
        """Test that an appropriate error is raised for an invalid location."""
        invalid_locations = [None, "", 123, []]
        
        for location in invalid_locations:
            with self.subTest(location=location):
                with self.assertRaises(ToolError) as context:
                    self.tool.run(location)
                
                # Verify error details
                self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
                self.assertIn("Location must be", str(context.exception))

    def test_invalid_units(self):
        """Test that an appropriate error is raised for invalid units."""
        invalid_units = ["kelvin", "c", "f", "CELSIUS", 123, None]
        
        for units in invalid_units:
            with self.subTest(units=units):
                with self.assertRaises(ToolError) as context:
                    self.tool.run(self.valid_location, units)
                
                # Verify error details
                self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
                self.assertIn("Invalid units", str(context.exception))
                
                # Verify error context contains valid units
                if hasattr(context.exception, 'context'):
                    self.assertIn("valid_units", context.exception.context)
                    self.assertIn("celsius", context.exception.context["valid_units"])
                    self.assertIn("fahrenheit", context.exception.context["valid_units"])

    def test_get_current_timestamp(self):
        """Test the _get_current_timestamp helper method."""
        # Mock datetime.now to return a fixed date
        fixed_date = datetime(2023, 4, 12, 12, 0, 0)
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = fixed_date
            
            # Call the method
            result = self.tool._get_current_timestamp()
            
            # Verify the result
            self.assertEqual(result, "2023-04-12T12:00:00Z")

    def test_get_day_name(self):
        """Test the _get_day_name helper method."""
        # Mock datetime to return a fixed date (Wednesday)
        fixed_date = datetime(2023, 4, 12)  # A Wednesday
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = fixed_date
            # Mock timedelta to avoid dependency on it
            mock_datetime.timedelta.side_effect = lambda days: days
            
            # Mock strftime to return day names
            mock_date = MagicMock()
            mock_datetime.return_value = mock_date
            mock_date.strftime.side_effect = [
                "Thursday", "Friday", "Saturday", "Sunday", "Monday"
            ]
            
            # Test for each day offset
            expected_days = ["Thursday", "Friday", "Saturday", "Sunday", "Monday"]
            for i, expected in enumerate(expected_days, 1):
                result = self.tool._get_day_name(i)
                self.assertEqual(result, expected)

    @patch('random.randint')
    @patch('random.choice')
    @patch('random.uniform')
    @patch('random.random')
    def test_generate_forecast(self, mock_random, mock_uniform, mock_choice, mock_randint):
        """Test the _generate_forecast helper method directly."""
        # Set up mocks
        mock_randint.side_effect = [55, 20, 65, 15, 70, 5, 75, 25, 80, 0]
        mock_choice.side_effect = ["Sunny", "Partly Cloudy", "Sunny", "Partly Cloudy", "Rainy"]
        mock_uniform.side_effect = [-2, 1, 0.5, -1, 2]
        mock_random.return_value = 0.8  # Above 0.7
        
        # Mock day name helper
        with patch.object(self.tool, '_get_day_name') as mock_day_name:
            mock_day_name.side_effect = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            
            # Call the method directly
            result = self.tool._generate_forecast(20, "celsius")
            
            # Verify result structure
            self.assertEqual(len(result), 5)
            for day_forecast in result:
                for field in self.expected_forecast_fields:
                    self.assertIn(field, day_forecast)
            
            # Check specific values
            self.assertEqual(result[0]["day"], 1)
            self.assertEqual(result[0]["day_name"], "Monday")
            self.assertEqual(result[0]["temperature"], 18.0)  # 20 - 2
            self.assertEqual(result[0]["conditions"], "Sunny")

            # Test fahrenheit conversion
            result_f = self.tool._generate_forecast(20, "fahrenheit")
            # 20 - 2 = 18°C, converted to F: 18 * 9/5 + 32 = 64.4°F
            celsius_temp = 20 - 2  # Mocked uniform returns -2
            expected_f_temp = celsius_temp * 9/5 + 32
            self.assertEqual(result_f[0]["temperature"], round(expected_f_temp, 1))


if __name__ == '__main__':
    unittest.main()