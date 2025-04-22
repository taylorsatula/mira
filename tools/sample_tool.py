import random
from typing import List, Dict, Any

from tools.repo import Tool
from errors import ErrorCode, error_context, ToolError


class WeatherTool(Tool):
    """
    Weather information retrieval tool for demonstration purposes.
    
    This tool serves as a reference implementation demonstrating proper structure
    and best practices for creating tools in the bot framework. It provides
    simulated weather data for any location, including current conditions and
    optional forecast data.

    Key features:
    1. Parameter validation with helpful error messages
    2. Proper error handling with the error_context pattern
    3. Clean separation of public interface and private helper methods
    4. Comprehensive documentation with detailed docstrings
    5. Logging of operations for debugging and auditing
    """

    name = "weather_tool"
    description = """
    Retrieves detailed weather information for any specified location. This tool provides current 
    weather data including temperature, conditions, humidity, and wind speed. Use this tool whenever 
    the user asks about weather conditions for a specific location or needs weather-related information 
    for planning purposes.
    
    The 'location' parameter accepts any city or region name and is required. The 'units' parameter 
    controls temperature format and defaults to celsius but can be set to fahrenheit. The optional 
    'include_forecast' parameter, when set to true, will return a 5-day weather forecast in addition 
    to current conditions.
    
    Note that this tool generates fictional weather data and should not be used for actual weather 
    planning or emergency situations. Response times are typically under 1 second. The tool does not 
    provide historical weather data or severe weather warnings.
    """
    usage_examples = [
        {
            "input": {"location": "New York", "units": "celsius"},
            "output": {
                "temperature": 22,
                "conditions": "Partly Cloudy",
                "humidity": 65,
                "wind_speed": 10
            }
        }
    ]

    def __init__(self):
        """
        Initialize the weather tool with sample data and configuration.
        
        Sets up the tool with predefined weather conditions and initializes
        logging for tracking operations. This simple initialization demonstrates
        proper setup patterns for more complex tools.
        """
        super().__init__()
        
        # Tool-specific state - predefined conditions for simulation
        self.weather_conditions = [
            "Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Thunderstorm",
            "Snowy", "Foggy", "Windy", "Clear"
        ]
        
        # Initialize logger
        self.logger.info("WeatherTool initialized with simulated data capabilities")

    def run(
        self,
        location: str,
        units: str = "celsius",
        include_forecast: bool = False
    ) -> Dict[str, Any]:
        """
        Get weather information for a specified location.

        This method serves as the main entry point for the tool. It validates inputs,
        generates weather data based on the provided parameters, and returns a structured
        response. The method demonstrates proper error handling and input validation
        patterns.

        Args:
            location: The city or location to get weather for (required)
            units: Temperature units ('celsius' or 'fahrenheit', defaults to 'celsius')
            include_forecast: Whether to include a 5-day forecast (defaults to False)

        Returns:
            Dictionary containing weather data with the following structure:
            {
                "location": str,
                "temperature": float,
                "units": str,
                "conditions": str,
                "humidity": int,
                "wind_speed": int,
                "forecast": list (optional)
            }

        Raises:
            ToolError: If units are invalid or other errors occur during execution
        """
        self.logger.info(f"Fetching weather for {location} in {units}")

        # Use the centralized error context for weather data generation
        with error_context(
            component_name=self.name,
            operation="generating weather data",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_INVALID_INPUT,
            logger=self.logger
        ):
            # Input validation
            if not location or not isinstance(location, str):
                raise ToolError(
                    "Location must be a non-empty string",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"provided_location": str(location)}
                )
            
            # Validate temperature units
            if units not in ["celsius", "fahrenheit"]:
                raise ToolError(
                    f"Invalid units: {units}. Must be 'celsius' or 'fahrenheit'",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"provided_units": units, "valid_units": ["celsius", "fahrenheit"]}
                )

            # Generate random weather data
            base_temp = random.randint(5, 35)
            temp = base_temp if units == "celsius" else base_temp * 9/5 + 32

            # Build response with detailed weather information
            weather_data = {
                "location": location,
                "temperature": round(temp, 1),
                "units": units,
                "conditions": random.choice(self.weather_conditions),
                "humidity": random.randint(30, 90),
                "wind_speed": random.randint(0, 30),
                "timestamp": self._get_current_timestamp()
            }

            # Add forecast if requested
            if include_forecast:
                self.logger.debug(f"Including forecast data for {location}")
                weather_data["forecast"] = self._generate_forecast(base_temp, units)

            return weather_data

    def _get_current_timestamp(self) -> str:
        """
        Generate a timestamp for the current weather data.
        
        This helper method demonstrates proper separation of concerns by
        moving functionality into discrete, testable methods.
        
        Returns:
            String representation of current timestamp in ISO format
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _generate_forecast(self, base_temp: float, units: str) -> List[Dict[str, Any]]:
        """
        Generate a fictional 5-day weather forecast.
        
        This private helper method creates weather predictions based on the current
        temperature, with realistic day-to-day variations. It demonstrates proper 
        method organization where complex functionality is separated from the main
        interface.

        Args:
            base_temp: Base temperature in celsius to build forecast around
            units: Temperature units ('celsius' or 'fahrenheit')

        Returns:
            List of forecast data dictionaries, each containing:
            - day: int (day number, 1-5)
            - temperature: float (predicted temperature)
            - conditions: str (weather conditions description)
            - humidity: int (percentage)
            - precipitation_chance: int (percentage)
            - wind_speed: int (speed value)
        """
        forecast = []
        self.logger.debug(f"Generating 5-day forecast starting from base temp: {base_temp}°C")

        for day in range(1, 6):  # 5-day forecast
            # Create realistic temperature variations
            temp_change = random.uniform(-5, 5)
            day_temp = base_temp + temp_change

            if units == "fahrenheit":
                day_temp = day_temp * 9/5 + 32

            # Generate varied conditions with weighted randomness
            # For a more realistic forecast (conditions tend to persist)
            if day > 1 and random.random() < 0.7:
                # 70% chance to maintain similar conditions as previous day
                conditions = forecast[-1]["conditions"]
            else:
                conditions = random.choice(self.weather_conditions)
                
            # Add forecast data with additional details
            forecast.append({
                "day": day,
                "day_name": self._get_day_name(day),
                "temperature": round(day_temp, 1),
                "conditions": conditions,
                "humidity": random.randint(30, 90),
                "precipitation_chance": random.randint(0, 100),
                "wind_speed": random.randint(0, 30)
            })

        return forecast
        
    def _get_day_name(self, day_offset: int) -> str:
        """
        Convert a day offset to a weekday name.
        
        Args:
            day_offset: Number of days from today (1-5)
            
        Returns:
            Name of the weekday (e.g., "Monday", "Tuesday")
        """
        import datetime
        
        today = datetime.datetime.now()
        future_date = today + datetime.timedelta(days=day_offset)
        return future_date.strftime("%A")


