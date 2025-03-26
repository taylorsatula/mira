import random
from typing import List, Dict, Any

from tools.repo import Tool
from errors import ErrorCode, error_context, ToolError


class WeatherTool(Tool):
    """
    Example tool that fetches weather information.

    This is a simple demonstration tool that returns fictional
    weather data for a given location.
    """

    name = "weather_tool"
    description = "Get current weather information for a location"
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
        """Initialize the weather tool with sample data."""
        super().__init__()
        # Tool-specific state
        self.weather_conditions = [
            "Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Thunderstorm",
            "Snowy", "Foggy", "Windy", "Clear"
        ]

    def run(
        self,
        location: str,
        units: str = "celsius",
        include_forecast: bool = False
    ) -> Dict[str, Any]:
        """
        Get weather information for a location.

        Args:
            location: The city or location to get weather for
            units: Temperature units ('celsius' or 'fahrenheit')
            include_forecast: Whether to include a forecast for upcoming days

        Returns:
            Weather data as a dictionary

        Raises:
            ToolError: If units are invalid or other errors occur
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
            # Validate units
            if units not in ["celsius", "fahrenheit"]:
                raise ValueError(f"Invalid units: {units}. Must be 'celsius' or 'fahrenheit'")

            # Generate random weather data
            base_temp = random.randint(5, 35)
            temp = base_temp if units == "celsius" else base_temp * 9/5 + 32

            # Build response
            weather_data = {
                "location": location,
                "temperature": round(temp, 1),
                "units": units,
                "conditions": random.choice(self.weather_conditions),
                "humidity": random.randint(30, 90),
                "wind_speed": random.randint(0, 30),
            }

            # Add forecast if requested
            if include_forecast:
                weather_data["forecast"] = self._generate_forecast(base_temp, units)

            return weather_data

    def _generate_forecast(self, base_temp: float, units: str) -> List[Dict[str, Any]]:
        """
        Generate a fictional weather forecast.

        Args:
            base_temp: Base temperature to build forecast around
            units: Temperature units

        Returns:
            List of forecast day data
        """
        forecast = []

        for day in range(1, 6):  # 5-day forecast
            temp_change = random.uniform(-5, 5)
            day_temp = base_temp + temp_change

            if units == "fahrenheit":
                day_temp = day_temp * 9/5 + 32

            forecast.append({
                "day": day,
                "temperature": round(day_temp, 1),
                "conditions": random.choice(self.weather_conditions),
                "humidity": random.randint(30, 90),
                "precipitation_chance": random.randint(0, 100)
            })

        return forecast


