"""
Natural Language Directions tool implementation.

This tool enhances the standard turn-by-turn directions with human-like natural
language descriptions that provide road context (e.g., "turn off the main road onto
the side street") rather than just standard instructions.

Requires Google Maps API key in the config file.
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple

from pydantic import BaseModel, Field
from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context
from config.registry import registry


# Define configuration class for DirectionsTool
class DirectionsToolConfig(BaseModel):
    """Configuration for the directions_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    timeout: int = Field(default=60, description="Timeout in seconds for Google Maps API requests")
    include_landmarks: bool = Field(default=True, description="Whether to include landmarks in enhanced directions")
    cache_timeout: int = Field(default=3600, description="Cache timeout in seconds (default: 1 hour)")


# Register with registry
registry.register("directions_tool", DirectionsToolConfig)


class DirectionsTool(Tool):
    """
    Tool for generating enhanced natural language directions.
    
    This tool transforms standard Google Maps turn-by-turn directions 
    into more natural, context-aware instructions. It detects road 
    classifications (main roads vs. side streets), road transitions, 
    and optionally includes landmarks to create directions that sound 
    more like how a human would give them.
    
    Features:
    1. Road Classification:
       - Identifies road hierarchy (highways, main roads, side streets)
       - Detects transitions between different road types
    
    2. Natural Language Directions:
       - Converts standard "Turn left onto X Street" to contextual instructions
       - Includes descriptive elements about the roads
    
    3. Optional Landmark Integration:
       - References nearby landmarks at decision points
       - Prioritizes highly visible landmarks for easier navigation
    """

    name = "directions_tool"
    description = """Provides natural language directions that describe routes in a more human-like way. Use this tool when the user needs directions with contextual information rather than just standard turn-by-turn instructions.

This tool enhances standard mapping directions to include road context and optional landmarks:

1. get_enhanced_directions: Get natural language directions between two locations.
   - Requires 'origin' and 'destination' parameters with addresses or coordinates
   - Optional 'include_landmarks' parameter to reference nearby landmarks (default: true)
   - Returns enhanced directions with contextual information about roads

Example enhanced directions:
- Standard: "Turn right onto Main Street"
- Enhanced: "Turn right off the side street onto the main road (Main Street)"
- With landmark: "Turn right at the Starbucks, onto the main road (Main Street)"

This provides a more intuitive navigation experience similar to how humans naturally give directions."""
    usage_examples = [
        {
            "input": {
                "origin": "123 Main St, Anytown, USA",
                "destination": "456 Oak Ave, Anytown, USA"
            },
            "output": {
                "origin": "123 Main St, Anytown, USA",
                "destination": "456 Oak Ave, Anytown, USA",
                "total_distance": "2.5 mi",
                "total_duration": "8 mins",
                "steps": [
                    {
                        "instruction": "Head south on Main Street (a main road)",
                        "distance": "0.2 mi",
                        "duration": "1 min"
                    },
                    {
                        "instruction": "Turn right off the main road onto the side street (Oak Avenue)",
                        "distance": "0.3 mi",
                        "duration": "2 mins"
                    }
                ]
            }
        }
    ]

    def __init__(self):
        """Initialize the Directions tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._client = None
        
        # Define road classification hierarchy
        self.road_hierarchy = {
            "highway": ["highway", "motorway", "freeway", "expressway", "interstate"],
            "main_road": ["primary", "trunk", "arterial", "major", "avenue", "boulevard", "main"],
            "secondary_road": ["secondary", "collector", "thoroughfare"],
            "side_street": ["residential", "local", "side", "minor", "lane", "way", "drive", "circle"]
        }

    @property
    def client(self):
        """
        Get the Google Maps client, initializing it if needed.
        Lazy loading approach.

        Returns:
            Google Maps client instance

        Raises:
            ToolError: If Google Maps API key is not set or client initialization fails
        """
        if self._client is None:
            try:
                from googlemaps import Client

                # Get API key from config
                from config import config
                api_key = config.google_maps_api_key
                if not api_key:
                    raise ToolError(
                        "Google Maps API key not found in configuration.",
                        ErrorCode.TOOL_EXECUTION_ERROR,
                    )

                # Create client with API key
                self.logger.info("Creating Google Maps client with API key")
                self._client = Client(key=api_key)
            except ImportError:
                raise ToolError(
                    "googlemaps library not installed. Run: pip install googlemaps",
                    ErrorCode.TOOL_INITIALIZATION_ERROR,
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize Google Maps client: {e}")
                raise ToolError(
                    f"Failed to initialize Google Maps client: {e}",
                    ErrorCode.TOOL_EXECUTION_ERROR,
                )
        return self._client

    def run(
        self,
        origin: str,
        destination: str,
        include_landmarks: bool = None,
        travel_mode: str = "driving"
    ) -> Dict[str, Any]:
        """
        Get enhanced natural language directions between two locations.

        This method serves as the main entry point for the tool. It retrieves
        standard directions from Google Maps API and enhances them with natural
        language descriptions that include road context and optional landmarks.

        Args:
            origin: Starting location (address or coordinates)
            destination: Ending location (address or coordinates)
            include_landmarks: Whether to enhance with landmark references (defaults to True)
            travel_mode: Mode of transportation (driving, walking, bicycling, transit)

        Returns:
            Dictionary containing enhanced directions with the following structure:
            {
                "origin": str,
                "destination": str,
                "total_distance": str,
                "total_duration": str,
                "steps": List[Dict] containing enhanced instructions
            }

        Raises:
            ToolError: If directions cannot be retrieved or enhanced
        """
        # Import config
        from config import config
        
        # Use the configured default for landmarks if none provided
        if include_landmarks is None:
            include_landmarks = config.directions_tool.include_landmarks
            
        self.logger.info(f"Getting enhanced directions from {origin} to {destination}")
        
        with error_context(
            component_name=self.name,
            operation="getting enhanced directions",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger,
        ):
            # Validate inputs
            if not origin or not destination:
                raise ToolError(
                    "Both origin and destination are required parameters",
                    ErrorCode.TOOL_INVALID_INPUT,
                )
                
            # Validate travel mode
            valid_modes = ["driving", "walking", "bicycling", "transit"]
            if travel_mode not in valid_modes:
                raise ToolError(
                    f"Invalid travel mode: {travel_mode}. Must be one of {valid_modes}",
                    ErrorCode.TOOL_INVALID_INPUT,
                )
            
            # Get directions from Google Maps API
            # The actual implementation will be added in subsequent tasks
            
            # Temporary placeholder return for now
            return {
                "origin": origin,
                "destination": destination,
                "total_distance": "Placeholder",
                "total_duration": "Placeholder",
                "steps": []
            }