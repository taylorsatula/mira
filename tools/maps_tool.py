"""
Maps API integration tool.

This tool enables the bot to interact with Maps APIs to resolve
natural language location queries to coordinates, retrieve place details,
and perform geocoding operations.

Requires Maps API key in the config file.
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Union

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context
from config import config


class MapsTool(Tool):
    """
    Tool for interacting with Maps APIs to resolve locations and places.

    Features:
    1. Geocoding:
       - Convert natural language queries to lat/long coordinates
       - Resolve place names to specific locations
       - Support for structured and unstructured address inputs

    2. Place Details:
       - Get detailed information about places
       - Retrieve business information, opening hours, ratings
       - Get contact information for businesses

    3. Reverse Geocoding:
       - Convert coordinates to formatted addresses
       - Get neighborhood, city, state information from coordinates
    """

    name = "maps_tool"
    description = """Provides comprehensive location intelligence and geographical services through Maps API integration.

This tool enables interaction with mapping services for various location-based operations including:

1. geocode: Convert natural language locations to precise coordinates.
   - Requires 'query' parameter with address, landmark, or place name
   - Returns formatted address, geographic coordinates, and place information

2. reverse_geocode: Convert coordinates to address information.
   - Requires 'lat' and 'lng' parameters as decimal degrees
   - Returns detailed address components for the specified location

3. place_details: Retrieve comprehensive information about specific places.
   - Requires 'place_id' parameter (unique place identifier)
   - Returns name, address, phone number, website, opening hours, and other details

4. places_nearby: Discover places around a specific location.
   - Requires 'lat' and 'lng' parameters for center point
   - Optional parameters: 'radius' (default 1000m), 'type' (e.g., restaurant), 'keyword', 'open_now'
   - Returns matching places sorted by proximity with ratings and details

5. find_place: Locate specific places by name or description.
   - Requires 'query' parameter with place name
   - Returns precise match results with location data

6. calculate_distance: Determine distance between two geographic points.
   - Requires 'lat1', 'lng1', 'lat2', 'lng2' parameters
   - Returns distance in meters, kilometers, and miles

Use this tool for any task requiring location resolution, place discovery, geocoding, or geographic calculations."""
    usage_examples = [
        # # Example 1: Geocode an address
        # {
        #     "input": {
        #         "operation": "geocode",
        #         "query": "Eiffel Tower, Paris"
        #     },
        #     "output": {
        #         "results": [
        #             {
        #                 "formatted_address": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
        #                 "place_id": "ChIJLU7jZClu5kcR4PcOOO6p3I0",
        #                 "location": {
        #                     "lat": 48.8583701,
        #                     "lng": 2.2944813
        #                 },
        #                 "types": ["tourist_attraction", "point_of_interest", "establishment"]
        #             }
        #         ]
        #     }
        # },
        # # Example 2: Get place details
        # {
        #     "input": {
        #         "operation": "place_details",
        #         "place_id": "ChIJLU7jZClu5kcR4PcOOO6p3I0"
        #     },
        #     "output": {
        #         "name": "Eiffel Tower",
        #         "formatted_address": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
        #         "formatted_phone_number": "+33 892 70 12 39",
        #         "location": {
        #             "lat": 48.8583701,
        #             "lng": 2.2944813
        #         },
        #         "opening_hours": {
        #             "open_now": True,
        #             "periods": [
        #                 {
        #                     "open": {"day": 0, "time": "0930"},
        #                     "close": {"day": 0, "time": "2245"}
        #                 }
        #             ]
        #         },
        #         "rating": 4.6,
        #         "website": "https://www.toureiffel.paris/en"
        #     }
        # },
        # # Example 3: Reverse geocode coordinates
        # {
        #     "input": {
        #         "operation": "reverse_geocode",
        #         "lat": 48.8583701,
        #         "lng": 2.2944813
        #     },
        #     "output": {
        #         "results": [
        #             {
        #                 "formatted_address": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
        #                 "place_id": "ChIJLU7jZClu5kcR4PcOOO6p3I0",
        #                 "types": ["tourist_attraction", "point_of_interest", "establishment"]
        #             }
        #         ]
        #     }
        # },
        # # Example 4: Places nearby search
        # {
        #     "input": {
        #         "operation": "places_nearby",
        #         "lat": 48.8583701, 
        #         "lng": 2.2944813,
        #         "radius": 1000,
        #         "type": "restaurant"
        #     },
        #     "output": {
        #         "results": [
        #             {
        #                 "name": "Le Jules Verne",
        #                 "place_id": "ChIJe8hLsClw5kcRCA4gYfL5Pxc",
        #                 "vicinity": "Avenue Gustave Eiffel, Paris",
        #                 "types": ["restaurant", "point_of_interest", "establishment"],
        #                 "rating": 4.1
        #             }
        #         ]
        #     }
        # },
        # # Example 5: Find place operation
        # {
        #     "input": {
        #         "operation": "find_place",
        #         "query": "Huntsville Hospital"
        #     },
        #     "output": {
        #         "results": [
        #             {
        #                 "name": "Huntsville Hospital",
        #                 "place_id": "ChIJJZZlLNdrYogR44PRuZ4EXU8",
        #                 "formatted_address": "101 Sivley Rd SW, Huntsville, AL 35801, USA",
        #                 "location": {
        #                     "lat": 34.7211561,
        #                     "lng": -86.5807587
        #                 }
        #             }
        #         ]
        #     }
        # },
        # # Example 6: Calculate distance between two points
        # {
        #     "input": {
        #         "operation": "calculate_distance",
        #         "lat1": 34.7211561,
        #         "lon1": -86.5807587,
        #         "lat2": 34.7304944,
        #         "lon2": -86.5860382
        #     },
        #     "output": {
        #         "distance_meters": 1057.84,
        #         "distance_kilometers": 1.05784,
        #         "distance_miles": 0.6573
        #     }
        # }
    ]

    def __init__(self):
        """Initialize the Google Maps tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._client = None

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

        
    def _geocode(self, query: str) -> List[Dict[str, Any]]:
        """
        Convert a natural language query to geographic coordinates.

        Args:
            query: Address, landmark name, or place description

        Returns:
            List of matching locations with coordinates and other details
        """
        try:
            # Get geocoding parameters
            params = {"address": query}
            
            # Call geocoding API
            results = self.client.geocode(**params)
            processed_results = []

            for result in results:
                # Extract and format the important information
                processed_result = {
                    "formatted_address": result.get("formatted_address", ""),
                    "place_id": result.get("place_id", ""),
                    "location": result.get("geometry", {}).get("location", {}),
                    "types": result.get("types", [])
                }
                processed_results.append(processed_result)

            return processed_results
        except Exception as e:
            self.logger.error(f"Geocoding error: {e}")
            raise ToolError(f"Failed to geocode query: {e}", ErrorCode.TOOL_EXECUTION_ERROR)

    def _reverse_geocode(self, lat: float, lng: float) -> List[Dict[str, Any]]:
        """
        Convert geographic coordinates to an address.

        Args:
            lat: Latitude
            lng: Longitude

        Returns:
            List of address information for the coordinates
        """
        try:
            results = self.client.reverse_geocode((lat, lng))
            processed_results = []

            for result in results:
                # Extract and format the important information
                processed_result = {
                    "formatted_address": result.get("formatted_address", ""),
                    "place_id": result.get("place_id", ""),
                    "types": result.get("types", [])
                }
                processed_results.append(processed_result)

            return processed_results
        except Exception as e:
            self.logger.error(f"Reverse geocoding error: {e}")
            raise ToolError(
                f"Failed to reverse geocode coordinates: {e}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )

    def _place_details(self, place_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a place.

        Args:
            place_id: Google Places ID

        Returns:
            Dict containing place details
        """
        try:
            result = self.client.place(place_id=place_id)
            
            # Extract place details from the result
            if "result" in result:
                place = result["result"]
                
                # Extract the necessary details
                details = {
                    "name": place.get("name", ""),
                    "formatted_address": place.get("formatted_address", ""),
                    "formatted_phone_number": place.get("formatted_phone_number", ""),
                    "international_phone_number": place.get("international_phone_number", ""),
                    "website": place.get("website", ""),
                    "url": place.get("url", ""),
                    "rating": place.get("rating", 0),
                    "types": place.get("types", []),
                }
                
                # Add location if available
                if "geometry" in place and "location" in place["geometry"]:
                    details["location"] = place["geometry"]["location"]
                
                # Add opening hours if available
                if "opening_hours" in place:
                    details["opening_hours"] = place["opening_hours"]
                
                return details
            else:
                raise ToolError(
                    f"No details found for place ID: {place_id}",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
        except Exception as e:
            self.logger.error(f"Place details error: {e}")
            raise ToolError(
                f"Failed to get place details: {e}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )

    def _places_nearby(
        self, 
        lat: float, 
        lng: float, 
        radius: int = 1000,
        keyword: Optional[str] = None,
        type: Optional[str] = None,
        language: Optional[str] = None,
        open_now: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Find places near a specific location.

        Args:
            lat: Latitude of the center point
            lng: Longitude of the center point
            radius: Search radius in meters (default 1000)
            keyword: Keywords describing the place
            type: Type of place (e.g., restaurant, cafe)
            language: Language for results
            open_now: Whether to return only places open at the time of request

        Returns:
            List of places near the specified location
        """
        try:
            # Prepare search parameters
            params = {
                "location": (lat, lng),
                "radius": radius
            }
            
            # Add optional parameters if provided
            if keyword:
                params["keyword"] = keyword
            if type:
                params["type"] = type
            if language:
                params["language"] = language
            if open_now is not None:
                params["open_now"] = open_now
                
            # Perform the search
            results = self.client.places_nearby(**params)
            
            # Process and return the results
            if "results" in results:
                processed_results = []
                
                for place in results["results"]:
                    processed_place = {
                        "name": place.get("name", ""),
                        "place_id": place.get("place_id", ""),
                        "vicinity": place.get("vicinity", ""),
                        "types": place.get("types", []),
                    }
                    
                    # Add location if available
                    if "geometry" in place and "location" in place["geometry"]:
                        processed_place["location"] = place["geometry"]["location"]
                    
                    # Add rating if available
                    if "rating" in place:
                        processed_place["rating"] = place["rating"]
                    
                    # Add open_now if available
                    if "opening_hours" in place and "open_now" in place["opening_hours"]:
                        processed_place["open_now"] = place["opening_hours"]["open_now"]
                        
                    processed_results.append(processed_place)
                    
                return processed_results
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Places nearby search error: {e}")
            raise ToolError(
                f"Failed to search nearby places: {e}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )

    def _find_place(self, query: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Find a specific place using a text query (more precise than places_text_search).
        
        Args:
            query: The text identifying the place (name, address, etc.)
            fields: Optional fields to include in the result
            
        Returns:
            Dictionary containing place information
        """
        try:
            # Default fields if not specified
            if fields is None:
                fields = [
                    'place_id', 'name', 'formatted_address', 'geometry', 
                    'types', 'business_status', 'rating'
                ]
            
            # Prepare parameters
            params = {
                "input": query,
                "input_type": "textquery",
                "fields": fields
            }
            
            # Call Find Place API
            results = self.client.find_place(**params)
            
            # Process and return results
            if "candidates" in results and results["candidates"]:
                processed_results = []
                
                for place in results["candidates"]:
                    processed_place = {
                        "name": place.get("name", ""),
                        "place_id": place.get("place_id", ""),
                        "formatted_address": place.get("formatted_address", ""),
                        "types": place.get("types", []),
                    }
                    
                    # Add location if available
                    if "geometry" in place and "location" in place["geometry"]:
                        processed_place["location"] = place["geometry"]["location"]
                    
                    # Add rating if available
                    if "rating" in place:
                        processed_place["rating"] = place["rating"]
                        
                    processed_results.append(processed_place)
                    
                return {"results": processed_results}
            else:
                return {"results": []}
                
        except Exception as e:
            self.logger.error(f"Find place error: {e}")
            raise ToolError(
                f"Failed to find place: {e}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _places_text_search(self, query: str, location: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Search for places using a text query.

        Args:
            query: Search query text
            location: Optional (latitude, longitude) tuple to bias results

        Returns:
            List of places matching the search query
        """
        try:
            # Prepare search parameters
            params = {"query": query}
            
            # Add location for result biasing if provided
            if location:
                params["location"] = location
                    
            # Perform the search
            results = self.client.places(**params)
            
            # Process and return the results
            if "results" in results:
                processed_results = []
                
                for place in results["results"]:
                    processed_place = {
                        "name": place.get("name", ""),
                        "place_id": place.get("place_id", ""),
                        "formatted_address": place.get("formatted_address", ""),
                        "types": place.get("types", []),
                    }
                    
                    # Add location if available
                    if "geometry" in place and "location" in place["geometry"]:
                        processed_place["location"] = place["geometry"]["location"]
                    
                    # Add rating if available
                    if "rating" in place:
                        processed_place["rating"] = place["rating"]
                        
                    processed_results.append(processed_place)
                    
                return processed_results
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Places text search error: {e}")
            raise ToolError(
                f"Failed to perform text search: {e}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )

    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        Calculate the great circle distance between two points on the Earth.
        Uses the haversine formula.
        
        Args:
            lat1: Latitude of point 1 (in degrees)
            lng1: Longitude of point 1 (in degrees)
            lat2: Latitude of point 2 (in degrees)
            lng2: Longitude of point 2 (in degrees)
            
        Returns:
            Distance in meters between the points
        """
        import math
        
        # Convert decimal degrees to radians
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        # Radius of Earth in meters
        r = 6371000
        return c * r
        
    def run(
        self,
        operation: str,
        query: Optional[str] = None,
        place_id: Optional[str] = None,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        lat1: Optional[float] = None,
        lng1: Optional[float] = None,
        lat2: Optional[float] = None,
        lng2: Optional[float] = None,
        radius: Optional[int] = None,
        type: Optional[str] = None,
        keyword: Optional[str] = None,
        open_now: Optional[bool] = None,
        language: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Google Maps API operation.

        Args:
            operation: Operation to perform (see below for valid operations)
            query: Search query for geocoding or places text search
            place_id: Google Places ID for place details
            lat: Latitude for operations requiring coordinates
            lng: Longitude for operations requiring coordinates
            lat1: First latitude for distance calculations
            lng1: First longitude for distance calculations 
            lat2: Second latitude for distance calculations
            lng2: Second longitude for distance calculations
            radius: Search radius in meters for nearby places
            type: Type of place (e.g., restaurant, cafe)
            keyword: Search keywords
            open_now: Filter results to those open at request time
            language: Language for results
            fields: Specific fields to request from the API (for find_place)

        Returns:
            Response data for the operation

        Raises:
            ToolError: If operation fails or parameters are invalid

        Valid Operations:

        1. geocode: Convert natural language query to coordinates
           - Required: query (address, landmark name, or place description)
           - Returns: List of matching locations with coordinates

        2. reverse_geocode: Convert coordinates to address
           - Required: lat, lng
           - Returns: List of address information for the coordinates

        3. place_details: Get detailed information about a place
           - Required: place_id
           - Returns: Dict with place details including address, phone, hours, etc.

        4. places_nearby: Find places near a specific location
           - Required: lat, lng
           - Optional: radius (default 1000m), type, keyword, open_now, language
           - Returns: List of places near the specified location

        5. places_text_search: Search for places using a text query
           - Required: query
           - Optional: lat, lng (for result biasing)
           - Returns: List of places matching the search query
           
        6. find_place: Find a specific place using a text query (most precise option)
           - Required: query (name of place, business, landmark, etc.)
           - Optional: fields (list of specific fields to request)
           - Returns: List of matching places with detailed information
           
        7. calculate_distance: Calculate the distance between two geographic points
           - Required: lat1, lng1, lat2, lng2 (coordinates in decimal degrees)
           - Returns: Distance in meters between the points
        """
        with error_context(
            component_name=self.name,
            operation=f"executing {operation}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger,
        ):
            # Check for missing coordinates in places operations where they're required
            coordinate_required_operations = ['places_nearby', 'reverse_geocode']
            if operation in coordinate_required_operations and (lat is None or lng is None):
                return {
                    "error": "This operation requires location data. Please allow location access in your app.",
                    "requires_location": True
                }
            
            # Handle JSON string passed in 'params' field
            if "params" in operation and isinstance(operation, dict) and isinstance(operation["params"], str):
                try:
                    params = json.loads(operation["params"])
                    operation = params.get("operation", operation)
                    query = params.get("query", query)
                    place_id = params.get("place_id", place_id)
                    lat = params.get("lat", lat)
                    lng = params.get("lng", lng)
                    radius = params.get("radius", radius)
                    type = params.get("type", type)
                    keyword = params.get("keyword", keyword)
                    open_now = params.get("open_now", open_now)
                    language = params.get("language", language)
                except json.JSONDecodeError as e:
                    raise ToolError(
                        f"Invalid JSON in params: {e}", ErrorCode.TOOL_INVALID_INPUT
                    )

            # Geocoding Operations
            if operation == "geocode":
                if not query:
                    raise ToolError(
                        "query parameter is required for geocode operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                results = self._geocode(query)
                return {"results": results}

            elif operation == "reverse_geocode":
                if lat is None or lng is None:
                    raise ToolError(
                        "lat and lng parameters are required for reverse_geocode operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                results = self._reverse_geocode(lat, lng)
                return {"results": results}

            # Place Operations
            elif operation == "place_details":
                if not place_id:
                    raise ToolError(
                        "place_id parameter is required for place_details operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self._place_details(place_id)

            elif operation == "places_nearby":
                if lat is None or lng is None:
                    raise ToolError(
                        "lat and lng parameters are required for places_nearby operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                results = self._places_nearby(
                    lat=lat,
                    lng=lng,
                    radius=radius or 1000,
                    keyword=keyword,
                    type=type,
                    language=language,
                    open_now=open_now
                )
                return {"results": results}

            elif operation == "places_text_search":
                if not query:
                    raise ToolError(
                        "query parameter is required for places_text_search operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                location = (lat, lng) if lat is not None and lng is not None else None
                results = self._places_text_search(
                    query=query, 
                    location=location
                )
                return {"results": results}

            elif operation == "find_place":
                if not query:
                    raise ToolError(
                        "query parameter is required for find_place operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self._find_place(
                    query=query, 
                    fields=fields
                )

            elif operation == "calculate_distance":
                if lat1 is None or lng1 is None or lat2 is None or lng2 is None:
                    raise ToolError(
                        "lat1, lng1, lat2, lng2 parameters are required for calculate_distance operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                
                try:
                    # Convert to float if they aren't already
                    lat1_val = float(lat1)
                    lng1_val = float(lng1)
                    lat2_val = float(lat2)
                    lng2_val = float(lng2)
                    
                    # Calculate distance
                    distance = self._haversine_distance(lat1_val, lng1_val, lat2_val, lng2_val)
                    
                    return {
                        "distance_meters": distance,
                        "distance_kilometers": distance / 1000,
                        "distance_miles": distance / 1609.34
                    }
                except ValueError:
                    raise ToolError(
                        "Coordinates must be valid numbers",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                
            else:
                raise ToolError(
                    f"Unknown operation: {operation}. Valid operations are: "
                    "geocode, reverse_geocode, place_details, places_nearby, places_text_search, find_place, calculate_distance",
                    ErrorCode.TOOL_INVALID_INPUT,
                )