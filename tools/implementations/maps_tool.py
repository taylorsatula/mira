"""
Maps integration tool backed by OpenStreetMap.

This tool enables the bot to resolve natural language location queries to
coordinates, retrieve place details, and perform geocoding operations using
the Nominatim and Overpass APIs. No API key required.

Place IDs are OpenStreetMap element references in the form "<T><osm_id>"
where T is N (node), W (way), or R (relation), e.g. "N123456789".
"""

import logging
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field
from tools.repo import Tool
from tools.registry import registry
from utils import nominatim_client

# Define configuration class for MapsTool
class MapsToolConfig(BaseModel):
    """Configuration for the maps_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")

# Register with registry
registry.register("maps_tool", MapsToolConfig)


# --- Input Models ---

class GeocodeInput(BaseModel):
    """Input for geocode operation."""
    query: str = Field(..., min_length=1, description="Address, landmark name, or place description")


class ReverseGeocodeInput(BaseModel):
    """Input for reverse geocode operation."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lng: float = Field(..., ge=-180, le=180, description="Longitude")


class PlaceDetailsInput(BaseModel):
    """Input for place details operation."""
    place_id: str = Field(..., min_length=1, description="OpenStreetMap place ID (e.g. 'N123456789') from a geocode, find_place, or places_nearby result")


class PlacesNearbyInput(BaseModel):
    """Input for places nearby operation."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude of center point")
    lng: float = Field(..., ge=-180, le=180, description="Longitude of center point")
    radius: int = Field(default=1000, ge=1, le=50000, description="Search radius in meters")
    keyword: Optional[str] = Field(default=None, description="Keywords to match against place names")
    type: Optional[str] = Field(default=None, description="Place type filter (e.g., 'restaurant')")


class FindPlaceInput(BaseModel):
    """Input for find place operation."""
    query: str = Field(..., min_length=1, description="Place name or description")


class CalculateDistanceInput(BaseModel):
    """Input for calculate distance operation."""
    lat1: float = Field(..., ge=-90, le=90, description="First point latitude")
    lng1: float = Field(..., ge=-180, le=180, description="First point longitude")
    lat2: float = Field(..., ge=-90, le=90, description="Second point latitude")
    lng2: float = Field(..., ge=-180, le=180, description="Second point longitude")


# --- Tool Implementation ---

class MapsTool(Tool):
    """
    Tool for interacting with OpenStreetMap to resolve locations and places.

    Features:
    1. Geocoding:
       - Convert natural language queries to lat/long coordinates
       - Resolve place names to specific locations
       - Support for structured and unstructured address inputs

    2. Place Details:
       - Get detailed information about places
       - Retrieve address components, website, phone, opening hours when available

    3. Reverse Geocoding:
       - Convert coordinates to formatted addresses
       - Get neighborhood, city, state information from coordinates

    4. Nearby Search:
       - Find points of interest within a radius of a coordinate
    """

    name = "maps_tool"

    tool_schema = {
        "name": "maps_tool",
        "description": "Provides comprehensive location intelligence and geographical services through OpenStreetMap integration. Use this tool for geocoding, place details, distance calculations, and location-based searches.",
        "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "geocode",
                            "reverse_geocode",
                            "place_details",
                            "places_nearby",
                            "find_place",
                            "calculate_distance"
                        ],
                        "description": "The maps operation to perform"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for geocode, find_place operations. Can be an address, landmark name, or place description"
                    },
                    "place_id": {
                        "type": "string",
                        "description": "OpenStreetMap place ID for place_details operation (e.g. 'N123456789'), as returned in geocode, find_place, or places_nearby results"
                    },
                    "lat": {
                        "type": "number",
                        "description": "Latitude for reverse_geocode, places_nearby operations",
                        "minimum": -90,
                        "maximum": 90
                    },
                    "lng": {
                        "type": "number",
                        "description": "Longitude for reverse_geocode, places_nearby operations",
                        "minimum": -180,
                        "maximum": 180
                    },
                    "lat1": {
                        "type": "number",
                        "description": "First point latitude for calculate_distance operation",
                        "minimum": -90,
                        "maximum": 90
                    },
                    "lng1": {
                        "type": "number",
                        "description": "First point longitude for calculate_distance operation",
                        "minimum": -180,
                        "maximum": 180
                    },
                    "lat2": {
                        "type": "number",
                        "description": "Second point latitude for calculate_distance operation",
                        "minimum": -90,
                        "maximum": 90
                    },
                    "lng2": {
                        "type": "number",
                        "description": "Second point longitude for calculate_distance operation",
                        "minimum": -180,
                        "maximum": 180
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Search radius in meters for places_nearby (default: 1000)",
                        "default": 1000
                    },
                    "type": {
                        "type": "string",
                        "description": "Type of place to filter (e.g., 'restaurant', 'cafe', 'hospital') for places_nearby"
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Keywords to match against place names in places_nearby operation"
                    }
                },
                "required": ["operation"]
            }
        }

    description = "Location services: geocoding, place details, nearby search, and distance calculation"

    def __init__(self):
        """Initialize the maps tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _process_nominatim_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a Nominatim result dict into the tool's output shape."""
        processed = {
            "formatted_address": result.get("display_name", ""),
            "place_id": nominatim_client.osm_place_id(result.get("osm_type", ""), result.get("osm_id")),
            "types": [t for t in (result.get("category"), result.get("type")) if t],
        }
        if result.get("lat") is not None and result.get("lon") is not None:
            processed["location"] = {"lat": float(result["lat"]), "lng": float(result["lon"])}
        return processed

    def _geocode(self, input: GeocodeInput) -> Dict[str, Any]:
        """Convert a natural language query to geographic coordinates."""
        try:
            results = nominatim_client.search(input.query)
            return {"results": [self._process_nominatim_result(r) for r in results]}
        except Exception as e:
            self.logger.error(f"Geocoding failed for '{input.query}': {e}")
            raise ValueError(f"Failed to geocode query: {e}")

    def _reverse_geocode(self, input: ReverseGeocodeInput) -> Dict[str, Any]:
        """Convert geographic coordinates to an address."""
        try:
            result = nominatim_client.reverse(input.lat, input.lng)
            return {"results": [self._process_nominatim_result(result)]}
        except Exception as e:
            self.logger.error(f"Reverse geocoding failed for ({input.lat}, {input.lng}): {e}")
            raise ValueError(f"Failed to reverse geocode coordinates: {e}")

    def _place_details(self, input: PlaceDetailsInput) -> Dict[str, Any]:
        """Get detailed information about a place."""
        try:
            place = nominatim_client.lookup(input.place_id)
            extratags = place.get("extratags", {})

            details = self._process_nominatim_result(place)
            details["name"] = place.get("name") or place.get("display_name", "").split(",")[0]

            if extratags.get("website"):
                details["website"] = extratags["website"]
            if extratags.get("phone"):
                details["phone"] = extratags["phone"]
            if extratags.get("opening_hours"):
                details["opening_hours"] = extratags["opening_hours"]

            return details

        except Exception as e:
            self.logger.error(f"Place details failed for '{input.place_id}': {e}")
            raise ValueError(f"Failed to get place details: {e}")

    def _places_nearby(self, input: PlacesNearbyInput) -> Dict[str, Any]:
        """Find places near a specific location."""
        try:
            elements = nominatim_client.nearby(
                input.lat, input.lng, input.radius,
                place_type=input.type, keyword=input.keyword,
            )
            processed_results = []

            for element in elements:
                tags = element.get("tags", {})

                # Way/relation elements report a center; nodes report lat/lon directly.
                center = element.get("center", element)
                if center.get("lat") is None or center.get("lon") is None:
                    continue

                processed_place = {
                    "name": tags.get("name", ""),
                    "place_id": nominatim_client.osm_place_id(element.get("type", ""), element.get("id")),
                    "location": {"lat": float(center["lat"]), "lng": float(center["lon"])},
                    "types": [tags[k] for k in ("amenity", "shop", "tourism", "leisure") if k in tags],
                }

                address_parts = [tags.get("addr:street"), tags.get("addr:city")]
                processed_place["vicinity"] = ", ".join(p for p in address_parts if p)

                processed_results.append(processed_place)

            return {"results": processed_results}

        except Exception as e:
            self.logger.error(f"Places nearby search failed for ({input.lat}, {input.lng}): {e}")
            raise ValueError(f"Failed to search nearby places: {e}")

    def _find_place(self, input: FindPlaceInput) -> Dict[str, Any]:
        """Find a specific place using a text query."""
        try:
            results = nominatim_client.search(input.query)
            return {"results": [self._process_nominatim_result(r) for r in results]}
        except Exception as e:
            self.logger.error(f"Find place failed for '{input.query}': {e}")
            raise ValueError(f"Failed to find place: {e}")

    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate the great circle distance between two points using the haversine formula."""
        import math

        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Earth radius in meters
        return c * r

    def _calculate_distance(self, input: CalculateDistanceInput) -> Dict[str, Any]:
        """Calculate the distance between two geographic points."""
        distance = self._haversine_distance(input.lat1, input.lng1, input.lat2, input.lng2)
        return {
            "distance_meters": distance,
            "distance_kilometers": distance / 1000,
            "distance_miles": distance / 1609.34
        }

    def run(self, **params) -> Dict[str, Any]:
        """Route to appropriate operation handler."""
        operation = params.pop("operation", None)
        if not operation:
            raise ValueError("Required parameter 'operation' not provided")

        if operation == "geocode":
            return self._geocode(GeocodeInput(**params))
        elif operation == "reverse_geocode":
            return self._reverse_geocode(ReverseGeocodeInput(**params))
        elif operation == "place_details":
            return self._place_details(PlaceDetailsInput(**params))
        elif operation == "places_nearby":
            return self._places_nearby(PlacesNearbyInput(**params))
        elif operation == "find_place":
            return self._find_place(FindPlaceInput(**params))
        elif operation == "calculate_distance":
            return self._calculate_distance(CalculateDistanceInput(**params))
        else:
            raise ValueError(
                f"Unknown operation: {operation}. Must be: geocode, reverse_geocode, "
                "place_details, places_nearby, find_place, or calculate_distance"
            )
