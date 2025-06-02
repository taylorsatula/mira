# Maps Tool Error Handling Improvements

This document provides specific error handling improvements for the Maps Tool, implementing the new error codes defined in the Error Code Implementation Guide.

## Current Limitations

The Maps Tool currently has limited error handling that uses generic error codes:
- Most errors use `ErrorCode.TOOL_EXECUTION_ERROR`
- Input validation uses `ErrorCode.TOOL_INVALID_INPUT`
- No specialized handling for network issues, geocoding failures, or API-specific errors

## Recommended Implementation

### 1. Parameter Validation Improvements

Replace general validation errors with more specific ones:

```python
# In run method (for lat/lng validation):
if operation in ['reverse_geocode', 'places_nearby'] and (lat is None or lng is None):
    raise ToolError(
        f"Required parameters missing for {operation}: lat and lng coordinates",
        ErrorCode.TOOL_PARAMETER_MISSING,
        {"operation": operation, "missing_params": ["lat", "lng"]}
    )

# For geocode operation:
if operation == "geocode" and not query:
    raise ToolError(
        "query parameter is required for geocode operation",
        ErrorCode.TOOL_PARAMETER_MISSING,
        {"operation": operation, "missing_params": ["query"]}
    )

# For calculate_distance (coordinate type checking):
try:
    # Convert to float if they aren't already
    lat1_val = float(lat1)
    lng1_val = float(lng1)
    lat2_val = float(lat2)
    lng2_val = float(lng2)
except ValueError:
    raise ToolError(
        "Coordinates must be valid numbers",
        ErrorCode.TOOL_DATA_TYPE_ERROR,
        {"invalid_params": ["lat1", "lng1", "lat2", "lng2"]}
    )

# For coordinate range validation:
for coord_name, coord_val in [("lat1", lat1_val), ("lat2", lat2_val)]:
    if coord_val < -90 or coord_val > 90:
        raise ToolError(
            f"Invalid {coord_name} value: {coord_val}. Latitude must be between -90 and 90",
            ErrorCode.TOOL_DATA_RANGE_ERROR,
            {"parameter": coord_name, "value": coord_val, "valid_range": "[-90, 90]"}
        )

for coord_name, coord_val in [("lng1", lng1_val), ("lng2", lng2_val)]:
    if coord_val < -180 or coord_val > 180:
        raise ToolError(
            f"Invalid {coord_name} value: {coord_val}. Longitude must be between -180 and 180",
            ErrorCode.TOOL_DATA_RANGE_ERROR,
            {"parameter": coord_name, "value": coord_val, "valid_range": "[-180, 180]"}
        )
```

### 2. Improved Network Error Handling

Enhance the client property method to handle different network errors:

```python
@property
def client(self):
    """Get the Google Maps client, initializing it if needed."""
    if self._client is None:
        try:
            from googlemaps import Client

            # Get API key from config
            from config import config
            api_key = config.google_maps_api_key
            if not api_key:
                raise ToolError(
                    "Google Maps API key not found in configuration.",
                    ErrorCode.CONFIG_NOT_FOUND,
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
            import socket
            
            # Check for different network error types
            error_str = str(e).lower()
            if "certificate" in error_str or "ssl" in error_str:
                raise ToolError(
                    f"SSL certificate error initializing Maps client: {e}",
                    ErrorCode.SSL_CERTIFICATE_ERROR,
                    {"error": str(e)}
                )
            elif "proxy" in error_str:
                raise ToolError(
                    f"Proxy error initializing Maps client: {e}",
                    ErrorCode.PROXY_CONNECTION_ERROR,
                    {"error": str(e)}
                )
            elif isinstance(e, socket.gaierror) or "name or service not known" in error_str:
                raise ToolError(
                    f"DNS resolution error initializing Maps client: {e}",
                    ErrorCode.DNS_RESOLUTION_ERROR,
                    {"error": str(e)}
                )
            else:
                self.logger.error(f"Failed to initialize Google Maps client: {e}")
                raise ToolError(
                    f"Failed to initialize Google Maps client: {e}",
                    ErrorCode.TOOL_INITIALIZATION_ERROR,
                    {"error": str(e)}
                )
    return self._client
```

### 3. Geocoding Error Improvements

Enhance geocoding error handling in the _geocode method:

```python
def _geocode(self, query: str) -> List[Dict[str, Any]]:
    """Convert a natural language query to geographic coordinates."""
    try:
        # Get geocoding parameters
        params = {"address": query}
        
        # Call geocoding API
        results = self.client.geocode(**params)
        
        # If no results found, raise specific error
        if not results:
            raise ToolError(
                f"No geocoding results found for query: {query}",
                ErrorCode.GEOLOCATION_ERROR,
                {"query": query}
            )
            
        processed_results = []
        # ... rest of the method remains the same
        
    except Exception as e:
        # If it's already a ToolError, re-raise it
        if isinstance(e, ToolError):
            raise
            
        # Otherwise, handle different error types
        error_str = str(e).lower()
        if "request denied" in error_str:
            raise ToolError(
                f"Maps API access denied: {e}",
                ErrorCode.API_AUTHENTICATION_ERROR,
                {"query": query, "error": str(e)}
            )
        elif "over query limit" in error_str:
            raise ToolError(
                f"Maps API rate limit exceeded: {e}",
                ErrorCode.API_RATE_LIMIT_ERROR,
                {"query": query, "error": str(e)}
            )
        elif "timeout" in error_str or "timed out" in error_str:
            raise ToolError(
                f"Maps API request timed out: {e}",
                ErrorCode.API_TIMEOUT_ERROR,
                {"query": query, "error": str(e)}
            )
        elif "unreachable" in error_str or "connection" in error_str:
            raise ToolError(
                f"Maps API network unreachable: {e}",
                ErrorCode.NETWORK_UNREACHABLE_ERROR,
                {"query": query, "error": str(e)}
            )
        else:
            self.logger.error(f"Geocoding error: {e}")
            raise ToolError(
                f"Failed to geocode query: {e}", 
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"query": query, "error": str(e)}
            )
```

### 4. Place Details Error Handling

Enhance the _place_details method with more specific error handling:

```python
def _place_details(self, place_id: str) -> Dict[str, Any]:
    """Get detailed information about a place."""
    try:
        result = self.client.place(place_id=place_id)
        
        # Extract place details from the result
        if "result" in result:
            # ... existing code...
        else:
            raise ToolError(
                f"No details found for place ID: {place_id}",
                ErrorCode.GEOLOCATION_ERROR,
                {"place_id": place_id}
            )
            
    except Exception as e:
        if isinstance(e, ToolError):
            raise
            
        error_str = str(e).lower()
        if "invalid" in error_str and "place_id" in error_str:
            raise ToolError(
                f"Invalid place ID: {place_id}",
                ErrorCode.TOOL_DATA_FORMAT_ERROR,
                {"place_id": place_id, "error": str(e)}
            )
        elif "request denied" in error_str:
            raise ToolError(
                f"Place details API access denied: {e}",
                ErrorCode.API_AUTHENTICATION_ERROR,
                {"place_id": place_id, "error": str(e)}
            )
        elif "over query limit" in error_str:
            raise ToolError(
                f"Place details API rate limit exceeded: {e}",
                ErrorCode.API_RATE_LIMIT_ERROR,
                {"place_id": place_id, "error": str(e)}
            )
        else:
            self.logger.error(f"Place details error: {e}")
            raise ToolError(
                f"Failed to get place details: {e}",
                ErrorCode.TOOL_EXECUTION_ERROR,
                {"place_id": place_id, "error": str(e)}
            )
```

### 5. Run Method Error Context Improvement

Update the error_context in the run method to provide better error classification:

```python
with error_context(
    component_name=self.name,
    operation=f"executing {operation}",
    error_class=ToolError,
    # Use specific error codes based on operation type
    error_code=ErrorCode.GEOLOCATION_ERROR if operation in ["geocode", "reverse_geocode", "find_place"] 
             else ErrorCode.TOOL_EXECUTION_ERROR,
    logger=self.logger,
):
    # Existing code
```

## Implementation Notes

1. The improvements maintain backward compatibility by handling exceptions the same way
2. New error codes provide more detailed information about what went wrong
3. Error details object includes relevant context for troubleshooting
4. Structured approach makes future debugging easier
5. Geographic-specific validation prevents common issues with coordinate data

These improvements will make the Maps Tool more robust, providing clearer error messages and more accurate diagnostics when problems occur.