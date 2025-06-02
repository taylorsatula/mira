# Natural Language Directions Implementation Guide

## Overview

This guide outlines how to enhance the maps_tool with human-like natural language directions that describe route context (e.g., "turn off the main road onto the side street") rather than just standard turn-by-turn instructions.

## Requirements

1. **Road Classification Detection**
   - Identify road hierarchy (main roads vs. side streets)
   - Detect transitions between different road types
   - Recognize highway exits, ramps, and service roads

2. **Enhanced Direction Phrasing**
   - Convert standard "Turn left onto X Street" to contextual instructions
   - Include descriptive elements about the roads
   - Maintain route accuracy while enhancing language

3. **Optional Landmark Integration**
   - Reference nearby landmarks at decision points
   - Prioritize highly visible landmarks (stores, parks, distinctive buildings)

4. **Performance Considerations**
   - Minimize additional API calls
   - Implement caching strategies
   - Maintain reasonable response times

## Implementation Architecture

### 1. Core Components

```
┌─────────────────────┐      ┌───────────────────────┐      ┌───────────────────────┐
│  Google Directions  │─────▶│  Direction Enhancer   │─────▶│  Natural Language     │
│  API Client         │      │  (Road Classification) │      │  Response Formatter   │
└─────────────────────┘      └───────────────────────┘      └───────────────────────┘
                                       │
                                       ▼
                              ┌───────────────────────┐
                              │  Optional Landmark    │
                              │  Enricher             │
                              └───────────────────────┘
```

### 2. Road Classification Rules

Establish a hierarchy of road types:
- Highway/Motorway (highest)
- Primary/Trunk roads
- Secondary/Arterial roads
- Tertiary/Collector roads
- Residential/Service roads (lowest)

## Implementation Steps

### Step 1: Extract Road Metadata

The Google Directions API provides information about roads in the `steps` of a route. Each step includes:
- `html_instructions`: The default turn instruction
- `distance` and `duration`: Length and expected travel time
- Road names and potentially types

```python
def extract_road_metadata(step):
    """Extract road metadata from a directions API step."""
    metadata = {
        "instruction": step.get("html_instructions", ""),
        "distance": step.get("distance", {}).get("value", 0),  # in meters
        "duration": step.get("duration", {}).get("value", 0),  # in seconds
        "start_location": step.get("start_location", {}),
        "end_location": step.get("end_location", {}),
        "maneuver": step.get("maneuver", ""),
        "travel_mode": step.get("travel_mode", "DRIVING")
    }
    
    # Extract road names from the instruction
    # This is a simplification - we'd need more sophisticated parsing in production
    instruction = metadata["instruction"]
    if "onto" in instruction:
        metadata["target_road"] = instruction.split("onto")[1].strip()
    
    return metadata
```

### Step 2: Determine Road Classification

Use Google Maps Places API to get additional information about the roads:

```python
def classify_road(road_name, location):
    """Determine the classification of a road."""
    # Query nearby roads using Places API
    places_result = maps_client.places_nearby(
        location=location,
        radius=50,
        keyword=road_name,
        type="route"  # This is a simplification
    )
    
    # Analyze results to determine road type
    # This would require more sophisticated logic in practice
    if not places_result.get("results"):
        return "unknown"
    
    types = places_result["results"][0].get("types", [])
    
    if "highway" in types:
        return "highway"
    elif "arterial" in types or "primary" in types:
        return "main_road"
    elif "residential" in types or "local" in types:
        return "side_street"
    else:
        return "standard_road"
```

### Step 3: Generate Enhanced Directions

Transform standard directions into more natural language based on road classifications:

```python
def enhance_direction(step, prev_step=None):
    """Generate enhanced natural language direction."""
    metadata = extract_road_metadata(step)
    
    # Skip if no previous step (first instruction)
    if not prev_step:
        return metadata["instruction"]
    
    prev_metadata = extract_road_metadata(prev_step)
    
    # Classify roads
    current_road_class = classify_road(
        metadata.get("target_road", ""), 
        metadata["end_location"]
    )
    
    prev_road_class = classify_road(
        prev_metadata.get("target_road", ""),
        prev_metadata["end_location"]
    )
    
    # Generate enhanced instruction based on transition
    if prev_road_class == "highway" and current_road_class != "highway":
        return f"Exit the highway onto {metadata.get('target_road', 'the exit')}"
        
    elif prev_road_class == "main_road" and current_road_class == "side_street":
        return f"Turn {metadata.get('maneuver', 'onto')} the side street ({metadata.get('target_road', '')})"
        
    elif prev_road_class == "side_street" and current_road_class == "main_road":
        return f"Turn {metadata.get('maneuver', 'onto')} the main road ({metadata.get('target_road', '')})"
    
    # Default to original instruction if no special case applies
    return metadata["instruction"]
```

### Step 4: Process Full Route

Process the complete route to generate a sequence of enhanced directions:

```python
def get_enhanced_directions(origin, destination):
    """Get directions with enhanced natural language instructions."""
    # Get standard directions from Google Maps API
    directions_result = maps_client.directions(origin, destination)
    
    if not directions_result:
        return {"error": "No directions found"}
    
    route = directions_result[0]
    legs = route.get("legs", [])
    if not legs:
        return {"error": "No route legs found"}
    
    leg = legs[0]  # For simplicity, focusing on first leg
    steps = leg.get("steps", [])
    
    # Process steps to generate enhanced directions
    enhanced_directions = []
    prev_step = None
    
    for step in steps:
        enhanced_instruction = enhance_direction(step, prev_step)
        enhanced_directions.append({
            "instruction": enhanced_instruction,
            "distance": step.get("distance", {}).get("text", ""),
            "duration": step.get("duration", {}).get("text", "")
        })
        prev_step = step
    
    return {
        "origin": leg.get("start_address", ""),
        "destination": leg.get("end_address", ""),
        "total_distance": leg.get("distance", {}).get("text", ""),
        "total_duration": leg.get("duration", {}).get("text", ""),
        "steps": enhanced_directions
    }
```

### Step 5: Add Optional Landmark Integration

Enhance directions with nearby landmarks at decision points:

```python
def add_landmarks_to_directions(directions):
    """Add landmark references to directions."""
    for i, step in enumerate(directions["steps"]):
        # Only add landmarks for turns and decision points
        if "turn" in step["instruction"].lower() or "exit" in step["instruction"].lower():
            # Get location for this step
            location = step.get("location", {})
            if not location:
                continue
                
            # Find nearby landmarks
            landmarks = maps_client.places_nearby(
                location=(location.get("lat"), location.get("lng")),
                radius=100,
                type=["store", "restaurant", "gas_station", "park"],
                fields=["name", "types", "vicinity"],
                rank_by="prominence"
            )
            
            # Select most prominent landmark
            if landmarks.get("results"):
                landmark = landmarks["results"][0]
                landmark_name = landmark.get("name", "")
                
                # Add landmark to instruction
                if landmark_name:
                    # Replace or enhance the instruction
                    if "turn" in step["instruction"].lower():
                        step["instruction"] = step["instruction"].replace(
                            "Turn", f"Turn at the {landmark_name},"
                        )
                    else:
                        step["instruction"] += f" (You'll see {landmark_name})"
    
    return directions
```

## Integration with maps_tool

Update the maps_tool to include a new operation for enhanced directions:

```python
def _get_enhanced_directions(self, origin, destination, include_landmarks=False):
    """
    Get directions with natural language enhancements.
    
    Args:
        origin: Starting location (address or coordinates)
        destination: Ending location (address or coordinates)
        include_landmarks: Whether to enhance with landmark references
        
    Returns:
        Dict containing enhanced directions
    """
    with error_context(
        component_name=self.name,
        operation="getting enhanced directions",
        error_class=ToolError,
        error_code=ErrorCode.TOOL_EXECUTION_ERROR,
        logger=self.logger,
    ):
        # Get enhanced directions
        directions = get_enhanced_directions(origin, destination)
        
        # Optionally add landmarks
        if include_landmarks:
            directions = add_landmarks_to_directions(directions)
            
        return directions
```

Update the `run` method of `MapsTool` to include this new operation:

```python
elif operation == "enhanced_directions":
    if not query and (origin is None or destination is None):
        raise ToolError(
            "Either query or origin/destination parameters are required for enhanced_directions operation",
            ErrorCode.TOOL_INVALID_INPUT,
        )
    
    # If query is provided, it should be in the format "from X to Y"
    if query:
        origin, destination = self._parse_directions_query(query)
    
    return self._get_enhanced_directions(
        origin=origin,
        destination=destination,
        include_landmarks=include_landmarks or False
    )
```

## Example Output

### Standard Google Directions:
```json
{
  "steps": [
    {
      "html_instructions": "Head south on Broadway toward W 58th St",
      "distance": {"text": "0.2 mi", "value": 322},
      "duration": {"text": "1 min", "value": 60}
    },
    {
      "html_instructions": "Turn right onto W 57th St",
      "distance": {"text": "0.3 mi", "value": 483},
      "duration": {"text": "2 mins", "value": 120},
      "maneuver": "turn-right"
    },
    {
      "html_instructions": "Turn left onto 6th Ave",
      "distance": {"text": "0.4 mi", "value": 644},
      "duration": {"text": "3 mins", "value": 180},
      "maneuver": "turn-left"
    }
  ]
}
```

### Enhanced Natural Language Directions:
```json
{
  "steps": [
    {
      "instruction": "Head south on Broadway (a main road)",
      "distance": "0.2 mi",
      "duration": "1 min"
    },
    {
      "instruction": "Turn right off the main road onto the side street (W 57th St)",
      "distance": "0.3 mi",
      "duration": "2 mins"
    },
    {
      "instruction": "Turn left at the Starbucks, back onto a main road (6th Ave)",
      "distance": "0.4 mi",
      "duration": "3 mins"
    }
  ]
}
```

## Testing Strategy

1. **Unit Tests:**
   - Test classification of various road types
   - Test direction enhancement logic for different scenarios
   - Test landmark integration

2. **Integration Tests:**
   - Test end-to-end direction generation with the Google Maps API
   - Compare enhanced directions with standard directions

3. **Manual Testing:**
   - Verify that enhanced directions make sense for various routes
   - Check that road classifications are accurate

## Challenges and Limitations

1. **Data Accuracy:**
   - Road classification data may not always be reliable
   - Visible landmarks might not be correctly prioritized

2. **API Costs:**
   - Additional API calls for road classification and landmarks increase costs
   - Consider implementing a caching strategy

3. **Language Naturalness:**
   - Some directions may still sound mechanical
   - Consider a hybrid approach with LLM post-processing for maximum naturalness

## Next Steps

1. Implement baseline road classification-based enhancements
2. Add optional landmark integration
3. Consider LLM post-processing for further refinement
4. Implement caching to reduce API costs
5. Test with diverse routes and refine logic