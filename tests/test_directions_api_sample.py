"""
Test script to explore Google Directions API responses.

This script makes sample queries to the Google Directions API and prints
the response structure to understand the data available for implementing
the natural language directions tool.

Usage:
    python -m tests.test_directions_api_sample
"""

import os
import json
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from googlemaps import Client
except ImportError:
    logger.error("googlemaps library not installed. Run: pip install googlemaps")
    exit(1)

def get_api_key():
    """Get Google Maps API key from environment variables."""
    # First try to get from environment
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    
    # If not in environment, try to read from config
    if not api_key:
        try:
            import sys
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from config import config
            api_key = config.google_maps_api_key
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load API key from config: {e}")
            
    if not api_key:
        logger.error("Google Maps API key not found. Set GOOGLE_MAPS_API_KEY environment variable.")
        exit(1)
        
    return api_key

def fetch_directions(client, origin, destination, mode="driving"):
    """Fetch directions from Google Maps API."""
    logger.info(f"Fetching directions from {origin} to {destination} via {mode}")
    
    try:
        directions = client.directions(
            origin=origin,
            destination=destination,
            mode=mode,
            alternatives=False,
            units="metric"
        )
        return directions
    except Exception as e:
        logger.error(f"Error fetching directions: {e}")
        return None

def analyze_step(step, indent="  "):
    """Analyze a step object to understand available fields."""
    logger.info(f"{indent}Step contains keys: {list(step.keys())}")
    
    # Log HTML instructions
    if "html_instructions" in step:
        logger.info(f"{indent}Instructions: {step['html_instructions']}")
    
    # Log maneuver if available
    if "maneuver" in step:
        logger.info(f"{indent}Maneuver: {step['maneuver']}")
    
    # Log distance and duration
    if "distance" in step:
        logger.info(f"{indent}Distance: {step['distance']}")
    
    if "duration" in step:
        logger.info(f"{indent}Duration: {step['duration']}")
    
    # Check for sub-steps (transit mode can have these)
    if "steps" in step:
        logger.info(f"{indent}Contains sub-steps:")
        for substep in step["steps"]:
            analyze_step(substep, indent + "  ")

def run_test_queries():
    """Run several test queries to understand API responses."""
    api_key = get_api_key()
    client = Client(key=api_key)
    
    # Define test cases with diverse road types
    test_cases = [
        {
            "name": "Urban route with main roads and side streets",
            "origin": "Times Square, New York, NY",
            "destination": "Central Park, New York, NY",
            "mode": "driving"
        },
        {
            "name": "Highway to local roads",
            "origin": "JFK Airport, New York, NY",
            "destination": "Empire State Building, New York, NY",
            "mode": "driving"
        },
        {
            "name": "Walking directions with landmarks",
            "origin": "Louvre Museum, Paris, France",
            "destination": "Eiffel Tower, Paris, France",
            "mode": "walking"
        }
    ]
    
    for i, test in enumerate(test_cases):
        logger.info(f"\n\n==== TEST CASE {i+1}: {test['name']} ====")
        directions = fetch_directions(
            client, 
            test["origin"], 
            test["destination"], 
            test["mode"]
        )
        
        if not directions:
            logger.error(f"No directions returned for test case {i+1}")
            continue
            
        # Print overall route info
        logger.info(f"Received {len(directions)} route(s)")
        
        for route_idx, route in enumerate(directions):
            logger.info(f"\nROUTE {route_idx + 1} contains keys: {list(route.keys())}")
            
            if "legs" in route:
                for leg_idx, leg in enumerate(route["legs"]):
                    logger.info(f"\nLEG {leg_idx + 1} contains keys: {list(leg.keys())}")
                    
                    logger.info(f"  Start address: {leg.get('start_address')}")
                    logger.info(f"  End address: {leg.get('end_address')}")
                    logger.info(f"  Total distance: {leg.get('distance')}")
                    logger.info(f"  Total duration: {leg.get('duration')}")
                    
                    # Analyze steps
                    if "steps" in leg:
                        logger.info(f"\n  STEPS ({len(leg['steps'])} total):")
                        
                        for step_idx, step in enumerate(leg["steps"]):
                            logger.info(f"\n  STEP {step_idx + 1}:")
                            analyze_step(step)
            
            # Save raw JSON for later reference
            output_dir = "data/directions_samples"
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{output_dir}/sample_{i+1}_{test['mode']}.json"
            with open(filename, "w") as f:
                json.dump(directions, f, indent=2)
            
            logger.info(f"Saved raw response to {filename}")

if __name__ == "__main__":
    run_test_queries()