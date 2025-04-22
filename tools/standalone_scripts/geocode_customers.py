#!/usr/bin/env python
"""
Geocode Square customers' addresses using Google Maps Geocoding API.

This standalone script takes Square customer IDs (one or more), geocodes their addresses
using Google Maps Geocoding API, and adds latitude/longitude coordinates to each customer
record in the local customer directory JSON file.

Requirements:
    - Google Maps API key in the GOOGLE_MAPS_API_KEY environment variable
    - requests library (pip install requests)
    - Existing Square customer directory with address data

Environment Variables:
    GOOGLE_MAPS_API_KEY: Required Google Maps Geocoding API key

Usage:
    # Process specific customers
    python geocode_customers.py --customer-id CUST123 [--customer-id CUST456 ...]

    # Process all customers in the directory
    python geocode_customers.py --all-customers

    # Enable verbose logging
    python geocode_customers.py --all-customers --verbose

Output:
    The script will add a "geocoding_data" field to each customer record containing:
    {
        "coordinates": {
            "lat": 37.7749,
            "lng": -122.4194
        },
        "geocoded_at": 1712616845
    }
"""

import os
import sys
import json
import time
import logging
import argparse
import pathlib
from typing import Dict, List, Any, Optional, Union
import requests

# Add parent directory to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config import config
from errors import ToolError, ErrorCode

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CustomerGeocoder:
    """
    Geocodes customer addresses in the local Square customer directory using Google Maps Geocoding API.
    """

    def __init__(self):
        """Initialize the geocoder with necessary configuration."""
        # Get API key from environment variable
        self.api_key = os.environ.get("GOOGLE_MAPS_API_KEY")

        if not self.api_key:
            raise ToolError(
                "Google Maps API key not found. Set the GOOGLE_MAPS_API_KEY environment variable.",
                ErrorCode.TOOL_EXECUTION_ERROR,
            )

        # Setup customer directory path
        data_dir = pathlib.Path(config.paths.data_dir)
        self.cache_dir = data_dir / "tools" / "square_tool"
        self.customer_directory_path = self.cache_dir / "customer_directory.json"

        # Load the customer directory
        self.customer_directory = self._load_customer_directory()

        logger.info("CustomerGeocoder initialized")

    def _load_customer_directory(self) -> Dict[str, Any]:
        """
        Load the customer directory from the cache file.

        Returns:
            Dict with customers indexed by ID and metadata
        """
        if not self.customer_directory_path.exists():
            logger.error("Customer directory cache not found")
            raise ToolError(
                "Customer directory cache not found at {}".format(
                    self.customer_directory_path
                ),
                ErrorCode.TOOL_EXECUTION_ERROR,
            )

        try:
            with open(self.customer_directory_path, "r") as f:
                directory = json.load(f)

                # Handle structure and validate
                if not isinstance(directory, dict):
                    logger.error("Invalid cache format")
                    raise ToolError(
                        "Invalid cache format", ErrorCode.TOOL_EXECUTION_ERROR
                    )

                # Ensure we have the expected structure
                if "customers" not in directory:
                    logger.error("Invalid cache structure, missing 'customers' key")
                    raise ToolError(
                        "Invalid cache structure, missing 'customers' key",
                        ErrorCode.TOOL_EXECUTION_ERROR,
                    )

                customer_count = len(directory.get("customers", {}))
                logger.info(f"Loaded {customer_count} customers from directory cache")
                return directory

        except Exception as e:
            logger.error(f"Error loading customer directory: {e}")
            raise ToolError(
                f"Error loading customer directory: {e}", ErrorCode.TOOL_EXECUTION_ERROR
            )

    def _save_customer_directory(self):
        """Save the customer directory to the cache file."""
        try:
            # Update timestamp before saving
            if isinstance(self.customer_directory, dict):
                self.customer_directory["last_updated"] = int(time.time())

            with open(self.customer_directory_path, "w") as f:
                json.dump(self.customer_directory, f, indent=2)

                # Get accurate customer count
                customer_count = len(self.customer_directory.get("customers", {}))
                logger.info(f"Saved {customer_count} customers to directory cache")
        except Exception as e:
            logger.error(f"Error saving customer directory: {e}")
            raise ToolError(
                f"Error saving customer directory: {e}", ErrorCode.TOOL_EXECUTION_ERROR
            )

    def geocode_address(self, address: Dict[str, str]) -> Optional[Dict[str, float]]:
        """
        Geocode a single address using Google Maps Geocoding API.

        Args:
            address: Dictionary containing address components
                Expected format:
                {
                    "address_line_1": str,
                    "address_line_2": str,
                    "locality": str,
                    "administrative_district_level_1": str,
                    "postal_code": str,
                    "country": str
                }

        Returns:
            Dict with latitude and longitude, or None if geocoding failed
                Format: {"lat": float, "lng": float}
        """
        # Skip if no address
        if not address:
            logger.warning("No address provided for geocoding")
            return None

        # Format address for geocoding
        address_str = ""
        if address.get("address_line_1"):
            address_str += address["address_line_1"] + " "
        if address.get("address_line_2"):
            address_str += address["address_line_2"] + " "
        if address.get("locality"):
            address_str += address["locality"] + " "
        if address.get("administrative_district_level_1"):
            address_str += address["administrative_district_level_1"] + " "
        if address.get("postal_code"):
            address_str += address["postal_code"] + " "
        if address.get("country"):
            address_str += address["country"]

        address_str = address_str.strip()
        if not address_str:
            logger.warning("Empty address string after formatting")
            return None

        logger.info(f"Geocoding address: {address_str}")

        # Make request to Google Maps Geocoding API
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": address_str, "key": self.api_key}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if data["status"] != "OK":
                logger.error(f"Geocoding error: {data['status']}")
                return None

            if not data.get("results"):
                logger.warning("No geocoding results found")
                return None

            # Get the first result's location
            location = data["results"][0]["geometry"]["location"]
            logger.info(f"Successfully geocoded: {location}")

            return {"lat": location["lat"], "lng": location["lng"]}

        except requests.RequestException as e:
            logger.error(f"Request error during geocoding: {e}")
            return None
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error parsing geocoding response: {e}")
            return None

    def geocode_customer(self, customer_id: str) -> bool:
        """
        Geocode a customer's address and update their record in the directory.

        Args:
            customer_id: Square customer ID

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Processing customer: {customer_id}")

        try:
            # Get customer from the directory
            customers = self.customer_directory.get("customers", {})

            if customer_id not in customers:
                logger.error(f"Customer {customer_id} not found in directory")
                return False

            customer = customers[customer_id]

            # Skip if customer already has geocoding data
            if customer.get("geocoding_data"):
                logger.info(f"Customer {customer_id} already geocoded")
                return True

            # Get address from customer
            address = customer.get("address")
            if not address:
                logger.warning(f"Customer {customer_id} has no address")
                return False

            # Geocode the address
            location = self.geocode_address(address)
            if not location:
                logger.error(f"Failed to geocode address for customer {customer_id}")
                return False

            # Add geocoding data to customer record
            customer["geocoding_data"] = {
                "coordinates": location,
                "geocoded_at": int(time.time()),
            }

            logger.info(f"Successfully geocoded customer {customer_id}")
            return True

        except Exception as e:
            logger.error(f"Unexpected error for customer {customer_id}: {e}")
            return False

    def geocode_customers(self, customer_ids: List[str]) -> Dict[str, bool]:
        """
        Geocode multiple customers.

        Args:
            customer_ids: List of Square customer IDs

        Returns:
            Dict mapping customer IDs to success status
        """
        results = {}
        modified = False

        for customer_id in customer_ids:
            # Add delay to avoid rate limiting
            time.sleep(0.5)
            success = self.geocode_customer(customer_id)
            results[customer_id] = success
            if success:
                modified = True

        # Save the directory once after processing all customers
        if modified:
            logger.info(
                f"Saving updated directory with {len(results)} processed customers"
            )
            self._save_customer_directory()

        return results

    def geocode_all_customers(self) -> Dict[str, bool]:
        """
        Geocode all customers in the local directory.

        Returns:
            Dict mapping customer IDs to success status
        """
        # Get all customer IDs
        customer_ids = list(self.customer_directory.get("customers", {}).keys())
        logger.info(f"Found {len(customer_ids)} customers to process")

        return self.geocode_customers(customer_ids)


def main():
    """Parse arguments and run the geocoding process."""
    parser = argparse.ArgumentParser(
        description="Geocode Square customer addresses using Google Maps API"
    )

    # Create mutually exclusive group for customer selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--customer-id",
        action="append",
        help="Square customer ID(s) to geocode (can be provided multiple times)",
    )
    group.add_argument(
        "--all-customers",
        action="store_true",
        help="Process all customers in the local directory",
    )

    # Add optional arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        geocoder = CustomerGeocoder()

        if args.all_customers:
            logger.info("Processing all customers")
            results = geocoder.geocode_all_customers()
        else:
            # Handle case where no customer IDs were provided
            if not args.customer_id:
                logger.error("No customer IDs provided")
                return 1

            logger.info(f"Processing {len(args.customer_id)} customer(s)")
            results = geocoder.geocode_customers(args.customer_id)

        # Summarize results
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Geocoding complete: {success_count}/{len(results)} successful")

        # Show results for each customer
        for customer_id, success in results.items():
            status = "Success" if success else "Failed"
            logger.info(f"Customer {customer_id}: {status}")

        return 0

    except ToolError as e:
        logger.error(f"Tool error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
