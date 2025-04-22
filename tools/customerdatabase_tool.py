"""
Customer management tool with database support.

This tool provides customer directory management capabilities including loading,
saving, searching, and location-based customer finding functionality. It uses a
SQLite database for persistent storage while maintaining compatibility with the
original JSON-based storage.
"""

import os
import logging
import pathlib
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Set, Tuple

from sqlalchemy import or_, and_, func, text

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context
from config import config
from db import Database, Customer, migrate_customers_from_json


class CustomerDatabaseTool(Tool):
    """
    Tool for managing a customer directory with search capabilities using SQLite database.
    
    Features:
    1. Customer Directory Management:
       - Load and save customer data to/from database
       - Maintain a local cache of customer information
       - Rebuild directory from external sources

    2. Customer Search:
       - Find customers by name, email, phone, address
       - Automatically categorize search queries
       - Location-based customer searching
    """

    name = "customerdatabase_tool"
    description = """Manages a comprehensive customer directory using SQLite database storage with robust search and location-based capabilities. 
    
This tool maintains a SQLite database for customer data with support for importing from external systems (currently Square). It provides efficient searching and retrieving of customer data through multiple operations:

1. search_customers: Find customers by various identifiers including name, email, phone number, or address. 
   - Requires 'query' parameter with your search term
   - Optional 'category' parameter to specify search type: 'name', 'given_name', 'family_name', 'email', 'phone', 'address', or 'any' (default)
   - Returns matching customer records with contact details

2. find_closest_customers: Locate nearby customers using geographical coordinates.
   - Requires 'lat' and 'lng' parameters (latitude/longitude)
   - Optional 'limit' parameter to specify maximum number of results (default: 1)
   - Optional 'max_distance' parameter to set maximum distance in meters
   - Optional 'exclude_customer_id' to omit a specific customer
   - Returns customers sorted by proximity with distance information

3. get_customer: Retrieve a specific customer record by ID.
   - Requires 'customer_id' parameter
   - Returns complete customer information

4. rebuild_directory: Refresh the customer database from external systems.
   - Optional 'source' parameter (currently only supports 'square')
   - Returns status information about the rebuild operation"""

    def __init__(self):
        """Initialize the Customer tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Setup database connection
        self.db = Database()
        
        # Setup cache
        self._customer_cache = {}
        self._cache_timestamp = 0
        self._cache_loaded = False
        
        # Check if we need to migrate from JSON
        self._check_migration()

    def _check_migration(self):
        """
        Check if we need to migrate data from JSON to the database.
        Performs migration if database is empty but JSON data exists.
        """
        # Check if database has any customers
        with self.db.get_session() as session:
            customer_count = session.query(Customer).count()
            
        if customer_count == 0:
            # Database is empty, check if JSON data exists
            data_dir = pathlib.Path(config.paths.data_dir)
            cache_dir = data_dir / "tools" / "customer_tool"
            json_path = cache_dir / "customer_directory.json"
            
            if json_path.exists():
                self.logger.info("Migrating customers from JSON to database...")
                result = migrate_customers_from_json()
                self.logger.info(f"Migration result: {result['message']}")

    def _load_customer_cache(self) -> Dict[str, Any]:
        """
        Load customers from database into memory cache for faster access.

        Returns:
            Dict with customers indexed by ID and metadata
        """
        # If cache is already loaded and fresh, return it
        now = time.time()
        if self._cache_loaded and now - self._cache_timestamp < 300:  # 5 minutes
            return self._customer_cache
            
        # Load customers from database
        with self.db.get_session() as session:
            customers = session.query(Customer).all()
            
        # Build cache
        cache = {}
        for customer in customers:
            cache[customer.id] = customer.to_dict()
            
        # Update cache
        self._customer_cache = cache
        self._cache_timestamp = now
        self._cache_loaded = True
        
        self.logger.info(f"Loaded {len(cache)} customers into cache")
        return cache

    def _invalidate_cache(self):
        """Invalidate the customer cache, forcing reload on next access."""
        self._cache_loaded = False
        self._customer_cache = {}
        self._cache_timestamp = 0

    def rebuild_directory(self, source="square") -> Dict[str, Any]:
        """
        Rebuild the customer directory from an external source.

        Args:
            source: Source system to rebuild the directory from (default: square)

        Returns:
            Dict with status and results
        """
        # For now, we only support Square as a source
        if source.lower() != "square":
            raise ToolError(
                f"Unsupported source for rebuild: {source}. Only 'square' is currently supported.",
                ErrorCode.TOOL_INVALID_INPUT,
            )
            
        try:
            # Import square_tool here to avoid circular imports
            from tools.square_tool import SquareTool
            
            self.logger.info("Rebuilding customer directory from Square API...")
            
            # Create a Square tool instance
            square_tool = SquareTool()
            
            # Track customer IDs we find for deletion logic
            found_customer_ids = set()
            
            # Create database instance
            db = self.db
            
            # Fetch all customers with pagination from Square
            cursor = None
            total_fetched = 0
            new_count = 0
            
            # Get existing customer IDs from database
            with db.get_session() as session:
                existing_ids = {id for id, in session.query(Customer.id).all()}

            while True:
                # Get a batch of customers from Square
                result = square_tool.run(
                    operation="list_customers",
                    cursor=cursor,
                    limit=100
                )

                customers = result.get("customers", [])
                if not customers:
                    break

                # Process each customer - only add new ones
                for customer_data in customers:
                    customer_id = customer_data.get("id")
                    if not customer_id:
                        continue

                    # Track that we found this customer
                    found_customer_ids.add(customer_id)

                    # Create or update customer in database
                    try:
                        customer = Customer.from_dict(customer_data)
                        
                        if customer_id in existing_ids:
                            # Update existing customer
                            db.update(customer)
                        else:
                            # Add new customer
                            db.add(customer)
                            new_count += 1
                    except Exception as e:
                        self.logger.error(f"Error processing customer {customer_id}: {e}")

                total_fetched += len(customers)
                self.logger.info(
                    f"Processed {len(customers)} customers, total: {total_fetched}, new: {new_count}"
                )

                # Get the next cursor for pagination
                cursor = result.get("cursor")
                if not cursor:
                    break

            # Delete customers that are no longer in Square
            deleted_count = 0
            if found_customer_ids:
                # Find customers that exist in our database but weren't found in Square
                to_delete = existing_ids - found_customer_ids
                
                # Delete them from the database
                with db.get_session() as session:
                    for customer_id in to_delete:
                        customer = session.query(Customer).get(customer_id)
                        if customer:
                            session.delete(customer)
                            deleted_count += 1
                    session.commit()

            # Invalidate cache after updates
            self._invalidate_cache()
            
            # If new customers were added, trigger geocoding on them
            if new_count > 0:
                # We need to identify which customer IDs need geocoding
                with db.get_session() as session:
                    new_ids = [
                        id for id, in session.query(Customer.id)
                        .filter(
                            and_(
                                Customer.latitude.is_(None),
                                Customer.longitude.is_(None),
                                Customer.id.in_(found_customer_ids)
                            )
                        ).all()
                    ]

                if new_ids:
                    self.logger.info(f"Geocoding {len(new_ids)} new customers")
                    self._geocode_customers(new_ids)

            # Get final customer count
            with db.get_session() as session:
                final_count = session.query(Customer).count()

            return {
                "success": True,
                "customer_count": final_count,
                "new_customers": new_count,
                "deleted_customers": deleted_count,
                "message": f"Customer directory successfully rebuilt with {final_count} customers"
            }

        except Exception as e:
            self.logger.error(f"Error rebuilding customer directory: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to rebuild customer directory"
            }

    def _geocode_customers(self, customer_ids: List[str]) -> Dict[str, bool]:
        """
        Geocode multiple customers using Maps API.
        
        Args:
            customer_ids: List of customer IDs to geocode
            
        Returns:
            Dict mapping customer IDs to success status
        """
        try:
            # Import maps_tool here to avoid circular imports
            from tools.maps_tool import MapsTool
            
            maps_tool = MapsTool()
            results = {}
            modified = False
            db = self.db
            
            for customer_id in customer_ids:
                # Get customer from database
                customer = db.get(Customer, customer_id)
                if not customer:
                    results[customer_id] = False
                    continue
                    
                # Skip if customer already has geocoding
                if customer.latitude is not None and customer.longitude is not None:
                    results[customer_id] = True
                    continue
                    
                # Format address for geocoding
                address_str = ""
                if customer.address_line1:
                    address_str += customer.address_line1 + " "
                if customer.address_line2:
                    address_str += customer.address_line2 + " "
                if customer.city:
                    address_str += customer.city + " "
                if customer.state:
                    address_str += customer.state + " "
                if customer.postal_code:
                    address_str += customer.postal_code + " "
                if customer.country:
                    address_str += customer.country
                    
                address_str = address_str.strip()
                if not address_str:
                    results[customer_id] = False
                    continue
                    
                try:
                    # Add delay to avoid rate limiting
                    time.sleep(0.5)
                    
                    # Call Maps API for geocoding
                    geocode_result = maps_tool.run(
                        operation="geocode",
                        query=address_str
                    )
                    
                    # Extract location data from result
                    if (geocode_result and 
                        "results" in geocode_result and 
                        geocode_result["results"] and 
                        "location" in geocode_result["results"][0]):
                        
                        location = geocode_result["results"][0]["location"]
                        
                        # Update customer with geocoding data
                        customer.latitude = location.get("lat")
                        customer.longitude = location.get("lng")
                        customer.geocoded_at = datetime.utcnow()
                        
                        # Save changes
                        db.update(customer)
                        
                        results[customer_id] = True
                        modified = True
                        self.logger.info(f"Successfully geocoded customer {customer_id}")
                    else:
                        results[customer_id] = False
                        self.logger.warning(f"No geocoding results found for customer {customer_id}")
                        
                except Exception as e:
                    self.logger.error(f"Error geocoding customer {customer_id}: {e}")
                    results[customer_id] = False
            
            # Invalidate cache if any customers were modified
            if modified:
                self._invalidate_cache()
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in geocoding process: {e}")
            return {customer_id: False for customer_id in customer_ids}

    def _search_customers(
        self, query: str, category: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for customers in the database.

        Args:
            query: The search term
            category: The search category (name, given_name, family_name, email, phone, address, any).
                     If not provided, the LLM will determine the appropriate category.

        Returns:
            List of matching customer records
        """
        self.logger.info(f"Searching customer database for: '{query}' with category: {category}")
        
        # Normalize query for comparison
        query = query.lower().strip()
        
        # For exact ID matching
        if category == "id":
            customer = self.db.get(Customer, query)
            if customer:
                return [customer.to_dict()]
            return []
        
        # Use SQL for all other searches for efficiency
        with self.db.get_session() as session:
            # Build base query
            db_query = session.query(Customer)
            
            # Apply search filters based on category
            if category == "name" or category in ["any", None]:
                # Search full name, given name, family name, nickname, company
                db_query = db_query.filter(
                    or_(
                        func.lower(Customer.given_name + " " + Customer.family_name).contains(query),
                        func.lower(Customer.given_name).contains(query),
                        func.lower(Customer.family_name).contains(query),
                        func.lower(Customer.company_name).contains(query)
                    )
                )
            elif category == "given_name":
                db_query = db_query.filter(func.lower(Customer.given_name).contains(query))
            elif category == "family_name":
                db_query = db_query.filter(func.lower(Customer.family_name).contains(query))
            elif category == "email" or category in ["any", None]:
                db_query = db_query.filter(func.lower(Customer.email_address).contains(query))
            elif category == "phone" or category in ["any", None]:
                db_query = db_query.filter(Customer.phone_number.contains(query))
            elif category == "address" or category in ["any", None]:
                # Search in address fields
                db_query = db_query.filter(
                    or_(
                        func.lower(Customer.address_line1).contains(query),
                        func.lower(Customer.address_line2).contains(query),
                        func.lower(Customer.city).contains(query),
                        func.lower(Customer.state).contains(query),
                        func.lower(Customer.postal_code).contains(query),
                        func.lower(Customer.country).contains(query)
                    )
                )
            
            # Execute query and convert results to dictionaries
            customers = db_query.all()
            results = [customer.to_dict() for customer in customers]
            
            self.logger.info(f"Database search found {len(results)} matching customers")
            return results

    def search_customers(
        self, query: str, category: str = None
    ) -> Dict[str, Any]:
        """
        Search for customers in the database with automatic rebuild if needed.

        Args:
            query: The search term
            category: Optional category override (name, email, phone, address, any)
                     If not provided, Claude will intelligently determine the category.
                     
                     Valid categories:
                     - "name": Full name search (first and last name together)
                     - "given_name": First/given name only search
                     - "family_name": Last/family name only search
                     - "address": Address search (any address component)
                     - "phone": Phone number search 
                     - "email": Email address search
                     - "any": Search across all fields (default if not specified)

        Returns:
            Dict with 'search_type' and 'customers' list containing matching records

        Raises:
            ToolError: If no matches found after rebuild
        """
        # Check if database is empty
        with self.db.get_session() as session:
            customer_count = session.query(Customer).count()
            
        if customer_count == 0:
            self.logger.info(
                "Customer database empty. Building it now..."
            )
            result = self.rebuild_directory()
            if not result.get("success", False):
                self.logger.error("Failed to build initial customer directory")
                raise ToolError(
                    "Failed to initialize customer directory",
                    ErrorCode.TOOL_EXECUTION_ERROR,
                )

        # Determine search type for the response
        search_type = "Search"
        if category == "name":
            search_type = "Full Name Search"
        elif category == "given_name":
            search_type = "Given Name Search"
        elif category == "family_name":
            search_type = "Family Name Search"
        elif category == "address":
            search_type = "Address Search"
        elif category == "phone":
            search_type = "Phone Number Search"
        elif category == "email":
            search_type = "Email Address Search"
        elif category == "any" or category is None:
            search_type = "Multi-field Search"
            category = "any"  # Default to any if not specified

        self.logger.info(f"Searching with category: {category}")

        # Now search using the database
        results = self._search_customers(query, category)

        # If no results, rebuild directory and try again
        if not results:
            self.logger.info(
                "No results found in database. Rebuilding customer directory..."
            )
            result = self.rebuild_directory()

            if result.get("success", False):
                self.logger.info("Directory rebuilt, searching again...")
                results = self._search_customers(query, category)
            else:
                self.logger.error("Failed to rebuild customer directory")

        # If still no results, raise error
        if not results:
            raise ToolError(
                f"No customers found matching '{query}' in category '{category}'",
                ErrorCode.TOOL_EXECUTION_ERROR,
            )

        return {
            "search_type": search_type,
            "customers": results
        }

    def find_closest_customers(
        self, lat: float, lng: float, limit: int = 1, max_distance: Optional[float] = None, 
        exclude_customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find the closest customers to a specific location.
        
        Args:
            lat: Latitude in decimal degrees
            lng: Longitude in decimal degrees
            limit: Maximum number of customers to return, sorted by proximity (default: 1)
            max_distance: Optional maximum distance in meters
            exclude_customer_id: Optional customer ID to exclude from results (e.g., to exclude the reference customer)
            
        Returns:
            Dict with 'customers' list containing nearby customers, sorted by distance.
            Each customer record includes a 'distance_meters' field.
            
        Raises:
            ToolError: If no customers with coordinates found
        """
        self.logger.info(f"Searching for customers near coordinates: {lat}, {lng}")
        
        # Check if database is empty
        with self.db.get_session() as session:
            customer_count = session.query(Customer).count()
            
        if customer_count == 0:
            self.logger.info("Customer database empty. Building it now...")
            result = self.rebuild_directory()
            if not result.get("success", False):
                self.logger.error("Failed to build initial customer directory")
                raise ToolError(
                    "Failed to initialize customer directory",
                    ErrorCode.TOOL_EXECUTION_ERROR,
                )
        
        try:
            # Import maps_tool to use distance calculation
            from tools.maps_tool import MapsTool
            maps_tool = MapsTool()
            
            # Query customers with geocoding data
            with self.db.get_session() as session:
                query = session.query(Customer).filter(
                    and_(
                        Customer.latitude.isnot(None),
                        Customer.longitude.isnot(None)
                    )
                )
                
                # Exclude specific customer if requested
                if exclude_customer_id:
                    query = query.filter(Customer.id != exclude_customer_id)
                    
                customers = query.all()
            
            results_with_distance = []
            
            # Calculate distances for each customer
            for customer in customers:
                try:
                    # Calculate distance using maps_tool
                    distance_result = maps_tool.run(
                        operation="calculate_distance",
                        lat1=lat,
                        lng1=lng,
                        lat2=float(customer.latitude),
                        lng2=float(customer.longitude)
                    )
                    
                    distance = distance_result.get("distance_meters")
                    
                    # Skip if beyond max distance
                    if max_distance is not None and distance > max_distance:
                        continue
                        
                    # Convert to dict and add distance
                    customer_dict = customer.to_dict()
                    customer_dict["distance_meters"] = distance
                    
                    results_with_distance.append(customer_dict)
                except Exception as e:
                    self.logger.error(f"Error calculating distance for customer {customer.id}: {e}")
                    continue
            
            # Sort results by distance
            results_with_distance.sort(key=lambda x: x["distance_meters"])
            
            # Limit the number of results
            results_with_distance = results_with_distance[:limit] if limit else results_with_distance
            
            # If no results with coordinates, try rebuilding the directory
            if not results_with_distance:
                self.logger.info(
                    "No customers with coordinates found. Rebuilding directory..."
                )
                result = self.rebuild_directory()
                
                if result.get("success", False):
                    # Re-run the search after rebuilding
                    return self.find_closest_customers(
                        lat=lat,
                        lng=lng,
                        limit=limit,
                        max_distance=max_distance,
                        exclude_customer_id=exclude_customer_id
                    )
                else:
                    self.logger.error("Failed to rebuild customer directory")
                    raise ToolError(
                        "No customers found with coordinates within the specified range",
                        ErrorCode.TOOL_EXECUTION_ERROR,
                    )
            
            self.logger.info(f"Found {len(results_with_distance)} nearby customers")
            return {"customers": results_with_distance}
            
        except Exception as e:
            self.logger.error(f"Error finding closest customers: {e}")
            raise ToolError(
                f"Error finding closest customers: {e}",
                ErrorCode.TOOL_EXECUTION_ERROR,
            )

    def get_customer(self, customer_id: str) -> Dict[str, Any]:
        """
        Get a customer by ID from the database.
        
        Args:
            customer_id: The ID of the customer to retrieve
            
        Returns:
            Dict with customer information
            
        Raises:
            ToolError: If customer is not found
        """
        customer = self.db.get(Customer, customer_id)
        
        if not customer:
            raise ToolError(
                f"Customer with ID '{customer_id}' not found",
                ErrorCode.TOOL_EXECUTION_ERROR,
            )
            
        return {"customer": customer.to_dict()}

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a customer directory operation.

        Args:
            operation: Operation to perform (see below for valid operations)
            **kwargs: Parameters for the specific operation

        Returns:
            Response data for the operation

        Raises:
            ToolError: If operation fails or parameters are invalid

        Valid Operations:

        1. search_customers: Search for customers in the local directory
           - Required: query (search term - name, email, phone, address)
           - Optional: category (name, given_name, family_name, email, phone, address, any)
           - Returns: Dict with search_type and list of matching customer records

        2. find_closest_customers: Find customers closest to specified coordinates
           - Required: latitude, longitude
           - Optional: limit (default: 1), max_distance (in meters)
           - Returns: List of customer records with distance information

        3. rebuild_directory: Rebuild the customer directory from external source
           - Optional: source (default: "square")
           - Returns: Status of rebuild operation

        4. get_customer: Get a customer by ID
           - Required: customer_id
           - Returns: Customer record
        """
        with error_context(
            component_name=self.name,
            operation=f"executing {operation}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger,
        ):
            # Initialize parameters
            params = {}
            
            # We only support parameters passed through kwargs as a JSON string
            if "kwargs" not in kwargs or not isinstance(kwargs["kwargs"], str):
                raise ToolError(
                    "Parameters must be passed as a JSON string in the 'kwargs' field",
                    ErrorCode.TOOL_INVALID_INPUT
                )
                
            # Parse kwargs JSON string
            try:
                params = json.loads(kwargs["kwargs"])
                self.logger.debug(f"Parsed kwargs JSON: {params}")
            except json.JSONDecodeError as e:
                raise ToolError(
                    f"Invalid JSON in kwargs: {e}", ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Directory Search Operations
            if operation == "search_customers":
                if "query" not in params:
                    raise ToolError(
                        "query parameter is required for search_customers operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                
                # Reject empty queries to prevent returning all records and token explosions
                if params["query"].strip() == "":
                    raise ToolError(
                        "Empty query not allowed. Please provide a specific search term or use find_closest_customers operation for location-based search.",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                    
                return self.search_customers(
                    query=params["query"], category=params.get("category")
                )

            elif operation == "find_closest_customers":
                # Check for required parameters
                lat_param = params.get("lat") or params.get("latitude")
                lng_param = params.get("lng") or params.get("longitude")
                
                if not lat_param or not lng_param:
                    raise ToolError(
                        "latitude/lat and longitude/lng parameters are required for find_closest_customers operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                
                try:
                    lat = float(lat_param)
                    lng = float(lng_param)
                    
                    # Optional parameters
                    limit = int(params["limit"]) if "limit" in params else 1
                    max_distance = float(params["max_distance"]) if "max_distance" in params else None
                    exclude_customer_id = params.get("exclude_customer_id")
                    
                    return self.find_closest_customers(
                        lat=lat,
                        lng=lng,
                        limit=limit,
                        max_distance=max_distance,
                        exclude_customer_id=exclude_customer_id
                    )
                except ValueError:
                    raise ToolError(
                        "lat, lng, limit, and max_distance must be valid numbers",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )

            elif operation == "rebuild_directory":
                source = params.get("source", "square")
                return self.rebuild_directory(source=source)
                
            elif operation == "get_customer":
                if "customer_id" not in params:
                    raise ToolError(
                        "customer_id parameter is required for get_customer operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.get_customer(customer_id=params["customer_id"])

            else:
                raise ToolError(
                    f"Unknown operation: {operation}. Valid operations are: "
                    "search_customers, find_closest_customers, rebuild_directory, get_customer",
                    ErrorCode.TOOL_INVALID_INPUT,
                )