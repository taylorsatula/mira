"""
Square integration tool for business management.

This tool provides a domain-specific interface to Square's APIs for managing
customers, appointments, and services. It abstracts the complexity of the
Square SDK and provides intuitive business-oriented operations.
"""

import logging
import os
import json
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dateutil import parser as date_parser

import httpx
from pydantic import BaseModel, Field

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context
from config.registry import registry
from utils.timezone_utils import validate_timezone, get_default_timezone, convert_to_timezone

# Define configuration class for SquareTool
class SquareToolConfig(BaseModel):
    """Configuration for the square_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    environment: str = Field(default="production", description="Square API environment (sandbox or production)")
    timeout: int = Field(default=30, description="Timeout in seconds for Square API requests")

# Register with registry
registry.register("square_tool", SquareToolConfig)


class SquareTool(Tool):
    """
    Tool for managing Square business operations.
    
    This tool provides domain-specific abstractions for Square's APIs, allowing
    intuitive management of customers, appointments, and services without
    needing to understand the underlying Square SDK complexity.
    
    Features:
    1. Customer Management:
       - Find customers by name, email, or phone
       - Add new customers with proper validation
       - Update customer details
       - Delete customers
       
    2. Appointment Management:
       - List appointments with various filters
       - Find available appointment slots
       - Schedule new appointments
       - Reschedule or cancel existing appointments
       
    3. Service Management:
       - List available services with categories
       - Get detailed service information
       - Add new services to the catalog
       - Update service details
    """
    
    name = "square_tool"
    simple_description = """Integrates with Square for business management operations including customer profiles, bookings/appointments, and service catalog."""
    
    implementation_details = """
    This tool provides intuitive business operations for Square integration:
    
    1. Customer Operations:
       - find_customer: Find customers by name, email, phone, or other criteria
       - add_customer: Create a new customer with contact information
       - update_customer: Modify a customer's information
       - delete_customer: Remove a customer from Square
    
    2. Appointment Operations:
       - list_appointments: View bookings with various filters
       - find_available_slots: Find open appointment times
       - schedule_appointment: Book a new appointment
       - reschedule_appointment: Change an appointment time
       - cancel_appointment: Cancel an existing appointment
       
    3. Service Operations:
       - list_services: Browse available services with optional categories
       - get_service_details: Get comprehensive details about a service
       - add_service: Create a new service in the catalog
       - update_service: Modify service details
    
    Each operation accepts simplified parameters that are mapped to Square's more complex API structure behind the scenes.
    All operations handle proper error reporting and data validation.
    """
    
    description = simple_description + implementation_details
    
    usage_examples = [
        # Example 1: Find a customer
        {
            "input": {
                "operation": "find_customer",
                "kwargs": "{\"query\": \"john.smith@example.com\"}"
            },
            "output": {
                "customers": [
                    {
                        "id": "ABCDEF123456",
                        "first_name": "John",
                        "last_name": "Smith",
                        "email": "john.smith@example.com",
                        "phone": "+1-555-123-4567"
                    }
                ],
                "count": 1
            }
        },
        # Example 2: Schedule an appointment
        {
            "input": {
                "operation": "schedule_appointment",
                "kwargs": "{\"service\": \"Haircut\", \"provider\": \"Jane Stylist\", \"date\": \"2023-12-01\", \"time\": \"10:00 AM\", \"customer_id\": \"ABCDEF123456\"}"
            },
            "output": {
                "appointment": {
                    "id": "BOOKING123456",
                    "service": "Haircut",
                    "provider": "Jane Stylist",
                    "datetime": "2023-12-01T10:00:00-05:00",
                    "duration_minutes": 30,
                    "customer_id": "ABCDEF123456",
                    "status": "ACCEPTED"
                },
                "message": "Appointment scheduled successfully"
            }
        }
    ]
    
    def __init__(self):
        """Initialize the Square tool with SDK client."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._client = None
        
        # Get config
        config = registry.get("square_tool")
        
        # Always prioritize environment variables for environment setting
        env_from_env_var = os.environ.get("SQUARE_ENVIRONMENT")
        if env_from_env_var:
            # Use environment variable value as-is, preserving case
            self.environment = env_from_env_var
        else:
            # Fall back to config if environment variable is not set
            self.environment = config.environment
            
        self.timeout = config.timeout
        
        # Create data directory if needed
        self.data_dir = os.path.join("data", "tools", "square_tool")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger.info("SquareTool initialized")
    
    @property
    def client(self):
        """
        Lazy-load the Square client when first needed.
        
        Returns:
            Square client instance
            
        Raises:
            ToolError: If Square SDK client cannot be initialized
        """
        if self._client is None:
            try:
                # Import Square SDK - this should be available since it's included in project
                from square import Square
                from square.environment import SquareEnvironment
                
                # Get API key from environment
                square_api_key = os.environ.get("SQUARE_API_KEY")
                if not square_api_key:
                    raise ToolError(
                        "Square API key not found in environment variables (SQUARE_API_KEY)",
                        ErrorCode.TOOL_INITIALIZATION_ERROR
                    )
                
                # Determine environment - check both exact case and lowercase for flexibility
                if self.environment == "production" or self.environment.lower() == "production":
                    square_env = SquareEnvironment.PRODUCTION
                else:
                    square_env = SquareEnvironment.SANDBOX
                
                # Create Square client
                self._client = Square(
                    token=square_api_key,
                    environment=square_env,
                    timeout=self.timeout
                )
                self.logger.info(f"Square client initialized in {self.environment} environment")
                
            except ImportError:
                self.logger.error("Failed to import Square SDK")
                raise ToolError(
                    "Failed to import Square SDK. Please make sure it's properly installed.",
                    ErrorCode.TOOL_INITIALIZATION_ERROR
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize Square client: {e}")
                raise ToolError(
                    f"Failed to initialize Square client: {str(e)}",
                    ErrorCode.TOOL_INITIALIZATION_ERROR
                )
                
        return self._client
    
    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a Square business operation.
        
        Args:
            operation: The business operation to perform
            **kwargs: Parameters specific to the operation
            
        Returns:
            Dict containing operation results
            
        Raises:
            ToolError: If operation fails or parameters are invalid
        """
        with error_context(
            component_name=self.name,
            operation=f"executing {operation}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Parse kwargs JSON string if provided
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    params = json.loads(kwargs["kwargs"])
                    kwargs = params
                except json.JSONDecodeError as e:
                    raise ToolError(
                        f"Invalid JSON in kwargs: {e}",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
            
            # Route to appropriate operation method
            # Customer operations
            if operation == "find_customer":
                return self._find_customer(**kwargs)
            elif operation == "add_customer":
                return self._add_customer(**kwargs)
            elif operation == "update_customer":
                return self._update_customer(**kwargs)
            elif operation == "delete_customer":
                return self._delete_customer(**kwargs)
            
            # Service operations
            elif operation == "list_services":
                return self._list_services(**kwargs)
            elif operation == "get_service_details":
                return self._get_service_details(**kwargs)
            
            # Appointment operations
            elif operation == "list_appointments":
                return self._list_appointments(**kwargs)
            elif operation == "find_available_slots":
                return self._find_available_slots(**kwargs)
            elif operation == "schedule_appointment":
                return self._schedule_appointment(**kwargs)
            elif operation == "reschedule_appointment":
                return self._reschedule_appointment(**kwargs)
            elif operation == "cancel_appointment":
                return self._cancel_appointment(**kwargs)
                
            # Handle unknown operations
            else:
                valid_operations = [
                    "find_customer", "add_customer", "update_customer", "delete_customer",
                    "list_services", "get_service_details", "add_service", "update_service",
                    "list_appointments", "find_available_slots", "schedule_appointment", 
                    "reschedule_appointment", "cancel_appointment"
                ]
                raise ToolError(
                    f"Unknown operation: {operation}. Valid operations are: {', '.join(valid_operations)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
    
    # ===== Helper Methods =====
    
    def _format_customer(self, square_customer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Square customer object to a simplified format.
        
        Args:
            square_customer: Square API customer object
            
        Returns:
            Dict with simplified customer information
        """
        result = {
            "id": square_customer.get("id", ""),
            "first_name": square_customer.get("given_name", ""),
            "last_name": square_customer.get("family_name", ""),
            "email": square_customer.get("email_address", ""),
            "phone": square_customer.get("phone_number", ""),
            "created_at": square_customer.get("created_at", ""),
        }
        
        # Add company name if available
        if square_customer.get("company_name"):
            result["company"] = square_customer["company_name"]
        
        # Add note if available
        if square_customer.get("note"):
            result["note"] = square_customer["note"]
            
        # Add reference_id if available
        if square_customer.get("reference_id"):
            result["reference_id"] = square_customer["reference_id"]
            
        # Add address if available
        if square_customer.get("address"):
            result["address"] = square_customer["address"]
            
        return result
    
    # ===== Customer Operations =====
    
    def _find_customer(
        self,
        query: str,
        search_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find customers matching the given query.
        
        Args:
            query: The search term (name, email, phone, etc.)
            search_type: Optional type of search (email, phone, name, or auto-detect)
            limit: Maximum number of results to return
            
        Returns:
            Dict with matching customers
            
        Raises:
            ToolError: If search fails
        """
        self.logger.info(f"Finding customer with query: {query}, type: {search_type}")
        
        # Validate input
        if not query:
            raise ToolError(
                "Query is required for finding customers",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Auto-detect search type if not provided
        if not search_type:
            if "@" in query:
                search_type = "email"
            elif any(c.isdigit() for c in query):
                search_type = "phone"
            else:
                search_type = "name"
        
        try:
            # Build search query based on search type
            search_query = {}
            
            if search_type == "email":
                # Email search
                search_query = {
                    "filter": {
                        "email_address": {"fuzzy": query}
                    }
                }
            elif search_type == "phone":
                # Phone search
                search_query = {
                    "filter": {
                        "phone_number": {"fuzzy": query}
                    }
                }
            elif search_type == "name" or search_type == "customer":
                # Name search - combines first and last name
                parts = query.strip().split()
                if len(parts) > 1:
                    # We have both first and last name
                    search_query = {
                        "filter": {
                            "given_name": {"fuzzy": parts[0]},
                            "family_name": {"fuzzy": parts[-1]}
                        }
                    }
                else:
                    # Just one name part, search in both fields
                    search_query = {
                        "filter": {
                            "given_name": {"fuzzy": query}
                        }
                    }
            
            # Handle sorting and limit
            if limit:
                search_query["limit"] = limit
            
            search_query["sort"] = {"field": "CREATED_AT", "order": "DESC"}
            
            # Execute the search using Square SDK
            result = self.client.customers.search(**search_query)
            
            # Format the customer objects into our simplified format
            formatted_customers = []
            if result.customers:
                for customer in result.customers:
                    formatted_customers.append(self._format_customer(customer))
            
            return {
                "customers": formatted_customers,
                "count": len(formatted_customers),
                "search_type": search_type,
                "query": query
            }
        
        except Exception as e:
            self.logger.error(f"Error finding customer: {e}")
            raise ToolError(
                f"Failed to search for customer: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _add_customer(
        self, 
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        address: Optional[Dict[str, Any]] = None,
        company: Optional[str] = None,
        note: Optional[str] = None,
        reference_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a new customer to Square.
        
        Args:
            first_name: Customer's first/given name
            last_name: Customer's last/family name
            email: Customer's email address
            phone: Customer's phone number
            address: Dict with address components
            company: Customer's company name
            note: Additional notes about the customer
            reference_id: External reference ID
            
        Returns:
            Dict with created customer information
            
        Raises:
            ToolError: If customer creation fails or required fields missing
        """
        self.logger.info(f"Adding customer: {first_name} {last_name}")
        
        # Validate that at least one identifying field is provided
        if not any([first_name, last_name, email, phone, company]):
            raise ToolError(
                "At least one of first_name, last_name, email, phone, or company must be provided",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            # Prepare parameters for Square API
            customer_params = {}
            
            # Add the parameters that are provided
            if first_name:
                customer_params["given_name"] = first_name
                
            if last_name:
                customer_params["family_name"] = last_name
                
            if email:
                customer_params["email_address"] = email
                
            if phone:
                customer_params["phone_number"] = phone
                
            if company:
                customer_params["company_name"] = company
                
            if note:
                customer_params["note"] = note
                
            if reference_id:
                customer_params["reference_id"] = reference_id
                
            if address:
                customer_params["address"] = address
            
            # Create the customer
            result = self.client.customers.create(**customer_params)
            
            # Format the response
            formatted_customer = self._format_customer(result.customer)
            
            return {
                "customer": formatted_customer,
                "message": f"Customer created successfully with ID: {formatted_customer['id']}"
            }
            
        except Exception as e:
            self.logger.error(f"Error adding customer: {e}")
            raise ToolError(
                f"Failed to add customer: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _update_customer(
        self,
        customer_id: str,
        **update_fields
    ) -> Dict[str, Any]:
        """
        Update an existing customer in Square.
        
        Args:
            customer_id: The ID of the customer to update
            **update_fields: Fields to update (first_name/given_name, last_name/family_name, email, etc.)
            
        Returns:
            Dict with updated customer information
            
        Raises:
            ToolError: If customer update fails or id is missing
        """
        self.logger.info(f"Updating customer: {customer_id}")
        
        # Validate customer_id
        if not customer_id:
            raise ToolError(
                "customer_id is required for updating a customer",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Validate that update fields are provided
        if not update_fields:
            raise ToolError(
                "At least one field to update must be provided",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            # Map our field names to Square's field names if needed
            mapped_fields = {}
            
            # Handle field name mapping
            field_mapping = {
                "first_name": "given_name",
                "last_name": "family_name",
                "email": "email_address",
                "phone": "phone_number",
                "company": "company_name"
            }
            
            for field, value in update_fields.items():
                # Map the field name if it needs mapping
                square_field = field_mapping.get(field, field)
                mapped_fields[square_field] = value
            
            # Get current customer version first (for optimistic concurrency)
            current = self.client.customers.get(customer_id=customer_id)
            version = current.customer.get("version", 0)
            mapped_fields["version"] = version
            
            # Update the customer
            result = self.client.customers.update(
                customer_id=customer_id,
                **mapped_fields
            )
            
            # Format the response
            formatted_customer = self._format_customer(result.customer)
            
            return {
                "customer": formatted_customer,
                "message": f"Customer updated successfully: {formatted_customer['id']}"
            }
            
        except Exception as e:
            self.logger.error(f"Error updating customer: {e}")
            raise ToolError(
                f"Failed to update customer: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _delete_customer(
        self,
        customer_id: str
    ) -> Dict[str, Any]:
        """
        Delete a customer from Square.
        
        Args:
            customer_id: The ID of the customer to delete
            
        Returns:
            Dict with deletion confirmation
            
        Raises:
            ToolError: If customer deletion fails or id is missing
        """
        self.logger.info(f"Deleting customer: {customer_id}")
        
        # Validate customer_id
        if not customer_id:
            raise ToolError(
                "customer_id is required for deleting a customer",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            # First, get the customer to confirm it exists
            try:
                current = self.client.customers.get(customer_id=customer_id)
                customer_name = f"{current.customer.get('given_name', '')} {current.customer.get('family_name', '')}".strip()
            except Exception:
                # Customer might not exist
                raise ToolError(
                    f"Customer with ID '{customer_id}' not found",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
            
            # Delete the customer
            self.client.customers.delete(customer_id=customer_id)
            
            return {
                "success": True,
                "customer_id": customer_id,
                "message": f"Customer {customer_name} (ID: {customer_id}) deleted successfully"
            }
            
        except ToolError as e:
            # Re-raise ToolErrors
            raise e
        except Exception as e:
            self.logger.error(f"Error deleting customer: {e}")
            raise ToolError(
                f"Failed to delete customer: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    # ===== Helper Methods - Catalog =====
    
    def _format_service(self, catalog_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Square catalog item object to a simplified service format.
        
        Args:
            catalog_item: Square API catalog item object
            
        Returns:
            Dict with simplified service information
        """
        result = {
            "id": catalog_item.get("id", ""),
            "type": catalog_item.get("type", ""),
            "name": catalog_item.get("item_data", {}).get("name", ""),
            "description": catalog_item.get("item_data", {}).get("description", ""),
            "version": catalog_item.get("version", 0),
        }
        
        # Add variations if available
        variations = catalog_item.get("item_data", {}).get("variations", [])
        if variations:
            result["variations"] = []
            for var in variations:
                var_data = var.get("item_variation_data", {})
                var_price = var_data.get("price_money", {})
                
                variation_info = {
                    "id": var.get("id", ""),
                    "name": var_data.get("name", "")
                }
                
                # Only add price if it exists
                if var_price and "amount" in var_price:
                    var_amount = var_price.get("amount", 0)
                    var_currency = var_price.get("currency", "USD")
                    
                    variation_info["price"] = {
                        "amount": var_amount,
                        "currency": var_currency,
                        "formatted": f"{(var_amount / 100.0):.2f} {var_currency}"
                    }
                
                # Add duration if available
                if var_data.get("service_duration"):
                    # Convert from milliseconds to minutes
                    duration_ms = var_data.get("service_duration")
                    duration_minutes = duration_ms / (60 * 1000)
                    variation_info["duration_minutes"] = duration_minutes
                
                result["variations"].append(variation_info)
        
        # Add category information if available
        if catalog_item.get("category_id"):
            result["category_id"] = catalog_item.get("category_id")
        
        # Add updated/created timestamps
        if catalog_item.get("updated_at"):
            result["updated_at"] = catalog_item.get("updated_at")
        if catalog_item.get("created_at"):
            result["created_at"] = catalog_item.get("created_at")
            
        return result
    
    def _resolve_service_id(self, service_name_or_id: str) -> Union[str, Dict[str, Any]]:
        """
        Resolve a service name or ID to the actual service ID.
        If no exact match is found, returns all potential matches for manual selection.
        
        Args:
            service_name_or_id: Name or ID of the service
            
        Returns:
            Either the service ID (string) if exact match is found,
            or a dictionary with potential matches for selection
            
        Raises:
            ToolError: If service cannot be found
        """
        # If it looks like an ID already, just return it
        if service_name_or_id.startswith("#") or "_" in service_name_or_id:
            return service_name_or_id
            
        # Search for the service by name
        try:
            # Build search query
            search_query = {
                "query": {
                    "prefix_query": {
                        "attribute_name": "name",
                        "attribute_prefix": service_name_or_id
                    }
                },
                "object_types": ["ITEM"]  # Only search for items
            }
            
            # Execute search
            result = self.client.catalog.search_catalog_objects(**search_query)
            
            if not result.objects:
                raise ToolError(
                    f"No service found with name '{service_name_or_id}'",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            # Filter items that are actual services
            service_items = [item for item in result.objects if item.get("type") == "ITEM"]
            
            if not service_items:
                raise ToolError(
                    f"No service found with name '{service_name_or_id}'",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            # First try exact case-sensitive match
            for item in service_items:
                item_name = item.get("item_data", {}).get("name", "")
                if item_name == service_name_or_id:
                    return item.get("id")
            
            # Then try case-insensitive match
            for item in service_items:
                item_name = item.get("item_data", {}).get("name", "")
                if item_name.lower() == service_name_or_id.lower():
                    return item.get("id")
            
            # If no exact match, return all potential matches
            potential_matches = {
                "matches": [
                    {
                        "id": item.get("id"),
                        "name": item.get("item_data", {}).get("name", "")
                    }
                    for item in service_items
                ],
                "message": f"Multiple potential matches found for '{service_name_or_id}'. Please specify the exact service name or ID."
            }
            return potential_matches
            
        except Exception as e:
            if isinstance(e, ToolError):
                raise e
            self.logger.error(f"Error resolving service ID: {e}")
            raise ToolError(
                f"Failed to resolve service '{service_name_or_id}': {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    # ===== Service Operations =====
    
    def _list_services(
        self,
        category: Optional[str] = None,
        include_deleted: bool = False,
        limit: Optional[int] = None,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """
        List available services/catalog items.
        
        Args:
            category: Optional category to filter services
            include_deleted: Whether to include deleted/archived items
            limit: Maximum number of services to return
            include_details: Whether to include complete variation details
            
        Returns:
            Dict with list of services
            
        Raises:
            ToolError: If listing services fails
        """
        self.logger.info(f"Listing services with category: {category}")
        
        try:
            # Build search query for catalog items
            search_query = {
                "object_types": ["ITEM"],  # Only include items, not taxes, discounts, etc.
            }
            
            # Add category filter if provided
            if category:
                # First, try to resolve category ID if it's a name
                category_id = category
                category_name = None
                try:
                    # Check if we need to resolve category name to ID
                    if not (category.startswith("#") or "_" in category):
                        # Search for category by name
                        cat_result = self.client.catalog.search_catalog_objects(
                            object_types=["CATEGORY"],
                            query={
                                "exact_query": {
                                    "attribute_name": "name",
                                    "attribute_value": category
                                }
                            }
                        )
                        if cat_result.objects:
                            category_obj = cat_result.objects[0]
                            category_id = category_obj.get("id")
                            category_name = category_obj.get("category_data", {}).get("name", category)
                    
                    # Build category filter query
                    search_query["query"] = {
                        "exact_query": {
                            "attribute_name": "category_id",
                            "attribute_value": category_id
                        }
                    }
                except Exception as e:
                    self.logger.warning(f"Error resolving category: {e}, proceeding without category filter")
            
            # Execute the search
            result = self.client.catalog.search_catalog_objects(**search_query)
            
            # Also get all categories to provide names
            try:
                categories_result = self.client.catalog.search_catalog_objects(
                    object_types=["CATEGORY"]
                )
                categories = {}
                if categories_result.objects:
                    for cat in categories_result.objects:
                        cat_id = cat.get("id")
                        cat_name = cat.get("category_data", {}).get("name", "")
                        if cat_id:
                            categories[cat_id] = cat_name
            except Exception as e:
                self.logger.warning(f"Error fetching categories: {e}")
                categories = {}
            
            # Format the results
            services = []
            
            if result.objects:
                for item in result.objects:
                    # Skip deleted items unless include_deleted is True
                    if item.get("is_deleted", False) and not include_deleted:
                        continue
                    
                    # Format the service with proper variation handling
                    service = {
                        "id": item.get("id", ""),
                        "type": item.get("type", ""),
                        "name": item.get("item_data", {}).get("name", ""),
                        "description": item.get("item_data", {}).get("description", ""),
                    }
                    
                    # Add category information
                    category_id = item.get("item_data", {}).get("category_id")
                    if category_id:
                        service["category_id"] = category_id
                        # Add category name if we have it
                        if category_id in categories:
                            service["category_name"] = categories[category_id]
                    
                    # Add variations with proper handling
                    variations = item.get("item_data", {}).get("variations", [])
                    if variations:
                        # Determine if this is a variable or fixed price service
                        has_fixed_price = False
                        has_variable_price = False
                        has_empty_default = False
                        
                        # Check variation pricing types
                        for var in variations:
                            var_data = var.get("item_variation_data", {})
                            pricing_type = var_data.get("pricing_type", "")
                            
                            if pricing_type == "FIXED_PRICING" and var_data.get("price_money"):
                                has_fixed_price = True
                            elif pricing_type == "VARIABLE_PRICING":
                                has_variable_price = True
                                
                            # Check for the empty default price case
                            if var_data.get("name", "").lower() == "regular" and pricing_type == "VARIABLE_PRICING":
                                has_empty_default = True
                        
                        # Add pricing type summary to help client understand
                        if has_fixed_price and has_variable_price:
                            service["pricing_type"] = "MIXED"
                        elif has_fixed_price:
                            service["pricing_type"] = "FIXED"
                        else:
                            service["pricing_type"] = "VARIABLE"
                            
                        if has_empty_default:
                            service["has_default_variation"] = True
                            
                        # Format all variations
                        if include_details:
                            service["variations"] = []
                            for var in variations:
                                var_data = var.get("item_variation_data", {})
                                var_info = {
                                    "id": var.get("id", ""),
                                    "name": var_data.get("name", ""),
                                    "pricing_type": var_data.get("pricing_type", "VARIABLE_PRICING")
                                }
                                
                                # Add price if available
                                if var_data.get("pricing_type") == "FIXED_PRICING" and var_data.get("price_money"):
                                    price_data = var_data.get("price_money", {})
                                    var_info["price"] = {
                                        "amount": price_data.get("amount", 0),
                                        "currency": price_data.get("currency", "USD"),
                                        "formatted": f"{(price_data.get('amount', 0) / 100.0):.2f} {price_data.get('currency', 'USD')}"
                                    }
                                    
                                # Add duration if available
                                if var_data.get("service_duration"):
                                    duration_ms = var_data.get("service_duration")
                                    duration_minutes = duration_ms / (60 * 1000)  # Convert to minutes
                                    var_info["duration_minutes"] = duration_minutes
                                    
                                service["variations"].append(var_info)
                        else:
                            # Just add variation count
                            service["variation_count"] = len(variations)
                            
                            # Add price range if fixed pricing exists
                            if has_fixed_price:
                                prices = []
                                for var in variations:
                                    var_data = var.get("item_variation_data", {})
                                    if var_data.get("pricing_type") == "FIXED_PRICING" and var_data.get("price_money"):
                                        prices.append(var_data.get("price_money", {}).get("amount", 0))
                                
                                if prices:
                                    min_price = min(prices)
                                    max_price = max(prices)
                                    
                                    if min_price == max_price:
                                        service["price"] = {
                                            "amount": min_price,
                                            "currency": "USD",
                                            "formatted": f"${(min_price / 100.0):.2f}"
                                        }
                                    else:
                                        service["price_range"] = {
                                            "min": {
                                                "amount": min_price,
                                                "formatted": f"${(min_price / 100.0):.2f}"
                                            },
                                            "max": {
                                                "amount": max_price,
                                                "formatted": f"${(max_price / 100.0):.2f}"
                                            },
                                            "currency": "USD"
                                        }
                    
                    # Add service to the list
                    services.append(service)
            
            # Apply limit if provided
            if limit and limit > 0 and len(services) > limit:
                services = services[:limit]
                
            return {
                "services": services,
                "count": len(services),
                "category": category_name if category_name else category
            }
            
        except Exception as e:
            self.logger.error(f"Error listing services: {e}")
            raise ToolError(
                f"Failed to list services: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _get_service_details(
        self,
        service_id: Optional[str] = None,
        service_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific service.
        
        Args:
            service_id: The ID of the service
            service_name: The name of the service (used if ID not provided)
            
        Returns:
            Dict with service details or potential matches if ambiguous
            
        Raises:
            ToolError: If service retrieval fails or identifiers missing
        """
        self.logger.info(f"Getting service details for: {service_id or service_name}")
        
        # Validate that at least one identifier is provided
        if not service_id and not service_name:
            raise ToolError(
                "Either service_id or service_name must be provided",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            # If only name is provided, resolve it to ID
            if not service_id and service_name:
                resolved = self._resolve_service_id(service_name)
                
                # Check if we got multiple matches
                if isinstance(resolved, dict) and "matches" in resolved:
                    return resolved
                
                service_id = resolved
            
            # Get the service details
            result = self.client.catalog.object.retrieve(object_id=service_id)
            
            # Format the response
            service = self._format_service(result.object)
            
            # Add related objects if available (like categories)
            if result.related_objects:
                for related in result.related_objects:
                    if related.get("type") == "CATEGORY" and related.get("id") == service.get("category_id"):
                        service["category_name"] = related.get("category_data", {}).get("name", "")
            
            return {
                "service": service
            }
            
        except Exception as e:
            self.logger.error(f"Error getting service details: {e}")
            raise ToolError(
                f"Failed to get service details: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _add_service(
        self,
        name: str,
        price: Optional[Union[float, int, str]] = None,
        duration_minutes: Optional[int] = None,
        category: Optional[str] = None,
        description: Optional[str] = None,
        variations: Optional[List[Dict[str, Any]]] = None,
        **additional_properties
    ) -> Dict[str, Any]:
        """
        Add a new service to the catalog.
        
        Args:
            name: Service name
            price: Service price in dollars (will be converted to cents)
            duration_minutes: Service duration in minutes
            category: Optional category for the service
            description: Optional service description
            variations: Optional list of variations with their own prices
            **additional_properties: Any additional service properties
            
        Returns:
            Dict with created service information
            
        Raises:
            ToolError: If service creation fails or required fields missing
        """
        self.logger.info(f"Adding service: {name}")
        
        # Validate required fields
        if not name:
            raise ToolError(
                "Service name is required",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Either price or variations should be provided
        if not price and not variations:
            raise ToolError(
                "Either service price or variations must be provided",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            import uuid
            
            # Convert price to cents for Square API if provided
            price_cents = None
            if price is not None:
                if isinstance(price, str):
                    # Try to parse from string with currency
                    try:
                        price_float = float(''.join(c for c in price if c.isdigit() or c == '.'))
                        price_cents = int(price_float * 100)
                    except ValueError:
                        raise ToolError(
                            f"Invalid price format: {price}",
                            ErrorCode.TOOL_INVALID_INPUT
                        )
                else:
                    # Convert from number
                    price_cents = int(float(price) * 100)
                
            # Resolve category ID if category name is provided
            category_id = None
            if category:
                # Try to find category by name
                try:
                    result = self.client.catalog.search_catalog_objects(
                        object_types=["CATEGORY"],
                        query={
                            "exact_query": {
                                "attribute_name": "name",
                                "attribute_value": category
                            }
                        }
                    )
                    
                    if result.objects:
                        category_id = result.objects[0].get("id")
                    else:
                        # Create the category
                        category_obj = {
                            "type": "CATEGORY",
                            "id": f"#{str(uuid.uuid4())[:8]}",
                            "category_data": {
                                "name": category
                            }
                        }
                        
                        # Batch create the category
                        category_result = self.client.catalog.batch_upsert_catalog_objects(
                            batches=[{
                                "objects": [category_obj]
                            }]
                        )
                        
                        if category_result.objects:
                            category_id = category_result.objects[0].get("id")
                            
                except Exception as e:
                    self.logger.warning(f"Error creating category: {e}, proceeding without category")
            
            # Create item variations
            item_variations = []
            
            if variations:
                # Format provided variations
                for i, var in enumerate(variations):
                    var_name = var.get("name", f"Option {i+1}")
                    var_price = var.get("price")
                    var_duration = var.get("duration_minutes", duration_minutes)
                    
                    # Skip variations without price - they'll have variable pricing
                    variation_data = {
                        "name": var_name,
                        "pricing_type": "VARIABLE_PRICING"  # Default to variable pricing
                    }
                    
                    # Add price if specified
                    if var_price is not None:
                        # Convert variation price to cents
                        if isinstance(var_price, str):
                            try:
                                var_price_float = float(''.join(c for c in var_price if c.isdigit() or c == '.'))
                                var_price_cents = int(var_price_float * 100)
                            except ValueError:
                                raise ToolError(
                                    f"Invalid price format for variation {var_name}: {var_price}",
                                    ErrorCode.TOOL_INVALID_INPUT
                                )
                        else:
                            var_price_cents = int(float(var_price) * 100)
                            
                        # Add fixed price to variation
                        variation_data["pricing_type"] = "FIXED_PRICING"
                        variation_data["price_money"] = {
                            "amount": var_price_cents,
                            "currency": "USD"
                        }
                        
                    # Add service duration if provided
                    if var_duration:
                        variation_data["service_duration"] = var_duration * 60 * 1000  # Convert to milliseconds
                        
                    # Create the variation object
                    variation_obj = {
                        "type": "ITEM_VARIATION",
                        "id": f"#{str(uuid.uuid4())[:8]}",
                        "item_variation_data": variation_data
                    }
                    
                    item_variations.append(variation_obj)
            elif price_cents is not None:
                # Create a single variation with the provided price
                variation_data = {
                    "name": "Regular",
                    "pricing_type": "FIXED_PRICING",
                    "price_money": {
                        "amount": price_cents,
                        "currency": "USD"
                    }
                }
                
                # Add duration if provided
                if duration_minutes:
                    variation_data["service_duration"] = duration_minutes * 60 * 1000  # Convert to milliseconds
                
                variation_obj = {
                    "type": "ITEM_VARIATION",
                    "id": f"#{str(uuid.uuid4())[:8]}",
                    "item_variation_data": variation_data
                }
                
                item_variations.append(variation_obj)
            else:
                # Create a variation with variable pricing
                variation_data = {
                    "name": "Regular",
                    "pricing_type": "VARIABLE_PRICING"
                }
                
                # Add duration if provided
                if duration_minutes:
                    variation_data["service_duration"] = duration_minutes * 60 * 1000  # Convert to milliseconds
                
                variation_obj = {
                    "type": "ITEM_VARIATION",
                    "id": f"#{str(uuid.uuid4())[:8]}",
                    "item_variation_data": variation_data
                }
                
                item_variations.append(variation_obj)
            
            # Create the item object
            item_id = f"#{str(uuid.uuid4())[:8]}"
            item_object = {
                "type": "ITEM",
                "id": item_id,
                "item_data": {
                    "name": name,
                    "description": description or "",
                    "category_id": category_id,
                    "variations": item_variations,
                    "product_type": "APPOINTMENTS_SERVICE"
                }
            }
            
            # Batch upsert the item
            result = self.client.catalog.batch_upsert_catalog_objects(
                idempotency_key=str(uuid.uuid4()),
                batches=[{
                    "objects": [item_object]
                }]
            )
            
            # Parse the result
            if not result.objects:
                raise ToolError(
                    "Failed to create service: No objects returned",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            # Get the created item
            created_item = None
            for obj in result.objects:
                if obj.get("type") == "ITEM" and obj.get("id") == item_id:
                    created_item = obj
                    break
                    
            if not created_item:
                raise ToolError(
                    "Failed to create service: Created item not found in response",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            # Format the response
            service = self._format_service(created_item)
            
            return {
                "service": service,
                "message": f"Service '{name}' created successfully with ID: {service['id']}"
            }
            
        except ToolError as e:
            # Re-raise existing ToolErrors
            raise e
        except Exception as e:
            self.logger.error(f"Error adding service: {e}")
            raise ToolError(
                f"Failed to add service: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _update_service(
        self,
        service_id: Optional[str] = None,
        service_name: Optional[str] = None,
        **update_fields
    ) -> Dict[str, Any]:
        """
        Update an existing service in the catalog.
        
        Args:
            service_id: The ID of the service to update
            service_name: The name of the service (used if ID not provided)
            **update_fields: Fields to update (name, price, description, etc.)
            
        Returns:
            Dict with updated service information or potential matches
            
        Raises:
            ToolError: If service update fails or id is missing
        """
        self.logger.info(f"Updating service: {service_id or service_name}")
        
        # Validate that at least one identifier is provided
        if not service_id and not service_name:
            raise ToolError(
                "Either service_id or service_name must be provided",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Validate that update fields are provided
        if not update_fields:
            raise ToolError(
                "At least one field to update must be provided",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            import uuid
            
            # If only name is provided, resolve it to ID
            if not service_id and service_name:
                resolved = self._resolve_service_id(service_name)
                
                # Check if we got multiple matches
                if isinstance(resolved, dict) and "matches" in resolved:
                    return resolved
                
                service_id = resolved
            
            # Get the current service
            current = self.client.catalog.object.retrieve(object_id=service_id)
            
            # Create a copy of the object to modify
            updated_object = current.object.copy()
            
            # Apply updates
            if "name" in update_fields:
                updated_object["item_data"]["name"] = update_fields["name"]
                
            if "description" in update_fields:
                updated_object["item_data"]["description"] = update_fields["description"]
                
            if "price" in update_fields and updated_object["item_data"].get("variations"):
                # Update the primary variation's price if the variation uses fixed pricing
                price = update_fields["price"]
                
                # Only update if the variation uses fixed pricing
                variation = updated_object["item_data"]["variations"][0]
                if variation["item_variation_data"].get("pricing_type") == "FIXED_PRICING":
                    # Convert price to cents
                    if isinstance(price, str):
                        try:
                            price_float = float(''.join(c for c in price if c.isdigit() or c == '.'))
                            price_cents = int(price_float * 100)
                        except ValueError:
                            raise ToolError(
                                f"Invalid price format: {price}",
                                ErrorCode.TOOL_INVALID_INPUT
                            )
                    else:
                        price_cents = int(float(price) * 100)
                    
                    # Update the price
                    if "price_money" not in variation["item_variation_data"]:
                        variation["item_variation_data"]["price_money"] = {
                            "amount": price_cents,
                            "currency": "USD"
                        }
                    else:
                        variation["item_variation_data"]["price_money"]["amount"] = price_cents
                
            if "duration_minutes" in update_fields and updated_object["item_data"].get("variations"):
                # Update all variations with the new duration
                duration_minutes = update_fields["duration_minutes"]
                if not isinstance(duration_minutes, int) or duration_minutes <= 0:
                    raise ToolError(
                        "duration_minutes must be a positive integer",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
                # Convert to milliseconds for Square API
                duration_ms = duration_minutes * 60 * 1000
                
                # Update all variations
                for variation in updated_object["item_data"]["variations"]:
                    variation["item_variation_data"]["service_duration"] = duration_ms
            
            if "category" in update_fields:
                # Resolve category ID if needed
                category = update_fields["category"]
                category_id = None
                
                # Try to find category by name
                try:
                    result = self.client.catalog.search_catalog_objects(
                        object_types=["CATEGORY"],
                        query={
                            "exact_query": {
                                "attribute_name": "name",
                                "attribute_value": category
                            }
                        }
                    )
                    
                    if result.objects:
                        category_id = result.objects[0].get("id")
                    else:
                        # Create the category
                        category_obj = {
                            "type": "CATEGORY",
                            "id": f"#{str(uuid.uuid4())[:8]}",
                            "category_data": {
                                "name": category
                            }
                        }
                        
                        # Batch create the category
                        category_result = self.client.catalog.batch_upsert_catalog_objects(
                            idempotency_key=str(uuid.uuid4()),
                            batches=[{
                                "objects": [category_obj]
                            }]
                        )
                        
                        if category_result.objects:
                            category_id = category_result.objects[0].get("id")
                            
                    # Update the category ID if found or created
                    if category_id:
                        updated_object["item_data"]["category_id"] = category_id
                        
                except Exception as e:
                    self.logger.warning(f"Error updating category: {e}, proceeding without category update")
            
            # Handle variation-specific updates
            if "variations" in update_fields and isinstance(update_fields["variations"], list):
                new_variations = []
                
                for i, var_update in enumerate(update_fields["variations"]):
                    # Find the existing variation if it has an ID
                    existing_var = None
                    var_id = var_update.get("id")
                    
                    if var_id:
                        for var in updated_object["item_data"]["variations"]:
                            if var.get("id") == var_id:
                                existing_var = var
                                break
                    
                    # If existing variation found, update it
                    if existing_var:
                        var_data = existing_var["item_variation_data"]
                        
                        # Update name if provided
                        if "name" in var_update:
                            var_data["name"] = var_update["name"]
                        
                        # Update price if provided
                        if "price" in var_update:
                            var_price = var_update["price"]
                            
                            # Skip empty prices - they will use variable pricing
                            if var_price:
                                # Convert to cents
                                if isinstance(var_price, str):
                                    try:
                                        var_price_float = float(''.join(c for c in var_price if c.isdigit() or c == '.'))
                                        var_price_cents = int(var_price_float * 100)
                                    except ValueError:
                                        raise ToolError(
                                            f"Invalid price format for variation: {var_price}",
                                            ErrorCode.TOOL_INVALID_INPUT
                                        )
                                else:
                                    var_price_cents = int(float(var_price) * 100)
                                
                                # Update pricing type and price
                                var_data["pricing_type"] = "FIXED_PRICING"
                                if "price_money" not in var_data:
                                    var_data["price_money"] = {
                                        "amount": var_price_cents,
                                        "currency": "USD"
                                    }
                                else:
                                    var_data["price_money"]["amount"] = var_price_cents
                            else:
                                # Set to variable pricing
                                var_data["pricing_type"] = "VARIABLE_PRICING"
                                if "price_money" in var_data:
                                    del var_data["price_money"]
                        
                        # Update duration if provided
                        if "duration_minutes" in var_update:
                            var_duration = var_update["duration_minutes"]
                            if var_duration and var_duration > 0:
                                var_data["service_duration"] = var_duration * 60 * 1000
                        
                        new_variations.append(existing_var)
                    else:
                        # Create a new variation
                        var_name = var_update.get("name", f"Option {i+1}")
                        var_price = var_update.get("price")
                        var_duration = var_update.get("duration_minutes")
                        
                        # Build new variation data
                        var_data = {
                            "name": var_name,
                            "pricing_type": "VARIABLE_PRICING"  # Default
                        }
                        
                        # Add price if provided
                        if var_price:
                            # Convert to cents
                            if isinstance(var_price, str):
                                try:
                                    var_price_float = float(''.join(c for c in var_price if c.isdigit() or c == '.'))
                                    var_price_cents = int(var_price_float * 100)
                                except ValueError:
                                    raise ToolError(
                                        f"Invalid price format for variation: {var_price}",
                                        ErrorCode.TOOL_INVALID_INPUT
                                    )
                            else:
                                var_price_cents = int(float(var_price) * 100)
                            
                            # Set fixed pricing
                            var_data["pricing_type"] = "FIXED_PRICING"
                            var_data["price_money"] = {
                                "amount": var_price_cents,
                                "currency": "USD"
                            }
                        
                        # Add duration if provided
                        if var_duration and var_duration > 0:
                            var_data["service_duration"] = var_duration * 60 * 1000
                        
                        # Create variation object
                        new_var = {
                            "type": "ITEM_VARIATION",
                            "id": f"#{str(uuid.uuid4())[:8]}",
                            "item_variation_data": var_data
                        }
                        
                        new_variations.append(new_var)
                
                # Replace variations if we have new ones
                if new_variations:
                    updated_object["item_data"]["variations"] = new_variations
            
            # Update the service
            result = self.client.catalog.batch_upsert_catalog_objects(
                idempotency_key=str(uuid.uuid4()),
                batches=[{
                    "objects": [updated_object]
                }]
            )
            
            # Get the updated object
            if not result.objects:
                raise ToolError(
                    "Failed to update service: No objects returned",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            # Format the response
            service = self._format_service(result.objects[0])
            
            return {
                "service": service,
                "message": f"Service updated successfully: {service['name']}"
            }
            
        except ToolError as e:
            # Re-raise existing ToolErrors
            raise e
        except Exception as e:
            self.logger.error(f"Error updating service: {e}")
            raise ToolError(
                f"Failed to update service: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    # ===== Appointment Operations =====
    
    def _list_appointments(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        customer_id: Optional[str] = None,
        customer_email: Optional[str] = None,
        provider_id: Optional[str] = None,
        location_id: Optional[str] = None,
        status: Optional[str] = None,
        timezone: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List appointments/bookings with various filters.
        
        Args:
            start_date: Optional start date for filtering (YYYY-MM-DD)
            end_date: Optional end date for filtering (YYYY-MM-DD)
            customer_id: Optional customer ID to filter by
            customer_email: Optional customer email to filter by (will be resolved to ID)
            provider_id: Optional provider/team member ID to filter by
            location_id: Optional location ID to filter by
            status: Optional status to filter by (ACCEPTED, PENDING, etc.)
            timezone: Optional timezone for date/time formatting
            
        Returns:
            Dict with list of appointments
            
        Raises:
            ToolError: If listing appointments fails
        """
        self.logger.info(f"Listing appointments from {start_date} to {end_date}")
        
        # Validate and process timezone if provided
        tz = None
        if timezone:
            try:
                tz = validate_timezone(timezone)
            except Exception as e:
                raise ToolError(
                    f"Invalid timezone: {str(e)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
        else:
            # Use default timezone
            tz = get_default_timezone()
        
        try:
            # Process date inputs
            start_datetime = None
            end_datetime = None
            
            if start_date:
                try:
                    # Parse start date
                    start_datetime = date_parser.parse(start_date)
                    # Set to beginning of day
                    start_datetime = start_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
                except Exception as e:
                    raise ToolError(
                        f"Invalid start_date format: {str(e)}. Use YYYY-MM-DD format.",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
            
            if end_date:
                try:
                    # Parse end date
                    end_datetime = date_parser.parse(end_date)
                    # Set to end of day
                    end_datetime = end_datetime.replace(hour=23, minute=59, second=59, microsecond=999999)
                except Exception as e:
                    raise ToolError(
                        f"Invalid end_date format: {str(e)}. Use YYYY-MM-DD format.",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
            
            # Default date range if not provided
            if not start_datetime:
                # Default to today
                start_datetime = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            if not end_datetime:
                # Default to 7 days from start
                end_datetime = start_datetime + datetime.timedelta(days=7)
                end_datetime = end_datetime.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            # Format dates for Square API - RFC 3339 format
            start_at_str = start_datetime.isoformat() + 'Z'
            end_at_str = end_datetime.isoformat() + 'Z'
            
            # Process customer filter
            if customer_email and not customer_id:
                # Try to resolve email to customer ID
                try:
                    self.logger.info(f"Resolving customer email: {customer_email}")
                    customer_results = self._find_customer(query=customer_email, search_type="email")
                    if customer_results.get("count") > 0:
                        customer_id = customer_results["customers"][0]["id"]
                        self.logger.info(f"Resolved customer email to ID: {customer_id}")
                    else:
                        raise ToolError(
                            f"No customer found with email: {customer_email}",
                            ErrorCode.TOOL_EXECUTION_ERROR
                        )
                except Exception as e:
                    if isinstance(e, ToolError):
                        raise e
                    self.logger.error(f"Error resolving customer email: {e}")
                    raise ToolError(
                        f"Failed to resolve customer email: {str(e)}",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
            
            # Prepare parameters for listing bookings
            list_params = {
                "start_at_min": start_at_str,
                "start_at_max": end_at_str,
                "limit": 100  # Use a reasonable limit
            }
            
            # Add optional filters
            if customer_id:
                list_params["customer_id"] = customer_id
            
            if provider_id:
                list_params["team_member_id"] = provider_id
            
            if location_id:
                list_params["location_id"] = location_id
            
            # Get all bookings matching the filters
            self.logger.info(f"Fetching bookings with params: {list_params}")
            
            # Use the SDK bookings.list method to get all bookings
            booking_list = self.client.bookings.list(**list_params)
            
            # Initialize services and team members caches to reduce API calls
            service_cache = {}
            team_member_cache = {}
            
            # Format the bookings into a simplified structure
            appointments = []
            
            # Iterate through paginated results
            for booking in booking_list:
                # Skip bookings that don't match the status filter
                if status and booking.status.upper() != status.upper():
                    continue
                
                # Parse the booking start time
                booking_time = None
                if booking.start_at:
                    try:
                        booking_time = date_parser.parse(booking.start_at)
                        # Convert to user's timezone
                        booking_time = booking_time.astimezone(tz)
                    except Exception as e:
                        self.logger.warning(f"Error parsing booking time {booking.start_at}: {e}")
                
                # Format a user-friendly time display
                formatted_time = None
                if booking_time:
                    formatted_time = booking_time.strftime("%Y-%m-%d %I:%M %p %Z")
                
                # Initialize the appointment info
                appointment_info = {
                    "id": booking.id,
                    "status": booking.status,
                    "start_at": booking.start_at,
                    "formatted_time": formatted_time,
                    "customer_id": booking.customer_id,
                    "location_id": booking.location_id,
                }
                
                # Add customer note if available
                if booking.customer_note:
                    appointment_info["customer_note"] = booking.customer_note
                
                # Add seller note if available (for internal use)
                if booking.seller_note:
                    appointment_info["seller_note"] = booking.seller_note
                
                # Process appointment segments
                if booking.appointment_segments and len(booking.appointment_segments) > 0:
                    segments = []
                    total_duration = 0
                    
                    for idx, segment in enumerate(booking.appointment_segments):
                        segment_info = {
                            "team_member_id": segment.team_member_id,
                            "service_variation_id": segment.service_variation_id,
                        }
                        
                        # Add duration if available
                        if segment.duration_minutes:
                            segment_info["duration_minutes"] = segment.duration_minutes
                            total_duration += segment.duration_minutes
                        
                        # Fetch service name if not in cache
                        if segment.service_variation_id:
                            if segment.service_variation_id not in service_cache:
                                try:
                                    # Get the service details from catalog
                                    service_result = self.client.catalog.object.retrieve(
                                        object_id=segment.service_variation_id
                                    )
                                    
                                    # Get the name from the parent item if this is a variation
                                    if service_result.object.type == "ITEM_VARIATION":
                                        parent_id = service_result.object.item_variation_data.item_id
                                        variation_name = service_result.object.item_variation_data.name
                                        
                                        # Try to get the parent item
                                        try:
                                            parent_result = self.client.catalog.object.retrieve(object_id=parent_id)
                                            parent_name = parent_result.object.item_data.name
                                            
                                            # Combine parent and variation name
                                            service_name = f"{parent_name} - {variation_name}"
                                        except Exception:
                                            # Just use variation name if parent not found
                                            service_name = variation_name
                                    else:
                                        # Use the item name directly
                                        service_name = service_result.object.item_data.name
                                    
                                    service_cache[segment.service_variation_id] = service_name
                                except Exception as e:
                                    self.logger.warning(f"Error retrieving service name: {e}")
                                    service_cache[segment.service_variation_id] = "Unknown Service"
                            
                            # Add service name to segment info
                            segment_info["service_name"] = service_cache[segment.service_variation_id]
                        
                        # Fetch team member name if not in cache
                        if segment.team_member_id:
                            if segment.team_member_id not in team_member_cache:
                                try:
                                    # Get the team member details
                                    member_result = self.client.team.get_team_member(
                                        team_member_id=segment.team_member_id
                                    )
                                    
                                    # Format the name
                                    given_name = member_result.team_member.given_name or ""
                                    family_name = member_result.team_member.family_name or ""
                                    member_name = f"{given_name} {family_name}".strip()
                                    
                                    if not member_name:
                                        member_name = "Unknown Provider"
                                    
                                    team_member_cache[segment.team_member_id] = member_name
                                except Exception as e:
                                    self.logger.warning(f"Error retrieving team member name: {e}")
                                    team_member_cache[segment.team_member_id] = "Unknown Provider"
                            
                            # Add team member name to segment info
                            segment_info["provider_name"] = team_member_cache[segment.team_member_id]
                        
                        segments.append(segment_info)
                    
                    # Add segments to appointment
                    appointment_info["segments"] = segments
                    appointment_info["duration_minutes"] = total_duration
                    
                    # Add primary service and provider for convenience
                    if segments:
                        primary_segment = segments[0]
                        if "service_name" in primary_segment:
                            appointment_info["service"] = primary_segment["service_name"]
                        if "provider_name" in primary_segment:
                            appointment_info["provider"] = primary_segment["provider_name"]
                
                # Add creation and update times
                if booking.created_at:
                    appointment_info["created_at"] = booking.created_at
                if booking.updated_at:
                    appointment_info["updated_at"] = booking.updated_at
                
                # Try to get customer name if ID is available
                if booking.customer_id:
                    try:
                        customer = self.client.customers.get(customer_id=booking.customer_id).customer
                        customer_name = f"{customer.get('given_name', '')} {customer.get('family_name', '')}".strip()
                        
                        if customer_name:
                            appointment_info["customer_name"] = customer_name
                        
                        # Add email and phone for reference
                        if customer.get("email_address"):
                            appointment_info["customer_email"] = customer.get("email_address")
                        if customer.get("phone_number"):
                            appointment_info["customer_phone"] = customer.get("phone_number")
                    except Exception as e:
                        self.logger.warning(f"Error retrieving customer details: {e}")
                
                # Try to get location name if ID is available
                if booking.location_id:
                    try:
                        location = self.client.locations.get(location_id=booking.location_id).location
                        if location.get("name"):
                            appointment_info["location_name"] = location.get("name")
                    except Exception as e:
                        self.logger.warning(f"Error retrieving location details: {e}")
                
                # Add appointment to the list
                appointments.append(appointment_info)
            
            # Sort appointments by start time
            appointments.sort(key=lambda x: x.get("start_at", ""), reverse=False)
            
            # Format the date range for response
            formatted_start = start_datetime.strftime("%Y-%m-%d")
            formatted_end = end_datetime.strftime("%Y-%m-%d")
            
            return {
                "appointments": appointments,
                "count": len(appointments),
                "start_date": formatted_start,
                "end_date": formatted_end,
                "filters": {
                    "customer_id": customer_id,
                    "customer_email": customer_email,
                    "provider_id": provider_id,
                    "location_id": location_id,
                    "status": status
                }
            }
            
        except ToolError as e:
            # Re-raise existing ToolErrors
            raise e
        except Exception as e:
            self.logger.error(f"Error listing appointments: {e}")
            raise ToolError(
                f"Failed to list appointments: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _find_available_slots(
        self,
        service: str,  # Can be name or ID
        date: Optional[str] = None,
        provider: Optional[str] = None,  # Can be name or ID
        location_id: Optional[str] = None,
        timezone: Optional[str] = None,
        duration_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find available appointment slots for a service.
        
        Args:
            service: Service name or ID
            date: Optional date to check (defaults to today)
            provider: Optional provider/team member name or ID
            location_id: Optional location ID
            timezone: Optional timezone for formatting times
            duration_minutes: Optional duration override (useful for variable duration services)
            
        Returns:
            Dict with available slots
            
        Raises:
            ToolError: If finding available slots fails
        """
        self.logger.info(f"Finding available slots for service: {service}, date: {date}")
        
        # Validate service
        if not service:
            raise ToolError(
                "Service name or ID is required for finding available slots",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Validate and process timezone
        tz = None
        if timezone:
            try:
                tz = validate_timezone(timezone)
            except Exception as e:
                raise ToolError(
                    f"Invalid timezone: {str(e)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
        else:
            # Use default timezone
            tz = get_default_timezone()
        
        try:
            # Resolve service ID if name was provided
            service_variation_id = None
            service_name = None
            
            # Check if service looks like a service variation ID or needs to be resolved
            if service.startswith("#") or "_" in service:
                service_variation_id = service
                
                # Try to get the service name for display
                try:
                    service_result = self.client.catalog.object.retrieve(object_id=service_variation_id)
                    if service_result.object.type == "ITEM_VARIATION":
                        variation_name = service_result.object.item_variation_data.name
                        parent_id = service_result.object.item_variation_data.item_id
                        
                        # Try to get parent name
                        try:
                            parent_result = self.client.catalog.object.retrieve(object_id=parent_id)
                            parent_name = parent_result.object.item_data.name
                            service_name = f"{parent_name} - {variation_name}"
                        except Exception:
                            service_name = variation_name
                    elif service_result.object.type == "ITEM":
                        service_name = service_result.object.item_data.name
                        # We need a variation, not an item
                        if service_result.object.item_data.variations and len(service_result.object.item_data.variations) > 0:
                            # Use first variation
                            service_variation_id = service_result.object.item_data.variations[0].id
                            variation_name = service_result.object.item_data.variations[0].item_variation_data.name
                            service_name = f"{service_name} - {variation_name}"
                except Exception as e:
                    self.logger.warning(f"Error retrieving service details: {e}")
            else:
                # Need to resolve service name to ID
                resolved = self._resolve_service_id(service)
                
                # Check if we got multiple matches
                if isinstance(resolved, dict) and "matches" in resolved:
                    return {
                        "status": "ambiguous",
                        "service_matches": resolved["matches"],
                        "message": resolved["message"]
                    }
                
                service_variation_id = resolved
                service_name = service  # Use provided name until we get the real one
                
                # Try to get the full service name
                try:
                    service_result = self.client.catalog.object.retrieve(object_id=service_variation_id)
                    if service_result.object.type == "ITEM_VARIATION":
                        variation_name = service_result.object.item_variation_data.name
                        parent_id = service_result.object.item_variation_data.item_id
                        
                        # Try to get parent name
                        try:
                            parent_result = self.client.catalog.object.retrieve(object_id=parent_id)
                            parent_name = parent_result.object.item_data.name
                            service_name = f"{parent_name} - {variation_name}"
                        except Exception:
                            service_name = variation_name
                            
                        # Get duration if not provided
                        if duration_minutes is None and service_result.object.item_variation_data.service_duration:
                            # Convert from milliseconds to minutes
                            service_duration_ms = service_result.object.item_variation_data.service_duration
                            duration_minutes = int(service_duration_ms / (60 * 1000))
                    elif service_result.object.type == "ITEM":
                        service_name = service_result.object.item_data.name
                        # We need a variation, not an item
                        if service_result.object.item_data.variations and len(service_result.object.item_data.variations) > 0:
                            # Use first variation
                            service_variation_id = service_result.object.item_data.variations[0].id
                            variation_name = service_result.object.item_data.variations[0].item_variation_data.name
                            service_name = f"{service_name} - {variation_name}"
                            
                            # Get duration if not provided
                            if duration_minutes is None and service_result.object.item_data.variations[0].item_variation_data.service_duration:
                                service_duration_ms = service_result.object.item_data.variations[0].item_variation_data.service_duration
                                duration_minutes = int(service_duration_ms / (60 * 1000))
                except Exception as e:
                    self.logger.warning(f"Error retrieving service details: {e}")
            
            # Resolve provider if name is provided
            team_member_id = None
            provider_name = None
            
            if provider:
                # Check if this looks like an ID already
                if provider.startswith("TMxxx") or "_" in provider:
                    team_member_id = provider
                    
                    # Try to get provider name
                    try:
                        member_result = self.client.team.get_team_member(team_member_id=team_member_id)
                        given_name = member_result.team_member.given_name or ""
                        family_name = member_result.team_member.family_name or ""
                        provider_name = f"{given_name} {family_name}".strip()
                    except Exception as e:
                        self.logger.warning(f"Error retrieving team member details: {e}")
                        provider_name = provider  # Default to ID if name can't be retrieved
                else:
                    # Need to search for team member by name
                    try:
                        # First try exact name match
                        team_result = self.client.team.search_team_members(
                            query={
                                "filter": {
                                    "status": "ACTIVE"
                                }
                            }
                        )
                        
                        # Try to find matching team member
                        found = False
                        if team_result.team_members:
                            for member in team_result.team_members:
                                given_name = member.given_name or ""
                                family_name = member.family_name or ""
                                member_name = f"{given_name} {family_name}".strip()
                                
                                # Check for match (case insensitive)
                                if member_name.lower() == provider.lower():
                                    team_member_id = member.id
                                    provider_name = member_name
                                    found = True
                                    break
                            
                            # If not found, try partial match
                            if not found:
                                matches = []
                                for member in team_result.team_members:
                                    given_name = member.given_name or ""
                                    family_name = member.family_name or ""
                                    member_name = f"{given_name} {family_name}".strip()
                                    
                                    # Check for partial match
                                    if (provider.lower() in given_name.lower() or 
                                        provider.lower() in family_name.lower() or
                                        provider.lower() in member_name.lower()):
                                        matches.append({
                                            "id": member.id,
                                            "name": member_name
                                        })
                                
                                # If multiple matches, return them
                                if len(matches) > 1:
                                    return {
                                        "status": "ambiguous",
                                        "provider_matches": matches,
                                        "message": f"Multiple providers found matching '{provider}'. Please specify the exact provider name or ID."
                                    }
                                elif len(matches) == 1:
                                    team_member_id = matches[0]["id"]
                                    provider_name = matches[0]["name"]
                                else:
                                    # No matches
                                    raise ToolError(
                                        f"No provider found with name '{provider}'",
                                        ErrorCode.TOOL_EXECUTION_ERROR
                                    )
                    except Exception as e:
                        if isinstance(e, ToolError):
                            raise e
                        self.logger.error(f"Error searching for team member: {e}")
                        raise ToolError(
                            f"Failed to find provider with name '{provider}': {str(e)}",
                            ErrorCode.TOOL_EXECUTION_ERROR
                        )
            
            # Resolve location if not provided
            if not location_id:
                try:
                    # Get the default location
                    locations = self.client.locations.list().locations
                    if locations and len(locations) > 0:
                        # Use first location as default
                        location_id = locations[0].id
                except Exception as e:
                    self.logger.warning(f"Error retrieving locations: {e}")
                    raise ToolError(
                        "Location ID is required, and no default location could be determined",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
            
            # Process date input
            start_date = None
            end_date = None
            
            if date:
                try:
                    # Parse the date
                    parsed_date = date_parser.parse(date)
                    # Set to beginning of day
                    start_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    # Set end to end of day
                    end_date = parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                except Exception as e:
                    raise ToolError(
                        f"Invalid date format: {str(e)}. Use YYYY-MM-DD format.",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
            else:
                # Default to today
                now = datetime.datetime.now()
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            # Format dates for Square API in RFC 3339 format
            start_at = start_date.isoformat() + 'Z'
            end_at = end_date.isoformat() + 'Z'
            
            # Build the availability search query
            search_query = {
                "query": {
                    "filter": {
                        "start_at_range": {
                            "start_at": start_at,
                            "end_at": end_at
                        },
                        "location_id": location_id,
                        "segment_filters": [
                            {
                                "service_variation_id": service_variation_id
                            }
                        ]
                    }
                }
            }
            
            # Add team member filter if provider specified
            if team_member_id:
                search_query["query"]["filter"]["segment_filters"][0]["team_member_id_filter"] = {
                    "any": [team_member_id]
                }
            
            # Execute the search
            self.logger.info(f"Searching availability with query: {search_query}")
            result = self.client.bookings.search_availability(query=search_query)
            
            # Format the results
            availability_slots = []
            
            if result.availabilities:
                for slot in result.availabilities:
                    # Parse the slot start time
                    slot_time = None
                    if slot.start_at:
                        try:
                            slot_time = date_parser.parse(slot.start_at)
                            # Convert to user's timezone
                            slot_time = slot_time.astimezone(tz)
                        except Exception as e:
                            self.logger.warning(f"Error parsing slot time {slot.start_at}: {e}")
                            continue  # Skip slots with parsing errors
                    else:
                        continue  # Skip slots without start time
                    
                    # Format the time for display
                    formatted_time = slot_time.strftime("%Y-%m-%d %I:%M %p %Z")
                    formatted_time_short = slot_time.strftime("%I:%M %p")
                    
                    # Get the slot's location
                    location_name = None
                    if slot.location_id:
                        try:
                            location = self.client.locations.get(location_id=slot.location_id).location
                            if location.get("name"):
                                location_name = location.get("name")
                        except Exception as e:
                            self.logger.warning(f"Error retrieving location details: {e}")
                    
                    # Process appointment segments
                    segments = []
                    slot_provider = None
                    slot_duration = 0
                    
                    if slot.appointment_segments and len(slot.appointment_segments) > 0:
                        for segment in slot.appointment_segments:
                            segment_info = {
                                "service_variation_id": segment.service_variation_id,
                                "team_member_id": segment.team_member_id,
                            }
                            
                            # Add duration if available
                            if segment.duration_minutes:
                                segment_info["duration_minutes"] = segment.duration_minutes
                                slot_duration += segment.duration_minutes
                            
                            # Get team member name if available
                            if segment.team_member_id and not slot_provider:
                                try:
                                    member_result = self.client.team.get_team_member(
                                        team_member_id=segment.team_member_id
                                    )
                                    given_name = member_result.team_member.given_name or ""
                                    family_name = member_result.team_member.family_name or ""
                                    slot_provider = f"{given_name} {family_name}".strip()
                                except Exception as e:
                                    self.logger.warning(f"Error retrieving team member details: {e}")
                            
                            segments.append(segment_info)
                    
                    # Use provided duration if no duration in segments
                    if slot_duration == 0 and duration_minutes:
                        slot_duration = duration_minutes
                    
                    # Format the availability slot
                    availability_info = {
                        "start_at": slot.start_at,
                        "formatted_time": formatted_time,
                        "formatted_time_short": formatted_time_short,
                        "location_id": slot.location_id,
                        "location_name": location_name,
                        "segments": segments,
                        "provider": slot_provider or provider_name,
                        "duration_minutes": slot_duration
                    }
                    
                    # Add to availability slots
                    availability_slots.append(availability_info)
            
            # Sort slots by start time
            availability_slots.sort(key=lambda x: x.get("start_at", ""))
            
            # Group slots by date for easier presentation
            slots_by_date = {}
            for slot in availability_slots:
                slot_time = date_parser.parse(slot["start_at"]).astimezone(tz)
                date_key = slot_time.strftime("%Y-%m-%d")
                date_display = slot_time.strftime("%A, %B %d, %Y")
                
                if date_key not in slots_by_date:
                    slots_by_date[date_key] = {
                        "date": date_key,
                        "display_date": date_display,
                        "slots": []
                    }
                
                slots_by_date[date_key]["slots"].append(slot)
            
            # Convert to list and sort by date
            dates_list = list(slots_by_date.values())
            dates_list.sort(key=lambda x: x["date"])
            
            return {
                "status": "success",
                "service": service_name,
                "service_id": service_variation_id,
                "provider": provider_name,
                "provider_id": team_member_id,
                "location_id": location_id,
                "location_name": location_name,
                "date": start_date.strftime("%Y-%m-%d"),
                "availability_dates": dates_list,
                "availability_count": len(availability_slots)
            }
            
        except ToolError as e:
            # Re-raise ToolErrors
            raise e
        except Exception as e:
            self.logger.error(f"Error finding available slots: {e}")
            raise ToolError(
                f"Failed to find available slots: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _resolve_team_member(self, provider: str) -> Tuple[str, str]:
        """
        Helper method to resolve a team member name or ID to an ID.
        
        Args:
            provider: Provider name or ID
            
        Returns:
            Tuple of (team_member_id, provider_name)
            
        Raises:
            ToolError: If provider resolution fails
        """
        # Check if provider looks like an ID already
        if provider.startswith("TMxxx") or "_" in provider:
            team_member_id = provider
            
            # Try to get provider name
            try:
                member_result = self.client.team.get_team_member(team_member_id=team_member_id)
                given_name = member_result.team_member.given_name or ""
                family_name = member_result.team_member.family_name or ""
                provider_name = f"{given_name} {family_name}".strip() or provider
                return team_member_id, provider_name
            except Exception as e:
                self.logger.warning(f"Error retrieving team member details: {e}")
                return team_member_id, provider  # Return ID as name if lookup fails
        
        # Search for team member by name
        try:
            team_result = self.client.team.search_team_members(
                query={"filter": {"status": "ACTIVE"}}
            )
            
            # Try exact name match first
            if team_result.team_members:
                for member in team_result.team_members:
                    given_name = member.given_name or ""
                    family_name = member.family_name or ""
                    member_name = f"{given_name} {family_name}".strip()
                    
                    # Check for exact match (case insensitive)
                    if member_name.lower() == provider.lower():
                        return member.id, member_name
                
                # Try partial match if no exact match
                matches = []
                for member in team_result.team_members:
                    given_name = member.given_name or ""
                    family_name = member.family_name or ""
                    member_name = f"{given_name} {family_name}".strip()
                    
                    # Check for partial match
                    if (provider.lower() in given_name.lower() or 
                        provider.lower() in family_name.lower() or
                        provider.lower() in member_name.lower()):
                        matches.append({
                            "id": member.id,
                            "name": member_name
                        })
                
                # Handle multiple or no matches
                if len(matches) > 1:
                    raise ToolError(
                        "multiple_providers",
                        ErrorCode.TOOL_AMBIGUOUS_INPUT,
                        data={"provider_matches": matches}
                    )
                elif len(matches) == 1:
                    return matches[0]["id"], matches[0]["name"]
            
            # No matches found
            raise ToolError(
                f"No provider found with name '{provider}'",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        except ToolError:
            # Re-raise ToolErrors with specific error info
            raise
        except Exception as e:
            self.logger.error(f"Error searching for team member: {e}")
            raise ToolError(
                f"Failed to find provider with name '{provider}': {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _resolve_service_variation(self, service: str) -> Tuple[str, str, int, Optional[int]]:
        """
        Helper method to resolve a service name or ID to a variation ID.
        
        Args:
            service: Service name or ID
            
        Returns:
            Tuple of (service_variation_id, service_name, service_version, duration_minutes)
            The duration_minutes may be None if not available
            
        Raises:
            ToolError: If service resolution fails
        """
        # First check if this is already an ID
        if service.startswith("#") or "_" in service:
            service_id = service
            
            # Get service details
            try:
                service_result = self.client.catalog.object.retrieve(object_id=service_id)
                
                # Handle variation vs item
                if service_result.object.type == "ITEM_VARIATION":
                    # Already a variation
                    variation = service_result.object
                    variation_id = variation.id
                    variation_name = variation.item_variation_data.name
                    version = variation.version
                    
                    # Get parent item name if available
                    parent_id = variation.item_variation_data.item_id
                    try:
                        parent_result = self.client.catalog.object.retrieve(object_id=parent_id)
                        parent_name = parent_result.object.item_data.name
                        service_name = f"{parent_name} - {variation_name}"
                    except Exception:
                        service_name = variation_name
                    
                    # Get duration if available (Square stores as milliseconds)
                    duration_minutes = None
                    if variation.item_variation_data.service_duration:
                        service_duration_ms = variation.item_variation_data.service_duration
                        duration_minutes = int(service_duration_ms / (60 * 1000))
                    
                    return variation_id, service_name, version, duration_minutes
                    
                elif service_result.object.type == "ITEM":
                    # It's an item, but we need a variation - use the first one
                    item = service_result.object
                    if not item.item_data.variations or len(item.item_data.variations) == 0:
                        raise ToolError(
                            f"Service {service} has no variations",
                            ErrorCode.TOOL_EXECUTION_ERROR
                        )
                    
                    variation = item.item_data.variations[0]
                    variation_id = variation.id
                    variation_name = variation.item_variation_data.name
                    version = variation.version
                    item_name = item.item_data.name
                    service_name = f"{item_name} - {variation_name}"
                    
                    # Get duration if available (Square stores as milliseconds)
                    duration_minutes = None
                    if variation.item_variation_data.service_duration:
                        service_duration_ms = variation.item_variation_data.service_duration
                        duration_minutes = int(service_duration_ms / (60 * 1000))
                    
                    return variation_id, service_name, version, duration_minutes
                
                else:
                    raise ToolError(
                        f"Object {service} is not a service item or variation",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
                    
            except ToolError:
                # Re-raise specific errors
                raise
            except Exception as e:
                self.logger.error(f"Error retrieving service details: {e}")
                raise ToolError(
                    f"Failed to retrieve service details for {service}: {str(e)}",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
        
        # Not an ID - need to resolve by name
        try:
            # Use existing resolver to find ID
            resolved = self._resolve_service_id(service)
            
            # Check if we got multiple matches
            if isinstance(resolved, dict) and "matches" in resolved:
                raise ToolError(
                    "multiple_services",
                    ErrorCode.TOOL_AMBIGUOUS_INPUT,
                    data={"service_matches": resolved["matches"]}
                )
            
            # Now that we have the ID, get full details recursively
            return self._resolve_service_variation(resolved)
            
        except ToolError:
            # Re-raise specific errors
            raise
        except Exception as e:
            self.logger.error(f"Error resolving service name: {e}")
            raise ToolError(
                f"Failed to resolve service name '{service}': {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _resolve_location(self, location_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Helper method to resolve a location ID or get default location.
        
        Args:
            location_id: Optional location ID
            
        Returns:
            Tuple of (location_id, location_name) - name may be None if unavailable
            
        Raises:
            ToolError: If location resolution fails and no default available
        """
        # If location ID provided, use it
        if location_id:
            try:
                # Try to get location name
                location = self.client.locations.get(location_id=location_id).location
                location_name = location.get("name")
                return location_id, location_name
            except Exception as e:
                self.logger.warning(f"Error retrieving location details: {e}")
                return location_id, None
        
        # No location provided, get default
        try:
            # Get the first active location
            locations = self.client.locations.list().locations
            for location in locations:
                if location.status == "ACTIVE":
                    return location.id, location.name
            
            # If no active locations, use first location
            if locations and len(locations) > 0:
                return locations[0].id, locations[0].name
                
            # No locations available
            raise ToolError(
                "No active locations found in your Square account",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
        except ToolError:
            # Re-raise specific errors
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving locations: {e}")
            raise ToolError(
                "Could not determine a default location. Please provide a location_id.",
                ErrorCode.TOOL_INVALID_INPUT
            )
    
    def _parse_appointment_datetime(self, date: str, time: str, timezone: Optional[str] = None) -> str:
        """
        Parse date and time strings into a timezone-aware datetime and return RFC 3339 format.
        
        Args:
            date: Date string (e.g., YYYY-MM-DD)
            time: Time string (e.g., HH:MM AM/PM)
            timezone: Optional timezone identifier
            
        Returns:
            RFC 3339 formatted datetime string
            
        Raises:
            ToolError: If date/time cannot be parsed
        """
        # Validate and process timezone
        tz = None
        if timezone:
            try:
                tz = validate_timezone(timezone)
            except Exception as e:
                raise ToolError(
                    f"Invalid timezone: {str(e)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
        else:
            # Use default timezone
            tz = get_default_timezone()
        
        try:
            # Combine date and time
            datetime_str = f"{date} {time}"
            # Parse with given timezone
            appointment_datetime = date_parser.parse(datetime_str)
            
            # If the parsed time doesn't have tzinfo, apply the timezone
            if appointment_datetime.tzinfo is None:
                appointment_datetime = tz.localize(appointment_datetime)
            
            # Convert to UTC for Square API
            utc_datetime = appointment_datetime.astimezone(datetime.timezone.utc)
            
            # Format in RFC 3339 format
            return utc_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            raise ToolError(
                f"Invalid date/time format: {str(e)}. Use YYYY-MM-DD for date and HH:MM AM/PM for time.",
                ErrorCode.TOOL_INVALID_INPUT
            )
    
    def _schedule_appointment(
        self,
        service: str,  # Can be name or ID
        provider: str,  # Can be name or ID
        date: str,
        time: str,
        customer_id: str,
        location_id: Optional[str] = None,
        customer_note: Optional[str] = None,
        seller_note: Optional[str] = None,
        duration_minutes: Optional[int] = None,
        timezone: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Schedule a new appointment/booking.
        
        Args:
            service: Service name or ID
            provider: Provider/team member name or ID
            date: Appointment date (YYYY-MM-DD)
            time: Appointment time (HH:MM AM/PM)
            customer_id: Customer ID
            location_id: Optional location ID
            customer_note: Optional customer-visible notes
            seller_note: Optional seller-only notes
            duration_minutes: Optional duration override for the service
            timezone: Optional timezone for date/time parsing
            
        Returns:
            Dict with scheduled appointment information
            
        Raises:
            ToolError: If appointment scheduling fails or required fields missing
        """
        self.logger.info(f"Scheduling appointment for service: {service}, date: {date}, time: {time}")
        
        # Validate required fields
        required_fields = {
            "service": service,
            "provider": provider,
            "date": date,
            "time": time,
            "customer_id": customer_id
        }
        
        for field, value in required_fields.items():
            if not value:
                raise ToolError(
                    f"{field} is required for scheduling an appointment",
                    ErrorCode.TOOL_INVALID_INPUT
                )
        
        try:
            # Step 1: Resolve service to get ID, name, version and duration
            try:
                service_variation_id, service_name, service_version, service_duration = self._resolve_service_variation(service)
                # Use provided duration if available, otherwise use service duration
                if duration_minutes is None:
                    duration_minutes = service_duration
            except ToolError as e:
                if e.code == "multiple_services":
                    # Return the service matches for user selection
                    return {
                        "status": "ambiguous",
                        "service_matches": e.data["service_matches"],
                        "message": f"Multiple services found matching '{service}'. Please specify the exact service name or ID."
                    }
                else:
                    # Re-raise other errors
                    raise
            
            # Step 2: Resolve provider to get ID and name
            try:
                team_member_id, provider_name = self._resolve_team_member(provider)
            except ToolError as e:
                if e.code == "multiple_providers":
                    # Return the provider matches for user selection
                    return {
                        "status": "ambiguous",
                        "provider_matches": e.data["provider_matches"],
                        "message": f"Multiple providers found matching '{provider}'. Please specify the exact provider name or ID."
                    }
                else:
                    # Re-raise other errors
                    raise
            
            # Step 3: Resolve location to get ID and name
            location_id, location_name = self._resolve_location(location_id)
            
            # Step 4: Parse date and time to create appointment start time
            start_at = self._parse_appointment_datetime(date, time, timezone)
            
            # Step 5: Build booking data
            booking_data = {
                "start_at": start_at,
                "location_id": location_id,
                "customer_id": customer_id,
                "appointment_segments": [
                    {
                        "service_variation_id": service_variation_id,
                        "service_variation_version": service_version,
                        "team_member_id": team_member_id
                    }
                ],
                "source": "FIRST_PARTY_BOOKING"
            }
            
            # Add optional duration if available
            if duration_minutes:
                booking_data["appointment_segments"][0]["duration_minutes"] = duration_minutes
            
            # Add optional notes if provided
            if customer_note:
                booking_data["customer_note"] = customer_note
            
            if seller_note:
                booking_data["seller_note"] = seller_note
            
            # Step 6: Create the booking with idempotency key
            import uuid
            idempotency_key = str(uuid.uuid4())
            
            self.logger.info(f"Creating booking with idempotency key: {idempotency_key}")
            result = self.client.bookings.create(
                booking=booking_data,
                idempotency_key=idempotency_key
            )
            
            # Step 7: Format the response
            booking = result.booking
            
            # Get timezone for formatting
            tz = timezone_utils.get_default_timezone() if not timezone else timezone_utils.validate_timezone(timezone)
            
            # Parse booking time for display
            formatted_time = None
            if booking.start_at:
                try:
                    booking_time = date_parser.parse(booking.start_at)
                    booking_time = booking_time.astimezone(tz)
                    formatted_time = booking_time.strftime("%Y-%m-%d %I:%M %p %Z")
                except Exception as e:
                    self.logger.warning(f"Error formatting booking time: {e}")
            
            # Create response
            appointment_info = {
                "id": booking.id,
                "status": booking.status,
                "created_at": booking.created_at,
                "start_at": booking.start_at,
                "formatted_time": formatted_time,
                "service": service_name,
                "service_id": service_variation_id,
                "provider": provider_name,
                "provider_id": team_member_id,
                "location_id": location_id,
                "location_name": location_name,
                "customer_id": booking.customer_id
            }
            
            # Add notes if available
            if booking.customer_note:
                appointment_info["customer_note"] = booking.customer_note
            
            if booking.seller_note:
                appointment_info["seller_note"] = booking.seller_note
            
            # Add duration if available
            if booking.appointment_segments and len(booking.appointment_segments) > 0:
                segment = booking.appointment_segments[0]
                if segment.duration_minutes:
                    appointment_info["duration_minutes"] = segment.duration_minutes
            elif duration_minutes:
                appointment_info["duration_minutes"] = duration_minutes
            
            return {
                "status": "success",
                "appointment": appointment_info,
                "message": "Appointment scheduled successfully"
            }
            
        except ToolError:
            # Re-raise specific errors
            raise
        except Exception as e:
            self.logger.error(f"Error scheduling appointment: {e}")
            raise ToolError(
                f"Failed to schedule appointment: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _reschedule_appointment(
        self,
        appointment_id: str,
        date: str,
        time: str,
        provider: Optional[str] = None,  # Can be name or ID
        location_id: Optional[str] = None,
        customer_note: Optional[str] = None,
        seller_note: Optional[str] = None,
        timezone: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reschedule an existing appointment/booking.
        
        Args:
            appointment_id: The ID of the appointment to reschedule
            date: New appointment date (YYYY-MM-DD)
            time: New appointment time (HH:MM AM/PM)
            provider: Optional new provider/team member name or ID
            location_id: Optional new location ID
            customer_note: Optional customer-visible notes to update
            seller_note: Optional seller-only notes to update
            timezone: Optional timezone for date/time parsing
            
        Returns:
            Dict with rescheduled appointment information
            
        Raises:
            ToolError: If appointment rescheduling fails or required fields missing
        """
        self.logger.info(f"Rescheduling appointment: {appointment_id} to {date} {time}")
        
        # Validate required fields
        if not appointment_id:
            raise ToolError(
                "appointment_id is required for rescheduling an appointment",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        if not date or not time:
            raise ToolError(
                "date and time are required for rescheduling an appointment",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            # Step 1: Get the existing booking to get its current details
            try:
                current_booking = self.client.bookings.get(appointment_id).booking
            except Exception as e:
                self.logger.error(f"Error retrieving appointment {appointment_id}: {e}")
                raise ToolError(
                    f"Could not find appointment with ID {appointment_id}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Step 2: Prepare the update booking data, starting with a copy of current booking
            booking_data = {
                # These fields cannot be changed
                "customer_id": current_booking.customer_id,
                "appointment_segments": current_booking.appointment_segments,
                "source": current_booking.source or "FIRST_PARTY_BOOKING"
            }
            
            # Step 3: Set the new start time
            start_at = self._parse_appointment_datetime(date, time, timezone)
            booking_data["start_at"] = start_at
            
            # Step 4: Update location if specified
            if location_id:
                # Validate location exists
                resolved_location_id, location_name = self._resolve_location(location_id)
                booking_data["location_id"] = resolved_location_id
            else:
                # Keep the current location
                booking_data["location_id"] = current_booking.location_id
            
            # Step 5: Update team member if specified
            if provider and current_booking.appointment_segments:
                try:
                    team_member_id, provider_name = self._resolve_team_member(provider)
                    
                    # Update all segments to use the new team member
                    for i, segment in enumerate(booking_data["appointment_segments"]):
                        booking_data["appointment_segments"][i]["team_member_id"] = team_member_id
                except ToolError as e:
                    if e.code == "multiple_providers":
                        # Return the provider matches for user selection
                        return {
                            "status": "ambiguous",
                            "provider_matches": e.data["provider_matches"],
                            "message": f"Multiple providers found matching '{provider}'. Please specify the exact provider name or ID."
                        }
                    else:
                        # Re-raise other errors
                        raise
            
            # Step 6: Update notes if provided
            if customer_note is not None:
                booking_data["customer_note"] = customer_note
            elif current_booking.customer_note:
                booking_data["customer_note"] = current_booking.customer_note
                
            if seller_note is not None:
                booking_data["seller_note"] = seller_note
            elif current_booking.seller_note:
                booking_data["seller_note"] = current_booking.seller_note
            
            # Step 7: Get version for optimistic concurrency control
            booking_version = current_booking.version
            
            # Step 8: Update the booking
            import uuid
            idempotency_key = str(uuid.uuid4())
            
            self.logger.info(f"Updating booking {appointment_id} with idempotency key: {idempotency_key}")
            result = self.client.bookings.update(
                booking_id=appointment_id,
                booking=booking_data,
                idempotency_key=idempotency_key
            )
            
            # Step 9: Format the response
            booking = result.booking
            
            # Get timezone for formatting
            tz = get_default_timezone() if not timezone else validate_timezone(timezone)
            
            # Parse booking time for display
            formatted_time = None
            if booking.start_at:
                try:
                    booking_time = date_parser.parse(booking.start_at)
                    booking_time = booking_time.astimezone(tz)
                    formatted_time = booking_time.strftime("%Y-%m-%d %I:%M %p %Z")
                except Exception as e:
                    self.logger.warning(f"Error formatting booking time: {e}")
            
            # Step 10: Get some additional details for a complete response
            service_name = "Unknown Service"
            provider_name = "Unknown Provider"
            location_name = None
            
            # Get service details
            if booking.appointment_segments and len(booking.appointment_segments) > 0:
                segment = booking.appointment_segments[0]
                
                # Get service name
                try:
                    service_variation_id = segment.service_variation_id
                    service_result = self.client.catalog.object.retrieve(object_id=service_variation_id)
                    
                    if service_result.object.type == "ITEM_VARIATION":
                        variation_name = service_result.object.item_variation_data.name
                        parent_id = service_result.object.item_variation_data.item_id
                        
                        try:
                            parent_result = self.client.catalog.object.retrieve(object_id=parent_id)
                            parent_name = parent_result.object.item_data.name
                            service_name = f"{parent_name} - {variation_name}"
                        except Exception:
                            service_name = variation_name
                except Exception as e:
                    self.logger.warning(f"Error retrieving service details: {e}")
                
                # Get provider name
                try:
                    team_member_id = segment.team_member_id
                    member_result = self.client.team.get_team_member(team_member_id=team_member_id)
                    given_name = member_result.team_member.given_name or ""
                    family_name = member_result.team_member.family_name or ""
                    provider_name = f"{given_name} {family_name}".strip() or "Unknown Provider"
                except Exception as e:
                    self.logger.warning(f"Error retrieving team member details: {e}")
            
            # Get location name
            try:
                location = self.client.locations.get(location_id=booking.location_id).location
                location_name = location.get("name")
            except Exception as e:
                self.logger.warning(f"Error retrieving location details: {e}")
            
            # Create response
            appointment_info = {
                "id": booking.id,
                "status": booking.status,
                "created_at": booking.created_at,
                "updated_at": booking.updated_at,
                "start_at": booking.start_at,
                "formatted_time": formatted_time,
                "service": service_name,
                "provider": provider_name,
                "location_id": booking.location_id,
                "customer_id": booking.customer_id
            }
            
            # Add location name if available
            if location_name:
                appointment_info["location_name"] = location_name
            
            # Add notes if available
            if booking.customer_note:
                appointment_info["customer_note"] = booking.customer_note
            
            if booking.seller_note:
                appointment_info["seller_note"] = booking.seller_note
            
            # Add duration if available
            if booking.appointment_segments and len(booking.appointment_segments) > 0:
                segment = booking.appointment_segments[0]
                if segment.duration_minutes:
                    appointment_info["duration_minutes"] = segment.duration_minutes
            
            return {
                "status": "success",
                "appointment": appointment_info,
                "message": "Appointment rescheduled successfully"
            }
            
        except ToolError:
            # Re-raise specific errors
            raise
        except Exception as e:
            self.logger.error(f"Error rescheduling appointment: {e}")
            raise ToolError(
                f"Failed to reschedule appointment: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _cancel_appointment(
        self,
        appointment_id: str,
        reason: Optional[str] = None,
        notify_customer: bool = True
    ) -> Dict[str, Any]:
        """
        Cancel an existing appointment/booking.
        
        Args:
            appointment_id: The ID of the appointment to cancel
            reason: Optional reason for cancellation (stored in seller notes)
            notify_customer: Whether to send cancellation notification to customer
            
        Returns:
            Dict with cancellation confirmation
            
        Raises:
            ToolError: If appointment cancellation fails or id is missing
        """
        self.logger.info(f"Cancelling appointment: {appointment_id}")
        
        # Validate appointment_id
        if not appointment_id:
            raise ToolError(
                "appointment_id is required for cancelling an appointment",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        try:
            # Step 1: Get the current booking to check status and get version
            try:
                current_booking = self.client.bookings.get(booking_id=appointment_id).booking
            except Exception as e:
                self.logger.error(f"Error retrieving appointment {appointment_id}: {e}")
                raise ToolError(
                    f"Could not find appointment with ID {appointment_id}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Step 2: Check if the booking is already cancelled
            if current_booking.status in ["CANCELLED_BY_SELLER", "CANCELLED_BY_CUSTOMER", "DECLINED", "NO_SHOW"]:
                return {
                    "status": "already_cancelled",
                    "appointment_id": appointment_id,
                    "message": f"Appointment is already in '{current_booking.status}' status and cannot be cancelled"
                }
            
            # Step 3: Get booking version for optimistic concurrency control
            booking_version = current_booking.version
            
            # Step 4: Add reason to seller note if provided
            new_seller_note = None
            if reason:
                if current_booking.seller_note:
                    new_seller_note = f"{current_booking.seller_note}\n\nCancellation reason: {reason}"
                else:
                    new_seller_note = f"Cancellation reason: {reason}"
            
            # Step 5: Generate idempotency key for safety
            import uuid
            idempotency_key = str(uuid.uuid4())
            
            # Step 6: Cancel the booking
            self.logger.info(f"Cancelling booking {appointment_id} with idempotency key: {idempotency_key}")
            result = self.client.bookings.cancel(
                booking_id=appointment_id,
                idempotency_key=idempotency_key,
                booking_version=booking_version
            )
            
            # Step 7: Format the response
            booking = result.booking
            
            # Step 8: Get additional details for the response
            service_name = "Unknown Service"
            provider_name = "Unknown Provider"
            customer_name = "Unknown Customer"
            location_name = None
            formatted_time = None
            
            # Get service and provider details
            if booking.appointment_segments and len(booking.appointment_segments) > 0:
                segment = booking.appointment_segments[0]
                
                # Get service name
                try:
                    service_variation_id = segment.service_variation_id
                    service_result = self.client.catalog.object.retrieve(object_id=service_variation_id)
                    
                    if service_result.object.type == "ITEM_VARIATION":
                        variation_name = service_result.object.item_variation_data.name
                        parent_id = service_result.object.item_variation_data.item_id
                        
                        try:
                            parent_result = self.client.catalog.object.retrieve(object_id=parent_id)
                            parent_name = parent_result.object.item_data.name
                            service_name = f"{parent_name} - {variation_name}"
                        except Exception:
                            service_name = variation_name
                except Exception as e:
                    self.logger.warning(f"Error retrieving service details: {e}")
                
                # Get provider name
                try:
                    team_member_id = segment.team_member_id
                    member_result = self.client.team.get_team_member(team_member_id=team_member_id)
                    given_name = member_result.team_member.given_name or ""
                    family_name = member_result.team_member.family_name or ""
                    provider_name = f"{given_name} {family_name}".strip() or "Unknown Provider"
                except Exception as e:
                    self.logger.warning(f"Error retrieving team member details: {e}")
            
            # Get customer name
            if booking.customer_id:
                try:
                    customer = self.client.customers.get(customer_id=booking.customer_id).customer
                    first_name = customer.get("given_name", "")
                    last_name = customer.get("family_name", "")
                    customer_name = f"{first_name} {last_name}".strip() or "Unknown Customer"
                except Exception as e:
                    self.logger.warning(f"Error retrieving customer details: {e}")
            
            # Get location name
            if booking.location_id:
                try:
                    location = self.client.locations.get(location_id=booking.location_id).location
                    location_name = location.get("name")
                except Exception as e:
                    self.logger.warning(f"Error retrieving location details: {e}")
            
            # Format booking time
            if booking.start_at:
                try:
                    booking_time = date_parser.parse(booking.start_at)
                    # Use default timezone for display
                    tz = get_default_timezone()
                    booking_time = booking_time.astimezone(tz)
                    formatted_time = booking_time.strftime("%Y-%m-%d %I:%M %p %Z")
                except Exception as e:
                    self.logger.warning(f"Error formatting booking time: {e}")
            
            # Create response
            cancellation_info = {
                "id": booking.id,
                "status": booking.status,  # Should be CANCELLED_BY_SELLER
                "created_at": booking.created_at,
                "updated_at": booking.updated_at,
                "start_at": booking.start_at,
                "formatted_time": formatted_time,
                "service": service_name,
                "provider": provider_name,
                "customer": customer_name,
                "customer_id": booking.customer_id,
                "location_id": booking.location_id
            }
            
            # Add location name if available
            if location_name:
                cancellation_info["location_name"] = location_name
            
            # Add cancellation reason if provided
            if reason:
                cancellation_info["cancellation_reason"] = reason
            
            # Step 9: Handle customer notification (placeholder for now)
            # In a real implementation, you might want to send a notification
            # to the customer about the cancellation
            if notify_customer:
                cancellation_info["notification_sent"] = True
            
            return {
                "status": "success",
                "appointment": cancellation_info,
                "message": "Appointment cancelled successfully"
            }
            
        except ToolError:
            # Re-raise specific errors
            raise
        except Exception as e:
            self.logger.error(f"Error cancelling appointment: {e}")
            raise ToolError(
                f"Failed to cancel appointment: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    # ===== Helper methods will be implemented in future steps =====