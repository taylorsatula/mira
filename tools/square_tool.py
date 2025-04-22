"""
Square API integration tool.

This tool enables the bot to interact with Square's API to manage customers, bookings, and catalog.
Features include customer management (create, retrieve, update), booking management
(create, list, search availability, cancel), and catalog management (list, create, retrieve, update, delete).

Requires Square API credentials in the .env file.
"""

import os
import logging
import pathlib
import time
import sys
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import json

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context
from config import config


class SquareTool(Tool):
    """
    Tool for interacting with Square's API to manage customers, bookings, and catalog.

    Features:
    1. Customer Management:
       - Search and retrieve customers
       - Create and update customer records
       - Manage customer groups

    2. Booking Management:
       - Create and manage bookings
       - Search for available time slots
       - Retrieve booking details
       - Cancel bookings
       
    3. Catalog Management:
       - List, retrieve, and search catalog items
       - Create and update catalog items, variations, and categories
       - Manage catalog images and modifiers
       - Batch operations for catalog objects
    """

    name = "square_tool"
    description = """Provides comprehensive access to Square's business management APIs for customer, booking, and catalog operations.

This tool enables interaction with Square's platform through the following operations:

1. Customer Management Operations:
   - list_customers: Retrieve a paginated list of all customer profiles
     - Optional parameters: 'limit', 'cursor', 'sort_field', 'sort_order', 'count'
   - retrieve_customer: Get detailed information about a specific customer
     - Required parameter: 'customer_id'
   - create_customer: Add a new customer to your Square account
     - Required parameter: 'body' containing customer information (name, email, phone, etc.)
   - update_customer: Modify an existing customer's information
     - Required parameters: 'customer_id', 'body' with fields to update
   - delete_customer: Remove a customer from your Square account
     - Required parameter: 'customer_id'
   - bulk_create_customers: Add multiple customers in a single operation
   - bulk_retrieve_customers: Get information about multiple customers
   - bulk_update_customers: Modify multiple customer profiles simultaneously
   - bulk_delete_customers: Remove multiple customers at once

2. Booking Management Operations:
   - create_booking: Schedule a new appointment
     - Required parameter: 'body' with booking details 
   - retrieve_booking: Get information about a specific booking
     - Required parameter: 'booking_id'
   - update_booking: Modify an existing booking's details
     - Required parameters: 'booking_id', 'body' with fields to update
   - cancel_booking: Cancel a scheduled appointment
     - Required parameters: 'booking_id', 'body' with cancellation details
   - list_bookings: Retrieve a list of bookings with filtering options
     - Optional parameters: 'customer_id', 'location_id', 'start_at_min', 'start_at_max'
   - search_availability: Find available appointment times
     - Required parameter: 'body' with search criteria
   - retrieve_business_booking_profile: Get business booking settings
   - list_team_member_booking_profiles: Get booking information for staff members

3. Catalog Management Operations:
   - list_catalog: Retrieve a list of catalog objects by type
     - Optional parameters: 'cursor', 'types', 'catalog_version'
   - retrieve_catalog_object: Get detailed information about a specific catalog object
     - Required parameter: 'object_id'
   - search_catalog_objects: Search for catalog objects using filters
     - Required parameter: 'body' with search criteria
   - search_catalog_items: Search specifically for catalog items with advanced filters
     - Required parameter: 'body' with search criteria
   - upsert_catalog_object: Create or update a catalog object
     - Required parameter: 'body' with the object details
   - delete_catalog_object: Remove a catalog object
     - Required parameter: 'object_id'
   - batch_delete_catalog_objects: Delete multiple catalog objects in one operation
     - Required parameter: 'body' with object IDs
   - batch_retrieve_catalog_objects: Get multiple catalog objects in one request
     - Required parameter: 'body' with object IDs
   - batch_upsert_catalog_objects: Create or update multiple catalog objects
     - Required parameter: 'body' with objects to upsert
   - catalog_info: Get information about the catalog API

Use this tool whenever you need to manage appointment scheduling or work with catalog items through Square's platform.

⚠️ CRITICAL: For creating a booking, you MUST enable and use BOTH square_tool AND customer_tool together. ⚠️

This tool does NOT have customer search capability:
- NEVER use square_tool.list_customers to search for a specific customer - it will list ALL customers and is inefficient
- You MUST enable and use customer_tool for all customer searching (by name, email, phone, address)
- When a user mentions a customer name for booking, ALWAYS enable the customer_tool and use it to search
- Square's booking system requires customer_ids that only customer_tool can find efficiently

For creating a booking:
1. First enable BOTH square_tool AND customer_tool - both are required
2. Use customer_tool.search_customers to find the customer by name
3. If customer exists, use their ID from customer_tool's search results
4. If customer doesn't exist, create them with square_tool.create_customer
5. Only then proceed with creating the booking with the appropriate customer ID"""
    usage_examples = [
        # Example: Proper workflow for creating a booking (multi-step)
        {
            "input": "I want to create a booking for a customer (either new or existing)",
            "output": "I'll help you create a booking. To do this properly, I need to use both customer_tool and square_tool together.",
            "description": "First, recognize that BOTH tools are needed for booking operations"
        },
        {
            "input": {"operation": "search_customers", "kwargs": "{\"query\": \"Customer Name\", \"category\": \"name\"}"},
            "output": {"search_type": "Name Search", "customers": []},
            "target_tool": "customer_tool",
            "description": "ALWAYS use customer_tool first to search for the customer"
        },
        {
            "input": {"operation": "create_customer", "body": {"given_name": "New", "family_name": "Customer", "email_address": "customer@example.com", "phone_number": "+15551234567"}},
            "output": {"customer": {"id": "CUSTOMER_ID", "given_name": "New", "family_name": "Customer"}},
            "description": "If customer doesn't exist, create them first"
        },
        {
            "input": {"operation": "create_booking", "body": {"booking": {"customer_id": "CUSTOMER_ID", "start_at": "2023-05-15T14:00:00Z", "appointment_segments": [{"service_variation_id": "SERVICE_ID", "service_variation_version": 1}]}}},
            "output": {"booking": {"id": "BOOKING_ID", "status": "ACCEPTED"}},
            "description": "Then create the booking using the customer ID"
        },
        
        # # Example 1: List customers
        # {
        #     "input": {"operation": "list_customers", "limit": 10},
        #     "output": {
        #         "customers": [
        #             {
        #                 "id": "JSJD9SJW4D5YFBQ47X1ADP1F7R",
        #                 "given_name": "Alex",
        #                 "family_name": "Tomlinson",
        #                 "email_address": "alex.w.tomlinson@gmail.com",
        #                 "phone_number": "+12565415960",
        #             },
        #             {
        #                 "id": "9E1WDA0K9MWKQA0XZS14853DNG",
        #                 "given_name": "Alexis",
        #                 "family_name": "Bentley",
        #                 "company_name": "Jeremiah's Italian Ice"
        #             }
        #         ]
        #     },
        # },
        # # Example 2: Retrieve a customer
        # {
        #     "input": {"operation": "retrieve_customer", "customer_id": "TM92KSDVD1XDXKS"},
        #     "output": {
        #         "customer": {
        #             "id": "TM92KSDVD1XDXKS",
        #             "given_name": "John",
        #             "family_name": "Smith",
        #             "email_address": "john.smith@example.com",
        #             "phone_number": "+1-555-123-4567",
        #         }
        #     },
        # },
        # # Example 3: List bookings for a specific customer
        # {
        #     "input": {
        #         "operation": "list_bookings",
        #         "limit": 10,
        #         "customer_id": "CUSTOMER_ID",
        #         "location_id": "LOCATION_ID",
        #         "start_at_min": "2023-04-01T00:00:00Z",
        #         "start_at_max": "2023-04-30T23:59:59Z",
        #     },
        #     "output": {
        #         "bookings": [
        #             {
        #                 "id": "booking_id",
        #                 "version": 0,
        #                 "status": "ACCEPTED",
        #                 "start_at": "2023-04-15T14:00:00Z",
        #                 "location_id": "LOCATION_ID",
        #                 "customer_id": "CUSTOMER_ID",
        #             }
        #         ]
        #     },
        # },
        # # Example 4: Create a new customer
        # {
        #     "input": {
        #         "operation": "create_customer",
        #         "body": {
        #             "given_name": "Sarah",
        #             "family_name": "Johnson",
        #             "email_address": "sarah.johnson@example.com",
        #             "phone_number": "+1-555-987-6543",
        #             "note": "VIP customer",
        #         },
        #     },
        #     "output": {
        #         "customer": {
        #             "id": "NEW_CUSTOMER_ID",
        #             "given_name": "Sarah",
        #             "family_name": "Johnson",
        #             "email_address": "sarah.johnson@example.com",
        #             "phone_number": "+1-555-987-6543",
        #             "note": "VIP customer",
        #             "created_at": "2023-05-15T14:00:00Z",
        #         }
        #     },
        # },
    ]

    def __init__(self):
        """Initialize the Square tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._client = None

    @property
    def client(self):
        """
        Get the Square client, initializing it if needed.
        Lazy loading approach.

        Returns:
            Square client instance

        Raises:
            ToolError: If Square API key is not set or client initialization fails
        """
        if self._client is None:
            # Create a direct Square client using the simplest approach
            try:
                from square.client import Client

                # Get API key from config
                api_key = config.square_api_key
                if not api_key:
                    raise ToolError(
                        "Square API key not found in configuration.",
                        ErrorCode.TOOL_EXECUTION_ERROR,
                    )

                # Create a simple client with just the essential parameters
                self.logger.info(f"Creating Square client with API key")
                self._client = Client(
                    access_token=api_key,
                    environment="production",  # Use production environment by default
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize Square client: {e}")
                raise ToolError(
                    f"Failed to initialize Square client: {e}",
                    ErrorCode.TOOL_EXECUTION_ERROR,
                )
        return self._client

    def _handle_response(self, result, operation):
        """
        Process Square API response and handle errors.

        Args:
            result: API response from Square
            operation: Operation name for error context

        Returns:
            Processed response data

        Raises:
            ToolError: If API returns an error
        """
        if result.is_success():
            self.logger.info(f"Square API {operation} completed successfully")
            return result.body
        elif result.is_error():
            error_message = "Unknown error"
            if hasattr(result, "errors") and result.errors:
                error_details = []
                for error in result.errors:
                    category = error.get("category", "UNKNOWN")
                    code = error.get("code", "UNKNOWN")
                    detail = error.get("detail", "No details provided")
                    error_details.append(f"{category}.{code}: {detail}")

                    # Log extra details for authentication errors
                    if category == "AUTHENTICATION_ERROR":
                        self.logger.error(
                            f"Square API authentication error: {code} - {detail}"
                        )
                        self.logger.error(
                            "Please verify your SQUARE_API_KEY environment variable"
                        )
                        masked_key = os.getenv("SQUARE_API_KEY", "")
                        if masked_key:
                            masked_key = (
                                masked_key[:4] + "..." + masked_key[-4:]
                                if len(masked_key) > 8
                                else "****"
                            )
                            self.logger.error(
                                f"Current API key starts with: {masked_key}"
                            )
                        else:
                            self.logger.error("No API key found in environment")

                error_message = "; ".join(error_details)

            self.logger.error(f"Square API error in {operation}: {error_message}")
            raise ToolError(
                f"Square API error in {operation}: {error_message}",
                ErrorCode.TOOL_EXECUTION_ERROR,
            )
        else:
            self.logger.error(
                f"Unexpected response type from Square API in {operation}"
            )
            raise ToolError(
                f"Unexpected response from Square API in {operation}",
                ErrorCode.TOOL_EXECUTION_ERROR,
            )

    def _handle_pagination(self, operation, api_method, **params):
        """
        Handle paginated API responses.

        Args:
            operation: Operation name for error context
            api_method: Square API method to call
            **params: Parameters to pass to the API method

        Returns:
            Combined results from all pages

        Example:
            For list_customers, this would combine all customer records
            across multiple pages into a single response.
        """
        # Initialize result with structure matching the operation
        if "customer" in operation:
            combined_result = {"customers": []}
        elif "booking" in operation:
            combined_result = {"bookings": []}
        else:
            combined_result = {}

        # Get the key for results based on operation
        if "customer" in operation:
            results_key = "customers"
        elif "booking" in operation:
            results_key = "bookings"
        else:
            results_key = None

        cursor = None

        # Keep fetching until no more pages
        while True:
            # Update cursor in params
            params_with_cursor = params.copy()
            params_with_cursor["cursor"] = cursor

            # Call API method
            result = api_method(**params_with_cursor)
            body = self._handle_response(result, operation)

            # Extract results and add to combined result
            if results_key and results_key in body:
                combined_result[results_key].extend(body[results_key])

            # Copy other fields from response
            for key, value in body.items():
                if key != results_key and key != "cursor":
                    combined_result[key] = value

            # Get next cursor
            cursor = body.get("cursor")

            # Break if no more pages
            if not cursor:
                break

        return combined_result

    # ===== Customer Management Methods =====

    def list_customers(
        self, cursor=None, limit=None, sort_field=None, sort_order=None, count=False
    ):
        """
        List customer profiles associated with a Square account.

        Args:
            cursor (str, optional): Pagination cursor
            limit (int, optional): Maximum number of results (1-100)
            sort_field (str, optional): Sort field ('DEFAULT', 'CREATED_AT')
            sort_order (str, optional): Sort order ('ASC', 'DESC')
            count (bool, optional): Whether to return the total count

        Returns:
            Dict containing customers list and optional pagination info
        """
        result = self.client.customers.list_customers(
            cursor=cursor,
            limit=limit,
            sort_field=sort_field,
            sort_order=sort_order,
            count=count,
        )
        return self._handle_response(result, "list_customers")

    def retrieve_customer(self, customer_id):
        """
        Retrieve a customer profile by ID.

        Args:
            customer_id (str): The ID of the customer to retrieve

        Returns:
            Dict containing customer information
        """
        result = self.client.customers.retrieve_customer(customer_id=customer_id)
        return self._handle_response(result, "retrieve_customer")

    def create_customer(self, body):
        """
        Create a new customer profile.

        Args:
            body (dict): The create request with customer data
                Structure:
                {
                    "given_name": str,
                    "family_name": str,
                    "company_name": str,
                    "nickname": str,
                    "email_address": str,
                    "address": {
                        "address_line_1": str,
                        "address_line_2": str,
                        "locality": str,
                        "administrative_district_level_1": str,
                        "postal_code": str,
                        "country": str
                    },
                    "phone_number": str,
                    "reference_id": str,
                    "note": str,
                    "birthday": str,  # YYYY-MM-DD format
                    "tax_ids": {
                        "eu_vat": str
                    }
                }

        Returns:
            Dict containing created customer information
        """
        result = self.client.customers.create_customer(body=body)
        return self._handle_response(result, "create_customer")

    def update_customer(self, customer_id, body):
        """
        Update an existing customer profile.

        Args:
            customer_id (str): The ID of the customer to update
            body (dict): The update request with customer data
                Structure:
                {
                    "version": int,  # Required for optimistic concurrency
                    "given_name": str,
                    "family_name": str,
                    "company_name": str,
                    "nickname": str,
                    "email_address": str,
                    "address": {
                        "address_line_1": str,
                        "address_line_2": str,
                        "locality": str,
                        "administrative_district_level_1": str,
                        "postal_code": str,
                        "country": str
                    },
                    "phone_number": str,
                    "reference_id": str,
                    "note": str,
                    "birthday": str,  # YYYY-MM-DD format
                    "tax_ids": {
                        "eu_vat": str
                    }
                }

        Returns:
            Dict containing updated customer information
        """
        result = self.client.customers.update_customer(
            customer_id=customer_id, body=body
        )
        return self._handle_response(result, "update_customer")

    def delete_customer(self, customer_id, version=None):
        """
        Delete a customer profile.

        Args:
            customer_id (str): The ID of the customer to delete
            version (int, optional): Current version for optimistic concurrency

        Returns:
            Dict containing deletion result
        """
        result = self.client.customers.delete_customer(
            customer_id=customer_id, version=version
        )
        return self._handle_response(result, "delete_customer")

    def bulk_create_customers(self, body):
        """
        Create multiple customer profiles.

        Args:
            body (dict): The bulk create request
                Structure:
                {
                    "customers": {
                        "key1": {
                            # customer data same as create_customer
                        },
                        "key2": {
                            # another customer
                        }
                    }
                }

        Returns:
            Dict containing results for each customer creation attempt
        """
        result = self.client.customers.bulk_create_customers(body=body)
        return self._handle_response(result, "bulk_create_customers")

    def bulk_retrieve_customers(self, body):
        """
        Retrieve multiple customer profiles by their IDs.

        Args:
            body (dict): The bulk retrieve request
                Structure:
                {
                    "customer_ids": [str, str, ...]
                }

        Returns:
            Dict containing results for each customer retrieval attempt
        """
        result = self.client.customers.bulk_retrieve_customers(body=body)
        return self._handle_response(result, "bulk_retrieve_customers")

    def bulk_update_customers(self, body):
        """
        Update multiple customer profiles.

        Args:
            body (dict): The bulk update request
                Structure:
                {
                    "customers": {
                        "customer_id_1": {
                            # customer data same as update_customer
                        },
                        "customer_id_2": {
                            # another customer update
                        }
                    }
                }

        Returns:
            Dict containing results for each customer update attempt
        """
        result = self.client.customers.bulk_update_customers(body=body)
        return self._handle_response(result, "bulk_update_customers")

    def bulk_delete_customers(self, body):
        """
        Delete multiple customer profiles.

        Args:
            body (dict): The bulk delete request
                Structure:
                {
                    "customer_ids": [str, str, ...],
                }

        Returns:
            Dict containing results for each customer deletion attempt
        """
        result = self.client.customers.bulk_delete_customers(body=body)
        return self._handle_response(result, "bulk_delete_customers")

    # ===== Catalog Management Methods =====

    def list_catalog(self, cursor=None, types=None, catalog_version=None):
        """
        List catalog objects of specified types.

        Args:
            cursor (str, optional): Pagination cursor from a previous response
            types (str, optional): Comma-separated list of object types to retrieve
                (e.g., "ITEM,CATEGORY,TAX")
            catalog_version (int, optional): Specific version of catalog objects to include

        Returns:
            Dict containing catalog objects and pagination info
        """
        result = self.client.catalog.list_catalog(
            cursor=cursor,
            types=types,
            catalog_version=catalog_version
        )
        return self._handle_response(result, "list_catalog")

    def retrieve_catalog_object(self, object_id, include_related_objects=False, 
                               catalog_version=None, include_category_path_to_root=False):
        """
        Retrieve a catalog object by ID.

        Args:
            object_id (str): The ID of the catalog object to retrieve
            include_related_objects (bool, optional): Include related objects
                referenced by the retrieved object
            catalog_version (int, optional): Specific version to retrieve
            include_category_path_to_root (bool, optional): Include the path from
                the category to the root category

        Returns:
            Dict containing the catalog object and related objects if requested
        """
        result = self.client.catalog.retrieve_catalog_object(
            object_id=object_id,
            include_related_objects=include_related_objects,
            catalog_version=catalog_version,
            include_category_path_to_root=include_category_path_to_root
        )
        return self._handle_response(result, "retrieve_catalog_object")

    def search_catalog_objects(self, body):
        """
        Search for catalog objects using a variety of filters.

        Args:
            body (dict): The search request
                Structure:
                {
                    "object_types": [str, ...],  # Types to search for
                    "query": {
                        "exact_query": {
                            "attribute_name": str,
                            "attribute_value": str
                        },
                        "prefix_query": {
                            "attribute_name": str,
                            "attribute_prefix": str
                        },
                        "text_query": {
                            "keywords": [str, ...]
                        }
                    },
                    "include_deleted_objects": bool,
                    "include_related_objects": bool,
                    "begin_time": str,  # RFC 3339 timestamp
                    "limit": int
                }

        Returns:
            Dict containing matching catalog objects
        """
        result = self.client.catalog.search_catalog_objects(body=body)
        return self._handle_response(result, "search_catalog_objects")

    def search_catalog_items(self, body):
        """
        Search specifically for catalog items with detailed query options.

        Args:
            body (dict): The search request
                Structure:
                {
                    "text_filter": str,  # Full-text search
                    "category_ids": [str, ...],
                    "stock_levels": [str, ...],  # "OUT", "LOW", "IN_STOCK"
                    "enabled_location_ids": [str, ...],
                    "product_types": [str, ...],
                    "custom_attribute_filters": [
                        {
                            "custom_attribute_definition_id": str,
                            "key": str,
                            "string_filter": str,
                            "number_filter": {
                                "min": str,
                                "max": str
                            },
                            "selection_uids_filter": [str, ...],
                            "bool_filter": bool
                        }
                    ],
                    "sort_order": str,  # "ASC" or "DESC"
                    "limit": int,
                    "cursor": str
                }

        Returns:
            Dict containing matching catalog items
        """
        result = self.client.catalog.search_catalog_items(body=body)
        return self._handle_response(result, "search_catalog_items")

    def upsert_catalog_object(self, body):
        """
        Create a new catalog object or update an existing one.

        Args:
            body (dict): The upsert request
                Structure:
                {
                    "idempotency_key": str,
                    "object": {
                        "type": str,  # ITEM, CATEGORY, TAX, etc.
                        "id": str,  # Optional, will be generated if not provided
                        "item_data": {},  # For ITEM type
                        "category_data": {},  # For CATEGORY type
                        # Other *_data fields based on type
                    }
                }

        Returns:
            Dict containing the created/updated catalog object
        """
        result = self.client.catalog.upsert_catalog_object(body=body)
        return self._handle_response(result, "upsert_catalog_object")

    def delete_catalog_object(self, object_id):
        """
        Delete a catalog object by ID.

        Args:
            object_id (str): The ID of the catalog object to delete

        Returns:
            Dict containing the IDs of all deleted objects (cascading deletion)
        """
        result = self.client.catalog.delete_catalog_object(object_id=object_id)
        return self._handle_response(result, "delete_catalog_object")

    def batch_delete_catalog_objects(self, body):
        """
        Delete multiple catalog objects in a single request.

        Args:
            body (dict): The batch delete request
                Structure:
                {
                    "object_ids": [str, ...]  # IDs of objects to delete
                }

        Returns:
            Dict containing the IDs of all deleted objects
        """
        result = self.client.catalog.batch_delete_catalog_objects(body=body)
        return self._handle_response(result, "batch_delete_catalog_objects")

    def batch_retrieve_catalog_objects(self, body):
        """
        Retrieve multiple catalog objects in a single request.

        Args:
            body (dict): The batch retrieve request
                Structure:
                {
                    "object_ids": [str, ...],  # IDs of objects to retrieve
                    "include_related_objects": bool
                }

        Returns:
            Dict containing the requested catalog objects
        """
        result = self.client.catalog.batch_retrieve_catalog_objects(body=body)
        return self._handle_response(result, "batch_retrieve_catalog_objects")

    def batch_upsert_catalog_objects(self, body):
        """
        Create or update multiple catalog objects in a single request.

        Args:
            body (dict): The batch upsert request
                Structure:
                {
                    "idempotency_key": str,
                    "batches": [
                        {
                            "objects": [
                                {
                                    "type": str,  # ITEM, CATEGORY, etc.
                                    "id": str,  # Optional
                                    # Various *_data fields
                                }
                            ]
                        }
                    ]
                }

        Returns:
            Dict containing the created/updated catalog objects
        """
        result = self.client.catalog.batch_upsert_catalog_objects(body=body)
        return self._handle_response(result, "batch_upsert_catalog_objects")

    def catalog_info(self):
        """
        Get information about the Square Catalog API.

        Returns:
            Dict containing information about the catalog API, like batch size limits
        """
        result = self.client.catalog.catalog_info()
        return self._handle_response(result, "catalog_info")

    def create_catalog_image(self, request, image_file):
        """
        Upload an image file for a CatalogImage.

        Args:
            request (dict): The create image request
                Structure:
                {
                    "idempotency_key": str,
                    "image": {
                        "type": "IMAGE",
                        "id": str,  # Optional
                        "image_data": {
                            "caption": str  # Optional
                        }
                    },
                    "object_id": str  # Optional, to associate with a catalog object
                }
            image_file (file): The image file to upload

        Returns:
            Dict containing the created catalog image information
        """
        result = self.client.catalog.create_catalog_image(request=request, image_file=image_file)
        return self._handle_response(result, "create_catalog_image")

    def update_item_taxes(self, body):
        """
        Update the tax settings for an item without replacing the entire item.

        Args:
            body (dict): The update request
                Structure:
                {
                    "item_ids": [str, ...],  # Item IDs to update
                    "taxes_to_enable": [str, ...],  # Tax IDs to apply
                    "taxes_to_disable": [str, ...]  # Tax IDs to remove
                }

        Returns:
            Dict containing the updated information
        """
        result = self.client.catalog.update_item_taxes(body=body)
        return self._handle_response(result, "update_item_taxes")

    def update_item_modifier_lists(self, body):
        """
        Update the modifier lists for an item without replacing the entire item.

        Args:
            body (dict): The update request
                Structure:
                {
                    "item_ids": [str, ...],  # Item IDs to update
                    "modifier_lists_to_enable": [str, ...],  # Modifier list IDs to apply
                    "modifier_lists_to_disable": [str, ...]  # Modifier list IDs to remove
                }

        Returns:
            Dict containing the updated information
        """
        result = self.client.catalog.update_item_modifier_lists(body=body)
        return self._handle_response(result, "update_item_modifier_lists")

    # ===== Booking Management Methods =====

    def create_booking(self, body):
        """
        Create a new booking.

        Args:
            body (dict): The create request with booking data
                Structure:
                {
                    "booking": {
                        "location_id": str,     # Required
                        "start_at": str,        # Required (RFC 3339 timestamp)
                        "customer_id": str,     # Optional
                        "customer_note": str,   # Optional
                        "seller_note": str,     # Optional
                        "appointment_segments": [  # Required (at least one)
                            {
                                "team_member_id": str,          # Required
                                "service_variation_id": str,    # Required
                                "service_variation_version": int,  # Required
                                "duration_minutes": int  # Optional (defaults to service variation duration)
                            }
                        ]
                    },
                    "idempotency_key": str  # Optional for ensuring request uniqueness
                }

        Returns:
            Dict containing created booking information
        """
        result = self.client.bookings.create_booking(body=body)
        return self._handle_response(result, "create_booking")

    def retrieve_booking(self, booking_id):
        """
        Retrieve a booking by ID.

        Args:
            booking_id (str): The ID of the booking to retrieve

        Returns:
            Dict containing booking information
        """
        result = self.client.bookings.retrieve_booking(booking_id=booking_id)
        return self._handle_response(result, "retrieve_booking")

    def update_booking(self, booking_id, body):
        """
        Update an existing booking.

        Args:
            booking_id (str): The ID of the booking to update
            body (dict): The update request with booking data
                Structure:
                {
                    "booking": {
                        "version": int,         # Required for optimistic concurrency
                        "status": str,          # Optional (ACCEPTED, DECLINED, CANCELLED, etc.)
                        "start_at": str,        # Optional (RFC 3339 timestamp)
                        "location_id": str,     # Optional
                        "customer_id": str,     # Optional
                        "customer_note": str,   # Optional
                        "seller_note": str,     # Optional
                        "appointment_segments": [  # Optional
                            {
                                "team_member_id": str,
                                "service_variation_id": str,
                                "service_variation_version": int,
                                "duration_minutes": int
                            }
                        ]
                    },
                    "idempotency_key": str  # Optional for ensuring request uniqueness
                }

        Returns:
            Dict containing updated booking information
        """
        result = self.client.bookings.update_booking(booking_id=booking_id, body=body)
        return self._handle_response(result, "update_booking")

    def cancel_booking(self, booking_id, body):
        """
        Cancel a booking.

        Args:
            booking_id (str): The ID of the booking to cancel
            body (dict): The cancel request
                Structure:
                {
                    "booking_version": int,  # Required for optimistic concurrency
                    "idempotency_key": str   # Optional for ensuring request uniqueness
                }

        Returns:
            Dict containing the cancelled booking information
        """
        result = self.client.bookings.cancel_booking(booking_id=booking_id, body=body)
        return self._handle_response(result, "cancel_booking")

    def list_bookings(
        self,
        limit=None,
        cursor=None,
        customer_id=None,
        team_member_id=None,
        location_id=None,
        start_at_min=None,
        start_at_max=None,
    ):
        """
        List bookings with filters.

        Args:
            limit (int, optional): Maximum number of results
            cursor (str, optional): Pagination cursor
            customer_id (str, optional): Filter by customer
            team_member_id (str, optional): Filter by team member
            location_id (str, optional): Filter by location
            start_at_min (str, optional): Minimum start time (RFC 3339 timestamp)
            start_at_max (str, optional): Maximum start time (RFC 3339 timestamp)

        Returns:
            Dict containing bookings list and pagination info
        """
        result = self.client.bookings.list_bookings(
            limit=limit,
            cursor=cursor,
            customer_id=customer_id,
            team_member_id=team_member_id,
            location_id=location_id,
            start_at_min=start_at_min,
            start_at_max=start_at_max,
        )
        return self._handle_response(result, "list_bookings")

    def search_availability(self, body):
        """
        Search for available booking times.

        Args:
            body (dict): The search availability request
                Structure:
                {
                    "query": {
                        "filter": {
                            "start_at_range": {
                                "start_at": str,  # RFC 3339 timestamp
                                "end_at": str     # RFC 3339 timestamp
                            },
                            "location_id": str,
                            "segment_filters": [
                                {
                                    "service_variation_id": str,
                                    "team_member_id_filter": {
                                        "any": [str, str, ...] or
                                        "all": [str, str, ...]
                                    }
                                }
                            ]
                        }
                    }
                }

        Returns:
            Dict containing available time slots
        """
        result = self.client.bookings.search_availability(body=body)
        return self._handle_response(result, "search_availability")

    def retrieve_business_booking_profile(self):
        """
        Retrieve the business's booking profile.

        Returns:
            Dict containing business booking profile information
        """
        result = self.client.bookings.retrieve_business_booking_profile()
        return self._handle_response(result, "retrieve_business_booking_profile")

    def list_team_member_booking_profiles(
        self, bookable_only=False, limit=None, cursor=None, location_id=None
    ):
        """
        List booking profiles for team members.

        Args:
            bookable_only (bool, optional): Include only bookable team members
            limit (int, optional): Maximum number of results
            cursor (str, optional): Pagination cursor
            location_id (str, optional): Filter by location

        Returns:
            Dict containing team member booking profiles
        """
        result = self.client.bookings.list_team_member_booking_profiles(
            bookable_only=bookable_only,
            limit=limit,
            cursor=cursor,
            location_id=location_id,
        )
        return self._handle_response(result, "list_team_member_booking_profiles")

    def bulk_retrieve_bookings(self, body):
        """
        Retrieve multiple bookings by their IDs.

        Args:
            body (dict): The bulk retrieve request
                Structure:
                {
                    "booking_ids": [str, str, ...]
                }

        Returns:
            Dict containing results for each booking retrieval attempt
        """
        result = self.client.bookings.bulk_retrieve_bookings(body=body)
        return self._handle_response(result, "bulk_retrieve_bookings")

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a Square API operation.

        Args:
            operation: Operation to perform (see below for valid operations)
            **kwargs: Parameters for the specific operation

        Returns:
            Response data for the operation

        Raises:
            ToolError: If operation fails or parameters are invalid

        Valid Operations:

        Customer Operations:
        - list_customers: List customer profiles
            params: limit, cursor, sort_field, sort_order, count
        - retrieve_customer: Get a customer by ID
            params: customer_id (required)
        - create_customer: Create a new customer
            params: body (required) - The create request
        - update_customer: Update an existing customer
            params: customer_id (required), body (required)
        - delete_customer: Delete a customer
            params: customer_id (required), version
        - bulk_create_customers: Create multiple customers
            params: body (required) - The bulk create request
        - bulk_retrieve_customers: Retrieve multiple customers
            params: body (required) - Contains customer_ids array
        - bulk_update_customers: Update multiple customers
            params: body (required) - The bulk update request
        - bulk_delete_customers: Delete multiple customers
            params: body (required) - Contains customer_ids array

        Booking Operations:
        - create_booking: Create a new booking
            params: body (required) - The create request
        - retrieve_booking: Get a booking by ID
            params: booking_id (required)
        - update_booking: Update an existing booking
            params: booking_id (required), body (required)
        - cancel_booking: Cancel a booking
            params: booking_id (required), body (required)
        - list_bookings: List bookings with filters
            params: limit, cursor, customer_id, team_member_id, location_id,
                   start_at_min, start_at_max
        - search_availability: Search for available booking times
            params: body (required) - The search request
        - retrieve_business_booking_profile: Get business booking profile
            params: none
        - list_team_member_booking_profiles: Get team member booking profiles
            params: bookable_only, limit, cursor, location_id
        - bulk_retrieve_bookings: Retrieve multiple bookings
            params: body (required) - Contains booking_ids array
        """
        with error_context(
            component_name=self.name,
            operation=f"executing {operation}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger,
        ):
            # Handle JSON string passed in 'params' field
            params = kwargs
            if "params" in kwargs and isinstance(kwargs["params"], str):
                try:
                    params = json.loads(kwargs["params"])
                    self.logger.debug(f"Parsed JSON params: {params}")
                except json.JSONDecodeError as e:
                    raise ToolError(
                        f"Invalid JSON in params: {e}", ErrorCode.TOOL_INVALID_INPUT
                    )

            # Process kwargs if passed as JSON string
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    kwargs_parsed = json.loads(kwargs["kwargs"])
                    params.update(kwargs_parsed)
                    self.logger.debug(f"Parsed kwargs JSON: {kwargs_parsed}")
                except json.JSONDecodeError as e:
                    raise ToolError(
                        f"Invalid JSON in kwargs: {e}", ErrorCode.TOOL_INVALID_INPUT
                    )

            # Customer Management Operations
            if operation == "list_customers":
                return self.list_customers(
                    cursor=params.get("cursor"),
                    limit=params.get("limit"),
                    sort_field=params.get("sort_field"),
                    sort_order=params.get("sort_order"),
                    count=params.get("count", False),
                )

            elif operation == "retrieve_customer":
                if "customer_id" not in params:
                    raise ToolError(
                        "customer_id parameter is required for retrieve_customer operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.retrieve_customer(params["customer_id"])

            elif operation == "create_customer":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for create_customer operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.create_customer(params["body"])

            elif operation == "update_customer":
                if "customer_id" not in params:
                    raise ToolError(
                        "customer_id parameter is required for update_customer operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for update_customer operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.update_customer(params["customer_id"], params["body"])

            elif operation == "delete_customer":
                if "customer_id" not in params:
                    raise ToolError(
                        "customer_id parameter is required for delete_customer operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.delete_customer(
                    params["customer_id"], params.get("version")
                )

            elif operation == "bulk_create_customers":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for bulk_create_customers operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.bulk_create_customers(params["body"])

            elif operation == "bulk_retrieve_customers":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for bulk_retrieve_customers operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.bulk_retrieve_customers(params["body"])

            elif operation == "bulk_update_customers":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for bulk_update_customers operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.bulk_update_customers(params["body"])

            elif operation == "bulk_delete_customers":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for bulk_delete_customers operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.bulk_delete_customers(params["body"])

            # Booking Management Operations
            elif operation == "create_booking":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for create_booking operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                
                # Validate the booking request for required fields
                booking_data = params["body"].get("booking", {})
                missing_fields = []
                
                # Check for required fields
                if not booking_data.get("location_id"):
                    missing_fields.append("location_id")
                
                if not booking_data.get("start_at"):
                    missing_fields.append("start_at")
                
                # Check if appointment segments exist
                if not booking_data.get("appointment_segments") or len(booking_data.get("appointment_segments", [])) == 0:
                    missing_fields.append("appointment_segments")
                else:
                    segments = booking_data.get("appointment_segments", [])
                    for idx, segment in enumerate(segments):
                        # Apply default team member ID if not provided
                        if not segment.get("team_member_id"):
                            self.logger.info(f"Using default team member ID for segment {idx}")
                            segment["team_member_id"] = config.square.default_team_member_id
                        
                        # Check for other required fields
                        if not segment.get("service_variation_id"):
                            missing_fields.append(f"appointment_segments[{idx}].service_variation_id")
                        if not segment.get("service_variation_version"):
                            missing_fields.append(f"appointment_segments[{idx}].service_variation_version")
                
                # If there are missing fields, return information about what's missing
                if missing_fields:
                    self.logger.info(f"Incomplete booking request, missing: {missing_fields}")
                    return {
                        "incomplete_booking": True,
                        "missing_fields": missing_fields,
                        "partial_booking_data": booking_data,
                        "message": "The booking request is incomplete. Please provide the missing information."
                    }
                
                # If all required fields are present, proceed with the booking
                return self.create_booking(params["body"])

            elif operation == "retrieve_booking":
                if "booking_id" not in params:
                    raise ToolError(
                        "booking_id parameter is required for retrieve_booking operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.retrieve_booking(params["booking_id"])

            elif operation == "update_booking":
                if "booking_id" not in params:
                    raise ToolError(
                        "booking_id parameter is required for update_booking operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for update_booking operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.update_booking(params["booking_id"], params["body"])

            elif operation == "cancel_booking":
                if "booking_id" not in params:
                    raise ToolError(
                        "booking_id parameter is required for cancel_booking operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for cancel_booking operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.cancel_booking(params["booking_id"], params["body"])

            elif operation == "list_bookings":
                return self.list_bookings(
                    limit=params.get("limit"),
                    cursor=params.get("cursor"),
                    customer_id=params.get("customer_id"),
                    team_member_id=params.get("team_member_id"),
                    location_id=params.get("location_id"),
                    start_at_min=params.get("start_at_min"),
                    start_at_max=params.get("start_at_max"),
                )

            elif operation == "search_availability":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for search_availability operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.search_availability(params["body"])

            elif operation == "retrieve_business_booking_profile":
                return self.retrieve_business_booking_profile()

            elif operation == "list_team_member_booking_profiles":
                return self.list_team_member_booking_profiles(
                    bookable_only=params.get("bookable_only", False),
                    limit=params.get("limit"),
                    cursor=params.get("cursor"),
                    location_id=params.get("location_id"),
                )

            elif operation == "bulk_retrieve_bookings":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for bulk_retrieve_bookings operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.bulk_retrieve_bookings(params["body"])
                
            # Catalog Management Operations
            elif operation == "list_catalog":
                return self.list_catalog(
                    cursor=params.get("cursor"),
                    types=params.get("types"),
                    catalog_version=params.get("catalog_version")
                )
                
            elif operation == "retrieve_catalog_object":
                if "object_id" not in params:
                    raise ToolError(
                        "object_id parameter is required for retrieve_catalog_object operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.retrieve_catalog_object(
                    object_id=params["object_id"],
                    include_related_objects=params.get("include_related_objects", False),
                    catalog_version=params.get("catalog_version"),
                    include_category_path_to_root=params.get("include_category_path_to_root", False)
                )
                
            elif operation == "search_catalog_objects":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for search_catalog_objects operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.search_catalog_objects(params["body"])
                
            elif operation == "search_catalog_items":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for search_catalog_items operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.search_catalog_items(params["body"])
                
            elif operation == "upsert_catalog_object":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for upsert_catalog_object operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.upsert_catalog_object(params["body"])
                
            elif operation == "delete_catalog_object":
                if "object_id" not in params:
                    raise ToolError(
                        "object_id parameter is required for delete_catalog_object operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.delete_catalog_object(params["object_id"])
                
            elif operation == "batch_delete_catalog_objects":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for batch_delete_catalog_objects operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.batch_delete_catalog_objects(params["body"])
                
            elif operation == "batch_retrieve_catalog_objects":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for batch_retrieve_catalog_objects operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.batch_retrieve_catalog_objects(params["body"])
                
            elif operation == "batch_upsert_catalog_objects":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for batch_upsert_catalog_objects operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.batch_upsert_catalog_objects(params["body"])
                
            elif operation == "catalog_info":
                return self.catalog_info()
                
            elif operation == "create_catalog_image":
                if "request" not in params:
                    raise ToolError(
                        "request parameter is required for create_catalog_image operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                if "image_file" not in params:
                    raise ToolError(
                        "image_file parameter is required for create_catalog_image operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.create_catalog_image(params["request"], params["image_file"])
                
            elif operation == "update_item_taxes":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for update_item_taxes operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.update_item_taxes(params["body"])
                
            elif operation == "update_item_modifier_lists":
                if "body" not in params:
                    raise ToolError(
                        "body parameter is required for update_item_modifier_lists operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                return self.update_item_modifier_lists(params["body"])

            else:
                # Special error message for common search mistakes
                if operation in ["search_customer", "search", "find_customer", "find"]:
                    raise ToolError(
                        f"To search for customers, please use the customer_tool instead with its search_customers operation",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                else:
                    raise ToolError(
                        f"Unknown operation: {operation}. Valid operations are: "
                        "list_customers, retrieve_customer, create_customer, "
                        "update_customer, delete_customer, bulk_create_customers, bulk_retrieve_customers, "
                        "bulk_update_customers, bulk_delete_customers, create_booking, retrieve_booking, "
                        "update_booking, cancel_booking, list_bookings, search_availability, "
                        "retrieve_business_booking_profile, list_team_member_booking_profiles, bulk_retrieve_bookings, "
                        "list_catalog, retrieve_catalog_object, search_catalog_objects, search_catalog_items, "
                        "upsert_catalog_object, delete_catalog_object, batch_delete_catalog_objects, "
                        "batch_retrieve_catalog_objects, batch_upsert_catalog_objects, catalog_info, "
                        "create_catalog_image, update_item_taxes, update_item_modifier_lists",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
