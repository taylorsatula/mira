"""
Square API integration tool.

This tool enables the bot to interact with Square API endpoints to manage
bookings, customers, and catalog information using a template-based approach
for better parameter validation and error handling.

Requires Square API token in the .env file as SQUARE_TOKEN.
"""

# Standard library imports
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

# Third-party imports
import dotenv
import requests
from pydantic import BaseModel, Field

# Local imports
from config.registry import registry
from errors import ToolError, ErrorCode, error_context
from tools.repo import Tool

# Load environment variables
dotenv.load_dotenv()

# Define configuration class for SquareApiTool
class SquareApiToolConfig(BaseModel):
    """Configuration for the Square API tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    timeout: int = Field(default=60, description="Timeout in seconds for Square API requests")
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")
    backoff_factor: float = Field(default=2.0, description="Backoff factor for retries")
    default_location_id: str = Field(default="LRBQZJ7CAJBV5", description="Default location ID for Square API requests")
    default_location_type: str = Field(default="CUSTOMER_LOCATION", description="Default location type for bookings")
    default_team_member_id: str = Field(default="TMjY-uYPS-Wb2hDU", description="Default team member ID for bookings")

# Register with registry
registry.register("squareapi_tool", SquareApiToolConfig)

class SquareApiTool(Tool):
    """
    Tool for interacting with Square API using a template-based approach.

    Features:
    1. Bookings API:
       - Create, manage, and search bookings
       - Handle availability and scheduling

    2. Customers API:
       - Create and manage customer profiles
       - Search for customers

    3. Catalog API:
       - Manage items, services, and categories
       - Search catalog items
    """

    name = "squareapi_tool"
    simple_description = """Provides access to Square API for managing bookings, customers, and catalog information. Use this tool when the user needs to interact with Square's business management platform."""
    
    implementation_details = """
    This tool enables interaction with Square API for various operations including:
   
    1. Bookings API:
       - Create bookings with customer details, services, and time slots
       - Retrieve, update, and cancel existing bookings
       - Search for available time slots and existing bookings
       
    2. Customers API:
       - Create and update customer profiles
       - Search for customers by name, email, phone
       - Retrieve customer details and history
       
    3. Catalog API:
       - Manage services, items, and categories
       - Search catalog for available services
       - Retrieve pricing and availability information
    
    Each operation provides automatic validation of required parameters and helpful
    error messages to guide correct API usage. Default values are provided for common
    parameters like location_id and team_member_id to simplify requests.
    """
    
    description = simple_description + implementation_details

    usage_examples = [
        # Create Booking Example
        {
            "input": {
                "operation": "create_booking",
                "start_at": "2023-05-30T15:00:00Z",
                "service_variation_id": "VBKQNFTZGCKHCH7ZYN3NP4YZ",
                "team_member_id": "TMjY-uYPS-Wb2hDU",
                "customer_id": "8FYVP54K71AGB1JGJN8SC72N24",
                "customer_note": "Please call when you arrive",
                "location_type": "CUSTOMER_LOCATION",
                "address": {
                    "address_line_1": "500 Electric Ave",
                    "locality": "New York",
                    "administrative_district_level_1": "NY",
                    "postal_code": "10003",
                    "country": "US"
                }
            },
            "output": {
                "booking": {
                    "id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
                    "version": 0,
                    "status": "ACCEPTED",
                    "created_at": "2023-05-01T15:47:41Z",
                    "updated_at": "2023-05-01T15:47:41Z",
                    "location_id": "LRBQZJ7CAJBV5",
                    "customer_id": "8FYVP54K71AGB1JGJN8SC72N24",
                    "customer_note": "Please call when you arrive",
                    "seller_note": "",
                    "start_at": "2023-05-30T15:00:00Z",
                    "location_type": "CUSTOMER_LOCATION",
                    "appointment_segments": [
                        {
                            "duration_minutes": 60,
                            "service_variation_id": "VBKQNFTZGCKHCH7ZYN3NP4YZ",
                            "team_member_id": "TMjY-uYPS-Wb2hDU",
                            "service_variation_version": 1599775456731
                        }
                    ],
                    "address": {
                        "address_line_1": "500 Electric Ave",
                        "locality": "New York",
                        "administrative_district_level_1": "NY",
                        "postal_code": "10003",
                        "country": "US"
                    }
                }
            }
        },
        
        # List Bookings Example
        {
            "input": {
                "operation": "list_bookings",
                "location_id": "LRBQZJ7CAJBV5",
                "start_at_min": "2023-05-01T00:00:00Z",
                "start_at_max": "2023-05-31T23:59:59Z",
                "limit": 50
            },
            "output": {
                "bookings": [
                    {
                        "id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
                        "version": 0,
                        "status": "ACCEPTED",
                        "created_at": "2023-05-01T15:47:41Z",
                        "updated_at": "2023-05-01T15:47:41Z",
                        "location_id": "LRBQZJ7CAJBV5",
                        "customer_id": "8FYVP54K71AGB1JGJN8SC72N24",
                        "customer_note": "Please call when you arrive",
                        "seller_note": "",
                        "start_at": "2023-05-30T15:00:00Z"
                    }
                ],
                "cursor": "CURSOR_STRING_FOR_NEXT_PAGE"
            }
        },
        
        # Retrieve Booking Example
        {
            "input": {
                "operation": "retrieve_booking",
                "booking_id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI"
            },
            "output": {
                "booking": {
                    "id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
                    "version": 0,
                    "status": "ACCEPTED",
                    "created_at": "2023-05-01T15:47:41Z",
                    "updated_at": "2023-05-01T15:47:41Z",
                    "location_id": "LRBQZJ7CAJBV5",
                    "customer_id": "8FYVP54K71AGB1JGJN8SC72N24",
                    "customer_note": "Please call when you arrive",
                    "seller_note": "",
                    "start_at": "2023-05-30T15:00:00Z",
                    "appointment_segments": [
                        {
                            "duration_minutes": 60,
                            "service_variation_id": "VBKQNFTZGCKHCH7ZYN3NP4YZ",
                            "team_member_id": "TMjY-uYPS-Wb2hDU",
                            "service_variation_version": 1599775456731
                        }
                    ]
                }
            }
        },
        
        # Update Booking Example
        {
            "input": {
                "operation": "update_booking",
                "booking_id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
                "version": 0,
                "start_at": "2023-05-30T16:00:00Z",
                "customer_note": "I'll be 15 minutes late"
            },
            "output": {
                "booking": {
                    "id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
                    "version": 1,
                    "status": "ACCEPTED",
                    "created_at": "2023-05-01T15:47:41Z",
                    "updated_at": "2023-05-01T16:05:12Z",
                    "location_id": "LRBQZJ7CAJBV5",
                    "customer_id": "8FYVP54K71AGB1JGJN8SC72N24",
                    "customer_note": "I'll be 15 minutes late",
                    "seller_note": "",
                    "start_at": "2023-05-30T16:00:00Z",
                    "appointment_segments": [
                        {
                            "duration_minutes": 60,
                            "service_variation_id": "VBKQNFTZGCKHCH7ZYN3NP4YZ",
                            "team_member_id": "TMjY-uYPS-Wb2hDU",
                            "service_variation_version": 1599775456731
                        }
                    ]
                }
            }
        },
        
        # Cancel Booking Example
        {
            "input": {
                "operation": "cancel_booking",
                "booking_id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
                "booking_version": 1
            },
            "output": {
                "booking": {
                    "id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
                    "version": 2,
                    "status": "CANCELLED_BY_SELLER",
                    "created_at": "2023-05-01T15:47:41Z",
                    "updated_at": "2023-05-01T16:15:35Z",
                    "location_id": "LRBQZJ7CAJBV5",
                    "customer_id": "8FYVP54K71AGB1JGJN8SC72N24",
                    "customer_note": "I'll be 15 minutes late",
                    "seller_note": "",
                    "start_at": "2023-05-30T16:00:00Z"
                }
            }
        },
        
        # List Customers Example
        {
            "input": {
                "operation": "list_customers",
                "limit": 50,
                "sort_field": "CREATED_AT",
                "sort_order": "DESC",
                "count": True
            },
            "output": {
                "customers": [
                    {
                        "id": "JDKYHBWT1D4F8MFH63DBMEN8Y4",
                        "created_at": "2016-03-23T20:21:54.859Z",
                        "updated_at": "2016-03-23T20:21:55Z",
                        "given_name": "Amelia",
                        "family_name": "Earhart",
                        "email_address": "Amelia.Earhart@example.com",
                        "address": {
                            "address_line_1": "500 Electric Ave",
                            "address_line_2": "Suite 600",
                            "locality": "New York",
                            "administrative_district_level_1": "NY",
                            "postal_code": "10003",
                            "country": "US"
                        },
                        "phone_number": "+1-212-555-4240",
                        "reference_id": "YOUR_REFERENCE_ID",
                        "note": "a customer",
                        "preferences": {
                            "email_unsubscribed": False
                        },
                        "creation_source": "THIRD_PARTY",
                        "version": 1
                    }
                ],
                "cursor": "CURSOR_STRING_FOR_NEXT_PAGE",
                "count": 1
            }
        },
        
        # Create Customer Example
        {
            "input": {
                "operation": "create_customer",
                "given_name": "Amelia",
                "family_name": "Earhart",
                "email_address": "Amelia.Earhart@example.com",
                "address": {
                    "address_line_1": "500 Electric Ave",
                    "address_line_2": "Suite 600",
                    "locality": "New York",
                    "administrative_district_level_1": "NY",
                    "postal_code": "10003",
                    "country": "US"
                },
                "phone_number": "+1-212-555-4240",
                "reference_id": "YOUR_REFERENCE_ID",
                "note": "a customer"
            },
            "output": {
                "customer": {
                    "id": "JDKYHBWT1D4F8MFH63DBMEN8Y4",
                    "created_at": "2016-03-23T20:21:54.859Z",
                    "updated_at": "2016-03-23T20:21:54.859Z",
                    "given_name": "Amelia",
                    "family_name": "Earhart",
                    "email_address": "Amelia.Earhart@example.com",
                    "address": {
                        "address_line_1": "500 Electric Ave",
                        "address_line_2": "Suite 600",
                        "locality": "New York",
                        "administrative_district_level_1": "NY",
                        "postal_code": "10003",
                        "country": "US"
                    },
                    "phone_number": "+1-212-555-4240",
                    "reference_id": "YOUR_REFERENCE_ID",
                    "note": "a customer",
                    "preferences": {
                        "email_unsubscribed": False
                    },
                    "creation_source": "THIRD_PARTY",
                    "version": 0
                }
            }
        },
        
        # Update Customer Example
        {
            "input": {
                "operation": "update_customer",
                "customer_id": "JDKYHBWT1D4F8MFH63DBMEN8Y4",
                "email_address": "New.Amelia.Earhart@example.com",
                "phone_number": None,  # Set to None to remove the value
                "note": "updated customer note",
                "version": 2
            },
            "output": {
                "customer": {
                    "id": "JDKYHBWT1D4F8MFH63DBMEN8Y4",
                    "created_at": "2016-03-23T20:21:54.859Z",
                    "updated_at": "2016-05-15T20:21:55Z",
                    "given_name": "Amelia",
                    "family_name": "Earhart",
                    "email_address": "New.Amelia.Earhart@example.com",
                    "address": {
                        "address_line_1": "500 Electric Ave",
                        "address_line_2": "Suite 600",
                        "locality": "New York",
                        "administrative_district_level_1": "NY",
                        "postal_code": "10003",
                        "country": "US"
                    },
                    "reference_id": "YOUR_REFERENCE_ID",
                    "note": "updated customer note",
                    "preferences": {
                        "email_unsubscribed": False
                    },
                    "creation_source": "THIRD_PARTY",
                    "version": 3
                }
            }
        },
        
        # List Catalog Example
        {
            "input": {
                "operation": "list_catalog",
                "types": "ITEM,ITEM_VARIATION,CATEGORY"
            },
            "output": {
                "objects": [
                    {
                        "type": "CATEGORY",
                        "id": "5ZYQZZ2IECPVJ2IJ5KQPRDC3",
                        "updated_at": "2017-02-21T14:50:26.495Z",
                        "version": 1487688626495,
                        "is_deleted": False,
                        "present_at_all_locations": True,
                        "category_data": {
                            "name": "Beverages"
                        }
                    },
                    {
                        "type": "ITEM",
                        "id": "R2TA2FOBUGCJZNIWJSOSNAI4",
                        "updated_at": "2021-06-14T15:51:39.021Z",
                        "version": 1623685899021,
                        "is_deleted": False,
                        "present_at_all_locations": True,
                        "item_data": {
                            "name": "Cocoa",
                            "description": "Hot Chocolate",
                            "abbreviation": "Ch",
                            "product_type": "REGULAR"
                        }
                    }
                ]
            }
        },
        
        # Upsert Catalog Object Example
        {
            "input": {
                "operation": "upsert_catalog_object",
                "idempotency_key": "af3d1afc-7212-4300-b463-0bfc5314a5ae",
                "object_type": "ITEM_VARIATION",
                "id": "#Large",
                "present_at_all_locations": True,
                "item_variation_data": {
                    "item_id": "R2TA2FOBUGCJZNIWJSOSNAI4",
                    "name": "Large",
                    "pricing_type": "FIXED_PRICING",
                    "price_money": {
                        "amount": 400,
                        "currency": "USD"
                    },
                    "service_duration": 1800000
                }
            },
            "output": {
                "catalog_object": {
                    "type": "ITEM_VARIATION",
                    "id": "NS77DKEIQ3AEQTCP727DSA7U",
                    "updated_at": "2021-06-14T15:51:39.021Z",
                    "version": 1623685899021,
                    "is_deleted": False,
                    "present_at_all_locations": True,
                    "item_variation_data": {
                        "item_id": "R2TA2FOBUGCJZNIWJSOSNAI4",
                        "name": "Large",
                        "ordinal": 1,
                        "pricing_type": "FIXED_PRICING",
                        "price_money": {
                            "amount": 400,
                            "currency": "USD"
                        },
                        "service_duration": 1800000,
                        "stockable": True
                    }
                },
                "id_mappings": [
                    {
                        "client_object_id": "#Large",
                        "object_id": "NS77DKEIQ3AEQTCP727DSA7U"
                    }
                ]
            }
        },
        
        # Create Variable Duration Booking Example
        {
            "input": {
                "operation": "create_variable_duration_booking",
                "base_item_id": "R2TA2FOBUGCJZNIWJSOSNAI4",
                "customer_id": "8FYVP54K71AGB1JGJN8SC72N24",
                "address": {
                    "address_line_1": "500 Electric Ave",
                    "locality": "New York",
                    "administrative_district_level_1": "NY",
                    "postal_code": "10003",
                    "country": "US"
                },
                "appointment_time": "2023-06-15T14:00:00Z",
                "duration_minutes": 90,
                "price_dollars": 125.50,
                "notes": "Please bring extra cleaning supplies"
            },
            "output": {
                "booking": {
                    "id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
                    "version": 0,
                    "status": "ACCEPTED",
                    "created_at": "2023-05-01T15:47:41Z",
                    "updated_at": "2023-05-01T15:47:41Z",
                    "location_id": "LRBQZJ7CAJBV5",
                    "customer_id": "8FYVP54K71AGB1JGJN8SC72N24",
                    "customer_note": "Please bring extra cleaning supplies",
                    "seller_note": "",
                    "start_at": "2023-06-15T14:00:00Z",
                    "location_type": "CUSTOMER_LOCATION",
                    "appointment_segments": [
                        {
                            "duration_minutes": 90,
                            "service_variation_id": "NS77DKEIQ3AEQTCP727DSA7U",
                            "team_member_id": "TMjY-uYPS-Wb2hDU",
                            "service_variation_version": 1623685899021
                        }
                    ],
                    "address": {
                        "address_line_1": "500 Electric Ave",
                        "locality": "New York",
                        "administrative_district_level_1": "NY",
                        "postal_code": "10003",
                        "country": "US"
                    }
                },
                "custom_variation": {
                    "id": "NS77DKEIQ3AEQTCP727DSA7U",
                    "version": 1623685899021,
                    "price_amount": 12550,
                    "price_currency": "USD",
                    "duration_minutes": 90
                }
            }
        }
    ]

    def __init__(self):
        """Initialize the Square API tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._session = None
        self.api_url = "https://connect.squareup.com/v2"
        
        # Get configuration
        config_class = registry.get("squareapi_tool")
        self.config = config_class() if config_class else None
        if not self.config:
            self.logger.error("No configuration found for squareapi_tool")
            
        # Get token from environment variables
        self.token = os.getenv("SQUARE_TOKEN")
        if not self.token:
            self.logger.error("SQUARE_TOKEN not found in environment variables")
            raise ToolError(
                "Square API token not configured. Please set SQUARE_TOKEN environment variable.",
                ErrorCode.TOOL_INITIALIZATION_ERROR
            )
        else:
            # Only log that token was found, never log the actual token
            self.logger.info("Square API token found in environment variables")

    def _get_session(self):
        """
        Get or create a requests Session object for connection pooling.
        
        Returns:
            requests.Session: A session object with default headers set
        """
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            })
            # Log that session was created but don't include the token
            self.logger.debug("Created Square API session with authentication headers")
        return self._session
    
    def _validate_duration_minutes(self, duration_minutes: int) -> None:
        """
        Validate that duration is within Square API limits.
        
        Args:
            duration_minutes: Duration in minutes to validate
            
        Raises:
            ToolError: If duration exceeds the maximum allowed (1500 minutes)
        """
        if duration_minutes > 1500:
            raise ToolError(
                "duration_minutes cannot exceed 1500",
                ErrorCode.TOOL_INVALID_INPUT
            )
    
    def _validate_rfc3339_date_format(self, date_str: str, param_name: str) -> None:
        """
        Validate that a string is in RFC 3339 format (YYYY-MM-DDTHH:MM:SSZ).
        
        Args:
            date_str: The date string to validate
            param_name: Name of the parameter for error messages
            
        Raises:
            ToolError: If the date format is invalid
        """
        try:
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            raise ToolError(
                f"Invalid {param_name} format. Please use RFC 3339 format (YYYY-MM-DDTHH:MM:SSZ)",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
        """
        Make an HTTP request to the Square API with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data for POST/PUT requests
            params: Query parameters for GET requests

        Returns:
            Dict containing the API response

        Raises:
            ToolError: If the request fails or returns an error after all retries
        """
        # Set defaults if no config is available
        timeout = getattr(self.config, "timeout", 60)
        max_retries = getattr(self.config, "max_retries", 3)
        backoff_factor = getattr(self.config, "backoff_factor", 2.0)
        
        url = f"{self.api_url}/{endpoint}"
        session = self._get_session()
        
        # Initialize variables for retry logic
        retries = 0
        last_exception = None
        
        while retries <= max_retries:
            try:
                # If this is a retry, log it and wait with exponential backoff
                if retries > 0:
                    wait_time = backoff_factor ** (retries - 1)
                    self.logger.info(f"Retry {retries}/{max_retries} for {method} request to {url} after {wait_time:.1f}s")
                    time.sleep(wait_time)
                
                self.logger.debug(f"Making {method} request to {url}")
                
                # Execute the appropriate HTTP method
                if method.upper() == "GET":
                    response = session.get(url, params=params, timeout=timeout)
                elif method.upper() == "POST":
                    response = session.post(url, json=data, timeout=timeout)
                elif method.upper() == "PUT":
                    response = session.put(url, json=data, timeout=timeout)
                elif method.upper() == "DELETE":
                    response = session.delete(url, timeout=timeout)
                else:
                    raise ToolError(f"Unsupported HTTP method: {method}", ErrorCode.TOOL_INVALID_INPUT)
                
                # Handle response - if we get here without exception, break the retry loop
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                last_exception = e
                
                # Determine if we should retry based on the error type
                should_retry = False
                error_code = ErrorCode.TOOL_EXECUTION_ERROR
                error_message = f"Square API request failed: {str(e)}"
                
                # Check if we have a response to parse
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        # Only retry on server errors (5xx) or rate limiting (429)
                        status_code = e.response.status_code
                        should_retry = (500 <= status_code < 600) or status_code == 429
                        
                        # Try to parse Square API error information
                        error_data = e.response.json()
                        if 'errors' in error_data and error_data['errors']:
                            error_details = error_data['errors'][0]
                            category = error_details.get('category', 'UNKNOWN')
                            detail = error_details.get('detail', 'No detail provided')
                            error_message = f"Square API error: {category}: {detail}"
                            
                            # Map Square error categories to our error codes
                            if category == "AUTHENTICATION_ERROR":
                                error_code = ErrorCode.API_AUTHENTICATION_ERROR
                                should_retry = False  # Don't retry auth errors
                            elif category == "RATE_LIMIT_ERROR":
                                error_code = ErrorCode.API_RATE_LIMIT_ERROR
                                should_retry = True   # Always retry rate limit errors
                            elif category == "INVALID_REQUEST_ERROR":
                                error_code = ErrorCode.TOOL_INVALID_INPUT
                                should_retry = False  # Don't retry invalid input
                    except Exception as parse_error:
                        self.logger.warning(f"Failed to parse error response: {parse_error}")
                
                # Also retry on connection and timeout errors
                if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                    should_retry = True
                
                # If we've reached max retries or shouldn't retry, raise the error
                if retries >= max_retries or not should_retry:
                    self.logger.error(f"{error_message} after {retries} retries")
                    raise ToolError(error_message, error_code)
                    
                # Otherwise, log the error and continue to the next retry
                self.logger.warning(f"{error_message} (attempt {retries+1}/{max_retries+1})")
                retries += 1
                
        # If we somehow exit the loop without returning or raising, raise with the last exception
        # This shouldn't happen due to the logic above, but is included as a safeguard
        if last_exception:
            self.logger.error(f"Request failed after {max_retries} retries")
            raise ToolError(f"Square API request failed after {max_retries} retries", ErrorCode.TOOL_EXECUTION_ERROR)

    # ==========================================================================
    # Bookings API Endpoints
    # ==========================================================================
    def _create_booking(self, 
                      start_at: str,
                      location_id: Optional[str] = None,
                      location_type: Optional[str] = None,
                      customer_id: Optional[str] = None,
                      customer_note: Optional[str] = None,
                      seller_note: Optional[str] = None,
                      service_variation_id: Optional[str] = None,
                      service_variation_version: Optional[int] = None,
                      team_member_id: Optional[str] = None,
                      duration_minutes: Optional[int] = None,
                      idempotency_key: Optional[str] = None,
                      address: Optional[Dict[str, str]] = None) -> Dict:
        """
        Create a new booking in Square.

        Args:
            start_at: RFC 3339 timestamp for when the booking starts (YYYY-MM-DDTHH:MM:SSZ)
            location_id: ID of the business location (default: LRBQZJ7CAJBV5)
            location_type: Type of location (BUSINESS_LOCATION, CUSTOMER_LOCATION, PHONE)
            customer_id: ID of the customer making the booking
            customer_note: Additional notes from the customer (max 4096 characters)
            seller_note: Notes from the seller (max 4096 characters, not visible to customers)
            service_variation_id: ID of the service variation being booked
            service_variation_version: Current version of the service variation
            team_member_id: ID of the team member providing the service (default: TMjY-uYPS-Wb2hDU)
            duration_minutes: Duration of the appointment in minutes (max 1500)
            idempotency_key: A unique key to make this request idempotent
            address: Customer address if location_type is CUSTOMER_LOCATION

        Returns:
            Dict containing the created booking details
            
        Required parameters:
            - start_at
            - service_variation_id (for appointment segments)
            - team_member_id (for appointment segments)
            - location_id
        """
        # Apply defaults from configuration if parameters are None
        if location_id is None and self.config:
            location_id = self.config.default_location_id
        
        if location_type is None and self.config:
            location_type = self.config.default_location_type
            
        if team_member_id is None and self.config:
            team_member_id = self.config.default_team_member_id
        
        # Validate required parameters
        if not start_at:
            raise ToolError(
                "start_at is required for creating a booking",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        if not service_variation_id:
            raise ToolError(
                "service_variation_id is required for creating a booking",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        if not location_id:
            raise ToolError(
                "location_id is required for creating a booking and no default is configured",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        if not team_member_id:
            raise ToolError(
                "team_member_id is required for creating a booking and no default is configured",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Validate start_at format
        self._validate_rfc3339_date_format(start_at, "start_at")

        # Check if location_type is valid
        valid_location_types = ["BUSINESS_LOCATION", "CUSTOMER_LOCATION", "PHONE"]
        if location_type and location_type not in valid_location_types:
            raise ToolError(
                f"Invalid location_type. Must be one of: {', '.join(valid_location_types)}",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # If location_type is CUSTOMER_LOCATION, address should be provided
        if location_type == "CUSTOMER_LOCATION" and not address:
            self.logger.warning("location_type is CUSTOMER_LOCATION but no address was provided")

        # Build the request body
        appointment_segment = {
            "team_member_id": team_member_id,
            "service_variation_id": service_variation_id,
        }
        
        if duration_minutes is not None:
            self._validate_duration_minutes(duration_minutes)
            appointment_segment["duration_minutes"] = duration_minutes
            
        if service_variation_version is not None:
            appointment_segment["service_variation_version"] = service_variation_version

        booking_data = {
            "booking": {
                "start_at": start_at,
                "location_id": location_id,
                "location_type": location_type
            }
        }
        
        # Add optional fields
        if customer_id:
            booking_data["booking"]["customer_id"] = customer_id
            
        if customer_note:
            booking_data["booking"]["customer_note"] = customer_note
            
        if seller_note:
            booking_data["booking"]["seller_note"] = seller_note
            
        if address and location_type == "CUSTOMER_LOCATION":
            booking_data["booking"]["address"] = address
        
        # Add appointment segments
        booking_data["booking"]["appointment_segments"] = [appointment_segment]
        
        # Add idempotency key if provided
        if idempotency_key:
            booking_data["idempotency_key"] = idempotency_key
        
        # Remove None values
        booking_data = self._remove_none_values(booking_data)
        
        # Make the API request
        return self._make_request("POST", "bookings", data=booking_data)

    def _list_bookings(self,
                      location_id: Optional[str] = None,
                      start_at_min: Optional[str] = None,
                      start_at_max: Optional[str] = None,
                      team_member_id: Optional[str] = None,
                      customer_id: Optional[str] = None,
                      limit: int = 100,
                      cursor: Optional[str] = None) -> Dict:
        """
        List bookings with optional filters.

        Args:
            location_id: ID of the business location (default: LRBQZJ7CAJBV5)
            start_at_min: Minimum start time (RFC 3339 format, YYYY-MM-DDTHH:MM:SSZ)
            start_at_max: Maximum start time (RFC 3339 format, YYYY-MM-DDTHH:MM:SSZ)
            team_member_id: Filter by team member ID
            customer_id: Filter by customer ID
            limit: Maximum number of results to return (up to 100)
            cursor: Pagination cursor from a previous response

        Returns:
            Dict containing the list of bookings and a pagination cursor
        """
        # Apply defaults from configuration if parameters are None
        if location_id is None and self.config:
            location_id = self.config.default_location_id
            
        # Validate date formats if provided
        if start_at_min:
            self._validate_rfc3339_date_format(start_at_min, "start_at_min")
        if start_at_max:
            self._validate_rfc3339_date_format(start_at_max, "start_at_max")
        
        # Validate limit
        if limit <= 0 or limit > 100:
            raise ToolError(
                "limit must be between 1 and 100",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Validate location_id
        if not location_id:
            raise ToolError(
                "location_id is required for listing bookings and no default is configured",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Build query parameters
        params = {
            "limit": limit,
            "location_id": location_id
        }
        
        # Add optional parameters if they exist
        if start_at_min:
            params["start_at_min"] = start_at_min
            
        if start_at_max:
            params["start_at_max"] = start_at_max
            
        if team_member_id:
            params["team_member_id"] = team_member_id
            
        if customer_id:
            params["customer_id"] = customer_id
            
        if cursor:
            params["cursor"] = cursor
        
        # Make the API request
        return self._make_request("GET", "bookings", params=params)

    def _retrieve_booking(self, booking_id: str) -> Dict:
        """
        Retrieve a specific booking by ID.

        Args:
            booking_id: The ID of the booking to retrieve (required)

        Returns:
            Dict containing the booking details including:
            - ID, version number, and status
            - Location, customer, and timing information
            - Notes and appointment segments
        """
        if not booking_id:
            raise ToolError(
                "booking_id is required for retrieving a booking",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        return self._make_request("GET", f"bookings/{booking_id}")

    def _update_booking(self,
                      booking_id: str,
                      version: int,
                      start_at: Optional[str] = None,
                      location_type: Optional[str] = None,
                      customer_id: Optional[str] = None,
                      customer_note: Optional[str] = None,
                      seller_note: Optional[str] = None,
                      service_variation_id: Optional[str] = None,
                      service_variation_version: Optional[int] = None,
                      team_member_id: Optional[str] = None, 
                      duration_minutes: Optional[int] = None,
                      idempotency_key: Optional[str] = None,
                      address: Optional[Dict[str, str]] = None) -> Dict:
        """
        Update an existing booking.

        Args:
            booking_id: ID of the booking to update (required)
            version: Revision number for optimistic concurrency (required)
            start_at: New RFC 3339 timestamp for when the booking starts
            location_type: Updated location type (BUSINESS_LOCATION, CUSTOMER_LOCATION, PHONE)
            customer_id: Updated customer ID
            customer_note: Updated notes from the customer
            seller_note: Updated notes from the seller (not visible to customers)
            service_variation_id: Updated service variation ID
            service_variation_version: Updated service variation version
            team_member_id: Updated team member ID
            duration_minutes: Updated duration in minutes (max 1500)
            idempotency_key: A unique key to make this request idempotent
            address: Updated customer address (if location_type is CUSTOMER_LOCATION)

        Returns:
            Dict containing the updated booking details
            
        Note:
            The booking version must be provided for optimistic concurrency control.
            Only include fields you want to update. Omitted fields remain unchanged.
        """
        # Validate required parameters
        if not booking_id:
            raise ToolError(
                "booking_id is required for updating a booking",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Validate start_at format
        if start_at:
            self._validate_rfc3339_date_format(start_at, "start_at")

        # Check if location_type is valid
        valid_location_types = ["BUSINESS_LOCATION", "CUSTOMER_LOCATION", "PHONE"]
        if location_type and location_type not in valid_location_types:
            raise ToolError(
                f"Invalid location_type. Must be one of: {', '.join(valid_location_types)}",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # If location_type is CUSTOMER_LOCATION, address should be provided
        if location_type == "CUSTOMER_LOCATION" and not address:
            self.logger.warning("location_type is CUSTOMER_LOCATION but no address was provided")

        # Build the request body
        booking_data = {
            "booking": {
                "version": version
            }
        }
        
        # Add optional booking fields if they exist
        if start_at:
            booking_data["booking"]["start_at"] = start_at
            
        if customer_id:
            booking_data["booking"]["customer_id"] = customer_id
            
        if customer_note is not None:  # Allow empty string
            booking_data["booking"]["customer_note"] = customer_note
            
        if seller_note is not None:  # Allow empty string
            booking_data["booking"]["seller_note"] = seller_note
            
        if location_type:
            booking_data["booking"]["location_type"] = location_type
            
        if address and location_type == "CUSTOMER_LOCATION":
            booking_data["booking"]["address"] = address
        
        # Only add appointment segments if at least one segment parameter exists
        segment_params = {
            "team_member_id": team_member_id,
            "service_variation_id": service_variation_id,
            "service_variation_version": service_variation_version,
            "duration_minutes": duration_minutes
        }
        
        # Filter out None values
        segment_params = {k: v for k, v in segment_params.items() if v is not None}
        
        if segment_params:
            booking_data["booking"]["appointment_segments"] = [segment_params]
            
        # Validate duration_minutes
        if duration_minutes is not None:
            self._validate_duration_minutes(duration_minutes)
            
        # Add idempotency key if provided
        if idempotency_key:
            booking_data["idempotency_key"] = idempotency_key
        
        # Remove None values
        booking_data = self._remove_none_values(booking_data)
        
        # Make the API request
        return self._make_request("PUT", f"bookings/{booking_id}", data=booking_data)

    def _cancel_booking(self, 
                      booking_id: str, 
                      booking_version: Optional[int] = None,
                      idempotency_key: Optional[str] = None) -> Dict:
        """
        Cancel a booking.

        Args:
            booking_id: ID of the booking to cancel (required)
            booking_version: Current version number of the booking for optimistic concurrency
            idempotency_key: A unique key to make this request idempotent

        Returns:
            Dict containing the cancelled booking with updated status
            
        Note:
            Providing booking_version is recommended for optimistic concurrency control.
            The status in response will be CANCELLED_BY_SELLER.
        """
        if not booking_id:
            raise ToolError(
                "booking_id is required for cancelling a booking",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        data = {}
        
        # Add booking version if provided
        if booking_version is not None:
            data["booking_version"] = booking_version
            
        # Add idempotency key if provided
        if idempotency_key:
            data["idempotency_key"] = idempotency_key
        
        return self._make_request("POST", f"bookings/{booking_id}/cancel", data=data)

    # ==========================================================================
    # Customers API Endpoints
    # ==========================================================================
    def _list_customers(self,
                      cursor: Optional[str] = None,
                      limit: int = 100,
                      sort_field: str = "DEFAULT",
                      sort_order: str = "ASC",
                      count: bool = False) -> Dict:
        """
        List customer profiles associated with a Square account.

        Args:
            cursor: A pagination cursor from a previous response to retrieve the next set of results
            limit: Maximum number of results to return (1-100, default 100)
            sort_field: How to sort customers (DEFAULT or CREATED_AT)
            sort_order: Sort order (ASC or DESC)
            count: Whether to return the total count of customers

        Returns:
            Dict containing the list of customers, optional count, and pagination cursor
            
        Note:
            Under normal conditions, newly created or updated customer profiles are available 
            in well under 30 seconds. Propagation can occasionally take longer during network incidents.
        """
        # Validate parameters
        if limit <= 0 or limit > 100:
            raise ToolError(
                "limit must be between 1 and 100",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        valid_sort_fields = ["DEFAULT", "CREATED_AT"]
        if sort_field not in valid_sort_fields:
            raise ToolError(
                f"Invalid sort_field. Must be one of: {', '.join(valid_sort_fields)}",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        valid_sort_orders = ["ASC", "DESC"]
        if sort_order not in valid_sort_orders:
            raise ToolError(
                f"Invalid sort_order. Must be one of: {', '.join(valid_sort_orders)}",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Build query parameters
        params = {
            "limit": limit,
            "sort_field": sort_field,
            "sort_order": sort_order,
            "count": str(count).lower()  # Convert to lowercase 'true' or 'false'
        }
        
        if cursor:
            params["cursor"] = cursor
        
        # Make the API request
        return self._make_request("GET", "customers", params=params)
        
    def _create_customer(self,
                       given_name: Optional[str] = None,
                       family_name: Optional[str] = None,
                       company_name: Optional[str] = None,
                       nickname: Optional[str] = None,
                       email_address: Optional[str] = None,
                       address: Optional[Dict[str, str]] = None,
                       phone_number: Optional[str] = None,
                       reference_id: Optional[str] = None,
                       note: Optional[str] = None,
                       birthday: Optional[str] = None,
                       tax_ids: Optional[Dict[str, str]] = None,
                       idempotency_key: Optional[str] = None) -> Dict:
        """
        Create a new customer profile.

        Args:
            given_name: First name (maximum length: 300 characters)
            family_name: Last name (maximum length: 300 characters)
            company_name: Business name (maximum length: 500 characters)
            nickname: Nickname (maximum length: 100 characters)
            email_address: Email address (maximum length: 254 characters)
            address: Physical address associated with the customer profile
            phone_number: Phone number (must be valid with optional + prefix and country code)
            reference_id: Second ID for association with external system (maximum length: 100 characters)
            note: Custom note associated with the customer profile
            birthday: Birthday in YYYY-MM-DD or MM-DD format
            tax_ids: Tax ID associated with the customer profile
            idempotency_key: A unique key to make this request idempotent

        Returns:
            Dict containing the created customer profile
            
        Note:
            At least one of the following fields must be provided: given_name, family_name, 
            company_name, email_address, or phone_number.
        """
        # Validate required parameters
        required_fields = [given_name, family_name, company_name, email_address, phone_number]
        if not any(required_fields):
            raise ToolError(
                "At least one of the following fields must be provided: given_name, family_name, "  
                "company_name, email_address, or phone_number",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Validate field lengths
        if given_name and len(given_name) > 300:
            raise ToolError("given_name cannot exceed 300 characters", ErrorCode.TOOL_INVALID_INPUT)
            
        if family_name and len(family_name) > 300:
            raise ToolError("family_name cannot exceed 300 characters", ErrorCode.TOOL_INVALID_INPUT)
            
        if company_name and len(company_name) > 500:
            raise ToolError("company_name cannot exceed 500 characters", ErrorCode.TOOL_INVALID_INPUT)
            
        if nickname and len(nickname) > 100:
            raise ToolError("nickname cannot exceed 100 characters", ErrorCode.TOOL_INVALID_INPUT)
            
        if email_address and len(email_address) > 254:
            raise ToolError("email_address cannot exceed 254 characters", ErrorCode.TOOL_INVALID_INPUT)
            
        if reference_id and len(reference_id) > 100:
            raise ToolError("reference_id cannot exceed 100 characters", ErrorCode.TOOL_INVALID_INPUT)
            
        # Validate birthday format
        if birthday:
            birthday_pattern = r'^(?:\d{4}-\d{2}-\d{2}|\d{2}-\d{2})$'
            if not re.match(birthday_pattern, birthday):
                raise ToolError(
                    "birthday must be in YYYY-MM-DD or MM-DD format",
                    ErrorCode.TOOL_INVALID_INPUT
                )
        
        # Build the request body
        customer_data = {}
        
        # Add all fields (will remove None values later)
        fields = {
            "given_name": given_name,
            "family_name": family_name,
            "company_name": company_name,
            "nickname": nickname,
            "email_address": email_address,
            "address": address,
            "phone_number": phone_number,
            "reference_id": reference_id,
            "note": note,
            "birthday": birthday,
            "tax_ids": tax_ids
        }
        
        # Filter out None values
        customer_data = {k: v for k, v in fields.items() if v is not None}
        
        # Add idempotency key if provided
        request_data = {}
        if idempotency_key:
            request_data["idempotency_key"] = idempotency_key
            
        request_data.update(customer_data)
        
        # Make the API request
        return self._make_request("POST", "customers", data=request_data)
        
    def _update_customer(self,
                       customer_id: str,
                       given_name: Optional[str] = None,
                       family_name: Optional[str] = None,
                       company_name: Optional[str] = None,
                       nickname: Optional[str] = None,
                       email_address: Optional[str] = None,
                       address: Optional[Dict[str, str]] = None,
                       phone_number: Optional[str] = None,
                       reference_id: Optional[str] = None,
                       note: Optional[str] = None,
                       birthday: Optional[str] = None,
                       version: Optional[int] = None,
                       tax_ids: Optional[Dict[str, str]] = None) -> Dict:
        """
        Update a customer profile.

        Args:
            customer_id: ID of the customer to update (required)
            given_name: First name (maximum length: 300 characters)
            family_name: Last name (maximum length: 300 characters)
            company_name: Business name (maximum length: 500 characters)
            nickname: Nickname (maximum length: 100 characters)
            email_address: Email address (maximum length: 254 characters)
            address: Physical address associated with the customer profile
            phone_number: Phone number (must be valid with optional + prefix and country code)
            reference_id: Second ID for association with external system (maximum length: 100 characters)
            note: Custom note associated with the customer profile
            birthday: Birthday in YYYY-MM-DD or MM-DD format
            version: Current version of the customer profile (for optimistic concurrency)
            tax_ids: Tax ID associated with the customer profile

        Returns:
            Dict containing the updated customer profile
            
        Note:
            This endpoint supports sparse updates, so only include fields you want to update.
            To remove a field, explicitly set it to null.
            For optimistic concurrency control, include the version number.
        """
        # Validate required parameters
        if not customer_id:
            raise ToolError(
                "customer_id is required for updating a customer",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Validate field lengths
        if given_name is not None and len(given_name) > 300:
            raise ToolError("given_name cannot exceed 300 characters", ErrorCode.TOOL_INVALID_INPUT)
            
        if family_name is not None and len(family_name) > 300:
            raise ToolError("family_name cannot exceed 300 characters", ErrorCode.TOOL_INVALID_INPUT)
            
        if company_name is not None and len(company_name) > 500:
            raise ToolError("company_name cannot exceed 500 characters", ErrorCode.TOOL_INVALID_INPUT)
            
        if nickname is not None and len(nickname) > 100:
            raise ToolError("nickname cannot exceed 100 characters", ErrorCode.TOOL_INVALID_INPUT)
            
        if email_address is not None and len(email_address) > 254:
            raise ToolError("email_address cannot exceed 254 characters", ErrorCode.TOOL_INVALID_INPUT)
            
        if reference_id is not None and len(reference_id) > 100:
            raise ToolError("reference_id cannot exceed 100 characters", ErrorCode.TOOL_INVALID_INPUT)
            
        # Validate birthday format
        if birthday is not None and birthday != "":
            birthday_pattern = r'^(?:\d{4}-\d{2}-\d{2}|\d{2}-\d{2})$'
            if not re.match(birthday_pattern, birthday):
                raise ToolError(
                    "birthday must be in YYYY-MM-DD or MM-DD format",
                    ErrorCode.TOOL_INVALID_INPUT
                )
        
        # Build the update request body - include fields even if None
        # In update context, None values mean to remove the field
        fields = {
            "given_name": given_name,
            "family_name": family_name,
            "company_name": company_name,
            "nickname": nickname,
            "email_address": email_address,
            "address": address,
            "phone_number": phone_number,
            "reference_id": reference_id,
            "note": note,
            "birthday": birthday,
            "tax_ids": tax_ids
        }
        
        # Get all local variables once to avoid multiple lookups
        local_vars = locals()
        
        # Filter fields to include only those explicitly passed to the function
        customer_data = {k: v for k, v in fields.items() if k in local_vars and local_vars[k] is not local_vars['self']}
        
        # Add version if provided
        if version is not None:
            customer_data["version"] = version
        
        # Make the API request
        return self._make_request("PUT", f"customers/{customer_id}", data=customer_data)

    # ==========================================================================
    # Catalog API Endpoints  
    # ==========================================================================
    def _list_catalog(self,
                    cursor: Optional[str] = None,
                    types: Optional[str] = None,
                    catalog_version: Optional[int] = None) -> Dict:
        """
        List catalog objects of the specified types.
        
        Args:
            cursor: Pagination cursor from a previous response for fetching the next set of results
            types: Comma-separated list of object types to retrieve (e.g., "ITEM,ITEM_VARIATION,CATEGORY")
                   If unspecified, returns objects of all top-level types
            catalog_version: Specific version of catalog objects to be included in the response
                            Allows retrieving historical versions of objects
            
        Returns:
            Dict containing the list of catalog objects and a pagination cursor
            
        Note:
            The valid types include: ITEM, ITEM_VARIATION, CATEGORY, DISCOUNT, TAX, MODIFIER, MODIFIER_LIST, 
            IMAGE, etc. If no types are specified, the operation returns all top-level types by default.
            This endpoint does not return deleted catalog items. To retrieve deleted catalog items, use
            the SearchCatalogObjects endpoint and set include_deleted_objects to true.
        """
        # Build query parameters
        params = {}
        
        if cursor:
            params["cursor"] = cursor
            
        if types:
            params["types"] = types
            
        if catalog_version is not None:
            params["catalog_version"] = catalog_version
        
        # Make the API request
        return self._make_request("GET", "catalog/list", params=params)
        
    def _upsert_catalog_object(self,
                             idempotency_key: str,
                             object_type: str,
                             id: Optional[str] = None,
                             present_at_all_locations: bool = True,
                             present_at_location_ids: Optional[List[str]] = None,
                             absent_at_location_ids: Optional[List[str]] = None,
                             item_data: Optional[Dict] = None,
                             category_data: Optional[Dict] = None,
                             item_variation_data: Optional[Dict] = None,
                             tax_data: Optional[Dict] = None,
                             discount_data: Optional[Dict] = None,
                             modifier_list_data: Optional[Dict] = None,
                             modifier_data: Optional[Dict] = None,
                             image_data: Optional[Dict] = None,
                             custom_attribute_values: Optional[Dict] = None,
                             quick_amounts_settings_data: Optional[Dict] = None) -> Dict:
        """
        Creates a new or updates an existing CatalogObject.
        
        Args:
            idempotency_key: A value that uniquely identifies this request (required)
            object_type: The type of this object (e.g., ITEM, ITEM_VARIATION, CATEGORY) (required)
            id: Identifier for the object (for creates, use a temporary identifier starting with #)
            present_at_all_locations: Whether this object is present at all locations (default: True)
            present_at_location_ids: List of locations where the object is present (even if present_at_all_locations is False)
            absent_at_location_ids: List of locations where the object is not present (even if present_at_all_locations is True)
            
            # Type-specific structured data fields (provide only the one that corresponds to object_type):
            item_data: Structured data for a CatalogItem (when type is ITEM)
            category_data: Structured data for CatalogCategory (when type is CATEGORY)
            item_variation_data: Structured data for CatalogItemVariation (when type is ITEM_VARIATION)
            tax_data: Structured data for CatalogTax (when type is TAX)
            discount_data: Structured data for CatalogDiscount (when type is DISCOUNT)
            modifier_list_data: Structured data for CatalogModifierList (when type is MODIFIER_LIST)
            modifier_data: Structured data for CatalogModifier (when type is MODIFIER)
            image_data: Structured data for CatalogImage (when type is IMAGE)
            custom_attribute_values: A map of custom attribute values
            quick_amounts_settings_data: Structured data for CatalogQuickAmountsSettings
        
        Returns:
            Dict containing the created or updated CatalogObject and id mappings
            
        Note:
            This endpoint is primarily used to create item variations that modify the price
            and duration of existing service items. For creates, the object ID must start
            with '#'. The provided ID is replaced with a server-generated ID.
            
            Only one update request is processed at a time per seller account for
            consistency. While one update request is being processed, others are rejected
            with a 429 error code.
        """
        # Validate required parameters
        if not idempotency_key:
            raise ToolError(
                "idempotency_key is required for upserting a catalog object",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        if not object_type:
            raise ToolError(
                "object_type is required for upserting a catalog object",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Validate object_type
        valid_types = ["ITEM", "ITEM_VARIATION", "CATEGORY", "TAX", "DISCOUNT", 
                      "MODIFIER_LIST", "MODIFIER", "TIME_PERIOD", "PRODUCT_SET", 
                      "PRICING_RULE", "IMAGE", "MEASUREMENT_UNIT", "SUBSCRIPTION_PLAN", 
                      "ITEM_OPTION", "ITEM_OPTION_VAL", "CUSTOM_ATTRIBUTE_DEFINITION", 
                      "QUICK_AMOUNTS_SETTINGS"]
                      
        if object_type not in valid_types:
            raise ToolError(
                f"Invalid object_type. Must be one of: {', '.join(valid_types)}",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Verify id starts with # if provided for a new object
        if id and not id.startswith("#") and not id.startswith("C"):
            self.logger.warning("For new catalog objects, id should start with '#'")
        
        # Validate idempotency_key length
        if len(idempotency_key) < 1 or len(idempotency_key) > 128:
            raise ToolError(
                "idempotency_key must be between 1 and 128 characters",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Build the catalog object based on the object_type
        catalog_object = {
            "type": object_type
        }
        
        if id:
            catalog_object["id"] = id
            
        if present_at_all_locations is not None:
            catalog_object["present_at_all_locations"] = present_at_all_locations
            
        if present_at_location_ids:
            catalog_object["present_at_location_ids"] = present_at_location_ids
            
        if absent_at_location_ids:
            catalog_object["absent_at_location_ids"] = absent_at_location_ids
        
        # Add the appropriate data field based on the object_type
        type_data_map = {
            "ITEM": ("item_data", item_data),
            "CATEGORY": ("category_data", category_data),
            "ITEM_VARIATION": ("item_variation_data", item_variation_data),
            "TAX": ("tax_data", tax_data),
            "DISCOUNT": ("discount_data", discount_data),
            "MODIFIER_LIST": ("modifier_list_data", modifier_list_data),
            "MODIFIER": ("modifier_data", modifier_data),
            "IMAGE": ("image_data", image_data),
            "QUICK_AMOUNTS_SETTINGS": ("quick_amounts_settings_data", quick_amounts_settings_data)
        }
        
        # Add the appropriate data field if it's provided
        if object_type in type_data_map:
            field_name, field_data = type_data_map[object_type]
            if field_data:
                catalog_object[field_name] = field_data
        
        # Add custom attribute values if provided
        if custom_attribute_values:
            catalog_object["custom_attribute_values"] = custom_attribute_values
        
        # Prepare request data
        request_data = {
            "idempotency_key": idempotency_key,
            "object": catalog_object
        }
        
        # Make the API request
        return self._make_request("POST", "catalog/object", data=request_data)

    # ==========================================================================
    # Convenience Methods
    # ==========================================================================
    def _create_variable_duration_booking(self,
                                       base_item_id: str,
                                       customer_id: str,
                                       address: Dict[str, str],
                                       appointment_time: str,
                                       duration_minutes: int,
                                       price_dollars: float,
                                       notes: str = "") -> Dict:
        """
        Create a booking with a custom price and duration for a service.
        
        This is a convenience method that performs two operations:
        1. Creates a custom service variation with the specified price and duration
        2. Creates a booking with the new custom service variation
        
        Args:
            base_item_id: ID of the base service item to create a variation for
            customer_id: ID of the customer to book the appointment for
            address: Customer address for service location (must include address_line_1, locality, etc.)
            appointment_time: RFC 3339 timestamp for when the booking starts (YYYY-MM-DDTHH:MM:SSZ)
            duration_minutes: Custom duration in minutes for the service (max 1500)
            price_dollars: Custom price in dollars for the service (will be converted to cents)
            notes: Optional notes from the customer
            
        Returns:
            Dict containing the completed booking details
            
        Note:
            This method creates a new catalog item variation for each booking.
            Each variation will have a unique name based on the duration.
        """
        
        # Input validation
        if not base_item_id:
            raise ToolError(
                "base_item_id is required for creating a variable duration booking",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        if not customer_id:
            raise ToolError(
                "customer_id is required for creating a variable duration booking",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        if not address:
            raise ToolError(
                "address is required for creating a variable duration booking",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        if not appointment_time:
            raise ToolError(
                "appointment_time is required for creating a variable duration booking",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Validate appointment_time format
        self._validate_rfc3339_date_format(appointment_time, "appointment_time")
            
        if duration_minutes <= 0:
            raise ToolError(
                "duration_minutes must be greater than 0",
                ErrorCode.TOOL_INVALID_INPUT
            )
        self._validate_duration_minutes(duration_minutes)
            
        if price_dollars <= 0:
            raise ToolError(
                "price_dollars must be greater than 0",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # 1. Create a service variation with custom duration and price
        price_cents = int(price_dollars * 100)  # Convert dollars to cents
        
        # Create unique idempotency key and variation ID using customer_id and timestamp
        var_idempotency_key = str(uuid.uuid4())
        var_id = f"#CustomVar-{int(time.time())}"
        
        self.logger.info(f"Creating service variation with duration={duration_minutes}min, price=${price_dollars}")
        
        try:
            variation_result = self._upsert_catalog_object(
                idempotency_key=var_idempotency_key,
                object_type="ITEM_VARIATION",
                id=var_id,
                present_at_all_locations=True,
                item_variation_data={
                    "item_id": base_item_id,
                    "name": f"Custom Service ({duration_minutes} min)",
                    "pricing_type": "FIXED_PRICING",
                    "price_money": {
                        "amount": price_cents,
                        "currency": "USD"
                    },
                    "service_duration": duration_minutes * 60 * 1000  # Convert to milliseconds
                }
            )
            
            service_variation_id = variation_result["catalog_object"]["id"]
            service_variation_version = variation_result["catalog_object"]["version"]
            
            self.logger.info(f"Created service variation with ID: {service_variation_id}")
            
            # 2. Create the booking with the new custom service variation
            booking_idempotency_key = str(uuid.uuid4())
            
            booking_result = self._create_booking(
                start_at=appointment_time,
                customer_id=customer_id,
                location_type="CUSTOMER_LOCATION",
                address=address,
                customer_note=notes,
                service_variation_id=service_variation_id,
                service_variation_version=service_variation_version,
                duration_minutes=duration_minutes,
                idempotency_key=booking_idempotency_key
            )
            
            self.logger.info(f"Created booking with ID: {booking_result.get('booking', {}).get('id')}")
            
            # Include both the booking and the variation in the result
            result = booking_result
            result["custom_variation"] = {
                "id": service_variation_id,
                "version": service_variation_version,
                "price_amount": price_cents,
                "price_currency": "USD",
                "duration_minutes": duration_minutes
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating variable duration booking: {str(e)}")
            raise ToolError(
                f"Failed to create variable duration booking: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    # ==========================================================================
    # Helper Methods
    # ==========================================================================
    def _remove_none_values(self, data: Dict) -> Dict:
        """
        Recursively remove None values from dictionaries.

        Args:
            data: The dictionary to clean

        Returns:
            Dict with None values removed
        """
        if not isinstance(data, dict):
            return data
            
        result = {}
        for key, value in data.items():
            if value is None:
                continue
                
            if isinstance(value, dict):
                nested_result = self._remove_none_values(value)
                if nested_result:  # Only add non-empty dicts
                    result[key] = nested_result
            elif isinstance(value, list):
                nested_result = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_item = self._remove_none_values(item)
                        if cleaned_item:  # Only add non-empty dicts
                            nested_result.append(cleaned_item)
                    else:
                        nested_result.append(item)
                if nested_result:  # Only add non-empty lists
                    result[key] = nested_result
            else:
                result[key] = value
                
        return result

    def run(self,
            operation: str,
            **kwargs) -> Dict[str, Any]:
        """
        Execute a Square API operation.

        Args:
            operation: Operation to perform (see below for valid operations)
            **kwargs: Operation-specific parameters

        Returns:
            Response data for the operation

        Raises:
            ToolError: If operation fails or parameters are invalid

        Valid Operations:

        1. create_booking: Create a new booking
           - Required: start_at (RFC 3339 format), service_variation_id, team_member_id
           - Optional: location_id, location_type, customer_id, customer_note, seller_note, 
                       service_variation_version, duration_minutes, idempotency_key, address
           
           Example: 
           {
             "operation": "create_booking",
             "start_at": "2023-05-30T15:00:00Z",
             "service_variation_id": "VBKQNFTZGCKHCH7ZYN3NP4YZ",
             "team_member_id": "TMjY-uYPS-Wb2hDU",
             "customer_id": "8FYVP54K71AGB1JGJN8SC72N24",
             "customer_note": "Please call when you arrive",
             "location_type": "CUSTOMER_LOCATION",
             "address": {
               "address_line_1": "500 Electric Ave",
               "locality": "New York",
               "administrative_district_level_1": "NY",
               "postal_code": "10003",
               "country": "US"
             }
           }

        2. list_bookings: List bookings with optional filters
           - Optional: location_id, start_at_min, start_at_max, team_member_id, customer_id, limit, cursor
           
           Example:
           {
             "operation": "list_bookings",
             "location_id": "LRBQZJ7CAJBV5",
             "start_at_min": "2023-05-01T00:00:00Z",
             "start_at_max": "2023-05-31T23:59:59Z",
             "limit": 50
           }

        3. retrieve_booking: Get a booking by ID
           - Required: booking_id
           
           Example:
           {
             "operation": "retrieve_booking",
             "booking_id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI"
           }

        4. update_booking: Update an existing booking
           - Required: booking_id, version
           - Optional: start_at, location_type, customer_id, customer_note, seller_note,
                       service_variation_id, service_variation_version, team_member_id,
                       duration_minutes, idempotency_key, address
           
           Example:
           {
             "operation": "update_booking",
             "booking_id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
             "version": 1,
             "start_at": "2023-05-30T16:00:00Z", 
             "customer_note": "I'll be 15 minutes late"
           }

        5. cancel_booking: Cancel a booking
           - Required: booking_id
           - Optional: booking_version, idempotency_key
           
           Example:
           {
             "operation": "cancel_booking",
             "booking_id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
             "booking_version": 1
           }
           
        6. list_customers: List customer profiles with optional filters
           - Optional: cursor, limit, sort_field, sort_order, count
           
           Example:
           {
             "operation": "list_customers",
             "limit": 50,
             "sort_field": "CREATED_AT",
             "sort_order": "DESC",
             "count": True
           }

        7. create_customer: Create a new customer profile
           - Required: At least one of: given_name, family_name, company_name, email_address, phone_number
           - Optional: nickname, address, reference_id, note, birthday, tax_ids, idempotency_key
           
           Example:
           {
             "operation": "create_customer",
             "given_name": "Amelia",
             "family_name": "Earhart",
             "email_address": "Amelia.Earhart@example.com",
             "address": {
               "address_line_1": "500 Electric Ave",
               "address_line_2": "Suite 600",
               "locality": "New York",
               "administrative_district_level_1": "NY",
               "postal_code": "10003",
               "country": "US"
             },
             "phone_number": "+1-212-555-4240",
             "reference_id": "YOUR_REFERENCE_ID",
             "note": "a customer"
           }

        8. update_customer: Update an existing customer profile
           - Required: customer_id
           - Optional: given_name, family_name, company_name, nickname, email_address,
                       address, phone_number, reference_id, note, birthday, version, tax_ids
           
           Example:
           {
             "operation": "update_customer",
             "customer_id": "JDKYHBWT1D4F8MFH63DBMEN8Y4",
             "email_address": "New.Amelia.Earhart@example.com",
             "phone_number": null,  # Set to null to remove the value
             "note": "updated customer note",
             "version": 2
           }

        9. list_catalog: List catalog objects of specified types
           - Optional: cursor, types, catalog_version
           
           Example:
           {
             "operation": "list_catalog",
             "types": "ITEM,ITEM_VARIATION,CATEGORY"
           }
           
        10. upsert_catalog_object: Create or update a catalog object
           - Required: idempotency_key, object_type
           - Optional fields depend on object_type (e.g., id, item_data, item_variation_data, etc.)
           
           Example for creating an item variation:
           {
             "operation": "upsert_catalog_object",
             "idempotency_key": "af3d1afc-7212-4300-b463-0bfc5314a5ae",
             "object_type": "ITEM_VARIATION",
             "id": "#Large",
             "present_at_all_locations": True,
             "item_variation_data": {
               "item_id": "R2TA2FOBUGCJZNIWJSOSNAI4",
               "name": "Large",
               "pricing_type": "FIXED_PRICING",
               "price_money": {
                 "amount": 400,
                 "currency": "USD"
               },
               "service_duration": 1800000
             }
           }
           
        11. create_variable_duration_booking: Create a booking with custom price and duration
           - Required: base_item_id, customer_id, address, appointment_time, duration_minutes, price_dollars
           - Optional: notes
           
           Example:
           {
             "operation": "create_variable_duration_booking",
             "base_item_id": "R2TA2FOBUGCJZNIWJSOSNAI4",
             "customer_id": "8FYVP54K71AGB1JGJN8SC72N24",
             "address": {
               "address_line_1": "500 Electric Ave",
               "locality": "New York",
               "administrative_district_level_1": "NY",
               "postal_code": "10003",
               "country": "US"
             },
             "appointment_time": "2023-06-15T14:00:00Z",
             "duration_minutes": 90,
             "price_dollars": 125.50,
             "notes": "Please bring extra cleaning supplies"
           }
        """
        # No need to import config - not used in this method
        with error_context(
            component_name=self.name,
            operation=f"executing {operation}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger,
        ):
            # Booking Operations
            if operation == "create_booking":
                required_params = ["start_at", "service_variation_id"]
                self._validate_params(required_params, kwargs)
                return self._create_booking(**kwargs)
                
            elif operation == "list_bookings":
                return self._list_bookings(**kwargs)
                
            elif operation == "retrieve_booking":
                required_params = ["booking_id"]
                self._validate_params(required_params, kwargs)
                return self._retrieve_booking(**kwargs)
                
            elif operation == "update_booking":
                required_params = ["booking_id", "version"]
                self._validate_params(required_params, kwargs)
                return self._update_booking(**kwargs)
                
            elif operation == "cancel_booking":
                required_params = ["booking_id"]
                self._validate_params(required_params, kwargs)
                return self._cancel_booking(**kwargs)
                
            # Customer Operations
            elif operation == "list_customers":
                return self._list_customers(**kwargs)
                
            elif operation == "create_customer":
                return self._create_customer(**kwargs)
                
            elif operation == "update_customer":
                required_params = ["customer_id"]
                self._validate_params(required_params, kwargs)
                return self._update_customer(**kwargs)
                
            # Catalog Operations
            elif operation == "list_catalog":
                return self._list_catalog(**kwargs)
                
            elif operation == "upsert_catalog_object":
                required_params = ["idempotency_key", "object_type"]
                self._validate_params(required_params, kwargs)
                return self._upsert_catalog_object(**kwargs)
                
            # Convenience Operations
            elif operation == "create_variable_duration_booking":
                required_params = ["base_item_id", "customer_id", "address", "appointment_time", "duration_minutes", "price_dollars"]
                self._validate_params(required_params, kwargs)
                return self._create_variable_duration_booking(**kwargs)
                
            else:
                valid_operations = [
                    "create_booking", "list_bookings", "retrieve_booking", "update_booking", "cancel_booking",
                    "list_customers", "create_customer", "update_customer",
                    "list_catalog", "upsert_catalog_object", "create_variable_duration_booking"
                ]
                raise ToolError(
                    f"Unknown operation: {operation}. Valid operations are: {', '.join(valid_operations)}",
                    ErrorCode.TOOL_INVALID_INPUT,
                )
                
    def _validate_params(self, required_params: List[str], params: Dict):
        """
        Validate that all required parameters are present.
        
        Args:
            required_params: List of parameter names that are required
            params: Dictionary of provided parameters
            
        Raises:
            ToolError: If any required parameter is missing
        """
        missing_params = [param for param in required_params if param not in params or params[param] is None]
        if missing_params:
            missing_str = ", ".join(missing_params)
            raise ToolError(
                f"Missing required parameters: {missing_str}",
                ErrorCode.TOOL_INVALID_INPUT,
            )