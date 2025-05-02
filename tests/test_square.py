"""
Tests for the Square tool.

This suite provides comprehensive test coverage for the SquareTool class,
ensuring reliable functionality across customer management and booking operations.
"""
import pytest
import os
from unittest.mock import patch, MagicMock, call
import json

from tools.square_tool import SquareTool
from errors import ToolError, ErrorCode
from config import config


class MockResponse:
    """Mock response object from Square API."""
    
    def __init__(self, success=True, body=None, errors=None, cursor=None):
        self._success = success
        self.body = body or {}
        self.errors = errors or []
        if cursor and success:
            self.body["cursor"] = cursor
        self.cursor = cursor  # For pagination testing
    
    def is_success(self):
        return self._success
    
    def is_error(self):
        return not self._success


@pytest.fixture
def mock_square_client():
    """Fixture to create a mock Square client."""
    with patch('square.client.Client') as mock_client_class:
        mock_client = MagicMock()
        
        # Mock customer-related APIs
        mock_customers_api = MagicMock()
        mock_client.customers = mock_customers_api
        
        # Mock booking-related APIs
        mock_bookings_api = MagicMock()
        mock_client.bookings = mock_bookings_api
        
        # Mock catalog-related APIs
        mock_catalog_api = MagicMock()
        mock_client.catalog = mock_catalog_api
        
        # Setup the mock client to be returned by the constructor
        mock_client_class.return_value = mock_client
        
        yield mock_client


@pytest.fixture
def square_tool():
    """Fixture to create a SquareTool instance."""
    return SquareTool()


def test_lazy_loading(square_tool, monkeypatch):
    """Test that Square SDK is lazily loaded only when needed."""
    # Import should not happen on initialization
    assert square_tool._client is None
    
    # Mock the import to fail
    monkeypatch.setattr("importlib.import_module", MagicMock(side_effect=ImportError("No module named 'square'")))
    
    # Access client should raise the proper error now
    with pytest.raises(ToolError) as exc_info:
        client = square_tool.client
    
    assert "Square SDK not installed" in str(exc_info.value)
    assert exc_info.value.code == ErrorCode.TOOL_EXECUTION_ERROR


def test_client_initialization():
    """Test that the Square client is correctly initialized."""
    # Mock environment variables
    os.environ["SQUARE_API_KEY"] = "test_api_key"
    os.environ["SQUARE_APPLICATION_ID"] = "test_app_id"

    # Mock config.square_api_key property to return a string value
    with patch('tools.square_tool.config') as mock_config:
        # Set up the mock to return a string when square_api_key is accessed
        mock_config.square_tool.api_key = "test_api_key"
        mock_config.square_tool.environment = "sandbox"
        mock_config.square_tool.timeout = 60
        mock_config.square_tool.max_retries = 3
        mock_config.square_tool.backoff_factor = 2.0
        
        with patch('square.client.Client') as mock_client:
            instance = mock_client.return_value
            
            tool = SquareTool()
            client = tool.client
            
            # Verify client was instantiated with correct parameters
            mock_client.assert_called_once()
            args, kwargs = mock_client.call_args
            assert kwargs["access_token"] == "test_api_key"
            assert kwargs["user_agent_detail"] == "Bot with Memory"
            assert "Square-Application-Id" in kwargs["additional_headers"]
            
            # Verify that client is only initialized once
            tool.client
            assert mock_client.call_count == 1


def test_handle_response_success(square_tool):
    """Test _handle_response with successful response."""
    expected_body = {"data": "success"}
    mock_response = MockResponse(success=True, body=expected_body)
    
    result = square_tool._handle_response(mock_response, "test_operation")
    
    assert result == expected_body


def test_handle_response_error(square_tool):
    """Test _handle_response with error response."""
    mock_errors = [
        {
            "category": "AUTHENTICATION_ERROR",
            "code": "UNAUTHORIZED",
            "detail": "Invalid access token"
        }
    ]
    mock_response = MockResponse(success=False, errors=mock_errors)
    
    with pytest.raises(ToolError) as exc_info:
        square_tool._handle_response(mock_response, "test_operation")
    
    assert "AUTHENTICATION_ERROR.UNAUTHORIZED" in str(exc_info.value)
    assert "Invalid access token" in str(exc_info.value)
    assert exc_info.value.code == ErrorCode.TOOL_EXECUTION_ERROR


def test_handle_response_unexpected(square_tool):
    """Test _handle_response with unexpected response type."""
    # Create a response that's neither success nor error
    mock_response = MagicMock()
    mock_response.is_success.return_value = False
    mock_response.is_error.return_value = False
    
    with pytest.raises(ToolError) as exc_info:
        square_tool._handle_response(mock_response, "test_operation")
    
    assert "Unexpected response" in str(exc_info.value)
    assert exc_info.value.code == ErrorCode.TOOL_EXECUTION_ERROR


# ---------- Customer API Tests ----------

def test_list_customers(square_tool, mock_square_client):
    """Test listing customers."""
    # Mock customer data
    mock_customers = [
        {
            "id": "customer1",
            "given_name": "John",
            "family_name": "Doe",
            "email_address": "john.doe@example.com",
        },
        {
            "id": "customer2",
            "given_name": "Jane",
            "family_name": "Smith",
            "email_address": "jane.smith@example.com",
        }
    ]
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body={"customers": mock_customers}
    )
    mock_square_client.customers.list_customers.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Test the list_customers operation
    result = square_tool.run(
        operation="list_customers",
        limit=10,
        sort_field="DEFAULT",
        sort_order="ASC"
    )
    
    # Verify the result
    assert "customers" in result
    assert len(result["customers"]) == 2
    assert result["customers"][0]["id"] == "customer1"
    assert result["customers"][1]["email_address"] == "jane.smith@example.com"
    
    # Verify the API was called with correct parameters
    mock_square_client.customers.list_customers.assert_called_once_with(
        cursor=None,
        limit=10,
        sort_field="DEFAULT",
        sort_order="ASC",
        count=False
    )


def test_retrieve_customer(square_tool, mock_square_client):
    """Test retrieving a customer by ID."""
    # Mock customer data
    mock_customer = {
        "id": "customer1",
        "given_name": "John",
        "family_name": "Doe",
        "email_address": "john.doe@example.com",
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body={"customer": mock_customer}
    )
    mock_square_client.customers.retrieve_customer.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Test the retrieve_customer operation
    result = square_tool.run(
        operation="retrieve_customer",
        customer_id="customer1"
    )
    
    # Verify the result
    assert "customer" in result
    assert result["customer"]["id"] == "customer1"
    assert result["customer"]["email_address"] == "john.doe@example.com"
    
    # Verify the API was called with correct parameters
    mock_square_client.customers.retrieve_customer.assert_called_once_with(
        customer_id="customer1"
    )


def test_create_customer(square_tool, mock_square_client):
    """Test creating a customer."""
    # Mock customer data
    mock_customer = {
        "id": "new_customer",
        "given_name": "Alice",
        "family_name": "Johnson",
        "email_address": "alice.johnson@example.com",
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body={"customer": mock_customer}
    )
    mock_square_client.customers.create_customer.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Customer data to create
    customer_data = {
        "customer": {
            "given_name": "Alice",
            "family_name": "Johnson",
            "email_address": "alice.johnson@example.com",
        }
    }
    
    # Test the create_customer operation
    result = square_tool.run(
        operation="create_customer",
        body=customer_data
    )
    
    # Verify the result
    assert "customer" in result
    assert result["customer"]["id"] == "new_customer"
    assert result["customer"]["email_address"] == "alice.johnson@example.com"
    
    # Verify the API was called with correct parameters
    mock_square_client.customers.create_customer.assert_called_once_with(body=customer_data)


def test_update_customer(square_tool, mock_square_client):
    """Test updating a customer."""
    # Mock updated customer data
    mock_customer = {
        "id": "customer1",
        "given_name": "John",
        "family_name": "Updated",
        "email_address": "john.updated@example.com",
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body={"customer": mock_customer}
    )
    mock_square_client.customers.update_customer.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Update data
    update_data = {
        "customer": {
            "family_name": "Updated",
            "email_address": "john.updated@example.com",
            "version": 1
        }
    }
    
    # Test the update_customer operation
    result = square_tool.run(
        operation="update_customer",
        customer_id="customer1",
        body=update_data
    )
    
    # Verify the result
    assert "customer" in result
    assert result["customer"]["id"] == "customer1"
    assert result["customer"]["family_name"] == "Updated"
    assert result["customer"]["email_address"] == "john.updated@example.com"
    
    # Verify the API was called with correct parameters
    mock_square_client.customers.update_customer.assert_called_once_with(
        customer_id="customer1",
        body=update_data
    )


def test_delete_customer(square_tool, mock_square_client):
    """Test deleting a customer."""
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body={}
    )
    mock_square_client.customers.delete_customer.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Test the delete_customer operation
    result = square_tool.run(
        operation="delete_customer",
        customer_id="customer1",
        version=2
    )
    
    # Verify the result is empty but successful
    assert result == {}
    
    # Verify the API was called with correct parameters
    mock_square_client.customers.delete_customer.assert_called_once_with(
        customer_id="customer1",
        version=2
    )


def test_search_customers(square_tool, mock_square_client):
    """Test searching for customers."""
    # Mock customer data
    mock_customers = [
        {
            "id": "customer1",
            "given_name": "John",
            "family_name": "Doe",
            "email_address": "john.doe@example.com",
        }
    ]
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body={"customers": mock_customers, "cursor": "next_cursor"}
    )
    mock_square_client.customers.search_customers.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Search query
    search_body = {
        "query": {
            "filter": {
                "email_address": {
                    "fuzzy": "example.com"
                }
            }
        }
    }
    
    # Test the search_customers operation
    result = square_tool.run(
        operation="search_customers",
        body=search_body
    )
    
    # Verify the result
    assert "customers" in result
    assert len(result["customers"]) == 1
    assert result["customers"][0]["id"] == "customer1"
    assert result["cursor"] == "next_cursor"
    
    # Verify the API was called with correct parameters
    mock_square_client.customers.search_customers.assert_called_once_with(body=search_body)


def test_bulk_create_customers(square_tool, mock_square_client):
    """Test bulk creating customers."""
    # Mock response data
    mock_response_data = {
        "responses": {
            "key1": {
                "customer": {
                    "id": "customer1",
                    "given_name": "John"
                }
            },
            "key2": {
                "customer": {
                    "id": "customer2",
                    "given_name": "Jane"
                }
            }
        }
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body=mock_response_data
    )
    mock_square_client.customers.bulk_create_customers.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Bulk create data
    bulk_create_body = {
        "customers": {
            "key1": {
                "given_name": "John"
            },
            "key2": {
                "given_name": "Jane"
            }
        }
    }
    
    # Test the bulk_create_customers operation
    result = square_tool.run(
        operation="bulk_create_customers",
        body=bulk_create_body
    )
    
    # Verify the result
    assert "responses" in result
    assert "key1" in result["responses"]
    assert "key2" in result["responses"]
    assert result["responses"]["key1"]["customer"]["id"] == "customer1"
    assert result["responses"]["key2"]["customer"]["id"] == "customer2"
    
    # Verify the API was called with correct parameters
    mock_square_client.customers.bulk_create_customers.assert_called_once_with(body=bulk_create_body)


def test_bulk_retrieve_customers(square_tool, mock_square_client):
    """Test bulk retrieving customers."""
    # Mock response data
    mock_response_data = {
        "responses": {
            "customer1": {
                "customer": {
                    "id": "customer1",
                    "given_name": "John"
                }
            },
            "customer2": {
                "customer": {
                    "id": "customer2",
                    "given_name": "Jane"
                }
            }
        }
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body=mock_response_data
    )
    mock_square_client.customers.bulk_retrieve_customers.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Bulk retrieve data
    bulk_retrieve_body = {
        "customer_ids": ["customer1", "customer2"]
    }
    
    # Test the bulk_retrieve_customers operation
    result = square_tool.run(
        operation="bulk_retrieve_customers",
        body=bulk_retrieve_body
    )
    
    # Verify the result
    assert "responses" in result
    assert "customer1" in result["responses"]
    assert "customer2" in result["responses"]
    
    # Verify the API was called with correct parameters
    mock_square_client.customers.bulk_retrieve_customers.assert_called_once_with(body=bulk_retrieve_body)


def test_bulk_update_customers(square_tool, mock_square_client):
    """Test bulk updating customers."""
    # Mock response data
    mock_response_data = {
        "responses": {
            "customer1": {
                "customer": {
                    "id": "customer1",
                    "given_name": "Updated John"
                }
            },
            "customer2": {
                "customer": {
                    "id": "customer2",
                    "given_name": "Updated Jane"
                }
            }
        }
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body=mock_response_data
    )
    mock_square_client.customers.bulk_update_customers.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Bulk update data
    bulk_update_body = {
        "customers": {
            "customer1": {
                "given_name": "Updated John",
                "version": 1
            },
            "customer2": {
                "given_name": "Updated Jane",
                "version": 2
            }
        }
    }
    
    # Test the bulk_update_customers operation
    result = square_tool.run(
        operation="bulk_update_customers",
        body=bulk_update_body
    )
    
    # Verify the result
    assert "responses" in result
    assert result["responses"]["customer1"]["customer"]["given_name"] == "Updated John"
    assert result["responses"]["customer2"]["customer"]["given_name"] == "Updated Jane"
    
    # Verify the API was called with correct parameters
    mock_square_client.customers.bulk_update_customers.assert_called_once_with(body=bulk_update_body)


def test_bulk_delete_customers(square_tool, mock_square_client):
    """Test bulk deleting customers."""
    # Mock response data
    mock_response_data = {
        "responses": {
            "customer1": {},
            "customer2": {
                "errors": [
                    {
                        "category": "NOT_FOUND",
                        "code": "NOT_FOUND",
                        "detail": "Customer not found"
                    }
                ]
            }
        }
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body=mock_response_data
    )
    mock_square_client.customers.bulk_delete_customers.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Bulk delete data
    bulk_delete_body = {
        "customer_ids": ["customer1", "customer2"]
    }
    
    # Test the bulk_delete_customers operation
    result = square_tool.run(
        operation="bulk_delete_customers",
        body=bulk_delete_body
    )
    
    # Verify the result
    assert "responses" in result
    assert "customer1" in result["responses"]
    assert "errors" in result["responses"]["customer2"]
    
    # Verify the API was called with correct parameters
    mock_square_client.customers.bulk_delete_customers.assert_called_once_with(body=bulk_delete_body)


# ---------- Booking API Tests ----------

def test_create_booking(square_tool, mock_square_client):
    """Test creating a booking."""
    # Mock booking data
    mock_booking = {
        "id": "new_booking",
        "start_at": "2023-05-15T14:00:00Z",
        "location_id": "location1",
        "customer_id": "customer1",
        "status": "ACCEPTED"
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body={"booking": mock_booking}
    )
    mock_square_client.bookings.create_booking.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Booking data to create
    booking_data = {
        "booking": {
            "start_at": "2023-05-15T14:00:00Z",
            "location_id": "location1",
            "customer_id": "customer1",
            "appointment_segments": [
                {
                    "team_member_id": "team_member1",
                    "service_variation_id": "service1",
                    "service_variation_version": 1599775456731
                }
            ]
        }
    }
    
    # Test the create_booking operation
    result = square_tool.run(
        operation="create_booking",
        body=booking_data
    )
    
    # Verify the result
    assert "booking" in result
    assert result["booking"]["id"] == "new_booking"
    assert result["booking"]["location_id"] == "location1"
    
    # Verify the API was called with correct parameters
    mock_square_client.bookings.create_booking.assert_called_once_with(body=booking_data)


def test_retrieve_booking(square_tool, mock_square_client):
    """Test retrieving a booking."""
    # Mock booking data
    mock_booking = {
        "id": "booking1",
        "start_at": "2023-05-15T14:00:00Z",
        "location_id": "location1",
        "customer_id": "customer1",
        "status": "ACCEPTED"
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body={"booking": mock_booking}
    )
    mock_square_client.bookings.retrieve_booking.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Test the retrieve_booking operation
    result = square_tool.run(
        operation="retrieve_booking",
        booking_id="booking1"
    )
    
    # Verify the result
    assert "booking" in result
    assert result["booking"]["id"] == "booking1"
    assert result["booking"]["status"] == "ACCEPTED"
    
    # Verify the API was called with correct parameters
    mock_square_client.bookings.retrieve_booking.assert_called_once_with(booking_id="booking1")


def test_update_booking(square_tool, mock_square_client):
    """Test updating a booking."""
    # Mock updated booking data
    mock_booking = {
        "id": "booking1",
        "start_at": "2023-06-15T15:00:00Z",  # Updated time
        "location_id": "location1",
        "customer_id": "customer1",
        "status": "ACCEPTED"
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body={"booking": mock_booking}
    )
    mock_square_client.bookings.update_booking.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Update data
    update_data = {
        "booking": {
            "start_at": "2023-06-15T15:00:00Z",
            "version": 1
        }
    }
    
    # Test the update_booking operation
    result = square_tool.run(
        operation="update_booking",
        booking_id="booking1",
        body=update_data
    )
    
    # Verify the result
    assert "booking" in result
    assert result["booking"]["id"] == "booking1"
    assert result["booking"]["start_at"] == "2023-06-15T15:00:00Z"
    
    # Verify the API was called with correct parameters
    mock_square_client.bookings.update_booking.assert_called_once_with(
        booking_id="booking1",
        body=update_data
    )


def test_cancel_booking(square_tool, mock_square_client):
    """Test canceling a booking."""
    # Mock canceled booking data
    mock_booking = {
        "id": "booking1",
        "start_at": "2023-05-15T14:00:00Z",
        "location_id": "location1",
        "customer_id": "customer1",
        "status": "CANCELLED_BY_CUSTOMER"
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body={"booking": mock_booking}
    )
    mock_square_client.bookings.cancel_booking.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Cancel data
    cancel_data = {
        "booking_version": 1,
        "idempotency_key": "unique-key"
    }
    
    # Test the cancel_booking operation
    result = square_tool.run(
        operation="cancel_booking",
        booking_id="booking1",
        body=cancel_data
    )
    
    # Verify the result
    assert "booking" in result
    assert result["booking"]["id"] == "booking1"
    assert result["booking"]["status"] == "CANCELLED_BY_CUSTOMER"
    
    # Verify the API was called with correct parameters
    mock_square_client.bookings.cancel_booking.assert_called_once_with(
        booking_id="booking1",
        body=cancel_data
    )


def test_list_bookings(square_tool, mock_square_client):
    """Test listing bookings."""
    # Mock booking data
    mock_bookings = [
        {
            "id": "booking1",
            "start_at": "2023-04-15T14:00:00Z",
            "location_id": "location1",
            "customer_id": "customer1",
            "status": "ACCEPTED"
        }
    ]
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body={"bookings": mock_bookings}
    )
    mock_square_client.bookings.list_bookings.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Test the list_bookings operation
    result = square_tool.run(
        operation="list_bookings",
        limit=10,
        location_id="location1",
        start_at_min="2023-04-01T00:00:00Z",
        start_at_max="2023-04-30T23:59:59Z"
    )
    
    # Verify the result
    assert "bookings" in result
    assert len(result["bookings"]) == 1
    assert result["bookings"][0]["id"] == "booking1"
    assert result["bookings"][0]["status"] == "ACCEPTED"
    
    # Verify the API was called with correct parameters
    mock_square_client.bookings.list_bookings.assert_called_once_with(
        limit=10,
        cursor=None,
        customer_id=None,
        team_member_id=None,
        location_id="location1",
        start_at_min="2023-04-01T00:00:00Z",
        start_at_max="2023-04-30T23:59:59Z"
    )


def test_search_availability(square_tool, mock_square_client):
    """Test searching for availability."""
    # Mock availability data
    mock_availabilities = [
        {
            "start_at": "2023-05-15T14:00:00Z",
            "location_id": "location1",
            "appointment_segments": [
                {
                    "duration_minutes": 60,
                    "team_member_id": "team_member1",
                    "service_variation_id": "service1"
                }
            ]
        }
    ]
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body={"availabilities": mock_availabilities}
    )
    mock_square_client.bookings.search_availability.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Search query
    search_body = {
        "query": {
            "filter": {
                "start_at_range": {
                    "start_at": "2023-05-15T00:00:00Z",
                    "end_at": "2023-05-15T23:59:59Z"
                },
                "location_id": "location1"
            }
        }
    }
    
    # Test the search_availability operation
    result = square_tool.run(
        operation="search_availability",
        body=search_body
    )
    
    # Verify the result
    assert "availabilities" in result
    assert len(result["availabilities"]) == 1
    assert result["availabilities"][0]["location_id"] == "location1"
    
    # Verify the API was called with correct parameters
    mock_square_client.bookings.search_availability.assert_called_once_with(body=search_body)


def test_retrieve_business_booking_profile(square_tool, mock_square_client):
    """Test retrieving business booking profile."""
    # Mock profile data
    mock_profile = {
        "business_booking_profile": {
            "seller_id": "seller1",
            "created_at": "2021-01-01T00:00:00Z",
            "booking_enabled": True,
            "customer_timezone_choice": "BUSINESS_LOCATION_TIMEZONE",
            "booking_policy": "ACCEPT_ALL"
        }
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body=mock_profile
    )
    mock_square_client.bookings.retrieve_business_booking_profile.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Test the retrieve_business_booking_profile operation
    result = square_tool.run(
        operation="retrieve_business_booking_profile"
    )
    
    # Verify the result
    assert "business_booking_profile" in result
    assert result["business_booking_profile"]["booking_enabled"] is True
    
    # Verify the API was called with correct parameters
    mock_square_client.bookings.retrieve_business_booking_profile.assert_called_once()


def test_list_team_member_booking_profiles(square_tool, mock_square_client):
    """Test listing team member booking profiles."""
    # Mock profile data
    mock_profiles = {
        "team_member_booking_profiles": [
            {
                "team_member_id": "team_member1",
                "description": "Senior Stylist",
                "display_name": "John Doe",
                "is_bookable": True
            }
        ]
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body=mock_profiles
    )
    mock_square_client.bookings.list_team_member_booking_profiles.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Test the list_team_member_booking_profiles operation
    result = square_tool.run(
        operation="list_team_member_booking_profiles",
        bookable_only=True,
        limit=10,
        location_id="location1"
    )
    
    # Verify the result
    assert "team_member_booking_profiles" in result
    assert len(result["team_member_booking_profiles"]) == 1
    assert result["team_member_booking_profiles"][0]["team_member_id"] == "team_member1"
    
    # Verify the API was called with correct parameters
    mock_square_client.bookings.list_team_member_booking_profiles.assert_called_once_with(
        bookable_only=True,
        limit=10,
        cursor=None,
        location_id="location1"
    )


def test_bulk_retrieve_bookings(square_tool, mock_square_client):
    """Test bulk retrieving bookings."""
    # Mock booking data
    mock_response_data = {
        "bookings": {
            "booking1": {
                "booking": {
                    "id": "booking1",
                    "status": "ACCEPTED"
                }
            },
            "booking2": {
                "booking": {
                    "id": "booking2",
                    "status": "PENDING"
                }
            }
        }
    }
    
    # Setup the mock response
    mock_response = MockResponse(
        success=True,
        body=mock_response_data
    )
    mock_square_client.bookings.bulk_retrieve_bookings.return_value = mock_response
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Bulk retrieve data
    bulk_retrieve_body = {
        "booking_ids": ["booking1", "booking2"]
    }
    
    # Test the bulk_retrieve_bookings operation
    result = square_tool.run(
        operation="bulk_retrieve_bookings",
        body=bulk_retrieve_body
    )
    
    # Verify the result
    assert "bookings" in result
    assert "booking1" in result["bookings"]
    assert "booking2" in result["bookings"]
    assert result["bookings"]["booking1"]["booking"]["status"] == "ACCEPTED"
    
    # Verify the API was called with correct parameters
    mock_square_client.bookings.bulk_retrieve_bookings.assert_called_once_with(body=bulk_retrieve_body)


# ---------- Error Handling Tests ----------

def test_invalid_operation(square_tool):
    """Test error handling for invalid operations."""
    with pytest.raises(ToolError) as exc_info:
        square_tool.run(operation="invalid_operation")
    
    assert "Unknown operation" in str(exc_info.value)
    assert exc_info.value.code == ErrorCode.TOOL_INVALID_INPUT


def test_missing_required_customer_params(square_tool):
    """Test error handling for missing required customer parameters."""
    with pytest.raises(ToolError) as exc_info:
        square_tool.run(operation="retrieve_customer")
    
    assert "customer_id parameter is required" in str(exc_info.value)
    assert exc_info.value.code == ErrorCode.TOOL_INVALID_INPUT


def test_missing_required_booking_params(square_tool):
    """Test error handling for missing required booking parameters."""
    with pytest.raises(ToolError) as exc_info:
        square_tool.run(operation="retrieve_booking")
    
    assert "booking_id parameter is required" in str(exc_info.value)
    assert exc_info.value.code == ErrorCode.TOOL_INVALID_INPUT


def test_missing_required_body_params(square_tool):
    """Test error handling for missing required body parameters."""
    with pytest.raises(ToolError) as exc_info:
        square_tool.run(operation="create_booking")
    
    assert "body parameter is required" in str(exc_info.value)
    assert exc_info.value.code == ErrorCode.TOOL_INVALID_INPUT


def test_cancel_booking_missing_body(square_tool):
    """Test error handling for missing body in cancel_booking."""
    with pytest.raises(ToolError) as exc_info:
        square_tool.run(
            operation="cancel_booking",
            booking_id="booking1"
        )
    
    assert "body parameter is required" in str(exc_info.value)
    assert exc_info.value.code == ErrorCode.TOOL_INVALID_INPUT


def test_update_customer_missing_params(square_tool):
    """Test error handling for missing parameters in update_customer."""
    # Test missing customer_id
    with pytest.raises(ToolError) as exc_info:
        square_tool.run(
            operation="update_customer",
            body={"customer": {}}
        )
    
    assert "customer_id parameter is required" in str(exc_info.value)
    
    # Test missing body
    with pytest.raises(ToolError) as exc_info:
        square_tool.run(
            operation="update_customer",
            customer_id="customer1"
        )
    
    assert "body parameter is required" in str(exc_info.value)


# ---------- Local Customer Search Tests ----------

def test_categorize_search_query(square_tool):
    """Test that search query categorization works correctly."""
    # Test email detection
    assert square_tool._categorize_search_query("user@example.com") == "email"
    assert square_tool._categorize_search_query("contact+name@company.co.uk") == "email"
    
    # Test phone detection
    assert square_tool._categorize_search_query("123-456-7890") == "phone"
    assert square_tool._categorize_search_query("(123) 456-7890") == "phone"
    assert square_tool._categorize_search_query("1234567890") == "phone"
    
    # Test address detection
    assert square_tool._categorize_search_query("123 Main St") == "address"
    assert square_tool._categorize_search_query("123 Baker Street, London") == "address"
    assert square_tool._categorize_search_query("10 Downing Street") == "address"
    
    # Test name detection
    assert square_tool._categorize_search_query("John Smith") == "name"
    assert square_tool._categorize_search_query("Robert Jones Jr") == "name"
    
    # Test fallback to any
    assert square_tool._categorize_search_query("search term") == "any"
    assert square_tool._categorize_search_query("abc") == "any"

def test_search_local_customers(square_tool, monkeypatch):
    """Test local customer directory search functionality."""
    # Mock customer directory with test data
    test_directory = {
        "customers": {
            "cust1": {
                "id": "cust1",
                "given_name": "John",
                "family_name": "Smith",
                "email_address": "john.smith@example.com",
                "phone_number": "+1-555-123-4567"
            },
            "cust2": {
                "id": "cust2",
                "given_name": "Jane",
                "family_name": "Doe",
                "email_address": "jane.doe@example.com",
                "phone_number": "+1-555-987-6543",
                "address": {
                    "address_line_1": "123 Main St",
                    "locality": "San Francisco",
                    "administrative_district_level_1": "CA",
                    "postal_code": "94105",
                    "country": "US"
                }
            }
        },
        "last_updated": int(time.time())
    }
    
    # Set the test directory for the tool
    square_tool.customer_directory = test_directory
    
    # Test name search
    results = square_tool._search_local_customers("John Smith")
    assert len(results) == 1
    assert results[0]["id"] == "cust1"
    
    # Test email search
    results = square_tool._search_local_customers("jane.doe")
    assert len(results) == 1
    assert results[0]["id"] == "cust2"
    
    # Test phone search (digits only)
    results = square_tool._search_local_customers("9876543")
    assert len(results) == 1
    assert results[0]["id"] == "cust2"
    
    # Test address search
    results = square_tool._search_local_customers("123 Main")
    assert len(results) == 1
    assert results[0]["id"] == "cust2"
    
    # Test no results
    results = square_tool._search_local_customers("Unknown Person")
    assert len(results) == 0

def test_search_local_customers_auto_rebuild(square_tool, monkeypatch):
    """Test automatic rebuilding of customer directory on search miss."""
    # Set up an empty directory first
    square_tool.customer_directory = {"customers": {}, "last_updated": 0}
    
    # Mock the rebuild method to add a customer matching our search
    def mock_rebuild():
        square_tool.customer_directory = {
            "customers": {
                "new_cust": {
                    "id": "new_cust",
                    "given_name": "New",
                    "family_name": "Customer",
                    "email_address": "new.customer@example.com"
                }
            },
            "last_updated": int(time.time())
        }
        return True
    
    # Apply the mock
    monkeypatch.setattr(square_tool, "_rebuild_customer_directory", mock_rebuild)
    
    # Should trigger rebuild and find the customer
    result = square_tool.search_local_customers("New Customer")
    
    assert "customers" in result
    assert len(result["customers"]) == 1
    assert result["customers"][0]["id"] == "new_cust"
    
    # Now make the rebuild fail
    def mock_rebuild_fail():
        return False
    
    monkeypatch.setattr(square_tool, "_rebuild_customer_directory", mock_rebuild_fail)
    
    # Make the search fail by using an unknown term
    with pytest.raises(ToolError) as exc_info:
        square_tool.search_local_customers("Unknown Person")
    
    assert "No customers found" in str(exc_info.value)

def test_run_search_local_customers(square_tool, monkeypatch):
    """Test running the search_local_customers operation via the run method."""
    # Mock the search method
    def mock_search(query, category=None):
        return {"customers": [{"id": "test_id", "given_name": "Test", "family_name": "User"}]}
    
    monkeypatch.setattr(square_tool, "search_local_customers", mock_search)
    
    # Test successful operation
    result = square_tool.run(operation="search_local_customers", query="Test User")
    assert "customers" in result
    assert len(result["customers"]) == 1
    assert result["customers"][0]["id"] == "test_id"
    
    # Test missing query parameter
    with pytest.raises(ToolError) as exc_info:
        square_tool.run(operation="search_local_customers")
    
    assert "query parameter is required" in str(exc_info.value)

# ---------- Additional Tests ----------

def test_pagination_handling(square_tool, mock_square_client):
    """Test handling of paginated responses with results spanning multiple pages."""
    # Create multiple mock responses for pagination
    mock_response1 = MockResponse(
        success=True,
        body={
            "customers": [
                {"id": "customer1", "given_name": "John"}
            ]
        },
        cursor="page2"
    )
    
    mock_response2 = MockResponse(
        success=True,
        body={
            "customers": [
                {"id": "customer2", "given_name": "Jane"}
            ]
        },
        cursor="page3"
    )
    
    mock_response3 = MockResponse(
        success=True,
        body={
            "customers": [
                {"id": "customer3", "given_name": "Bob"}
            ]
        },
        cursor=None  # No more pages
    )
    
    # Set up the mock client to return different responses on subsequent calls
    mock_square_client.customers.list_customers.side_effect = [
        mock_response1, mock_response2, mock_response3
    ]
    
    # Force the tool to use our mock client
    square_tool._client = mock_square_client
    
    # Patch _list_customers to use _handle_pagination directly
    with patch.object(square_tool, '_list_customers') as mock_list_customers:
        # Make it use our pagination method
        mock_list_customers.side_effect = lambda **kwargs: square_tool._handle_pagination(
            "list_customers",
            mock_square_client.customers.list_customers,
            **kwargs
        )
        
        # Call list_customers which should trigger our mock
        square_tool.run(
            operation="list_customers",
            limit=10
        )
    
    # Verify that list_customers was called three times with different cursor values
    assert mock_square_client.customers.list_customers.call_count == 3
    
    calls = [
        call(cursor=None, limit=10, sort_field=None, sort_order=None, count=False),
        call(cursor="page2", limit=10, sort_field=None, sort_order=None, count=False),
        call(cursor="page3", limit=10, sort_field=None, sort_order=None, count=False)
    ]
    mock_square_client.customers.list_customers.assert_has_calls(calls, any_order=False)