# Square API Tool Guide

This guide provides detailed instructions for interacting with the Square API through the `squareapi_tool`. This tool uses a template-based approach for handling complex Square API endpoints, making it easier to use in conversations.

## Table of Contents

1. [Overview](#overview)
2. [Common Parameters](#common-parameters)
3. [Bookings API](#bookings-api)
   - [Create Booking](#create-booking)
   - [List Bookings](#list-bookings)
   - [Retrieve Booking](#retrieve-booking)
   - [Update Booking](#update-booking)
   - [Cancel Booking](#cancel-booking)
4. [Working with Address Data](#working-with-address-data)
5. [Error Handling](#error-handling)
6. [Versioning and Concurrency](#versioning-and-concurrency)
7. [Common Patterns and Use Cases](#common-patterns-and-use-cases)

## Overview

The Square API provides access to various Square services including Bookings, Customers, Catalog, and more. This tool simplifies interactions with these endpoints by providing:

- Clear parameter validation and formatting
- Default values for common parameters
- Automatic handling of deeply nested JSON structures
- Comprehensive error messages
- Example usage for each endpoint

All date/time values should be in RFC 3339 format: `YYYY-MM-DDTHH:MM:SSZ`. For example: `2023-05-30T15:00:00Z`.

## Common Parameters

These parameters are used across multiple endpoints:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `location_id` | string | Square location ID | "LRBQZJ7CAJBV5" |
| `team_member_id` | string | Square team member ID | "TMjY-uYPS-Wb2hDU" |
| `location_type` | string | Type of appointment location (BUSINESS_LOCATION, CUSTOMER_LOCATION, PHONE) | "CUSTOMER_LOCATION" |
| `idempotency_key` | string | Unique key to ensure request is processed only once | None |

## Bookings API

The Bookings API allows you to create, list, retrieve, update, and cancel appointments.

### Create Booking

Creates a new booking in Square's system.

**Operation:** `create_booking`

**Required Parameters:**
- `start_at`: RFC 3339 timestamp for when the booking starts
- `service_variation_id`: ID of the service variation being booked

**Optional Parameters:**
- `location_id`: ID of the business location (default: LRBQZJ7CAJBV5)
- `location_type`: Type of location (BUSINESS_LOCATION, CUSTOMER_LOCATION, PHONE)
- `customer_id`: ID of the customer making the booking
- `customer_note`: Additional notes from the customer (max 4096 characters)
- `seller_note`: Notes from the seller (max 4096 characters, not visible to customers)
- `service_variation_version`: Current version of the service variation
- `team_member_id`: ID of the team member providing the service (default: TMjY-uYPS-Wb2hDU)
- `duration_minutes`: Duration of the appointment in minutes (max 1500)
- `idempotency_key`: A unique key to make this request idempotent
- `address`: Customer address if location_type is CUSTOMER_LOCATION (see Address format below)

**Example:**

```json
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
```

### List Bookings

Lists bookings with optional filters.

**Operation:** `list_bookings`

**Optional Parameters:**
- `location_id`: ID of the business location (default: LRBQZJ7CAJBV5)
- `start_at_min`: Minimum start time (RFC 3339 format)
- `start_at_max`: Maximum start time (RFC 3339 format)
- `team_member_id`: Filter by team member ID
- `customer_id`: Filter by customer ID
- `limit`: Maximum number of results to return (up to 100)
- `cursor`: Pagination cursor from a previous response

**Example:**

```json
{
  "operation": "list_bookings",
  "location_id": "LRBQZJ7CAJBV5",
  "start_at_min": "2023-05-01T00:00:00Z",
  "start_at_max": "2023-05-31T23:59:59Z",
  "limit": 50
}
```

**Pagination:**
The response includes a `cursor` field that can be passed in a subsequent request to retrieve the next page of results.

### Retrieve Booking

Retrieves a specific booking by ID.

**Operation:** `retrieve_booking`

**Required Parameters:**
- `booking_id`: The ID of the booking to retrieve

**Example:**

```json
{
  "operation": "retrieve_booking",
  "booking_id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI"
}
```

### Update Booking

Updates an existing booking.

**Operation:** `update_booking`

**Required Parameters:**
- `booking_id`: ID of the booking to update
- `version`: Revision number for optimistic concurrency

**Optional Parameters:**
- `start_at`: New RFC 3339 timestamp for when the booking starts
- `location_type`: Updated location type
- `customer_id`: Updated customer ID
- `customer_note`: Updated notes from the customer
- `seller_note`: Updated notes from the seller
- `service_variation_id`: Updated service variation ID
- `service_variation_version`: Updated service variation version
- `team_member_id`: Updated team member ID
- `duration_minutes`: Updated duration in minutes (max 1500)
- `idempotency_key`: A unique key to make this request idempotent
- `address`: Updated customer address

**Important:** Only include fields you want to update. Omitted fields remain unchanged.

**Example:**

```json
{
  "operation": "update_booking",
  "booking_id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
  "version": 0,
  "start_at": "2023-05-30T16:00:00Z",
  "customer_note": "I'll be 15 minutes late"
}
```

### Cancel Booking

Cancels a booking.

**Operation:** `cancel_booking`

**Required Parameters:**
- `booking_id`: ID of the booking to cancel

**Optional Parameters:**
- `booking_version`: Current version number of the booking for optimistic concurrency
- `idempotency_key`: A unique key to make this request idempotent

**Example:**

```json
{
  "operation": "cancel_booking",
  "booking_id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
  "booking_version": 1
}
```

## Working with Address Data

When the `location_type` is set to `CUSTOMER_LOCATION`, you should provide an address object with the following format:

```json
{
  "address_line_1": "500 Electric Ave",
  "address_line_2": "Suite 600",  // Optional
  "locality": "New York",  // City
  "administrative_district_level_1": "NY",  // State/Province
  "postal_code": "10003",
  "country": "US"  // Two-letter ISO country code
}
```

Fields like `sublocality`, `sublocality_2`, `sublocality_3`, `administrative_district_level_2`, `administrative_district_level_3`, `first_name`, and `last_name` are also available if needed.

## Error Handling

The tool provides detailed error messages to help diagnose issues:

**Common Errors:**

1. **Missing Required Parameters:**
   ```
   Missing required parameters: start_at, service_variation_id
   ```

2. **Invalid Date Format:**
   ```
   Invalid start_at format. Please use RFC 3339 format (YYYY-MM-DDTHH:MM:SSZ)
   ```

3. **Invalid Location Type:**
   ```
   Invalid location_type. Must be one of: BUSINESS_LOCATION, CUSTOMER_LOCATION, PHONE
   ```

4. **Exceeding Limits:**
   ```
   duration_minutes cannot exceed 1500
   ```

5. **API Error:**
   ```
   Square API error: INVALID_REQUEST_ERROR: Start time must be in the future.
   ```

## Versioning and Concurrency

Square uses optimistic concurrency control through version numbers. Each booking has a `version` field that increments with each update.

**Best Practices:**

1. Always retrieve a booking before updating it to get the current version
2. Include the version number in update and cancel requests
3. Handle potential version conflicts by retrying the operation with the updated version

Example workflow:

```python
# 1. Retrieve the booking
booking_info = squareapi_tool.run("retrieve_booking", booking_id="Z5AQ3IUAXDV2WQ2PWMMW3QKXDI")

# 2. Extract the current version
current_version = booking_info["booking"]["version"]

# 3. Update the booking with the correct version
squareapi_tool.run(
    "update_booking",
    booking_id="Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
    version=current_version,
    start_at="2023-05-30T16:00:00Z"
)
```

## Common Patterns and Use Cases

### 1. Creating a Booking for a Customer at a Business Location

```json
{
  "operation": "create_booking",
  "start_at": "2023-06-15T14:00:00Z",
  "service_variation_id": "VBKQNFTZGCKHCH7ZYN3NP4YZ",
  "location_type": "BUSINESS_LOCATION",
  "customer_id": "8FYVP54K71AGB1JGJN8SC72N24",
  "duration_minutes": 60
}
```

### 2. Searching for Bookings in a Date Range

```json
{
  "operation": "list_bookings",
  "start_at_min": "2023-06-01T00:00:00Z",
  "start_at_max": "2023-06-30T23:59:59Z"
}
```

### 3. Rescheduling an Appointment

```json
{
  "operation": "update_booking",
  "booking_id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
  "version": 0,
  "start_at": "2023-06-16T10:00:00Z"
}
```

### 4. Changing the Service Provider

```json
{
  "operation": "update_booking",
  "booking_id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
  "version": 0,
  "appointment_segments": [
    {
      "team_member_id": "TMSt7IvYgwGHGf-XNRYCC-g",
      "service_variation_id": "VBKQNFTZGCKHCH7ZYN3NP4YZ"
    }
  ]
}
```

### 5. Changing the Location Type from Business to Customer's Home

```json
{
  "operation": "update_booking",
  "booking_id": "Z5AQ3IUAXDV2WQ2PWMMW3QKXDI",
  "version": 0,
  "location_type": "CUSTOMER_LOCATION",
  "address": {
    "address_line_1": "123 Main St",
    "locality": "Brooklyn",
    "administrative_district_level_1": "NY",
    "postal_code": "11201",
    "country": "US"
  }
}
```

Remember to always handle errors gracefully and provide clear explanations to users when operations fail.