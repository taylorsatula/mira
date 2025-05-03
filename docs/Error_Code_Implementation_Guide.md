# Error Code Implementation Guide

This document provides guidance on implementing the newly proposed error codes across the codebase. It identifies specific locations in each tool where the new error codes would be most appropriate.

## Table of Contents
1. [Updated Error Codes](#updated-error-codes)
2. [Implementation Locations](#implementation-locations)
   - [Network Tools](#network-tools)
   - [Data Validation](#data-validation)
   - [Database Operations](#database-operations)
   - [Workflow Management](#workflow-management)
   - [Service-Specific Tools](#service-specific-tools)
3. [Migration Strategy](#migration-strategy)

## Updated Error Codes

### Network Error Subcategories (2xx)
```python
# API errors (2xx)
API_CONNECTION_ERROR = 201       # Existing
API_AUTHENTICATION_ERROR = 202   # Existing
API_RATE_LIMIT_ERROR = 203       # Existing
API_RESPONSE_ERROR = 204         # Existing
API_TIMEOUT_ERROR = 205          # Existing
DNS_RESOLUTION_ERROR = 206       # New
SSL_CERTIFICATE_ERROR = 207      # New
PROXY_CONNECTION_ERROR = 208     # New
NETWORK_UNREACHABLE_ERROR = 209  # New
```

### Data Validation (4xx)
```python
# Tool errors (4xx)
TOOL_NOT_FOUND = 401                   # Existing
TOOL_EXECUTION_ERROR = 402             # Existing
TOOL_INVALID_INPUT = 403               # Existing
TOOL_INVALID_OUTPUT = 404              # Existing
TOOL_INITIALIZATION_ERROR = 405        # Existing
TOOL_REGISTRATION_ERROR = 406          # Existing
TOOL_DUPLICATE_NAME = 407              # Existing
TOOL_NOT_ENABLED = 408                 # Existing
TOOL_INVALID_PARAMETERS = 409          # Existing
TOOL_CIRCULAR_DEPENDENCY = 410         # Existing
TOOL_DATA_TYPE_ERROR = 411             # New
TOOL_DATA_FORMAT_ERROR = 412           # New
TOOL_DATA_RANGE_ERROR = 413            # New
TOOL_PARAMETER_MISSING = 414           # New
TOOL_INSUFFICIENT_PERMISSIONS = 415    # New
EMAIL_AUTHENTICATION_ERROR = 420       # New
CALENDAR_SYNC_ERROR = 421              # New
GEOLOCATION_ERROR = 422                # New
```

### Database Errors (7xx - New Category)
```python
# Database errors (7xx)
DATABASE_CONNECTION_ERROR = 701
DATABASE_QUERY_ERROR = 702
DATABASE_CONSTRAINT_ERROR = 703
DATABASE_TRANSACTION_ERROR = 704
```

### Workflow Errors (8xx - New Category)
```python
# Workflow errors (8xx)
WORKFLOW_NOT_FOUND = 801
WORKFLOW_INVALID_STATE = 802
WORKFLOW_STEP_ERROR = 803
WORKFLOW_VALIDATION_ERROR = 804
WORKFLOW_DEPENDENCY_ERROR = 805
```

## Implementation Locations

### Network Tools

#### HTTP Tool (tools/http_tool.py)
```python
# Replace generic API errors with more specific ones
except requests.exceptions.ConnectionError as e:
    # Check DNS resolution error
    if "Name or service not known" in str(e) or "getaddrinfo failed" in str(e):
        raise ToolError(
            f"DNS resolution failed for {url}", 
            ErrorCode.DNS_RESOLUTION_ERROR,
            {"url": url, "error": str(e)}
        )
    # Check SSL error
    elif "SSL" in str(e) or "certificate" in str(e).lower():
        raise ToolError(
            f"SSL certificate error for {url}", 
            ErrorCode.SSL_CERTIFICATE_ERROR,
            {"url": url, "error": str(e)}
        )
    # Check proxy error
    elif "Proxy" in str(e) or "407" in str(e):
        raise ToolError(
            f"Proxy connection error for {url}", 
            ErrorCode.PROXY_CONNECTION_ERROR,
            {"url": url, "error": str(e)}
        )
    # General network error
    else:
        raise ToolError(
            f"Network error while connecting to {url}", 
            ErrorCode.NETWORK_UNREACHABLE_ERROR,
            {"url": url, "error": str(e)}
        )
```

#### Email Tool (tools/email_tool.py)
```python
# Add to IMAP connection errors:
except imaplib.IMAP4.error as e:
    if "authentication failed" in str(e).lower() or "login failed" in str(e).lower():
        raise ToolError(
            f"Email authentication failed: {str(e)}",
            ErrorCode.EMAIL_AUTHENTICATION_ERROR,
            {"server": self.imap_server, "user": self.username}
        )
    else:
        raise ToolError(
            f"Error connecting to email server: {str(e)}",
            ErrorCode.API_CONNECTION_ERROR,
            {"server": self.imap_server}
        )
```

### Data Validation

#### Calendar Tool (tools/calendar_tool.py)
```python
# Replace generic validation error:
if not is_iso_format(start_time):
    raise ToolError(
        f"Invalid start_time format: {start_time}. Expected ISO format (YYYY-MM-DDTHH:MM:SS)",
        ErrorCode.TOOL_DATA_FORMAT_ERROR,
        {"parameter": "start_time", "value": start_time, "expected_format": "ISO 8601"}
    )

# Add for missing required parameters:
if not calendar_name:
    raise ToolError(
        "Missing required parameter: calendar_name",
        ErrorCode.TOOL_PARAMETER_MISSING,
        {"missing_parameter": "calendar_name"}
    )

# Add for calendar sync issues:
except caldav.error.DAVError as e:
    raise ToolError(
        f"Failed to synchronize with calendar: {str(e)}",
        ErrorCode.CALENDAR_SYNC_ERROR,
        {"calendar": calendar_name, "error": str(e)}
    )
```

#### Maps Tool (tools/maps_tool.py)
```python
# For geocoding errors:
if not result:
    raise ToolError(
        f"Could not geocode address: {address}",
        ErrorCode.GEOLOCATION_ERROR,
        {"address": address}
    )

# For type errors:
if not isinstance(latitude, (int, float)):
    raise ToolError(
        f"Latitude must be a number, got {type(latitude).__name__}",
        ErrorCode.TOOL_DATA_TYPE_ERROR,
        {"parameter": "latitude", "value": latitude, "expected_type": "number"}
    )

# For value range errors:
if latitude < -90 or latitude > 90:
    raise ToolError(
        f"Latitude out of range: {latitude}. Must be between -90 and 90",
        ErrorCode.TOOL_DATA_RANGE_ERROR,
        {"parameter": "latitude", "value": latitude, "valid_range": "[-90, 90]"}
    )
```

### Database Operations

#### Customer Database Tool (tools/customerdatabase_tool.py)
```python
# For database connection issues:
try:
    conn = sqlite3.connect(self.db_path)
except sqlite3.Error as e:
    raise ToolError(
        f"Failed to connect to customer database: {str(e)}",
        ErrorCode.DATABASE_CONNECTION_ERROR,
        {"db_path": self.db_path, "error": str(e)}
    )

# For query errors:
try:
    cursor.execute(query, params)
except sqlite3.Error as e:
    raise ToolError(
        f"Database query error: {str(e)}",
        ErrorCode.DATABASE_QUERY_ERROR,
        {"query": query, "params": params, "error": str(e)}
    )

# For constraint violations:
except sqlite3.IntegrityError as e:
    if "UNIQUE constraint failed" in str(e):
        raise ToolError(
            f"Customer with this information already exists: {str(e)}",
            ErrorCode.DATABASE_CONSTRAINT_ERROR,
            {"customer_info": customer_info, "error": str(e)}
        )
    else:
        raise ToolError(
            f"Database constraint violation: {str(e)}",
            ErrorCode.DATABASE_CONSTRAINT_ERROR,
            {"customer_info": customer_info, "error": str(e)}
        )
```

### Workflow Management

#### Workflow Manager (tools/workflows/workflow_manager.py)
```python
# For workflow not found:
if workflow_id not in self.workflows:
    raise ToolError(
        f"Workflow with ID '{workflow_id}' doesn't exist",
        ErrorCode.WORKFLOW_NOT_FOUND,
        {"workflow_id": workflow_id, "available_workflows": list(self.workflows.keys())}
    )

# For invalid state transitions:
if self.active_step_index >= len(workflow["steps"]):
    raise ToolError(
        f"Invalid workflow step index: {self.active_step_index}",
        ErrorCode.WORKFLOW_INVALID_STATE,
        {"workflow_id": self.active_workflow_id, "max_steps": len(workflow["steps"])}
    )

# For step execution errors:
try:
    # Execute step logic
except Exception as e:
    raise ToolError(
        f"Error executing workflow step: {str(e)}",
        ErrorCode.WORKFLOW_STEP_ERROR,
        {"workflow_id": self.active_workflow_id, "step_index": self.active_step_index}
    )

# For workflow definition validation:
if not self._validate_workflow(workflow):
    raise ToolError(
        f"Invalid workflow definition for {workflow.get('id', 'unknown')}",
        ErrorCode.WORKFLOW_VALIDATION_ERROR,
        {"workflow": workflow}
    )

# For tool dependency errors:
for tool_name in tools_for_current_step:
    if not self.tool_repo.has_tool(tool_name):
        raise ToolError(
            f"Workflow depends on missing tool: {tool_name}",
            ErrorCode.WORKFLOW_DEPENDENCY_ERROR,
            {"workflow_id": self.active_workflow_id, "missing_tool": tool_name}
        )
```

### Service-Specific Tools

#### Square Tool (tools/square_tool.py)
```python
# For authentication issues:
except square.exceptions.ApiException as e:
    if e.status == 401:
        raise ToolError(
            "Square API authentication error",
            ErrorCode.API_AUTHENTICATION_ERROR,
            {"status": e.status, "error": str(e)}
        )
    elif e.status == 403:
        raise ToolError(
            "Insufficient permissions for Square API operation",
            ErrorCode.TOOL_INSUFFICIENT_PERMISSIONS,
            {"status": e.status, "error": str(e)}
        )
    else:
        raise ToolError(
            f"Square API error: {str(e)}",
            ErrorCode.API_RESPONSE_ERROR,
            {"status": e.status, "error": str(e)}
        )
```

#### Reminder Tool (tools/reminder_tool.py)
```python
# For validation errors:
if not self._validate_date_format(reminder_date):
    raise ToolError(
        f"Invalid date format: {reminder_date}. Expected YYYY-MM-DD",
        ErrorCode.TOOL_DATA_FORMAT_ERROR,
        {"parameter": "reminder_date", "value": reminder_date, "expected_format": "YYYY-MM-DD"}
    )

# For database errors:
try:
    conn = sqlite3.connect(self.db_path)
except sqlite3.Error as e:
    raise ToolError(
        f"Failed to connect to reminder database: {str(e)}",
        ErrorCode.DATABASE_CONNECTION_ERROR,
        {"db_path": self.db_path, "error": str(e)}
    )
```

## Migration Strategy

1. **Update errors.py first**: Add all new error codes to the ErrorCode enum
2. **Create database error class**: Add a new DatabaseError class that inherits from AgentError
3. **Create workflow error class**: Add a new WorkflowError class that inherits from AgentError
4. **Update tools one by one**: Start with HTTP and Email tools that have the most network interactions
5. **Add integration tests**: Create tests to verify error handling is consistent
6. **Update documentation**: Ensure all new error codes are documented properly

By implementing these more specific error codes, the system will provide better diagnostics, enable more targeted error handling, and improve overall reliability.