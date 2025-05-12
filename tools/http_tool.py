import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
import requests
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from tools.repo import Tool
from errors import ErrorCode, error_context, ToolError
from config.registry import registry

# Define configuration class for HTTPTool
class HttpToolConfig(BaseModel):
    """Configuration for the http_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    timeout: int = Field(default=30, description="Timeout in seconds for HTTP requests")
    max_timeout: int = Field(default=120, description="Maximum timeout allowed for HTTP requests in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")
    allowed_domains: List[str] = Field(default=[], description="List of allowed domains for requests (empty for all)")

# Register with registry
registry.register("http_tool", HttpToolConfig)


class HTTPTool(Tool):
    """
    HTTP request tool for making API calls and web requests.
    
    This tool allows for performing HTTP requests to external services and APIs.
    It supports the standard HTTP methods (GET, POST, PUT, DELETE) and provides
    flexible control over request parameters, headers, and response formatting.
    """
    
    name = "http_tool"
    simple_description = """
    Makes HTTP requests to external APIs and web services. This tool lets you directly interact with 
    web APIs and services by sending HTTP requests with various methods, parameters, and headers. Use this tool when you need to contact a remote server that is not handled by another tool AND you know the correct API format for the service. If you are unsure of the format but confident that you need this tool please ask the user for the proper format and then try again.
    """
    
    anthropic_details = """

    OPERATIONS:
    - GET: Retrieve data from a specified URL
      Parameters:
        url (required): The URL to send the request to
        params (optional): Query parameters as a dictionary
        headers (optional): HTTP headers as a dictionary
        timeout (optional, default=30): Request timeout in seconds
        response_format (optional, default="json"): Format to return the response in ("json", "text", or "full")

    - POST: Send data to a specified URL
      Parameters:
        url (required): The URL to send the request to
        data (optional): Form data to send (as a string or dictionary)
        json (optional): JSON data to send (as a dictionary)
        params (optional): Query parameters as a dictionary
        headers (optional): HTTP headers as a dictionary
        timeout (optional, default=30): Request timeout in seconds
        response_format (optional, default="json"): Format to return the response in ("json", "text", or "full")

    - PUT: Update data at a specified URL
      Parameters:
        url (required): The URL to send the request to
        data (optional): Form data to send (as a string or dictionary)
        json (optional): JSON data to send (as a dictionary)
        params (optional): Query parameters as a dictionary
        headers (optional): HTTP headers as a dictionary
        timeout (optional, default=30): Request timeout in seconds
        response_format (optional, default="json"): Format to return the response in ("json", "text", or "full")

    - DELETE: Delete data at a specified URL
      Parameters:
        url (required): The URL to send the request to
        params (optional): Query parameters as a dictionary
        headers (optional): HTTP headers as a dictionary
        timeout (optional, default=30): Request timeout in seconds
        response_format (optional, default="json"): Format to return the response in ("json", "text", or "full")

    RESPONSE FORMAT OPTIONS:
    - "json": Automatically parse and return the JSON response (default)
    - "text": Return the raw text response
    - "full": Return a comprehensive response object with status, headers, and body

    USAGE NOTES:
    - Always validate the URL before sending sensitive information
    - Use appropriate headers for authentication (e.g., Authorization header)
    - For GET requests, use params to send query parameters
    - For POST/PUT requests, use either data (for form data) or json (for JSON data)
    - The response_format parameter controls how the response is returned
    - Check the status_code in the response to verify success (200-299 is success)

    LIMITATIONS:
    - Cannot make requests to internal network addresses (security restriction)
    - File uploads are not supported in the current version
    - Redirects are followed by default (up to 5)
    - Cookie persistence is not maintained between requests
    - Binary responses are not supported (images, files, etc.)
    """

    openai_schema = {
        "type": "function",
        "function": {
            "name": "http_tool",
            "description": "Makes HTTP requests to external APIs and web services. Use when you need to directly interact with web APIs and services.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "description": "HTTP method to use for the request"
                    },
                    "url": {
                        "type": "string",
                        "description": "The URL to send the request to"
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters as key-value pairs"
                    },
                    "headers": {
                        "type": "object",
                        "description": "HTTP headers as key-value pairs"
                    },
                    "data": {
                        "type": ["object", "string"],
                        "description": "Form data to send (as a string or object)"
                    },
                    "json": {
                        "type": "object",
                        "description": "JSON data to send in the request body"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (default: 30)"
                    },
                    "response_format": {
                        "type": "string",
                        "enum": ["json", "text", "full"],
                        "description": "Format to return the response in"
                    }
                },
                "required": ["method", "url"]
            }
        }
    }
    
    usage_examples = [
        {
            "input": {"method": "GET", "url": "https://api.example.com/data", "params": {"key": "value"}},
            "output": {
                "status_code": 200,
                "success": True,
                "data": {"example": "response"}
            }
        },
        {
            "input": {"method": "POST", "url": "https://api.example.com/create", "json": {"name": "Test"}, "headers": {"Authorization": "Bearer token"}},
            "output": {
                "status_code": 201,
                "success": True,
                "data": {"id": 123, "created": True}
            }
        }
    ]

    def __init__(self):
        """
        Initialize the HTTP tool with configuration and setup.
        """
        super().__init__()
        self.logger.info("HTTPTool initialized")
        
        # List of blocked URL patterns for security
        self._blocked_url_patterns = [
            r'^https?://localhost',
            r'^https?://127\.',
            r'^https?://10\.',
            r'^https?://172\.(1[6-9]|2[0-9]|3[0-1])\.',
            r'^https?://192\.168\.',
            r'^https?://0\.0\.0\.0',
        ]
        
        # Default timeout and max timeout will be loaded from config when needed

    def run(self, **params) -> Dict[str, Any]:
        """
        Execute an HTTP request with the specified parameters.

        This is the main entry point for the HTTP tool. It validates the inputs,
        prepares and sends the HTTP request, and formats the response according
        to the specified format.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: The URL to send the request to
            params: Optional query parameters as a dictionary
            headers: Optional HTTP headers as a dictionary
            data: Optional data to send (for POST/PUT)
            json: Optional JSON data to send (for POST/PUT)
            timeout: Optional request timeout in seconds (default: 30)
            response_format: Format to return the response in (json, text, full)

        Returns:
            Dictionary containing the response with the following structure:
            {
                "success": bool,
                "status_code": int,
                "data": Any,  # Parsed JSON, text, or full response object
                "headers": Dict[str, str],  # Only included if response_format="full"
                "url": str,  # The final URL (may differ from request URL due to redirects)
                "error": str  # Only included if an error occurred
            }

        Raises:
            ToolError: If inputs are invalid or if the request fails
        """
        # Extract expected parameters
        method = params.get("method")
        url = params.get("url")
        request_params = params.get("params")
        headers = params.get("headers")
        data = params.get("data")
        json_data = params.get("json")
        timeout = params.get("timeout")
        response_format = params.get("response_format", "json")

        # Validate required parameters
        if not method or not url:
            raise ToolError(
                "Required parameters 'method' and 'url' must be provided",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_params": list(params.keys())}
            )

        # Import config when needed (avoids circular imports)
        from config import config
        
        self.logger.info(f"Executing {method} request to {url}")

        # Use the main error context for the entire operation
        with error_context(
            component_name=self.name,
            operation=f"executing {method} request",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Input validation
            self._validate_inputs(method, url, response_format, timeout)

            # Set default timeout if not provided
            if timeout is None:
                timeout = config.http_tool.timeout

            # Prepare the request
            method = str(method).upper()
            headers = headers or {}
            request_params = request_params or {}
            
            # Attempt to execute the request
            try:
                response = requests.request(
                    method=str(method).upper(),
                    url=str(url),
                    params=request_params,
                    headers=headers,
                    data=data,
                    json=json_data,
                    timeout=timeout,
                    allow_redirects=True
                )
                
                # Log response info
                self.logger.debug(f"Response received: Status {response.status_code}")
                
                # Format and return the response
                return self._format_response(response, response_format)
                
            except requests.exceptions.Timeout:
                self.logger.error(f"Request to {url} timed out after {timeout} seconds")
                raise ToolError(
                    f"Request timed out after {timeout} seconds",
                    ErrorCode.API_TIMEOUT_ERROR,
                    {"url": url, "method": method, "timeout": timeout}
                )
                
            except requests.exceptions.ConnectionError as e:
                self.logger.error(f"Connection error for {url}: {str(e)}")
                raise ToolError(
                    f"Connection error: {str(e)}",
                    ErrorCode.API_CONNECTION_ERROR,
                    {"url": url, "method": method, "error": str(e)}
                )
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request error for {url}: {str(e)}")
                raise ToolError(
                    f"Request error: {str(e)}",
                    ErrorCode.API_RESPONSE_ERROR,
                    {"url": url, "method": method, "error": str(e)}
                )
    
    def _validate_inputs(self, method, url, response_format, timeout):
        """
        Validate input parameters before executing the request.
        
        Performs comprehensive validation of all input parameters to ensure
        they are valid and safe for execution.
        
        Args:
            method: HTTP method string
            url: URL string
            response_format: Response format string
            timeout: Timeout value in seconds
            
        Raises:
            ToolError: If any inputs are invalid
        """
        # Validate HTTP method
        valid_methods = ["GET", "POST", "PUT", "DELETE"]
        if not method or method.upper() not in valid_methods:
            raise ToolError(
                f"Invalid HTTP method: {method}. Must be one of: {', '.join(valid_methods)}",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_method": method, "valid_methods": valid_methods}
            )
        
        # Validate URL format
        if not url or not isinstance(url, str):
            raise ToolError(
                "URL must be a non-empty string",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_url": str(url)}
            )
            
        # Check URL scheme
        parsed_url = urlparse(url)
        if not parsed_url.scheme or parsed_url.scheme not in ["http", "https"]:
            raise ToolError(
                f"Invalid URL scheme: {parsed_url.scheme}. Must be http or https",
                ErrorCode.TOOL_INVALID_INPUT,
                {"url": url, "scheme": parsed_url.scheme}
            )
            
        # Security check - validate against blocked URL patterns
        for pattern in self._blocked_url_patterns:
            if re.match(pattern, url, re.IGNORECASE):
                raise ToolError(
                    "URL is restricted for security reasons (internal/private network)",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"url": url}
                )
                
        # Validate response format
        valid_formats = ["json", "text", "full"]
        if response_format not in valid_formats:
            raise ToolError(
                f"Invalid response format: {response_format}. Must be one of: {', '.join(valid_formats)}",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_format": response_format, "valid_formats": valid_formats}
            )
            
        # Validate timeout
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ToolError(
                    "Timeout must be a positive number",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"provided_timeout": timeout}
                )
                
            # Get max timeout from config
            max_timeout = 120  # Default fallback
            try:
                from config import config
                max_timeout = config.http_tool.max_timeout if hasattr(config.http_tool, 'max_timeout') else 120
            except Exception:
                # If config access fails, use default
                pass
                
            if timeout > max_timeout:
                raise ToolError(
                    f"Timeout value exceeds maximum allowed ({max_timeout} seconds)",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"provided_timeout": timeout, "max_timeout": max_timeout}
                )
    
    def _format_response(self, response, response_format):
        """
        Format the HTTP response according to the specified format.
        
        Args:
            response: The requests.Response object
            response_format: The format to return ("json", "text", or "full")
            
        Returns:
            Formatted response dictionary
        """
        result = {
            "success": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "url": response.url
        }
        
        # Format based on specified format
        if response_format == "json":
            try:
                # Try to parse as JSON
                result["data"] = response.json()
            except ValueError:
                # If parsing fails, include text and a warning
                result["data"] = response.text
                result["warning"] = "Response could not be parsed as JSON"
                
        elif response_format == "text":
            result["data"] = response.text
            
        elif response_format == "full":
            result["data"] = response.text
            result["headers"] = dict(response.headers)
            
            # Try to include JSON if response is JSON
            try:
                result["json"] = response.json()
            except ValueError:
                # Not JSON, so we don't include it
                pass
        
        return result