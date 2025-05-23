"""
Web access tool that combines HTTP requests, web search, and webpage extraction.

This tool provides a comprehensive web interaction interface, allowing the system to:
1. Make HTTP requests to external APIs and web services
2. Perform web searches using Anthropic's built-in web search API
3. Extract and parse content from webpages using Claude's AI capabilities
"""
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

# Define configuration class for WebAccessTool
class WebAccessConfig(BaseModel):
    """Configuration for the webaccess_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    timeout: int = Field(default=30, description="Timeout in seconds for HTTP requests")
    max_timeout: int = Field(default=120, description="Maximum timeout allowed for HTTP requests in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")
    allowed_domains: List[str] = Field(default=[], description="List of allowed domains for requests (empty for all)")
    blocked_domains: List[str] = Field(default=[], description="List of domains to exclude from results")
    max_searches_per_request: int = Field(default=3, description="Maximum number of searches allowed per request")
    default_extraction_prompt: str = Field(
        default="Please extract the main content from this webpage. Focus on the article text, headings, and important information. Ignore navigation, ads, footers, and other non-essential elements.",
        description="Default prompt to use for content extraction"
    )

# Register with registry
registry.register("webaccess_tool", WebAccessConfig)


class WebAccessTool(Tool):
    """
    Web access tool that combines HTTP requests, web search, and webpage extraction.
    
    This integrated tool provides three main capabilities:
    1. HTTP Tool: Make direct HTTP requests to external APIs and web services
    2. Web Search Tool: Perform web searches using Anthropic's built-in search API
    3. Webpage Extraction Tool: Extract and parse content from webpages
    
    Having these capabilities combined in one tool allows for more efficient web
    interaction workflows, such as searching for information, accessing specific
    URLs found in search results, and extracting content from those pages.
    """
    
    name = "webaccess_tool"
    simple_description = """
    Provides comprehensive web access capabilities including HTTP requests, web searches, and webpage content extraction.
    This integrated tool allows you to interact with web resources through direct HTTP requests to APIs,
    perform web searches for up-to-date information, and extract clean content from webpages.
    Use this tool whenever you need to access, search, or extract information from the web.
    """
    
    implementation_details = """
    This tool combines three web access capabilities:
    
    1. HTTP REQUEST FUNCTIONALITY:
       Makes HTTP requests to external APIs and web services with various methods, parameters, and headers.
       
       Operations:
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
    
       - PUT: Update data at a specified URL (parameters same as POST)
    
       - DELETE: Delete data at a specified URL (parameters similar to GET)
    
    2. WEB SEARCH FUNCTIONALITY:
       Performs web searches to find up-to-date information from the internet.
       
       Operations:
       - Search the web for information
         Parameters:
           query (required): The search query to send to the search engine
           max_results (optional, default=3): Maximum number of results to return
           allowed_domains (optional): List of domains to include in results
           blocked_domains (optional): List of domains to exclude from results
    
    3. WEBPAGE EXTRACTION FUNCTIONALITY:
       Extracts content from webpages using Claude's understanding of web content.
       
       Operations:
       - Extract content from a webpage at a given URL
         Parameters:
           url (required): The URL of the webpage to extract content from
           extraction_prompt (optional): Custom prompt to guide the extraction (default focuses on main content)
           format (optional, default="text"): Output format - "text", "markdown", or "html"
           include_metadata (optional, default=False): Whether to include page metadata in the output
           timeout (optional, default=30): Request timeout in seconds
    
    USAGE NOTES:
    - Use http_request for direct API calls when you know the specific endpoint and parameters
    - Use web_search when you need to find information but don't have a specific URL
    - Use webpage_extract when you have a URL but need to get clean, readable content from it
    - These capabilities can be used together: search for content, then extract from the best result URLs
    
    LIMITATIONS:
    - Cannot make requests to internal network addresses (security restriction)
    - Search quality depends on query formulation
    - Very large pages may exceed processing limits
    - Some websites may block automated access
    - JavaScript-rendered content may not be fully captured
    """
    
    description = simple_description + implementation_details
    
    openai_schema = {
        "type": "function",
        "function": {
            "name": "webaccess_tool",
            "description": "Provides comprehensive web access with HTTP requests, web searches, and webpage extraction",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["http_request", "web_search", "webpage_extract"],
                        "description": "The web access operation to perform"
                    },
                    # HTTP request parameters
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "description": "HTTP method for http_request operation"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL for http_request or webpage_extract operations"
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters for http_request operation"
                    },
                    "headers": {
                        "type": "object",
                        "description": "HTTP headers for http_request operation"
                    },
                    "data": {
                        "type": ["object", "string"],
                        "description": "Form data for http_request operation"
                    },
                    "json": {
                        "type": "object",
                        "description": "JSON data for http_request operation"
                    },
                    "response_format": {
                        "type": "string",
                        "enum": ["json", "text", "full"],
                        "description": "Response format for http_request operation"
                    },
                    # Web search parameters
                    "query": {
                        "type": "string",
                        "description": "Search query for web_search operation"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results for web_search operation"
                    },
                    # Webpage extraction parameters
                    "extraction_prompt": {
                        "type": "string",
                        "description": "Content extraction prompt for webpage_extract operation"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["text", "markdown", "html"],
                        "description": "Output format for webpage_extract operation"
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Whether to include page metadata in webpage_extract operation"
                    },
                    # Shared parameters
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds"
                    },
                    "allowed_domains": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of domains to include in results"
                    },
                    "blocked_domains": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of domains to exclude from results"
                    }
                },
                "required": ["operation"]
            }
        }
    }
    
    usage_examples = [
        {
            "input": {
                "operation": "http_request", 
                "method": "GET", 
                "url": "https://api.example.com/data", 
                "params": {"key": "value"}
            },
            "output": {
                "success": True,
                "status_code": 200,
                "data": {"example": "response"}
            }
        },
        {
            "input": {
                "operation": "web_search", 
                "query": "latest Mars rover discoveries"
            },
            "output": {
                "success": True,
                "results": [
                    {
                        "title": "NASA's Perseverance Mars Rover Makes New Discovery",
                        "url": "https://example.com/mars-rover-news",
                        "content": "Example content about Mars rover discoveries..."
                    }
                ]
            }
        },
        {
            "input": {
                "operation": "webpage_extract", 
                "url": "https://example.com/article", 
                "format": "markdown"
            },
            "output": {
                "success": True,
                "url": "https://example.com/article",
                "content": "# Article Title\n\nExample extracted content in markdown format..."
            }
        }
    ]

    def __init__(self):
        """
        Initialize the web access tool with configuration and setup.
        """
        super().__init__()
        self.logger.info("WebAccessTool initialized")
        
        # List of blocked URL patterns for security
        self._blocked_url_patterns = [
            r'^https?://localhost',
            r'^https?://127\.',
            r'^https?://10\.',
            r'^https?://172\.(1[6-9]|2[0-9]|3[0-1])\.',
            r'^https?://192\.168\.',
            r'^https?://0\.0\.0\.0',
        ]

    def run(self, **params) -> Dict[str, Any]:
        """
        Execute a web access operation based on the specified parameters.

        This is the main entry point for the web access tool. It routes to the
        appropriate operation handler based on the 'operation' parameter.

        Args:
            operation: The operation to perform ('http_request', 'web_search', or 'webpage_extract')
            [Other parameters depend on the operation chosen]

        Returns:
            Dictionary containing the result of the operation (structure depends on operation)

        Raises:
            ToolError: If inputs are invalid or if the operation fails
        """
        # Extract operation
        operation = params.get("operation")
        
        # Validate operation parameter
        if not operation:
            raise ToolError(
                "Required parameter 'operation' must be provided",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_params": list(params.keys())}
            )
            
        # Route to appropriate handler based on operation
        if operation == "http_request":
            return self._handle_http_request(params)
        elif operation == "web_search":
            return self._handle_web_search(params)
        elif operation == "webpage_extract":
            return self._handle_webpage_extract(params)
        else:
            raise ToolError(
                f"Invalid operation: {operation}. Must be one of: http_request, web_search, webpage_extract",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_operation": operation, "valid_operations": ["http_request", "web_search", "webpage_extract"]}
            )

    def _make_http_request(self, method, url, params=None, headers=None, data=None, json_data=None, 
                       timeout=None, is_browser=False, max_content_size=10*1024*1024,
                       retries=None, retry_status_codes=None, retry_delay=1.0):
        """
        Shared function for making HTTP requests.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: The URL to send the request to
            params: Optional query parameters as a dictionary
            headers: Optional HTTP headers as a dictionary
            data: Optional form data (for POST/PUT)
            json_data: Optional JSON data (for POST/PUT)
            timeout: Request timeout in seconds
            is_browser: Whether to use browser-like headers
            max_content_size: Maximum content size in bytes (default 10MB)
            retries: Number of retries for transient errors (default from config)
            retry_status_codes: HTTP status codes to retry on (default: 429, 500, 502, 503, 504)
            retry_delay: Base delay between retries in seconds (will increase exponentially)
            
        Returns:
            requests.Response object
            
        Raises:
            ToolError: If the request fails after all retries
        """
        self.logger.debug(f"Making {method} request to {url}")
        
        # Set default headers for browser-like requests
        if is_browser and (headers is None or len(headers) == 0):
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
        elif headers is None:
            headers = {}
            
        # Merge with any provided headers
        if headers and is_browser:
            browser_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            }
            if "User-Agent" not in headers:
                headers.update(browser_headers)
        
        # Get configuration for retries
        from config import config
        
        # Set up retry configuration
        if retries is None:
            try:
                retries = config.webaccess_tool.max_retries
            except AttributeError:
                retries = 3  # Default to 3 retries
                
        if retry_status_codes is None:
            retry_status_codes = [429, 500, 502, 503, 504]
        
        # Use the error context for HTTP requests
        operation_description = f"executing {method} request to {url}"
        with error_context(
            component_name=self.name,
            operation=operation_description,
            error_class=ToolError,
            error_code=ErrorCode.API_CONNECTION_ERROR,
            logger=self.logger
        ):
            # Initialize variables for retry logic
            attempts = 0
            last_exception = None
            
            while attempts <= retries:
                attempts += 1
                try:
                    # Use streaming for large responses to avoid memory issues
                    with requests.request(
                        method=method,
                        url=url,
                        params=params,
                        headers=headers,
                        data=data,
                        json=json_data,
                        timeout=timeout,
                        allow_redirects=True,
                        stream=True  # Use streaming mode
                    ) as response:
                        # Check if we got a status code that should trigger a retry
                        if response.status_code in retry_status_codes and attempts <= retries:
                            # Get retry-after header if it exists (used by many APIs for rate limiting)
                            retry_after = response.headers.get('Retry-After')
                            if retry_after:
                                try:
                                    # Try to parse as integer seconds
                                    wait_time = float(retry_after)
                                except ValueError:
                                    # Default to exponential backoff
                                    wait_time = retry_delay * (2 ** (attempts - 1))
                            else:
                                # Use exponential backoff
                                wait_time = retry_delay * (2 ** (attempts - 1))
                                
                            self.logger.warning(
                                f"Request to {url} returned status {response.status_code}, "
                                f"retrying in {wait_time:.1f} seconds (attempt {attempts}/{retries})"
                            )
                            
                            # Small jitter to avoid thundering herd problems
                            import random
                            import time
                            time.sleep(wait_time + random.uniform(0, 0.5))
                            continue
                    # Log response info
                    self.logger.debug(f"Response received: Status {response.status_code}")
                    
                    # Only raise for status if specified (we may want to handle errors differently in some cases)
                    response.raise_for_status()
                    
                    # Check content length header
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > max_content_size:
                        raise ToolError(
                            f"Content too large: {int(content_length) // (1024*1024)}MB exceeds limit of {max_content_size // (1024*1024)}MB",
                            ErrorCode.API_RESPONSE_ERROR,
                            {"url": url, "content_length": content_length, "max_size": max_content_size}
                        )
                    
                    # Load content in chunks to avoid memory issues
                    content = bytearray()
                    total_size = 0
                    
                    for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                        if not chunk:
                            continue
                        
                        total_size += len(chunk)
                        if total_size > max_content_size:
                            # Clear any partial content to free memory
                            content = None
                            # Immediately close the response
                            response.close()
                            
                            raise ToolError(
                                f"Content too large: exceeds limit of {max_content_size // (1024*1024)}MB",
                                ErrorCode.API_RESPONSE_ERROR,
                                {"url": url, "max_size": max_content_size}
                            )
                        
                        content.extend(chunk)
                    
                    # Create a copy of the response with the full content loaded
                    response._content = bytes(content)
                    response._content_consumed = True
                    
                    return response
                except (requests.exceptions.Timeout,
                       requests.exceptions.ConnectionError,
                       requests.exceptions.HTTPError,
                       requests.exceptions.RequestException) as e:
                    # Save the exception for potential re-raising
                    last_exception = e
                    
                    # Determine if this exception should trigger a retry
                    should_retry = False
                    
                    # Check exception type to determine if it's retryable
                    if isinstance(e, requests.exceptions.Timeout) or isinstance(e, requests.exceptions.ConnectionError):
                        # Network-related errors are generally retryable
                        should_retry = True
                    elif isinstance(e, requests.exceptions.HTTPError) and hasattr(e, 'response'):
                        # Retry specific HTTP status codes
                        if e.response.status_code in retry_status_codes:
                            should_retry = True
                    
                    # Check if we should retry
                    if should_retry and attempts < retries:
                        # Calculate backoff time
                        wait_time = retry_delay * (2 ** (attempts - 1))
                        
                        self.logger.warning(
                            f"Request to {url} failed with {type(e).__name__}, "
                            f"retrying in {wait_time:.1f} seconds (attempt {attempts}/{retries}). "
                            f"Error: {str(e)}"
                        )
                        
                        # Add jitter to avoid thundering herd problems
                        import random
                        import time
                        time.sleep(wait_time + random.uniform(0, 0.5))
                        continue
                    
                    # If we reach here, we either exhausted retries or the exception isn't retryable
                    if isinstance(e, requests.exceptions.Timeout):
                        self.logger.error(f"Request to {url} timed out after {timeout} seconds (after {attempts} attempts)")
                        raise ToolError(
                            f"Request timed out after {timeout} seconds",
                            ErrorCode.API_TIMEOUT_ERROR,
                            {"url": url, "method": method, "timeout": timeout, "attempts": attempts}
                        )
                    elif isinstance(e, requests.exceptions.ConnectionError):
                        self.logger.error(f"Connection error for {url}: {str(e)} (after {attempts} attempts)")
                        raise ToolError(
                            f"Connection error: {str(e)}",
                            ErrorCode.API_CONNECTION_ERROR,
                            {"url": url, "method": method, "error": str(e), "attempts": attempts}
                        )
                    elif isinstance(e, requests.exceptions.HTTPError):
                        status_code = e.response.status_code if hasattr(e, 'response') and hasattr(e.response, 'status_code') else "unknown"
                        self.logger.error(f"HTTP error {status_code} for {url}: {str(e)} (after {attempts} attempts)")
                        raise ToolError(
                            f"HTTP error {status_code}: {str(e)}",
                            ErrorCode.API_RESPONSE_ERROR,
                            {"url": url, "method": method, "status_code": status_code, "error": str(e), "attempts": attempts}
                        )
                    else:
                        self.logger.error(f"Request error for {url}: {str(e)} (after {attempts} attempts)")
                        raise ToolError(
                            f"Request error: {str(e)}",
                            ErrorCode.API_RESPONSE_ERROR,
                            {"url": url, "method": method, "error": str(e), "attempts": attempts}
                        )
            # If we've exhausted all retries without success
            if last_exception:
                self.logger.error(f"Request to {url} failed after {retries} retries")
                raise ToolError(
                    f"Request failed after {retries} retries: {str(last_exception)}",
                    ErrorCode.API_RESPONSE_ERROR,
                    {"url": url, "method": method, "error": str(last_exception), "attempts": attempts}
                )

    def _handle_http_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle HTTP request operations.
        
        This function handles direct HTTP requests to external APIs and services.
        
        Args:
            params: Dictionary of parameters for the HTTP request
            
        Returns:
            Dictionary containing the HTTP response
            
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
                "Required parameters 'method' and 'url' must be provided for http_request operation",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_params": list(params.keys())}
            )

        self.logger.info(f"Executing HTTP {method} request to {url}")

        # Use the main error context for the entire operation
        with error_context(
            component_name=self.name,
            operation=f"executing HTTP {method} request",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Input validation
            self._validate_http_inputs(method, url, response_format, timeout)

            # Set default timeout if not provided
            timeout = self._validate_timeout(timeout)

            # Prepare the request
            method = str(method).upper()
            
            # Use shared HTTP request function
            try:
                response = self._make_http_request(
                    method=method,
                    url=url,
                    params=request_params,
                    headers=headers,
                    data=data,
                    json_data=json_data,
                    timeout=timeout
                )
                
                # Format and return the response
                return self._format_http_response(response, response_format)
                
            except ToolError:
                # Re-raise ToolErrors without modification since they're already formatted
                raise
                
    def _validate_http_inputs(self, method, url, response_format, timeout):
        """
        Validate input parameters before executing an HTTP request.
        
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
        
        # Validate URL with operation context for domain restrictions
        self._validate_url(url, operation="http_request")
            
        # Validate response format
        valid_formats = ["json", "text", "full"]
        if response_format not in valid_formats:
            raise ToolError(
                f"Invalid response format: {response_format}. Must be one of: {', '.join(valid_formats)}",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_format": response_format, "valid_formats": valid_formats}
            )
            
        # Timeout validation is handled by the shared _validate_timeout method
        
    def _format_http_response(self, response, response_format):
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
        
        # Detect content type
        content_info = self._detect_content_type(response=response)
        result["content_type"] = f"{content_info['mimetype']}/{content_info['subtype']}"
        
        # Format based on specified format
        if response_format == "json":
            try:
                # Use response's built-in JSON parser first for efficiency
                result["data"] = response.json()
            except ValueError:
                # If built-in parsing fails, check if content appears to be JSON
                if content_info["format"] == "json":
                    # If content type detection suggests JSON, use our robust parser
                    try:
                        result["data"] = self._parse_json_response(
                            response.text, 
                            expected_format=None  # Auto-detect format
                        )
                    except Exception:
                        # If that also fails, include the raw text
                        result["data"] = response.text
                        result["warning"] = "Response could not be parsed as JSON despite Content-Type"
                else:
                    # Not JSON content type
                    result["data"] = response.text
                    result["warning"] = "Response could not be parsed as JSON"
                
        elif response_format == "text":
            result["data"] = response.text
            
        elif response_format == "full":
            result["data"] = response.text
            result["headers"] = dict(response.headers)
            result["content_info"] = content_info
            
            # Try to include JSON if content might be JSON
            if content_info["format"] == "json":
                try:
                    # Use our robust parser for the JSON field
                    result["json"] = self._parse_json_response(
                        response.text,
                        expected_format=None  # Auto-detect format
                    )
                except Exception:
                    # If parsing fails, don't include JSON
                    pass
            else:
                # Try standard parsing for non-JSON content types that might contain JSON
                try:
                    result["json"] = response.json()
                except ValueError:
                    # Not JSON, so we don't include it
                    pass
        
        return result
        
    def _handle_web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle web search operations.
        
        This function performs web searches using Anthropic's search API.
        
        Args:
            params: Dictionary of parameters for the web search
            
        Returns:
            Dictionary containing the search results
            
        Raises:
            ToolError: If inputs are invalid or if the search fails
        """
        # Extract expected parameters
        query = params.get("query")
        max_results = params.get("max_results", 3)
        allowed_domains = params.get("allowed_domains", [])
        blocked_domains = params.get("blocked_domains", [])

        # Validate required parameters
        if not query:
            raise ToolError(
                "Required parameter 'query' must be provided for web_search operation",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_params": list(params.keys())}
            )

        self.logger.info(f"Executing web search for: {query}")

        # Use the main error context for the entire operation
        with error_context(
            component_name=self.name,
            operation=f"searching web for '{query}'",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Validate inputs
            self._validate_web_search_inputs(query, max_results, allowed_domains, blocked_domains)
            
            # Configure the web search tool for Claude
            web_search_tool = self._configure_web_search_tool(max_results, allowed_domains, blocked_domains)
            
            # Execute the search via the LLM
            search_results = self._execute_search(query, web_search_tool)
            
            # Process and return results
            return {
                "success": True,
                "results": search_results
            }
            
    def _validate_web_search_inputs(self, query, max_results, allowed_domains, blocked_domains):
        """
        Validate input parameters before executing the search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            allowed_domains: List of domains to include
            blocked_domains: List of domains to exclude
            
        Raises:
            ToolError: If any inputs are invalid
        """
        # Validate query
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            raise ToolError(
                "Search query must be a non-empty string",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_query": str(query)}
            )
        
        # Validate max_results
        if max_results is not None:
            if not isinstance(max_results, int) or max_results <= 0:
                raise ToolError(
                    "max_results must be a positive integer",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"provided_max_results": max_results}
                )
        
        # Get consolidated domains before validating them
        final_allowed, final_blocked = self._get_domain_restrictions(
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            operation="web_search"
        )
        
        # Validate domain restrictions using the shared domain validation method
        self._validate_domains(final_allowed, final_blocked)

    def _configure_web_search_tool(self, max_results, allowed_domains, blocked_domains):
        """
        Configure the web search tool for Claude.
        
        This creates the tool definition to pass to Anthropic's API.
        
        Args:
            max_results: Maximum number of searches to perform
            allowed_domains: List of domains to include
            blocked_domains: List of domains to exclude
            
        Returns:
            Dictionary with the web_search tool configuration
        """
        # Import config when needed
        from config import config
        
        # Get tool configuration
        web_access_config = None
        try:
            web_access_config = config.webaccess_tool
        except AttributeError:
            # Use defaults if not in config
            web_access_config = WebAccessConfig()
            
        # Build the web search tool definition
        web_search_tool = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": max_results or web_access_config.max_searches_per_request,
        }
        
        # Use consolidated domain management for web search
        final_allowed, final_blocked = self._get_domain_restrictions(
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            operation="web_search"
        )
        
        # Add domain filters if available
        if final_allowed:
            web_search_tool["allowed_domains"] = final_allowed
            
        if final_blocked:
            web_search_tool["blocked_domains"] = final_blocked
            
        return web_search_tool

    def _detect_content_type(self, response=None, content=None, headers=None):
        """
        Detect the content type of a response or raw content.
        
        This function analyzes response headers and/or content to determine
        the content type and format for better handling.
        
        Args:
            response: Optional requests.Response object
            content: Optional string content to analyze
            headers: Optional headers dictionary
            
        Returns:
            Dictionary with detected content properties:
            {
                "mimetype": Primary MIME type (e.g., "text", "application")
                "subtype": MIME subtype (e.g., "html", "json")
                "format": Detected format ("json", "html", "xml", "text", etc.)
                "encoding": Detected character encoding
                "is_binary": Boolean indicating if content appears to be binary
            }
        """
        result = {
            "mimetype": "text",
            "subtype": "plain",
            "format": "text",
            "encoding": "utf-8",
            "is_binary": False
        }
        
        # Check response and headers
        if response and response.headers:
            headers = response.headers
        
        # Extract content type from headers
        if headers:
            content_type = headers.get('Content-Type', '').lower()
            if content_type:
                # Parse the content type
                parts = content_type.split(';', 1)
                mimetype_full = parts[0].strip()
                
                # Extract encoding if present
                if len(parts) > 1 and 'charset=' in parts[1].lower():
                    encoding = parts[1].lower().split('charset=', 1)[1].strip()
                    result["encoding"] = encoding
                
                # Split mimetype into main type and subtype
                if '/' in mimetype_full:
                    mimetype, subtype = mimetype_full.split('/', 1)
                    result["mimetype"] = mimetype.strip()
                    result["subtype"] = subtype.strip()
                    
                    # Set format based on mimetype/subtype
                    if subtype in ['json', 'html', 'xml', 'javascript', 'css']:
                        result["format"] = subtype
                    elif 'json' in subtype:  # application/ld+json, etc.
                        result["format"] = "json"
                    elif mimetype == "application" and subtype in ["octet-stream", "pdf", "zip"]:
                        result["is_binary"] = True
                        result["format"] = subtype
                    elif mimetype in ["image", "audio", "video"]:
                        result["is_binary"] = True
                        result["format"] = mimetype
        
        # Analyze content if available and format not definitively determined
        if content and result["format"] == "text":
            # Check for JSON-like content
            if content.strip().startswith('{') and content.strip().endswith('}'):
                result["format"] = "json"
            elif content.strip().startswith('[') and content.strip().endswith(']'):
                result["format"] = "json"
            # Check for HTML-like content
            elif '<html' in content.lower() or '<!doctype html' in content.lower():
                result["format"] = "html"
                result["mimetype"] = "text"
                result["subtype"] = "html"
            # Check for XML-like content
            elif content.strip().startswith('<?xml'):
                result["format"] = "xml"
                result["mimetype"] = "text"
                result["subtype"] = "xml"
                
        return result
    
    def _parse_json_response(self, content, expected_format="list", required_fields=None):
        """
        Parse and validate JSON responses from APIs or LLMs.
        
        This function handles common JSON parsing issues and enforces format requirements.
        
        Args:
            content: String content to parse as JSON
            expected_format: Expected format of the parsed JSON ("list" or "dict")
            required_fields: List of required fields for dict items
            
        Returns:
            Parsed JSON object with validated format
            
        Raises:
            ToolError: If parsing fails or format validation fails
        """
        try:
            # Clean up content that might have markdown code blocks
            content = content.strip()
            if content.startswith("```json"):
                content = content.split("```json", 1)[1]
            elif content.startswith("```"):
                content = content.split("```", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
                
            # Try to parse as JSON
            parsed_content = json.loads(content)
            
            # Validate expected format
            if expected_format == "list" and not isinstance(parsed_content, list):
                self.logger.warning(f"Expected list format but got {type(parsed_content).__name__}")
                # Try to adapt non-list response
                if isinstance(parsed_content, dict):
                    # Extract values if it's a dict
                    parsed_content = list(parsed_content.values())
                else:
                    # Wrap any other type in a list
                    parsed_content = [parsed_content]
            elif expected_format == "dict" and not isinstance(parsed_content, dict):
                self.logger.warning(f"Expected dict format but got {type(parsed_content).__name__}")
                # Try to adapt non-dict response
                if isinstance(parsed_content, list) and len(parsed_content) > 0:
                    # Use first item if it's a list of dicts
                    if isinstance(parsed_content[0], dict):
                        parsed_content = parsed_content[0]
                    else:
                        # Create a generic wrapper
                        parsed_content = {"content": parsed_content}
                else:
                    # Create a generic wrapper for other types
                    parsed_content = {"content": parsed_content}
            
            # Validate required fields if any are specified
            if required_fields and isinstance(parsed_content, list):
                for i, item in enumerate(parsed_content):
                    if not isinstance(item, dict):
                        self.logger.warning(f"Item {i} is not a dictionary: {item}")
                        parsed_content[i] = {field: "N/A" for field in required_fields}
                        parsed_content[i]["content"] = str(item)
                        continue
                    
                    # Check for missing fields
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        self.logger.warning(f"Item {i} is missing fields {missing_fields}: {item}")
                        # Add missing fields with default values
                        for field in missing_fields:
                            item[field] = "N/A"
            
            return parsed_content
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.error(f"Raw content: {content}")
            
            # Return a fallback response based on expected format
            if expected_format == "list":
                if required_fields:
                    # Create a valid structure with required fields
                    return [{field: "N/A" for field in required_fields}]
                else:
                    return [{"content": content}]
            else:
                return {"content": content}
    
    def _get_llm_bridge(self):
        """
        Get or create an LLM bridge instance.
        
        This helper function centralizes LLM bridge acquisition logic.
        
        Returns:
            LLMBridge instance
        """
        from api.llm_bridge import LLMBridge
        
        # Create or get LLM bridge instance
        try:
            # Try to import llm_bridge from the system context first
            from main import get_system
            system = get_system()
            llm_bridge = system.get('llm_bridge')
            if not llm_bridge:
                llm_bridge = LLMBridge()
        except (ImportError, AttributeError):
            # If not available, create a new instance
            llm_bridge = LLMBridge()
            
        return llm_bridge
    
    def _execute_search(self, query, web_search_tool):
        """
        Execute the search using Anthropic's API.
        
        Args:
            query: The search query to execute
            web_search_tool: The configured web search tool
            
        Returns:
            List of search result objects
            
        Raises:
            ToolError: If the search fails or returns invalid results
        """
        # Get the LLM bridge instance
        llm_bridge = self._get_llm_bridge()
        
        # Create a system prompt that instructs Claude to search and return structured results
        system_prompt = """
        You are a helpful research assistant. Your task is to search the web for information 
        on the provided query and return the results in a structured format. Please follow these guidelines:
        
        - Use the web_search tool to find relevant information
        - For each search result, provide a summary of the key information
        - Include the source URL for each result
        - Focus on factual information and recent developments
        
        Please structure your response as a JSON array of objects with these properties:
        - title: A descriptive title for the search result
        - url: The source URL
        - content: A summary of the key information from this source
        
        Only return the JSON array, with no additional text or explanation.
        """
        
        # Prepare the user message with the search query
        user_message = f"Search for information about: {query}"
        
        # Use the error context for API interactions
        with error_context(
            component_name=self.name,
            operation=f"executing web search for '{query}'",
            error_class=ToolError,
            error_code=ErrorCode.API_RESPONSE_ERROR,
            logger=self.logger
        ):
            try:
                # Call the LLM with the web search tool
                messages = [{"role": "user", "content": user_message}]
                response = llm_bridge.generate_response(
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=[web_search_tool],
                    temperature=0.1,  # Low temperature for deterministic results
                )
                
                # Extract the text content
                result_content = llm_bridge.extract_text_content(response)
                
                # Parse the results using the shared JSON parsing function
                return self._parse_json_response(
                    result_content, 
                    expected_format="list",
                    required_fields=["title", "url", "content"]
                )
                
            except Exception as e:
                self.logger.error(f"Error executing web search: {str(e)}")
                raise ToolError(
                    f"Web search failed: {str(e)}",
                    ErrorCode.API_RESPONSE_ERROR,
                    {"error": str(e), "query": query}
                )
        
    def _handle_webpage_extract(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle webpage extraction operations.
        
        This function extracts content from webpages using Claude's API.
        
        Args:
            params: Dictionary of parameters for the webpage extraction
            
        Returns:
            Dictionary containing the extracted content
            
        Raises:
            ToolError: If inputs are invalid or if the extraction fails
        """
        # Extract expected parameters
        url = params.get("url")
        extraction_prompt = params.get("extraction_prompt")
        format_type = params.get("format", "text")
        include_metadata = params.get("include_metadata", False)
        timeout = params.get("timeout")

        # Validate required parameters
        if not url:
            raise ToolError(
                "Required parameter 'url' must be provided for webpage_extract operation",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_params": list(params.keys())}
            )

        self.logger.info(f"Extracting content from {url}")

        # Use the main error context for the entire operation
        with error_context(
            component_name=self.name,
            operation=f"extracting content from {url}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Input validation
            self._validate_webpage_extract_inputs(url, format_type, timeout)

            # Set default timeout if not provided
            timeout = self._validate_timeout(timeout)

            # Set default extraction prompt if not provided
            if extraction_prompt is None:
                # Import config when needed (avoids circular imports)
                from config import config
                
                try:
                    extraction_prompt = config.webaccess_tool.default_extraction_prompt
                except AttributeError:
                    extraction_prompt = (
                        "Please extract the main content from this webpage. "
                        "Focus on the article text, headings, and important information. "
                        "Ignore navigation, ads, footers, and other non-essential elements."
                    )

            # Fetch the webpage HTML
            html_content, response = self._fetch_webpage(url, timeout)
            
            if not html_content:
                return {
                    "success": False,
                    "url": url,
                    "error": "Failed to fetch webpage content or content was empty"
                }

            # Extract content from HTML using LLM
            extracted_content = self._extract_content_with_llm(
                html_content, 
                url, 
                extraction_prompt, 
                format_type
            )
            
            result = {
                "success": True,
                "url": url,
                "content": extracted_content
            }
            
            # Add metadata if requested
            if include_metadata:
                title = self._extract_title(html_content)
                metadata = self._extract_metadata(html_content, response)
                result["title"] = title
                result["metadata"] = metadata
                
            return result
            
    def _validate_webpage_extract_inputs(self, url, format_type, timeout):
        """
        Validate input parameters for webpage extraction.
        
        Args:
            url: URL string
            format_type: Format type string
            timeout: Timeout value in seconds
            
        Raises:
            ToolError: If any inputs are invalid
        """
        # Validate URL with operation context for domain restrictions
        self._validate_url(url, operation="webpage_extract")
            
        # Validate format type
        valid_formats = ["text", "markdown", "html"]
        if format_type not in valid_formats:
            raise ToolError(
                f"Invalid format type: {format_type}. Must be one of: {', '.join(valid_formats)}",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_format": format_type, "valid_formats": valid_formats}
            )
            
        # Timeout validation is handled by the shared _validate_timeout method
    
    def _extract_content_with_llm(self, html_content, url, extraction_prompt, format_type):
        """
        Use the LLM to extract content from the HTML.
        
        Args:
            html_content: The HTML content to extract from
            url: The URL of the webpage (for context)
            extraction_prompt: The prompt to guide the extraction
            format_type: The desired output format
            
        Returns:
            Extracted content as string
            
        Raises:
            ToolError: If extraction fails
        """
        self.logger.debug("Extracting content with LLM")
        
        # Get the LLM bridge instance
        llm_bridge = self._get_llm_bridge()
        
        # Construct the prompt
        format_instruction = ""
        if format_type == "markdown":
            format_instruction = "Format your output as Markdown to preserve the structure."
        elif format_type == "html":
            format_instruction = "Return a filtered, clean HTML that preserves the structure but removes unnecessary elements."
            
        # Construct the system prompt
        system_prompt = f"""
        You are an expert at extracting content from webpages. Your task is to extract the main content from the HTML provided.
        
        URL: {url}
        
        {extraction_prompt}
        
        {format_instruction}
        
        Only extract content that's visible to a user viewing the page. Ignore navigation menus, headers, footers, ads, and other
        non-essential elements. Focus on the main content, including text, headings, and important information.
        """
        
        # Construct the user message
        user_message = f"Here is the HTML content to extract from:\n\n```html\n{html_content}\n```"
        
        # Use the error context for LLM extraction
        with error_context(
            component_name=self.name,
            operation="extracting content with LLM",
            error_class=ToolError,
            error_code=ErrorCode.API_RESPONSE_ERROR,
            logger=self.logger
        ):
            try:
                # Call the LLM
                messages = [{"role": "user", "content": user_message}]
                response = llm_bridge.generate_response(
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=0.1,  # Low temperature for deterministic extraction
                )
                
                # Extract the text content
                extracted_content = llm_bridge.extract_text_content(response)
                
                return extracted_content
                
            except Exception as e:
                self.logger.error(f"Error in LLM content extraction: {str(e)}")
                raise ToolError(
                    f"LLM content extraction failed: {str(e)}",
                    ErrorCode.API_RESPONSE_ERROR,
                    {"error": str(e)}
                )
    
    def _extract_title(self, html_content):
        """
        Extract the title from HTML content.
        
        Args:
            html_content: HTML content as string
            
        Returns:
            Title string or empty string if not found
        """
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1).strip()
        return ""
    
    def _extract_metadata(self, html_content, response):
        """
        Extract basic metadata from HTML and response.
        
        Args:
            html_content: HTML content as string
            response: Requests response object
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "content_type": response.headers.get('Content-Type', ''),
            "last_modified": response.headers.get('Last-Modified', ''),
            "size": len(html_content),
            "status_code": response.status_code,
            "final_url": response.url
        }
        
        # Extract meta tags
        meta_tags = {}
        description_match = re.search(r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']\s*/?>', html_content, re.IGNORECASE)
        if description_match:
            meta_tags["description"] = description_match.group(1).strip()
            
        keywords_match = re.search(r'<meta\s+name=["\']keywords["\']\s+content=["\'](.*?)["\']\s*/?>', html_content, re.IGNORECASE)
        if keywords_match:
            meta_tags["keywords"] = keywords_match.group(1).strip()
            
        # Extract Open Graph metadata
        og_title_match = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\'](.*?)["\']\s*/?>', html_content, re.IGNORECASE)
        if og_title_match:
            meta_tags["og:title"] = og_title_match.group(1).strip()
            
        og_description_match = re.search(r'<meta\s+property=["\']og:description["\']\s+content=["\'](.*?)["\']\s*/?>', html_content, re.IGNORECASE)
        if og_description_match:
            meta_tags["og:description"] = og_description_match.group(1).strip()
            
        metadata["meta_tags"] = meta_tags
        
        return metadata
        
    def _validate_url(self, url: str, operation: str = None) -> None:
        """
        Validate URL format and security restrictions.
        
        Args:
            url: The URL to validate
            operation: Optional operation context for domain restrictions
            
        Raises:
            ToolError: If the URL is invalid or restricted
        """
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
                
        # Check domain restrictions if operation is provided
        if operation:
            # Get domain restrictions for this operation
            allowed_domains, blocked_domains = self._get_domain_restrictions(operation=operation)
            
            # Check if domain is explicitly blocked
            if blocked_domains and parsed_url.netloc:
                domain = parsed_url.netloc.lower()
                
                for blocked in blocked_domains:
                    if domain == blocked.lower() or domain.endswith('.' + blocked.lower()):
                        raise ToolError(
                            f"Domain '{domain}' is blocked by configuration",
                            ErrorCode.TOOL_INVALID_INPUT,
                            {"url": url, "domain": domain, "blocked_by": blocked}
                        )
            
            # Check if domain is allowed (when allowlist is active)
            if allowed_domains and parsed_url.netloc:
                domain = parsed_url.netloc.lower()
                is_allowed = False
                
                for allowed in allowed_domains:
                    if domain == allowed.lower() or domain.endswith('.' + allowed.lower()):
                        is_allowed = True
                        break
                        
                if not is_allowed:
                    raise ToolError(
                        f"Domain '{domain}' is not in the allowed domains list",
                        ErrorCode.TOOL_INVALID_INPUT,
                        {"url": url, "domain": domain, "allowed_domains": allowed_domains}
                    )
    
    def _validate_timeout(self, timeout: Optional[int]) -> int:
        """
        Validate and return the timeout value.
        
        Args:
            timeout: The timeout value to validate
            
        Returns:
            Validated timeout value
            
        Raises:
            ToolError: If the timeout is invalid
        """
        from config import config
        
        # Get config
        try:
            tool_config = config.webaccess_tool
            default_timeout = tool_config.timeout
            max_timeout = tool_config.max_timeout
        except AttributeError:
            default_timeout = 30
            max_timeout = 120
            
        # Use default if not provided
        if timeout is None:
            return default_timeout
            
        # Validate timeout
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ToolError(
                "Timeout must be a positive number",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_timeout": timeout}
            )
            
        # Check against max timeout
        if timeout > max_timeout:
            raise ToolError(
                f"Timeout value exceeds maximum allowed ({max_timeout} seconds)",
                ErrorCode.TOOL_INVALID_INPUT,
                {"provided_timeout": timeout, "max_timeout": max_timeout}
            )
            
        return timeout
        
    def _get_domain_restrictions(self, allowed_domains=None, blocked_domains=None, operation=None):
        """
        Get and merge domain restrictions from various sources.
        
        This function consolidates domain restrictions from:
        - Provided parameters
        - Tool-specific configuration
        - Global tool configuration
        
        Args:
            allowed_domains: Optional list of explicitly allowed domains
            blocked_domains: Optional list of explicitly blocked domains
            operation: Optional specific operation to get restrictions for
            
        Returns:
            Tuple of (final_allowed_domains, final_blocked_domains)
        """
        from config import config
        
        final_allowed = set()
        final_blocked = set()
        
        # First, add global restrictions from config
        try:
            if config.webaccess_tool.allowed_domains:
                final_allowed.update(config.webaccess_tool.allowed_domains)
                
            if config.webaccess_tool.blocked_domains:
                final_blocked.update(config.webaccess_tool.blocked_domains)
        except AttributeError:
            pass
            
        # Add operation-specific restrictions if specified
        if operation == "http_request":
            # No specific restrictions for HTTP requests beyond globals
            pass
        elif operation == "web_search":
            # Web search might have specific allowed/blocked domains
            try:
                if hasattr(config, 'web_search_tool') and config.web_search_tool.allowed_domains:
                    final_allowed.update(config.web_search_tool.allowed_domains)
                    
                if hasattr(config, 'web_search_tool') and config.web_search_tool.blocked_domains:
                    final_blocked.update(config.web_search_tool.blocked_domains)
            except AttributeError:
                pass
        elif operation == "webpage_extract":
            # Webpage extraction might have specific allowed domains
            try:
                if hasattr(config, 'webpage_extraction_tool') and config.webpage_extraction_tool.allowed_domains:
                    final_allowed.update(config.webpage_extraction_tool.allowed_domains)
            except AttributeError:
                pass
                
        # Finally, add explicitly provided domains (these take precedence)
        if allowed_domains:
            final_allowed.update(allowed_domains)
            
        if blocked_domains:
            final_blocked.update(blocked_domains)
            
        # Convert back to lists for return
        return list(final_allowed) if final_allowed else None, list(final_blocked) if final_blocked else None
                
    def _validate_domains(self, allowed_domains: Optional[List[str]], blocked_domains: Optional[List[str]]) -> None:
        """
        Validate domain restriction lists.
        
        Args:
            allowed_domains: List of allowed domains
            blocked_domains: List of blocked domains
            
        Raises:
            ToolError: If domain lists are invalid or conflicting
        """
        # Validate allowed_domains
        if allowed_domains is not None:
            if not isinstance(allowed_domains, list):
                raise ToolError(
                    "allowed_domains must be a list of strings",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"provided_allowed_domains": allowed_domains}
                )
                
            for domain in allowed_domains:
                if not isinstance(domain, str):
                    raise ToolError(
                        "Each allowed domain must be a string",
                        ErrorCode.TOOL_INVALID_INPUT,
                        {"invalid_domain": domain}
                    )
        
        # Validate blocked_domains
        if blocked_domains is not None:
            if not isinstance(blocked_domains, list):
                raise ToolError(
                    "blocked_domains must be a list of strings",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"provided_blocked_domains": blocked_domains}
                )
                
            for domain in blocked_domains:
                if not isinstance(domain, str):
                    raise ToolError(
                        "Each blocked domain must be a string",
                        ErrorCode.TOOL_INVALID_INPUT,
                        {"invalid_domain": domain}
                    )
        
        # Check for conflict between allowed and blocked domains
        if allowed_domains and blocked_domains:
            overlap = set(allowed_domains).intersection(set(blocked_domains))
            if overlap:
                raise ToolError(
                    "Domain cannot be both allowed and blocked",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"conflicting_domains": list(overlap)}
                )
                
    def _fetch_webpage(self, url: str, timeout: int) -> tuple:
        """
        Fetch webpage content from the specified URL.
        
        Args:
            url: The URL to fetch
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (html_content, response)
            
        Raises:
            ToolError: If fetching fails
        """
        self.logger.debug(f"Fetching webpage: {url}")
        
        # Use shared HTTP request function with browser-like headers
        response = self._make_http_request(
            method="GET",
            url=url,
            timeout=timeout,
            is_browser=True  # Use browser-like headers for better compatibility
        )
        
        return response.text, response