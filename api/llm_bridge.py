"""
Abstraction layer for Anthropic API communication.

This module handles communications with the Anthropic API, including
prompt construction, response handling, and error management for API interactions.
"""
import json
import logging
import time
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Union, Callable

import anthropic
import requests
from requests.exceptions import RequestException

from errors import APIError, ErrorCode, error_context
from config import config


class LLMBridge:
    """
    Bridge to the Anthropic Claude API.

    Handles communication with the Anthropic API, including:
    - Prompt construction
    - Response handling
    - Error management
    - Rate limiting
    """

    def __init__(self):
        """
        Initialize the LLM bridge.

        Sets up the API client, logging, and rate limiting.
        """
        self.logger = logging.getLogger("llm_bridge")

        # Get API settings from config
        self.api_key = config.api_key
        self.model = config.api.model
        self.max_tokens = config.api.max_tokens
        self.temperature = config.api.temperature
        self.max_retries = config.api.max_retries
        self.timeout = config.api.timeout

        # Initialize the API client
        try:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except Exception as e:
            raise APIError(
                f"Failed to initialize Anthropic API client: {e}",
                ErrorCode.API_AUTHENTICATION_ERROR
            )

        # Rate limiting variables
        self.rate_limit_rpm = config.api.rate_limit_rpm
        self.min_request_interval = 60.0 / self.rate_limit_rpm
        self.last_request_time = 0.0
        
        # Token bucket for burst handling
        self.token_bucket_size = config.api.burst_limit if hasattr(config.api, "burst_limit") else 3
        self.tokens = self.token_bucket_size  # Start with a full bucket
        self.last_token_refill = time.time()

        self.logger.debug(f"LLM Bridge initialized with model {self.model}")

    def _enforce_rate_limit(self) -> None:
        """
        Enforce API rate limiting with burst allowance using token bucket algorithm.
        
        This implementation allows for burst API calls while maintaining the overall
        rate limit over time.
        """
        now = time.time()
        
        # Refill tokens based on time elapsed (at the rate of RPM)
        time_since_refill = now - self.last_token_refill
        new_tokens = time_since_refill * (self.rate_limit_rpm / 60.0)
        
        if new_tokens > 0:
            self.tokens = min(self.token_bucket_size, self.tokens + new_tokens)
            self.last_token_refill = now
        
        # If we have at least one token, allow the request immediately
        if self.tokens >= 1:
            self.tokens -= 1
            self.last_request_time = now
            self.logger.debug(f"Burst mode: using token, {self.tokens:.2f} tokens remaining")
            return
            
        # If we're out of tokens, calculate wait time
        time_since_last_request = now - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            # Wait to respect rate limit
            wait_time = self.min_request_interval - time_since_last_request
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()

    def _handle_api_error(self, error: Exception, attempt: int) -> None:
        """
        Handle API errors, with appropriate retry logic.

        Args:
            error: The API error that occurred
            attempt: The current retry attempt number

        Raises:
            APIError: With appropriate error code and details
        """
        # Check for overloaded error in dictionary format
        if hasattr(error, "get") and callable(getattr(error, "get", None)):
            try:
                if error.get("type") == "error" and error.get("error", {}).get("type") == "overloaded_error":
                    raise APIError(
                        "The API is currently experiencing high traffic. Please try again later.",
                        ErrorCode.API_RATE_LIMIT_ERROR,
                        {"details": error.get("error", {})}
                    )
            except (AttributeError, TypeError):
                pass

        # Check for overloaded error in string representation
        error_str = str(error)
        if "overloaded_error" in error_str and "Overloaded" in error_str:
            raise APIError(
                "The API is currently experiencing high traffic. Please try again later.",
                ErrorCode.API_RATE_LIMIT_ERROR,
                {"original_error": error_str}
            )

        if isinstance(error, anthropic.APIError):
            # Handle specific Anthropic API errors
            if error.status_code == 401:
                raise APIError(
                    "Authentication error: Invalid API key",
                    ErrorCode.API_AUTHENTICATION_ERROR,
                    {"status_code": 401}
                )
            elif error.status_code == 429:
                # Rate limit exceeded
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(
                        f"Rate limit exceeded. Retrying in {wait_time} seconds (attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                    return
                else:
                    raise APIError(
                        "API rate limit exceeded. Too many requests.",
                        ErrorCode.API_RATE_LIMIT_ERROR,
                        {"status_code": 429, "attempts": attempt}
                    )
            elif error.status_code >= 500:
                # Server error
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(
                        f"Server error. Retrying in {wait_time} seconds (attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                    return
                else:
                    raise APIError(
                        f"API server error: {error}",
                        ErrorCode.API_RESPONSE_ERROR,
                        {"status_code": error.status_code, "attempts": attempt}
                    )
            else:
                # Other API errors
                raise APIError(
                    f"API error: {error}",
                    ErrorCode.API_RESPONSE_ERROR,
                    {"status_code": error.status_code}
                )

        elif isinstance(error, RequestException):
            # Network-related errors
            if attempt < self.max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.warning(
                    f"Connection error. Retrying in {wait_time} seconds (attempt {attempt}/{self.max_retries})"
                )
                time.sleep(wait_time)
                return
            else:
                raise APIError(
                    f"API connection error: {error}",
                    ErrorCode.API_CONNECTION_ERROR,
                    {"attempts": attempt}
                )

        elif isinstance(error, requests.Timeout):
            # Timeout errors
            if attempt < self.max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.warning(
                    f"Timeout error. Retrying in {wait_time} seconds (attempt {attempt}/{self.max_retries})"
                )
                time.sleep(wait_time)
                return
            else:
                raise APIError(
                    f"API timeout error: {error}",
                    ErrorCode.API_TIMEOUT_ERROR,
                    {"attempts": attempt}
                )

        else:
            # Other unexpected errors
            raise APIError(
                f"Unexpected API error: {error}",
                ErrorCode.API_RESPONSE_ERROR
            )

        # If we got here, we should not retry
        raise APIError(
            f"API error after {attempt} attempts: {error}",
            ErrorCode.API_RESPONSE_ERROR,
            {"attempts": attempt}
        )

    @contextmanager
    def api_error_context(self, operation, retry_allowed=True, max_attempts=None):
        """
        Context manager for API error handling with retry support.
        
        Args:
            operation: Description of the API operation
            retry_allowed: Whether retries are allowed for this operation
            max_attempts: Override for max retry attempts
            
        Yields:
            Tuple: (current_attempt, max_attempts) to track retry state
            
        Raises:
            APIError: With appropriate error information after retries
        """
        max_retries = max_attempts if max_attempts is not None else self.max_retries
        
        for attempt in range(1, max_retries + 1):
            try:
                # Enforce rate limiting
                self._enforce_rate_limit()
                
                # Provide attempt information to the caller
                yield (attempt, max_retries)
                
                # If we get here, the operation succeeded
                return
                
            except anthropic.APIError as e:
                # Handle API-specific errors with retries
                if e.status_code == 429 and retry_allowed and attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(
                        f"Rate limit exceeded. Retrying in {wait_time}s (attempt {attempt}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                elif e.status_code >= 500 and retry_allowed and attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(
                        f"Server error. Retrying in {wait_time}s (attempt {attempt}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                
                # If retries not allowed or we're out of retries
                with error_context(
                    component_name="API",
                    operation=operation,
                    error_class=APIError,
                    error_code=ErrorCode.API_RATE_LIMIT_ERROR if e.status_code == 429 else ErrorCode.API_RESPONSE_ERROR,
                    logger=self.logger
                ):
                    raise
                    
            except (RequestException, requests.Timeout) as e:
                # Handle network errors with retries
                if retry_allowed and attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(
                        f"Connection error. Retrying in {wait_time}s (attempt {attempt}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                
                # If retries not allowed or we're out of retries
                with error_context(
                    component_name="API",
                    operation=operation,
                    error_class=APIError,
                    error_code=ErrorCode.API_CONNECTION_ERROR if isinstance(e, RequestException) else ErrorCode.API_TIMEOUT_ERROR,
                    logger=self.logger
                ):
                    raise
                    
            except Exception as e:
                # Handle any other errors (no retries)
                with error_context(
                    component_name="API",
                    operation=operation,
                    error_class=APIError,
                    error_code=ErrorCode.API_RESPONSE_ERROR,
                    logger=self.logger
                ):
                    raise
        
        # If we've exhausted all retries
        raise APIError(
            f"Failed to complete API operation '{operation}' after {max_retries} attempts",
            ErrorCode.API_RESPONSE_ERROR
        )

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        callback: Optional[Callable] = None,
        cache_control: Optional[Dict[str, str]] = None,
        dynamic_content: Optional[str] = None
    ) -> Union[Dict[str, Any], Any]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt for the conversation
            temperature: Optional temperature parameter (0.0 to 1.0)
            max_tokens: Optional maximum tokens for the response
            tools: Optional list of tool definitions
            stream: Whether to stream the response (default: False)
            callback: Optional callback function to process streaming chunks
            cache_control: Optional cache control parameters for prompt caching
            dynamic_content: Optional dynamic content to append after cached system prompt

        Returns:
            If stream=False: API response as a dictionary
            If stream=True: A stream object to be processed

        Raises:
            APIError: If the API request fails
        """
        # Use provided values or defaults from config
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Prepare request parameters
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add optional parameters if provided
        if system_prompt:
            # Format system content as structured blocks with caching
            if dynamic_content and cache_control:
                # Create structured system blocks
                system_blocks = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": cache_control
                    },
                    {
                        "type": "text",
                        "text": dynamic_content
                    }
                ]
                params["system"] = system_blocks
                self.logger.debug("Using structured system blocks with caching")
            else:
                # Simple string system prompt without caching
                params["system"] = system_prompt
        
        if tools:
            params["tools"] = tools

        # Add streaming parameter if requested
        if stream:
            params["stream"] = True

        # Log the request (sanitized for security)
        self._log_request(params)

        # Use the API error context for handling errors with retries
        with self.api_error_context("generate response"):
            # Make the API call
            if stream:
                # Stream mode - return the stream or process with callback
                # Remove 'stream' param since it's not needed for stream() method
                stream_params = params.copy()
                if 'stream' in stream_params:
                    del stream_params['stream']

                stream_response = self.client.messages.stream(**stream_params)

                if callback:
                    # Process stream with callback
                    return self._process_stream_with_callback(stream_response, callback)
                else:
                    # Return stream directly
                    self.logger.debug("Returning stream object directly")
                    return stream_response
            else:
                # Normal mode - make regular API call
                response = self.client.messages.create(**params)

                # Log the response
                self._log_response(response)
                
                return response

    def _process_stream_with_callback(self, stream, callback):
        """
        Process a streaming response with a callback function.

        Args:
            stream: The streaming response from Anthropic API
            callback: A callback function to process each chunk

        Returns:
            The final complete response
        """
        with error_context(
            component_name="API", 
            operation="processing stream", 
            error_class=APIError,
            error_code=ErrorCode.API_RESPONSE_ERROR,
            logger=self.logger
        ):
            final_response = None

            with stream as response:
                # Store the final response for later
                final_response = response

                # Process each text chunk with the callback
                for text in response.text_stream:
                    callback(text)

            return final_response

    def _log_request(self, params: Dict[str, Any]) -> None:
        """
        Log API request parameters (with sensitive data removed).

        Args:
            params: Request parameters
        """
        # Create a sanitized copy of the parameters
        sanitized = params.copy()

        # Log at debug level
        self.logger.debug(f"API Request: {json.dumps(sanitized, indent=2, default=str)}")

    def _log_response(self, response: Any) -> None:
        """
        Log API response for debugging.

        Args:
            response: API response
        """
        # Log at debug level - sanitize or truncate if needed
        self.logger.debug(f"API Response received: {type(response)}")

    def extract_text_content(self, response: Any) -> str:
        """
        Extract text content from an Anthropic API response.

        Args:
            response: API response (could be a standard Message or a MessageStream.message)

        Returns:
            Extracted text content from all text blocks
        """
        text_content = []

        # First check if we have a content attribute
        if not hasattr(response, 'content'):
            self.logger.warning("Response has no content attribute")
            return ""

        # Process all content blocks to extract text
        for content_block in response.content:
            if hasattr(content_block, 'type') and content_block.type == "text":
                text_content.append(content_block.text)
            elif hasattr(content_block, 'text'):  # Fallback for other content types with text
                text_content.append(content_block.text)

        # If no text content found with either approach
        if not text_content and len(response.content) > 0:
            if hasattr(response.content[0], 'text'):  # Last resort fallback
                return response.content[0].text
            else:
                self.logger.warning("Could not extract text from response content")
                return ""

        return " ".join(text_content).strip()

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """
        Extract tool calls from an Anthropic API response.

        Args:
            response: API response (could be a standard Message or a MessageStream.message)

        Returns:
            List of tool call objects
        """
        tool_calls = []

        # First check if we have a content attribute
        if not hasattr(response, 'content'):
            self.logger.warning("Response has no content attribute")
            return []

        # Process all content blocks in the response
        for content_block in response.content:
            # Look for Anthropic's native tool_use format
            if hasattr(content_block, 'type') and content_block.type == "tool_use":
                self.logger.debug(f"Found tool_use block: {content_block.name}")
                tool_calls.append({
                    "id": content_block.id,
                    "tool_name": content_block.name,
                    "input": content_block.input
                })

        self.logger.debug(f"Extracted {len(tool_calls)} tool calls from response")
        return tool_calls
