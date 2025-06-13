"""
Unified LLM provider for OpenAI-compatible APIs.

This module provides a single interface for interacting with any OpenAI-compatible
LLM API, whether local (like Ollama) or remote (like OpenAI, Anthropic via proxy, etc).
"""
import json
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Union, Callable
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from errors import APIError, ErrorCode, error_context
from config import config


class LLMProvider:
    """
    Unified provider for OpenAI-compatible LLM APIs.
    
    Supports both local and remote providers through a standardized
    OpenAI chat completions API interface.
    """
    
    def __init__(self, 
                 provider_type: Optional[str] = None,
                 api_endpoint: Optional[str] = None,
                 model: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 max_retries: Optional[int] = None,
                 timeout: Optional[int] = None,
                 api_key: Optional[str] = None):
        """Initialize the LLM provider with configuration.
        
        Args:
            provider_type: Override provider type ("local" or "remote")
            api_endpoint: Override API endpoint URL
            model: Override model name
            max_tokens: Override max tokens
            temperature: Override temperature
            max_retries: Override max retries
            timeout: Override timeout
            api_key: Override API key
        """
        self.logger = logging.getLogger("llm_provider")
        
        # Get configuration with optional overrides
        self.provider_type = provider_type if provider_type is not None else config.api.provider
        self.api_endpoint = api_endpoint if api_endpoint is not None else config.api.api_endpoint
        self.model = model if model is not None else config.api.model
        self.max_tokens = max_tokens if max_tokens is not None else config.api.max_tokens
        self.temperature = temperature if temperature is not None else config.api.temperature
        self.max_retries = max_retries if max_retries is not None else config.api.max_retries
        self.timeout = timeout if timeout is not None else config.api.timeout
        
        # Get API key with override
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = config.api_key
        
        # Thread lock to serialize requests and prevent race conditions
        self._request_lock = threading.Lock()
        
        self.logger.info(
            f"LLM Provider initialized: type={self.provider_type}, "
            f"endpoint={self.api_endpoint}, model={self.model}"
        )
    
    def _create_session(self) -> requests.Session:
        """Create a fresh session for each request to avoid race conditions."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        callback: Optional[Callable] = None,
        dynamic_content: Optional[str] = None,
        **kwargs  # Ignore provider-specific parameters
    ) -> Dict[str, Any]:
        """
        Generate a response using the configured LLM provider.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt to prepend
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            tools: Optional list of tools in OpenAI format
            stream: Whether to stream the response
            callback: Optional callback for streaming responses
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            Response object with standardized format
            
        Raises:
            APIError: If the request fails
        """
        # Build request body
        request_body = self._build_request_body(
            messages, system_prompt, temperature, max_tokens, tools, dynamic_content
        )
        
        # Make the request with thread safety
        with self._request_lock:
            with error_context(
                component_name="LLMProvider",
                operation="generate_response",
                error_class=APIError,
                error_code=ErrorCode.API_RESPONSE_ERROR,
                logger=self.logger
            ):
                if stream and callback:
                    return self._handle_streaming_request(request_body, callback)
                else:
                    return self._handle_standard_request(request_body)
    
    def _build_request_body(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        dynamic_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build the OpenAI-format request body."""
        # Format messages
        formatted_messages = []
        
        # Combine system prompt with dynamic content
        if system_prompt or dynamic_content:
            content_parts = []
            if system_prompt:
                content_parts.append(system_prompt)
            if dynamic_content:
                content_parts.append(dynamic_content)
            
            formatted_messages.append({
                "role": "system",
                "content": "\n\n".join(content_parts)
            })
        
        # Add conversation messages
        formatted_messages.extend(messages)
        
        # Build request
        request_body = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        
        # Add tools if provided
        if tools:
            request_body["tools"] = tools
            
        return request_body
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add authorization for remote providers
        if self.provider_type == "remote" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        return headers
    
    def _handle_standard_request(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a standard (non-streaming) request."""
        try:
            session = self._create_session()
            response = session.post(
                self.api_endpoint,
                json=request_body,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Convert to standardized format
            return self._standardize_response(response_data)
            
        except requests.exceptions.Timeout:
            raise APIError(
                f"Request timed out after {self.timeout} seconds",
                ErrorCode.API_TIMEOUT_ERROR
            )
        except requests.exceptions.HTTPError as e:
            self._handle_http_error(e)
        except requests.exceptions.RequestException as e:
            raise APIError(
                f"Network error: {str(e)}",
                ErrorCode.API_CONNECTION_ERROR
            )
        except json.JSONDecodeError:
            raise APIError(
                "Invalid JSON response from API",
                ErrorCode.API_RESPONSE_ERROR
            )
    
    def _handle_streaming_request(
        self, 
        request_body: Dict[str, Any], 
        callback: Callable
    ) -> Dict[str, Any]:
        """Handle a streaming request."""
        request_body["stream"] = True
        
        try:
            session = self._create_session()
            response = session.post(
                self.api_endpoint,
                json=request_body,
                headers=self._get_headers(),
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            # Process SSE stream
            full_content = ""
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_str = line_str[6:]  # Remove "data: " prefix
                    
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        chunk = json.loads(data_str)
                        # Extract content from chunk
                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                content = delta["content"]
                                full_content += content
                                callback(content)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse streaming chunk: {data_str}")
            
            # Return final response in standard format
            return self._create_response_object(full_content)
            
        except requests.exceptions.Timeout:
            raise APIError(
                f"Streaming request timed out after {self.timeout} seconds",
                ErrorCode.API_TIMEOUT_ERROR
            )
        except requests.exceptions.HTTPError as e:
            self._handle_http_error(e)
        except requests.exceptions.RequestException as e:
            raise APIError(
                f"Network error during streaming: {str(e)}",
                ErrorCode.API_CONNECTION_ERROR
            )
    
    def _handle_http_error(self, error: requests.exceptions.HTTPError):
        """Handle HTTP errors with appropriate error codes."""
        if error.response is None:
            raise APIError(
                "No response from API",
                ErrorCode.API_CONNECTION_ERROR
            )
            
        status_code = error.response.status_code
        
        # Try to get error message from response
        try:
            error_data = error.response.json()
            error_message = error_data.get("error", {}).get("message", str(error))
        except:
            error_message = error.response.text or str(error)
        
        if status_code == 401:
            raise APIError(
                "Authentication failed. Check your API key.",
                ErrorCode.API_AUTHENTICATION_ERROR
            )
        elif status_code == 429:
            raise APIError(
                "Rate limit exceeded. Please try again later.",
                ErrorCode.API_RATE_LIMIT_ERROR
            )
        elif status_code >= 500:
            raise APIError(
                f"Server error: {error_message}",
                ErrorCode.API_RESPONSE_ERROR
            )
        else:
            raise APIError(
                f"API error (status {status_code}): {error_message}",
                ErrorCode.API_RESPONSE_ERROR
            )
    
    def _standardize_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize response format across different providers.
        
        Ensures consistent response structure regardless of provider.
        """
        # Create standardized response
        standardized = {
            "content": [],
            "usage": {}
        }
        
        # Extract content
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            message = choice.get("message", {})
            
            # Add text content
            if "content" in message and message["content"]:
                standardized["content"].append({
                    "type": "text",
                    "text": message["content"]
                })
            
            # Add tool calls if present
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    function = tool_call.get("function", {})
                    standardized["content"].append({
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "input": json.loads(function.get("arguments", "{}"))
                    })
        
        # Extract usage information
        if "usage" in response_data:
            standardized["usage"] = {
                "input_tokens": response_data["usage"].get("prompt_tokens", 0),
                "output_tokens": response_data["usage"].get("completion_tokens", 0),
                "total_tokens": response_data["usage"].get("total_tokens", 0)
            }
        
        return standardized
    
    def _create_response_object(self, content: str) -> Dict[str, Any]:
        """Create a standardized response object from content."""
        return {
            "content": [{"type": "text", "text": content}],
            "usage": {}  # Usage not available for streaming
        }
    
    def extract_text_content(self, response: Dict[str, Any]) -> str:
        """
        Extract text content from response.
        
        Args:
            response: Standardized response dictionary
            
        Returns:
            Extracted text content
        """
        text_parts = []
        
        for content_block in response.get("content", []):
            if content_block.get("type") == "text":
                text_parts.append(content_block.get("text", ""))
                
        return " ".join(text_parts)
    
    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tool calls from response.
        
        Args:
            response: Standardized response dictionary
            
        Returns:
            List of tool calls in standardized format
        """
        tool_calls = []
        
        for content_block in response.get("content", []):
            if content_block.get("type") == "tool_use":
                tool_calls.append({
                    "id": content_block.get("id", ""),
                    "tool_name": content_block.get("name", ""),
                    "input": content_block.get("input", {})
                })
                
        return tool_calls