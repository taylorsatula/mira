"""
Abstraction layer for Ollama API communication.

This module provides the Ollama API bridge that supports multi-user access
to a single local Ollama instance. It follows the same interface as the
Anthropic API bridge but connects to an Ollama instance.
"""
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Callable

from api.request_queue import RequestQueue
from errors import APIError, ErrorCode, error_context

class OllamaBridge:
    """
    Bridge to the Ollama API for local LLM inference.
    
    This implements the same interface as LLMBridge but connects
    to a local Ollama instance with multi-user support via queue.
    """
    
    def __init__(self, base_url="http://localhost:11434", model="qwen"):
        """
        Initialize the Ollama bridge.
        
        Args:
            base_url: URL of the Ollama API server
            model: Name of the model to use
        """
        self.logger = logging.getLogger("ollama_bridge")
        self.base_url = base_url
        self.model = model
        
        # Get request queue singleton
        self.request_queue = RequestQueue.get_instance()
        
        self.logger.info(f"Ollama Bridge initialized with model {self.model} at {self.base_url}")
    
    def generate_response(
        self,
        messages: List[Dict[str, Any]],
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
        Generate a response using Ollama.
        
        This method follows the same interface as LLMBridge.generate_response
        for compatibility with the existing application architecture.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt for the conversation
            temperature: Optional temperature parameter (0.0 to 1.0)
            max_tokens: Optional maximum tokens for the response
            tools: Optional list of tool definitions
            stream: Whether to stream the response
            callback: Optional callback function for processing streamed chunks
            cache_control: Optional cache control parameters (ignored)
            dynamic_content: Optional dynamic content to append to system prompt
            
        Returns:
            Response object with compatible interface to Anthropic API
            
        Raises:
            APIError: If the API request fails
        """
        with error_context(
            component_name="OllamaBridge",
            operation="generate response",
            error_class=APIError,
            error_code=ErrorCode.API_RESPONSE_ERROR,
            logger=self.logger
        ):
            # Format messages for OpenAI format
            formatted_messages = self._format_messages(messages, system_prompt, dynamic_content)
            
            # Extract OpenAI schemas from tools
            openai_tools = self._extract_openai_schemas(tools) if tools else None
            
            # Build request body
            request_body = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": temperature if temperature is not None else 0.7,
            }
            
            if max_tokens:
                request_body["max_tokens"] = max_tokens
                
            if openai_tools:
                request_body["tools"] = openai_tools
                
            if stream and not callback:
                # Streaming requested but no callback provided
                self.logger.warning("Streaming requested without callback, using standard request")
                stream = False
            
            # Log the request metadata
            self.logger.debug(f"Preparing Ollama request for model {self.model}" +
                             (f" with {len(openai_tools)} tools" if openai_tools else ""))
            
            # Process the request through the queue
            if stream and callback:
                # Streaming request
                request_id = self.request_queue.add_streaming_request(
                    {
                        "base_url": self.base_url,
                        "body": request_body
                    },
                    callback
                )
                
                # Wait for streaming to complete and get final response
                final_response = self.request_queue.get_result(request_id)
            else:
                # Standard request
                request_id = self.request_queue.add_request({
                    "base_url": self.base_url,
                    "body": request_body
                })
                
                # Wait for response
                final_response = self.request_queue.get_result(request_id)
            
            # Check for errors
            if isinstance(final_response, dict) and "error" in final_response:
                error_msg = final_response.get("error", "Unknown error")
                raise APIError(
                    f"Ollama API error: {error_msg}",
                    ErrorCode.API_RESPONSE_ERROR
                )
            
            # Create a response object compatible with Anthropic API
            return self._create_response_object(final_response)
    
    def _format_messages(self, messages, system_prompt=None, dynamic_content=None):
        """
        Format messages for OpenAI-compatible API.
        
        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt
            dynamic_content: Optional dynamic content to append to system prompt
            
        Returns:
            Formatted messages list for OpenAI API
        """
        formatted_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            system_content = system_prompt
            if dynamic_content:
                system_content += "\n\n" + dynamic_content
                
            formatted_messages.append({
                "role": "system", 
                "content": system_content
            })
        
        # Add conversation messages
        for message in messages:
            # Handle messages with tool results (content as list)
            if isinstance(message.get("content", ""), list):
                # Convert Anthropic content blocks to OpenAI format
                text_parts = []
                tool_results = []
                
                for block in message["content"]:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        tool_results.append({
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": block.get("content", "")
                        })
                
                if message["role"] == "user" and tool_results:
                    # This is a user message with tool results
                    # First add regular content
                    if text_parts:
                        formatted_messages.append({
                            "role": "user",
                            "content": "\n".join(text_parts)
                        })
                    
                    # Then add tool results as separate messages
                    for result in tool_results:
                        formatted_messages.append({
                            "role": "tool", 
                            "content": result["content"],
                            "tool_call_id": result["tool_call_id"]
                        })
                else:
                    # Regular message with content blocks
                    formatted_messages.append({
                        "role": message["role"],
                        "content": "\n".join(text_parts) if text_parts else ""
                    })
            else:
                # Regular message with string content
                formatted_messages.append(message)
        
        return formatted_messages
    
    def _extract_openai_schemas(self, tools):
        """
        Extract OpenAI schemas from tools.
        
        Only includes tools that explicitly have an openai_schema defined.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            List of OpenAI-compatible tool definitions or None if no compatible tools
        """
        openai_tools = []
        skipped_tools = []
        
        for tool in tools:
            # Check if the tool has an openai_schema defined
            if "openai_schema" in tool:
                openai_tools.append(tool["openai_schema"])
            else:
                skipped_tools.append(tool.get("name", "unnamed"))
        
        if skipped_tools:
            self.logger.info(f"Skipped {len(skipped_tools)} tools without OpenAI schema: {', '.join(skipped_tools)}")
            
        return openai_tools if openai_tools else None
    
    def _create_response_object(self, ollama_response):
        """
        Create a response object compatible with Anthropic API.

        Args:
            ollama_response: Raw response from Ollama

        Returns:
            Response object with Anthropic-compatible interface
        """

        # Create response object with compatible interface
        class AnthropicResponse:
            def __init__(self, content, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls

                # Add usage info (estimates)
                class Usage:
                    def __init__(self):
                        self.input_tokens = ollama_response.get("usage", {}).get("prompt_tokens", 0)
                        self.output_tokens = ollama_response.get("usage", {}).get("completion_tokens", 0)

                self.usage = Usage()

        # Extract content from Ollama response
        content = []
        
        # Handle different response formats
        if "message" in ollama_response:
            response_message = ollama_response["message"]
            response_text = response_message.get("content", "")
        elif "choices" in ollama_response and len(ollama_response["choices"]) > 0:
            response_message = ollama_response["choices"][0].get("message", {})
            response_text = response_message.get("content", "")
        # Handle final streaming response which might be formatted differently
        elif isinstance(ollama_response, dict) and "done" in ollama_response and ollama_response.get("done") is True:
            # This handles the final message in a streaming response
            if "message" in ollama_response and "content" in ollama_response["message"]:
                response_text = ollama_response["message"]["content"]
                response_message = ollama_response["message"]
            else:
                self.logger.warning(f"Final streaming message missing content: {ollama_response}")
                response_message = {}
                response_text = ""
        else:
            self.logger.warning(f"Unexpected Ollama response format: {ollama_response}")
            response_message = {}
            response_text = "Unable to parse response from Ollama"
        
        # Add text block
        content.append({"type": "text", "text": response_text})
        
        # Process tool calls if present
        tool_calls = []
        
        # Extract tool calls from the response
        tool_call_list = []
        if "tool_calls" in response_message:
            tool_call_list = response_message["tool_calls"]
        elif "choices" in ollama_response and len(ollama_response["choices"]) > 0:
            choice_message = ollama_response["choices"][0].get("message", {})
            if "tool_calls" in choice_message:
                tool_call_list = choice_message["tool_calls"]
        
        # Process each tool call
        for tool_call in tool_call_list:
            try:
                # Extract function call data
                function_data = tool_call.get("function", {})
                function_name = function_data.get("name", "unknown_tool")
                arguments_str = function_data.get("arguments", "{}")
                
                # Parse arguments
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse tool arguments: {arguments_str}")
                    arguments = {}
                
                # Add tool use block in Anthropic format
                content.append({
                    "type": "tool_use",
                    "id": tool_call.get("id", str(uuid.uuid4())),
                    "name": function_name,
                    "input": arguments
                })
                
                tool_calls.append(tool_call)
            except Exception as e:
                self.logger.error(f"Error processing tool call: {e}")
        
        return AnthropicResponse(content, tool_calls)
    
    def extract_text_content(self, response):
        """
        Extract text content from response.
        
        Args:
            response: Response object from generate_response
            
        Returns:
            Extracted text content
        """
        if hasattr(response, "content"):
            text_parts = []
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "".join(text_parts)
        return ""
    
    def extract_tool_calls(self, response):
        """
        Extract tool calls from response.
        
        Args:
            response: Response object from generate_response
            
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        
        if hasattr(response, "content"):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", str(uuid.uuid4())),
                        "tool_name": block.get("name"),
                        "input": block.get("input", {})
                    })
        
        return tool_calls