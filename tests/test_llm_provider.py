"""
Production-grade tests for LLMProvider system.

Testing philosophy: TEST THE CONTRACT, NOT THE IMPLEMENTATION
Before every test: "What real production bug in OUR CODE would this catch?"
"""

import pytest
import time
from api.llm_provider import LLMProvider


@pytest.fixture
def real_provider():
    """Real LLMProvider instance connecting to actual Ollama."""
    return LLMProvider()


class TestBasicResponseContract:
    """Test the fundamental contract: messages in, structured response out."""
    
    def test_response_structure_is_consistent(self, real_provider):
        """
        Test that our response standardization produces consistent structure.
        
        REAL BUG THIS CATCHES: If _standardize_response() fails to convert
        the Ollama response format correctly, breaking downstream code that
        expects our standardized format.
        """
        messages = [{"role": "user", "content": "Hello"}]
        
        response = real_provider.generate_response(messages)
        
        # These are the contracts our code promises to downstream users
        assert isinstance(response, dict), "Response must be dict"
        assert "content" in response, "Response must have 'content' key"
        assert "usage" in response, "Response must have 'usage' key"
        assert isinstance(response["content"], list), "Content must be list"
        
        # If content exists, it must follow our standardized format
        if response["content"]:
            for content_block in response["content"]:
                assert isinstance(content_block, dict), "Content blocks must be dicts"
                assert "type" in content_block, "Content blocks must have 'type'"
                
                if content_block["type"] == "text":
                    assert "text" in content_block, "Text blocks must have 'text' key"
                    assert isinstance(content_block["text"], str), "Text must be string"


class TestRequestFormatting:
    """Test that we format requests correctly for the API."""
    
    def test_system_prompt_positioning(self, real_provider):
        """
        Test that system prompts are correctly positioned in the message array.
        
        REAL BUG THIS CATCHES: If our _build_request_body() method puts the
        system prompt in the wrong position or formats it incorrectly, 
        the LLM won't follow the system instructions correctly.
        """
        messages = [{"role": "user", "content": "Say 'CORRECT' if you understand"}]
        system_prompt = "You must always respond with exactly one word: CORRECT"
        
        # This tests our request building logic by seeing if the system prompt actually works
        response = real_provider.generate_response(messages, system_prompt=system_prompt)
        text = real_provider.extract_text_content(response)
        
        # If our request formatting is wrong, the system prompt won't be followed
        assert "CORRECT" in text.upper(), f"System prompt not followed. Got: {text}"


class TestErrorHandling:
    """Test that we handle real network/API errors correctly."""
    
    def test_nonexistent_endpoint_gives_clear_error(self, real_provider):
        """
        Test that connection failures give proper error classification.
        
        REAL BUG THIS CATCHES: If our error handling in _handle_http_error() 
        doesn't correctly classify connection errors, users get confusing 
        error messages and can't debug issues.
        """
        from unittest.mock import patch
        from errors import APIError, ErrorCode
        
        # Create provider then override endpoint to force connection error
        bad_provider = real_provider  # Start with working provider
        
        # Override the endpoint to point to a closed port
        bad_provider.api_endpoint = "http://127.0.0.1:99999/v1/chat/completions"
        bad_provider.timeout = 2
        bad_provider.max_retries = 0
        
        print(f"DEBUG: Provider endpoint: {bad_provider.api_endpoint}")
        print(f"DEBUG: Provider timeout: {bad_provider.timeout}")
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Test should raise APIError with connection error
        with pytest.raises(APIError) as exc_info:
            bad_provider.generate_response(messages)
        
        # Our error handling should classify this as connection error
        error = exc_info.value
        print(f"DEBUG: Got APIError: {error}")
        
        # Should contain helpful error message about network/connection issue
        error_str = str(error).lower()
        assert "network error" in error_str or "connection" in error_str or "failed to parse" in error_str


class TestContentExtraction:
    """Test that we extract content correctly from real responses."""
    
    def test_extract_text_from_real_response(self, real_provider):
        """
        Test that extract_text_content() works with actual LLM responses.
        
        REAL BUG THIS CATCHES: If our _standardize_response() creates a 
        response format that our extract_text_content() can't handle,
        downstream code will break when trying to get the LLM's answer.
        """
        messages = [{"role": "user", "content": "Say exactly: The cat is on the mat"}]
        
        # Get a real response from the LLM
        response = real_provider.generate_response(messages)
        
        # Test our extraction utility
        extracted_text = real_provider.extract_text_content(response)
        
        # Our extraction should work and return the actual text
        assert isinstance(extracted_text, str), "Extracted text must be string"
        assert len(extracted_text) > 0, "Should extract non-empty text"
        assert "cat" in extracted_text.lower(), f"Should contain expected content. Got: {extracted_text}"


class TestParameterOverrides:
    """Test that parameter overrides actually work."""
    
    def test_max_tokens_override_limits_response(self, real_provider):
        """
        Test that max_tokens parameter actually limits response length.
        
        REAL BUG THIS CATCHES: If our _build_request_body() method doesn't 
        correctly apply parameter overrides, users can't control LLM behavior.
        This could cause runaway costs or unexpected response lengths.
        """
        messages = [{"role": "user", "content": "Write a very long story about a dragon"}]
        
        # Test with very low max_tokens
        short_response = real_provider.generate_response(messages, max_tokens=10)
        short_text = real_provider.extract_text_content(short_response)
        
        # Test with higher max_tokens
        long_response = real_provider.generate_response(messages, max_tokens=100)
        long_text = real_provider.extract_text_content(long_response)
        
        # If our parameter override works, short should be shorter
        short_word_count = len(short_text.split())
        long_word_count = len(long_text.split())
        
        assert short_word_count > 0, "Should get some response even with low max_tokens"
        assert long_word_count > 0, "Should get response with higher max_tokens"
        assert short_word_count <= long_word_count, f"Short response ({short_word_count} words) should be <= long response ({long_word_count} words)"


class TestStreaming:
    """Test that streaming functionality works correctly."""
    
    def test_streaming_calls_callback_with_real_content(self, real_provider):
        """
        Test that streaming actually calls the callback with content chunks.
        
        REAL BUG THIS CATCHES: If our _handle_streaming_request() method has bugs
        in parsing the SSE stream or calling the callback, streaming appears to work
        but users get no progressive content - breaking the user experience.
        """
        messages = [{"role": "user", "content": "Count from 1 to 5"}]
        
        # Collect streaming chunks
        collected_chunks = []
        def test_callback(content: str):
            collected_chunks.append(content)
        
        # Test streaming
        response = real_provider.generate_response(
            messages, 
            stream=True, 
            callback=test_callback
        )
        
        # Our streaming should have called the callback
        assert len(collected_chunks) > 0, "Callback should receive content chunks"
        
        # Streamed content should match final response  
        streamed_text = "".join(collected_chunks)
        final_text = real_provider.extract_text_content(response)
        assert streamed_text == final_text, f"Streamed '{streamed_text}' != final '{final_text}'"
        
        # Should contain the counting content
        assert any(str(i) in streamed_text for i in range(1, 6)), f"Should contain counting. Got: {streamed_text}"


class TestThreadSafety:
    """Test that concurrent requests don't break our code."""
    
    def test_concurrent_requests_dont_corrupt_responses(self, real_provider):
        """
        Test that multiple concurrent requests don't interfere with each other.
        
        REAL BUG THIS CATCHES: If our HTTP session sharing or any instance variables
        have thread safety issues, concurrent requests could get mixed up responses,
        corrupted data, or crashes. This happens in production with multiple users.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def make_unique_request(request_id):
            messages = [{"role": "user", "content": f"Reply with only this JSON: {{\"request_id\": {request_id}, \"status\": \"success\"}}"}]
            try:
                response = real_provider.generate_response(messages)
                text = real_provider.extract_text_content(response)
                return {
                    "success": True,
                    "request_id": request_id,
                    "text": text
                }
            except Exception as e:
                return {
                    "success": False, 
                    "request_id": request_id,
                    "error": str(e)
                }
        
        # Run 4 concurrent requests with unique identifiers
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_unique_request, i) for i in range(1, 5)]
            results = [future.result() for future in as_completed(futures)]
        
        # All requests should succeed
        successful = [r for r in results if r["success"]]
        assert len(successful) == 4, f"Expected 4 successes, got {len(successful)}"
        
        # Debug: Print all responses to understand what's happening
        print("\nDEBUG: All responses:")
        for result in successful:
            print(f"  Request {result['request_id']} -> Response: {result['text']}")
        
        # Each should get the correct response (no cross-contamination)
        for result in successful:
            request_id = result["request_id"]
            text = result["text"]
            # Should contain the unique request_id in JSON format
            assert f"\"request_id\": {request_id}" in text or f"{request_id}" in text, f"Request {request_id} got wrong response: {text}"


class TestTimeoutHandling:
    """Test that timeouts are handled correctly."""
    
    def test_timeout_gives_proper_error_code(self):
        """
        Test that actual timeouts result in correct error classification.
        
        REAL BUG THIS CATCHES: If our timeout handling in _handle_standard_request()
        doesn't properly catch TimeoutException or classifies it wrong, users get
        confusing errors instead of clear "request timed out" messages.
        """
        from unittest.mock import patch
        from errors import APIError, ErrorCode
        
        # Create provider with very short timeout and slow endpoint
        with patch('config.config') as mock_config:
            mock_config.api.provider = "local"
            mock_config.api.api_endpoint = "http://httpbin.org/delay/5"  # 5 second delay
            mock_config.api.model = "test-model"
            mock_config.api.max_tokens = 100
            mock_config.api.temperature = 0.7
            mock_config.api.max_retries = 0  # No retries to speed up test
            mock_config.api.timeout = 1  # 1 second timeout
            mock_config.api_key = ""
            
            timeout_provider = LLMProvider()
        
        messages = [{"role": "user", "content": "Hello"}]
        
        start_time = time.time()
        with pytest.raises(APIError) as exc_info:
            timeout_provider.generate_response(messages)
        elapsed = time.time() - start_time
        
        # Should timeout quickly (not wait for the full 5 second delay)
        assert elapsed < 3, f"Should timeout quickly, took {elapsed:.2f}s"
        
        # Our error handling should classify this as timeout
        error = exc_info.value
        assert error.error_code == ErrorCode.API_TIMEOUT_ERROR
        assert "timed out" in str(error).lower() or "timeout" in str(error).lower()


class TestMalformedResponseHandling:
    """Test that we handle non-JSON responses correctly."""
    
    def test_html_response_gives_json_decode_error(self):
        """
        Test that non-JSON responses are handled with proper error messages.
        
        REAL BUG THIS CATCHES: If our JSON parsing in _handle_standard_request()
        doesn't properly catch JSONDecodeError, or gives unhelpful error messages,
        users can't debug when APIs return HTML error pages instead of JSON.
        """
        from unittest.mock import patch
        from errors import APIError, ErrorCode
        
        # Use endpoint that returns HTML instead of JSON
        with patch('config.config') as mock_config:
            mock_config.api.provider = "local"
            mock_config.api.api_endpoint = "http://httpbin.org/html"  # Returns HTML page
            mock_config.api.model = "test-model"
            mock_config.api.max_tokens = 100
            mock_config.api.temperature = 0.7
            mock_config.api.max_retries = 0
            mock_config.api.timeout = 10
            mock_config.api_key = ""
            
            html_provider = LLMProvider()
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(APIError) as exc_info:
            html_provider.generate_response(messages)
        
        # Our JSON parsing error handling should catch this
        error = exc_info.value
        assert error.error_code == ErrorCode.API_RESPONSE_ERROR
        assert "json" in str(error).lower(), f"Should mention JSON parsing issue. Got: {error}"


class TestIntegrationWorkflow:
    """Test complete workflows that users actually do."""
    
    def test_conversation_with_context_works_end_to_end(self, real_provider):
        """
        Test that multi-turn conversations work correctly from start to finish.
        
        REAL BUG THIS CATCHES: If our message formatting, request building, 
        response parsing, or content extraction has bugs, multi-turn conversations
        break - which is a core use case that would break the entire application.
        """
        # Start a conversation
        messages = [{"role": "user", "content": "My favorite color is blue"}]
        
        response1 = real_provider.generate_response(messages)
        assistant_reply = real_provider.extract_text_content(response1)
        
        # Should get a response about blue
        assert len(assistant_reply) > 0, "Should get initial response"
        
        # Continue the conversation with context
        messages.append({"role": "assistant", "content": assistant_reply})
        messages.append({"role": "user", "content": "What's my favorite color?"})
        
        response2 = real_provider.generate_response(messages)
        final_reply = real_provider.extract_text_content(response2)
        
        # Should remember the context (blue)
        assert len(final_reply) > 0, "Should get contextual response"
        assert "blue" in final_reply.lower(), f"Should remember favorite color. Got: {final_reply}"
        
        # Test that our response format is consistent across conversation
        for response in [response1, response2]:
            assert "content" in response
            assert "usage" in response
            assert isinstance(response["content"], list)


class TestToolUsage:
    """Test that tool calling functionality works correctly."""
    
    def test_tool_definitions_passed_correctly_to_api(self, real_provider):
        """
        Test that tool definitions are correctly formatted and passed to the LLM API.
        
        REAL BUG THIS CATCHES: If our _build_request_body() method doesn't correctly
        format or include tool definitions, the LLM won't know about available tools
        and can't make tool calls - breaking function calling entirely.
        """
        # Define a simple tool
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country, e.g. 'London, UK'"
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
        
        messages = [{"role": "user", "content": "What's the weather like in Paris?"}]
        
        # Test that tools parameter is accepted without errors
        response = real_provider.generate_response(messages, tools=tools)
        
        # Should get a valid response structure
        assert isinstance(response, dict)
        assert "content" in response
        assert "usage" in response
        
        # Response should either contain text or tool calls (or both)
        text = real_provider.extract_text_content(response)
        tool_calls = real_provider.extract_tool_calls(response)
        
        # Must have some response content
        assert len(text) > 0 or len(tool_calls) > 0, "Should get either text response or tool calls"
        
        # If tool calls were made, they should be about weather
        for tool_call in tool_calls:
            assert tool_call["tool_name"] == "get_weather", f"Unexpected tool call: {tool_call}"
            assert "location" in tool_call["input"], f"Tool call missing location: {tool_call}"
    
    def test_tool_call_extraction_from_real_llm_response(self, real_provider):
        """
        Test that tool call extraction works with actual LLM tool call responses.
        
        REAL BUG THIS CATCHES: If our _standardize_response() method doesn't correctly
        convert tool_calls from the actual API response format, or if extract_tool_calls()
        has bugs parsing the standardized format, tool calling breaks end-to-end.
        """
        # Define a tool that the LLM might actually call
        tools = [{
            "type": "function", 
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to calculate, e.g. '2 + 2'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }]
        
        # Ask a question that should trigger tool use
        messages = [{"role": "user", "content": "What is 15 + 27? Please use the calculate tool."}]
        
        # Get real response from LLM with tools
        response = real_provider.generate_response(messages, tools=tools)
        
        # Test our extraction methods work with the real response
        text = real_provider.extract_text_content(response)
        tool_calls = real_provider.extract_tool_calls(response)
        
        # Should get valid response structure from our standardization
        assert isinstance(response, dict)
        assert "content" in response
        assert isinstance(response["content"], list)
        
        # Should get either text response or tool calls (or both)
        assert len(text) > 0 or len(tool_calls) > 0, "Should get some response content"
        
        # If tool calls were made, verify our extraction works correctly
        for tool_call in tool_calls:
            # Our extract_tool_calls should return this format
            assert "id" in tool_call, "Tool call should have id"
            assert "tool_name" in tool_call, "Tool call should have tool_name" 
            assert "input" in tool_call, "Tool call should have input"
            assert isinstance(tool_call["input"], dict), "Tool call input should be dict"
            
            # For math tool calls, should have reasonable content
            if tool_call["tool_name"] == "calculate":
                assert "expression" in tool_call["input"], "Calculate call should have expression"
                expr = tool_call["input"]["expression"]
                assert any(char in expr for char in "0123456789+"), f"Expression should contain math: {expr}"