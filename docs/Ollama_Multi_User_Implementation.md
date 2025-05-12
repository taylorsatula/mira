# Step-by-Step Implementation Guide: Multi-User Ollama Integration

This guide outlines how to add multi-user Ollama support to the existing application architecture.

## Phase 1: Core Implementation

### Step 1: Create RequestQueue Class
```python
# api/request_queue.py

import time
import threading
import queue
import uuid
import logging
from typing import Dict, Any, Optional

class RequestQueue:
    """Thread-safe request queue for managing Ollama API requests."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Singleton pattern to ensure one queue across all instances."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.logger = logging.getLogger("request_queue")
        self.request_queue = queue.Queue()
        self.results = {}
        self.lock = threading.RLock()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        self.logger.info("Request queue initialized with worker thread")
    
    def add_request(self, request_data):
        """Add a request to the queue."""
        request_id = str(uuid.uuid4())
        with self.lock:
            self.results[request_id] = {"status": "pending"}
        
        self.request_queue.put((request_id, request_data))
        self.logger.debug(f"Added request {request_id} to queue")
        return request_id
    
    def get_result(self, request_id, timeout=120):
        """Get the result for a request, waiting up to timeout seconds."""
        start_time = time.time()
        while timeout is None or (time.time() - start_time) < timeout:
            with self.lock:
                result = self.results.get(request_id, {})
                if result.get("status") in ["completed", "error"]:
                    # Mark for cleanup after delivery
                    result["retrieved"] = True
                    return result.get("data")
            
            # Small sleep to reduce CPU usage
            time.sleep(0.1)
        
        # Timeout occurred
        with self.lock:
            if request_id in self.results:
                self.results[request_id]["status"] = "timeout"
                self.results[request_id]["retrieved"] = True
        
        self.logger.warning(f"Request {request_id} timed out after {timeout}s")
        return {"error": "Request timed out"}
    
    def get_queue_position(self, request_id):
        """Get position in queue for a request."""
        with self.lock:
            if request_id not in self.results:
                return None
            
            # Count items in queue ahead of this one
            position = 0
            for item_id, _ in list(self.request_queue.queue):
                if item_id == request_id:
                    break
                position += 1
            
            queue_size = self.request_queue.qsize()
            return {"position": position if position < queue_size else "processing", 
                    "total": queue_size}
    
    def _process_queue(self):
        """Worker thread to process requests from the queue."""
        import requests
        
        while True:
            try:
                # Get next request from queue
                request_id, request_data = self.request_queue.get()
                self.logger.debug(f"Processing request {request_id}")
                
                try:
                    # Make the API call
                    response = requests.post(
                        f"{request_data['base_url']}/api/chat",
                        json=request_data['body'],
                        timeout=60
                    )
                    
                    if response.status_code != 200:
                        raise Exception(f"API error: {response.status_code} - {response.text}")
                    
                    # Store result
                    with self.lock:
                        self.results[request_id] = {
                            "status": "completed",
                            "data": response.json(),
                            "completed_at": time.time()
                        }
                    
                    self.logger.debug(f"Request {request_id} completed successfully")
                
                except Exception as e:
                    # Handle errors
                    with self.lock:
                        self.results[request_id] = {
                            "status": "error",
                            "data": {"error": str(e)},
                            "completed_at": time.time()
                        }
                    
                    self.logger.error(f"Error processing request {request_id}: {e}")
                
                # Mark task as done
                self.request_queue.task_done()
                
                # Cleanup old results
                self._cleanup_old_results()
            
            except Exception as e:
                self.logger.error(f"Unexpected error in queue processor: {e}")
                # Prevent tight loop on persistent errors
                time.sleep(1)
    
    def _cleanup_old_results(self):
        """Periodically clean up old results to prevent memory leaks."""
        with self.lock:
            current_time = time.time()
            to_remove = []
            
            for req_id, result in self.results.items():
                # Remove results that have been retrieved and are older than 5 minutes
                if (result.get("retrieved", False) and 
                    current_time - result.get("completed_at", 0) > 300):
                    to_remove.append(req_id)
                
                # Remove any results older than 30 minutes regardless
                elif current_time - result.get("completed_at", current_time) > 1800:
                    to_remove.append(req_id)
            
            # Remove identified items
            for req_id in to_remove:
                del self.results[req_id]
            
            if to_remove:
                self.logger.debug(f"Cleaned up {len(to_remove)} old request results")
```

### Step 2: Create OllamaBridge Class
```python
# api/ollama_bridge.py

import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Callable

from api.request_queue import RequestQueue
from errors import APIError, ErrorCode, error_context

class OllamaBridge:
    """
    Bridge to the Ollama API for local LLM inference.
    
    This provides the same interface as LLMBridge but connects
    to a local Ollama instance with multi-user support via queue.
    """
    
    def __init__(self, base_url="http://localhost:11434", model="qwen"):
        """
        Initialize the Ollama bridge.
        
        Args:
            base_url: URL of the Ollama server
            model: Name of the model to use
        """
        self.logger = logging.getLogger("ollama_bridge")
        self.base_url = base_url
        self.model = model
        
        # Get request queue singleton
        self.request_queue = RequestQueue.get_instance()
        
        self.logger.info(f"Ollama Bridge initialized with model {self.model}")
    
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
        
        This method follows the same interface as LLMBridge.generate_response.
        """
        with error_context(
            component_name="OllamaBridge",
            operation="generate response",
            error_class=APIError,
            error_code=ErrorCode.API_RESPONSE_ERROR,
            logger=self.logger
        ):
            # Prepare messages for OpenAI format
            formatted_messages = self._format_messages(messages, system_prompt, dynamic_content)
            
            # Convert tools to OpenAI format if present
            openai_tools = self._convert_tools(tools) if tools else None
            
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
                
            # Stream is not fully supported in this version
            if stream:
                self.logger.warning("Streaming is not fully supported with Ollama bridge")
            
            # Log the request (sanitized)
            self.logger.debug(f"Preparing Ollama request for model {self.model}")
            
            # Submit request to queue
            request_id = self.request_queue.add_request({
                "base_url": self.base_url,
                "body": request_body
            })
            
            # Wait for response
            response = self.request_queue.get_result(request_id)
            
            # Check for errors
            if "error" in response:
                raise APIError(
                    f"Ollama API error: {response['error']}",
                    ErrorCode.API_RESPONSE_ERROR
                )
            
            # Convert response to Anthropic format for compatibility
            return self._convert_to_anthropic_format(response)
    
    def _format_messages(self, messages, system_prompt=None, dynamic_content=None):
        """Format messages for OpenAI-compatible API."""
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
                    formatted_messages.append({
                        "role": "assistant",
                        "content": "\n".join(text_parts) if text_parts else ""
                    })
                    
                    # Add tool results as separate messages
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
                        "content": "\n".join(text_parts)
                    })
            else:
                # Regular message with string content
                formatted_messages.append(message)
        
        return formatted_messages
    
    def _convert_tools(self, tools):
        """Convert Anthropic tool definitions to OpenAI format."""
        openai_tools = []
        
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            }
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    def _convert_to_anthropic_format(self, ollama_response):
        """Convert Ollama/OpenAI response to Anthropic format."""
        
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
        response_message = ollama_response.get("choices", [{}])[0].get("message", {})
        response_text = response_message.get("content", "")
        
        # Add text block
        content.append({"type": "text", "text": response_text})
        
        # Process tool calls if present
        tool_calls = []
        if "tool_calls" in response_message:
            for tool_call in response_message["tool_calls"]:
                # Add tool use block in Anthropic format
                content.append({
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "input": json.loads(tool_call["function"]["arguments"])
                })
                
                tool_calls.append(tool_call)
        
        return AnthropicResponse(content, tool_calls)
    
    def extract_text_content(self, response):
        """Extract text content from response."""
        if hasattr(response, "content"):
            text_parts = []
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "".join(text_parts)
        return ""
    
    def extract_tool_calls(self, response):
        """Extract tool calls from response."""
        tool_calls = []
        
        if hasattr(response, "content"):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", str(uuid.uuid4())),
                        "tool_name": block.get("name"),
                        "input": block.get("input")
                    })
        
        return tool_calls
```

### Step 3: Update Config Structure
```python
# config/config.py
# (Add these sections to your existing config)

# LLM API configuration
api:
  provider: "anthropic"  # Options: "anthropic", "ollama"
  model: "claude-3-sonnet-20240229" 
  ollama_url: "http://localhost:11434"
  ollama_model: "qwen"  # Model to use with Ollama
```

### Step 4: Modify Main.py for Provider Selection
```python
# In main.py, modify initialize_system function

def initialize_system(args):
    # ...existing code...
    
    # Initialize LLM bridge based on configured provider
    provider = config.api.provider.lower()
    
    if provider == "ollama":
        from api.ollama_bridge import OllamaBridge
        llm_bridge = OllamaBridge(
            base_url=config.api.ollama_url,
            model=config.api.ollama_model
        )
        logger.info(f"Initialized Ollama bridge with model {config.api.ollama_model}")
    else:
        # Default to Anthropic
        from api.llm_bridge import LLMBridge
        llm_bridge = LLMBridge()
        logger.info(f"Initialized Anthropic bridge with model {config.api.model}")
    
    # ...continue with existing code...
```

## Phase 2: Enhancing the Implementation

### Step 5: Add Queue Status Information
```python
# api/request_queue.py
# Add this method to the RequestQueue class

def get_queue_stats(self):
    """Get current queue statistics."""
    with self.lock:
        queue_size = self.request_queue.qsize()
        active_requests = len([r for r in self.results.values() 
                              if r.get("status") == "pending"])
        
        # Calculate average processing time from recent completions
        processing_times = []
        cutoff_time = time.time() - 300  # Last 5 minutes
        
        for result in self.results.values():
            if (result.get("status") == "completed" and 
                result.get("completed_at", 0) > cutoff_time and
                "processing_time" in result):
                processing_times.append(result["processing_time"])
        
        avg_time = sum(processing_times) / len(processing_times) if processing_times else None
        
        return {
            "queue_size": queue_size,
            "active_requests": active_requests,
            "avg_processing_time": avg_time,
            "recent_completions": len(processing_times)
        }
```

### Step 6: Add Queue Status to Conversation.py
```python
# In conversation.py, modify generate_response method

def generate_response(self, user_input, ...):
    # Existing code...
    
    # Check if we're using Ollama and add queue status information
    if hasattr(self.llm_bridge, 'request_queue'):
        # Get request queue
        queue = self.llm_bridge.request_queue
        
        # If queue has multiple items, notify the user
        stats = queue.get_queue_stats()
        if stats['queue_size'] > 1:
            wait_estimate = stats['avg_processing_time'] * stats['queue_size'] / 2
            if wait_estimate:
                wait_msg = f"Your request is in a queue with {stats['queue_size']} other requests. "
                wait_msg += f"Estimated wait time: {int(wait_estimate)} seconds."
                
                # Add notification message
                self.add_message(
                    "assistant", 
                    wait_msg,
                    {"is_notification": True}
                )
    
    # Continue with normal processing...
```

### Step 7: Create Simple Monitoring Endpoint
```python
# Add to flask_app.py

@app.route('/api/status', methods=['GET'])
def get_api_status():
    """Return status information about the API."""
    try:
        from api.request_queue import RequestQueue
        
        # If using Ollama, include queue stats
        queue_stats = {}
        try:
            queue = RequestQueue.get_instance()
            queue_stats = queue.get_queue_stats()
        except Exception:
            pass
        
        return jsonify({
            "status": "ok",
            "provider": config.api.provider,
            "queue_stats": queue_stats
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
```

## Phase 3: Testing and Validation

### Step 8: Create Test for OllamaBridge
```python
# tests/test_ollama_bridge.py

import unittest
from unittest.mock import patch, MagicMock
import json

from api.ollama_bridge import OllamaBridge
from api.request_queue import RequestQueue

class TestOllamaBridge(unittest.TestCase):
    """Test cases for OllamaBridge."""
    
    def setUp(self):
        self.bridge = OllamaBridge(base_url="http://test", model="test-model")
        
        # Create a mock for the request queue
        patcher = patch('api.request_queue.RequestQueue.get_instance')
        self.mock_queue = MagicMock()
        self.mock_get_instance = patcher.start()
        self.mock_get_instance.return_value = self.mock_queue
        self.addCleanup(patcher.stop)
    
    def test_message_formatting(self):
        """Test formatting messages for Ollama."""
        system_prompt = "You are a helpful assistant."
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        formatted = self.bridge._format_messages(messages, system_prompt)
        
        self.assertEqual(len(formatted), 3)
        self.assertEqual(formatted[0]["role"], "system")
        self.assertEqual(formatted[0]["content"], system_prompt)
        self.assertEqual(formatted[1]["role"], "user")
        self.assertEqual(formatted[2]["role"], "assistant")
    
    def test_tool_conversion(self):
        """Test converting tool definitions."""
        tools = [{
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "test": {"type": "string"}
                }
            }
        }]
        
        converted = self.bridge._convert_tools(tools)
        
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["type"], "function")
        self.assertEqual(converted[0]["function"]["name"], "test_tool")
    
    def test_generate_response_flow(self):
        """Test the overall response generation flow."""
        # Mock queue response
        mock_response = {
            "choices": [{
                "message": {
                    "content": "Test response",
                    "tool_calls": [{
                        "id": "test-id",
                        "function": {
                            "name": "test_tool",
                            "arguments": "{\"test\": \"value\"}"
                        }
                    }]
                }
            }]
        }
        self.mock_queue.add_request.return_value = "test-id"
        self.mock_queue.get_result.return_value = mock_response
        
        # Test generate_response
        response = self.bridge.generate_response([
            {"role": "user", "content": "Test"}
        ])
        
        # Verify queue was used
        self.mock_queue.add_request.assert_called_once()
        self.mock_queue.get_result.assert_called_once()
        
        # Verify response conversion
        self.assertTrue(hasattr(response, "content"))
        self.assertEqual(len(response.content), 2)  # Text and tool_use
        self.assertEqual(response.content[0].get("text"), "Test response")
        self.assertEqual(response.content[1].get("type"), "tool_use")
    
    def test_extract_tool_calls(self):
        """Test extracting tool calls from response."""
        # Create mock response with tool calls
        class MockResponse:
            def __init__(self):
                self.content = [
                    {"type": "text", "text": "Test"},
                    {"type": "tool_use", "id": "test-id", "name": "test_tool", "input": {"test": "value"}}
                ]
        
        tool_calls = self.bridge.extract_tool_calls(MockResponse())
        
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["tool_name"], "test_tool")
        self.assertEqual(tool_calls[0]["input"], {"test": "value"})

if __name__ == '__main__':
    unittest.main()
```

### Step 9: Create Test for RequestQueue
```python
# tests/test_request_queue.py

import unittest
from unittest.mock import patch, MagicMock
import time
import threading

from api.request_queue import RequestQueue

class TestRequestQueue(unittest.TestCase):
    """Test cases for RequestQueue."""
    
    def setUp(self):
        # Reset singleton instance for each test
        RequestQueue._instance = None
        
        # Patch the worker thread to prevent actual processing
        patcher = patch.object(threading.Thread, 'start')
        self.mock_start = patcher.start()
        self.addCleanup(patcher.stop)
        
        self.queue = RequestQueue.get_instance()
    
    def test_singleton_pattern(self):
        """Test the singleton pattern."""
        queue1 = RequestQueue.get_instance()
        queue2 = RequestQueue.get_instance()
        self.assertIs(queue1, queue2)
    
    def test_add_request(self):
        """Test adding a request to the queue."""
        request_id = self.queue.add_request({"test": "data"})
        self.assertIsNotNone(request_id)
        self.assertEqual(self.queue.request_queue.qsize(), 1)
        
        with self.queue.lock:
            self.assertIn(request_id, self.queue.results)
            self.assertEqual(self.queue.results[request_id]["status"], "pending")
    
    def test_get_result_timeout(self):
        """Test getting a result with timeout."""
        request_id = self.queue.add_request({"test": "data"})
        
        # Get result with a short timeout
        result = self.queue.get_result(request_id, timeout=0.1)
        
        self.assertIn("error", result)
        self.assertIn("timeout", result["error"])
        
        with self.queue.lock:
            self.assertEqual(self.queue.results[request_id]["status"], "timeout")
    
    def test_get_result_complete(self):
        """Test getting a completed result."""
        request_id = self.queue.add_request({"test": "data"})
        
        # Manually set the result as completed
        with self.queue.lock:
            self.queue.results[request_id] = {
                "status": "completed",
                "data": {"result": "success"},
                "completed_at": time.time()
            }
        
        # Get the result
        result = self.queue.get_result(request_id)
        
        self.assertEqual(result, {"result": "success"})
        
        # Verify retrieved flag is set
        with self.queue.lock:
            self.assertTrue(self.queue.results[request_id].get("retrieved", False))
    
    def test_get_queue_position(self):
        """Test getting position in queue."""
        # Add three items to the queue
        ids = []
        for i in range(3):
            ids.append(self.queue.add_request({"test": i}))
        
        # Check position of second item
        position = self.queue.get_queue_position(ids[1])
        
        self.assertIsNotNone(position)
        self.assertEqual(position["position"], 1)  # Zero-indexed
        self.assertEqual(position["total"], 3)
    
    def test_cleanup_mechanism(self):
        """Test the cleanup mechanism for old results."""
        # Create old results
        with self.queue.lock:
            old_time = time.time() - 1800  # 30 minutes ago
            self.queue.results["old1"] = {
                "status": "completed",
                "completed_at": old_time,
                "data": {}
            }
            self.queue.results["old2"] = {
                "status": "error",
                "completed_at": old_time,
                "data": {}
            }
            
            # Create retrieved result that should be cleaned up
            retrieved_time = time.time() - 600  # 10 minutes ago
            self.queue.results["retrieved"] = {
                "status": "completed",
                "completed_at": retrieved_time,
                "retrieved": True,
                "data": {}
            }
            
            # Create recent result that should stay
            recent_time = time.time() - 60  # 1 minute ago
            self.queue.results["recent"] = {
                "status": "completed",
                "completed_at": recent_time,
                "data": {}
            }
        
        # Run cleanup
        self.queue._cleanup_old_results()
        
        # Check what remains
        with self.queue.lock:
            self.assertNotIn("old1", self.queue.results)
            self.assertNotIn("old2", self.queue.results)
            self.assertNotIn("retrieved", self.queue.results)
            self.assertIn("recent", self.queue.results)

if __name__ == '__main__':
    unittest.main()
```

## Deployment Steps

### Step 10: Install Ollama
```bash
# MacOS
curl -fsSL https://ollama.com/install.sh | sh

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Install Qwen model
ollama pull qwen
```

### Step 11: Update Configuration
```bash
# Update config file with Ollama settings
echo 'api:
  provider: "ollama"
  ollama_url: "http://localhost:11434"
  ollama_model: "qwen"' >> config/config.local.yml
```

### Step 12: Test the Integration
```bash
# Run in test mode first
python main.py --log-level=DEBUG
```

## Implementation Principles

This implementation follows key design principles:

1. **Minimal Changes**: Maintains the exact same interfaces as the existing LLMBridge
2. **Simple First Approach**: Starts with a simple queue-based solution that can be enhanced later
3. **Progressive Implementation**: Can be deployed incrementally with tests at each stage
4. **Clean Architecture**: Uses proper abstraction layers and separation of concerns
5. **Thread Safety**: Carefully handles concurrency issues with proper locking
6. **Memory Management**: Includes cleanup mechanisms to prevent memory leaks
7. **User Experience**: Provides feedback on queue position and expected wait times
8. **Monitoring**: Includes basic stats and monitoring endpoints

## Scaling Considerations

This implementation provides a foundation that can be scaled as your user base grows:

1. **For 1-10 users**: The simple queue system is sufficient
2. **For 10-50 users**: Consider:
   - Separating the queue into priority levels
   - Adding more performance monitoring
   - Implementing response caching for common requests

3. **For 50+ users**: Consider:
   - Deploying multiple Ollama instances with load balancing
   - Adding cloud API fallback during peak times
   - Implementing more sophisticated request routing

By starting with this clean design, you can easily extend the system as your needs grow without requiring a full rewrite.