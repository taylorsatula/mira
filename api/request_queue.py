"""
Request queue system for managing concurrent Ollama API requests.

This module implements a thread-safe queue for handling multiple requests to
the Ollama API, enabling multi-user support for a single Ollama instance.
It includes proper cancellation of timed-out or abandoned requests and
supports streaming responses.
"""
import time
import threading
import queue
import uuid
import logging
import json
import requests
from typing import Dict, Any, Optional, Callable

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
        # Use priority queue so we can cancel items
        self.request_queue = queue.PriorityQueue()
        self.results = {}
        self.lock = threading.RLock()
        
        # Currently processing request_id
        self.current_processing = None
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        self.logger.info("Request queue initialized with worker thread")
    
    def add_request(self, request_data, priority=1):
        """
        Add a request to the queue.
        
        Args:
            request_data: Dictionary containing request information
            priority: Priority level (lower is higher priority)
            
        Returns:
            request_id: Unique ID for tracking the request
        """
        request_id = str(uuid.uuid4())
        with self.lock:
            self.results[request_id] = {
                "status": "pending", 
                "created_at": time.time(),
                "cancelled": False
            }
        
        # Put in queue with priority and timestamp for sorting
        queue_item = (priority, time.time(), request_id, request_data)
        self.request_queue.put(queue_item)
        self.logger.debug(f"Added request {request_id} to queue with priority {priority}")
        return request_id
    
    def add_streaming_request(self, request_data, callback, priority=1):
        """
        Add a streaming request to the queue.
        
        Args:
            request_data: Dictionary containing request information
            callback: Callback function to receive streaming chunks
            priority: Priority level (lower is higher priority)
            
        Returns:
            request_id: Unique ID for tracking the request
        """
        request_id = str(uuid.uuid4())
        with self.lock:
            self.results[request_id] = {
                "status": "pending", 
                "created_at": time.time(),
                "cancelled": False,
                "is_streaming": True,
                "stream_callback": callback,
                "stream_complete": False
            }
        
        # Put in queue with priority and timestamp for sorting
        queue_item = (priority, time.time(), request_id, request_data)
        self.request_queue.put(queue_item)
        self.logger.debug(f"Added streaming request {request_id} to queue with priority {priority}")
        return request_id
    
    def cancel_request(self, request_id):
        """
        Cancel a pending request if it hasn't been processed yet.
        
        Args:
            request_id: The ID of the request to cancel
            
        Returns:
            bool: True if request was cancelled, False if not found or already processed
        """
        with self.lock:
            # Check if request exists
            if request_id not in self.results:
                return False
                
            result = self.results[request_id]
            
            # If already completed, can't cancel
            if result.get("status") in ["completed", "error"]:
                return False
                
            # Mark as cancelled
            result["cancelled"] = True
            result["status"] = "cancelled"
            
            # If this is the currently processing request, we can't remove it
            # from the queue, but the worker will check the cancelled flag
            if self.current_processing == request_id:
                self.logger.debug(f"Marked in-progress request {request_id} as cancelled")
                return True
                
            self.logger.debug(f"Cancelled pending request {request_id}")
            return True
    
    def get_result(self, request_id, timeout=120):
        """
        Get the result for a request, waiting up to timeout seconds.
        
        Args:
            request_id: The ID of the request to get results for
            timeout: Maximum time to wait in seconds
            
        Returns:
            The result data, or an error dictionary if timeout occurs
        """
        start_time = time.time()
        
        with self.lock:
            # Check if this is a streaming request
            result = self.results.get(request_id, {})
            if result.get("is_streaming", False):
                # For streaming requests, we wait for completion rather than results
                is_streaming = True
            else:
                is_streaming = False
        
        if is_streaming:
            # For streaming requests, wait until streaming is complete or timeout
            while timeout is None or (time.time() - start_time) < timeout:
                with self.lock:
                    result = self.results.get(request_id, {})
                    if result.get("stream_complete", False):
                        # Streaming is complete, return the final result if available
                        result["retrieved"] = True
                        final_result = result.get("final_result", {"choices": [{"message": {"content": ""}}]})
                        return final_result
                
                # Small sleep to reduce CPU usage
                time.sleep(0.1)
            
            # Timeout occurred - cancel the request
            self.cancel_request(request_id)
            return {"error": "Request timed out and was cancelled"}
        else:
            # Standard non-streaming request
            while timeout is None or (time.time() - start_time) < timeout:
                with self.lock:
                    result = self.results.get(request_id, {})
                    if result.get("status") in ["completed", "error"]:
                        # Mark for cleanup after delivery
                        result["retrieved"] = True
                        return result.get("data")
                
                # Small sleep to reduce CPU usage
                time.sleep(0.1)
            
            # Timeout occurred - cancel the request
            self.cancel_request(request_id)
            
            self.logger.warning(f"Request {request_id} timed out after {timeout}s and was cancelled")
            return {"error": "Request timed out and was cancelled"}
    
    def get_queue_position(self, request_id):
        """
        Get position in queue for a request.
        
        Args:
            request_id: The ID of the request to get position for
            
        Returns:
            Dictionary with position and total queue size, or None if not found
        """
        with self.lock:
            if request_id not in self.results:
                return None
                
            # If this is currently processing, report that
            if self.current_processing == request_id:
                return {
                    "position": "processing", 
                    "total": self.request_queue.qsize()
                }
                
            # Check if cancelled
            if self.results[request_id].get("cancelled", False):
                return {
                    "position": "cancelled",
                    "total": self.request_queue.qsize()
                }
                
            # If already completed, it's not in queue
            if self.results[request_id].get("status") in ["completed", "error"]:
                return {
                    "position": "completed",
                    "total": self.request_queue.qsize()
                }
            
            # Approximate position by checking all items in queue
            # Note: This is inefficient for large queues but works for our scale
            position = None
            queue_items = list(self.request_queue.queue)
            for i, item in enumerate(queue_items):
                _, _, item_id, _ = item
                if item_id == request_id:
                    position = i
                    break
            
            # If not found in queue but pending, it might have just been picked up
            queue_size = self.request_queue.qsize()
            return {
                "position": position if position is not None else "processing", 
                "total": queue_size
            }
    
    def get_queue_stats(self):
        """
        Get current queue statistics.
        
        Returns:
            Dictionary with queue statistics
        """
        with self.lock:
            queue_size = self.request_queue.qsize()
            active_requests = len([r for r in self.results.values() 
                                  if r.get("status") == "pending" and 
                                     not r.get("cancelled", False)])
            
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
    
    def _process_queue(self):
        """Worker thread to process requests from the queue."""
        while True:
            try:
                # Get next request from queue
                priority, _, request_id, request_data = self.request_queue.get()
                
                # Check if this request was cancelled while in queue
                with self.lock:
                    result = self.results.get(request_id, {})
                    if result.get("cancelled", False):
                        self.logger.debug(f"Skipping cancelled request {request_id}")
                        self.request_queue.task_done()
                        continue
                    
                    # Mark as currently processing
                    self.current_processing = request_id
                
                self.logger.debug(f"Processing request {request_id} (priority {priority})")
                
                # Check if this is a streaming request
                is_streaming = result.get("is_streaming", False)
                
                start_time = time.time()
                
                try:
                    if is_streaming:
                        # Handle streaming request
                        self._process_streaming_request(request_id, request_data, result)
                    else:
                        # Handle standard request
                        self._process_standard_request(request_id, request_data)
                    
                except Exception as e:
                    # Handle errors
                    with self.lock:
                        if self.results[request_id].get("cancelled", False):
                            self.logger.debug(f"Cancelled request {request_id} encountered error: {e}")
                        else:
                            self.results[request_id] = {
                                "status": "error",
                                "data": {"error": str(e)},
                                "completed_at": time.time()
                            }
                        self.current_processing = None
                    
                    self.logger.error(f"Error processing request {request_id}: {e}")
                
                # Mark task as done
                self.request_queue.task_done()
                
                # Cleanup old results
                self._cleanup_old_results()
            
            except Exception as e:
                # Reset current processing in case of unexpected error
                self.current_processing = None
                self.logger.error(f"Unexpected error in queue processor: {e}")
                # Prevent tight loop on persistent errors
                time.sleep(1)
    
    def _process_standard_request(self, request_id, request_data):
        """Process a standard (non-streaming) request."""
        response = requests.post(
            f"{request_data['base_url']}/api/chat",
            json=request_data['body'],
            timeout=60
        )
        
        processing_time = time.time() - self.results[request_id]["created_at"]
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        # Before storing result, check if request was cancelled during processing
        with self.lock:
            if self.results[request_id].get("cancelled", False):
                self.logger.debug(f"Request {request_id} was cancelled during processing")
                self.current_processing = None
                return
        
        # Store result
        with self.lock:
            self.results[request_id] = {
                "status": "completed",
                "data": response.json(),
                "completed_at": time.time(),
                "processing_time": processing_time
            }
            self.current_processing = None
        
        self.logger.debug(f"Request {request_id} completed successfully in {processing_time:.2f}s")
    
    def _process_streaming_request(self, request_id, request_data, result):
        """Process a streaming request."""
        # Modify request to enable streaming
        body = request_data['body'].copy()
        body['stream'] = True
        
        # Get the callback function
        callback = result.get("stream_callback")
        if not callback:
            raise Exception("Streaming request missing callback function")
        
        # Keep tracking this as currently processing to block the queue
        # Unlike normal requests, we'll keep this marked as processing until
        # streaming completes
        
        self.logger.debug(f"Starting streaming request {request_id}")
        
        # Make streaming request
        with requests.post(
            f"{request_data['base_url']}/api/chat",
            json=body,
            stream=True,
            timeout=120
        ) as response:
            
            if response.status_code != 200:
                with self.lock:
                    self.results[request_id] = {
                        "status": "error",
                        "data": {"error": f"API error: {response.status_code}"},
                        "completed_at": time.time(),
                        "stream_complete": True
                    }
                    self.current_processing = None
                raise Exception(f"API error: {response.status_code}")
            
            # Initialize final response
            final_response = {
                "choices": [
                    {
                        "message": {
                            "content": ""
                        }
                    }
                ]
            }
            
            # Process streaming response
            for line in response.iter_lines():
                # Check if request was cancelled
                with self.lock:
                    if self.results[request_id].get("cancelled", False):
                        self.logger.debug(f"Streaming request {request_id} was cancelled during processing")
                        self.current_processing = None
                        return
                
                if not line:
                    continue
                
                try:
                    # Parse the line
                    data = line.decode('utf-8')
                    if data.startswith('data: '):
                        data = data[6:]  # Remove 'data: ' prefix
                    
                    # Handle special case of [DONE] marker
                    if data == '[DONE]':
                        break
                    
                    # Parse JSON chunk
                    chunk = json.loads(data)

                    # Extract the content delta
                    delta = ""
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        choice = chunk['choices'][0]
                        if 'delta' in choice and 'content' in choice['delta']:
                            delta = choice['delta']['content']
                        elif 'text' in choice:
                            delta = choice['text']
                    # Handle Ollama-specific response format
                    elif 'message' in chunk and 'content' in chunk['message']:
                        delta = chunk['message']['content']

                    # Don't include internal <think> tags from Ollama
                    if delta == "<think>" or delta == "</think>":
                        delta = ""


                    # Append to final response
                    if delta:
                        final_response['choices'][0]['message']['content'] += delta

                        # Call the callback with the delta
                        callback(delta)
                    
                    # Check for tool calls in the chunk
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        choice = chunk['choices'][0]
                        if 'delta' in choice and 'tool_calls' in choice['delta']:
                            # Tool calls in streaming aren't fully supported yet
                            # Just note their presence for now
                            self.logger.debug(f"Tool call detected in stream for request {request_id}")
                
                except Exception as e:
                    self.logger.error(f"Error processing streaming chunk: {e}")
            
            # Streaming complete
            processing_time = time.time() - self.results[request_id]["created_at"]
            
            with self.lock:
                self.results[request_id].update({
                    "status": "completed",
                    "final_result": final_response,
                    "completed_at": time.time(),
                    "processing_time": processing_time,
                    "stream_complete": True
                })
                self.current_processing = None
            
            self.logger.debug(f"Streaming request {request_id} completed in {processing_time:.2f}s")
    
    def _cleanup_old_results(self):
        """Periodically clean up old results to prevent memory leaks."""
        with self.lock:
            current_time = time.time()
            to_remove = []
            
            for req_id, result in self.results.items():
                # Skip currently processing requests
                if self.current_processing == req_id:
                    continue
                
                # Remove cancelled requests after 5 minutes
                if (result.get("cancelled", False) and 
                    current_time - result.get("created_at", 0) > 300):
                    to_remove.append(req_id)
                    
                # Remove results that have been retrieved and are older than 5 minutes
                elif (result.get("retrieved", False) and 
                      current_time - result.get("completed_at", 0) > 300):
                    to_remove.append(req_id)
                
                # Remove completed streaming requests after 5 minutes
                elif (result.get("is_streaming", False) and 
                      result.get("stream_complete", False) and
                      current_time - result.get("completed_at", 0) > 300):
                    to_remove.append(req_id)
                
                # Remove any results older than 30 minutes regardless
                elif current_time - result.get("created_at", current_time) > 1800:
                    to_remove.append(req_id)
            
            # Remove identified items
            for req_id in to_remove:
                del self.results[req_id]
            
            if to_remove:
                self.logger.debug(f"Cleaned up {len(to_remove)} old request results")