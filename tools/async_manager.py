"""
Asynchronous task management for the AI agent system.

This module provides the ability to run tools asynchronously
in the background without blocking the main conversation flow.
"""
import logging
import threading
import time
import uuid
from queue import Queue
from typing import Dict, Any, Optional, List

from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository
from errors import ToolError, ErrorCode


class TaskStatus:
    """Constants for task status tracking."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AsyncTask:
    """
    Representation of an asynchronous task.
    
    Tracks execution state, results, and metadata for
    a background tool operation.
    """
    
    def __init__(
        self,
        task_id: str,
        description: str,
        notify_on_completion: bool = False
    ):
        """
        Initialize an async task.
        
        Args:
            task_id: Unique identifier for the task
            description: Human-readable description of the task
            notify_on_completion: Whether to send notification when complete
        """
        self.task_id = task_id
        self.description = description
        self.notify_on_completion = notify_on_completion
        
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary representation.
        
        Returns:
            Dictionary with task details
        """
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "notify_on_completion": self.notify_on_completion
        }


class AsyncTaskManager:
    """
    Manager for asynchronous task execution.
    
    Handles task scheduling, execution, and status tracking
    for background operations.
    """
    
    def __init__(self, tool_repo: Optional[ToolRepository] = None, llm_bridge: Optional[LLMBridge] = None):
        """
        Initialize the async task manager.
        
        Args:
            tool_repo: Tool repository for tool execution
            llm_bridge: LLM bridge for assistant interactions
        """
        self.logger = logging.getLogger("async_task_manager")
        self.tasks: Dict[str, AsyncTask] = {}
        self.task_queue = Queue()
        
        # Components for task execution
        self.tool_repo = tool_repo
        self.llm_bridge = llm_bridge
        
        # Create directory for async results if it doesn't exist
        import os
        os.makedirs("persistent/async_results", exist_ok=True)
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        # Notification callback
        self.notification_callback = None
        
        self.logger.info("AsyncTaskManager initialized")
    
    def schedule_task(
        self,
        description: str,
        task_prompt: str,
        notify_on_completion: bool = False
    ) -> str:
        """
        Schedule a task for background execution.
        
        Args:
            description: Human-readable description of the task
            task_prompt: Prompt for the background LLM assistant
            notify_on_completion: Whether to notify when complete
            
        Returns:
            Task ID for tracking the task
        """
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create and store task
        task = AsyncTask(
            task_id=task_id,
            description=description,
            notify_on_completion=notify_on_completion
        )
        self.tasks[task_id] = task
        
        # Queue task for execution
        self.task_queue.put((task, task_prompt))
        
        self.logger.info(f"Scheduled async task: {task_id} - {description}")
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            Dictionary with task status information
            
        Raises:
            ToolError: If task not found
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ToolError(
                f"Task not found: {task_id}",
                ErrorCode.TOOL_NOT_FOUND
            )
        
        return task.to_dict()
    
    def set_notification_callback(self, callback):
        """
        Set callback function for task completion notifications.
        
        Args:
            callback: Function to call with task info on completion
        """
        self.notification_callback = callback
    
    def _worker_loop(self):
        """
        Background worker thread for executing tasks.
        
        Runs continuously, processing tasks from the queue.
        """
        while True:
            try:
                # Get next task from queue
                task, task_prompt = self.task_queue.get()
                
                # Handle None task (for testing) or skip completed tasks
                if task is None:
                    self.task_queue.task_done()
                    continue
                
                # Skip if task already completed or failed
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    self.task_queue.task_done()
                    continue
                
                # Mark task as running
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                
                # Ensure we have the required components
                if not self.llm_bridge or not self.tool_repo:
                    task.status = TaskStatus.FAILED
                    task.error = "Task manager not properly initialized (missing LLM Bridge or Tool Repository)"
                    task.completed_at = time.time()
                    self.task_queue.task_done()
                    continue
                
                try:
                    # Execute task using background LLM
                    result = self._execute_task_with_llm(task, task_prompt)
                    
                    # Mark task as completed
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = time.time()
                    
                    # Send notification if requested
                    if task.notify_on_completion and self.notification_callback:
                        self.notification_callback(task)
                    
                    self.logger.info(f"Completed async task: {task.task_id}")
                
                except Exception as e:
                    # Handle execution error
                    self.logger.error(f"Error executing async task {task.task_id}: {e}")
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = time.time()
                    
                    # Send notification for failures too
                    if task.notify_on_completion and self.notification_callback:
                        self.notification_callback(task)
                
                finally:
                    # Mark task as done in queue
                    self.task_queue.task_done()
            
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
    
    def _execute_task_with_llm(self, task: AsyncTask, task_prompt: str) -> Any:
        """
        Execute a task using the background LLM.
        
        Args:
            task: The task to execute
            task_prompt: Prompt for the background LLM
            
        Returns:
            Result of the task execution
        """
        # Prepare system prompt for background LLM
        system_prompt = """You are a background task executor that uses tools to complete tasks. You are NOT having a conversation - your only job is to execute tools correctly.

PERSISTENCE TOOL USAGE:
- CORRECT operations: "get", "set", "delete", "list" (NEVER use "read", "write", or other invalid operations)
- CORRECT format: persistence(filename="async_results/{task_id}.json", operation="set", key="result", value=data)
- ALWAYS save your final results with: persistence(filename="async_results/{task_id}.json", operation="set", key="result", value=data)

TASK EXECUTION STEPS:
1. Process the assigned task
2. Execute necessary tools in sequence
3. Save the FINAL result using ONLY the persistence tool with operation="set"
4. Return a SHORT completion confirmation message

CRITICAL RULES:
- Use EXACT operation names: "get"/"set"/"delete"/"list" for persistence tool
- When extracting information, specify the correct template
- ALL results MUST be saved to "async_results/{task_id}.json"
- Use simple text for your final response, NOT JSON or other structured formats
- DO NOT use markdown formatting in your responses

EXAMPLE CORRECT USAGE:
1. extract(message="user message", template="food_preferences")  
2. persistence(filename="preferences.json", operation="get", key="preferences")
3. persistence(filename="async_results/{task_id}.json", operation="set", key="result", value=analyzed_data)
4. Return: "Task completed and results saved."

You are working autonomously without user interaction."""
        
        # Add task ID to system prompt
        system_prompt = system_prompt.replace("{task_id}", task.task_id)
        
        # Create message for the background LLM
        messages = [
            {"role": "user", "content": task_prompt}
        ]
        
        # Get tool definitions for the background LLM
        tools = self.tool_repo.get_all_tool_definitions()
        
        # Generate response with tools
        response = self.llm_bridge.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools
        )
        
        # Process response and tool calls
        result = self._process_background_llm_response(response, task)
        
        return result
    
    def _process_background_llm_response(self, response, task: AsyncTask) -> Any:
        """
        Process the background LLM response.
        
        Handles tool calls and result extraction from the background LLM.
        
        Args:
            response: LLM response
            task: The async task being processed
            
        Returns:
            Task execution result
        """
        # Maximum iterations to prevent infinite loops
        max_iterations = 10
        iterations = 0
        final_result = None
        
        while iterations < max_iterations:
            iterations += 1
            
            # Extract tool calls from response
            tool_calls = self.llm_bridge.extract_tool_calls(response)
            
            # If no tool calls, use text response as result
            if not tool_calls:
                text_content = self.llm_bridge.extract_text_content(response)
                final_result = text_content
                break
            
            # Process tool calls
            tool_results = {}
            for tool_call in tool_calls:
                tool_name = tool_call["tool_name"]
                tool_input = tool_call["input"]
                tool_id = tool_call["id"]
                
                try:
                    # Invoke the tool
                    result = self.tool_repo.invoke_tool(tool_name, tool_input)
                    tool_results[tool_id] = {
                        "content": str(result),
                        "is_error": False
                    }
                    
                    self.logger.debug(f"Background tool call successful: {tool_name}")
                    
                    # Check if this was a persistence tool call saving the final result
                    if tool_name == "persistence" and tool_input.get("operation") == "set":
                        if "async_results" in str(tool_input.get("filename", "")):
                            final_result = f"Task completed and result saved to {tool_input.get('filename')}"
                
                except Exception as e:
                    # Handle tool execution errors
                    self.logger.error(f"Background tool execution error: {tool_name}: {e}")
                    tool_results[tool_id] = {
                        "content": f"Error: {e}",
                        "is_error": True
                    }
            
            # Prepare tool result blocks for next message
            tool_result_blocks = []
            for tool_id, result in tool_results.items():
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result["content"],
                    "is_error": result.get("is_error", False)
                })
            
            # Ensure the assistant response is a string if it's not already
            if not isinstance(response, str):
                if hasattr(response, 'content'):
                    assistant_content = response.content
                else:
                    # Extract text content as fallback
                    assistant_content = self.llm_bridge.extract_text_content(response)
            else:
                assistant_content = response
                
            # Format messages properly for the next response
            messages = [
                {"role": "user", "content": task.description},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": tool_result_blocks}
            ]
            
            # Generate next response using the same system prompt from above
            # Get the existing system prompt with task ID replaced
            task_system_prompt = """You are a background task executor that uses tools to complete tasks. You are NOT having a conversation - your only job is to execute tools correctly.

PERSISTENCE TOOL USAGE:
- CORRECT operations: "get", "set", "delete", "list" (NEVER use "read", "write", or other invalid operations)
- CORRECT format: persistence(filename="async_results/{task_id}.json", operation="set", key="result", value=data)
- ALWAYS save your final results with: persistence(filename="async_results/{task_id}.json", operation="set", key="result", value=data)

TASK EXECUTION STEPS:
1. Process the assigned task
2. Execute necessary tools in sequence
3. Save the FINAL result using ONLY the persistence tool with operation="set"
4. Return a SHORT completion confirmation message

CRITICAL RULES:
- Use EXACT operation names: "get"/"set"/"delete"/"list" for persistence tool
- When extracting information, specify the correct template
- ALL results MUST be saved to "async_results/{task_id}.json"
- Use simple text for your final response, NOT JSON or other structured formats
- DO NOT use markdown formatting in your responses

EXAMPLE CORRECT USAGE:
1. extract(message="user message", template="food_preferences")  
2. persistence(filename="preferences.json", operation="get", key="preferences")
3. persistence(filename="async_results/{task_id}.json", operation="set", key="result", value=analyzed_data)
4. Return: "Task completed and results saved."

You are working autonomously without user interaction.""".replace("{task_id}", task.task_id)
            
            response = self.llm_bridge.generate_response(
                messages=messages,
                system_prompt=task_system_prompt,
                tools=self.tool_repo.get_all_tool_definitions()
            )
            
            # If we provided final_result from a persistence operation, break the loop
            if final_result:
                break
        
        # If we reached max iterations without getting a final result
        if iterations >= max_iterations and not final_result:
            final_result = f"Task processing reached maximum iterations ({max_iterations})"
        
        return final_result
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed/failed tasks.
        
        Args:
            max_age_hours: Maximum age in hours for tasks to keep
            
        Returns:
            Number of tasks cleaned up
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                if task.completed_at and task.completed_at < cutoff_time:
                    tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        
        self.logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
        return len(tasks_to_remove)
    
    def shutdown(self):
        """Shut down the task manager gracefully."""
        # Signal worker thread to terminate
        self.task_queue.put((None, None))
        
        # Wait for worker thread to finish
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        self.logger.info("AsyncTaskManager shut down")