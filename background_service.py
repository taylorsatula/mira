#!/usr/bin/env python3
"""
Background service for asynchronous task execution.

This standalone service monitors for task requests and executes them
independently from the main conversation process.
"""
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository
from config import config
from errors import ToolError, ErrorCode, error_context


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("background_service.log")
    ]
)


class BackgroundService:
    """
    Service for executing tasks in the background.
    
    This service monitors a directory for task requests, executes them
    using LLM and tools, and saves the results for the main process.
    """
    
    def __init__(self):
        """Initialize the background service."""
        # Initialize directories
        self.tasks_dir = Path(config.paths.persistent_dir) / "tasks"
        self.pending_dir = self.tasks_dir / "pending"
        self.running_dir = self.tasks_dir / "running"
        self.completed_dir = self.tasks_dir / "completed"
        self.failed_dir = self.tasks_dir / "failed"
        self.notifications_dir = Path(config.paths.persistent_dir) / "notifications"
        
        # Create directories
        for directory in [self.pending_dir, self.running_dir, self.completed_dir, 
                         self.failed_dir, self.notifications_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components (dedicated instances)
        self.llm_bridge = LLMBridge()
        self.tool_repo = ToolRepository()
        
        # Discover and enable ALL tools for background operations
        self.tool_repo.discover_tools()
        self.tool_repo.enable_all_tools()
        
        self.running = True
        self.logger = logging.getLogger("background_service")
        self.logger.info("Background service initialized")
    
    def start(self):
        """Start the background service processing loop."""
        self.logger.info("Background service started")
        print(f"Monitoring for tasks in: {self.pending_dir}")
        print("Press Ctrl+C to stop the service")
        
        try:
            while self.running:
                # Process any pending tasks
                self._process_pending_tasks()
                # Sleep briefly to avoid high CPU usage
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Background service stopping due to keyboard interrupt")
            self.running = False
        except Exception as e:
            self.logger.error(f"Background service error: {e}")
            self.running = False
    
    def _process_pending_tasks(self):
        """Check for and process any pending tasks."""
        try:
            pending_files = list(self.pending_dir.glob("*.json"))
            if pending_files:
                self.logger.info(f"Found {len(pending_files)} pending tasks")
            
            for task_file in pending_files:
                try:
                    # Load task data
                    with open(task_file, "r") as f:
                        task_data = json.load(f)
                    
                    # Move to running state
                    task_id = task_data["task_id"]
                    task_data["status"] = "running"
                    task_data["started_at"] = time.time()
                    
                    with open(self.running_dir / f"{task_id}.json", "w") as f:
                        json.dump(task_data, f)
                    
                    # Remove from pending
                    task_file.unlink()
                    
                    self.logger.info(f"Starting task: {task_id} - {task_data.get('description', 'No description')}")
                    
                    # Execute in a separate thread
                    thread = threading.Thread(
                        target=self._execute_task,
                        args=(task_data,),
                        daemon=True
                    )
                    thread.start()
                    
                except Exception as e:
                    self.logger.error(f"Error processing task {task_file}: {e}")
        except Exception as e:
            self.logger.error(f"Error in task processing loop: {e}")
    
    def _execute_task(self, task_data):
        """
        Execute a task with full LLM and tool access.
        
        Args:
            task_data: Dictionary with task information
        """
        task_id = task_data["task_id"]
        task_prompt = task_data["task_prompt"]
        
        try:
            # Add detailed logging
            self.logger.info(f"Executing task {task_id}: {task_data.get('description', 'No description')}")
            self.logger.debug(f"Task prompt: {task_prompt}")
            
            # Get specialized system prompt for background tasks
            system_prompt = config.get_system_prompt(
                "background_system_prompt", 
                replacements={"{task_id}": task_id}
            )
            
            # Create message for the LLM
            messages = [{"role": "user", "content": task_prompt}]
            
            # Get ALL tools (background has full access)
            tools = self.tool_repo.get_all_tool_definitions()
            self.logger.info(f"Using {len(tools)} tools for task execution")
            
            # Generate and process the response
            response = self.llm_bridge.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools
            )
            
            # Process the response with all tool calls
            result = self._process_llm_response(task_id, task_prompt, response)
            
            # Mark as completed
            task_data["status"] = "completed"
            task_data["result"] = result
            task_data["completed_at"] = time.time()
            
            with open(self.completed_dir / f"{task_id}.json", "w") as f:
                json.dump(task_data, f)
            
            # Remove from running
            running_file = self.running_dir / f"{task_id}.json"
            if running_file.exists():
                running_file.unlink()
            
            self.logger.info(f"Task {task_id} completed successfully")
            
            # Send notification if requested
            if task_data.get("notify_on_completion", False):
                self._send_notification(task_data)
                
        except Exception as e:
            self.logger.error(f"Error executing task {task_id}: {e}")
            
            # Mark as failed
            task_data["status"] = "failed"
            task_data["error"] = str(e)
            task_data["completed_at"] = time.time()
            
            with open(self.failed_dir / f"{task_id}.json", "w") as f:
                json.dump(task_data, f)
            
            # Remove from running
            running_file = self.running_dir / f"{task_id}.json"
            if running_file.exists():
                running_file.unlink()
            
            # Send notification about failure
            if task_data.get("notify_on_completion", False):
                self._send_notification(task_data)
    
    def _process_llm_response(self, task_id: str, task_prompt: str, response) -> Any:
        """
        Process LLM response with tool calls.
        
        Args:
            task_id: The ID of the task
            task_prompt: The original task prompt
            response: The LLM response
            
        Returns:
            The final task result
        """
        # Maximum iterations to prevent infinite loops
        max_iterations = config.tools.max_background_iterations
        iterations = 0
        final_result = None
        
        while iterations < max_iterations:
            iterations += 1
            
            # Extract tool calls from response
            tool_calls = self.llm_bridge.extract_tool_calls(response)
            
            # Extract text content for logging
            text_content = self.llm_bridge.extract_text_content(response)
            self.logger.debug(f"LLM response (iteration {iterations}): {text_content[:500]}...")
            
            # If no tool calls, use text response as result
            if not tool_calls:
                final_result = text_content
                break
            
            # Process tool calls
            tool_results = {}
            for tool_call in tool_calls:
                tool_name = tool_call["tool_name"]
                tool_input = tool_call["input"]
                tool_id = tool_call["id"]
                
                self.logger.info(f"Executing tool: {tool_name}")
                self.logger.debug(f"Tool input: {tool_input}")
                
                try:
                    # Invoke the tool
                    result = self.tool_repo.invoke_tool(tool_name, tool_input)
                    result_str = str(result)
                    tool_results[tool_id] = {
                        "content": result_str,
                        "is_error": False
                    }
                    
                    self.logger.debug(f"Tool result: {result_str[:500]}...")
                    
                    # Check if this was a persistence operation for results
                    if tool_name == "persistence":
                        operation = tool_input.get("operation", "")
                        
                        # Check for async_results operations
                        if operation in ["set_data", "set_file"] and "location" in tool_input:
                            location = str(tool_input.get("location", ""))
                            if "async_results" in location:
                                # Store full path for reporting
                                full_path = f"{config.paths.persistent_dir}/{location}"
                                if not full_path.endswith('.json'):
                                    full_path += '.json'
                                final_result = f"Task completed and result saved to {full_path}"
                    
                except Exception as e:
                    error_msg = f"Tool execution error: {tool_name}: {e}"
                    self.logger.error(error_msg)
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
            
            # Get assistant content for next message
            if hasattr(response, 'content'):
                assistant_content = response.content
            else:
                assistant_content = self.llm_bridge.extract_text_content(response)
            
            # Format messages for the next response
            messages = [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": tool_result_blocks}
            ]
            
            # Generate next response
            response = self.llm_bridge.generate_response(
                messages=messages,
                system_prompt=config.get_system_prompt("background_system_prompt", 
                                                     replacements={"{task_id}": task_id}),
                tools=self.tool_repo.get_all_tool_definitions()
            )
            
            # If we have a final result, break the loop
            if final_result:
                break
        
        # If we reached max iterations without result
        if iterations >= max_iterations and not final_result:
            final_result = f"Task processing reached maximum iterations ({max_iterations})"
            self.logger.warning(f"Task {task_id} reached max iterations without final result")
        
        return final_result
    
    def _send_notification(self, task_data):
        """
        Send notification back to the main process.
        
        Args:
            task_data: Dictionary with task information
        """
        task_id = task_data["task_id"]
        conversation_id = task_data.get("conversation_id", "default")
        
        # Create notification directory if needed
        notification_dir = self.notifications_dir / conversation_id
        notification_dir.mkdir(exist_ok=True)
        
        # Create notification
        notification = {
            "notification_id": str(uuid.uuid4()),
            "task_id": task_id,
            "description": task_data.get("description", ""),
            "status": task_data.get("status", "unknown"),
            "result_summary": str(task_data.get("result", ""))[:200],
            "error": task_data.get("error", None),
            "timestamp": time.time()
        }
        
        # Write notification file
        notification_path = notification_dir / f"{notification['notification_id']}.json"
        with open(notification_path, "w") as f:
            json.dump(notification, f)
        
        self.logger.info(f"Sent notification for task {task_id} to conversation {conversation_id}")


if __name__ == "__main__":
    # Create and start the background service
    service = BackgroundService()
    try:
        service.start()
    except Exception as e:
        logging.error(f"Background service failed: {e}")
    sys.exit(0)