"""
Asynchronous task client for the main conversation process.

This module provides a client interface for scheduling and
checking background tasks handled by the background service.
"""
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from errors import ToolError, ErrorCode, error_context
from config import config


class AsyncClient:
    """
    Client for the background task service.
    
    Provides an interface for the main conversation process to
    schedule tasks, check their status, and receive notifications.
    """
    
    def __init__(self, conversation_id: str):
        """
        Initialize the async client.
        
        Args:
            conversation_id: ID of the conversation this client belongs to
        """
        self.conversation_id = conversation_id
        self.logger = logging.getLogger("async_client")
        
        # Initialize directories (same structure as background service)
        self.tasks_dir = Path(config.paths.persistent_dir) / "tasks"
        self.pending_dir = self.tasks_dir / "pending"
        self.running_dir = self.tasks_dir / "running" 
        self.completed_dir = self.tasks_dir / "completed"
        self.failed_dir = self.tasks_dir / "failed"
        self.notifications_dir = Path(config.paths.persistent_dir) / "notifications" / conversation_id
        
        # Ensure directories exist
        for directory in [self.pending_dir, self.running_dir, self.completed_dir, 
                         self.failed_dir, self.notifications_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"AsyncClient initialized for conversation: {conversation_id}")
    
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
            task_prompt: Prompt for the background LLM
            notify_on_completion: Whether to notify when complete
            
        Returns:
            Task ID for tracking
        """
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create task data
        task_data = {
            "task_id": task_id,
            "description": description, 
            "task_prompt": task_prompt,
            "conversation_id": self.conversation_id,
            "notify_on_completion": notify_on_completion,
            "status": "pending",
            "created_at": time.time()
        }
        
        # Write to pending directory
        task_path = self.pending_dir / f"{task_id}.json"
        with open(task_path, "w") as f:
            json.dump(task_data, f)
        
        self.logger.info(f"Scheduled task {task_id}: {description}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            Dictionary with task status information
            
        Raises:
            ToolError: If the task is not found
        """
        # Check all possible locations in order
        for status, directory in [
            ("completed", self.completed_dir),
            ("failed", self.failed_dir),
            ("running", self.running_dir),
            ("pending", self.pending_dir)
        ]:
            task_path = directory / f"{task_id}.json"
            if task_path.exists():
                with open(task_path, "r") as f:
                    return json.load(f)
        
        # Not found
        error_msg = f"Task not found: {task_id}"
        self.logger.warning(error_msg)
        raise ToolError(error_msg, ErrorCode.TOOL_NOT_FOUND)
    
    def get_notifications(self, clear: bool = True) -> List[Dict[str, Any]]:
        """
        Get and optionally clear pending notifications.
        
        Args:
            clear: Whether to remove notifications after retrieving
            
        Returns:
            List of notification dictionaries
        """
        notifications = []
        
        # Check if directory exists
        if not self.notifications_dir.exists():
            return notifications
        
        # Get notification files
        for notification_file in self.notifications_dir.glob("*.json"):
            try:
                with open(notification_file, "r") as f:
                    notification = json.load(f)
                
                notifications.append(notification)
                
                # Clear if requested
                if clear:
                    notification_file.unlink()
            except Exception as e:
                self.logger.error(f"Error reading notification {notification_file}: {e}")
        
        if notifications:
            self.logger.info(f"Retrieved {len(notifications)} notifications")
        
        return notifications