"""
Scheduler tool for managing scheduled tasks.

This tool allows users to create, view, update, and delete scheduled tasks.
It supports both direct tool execution and LLM-orchestrated tasks, with
various scheduling options.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, cast
import pytz

from pydantic import BaseModel, Field

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context
from scheduled_task import ScheduledTask, TaskFrequency, TaskStatus, ExecutionMode
from utils import scheduler_service
from config import config
from config.registry import registry

# Configure logger
logger = logging.getLogger(__name__)


# Define configuration class for SchedulerTool
class SchedulerToolConfig(BaseModel):
    """Configuration for the scheduler_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    max_tasks_per_user: int = Field(default=20, description="Maximum number of tasks a user can create")


# Register with registry
registry.register("scheduler_tool", SchedulerToolConfig)


class SchedulerTool(Tool):
    """
    Tool for managing scheduled tasks.
    
    This tool allows the creation and management of scheduled tasks that can
    execute at specified times without user intervention. It supports both
    direct tool execution and LLM-orchestrated tasks.
    """
    
    name = "scheduler_tool"
    description = """
    Manages scheduled tasks that run automatically at specified times. Use this tool 
    when the user wants to automate tasks or set up recurring activities.
    
    IMPORTANT: This tool requires parameters to be passed as a JSON string in the "kwargs" field.
    The tool supports these operations:
    
    1. create_task: Schedule a new task for future execution.
       - Required for all tasks: name, frequency (once, minutely, hourly, daily, weekly, monthly, custom)
       - Required for direct tasks: tool_name, operation
       - Required for orchestrated tasks: task_description
       - Optional: scheduled_time, day_of_week (for weekly), day_of_month (for monthly), 
         end_time, timezone, max_runs, timeout, max_retries
       - Returns the created task with a unique identifier
    
    2. get_tasks: Retrieve tasks filtered by criteria.
       - Optional: status (scheduled, running, completed, failed, cancelled),
         frequency, limit (default 10), offset (default 0)
       - Returns list of tasks matching the criteria
    
    3. get_task: Get details of a specific task.
       - Required: task_id
       - Returns the task details
    
    4. update_task: Modify an existing task.
       - Required: task_id
       - Optional: Any fields to update (name, scheduled_time, etc.)
       - Returns the updated task
    
    5. delete_task: Remove a task.
       - Required: task_id
       - Returns confirmation of deletion
    
    6. execute_task_now: Run a task immediately.
       - Required: task_id
       - Returns confirmation that the task was submitted for execution
       
    7. cancel_task: Cancel a scheduled task.
       - Required: task_id
       - Returns confirmation of cancellation
    
    This tool works with two types of tasks:
    
    1. Direct tasks: Execute a specific tool operation with predetermined parameters.
    2. Orchestrated tasks: Use the LLM to interpret a natural language task description
       and determine which tools to use.
    
    Examples:
    - "Schedule a daily reminder to check inventory at 9am"
    - "Create a task to send a weekly sales report every Monday"
    - "Set up a monthly database backup"
    """
    
    usage_examples = [
        {
            "input": {
                "operation": "create_task",
                "kwargs": json.dumps({
                    "name": "Daily Inventory Check",
                    "frequency": "daily",
                    "scheduled_time": "09:00:00",
                    "execution_mode": "direct",
                    "tool_name": "reminder_tool",
                    "operation": "add_reminder",
                    "parameters": {
                        "title": "Check inventory levels", 
                        "date": "today",
                        "description": "Review stock levels for popular items"
                    }
                })
            },
            "output": {
                "task": {
                    "id": "task_12345",
                    "name": "Daily Inventory Check",
                    "frequency": "daily",
                    "scheduled_time": "2024-04-20T09:00:00+00:00",
                    "next_run_time": "2024-04-20T09:00:00+00:00",
                    "status": "scheduled"
                },
                "message": "Task 'Daily Inventory Check' scheduled to run daily at 09:00:00"
            }
        },
        {
            "input": {
                "operation": "create_task",
                "kwargs": json.dumps({
                    "name": "Weekly Sales Analysis",
                    "frequency": "weekly",
                    "day_of_week": 0,  # Monday
                    "scheduled_time": "08:00:00",
                    "execution_mode": "orchestrated",
                    "task_description": "Generate a sales report for the past week, analyze top-selling products, and email a summary to the sales team."
                })
            },
            "output": {
                "task": {
                    "id": "task_67890",
                    "name": "Weekly Sales Analysis",
                    "frequency": "weekly",
                    "scheduled_time": "2024-04-22T08:00:00+00:00",
                    "day_of_week": 0,
                    "next_run_time": "2024-04-22T08:00:00+00:00",
                    "status": "scheduled"
                },
                "message": "Task 'Weekly Sales Analysis' scheduled to run weekly on Monday at 08:00:00"
            }
        },
        {
            "input": {
                "operation": "get_tasks",
                "kwargs": json.dumps({
                    "status": "scheduled"
                })
            },
            "output": {
                "tasks": [
                    {
                        "id": "task_12345",
                        "name": "Daily Inventory Check",
                        "frequency": "daily",
                        "next_run_time": "2024-04-20T09:00:00+00:00",
                        "status": "scheduled"
                    },
                    {
                        "id": "task_67890",
                        "name": "Weekly Sales Analysis",
                        "frequency": "weekly",
                        "next_run_time": "2024-04-22T08:00:00+00:00",
                        "status": "scheduled"
                    }
                ],
                "count": 2,
                "message": "Found 2 scheduled tasks"
            }
        }
    ]
    
    def __init__(self):
        """Initialize the scheduler tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Get the scheduler instance
        self.scheduler = scheduler_service.get_scheduler()
        
        if not self.scheduler:
            self.logger.warning("Scheduler service is not initialized")
    
    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a scheduler operation.
        
        Args:
            operation: Operation to perform
            **kwargs: Parameters for the operation
            
        Returns:
            Response data for the operation
            
        Raises:
            ToolError: If operation fails or parameters are invalid
        """
        with error_context(
            component_name=self.name,
            operation=f"executing {operation}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Check if scheduler is available
            if not self.scheduler:
                raise ToolError(
                    "Scheduler service is not available",
                    ErrorCode.TOOL_UNAVAILABLE
                )
            
            # Parse kwargs JSON string if provided that way
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    params = json.loads(kwargs["kwargs"])
                    kwargs = params
                except json.JSONDecodeError as e:
                    raise ToolError(
                        f"Invalid JSON in kwargs: {e}",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
            
            # Route to the appropriate operation
            if operation == "create_task":
                return self._create_task(**kwargs)
            elif operation == "get_tasks":
                return self._get_tasks(**kwargs)
            elif operation == "get_task":
                return self._get_task(**kwargs)
            elif operation == "update_task":
                return self._update_task(**kwargs)
            elif operation == "delete_task":
                return self._delete_task(**kwargs)
            elif operation == "execute_task_now":
                return self._execute_task_now(**kwargs)
            elif operation == "cancel_task":
                return self._cancel_task(**kwargs)
            else:
                raise ToolError(
                    f"Unknown operation: {operation}. Valid operations are: "
                    "create_task, get_tasks, get_task, update_task, delete_task, "
                    "execute_task_now, cancel_task",
                    ErrorCode.TOOL_INVALID_INPUT
                )
    
    def _create_task(self, **kwargs) -> Dict[str, Any]:
        """
        Create a new scheduled task.
        
        Args:
            **kwargs: Task parameters
            
        Returns:
            Dict containing the created task
            
        Raises:
            ToolError: If required parameters are missing or invalid
        """
        self.logger.info(f"Creating task: {kwargs.get('name')}")
        
        # Process scheduled_time
        scheduled_time = kwargs.get("scheduled_time")
        if scheduled_time:
            # Handle time-only strings (add today's date)
            if isinstance(scheduled_time, str) and ":" in scheduled_time and "T" not in scheduled_time:
                try:
                    # Split into hours, minutes, seconds
                    time_parts = scheduled_time.split(":")
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
                    seconds = int(time_parts[2]) if len(time_parts) > 2 else 0
                    
                    # Get current date in the system timezone
                    tz = pytz.timezone(kwargs.get("timezone") or config.system.timezone)
                    now = datetime.now(tz)
                    
                    # Create a datetime with today's date and the specified time
                    scheduled_datetime = now.replace(
                        hour=hours,
                        minute=minutes,
                        second=seconds,
                        microsecond=0
                    )
                    
                    # If this time has already passed today, use tomorrow
                    if scheduled_datetime < now:
                        scheduled_datetime += timedelta(days=1)
                    
                    # Convert to UTC for storage
                    kwargs["scheduled_time"] = scheduled_datetime.astimezone(timezone.utc)
                    
                except (ValueError, IndexError) as e:
                    raise ToolError(
                        f"Invalid time format '{scheduled_time}'. Use HH:MM:SS format.",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
        
        # Create the task
        try:
            task = self.scheduler.create_task(kwargs)
            
            # Generate a human-readable message
            frequency = task.frequency.value
            time_str = task.scheduled_time.strftime('%H:%M:%S')
            
            if frequency == "once":
                date_str = task.scheduled_time.strftime('%Y-%m-%d')
                message = f"Task '{task.name}' scheduled to run once on {date_str} at {time_str}"
            elif frequency == "daily":
                message = f"Task '{task.name}' scheduled to run daily at {time_str}"
            elif frequency == "weekly":
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_name = day_names[task.day_of_week] if task.day_of_week is not None else "unknown day"
                message = f"Task '{task.name}' scheduled to run weekly on {day_name} at {time_str}"
            elif frequency == "monthly":
                day = task.day_of_month or task.scheduled_time.day
                message = f"Task '{task.name}' scheduled to run monthly on day {day} at {time_str}"
            else:
                message = f"Task '{task.name}' scheduled with {frequency} frequency"
            
            return {
                "task": task.to_dict(),
                "message": message
            }
            
        except Exception as e:
            raise ToolError(
                f"Failed to create task: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
    
    def _get_tasks(
        self,
        status: Optional[str] = None,
        frequency: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get tasks filtered by criteria.
        
        Args:
            status: Filter by task status
            frequency: Filter by task frequency
            limit: Maximum number of tasks to return
            offset: Offset for pagination
            
        Returns:
            Dict containing list of tasks
            
        Raises:
            ToolError: If parameters are invalid
        """
        self.logger.info(f"Getting tasks with filters: status={status}, frequency={frequency}")
        
        # Validate status
        if status and status not in [s.value for s in TaskStatus]:
            raise ToolError(
                f"Invalid status: {status}. Valid values are: " +
                ", ".join([s.value for s in TaskStatus]),
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Validate frequency
        if frequency and frequency not in [f.value for f in TaskFrequency]:
            raise ToolError(
                f"Invalid frequency: {frequency}. Valid values are: " +
                ", ".join([f.value for f in TaskFrequency]),
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Get tasks
        tasks = self.scheduler.get_tasks(
            status=status,
            frequency=frequency,
            limit=limit,
            offset=offset
        )
        
        # Convert to dictionaries
        task_dicts = [task.to_dict() for task in tasks]
        
        # Create simplified view for list display
        simplified_tasks = []
        for task in task_dicts:
            simplified_tasks.append({
                "id": task["id"],
                "name": task["name"],
                "frequency": task["frequency"],
                "next_run_time": task["next_run_time"],
                "status": task["status"],
                "execution_mode": task["execution_mode"]
            })
        
        status_str = f" with status '{status}'" if status else ""
        frequency_str = f" with {frequency} frequency" if frequency else ""
        
        return {
            "tasks": simplified_tasks,
            "count": len(tasks),
            "message": f"Found {len(tasks)} task(s){status_str}{frequency_str}"
        }
    
    def _get_task(self, task_id: str) -> Dict[str, Any]:
        """
        Get details of a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dict containing task details
            
        Raises:
            ToolError: If task not found
        """
        self.logger.info(f"Getting task: {task_id}")
        
        task = self.scheduler.get_task(task_id)
        if not task:
            raise ToolError(
                f"Task with ID '{task_id}' not found",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
        
        return {
            "task": task.to_dict(),
            "message": f"Retrieved task '{task.name}'"
        }
    
    def _update_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """
        Update an existing task.
        
        Args:
            task_id: ID of the task to update
            **kwargs: Fields to update
            
        Returns:
            Dict containing the updated task
            
        Raises:
            ToolError: If task not found or update invalid
        """
        self.logger.info(f"Updating task: {task_id}")
        
        # Process scheduled_time similar to create_task
        scheduled_time = kwargs.get("scheduled_time")
        if scheduled_time and isinstance(scheduled_time, str) and ":" in scheduled_time and "T" not in scheduled_time:
            try:
                # Get the task to get its current date
                task = self.scheduler.get_task(task_id)
                if not task:
                    raise ToolError(
                        f"Task with ID '{task_id}' not found",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
                
                # Split into hours, minutes, seconds
                time_parts = scheduled_time.split(":")
                hours = int(time_parts[0])
                minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
                seconds = int(time_parts[2]) if len(time_parts) > 2 else 0
                
                # Get tz
                tz = pytz.timezone(kwargs.get("timezone") or task.timezone or config.system.timezone)
                
                # Use the current scheduled date with the new time
                current_date = task.scheduled_time.astimezone(tz).date()
                new_datetime = datetime.combine(
                    current_date,
                    datetime.time(hours, minutes, seconds),
                    tzinfo=tz
                )
                
                # Convert to UTC for storage
                kwargs["scheduled_time"] = new_datetime.astimezone(timezone.utc)
                
            except (ValueError, IndexError) as e:
                raise ToolError(
                    f"Invalid time format '{scheduled_time}'. Use HH:MM:SS format.",
                    ErrorCode.TOOL_INVALID_INPUT
                )
        
        # Update the task
        updated_task = self.scheduler.update_task(task_id, kwargs)
        if not updated_task:
            raise ToolError(
                f"Task with ID '{task_id}' not found",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
        
        return {
            "task": updated_task.to_dict(),
            "message": f"Task '{updated_task.name}' updated successfully"
        }
    
    def _delete_task(self, task_id: str) -> Dict[str, Any]:
        """
        Delete a task.
        
        Args:
            task_id: ID of the task to delete
            
        Returns:
            Dict containing deletion confirmation
            
        Raises:
            ToolError: If task not found
        """
        self.logger.info(f"Deleting task: {task_id}")
        
        # Get the task name first for the message
        task = self.scheduler.get_task(task_id)
        if not task:
            raise ToolError(
                f"Task with ID '{task_id}' not found",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
        
        task_name = task.name
        
        # Delete the task
        success = self.scheduler.delete_task(task_id)
        if not success:
            raise ToolError(
                f"Failed to delete task with ID '{task_id}'",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
        
        return {
            "task_id": task_id,
            "message": f"Task '{task_name}' deleted successfully"
        }
    
    def _execute_task_now(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a task immediately.
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            Dict containing execution confirmation
            
        Raises:
            ToolError: If task not found or execution fails
        """
        self.logger.info(f"Executing task now: {task_id}")
        
        # Get the task name first for the message
        task = self.scheduler.get_task(task_id)
        if not task:
            raise ToolError(
                f"Task with ID '{task_id}' not found",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
        
        task_name = task.name
        
        # Execute the task
        success = self.scheduler.execute_task_now(task_id)
        if not success:
            raise ToolError(
                f"Failed to execute task with ID '{task_id}'",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
        
        return {
            "task_id": task_id,
            "message": f"Task '{task_name}' submitted for immediate execution"
        }
    
    def _cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel a scheduled task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            Dict containing cancellation confirmation
            
        Raises:
            ToolError: If task not found
        """
        self.logger.info(f"Cancelling task: {task_id}")
        
        # Get the task name first for the message
        task = self.scheduler.get_task(task_id)
        if not task:
            raise ToolError(
                f"Task with ID '{task_id}' not found",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
        
        task_name = task.name
        
        # Cancel the task
        success = self.scheduler.cancel_task(task_id)
        if not success:
            raise ToolError(
                f"Failed to cancel task with ID '{task_id}'",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
        
        return {
            "task_id": task_id,
            "message": f"Task '{task_name}' cancelled successfully"
        }