"""
Task scheduler module for managing and executing scheduled tasks.

This module provides the TaskScheduler class which is responsible for
managing scheduled tasks, calculating next run times, and executing tasks
based on their configurations. It supports both direct tool execution and
LLM-orchestrated tasks.
"""

import datetime
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union, Set, cast
from pytz import timezone as pytz_timezone, utc
from sqlalchemy import and_, or_, func, desc

from api.llm_bridge import LLMBridge
from config import config
from db import Database
from errors import ToolError, ErrorCode, error_context
from task_manager.scheduled_task import (
    ScheduledTask, TaskFrequency, TaskStatus, ExecutionMode, SchedulerConfig
)
from task_manager.task_notification import NotificationManager
from tools.repo import ToolRepository

# Configure logger
logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Task scheduler for managing and executing scheduled tasks.
    
    This class handles scheduling, executing, and managing recurring tasks with
    support for both direct tool execution and LLM-orchestrated execution.
    """
    
    def __init__(
        self,
        tool_repo: Optional[ToolRepository] = None,
        llm_bridge: Optional[LLMBridge] = None
    ):
        """
        Initialize the task scheduler.
        
        Args:
            tool_repo: Repository of available tools
            llm_bridge: LLM bridge for orchestrated tasks
        """
        self.db = Database()
        self.tool_repo = tool_repo or ToolRepository()
        self.llm_bridge = llm_bridge or LLMBridge()
        self.notification_manager = NotificationManager()
        
        # Load configuration
        self.config = config.get("scheduler", SchedulerConfig())
        self.system_timezone = pytz_timezone(config.system.timezone)
        
        # Initialize execution state
        self.running = False
        self.executor = None
        self.active_tasks: Set[str] = set()  # Track task IDs currently being executed
        
        # Initialize locks
        self.task_lock = threading.Lock()
        
        logger.info("Task scheduler initialized")
    
    def start(self) -> None:
        """
        Start the task scheduler.
        
        This method begins the background thread that periodically checks for
        and executes scheduled tasks.
        """
        if self.running:
            logger.warning("Task scheduler is already running")
            return
            
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)
        
        # Start the scheduler thread
        threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="TaskSchedulerThread"
        ).start()
        
        logger.info(f"Task scheduler started with check interval of {self.config.check_interval} seconds")
    
    def stop(self) -> None:
        """
        Stop the task scheduler.
        
        This method stops the scheduler thread and waits for any executing
        tasks to complete.
        """
        if not self.running:
            logger.warning("Task scheduler is not running")
            return
            
        logger.info("Stopping task scheduler...")
        self.running = False
        
        # Shutdown the executor and wait for tasks to complete
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
            
        logger.info("Task scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """
        Main scheduler loop that periodically checks for and executes due tasks.
        """
        while self.running:
            try:
                # Find and execute due tasks
                self._process_due_tasks()
                
                # Sleep until next check
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                # Continue running even if an error occurs
                time.sleep(60)  # Wait a bit longer after an error
    
    def _process_due_tasks(self) -> None:
        """
        Find and execute tasks that are due to run.
        """
        now = datetime.now(timezone.utc)
        
        with self.db.get_session() as session:
            # Find scheduled tasks that are due to run
            due_tasks = session.query(ScheduledTask).filter(
                and_(
                    ScheduledTask.status == TaskStatus.SCHEDULED,
                    ScheduledTask.next_run_time <= now,
                    or_(
                        ScheduledTask.max_runs.is_(None),
                        ScheduledTask.run_count < ScheduledTask.max_runs
                    )
                )
            ).all()
            
            if due_tasks:
                logger.info(f"Found {len(due_tasks)} tasks due for execution")
                
                for task in due_tasks:
                    # Skip tasks that are already running
                    if task.id in self.active_tasks:
                        logger.debug(f"Task {task.id} is already running, skipping")
                        continue
                    
                    # Check if we have capacity to run the task
                    with self.task_lock:
                        if len(self.active_tasks) >= self.config.max_concurrent_tasks:
                            logger.warning("Maximum concurrent tasks reached, deferring execution")
                            break
                        
                        # Mark task as running and submit for execution
                        task.status = TaskStatus.RUNNING
                        task.last_run_time = now
                        self.active_tasks.add(task.id)
                        
                    # Submit task for execution in the thread pool
                    if self.executor:
                        self.executor.submit(self._execute_task, task.id)
                    
                    # Calculate and set the next run time immediately
                    # This prevents schedule drift by basing next run on the schedule, not completion time
                    next_run = self._calculate_next_run_time(task)
                    if next_run:
                        task.next_run_time = next_run
                        logger.debug(f"Next run for task {task.id} scheduled for {next_run.isoformat()}")
                    
                    session.commit()
    
    def _calculate_next_run_time(self, task: ScheduledTask) -> Optional[datetime]:
        """
        Calculate the next run time for a recurring task based on its schedule,
        not its completion time. This prevents schedule drift.
        
        Args:
            task: The task to calculate next run time for
            
        Returns:
            The next run time or None if no more runs are scheduled
        """
        now = datetime.now(timezone.utc)
        
        # Check if the task has an end time and we've passed it
        if task.end_time and task.end_time <= now:
            return None
            
        # Check if we've reached the maximum number of runs
        if task.max_runs and task.run_count >= task.max_runs:
            return None
            
        # For one-time tasks, there's no next run
        if task.frequency == TaskFrequency.ONCE:
            return None
            
        # Get the base scheduled time (e.g., 9:00 AM)
        base_time = task.scheduled_time
        if not base_time:
            return None
            
        # Convert to the task's timezone if specified
        task_timezone = task.timezone or config.system.timezone
        tz = pytz_timezone(task_timezone)
        
        # Get the time components (hour, minute, second)
        scheduled_hour = base_time.hour
        scheduled_minute = base_time.minute
        scheduled_second = base_time.second
        
        # Start with the current time in the task's timezone
        current_time = now.astimezone(tz)
        
        # Calculate next run based on frequency
        if task.frequency == TaskFrequency.MINUTELY:
            # Next minute, same second
            next_run = current_time.replace(second=scheduled_second) + timedelta(minutes=1)
            
        elif task.frequency == TaskFrequency.HOURLY:
            # Next hour, same minute and second
            next_run = current_time.replace(minute=scheduled_minute, second=scheduled_second)
            if next_run <= current_time:
                next_run += timedelta(hours=1)
                
        elif task.frequency == TaskFrequency.DAILY:
            # Next day, same hour, minute, and second
            next_run = current_time.replace(
                hour=scheduled_hour, 
                minute=scheduled_minute, 
                second=scheduled_second
            )
            if next_run <= current_time:
                next_run += timedelta(days=1)
                
        elif task.frequency == TaskFrequency.WEEKLY:
            # Next occurrence of the specified day of week
            if task.day_of_week is None:
                # Default to same day of week as scheduled_time
                day_of_week = base_time.weekday()
            else:
                day_of_week = task.day_of_week
                
            # Start with today at the scheduled time
            next_run = current_time.replace(
                hour=scheduled_hour, 
                minute=scheduled_minute, 
                second=scheduled_second
            )
            
            # Add days until we reach the correct day of week
            days_ahead = day_of_week - current_time.weekday()
            if days_ahead < 0 or (days_ahead == 0 and next_run <= current_time):
                days_ahead += 7
                
            next_run += timedelta(days=days_ahead)
                
        elif task.frequency == TaskFrequency.MONTHLY:
            # Next occurrence of the specified day of month
            if task.day_of_month is None:
                # Default to same day of month as scheduled_time
                day_of_month = base_time.day
            else:
                day_of_month = task.day_of_month
                
            # Start with current month at the scheduled time and day
            try:
                next_run = current_time.replace(
                    day=day_of_month,
                    hour=scheduled_hour, 
                    minute=scheduled_minute, 
                    second=scheduled_second
                )
            except ValueError:
                # Handle invalid day for month (e.g., Feb 30)
                # Use the last day of the month instead
                next_month = current_time.replace(day=1) + relativedelta(months=1)
                last_day = (next_month - timedelta(days=1)).day
                next_run = current_time.replace(
                    day=min(day_of_month, last_day),
                    hour=scheduled_hour, 
                    minute=scheduled_minute, 
                    second=scheduled_second
                )
                
            # If this time has already passed, move to next month
            if next_run <= current_time:
                next_month = next_run + relativedelta(months=1)
                try:
                    next_run = next_month.replace(day=day_of_month)
                except ValueError:
                    # Handle invalid day for month (e.g., Feb 30)
                    next_month_plus_1 = next_month.replace(day=1) + relativedelta(months=1)
                    last_day = (next_month_plus_1 - timedelta(days=1)).day
                    next_run = next_month.replace(day=min(day_of_month, last_day))
                
        elif task.frequency == TaskFrequency.CUSTOM and task.custom_schedule:
            # Custom schedules would need a more complex parser
            # For now, assume a simple format like "every 3 days" or similar
            try:
                parts = task.custom_schedule.lower().split()
                if len(parts) >= 3 and parts[0] == "every":
                    amount = int(parts[1])
                    unit = parts[2]
                    
                    # Start with the current time
                    next_run = current_time.replace(
                        hour=scheduled_hour, 
                        minute=scheduled_minute, 
                        second=scheduled_second
                    )
                    
                    # If this time has already passed, use last_run_time as base
                    if next_run <= current_time and task.last_run_time:
                        base_time = task.last_run_time.astimezone(tz)
                        next_run = base_time
                    
                    # Add the appropriate interval
                    if unit.startswith("minute"):
                        next_run += timedelta(minutes=amount)
                    elif unit.startswith("hour"):
                        next_run += timedelta(hours=amount)
                    elif unit.startswith("day"):
                        next_run += timedelta(days=amount)
                    elif unit.startswith("week"):
                        next_run += timedelta(weeks=amount)
                    elif unit.startswith("month"):
                        next_run += relativedelta(months=amount)
                    else:
                        # Unknown unit, default to daily
                        next_run += timedelta(days=1)
                        
                else:
                    # Unsupported format, default to daily
                    next_run = current_time.replace(
                        hour=scheduled_hour, 
                        minute=scheduled_minute, 
                        second=scheduled_second
                    )
                    if next_run <= current_time:
                        next_run += timedelta(days=1)
                        
            except Exception as e:
                logger.error(f"Error parsing custom schedule '{task.custom_schedule}': {e}")
                # Default to daily if we can't parse
                next_run = current_time.replace(
                    hour=scheduled_hour, 
                    minute=scheduled_minute, 
                    second=scheduled_second
                )
                if next_run <= current_time:
                    next_run += timedelta(days=1)
        else:
            # Unknown frequency, no next run
            return None
            
        # Convert back to UTC for storage
        return next_run.astimezone(utc)
    
    def _execute_task(self, task_id: str) -> None:
        """
        Execute a task by its ID.
        
        Args:
            task_id: ID of the task to execute
        """
        task = self.db.get(ScheduledTask, task_id)
        
        if not task:
            logger.error(f"Task with ID {task_id} not found")
            return
            
        logger.info(f"Executing task {task_id}: '{task.name}'")
        
        start_time = datetime.now(timezone.utc)
        result = None
        error = None
        
        try:
            # Execute based on the execution mode
            if task.execution_mode == ExecutionMode.DIRECT:
                result = self._execute_direct_task(task)
            else:
                result = self._execute_orchestrated_task(task)
                
            # Update task status and result
            task.status = TaskStatus.COMPLETED
            task.last_result = result
            task.run_count += 1
            
            # Create success notification
            self.notification_manager.create_task_result_notification(
                task_id=task.id,
                task_name=task.name,
                success=True,
                result=result
            )
            
            logger.info(f"Task {task_id} executed successfully")
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error executing task {task_id}: {error_message}", exc_info=True)
            
            # Update task status and error
            task.status = TaskStatus.FAILED
            task.last_error = error_message
            task.retry_count += 1
            
            # If retries are available, reschedule
            if task.retry_count <= task.max_retries:
                task.status = TaskStatus.SCHEDULED
                task.next_run_time = datetime.now(timezone.utc) + timedelta(minutes=5)
                logger.info(f"Task {task_id} will be retried in 5 minutes (attempt {task.retry_count}/{task.max_retries})")
            else:
                logger.info(f"Task {task_id} failed after {task.retry_count-1} retries")
            
            # Create failure notification
            self.notification_manager.create_task_result_notification(
                task_id=task.id,
                task_name=task.name,
                success=False,
                error=error_message
            )
            
            error = {"error": error_message}
            
        finally:
            # Calculate duration
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            # Update the task in the database
            self.db.update(task)
            
            # Remove from active tasks
            with self.task_lock:
                self.active_tasks.discard(task_id)
    
    def _execute_direct_task(self, task: ScheduledTask) -> Dict[str, Any]:
        """
        Execute a task in direct execution mode.
        
        Args:
            task: Task to execute
            
        Returns:
            Result of the task execution
            
        Raises:
            ToolError: If the task execution fails
        """
        # Get the tool
        tool = self.tool_repo.get_tool(task.tool_name)
        if not tool:
            raise ToolError(
                f"Tool '{task.tool_name}' not found or not enabled",
                ErrorCode.TOOL_NOT_FOUND
            )
        
        # Execute the tool with the specified operation and parameters
        logger.debug(f"Executing tool {task.tool_name}.{task.operation} with parameters: {task.parameters}")
        result = tool.run(task.operation, **task.parameters)
        
        return result
    
    def _execute_orchestrated_task(self, task: ScheduledTask) -> Dict[str, Any]:
        """
        Execute a task in LLM-orchestrated execution mode.
        
        Args:
            task: Task to execute
            
        Returns:
            Result of the task execution
            
        Raises:
            ToolError: If the task execution fails
        """
        logger.debug(f"Executing orchestrated task: {task.task_description}")
        
        # Create a mini conversation context
        system_prompt = task.task_prompt or self.config.default_system_prompt
        
        # Add task context
        task_context = {
            "task_name": task.name,
            "task_description": task.task_description,
            "schedule": {
                "frequency": task.frequency.value,
                "run_count": task.run_count
            }
        }
        
        # Enhance system prompt with task context
        enhanced_prompt = f"""
{system_prompt}

You are executing a scheduled task with the following details:
Task: {task.name}
Description: {task.task_description}
Schedule: {task.frequency.value}
Run count: {task.run_count}

Your objective is to accomplish this task using the tools available to you.
When you're done, provide a concise summary of what you did and any results or
findings. Be thorough but efficient.
"""
        
        # Enable the tools for this task
        enabled_tools = []
        if task.available_tools:
            # Only enable specified tools
            for tool_name in task.available_tools:
                if self.tool_repo.enable_tool(tool_name):
                    enabled_tools.append(tool_name)
        else:
            # If no specific tools are provided, use all available tools
            enabled_tools = self.tool_repo.get_enabled_tools()
        
        # Create minimal conversation for the task
        from conversation import Conversation
        
        # Create a temporary conversation for this task
        conversation = Conversation(
            conversation_id=f"task_{task.id}_{int(time.time())}",
            system_prompt=enhanced_prompt,
            llm_bridge=self.llm_bridge,
            tool_repo=self.tool_repo
        )
        
        # Execute the task
        response = conversation.generate_response(task.task_description)
        
        # Extract results
        result = {
            "llm_response": response,
            "enabled_tools": enabled_tools,
            "tools_used": conversation.get_tool_usage_summary() if hasattr(conversation, 'get_tool_usage_summary') else []
        }
        
        return result
    
    # Task management methods
    def create_task(self, task_data: Dict[str, Any]) -> ScheduledTask:
        """
        Create a new scheduled task.
        
        Args:
            task_data: Task definition data
            
        Returns:
            The created task
            
        Raises:
            ToolError: If the task is invalid
        """
        with error_context(
            component_name="TaskScheduler",
            operation="create_task",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_INVALID_INPUT,
            logger=logger
        ):
            # Create the task object
            task = ScheduledTask.from_dict(task_data)
            
            # Set timezone if not provided
            if not task.timezone:
                task.timezone = config.system.timezone
            
            # Calculate next run time if not provided
            if not task.next_run_time:
                # For immediate first run, use scheduled_time
                task.next_run_time = task.scheduled_time
            
            # Validate the task
            errors = task.validate()
            if errors:
                raise ToolError(
                    f"Invalid task definition: {', '.join(errors)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Save to database
            self.db.add(task)
            logger.info(f"Created new task: {task.id} ('{task.name}')")
            
            return task
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            The task or None if not found
        """
        return self.db.get(ScheduledTask, task_id)
    
    def update_task(self, task_id: str, task_data: Dict[str, Any]) -> Optional[ScheduledTask]:
        """
        Update an existing task.
        
        Args:
            task_id: ID of the task to update
            task_data: New task data
            
        Returns:
            The updated task or None if not found
            
        Raises:
            ToolError: If the task update is invalid
        """
        with error_context(
            component_name="TaskScheduler",
            operation="update_task",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=logger
        ):
            # Get the existing task
            task = self.db.get(ScheduledTask, task_id)
            if not task:
                return None
            
            # Create a new task with the updated data
            updated_data = task.to_dict()
            updated_data.update(task_data)
            
            updated_task = ScheduledTask.from_dict(updated_data)
            
            # Validate the updated task
            errors = updated_task.validate()
            if errors:
                raise ToolError(
                    f"Invalid task update: {', '.join(errors)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Save to database
            self.db.update(updated_task)
            logger.info(f"Updated task: {task_id}")
            
            return updated_task
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task.
        
        Args:
            task_id: ID of the task to delete
            
        Returns:
            True if the task was deleted, False if not found
        """
        task = self.db.get(ScheduledTask, task_id)
        if not task:
            return False
            
        self.db.delete(task)
        logger.info(f"Deleted task: {task_id}")
        
        return True
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if the task was cancelled, False if not found
        """
        task = self.db.get(ScheduledTask, task_id)
        if not task:
            return False
            
        task.status = TaskStatus.CANCELLED
        self.db.update(task)
        logger.info(f"Cancelled task: {task_id}")
        
        return True
    
    def get_tasks(
        self,
        status: Optional[str] = None,
        frequency: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ScheduledTask]:
        """
        Get tasks filtered by criteria.
        
        Args:
            status: Filter by task status
            frequency: Filter by task frequency
            limit: Maximum number of tasks to return
            offset: Offset for pagination
            
        Returns:
            List of tasks matching the criteria
        """
        with self.db.get_session() as session:
            query = session.query(ScheduledTask)
            
            # Apply filters
            if status:
                query = query.filter(ScheduledTask.status == TaskStatus(status))
                
            if frequency:
                query = query.filter(ScheduledTask.frequency == TaskFrequency(frequency))
                
            # Apply sorting, limit and offset
            query = query.order_by(desc(ScheduledTask.scheduled_time))
            query = query.limit(limit).offset(offset)
            
            # Execute query
            return query.all()
    
    def execute_task_now(self, task_id: str) -> bool:
        """
        Execute a task immediately.
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            True if the task was submitted for execution, False if not found
            
        Raises:
            ToolError: If the task cannot be executed
        """
        with error_context(
            component_name="TaskScheduler",
            operation="execute_task_now",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=logger
        ):
            task = self.db.get(ScheduledTask, task_id)
            if not task:
                return False
                
            # Check if the task is already running
            if task.id in self.active_tasks:
                raise ToolError(
                    f"Task {task_id} is already running",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
                
            # Update task status and submit for execution
            task.status = TaskStatus.RUNNING
            task.last_run_time = datetime.now(timezone.utc)
            self.db.update(task)
            
            # Add to active tasks
            with self.task_lock:
                self.active_tasks.add(task.id)
                
            # Submit for execution
            if self.executor:
                self.executor.submit(self._execute_task, task.id)
                logger.info(f"Task {task_id} submitted for immediate execution")
                return True
            else:
                # No executor available, execute in a separate thread
                threading.Thread(
                    target=self._execute_task,
                    args=(task.id,),
                    daemon=True
                ).start()
                logger.info(f"Task {task_id} started in separate thread")
                return True