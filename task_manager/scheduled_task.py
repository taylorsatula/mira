"""
Scheduled task module for managing time-based tasks.

This module defines the database model for scheduled tasks and provides
functionality for scheduling, executing, and managing recurring tasks with
support for both direct tool execution and LLM-orchestrated task execution.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List

from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, JSON, Enum as SQLAEnum, Index, ForeignKey
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

from db import Base, Database
from config.registry import registry
from errors import ToolError, ErrorCode, error_context

# Configure logger
logger = logging.getLogger(__name__)


class TaskFrequency(str, Enum):
    """Enumeration of supported task frequencies."""
    ONCE = "once"             # Run once at specified time
    MINUTELY = "minutely"     # Run every minute
    HOURLY = "hourly"         # Run every hour
    DAILY = "daily"           # Run every day
    WEEKLY = "weekly"         # Run every week
    MONTHLY = "monthly"       # Run every month
    CUSTOM = "custom"         # Custom cron-like specification


class TaskStatus(str, Enum):
    """Enumeration of task status values."""
    SCHEDULED = "scheduled"   # Task is scheduled to run
    RUNNING = "running"       # Task is currently running
    COMPLETED = "completed"   # Task has completed successfully
    FAILED = "failed"         # Task execution failed
    CANCELLED = "cancelled"   # Task was cancelled


class ExecutionMode(str, Enum):
    """Enumeration of task execution modes."""
    DIRECT = "direct"         # Direct tool execution with specific parameters
    ORCHESTRATED = "orchestrated"  # LLM orchestrates task execution based on description


class ScheduledTask(Base):
    """
    Scheduled task model for storing task data.
    
    Maps to the 'scheduled_tasks' table with columns for task details
    including timing, tool information, and execution status. Supports both
    direct tool execution and LLM-orchestrated execution.
    """
    __tablename__ = 'scheduled_tasks'
    
    # Primary key
    id = Column(String, primary_key=True)
    
    # Task details
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Execution mode and details
    execution_mode = Column(SQLAEnum(ExecutionMode), default=ExecutionMode.DIRECT)
    
    # For direct execution
    tool_name = Column(String)
    operation = Column(String)
    parameters = Column(MutableDict.as_mutable(JSON), default={})
    
    # For orchestrated execution
    task_description = Column(Text)
    task_prompt = Column(Text)  # Custom system prompt for the LLM
    available_tools = Column(JSON)  # List of tool names the LLM can use
    
    # Schedule information
    frequency = Column(SQLAEnum(TaskFrequency), nullable=False)
    scheduled_time = Column(DateTime, nullable=False)  # Base time for scheduling (e.g., 9:00 AM)
    day_of_week = Column(Integer)  # For weekly tasks (0=Monday, 6=Sunday)
    day_of_month = Column(Integer)  # For monthly tasks
    end_time = Column(DateTime, nullable=True)  # For recurring tasks with end date
    custom_schedule = Column(String)  # For custom schedules
    timezone = Column(String)  # Store the timezone for this task
    
    # Status tracking
    status = Column(SQLAEnum(TaskStatus), default=TaskStatus.SCHEDULED)
    last_run_time = Column(DateTime)
    next_run_time = Column(DateTime)
    run_count = Column(Integer, default=0)
    max_runs = Column(Integer)  # Maximum number of runs (optional)
    
    # Error tracking
    last_error = Column(Text)
    last_result = Column(MutableDict.as_mutable(JSON))
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                        onupdate=lambda: datetime.now(timezone.utc))
    created_by = Column(String)  # User or system identifier
    
    # Notification settings
    notification_id = Column(String, ForeignKey('task_notifications.id', ondelete='SET NULL'), nullable=True)  # ID of the notification to create for results
    
    # Execution settings
    timeout = Column(Integer, default=300)  # Timeout in seconds
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Define indices for frequently queried fields
    __table_args__ = (
        Index('ix_scheduled_tasks_status', 'status'),
        Index('ix_scheduled_tasks_next_run_time', 'next_run_time'),
        Index('ix_scheduled_tasks_created_at', 'created_at'),
        Index('ix_scheduled_tasks_notification_id', 'notification_id'),
    )
    
    # Relationship with notifications
    notification = relationship("TaskNotification", foreign_keys=[notification_id], back_populates="related_task", uselist=False)
    notifications = relationship("TaskNotification", back_populates="task", 
                               primaryjoin="ScheduledTask.id == TaskNotification.task_id",
                               cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict representation of the scheduled task
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "execution_mode": self.execution_mode.value if self.execution_mode else None,
            
            # Direct execution fields
            "tool_name": self.tool_name,
            "operation": self.operation,
            "parameters": self.parameters,
            
            # Orchestrated execution fields
            "task_description": self.task_description,
            "task_prompt": self.task_prompt,
            "available_tools": self.available_tools,
            
            # Schedule fields
            "frequency": self.frequency.value if self.frequency else None,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "day_of_week": self.day_of_week,
            "day_of_month": self.day_of_month,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "custom_schedule": self.custom_schedule,
            "timezone": self.timezone,
            
            # Status fields
            "status": self.status.value if self.status else None,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "next_run_time": self.next_run_time.isoformat() if self.next_run_time else None,
            "run_count": self.run_count,
            "max_runs": self.max_runs,
            "last_error": self.last_error,
            "last_result": self.last_result,
            
            # Metadata
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            
            # Notification settings
            "notification_id": self.notification_id,
            
            # Execution settings
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            
            # Include notification summaries if loaded
            "notifications_count": len(self.notifications) if hasattr(self, 'notifications') and self.notifications is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduledTask':
        """
        Create a ScheduledTask instance from a dictionary.
        
        Args:
            data: Dictionary representation of a scheduled task
            
        Returns:
            ScheduledTask instance
        """
        task = cls(
            id=data.get("id") or str(uuid.uuid4()),
            name=data.get("name"),
            description=data.get("description"),
            tool_name=data.get("tool_name"),
            operation=data.get("operation"),
            parameters=data.get("parameters", {}),
            task_description=data.get("task_description"),
            task_prompt=data.get("task_prompt"),
            available_tools=data.get("available_tools"),
            frequency=TaskFrequency(data.get("frequency")) if data.get("frequency") else TaskFrequency.ONCE,
            day_of_week=data.get("day_of_week"),
            day_of_month=data.get("day_of_month"),
            custom_schedule=data.get("custom_schedule"),
            timezone=data.get("timezone"),
            run_count=data.get("run_count", 0),
            max_runs=data.get("max_runs"),
            last_error=data.get("last_error"),
            last_result=data.get("last_result"),
            created_by=data.get("created_by"),
            notification_id=data.get("notification_id"),
            timeout=data.get("timeout", 300),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )
        
        # Set execution mode
        if "execution_mode" in data and data["execution_mode"]:
            task.execution_mode = ExecutionMode(data["execution_mode"])
        else:
            # Determine execution mode based on available fields
            if data.get("task_description"):
                task.execution_mode = ExecutionMode.ORCHESTRATED
            else:
                task.execution_mode = ExecutionMode.DIRECT
        
        # Handle datetime fields
        if "scheduled_time" in data and data["scheduled_time"]:
            if isinstance(data["scheduled_time"], str):
                task.scheduled_time = datetime.fromisoformat(data["scheduled_time"])
            else:
                task.scheduled_time = data["scheduled_time"]
                
        if "end_time" in data and data["end_time"]:
            if isinstance(data["end_time"], str):
                task.end_time = datetime.fromisoformat(data["end_time"])
            else:
                task.end_time = data["end_time"]
                
        if "last_run_time" in data and data["last_run_time"]:
            if isinstance(data["last_run_time"], str):
                task.last_run_time = datetime.fromisoformat(data["last_run_time"])
            else:
                task.last_run_time = data["last_run_time"]
                
        if "next_run_time" in data and data["next_run_time"]:
            if isinstance(data["next_run_time"], str):
                task.next_run_time = datetime.fromisoformat(data["next_run_time"])
            else:
                task.next_run_time = data["next_run_time"]
                
        # Set status
        if "status" in data and data["status"]:
            task.status = TaskStatus(data["status"])
        else:
            task.status = TaskStatus.SCHEDULED
            
        # Set timestamps if provided
        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                task.created_at = datetime.fromisoformat(data["created_at"])
            else:
                task.created_at = data["created_at"]
                
        if "updated_at" in data and data["updated_at"]:
            if isinstance(data["updated_at"], str):
                task.updated_at = datetime.fromisoformat(data["updated_at"])
            else:
                task.updated_at = data["updated_at"]
                
        return task

    def validate(self) -> List[str]:
        """
        Validate task data and return any errors.
        
        Returns:
            List of validation error messages, empty if valid
        """
        errors = []
        
        # Required fields
        if not self.name:
            errors.append("Task name is required")
            
        if not self.scheduled_time:
            errors.append("Scheduled time is required")
            
        if not self.frequency:
            errors.append("Task frequency is required")
            
        # Validate execution mode specific fields
        if self.execution_mode == ExecutionMode.DIRECT:
            if not self.tool_name:
                errors.append("Tool name is required for direct execution mode")
            if not self.operation:
                errors.append("Operation is required for direct execution mode")
                
        elif self.execution_mode == ExecutionMode.ORCHESTRATED:
            if not self.task_description:
                errors.append("Task description is required for orchestrated execution mode")
                
        # Frequency-specific validations
        if self.frequency == TaskFrequency.WEEKLY and self.day_of_week is None:
            errors.append("Day of week is required for weekly tasks")
            
        if self.frequency == TaskFrequency.MONTHLY and self.day_of_month is None:
            errors.append("Day of month is required for monthly tasks")
                
        return errors


# Configuration class for scheduler
class SchedulerConfig(BaseModel):
    """Configuration for the task scheduler."""
    enabled: bool = Field(default=True, description="Whether the scheduler is enabled by default")
    check_interval: int = Field(default=60, description="Interval in seconds to check for scheduled tasks")
    log_level: str = Field(default="INFO", description="Logging level for the scheduler")
    max_concurrent_tasks: int = Field(default=5, description="Maximum number of tasks to run concurrently")
    task_timeout: int = Field(default=300, description="Default timeout for tasks in seconds")
    orchestration_model: str = Field(default="claude-3-haiku-20240307", 
                                   description="Model to use for orchestrated tasks")
    default_system_prompt: str = Field(default="You are a task automation assistant that helps execute scheduled tasks. "
                                             "Your goal is to accomplish the task described using the available tools. "
                                             "Be thorough, efficient, and provide clear explanations of your actions.",
                                     description="Default system prompt for orchestrated tasks")


# Register with the configuration registry
registry.register("scheduler", SchedulerConfig)