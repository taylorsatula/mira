"""
Automation models for creating and managing automated tasks and sequences.

This module defines the database models for the unified automation system,
providing a way to create both simple tasks and multi-step sequences with
data passing, error handling, and conditional logic.
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Union

from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, JSON, ForeignKey, Index
from sqlalchemy import Enum as SQLAEnum
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship

from db import Base, Database
from config.registry import registry
from errors import ToolError, ErrorCode, error_context

# Configure logger
logger = logging.getLogger(__name__)


class AutomationType(str, Enum):
    """Enumeration of automation types."""
    SIMPLE_TASK = "simple_task"   # Single action task
    SEQUENCE = "sequence"         # Multi-step sequence


class AutomationStatus(str, Enum):
    """Enumeration of automation status values."""
    ACTIVE = "active"         # Automation is active and can be executed
    PAUSED = "paused"         # Automation is temporarily paused
    COMPLETED = "completed"   # Automation has completed all executions
    FAILED = "failed"         # Automation failed and needs attention
    ARCHIVED = "archived"     # Automation is no longer active


class ExecutionMode(str, Enum):
    """Enumeration of execution modes."""
    DIRECT = "direct"             # Direct tool execution
    ORCHESTRATED = "orchestrated" # LLM-orchestrated execution


class ExecutionStatus(str, Enum):
    """Enumeration of execution status values."""
    PENDING = "pending"       # Execution is waiting to start
    RUNNING = "running"       # Currently running
    COMPLETED = "completed"   # Completed successfully
    FAILED = "failed"         # Execution failed
    CANCELLED = "cancelled"   # Execution was cancelled


class StepExecutionStatus(str, Enum):
    """Enumeration of step execution status values."""
    PENDING = "pending"       # Step is waiting to be executed
    RUNNING = "running"       # Step is currently running
    COMPLETED = "completed"   # Step completed successfully
    FAILED = "failed"         # Step execution failed
    SKIPPED = "skipped"       # Step was skipped due to conditions


class ErrorPolicy(str, Enum):
    """Enumeration of error handling policies."""
    STOP = "stop"             # Stop execution on error
    CONTINUE = "continue"     # Continue to next step, ignoring error
    RETRY = "retry"           # Retry the failed step
    ALTERNATIVE = "alternative"  # Use alternative step on failure
    SILENT = "silent"         # Fail silently without notification


class TaskFrequency(str, Enum):
    """Enumeration of task frequency values."""
    ONCE = "once"             # Run once at the specified time
    MINUTELY = "minutely"     # Run every minute
    HOURLY = "hourly"         # Run every hour
    DAILY = "daily"           # Run every day
    WEEKLY = "weekly"         # Run every week
    MONTHLY = "monthly"       # Run every month
    CUSTOM = "custom"         # Custom schedule


class TriggerType(str, Enum):
    """Enumeration of trigger types."""
    SCHEDULED = "scheduled"   # Triggered by schedule
    MANUAL = "manual"         # Triggered manually
    EVENT = "event"           # Triggered by an event
    API = "api"               # Triggered by API call


class Automation(Base):
    """
    Automation model for storing unified automation data.
    
    Maps to the 'automations' table with columns for both simple tasks
    and multi-step sequences with common scheduling and execution tracking.
    """
    __tablename__ = 'automations'
    
    # Primary key
    id = Column(String, primary_key=True)
    
    # Basic information
    name = Column(String, nullable=False)
    description = Column(Text)
    type = Column(SQLAEnum(AutomationType), nullable=False)
    
    # Schedule information
    frequency = Column(SQLAEnum(TaskFrequency), nullable=False)
    scheduled_time = Column(DateTime, nullable=False)  # Base time for scheduling
    day_of_week = Column(Integer)  # For weekly (0=Monday, 6=Sunday)
    day_of_month = Column(Integer)  # For monthly
    end_time = Column(DateTime)  # For recurring with end date
    custom_schedule = Column(String)  # For custom schedules
    timezone = Column(String)  # Store the timezone for this automation
    
    # Status tracking
    status = Column(SQLAEnum(AutomationStatus), default=AutomationStatus.ACTIVE)
    last_execution_id = Column(String)  # ID of the last execution
    last_execution_time = Column(DateTime)
    next_execution_time = Column(DateTime)
    execution_count = Column(Integer, default=0)
    max_executions = Column(Integer)  # Maximum number of executions (optional)
    
    # Simple task fields (used when type is SIMPLE_TASK)
    execution_mode = Column(SQLAEnum(ExecutionMode))
    tool_name = Column(String)
    operation = Column(String)
    parameters = Column(MutableDict.as_mutable(JSON), default={})
    task_description = Column(Text)  # For orchestrated mode
    available_tools = Column(JSON)  # List of tool names for orchestrated execution
    
    # Sequence fields (used when type is SEQUENCE)
    error_policy = Column(SQLAEnum(ErrorPolicy), default=ErrorPolicy.CONTINUE)
    timeout = Column(Integer, default=3600)  # seconds
    
    # User-specific fields
    user_id = Column(String)  # User who created this automation
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                         onupdate=lambda: datetime.now(timezone.utc))
    version = Column(Integer, default=1)
    
    # Define indices for frequently queried fields
    __table_args__ = (
        Index('ix_automations_status', 'status'),
        Index('ix_automations_type', 'type'),
        Index('ix_automations_next_execution_time', 'next_execution_time'),
        Index('ix_automations_created_at', 'created_at'),
        Index('ix_automations_user_id', 'user_id'),
    )
    
    # Relationships
    steps = relationship("AutomationStep", back_populates="automation", 
                          order_by="AutomationStep.position", cascade="all, delete-orphan")
    executions = relationship("AutomationExecution", back_populates="automation",
                              cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict representation of the automation with timestamps converted to user timezone
        """
        # Import here to avoid circular imports
        from utils.timezone_utils import convert_to_timezone, get_default_timezone
        
        # Always perform timezone conversion for consistency
        # Use automation's timezone if available, otherwise system default
        tz_name = self.timezone or get_default_timezone()
        
        # Function to handle datetime conversion
        def format_datetime(dt: Optional[datetime]) -> Optional[str]:
            if not dt:
                return None
            
            try:
                # Always convert to ensure timezone is handled consistently
                dt = convert_to_timezone(dt, tz_name)
            except Exception as e:
                # Log but don't fail if timezone conversion fails
                logger.warning(f"Failed to convert datetime to timezone {tz_name}: {e}")
            
            return dt.isoformat()
        
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.value if self.type else None,
            
            # Schedule fields
            "frequency": self.frequency.value if self.frequency else None,
            "scheduled_time": format_datetime(self.scheduled_time),
            "day_of_week": self.day_of_week,
            "day_of_month": self.day_of_month,
            "end_time": format_datetime(self.end_time),
            "custom_schedule": self.custom_schedule,
            "timezone": self.timezone,
            
            # Status fields
            "status": self.status.value if self.status else None,
            "last_execution_time": format_datetime(self.last_execution_time),
            "next_execution_time": format_datetime(self.next_execution_time),
            "execution_count": self.execution_count,
            "max_executions": self.max_executions,
            
            # User identification
            "user_id": self.user_id,
            
            # Metadata
            "created_at": format_datetime(self.created_at),
            "updated_at": format_datetime(self.updated_at),
            "version": self.version,
        }
        
        # Add type-specific fields
        if self.type == AutomationType.SIMPLE_TASK:
            result.update({
                "execution_mode": self.execution_mode.value if self.execution_mode else None,
                "tool_name": self.tool_name,
                "operation": self.operation,
                "parameters": self.parameters,
                "task_description": self.task_description,
                "available_tools": self.available_tools,
            })
        elif self.type == AutomationType.SEQUENCE:
            result.update({
                "error_policy": self.error_policy.value if self.error_policy else None,
                "timeout": self.timeout,
                # Include steps if loaded
                "steps": [step.to_dict() for step in self.steps] if self.steps else []
            })
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Automation':
        """
        Create an Automation instance from a dictionary.
        
        Args:
            data: Dictionary representation of an automation
            
        Returns:
            Automation instance
        """
        automation = cls(
            id=data.get("id") or str(uuid.uuid4()),
            name=data.get("name"),
            description=data.get("description"),
            day_of_week=data.get("day_of_week"),
            day_of_month=data.get("day_of_month"),
            custom_schedule=data.get("custom_schedule"),
            timezone=data.get("timezone"),
            execution_count=data.get("execution_count", 0),
            max_executions=data.get("max_executions"),
            user_id=data.get("user_id"),
            version=data.get("version", 1)
        )
        
        # Set type-specific fields
        automation_type = data.get("type")
        if automation_type:
            automation.type = AutomationType(automation_type)
            
            # Set fields based on type
            if automation.type == AutomationType.SIMPLE_TASK:
                if "execution_mode" in data and data["execution_mode"]:
                    automation.execution_mode = ExecutionMode(data["execution_mode"])
                automation.tool_name = data.get("tool_name")
                automation.operation = data.get("operation")
                automation.parameters = data.get("parameters", {})
                automation.task_description = data.get("task_description")
                automation.available_tools = data.get("available_tools")
            elif automation.type == AutomationType.SEQUENCE:
                if "error_policy" in data and data["error_policy"]:
                    automation.error_policy = ErrorPolicy(data["error_policy"])
                automation.timeout = data.get("timeout", 3600)
        else:
            # Default to simple task if not specified
            automation.type = AutomationType.SIMPLE_TASK
        
        # Set frequency
        if "frequency" in data and data["frequency"]:
            automation.frequency = TaskFrequency(data["frequency"])
        else:
            automation.frequency = TaskFrequency.ONCE
            
        # Set status
        if "status" in data and data["status"]:
            automation.status = AutomationStatus(data["status"])
        else:
            automation.status = AutomationStatus.ACTIVE
        
        # Handle datetime fields
        if "scheduled_time" in data and data["scheduled_time"]:
            if isinstance(data["scheduled_time"], str):
                automation.scheduled_time = datetime.fromisoformat(data["scheduled_time"])
            else:
                automation.scheduled_time = data["scheduled_time"]
                
        if "end_time" in data and data["end_time"]:
            if isinstance(data["end_time"], str):
                automation.end_time = datetime.fromisoformat(data["end_time"])
            else:
                automation.end_time = data["end_time"]
                
        if "last_execution_time" in data and data["last_execution_time"]:
            if isinstance(data["last_execution_time"], str):
                automation.last_execution_time = datetime.fromisoformat(data["last_execution_time"])
            else:
                automation.last_execution_time = data["last_execution_time"]
                
        if "next_execution_time" in data and data["next_execution_time"]:
            if isinstance(data["next_execution_time"], str):
                automation.next_execution_time = datetime.fromisoformat(data["next_execution_time"])
            else:
                automation.next_execution_time = data["next_execution_time"]
                
        # Set timestamps if provided
        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                automation.created_at = datetime.fromisoformat(data["created_at"])
            else:
                automation.created_at = data["created_at"]
                
        if "updated_at" in data and data["updated_at"]:
            if isinstance(data["updated_at"], str):
                automation.updated_at = datetime.fromisoformat(data["updated_at"])
            else:
                automation.updated_at = data["updated_at"]
                
        # Handle steps for sequences
        if automation.type == AutomationType.SEQUENCE and "steps" in data and data["steps"]:
            position = 0
            for step_data in data["steps"]:
                position += 1
                step_data["position"] = position
                # Add automation_id to the step data
                step_data["automation_id"] = automation.id
                step = AutomationStep.from_dict(step_data)
                automation.steps.append(step)
                
        return automation
    
    def validate(self) -> List[str]:
        """
        Validate automation data and return any errors.
        
        Returns:
            List of validation error messages, empty if valid
        """
        errors = []
        
        # Required fields
        if not self.name:
            errors.append("Automation name is required")
            
        if not self.scheduled_time:
            errors.append("Scheduled time is required")
            
        if not self.frequency:
            errors.append("Frequency is required")
            
        # Frequency-specific validations
        if self.frequency == TaskFrequency.WEEKLY and self.day_of_week is None:
            errors.append("Day of week is required for weekly frequency")
            
        if self.frequency == TaskFrequency.MONTHLY and self.day_of_month is None:
            errors.append("Day of month is required for monthly frequency")
            
        # Type-specific validations
        if self.type == AutomationType.SIMPLE_TASK:
            if self.execution_mode == ExecutionMode.DIRECT:
                if not self.tool_name:
                    errors.append("Tool name is required for direct execution mode")
                if not self.operation:
                    errors.append("Operation is required for direct execution mode")
            elif self.execution_mode == ExecutionMode.ORCHESTRATED:
                if not self.task_description:
                    errors.append("Task description is required for orchestrated execution mode")
            else:
                errors.append("Execution mode is required for simple tasks")
        elif self.type == AutomationType.SEQUENCE:
            # Validate steps
            if not self.steps:
                errors.append("At least one step is required for sequence type")
            else:
                # Check for duplicate positions
                positions = [step.position for step in self.steps]
                if len(positions) != len(set(positions)):
                    errors.append("Duplicate step positions detected")
                    
                # Validate each step
                for step in self.steps:
                    step_errors = step.validate()
                    if step_errors:
                        errors.extend([f"Step '{step.name}': {err}" for err in step_errors])
                
        return errors


class ConditionType(str, Enum):
    """Enumeration of condition types for steps."""
    ALWAYS = "always"           # Always execute (default)
    IF_SUCCESS = "if_success"   # Execute if previous step succeeded
    IF_FAILURE = "if_failure"   # Execute if previous step failed
    IF_DATA = "if_data"         # Execute if data condition is met
    IF_NO_DATA = "if_no_data"   # Execute if data condition is not met


class AutomationStep(Base):
    """
    Automation step model for storing step data.
    
    Maps to the 'automation_steps' table with columns for step details
    including tool operations, parameters, and execution configuration.
    """
    __tablename__ = 'automation_steps'
    
    # Primary key
    id = Column(String, primary_key=True)
    
    # Relationship to automation
    automation_id = Column(String, ForeignKey('automations.id'), nullable=False)
    
    # Step details
    name = Column(String, nullable=False)
    position = Column(Integer, nullable=False)  # Order within the sequence
    description = Column(Text)  # Optional human-readable description
    
    # Execution configuration
    execution_mode = Column(SQLAEnum(ExecutionMode), default=ExecutionMode.DIRECT)
    tool_name = Column(String)
    operation = Column(String)
    task_description = Column(Text)  # For orchestrated steps
    available_tools = Column(JSON)  # List of tool names for orchestrated steps
    
    # Parameters and results
    parameters = Column(MutableDict.as_mutable(JSON), default={})
    output_key = Column(String, nullable=False)  # Key to store result under
    
    # Conditional execution
    condition_type = Column(SQLAEnum(ConditionType), default=ConditionType.ALWAYS)
    condition_data_key = Column(String)  # Data key to check for condition
    condition_value = Column(String)  # Value to compare against
    condition_operator = Column(String, default="eq")  # eq, neq, gt, lt, etc.
    
    # Error handling
    error_policy = Column(SQLAEnum(ErrorPolicy))  # Step-specific error policy
    timeout = Column(Integer)  # Step-specific timeout in seconds
    max_retries = Column(Integer, default=0)
    retry_delay = Column(Integer, default=60)  # Seconds between retries
    
    # Alternative path (branching)
    on_success_step_id = Column(String)  # Override next step if successful
    on_failure_step_id = Column(String)  # Override next step if failed
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                        onupdate=lambda: datetime.now(timezone.utc))
    
    # Define indices for frequently queried fields
    __table_args__ = (
        Index('ix_automation_steps_automation_id', 'automation_id'),
        Index('ix_automation_steps_position', 'automation_id', 'position'),
    )
    
    # Relationships
    automation = relationship("Automation", back_populates="steps")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict representation of the automation step
        """
        return {
            "id": self.id,
            "automation_id": self.automation_id,
            "name": self.name,
            "position": self.position,
            "description": self.description,
            
            # Execution configuration
            "execution_mode": self.execution_mode.value if self.execution_mode else None,
            "tool_name": self.tool_name,
            "operation": self.operation,
            "task_description": self.task_description,
            "available_tools": self.available_tools,
            
            # Parameters and results
            "parameters": self.parameters,
            "output_key": self.output_key,
            
            # Conditional execution
            "condition_type": self.condition_type.value if self.condition_type else None,
            "condition_data_key": self.condition_data_key,
            "condition_value": self.condition_value,
            "condition_operator": self.condition_operator,
            
            # Error handling
            "error_policy": self.error_policy.value if self.error_policy else None,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            
            # Alternative paths
            "on_success_step_id": self.on_success_step_id,
            "on_failure_step_id": self.on_failure_step_id,
            
            # Metadata
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutomationStep':
        """
        Create an AutomationStep instance from a dictionary.
        
        Args:
            data: Dictionary representation of an automation step
            
        Returns:
            AutomationStep instance
        """
        step = cls(
            id=data.get("id") or str(uuid.uuid4()),
            automation_id=data.get("automation_id"),
            name=data.get("name"),
            position=data.get("position", 0),
            description=data.get("description"),
            tool_name=data.get("tool_name"),
            operation=data.get("operation"),
            task_description=data.get("task_description"),
            available_tools=data.get("available_tools"),
            parameters=data.get("parameters", {}),
            output_key=data.get("output_key"),
            condition_data_key=data.get("condition_data_key"),
            condition_value=data.get("condition_value"),
            condition_operator=data.get("condition_operator", "eq"),
            timeout=data.get("timeout"),
            max_retries=data.get("max_retries", 0),
            retry_delay=data.get("retry_delay", 60),
            on_success_step_id=data.get("on_success_step_id"),
            on_failure_step_id=data.get("on_failure_step_id")
        )
        
        # Set execution mode
        if "execution_mode" in data and data["execution_mode"]:
            step.execution_mode = ExecutionMode(data["execution_mode"])
        else:
            # Determine execution mode based on available fields
            if data.get("task_description"):
                step.execution_mode = ExecutionMode.ORCHESTRATED
            else:
                step.execution_mode = ExecutionMode.DIRECT
        
        # Set condition type
        if "condition_type" in data and data["condition_type"]:
            step.condition_type = ConditionType(data["condition_type"])
        else:
            step.condition_type = ConditionType.ALWAYS
        
        # Set error policy if provided
        if "error_policy" in data and data["error_policy"]:
            step.error_policy = ErrorPolicy(data["error_policy"])
            
        # Set timestamps if provided
        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                step.created_at = datetime.fromisoformat(data["created_at"])
            else:
                step.created_at = data["created_at"]
                
        if "updated_at" in data and data["updated_at"]:
            if isinstance(data["updated_at"], str):
                step.updated_at = datetime.fromisoformat(data["updated_at"])
            else:
                step.updated_at = data["updated_at"]
                
        return step
    
    def validate(self) -> List[str]:
        """
        Validate step data and return any errors.
        
        Returns:
            List of validation error messages, empty if valid
        """
        errors = []
        
        # Required fields
        if not self.name:
            errors.append("Step name is required")
            
        if not self.output_key:
            errors.append("Output key is required")
            
        # Validate execution mode specific fields
        if self.execution_mode == ExecutionMode.DIRECT:
            if not self.tool_name:
                errors.append("Tool name is required for direct execution mode")
            if not self.operation:
                errors.append("Operation is required for direct execution mode")
                
        elif self.execution_mode == ExecutionMode.ORCHESTRATED:
            if not self.task_description:
                errors.append("Task description is required for orchestrated execution mode")
                
        # Validate condition type specific fields
        if self.condition_type in [ConditionType.IF_DATA, ConditionType.IF_NO_DATA]:
            if not self.condition_data_key:
                errors.append("Condition data key is required for data conditions")
        
        return errors


class AutomationExecution(Base):
    """
    Automation execution model for tracking executions.
    
    Maps to the 'automation_executions' table with columns for execution
    details, timing, and results.
    """
    __tablename__ = 'automation_executions'
    
    # Primary key
    id = Column(String, primary_key=True)
    
    # Relationship to automation
    automation_id = Column(String, ForeignKey('automations.id'), nullable=False)
    
    # Execution details
    status = Column(SQLAEnum(ExecutionStatus), default=ExecutionStatus.PENDING)
    trigger_type = Column(SQLAEnum(TriggerType), default=TriggerType.SCHEDULED)
    
    # Timing information
    scheduled_time = Column(DateTime)  # When it was scheduled to run
    started_at = Column(DateTime)  # When it actually started
    completed_at = Column(DateTime)  # When it completed or failed
    runtime_seconds = Column(Integer)  # Total execution time
    
    # Results
    execution_context = Column(MutableDict.as_mutable(JSON), default={})  # Data passed between steps
    result = Column(MutableDict.as_mutable(JSON))  # Final result data
    error = Column(Text)  # Error message if failed
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Define indices for frequently queried fields
    __table_args__ = (
        Index('ix_automation_executions_automation_id', 'automation_id'),
        Index('ix_automation_executions_status', 'status'),
        Index('ix_automation_executions_created_at', 'created_at'),
        Index('ix_automation_executions_scheduled_time', 'scheduled_time'),
    )
    
    # Relationships
    automation = relationship("Automation", back_populates="executions")
    step_executions = relationship("StepExecution", back_populates="execution",
                                  cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict representation of the execution with timestamps in automation's timezone
        """
        # Import here to avoid circular imports
        from utils.timezone_utils import convert_to_timezone, get_default_timezone
        
        # Get the automation's timezone if available, otherwise use system default
        timezone_name = None
        if self.automation:
            timezone_name = self.automation.timezone
        
        # Always perform timezone conversion for consistency
        tz_name = timezone_name or get_default_timezone()
        
        # Function to handle datetime conversion
        def format_datetime(dt: Optional[datetime]) -> Optional[str]:
            if not dt:
                return None
            
            try:
                # Always convert to ensure timezone is handled consistently
                dt = convert_to_timezone(dt, tz_name)
            except Exception as e:
                # Log but don't fail if timezone conversion fails
                logger.warning(f"Failed to convert datetime to timezone {tz_name}: {e}")
            
            return dt.isoformat()
        
        return {
            "id": self.id,
            "automation_id": self.automation_id,
            "status": self.status.value if self.status else None,
            "trigger_type": self.trigger_type.value if self.trigger_type else None,
            
            # Timing information
            "scheduled_time": format_datetime(self.scheduled_time),
            "started_at": format_datetime(self.started_at),
            "completed_at": format_datetime(self.completed_at),
            "runtime_seconds": self.runtime_seconds,
            
            # Results
            "execution_context": self.execution_context,
            "result": self.result,
            "error": self.error,
            
            # Metadata
            "created_at": format_datetime(self.created_at),
            
            # Include steps if loaded - pass timezone to ensure consistency
            "step_executions": [step.to_dict(tz_name) for step in self.step_executions] if self.step_executions else []
        }


class StepExecution(Base):
    """
    Step execution model for tracking step executions.
    
    Maps to the 'step_executions' table with columns for step-specific
    execution details, timing, and results.
    """
    __tablename__ = 'step_executions'
    
    # Primary key
    id = Column(String, primary_key=True)
    
    # Relationships
    execution_id = Column(String, ForeignKey('automation_executions.id'), nullable=False)
    step_id = Column(String, ForeignKey('automation_steps.id'))
    
    # Execution details
    position = Column(Integer, nullable=False)  # Position in the execution order
    status = Column(SQLAEnum(StepExecutionStatus), default=StepExecutionStatus.PENDING)
    
    # Timing information
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    runtime_seconds = Column(Integer)
    
    # Results
    resolved_parameters = Column(MutableDict.as_mutable(JSON))  # Parameters after template substitution
    result = Column(MutableDict.as_mutable(JSON))  # Step execution result
    error = Column(Text)  # Error message if failed
    
    # Retry information
    attempts = Column(Integer, default=1)  # Number of attempts made
    next_step_id = Column(String)  # ID of the next step to execute (for conditional branching)
    condition_result = Column(Boolean)  # Result of condition evaluation
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Define indices for frequently queried fields
    __table_args__ = (
        Index('ix_step_executions_execution_id', 'execution_id'),
        Index('ix_step_executions_step_id', 'step_id'),
        Index('ix_step_executions_status', 'status'),
    )
    
    # Relationships
    execution = relationship("AutomationExecution", back_populates="step_executions")
    
    def to_dict(self, timezone_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Args:
            timezone_name: Timezone to convert timestamps to (defaults to UTC)
            
        Returns:
            Dict representation of the step execution with converted timestamps
        """
        # Import here to avoid circular imports
        from utils.timezone_utils import convert_to_timezone, get_default_timezone
        
        # Use provided timezone, or get from execution's automation, or use system default
        tz_name = timezone_name
        if not tz_name and self.execution and hasattr(self.execution, 'automation') and self.execution.automation:
            tz_name = self.execution.automation.timezone
        
        # Always perform timezone conversion for consistency
        tz_name = tz_name or get_default_timezone()
            
        # Function to handle datetime conversion
        def format_datetime(dt: Optional[datetime]) -> Optional[str]:
            if not dt:
                return None
            
            try:
                # Always convert to ensure timezone is handled consistently
                dt = convert_to_timezone(dt, tz_name)
            except Exception as e:
                # Log but don't fail if timezone conversion fails
                logger.warning(f"Failed to convert datetime to timezone {tz_name}: {e}")
            
            return dt.isoformat()
        
        return {
            "id": self.id,
            "execution_id": self.execution_id,
            "step_id": self.step_id,
            "position": self.position,
            "status": self.status.value if self.status else None,
            
            # Timing information
            "started_at": format_datetime(self.started_at),
            "completed_at": format_datetime(self.completed_at),
            "runtime_seconds": self.runtime_seconds,
            
            # Results
            "resolved_parameters": self.resolved_parameters,
            "result": self.result,
            "error": self.error,
            
            # Retry information
            "attempts": self.attempts,
            "next_step_id": self.next_step_id,
            "condition_result": self.condition_result,
            
            # Metadata
            "created_at": format_datetime(self.created_at)
        }