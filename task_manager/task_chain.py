"""
Task chain module for creating and managing chains of related tasks.

This module defines the database models for task chains and chain steps,
providing a way to create sequences of operations that pass data between
steps, handle errors, and implement conditional logic.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Union

from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, JSON, ForeignKey, Index
from sqlalchemy import Enum as SQLAEnum
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

from db import Base, Database
from config.registry import registry
from errors import ToolError, ErrorCode, error_context
from task_manager.scheduled_task import TaskFrequency, ExecutionMode

# Configure logger
logger = logging.getLogger(__name__)


class ChainStatus(str, Enum):
    """Enumeration of task chain status values."""
    ACTIVE = "active"         # Chain is active and can be executed
    PAUSED = "paused"         # Chain is temporarily paused
    COMPLETED = "completed"   # Chain has completed all executions
    FAILED = "failed"         # Chain failed and needs attention
    ARCHIVED = "archived"     # Chain is archived and no longer active


class ErrorPolicy(str, Enum):
    """Enumeration of error handling policies."""
    STOP = "stop"             # Stop chain execution on error
    CONTINUE = "continue"     # Continue to next step, ignoring error
    RETRY = "retry"           # Retry the failed step
    ALTERNATIVE = "alternative"  # Use alternative step on failure
    ROLLBACK = "rollback"     # Execute compensation steps


class StepExecutionStatus(str, Enum):
    """Enumeration of step execution status values."""
    PENDING = "pending"       # Step is waiting to be executed
    RUNNING = "running"       # Step is currently running
    COMPLETED = "completed"   # Step completed successfully
    FAILED = "failed"         # Step execution failed
    SKIPPED = "skipped"       # Step was skipped due to conditions


class ChainExecutionStatus(str, Enum):
    """Enumeration of chain execution status values."""
    PENDING = "pending"       # Chain execution is waiting to start
    RUNNING = "running"       # Chain is currently running
    COMPLETED = "completed"   # Chain completed successfully
    FAILED = "failed"         # Chain execution failed
    CANCELLED = "cancelled"   # Chain execution was cancelled


class TaskChain(Base):
    """
    Task chain model for storing chain data.
    
    Maps to the 'task_chains' table with columns for chain details
    including scheduling, steps, and execution status.
    """
    __tablename__ = 'task_chains'
    
    # Primary key
    id = Column(String, primary_key=True)
    
    # Chain details
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Schedule information
    frequency = Column(SQLAEnum(TaskFrequency), nullable=False)
    scheduled_time = Column(DateTime, nullable=False)  # Base time for scheduling (e.g., 9:00 AM)
    day_of_week = Column(Integer)  # For weekly chains (0=Monday, 6=Sunday)
    day_of_month = Column(Integer)  # For monthly chains
    end_time = Column(DateTime, nullable=True)  # For recurring chains with end date
    custom_schedule = Column(String)  # For custom schedules
    timezone = Column(String)  # Store the timezone for this chain
    
    # Status tracking
    status = Column(SQLAEnum(ChainStatus), default=ChainStatus.ACTIVE)
    last_execution_id = Column(String)  # ID of the last execution
    last_execution_time = Column(DateTime)
    next_execution_time = Column(DateTime)
    execution_count = Column(Integer, default=0)
    max_executions = Column(Integer)  # Maximum number of executions (optional)
    
    # Configuration
    error_policy = Column(SQLAEnum(ErrorPolicy), default=ErrorPolicy.CONTINUE)
    timeout = Column(Integer, default=3600)  # seconds
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                       onupdate=lambda: datetime.now(timezone.utc))
    created_by = Column(String)  # User or system identifier
    version = Column(Integer, default=1)
    
    # Define indices for frequently queried fields
    __table_args__ = (
        Index('ix_task_chains_status', 'status'),
        Index('ix_task_chains_next_execution_time', 'next_execution_time'),
        Index('ix_task_chains_created_at', 'created_at'),
    )
    
    # Relationships
    steps = relationship("ChainStep", back_populates="chain", 
                        order_by="ChainStep.position", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict representation of the task chain
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            
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
            "last_execution_time": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "next_execution_time": self.next_execution_time.isoformat() if self.next_execution_time else None,
            "execution_count": self.execution_count,
            "max_executions": self.max_executions,
            
            # Configuration
            "error_policy": self.error_policy.value if self.error_policy else None,
            "timeout": self.timeout,
            
            # Metadata
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "version": self.version,
            
            # Include steps if loaded
            "steps": [step.to_dict() for step in self.steps] if self.steps else []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskChain':
        """
        Create a TaskChain instance from a dictionary.
        
        Args:
            data: Dictionary representation of a task chain
            
        Returns:
            TaskChain instance
        """
        chain = cls(
            id=data.get("id") or str(uuid.uuid4()),
            name=data.get("name"),
            description=data.get("description"),
            day_of_week=data.get("day_of_week"),
            day_of_month=data.get("day_of_month"),
            custom_schedule=data.get("custom_schedule"),
            timezone=data.get("timezone"),
            execution_count=data.get("execution_count", 0),
            max_executions=data.get("max_executions"),
            created_by=data.get("created_by"),
            timeout=data.get("timeout", 3600),
            version=data.get("version", 1)
        )
        
        # Set frequency
        if "frequency" in data and data["frequency"]:
            chain.frequency = TaskFrequency(data["frequency"])
        else:
            chain.frequency = TaskFrequency.ONCE
        
        # Set error policy
        if "error_policy" in data and data["error_policy"]:
            chain.error_policy = ErrorPolicy(data["error_policy"])
        else:
            chain.error_policy = ErrorPolicy.CONTINUE
            
        # Set status
        if "status" in data and data["status"]:
            chain.status = ChainStatus(data["status"])
        else:
            chain.status = ChainStatus.ACTIVE
        
        # Handle datetime fields
        if "scheduled_time" in data and data["scheduled_time"]:
            if isinstance(data["scheduled_time"], str):
                chain.scheduled_time = datetime.fromisoformat(data["scheduled_time"])
            else:
                chain.scheduled_time = data["scheduled_time"]
                
        if "end_time" in data and data["end_time"]:
            if isinstance(data["end_time"], str):
                chain.end_time = datetime.fromisoformat(data["end_time"])
            else:
                chain.end_time = data["end_time"]
                
        if "last_execution_time" in data and data["last_execution_time"]:
            if isinstance(data["last_execution_time"], str):
                chain.last_execution_time = datetime.fromisoformat(data["last_execution_time"])
            else:
                chain.last_execution_time = data["last_execution_time"]
                
        if "next_execution_time" in data and data["next_execution_time"]:
            if isinstance(data["next_execution_time"], str):
                chain.next_execution_time = datetime.fromisoformat(data["next_execution_time"])
            else:
                chain.next_execution_time = data["next_execution_time"]
                
        # Set timestamps if provided
        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                chain.created_at = datetime.fromisoformat(data["created_at"])
            else:
                chain.created_at = data["created_at"]
                
        if "updated_at" in data and data["updated_at"]:
            if isinstance(data["updated_at"], str):
                chain.updated_at = datetime.fromisoformat(data["updated_at"])
            else:
                chain.updated_at = data["updated_at"]
                
        # Handle steps if provided
        if "steps" in data and data["steps"]:
            position = 0
            for step_data in data["steps"]:
                position += 1
                step_data["position"] = position
                step = ChainStep.from_dict(step_data)
                chain.steps.append(step)
                
        return chain
    
    def validate(self) -> List[str]:
        """
        Validate chain data and return any errors.
        
        Returns:
            List of validation error messages, empty if valid
        """
        errors = []
        
        # Required fields
        if not self.name:
            errors.append("Chain name is required")
            
        if not self.scheduled_time:
            errors.append("Scheduled time is required")
            
        if not self.frequency:
            errors.append("Chain frequency is required")
            
        # Frequency-specific validations
        if self.frequency == TaskFrequency.WEEKLY and self.day_of_week is None:
            errors.append("Day of week is required for weekly chains")
            
        if self.frequency == TaskFrequency.MONTHLY and self.day_of_month is None:
            errors.append("Day of month is required for monthly chains")
            
        # Validate steps
        if not self.steps:
            errors.append("At least one step is required")
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


class ChainStep(Base):
    """
    Chain step model for storing step data.
    
    Maps to the 'chain_steps' table with columns for step details
    including tool operations, parameters, and execution configuration.
    """
    __tablename__ = 'chain_steps'
    
    # Primary key
    id = Column(String, primary_key=True)
    
    # Relationship to chain
    chain_id = Column(String, ForeignKey('task_chains.id'), nullable=False)
    
    # Step details
    name = Column(String, nullable=False)
    position = Column(Integer, nullable=False)  # Order within the chain
    
    # Execution configuration
    execution_mode = Column(SQLAEnum(ExecutionMode), default=ExecutionMode.DIRECT)
    tool_name = Column(String)
    operation = Column(String)
    task_description = Column(Text)  # For orchestrated steps
    available_tools = Column(JSON)  # List of tool names for orchestrated steps
    
    # Parameters and results
    parameters = Column(MutableDict.as_mutable(JSON), default={})
    output_key = Column(String, nullable=False)  # Key to store result under
    
    # Flow control
    next_step_logic = Column(MutableDict.as_mutable(JSON), default={})  # For conditional branching
    timeout = Column(Integer)  # Step-specific timeout in seconds
    error_policy = Column(SQLAEnum(ErrorPolicy))  # Step-specific error policy
    
    # Retry configuration
    max_retries = Column(Integer, default=0)
    retry_delay = Column(Integer, default=60)  # Seconds between retries
    
    # Repeat configuration
    repeat_config = Column(MutableDict.as_mutable(JSON), default={})  # For repeating steps
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                       onupdate=lambda: datetime.now(timezone.utc))
    
    # Define indices for frequently queried fields
    __table_args__ = (
        Index('ix_chain_steps_chain_id', 'chain_id'),
        Index('ix_chain_steps_position', 'chain_id', 'position'),
    )
    
    # Relationships
    chain = relationship("TaskChain", back_populates="steps")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict representation of the chain step
        """
        return {
            "id": self.id,
            "name": self.name,
            "position": self.position,
            
            # Execution configuration
            "execution_mode": self.execution_mode.value if self.execution_mode else None,
            "tool_name": self.tool_name,
            "operation": self.operation,
            "task_description": self.task_description,
            "available_tools": self.available_tools,
            
            # Parameters and results
            "parameters": self.parameters,
            "output_key": self.output_key,
            
            # Flow control
            "next_step_logic": self.next_step_logic,
            "timeout": self.timeout,
            "error_policy": self.error_policy.value if self.error_policy else None,
            
            # Retry configuration
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            
            # Repeat configuration
            "repeat_config": self.repeat_config,
            
            # Metadata
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChainStep':
        """
        Create a ChainStep instance from a dictionary.
        
        Args:
            data: Dictionary representation of a chain step
            
        Returns:
            ChainStep instance
        """
        step = cls(
            id=data.get("id") or str(uuid.uuid4()),
            chain_id=data.get("chain_id"),
            name=data.get("name"),
            position=data.get("position", 0),
            tool_name=data.get("tool_name"),
            operation=data.get("operation"),
            task_description=data.get("task_description"),
            available_tools=data.get("available_tools"),
            parameters=data.get("parameters", {}),
            output_key=data.get("output_key"),
            next_step_logic=data.get("next_step_logic", {}),
            timeout=data.get("timeout"),
            max_retries=data.get("max_retries", 0),
            retry_delay=data.get("retry_delay", 60),
            repeat_config=data.get("repeat_config", {})
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
                
        return errors


# Configuration class for task chain system
class TaskChainConfig(BaseModel):
    """Configuration for the task chain system."""
    enabled: bool = Field(default=True, description="Whether the task chain system is enabled by default")
    check_interval: int = Field(default=60, description="Interval in seconds to check for scheduled chains")
    log_level: str = Field(default="INFO", description="Logging level for the task chain system")
    max_concurrent_chains: int = Field(default=3, description="Maximum number of chains to run concurrently")
    max_concurrent_steps: int = Field(default=5, description="Maximum number of steps to run concurrently within chains")
    default_timeout: int = Field(default=3600, description="Default timeout for chains in seconds")
    orchestration_model: str = Field(default="claude-3-haiku-20240307", 
                                  description="Model to use for orchestrated steps")
    default_system_prompt: str = Field(default="You are a task automation assistant that helps execute scheduled tasks. "
                                         "Your goal is to accomplish the task described using the available tools. "
                                         "Be thorough, efficient, and provide clear explanations of your actions.",
                                description="Default system prompt for orchestrated steps")


# Register with the configuration registry
registry.register("task_chain", TaskChainConfig)