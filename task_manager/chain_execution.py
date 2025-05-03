"""
Chain execution module for tracking and managing task chain executions.

This module defines the database models for chain executions and step executions,
providing a way to track the progress and results of task chain runs.
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
from errors import ToolError, ErrorCode, error_context
from task_manager.task_chain import ChainExecutionStatus, StepExecutionStatus, TaskChain, ChainStep

# Configure logger
logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    """Enumeration of execution trigger types."""
    SCHEDULED = "scheduled"   # Triggered by scheduler at scheduled time
    MANUAL = "manual"         # Triggered manually by user
    API = "api"               # Triggered via API call
    EVENT = "event"           # Triggered by an event


class ChainExecution(Base):
    """
    Chain execution model for tracking chain execution instances.
    
    Maps to the 'chain_executions' table with columns for execution details
    including timing, status, and results.
    """
    __tablename__ = 'chain_executions'
    
    # Primary key
    id = Column(String, primary_key=True)
    
    # Relationship to chain
    chain_id = Column(String, ForeignKey('task_chains.id'), nullable=False)
    
    # Execution details
    status = Column(SQLAEnum(ChainExecutionStatus), default=ChainExecutionStatus.PENDING)
    trigger_type = Column(SQLAEnum(TriggerType), default=TriggerType.SCHEDULED)
    
    # Timing
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    runtime_seconds = Column(Integer)  # Total execution time
    
    # Result information
    error = Column(Text)  # Error message if failed
    execution_context = Column(MutableDict.as_mutable(JSON), default={})  # Execution context including results
    
    # Metadata
    created_by = Column(String)  # User or system identifier
    
    # Define indices for frequently queried fields
    __table_args__ = (
        Index('ix_chain_executions_chain_id', 'chain_id'),
        Index('ix_chain_executions_status', 'status'),
        Index('ix_chain_executions_started_at', 'started_at'),
    )
    
    # Relationships
    chain = relationship("TaskChain", foreign_keys=[chain_id])
    step_executions = relationship("StepExecution", back_populates="chain_execution",
                                 order_by="StepExecution.started_at",
                                 cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict representation of the chain execution
        """
        return {
            "id": self.id,
            "chain_id": self.chain_id,
            
            # Execution details
            "status": self.status.value if self.status else None,
            "trigger_type": self.trigger_type.value if self.trigger_type else None,
            
            # Timing
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "runtime_seconds": self.runtime_seconds,
            
            # Result information
            "error": self.error,
            "execution_context": self.execution_context,
            
            # Metadata
            "created_by": self.created_by,
            
            # Include step executions if loaded
            "step_executions": [step_exec.to_dict() for step_exec in self.step_executions] 
                              if self.step_executions else []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChainExecution':
        """
        Create a ChainExecution instance from a dictionary.
        
        Args:
            data: Dictionary representation of a chain execution
            
        Returns:
            ChainExecution instance
        """
        execution = cls(
            id=data.get("id") or str(uuid.uuid4()),
            chain_id=data.get("chain_id"),
            error=data.get("error"),
            execution_context=data.get("execution_context", {}),
            created_by=data.get("created_by"),
            runtime_seconds=data.get("runtime_seconds")
        )
        
        # Set status
        if "status" in data and data["status"]:
            execution.status = ChainExecutionStatus(data["status"])
        else:
            execution.status = ChainExecutionStatus.PENDING
            
        # Set trigger type
        if "trigger_type" in data and data["trigger_type"]:
            execution.trigger_type = TriggerType(data["trigger_type"])
        else:
            execution.trigger_type = TriggerType.SCHEDULED
            
        # Handle datetime fields
        if "started_at" in data and data["started_at"]:
            if isinstance(data["started_at"], str):
                execution.started_at = datetime.fromisoformat(data["started_at"])
            else:
                execution.started_at = data["started_at"]
                
        if "completed_at" in data and data["completed_at"]:
            if isinstance(data["completed_at"], str):
                execution.completed_at = datetime.fromisoformat(data["completed_at"])
            else:
                execution.completed_at = data["completed_at"]
                
        return execution


class StepExecution(Base):
    """
    Step execution model for tracking step execution instances.
    
    Maps to the 'step_executions' table with columns for step execution details
    including timing, status, and results.
    """
    __tablename__ = 'step_executions'
    
    # Primary key
    id = Column(String, primary_key=True)
    
    # Relationships
    chain_execution_id = Column(String, ForeignKey('chain_executions.id'), nullable=False)
    step_id = Column(String, ForeignKey('chain_steps.id'), nullable=False)
    
    # Execution details
    status = Column(SQLAEnum(StepExecutionStatus), default=StepExecutionStatus.PENDING)
    position = Column(Integer, nullable=False)  # Position in the execution sequence
    
    # Parameters used for this execution
    resolved_parameters = Column(MutableDict.as_mutable(JSON), default={})
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    runtime_seconds = Column(Integer)  # Execution time in seconds
    
    # Result information
    result = Column(MutableDict.as_mutable(JSON))  # Result of step execution
    error = Column(Text)  # Error message if failed
    next_step_id = Column(String)  # ID of the next step to execute
    
    # Retry information
    attempts = Column(Integer, default=0)  # Number of attempts
    
    # Define indices for frequently queried fields
    __table_args__ = (
        Index('ix_step_executions_chain_execution_id', 'chain_execution_id'),
        Index('ix_step_executions_step_id', 'step_id'),
        Index('ix_step_executions_status', 'status'),
    )
    
    # Relationships
    chain_execution = relationship("ChainExecution", back_populates="step_executions")
    step = relationship("ChainStep", foreign_keys=[step_id])
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict representation of the step execution
        """
        return {
            "id": self.id,
            "chain_execution_id": self.chain_execution_id,
            "step_id": self.step_id,
            
            # Execution details
            "status": self.status.value if self.status else None,
            "position": self.position,
            "resolved_parameters": self.resolved_parameters,
            
            # Timing
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "runtime_seconds": self.runtime_seconds,
            
            # Result information
            "result": self.result,
            "error": self.error,
            "next_step_id": self.next_step_id,
            
            # Retry information
            "attempts": self.attempts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StepExecution':
        """
        Create a StepExecution instance from a dictionary.
        
        Args:
            data: Dictionary representation of a step execution
            
        Returns:
            StepExecution instance
        """
        step_execution = cls(
            id=data.get("id") or str(uuid.uuid4()),
            chain_execution_id=data.get("chain_execution_id"),
            step_id=data.get("step_id"),
            position=data.get("position", 0),
            resolved_parameters=data.get("resolved_parameters", {}),
            result=data.get("result"),
            error=data.get("error"),
            next_step_id=data.get("next_step_id"),
            attempts=data.get("attempts", 0),
            runtime_seconds=data.get("runtime_seconds")
        )
        
        # Set status
        if "status" in data and data["status"]:
            step_execution.status = StepExecutionStatus(data["status"])
        else:
            step_execution.status = StepExecutionStatus.PENDING
            
        # Handle datetime fields
        if "started_at" in data and data["started_at"]:
            if isinstance(data["started_at"], str):
                step_execution.started_at = datetime.fromisoformat(data["started_at"])
            else:
                step_execution.started_at = data["started_at"]
                
        if "completed_at" in data and data["completed_at"]:
            if isinstance(data["completed_at"], str):
                step_execution.completed_at = datetime.fromisoformat(data["completed_at"])
            else:
                step_execution.completed_at = data["completed_at"]
                
        return step_execution