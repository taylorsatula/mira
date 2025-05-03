"""
Chain tool for managing task chains.

This tool allows users to create, view, update, and delete task chains.
It supports both direct tool execution and LLM-orchestrated steps, with
various scheduling options and dependency handling.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, cast, Union
import pytz

from pydantic import BaseModel, Field

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context
from task_manager.task_chain import TaskChain, ChainStep, ChainStatus, ErrorPolicy
from task_manager.chain_execution import ChainExecution, StepExecution, ChainExecutionStatus, TriggerType
from task_manager.scheduled_task import TaskFrequency
from task_manager.task_scheduler import TaskScheduler
from utils import chain_scheduler_service
from config import config
from config.registry import registry

# Configure logger
logger = logging.getLogger(__name__)


# Define configuration class for ChainTool
class ChainToolConfig(BaseModel):
    """Configuration for the chain_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    max_chains_per_user: int = Field(default=20, description="Maximum number of chains a user can create")


# Register with registry
registry.register("chain_tool", ChainToolConfig)


class ChainTool(Tool):
    """
    Tool for managing task chains.
    
    This tool allows the creation and management of task chains that can
    execute sequences of operations with data passing between steps.
    It supports conditional branching, error handling, and both direct
    and LLM-orchestrated execution.
    """
    
    name = "chain_tool"
    description = """
    Manages task chains for automating sequences of operations. Use this tool 
    when the user wants to create workflows that connect multiple operations together.
    
    IMPORTANT: This tool requires parameters to be passed as a JSON string in the "kwargs" field.
    The tool supports these operations:
    
    1. create_chain: Create a new task chain with steps.
       - Required fields: name, frequency (once, minutely, hourly, daily, weekly, monthly, custom), steps
       - Each step requires: name, tool_name/operation OR task_description, output_key
       - Optional fields: description, scheduled_time, day_of_week (for weekly), day_of_month (for monthly),
         error_policy, timeout
       - Returns the created chain with its ID
    
    2. get_chains: Retrieve chains filtered by criteria.
       - Optional: status (active, paused, completed, failed, archived),
         frequency, limit (default 10), offset (default 0)
       - Returns list of chains matching the criteria
    
    3. get_chain: Get details of a specific chain with its steps.
       - Required: chain_id
       - Returns the chain details with all steps
    
    4. update_chain: Modify an existing chain.
       - Required: chain_id
       - Optional: Any fields to update (name, steps, etc.)
       - Returns the updated chain
    
    5. delete_chain: Remove a chain.
       - Required: chain_id, confirm (must be true)
       - Returns confirmation of deletion
    
    6. execute_chain_now: Run a chain immediately.
       - Required: chain_id
       - Optional: initial_context (dictionary of initial values)
       - Returns confirmation that the chain was submitted for execution
    
    7. pause_chain: Pause a chain's scheduled execution.
       - Required: chain_id
       - Returns confirmation of pausing
    
    8. resume_chain: Resume a paused chain.
       - Required: chain_id
       - Returns confirmation of resuming
    
    9. get_chain_executions: Get execution history for a chain.
       - Required: chain_id
       - Optional: limit (default 10), offset (default 0)
       - Returns list of executions with their status and results
    
    10. get_execution_details: Get detailed information about a specific execution.
       - Required: execution_id
       - Returns execution details including all step executions
       
    This tool works with two types of steps:
    
    1. Direct steps: Execute a specific tool operation with parameters
    2. Orchestrated steps: Use the LLM to interpret a task description
       and determine which tools to use
    
    Step parameters can reference results from previous steps using template
    syntax: {previous_step_output_key.field_name}
    
    Examples:
    - "Create a chain that checks weather and sends a notification if it will rain"
    - "Set up a workflow that processes customer data and generates a report"
    - "Make a chain of steps to backup my database and notify me when it's done"
    """
    
    def run(self, operation: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Run a chain_tool operation.
        
        Args:
            operation: The operation to perform
            **kwargs: Operation-specific parameters
            
        Returns:
            Operation result
            
        Raises:
            ToolError: If the operation fails
        """
        with error_context(
            component_name=self.name,
            operation=operation,
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=logger
        ):
            # Initialize scheduler service if needed
            if not chain_scheduler_service.get_chain_executor():
                chain_scheduler_service.initialize_chain_scheduler()
            
            # Route to appropriate operation
            if operation == "create_chain":
                return self._create_chain(**kwargs)
            elif operation == "get_chains":
                return self._get_chains(**kwargs)
            elif operation == "get_chain":
                return self._get_chain(**kwargs)
            elif operation == "update_chain":
                return self._update_chain(**kwargs)
            elif operation == "delete_chain":
                return self._delete_chain(**kwargs)
            elif operation == "execute_chain_now":
                return self._execute_chain_now(**kwargs)
            elif operation == "pause_chain":
                return self._pause_chain(**kwargs)
            elif operation == "resume_chain":
                return self._resume_chain(**kwargs)
            elif operation == "get_chain_executions":
                return self._get_chain_executions(**kwargs)
            elif operation == "get_execution_details":
                return self._get_execution_details(**kwargs)
            else:
                raise ToolError(
                    f"Unknown operation: {operation}",
                    ErrorCode.TOOL_INVALID_OPERATION
                )
    
    def _create_chain(
        self,
        name: str,
        frequency: str,
        steps: List[Dict[str, Any]],
        description: Optional[str] = None,
        scheduled_time: Optional[str] = None,
        day_of_week: Optional[int] = None,
        day_of_month: Optional[int] = None,
        end_time: Optional[str] = None,
        custom_schedule: Optional[str] = None,
        timezone: Optional[str] = None,
        error_policy: Optional[str] = None,
        timeout: Optional[int] = None,
        max_executions: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Create a new task chain.
        
        Args:
            name: Name of the chain
            frequency: Execution frequency (once, minutely, hourly, daily, weekly, monthly, custom)
            steps: List of steps in the chain
            description: Description of the chain
            scheduled_time: Time for execution (e.g., "09:00:00")
            day_of_week: Day of week for weekly chains (0=Monday, 6=Sunday)
            day_of_month: Day of month for monthly chains
            end_time: End date/time for recurring chains
            custom_schedule: Custom schedule specification
            timezone: Timezone for scheduling (e.g., "America/New_York")
            error_policy: Error handling policy (stop, continue, retry, alternative, rollback)
            timeout: Timeout in seconds
            max_executions: Maximum number of executions
            
        Returns:
            The created chain
            
        Raises:
            ToolError: If chain creation fails
        """
        # Validate required parameters with clear error messages
        if not name:
            raise ToolError(
                "Chain name is required. Please provide a descriptive name for the chain.",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        if not frequency:
            raise ToolError(
                "Frequency is required. Please specify one of: once, minutely, hourly, daily, weekly, monthly, custom.",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        if not steps or not isinstance(steps, list) or len(steps) == 0:
            raise ToolError(
                "At least one step is required. Each step must include name, tool_name/operation or task_description, and output_key.",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Check frequency-specific requirements
        try:
            freq = TaskFrequency(frequency)
        except ValueError:
            raise ToolError(
                f"Invalid frequency: {frequency}. Must be one of: {', '.join(f.value for f in TaskFrequency)}",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        if freq == TaskFrequency.WEEKLY and day_of_week is None:
            raise ToolError(
                "day_of_week is required for weekly frequency (0=Monday, 6=Sunday)",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        if freq == TaskFrequency.MONTHLY and day_of_month is None:
            raise ToolError(
                "day_of_month is required for monthly frequency (1-31)",
                ErrorCode.TOOL_INVALID_INPUT
            )

        # Create chain data
        chain_data = {
            "name": name,
            "description": description,
            "frequency": frequency,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "custom_schedule": custom_schedule,
            "timezone": timezone or config.system.timezone,
            "error_policy": error_policy,
            "timeout": timeout,
            "max_executions": max_executions,
            "created_by": kwargs.get("user_id", "system")
        }
        
        # Handle scheduled_time
        if scheduled_time:
            # If only time is provided (no date), use today's date
            if ":" in scheduled_time and len(scheduled_time) <= 8:
                today = datetime.now().strftime("%Y-%m-%d")
                scheduled_time = f"{today}T{scheduled_time}"
                
            try:
                chain_data["scheduled_time"] = datetime.fromisoformat(scheduled_time)
            except ValueError:
                raise ToolError(
                    f"Invalid scheduled_time format: {scheduled_time}. Use ISO format (YYYY-MM-DDTHH:MM:SS)",
                    ErrorCode.TOOL_INVALID_INPUT
                )
        else:
            # Default to current time
            chain_data["scheduled_time"] = datetime.now(timezone.utc)
        
        # Handle end_time
        if end_time:
            try:
                chain_data["end_time"] = datetime.fromisoformat(end_time)
            except ValueError:
                raise ToolError(
                    f"Invalid end_time format: {end_time}. Use ISO format (YYYY-MM-DDTHH:MM:SS)",
                    ErrorCode.TOOL_INVALID_INPUT
                )
        
        # Calculate next execution time using task_scheduler logic
        scheduler = TaskScheduler()
        
        # Create a temporary task to calculate next run time
        temp_task = {
            "frequency": frequency,
            "scheduled_time": chain_data["scheduled_time"],
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "custom_schedule": custom_schedule,
            "timezone": chain_data.get("timezone")
        }
        
        # Use the scheduler's logic to calculate the next run time
        next_run = scheduler._calculate_next_run_time(temp_task)
        chain_data["next_execution_time"] = next_run or chain_data["scheduled_time"]
        
        # Add steps to the chain
        chain_data["steps"] = steps
        
        # Create the chain
        db = chain_scheduler_service.get_chain_executor().db
        chain = TaskChain.from_dict(chain_data)
        
        # Validate the chain
        errors = chain.validate()
        if errors:
            raise ToolError(
                f"Invalid chain definition: {', '.join(errors)}",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Validate steps
        for i, step in enumerate(steps):
            if not step.get("name"):
                raise ToolError(
                    f"Step {i+1} is missing a name",
                    ErrorCode.TOOL_INVALID_INPUT
                )
                
            if not step.get("output_key"):
                raise ToolError(
                    f"Step '{step.get('name', f'at position {i+1}')}' is missing an output_key",
                    ErrorCode.TOOL_INVALID_INPUT
                )
                
            if step.get("execution_mode") == "direct" or (not step.get("execution_mode") and not step.get("task_description")):
                if not step.get("tool_name"):
                    raise ToolError(
                        f"Step '{step.get('name')}' is missing tool_name for direct execution",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                if not step.get("operation"):
                    raise ToolError(
                        f"Step '{step.get('name')}' is missing operation for direct execution",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
            elif step.get("execution_mode") == "orchestrated" or step.get("task_description"):
                if not step.get("task_description"):
                    raise ToolError(
                        f"Step '{step.get('name')}' is missing task_description for orchestrated execution",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
        
        # Save to database
        db.add(chain)
        
        logger.info(f"Created new chain: {chain.id} ('{chain.name}')")
        
        return chain.to_dict()
    
    def _get_chains(
        self,
        status: Optional[str] = None,
        frequency: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get chains filtered by criteria.
        
        Args:
            status: Filter by chain status
            frequency: Filter by execution frequency
            limit: Maximum number of chains to return
            offset: Offset for pagination
            
        Returns:
            List of chains matching the criteria
        """
        db = chain_scheduler_service.get_chain_executor().db
        
        with db.get_session() as session:
            query = session.query(TaskChain)
            
            # Apply filters
            if status:
                try:
                    query = query.filter(TaskChain.status == ChainStatus(status))
                except ValueError:
                    raise ToolError(
                        f"Invalid status: {status}. Must be one of: {', '.join(s.value for s in ChainStatus)}",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
            if frequency:
                try:
                    query = query.filter(TaskChain.frequency == TaskFrequency(frequency))
                except ValueError:
                    raise ToolError(
                        f"Invalid frequency: {frequency}. Must be one of: {', '.join(f.value for f in TaskFrequency)}",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                
            # Get total count
            total_count = query.count()
            
            # Apply pagination
            query = query.order_by(TaskChain.created_at.desc())
            query = query.limit(limit).offset(offset)
            
            # Execute query
            chains = query.all()
            
            return {
                "chains": [chain.to_dict() for chain in chains],
                "total": total_count,
                "limit": limit,
                "offset": offset
            }
    
    def _get_chain(
        self,
        chain_id: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get a chain by ID with all its steps.
        
        Args:
            chain_id: ID of the chain to retrieve
            
        Returns:
            The chain with its steps
            
        Raises:
            ToolError: If the chain is not found
        """
        if not chain_id:
            raise ToolError(
                "chain_id is required",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        db = chain_scheduler_service.get_chain_executor().db
        chain = db.get(TaskChain, chain_id)
        
        if not chain:
            raise ToolError(
                f"Chain {chain_id} not found",
                ErrorCode.TOOL_NOT_FOUND
            )
            
        return chain.to_dict()
    
    def _update_chain(
        self,
        chain_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        frequency: Optional[str] = None,
        scheduled_time: Optional[str] = None,
        day_of_week: Optional[int] = None,
        day_of_month: Optional[int] = None,
        end_time: Optional[str] = None,
        custom_schedule: Optional[str] = None,
        timezone: Optional[str] = None,
        status: Optional[str] = None,
        error_policy: Optional[str] = None,
        timeout: Optional[int] = None,
        max_executions: Optional[int] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Update an existing chain.
        
        Args:
            chain_id: ID of the chain to update
            name: New name for the chain
            description: New description
            frequency: New execution frequency
            scheduled_time: New scheduled time
            day_of_week: New day of week
            day_of_month: New day of month
            end_time: New end time
            custom_schedule: New custom schedule
            timezone: New timezone
            status: New status
            error_policy: New error policy
            timeout: New timeout
            max_executions: New maximum executions
            steps: New list of steps
            
        Returns:
            The updated chain
            
        Raises:
            ToolError: If the update fails
        """
        if not chain_id:
            raise ToolError(
                "chain_id is required",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        db = chain_scheduler_service.get_chain_executor().db
        chain = db.get(TaskChain, chain_id)
        
        if not chain:
            raise ToolError(
                f"Chain {chain_id} not found",
                ErrorCode.TOOL_NOT_FOUND
            )
        
        # Create update data
        update_data = {}
        
        # Set simple fields
        if name is not None:
            update_data["name"] = name
        if description is not None:
            update_data["description"] = description
        if day_of_week is not None:
            update_data["day_of_week"] = day_of_week
        if day_of_month is not None:
            update_data["day_of_month"] = day_of_month
        if custom_schedule is not None:
            update_data["custom_schedule"] = custom_schedule
        if timezone is not None:
            update_data["timezone"] = timezone
        if timeout is not None:
            update_data["timeout"] = timeout
        if max_executions is not None:
            update_data["max_executions"] = max_executions
        
        # Set enum fields with validation
        if frequency is not None:
            try:
                freq = TaskFrequency(frequency)
                update_data["frequency"] = frequency
            except ValueError:
                raise ToolError(
                    f"Invalid frequency: {frequency}. Must be one of: {', '.join(f.value for f in TaskFrequency)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
                
            # Check frequency-specific requirements
            if freq == TaskFrequency.WEEKLY and day_of_week is None and chain.day_of_week is None:
                raise ToolError(
                    "day_of_week is required for weekly frequency (0=Monday, 6=Sunday)",
                    ErrorCode.TOOL_INVALID_INPUT
                )
                
            if freq == TaskFrequency.MONTHLY and day_of_month is None and chain.day_of_month is None:
                raise ToolError(
                    "day_of_month is required for monthly frequency (1-31)",
                    ErrorCode.TOOL_INVALID_INPUT
                )
                
        if status is not None:
            try:
                update_data["status"] = ChainStatus(status).value
            except ValueError:
                raise ToolError(
                    f"Invalid status: {status}. Must be one of: {', '.join(s.value for s in ChainStatus)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
                
        if error_policy is not None:
            try:
                update_data["error_policy"] = ErrorPolicy(error_policy).value
            except ValueError:
                raise ToolError(
                    f"Invalid error_policy: {error_policy}. Must be one of: {', '.join(p.value for p in ErrorPolicy)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
        
        # Handle scheduled_time
        if scheduled_time:
            try:
                # If only time is provided (no date), use today's date
                if ":" in scheduled_time and len(scheduled_time) <= 8:
                    today = datetime.now().strftime("%Y-%m-%d")
                    scheduled_time = f"{today}T{scheduled_time}"
                
                update_data["scheduled_time"] = datetime.fromisoformat(scheduled_time)
            except ValueError:
                raise ToolError(
                    f"Invalid scheduled_time format: {scheduled_time}. Use ISO format (YYYY-MM-DDTHH:MM:SS)",
                    ErrorCode.TOOL_INVALID_INPUT
                )
        
        # Handle end_time
        if end_time:
            try:
                update_data["end_time"] = datetime.fromisoformat(end_time)
            except ValueError:
                raise ToolError(
                    f"Invalid end_time format: {end_time}. Use ISO format (YYYY-MM-DDTHH:MM:SS)",
                    ErrorCode.TOOL_INVALID_INPUT
                )
        
        # Recalculate next execution time if scheduling parameters changed
        if any(key in update_data for key in ["frequency", "scheduled_time", "day_of_week", "day_of_month", "custom_schedule", "timezone"]):
            scheduler = TaskScheduler()
            
            # Create a temporary task to calculate next run time
            temp_task = {
                "frequency": update_data.get("frequency", chain.frequency.value if chain.frequency else None),
                "scheduled_time": update_data.get("scheduled_time", chain.scheduled_time),
                "day_of_week": update_data.get("day_of_week", chain.day_of_week),
                "day_of_month": update_data.get("day_of_month", chain.day_of_month),
                "custom_schedule": update_data.get("custom_schedule", chain.custom_schedule),
                "timezone": update_data.get("timezone", chain.timezone)
            }
            
            # Use the scheduler's logic to calculate the next run time
            next_run = scheduler._calculate_next_run_time(temp_task)
            if next_run:
                update_data["next_execution_time"] = next_run
        
        # Handle steps update
        if steps is not None:
            # Validate steps
            for i, step in enumerate(steps):
                if not step.get("name"):
                    raise ToolError(
                        f"Step {i+1} is missing a name",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                    
                if not step.get("output_key"):
                    raise ToolError(
                        f"Step '{step.get('name', f'at position {i+1}')}' is missing an output_key",
                        ErrorCode.TOOL_INVALID_INPUT
                    )
                    
                if step.get("execution_mode") == "direct" or (not step.get("execution_mode") and not step.get("task_description")):
                    if not step.get("tool_name"):
                        raise ToolError(
                            f"Step '{step.get('name')}' is missing tool_name for direct execution",
                            ErrorCode.TOOL_INVALID_INPUT
                        )
                    if not step.get("operation"):
                        raise ToolError(
                            f"Step '{step.get('name')}' is missing operation for direct execution",
                            ErrorCode.TOOL_INVALID_INPUT
                        )
                elif step.get("execution_mode") == "orchestrated" or step.get("task_description"):
                    if not step.get("task_description"):
                        raise ToolError(
                            f"Step '{step.get('name')}' is missing task_description for orchestrated execution",
                            ErrorCode.TOOL_INVALID_INPUT
                        )
            
            update_data["steps"] = steps
            
            # Increment version
            update_data["version"] = chain.version + 1
        
        # Update chain
        chain_dict = chain.to_dict()
        chain_dict.update(update_data)
        
        updated_chain = TaskChain.from_dict(chain_dict)
        
        # Validate the updated chain
        errors = updated_chain.validate()
        if errors:
            raise ToolError(
                f"Invalid chain update: {', '.join(errors)}",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Save to database
        db.update(updated_chain)
        
        logger.info(f"Updated chain: {chain_id}")
        
        return updated_chain.to_dict()
    
    def _delete_chain(
        self,
        chain_id: str,
        confirm: Optional[bool] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Delete a chain.
        
        Args:
            chain_id: ID of the chain to delete
            confirm: Explicit confirmation required (must be True)
            
        Returns:
            Success confirmation
            
        Raises:
            ToolError: If deletion fails or not confirmed
        """
        if not chain_id:
            raise ToolError(
                "chain_id is required",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Require explicit confirmation
        if not confirm:
            raise ToolError(
                "Explicit confirmation is required to delete a chain. Set confirm=true to proceed.",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        db = chain_scheduler_service.get_chain_executor().db
        chain = db.get(TaskChain, chain_id)
        
        if not chain:
            raise ToolError(
                f"Chain {chain_id} not found",
                ErrorCode.TOOL_NOT_FOUND
            )
            
        # Check if chain is running
        if chain_id in chain_scheduler_service.get_chain_executor().active_chains:
            raise ToolError(
                f"Cannot delete chain {chain_id} while it is running. Try pausing it first.",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        # Check if chain has executions
        with db.get_session() as session:
            execution_count = session.query(ChainExecution).filter(
                ChainExecution.chain_id == chain_id
            ).count()
            
        # Delete the chain (and cascading steps)
        db.delete(chain)
        
        logger.info(f"Deleted chain: {chain_id}")
        
        return {
            "success": True,
            "message": f"Chain '{chain.name}' (ID: {chain_id}) deleted successfully",
            "execution_count": execution_count
        }
    
    def _execute_chain_now(
        self,
        chain_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute a chain immediately.
        
        Args:
            chain_id: ID of the chain to execute
            initial_context: Initial context values
            
        Returns:
            Execution confirmation
            
        Raises:
            ToolError: If execution fails
        """
        if not chain_id:
            raise ToolError(
                "chain_id is required",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        # Verify chain exists before attempting execution
        db = chain_scheduler_service.get_chain_executor().db
        chain = db.get(TaskChain, chain_id)
        
        if not chain:
            raise ToolError(
                f"Chain {chain_id} not found",
                ErrorCode.TOOL_NOT_FOUND
            )
            
        # Check if chain is active
        if chain.status != ChainStatus.ACTIVE:
            raise ToolError(
                f"Cannot execute chain {chain_id} with status '{chain.status.value}'. Only active chains can be executed.",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        # Check if chain is already running
        if chain_id in chain_scheduler_service.get_chain_executor().active_chains:
            raise ToolError(
                f"Chain {chain_id} is already running. Only one execution can run at a time.",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
            
        # Execute the chain
        execution = chain_scheduler_service.execute_chain_now(
            chain_id=chain_id,
            initial_context=initial_context or {}
        )
        
        return {
            "success": True,
            "message": f"Chain execution started with ID: {execution.id}",
            "execution_id": execution.id,
            "chain_id": chain_id,
            "chain_name": chain.name,
            "started_at": execution.started_at.isoformat() if execution.started_at else None
        }
    
    def _pause_chain(
        self,
        chain_id: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Pause a chain's scheduled execution.
        
        Args:
            chain_id: ID of the chain to pause
            
        Returns:
            Success confirmation
            
        Raises:
            ToolError: If pause fails
        """
        if not chain_id:
            raise ToolError(
                "chain_id is required",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        db = chain_scheduler_service.get_chain_executor().db
        chain = db.get(TaskChain, chain_id)
        
        if not chain:
            raise ToolError(
                f"Chain {chain_id} not found",
                ErrorCode.TOOL_NOT_FOUND
            )
            
        # Check current status
        if chain.status == ChainStatus.PAUSED:
            return {
                "success": True,
                "message": f"Chain '{chain.name}' is already paused",
                "chain_id": chain_id
            }
            
        # Update status
        chain.status = ChainStatus.PAUSED
        db.update(chain)
        
        logger.info(f"Paused chain: {chain_id}")
        
        return {
            "success": True,
            "message": f"Chain '{chain.name}' paused successfully",
            "chain_id": chain_id
        }
    
    def _resume_chain(
        self,
        chain_id: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Resume a paused chain.
        
        Args:
            chain_id: ID of the chain to resume
            
        Returns:
            Success confirmation
            
        Raises:
            ToolError: If resume fails
        """
        if not chain_id:
            raise ToolError(
                "chain_id is required",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        db = chain_scheduler_service.get_chain_executor().db
        chain = db.get(TaskChain, chain_id)
        
        if not chain:
            raise ToolError(
                f"Chain {chain_id} not found",
                ErrorCode.TOOL_NOT_FOUND
            )
            
        # Check current status
        if chain.status == ChainStatus.ACTIVE:
            return {
                "success": True,
                "message": f"Chain '{chain.name}' is already active",
                "chain_id": chain_id
            }
            
        # Update status
        chain.status = ChainStatus.ACTIVE
        
        # Recalculate next execution time
        scheduler = TaskScheduler()
        now = datetime.now(timezone.utc)
        
        # If next_execution_time is in the past, recalculate
        if not chain.next_execution_time or chain.next_execution_time < now:
            # Create a temporary task to calculate next run time
            temp_task = {
                "frequency": chain.frequency.value,
                "scheduled_time": chain.scheduled_time,
                "day_of_week": chain.day_of_week,
                "day_of_month": chain.day_of_month,
                "custom_schedule": chain.custom_schedule,
                "timezone": chain.timezone
            }
            
            # Use the scheduler's logic to calculate the next run time
            next_run = scheduler._calculate_next_run_time(temp_task)
            
            # If no future runs are scheduled, default to now + 1 minute
            chain.next_execution_time = next_run or (now + timedelta(minutes=1))
            
        db.update(chain)
        
        logger.info(f"Resumed chain: {chain_id}")
        
        return {
            "success": True,
            "message": f"Chain '{chain.name}' resumed successfully. Next execution at {chain.next_execution_time.isoformat()}",
            "chain_id": chain_id,
            "next_execution": chain.next_execution_time.isoformat() if chain.next_execution_time else None
        }
    
    def _get_chain_executions(
        self,
        chain_id: str,
        limit: int = 10,
        offset: int = 0,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get execution history for a chain.
        
        Args:
            chain_id: ID of the chain
            limit: Maximum number of executions to return
            offset: Offset for pagination
            
        Returns:
            List of executions
            
        Raises:
            ToolError: If the chain is not found
        """
        if not chain_id:
            raise ToolError(
                "chain_id is required",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        db = chain_scheduler_service.get_chain_executor().db
        
        # Check if chain exists
        chain = db.get(TaskChain, chain_id)
        if not chain:
            raise ToolError(
                f"Chain {chain_id} not found",
                ErrorCode.TOOL_NOT_FOUND
            )
            
        with db.get_session() as session:
            # Query executions
            query = session.query(ChainExecution).filter(
                ChainExecution.chain_id == chain_id
            )
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination
            query = query.order_by(ChainExecution.started_at.desc())
            query = query.limit(limit).offset(offset)
            
            # Execute query
            executions = query.all()
            
            return {
                "chain_id": chain_id,
                "chain_name": chain.name,
                "executions": [
                    {
                        "id": execution.id,
                        "status": execution.status.value if execution.status else None,
                        "started_at": execution.started_at.isoformat() if execution.started_at else None,
                        "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                        "runtime_seconds": execution.runtime_seconds,
                        "trigger_type": execution.trigger_type.value if execution.trigger_type else None,
                        "error": execution.error
                    }
                    for execution in executions
                ],
                "total": total_count,
                "limit": limit,
                "offset": offset
            }
    
    def _get_execution_details(
        self,
        execution_id: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific execution.
        
        Args:
            execution_id: ID of the execution
            
        Returns:
            Execution details including all step executions
            
        Raises:
            ToolError: If the execution is not found
        """
        if not execution_id:
            raise ToolError(
                "execution_id is required",
                ErrorCode.TOOL_INVALID_INPUT
            )
            
        db = chain_scheduler_service.get_chain_executor().db
        
        # Get execution
        execution = db.get(ChainExecution, execution_id)
        if not execution:
            raise ToolError(
                f"Execution {execution_id} not found",
                ErrorCode.TOOL_NOT_FOUND
            )
            
        # Get chain
        chain = db.get(TaskChain, execution.chain_id)
        if not chain:
            logger.warning(f"Chain {execution.chain_id} not found for execution {execution_id}")
            
        # Get step executions
        with db.get_session() as session:
            step_executions = session.query(StepExecution).filter(
                StepExecution.chain_execution_id == execution_id
            ).order_by(StepExecution.position).all()
            
            # Get details for each step
            step_details = []
            for step_execution in step_executions:
                step = db.get(ChainStep, step_execution.step_id)
                if step:
                    step_details.append({
                        "execution_id": step_execution.id,
                        "step_id": step.id,
                        "name": step.name,
                        "position": step_execution.position,
                        "status": step_execution.status.value if step_execution.status else None,
                        "started_at": step_execution.started_at.isoformat() if step_execution.started_at else None,
                        "completed_at": step_execution.completed_at.isoformat() if step_execution.completed_at else None,
                        "runtime_seconds": step_execution.runtime_seconds,
                        "attempts": step_execution.attempts,
                        "error": step_execution.error,
                        "parameters": step_execution.resolved_parameters,
                        "result": step_execution.result
                    })
                else:
                    logger.warning(f"Step {step_execution.step_id} not found for step execution {step_execution.id}")
            
            return {
                "execution_id": execution.id,
                "chain_id": execution.chain_id,
                "chain_name": chain.name if chain else None,
                "status": execution.status.value if execution.status else None,
                "trigger_type": execution.trigger_type.value if execution.trigger_type else None,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "runtime_seconds": execution.runtime_seconds,
                "error": execution.error,
                "context": execution.execution_context,
                "steps": step_details
            }