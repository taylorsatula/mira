"""
Automation tool for managing automated tasks and sequences.

This tool provides a unified interface for creating and managing both simple
recurring tasks and multi-step sequences with a common scheduling system.
It replaces the separate scheduler_tool and chain_tool with a single,
more intuitive interface.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context
from task_manager.automation import AutomationType, AutomationStatus, TaskFrequency
from task_manager.automation_engine import get_automation_engine
from config import config
from config.registry import registry

# Configure logger
logger = logging.getLogger(__name__)


# Define configuration class for AutomationTool
class AutomationToolConfig(BaseModel):
    """Configuration for the automation_tool."""

    enabled: bool = Field(
        default=True, description="Whether this tool is enabled by default"
    )
    max_automations_per_user: int = Field(
        default=20, description="Maximum number of automations a user can create"
    )
    confirm_before_delete: bool = Field(
        default=True,
        description="Whether to require explicit confirmation before deleting automations",
    )


# Register with registry
registry.register("automation_tool", AutomationToolConfig)


class AutomationTool(Tool):
    """
    Tool for managing automated tasks and sequences.

    This unified tool allows the creation and management of both simple tasks
    and multi-step sequences with a common scheduling system. It includes features
    for conditional execution, data passing between steps, error handling,
    and a variety of scheduling options.
    """

    name = "automation_tool"
    simple_description = """
    Creates and manages automated tasks and sequences that run at scheduled times without user intervention.
    Use this tool when the user wants to automate tasks, create workflows, or set up recurring activities.
    """
    
    implementation_details = """
    
    This tool supports two types of automations:
    
    1. Simple Tasks: Execute a single operation at scheduled times
       - Direct execution: Run a specific tool operation with predetermined parameters
       - Orchestrated execution: Use Claude to interpret a task description and determine which tools to use
       
    2. Sequences: Execute a series of connected steps with data passing between them
       - Each step can execute either directly or through Claude
       - Support for conditional execution, error handling, and alternative paths
       - Parameters can reference results from previous steps using template syntax
    
    The tool supports these operations:
    
    1. create_automation: Create a new task or sequence
       - Required fields: name, type (simple_task or sequence), frequency
       - For simple_task: execution_mode, tool_name/operation OR task_description
       - For sequence: steps (list of step definitions)
       - Optional: scheduled_time, day_of_week, day_of_month, timezone, etc.
    
    2. get_automations: List automations filtered by criteria
       - Optional filters: type, status, frequency, limit, offset
    
    3. get_automation: Get detailed information about a specific automation
       - Required: automation_id
    
    4. update_automation: Modify an existing automation
       - Required: automation_id
       - Optional: Any fields to update
    
    5. delete_automation: Remove an automation
       - Required: automation_id
    
    6. execute_now: Run an automation immediately
       - Required: automation_id
       - Optional: initial_context (for sequences)
    
    7. pause_automation: Temporarily pause scheduled execution
       - Required: automation_id
    
    8. resume_automation: Resume a paused automation
       - Required: automation_id
    
    9. get_executions: View execution history for an automation
       - Required: automation_id
       - Optional: limit, offset
    
    10. get_execution_details: Get detailed information about a specific execution
        - Required: execution_id
    
    This tool requires parameters to be passed as a JSON string in the "kwargs" field when using the direct APIs.
    When interacting through Claude, you can use natural language to describe the automation you want to create.
    
    Examples:
    - "Create a daily reminder to check inventory at 9am"
    - "Set up a weekly chain that generates a sales report and emails it to the team every Monday"
    - "Make a sequence that processes customer data and sends personalized messages"
    """
    
    description = simple_description + implementation_details

    usage_examples = [
        {
            "input": {
                "operation": "create_automation",
                "kwargs": json.dumps(
                    {
                        "name": "Daily Inventory Check",
                        "type": "simple_task",
                        "frequency": "daily",
                        "scheduled_time": "09:00:00",
                        "execution_mode": "direct",
                        "tool_name": "reminder_tool",
                        "operation": "add_reminder",
                        "parameters": {
                            "title": "Check inventory levels",
                            "date": "today",
                            "description": "Review stock levels for popular items",
                        },
                    }
                ),
            },
            "output": {
                "automation": {
                    "id": "auto_12345",
                    "name": "Daily Inventory Check",
                    "type": "simple_task",
                    "frequency": "daily",
                    "scheduled_time": "2024-04-20T09:00:00+00:00",
                    "next_execution_time": "2024-04-20T09:00:00+00:00",
                    "status": "active",
                },
                "message": "Automation 'Daily Inventory Check' scheduled to run daily at 09:00:00",
            },
        },
        {
            "input": {
                "operation": "create_automation",
                "kwargs": json.dumps(
                    {
                        "name": "Weekly Sales Report",
                        "type": "sequence",
                        "frequency": "weekly",
                        "day_of_week": 1,  # Tuesday
                        "scheduled_time": "07:00:00",
                        "steps": [
                            {
                                "name": "Generate Report",
                                "position": 1,
                                "execution_mode": "orchestrated",
                                "task_description": "Generate a sales report for the previous week",
                                "output_key": "report",
                            },
                            {
                                "name": "Send Email",
                                "position": 2,
                                "execution_mode": "direct",
                                "tool_name": "email_tool",
                                "operation": "send_email",
                                "parameters": {
                                    "to": "sales@example.com",
                                    "subject": "Weekly Sales Report",
                                    "body": "{report.summary}",
                                },
                                "output_key": "email_result",
                            },
                        ],
                    }
                ),
            },
            "output": {
                "automation": {
                    "id": "auto_67890",
                    "name": "Weekly Sales Report",
                    "type": "sequence",
                    "frequency": "weekly",
                    "day_of_week": 1,
                    "scheduled_time": "2024-04-23T07:00:00+00:00",
                    "next_execution_time": "2024-04-23T07:00:00+00:00",
                    "status": "active",
                },
                "message": "Automation 'Weekly Sales Report' scheduled to run weekly on Tuesday at 07:00:00",
            },
        },
        {
            "input": {
                "operation": "get_automations",
                "kwargs": json.dumps({"status": "active"}),
            },
            "output": {
                "automations": [
                    {
                        "id": "auto_12345",
                        "name": "Daily Inventory Check",
                        "type": "simple_task",
                        "frequency": "daily",
                        "next_execution_time": "2024-04-20T09:00:00+00:00",
                        "status": "active",
                    },
                    {
                        "id": "auto_67890",
                        "name": "Weekly Sales Report",
                        "type": "sequence",
                        "frequency": "weekly",
                        "next_execution_time": "2024-04-23T07:00:00+00:00",
                        "status": "active",
                    },
                ],
                "count": 2,
                "total": 2,
                "message": "Found 2 active automations",
            },
        },
    ]

    def __init__(self):
        """Initialize the automation tool."""
        super().__init__()

        # Get or initialize the automation engine
        self.engine = get_automation_engine()

        if not self.engine:
            logger.warning("Automation engine is not available")

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an automation tool operation.

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
            logger=logger,
        ):
            # Check if automation engine is available
            if not self.engine:
                raise ToolError(
                    "Automation engine is not available", ErrorCode.TOOL_UNAVAILABLE
                )

            # Parse kwargs JSON string if provided that way
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    params = json.loads(kwargs["kwargs"])
                    kwargs = params
                except json.JSONDecodeError as e:
                    raise ToolError(
                        f"Invalid JSON in kwargs: {e}", ErrorCode.TOOL_INVALID_INPUT
                    )

            # Route to the appropriate operation
            if operation == "create_automation":
                return self._create_automation(**kwargs)
            elif operation == "get_automations":
                return self._get_automations(**kwargs)
            elif operation == "get_automation":
                return self._get_automation(**kwargs)
            elif operation == "update_automation":
                return self._update_automation(**kwargs)
            elif operation == "delete_automation":
                return self._delete_automation(**kwargs)
            elif operation == "execute_now":
                return self._execute_now(**kwargs)
            elif operation == "pause_automation":
                return self._pause_automation(**kwargs)
            elif operation == "resume_automation":
                return self._resume_automation(**kwargs)
            elif operation == "get_executions":
                return self._get_executions(**kwargs)
            elif operation == "get_execution_details":
                return self._get_execution_details(**kwargs)
            else:
                raise ToolError(
                    f"Unknown operation: {operation}. Valid operations are: "
                    "create_automation, get_automations, get_automation, "
                    "update_automation, delete_automation, execute_now, "
                    "pause_automation, resume_automation, get_executions, "
                    "get_execution_details",
                    ErrorCode.TOOL_INVALID_INPUT,
                )

    def _create_automation(self, **kwargs) -> Dict[str, Any]:
        """
        Create a new automation (simple task or sequence).

        Args:
            **kwargs: Automation parameters including:
                name: Name of the automation
                type: Type of automation (simple_task or sequence)
                frequency: Execution frequency
                scheduled_time: When to execute
                various type-specific parameters

        Returns:
            Dict containing the created automation

        Raises:
            ToolError: If creation fails
        """
        logger.info(f"Creating automation: {kwargs.get('name')}")

        # Validate required parameters
        if not kwargs.get("name"):
            raise ToolError("Automation name is required", ErrorCode.TOOL_INVALID_INPUT)

        # Check automation type
        automation_type = kwargs.get("type")
        if not automation_type:
            raise ToolError(
                "Automation type is required (simple_task or sequence)",
                ErrorCode.TOOL_INVALID_INPUT,
            )

        try:
            auto_type = AutomationType(automation_type)
        except ValueError:
            raise ToolError(
                f"Invalid automation type: {automation_type}. Must be 'simple_task' or 'sequence'",
                ErrorCode.TOOL_INVALID_INPUT,
            )

        # Type-specific validation
        if auto_type == AutomationType.SIMPLE_TASK:
            execution_mode = kwargs.get("execution_mode")

            if not execution_mode:
                # Infer execution mode from provided parameters
                if kwargs.get("task_description"):
                    execution_mode = "orchestrated"
                    kwargs["execution_mode"] = "orchestrated"
                else:
                    execution_mode = "direct"
                    kwargs["execution_mode"] = "direct"

            if execution_mode == "direct":
                if not kwargs.get("tool_name"):
                    raise ToolError(
                        "tool_name is required for direct execution",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
                if not kwargs.get("operation"):
                    raise ToolError(
                        "operation is required for direct execution",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
            elif execution_mode == "orchestrated":
                if not kwargs.get("task_description"):
                    raise ToolError(
                        "task_description is required for orchestrated execution",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )
            else:
                raise ToolError(
                    f"Invalid execution_mode: {execution_mode}. Must be 'direct' or 'orchestrated'",
                    ErrorCode.TOOL_INVALID_INPUT,
                )

        elif auto_type == AutomationType.SEQUENCE:
            steps = kwargs.get("steps")
            if not steps or not isinstance(steps, list) or len(steps) == 0:
                raise ToolError(
                    "At least one step is required for sequence type",
                    ErrorCode.TOOL_INVALID_INPUT,
                )

            # Set position for steps if not provided
            for i, step in enumerate(steps):
                if "position" not in step:
                    step["position"] = i + 1

                # Validate step fields
                if not step.get("name"):
                    raise ToolError(
                        f"Step {i+1} is missing a name", ErrorCode.TOOL_INVALID_INPUT
                    )

                if not step.get("output_key"):
                    raise ToolError(
                        f"Step '{step.get('name', f'at position {i+1}')}' is missing an output_key",
                        ErrorCode.TOOL_INVALID_INPUT,
                    )

                # Check or infer execution mode
                step_exec_mode = step.get("execution_mode")
                if not step_exec_mode:
                    if step.get("task_description"):
                        step["execution_mode"] = "orchestrated"
                    else:
                        step["execution_mode"] = "direct"

                # Validate based on execution mode
                if step.get("execution_mode") == "direct":
                    if not step.get("tool_name"):
                        raise ToolError(
                            f"Step '{step.get('name')}' is missing tool_name for direct execution",
                            ErrorCode.TOOL_INVALID_INPUT,
                        )
                    if not step.get("operation"):
                        raise ToolError(
                            f"Step '{step.get('name')}' is missing operation for direct execution",
                            ErrorCode.TOOL_INVALID_INPUT,
                        )
                elif step.get("execution_mode") == "orchestrated":
                    if not step.get("task_description"):
                        raise ToolError(
                            f"Step '{step.get('name')}' is missing task_description for orchestrated execution",
                            ErrorCode.TOOL_INVALID_INPUT,
                        )

        # Create the automation
        try:
            automation = self.engine.create_automation(kwargs)

            # Import timezone utilities
            from utils.timezone_utils import convert_to_timezone, format_datetime

            # Generate a human-readable message with proper timezone
            frequency = automation.frequency.value
            
            # Convert time to user's timezone for display
            user_tz = automation.timezone
            time_display = format_datetime(automation.scheduled_time, 'short', user_tz)

            if frequency == "once":
                date_display = format_datetime(automation.scheduled_time, 'date', user_tz)
                message = f"Automation '{automation.name}' scheduled to run once on {date_display} at {time_display}"
            elif frequency == "daily":
                message = f"Automation '{automation.name}' scheduled to run daily at {time_display}"
            elif frequency == "weekly":
                day_names = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                day_name = (
                    day_names[automation.day_of_week]
                    if automation.day_of_week is not None
                    else "unknown day"
                )
                message = f"Automation '{automation.name}' scheduled to run weekly on {day_name} at {time_display}"
            elif frequency == "monthly":
                day = automation.day_of_month or automation.scheduled_time.day
                message = f"Automation '{automation.name}' scheduled to run monthly on day {day} at {time_display}"
            else:
                message = f"Automation '{automation.name}' scheduled with {frequency} frequency at {time_display}"

            return {"automation": automation.to_dict(), "message": message}

        except Exception as e:
            raise ToolError(
                f"Failed to create automation: {str(e)}", ErrorCode.TOOL_EXECUTION_ERROR
            )

    def _get_automations(
        self,
        type: Optional[str] = None,
        status: Optional[str] = None,
        frequency: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get automations filtered by criteria.

        Args:
            type: Filter by automation type
            status: Filter by automation status
            frequency: Filter by execution frequency
            limit: Maximum number of automations to return
            offset: Offset for pagination

        Returns:
            Dict containing list of automations

        Raises:
            ToolError: If parameters are invalid
        """
        logger.info(
            f"Getting automations with filters: type={type}, status={status}, frequency={frequency}"
        )

        # Validate parameters
        if status and status not in [s.value for s in AutomationStatus]:
            raise ToolError(
                f"Invalid status: {status}. Valid values are: "
                + ", ".join([s.value for s in AutomationStatus]),
                ErrorCode.TOOL_INVALID_INPUT,
            )

        if frequency and frequency not in [f.value for f in TaskFrequency]:
            raise ToolError(
                f"Invalid frequency: {frequency}. Valid values are: "
                + ", ".join([f.value for f in TaskFrequency]),
                ErrorCode.TOOL_INVALID_INPUT,
            )

        if type and type not in [t.value for t in AutomationType]:
            raise ToolError(
                f"Invalid type: {type}. Valid values are: "
                + ", ".join([t.value for t in AutomationType]),
                ErrorCode.TOOL_INVALID_INPUT,
            )

        # Get automations
        result = self.engine.get_automations(
            automation_type=type,
            status=status,
            frequency=frequency,
            limit=limit,
            offset=offset,
            user_id=kwargs.get("user_id"),
        )

        # Create descriptive message
        type_str = f" of type '{type}'" if type else ""
        status_str = f" with status '{status}'" if status else ""
        frequency_str = f" with {frequency} frequency" if frequency else ""

        message = f"Found {result['count']} automation(s){type_str}{status_str}{frequency_str}"

        result["message"] = message
        return result

    def _get_automation(self, automation_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get details of a specific automation.

        Args:
            automation_id: ID of the automation

        Returns:
            Dict containing automation details

        Raises:
            ToolError: If automation not found
        """
        logger.info(f"Getting automation: {automation_id}")

        automation = self.engine.get_automation(automation_id)
        if not automation:
            raise ToolError(
                f"Automation with ID '{automation_id}' not found",
                ErrorCode.TOOL_NOT_FOUND,
            )

        return {
            "automation": automation.to_dict(),
            "message": f"Retrieved automation '{automation.name}'",
        }

    def _update_automation(self, automation_id: str, **kwargs) -> Dict[str, Any]:
        """
        Update an existing automation.

        Args:
            automation_id: ID of the automation to update
            **kwargs: Fields to update

        Returns:
            Dict containing the updated automation

        Raises:
            ToolError: If automation not found or update invalid
        """
        logger.info(f"Updating automation: {automation_id}")

        # Update the automation
        updated_automation = self.engine.update_automation(automation_id, kwargs)
        if not updated_automation:
            raise ToolError(
                f"Automation with ID '{automation_id}' not found",
                ErrorCode.TOOL_NOT_FOUND,
            )

        return {
            "automation": updated_automation.to_dict(),
            "message": f"Automation '{updated_automation.name}' updated successfully",
        }

    def _delete_automation(
        self, automation_id: str, confirm: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """
        Delete an automation.

        Args:
            automation_id: ID of the automation to delete
            confirm: Confirmation flag (may be required based on config)

        Returns:
            Dict containing deletion confirmation

        Raises:
            ToolError: If automation not found or deletion not confirmed
        """
        logger.info(f"Deleting automation: {automation_id}")

        # Check if confirmation is required
        confirm_required = config.automation_tool.confirm_before_delete
        if confirm_required and not confirm:
            raise ToolError(
                "Explicit confirmation is required to delete an automation. Set confirm=true to proceed.",
                ErrorCode.TOOL_INVALID_INPUT,
            )

        # Get automation name first for the message
        automation = self.engine.get_automation(automation_id)
        if not automation:
            raise ToolError(
                f"Automation with ID '{automation_id}' not found",
                ErrorCode.TOOL_NOT_FOUND,
            )

        automation_name = automation.name

        # Delete the automation
        success = self.engine.delete_automation(automation_id)
        if not success:
            raise ToolError(
                f"Failed to delete automation with ID '{automation_id}'",
                ErrorCode.TOOL_EXECUTION_ERROR,
            )

        return {
            "automation_id": automation_id,
            "message": f"Automation '{automation_name}' deleted successfully",
        }

    def _execute_now(
        self,
        automation_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute an automation immediately.

        Args:
            automation_id: ID of the automation to execute
            initial_context: Initial context data for sequences

        Returns:
            Dict containing execution confirmation

        Raises:
            ToolError: If automation not found or execution fails
        """
        logger.info(f"Executing automation now: {automation_id}")

        try:
            execution = self.engine.execute_now(
                automation_id=automation_id, initial_context=initial_context or {}
            )

            # Get automation name
            automation = self.engine.get_automation(automation_id)
            automation_name = automation.name if automation else "Unknown"

            return {
                "success": True,
                "message": f"Automation '{automation_name}' submitted for immediate execution",
                "execution_id": execution.id,
                "automation_id": automation_id,
            }
        except Exception as e:
            raise ToolError(
                f"Failed to execute automation: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR,
            )

    def _pause_automation(self, automation_id: str, **kwargs) -> Dict[str, Any]:
        """
        Pause an automation's scheduled execution.

        Args:
            automation_id: ID of the automation to pause

        Returns:
            Dict containing pause confirmation

        Raises:
            ToolError: If automation not found or pause fails
        """
        logger.info(f"Pausing automation: {automation_id}")

        # Pause the automation
        success = self.engine.pause_automation(automation_id)
        if not success:
            raise ToolError(
                f"Failed to pause automation with ID '{automation_id}'",
                ErrorCode.TOOL_EXECUTION_ERROR,
            )

        # Get automation name
        automation = self.engine.get_automation(automation_id)
        automation_name = automation.name if automation else "Unknown"

        return {
            "automation_id": automation_id,
            "message": f"Automation '{automation_name}' paused successfully",
        }

    def _resume_automation(self, automation_id: str, **kwargs) -> Dict[str, Any]:
        """
        Resume a paused automation.

        Args:
            automation_id: ID of the automation to resume

        Returns:
            Dict containing resume confirmation

        Raises:
            ToolError: If automation not found or resume fails
        """
        logger.info(f"Resuming automation: {automation_id}")

        # Resume the automation
        result = self.engine.resume_automation(automation_id)

        return {
            "automation_id": automation_id,
            "message": result.get("message", "Automation resumed successfully"),
            "next_execution": result.get("next_execution"),
        }

    def _get_executions(
        self, automation_id: str, limit: int = 10, offset: int = 0, **kwargs
    ) -> Dict[str, Any]:
        """
        Get execution history for an automation.

        Args:
            automation_id: ID of the automation
            limit: Maximum number of executions to return
            offset: Offset for pagination

        Returns:
            Dict containing list of executions

        Raises:
            ToolError: If automation not found
        """
        logger.info(f"Getting executions for automation: {automation_id}")

        # Get executions
        result = self.engine.get_executions(
            automation_id=automation_id, limit=limit, offset=offset
        )

        # Get automation name
        automation = self.engine.get_automation(automation_id)
        automation_name = automation.name if automation else "Unknown"

        result["automation_name"] = automation_name
        result["message"] = (
            f"Found {result['count']} execution(s) for automation '{automation_name}'"
        )

        return result

    def _get_execution_details(self, execution_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get detailed information about a specific execution.

        Args:
            execution_id: ID of the execution

        Returns:
            Dict containing execution details

        Raises:
            ToolError: If execution not found
        """
        logger.info(f"Getting execution details: {execution_id}")

        # Get execution details
        execution = self.engine.get_execution_details(execution_id)
        if not execution:
            raise ToolError(
                f"Execution with ID '{execution_id}' not found",
                ErrorCode.TOOL_NOT_FOUND,
            )

        # Get automation name
        automation = self.engine.get_automation(execution.automation_id)
        automation_name = automation.name if automation else "Unknown"

        return {
            "execution": execution.to_dict(),
            "automation_name": automation_name,
            "message": f"Retrieved execution details for '{automation_name}'",
        }
