"""
Automation engine for executing automations.

This module provides the AutomationEngine class which is responsible for
creating, scheduling, and executing automations, including both simple tasks
and multi-step sequences.
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta, UTC
from typing import Dict, Any, Optional, List, Set, Union, cast

import pytz
from dateutil.relativedelta import relativedelta

from db import Database
from errors import ToolError, ErrorCode, error_context
from task_manager.automation import (
    Automation, AutomationType, AutomationStatus, ExecutionMode,
    TaskFrequency, ErrorPolicy, TriggerType, ExecutionStatus,
    AutomationExecution, StepExecution, AutomationStep, 
    StepExecutionStatus, ConditionType
)
from tools.repo import ToolRepository
from api.llm_bridge import LLMBridge
from config import config
from utils.timezone_utils import (
    utc_now, ensure_utc, convert_to_timezone, convert_from_utc,
    parse_time_string, validate_timezone
)

# Configure logger
logger = logging.getLogger(__name__)

# Global engine instance
_automation_engine = None
_engine_lock = threading.Lock()


class TemplateEngine:
    """
    Template substitution engine for parameter passing between steps.
    
    Handles the substitution of placeholders in parameters with values
    from the execution context.
    """
    
    def __init__(self):
        """Initialize the template engine."""
        self.logger = logging.getLogger(__name__ + ".TemplateEngine")
    
    def resolve_template(self, template: Any, context: Dict[str, Any]) -> Any:
        """
        Resolve templates in a value using the provided context.
        
        Args:
            template: Template value which may contain placeholders
            context: Dictionary of values to substitute
            
        Returns:
            Template with placeholders substituted with actual values
        """
        if isinstance(template, str):
            return self._resolve_string_template(template, context)
        elif isinstance(template, dict):
            return self._resolve_dict_template(template, context)
        elif isinstance(template, list):
            return self._resolve_list_template(template, context)
        else:
            # Return primitive values unchanged
            return template
    
    def _resolve_string_template(self, template: str, context: Dict[str, Any]) -> Any:
        """
        Resolve a string template using the provided context.
        
        Args:
            template: String template which may contain placeholders
            context: Dictionary of values to substitute
            
        Returns:
            String with placeholders substituted or original value
        """
        # Check if this string is a template
        if not template.startswith("{") or not template.endswith("}"):
            # Not a direct reference template, check if it contains placeholders
            if "{" in template and "}" in template:
                # Try string formatting
                try:
                    return template.format(**context)
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Error formatting template: {e}")
                    return template
            else:
                # No placeholders, return as is
                return template
        
        # This is a direct reference template like "{step1.result}"
        reference = template[1:-1]  # Remove { and }
        parts = reference.split(".")
        
        if len(parts) < 2:
            self.logger.warning(f"Invalid template reference: {template}")
            return template
        
        # Get the root object
        root_key = parts[0]
        if root_key not in context:
            self.logger.warning(f"Template root not found in context: {root_key}")
            return template
        
        # Navigate the object hierarchy
        value = context[root_key]
        
        for part in parts[1:]:
            # Handle array indices
            if part.isdigit() and isinstance(value, list):
                index = int(part)
                if 0 <= index < len(value):
                    value = value[index]
                else:
                    self.logger.warning(f"Array index out of bounds: {part}")
                    return template
            # Handle object properties
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                self.logger.warning(f"Template part not found: {part}")
                return template
        
        return value
    
    def _resolve_dict_template(self, template: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve templates in a dictionary using the provided context.
        
        Args:
            template: Dictionary with values that may contain placeholders
            context: Dictionary of values to substitute
            
        Returns:
            Dictionary with placeholders substituted
        """
        result = {}
        
        for key, value in template.items():
            # Resolve the key if it's a template
            resolved_key = key
            if isinstance(key, str):
                resolved_key = self._resolve_string_template(key, context)
            
            # Resolve the value recursively
            resolved_value = self.resolve_template(value, context)
            
            # Add to result
            result[resolved_key] = resolved_value
        
        return result
    
    def _resolve_list_template(self, template: List[Any], context: Dict[str, Any]) -> List[Any]:
        """
        Resolve templates in a list using the provided context.
        
        Args:
            template: List with values that may contain placeholders
            context: Dictionary of values to substitute
            
        Returns:
            List with placeholders substituted
        """
        return [self.resolve_template(item, context) for item in template]
    
    def evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Evaluate a condition against the provided context.
        
        Args:
            condition: Condition to evaluate, containing field, operator, and value
            context: Dictionary of values to check against
            
        Returns:
            True if the condition is met, False otherwise
        """
        if "field" not in condition or "operator" not in condition:
            self.logger.warning("Invalid condition format: missing field or operator")
            return False
        
        field = condition["field"]
        operator = condition["operator"]
        expected_value = condition.get("value")
        
        # Resolve field to get actual value
        actual_value = self.resolve_template(f"{{{field}}}", context)
        
        # Evaluate based on operator
        if operator == "eq":
            return actual_value == expected_value
        elif operator == "neq":
            return actual_value != expected_value
        elif operator == "gt":
            return actual_value > expected_value
        elif operator == "gte":
            return actual_value >= expected_value
        elif operator == "lt":
            return actual_value < expected_value
        elif operator == "lte":
            return actual_value <= expected_value
        elif operator == "in":
            return actual_value in expected_value
        elif operator == "nin":
            return actual_value not in expected_value
        elif operator == "contains":
            return expected_value in actual_value
        elif operator == "exists":
            return actual_value is not None
        elif operator == "missing":
            return actual_value is None
        else:
            self.logger.warning(f"Unknown operator: {operator}")
            return False


class AutomationEngine:
    """
    Engine for managing and executing automations.
    
    This class is responsible for creating, scheduling, and executing
    automations, including both simple tasks and multi-step sequences.
    """
    
    def __init__(
        self, 
        tool_repo: Optional[ToolRepository] = None,
        llm_bridge: Optional[LLMBridge] = None
    ):
        """
        Initialize the automation engine.
        
        Args:
            tool_repo: Repository of available tools
            llm_bridge: LLM bridge for orchestrated steps
        """
        self.db = Database()
        self.tool_repo = tool_repo or ToolRepository()
        self.llm_bridge = llm_bridge or LLMBridge()
        self.template_engine = TemplateEngine()
        
        # Keep track of which automations are currently being executed
        self.active_automations = {}  # automation_id -> execution_id
        self.automation_lock = threading.Lock()
        
        # Flag to indicate if the scheduler is running
        self.scheduler_running = False
        
        # Thread for the scheduler
        self.scheduler_thread = None
        
        # Configuration
        self.check_interval = config.get("automation", {}).get("check_interval", 30)
        self.max_concurrent_automations = config.get("automation", {}).get("max_concurrent_automations", 5)
        
        logger.info("Automation engine initialized")
    
    def start_scheduler(self) -> None:
        """
        Start the automation scheduler.
        
        This method begins the background thread that periodically checks for
        and executes scheduled automations.
        """
        if self.scheduler_running:
            logger.warning("Automation scheduler is already running")
            return
        
        self.scheduler_running = True
        
        # Start the scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="AutomationSchedulerThread"
        )
        self.scheduler_thread.start()
        
        logger.info(f"Automation scheduler started with check interval of {self.check_interval} seconds")
    
    def stop_scheduler(self) -> None:
        """
        Stop the automation scheduler.
        
        This method stops the scheduler thread and waits for any executing
        automations to complete.
        """
        if not self.scheduler_running:
            logger.warning("Automation scheduler is not running")
            return
        
        logger.info("Stopping automation scheduler...")
        self.scheduler_running = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        logger.info("Automation scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """
        Main scheduler loop that periodically checks for and executes due automations.
        """
        while self.scheduler_running:
            try:
                # Find and execute due automations
                self._process_due_automations()
                
                # Sleep until next check
                time.sleep(self.check_interval)
            
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                # Continue running even if an error occurs
                time.sleep(60)  # Wait a bit longer after an error
    
    def _process_due_automations(self) -> None:
        """
        Find and execute automations that are due to run.
        """
        now = utc_now()
        
        with self.db.get_session() as session:
            # Find automations that are due to run
            due_automations = session.query(Automation).filter(
                Automation.status == AutomationStatus.ACTIVE,
                Automation.next_execution_time <= now
            ).all()
            
            if due_automations:
                logger.info(f"Found {len(due_automations)} automations due for execution")
                
                for automation in due_automations:
                    # Skip if already running
                    if automation.id in self.active_automations:
                        logger.debug(f"Automation {automation.id} is already running, skipping")
                        continue
                    
                    # Check if we've reached the maximum concurrent automations
                    with self.automation_lock:
                        if len(self.active_automations) >= self.max_concurrent_automations:
                            logger.warning("Maximum concurrent automations reached, deferring execution")
                            break
                    
                    # Submit for execution
                    threading.Thread(
                        target=self._execute_automation_safely,
                        args=(automation.id,),
                        daemon=True,
                        name=f"Automation-{automation.id[:8]}"
                    ).start()
                    
                    # Prevent overloading by staggering executions
                    time.sleep(1)
    
    def _execute_automation_safely(self, automation_id: str) -> None:
        """
        Execute an automation safely in a separate thread.
        
        Args:
            automation_id: ID of the automation to execute
        """
        try:
            # Check if automation is already running
            if automation_id in self.active_automations:
                logger.warning(f"Automation {automation_id} is already running, skipping")
                return
            
            # Get the automation with full details
            automation = self.db.get(Automation, automation_id)
            if not automation:
                logger.error(f"Automation {automation_id} not found")
                return
            
            # Execute the automation
            self.execute_automation(automation, TriggerType.SCHEDULED)
            
        except Exception as e:
            logger.error(f"Error executing automation {automation_id}: {e}", exc_info=True)
    
    def execute_automation(self, automation: Automation, trigger_type: TriggerType) -> None:
        """
        Execute an automation.
        
        Args:
            automation: The automation to execute
            trigger_type: What triggered this execution
        """
        # Create execution record
        execution = AutomationExecution(
            id=str(uuid.uuid4()),
            automation_id=automation.id,
            status=ExecutionStatus.RUNNING,
            trigger_type=trigger_type,
            scheduled_time=automation.next_execution_time,
            started_at=utc_now()
        )
        
        # Save execution record
        self.db.add(execution)
        
        # Update automation with execution info
        automation.last_execution_id = execution.id
        automation.last_execution_time = execution.started_at
        automation.execution_count += 1
        
        # Calculate next execution time
        next_run = self._calculate_next_run_time(automation)
        if next_run:
            automation.next_execution_time = next_run
        elif automation.frequency == TaskFrequency.ONCE:
            # One-time automations should be marked as completed after execution
            automation.status = AutomationStatus.COMPLETED
        
        self.db.update(automation)
        
        # Mark automation as active
        with self.automation_lock:
            self.active_automations[automation.id] = execution.id
        
        try:
            # Execute based on type
            if automation.type == AutomationType.SIMPLE_TASK:
                self._execute_simple_task(automation, execution)
            else:  # sequence
                self._execute_sequence(automation, execution)
            
            # Mark execution as completed
            execution.status = ExecutionStatus.COMPLETED
            execution.completed_at = utc_now()
            execution.runtime_seconds = (execution.completed_at - execution.started_at).total_seconds()
            self.db.update(execution)
            
            logger.info(f"Automation {automation.id} executed successfully")
            
        except Exception as e:
            # Log the error
            error_message = str(e)
            logger.error(f"Error executing automation {automation.id}: {error_message}", exc_info=True)
            
            # Update execution with error
            execution.status = ExecutionStatus.FAILED
            execution.completed_at = utc_now()
            
            # Ensure both datetimes have timezone info before calculating runtime
            if execution.started_at and execution.started_at.tzinfo is None:
                started_at = ensure_utc(execution.started_at)
            else:
                started_at = execution.started_at
                
            execution.runtime_seconds = (execution.completed_at - started_at).total_seconds()
            execution.error = error_message
            self.db.update(execution)
            
        finally:
            # Remove automation from active automations
            with self.automation_lock:
                self.active_automations.pop(automation.id, None)
    
    def _execute_simple_task(self, automation: Automation, execution: AutomationExecution) -> None:
        """
        Execute a simple task automation.
        
        Args:
            automation: The automation to execute
            execution: The execution record
            
        Raises:
            ToolError: If the task execution fails
        """
        with error_context(
            component_name="AutomationEngine",
            operation="execute_simple_task",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=logger
        ):
            # Execute based on the execution mode
            if automation.execution_mode == ExecutionMode.DIRECT:
                result = self._execute_direct_task(automation)
            else:  # orchestrated
                result = self._execute_orchestrated_task(automation)
            
            # Update execution with result
            execution.result = result
            self.db.update(execution)
    
    def _execute_sequence(self, automation: Automation, execution: AutomationExecution) -> None:
        """
        Execute a sequence automation.
        
        Args:
            automation: The automation to execute
            execution: The execution record
            
        Raises:
            ToolError: If the sequence execution fails
        """
        with error_context(
            component_name="AutomationEngine",
            operation="execute_sequence",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=logger
        ):
            # Make sure steps are loaded
            with self.db.get_session() as session:
                from sqlalchemy.orm import joinedload
                automation = session.query(Automation).options(
                    joinedload(Automation.steps)
                ).filter(Automation.id == automation.id).first()
                
                if not automation:
                    raise ToolError(
                        f"Automation {execution.automation_id} not found during execution",
                        ErrorCode.TOOL_NOT_FOUND
                    )
                
                # Create a dictionary mapping step IDs to steps for quick lookups
                step_map = {step.id: step for step in automation.steps}
                
                # Ensure steps are sorted by position
                steps = sorted(automation.steps, key=lambda s: s.position)
            
            if not steps:
                logger.warning(f"Automation {automation.id} has no steps")
                return
            
            # Execute steps
            step_index = 0
            current_step = steps[0]
            
            # Initialize execution context
            execution.execution_context = {}
            self.db.update(execution)
            
            # Execute steps until done
            while current_step and step_index < len(steps):
                try:
                    # Execute the step
                    next_step_id = self._execute_step(
                        step=current_step,
                        automation=automation,
                        execution=execution
                    )
                    
                    # Find the next step
                    if next_step_id:
                        # Find step by ID using the map instead of linear search
                        next_step = step_map.get(next_step_id)
                        if not next_step:
                            raise ToolError(
                                f"Next step {next_step_id} not found in automation {automation.id}",
                                ErrorCode.TOOL_EXECUTION_ERROR
                            )
                        current_step = next_step
                    else:
                        # Move to next step by position
                        step_index += 1
                        if step_index < len(steps):
                            current_step = steps[step_index]
                        else:
                            current_step = None
                
                except Exception as e:
                    error_policy = current_step.error_policy or automation.error_policy or ErrorPolicy.STOP
                    
                    if error_policy == ErrorPolicy.STOP:
                        # Stop sequence execution
                        raise ToolError(
                            f"Automation {automation.id} execution stopped due to error in step '{current_step.name}': {str(e)}",
                            ErrorCode.TOOL_EXECUTION_ERROR
                        ) from e
                    
                    elif error_policy == ErrorPolicy.CONTINUE:
                        # Log error and continue with next step
                        logger.error(
                            f"Error in step '{current_step.name}' (automation {automation.id}): {str(e)}. "
                            f"Continuing with next step as per error policy.",
                            exc_info=True
                        )
                        
                        # Move to next step
                        step_index += 1
                        if step_index < len(steps):
                            current_step = steps[step_index]
                        else:
                            current_step = None
                    
                    elif error_policy == ErrorPolicy.RETRY:
                        # Retry is handled at the step level, but we still need to decide
                        # whether to continue or stop after maximum retries
                        # For now, continue to next step after max retries
                        logger.error(
                            f"Error in step '{current_step.name}' (automation {automation.id}) after max retries: {str(e)}. "
                            f"Continuing with next step.",
                            exc_info=True
                        )
                        
                        # Move to next step
                        step_index += 1
                        if step_index < len(steps):
                            current_step = steps[step_index]
                        else:
                            current_step = None
                    
                    elif error_policy == ErrorPolicy.ALTERNATIVE:
                        # Use the on_failure_step_id if specified
                        if current_step.on_failure_step_id:
                            next_step = step_map.get(current_step.on_failure_step_id)
                            if next_step:
                                current_step = next_step
                            else:
                                logger.error(
                                    f"Alternative step {current_step.on_failure_step_id} not found for step '{current_step.name}'",
                                    exc_info=True
                                )
                                # Move to next step
                                step_index += 1
                                if step_index < len(steps):
                                    current_step = steps[step_index]
                                else:
                                    current_step = None
                        else:
                            # No alternative specified, move to next step
                            step_index += 1
                            if step_index < len(steps):
                                current_step = steps[step_index]
                            else:
                                current_step = None
                    
                    elif error_policy == ErrorPolicy.SILENT:
                        # End execution quietly
                        logger.info(
                            f"Ending automation {automation.id} silently due to error in step '{current_step.name}'"
                        )
                        break
                    
                    else:
                        # Unknown error policy, default to stop
                        raise ToolError(
                            f"Unknown error policy {error_policy} for step '{current_step.name}'",
                            ErrorCode.TOOL_EXECUTION_ERROR
                        ) from e
    
    def _execute_step(
        self, 
        step: AutomationStep,
        automation: Automation,
        execution: AutomationExecution
    ) -> Optional[str]:
        """
        Execute a single step of an automation.
        
        Args:
            step: The step to execute
            automation: The parent automation
            execution: The automation execution record
            
        Returns:
            ID of the next step to execute, or None for linear progression
            
        Raises:
            ToolError: If step execution fails
        """
        # Create step execution record
        step_execution = StepExecution(
            id=str(uuid.uuid4()),
            execution_id=execution.id,
            step_id=step.id,
            position=step.position,
            status=StepExecutionStatus.RUNNING,
            started_at=utc_now()
        )
        
        # Check if this step should be skipped based on its condition
        should_execute = True
        if step.condition_type != ConditionType.ALWAYS:
            if step.condition_type == ConditionType.IF_SUCCESS:
                # Check if the previous step succeeded
                with self.db.get_session() as session:
                    prev_step = session.query(StepExecution).filter(
                        StepExecution.execution_id == execution.id,
                        StepExecution.position == step.position - 1
                    ).first()
                    
                    if prev_step and prev_step.status != StepExecutionStatus.COMPLETED:
                        should_execute = False
                        step_execution.condition_result = False
            
            elif step.condition_type == ConditionType.IF_FAILURE:
                # Check if the previous step failed
                with self.db.get_session() as session:
                    prev_step = session.query(StepExecution).filter(
                        StepExecution.execution_id == execution.id,
                        StepExecution.position == step.position - 1
                    ).first()
                    
                    if prev_step and prev_step.status != StepExecutionStatus.FAILED:
                        should_execute = False
                        step_execution.condition_result = False
            
            elif step.condition_type in [ConditionType.IF_DATA, ConditionType.IF_NO_DATA]:
                # Check data condition
                if step.condition_data_key:
                    has_data = False
                    if step.condition_data_key in execution.execution_context:
                        data_value = execution.execution_context[step.condition_data_key]
                        has_data = data_value is not None and data_value != ""
                    
                    if step.condition_type == ConditionType.IF_DATA:
                        should_execute = has_data
                    else:  # IF_NO_DATA
                        should_execute = not has_data
                    
                    step_execution.condition_result = should_execute
        
        # Add to database
        self.db.add(step_execution)
        
        # If should be skipped, mark as skipped and return
        if not should_execute:
            step_execution.status = StepExecutionStatus.SKIPPED
            step_execution.completed_at = utc_now()
            step_execution.runtime_seconds = 0
            self.db.update(step_execution)
            
            logger.info(f"Skipped step '{step.name}' (automation {automation.id}) due to condition")
            
            # Return next step based on step logic
            if step.on_failure_step_id:
                return step.on_failure_step_id
            else:
                return None
        
        # Resolve parameters using template engine
        try:
            resolved_parameters = self.template_engine.resolve_template(
                step.parameters,
                execution.execution_context
            )
            step_execution.resolved_parameters = resolved_parameters
            self.db.update(step_execution)
        except Exception as e:
            # Update step execution status
            step_execution.status = StepExecutionStatus.FAILED
            step_execution.error = f"Parameter resolution error: {str(e)}"
            step_execution.completed_at = utc_now()
            step_execution.runtime_seconds = (step_execution.completed_at - step_execution.started_at).total_seconds()
            self.db.update(step_execution)
            
            raise ToolError(
                f"Error resolving parameters for step '{step.name}': {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            ) from e
        
        # Maximum retry attempts
        max_retries = step.max_retries or 0
        
        # Execute with retry logic
        for attempt in range(max_retries + 1):
            try:
                step_execution.attempts = attempt + 1
                
                # Log attempt information
                if attempt > 0:
                    logger.info(
                        f"Retry attempt {attempt}/{max_retries} for step '{step.name}' (automation {automation.id})"
                    )
                
                # Execute based on execution mode
                if step.execution_mode == ExecutionMode.DIRECT:
                    result = self._execute_direct_step(step, resolved_parameters)
                else:
                    result = self._execute_orchestrated_step(step, resolved_parameters)
                
                # Update step execution and context
                with self.db.get_session() as session:
                    # Refresh from database for latest state
                    step_execution = session.query(StepExecution).filter(
                        StepExecution.id == step_execution.id
                    ).first()
                    
                    execution = session.query(AutomationExecution).filter(
                        AutomationExecution.id == execution.id
                    ).first()
                    
                    if not step_execution or not execution:
                        raise ToolError(
                            f"Step execution or automation execution record not found",
                            ErrorCode.TOOL_EXECUTION_ERROR
                        )
                    
                    # Update step execution
                    step_execution.status = StepExecutionStatus.COMPLETED
                    step_execution.result = result
                    step_execution.completed_at = utc_now()
                    step_execution.runtime_seconds = (step_execution.completed_at - step_execution.started_at).total_seconds()
                    
                    # Update execution context
                    if step.output_key:
                        if not execution.execution_context:
                            execution.execution_context = {}
                        execution.execution_context[step.output_key] = result
                    
                    # Determine next step based on logic
                    next_step_id = self._determine_next_step(step, result, execution.execution_context)
                    step_execution.next_step_id = next_step_id
                    
                    # Save changes
                    session.commit()
                
                return next_step_id
                
            except Exception as e:
                # Handle retry logic
                if attempt < max_retries:
                    # Log retry
                    logger.warning(
                        f"Error in step '{step.name}' (automation {automation.id}), attempt {attempt + 1}/{max_retries + 1}: {str(e)}. "
                        f"Retrying in {step.retry_delay or 60} seconds.",
                        exc_info=True
                    )
                    
                    # Wait before retry
                    time.sleep(step.retry_delay or 60)
                    continue
                
                # Last attempt failed, update step execution with failure
                with self.db.get_session() as session:
                    # Refresh from database
                    step_execution = session.query(StepExecution).filter(
                        StepExecution.id == step_execution.id
                    ).first()
                    
                    if step_execution:
                        step_execution.status = StepExecutionStatus.FAILED
                        step_execution.error = str(e)
                        step_execution.completed_at = utc_now()
                        step_execution.runtime_seconds = (step_execution.completed_at - step_execution.started_at).total_seconds()
                        session.commit()
                
                # Propagate the error
                raise ToolError(
                    f"Error executing step '{step.name}': {str(e)}",
                    ErrorCode.TOOL_EXECUTION_ERROR
                ) from e
    
    def _execute_direct_task(self, automation: Automation) -> Dict[str, Any]:
        """
        Execute a direct task.
        
        Args:
            automation: The automation to execute
            
        Returns:
            Result of the task execution
            
        Raises:
            ToolError: If the task execution fails
        """
        # Get the tool
        tool = self.tool_repo.get_tool(automation.tool_name)
        if not tool:
            raise ToolError(
                f"Tool '{automation.tool_name}' not found or not enabled",
                ErrorCode.TOOL_NOT_FOUND
            )
        
        # Execute the tool with the specified operation and parameters
        logger.debug(f"Executing tool {automation.tool_name}.{automation.operation} with parameters: {automation.parameters}")
        result = tool.run(automation.operation, **automation.parameters)
        
        return result
    
    def _execute_orchestrated_task(self, automation: Automation) -> Dict[str, Any]:
        """
        Execute an orchestrated task.
        
        Args:
            automation: The automation to execute
            
        Returns:
            Result of the task execution
            
        Raises:
            ToolError: If the task execution fails
        """
        logger.debug(f"Executing orchestrated task: {automation.task_description}")
        
        # Create a system prompt
        system_prompt = """
        You are an automation assistant executing a scheduled task.
        Your job is to accomplish the task described below using the available tools.
        Be thorough, efficient, and focused on the task at hand.
        
        You are executing an automated task with the following details:
        Task: {task_name}
        Description: {task_description}
        
        Once you have completed the task, provide a clear summary of what you did
        and any relevant results or findings.
        """.format(
            task_name=automation.name,
            task_description=automation.task_description
        )
        
        # Enable the tools for this task
        enabled_tools = []
        if automation.available_tools:
            # Only enable specified tools
            for tool_name in automation.available_tools:
                if self.tool_repo.enable_tool(tool_name):
                    enabled_tools.append(tool_name)
        else:
            # If no specific tools are provided, use all available tools
            enabled_tools = self.tool_repo.get_enabled_tools()
        
        # Create minimal conversation for the task
        from conversation import Conversation
        
        # Create a temporary conversation for this task
        conversation_id = f"automation_{automation.id}_{int(time.time())}"
        conversation = Conversation(
            conversation_id=conversation_id,
            system_prompt=system_prompt,
            llm_bridge=self.llm_bridge,
            tool_repo=self.tool_repo
        )
        
        # Execute the task
        response = conversation.generate_response(automation.task_description)
        
        # Extract results
        result = {
            "llm_response": response,
            "enabled_tools": enabled_tools,
            "tools_used": conversation.get_tool_usage_summary() if hasattr(conversation, 'get_tool_usage_summary') else []
        }
        
        return result
    
    def _execute_direct_step(self, step: AutomationStep, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a direct step.
        
        Args:
            step: The step to execute
            parameters: The resolved parameters
            
        Returns:
            Result of the step execution
            
        Raises:
            ToolError: If the step execution fails
        """
        # Get the tool
        tool = self.tool_repo.get_tool(step.tool_name)
        if not tool:
            raise ToolError(
                f"Tool '{step.tool_name}' not found or not enabled",
                ErrorCode.TOOL_NOT_FOUND
            )
        
        # Execute the tool with the specified operation and parameters
        logger.debug(f"Executing tool {step.tool_name}.{step.operation} with parameters: {parameters}")
        result = tool.run(step.operation, **parameters)
        
        return result
    
    def _execute_orchestrated_step(self, step: AutomationStep, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an orchestrated step.
        
        Args:
            step: The step to execute
            parameters: The resolved parameters
            
        Returns:
            Result of the step execution
            
        Raises:
            ToolError: If the step execution fails
        """
        logger.debug(f"Executing orchestrated step: {step.task_description}")
        
        # Create a system prompt
        system_prompt = """
        You are an automation assistant executing a sequence step.
        Your job is to accomplish the step described below using the available tools.
        Be thorough, efficient, and focused on the task at hand.
        
        You are executing a step with the following details:
        Step: {step_name}
        Description: {task_description}
        
        Once you have completed the step, provide a clear summary of what you did
        and any relevant results or findings.
        """.format(
            step_name=step.name,
            task_description=step.task_description
        )
        
        # Enable the tools for this step
        enabled_tools = []
        if step.available_tools:
            # Only enable specified tools
            for tool_name in step.available_tools:
                if self.tool_repo.enable_tool(tool_name):
                    enabled_tools.append(tool_name)
        else:
            # If no specific tools are provided, use all available tools
            enabled_tools = self.tool_repo.get_enabled_tools()
        
        # Create minimal conversation for the task
        from conversation import Conversation
        
        # Create a temporary conversation for this step
        conversation_id = f"step_{step.id}_{int(time.time())}"
        conversation = Conversation(
            conversation_id=conversation_id,
            system_prompt=system_prompt,
            llm_bridge=self.llm_bridge,
            tool_repo=self.tool_repo
        )
        
        # Construct user message with parameters
        user_message = step.task_description
        if parameters:
            param_str = "\n\nParameters:\n"
            for key, value in parameters.items():
                if isinstance(value, (dict, list)):
                    param_str += f"- {key}: {json.dumps(value)}\n"
                else:
                    param_str += f"- {key}: {value}\n"
            user_message += param_str
        
        # Execute the task
        response = conversation.generate_response(user_message)
        
        # Extract results
        result = {
            "llm_response": response,
            "enabled_tools": enabled_tools,
            "tools_used": conversation.get_tool_usage_summary() if hasattr(conversation, 'get_tool_usage_summary') else []
        }
        
        return result
    
    def _determine_next_step(
        self, 
        step: AutomationStep, 
        result: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Determine the next step based on step logic and results.
        
        Args:
            step: The current step
            result: Result of the step execution
            context: The execution context
            
        Returns:
            ID of the next step or None for linear progression
        """
        # Check for explicit next step overrides first
        if step.on_success_step_id:
            return step.on_success_step_id
        
        # If no explicit next step is defined, return None for linear progression
        return None
    
    def _calculate_next_run_time(self, automation: Automation) -> Optional[datetime]:
        """
        Calculate the next run time for a recurring automation efficiently.
        
        Args:
            automation: The automation to calculate next run time for
            
        Returns:
            The next run time or None if no more runs are scheduled
        """
        # Early termination checks
        if automation.frequency == TaskFrequency.ONCE:
            return None
            
        if (automation.end_time and automation.end_time <= utc_now()) or \
           (automation.max_executions and automation.execution_count >= automation.max_executions):
            return None
        
        # Get base time components
        base_time = automation.scheduled_time
        if not base_time:
            return None
        
        # Use automation's timezone or default
        tz_name = automation.timezone or config.system.timezone
        tz_name = validate_timezone(tz_name)
        
        # Get current time in the user's timezone once
        now_utc = utc_now()
        now_local = convert_from_utc(now_utc, tz_name)
        
        # Extract time components from base time for reuse
        time_components = {
            'hour': base_time.hour,
            'minute': base_time.minute,
            'second': base_time.second,
            'microsecond': 0
        }
        
        # Calculate next run based on frequency in the user's timezone
        if automation.frequency == TaskFrequency.MINUTELY:
            next_run = now_local.replace(second=time_components['second']) + timedelta(minutes=1)
        
        elif automation.frequency == TaskFrequency.HOURLY:
            next_run = now_local.replace(minute=time_components['minute'], 
                                      second=time_components['second'])
            if next_run <= now_local:
                next_run += timedelta(hours=1)
        
        elif automation.frequency == TaskFrequency.DAILY:
            next_run = now_local.replace(**time_components)
            if next_run <= now_local:
                next_run += timedelta(days=1)
        
        elif automation.frequency == TaskFrequency.WEEKLY:
            # Get target day of week (0=Monday, 6=Sunday)
            target_day = automation.day_of_week if automation.day_of_week is not None else base_time.weekday()
            
            # Create a time on the target day
            next_run = now_local.replace(**time_components)
            days_ahead = (target_day - now_local.weekday()) % 7  # Positive modulo ensures days_ahead is 0-6
            
            # If it's the same day but time has passed, or it's a future day
            if (days_ahead == 0 and next_run <= now_local) or days_ahead > 0:
                next_run += timedelta(days=days_ahead)
            else:
                next_run += timedelta(days=7)  # Next week
        
        elif automation.frequency == TaskFrequency.MONTHLY:
            # Get target day of month
            target_day = automation.day_of_month if automation.day_of_month is not None else base_time.day
            
            # Start with current month
            next_run = self._get_valid_day_in_month(now_local.year, now_local.month, target_day)
            next_run = next_run.replace(**time_components)
            
            # If this time has already passed, move to next month
            if next_run <= now_local:
                if now_local.month == 12:
                    next_run = self._get_valid_day_in_month(now_local.year + 1, 1, target_day)
                else:
                    next_run = self._get_valid_day_in_month(now_local.year, now_local.month + 1, target_day)
                next_run = next_run.replace(**time_components)
        
        elif automation.frequency == TaskFrequency.CUSTOM and automation.custom_schedule:
            # Handle custom schedules efficiently
            parts = automation.custom_schedule.lower().split()
            if len(parts) >= 3 and parts[0] == "every":
                try:
                    amount = int(parts[1])
                    unit = parts[2]
                    
                    # Start with today at the scheduled time
                    next_run = now_local.replace(**time_components)
                    
                    # If this time has already passed, use the last run time + interval 
                    if next_run <= now_local:
                        if unit.startswith("minute"):
                            next_run = now_local + timedelta(minutes=amount)
                        elif unit.startswith("hour"):
                            next_run = now_local + timedelta(hours=amount)
                        elif unit.startswith("day"):
                            next_run = now_local + timedelta(days=amount)
                        elif unit.startswith("week"):
                            next_run = now_local + timedelta(weeks=amount)
                        elif unit.startswith("month"):
                            next_run = now_local + relativedelta(months=amount)
                        else:
                            # Unknown unit, default to daily
                            next_run = now_local + timedelta(days=1)
                except ValueError:
                    # Invalid format, default to daily
                    next_run = now_local.replace(**time_components)
                    if next_run <= now_local:
                        next_run += timedelta(days=1)
            else:
                # Unsupported format, default to daily
                next_run = now_local.replace(**time_components)
                if next_run <= now_local:
                    next_run += timedelta(days=1)
        else:
            # Unknown frequency, default to daily
            next_run = now_local.replace(**time_components)
            if next_run <= now_local:
                next_run += timedelta(days=1)
        
        # Convert back to UTC for storage
        # next_run is in user's timezone, convert to UTC for storage
        return convert_to_utc(next_run)
    
    def _get_valid_day_in_month(self, year: int, month: int, day: int) -> datetime:
        """
        Get a valid day in the specified month, handling edge cases like February 30.
        
        Args:
            year: The year
            month: The month
            day: The target day
            
        Returns:
            A datetime object with a valid day in the month
        """
        # Get the number of days in the month
        if month == 12:
            last_day = (datetime(year + 1, 1, 1) - timedelta(days=1)).day
        else:
            last_day = (datetime(year, month + 1, 1) - timedelta(days=1)).day
        
        # Use the minimum of the requested day and the last day of the month
        return datetime(year, month, min(day, last_day))
    
    # Public API methods
    def create_automation(self, data: Dict[str, Any]) -> Automation:
        """
        Create a new automation.
        
        Args:
            data: Automation definition data
            
        Returns:
            The created automation
            
        Raises:
            ToolError: If the automation is invalid
        """
        with error_context(
            component_name="AutomationEngine",
            operation="create_automation",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_INVALID_INPUT,
            logger=logger
        ):
            # Create the automation object
            automation = Automation.from_dict(data)
            
            # Set timezone if not provided
            if not automation.timezone:
                automation.timezone = config.system.timezone
            
            # Process scheduled_time
            if isinstance(data.get("scheduled_time"), str):
                scheduled_time = data["scheduled_time"]
                # Use our timezone utilities to properly parse time strings
                if ":" in scheduled_time:
                    # Parse the time using our utility
                    tz_name = automation.timezone
                    parsed_dt = parse_time_string(scheduled_time, tz_name)
                    
                    # If time has already passed today and it's a time-only string,
                    # use tomorrow's date
                    if "T" not in scheduled_time and parsed_dt < utc_now():
                        parsed_dt += timedelta(days=1)
                    
                    # Convert to UTC for storage
                    automation.scheduled_time = ensure_utc(parsed_dt)
            
            # Calculate next run time if not provided
            if not automation.next_execution_time:
                # For immediate first run, use scheduled_time
                if automation.scheduled_time:
                    automation.next_execution_time = automation.scheduled_time
                else:
                    automation.next_execution_time = utc_now()
            
            # Validate the automation
            errors = automation.validate()
            if errors:
                raise ToolError(
                    f"Invalid automation definition: {', '.join(errors)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Save to database
            self.db.add(automation)
            logger.info(f"Created new automation: {automation.id} ('{automation.name}')")
            
            return automation
    
    def get_automation(self, automation_id: str) -> Optional[Automation]:
        """
        Get an automation by ID.
        
        Args:
            automation_id: ID of the automation
            
        Returns:
            The automation or None if not found
        """
        return self.db.get(Automation, automation_id)
    
    def get_automations(
        self,
        automation_type: Optional[str] = None,
        status: Optional[str] = None,
        frequency: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get automations filtered by criteria.
        
        Args:
            automation_type: Filter by automation type
            status: Filter by automation status
            frequency: Filter by execution frequency
            user_id: Filter by user ID
            limit: Maximum number of automations to return
            offset: Offset for pagination
            
        Returns:
            Dict containing automations and count
        """
        with self.db.get_session() as session:
            query = session.query(Automation)
            
            # Apply filters
            if automation_type:
                query = query.filter(Automation.type == AutomationType(automation_type))
                
            if status:
                query = query.filter(Automation.status == AutomationStatus(status))
                
            if frequency:
                query = query.filter(Automation.frequency == TaskFrequency(frequency))
                
            if user_id:
                query = query.filter(Automation.user_id == user_id)
                
            # Get total count
            total_count = query.count()
            
            # Apply sorting and pagination
            query = query.order_by(Automation.next_execution_time)
            query = query.limit(limit).offset(offset)
            
            # Execute query
            automations = query.all()
            
            return {
                "automations": [a.to_dict() for a in automations],
                "count": len(automations),
                "total": total_count
            }
    
    def update_automation(self, automation_id: str, data: Dict[str, Any]) -> Optional[Automation]:
        """
        Update an existing automation.
        
        Args:
            automation_id: ID of the automation to update
            data: New automation data
            
        Returns:
            The updated automation or None if not found
            
        Raises:
            ToolError: If the automation update is invalid
        """
        with error_context(
            component_name="AutomationEngine",
            operation="update_automation",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=logger
        ):
            # Get the existing automation
            automation = self.db.get(Automation, automation_id)
            if not automation:
                return None
            
            # Process scheduled_time if provided as string
            if isinstance(data.get("scheduled_time"), str):
                scheduled_time = data["scheduled_time"]
                # Use our timezone utilities to properly parse the time string
                tz_name = data.get("timezone") or automation.timezone or config.system.timezone
                parsed_dt = parse_time_string(scheduled_time, tz_name)
                
                # If time has already passed today and it's a time-only string,
                # use tomorrow's date
                if "T" not in scheduled_time and ":" in scheduled_time and parsed_dt < utc_now():
                    parsed_dt += timedelta(days=1)
                
                # Convert to UTC for storage
                data["scheduled_time"] = ensure_utc(parsed_dt)
            
            # Create a new automation with the updated data
            updated_data = automation.to_dict()
            updated_data.update(data)
            
            updated_automation = Automation.from_dict(updated_data)
            
            # Validate the updated automation
            errors = updated_automation.validate()
            if errors:
                raise ToolError(
                    f"Invalid automation update: {', '.join(errors)}",
                    ErrorCode.TOOL_INVALID_INPUT
                )
            
            # Recalculate next_execution_time if schedule parameters changed
            schedule_params = ["frequency", "scheduled_time", "day_of_week", "day_of_month", "custom_schedule", "timezone"]
            if any(param in data for param in schedule_params):
                next_run = self._calculate_next_run_time(updated_automation)
                if next_run:
                    updated_automation.next_execution_time = next_run
            
            # Increment version
            updated_automation.version += 1
            
            # Save to database
            self.db.update(updated_automation)
            logger.info(f"Updated automation: {automation_id}")
            
            return updated_automation
    
    def delete_automation(self, automation_id: str) -> bool:
        """
        Delete an automation.
        
        Args:
            automation_id: ID of the automation to delete
            
        Returns:
            True if the automation was deleted, False if not found
        """
        automation = self.db.get(Automation, automation_id)
        if not automation:
            return False
            
        # Check if automation is currently running
        if automation_id in self.active_automations:
            logger.warning(f"Cannot delete automation {automation_id} while it's running")
            return False
            
        self.db.delete(automation)
        logger.info(f"Deleted automation: {automation_id}")
        
        return True
    
    def pause_automation(self, automation_id: str) -> bool:
        """
        Pause an automation.
        
        Args:
            automation_id: ID of the automation to pause
            
        Returns:
            True if the automation was paused, False if not found
        """
        automation = self.db.get(Automation, automation_id)
        if not automation:
            return False
            
        # Already paused?
        if automation.status == AutomationStatus.PAUSED:
            return True
            
        automation.status = AutomationStatus.PAUSED
        self.db.update(automation)
        logger.info(f"Paused automation: {automation_id}")
        
        return True
    
    def resume_automation(self, automation_id: str) -> Dict[str, Any]:
        """
        Resume a paused automation.
        
        Args:
            automation_id: ID of the automation to resume
            
        Returns:
            Dict with success status and next execution time
        """
        automation = self.db.get(Automation, automation_id)
        if not automation:
            return {"success": False, "message": "Automation not found"}
            
        # Already active?
        if automation.status == AutomationStatus.ACTIVE:
            return {
                "success": True, 
                "message": "Automation is already active",
                "next_execution": automation.next_execution_time.isoformat() if automation.next_execution_time else None
            }
            
        # Update status
        automation.status = AutomationStatus.ACTIVE
        
        # Recalculate next execution time if needed
        now = utc_now()
        
        if not automation.next_execution_time or automation.next_execution_time < now:
            next_run = self._calculate_next_run_time(automation)
            automation.next_execution_time = next_run or (now + timedelta(minutes=1))
            
        self.db.update(automation)
        logger.info(f"Resumed automation: {automation_id}")
        
        return {
            "success": True,
            "message": f"Automation resumed successfully",
            "next_execution": automation.next_execution_time.isoformat() if automation.next_execution_time else None
        }
    
    def execute_now(self, automation_id: str, initial_context: Dict[str, Any] = None) -> AutomationExecution:
        """
        Execute an automation immediately.
        
        Args:
            automation_id: ID of the automation to execute
            initial_context: Initial context data
            
        Returns:
            The automation execution record
            
        Raises:
            ToolError: If execution fails
        """
        with error_context(
            component_name="AutomationEngine",
            operation="execute_now",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=logger
        ):
            # Get the automation
            automation = self.db.get(Automation, automation_id)
            if not automation:
                raise ToolError(
                    f"Automation {automation_id} not found",
                    ErrorCode.TOOL_NOT_FOUND
                )
            
            # Check if automation is active
            if automation.status != AutomationStatus.ACTIVE:
                raise ToolError(
                    f"Cannot execute automation {automation_id} with status '{automation.status.value}'",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
            
            # Check if already running
            if automation_id in self.active_automations:
                raise ToolError(
                    f"Automation {automation_id} is already running",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
            
            # Create execution record
            execution = AutomationExecution(
                id=str(uuid.uuid4()),
                automation_id=automation.id,
                status=ExecutionStatus.PENDING,
                trigger_type=TriggerType.MANUAL,
                scheduled_time=utc_now(),
                execution_context=initial_context or {}
            )
            
            # Save to database
            self.db.add(execution)
            
            # Launch execution in a separate thread
            threading.Thread(
                target=self._execute_automation_safely,
                args=(automation.id,),
                daemon=True,
                name=f"ManualAutomation-{automation.id[:8]}"
            ).start()
            
            return execution
    
    def get_executions(
        self,
        automation_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get execution history for an automation.
        
        Args:
            automation_id: ID of the automation
            limit: Maximum number of executions to return
            offset: Offset for pagination
            
        Returns:
            Dict containing executions and count
        """
        with self.db.get_session() as session:
            query = session.query(AutomationExecution).filter(
                AutomationExecution.automation_id == automation_id
            )
            
            # Get total count
            total_count = query.count()
            
            # Apply sorting and pagination
            query = query.order_by(AutomationExecution.created_at.desc())
            query = query.limit(limit).offset(offset)
            
            # Execute query
            executions = query.all()
            
            # Create simplified view for list display
            simplified_executions = []
            for execution in executions:
                simplified_executions.append({
                    "id": execution.id,
                    "status": execution.status.value if execution.status else None,
                    "trigger_type": execution.trigger_type.value if execution.trigger_type else None,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None,
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "runtime_seconds": execution.runtime_seconds,
                    "error": execution.error
                })
                
            return {
                "executions": simplified_executions,
                "count": len(executions),
                "total": total_count
            }
    
    def get_execution_details(self, execution_id: str) -> Optional[AutomationExecution]:
        """
        Get detailed information about a specific execution.
        
        Args:
            execution_id: ID of the execution
            
        Returns:
            The execution record or None if not found
        """
        with self.db.get_session() as session:
            from sqlalchemy.orm import joinedload
            execution = session.query(AutomationExecution).options(
                joinedload(AutomationExecution.step_executions)
            ).filter(
                AutomationExecution.id == execution_id
            ).first()
            
            return execution
    
    def get_upcoming_automations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming automations for a given time period.
        
        Args:
            start_time: Start of the time period
            end_time: End of the time period
            user_id: Filter by user ID
            limit: Maximum number of automations to return
            
        Returns:
            List of upcoming automations
        """
        if not start_time:
            start_time = utc_now()
            
        if not end_time:
            # Default to 24 hours from now
            end_time = start_time + timedelta(hours=24)
            
        with self.db.get_session() as session:
            query = session.query(Automation).filter(
                Automation.status == AutomationStatus.ACTIVE,
                Automation.next_execution_time >= start_time,
                Automation.next_execution_time <= end_time
            )
            
            if user_id:
                query = query.filter(Automation.user_id == user_id)
                
            # Apply sorting and limit
            query = query.order_by(Automation.next_execution_time)
            query = query.limit(limit)
            
            # Execute query
            automations = query.all()
            
            # Create simplified view for display
            result = []
            for automation in automations:
                # Convert next_execution_time to user's timezone happens in to_dict()
                auto_dict = automation.to_dict()
                
                # Keep only the fields we need
                result.append({
                    "id": automation.id,
                    "name": automation.name,
                    "type": automation.type.value if automation.type else None,
                    "next_execution_time": auto_dict["next_execution_time"],
                    # Include the raw UTC time for API consumers that need it
                    "next_execution_time_utc": automation.next_execution_time.isoformat() 
                    if automation.next_execution_time else None
                })
                
            return result
    
    def find_automations_by_name_or_description(
        self,
        search_text: str,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find automations by name or description.
        
        Args:
            search_text: Text to search for
            user_id: Filter by user ID
            limit: Maximum number of automations to return
            
        Returns:
            List of matching automations
        """
        with self.db.get_session() as session:
            from sqlalchemy import or_
            
            query = session.query(Automation).filter(
                or_(
                    Automation.name.ilike(f"%{search_text}%"),
                    Automation.description.ilike(f"%{search_text}%")
                )
            )
            
            if user_id:
                query = query.filter(Automation.user_id == user_id)
                
            # Apply sorting and limit
            query = query.order_by(Automation.next_execution_time)
            query = query.limit(limit)
            
            # Execute query
            automations = query.all()
            
            # Create simplified view for display
            result = []
            for automation in automations:
                result.append({
                    "id": automation.id,
                    "name": automation.name,
                    "type": automation.type.value if automation.type else None,
                    "status": automation.status.value if automation.status else None,
                    "frequency": automation.frequency.value if automation.frequency else None,
                    "next_execution_time": automation.next_execution_time.isoformat() if automation.next_execution_time else None
                })
                
            return result


def get_automation_engine() -> AutomationEngine:
    """
    Get or create the automation engine.
    
    Returns:
        The automation engine instance
    """
    global _automation_engine
    
    with _engine_lock:
        if _automation_engine is None:
            _automation_engine = AutomationEngine()
            
    return _automation_engine


def initialize_automation_engine() -> AutomationEngine:
    """
    Initialize the automation engine and start the scheduler.
    
    Returns:
        The initialized automation engine
    """
    engine = get_automation_engine()
    engine.start_scheduler()
    return engine