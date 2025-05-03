"""
Chain executor module for executing task chains.

This module provides the ChainExecutor class for executing task chains,
handling step execution, parameter substitution, and error handling.
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple

from db import Database
from errors import ToolError, ErrorCode, error_context
from task_manager.task_chain import (
    TaskChain, ChainStep, ChainStatus, ErrorPolicy
)
from task_manager.chain_execution import (
    ChainExecution, StepExecution, ChainExecutionStatus, StepExecutionStatus, TriggerType
)
from task_manager.chain_template import TemplateEngine
from task_manager.scheduled_task import ExecutionMode
from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository
from task_manager.task_notification import NotificationManager, NotificationType, NotificationPriority

# Configure logger
logger = logging.getLogger(__name__)


class ChainExecutor:
    """
    Chain executor for executing task chains.
    
    Handles the execution of task chains including parameter substitution,
    step execution, error handling, and result tracking.
    """
    
    def __init__(
        self,
        tool_repo: Optional[ToolRepository] = None,
        llm_bridge: Optional[LLMBridge] = None,
        notification_manager: Optional[NotificationManager] = None
    ):
        """
        Initialize the chain executor.
        
        Args:
            tool_repo: Repository of available tools
            llm_bridge: LLM bridge for orchestrated steps
            notification_manager: Manager for creating notifications
        """
        self.db = Database()
        self.tool_repo = tool_repo or ToolRepository()
        self.llm_bridge = llm_bridge or LLMBridge()
        self.notification_manager = notification_manager or NotificationManager()
        self.template_engine = TemplateEngine()
        
        # Initialize execution state
        self.active_chains = {}  # chain_id -> chain_execution_id
        self.chain_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def execute_chain(
        self, 
        chain_id: str,
        trigger_type: TriggerType = TriggerType.SCHEDULED,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> ChainExecution:
        """
        Execute a chain by its ID.
        
        Args:
            chain_id: ID of the chain to execute
            trigger_type: What triggered this execution
            initial_context: Initial context for the execution
            
        Returns:
            The chain execution record
            
        Raises:
            ToolError: If the chain execution fails
        """
        with error_context(
            component_name="ChainExecutor",
            operation="execute_chain",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Check if chain is already running
            with self.chain_lock:
                if chain_id in self.active_chains:
                    raise ToolError(
                        f"Chain {chain_id} is already running (execution {self.active_chains[chain_id]})",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    )
            
            # Load the chain
            chain = self.db.get(TaskChain, chain_id)
            if not chain:
                raise ToolError(
                    f"Chain {chain_id} not found",
                    ErrorCode.TOOL_NOT_FOUND
                )
            
            # Check if chain is active
            if chain.status != ChainStatus.ACTIVE:
                raise ToolError(
                    f"Chain {chain_id} is not active (status: {chain.status.value})",
                    ErrorCode.TOOL_EXECUTION_ERROR
                )
            
            # Create execution record
            execution = ChainExecution(
                id=str(uuid.uuid4()),
                chain_id=chain_id,
                status=ChainExecutionStatus.RUNNING,
                trigger_type=trigger_type,
                started_at=datetime.now(timezone.utc),
                execution_context=initial_context or {}
            )
            
            # Save execution record
            self.db.add(execution)
            
            # Update chain with execution info
            chain.last_execution_id = execution.id
            chain.last_execution_time = execution.started_at
            chain.execution_count += 1
            self.db.update(chain)
            
            # Mark chain as active
            with self.chain_lock:
                self.active_chains[chain_id] = execution.id
            
            try:
                # Execute the chain
                self._execute_chain_steps(chain, execution)
                
                # Mark execution as completed
                execution.status = ChainExecutionStatus.COMPLETED
                execution.completed_at = datetime.now(timezone.utc)
                execution.runtime_seconds = (execution.completed_at - execution.started_at).total_seconds()
                self.db.update(execution)
                
                # Create success notification
                self._create_execution_notification(
                    chain=chain,
                    execution=execution,
                    success=True
                )
                
                self.logger.info(f"Chain {chain_id} executed successfully (execution {execution.id})")
                
            except Exception as e:
                # Log the error
                error_message = str(e)
                self.logger.error(f"Error executing chain {chain_id}: {error_message}", exc_info=True)
                
                # Update execution with error
                execution.status = ChainExecutionStatus.FAILED
                execution.completed_at = datetime.now(timezone.utc)
                execution.runtime_seconds = (execution.completed_at - execution.started_at).total_seconds()
                execution.error = error_message
                self.db.update(execution)
                
                # Create failure notification
                self._create_execution_notification(
                    chain=chain,
                    execution=execution,
                    success=False,
                    error=error_message
                )
                
                # Propagate the error
                raise
                
            finally:
                # Remove chain from active chains
                with self.chain_lock:
                    self.active_chains.pop(chain_id, None)
            
            return execution
    
    def _execute_chain_steps(self, chain: TaskChain, execution: ChainExecution) -> None:
        """
        Execute the steps of a chain.
        
        Args:
            chain: The chain to execute
            execution: The execution record
            
        Raises:
            ToolError: If step execution fails and error policy is STOP
        """
        # Ensure steps are sorted by position
        steps = sorted(chain.steps, key=lambda s: s.position)
        
        if not steps:
            self.logger.warning(f"Chain {chain.id} has no steps")
            return
        
        # Get first step
        current_step = steps[0]
        step_index = 0
        
        # Execute steps until done
        while current_step and step_index < len(steps):
            try:
                # Execute the step
                next_step_id = self._execute_step(
                    step=current_step,
                    chain=chain,
                    execution=execution
                )
                
                # Find the next step
                if next_step_id:
                    # Find step by ID
                    next_step = next((s for s in steps if s.id == next_step_id), None)
                    if not next_step:
                        raise ToolError(
                            f"Next step {next_step_id} not found in chain {chain.id}",
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
                error_policy = current_step.error_policy or chain.error_policy
                
                if error_policy == ErrorPolicy.STOP:
                    # Stop chain execution
                    raise ToolError(
                        f"Chain {chain.id} execution stopped due to error in step '{current_step.name}': {str(e)}",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    ) from e
                
                elif error_policy == ErrorPolicy.CONTINUE:
                    # Log error and continue with next step
                    self.logger.error(
                        f"Error in step '{current_step.name}' (chain {chain.id}): {str(e)}. "
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
                    self.logger.error(
                        f"Error in step '{current_step.name}' (chain {chain.id}) after max retries: {str(e)}. "
                        f"Continuing with next step.",
                        exc_info=True
                    )
                    
                    # Move to next step
                    step_index += 1
                    if step_index < len(steps):
                        current_step = steps[step_index]
                    else:
                        current_step = None
                
                else:
                    # Other error policies are not implemented yet
                    self.logger.error(
                        f"Unsupported error policy {error_policy} for step '{current_step.name}' (chain {chain.id}). "
                        f"Stopping chain execution.",
                        exc_info=True
                    )
                    raise ToolError(
                        f"Unsupported error policy {error_policy} for step '{current_step.name}'",
                        ErrorCode.TOOL_EXECUTION_ERROR
                    ) from e
    
    def _execute_step(
        self, 
        step: ChainStep,
        chain: TaskChain,
        execution: ChainExecution
    ) -> Optional[str]:
        """
        Execute a single step of a chain.
        
        Args:
            step: The step to execute
            chain: The parent chain
            execution: The chain execution record
            
        Returns:
            ID of the next step to execute, or None for linear progression
            
        Raises:
            ToolError: If step execution fails
        """
        # Create step execution record
        step_execution = StepExecution(
            id=str(uuid.uuid4()),
            chain_execution_id=execution.id,
            step_id=step.id,
            status=StepExecutionStatus.RUNNING,
            position=step.position,
            started_at=datetime.now(timezone.utc)
        )
        
        # Resolve parameters using template engine
        try:
            resolved_parameters = self.template_engine.resolve_template(
                step.parameters,
                execution.execution_context
            )
            step_execution.resolved_parameters = resolved_parameters
        except Exception as e:
            step_execution.status = StepExecutionStatus.FAILED
            step_execution.error = f"Parameter resolution error: {str(e)}"
            step_execution.completed_at = datetime.now(timezone.utc)
            step_execution.runtime_seconds = (step_execution.completed_at - step_execution.started_at).total_seconds()
            self.db.add(step_execution)
            
            raise ToolError(
                f"Error resolving parameters for step '{step.name}': {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            ) from e
        
        # Save initial step execution record
        self.db.add(step_execution)
        
        # Maximum retry attempts
        max_retries = step.max_retries or 0
        
        # Execute with retry logic
        for attempt in range(max_retries + 1):
            try:
                step_execution.attempts = attempt + 1
                
                # Log attempt information
                if attempt > 0:
                    self.logger.info(
                        f"Retry attempt {attempt}/{max_retries} for step '{step.name}' (chain {chain.id})"
                    )
                
                # Execute based on execution mode
                if step.execution_mode == ExecutionMode.DIRECT:
                    result = self._execute_direct_step(step, resolved_parameters)
                else:
                    result = self._execute_orchestrated_step(step, resolved_parameters)
                
                # Update step execution with success
                step_execution.status = StepExecutionStatus.COMPLETED
                step_execution.result = result
                step_execution.completed_at = datetime.now(timezone.utc)
                step_execution.runtime_seconds = (step_execution.completed_at - step_execution.started_at).total_seconds()
                self.db.update(step_execution)
                
                # Store result in execution context
                if step.output_key:
                    execution.execution_context[step.output_key] = result
                    self.db.update(execution)
                
                # Determine next step based on step logic
                next_step_id = self._determine_next_step(step, result, execution.execution_context)
                step_execution.next_step_id = next_step_id
                self.db.update(step_execution)
                
                return next_step_id
                
            except Exception as e:
                # Handle retry logic
                if attempt < max_retries:
                    # Log retry
                    self.logger.warning(
                        f"Error in step '{step.name}' (chain {chain.id}), attempt {attempt + 1}/{max_retries + 1}: {str(e)}. "
                        f"Retrying in {step.retry_delay or 60} seconds.",
                        exc_info=True
                    )
                    
                    # Wait before retry
                    time.sleep(step.retry_delay or 60)
                    continue
                
                # Last attempt failed, update step execution with failure
                step_execution.status = StepExecutionStatus.FAILED
                step_execution.error = str(e)
                step_execution.completed_at = datetime.now(timezone.utc)
                step_execution.runtime_seconds = (step_execution.completed_at - step_execution.started_at).total_seconds()
                self.db.update(step_execution)
                
                # Propagate the error
                raise ToolError(
                    f"Error executing step '{step.name}': {str(e)}",
                    ErrorCode.TOOL_EXECUTION_ERROR
                ) from e
    
    def _execute_direct_step(self, step: ChainStep, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a step in direct execution mode.
        
        Args:
            step: The step to execute
            parameters: The resolved parameters for execution
            
        Returns:
            Result of the step execution
            
        Raises:
            ToolError: If step execution fails
        """
        # Get the tool
        tool = self.tool_repo.get_tool(step.tool_name)
        if not tool:
            raise ToolError(
                f"Tool '{step.tool_name}' not found or not enabled",
                ErrorCode.TOOL_NOT_FOUND
            )
        
        # Execute the tool with the specified operation and parameters
        self.logger.debug(f"Executing tool {step.tool_name}.{step.operation} with parameters: {parameters}")
        result = tool.run(step.operation, **parameters)
        
        return result
    
    def _execute_orchestrated_step(self, step: ChainStep, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a step in LLM-orchestrated execution mode.
        
        Args:
            step: The step to execute
            parameters: The resolved parameters for execution
            
        Returns:
            Result of the step execution
            
        Raises:
            ToolError: If step execution fails
        """
        self.logger.debug(f"Executing orchestrated step: {step.task_description}")
        
        # Create a system prompt
        system_prompt = """
        You are a task automation assistant that helps execute scheduled tasks.
        Your goal is to accomplish the task described using the available tools.
        Be thorough, efficient, and provide clear explanations of your actions.
        
        You are executing an automated task step with the following details:
        Step: {step_name}
        Task: {task_description}
        
        Your objective is to accomplish this task using the tools available to you.
        When you're done, provide a concise summary of what you did and any results or
        findings.
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
        step: ChainStep, 
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
        # If no next step logic is defined, return None for linear progression
        if not step.next_step_logic or not isinstance(step.next_step_logic, dict):
            return None
        
        # Get the type of next step logic
        logic_type = step.next_step_logic.get("type")
        
        if logic_type == "conditional":
            # Conditional branching based on rules
            rules = step.next_step_logic.get("rules", [])
            
            # Evaluate each rule in order
            for rule in rules:
                if "condition" in rule and "next_step" in rule:
                    try:
                        # Evaluate the condition
                        if self.template_engine.evaluate_condition(rule["condition"], context):
                            return rule["next_step"]
                    except Exception as e:
                        self.logger.warning(f"Error evaluating condition '{rule['condition']}': {str(e)}")
                
                # Check for default rule
                if "default" in rule:
                    return rule["default"]
            
            # No matching rule
            return None
            
        elif logic_type == "fixed":
            # Fixed next step
            return step.next_step_logic.get("next_step")
            
        # Default to linear progression
        return None
    
    def _create_execution_notification(
        self,
        chain: TaskChain,
        execution: ChainExecution,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Create a notification for chain execution result.
        
        Args:
            chain: The chain that was executed
            execution: The chain execution record
            success: Whether the execution succeeded
            error: Error message if failed
        """
        # Get all successful step results
        successful_steps = []
        for step_execution in execution.step_executions:
            if step_execution.status == StepExecutionStatus.COMPLETED:
                successful_steps.append({
                    "name": self.db.get(ChainStep, step_execution.step_id).name,
                    "result": step_execution.result
                })
        
        # Create notification data
        notification_type = NotificationType.TASK_COMPLETED if success else NotificationType.TASK_FAILED
        priority = NotificationPriority.NORMAL if success else NotificationPriority.HIGH
        
        if success:
            title = f"✅ Chain '{chain.name}' completed successfully"
            content = f"Task chain '{chain.name}' completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Add step summary for successful chains
            if successful_steps:
                content += f"\n\nCompleted steps: {len(successful_steps)}/{len(execution.step_executions)}"
                
                # Include brief summary of what was done in each step
                content += "\n\nSummary of steps:"
                for i, step in enumerate(successful_steps, 1):
                    step_name = step["name"]
                    content += f"\n{i}. {step_name}"
        else:
            title = f"❌ Chain '{chain.name}' failed"
            content = f"Task chain '{chain.name}' failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            if error:
                content += f"\n\nError: {error}"
                
            # Add information about which step failed
            failed_step = next((s for s in execution.step_executions if s.status == StepExecutionStatus.FAILED), None)
            if failed_step:
                step = self.db.get(ChainStep, failed_step.step_id)
                if step:
                    content += f"\n\nFailed at step: {step.name}"
        
        # Add runtime information
        if execution.runtime_seconds:
            content += f"\n\nTotal runtime: {execution.runtime_seconds:.1f} seconds"
        
        # Create notification
        self.notification_manager.create_notification({
            "title": title,
            "content": content,
            "notification_type": notification_type.value,
            "priority": priority.value,
            "result": {
                "chain_id": chain.id,
                "chain_name": chain.name,
                "execution_id": execution.id,
                "successful_steps": len(successful_steps),
                "total_steps": len(execution.step_executions),
                "runtime_seconds": execution.runtime_seconds
            }
        })