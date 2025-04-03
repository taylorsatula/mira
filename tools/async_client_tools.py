"""
Tools for working with the asynchronous task client.

This module provides tools for the main conversation process
to interact with the background task service.
"""
from typing import Dict, Any, Optional

from tools.repo import Tool
from errors import ToolError, ErrorCode, error_context
from async_client import AsyncClient


class ScheduleAsyncTaskTool(Tool):
    """
    Tool for scheduling tasks to run in the background.
    
    Allows the LLM to delegate complex operations to run
    in the background without blocking the conversation.
    """
    
    name = "schedule_async_task"
    description = "Schedule a task to run asynchronously in the background"
    background_capable = True
    usage_examples = [
        {
            "input": {
                "description": "Generate report on customer data",
                "task_prompt": "Analyze customer data and create a summary report. Save the results using the persistence tool.",
                "notify_on_completion": True
            },
            "output": {"task_id": "123e4567-e89b-12d3-a456-426614174000"}
        }
    ]
    
    def __init__(self, async_client: AsyncClient):
        """
        Initialize the schedule async task tool.
        
        Args:
            async_client: AsyncClient instance (required)
        """
        super().__init__()
        self.async_client = async_client
    
    def run(
        self,
        description: str,
        task_prompt: str,
        notify_on_completion: bool = False
    ) -> Dict[str, Any]:
        """
        Schedule a task to run asynchronously.
        
        Args:
            description: Human-readable description of the task
            task_prompt: Detailed instructions for the background LLM
            notify_on_completion: Whether to notify the user when complete
            
        Returns:
            Dictionary with the task ID
            
        Raises:
            ToolError: If scheduling fails
        """
        with error_context(
            component_name=self.name,
            operation="scheduling async task",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Enhance the task prompt with explicit tool usage instructions
            enhanced_prompt = self._enhance_task_prompt(description, task_prompt)
            
            # Schedule the task using the client
            task_id = self.async_client.schedule_task(
                description=description,
                task_prompt=enhanced_prompt,
                notify_on_completion=notify_on_completion
            )
            
            return {"task_id": task_id}
    
    def _enhance_task_prompt(self, description: str, task_prompt: str) -> str:
        """
        Enhance the task prompt with explicit instructions for tool usage.
        
        Args:
            description: Task description
            task_prompt: Original task prompt
            
        Returns:
            Enhanced task prompt
        """
        # Define task-specific enhancements
        enhancements = []
        
        # Check for weather-related tasks
        if any(term in description.lower() or term in task_prompt.lower() 
               for term in ["weather", "temperature", "forecast", "climate"]):
            enhancements.append(
                "- Use tool_finder to enable the weather_tool first\n"
                "- Use weather_tool to get weather data for the location\n"
                "- Save results with the persistence tool"
            )
        
        # Check for questionnaire-related tasks
        elif any(term in description.lower() or term in task_prompt.lower() 
                for term in ["questionnaire", "survey", "form", "questions"]):
            enhancements.append(
                "- Use tool_finder to enable the questionnaire_tool first\n"
                "- Use questionnaire_tool to create or run questionnaires\n"
                "- Save results with the persistence tool"
            )
        
        # Add general instructions if no specific enhancement
        if not enhancements:
            enhancements.append(
                "- First use tool_finder to identify and enable any needed tools\n"
                "- Execute the required operations with the enabled tools\n"
                "- Always save your final results with the persistence tool"
            )
        
        # Create the enhanced prompt
        enhanced_prompt = f"{task_prompt}\n\n"
        enhanced_prompt += "IMPORTANT INSTRUCTIONS:\n"
        enhanced_prompt += enhancements[0]
        
        return enhanced_prompt


class CheckAsyncTaskTool(Tool):
    """
    Tool for checking the status of asynchronous tasks.
    
    Allows the LLM to query the progress and results of
    background tasks.
    """
    
    name = "check_async_task"
    description = "Check the status of an asynchronous task"
    usage_examples = [
        {
            "input": {"task_id": "123e4567-e89b-12d3-a456-426614174000"},
            "output": {
                "status": "completed",
                "description": "Generate report on customer data",
                "result": "Task completed and result saved to persistent/async_results/123e4567-e89b-12d3-a456-426614174000.json"
            }
        }
    ]
    
    def __init__(self, async_client: AsyncClient):
        """
        Initialize the check async task tool.
        
        Args:
            async_client: AsyncClient instance (required)
        """
        super().__init__()
        self.async_client = async_client
    
    def run(self, task_id: str) -> Dict[str, Any]:
        """
        Check the status of an asynchronous task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            Dictionary with task status information
            
        Raises:
            ToolError: If the task is not found or checking fails
        """
        with error_context(
            component_name=self.name,
            operation="checking task status",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Get task status from client
            status = self.async_client.get_task_status(task_id)
            
            # Format for user-friendly output
            formatted_status = {
                "task_id": task_id,
                "description": status.get("description", "No description"),
                "status": status.get("status", "unknown"),
            }
            
            # Add result if available
            if "result" in status:
                formatted_status["result"] = status["result"]
            
            # Add error if available
            if "error" in status and status["error"]:
                formatted_status["error"] = status["error"]
            
            # Add timing information
            if "created_at" in status:
                formatted_status["created_at"] = status["created_at"]
            if "completed_at" in status:
                formatted_status["completed_at"] = status["completed_at"]
            
            return formatted_status