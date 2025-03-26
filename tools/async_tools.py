"""
Tools for working with asynchronous tasks.

This module provides tools for scheduling background tasks
and checking their status.
"""
from typing import Dict, Any

from tools.repo import Tool
from tools.async_manager import AsyncTaskManager
from errors import ToolError, ErrorCode


class ScheduleAsyncTaskTool(Tool):
    """
    Tool for scheduling asynchronous tasks to run in the background.

    Allows the LLM to delegate complex operations to run
    in the background without blocking the conversation.
    """

    name = "schedule_async_task"
    description = "Schedule a task to run asynchronously in the background"
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

    def __init__(self, task_manager: AsyncTaskManager):
        """
        Initialize the schedule async task tool.

        Args:
            task_manager: Async task manager instance (required)
        """
        super().__init__()
        self.task_manager = task_manager

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

        try:
            task_id = self.task_manager.schedule_task(
                description=description,
                task_prompt=task_prompt,
                notify_on_completion=notify_on_completion
            )

            return {"task_id": task_id}

        except Exception as e:
            raise ToolError(
                f"Error scheduling async task: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )


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

    def __init__(self, task_manager: AsyncTaskManager):
        """
        Initialize the check async task tool.

        Args:
            task_manager: Async task manager instance (required)
        """
        super().__init__()
        self.task_manager = task_manager

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

        try:
            task_status = self.task_manager.get_task_status(task_id)
            return task_status

        except Exception as e:
            if isinstance(e, ToolError):
                raise

            raise ToolError(
                f"Error checking task status: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )
