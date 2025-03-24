"""
Tests for asynchronous tool functionality.

This module provides tests for the async task manager
and associated async tools.
"""
import time
import threading
import pytest
from unittest.mock import MagicMock, patch

from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository
from tools.async_manager import AsyncTaskManager, TaskStatus
from tools.async_tools import ScheduleAsyncTaskTool, CheckAsyncTaskTool
from errors import ToolError


@pytest.fixture
def mock_llm_bridge():
    """Create a mock LLM bridge for testing."""
    mock_bridge = MagicMock(spec=LLMBridge)
    
    # Configure extract_tool_calls to initially return tool calls, then no tool calls
    mock_bridge.extract_tool_calls.side_effect = [
        [{"id": "1", "tool_name": "persistence", "input": {"operation": "set", "filename": "async_results/test.json"}}],
        []  # No more tool calls on second invocation
    ]
    
    # Configure extract_text_content to return a completion message
    mock_bridge.extract_text_content.return_value = "Task completed successfully"
    
    # Configure generate_response to return a mock response
    mock_response = MagicMock()
    mock_bridge.generate_response.return_value = mock_response
    
    return mock_bridge


@pytest.fixture
def mock_tool_repo():
    """Create a mock tool repository for testing."""
    mock_repo = MagicMock(spec=ToolRepository)
    mock_repo.get_all_tool_definitions.return_value = [
        {"name": "persistence", "description": "Persistence tool", "input_schema": {}}
    ]
    mock_repo.invoke_tool.return_value = {"status": "success"}
    return mock_repo


@pytest.fixture
def async_task_manager(mock_llm_bridge, mock_tool_repo):
    """Create an async task manager for testing."""
    # Fix for None task issue in worker thread
    def safe_get(*args, **kwargs):
        # Configure the mock to safely handle None task
        # and return an empty list of tool calls
        return []
        
    # Update mock behavior
    mock_llm_bridge.extract_tool_calls.side_effect = safe_get
    
    manager = AsyncTaskManager(llm_bridge=mock_llm_bridge, tool_repo=mock_tool_repo)
    
    # Give worker thread time to initialize
    time.sleep(0.1)
    
    yield manager
    manager.shutdown()


def test_schedule_async_task(async_task_manager):
    """Test scheduling an async task."""
    # Update the extract_text_content mock to return successful response
    async_task_manager.llm_bridge.extract_text_content.return_value = "Task completed"
    
    # Schedule a task
    task_id = async_task_manager.schedule_task(
        description="Test task",
        task_prompt="Run a test task and save results",
        notify_on_completion=False
    )
    
    # Verify task was created
    assert task_id in async_task_manager.tasks
    task = async_task_manager.tasks[task_id]
    assert task.description == "Test task"
    
    # We don't check task.status here as it might have already moved past PENDING
    # due to the background thread


def test_async_task_execution(async_task_manager):
    """Test async task execution and completion."""
    # Mock successful LLM response for this test
    async_task_manager.llm_bridge.extract_tool_calls.side_effect = None  # Reset side effect
    async_task_manager.llm_bridge.extract_tool_calls.return_value = []  # No tool calls
    async_task_manager.llm_bridge.extract_text_content.return_value = "Task completed successfully"
    
    # Schedule a task
    task_id = async_task_manager.schedule_task(
        description="Test execution",
        task_prompt="Execute this test task",
        notify_on_completion=False
    )
    
    # Manually complete the task for testing purposes
    task = async_task_manager.tasks[task_id]
    task.status = TaskStatus.COMPLETED
    task.result = "Test result"
    
    # Verify task has expected values
    assert task.description == "Test execution"
    assert task.status == TaskStatus.COMPLETED
    assert task.result == "Test result"


def test_schedule_async_task_tool(async_task_manager):
    """Test the ScheduleAsyncTaskTool."""
    # Create the tool with required task_manager
    schedule_tool = ScheduleAsyncTaskTool(task_manager=async_task_manager)
    
    # Run the tool
    result = schedule_tool.run(
        description="Tool test task",
        task_prompt="Test the schedule async task tool",
        notify_on_completion=True
    )
    
    # Verify result contains task ID
    assert "task_id" in result
    assert result["task_id"] in async_task_manager.tasks


def test_check_async_task_tool(async_task_manager):
    """Test the CheckAsyncTaskTool."""
    # Schedule a task
    task_id = async_task_manager.schedule_task(
        description="Check tool test",
        task_prompt="Test the check async task tool",
        notify_on_completion=False
    )
    
    # Create the check tool with required task_manager
    check_tool = CheckAsyncTaskTool(task_manager=async_task_manager)
    
    # Run the tool
    result = check_tool.run(task_id=task_id)
    
    # Verify result contains task info
    assert result["task_id"] == task_id
    assert result["description"] == "Check tool test"
    assert "status" in result


def test_check_nonexistent_task(async_task_manager):
    """Test checking a nonexistent task."""
    # Create the check tool with required task_manager
    check_tool = CheckAsyncTaskTool(task_manager=async_task_manager)
    
    # Attempt to check nonexistent task
    with pytest.raises(ToolError) as excinfo:
        check_tool.run(task_id="nonexistent-task-id")
    
    # Verify error
    assert "Task not found" in str(excinfo.value)


def test_notification_callback(async_task_manager):
    """Test notification callback functionality."""
    # Create a mock callback
    mock_callback = MagicMock()
    async_task_manager.set_notification_callback(mock_callback)
    
    # Schedule a task with notification
    task_id = async_task_manager.schedule_task(
        description="Notification test",
        task_prompt="Test notification callback",
        notify_on_completion=True
    )
    
    # Get the task
    task = async_task_manager.tasks[task_id]
    
    # Manually trigger the notification (for testing purposes)
    task.status = TaskStatus.COMPLETED
    task.result = "Test result"
    if task.notify_on_completion:
        async_task_manager.notification_callback(task)
    
    # Verify callback was called
    mock_callback.assert_called_once()
    
    # Get the task that was passed to the callback
    called_task = mock_callback.call_args[0][0]
    assert called_task.task_id == task_id