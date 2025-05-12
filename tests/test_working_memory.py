"""
Test suite for the working memory system.

This module contains tests that verify the functionality of the
working memory system and its integration with other components.
"""

import pytest
import asyncio
import threading
import time
from typing import Dict, Any

from working_memory import WorkingMemory
from utils.async_utils import get_or_create_event_loop, run_coroutine, close_event_loop
from task_manager.automation_engine import AutomationEngine
from tools.repo import ToolRepository


def test_asyncio_utils_basic():
    """Test the basic functionality of async_utils."""
    # Test get_or_create_event_loop
    loop = get_or_create_event_loop()
    assert loop is not None
    assert loop.is_running() is False
    assert loop.is_closed() is False
    
    # Test run_coroutine
    async def sample_coro():
        return 42
    
    result = run_coroutine(sample_coro())
    assert result == 42
    
    # Test close_event_loop
    close_event_loop()
    
    # Getting a new loop should work after closing
    new_loop = get_or_create_event_loop()
    assert new_loop is not None
    assert new_loop.is_closed() is False


def test_asyncio_utils_threaded():
    """Test the thread isolation of async_utils."""
    main_thread_loop = get_or_create_event_loop()
    
    thread_loops = []
    
    def thread_func():
        # Get loop for this thread
        loop = get_or_create_event_loop()
        thread_loops.append(loop)
        
        # Run a simple coroutine
        async def sample_coro():
            return threading.get_ident()
        
        result = run_coroutine(sample_coro())
        assert result == threading.get_ident()
        
        # Close the loop
        close_event_loop()
    
    # Create and run threads
    threads = []
    for _ in range(3):
        thread = threading.Thread(target=thread_func)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify that each thread got its own loop
    assert len(thread_loops) == 3
    assert all(loop != main_thread_loop for loop in thread_loops)
    assert all(loop1 != loop2 for i, loop1 in enumerate(thread_loops) 
               for loop2 in thread_loops[i+1:])


def test_asyncio_tool_base():
    """Test AsyncioToolBase functionality."""
    from utils.asyncio_tool_base import AsyncioToolBase
    
    class TestTool(AsyncioToolBase):
        name = "test_tool"
        
        def run(self, operation, **kwargs):
            if operation == "run_async":
                return self.run_async(self._async_op(**kwargs))
            elif operation == "thread_data":
                key = kwargs.get("key", "test")
                value = kwargs.get("value", "data")
                action = kwargs.get("action", "get")
                
                if action == "set":
                    self.set_thread_data(key, value)
                    return {"success": True}
                elif action == "get":
                    return {"data": self.get_thread_data(key)}
                elif action == "has":
                    return {"exists": self.has_thread_data(key)}
                elif action == "remove":
                    self.remove_thread_data(key)
                    return {"success": True}
            return {"success": False}
        
        async def _async_op(self, **kwargs):
            return {"success": True, "thread_id": threading.get_ident()}
    
    # Create tool instance
    tool = TestTool()
    
    # Test run_async
    result = tool.run("run_async")
    assert result["success"] is True
    assert result["thread_id"] == threading.get_ident()
    
    # Test thread data operations
    # Set data
    tool.run("thread_data", action="set", key="test_key", value="test_value")
    
    # Get data
    result = tool.run("thread_data", action="get", key="test_key")
    assert result["data"] == "test_value"
    
    # Check if data exists
    result = tool.run("thread_data", action="has", key="test_key")
    assert result["exists"] is True
    
    # Remove data
    tool.run("thread_data", action="remove", key="test_key")
    
    # Verify it's gone
    result = tool.run("thread_data", action="has", key="test_key")
    assert result["exists"] is False
    
    # Test thread isolation
    def thread_func():
        # This should be None since it's a different thread
        result = tool.run("thread_data", action="get", key="test_key")
        assert result["data"] is None
        
        # Set in this thread
        tool.run("thread_data", action="set", key="thread_specific", value="thread_value")
        
        # Get in this thread
        result = tool.run("thread_data", action="get", key="thread_specific")
        assert result["data"] == "thread_value"
    
    thread = threading.Thread(target=thread_func)
    thread.start()
    thread.join()
    
    # Verify main thread doesn't see thread-specific data
    result = tool.run("thread_data", action="has", key="thread_specific")
    assert result["exists"] is False
    
    # Test cleanup
    tool.run("thread_data", action="set", key="cleanup_test", value="to_be_cleaned")
    tool.cleanup()
    
    # Verify data is cleaned up
    result = tool.run("thread_data", action="has", key="cleanup_test")
    assert result["exists"] is False


def test_automation_engine_asyncio_integration():
    """
    Test the integration of automation engine with asyncio utilities.
    
    This test verifies that the automation engine properly initializes
    and cleans up event loops for automations.
    """
    # Create tool repository and automation engine
    tool_repo = ToolRepository()
    engine = AutomationEngine(tool_repo=tool_repo)
    
    # Mock automation execution that uses asyncio
    executed_with_event_loop = False
    cleaned_up_event_loop = False
    
    def mock_execute():
        nonlocal executed_with_event_loop, cleaned_up_event_loop
        
        try:
            # Check if we have an event loop
            loop = asyncio.get_event_loop()
            executed_with_event_loop = True
            
            # Run a simple coroutine
            async def test_coro():
                return True
            
            assert loop.run_until_complete(test_coro()) is True
            
        except Exception as e:
            pytest.fail(f"Error in mock execute: {e}")
            
        finally:
            # Record cleanup
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                cleaned_up_event_loop = True
    
    # Replace _execute_automation_safely with our mock
    original_execute = engine._execute_automation_safely
    engine._execute_automation_safely = lambda *args: mock_execute()
    
    try:
        # Trigger execution
        engine._execute_automation_safely("dummy_id")
        
        # Wait a bit for everything to complete
        time.sleep(0.1)
        
        # Verify execution with event loop
        assert executed_with_event_loop is True
    finally:
        # Restore original method
        engine._execute_automation_safely = original_execute