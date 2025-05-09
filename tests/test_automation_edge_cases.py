"""
Edge case tests for the automation system.

This module provides tests that verify the automation system handles
edge cases and error scenarios correctly.
"""

import os
import json
import unittest
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import threading

import pytest

from task_manager.automation import (
    AutomationType, AutomationStatus, TaskFrequency, 
    Automation, AutomationStep, AutomationExecution, ExecutionMode
)
from task_manager.automation_engine import (
    AutomationEngine, get_automation_engine, initialize_automation_engine
)
from tools.automation_tool import AutomationTool
from errors import ToolError, ErrorCode
from db import Database


class TestAutomationEdgeCases(unittest.TestCase):
    """Edge case tests for the automation system."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test class."""
        # Use a temporary database for testing
        os.environ["TEST_MODE"] = "true"
        os.environ["TEST_DB_PATH"] = ":memory:"
        
        # Initialize the database
        cls.db = Database()
        cls.db.create_tables()
        
        # Initialize the automation engine
        cls.engine = initialize_automation_engine()
        
        # Create automation tool instance
        cls.tool = AutomationTool()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Stop the scheduler to prevent background tasks
        cls.engine.stop_scheduler()
        
        # Remove environment variables
        del os.environ["TEST_MODE"]
        del os.environ["TEST_DB_PATH"]
    
    def setUp(self):
        """Set up test fixtures for each test."""
        # Clean up any automations from previous tests
        with self.db.get_session() as session:
            session.query(Automation).delete()
            session.query(AutomationStep).delete()
            session.query(AutomationExecution).delete()
            session.commit()
    
    def test_automation_with_very_long_name(self):
        """Test creating an automation with a very long name."""
        # Create a name that's 1000 characters long
        very_long_name = "A" * 1000
        
        # Create automation with very long name
        create_result = self.tool.run(
            operation="create_automation",
            name=very_long_name,
            type="simple_task",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Test message"}
        )
        
        # Verify the automation was created and name was stored correctly
        self.assertEqual(create_result["automation"]["name"], very_long_name)
    
    def test_automation_with_far_future_date(self):
        """Test creating an automation with a scheduled time far in the future."""
        # Create a date 10 years in the future
        future_date = datetime.now(timezone.utc) + timedelta(days=365 * 10)
        
        # Create automation with future date
        create_result = self.tool.run(
            operation="create_automation",
            name="Far Future Automation",
            type="simple_task",
            frequency="once",
            scheduled_time=future_date.isoformat(),
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Future message"}
        )
        
        # Verify the automation was created and date was stored correctly
        scheduled_time = datetime.fromisoformat(create_result["automation"]["scheduled_time"])
        self.assertGreaterEqual(scheduled_time.year, future_date.year)
    
    def test_automation_with_past_date(self):
        """Test creating an automation with a scheduled time in the past."""
        # Create a date in the past
        past_date = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Create automation with past date
        create_result = self.tool.run(
            operation="create_automation",
            name="Past Date Automation",
            type="simple_task",
            frequency="once",
            scheduled_time=past_date.isoformat(),
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Past message"}
        )
        
        # For one-time automations with past dates, the automation should be created
        # but will execute as soon as possible
        self.assertEqual(create_result["automation"]["status"], "active")
    
    def test_sequence_with_100_steps(self):
        """Test creating and executing a sequence with a large number of steps."""
        # Create step definitions
        steps = []
        for i in range(1, 101):
            steps.append({
                "name": f"Step {i}",
                "position": i,
                "execution_mode": "direct",
                "tool_name": "http_tool",
                "operation": "echo",
                "parameters": {"message": f"Message from step {i}"},
                "output_key": f"step{i}_result"
            })
        
        # Create automation with 100 steps
        create_result = self.tool.run(
            operation="create_automation",
            name="Many Steps Sequence",
            type="sequence",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            steps=steps
        )
        
        # Verify all steps were created
        self.assertEqual(len(create_result["automation"]["steps"]), 100)
        
        # Execute the automation
        execute_result = self.tool.run(
            operation="execute_now",
            automation_id=create_result["automation"]["id"]
        )
        
        # Allow sufficient time for execution (this might need to be adjusted)
        time.sleep(5)
        
        # Get execution details
        details_result = self.tool.run(
            operation="get_execution_details",
            execution_id=execute_result["execution_id"]
        )
        
        # Verify all steps were executed
        self.assertEqual(len(details_result["execution"]["steps"]), 100)
        
        # Check the final context has all step results
        context = details_result["execution"]["context"]
        self.assertEqual(len(context), 100)
    
    def test_concurrent_executions_of_same_automation(self):
        """Test attempting to execute the same automation concurrently."""
        # Create an automation
        create_result = self.tool.run(
            operation="create_automation",
            name="Concurrent Test",
            type="sequence",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            steps=[
                {
                    "name": "Long Running Step",
                    "position": 1,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "This step takes time"},
                    "output_key": "step_result"
                }
            ]
        )
        
        automation_id = create_result["automation"]["id"]
        
        # Start first execution
        execute_result1 = self.tool.run(
            operation="execute_now",
            automation_id=automation_id
        )
        
        # Immediately try to start a second execution
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                operation="execute_now",
                automation_id=automation_id
            )
        
        # Verify the error is about concurrent execution
        self.assertEqual(context.exception.code, ErrorCode.TOOL_EXECUTION_ERROR)
        self.assertIn("already running", str(context.exception))
    
    def test_automation_with_empty_parameters(self):
        """Test creating an automation with empty parameters."""
        # Create automation with empty parameters
        create_result = self.tool.run(
            operation="create_automation",
            name="Empty Params Test",
            type="simple_task",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={}
        )
        
        # Verify the automation was created with empty parameters
        self.assertEqual(create_result["automation"]["parameters"], {})
        
        # Execute the automation
        execute_result = self.tool.run(
            operation="execute_now",
            automation_id=create_result["automation"]["id"]
        )
        
        # Allow time for execution to complete
        time.sleep(1)
        
        # Get execution details
        details_result = self.tool.run(
            operation="get_execution_details",
            execution_id=execute_result["execution_id"]
        )
        
        # Verify the execution completed (might succeed or fail depending on the tool)
        self.assertIn(details_result["execution"]["status"], ["completed", "failed"])
    
    def test_sequence_with_circular_references(self):
        """Test a sequence with circular references in parameter templates."""
        # Create a sequence with steps that have circular template references
        create_result = self.tool.run(
            operation="create_automation",
            name="Circular Reference Test",
            type="sequence",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            steps=[
                {
                    "name": "Step 1",
                    "position": 1,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "Initial message"},
                    "output_key": "step1_result"
                },
                {
                    "name": "Step 2",
                    "position": 2,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "{step3_result.message}"},  # Reference to step 3 which doesn't exist yet
                    "output_key": "step2_result"
                },
                {
                    "name": "Step 3",
                    "position": 3,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "{step2_result.message}"},  # Reference to step 2
                    "output_key": "step3_result"
                }
            ]
        )
        
        # Execute the automation
        execute_result = self.tool.run(
            operation="execute_now",
            automation_id=create_result["automation"]["id"]
        )
        
        # Allow time for execution to complete
        time.sleep(2)
        
        # Get execution details
        details_result = self.tool.run(
            operation="get_execution_details",
            execution_id=execute_result["execution_id"]
        )
        
        # Verify the steps with circular references failed
        steps = details_result["execution"]["steps"]
        self.assertEqual(steps[0]["status"], "completed")  # First step should succeed
        
        # At least one of the steps with circular references should fail
        self.assertTrue(
            steps[1]["status"] == "failed" or steps[2]["status"] == "failed",
            "At least one step with circular references should fail"
        )
    
    def test_update_nonexistent_automation(self):
        """Test updating a non-existent automation."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                operation="update_automation",
                automation_id="nonexistent_id",
                name="New Name"
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_NOT_FOUND)
    
    def test_execute_nonexistent_automation(self):
        """Test executing a non-existent automation."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                operation="execute_now",
                automation_id="nonexistent_id"
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_EXECUTION_ERROR)
    
    def test_invalid_frequency_values(self):
        """Test creating automations with invalid frequency values."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                operation="create_automation",
                name="Invalid Frequency Test",
                type="simple_task",
                frequency="every_decade",  # Invalid frequency
                scheduled_time=datetime.now(timezone.utc).isoformat(),
                execution_mode="direct",
                tool_name="http_tool",
                operation="echo",
                parameters={"message": "Test message"}
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
    
    def test_invalid_day_of_week(self):
        """Test creating weekly automation with invalid day of week."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                operation="create_automation",
                name="Invalid Day Test",
                type="simple_task",
                frequency="weekly",
                day_of_week=8,  # Invalid (valid range is 0-6)
                scheduled_time=datetime.now(timezone.utc).isoformat(),
                execution_mode="direct",
                tool_name="http_tool",
                operation="echo",
                parameters={"message": "Test message"}
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_EXECUTION_ERROR)
    
    def test_invalid_day_of_month(self):
        """Test creating monthly automation with invalid day of month."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                operation="create_automation",
                name="Invalid Day Test",
                type="simple_task",
                frequency="monthly",
                day_of_month=32,  # Invalid (valid range is 1-31)
                scheduled_time=datetime.now(timezone.utc).isoformat(),
                execution_mode="direct",
                tool_name="http_tool",
                operation="echo",
                parameters={"message": "Test message"}
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_EXECUTION_ERROR)
    
    def test_leap_year_day_handling(self):
        """Test handling of February 29 in non-leap years."""
        # Create a monthly automation scheduled for the 29th
        create_result = self.tool.run(
            operation="create_automation",
            name="Leap Year Test",
            type="simple_task",
            frequency="monthly",
            day_of_month=29,
            scheduled_time="12:00:00",
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Leap year test"}
        )
        
        # The system should accept this configuration
        self.assertEqual(create_result["automation"]["day_of_month"], 29)
        
        # The _get_valid_day_in_month method should handle this edge case when calculating
        # next run times, but we can't easily test that here directly.
    
    def test_automation_with_unicode_characters(self):
        """Test creating an automation with Unicode characters in fields."""
        # Create automation with Unicode characters
        create_result = self.tool.run(
            operation="create_automation",
            name="Unicode Test 🌍 🚀 💻",
            type="simple_task",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Unicode message: 你好，世界！"}
        )
        
        # Verify the automation was created with Unicode characters
        self.assertEqual(create_result["automation"]["name"], "Unicode Test 🌍 🚀 💻")
        self.assertEqual(create_result["automation"]["parameters"]["message"], "Unicode message: 你好，世界！")
    
    def test_automation_with_special_characters(self):
        """Test creating an automation with special characters in fields."""
        # Create automation with special characters
        create_result = self.tool.run(
            operation="create_automation",
            name="Special Chars Test !@#$%^&*()_+-=[]{}|;':\",./<>?",
            type="simple_task",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"}
        )
        
        # Verify the automation was created with special characters
        self.assertEqual(
            create_result["automation"]["name"],
            "Special Chars Test !@#$%^&*()_+-=[]{}|;':\",./<>?"
        )
    
    def test_sequence_with_template_escaping(self):
        """Test sequence with template syntax that should be escaped."""
        # Create a sequence with parameters that include template-like syntax
        create_result = self.tool.run(
            operation="create_automation",
            name="Template Escaping Test",
            type="sequence",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            steps=[
                {
                    "name": "Step 1",
                    "position": 1,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "This is a template-like string: {{not_a_template}}"},
                    "output_key": "step1_result"
                },
                {
                    "name": "Step 2",
                    "position": 2,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "{step1_result.message}"},
                    "output_key": "step2_result"
                }
            ]
        )
        
        # Execute the automation
        execute_result = self.tool.run(
            operation="execute_now",
            automation_id=create_result["automation"]["id"]
        )
        
        # Allow time for execution to complete
        time.sleep(2)
        
        # Get execution details
        details_result = self.tool.run(
            operation="get_execution_details",
            execution_id=execute_result["execution_id"]
        )
        
        # Verify the template was handled correctly
        context = details_result["execution"]["context"]
        self.assertEqual(
            context["step1_result"]["message"],
            "This is a template-like string: {{not_a_template}}"
        )
        self.assertEqual(
            context["step2_result"]["message"],
            "This is a template-like string: {{not_a_template}}"
        )
    
    def test_nonexistent_tool_in_automation(self):
        """Test creating an automation that references a non-existent tool."""
        # Create automation with non-existent tool
        create_result = self.tool.run(
            operation="create_automation",
            name="Non-existent Tool Test",
            type="simple_task",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            execution_mode="direct",
            tool_name="nonexistent_tool",
            operation="echo",
            parameters={"message": "Test message"}
        )
        
        # The automation should be created successfully
        self.assertEqual(create_result["automation"]["tool_name"], "nonexistent_tool")
        
        # Execute the automation (should fail during execution)
        execute_result = self.tool.run(
            operation="execute_now",
            automation_id=create_result["automation"]["id"]
        )
        
        # Allow time for execution to complete
        time.sleep(1)
        
        # Get execution details
        details_result = self.tool.run(
            operation="get_execution_details",
            execution_id=execute_result["execution_id"]
        )
        
        # Verify the execution failed
        self.assertEqual(details_result["execution"]["status"], "failed")
        self.assertIn("error", details_result["execution"])
    
    def test_very_large_data_passing(self):
        """Test sequence with very large data passing between steps."""
        # Create a large data string (100 KB)
        large_data = "X" * (100 * 1024)
        
        # Create a sequence that passes large data between steps
        create_result = self.tool.run(
            operation="create_automation",
            name="Large Data Test",
            type="sequence",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            steps=[
                {
                    "name": "Generate Large Data",
                    "position": 1,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": large_data},
                    "output_key": "large_data"
                },
                {
                    "name": "Process Large Data",
                    "position": 2,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "Received large data: {large_data.message}"},
                    "output_key": "processed_data"
                }
            ]
        )
        
        # Execute the automation
        execute_result = self.tool.run(
            operation="execute_now",
            automation_id=create_result["automation"]["id"]
        )
        
        # Allow time for execution to complete
        time.sleep(2)
        
        # Get execution details
        details_result = self.tool.run(
            operation="get_execution_details",
            execution_id=execute_result["execution_id"]
        )
        
        # Verify the execution worked with large data
        # The second step might truncate the output, but it should have run
        self.assertEqual(details_result["execution"]["steps"][0]["status"], "completed")
        self.assertEqual(details_result["execution"]["steps"][1]["status"], "completed")


if __name__ == "__main__":
    unittest.main()