"""
Integration tests for the automation system.

This module provides tests that verify the automation system components
work together correctly in real-world scenarios.
"""

import os
import json
import unittest
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import pytest

from task_manager.automation import (
    AutomationType, AutomationStatus, TaskFrequency, 
    Automation, AutomationStep, AutomationExecution,
    ExecutionMode, ExecutionStatus, TriggerType, StepExecutionStatus
)
from task_manager.automation_engine import (
    AutomationEngine, get_automation_engine, initialize_automation_engine
)
from tools.automation_tool import AutomationTool
from errors import ToolError, ErrorCode
from db import Database


class TestAutomationIntegration(unittest.TestCase):
    """Integration tests for the automation system."""

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
    
    def test_create_and_execute_simple_task(self):
        """
        Test creating and executing a simple direct task automation.
        
        This test verifies the end-to-end flow of:
        1. Creating a simple task automation
        2. Executing it immediately
        3. Verifying the execution was successful
        """
        # Create a simple task automation using the tool
        create_result = self.tool.run(
            operation="create_automation",
            name="Test Simple Task",
            type="simple_task",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            execution_mode="direct",
            tool_name="http_tool",
            tool_operation="echo",
            parameters={"message": "Hello World"}
        )
        
        # Get the automation ID
        automation_id = create_result["automation"]["id"]
        
        # Execute the automation immediately
        execute_result = self.tool.run(
            operation="execute_now",
            automation_id=automation_id
        )
        
        # Get the execution ID
        execution_id = execute_result["execution_id"]
        
        # Allow time for execution to complete
        time.sleep(1)
        
        # Get execution details
        details_result = self.tool.run(
            operation="get_execution_details",
            execution_id=execution_id
        )
        
        # Verify the execution completed successfully
        self.assertEqual(details_result["execution"]["status"], "completed")
        
        # Verify the execution result contains the expected output
        self.assertIn("result", details_result["execution"])
        self.assertIn("message", details_result["execution"]["result"])
        self.assertEqual(details_result["execution"]["result"]["message"], "Hello World")
    
    def test_create_and_execute_sequence(self):
        """
        Test creating and executing a sequence automation.
        
        This test verifies the end-to-end flow of:
        1. Creating a sequence automation with multiple steps
        2. Executing it immediately
        3. Verifying each step was executed in order and with correct results
        4. Verifying data passing between steps works correctly
        """
        # Create a sequence automation using the tool
        create_result = self.tool.run(
            operation="create_automation",
            name="Test Sequence",
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
                    "parameters": {"message": "Step 1 output"},
                    "output_key": "step1_result"
                },
                {
                    "name": "Step 2",
                    "position": 2,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "{step1_result.message} processed"},
                    "output_key": "step2_result"
                }
            ]
        )
        
        # Get the automation ID
        automation_id = create_result["automation"]["id"]
        
        # Execute the automation immediately
        execute_result = self.tool.run(
            operation="execute_now",
            automation_id=automation_id
        )
        
        # Get the execution ID
        execution_id = execute_result["execution_id"]
        
        # Allow time for execution to complete
        time.sleep(2)
        
        # Get execution details
        details_result = self.tool.run(
            operation="get_execution_details",
            execution_id=execution_id
        )
        
        # Verify the execution completed successfully
        self.assertEqual(details_result["execution"]["status"], "completed")
        
        # Verify each step executed correctly
        for step in details_result["execution"]["steps"]:
            self.assertEqual(step["status"], "completed")
        
        # Verify data passing between steps worked
        context = details_result["execution"]["context"]
        self.assertIn("step1_result", context)
        self.assertIn("step2_result", context)
        self.assertEqual(context["step1_result"]["message"], "Step 1 output")
        self.assertEqual(context["step2_result"]["message"], "Step 1 output processed")
    
    def test_scheduled_execution(self):
        """
        Test scheduled execution of an automation.
        
        This test verifies:
        1. Creating an automation scheduled to run in the near future
        2. Verifying it runs at the scheduled time
        3. Verifying the next_execution_time is calculated correctly
        """
        # Schedule for 2 seconds in the future
        scheduled_time = datetime.now(timezone.utc) + timedelta(seconds=2)
        
        # Create a simple task automation using the tool
        create_result = self.tool.run(
            operation="create_automation",
            name="Test Scheduled Task",
            type="simple_task",
            frequency="once",
            scheduled_time=scheduled_time.isoformat(),
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Scheduled execution"}
        )
        
        # Get the automation ID
        automation_id = create_result["automation"]["id"]
        
        # Verify the scheduled time was set correctly
        self.assertIn("next_execution_time", create_result["automation"])
        
        # Wait for the scheduled time plus a bit more
        time.sleep(3)
        
        # Get the automation executions
        executions_result = self.tool.run(
            operation="get_executions",
            automation_id=automation_id
        )
        
        # Verify there was an execution
        self.assertGreater(executions_result["count"], 0)
        
        # Get the details of the first execution
        execution_id = executions_result["executions"][0]["id"]
        details_result = self.tool.run(
            operation="get_execution_details",
            execution_id=execution_id
        )
        
        # Verify the execution completed successfully
        self.assertEqual(details_result["execution"]["status"], "completed")
        
        # Verify the execution has the correct trigger type
        self.assertEqual(details_result["execution"]["trigger_type"], "scheduled")
    
    def test_pause_and_resume(self):
        """
        Test pausing and resuming an automation.
        
        This test verifies:
        1. Creating a scheduled automation
        2. Pausing it
        3. Verifying it's status is updated
        4. Resuming it
        5. Verifying the status is updated again
        """
        # Schedule for 1 hour in the future to avoid actual execution
        scheduled_time = datetime.now(timezone.utc) + timedelta(hours=1)
        
        # Create a recurring automation
        create_result = self.tool.run(
            operation="create_automation",
            name="Test Pause Resume",
            type="simple_task",
            frequency="daily",
            scheduled_time=scheduled_time.isoformat(),
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Daily execution"}
        )
        
        # Get the automation ID
        automation_id = create_result["automation"]["id"]
        
        # Pause the automation
        pause_result = self.tool.run(
            operation="pause_automation",
            automation_id=automation_id
        )
        
        # Verify the pause was successful
        self.assertIn("message", pause_result)
        self.assertIn("paused", pause_result["message"])
        
        # Get the automation to verify its status
        get_result = self.tool.run(
            operation="get_automation",
            automation_id=automation_id
        )
        
        # Verify the status is paused
        self.assertEqual(get_result["automation"]["status"], "paused")
        
        # Resume the automation
        resume_result = self.tool.run(
            operation="resume_automation",
            automation_id=automation_id
        )
        
        # Verify the resume was successful
        self.assertIn("message", resume_result)
        self.assertIn("resumed", resume_result["message"].lower())
        
        # Get the automation again to verify its status
        get_result = self.tool.run(
            operation="get_automation",
            automation_id=automation_id
        )
        
        # Verify the status is active again
        self.assertEqual(get_result["automation"]["status"], "active")
    
    def test_conditional_execution(self):
        """
        Test conditional execution of steps in a sequence.
        
        This test verifies:
        1. Creating a sequence with steps that have conditions
        2. Executing it with different initial contexts
        3. Verifying steps are executed or skipped based on conditions
        """
        # Create a sequence with conditional steps
        create_result = self.tool.run(
            operation="create_automation",
            name="Conditional Sequence",
            type="sequence",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            steps=[
                {
                    "name": "First Step",
                    "position": 1,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "First step output"},
                    "output_key": "first_result",
                    "condition_type": "always"
                },
                {
                    "name": "Conditional Step",
                    "position": 2,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "Conditional step executed"},
                    "output_key": "conditional_result",
                    "condition_type": "if_data",
                    "condition_data_key": "should_execute"
                }
            ]
        )
        
        # Get the automation ID
        automation_id = create_result["automation"]["id"]
        
        # Execute with condition not met
        execute_result1 = self.tool.run(
            operation="execute_now",
            automation_id=automation_id,
            initial_context={}
        )
        
        # Get execution details
        execution_id1 = execute_result1["execution_id"]
        time.sleep(1)
        details_result1 = self.tool.run(
            operation="get_execution_details",
            execution_id=execution_id1
        )
        
        # Verify first step executed, conditional step skipped
        steps1 = details_result1["execution"]["steps"]
        self.assertEqual(len(steps1), 2)
        self.assertEqual(steps1[0]["status"], "completed")
        self.assertEqual(steps1[1]["status"], "skipped")
        
        # Execute with condition met
        execute_result2 = self.tool.run(
            operation="execute_now",
            automation_id=automation_id,
            initial_context={"should_execute": True}
        )
        
        # Get execution details
        execution_id2 = execute_result2["execution_id"]
        time.sleep(1)
        details_result2 = self.tool.run(
            operation="get_execution_details",
            execution_id=execution_id2
        )
        
        # Verify both steps executed
        steps2 = details_result2["execution"]["steps"]
        self.assertEqual(len(steps2), 2)
        self.assertEqual(steps2[0]["status"], "completed")
        self.assertEqual(steps2[1]["status"], "completed")
    
    def test_error_handling(self):
        """
        Test error handling policies in a sequence.
        
        This test verifies:
        1. Creating a sequence with steps and error policies
        2. Executing it with steps that will fail
        3. Verifying error handling works according to policies
        """
        # Create a sequence with error handling
        create_result = self.tool.run(
            operation="create_automation",
            name="Error Handling Sequence",
            type="sequence",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            error_policy="continue",  # Continue on error
            steps=[
                {
                    "name": "First Step",
                    "position": 1,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "First step output"},
                    "output_key": "first_result"
                },
                {
                    "name": "Failing Step",
                    "position": 2,
                    "execution_mode": "direct",
                    "tool_name": "non_existent_tool",  # This will fail
                    "operation": "echo",
                    "parameters": {"message": "This should fail"},
                    "output_key": "failing_result"
                },
                {
                    "name": "After Error Step",
                    "position": 3,
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "After error output"},
                    "output_key": "after_error_result"
                }
            ]
        )
        
        # Get the automation ID
        automation_id = create_result["automation"]["id"]
        
        # Execute the automation
        execute_result = self.tool.run(
            operation="execute_now",
            automation_id=automation_id
        )
        
        # Get execution details
        execution_id = execute_result["execution_id"]
        time.sleep(1)
        details_result = self.tool.run(
            operation="get_execution_details",
            execution_id=execution_id
        )
        
        # Verify the overall execution completed (with errors)
        self.assertEqual(details_result["execution"]["status"], "completed")
        
        # Verify the step statuses
        steps = details_result["execution"]["steps"]
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]["status"], "completed")  # First step succeeds
        self.assertEqual(steps[1]["status"], "failed")     # Second step fails
        self.assertEqual(steps[2]["status"], "completed")  # Third step runs due to continue policy
    
    def test_update_automation(self):
        """
        Test updating an automation.
        
        This test verifies:
        1. Creating an automation
        2. Updating various fields
        3. Verifying the updates are applied correctly
        """
        # Create a simple automation
        create_result = self.tool.run(
            operation="create_automation",
            name="Update Test",
            type="simple_task",
            frequency="daily",
            scheduled_time="12:00:00",
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Original message"}
        )
        
        # Get the automation ID
        automation_id = create_result["automation"]["id"]
        
        # Update various fields
        update_result = self.tool.run(
            operation="update_automation",
            automation_id=automation_id,
            name="Updated Name",
            frequency="weekly",
            day_of_week=3,  # Wednesday
            scheduled_time="15:00:00",
            parameters={"message": "Updated message"}
        )
        
        # Verify the update was successful
        self.assertEqual(update_result["automation"]["name"], "Updated Name")
        self.assertEqual(update_result["automation"]["frequency"], "weekly")
        self.assertEqual(update_result["automation"]["day_of_week"], 3)
        
        # Check the parameters were updated
        self.assertEqual(update_result["automation"]["parameters"]["message"], "Updated message")
        
        # Get the automation to double-check
        get_result = self.tool.run(
            operation="get_automation",
            automation_id=automation_id
        )
        
        # Verify the updates were really applied
        self.assertEqual(get_result["automation"]["name"], "Updated Name")
        self.assertEqual(get_result["automation"]["frequency"], "weekly")
    
    def test_delete_automation(self):
        """
        Test deleting an automation.
        
        This test verifies:
        1. Creating an automation
        2. Deleting it
        3. Verifying it can no longer be found
        """
        # Create a simple automation
        create_result = self.tool.run(
            operation="create_automation",
            name="Delete Test",
            type="simple_task",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "To be deleted"}
        )
        
        # Get the automation ID
        automation_id = create_result["automation"]["id"]
        
        # Delete the automation
        delete_result = self.tool.run(
            operation="delete_automation",
            automation_id=automation_id,
            confirm=True
        )
        
        # Verify the delete was successful
        self.assertEqual(delete_result["automation_id"], automation_id)
        self.assertIn("deleted", delete_result["message"])
        
        # Try to get the deleted automation
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                operation="get_automation",
                automation_id=automation_id
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_NOT_FOUND)
    
    def test_list_and_filter_automations(self):
        """
        Test listing and filtering automations.
        
        This test verifies:
        1. Creating multiple automations of different types
        2. Listing all automations
        3. Filtering by type, status, and frequency
        """
        # Create a simple task automation
        self.tool.run(
            operation="create_automation",
            name="Simple Task",
            type="simple_task",
            frequency="daily",
            scheduled_time="12:00:00",
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Simple task message"}
        )
        
        # Create a sequence automation
        self.tool.run(
            operation="create_automation",
            name="Sequence",
            type="sequence",
            frequency="weekly",
            day_of_week=1,
            scheduled_time="15:00:00",
            steps=[
                {
                    "name": "Step 1",
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": "Step 1 message"},
                    "output_key": "step1_result"
                }
            ]
        )
        
        # Create a paused automation
        paused_result = self.tool.run(
            operation="create_automation",
            name="Paused Task",
            type="simple_task",
            frequency="monthly",
            day_of_month=15,
            scheduled_time="09:00:00",
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Paused task message"}
        )
        
        # Pause the automation
        self.tool.run(
            operation="pause_automation",
            automation_id=paused_result["automation"]["id"]
        )
        
        # List all automations
        list_all_result = self.tool.run(
            operation="get_automations"
        )
        
        # Verify all automations are listed
        self.assertEqual(list_all_result["count"], 3)
        
        # Filter by type
        list_simple_result = self.tool.run(
            operation="get_automations",
            type="simple_task"
        )
        
        # Verify only simple tasks are listed
        self.assertEqual(list_simple_result["count"], 2)
        for automation in list_simple_result["automations"]:
            self.assertEqual(automation["type"], "simple_task")
        
        # Filter by status
        list_paused_result = self.tool.run(
            operation="get_automations",
            status="paused"
        )
        
        # Verify only paused automations are listed
        self.assertEqual(list_paused_result["count"], 1)
        self.assertEqual(list_paused_result["automations"][0]["name"], "Paused Task")
        
        # Filter by frequency
        list_weekly_result = self.tool.run(
            operation="get_automations",
            frequency="weekly"
        )
        
        # Verify only weekly automations are listed
        self.assertEqual(list_weekly_result["count"], 1)
        self.assertEqual(list_weekly_result["automations"][0]["frequency"], "weekly")
        
        # Combine filters
        list_combined_result = self.tool.run(
            operation="get_automations",
            type="simple_task",
            status="active"
        )
        
        # Verify filtering works correctly
        self.assertEqual(list_combined_result["count"], 1)
        self.assertEqual(list_combined_result["automations"][0]["name"], "Simple Task")


if __name__ == "__main__":
    unittest.main()