"""
Tests for the automation tool.

This module provides comprehensive test coverage for the automation tool,
ensuring it correctly handles both normal cases and edge cases.
"""

import json
import unittest
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta, timezone
import pytest
from typing import Dict, Any, List

from tools.automation_tool import AutomationTool
from task_manager.automation import (
    AutomationType, AutomationStatus, TaskFrequency, 
    Automation, AutomationStep, AutomationExecution
)
from task_manager.automation_engine import AutomationEngine
from errors import ToolError, ErrorCode


class TestAutomationTool(unittest.TestCase):
    """Test cases for the AutomationTool class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock automation engine
        self.mock_engine = MagicMock(spec=AutomationEngine)
        
        # Initialize the automation tool with the mock engine
        with patch('tools.automation_tool.get_automation_engine') as mock_get_engine:
            mock_get_engine.return_value = self.mock_engine
            self.tool = AutomationTool()
    
    def test_tool_initialization(self):
        """Test that the tool initializes correctly."""
        self.assertEqual(self.tool.name, "automation_tool")
        self.assertIsNotNone(self.tool.description)
        self.assertIsNotNone(self.tool.usage_examples)
        self.assertEqual(self.tool.engine, self.mock_engine)
    
    def test_tool_run_invalid_operation(self):
        """Test that the tool handles invalid operations correctly."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation="invalid_operation")
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
    
    def test_tool_run_engine_unavailable(self):
        """Test that the tool handles unavailable engine correctly."""
        # Set up the tool with no engine
        with patch('tools.automation_tool.get_automation_engine') as mock_get_engine:
            mock_get_engine.return_value = None
            tool = AutomationTool()
        
        with self.assertRaises(ToolError) as context:
            tool.run(operation="get_automations")
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_UNAVAILABLE)
    
    def test_tool_run_with_json_kwargs(self):
        """Test that the tool processes JSON kwargs correctly."""
        # Setup mock for get_automations
        self.mock_engine.get_automations.return_value = {
            "automations": [],
            "count": 0,
            "total": 0
        }
        
        # Call with JSON string in kwargs
        result = self.tool.run(
            operation="get_automations",
            kwargs=json.dumps({"status": "active"})
        )
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.get_automations.assert_called_once_with(
            automation_type=None,
            status="active",
            frequency=None,
            limit=10,
            offset=0,
            user_id=None
        )
        
        self.assertIn("message", result)
    
    def test_tool_run_with_invalid_json_kwargs(self):
        """Test that the tool handles invalid JSON kwargs correctly."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(
                operation="get_automations",
                kwargs="invalid json"
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)

    # Tests for create_automation
    
    def test_create_simple_task_automation(self):
        """Test creating a simple task automation."""
        # Create a mock automation object
        mock_automation = MagicMock(spec=Automation)
        mock_automation.name = "Test Task"
        mock_automation.frequency = TaskFrequency.DAILY
        mock_automation.scheduled_time = datetime.now(timezone.utc)
        mock_automation.day_of_week = None
        mock_automation.day_of_month = None
        mock_automation.to_dict.return_value = {
            "id": "auto_12345",
            "name": "Test Task",
            "type": "simple_task",
            "frequency": "daily",
            "scheduled_time": mock_automation.scheduled_time.isoformat(),
            "next_execution_time": mock_automation.scheduled_time.isoformat(),
            "status": "active"
        }
        
        # Set up the mock engine to return the mock automation
        self.mock_engine.create_automation.return_value = mock_automation
        
        # Call the create_automation function
        result = self.tool._create_automation(
            name="Test Task",
            type="simple_task",
            frequency="daily",
            scheduled_time="09:00:00",
            execution_mode="direct",
            tool_name="reminder_tool",
            operation="add_reminder",
            parameters={"title": "Test", "description": "Test task"}
        )
        
        # Check the result
        self.assertEqual(result["automation"]["name"], "Test Task")
        self.assertEqual(result["automation"]["type"], "simple_task")
        self.assertIn("message", result)
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.create_automation.assert_called_once()
        # Extract the actual call arguments
        args, _ = self.mock_engine.create_automation.call_args
        self.assertEqual(args[0]["name"], "Test Task")
        self.assertEqual(args[0]["type"], "simple_task")
        self.assertEqual(args[0]["frequency"], "daily")
    
    def test_create_sequence_automation(self):
        """Test creating a sequence automation."""
        # Create a mock automation object
        mock_automation = MagicMock(spec=Automation)
        mock_automation.name = "Test Sequence"
        mock_automation.frequency = TaskFrequency.WEEKLY
        mock_automation.scheduled_time = datetime.now(timezone.utc)
        mock_automation.day_of_week = 1  # Tuesday
        mock_automation.day_of_month = None
        mock_automation.to_dict.return_value = {
            "id": "auto_67890",
            "name": "Test Sequence",
            "type": "sequence",
            "frequency": "weekly",
            "day_of_week": 1,
            "scheduled_time": mock_automation.scheduled_time.isoformat(),
            "next_execution_time": mock_automation.scheduled_time.isoformat(),
            "status": "active"
        }
        
        # Set up the mock engine to return the mock automation
        self.mock_engine.create_automation.return_value = mock_automation
        
        # Call the create_automation function
        result = self.tool._create_automation(
            name="Test Sequence",
            type="sequence",
            frequency="weekly",
            day_of_week=1,  # Tuesday
            scheduled_time="09:00:00",
            steps=[
                {
                    "name": "Step 1",
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "get",
                    "parameters": {"url": "https://example.com"},
                    "output_key": "step1_result"
                },
                {
                    "name": "Step 2",
                    "execution_mode": "orchestrated",
                    "task_description": "Process data from step 1",
                    "output_key": "step2_result"
                }
            ]
        )
        
        # Check the result
        self.assertEqual(result["automation"]["name"], "Test Sequence")
        self.assertEqual(result["automation"]["type"], "sequence")
        self.assertIn("message", result)
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.create_automation.assert_called_once()
        # Extract the actual call arguments
        args, _ = self.mock_engine.create_automation.call_args
        self.assertEqual(args[0]["name"], "Test Sequence")
        self.assertEqual(args[0]["type"], "sequence")
        self.assertEqual(args[0]["frequency"], "weekly")
        self.assertEqual(args[0]["day_of_week"], 1)
        self.assertEqual(len(args[0]["steps"]), 2)
    
    def test_create_automation_validation_errors(self):
        """Test validation errors when creating an automation."""
        # Test missing name
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                type="simple_task",
                frequency="daily"
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test missing type
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Task",
                frequency="daily"
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test invalid type
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Task",
                type="invalid_type",
                frequency="daily"
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test missing tool_name for direct execution
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Task",
                type="simple_task",
                frequency="daily",
                execution_mode="direct",
                operation="operation"
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test missing operation for direct execution
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Task",
                type="simple_task",
                frequency="daily",
                execution_mode="direct",
                tool_name="tool_name"
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test missing task_description for orchestrated execution
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Task",
                type="simple_task",
                frequency="daily",
                execution_mode="orchestrated"
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test missing steps for sequence
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Sequence",
                type="sequence",
                frequency="daily"
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test invalid steps for sequence (empty list)
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Sequence",
                type="sequence",
                frequency="daily",
                steps=[]
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
    
    def test_create_automation_step_validation(self):
        """Test validation of steps when creating a sequence automation."""
        # Test missing step name
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Sequence",
                type="sequence",
                frequency="daily",
                steps=[
                    {
                        "execution_mode": "direct",
                        "tool_name": "http_tool",
                        "operation": "get",
                        "parameters": {"url": "https://example.com"},
                        "output_key": "step1_result"
                    }
                ]
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test missing output_key
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Sequence",
                type="sequence",
                frequency="daily",
                steps=[
                    {
                        "name": "Step 1",
                        "execution_mode": "direct",
                        "tool_name": "http_tool",
                        "operation": "get",
                        "parameters": {"url": "https://example.com"}
                    }
                ]
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test direct step missing tool_name
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Sequence",
                type="sequence",
                frequency="daily",
                steps=[
                    {
                        "name": "Step 1",
                        "execution_mode": "direct",
                        "operation": "get",
                        "parameters": {"url": "https://example.com"},
                        "output_key": "step1_result"
                    }
                ]
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test direct step missing operation
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Sequence",
                type="sequence",
                frequency="daily",
                steps=[
                    {
                        "name": "Step 1",
                        "execution_mode": "direct",
                        "tool_name": "http_tool",
                        "parameters": {"url": "https://example.com"},
                        "output_key": "step1_result"
                    }
                ]
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test orchestrated step missing task_description
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Sequence",
                type="sequence",
                frequency="daily",
                steps=[
                    {
                        "name": "Step 1",
                        "execution_mode": "orchestrated",
                        "output_key": "step1_result"
                    }
                ]
            )
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
    
    def test_create_automation_infer_execution_mode(self):
        """Test inferring execution mode when creating an automation."""
        # Create a mock automation object
        mock_automation = MagicMock(spec=Automation)
        mock_automation.name = "Test Task"
        mock_automation.frequency = TaskFrequency.DAILY
        mock_automation.scheduled_time = datetime.now(timezone.utc)
        mock_automation.to_dict.return_value = {"name": "Test Task"}
        
        # Set up the mock engine to return the mock automation
        self.mock_engine.create_automation.return_value = mock_automation
        
        # Test inferring direct execution mode
        self.tool._create_automation(
            name="Test Task",
            type="simple_task",
            frequency="daily",
            tool_name="tool_name",
            operation="operation"
        )
        
        # Check that the engine method was called with direct execution mode
        args, _ = self.mock_engine.create_automation.call_args
        self.assertEqual(args[0]["execution_mode"], "direct")
        
        # Reset the mock
        self.mock_engine.create_automation.reset_mock()
        
        # Test inferring orchestrated execution mode
        self.tool._create_automation(
            name="Test Task",
            type="simple_task",
            frequency="daily",
            task_description="Do something"
        )
        
        # Check that the engine method was called with orchestrated execution mode
        args, _ = self.mock_engine.create_automation.call_args
        self.assertEqual(args[0]["execution_mode"], "orchestrated")
    
    def test_create_automation_engine_error(self):
        """Test handling of engine errors when creating an automation."""
        # Set up the mock engine to raise an exception
        self.mock_engine.create_automation.side_effect = Exception("Engine error")
        
        # Call the create_automation function with valid parameters
        with self.assertRaises(ToolError) as context:
            self.tool._create_automation(
                name="Test Task",
                type="simple_task",
                frequency="daily",
                execution_mode="direct",
                tool_name="tool_name",
                operation="operation"
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_EXECUTION_ERROR)
    
    # Tests for get_automations
    
    def test_get_automations(self):
        """Test getting automations."""
        # Create a mock automation list
        mock_automations = [
            {
                "id": "auto_12345",
                "name": "Test Task",
                "type": "simple_task",
                "frequency": "daily",
                "next_execution_time": datetime.now(timezone.utc).isoformat(),
                "status": "active"
            },
            {
                "id": "auto_67890",
                "name": "Test Sequence",
                "type": "sequence",
                "frequency": "weekly",
                "next_execution_time": datetime.now(timezone.utc).isoformat(),
                "status": "active"
            }
        ]
        
        # Set up the mock engine to return the mock automation list
        self.mock_engine.get_automations.return_value = {
            "automations": mock_automations,
            "count": len(mock_automations),
            "total": len(mock_automations)
        }
        
        # Call the get_automations function
        result = self.tool._get_automations(status="active")
        
        # Check the result
        self.assertEqual(len(result["automations"]), 2)
        self.assertEqual(result["count"], 2)
        self.assertIn("message", result)
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.get_automations.assert_called_once_with(
            automation_type=None,
            status="active",
            frequency=None,
            limit=10,
            offset=0,
            user_id=None
        )
    
    def test_get_automations_with_filters(self):
        """Test getting automations with multiple filters."""
        # Set up the mock engine to return an empty list
        self.mock_engine.get_automations.return_value = {
            "automations": [],
            "count": 0,
            "total": 0
        }
        
        # Call the get_automations function with filters
        result = self.tool._get_automations(
            type="simple_task",
            status="active",
            frequency="daily",
            limit=5,
            offset=10,
            user_id="user_123"
        )
        
        # Check the result
        self.assertEqual(len(result["automations"]), 0)
        self.assertEqual(result["count"], 0)
        self.assertIn("message", result)
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.get_automations.assert_called_once_with(
            automation_type="simple_task",
            status="active",
            frequency="daily",
            limit=5,
            offset=10,
            user_id="user_123"
        )
    
    def test_get_automations_validation_errors(self):
        """Test validation errors when getting automations."""
        # Test invalid status
        with self.assertRaises(ToolError) as context:
            self.tool._get_automations(status="invalid_status")
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test invalid frequency
        with self.assertRaises(ToolError) as context:
            self.tool._get_automations(frequency="invalid_frequency")
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
        
        # Test invalid type
        with self.assertRaises(ToolError) as context:
            self.tool._get_automations(type="invalid_type")
        self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
    
    # Tests for get_automation
    
    def test_get_automation(self):
        """Test getting a specific automation."""
        # Create a mock automation
        mock_automation = MagicMock(spec=Automation)
        mock_automation.name = "Test Automation"
        mock_automation.to_dict.return_value = {
            "id": "auto_12345",
            "name": "Test Automation",
            "type": "simple_task",
            "frequency": "daily",
            "next_execution_time": datetime.now(timezone.utc).isoformat(),
            "status": "active"
        }
        
        # Set up the mock engine to return the mock automation
        self.mock_engine.get_automation.return_value = mock_automation
        
        # Call the get_automation function
        result = self.tool._get_automation(automation_id="auto_12345")
        
        # Check the result
        self.assertEqual(result["automation"]["name"], "Test Automation")
        self.assertEqual(result["automation"]["id"], "auto_12345")
        self.assertIn("message", result)
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.get_automation.assert_called_once_with("auto_12345")
    
    def test_get_automation_not_found(self):
        """Test getting a non-existent automation."""
        # Set up the mock engine to return None
        self.mock_engine.get_automation.return_value = None
        
        # Call the get_automation function with an invalid ID
        with self.assertRaises(ToolError) as context:
            self.tool._get_automation(automation_id="invalid_id")
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_NOT_FOUND)
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.get_automation.assert_called_once_with("invalid_id")
    
    # Tests for update_automation
    
    def test_update_automation(self):
        """Test updating an automation."""
        # Create a mock updated automation
        mock_automation = MagicMock(spec=Automation)
        mock_automation.name = "Updated Automation"
        mock_automation.to_dict.return_value = {
            "id": "auto_12345",
            "name": "Updated Automation",
            "type": "simple_task",
            "frequency": "weekly",
            "next_execution_time": datetime.now(timezone.utc).isoformat(),
            "status": "active"
        }
        
        # Set up the mock engine to return the mock automation
        self.mock_engine.update_automation.return_value = mock_automation
        
        # Call the update_automation function
        result = self.tool._update_automation(
            automation_id="auto_12345",
            name="Updated Automation",
            frequency="weekly"
        )
        
        # Check the result
        self.assertEqual(result["automation"]["name"], "Updated Automation")
        self.assertEqual(result["automation"]["frequency"], "weekly")
        self.assertIn("message", result)
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.update_automation.assert_called_once_with(
            "auto_12345",
            {
                "name": "Updated Automation",
                "frequency": "weekly"
            }
        )
    
    def test_update_automation_not_found(self):
        """Test updating a non-existent automation."""
        # Set up the mock engine to return None
        self.mock_engine.update_automation.return_value = None
        
        # Call the update_automation function with an invalid ID
        with self.assertRaises(ToolError) as context:
            self.tool._update_automation(
                automation_id="invalid_id",
                name="Updated Automation"
            )
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_NOT_FOUND)
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.update_automation.assert_called_once_with(
            "invalid_id",
            {"name": "Updated Automation"}
        )
    
    # Tests for delete_automation
    
    def test_delete_automation(self):
        """Test deleting an automation."""
        # Create a mock automation
        mock_automation = MagicMock(spec=Automation)
        mock_automation.name = "Test Automation"
        
        # Set up the mock engine
        self.mock_engine.get_automation.return_value = mock_automation
        self.mock_engine.delete_automation.return_value = True
        
        # Mock the config to require confirmation
        with patch('tools.automation_tool.config') as mock_config:
            mock_config.automation_tool.confirm_before_delete = True
            
            # Call the delete_automation function with confirmation
            result = self.tool._delete_automation(
                automation_id="auto_12345",
                confirm=True
            )
            
            # Check the result
            self.assertEqual(result["automation_id"], "auto_12345")
            self.assertIn("message", result)
            self.assertIn("Test Automation", result["message"])
            
            # Check that the engine methods were called with the correct parameters
            self.mock_engine.get_automation.assert_called_once_with("auto_12345")
            self.mock_engine.delete_automation.assert_called_once_with("auto_12345")
    
    def test_delete_automation_without_confirmation(self):
        """Test deleting an automation without confirmation."""
        # Mock the config to require confirmation
        with patch('tools.automation_tool.config') as mock_config:
            mock_config.automation_tool.confirm_before_delete = True
            
            # Call the delete_automation function without confirmation
            with self.assertRaises(ToolError) as context:
                self.tool._delete_automation(automation_id="auto_12345")
            
            self.assertEqual(context.exception.code, ErrorCode.TOOL_INVALID_INPUT)
            
            # Check that the engine methods were not called
            self.mock_engine.get_automation.assert_not_called()
            self.mock_engine.delete_automation.assert_not_called()
    
    def test_delete_automation_not_found(self):
        """Test deleting a non-existent automation."""
        # Set up the mock engine to return None for get_automation
        self.mock_engine.get_automation.return_value = None
        
        # Mock the config to not require confirmation
        with patch('tools.automation_tool.config') as mock_config:
            mock_config.automation_tool.confirm_before_delete = False
            
            # Call the delete_automation function with an invalid ID
            with self.assertRaises(ToolError) as context:
                self.tool._delete_automation(automation_id="invalid_id")
            
            self.assertEqual(context.exception.code, ErrorCode.TOOL_NOT_FOUND)
            
            # Check that the engine methods were called correctly
            self.mock_engine.get_automation.assert_called_once_with("invalid_id")
            self.mock_engine.delete_automation.assert_not_called()
    
    def test_delete_automation_failure(self):
        """Test handling of failure when deleting an automation."""
        # Create a mock automation
        mock_automation = MagicMock(spec=Automation)
        mock_automation.name = "Test Automation"
        
        # Set up the mock engine
        self.mock_engine.get_automation.return_value = mock_automation
        self.mock_engine.delete_automation.return_value = False
        
        # Mock the config to not require confirmation
        with patch('tools.automation_tool.config') as mock_config:
            mock_config.automation_tool.confirm_before_delete = False
            
            # Call the delete_automation function
            with self.assertRaises(ToolError) as context:
                self.tool._delete_automation(automation_id="auto_12345")
            
            self.assertEqual(context.exception.code, ErrorCode.TOOL_EXECUTION_ERROR)
            
            # Check that the engine methods were called correctly
            self.mock_engine.get_automation.assert_called_once_with("auto_12345")
            self.mock_engine.delete_automation.assert_called_once_with("auto_12345")
    
    # Tests for execute_now
    
    def test_execute_now(self):
        """Test executing an automation immediately."""
        # Create a mock automation
        mock_automation = MagicMock(spec=Automation)
        mock_automation.name = "Test Automation"
        
        # Create a mock execution
        mock_execution = MagicMock(spec=AutomationExecution)
        mock_execution.id = "exec_12345"
        
        # Set up the mock engine
        self.mock_engine.get_automation.return_value = mock_automation
        self.mock_engine.execute_now.return_value = mock_execution
        
        # Call the execute_now function
        result = self.tool._execute_now(
            automation_id="auto_12345",
            initial_context={"param1": "value1"}
        )
        
        # Check the result
        self.assertEqual(result["automation_id"], "auto_12345")
        self.assertEqual(result["execution_id"], "exec_12345")
        self.assertTrue(result["success"])
        self.assertIn("message", result)
        self.assertIn("Test Automation", result["message"])
        
        # Check that the engine methods were called with the correct parameters
        self.mock_engine.execute_now.assert_called_once_with(
            automation_id="auto_12345",
            initial_context={"param1": "value1"}
        )
        self.mock_engine.get_automation.assert_called_once_with("auto_12345")
    
    def test_execute_now_with_default_context(self):
        """Test executing an automation with a default empty context."""
        # Create a mock automation and execution
        mock_automation = MagicMock(spec=Automation)
        mock_automation.name = "Test Automation"
        mock_execution = MagicMock(spec=AutomationExecution)
        mock_execution.id = "exec_12345"
        
        # Set up the mock engine
        self.mock_engine.get_automation.return_value = mock_automation
        self.mock_engine.execute_now.return_value = mock_execution
        
        # Call the execute_now function without initial_context
        result = self.tool._execute_now(automation_id="auto_12345")
        
        # Check that the engine method was called with an empty context
        self.mock_engine.execute_now.assert_called_once_with(
            automation_id="auto_12345",
            initial_context={}
        )
    
    def test_execute_now_engine_error(self):
        """Test handling of engine errors when executing an automation."""
        # Set up the mock engine to raise an exception
        self.mock_engine.execute_now.side_effect = Exception("Engine error")
        
        # Call the execute_now function
        with self.assertRaises(ToolError) as context:
            self.tool._execute_now(automation_id="auto_12345")
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_EXECUTION_ERROR)
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.execute_now.assert_called_once_with(
            automation_id="auto_12345",
            initial_context={}
        )
    
    # Tests for pause_automation
    
    def test_pause_automation(self):
        """Test pausing an automation."""
        # Create a mock automation
        mock_automation = MagicMock(spec=Automation)
        mock_automation.name = "Test Automation"
        
        # Set up the mock engine
        self.mock_engine.get_automation.return_value = mock_automation
        self.mock_engine.pause_automation.return_value = True
        
        # Call the pause_automation function
        result = self.tool._pause_automation(automation_id="auto_12345")
        
        # Check the result
        self.assertEqual(result["automation_id"], "auto_12345")
        self.assertIn("message", result)
        self.assertIn("Test Automation", result["message"])
        
        # Check that the engine methods were called with the correct parameters
        self.mock_engine.pause_automation.assert_called_once_with("auto_12345")
        self.mock_engine.get_automation.assert_called_once_with("auto_12345")
    
    def test_pause_automation_failure(self):
        """Test handling of failure when pausing an automation."""
        # Set up the mock engine
        self.mock_engine.pause_automation.return_value = False
        
        # Call the pause_automation function
        with self.assertRaises(ToolError) as context:
            self.tool._pause_automation(automation_id="auto_12345")
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_EXECUTION_ERROR)
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.pause_automation.assert_called_once_with("auto_12345")
    
    # Tests for resume_automation
    
    def test_resume_automation(self):
        """Test resuming an automation."""
        # Set up the mock engine
        self.mock_engine.resume_automation.return_value = {
            "success": True,
            "message": "Automation resumed successfully",
            "next_execution": datetime.now(timezone.utc).isoformat()
        }
        
        # Call the resume_automation function
        result = self.tool._resume_automation(automation_id="auto_12345")
        
        # Check the result
        self.assertEqual(result["automation_id"], "auto_12345")
        self.assertIn("message", result)
        self.assertIn("next_execution", result)
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.resume_automation.assert_called_once_with("auto_12345")
    
    # Tests for get_executions
    
    def test_get_executions(self):
        """Test getting execution history for an automation."""
        # Create a mock automation
        mock_automation = MagicMock(spec=Automation)
        mock_automation.name = "Test Automation"
        
        # Create mock execution results
        mock_executions = {
            "executions": [
                {
                    "id": "exec_12345",
                    "status": "completed",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "runtime_seconds": 10
                }
            ],
            "count": 1,
            "total": 1
        }
        
        # Set up the mock engine
        self.mock_engine.get_automation.return_value = mock_automation
        self.mock_engine.get_executions.return_value = mock_executions
        
        # Call the get_executions function
        result = self.tool._get_executions(
            automation_id="auto_12345",
            limit=5,
            offset=0
        )
        
        # Check the result
        self.assertEqual(result["automation_name"], "Test Automation")
        self.assertEqual(result["count"], 1)
        self.assertEqual(len(result["executions"]), 1)
        self.assertIn("message", result)
        
        # Check that the engine methods were called with the correct parameters
        self.mock_engine.get_executions.assert_called_once_with(
            automation_id="auto_12345",
            limit=5,
            offset=0
        )
        self.mock_engine.get_automation.assert_called_once_with("auto_12345")
    
    # Tests for get_execution_details
    
    def test_get_execution_details(self):
        """Test getting detailed information about a specific execution."""
        # Create a mock execution
        mock_execution = MagicMock(spec=AutomationExecution)
        mock_execution.automation_id = "auto_12345"
        mock_execution.to_dict.return_value = {
            "id": "exec_12345",
            "automation_id": "auto_12345",
            "status": "completed",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": 10,
            "step_executions": []
        }
        
        # Create a mock automation
        mock_automation = MagicMock(spec=Automation)
        mock_automation.name = "Test Automation"
        
        # Set up the mock engine
        self.mock_engine.get_execution_details.return_value = mock_execution
        self.mock_engine.get_automation.return_value = mock_automation
        
        # Call the get_execution_details function
        result = self.tool._get_execution_details(execution_id="exec_12345")
        
        # Check the result
        self.assertEqual(result["automation_name"], "Test Automation")
        self.assertEqual(result["execution"]["id"], "exec_12345")
        self.assertIn("message", result)
        
        # Check that the engine methods were called with the correct parameters
        self.mock_engine.get_execution_details.assert_called_once_with("exec_12345")
        self.mock_engine.get_automation.assert_called_once_with("auto_12345")
    
    def test_get_execution_details_not_found(self):
        """Test getting details for a non-existent execution."""
        # Set up the mock engine to return None
        self.mock_engine.get_execution_details.return_value = None
        
        # Call the get_execution_details function with an invalid ID
        with self.assertRaises(ToolError) as context:
            self.tool._get_execution_details(execution_id="invalid_id")
        
        self.assertEqual(context.exception.code, ErrorCode.TOOL_NOT_FOUND)
        
        # Check that the engine method was called with the correct parameters
        self.mock_engine.get_execution_details.assert_called_once_with("invalid_id")


if __name__ == "__main__":
    unittest.main()