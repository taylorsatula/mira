"""
Concurrency tests for the automation system.

This module provides tests that verify the automation system correctly
handles concurrent operations and maintains data integrity under load.
"""

import os
import json
import unittest
import threading
import time
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from task_manager.automation import (
    AutomationType, AutomationStatus, TaskFrequency, 
    Automation, AutomationStep, AutomationExecution,
    ExecutionMode, ExecutionStatus
)
from task_manager.automation_engine import (
    AutomationEngine, get_automation_engine, initialize_automation_engine
)
from tools.automation_tool import AutomationTool
from errors import ToolError, ErrorCode
from db import Database


class TestAutomationConcurrency(unittest.TestCase):
    """Concurrency tests for the automation system."""

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
    
    def test_concurrent_automation_creation(self):
        """Test creating multiple automations concurrently."""
        # Number of automations to create
        num_automations = 10
        
        # Function to create an automation
        def create_automation(index):
            try:
                result = self.tool.run(
                    operation="create_automation",
                    name=f"Concurrent Automation {index}",
                    type="simple_task",
                    frequency="once",
                    scheduled_time=datetime.now(timezone.utc).isoformat(),
                    execution_mode="direct",
                    tool_name="http_tool",
                    operation="echo",
                    parameters={"message": f"Message from automation {index}"}
                )
                return result["automation"]["id"]
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Create automations concurrently
        with ThreadPoolExecutor(max_workers=num_automations) as executor:
            future_to_index = {executor.submit(create_automation, i): i for i in range(num_automations)}
            
            # Collect results
            automation_ids = []
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    automation_id = future.result()
                    if not automation_id.startswith("Error"):
                        automation_ids.append(automation_id)
                except Exception as e:
                    print(f"Automation {index} generated an exception: {e}")
        
        # Verify all automations were created
        self.assertEqual(len(automation_ids), num_automations)
        
        # Verify we can retrieve all the automations
        all_automations = self.tool.run(
            operation="get_automations",
            limit=num_automations * 2  # Set limit high enough to get all
        )
        
        self.assertEqual(all_automations["count"], num_automations)
    
    def test_concurrent_sequence_creation(self):
        """Test creating multiple sequence automations concurrently."""
        # Number of sequences to create
        num_sequences = 5
        
        # Function to create a sequence with multiple steps
        def create_sequence(index):
            try:
                # Create steps
                steps = []
                num_steps = random.randint(2, 5)
                
                for step_index in range(1, num_steps + 1):
                    steps.append({
                        "name": f"Step {step_index} of Sequence {index}",
                        "position": step_index,
                        "execution_mode": "direct",
                        "tool_name": "http_tool",
                        "operation": "echo",
                        "parameters": {"message": f"Message from step {step_index} of sequence {index}"},
                        "output_key": f"step{step_index}_result"
                    })
                
                result = self.tool.run(
                    operation="create_automation",
                    name=f"Concurrent Sequence {index}",
                    type="sequence",
                    frequency="once",
                    scheduled_time=datetime.now(timezone.utc).isoformat(),
                    steps=steps
                )
                return result["automation"]["id"]
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Create sequences concurrently
        with ThreadPoolExecutor(max_workers=num_sequences) as executor:
            future_to_index = {executor.submit(create_sequence, i): i for i in range(num_sequences)}
            
            # Collect results
            sequence_ids = []
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    sequence_id = future.result()
                    if not sequence_id.startswith("Error"):
                        sequence_ids.append(sequence_id)
                except Exception as e:
                    print(f"Sequence {index} generated an exception: {e}")
        
        # Verify all sequences were created
        self.assertEqual(len(sequence_ids), num_sequences)
        
        # Verify we can retrieve all sequences
        all_sequences = self.tool.run(
            operation="get_automations",
            type="sequence",
            limit=num_sequences * 2
        )
        
        self.assertEqual(all_sequences["count"], num_sequences)
    
    def test_concurrent_update_of_automations(self):
        """Test updating multiple automations concurrently."""
        # Create automations first
        automation_ids = []
        for i in range(5):
            create_result = self.tool.run(
                operation="create_automation",
                name=f"Automation {i}",
                type="simple_task",
                frequency="once",
                scheduled_time=datetime.now(timezone.utc).isoformat(),
                execution_mode="direct",
                tool_name="http_tool",
                operation="echo",
                parameters={"message": f"Original message {i}"}
            )
            automation_ids.append(create_result["automation"]["id"])
        
        # Function to update an automation
        def update_automation(automation_id, index):
            try:
                result = self.tool.run(
                    operation="update_automation",
                    automation_id=automation_id,
                    name=f"Updated Automation {index}",
                    parameters={"message": f"Updated message {index}"}
                )
                return True
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Update automations concurrently
        with ThreadPoolExecutor(max_workers=len(automation_ids)) as executor:
            future_to_index = {
                executor.submit(update_automation, automation_id, i): i 
                for i, automation_id in enumerate(automation_ids)
            }
            
            # Collect results
            success_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result is True:
                        success_count += 1
                except Exception as e:
                    print(f"Update {index} generated an exception: {e}")
        
        # Verify all updates succeeded
        self.assertEqual(success_count, len(automation_ids))
        
        # Verify the updates were applied
        for i, automation_id in enumerate(automation_ids):
            get_result = self.tool.run(
                operation="get_automation",
                automation_id=automation_id
            )
            self.assertEqual(get_result["automation"]["name"], f"Updated Automation {i}")
            self.assertEqual(get_result["automation"]["parameters"]["message"], f"Updated message {i}")
    
    def test_concurrent_execution_of_automations(self):
        """Test executing multiple different automations concurrently."""
        # Create automations first
        automation_ids = []
        for i in range(5):
            create_result = self.tool.run(
                operation="create_automation",
                name=f"Automation {i}",
                type="simple_task",
                frequency="once",
                scheduled_time=datetime.now(timezone.utc).isoformat(),
                execution_mode="direct",
                tool_name="http_tool",
                operation="echo",
                parameters={"message": f"Message {i}"}
            )
            automation_ids.append(create_result["automation"]["id"])
        
        # Function to execute an automation
        def execute_automation(automation_id):
            try:
                result = self.tool.run(
                    operation="execute_now",
                    automation_id=automation_id
                )
                return result["execution_id"]
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Execute automations concurrently
        with ThreadPoolExecutor(max_workers=len(automation_ids)) as executor:
            future_to_id = {
                executor.submit(execute_automation, automation_id): automation_id 
                for automation_id in automation_ids
            }
            
            # Collect results
            execution_ids = []
            for future in as_completed(future_to_id):
                automation_id = future_to_id[future]
                try:
                    execution_id = future.result()
                    if not execution_id.startswith("Error"):
                        execution_ids.append(execution_id)
                except Exception as e:
                    print(f"Execution of {automation_id} generated an exception: {e}")
        
        # Verify all executions were initiated
        self.assertEqual(len(execution_ids), len(automation_ids))
        
        # Allow time for executions to complete
        time.sleep(2)
        
        # Verify all executions completed
        for execution_id in execution_ids:
            details_result = self.tool.run(
                operation="get_execution_details",
                execution_id=execution_id
            )
            self.assertEqual(details_result["execution"]["status"], "completed")
    
    def test_mixed_concurrent_operations(self):
        """Test performing different operations concurrently."""
        # Set up initial automations
        automation_ids = []
        for i in range(3):
            create_result = self.tool.run(
                operation="create_automation",
                name=f"Initial Automation {i}",
                type="simple_task",
                frequency="once",
                scheduled_time=datetime.now(timezone.utc).isoformat(),
                execution_mode="direct",
                tool_name="http_tool",
                operation="echo",
                parameters={"message": f"Initial message {i}"}
            )
            automation_ids.append(create_result["automation"]["id"])
        
        # Define different operations
        operations = []
        
        # Create operations
        for i in range(3, 6):
            operations.append({
                "type": "create",
                "params": {
                    "name": f"New Automation {i}",
                    "type": "simple_task",
                    "frequency": "once",
                    "scheduled_time": datetime.now(timezone.utc).isoformat(),
                    "execution_mode": "direct",
                    "tool_name": "http_tool",
                    "operation": "echo",
                    "parameters": {"message": f"New message {i}"}
                }
            })
        
        # Update operations
        for i, automation_id in enumerate(automation_ids):
            operations.append({
                "type": "update",
                "automation_id": automation_id,
                "params": {
                    "name": f"Updated Automation {i}",
                    "parameters": {"message": f"Updated message {i}"}
                }
            })
        
        # Execute operations
        for automation_id in automation_ids:
            operations.append({
                "type": "execute",
                "automation_id": automation_id
            })
        
        # Get operations
        operations.append({"type": "get_all"})
        
        # Function to perform an operation
        def perform_operation(operation):
            try:
                if operation["type"] == "create":
                    result = self.tool.run(
                        operation="create_automation",
                        **operation["params"]
                    )
                    return {"type": "create", "id": result["automation"]["id"]}
                
                elif operation["type"] == "update":
                    result = self.tool.run(
                        operation="update_automation",
                        automation_id=operation["automation_id"],
                        **operation["params"]
                    )
                    return {"type": "update", "id": operation["automation_id"]}
                
                elif operation["type"] == "execute":
                    result = self.tool.run(
                        operation="execute_now",
                        automation_id=operation["automation_id"]
                    )
                    return {"type": "execute", "id": result["execution_id"]}
                
                elif operation["type"] == "get_all":
                    result = self.tool.run(
                        operation="get_automations",
                        limit=20
                    )
                    return {"type": "get_all", "count": result["count"]}
                
                return None
                
            except Exception as e:
                return {"type": operation["type"], "error": str(e)}
        
        # Perform operations concurrently
        results = []
        with ThreadPoolExecutor(max_workers=len(operations)) as executor:
            future_to_op = {
                executor.submit(perform_operation, op): i 
                for i, op in enumerate(operations)
            }
            
            # Collect results
            for future in as_completed(future_to_op):
                op_index = future_to_op[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Operation {op_index} generated an exception: {e}")
        
        # Verify results
        create_results = [r for r in results if r["type"] == "create" and "error" not in r]
        update_results = [r for r in results if r["type"] == "update" and "error" not in r]
        execute_results = [r for r in results if r["type"] == "execute" and "error" not in r]
        get_all_results = [r for r in results if r["type"] == "get_all" and "error" not in r]
        
        # All operations should have succeeded
        self.assertEqual(len(create_results), 3)  # 3 creates
        self.assertEqual(len(update_results), 3)  # 3 updates
        self.assertEqual(len(execute_results), 3)  # 3 executes
        
        # If the get_all was among the last operations, it should see all automations
        # But since operations are concurrent, we can't guarantee order, so we check
        # that at least it found some automations
        if get_all_results:
            self.assertGreater(get_all_results[0]["count"], 0)
    
    def test_concurrent_reads_during_update(self):
        """Test concurrent reading of an automation while it's being updated."""
        # Create an automation
        create_result = self.tool.run(
            operation="create_automation",
            name="Read-Update Test",
            type="simple_task",
            frequency="once",
            scheduled_time=datetime.now(timezone.utc).isoformat(),
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Original message"}
        )
        
        automation_id = create_result["automation"]["id"]
        
        # Variables to track completion
        update_started = threading.Event()
        update_done = threading.Event()
        read_results = []
        
        # Function to update the automation (slow)
        def update_automation():
            try:
                update_started.set()
                # Small delay to ensure reads can start
                time.sleep(0.2)
                result = self.tool.run(
                    operation="update_automation",
                    automation_id=automation_id,
                    name="Updated Read-Update Test",
                    parameters={"message": "Updated message"}
                )
                update_done.set()
                return True
            except Exception as e:
                update_done.set()
                return f"Error: {str(e)}"
        
        # Function to read the automation
        def read_automation():
            # Wait for the update to start
            update_started.wait(timeout=1.0)
            try:
                result = self.tool.run(
                    operation="get_automation",
                    automation_id=automation_id
                )
                # Record what version we saw
                read_results.append(result["automation"]["name"])
                return True
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Start the update thread
        update_thread = threading.Thread(target=update_automation)
        update_thread.start()
        
        # Start multiple read threads
        read_threads = []
        for _ in range(5):
            read_thread = threading.Thread(target=read_automation)
            read_thread.start()
            read_threads.append(read_thread)
        
        # Wait for all threads to complete
        update_thread.join()
        for thread in read_threads:
            thread.join()
        
        # Verify the update succeeded
        self.assertTrue(update_done.is_set())
        
        # Get the final state
        final_state = self.tool.run(
            operation="get_automation",
            automation_id=automation_id
        )
        
        # Verify the update was applied
        self.assertEqual(final_state["automation"]["name"], "Updated Read-Update Test")
        
        # Verify reads returned valid data (either original or updated version)
        for name in read_results:
            self.assertTrue(
                name in ["Read-Update Test", "Updated Read-Update Test"],
                f"Unexpected name: {name}"
            )
    
    def test_scheduler_restartability(self):
        """Test stopping and restarting the scheduler."""
        # Create automations
        create_result = self.tool.run(
            operation="create_automation",
            name="Scheduler Test",
            type="simple_task",
            frequency="once",
            # Schedule 3 seconds in the future
            scheduled_time=(datetime.now(timezone.utc) + timedelta(seconds=3)).isoformat(),
            execution_mode="direct",
            tool_name="http_tool",
            operation="echo",
            parameters={"message": "Scheduler test message"}
        )
        
        automation_id = create_result["automation"]["id"]
        
        # Stop the scheduler
        self.engine.stop_scheduler()
        
        # Wait for 1 second (scheduler is stopped)
        time.sleep(1)
        
        # Restart the scheduler
        self.engine.start_scheduler()
        
        # Wait for scheduler to execute the automation
        time.sleep(5)
        
        # Check if the automation was executed
        executions_result = self.tool.run(
            operation="get_executions",
            automation_id=automation_id
        )
        
        # Verify there was at least one execution
        self.assertGreater(executions_result["count"], 0)
        
        # Get the details of the execution
        execution_id = executions_result["executions"][0]["id"]
        details_result = self.tool.run(
            operation="get_execution_details",
            execution_id=execution_id
        )
        
        # Verify the execution completed
        self.assertEqual(details_result["execution"]["status"], "completed")
    
    def test_concurrent_same_name_automations(self):
        """Test creating multiple automations with the same name concurrently."""
        # Function to create an automation with the same name
        def create_same_name_automation(index):
            try:
                result = self.tool.run(
                    operation="create_automation",
                    name="Same Name Automation",
                    type="simple_task",
                    frequency="once",
                    scheduled_time=datetime.now(timezone.utc).isoformat(),
                    execution_mode="direct",
                    tool_name="http_tool",
                    operation="echo",
                    parameters={"message": f"Message {index}"}
                )
                return result["automation"]["id"]
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Create automations concurrently
        num_automations = 5
        with ThreadPoolExecutor(max_workers=num_automations) as executor:
            future_to_index = {
                executor.submit(create_same_name_automation, i): i 
                for i in range(num_automations)
            }
            
            # Collect results
            automation_ids = []
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    automation_id = future.result()
                    if not automation_id.startswith("Error"):
                        automation_ids.append(automation_id)
                except Exception as e:
                    print(f"Automation {index} generated an exception: {e}")
        
        # Verify all automations were created
        self.assertEqual(len(automation_ids), num_automations)
        
        # Verify they all have the same name
        for automation_id in automation_ids:
            get_result = self.tool.run(
                operation="get_automation",
                automation_id=automation_id
            )
            self.assertEqual(get_result["automation"]["name"], "Same Name Automation")


if __name__ == "__main__":
    unittest.main()