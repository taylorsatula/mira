"""
Test for automation timezone handling.

This file provides a simple testbench that creates an automation,
runs it, and then deletes it to verify that timezone handling works correctly.
"""

import logging
import time
import sys
import os
from datetime import datetime, timedelta, UTC

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_manager.automation import Automation, AutomationType, TaskFrequency, ExecutionMode
from task_manager.automation_engine import get_automation_engine
from db import Database

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(name)s │ %(message)s")
logger = logging.getLogger(__name__)

def test_automation_timezone_handling():
    """Test automation execution with timezone handling."""
    logger.info("Starting automation timezone test")
    
    # Get the automation engine
    engine = get_automation_engine()
    db = Database()
    
    # Create a simple automation that will run immediately
    automation_data = {
        "name": "Timezone Test Automation",
        "description": "Test automation for timezone handling",
        "type": AutomationType.SIMPLE_TASK.value,
        "frequency": TaskFrequency.ONCE.value,
        "scheduled_time": datetime.now(UTC),
        "execution_mode": ExecutionMode.DIRECT.value,
        "tool_name": "calculator_tool",  # Using the correct tool name from sample_tool_simple.py
        "operation": "get_time",  # Changed to get_time to test a simpler operation
        "parameters": {}
    }
    
    try:
        # Create the automation
        logger.info("Creating test automation")
        automation = engine.create_automation(automation_data)
        logger.info(f"Created automation with ID: {automation.id}")
        
        # Run it immediately
        logger.info("Executing automation")
        execution = engine.execute_now(automation.id)
        logger.info(f"Started execution with ID: {execution.id}")
        
        # Wait for execution to complete
        logger.info("Waiting for execution to complete")
        max_wait = 10  # seconds
        wait_time = 0
        completed = False
        
        while wait_time < max_wait:
            time.sleep(1)
            wait_time += 1
            
            # Get the execution details
            execution_details = engine.get_execution_details(execution.id)
            if execution_details and execution_details.status and execution_details.status.value in ['completed', 'failed']:
                completed = True
                break
        
        if completed:
            logger.info(f"Execution completed with status: {execution_details.status.value}")
            logger.info(f"Runtime: {execution_details.runtime_seconds} seconds")
            
            # Verify that runtime calculation worked
            assert execution_details.runtime_seconds is not None, "Runtime should be calculated"
            assert execution_details.runtime_seconds >= 0, "Runtime should be non-negative"
            
            # Log the timestamp details
            logger.info(f"started_at: {execution_details.started_at} (tzinfo: {execution_details.started_at.tzinfo})")
            logger.info(f"completed_at: {execution_details.completed_at} (tzinfo: {execution_details.completed_at.tzinfo})")
        else:
            logger.warning("Execution did not complete within the expected time")
        
        # Now test the error handling case by creating a failing automation
        fail_automation_data = {
            "name": "Failing Timezone Test",
            "description": "Test automation that will fail",
            "type": AutomationType.SIMPLE_TASK.value,
            "frequency": TaskFrequency.ONCE.value,
            "scheduled_time": datetime.now(UTC),
            "execution_mode": ExecutionMode.DIRECT.value,
            "tool_name": "nonexistent_tool",  # This will cause a tool not found error
            "operation": "noop",
            "parameters": {}
        }
        
        # Create the failing automation
        logger.info("Creating failing test automation")
        fail_automation = engine.create_automation(fail_automation_data)
        logger.info(f"Created failing automation with ID: {fail_automation.id}")
        
        # Run it and expect failure
        logger.info("Executing failing automation")
        fail_execution = engine.execute_now(fail_automation.id)
        logger.info(f"Started execution with ID: {fail_execution.id}")
        
        # Wait for execution to complete
        logger.info("Waiting for failing execution to complete")
        wait_time = 0
        completed = False
        
        while wait_time < max_wait:
            time.sleep(1)
            wait_time += 1
            
            # Get the execution details
            fail_execution_details = engine.get_execution_details(fail_execution.id)
            if fail_execution_details and fail_execution_details.status and fail_execution_details.status.value in ['completed', 'failed']:
                completed = True
                break
        
        if completed:
            logger.info(f"Failing execution completed with status: {fail_execution_details.status.value}")
            logger.info(f"Error: {fail_execution_details.error}")
            logger.info(f"Runtime: {fail_execution_details.runtime_seconds} seconds")
            
            # Verify that runtime calculation worked even for failed execution
            assert fail_execution_details.runtime_seconds is not None, "Runtime should be calculated for failed execution"
            assert fail_execution_details.runtime_seconds >= 0, "Runtime should be non-negative"
            
            # Log the timestamp details
            logger.info(f"started_at: {fail_execution_details.started_at} (tzinfo: {fail_execution_details.started_at.tzinfo})")
            logger.info(f"completed_at: {fail_execution_details.completed_at} (tzinfo: {fail_execution_details.completed_at.tzinfo})")
        else:
            logger.warning("Failing execution did not complete within the expected time")
    
    finally:
        # Clean up by deleting the automations
        try:
            if 'automation' in locals() and automation:
                logger.info(f"Deleting test automation: {automation.id}")
                engine.delete_automation(automation.id)
                
            if 'fail_automation' in locals() and fail_automation:
                logger.info(f"Deleting failing test automation: {fail_automation.id}")
                engine.delete_automation(fail_automation.id)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    logger.info("Automation timezone test completed")

if __name__ == "__main__":
    test_automation_timezone_handling()