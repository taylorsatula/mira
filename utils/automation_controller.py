"""
Automation controller for managing the unified automation engine.

This module provides functions to initialize, start, and stop the automation
system as part of the main application lifecycle, presenting a unified interface
for the main application.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from task_manager.automation_engine import (
    get_automation_engine, initialize_automation_engine, AutomationEngine
)
from task_manager.automation import (
    AutomationExecution, ExecutionStatus, AutomationType, TriggerType
)
from api.llm_bridge import LLMBridge
from tools.repo import ToolRepository
from db import Database

# Configure logger
logger = logging.getLogger(__name__)

# Global instances
_db = None
_is_initialized = False
_is_running = False


def initialize_systems(
    tool_repo: Optional[ToolRepository] = None,
    llm_bridge: Optional[LLMBridge] = None
) -> Dict[str, Any]:
    """
    Initialize all automation systems.
    
    This function initializes the unified automation engine which handles
    both simple tasks and multi-step sequences.
    
    Args:
        tool_repo: Repository of available tools
        llm_bridge: LLM bridge for LLM-orchestrated execution
        
    Returns:
        Dictionary containing all initialized components
    """
    global _db, _is_initialized
    
    logger.info("Initializing automation systems...")
    
    # Initialize the database
    _db = Database()
    
    # Initialize the automation engine with the shared tool repository
    engine = initialize_automation_engine(
        tool_repo=tool_repo,
        llm_bridge=llm_bridge
    )

    # Set flag
    _is_initialized = True
    
    # Build components dictionary that matches what main.py expects
    components = {
        'scheduler': engine,  # Map to what main.py expects
        'notification_manager': None,  # Will be handled internally by the engine
        'chain_executor': engine  # Map to what main.py expects
    }
    
    logger.info("Automation system initialized")
    
    return components


def start_systems() -> None:
    """
    Start all automation systems.
    
    This function starts the automation engine's scheduler.
    """
    global _is_running
    
    if not _is_initialized:
        logger.warning("Cannot start systems: not initialized")
        return
    
    logger.info("Starting automation systems...")
    
    # Get the automation engine
    engine = get_automation_engine()
    
    # Start the scheduler
    if engine:
        engine.start_scheduler()
        logger.info("Automation scheduler started")
        _is_running = True
    else:
        logger.warning("Cannot start automation system: engine not initialized")


def stop_systems() -> None:
    """
    Stop all automation systems.
    
    This function stops the automation engine's scheduler.
    """
    global _is_running
    
    if not _is_initialized:
        return
    
    logger.info("Stopping automation systems...")
    
    # Get the automation engine
    engine = get_automation_engine()
    
    # Stop the scheduler
    if engine:
        engine.stop_scheduler()
        logger.info("Automation scheduler stopped")
        
        _is_running = False
    else:
        logger.warning("Cannot stop automation system: engine not initialized")


def is_scheduler_running() -> bool:
    """
    Check if the scheduler is running.
    
    Returns:
        True if the scheduler is running, False otherwise
    """
    return _is_running


def is_chain_scheduler_running() -> bool:
    """
    Check if the chain scheduler is running.
    
    This is maintained for compatibility with older code.
    In the new system, chains are part of the automation engine.
    
    Returns:
        True if the automation engine is running, False otherwise
    """
    return _is_running


def get_recent_executions(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent automation executions.
    
    Args:
        limit: Maximum number of executions to return
        
    Returns:
        List of recent automation executions
    """
    if not _db:
        return []
    
    executions = []
    
    with _db.get_session() as session:
        query = session.query(AutomationExecution).filter(
            AutomationExecution.status.in_([
                ExecutionStatus.COMPLETED.value, 
                ExecutionStatus.FAILED.value
            ])
        ).order_by(AutomationExecution.completed_at.desc()).limit(limit)
        
        for execution in query.all():
            automation_name = execution.automation.name if execution.automation else "Unknown Automation"
            success = execution.status == ExecutionStatus.COMPLETED.value
            
            executions.append({
                "id": execution.id,
                "automation_id": execution.automation_id,
                "automation_name": automation_name,
                "status": execution.status.value if execution.status else None,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "result": execution.result,
                "error": execution.error
            })
    
    return executions


def get_execution_details(execution_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific execution.
    
    Args:
        execution_id: ID of the execution
        
    Returns:
        Execution details or None if not found
    """
    engine = get_automation_engine()
    if not engine:
        return None
        
    execution = engine.get_execution_details(execution_id)
    if not execution:
        return None
        
    return execution.to_dict()