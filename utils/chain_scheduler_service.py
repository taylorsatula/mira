"""
Chain scheduler service for managing and executing task chains.

This module provides the integration point between the task chain system
and the main application, handling chain scheduling, execution, and lifecycle.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set

from db import Database
from task_manager.task_chain import TaskChain, ChainStatus
from task_manager.chain_execution import ChainExecution, TriggerType
from task_manager.chain_executor import ChainExecutor
from task_manager.task_scheduler import TaskScheduler
from tools.repo import ToolRepository
from api.llm_bridge import LLMBridge
from task_manager.task_notification import NotificationManager
from config import config

# Configure logger
logger = logging.getLogger(__name__)

# Global executor reference
_chain_executor = None
_scheduler_thread = None
_running = False
_chain_scheduler_lock = threading.Lock()


def initialize_chain_scheduler(
    tool_repo: Optional[ToolRepository] = None,
    llm_bridge: Optional[LLMBridge] = None,
    notification_manager: Optional[NotificationManager] = None
) -> Dict[str, Any]:
    """
    Initialize the chain scheduler system.
    
    Args:
        tool_repo: Repository of available tools
        llm_bridge: LLM bridge for orchestrated steps
        notification_manager: Manager for creating notifications
        
    Returns:
        Dictionary with initialized components
    """
    global _chain_executor
    
    with _chain_scheduler_lock:
        if _chain_executor is None:
            logger.info("Initializing chain scheduler system")
            
            # Create chain executor
            _chain_executor = ChainExecutor(
                tool_repo=tool_repo,
                llm_bridge=llm_bridge,
                notification_manager=notification_manager
            )
            
            logger.info("Chain scheduler system initialized")
    
    return {
        "executor": _chain_executor
    }


def start_chain_scheduler() -> None:
    """
    Start the chain scheduler thread.
    
    This method begins the background thread that periodically checks for
    and executes scheduled chains.
    """
    global _scheduler_thread, _running
    
    with _chain_scheduler_lock:
        if _running:
            logger.warning("Chain scheduler is already running")
            return
            
        _running = True
        
        # Make sure executor is initialized
        if _chain_executor is None:
            initialize_chain_scheduler()
        
        # Start the scheduler thread
        _scheduler_thread = threading.Thread(
            target=_scheduler_loop,
            daemon=True,
            name="ChainSchedulerThread"
        )
        _scheduler_thread.start()
        
        logger.info("Chain scheduler started")


def stop_chain_scheduler() -> None:
    """
    Stop the chain scheduler thread.
    
    This method stops the scheduler thread and waits for any executing
    chains to complete.
    """
    global _running
    
    with _chain_scheduler_lock:
        if not _running:
            logger.warning("Chain scheduler is not running")
            return
            
        logger.info("Stopping chain scheduler...")
        _running = False
        
        # Wait for scheduler thread to complete
        if _scheduler_thread and _scheduler_thread.is_alive():
            _scheduler_thread.join(timeout=5.0)
            
        logger.info("Chain scheduler stopped")


def is_chain_scheduler_running() -> bool:
    """
    Check if the chain scheduler is running.
    
    Returns:
        True if the scheduler is running, False otherwise
    """
    return _running


def get_chain_executor() -> Optional[ChainExecutor]:
    """
    Get the chain executor instance.
    
    Returns:
        The chain executor instance or None if not initialized
    """
    return _chain_executor


def execute_chain_now(
    chain_id: str,
    initial_context: Optional[Dict[str, Any]] = None
) -> ChainExecution:
    """
    Execute a chain immediately.
    
    Args:
        chain_id: ID of the chain to execute
        initial_context: Initial context for the execution
        
    Returns:
        The chain execution record
        
    Raises:
        ToolError: If the chain execution fails
    """
    if _chain_executor is None:
        initialize_chain_scheduler()
        
    return _chain_executor.execute_chain(
        chain_id=chain_id,
        trigger_type=TriggerType.MANUAL,
        initial_context=initial_context
    )


def _scheduler_loop() -> None:
    """
    Main scheduler loop that periodically checks for and executes due chains.
    """
    db = Database()
    check_interval = config.get("task_chain", {}).get("check_interval", 60)
    
    while _running:
        try:
            # Find and execute due chains
            _process_due_chains(db)
            
            # Sleep until next check
            time.sleep(check_interval)
            
        except Exception as e:
            logger.error(f"Error in chain scheduler loop: {e}", exc_info=True)
            # Continue running even if an error occurs
            time.sleep(60)  # Wait a bit longer after an error


def _process_due_chains(db: Database) -> None:
    """
    Find and execute chains that are due to run.
    
    Args:
        db: Database instance for querying chains
    """
    now = datetime.now(timezone.utc)
    
    with db.get_session() as session:
        # Find chains that are due to run
        due_chains = session.query(TaskChain).filter(
            TaskChain.status == ChainStatus.ACTIVE,
            TaskChain.next_execution_time <= now
        ).all()
        
        if due_chains:
            logger.info(f"Found {len(due_chains)} chains due for execution")
            
            for chain in due_chains:
                # Execute the chain in a separate thread
                threading.Thread(
                    target=_execute_chain_safely,
                    args=(chain.id,),
                    daemon=True,
                    name=f"ChainExecution-{chain.id[:8]}"
                ).start()
                
                # Prevent overloading by staggering executions
                time.sleep(1)


def _execute_chain_safely(chain_id: str) -> None:
    """
    Execute a chain safely in a separate thread.
    
    Args:
        chain_id: ID of the chain to execute
    """
    try:
        # Check if chain is already running
        if _chain_executor and chain_id in _chain_executor.active_chains:
            logger.warning(f"Chain {chain_id} is already running, skipping")
            return
            
        # Execute the chain
        if _chain_executor:
            _chain_executor.execute_chain(
                chain_id=chain_id,
                trigger_type=TriggerType.SCHEDULED
            )
        else:
            logger.error(f"Cannot execute chain {chain_id}: chain executor not initialized")
            
    except Exception as e:
        logger.error(f"Error executing chain {chain_id}: {e}", exc_info=True)