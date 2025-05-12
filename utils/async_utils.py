"""
Asyncio utilities for managing event loops in a thread-safe manner.

This module provides utilities for managing asyncio event loops in a
thread-safe way, ensuring that each thread has its own event loop and
that event loops are properly initialized and cleaned up.
"""

import asyncio
import threading
import logging
from typing import Any, Coroutine

# Configure logger
logger = logging.getLogger(__name__)

# Thread-local storage for event loops
_thread_local = threading.local()


def get_or_create_event_loop():
    """
    Get the current thread's event loop or create a new one.
    
    Returns:
        The thread's event loop
    """
    # Return existing loop if it exists and is not closed
    if hasattr(_thread_local, 'event_loop'):
        loop = _thread_local.event_loop
        if not loop.is_closed():
            return loop
    
    # Create new event loop
    _thread_local.event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_thread_local.event_loop)
    logger.debug("Created new event loop for thread")
    return _thread_local.event_loop


def run_coroutine(coro: Coroutine) -> Any:
    """
    Run a coroutine using the thread's event loop.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
    """
    loop = get_or_create_event_loop()
    return loop.run_until_complete(coro)


def close_event_loop():
    """
    Close the current thread's event loop if it exists.
    
    This ensures all pending tasks are properly canceled and
    the event loop is closed, preventing resource leaks.
    """
    if hasattr(_thread_local, 'event_loop'):
        loop = _thread_local.event_loop
        
        # Cancel pending tasks
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                logger.debug(f"Cancelling {len(pending)} pending tasks")
                for task in pending:
                    task.cancel()
                
                # Allow tasks to be properly cancelled
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except asyncio.CancelledError:
                    pass
        except RuntimeError as e:
            # Handle case where loop is already closed or running
            logger.debug(f"Error cleaning up tasks: {e}")
        
        # Close the loop
        try:
            if not loop.is_closed():
                loop.close()
                logger.debug("Closed event loop for thread")
        except Exception as e:
            logger.warning(f"Error closing event loop: {e}")
        
        # Remove reference
        del _thread_local.event_loop