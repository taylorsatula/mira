"""
Base class for tools that use asyncio.

This module provides a base class for tools that use asyncio, providing
common functionality for running coroutines and managing thread-local state.
"""

import asyncio
import threading
import logging
from typing import Dict, Any, Coroutine, Optional

from tools.repo import Tool
from utils.async_utils import run_coroutine
from errors import ToolError, ErrorCode, error_context

# Configure logger
logger = logging.getLogger(__name__)


class AsyncioToolBase(Tool):
    """
    Base class for tools that use asyncio.
    
    This class provides common functionality for tools that use asyncio,
    including thread-local state management and coroutine execution.
    Subclasses should override the cleanup method to perform any necessary
    resource cleanup.
    
    Note: This class is meant to be subclassed and not used directly.
    It does not implement the abstract 'run' method from the Tool base class.
    """
    
    
    def __init__(self):
        """Initialize the asyncio tool base."""
        super().__init__()
        self._thread_local = threading.local()
        logger.debug(f"Initialized AsyncioToolBase for {self.name}")
    
    def run_async(self, coro: Coroutine) -> Any:
        """
        Run a coroutine using the thread's event loop.
        
        Args:
            coro: The coroutine to run
            
        Returns:
            The result of the coroutine
        """
        return run_coroutine(coro)
    
    def get_thread_data(self, key: str, default: Any = None) -> Any:
        """
        Get thread-specific data for this tool.
        
        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The value associated with the key, or the default if not found
        """
        if not hasattr(self._thread_local, 'data'):
            self._thread_local.data = {}
        return self._thread_local.data.get(key, default)
    
    def set_thread_data(self, key: str, value: Any) -> None:
        """
        Set thread-specific data for this tool.
        
        Args:
            key: The key to set
            value: The value to store
        """
        if not hasattr(self._thread_local, 'data'):
            self._thread_local.data = {}
        self._thread_local.data[key] = value
        
    def has_thread_data(self, key: str) -> bool:
        """
        Check if thread-specific data exists for this key.
        
        Args:
            key: The key to check
            
        Returns:
            True if the key exists, False otherwise
        """
        if not hasattr(self._thread_local, 'data'):
            return False
        return key in self._thread_local.data
    
    def remove_thread_data(self, key: str) -> None:
        """
        Remove thread-specific data for this key.
        
        Args:
            key: The key to remove
        """
        if hasattr(self._thread_local, 'data') and key in self._thread_local.data:
            del self._thread_local.data[key]
    
    def cleanup(self) -> None:
        """
        Clean up resources for this thread.
        
        This method should be overridden by subclasses to perform any
        necessary resource cleanup. The base implementation simply
        clears the thread-local data.
        """
        if hasattr(self._thread_local, 'data'):
            # Attempt to close any asyncio resources
            for key, value in list(self._thread_local.data.items()):
                if hasattr(value, 'close'):
                    try:
                        if asyncio.iscoroutinefunction(value.close):
                            self.run_async(value.close())
                        else:
                            value.close()
                    except Exception as e:
                        logger.warning(f"Error closing resource {key}: {e}")
            
            # Clear the data
            self._thread_local.data = {}
            logger.debug(f"Cleaned up thread-local data for {self.name}")
    
    @property
    def thread_id(self) -> int:
        """
        Get the current thread ID.
        
        Returns:
            The current thread ID
        """
        return threading.get_ident()