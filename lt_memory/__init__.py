"""
LT_Memory - Long-term memory system for MIRA.

This package provides persistent, self-managed memory capabilities inspired by MemGPT,
with advanced features for vector search, knowledge graphs, and automated consolidation.
"""

from lt_memory.managers.memory_manager import MemoryManager
from lt_memory.bridge import MemoryBridge
from lt_memory.tools.memory_tool import LTMemoryTool

__version__ = "0.1.0"
__all__ = ["MemoryManager", "MemoryBridge", "LTMemoryTool"]