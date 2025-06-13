"""
Tool interface for LT_Memory operations.
"""

import json
import logging
from typing import Dict, Any, Optional, List

from errors import ToolError, ErrorCode, error_context

logger = logging.getLogger(__name__)


class LTMemoryTool:
    """
    Tool interface for LT_Memory operations.
    
    Provides access to memory management functions for use
    by the AI assistant and automation system.
    """
    
    def __init__(self, memory_manager):
        """
        Initialize memory tool.
        
        Args:
            memory_manager: MemoryManager instance
        """
        self.memory_manager = memory_manager
        self.name = "lt_memory"
        self.description = """Manage long-term memory with core memory editing and archival search.

Operations:
- core_memory_append: Add content to a memory block
- core_memory_replace: Replace content in a memory block  
- memory_insert: Insert content at specific line
- memory_rethink: Completely rewrite a memory block
- search_archival: Search through archived memories
- process_recent_conversations: Process recent conversations into memory
- consolidate_memories: Trigger memory consolidation
- get_memory_stats: Get memory system statistics

Examples:
- Append to persona: {"operation": "core_memory_append", "label": "persona", "content": "I enjoy helping with coding."}
- Search memories: {"operation": "search_archival", "query": "user preferences", "limit": 5}
"""
    
    def run(self, operation: str, **params) -> Dict[str, Any]:
        """
        Execute memory operation.
        
        Args:
            operation: Operation to perform
            **params: Operation-specific parameters
            
        Returns:
            Operation results
            
        Raises:
            ToolError: If operation fails
        """
        with error_context(ErrorCode.TOOL_EXECUTION_ERROR, {
            "tool": self.name,
            "operation": operation,
            "params": params
        }):
            # Core memory operations
            if operation == "core_memory_append":
                return self._core_memory_append(**params)
            elif operation == "core_memory_replace":
                return self._core_memory_replace(**params)
            elif operation == "memory_insert":
                return self._memory_insert(**params)
            elif operation == "memory_rethink":
                return self._memory_rethink(**params)
            
            # Archival memory operations
            elif operation == "search_archival":
                return self._search_archival(**params)
            
            
            # Batch processing operations
            elif operation == "process_recent_conversations":
                return self._process_recent_conversations(**params)
            elif operation == "process_all_pending":
                return self._process_all_pending(**params)
            
            # Maintenance operations
            elif operation == "consolidate_memories":
                return self._consolidate_memories(**params)
            elif operation == "update_importance_scores":
                return self._update_importance_scores(**params)
            
            # Information operations
            elif operation == "get_memory_stats":
                return self._get_memory_stats(**params)
            elif operation == "get_core_memory":
                return self._get_core_memory(**params)
            
            else:
                raise ToolError(
                    f"Unknown operation: {operation}",
                    error_code=ErrorCode.INVALID_INPUT
                )
    
    def _core_memory_append(self, label: str, content: str) -> Dict[str, Any]:
        """Append to core memory block."""
        result = self.memory_manager.block_manager.core_memory_append(
            label=label,
            content=content,
            actor="tool"
        )
        
        return {
            "success": True,
            "message": f"Appended {len(content)} characters to {label}",
            "block": result
        }
    
    def _core_memory_replace(self, label: str, old_content: str, 
                            new_content: str) -> Dict[str, Any]:
        """Replace content in core memory block."""
        result = self.memory_manager.block_manager.core_memory_replace(
            label=label,
            old_content=old_content,
            new_content=new_content,
            actor="tool"
        )
        
        return {
            "success": True,
            "message": f"Replaced content in {label}",
            "block": result
        }
    
    def _memory_insert(self, label: str, content: str, 
                      line_number: int) -> Dict[str, Any]:
        """Insert content at specific line in core memory."""
        result = self.memory_manager.block_manager.memory_insert(
            label=label,
            content=content,
            line_number=line_number,
            actor="tool"
        )
        
        return {
            "success": True,
            "message": f"Inserted content at line {line_number} in {label}",
            "block": result
        }
    
    def _memory_rethink(self, label: str, new_content: str) -> Dict[str, Any]:
        """Completely rewrite a core memory block."""
        result = self.memory_manager.block_manager.memory_rethink(
            label=label,
            new_content=new_content,
            actor="tool"
        )
        
        return {
            "success": True,
            "message": f"Rewrote {label} block",
            "block": result
        }
    
    def _search_archival(self, query: str, limit: int = 10,
                        source: Optional[str] = None,
                        min_importance: Optional[float] = None) -> Dict[str, Any]:
        """Search archival memory."""
        filters = {}
        if source:
            filters["source"] = source
        if min_importance is not None:
            filters["min_importance"] = min_importance
        
        results = self.memory_manager.passage_manager.search_passages(
            query=query,
            limit=limit,
            filters=filters
        )
        
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results
        }
    
    
    def _process_recent_conversations(self, hours: Optional[int] = None) -> Dict[str, Any]:
        """Process recent conversations into memory via conversation archive."""
        # This operation is now handled automatically by the conversation archive system
        # when conversations are archived daily
        return {
            "success": True,
            "message": "Conversation processing is handled automatically by the archive system",
            "note": "Use conversation archive operations for manual processing"
        }
    
    def _process_all_pending(self) -> Dict[str, Any]:
        """Process all pending conversations."""
        # This operation is now handled automatically by the conversation archive system
        return {
            "success": True,
            "message": "All conversation processing is handled automatically by the archive system",
            "note": "Conversations are processed when archived daily"
        }
    
    def _consolidate_memories(self) -> Dict[str, Any]:
        """Trigger memory consolidation."""
        results = self.memory_manager.consolidation_engine.consolidate_memories()
        
        return {
            "success": True,
            "message": "Memory consolidation completed",
            "results": results
        }
    
    def _update_importance_scores(self) -> Dict[str, Any]:
        """Update importance scores for all memories."""
        updated = self.memory_manager.consolidation_engine._update_importance_scores()
        
        return {
            "success": True,
            "message": f"Updated importance scores for {updated} items",
            "count": updated
        }
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = self.memory_manager.get_memory_stats()
        health = self.memory_manager.health_check()
        processing_status = {"status": "handled_by_conversation_archive"}
        
        return {
            "success": True,
            "stats": stats,
            "health": health,
            "processing": processing_status
        }
    
    def _get_core_memory(self, label: Optional[str] = None) -> Dict[str, Any]:
        """Get core memory blocks."""
        if label:
            block = self.memory_manager.block_manager.get_block(label)
            if not block:
                return {
                    "success": False,
                    "message": f"Memory block '{label}' not found"
                }
            return {
                "success": True,
                "block": block
            }
        else:
            blocks = self.memory_manager.block_manager.get_all_blocks()
            return {
                "success": True,
                "blocks": blocks
            }
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get tool metadata for registration.
        
        Returns:
            Tool metadata dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": "1.0.0",
            "author": "LT_Memory System"
        }
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Get tool definition for registration.
        
        Returns:
            Tool definition dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform",
                        "enum": [
                            "core_memory_append",
                            "core_memory_replace",
                            "memory_insert",
                            "memory_rethink",
                            "search_archival",
                            "process_recent_conversations",
                            "process_all_pending",
                            "consolidate_memories",
                            "update_importance_scores",
                            "get_memory_stats",
                            "get_core_memory"
                        ]
                    },
                    "label": {
                        "type": "string",
                        "description": "Memory block label (for core memory operations)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to add/insert"
                    },
                    "old_content": {
                        "type": "string",
                        "description": "Content to replace (for replace operation)"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content (for replace/rethink operations)"
                    },
                    "line_number": {
                        "type": "integer",
                        "description": "Line number for insert operation"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return"
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Hours to process"
                    }
                },
                "required": ["operation"]
            },
            "examples": [
                {
                    "operation": "core_memory_append",
                    "label": "human",
                    "content": "The user prefers dark mode interfaces."
                },
                {
                    "operation": "search_archival",
                    "query": "coding preferences",
                    "limit": 5
                },
            ]
        }