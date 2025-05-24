"""
Manager for core memory blocks with self-editing capabilities.
"""

import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, UTC

from sqlalchemy.orm import Session
from jinja2 import Template

from lt_memory.models.base import MemoryBlock, BlockHistory
from errors import ToolError, ErrorCode, error_context

logger = logging.getLogger(__name__)


class BlockManager:
    """
    Manages core memory blocks.
    
    Provides the self-editing memory functions that allow the system
    to modify its own persistent context.
    """
    
    def __init__(self, memory_manager):
        """
        Initialize block manager.
        
        Args:
            memory_manager: Parent MemoryManager instance
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
    
    def get_block(self, label: str) -> Optional[Dict[str, Any]]:
        """
        Get a memory block by label.
        
        Args:
            label: Block label to retrieve
            
        Returns:
            Block data dictionary or None if not found
        """
        with self.memory_manager.get_session() as session:
            block = session.query(MemoryBlock).filter_by(label=label).first()
            if block:
                return {
                    "id": str(block.id),
                    "label": block.label,
                    "value": block.value,
                    "limit": block.character_limit,
                    "characters": len(block.value),
                    "version": block.version,
                    "updated_at": block.updated_at.isoformat()
                }
        return None
    
    def get_all_blocks(self) -> List[Dict[str, Any]]:
        """Get all memory blocks."""
        with self.memory_manager.get_session() as session:
            blocks = session.query(MemoryBlock).order_by(MemoryBlock.label).all()
            return [
                {
                    "id": str(block.id),
                    "label": block.label,
                    "value": block.value,
                    "limit": block.character_limit,
                    "characters": len(block.value),
                    "version": block.version,
                    "updated_at": block.updated_at.isoformat()
                }
                for block in blocks
            ]
    
    def core_memory_append(self, label: str, content: str, 
                          actor: str = "system") -> Dict[str, Any]:
        """
        Append content to a memory block.
        
        Args:
            label: Block label to append to
            content: Content to append
            actor: Who is making the change
            
        Returns:
            Updated block data
            
        Raises:
            ToolError: If block not found or would exceed limit
        """
        with error_context("block_manager", "append", ToolError, ErrorCode.MEMORY_ERROR):
            with self.memory_manager.get_session() as session:
                block = session.query(MemoryBlock).filter_by(label=label).first()
                if not block:
                    raise ToolError(
                        f"Memory block '{label}' not found",
                        error_code=ErrorCode.NOT_FOUND
                    )
                
                old_value = block.value
                # Add newline if block has content
                separator = "\n" if old_value and not old_value.endswith("\n") else ""
                new_value = old_value + separator + content
                
                if len(new_value) > block.character_limit:
                    raise ToolError(
                        f"Content would exceed character limit "
                        f"({len(new_value)}/{block.character_limit}). "
                        f"Consider using core_memory_replace to update existing content.",
                        error_code=ErrorCode.INVALID_INPUT
                    )
                
                # Update block
                block.value = new_value
                block.version += 1
                block.updated_at = datetime.now(UTC)
                
                # Record history
                history = BlockHistory(
                    block_id=block.id,
                    label=label,
                    old_value=old_value,
                    new_value=new_value,
                    operation="append",
                    actor=actor
                )
                session.add(history)
                session.commit()
                
                self.logger.info(f"Appended {len(content)} chars to block '{label}'")
                return self.get_block(label)
    
    def core_memory_replace(self, label: str, old_content: str, new_content: str, 
                           actor: str = "system") -> Dict[str, Any]:
        """
        Replace content in a memory block.
        
        Args:
            label: Block label to modify
            old_content: Exact content to replace
            new_content: New content to insert
            actor: Who is making the change
            
        Returns:
            Updated block data
            
        Raises:
            ToolError: If block not found, content not found, or would exceed limit
        """
        with error_context("block_manager", "replace", ToolError, ErrorCode.MEMORY_ERROR):
            with self.memory_manager.get_session() as session:
                block = session.query(MemoryBlock).filter_by(label=label).first()
                if not block:
                    raise ToolError(
                        f"Memory block '{label}' not found",
                        error_code=ErrorCode.NOT_FOUND
                    )
                
                if old_content not in block.value:
                    # Provide helpful error with snippet of block content
                    preview = block.value[:200] + "..." if len(block.value) > 200 else block.value
                    raise ToolError(
                        f"Content to replace not found in block '{label}'. "
                        f"Block starts with: {preview}",
                        error_code=ErrorCode.INVALID_INPUT
                    )
                
                old_value = block.value
                # Only replace first occurrence
                new_value = old_value.replace(old_content, new_content, 1)
                
                if len(new_value) > block.character_limit:
                    raise ToolError(
                        f"Replacement would exceed character limit "
                        f"({len(new_value)}/{block.character_limit}). "
                        f"Consider removing content first or using memory_rethink.",
                        error_code=ErrorCode.INVALID_INPUT
                    )
                
                # Update block
                block.value = new_value
                block.version += 1
                block.updated_at = datetime.now(UTC)
                
                # Record history
                history = BlockHistory(
                    block_id=block.id,
                    label=label,
                    old_value=old_value,
                    new_value=new_value,
                    operation="replace",
                    actor=actor
                )
                session.add(history)
                session.commit()
                
                self.logger.info(f"Replaced content in block '{label}'")
                return self.get_block(label)
    
    def memory_insert(self, label: str, content: str, line_number: int, 
                     actor: str = "system") -> Dict[str, Any]:
        """
        Insert content at a specific line number.
        
        Args:
            label: Block label to modify
            content: Content to insert
            line_number: Line number to insert at (1-based)
            actor: Who is making the change
            
        Returns:
            Updated block data
            
        Raises:
            ToolError: If block not found, invalid line number, or would exceed limit
        """
        with error_context("block_manager", "insert", ToolError, ErrorCode.MEMORY_ERROR):
            with self.memory_manager.get_session() as session:
                block = session.query(MemoryBlock).filter_by(label=label).first()
                if not block:
                    raise ToolError(
                        f"Memory block '{label}' not found",
                        error_code=ErrorCode.NOT_FOUND
                    )
                
                # Split into lines
                lines = block.value.split('\n') if block.value else []
                
                # Validate line number
                if line_number < 1 or line_number > len(lines) + 1:
                    raise ToolError(
                        f"Invalid line number: {line_number}. "
                        f"Block has {len(lines)} lines. "
                        f"Valid range: 1 to {len(lines) + 1}",
                        error_code=ErrorCode.INVALID_INPUT
                    )
                
                old_value = block.value
                # Insert at specified position (convert to 0-based index)
                lines.insert(line_number - 1, content)
                new_value = '\n'.join(lines)
                
                if len(new_value) > block.character_limit:
                    raise ToolError(
                        f"Insertion would exceed character limit "
                        f"({len(new_value)}/{block.character_limit})",
                        error_code=ErrorCode.INVALID_INPUT
                    )
                
                # Update block
                block.value = new_value
                block.version += 1
                block.updated_at = datetime.now(UTC)
                
                # Record history
                history = BlockHistory(
                    block_id=block.id,
                    label=label,
                    old_value=old_value,
                    new_value=new_value,
                    operation="insert",
                    actor=actor
                )
                session.add(history)
                session.commit()
                
                self.logger.info(f"Inserted content at line {line_number} in block '{label}'")
                return self.get_block(label)
    
    def memory_rethink(self, label: str, new_content: str, 
                      actor: str = "system") -> Dict[str, Any]:
        """
        Completely rewrite a memory block.
        
        Args:
            label: Block label to rewrite
            new_content: New content for the block
            actor: Who is making the change
            
        Returns:
            Updated block data
            
        Raises:
            ToolError: If block not found or content exceeds limit
        """
        with error_context("block_manager", "rethink", ToolError, ErrorCode.MEMORY_ERROR):
            with self.memory_manager.get_session() as session:
                block = session.query(MemoryBlock).filter_by(label=label).first()
                if not block:
                    raise ToolError(
                        f"Memory block '{label}' not found",
                        error_code=ErrorCode.NOT_FOUND
                    )
                
                if len(new_content) > block.character_limit:
                    raise ToolError(
                        f"New content exceeds character limit "
                        f"({len(new_content)}/{block.character_limit})",
                        error_code=ErrorCode.INVALID_INPUT
                    )
                
                old_value = block.value
                
                # Update block
                block.value = new_content
                block.version += 1
                block.updated_at = datetime.now(UTC)
                
                # Record history
                history = BlockHistory(
                    block_id=block.id,
                    label=label,
                    old_value=old_value,
                    new_value=new_content,
                    operation="rethink",
                    actor=actor
                )
                session.add(history)
                session.commit()
                
                self.logger.info(f"Rewrote block '{label}'")
                return self.get_block(label)
    
    def render_blocks(self, template: Optional[str] = None) -> str:
        """
        Render memory blocks for inclusion in prompt.
        
        Args:
            template: Optional Jinja2 template string
            
        Returns:
            Rendered memory blocks as string
        """
        blocks = self.get_all_blocks()
        
        if not template:
            # Default template mimicking MemGPT format
            template = """{% for block in blocks -%}
<{{ block.label }} characters="{{ block.characters }}/{{ block.limit }}">
{{ block.value }}
</{{ block.label }}>
{% endfor %}"""
        
        tmpl = Template(template)
        return tmpl.render(blocks=blocks)
    
    def get_block_history(self, label: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of changes for a block.
        
        Args:
            label: Block label
            limit: Maximum number of history entries
            
        Returns:
            List of history entries
        """
        with self.memory_manager.get_session() as session:
            # First get the block
            block = session.query(MemoryBlock).filter_by(label=label).first()
            if not block:
                return []
            
            # Get history
            history = session.query(BlockHistory).filter_by(
                block_id=block.id
            ).order_by(
                BlockHistory.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    "id": str(h.id),
                    "operation": h.operation,
                    "actor": h.actor,
                    "created_at": h.created_at.isoformat(),
                    "old_value_preview": h.old_value[:100] + "..." if len(h.old_value) > 100 else h.old_value,
                    "new_value_preview": h.new_value[:100] + "..." if len(h.new_value) > 100 else h.new_value
                }
                for h in history
            ]
    
    def rollback_block(self, label: str, version: Optional[int] = None,
                      actor: str = "system") -> Dict[str, Any]:
        """
        Rollback a block to a previous version.
        
        Args:
            label: Block label
            version: Version to rollback to (or previous if None)
            actor: Who is performing the rollback
            
        Returns:
            Updated block data
        """
        with self.memory_manager.get_session() as session:
            block = session.query(MemoryBlock).filter_by(label=label).first()
            if not block:
                raise ToolError(
                    f"Memory block '{label}' not found",
                    error_code=ErrorCode.NOT_FOUND
                )
            
            # Get the history entry to rollback to
            history_query = session.query(BlockHistory).filter_by(
                block_id=block.id
            ).order_by(BlockHistory.created_at.desc())
            
            if version:
                # Skip to specific version
                history_entries = history_query.limit(block.version - version + 1).all()
                if not history_entries:
                    raise ToolError(
                        f"Version {version} not found in history",
                        error_code=ErrorCode.NOT_FOUND
                    )
                target_history = history_entries[-1]
            else:
                # Just get the previous version
                target_history = history_query.first()
                if not target_history:
                    raise ToolError(
                        "No history available for rollback",
                        error_code=ErrorCode.NOT_FOUND
                    )
            
            # Perform rollback
            old_value = block.value
            block.value = target_history.old_value
            block.version += 1
            block.updated_at = datetime.now(UTC)
            
            # Record rollback in history
            history = BlockHistory(
                block_id=block.id,
                label=label,
                old_value=old_value,
                new_value=target_history.old_value,
                operation="rollback",
                actor=actor
            )
            session.add(history)
            session.commit()
            
            self.logger.info(f"Rolled back block '{label}'")
            return self.get_block(label)