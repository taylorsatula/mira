"""
Manager for core memory blocks with self-editing capabilities.
"""

import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, UTC

from sqlalchemy.orm import Session
from jinja2 import Template, Environment
from markupsafe import escape

from lt_memory.models.base import MemoryBlock, BlockHistory
from lt_memory.utils.diff_engine import DiffEngine
from utils.timezone_utils import utc_now, format_utc_iso
from errors import ToolError, ErrorCode, error_context

logger = logging.getLogger(__name__)


class BlockManager:
    """
    Manages core memory blocks.
    
    Provides the self-editing memory functions that allow the system
    to modify its own persistent context.
    """
    
    # Protected block labels that require specific actor permissions
    PROTECTED_BLOCKS = {
        # Example: "core_identity": ["self_reflection", "persona_optimizer"],
        # Add protected blocks and their allowed actors when needed
    }
    
    def __init__(self, memory_manager):
        """
        Initialize block manager.
        
        Args:
            memory_manager: Parent MemoryManager instance
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
        
        # Determine once if write protection is needed
        self.write_protection_enabled = bool(self.PROTECTED_BLOCKS)
    
    def _check_write_permission(self, label: str, actor: str) -> None:
        """
        Check if the actor has permission to modify a protected block.
        
        Args:
            label: Block label to check
            actor: Actor attempting the modification
            
        Raises:
            ToolError: If the block is protected and actor lacks permission
        """
        # Fast path - skip entirely if no protection needed
        if not self.write_protection_enabled:
            return
            
        if label in self.PROTECTED_BLOCKS:
            allowed_actors = self.PROTECTED_BLOCKS[label]
            if actor not in allowed_actors and actor != "system":
                raise ToolError(
                    f"Block '{label}' is protected. Only {allowed_actors} can modify it. "
                    f"Current actor: '{actor}'",
                    code=ErrorCode.PERMISSION_DENIED
                )
    
    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize user input to prevent database issues and security problems.
        
        Args:
            text: Raw user input text
            
        Returns:
            Sanitized text safe for database storage
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove null bytes and other problematic control characters that PostgreSQL doesn't like
        # Keep standard printable chars, newlines, tabs, and common unicode
        sanitized = ""
        for char in text:
            # Allow normal printable ASCII, newlines, tabs, and unicode above 127
            if (32 <= ord(char) <= 126) or char in '\n\r\t' or ord(char) > 127:
                sanitized += char
            # Replace problematic control characters with space
            elif ord(char) < 32:
                sanitized += " "
        
        return sanitized
    
    def create_block(self, label: str, content: str, character_limit: int = 2048,
                    actor: str = "system") -> Dict[str, Any]:
        """
        Create a new memory block with base version history.
        
        Args:
            label: Block label
            content: Initial content
            character_limit: Character limit for the block
            actor: Who is creating the block
            
        Returns:
            Created block data
        """
        with error_context("block_manager", "create", ToolError, ErrorCode.MEMORY_ERROR):
            # Sanitize all user inputs
            label = self._sanitize_input(label)
            content = self._sanitize_input(content)
            actor = self._sanitize_input(actor)
            
            with self.memory_manager.get_session() as session:
                # Check if block already exists
                existing = session.query(MemoryBlock).filter_by(label=label).first()
                if existing:
                    raise ToolError(
                        f"Memory block '{label}' already exists",
                        code=ErrorCode.MEMORY_BLOCK_ALREADY_EXISTS
                    )
                
                if len(content) > character_limit:
                    raise ToolError(
                        f"Content exceeds character limit ({len(content)}/{character_limit})",
                        code=ErrorCode.INVALID_INPUT
                    )
                
                # Create block
                block = MemoryBlock(
                    label=label,
                    value=content,
                    character_limit=character_limit,
                    version=1
                )
                session.add(block)
                session.flush()  # Get the ID
                
                # Create base version history entry
                base_data = DiffEngine.create_base_version(content)
                history = BlockHistory(
                    block_id=block.id,
                    label=label,
                    version=1,
                    diff_data=base_data,
                    operation="base",
                    actor=actor
                )
                session.add(history)
                session.commit()
                
                self.logger.info(f"Created block '{label}' with {len(content)} characters")
                return self.get_block(label)
    
    def get_block(self, label: str) -> Optional[Dict[str, Any]]:
        """
        Get a memory block by label.
        
        Args:
            label: Block label to retrieve
            
        Returns:
            Block data dictionary or None if not found
        """
        # Sanitize user input
        label = self._sanitize_input(label)
        
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
                    "updated_at": format_utc_iso(block.updated_at)
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
            # Sanitize user inputs
            label = self._sanitize_input(label)
            content = self._sanitize_input(content)
            actor = self._sanitize_input(actor)
            
            # Check write permissions
            self._check_write_permission(label, actor)
            
            with self.memory_manager.get_session() as session:
                block = session.query(MemoryBlock).filter_by(label=label).with_for_update().first()
                if not block:
                    raise ToolError(
                        f"Memory block '{label}' not found",
                        code=ErrorCode.NOT_FOUND
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
                        code=ErrorCode.INVALID_INPUT
                    )
                
                # Update block
                block.value = new_value
                block.version += 1
                block.updated_at = utc_now()
                
                # Record history with differential storage or snapshot
                if DiffEngine.should_create_snapshot(block.version):
                    # Create snapshot for recovery
                    diff_data = DiffEngine.create_snapshot_version(new_value, block.version)
                    self.logger.info(f"Created snapshot for block '{label}' version {block.version}")
                else:
                    # Create diff
                    diff_data = DiffEngine.create_diff_version(old_value, new_value, block.version)
                
                history = BlockHistory(
                    block_id=block.id,
                    label=label,
                    version=block.version,
                    diff_data=diff_data,
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
            # Sanitize user inputs
            label = self._sanitize_input(label)
            old_content = self._sanitize_input(old_content)
            new_content = self._sanitize_input(new_content)
            actor = self._sanitize_input(actor)
            
            # Check write permissions
            self._check_write_permission(label, actor)
            
            with self.memory_manager.get_session() as session:
                block = session.query(MemoryBlock).filter_by(label=label).with_for_update().first()
                if not block:
                    raise ToolError(
                        f"Memory block '{label}' not found",
                        code=ErrorCode.NOT_FOUND
                    )
                
                if old_content not in block.value:
                    # Provide helpful error with snippet of block content
                    preview = block.value[:200] + "..." if len(block.value) > 200 else block.value
                    raise ToolError(
                        f"Content to replace not found in block '{label}'. "
                        f"Block starts with: {preview}",
                        code=ErrorCode.INVALID_INPUT
                    )
                
                old_value = block.value
                # Only replace first occurrence
                new_value = old_value.replace(old_content, new_content, 1)
                
                if len(new_value) > block.character_limit:
                    raise ToolError(
                        f"Replacement would exceed character limit "
                        f"({len(new_value)}/{block.character_limit}). "
                        f"Consider removing content first or using memory_rethink.",
                        code=ErrorCode.INVALID_INPUT
                    )
                
                # Update block
                block.value = new_value
                block.version += 1
                block.updated_at = utc_now()
                
                # Record history with differential storage
                diff_data = DiffEngine.create_diff_version(old_value, new_value, block.version)
                history = BlockHistory(
                    block_id=block.id,
                    label=label,
                    version=block.version,
                    diff_data=diff_data,
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
            # Sanitize user inputs
            label = self._sanitize_input(label)
            content = self._sanitize_input(content)
            actor = self._sanitize_input(actor)
            
            # Check write permissions
            self._check_write_permission(label, actor)
            
            with self.memory_manager.get_session() as session:
                block = session.query(MemoryBlock).filter_by(label=label).with_for_update().first()
                if not block:
                    raise ToolError(
                        f"Memory block '{label}' not found",
                        code=ErrorCode.NOT_FOUND
                    )
                
                # Split into lines
                lines = block.value.split('\n') if block.value else []
                
                # Validate line number
                if line_number < 1 or line_number > len(lines) + 1:
                    raise ToolError(
                        f"Invalid line number: {line_number}. "
                        f"Block has {len(lines)} lines. "
                        f"Valid range: 1 to {len(lines) + 1}",
                        code=ErrorCode.INVALID_INPUT
                    )
                
                old_value = block.value
                # Insert at specified position (convert to 0-based index)
                lines.insert(line_number - 1, content)
                new_value = '\n'.join(lines)
                
                if len(new_value) > block.character_limit:
                    raise ToolError(
                        f"Insertion would exceed character limit "
                        f"({len(new_value)}/{block.character_limit})",
                        code=ErrorCode.INVALID_INPUT
                    )
                
                # Update block
                block.value = new_value
                block.version += 1
                block.updated_at = utc_now()
                
                # Record history with differential storage
                diff_data = DiffEngine.create_diff_version(old_value, new_value, block.version)
                history = BlockHistory(
                    block_id=block.id,
                    label=label,
                    version=block.version,
                    diff_data=diff_data,
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
            # Sanitize user inputs
            label = self._sanitize_input(label)
            new_content = self._sanitize_input(new_content)
            actor = self._sanitize_input(actor)
            
            # Check write permissions
            self._check_write_permission(label, actor)
            
            with self.memory_manager.get_session() as session:
                block = session.query(MemoryBlock).filter_by(label=label).with_for_update().first()
                if not block:
                    raise ToolError(
                        f"Memory block '{label}' not found",
                        code=ErrorCode.NOT_FOUND
                    )
                
                if len(new_content) > block.character_limit:
                    raise ToolError(
                        f"New content exceeds character limit "
                        f"({len(new_content)}/{block.character_limit})",
                        code=ErrorCode.INVALID_INPUT
                    )
                
                old_value = block.value
                
                # Update block
                block.value = new_content
                block.version += 1
                block.updated_at = utc_now()
                
                # Record history with differential storage
                diff_data = DiffEngine.create_diff_version(old_value, new_content, block.version)
                history = BlockHistory(
                    block_id=block.id,
                    label=label,
                    version=block.version,
                    diff_data=diff_data,
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
        
        # Pre-escape all user content to prevent template injection
        safe_blocks = []
        for block in blocks:
            safe_block = block.copy()
            safe_block["value"] = escape(block["value"])  # Escape user content
            safe_block["label"] = escape(block["label"])  # Escape label too
            safe_blocks.append(safe_block)
        
        if not template:
            # Default template mimicking MemGPT format
            template = """{% for block in blocks -%}
<{{ block.label }} characters="{{ block.characters }}/{{ block.limit }}">
{{ block.value }}
</{{ block.label }}>
{% endfor %}"""
        
        # Create environment with auto-escaping disabled since we pre-escaped
        env = Environment(autoescape=False)
        tmpl = env.from_string(template)
        return tmpl.render(blocks=safe_blocks)
    
    def get_block_history(self, label: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of changes for a block.
        
        Args:
            label: Block label
            limit: Maximum number of history entries
            
        Returns:
            List of history entries
        """
        # Sanitize user input
        label = self._sanitize_input(label)
        
        with self.memory_manager.get_session() as session:
            # First get the block
            block = session.query(MemoryBlock).filter_by(label=label).first()
            if not block:
                return []
            
            # Get history
            history = session.query(BlockHistory).filter_by(
                block_id=block.id
            ).order_by(
                BlockHistory.version.desc()
            ).limit(limit).all()
            
            return [
                {
                    "id": str(h.id),
                    "version": h.version,
                    "operation": h.operation,
                    "actor": h.actor,
                    "created_at": format_utc_iso(h.created_at)
                }
                for h in history
            ]
    
    def rollback_block(self, label: str, version: Optional[int] = None,
                      actor: str = "system") -> Dict[str, Any]:
        """
        Rollback a block to a previous version using differential reconstruction.
        
        Args:
            label: Block label
            version: Version to rollback to (or previous if None)
            actor: Who is performing the rollback
            
        Returns:
            Updated block data
        """
        # Sanitize user input
        label = self._sanitize_input(label)
        actor = self._sanitize_input(actor)
        
        with self.memory_manager.get_session() as session:
            block = session.query(MemoryBlock).filter_by(label=label).with_for_update().first()
            if not block:
                raise ToolError(
                    f"Memory block '{label}' not found",
                    code=ErrorCode.NOT_FOUND
                )
            
            target_version = version if version else block.version - 1
            if target_version < 1:
                raise ToolError(
                    "Cannot rollback to version less than 1",
                    code=ErrorCode.INVALID_INPUT
                )
            
            # Get all history for this block, ordered by version
            all_history = session.query(BlockHistory).filter_by(
                block_id=block.id
            ).order_by(BlockHistory.version.asc()).all()
            
            if not all_history:
                raise ToolError(
                    "No history available for rollback",
                    code=ErrorCode.NOT_FOUND
                )
            
            # Find base version (version 1)
            base_history = next((h for h in all_history if h.version == 1), None)
            if not base_history:
                raise ToolError(
                    "No base version found in history",
                    code=ErrorCode.DATA_CORRUPTION
                )
            
            # Get base content
            base_content = base_history.diff_data.get("content", "")
            
            # Reconstruct target version using new DiffEngine API
            try:
                rollback_content = DiffEngine.reconstruct_version(
                    base_content, 
                    [h.diff_data for h in all_history], 
                    target_version
                )
            except Exception as e:
                raise ToolError(
                    f"Failed to reconstruct version {target_version}: {str(e)}",
                    code=ErrorCode.DATA_CORRUPTION
                )
            
            # Perform rollback
            old_value = block.value
            block.value = rollback_content
            block.version += 1
            block.updated_at = utc_now()
            
            # Record rollback in history with differential storage
            diff_data = DiffEngine.create_diff_version(old_value, rollback_content, block.version)
            history = BlockHistory(
                block_id=block.id,
                label=label,
                version=block.version,
                diff_data=diff_data,
                operation="rollback",
                actor=actor
            )
            session.add(history)
            session.commit()
            
            self.logger.info(f"Rolled back block '{label}' to version {target_version}")
            return self.get_block(label)