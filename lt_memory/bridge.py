"""
Bridge between WorkingMemory and LT_Memory systems.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, UTC, timedelta

from working_memory import WorkingMemory

logger = logging.getLogger(__name__)


class MemoryBridge:
    """
    Bridge between WorkingMemory and LT_Memory systems.
    
    Updates working memory with relevant long-term memories
    during conversation processing.
    """
    
    def __init__(self, working_memory: WorkingMemory, memory_manager):
        """
        Initialize memory bridge.
        
        Args:
            working_memory: WorkingMemory instance
            memory_manager: MemoryManager instance
        """
        self.working_memory = working_memory
        self.memory_manager = memory_manager
        self._memory_ids = {}  # Track working memory item IDs
        
        # Register with working memory as a manager
        working_memory.register_manager(self)
        
        logger.info("Memory bridge initialized")
    
    def update_working_memory(self):
        """
        Update working memory with relevant LT_Memory content.
        
        This is called before each response generation to ensure
        relevant memories are available in context.
        """
        try:
            # Clear previous memory items
            for item_id in self._memory_ids.values():
                self.working_memory.remove(item_id)
            self._memory_ids.clear()
            
            # Add core memory blocks
            self._add_core_memory()
            
            # Add recent insights from batch processing
            self._add_recent_insights()
            
            # Add relevant entities if any are active
            self._add_active_entities()
            
            # Add memory statistics for transparency
            self._add_memory_stats()
            
        except Exception as e:
            logger.error(f"Error updating working memory from LT_Memory: {e}")
    
    def _add_core_memory(self):
        """Add core memory blocks to working memory."""
        try:
            blocks_content = self.memory_manager.block_manager.render_blocks()
            if blocks_content:
                item_id = self.working_memory.add(
                    content=f"# Core Memory\n{blocks_content}",
                    category="lt_memory_core"
                )
                self._memory_ids["core"] = item_id
                logger.debug("Added core memory blocks to working memory")
        except Exception as e:
            logger.error(f"Failed to add core memory: {e}")
    
    def _add_recent_insights(self):
        """Add recently processed memories as insights."""
        try:
            # Get high-importance memories from recent processing
            recent_memories = self._get_recent_relevant_memories()
            
            if recent_memories:
                memory_text = self._format_memories(recent_memories)
                item_id = self.working_memory.add(
                    content=f"# Recent Insights\n{memory_text}",
                    category="lt_memory_insights"
                )
                self._memory_ids["insights"] = item_id
                logger.debug(f"Added {len(recent_memories)} recent insights")
        except Exception as e:
            logger.error(f"Failed to add recent insights: {e}")
    
    def _add_active_entities(self):
        """Add recently active entities to context."""
        try:
            active_entities = self._get_active_entities()
            
            if active_entities:
                entity_text = self._format_entities(active_entities)
                item_id = self.working_memory.add(
                    content=f"# Known Entities\n{entity_text}",
                    category="lt_memory_entities"
                )
                self._memory_ids["entities"] = item_id
                logger.debug(f"Added {len(active_entities)} active entities")
        except Exception as e:
            logger.error(f"Failed to add active entities: {e}")
    
    def _add_memory_stats(self):
        """Add memory system statistics for transparency."""
        try:
            stats = self.memory_manager.get_memory_stats()
            
            # Format key statistics
            stats_text = (
                f"Memory System: "
                f"{stats['passages']['count']} memories, "
                f"{stats['entities']['count']} entities, "
                f"{stats['relations']['count']} relationships"
            )
            
            item_id = self.working_memory.add(
                content=f"# Memory Status\n{stats_text}",
                category="lt_memory_status"
            )
            self._memory_ids["status"] = item_id
            
        except Exception as e:
            logger.error(f"Failed to add memory stats: {e}")
    
    def _get_recent_relevant_memories(self, hours: int = 24, 
                                     limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recently processed memories with high importance.
        
        Args:
            hours: How many hours back to look
            limit: Maximum memories to return
            
        Returns:
            List of memory passages
        """
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        
        with self.memory_manager.get_session() as session:
            from lt_memory.models.base import MemoryPassage
            
            recent = session.query(MemoryPassage).filter(
                (MemoryPassage.created_at > cutoff) &
                (MemoryPassage.importance_score > 0.7)
            ).order_by(
                MemoryPassage.importance_score.desc()
            ).limit(limit).all()
            
            return [
                {
                    "text": p.text,
                    "importance": p.importance_score,
                    "source": p.source,
                    "created_at": p.created_at
                }
                for p in recent
            ]
    
    def _get_active_entities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently mentioned entities.
        
        Args:
            limit: Maximum entities to return
            
        Returns:
            List of active entities
        """
        with self.memory_manager.get_session() as session:
            from lt_memory.models.base import MemoryEntity
            
            # Get entities seen in last 7 days with high importance
            cutoff = datetime.now(UTC) - timedelta(days=7)
            
            entities = session.query(MemoryEntity).filter(
                (MemoryEntity.last_seen > cutoff) &
                (MemoryEntity.importance_score > 0.6)
            ).order_by(
                MemoryEntity.importance_score.desc(),
                MemoryEntity.last_seen.desc()
            ).limit(limit).all()
            
            return [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "importance": e.importance_score,
                    "mentions": e.mention_count
                }
                for e in entities
            ]
    
    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format memories for display in working memory.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            Formatted string
        """
        lines = []
        
        for mem in memories:
            # Truncate long memories
            text = mem['text']
            if len(text) > 200:
                text = text[:197] + "..."
            
            lines.append(
                f"- [{mem['source']}] {text} "
                f"(importance: {mem['importance']:.2f})"
            )
        
        return "\n".join(lines)
    
    def _format_entities(self, entities: List[Dict[str, Any]]) -> str:
        """
        Format entities for display in working memory.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Formatted string
        """
        lines = []
        
        # Group by type
        by_type = {}
        for entity in entities:
            entity_type = entity['type'] or 'other'
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append(entity)
        
        # Format by type
        for entity_type, type_entities in by_type.items():
            lines.append(f"\n{entity_type.title()}s:")
            for entity in type_entities:
                lines.append(
                    f"- {entity['name']} "
                    f"({entity['mentions']} mentions)"
                )
        
        return "\n".join(lines)
    
    def search_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for memories relevant to a query.
        
        This can be called by tools or other components to find
        relevant historical context.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of relevant memories
        """
        return self.memory_manager.passage_manager.search_passages(
            query=query,
            limit=limit,
            filters={"min_importance": 0.5}
        )
    
    def get_entity_context(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Get full context for a specific entity.
        
        Args:
            entity_name: Name of entity to look up
            
        Returns:
            Entity graph data or None
        """
        return self.memory_manager.entity_manager.get_entity_graph(
            entity_name=entity_name,
            depth=2
        )