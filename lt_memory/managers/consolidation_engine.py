"""
Memory consolidation and optimization engine.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from sqlalchemy import text, and_, or_, func, exists

from lt_memory.models.base import MemoryPassage
from errors import error_context, ErrorCode, ToolError
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """
    Handles memory consolidation and optimization.
    
    Performs scheduled maintenance tasks like pruning old memories,
    merging duplicates, and updating importance scores.
    """
    
    def __init__(self, memory_manager):
        """
        Initialize consolidation engine.
        
        Args:
            memory_manager: Parent MemoryManager instance
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
    
    def consolidate_memories(self, conversation_id: Optional[str] = None) -> Dict[str, int]:
        """
        Run full memory consolidation process.
        
        Args:
            conversation_id: Optional conversation triggering consolidation
            
        Returns:
            Results dictionary with operation counts
        """
        results = {
            "pruned_passages": 0,
            "updated_scores": 0,
            "archived_conversations": 0,
            "optimized_indexes": False
        }
        
        with error_context("consolidation_engine", "consolidate_memories", ToolError, ErrorCode.MEMORY_ERROR):
            # Step 1: Prune old, low-importance passages
            results["pruned_passages"] = self._prune_old_passages()
            
            
            # Step 3: Update importance scores based on usage
            results["updated_scores"] = self._update_importance_scores()
            
            # Step 4: Archive completed conversations
            if conversation_id:
                # This would integrate with conversation history
                # For now, just note it in results
                results["archived_conversations"] = 1
            
            # Step 5: Optimize vector indexes if needed
            if self._should_optimize_indexes():
                self.memory_manager.vector_store.optimize_index(table="memory_passages")
                results["optimized_indexes"] = True
            
            # Step 6: Create consolidation snapshot
            self.memory_manager.create_snapshot(
                conversation_id or "system",
                reason="consolidation"
            )
            
            # Step 7: Log completion (no cache cleanup needed as embeddings are not cached)
            
            self.logger.info(f"Consolidation complete: {results}")
            
        return results
    
    def _prune_old_passages(self) -> int:
        """
        Remove old, low-importance passages.
        
        Returns:
            Number of passages pruned
        """
        cutoff_date = utc_now() - timedelta(
            days=self.memory_manager.config.memory.max_memory_age_days
        )
        
        pruned = 0
        
        with self.memory_manager.get_session() as session:
            # Find candidates for pruning
            candidates = session.query(MemoryPassage).filter(
                and_(
                    MemoryPassage.created_at < cutoff_date,
                    MemoryPassage.importance_score < 0.3,
                    MemoryPassage.access_count < 2
                )
            ).all()
            
            for passage in candidates:
                # Safe to delete - no relationships to check
                session.delete(passage)
                pruned += 1
            
            session.commit()
        
        if pruned > 0:
            self.logger.info(f"Pruned {pruned} old passages")
        
        return pruned
    
    
    
    
    def _update_importance_scores(self) -> int:
        """
        Update importance scores based on usage patterns.
        
        Returns:
            Number of items with updated scores
        """
        updated = 0
        
        with self.memory_manager.get_session() as session:
            # Update passage importance scores
            passages = session.query(MemoryPassage).all()
            
            for passage in passages:
                old_score = passage.importance_score
                
                # Calculate age factor (decay over time)
                age_days = (utc_now() - passage.created_at).days
                age_factor = max(0.5, 1.0 - (age_days * 0.002))  # 0.2% decay per day
                
                # Calculate access factor (boost for frequent access)
                access_factor = min(1.5, 1.0 + (passage.access_count * 0.05))
                
                # Calculate recency factor
                if passage.last_accessed:
                    days_since_access = (utc_now() - passage.last_accessed).days
                    recency_factor = max(0.7, 1.0 - (days_since_access * 0.003))
                else:
                    recency_factor = age_factor
                
                # New score is combination of factors
                new_score = old_score * age_factor * access_factor * recency_factor
                new_score = max(0.1, min(1.0, new_score))  # Clamp between 0.1 and 1.0
                
                if abs(new_score - old_score) > 0.01:  # Significant change
                    passage.importance_score = new_score
                    updated += 1
            
            
            session.commit()
        
        if updated > 0:
            self.logger.info(f"Updated importance scores for {updated} items")
        
        return updated
    
    def _should_optimize_indexes(self) -> bool:
        """
        Determine if vector indexes need optimization.
        
        Returns:
            True if optimization is recommended
        """
        with self.memory_manager.get_session() as session:
            # Get passage count
            passage_count = session.query(MemoryPassage).count()
            
            # Get last optimization time from metadata
            result = session.execute(
                text("""
                SELECT obj_description(c.oid, 'pg_class') as comment
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'idx_memory_passages_embedding'
                AND n.nspname = 'public'
                """)
            ).fetchone()
            
            # Simple heuristic: optimize if >10k passages
            # In production, would check index statistics
            return passage_count > 10000
    
    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """
        Analyze memory patterns for insights.
        
        Returns:
            Analysis results
        """
        with self.memory_manager.get_session() as session:
            # Time-based analysis
            now = utc_now()
            time_windows = {
                "last_hour": now - timedelta(hours=1),
                "last_day": now - timedelta(days=1),
                "last_week": now - timedelta(weeks=1),
                "last_month": now - timedelta(days=30)
            }
            
            analysis = {
                "passage_creation_rate": {},
                "memory_health": {}
            }
            
            # Passage creation rate
            for window_name, cutoff in time_windows.items():
                count = session.query(MemoryPassage).filter(
                    MemoryPassage.created_at > cutoff
                ).count()
                analysis["passage_creation_rate"][window_name] = count
            
            # Memory health metrics
            total_passages = session.query(MemoryPassage).count()
            low_importance_passages = session.query(MemoryPassage).filter(
                MemoryPassage.importance_score < 0.3
            ).count()
            
            analysis["memory_health"] = {
                "total_passages": total_passages,
                "low_importance_ratio": low_importance_passages / total_passages if total_passages > 0 else 0,
                "avg_passage_importance": session.query(
                    func.avg(MemoryPassage.importance_score)
                ).scalar() or 0,
            }
            
            return analysis