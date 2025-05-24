"""
Memory consolidation and optimization engine.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC, timedelta

from sqlalchemy import text, and_, or_

from lt_memory.models.base import MemoryPassage, MemoryEntity, MemoryRelation
from errors import error_context, ErrorCode, ToolError

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
            "merged_entities": 0,
            "updated_scores": 0,
            "archived_conversations": 0,
            "optimized_indexes": False
        }
        
        with error_context("consolidation_engine", "consolidate_memories", ToolError, ErrorCode.MEMORY_ERROR):
            # Step 1: Prune old, low-importance passages
            results["pruned_passages"] = self._prune_old_passages()
            
            # Step 2: Merge duplicate entities
            results["merged_entities"] = self._merge_duplicate_entities()
            
            # Step 3: Update importance scores based on usage
            results["updated_scores"] = self._update_importance_scores()
            
            # Step 4: Archive completed conversations
            if conversation_id:
                # This would integrate with conversation history
                # For now, just note it in results
                results["archived_conversations"] = 1
            
            # Step 5: Optimize vector indexes if needed
            if self._should_optimize_indexes():
                self.memory_manager.vector_store.optimize_index()
                results["optimized_indexes"] = True
            
            # Step 6: Create consolidation snapshot
            self.memory_manager.create_snapshot(
                conversation_id or "system",
                reason="consolidation"
            )
            
            # Step 7: Clean up embedding cache
            cache_stats = self.memory_manager.embedding_cache.get_stats()
            if cache_stats["disk_cache_size"] > 10000:
                self.memory_manager.embedding_cache.clear_disk_cache()
                self.logger.info("Cleared embedding cache during consolidation")
            
            self.logger.info(f"Consolidation complete: {results}")
            
        return results
    
    def _prune_old_passages(self) -> int:
        """
        Remove old, low-importance passages.
        
        Returns:
            Number of passages pruned
        """
        cutoff_date = datetime.now(UTC) - timedelta(
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
                # Check if passage is evidence for any relationships
                relations_using = session.query(MemoryRelation).filter(
                    MemoryRelation.evidence_ids.contains([str(passage.id)])
                ).count()
                
                if relations_using == 0:
                    # Safe to delete
                    session.delete(passage)
                    pruned += 1
                else:
                    # Just reduce importance further
                    passage.importance_score *= 0.5
            
            session.commit()
        
        if pruned > 0:
            self.logger.info(f"Pruned {pruned} old passages")
        
        return pruned
    
    def _merge_duplicate_entities(self) -> int:
        """
        Merge entities that likely refer to the same thing.
        
        Returns:
            Number of entities merged
        """
        merged = 0
        
        with self.memory_manager.get_session() as session:
            # Get all entities grouped by type
            entity_types = session.query(MemoryEntity.entity_type).distinct().all()
            
            for (entity_type,) in entity_types:
                if not entity_type:
                    continue
                
                # Get entities of this type
                entities = session.query(MemoryEntity).filter_by(
                    entity_type=entity_type
                ).order_by(
                    MemoryEntity.importance_score.desc()
                ).all()
                
                # Check each pair for similarity
                for i, entity1 in enumerate(entities):
                    if not entity1:  # Might have been deleted
                        continue
                    
                    for entity2 in entities[i+1:]:
                        if not entity2:  # Might have been deleted
                            continue
                        
                        if self._entities_similar(entity1.name, entity2.name):
                            # Merge entity2 into entity1
                            self._merge_entities(session, entity1, entity2)
                            merged += 1
                            
                            # Mark entity2 as None so we skip it
                            entities[entities.index(entity2)] = None
            
            session.commit()
        
        if merged > 0:
            self.logger.info(f"Merged {merged} duplicate entities")
        
        return merged
    
    def _entities_similar(self, name1: str, name2: str) -> bool:
        """
        Check if two entity names are similar enough to merge.
        
        Args:
            name1: First entity name
            name2: Second entity name
            
        Returns:
            True if entities should be merged
        """
        # Normalize names
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()
        
        # Exact match
        if n1 == n2:
            return True
        
        # One is substring of other (e.g., "John" and "John Smith")
        if n1 in n2 or n2 in n1:
            return True
        
        # Edit distance check (simple implementation)
        if abs(len(n1) - len(n2)) <= 2:
            # Count character differences
            differences = sum(1 for a, b in zip(n1, n2) if a != b)
            if differences <= 2:
                return True
        
        # Check common abbreviations
        abbreviations = [
            ("company", "co"),
            ("corporation", "corp"),
            ("incorporated", "inc"),
            ("limited", "ltd"),
            ("university", "univ"),
            ("mister", "mr"),
            ("doctor", "dr")
        ]
        
        for full, abbr in abbreviations:
            if full in n1 and abbr in n2:
                n1_normalized = n1.replace(full, abbr)
                if n1_normalized == n2:
                    return True
            elif abbr in n1 and full in n2:
                n2_normalized = n2.replace(full, abbr)
                if n1 == n2_normalized:
                    return True
        
        return False
    
    def _merge_entities(self, session, keep: MemoryEntity, merge: MemoryEntity):
        """
        Merge one entity into another.
        
        Args:
            session: Database session
            keep: Entity to keep
            merge: Entity to merge and delete
        """
        # Update all relationships
        relations = session.query(MemoryRelation).filter(
            or_(
                MemoryRelation.subject_id == merge.id,
                MemoryRelation.object_id == merge.id
            )
        ).all()
        
        for relation in relations:
            # Check if this would create a duplicate relationship
            if relation.subject_id == merge.id:
                # Check if same relationship exists with keep entity
                existing = session.query(MemoryRelation).filter(
                    and_(
                        MemoryRelation.subject_id == keep.id,
                        MemoryRelation.predicate == relation.predicate,
                        MemoryRelation.object_id == relation.object_id
                    )
                ).first()
                
                if existing:
                    # Merge evidence and confidence
                    existing.confidence = max(existing.confidence, relation.confidence)
                    evidence = list(set(existing.evidence_ids + relation.evidence_ids))
                    existing.evidence_ids = evidence
                    session.delete(relation)
                else:
                    relation.subject_id = keep.id
            
            if relation.object_id == merge.id:
                # Similar check for object relationships
                existing = session.query(MemoryRelation).filter(
                    and_(
                        MemoryRelation.subject_id == relation.subject_id,
                        MemoryRelation.predicate == relation.predicate,
                        MemoryRelation.object_id == keep.id
                    )
                ).first()
                
                if existing:
                    existing.confidence = max(existing.confidence, relation.confidence)
                    evidence = list(set(existing.evidence_ids + relation.evidence_ids))
                    existing.evidence_ids = evidence
                    session.delete(relation)
                else:
                    relation.object_id = keep.id
        
        # Merge attributes
        if merge.attributes:
            keep_attrs = keep.attributes or {}
            keep_attrs.update(merge.attributes)
            keep.attributes = keep_attrs
        
        # Update statistics
        keep.mention_count += merge.mention_count
        keep.importance_score = max(keep.importance_score, merge.importance_score)
        keep.first_seen = min(keep.first_seen, merge.first_seen)
        keep.last_seen = max(keep.last_seen, merge.last_seen)
        
        # Delete merged entity
        session.delete(merge)
        
        self.logger.debug(f"Merged entity '{merge.name}' into '{keep.name}'")
    
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
                age_days = (datetime.now(UTC) - passage.created_at).days
                age_factor = max(0.5, 1.0 - (age_days * 0.002))  # 0.2% decay per day
                
                # Calculate access factor (boost for frequent access)
                access_factor = min(1.5, 1.0 + (passage.access_count * 0.05))
                
                # Calculate recency factor
                if passage.last_accessed:
                    days_since_access = (datetime.now(UTC) - passage.last_accessed).days
                    recency_factor = max(0.7, 1.0 - (days_since_access * 0.003))
                else:
                    recency_factor = age_factor
                
                # New score is combination of factors
                new_score = old_score * age_factor * access_factor * recency_factor
                new_score = max(0.1, min(1.0, new_score))  # Clamp between 0.1 and 1.0
                
                if abs(new_score - old_score) > 0.01:  # Significant change
                    passage.importance_score = new_score
                    updated += 1
            
            # Update entity importance scores
            entities = session.query(MemoryEntity).all()
            
            for entity in entities:
                old_score = entity.importance_score
                
                # Base score on mention frequency
                mention_score = min(1.0, 0.3 + (entity.mention_count * 0.02))
                
                # Recency factor
                days_since_seen = (datetime.now(UTC) - entity.last_seen).days
                recency_factor = max(0.5, 1.0 - (days_since_seen * 0.005))
                
                # Relationship factor (more relationships = more important)
                relation_count = session.query(MemoryRelation).filter(
                    or_(
                        MemoryRelation.subject_id == entity.id,
                        MemoryRelation.object_id == entity.id
                    )
                ).count()
                
                relation_factor = min(1.5, 1.0 + (relation_count * 0.1))
                
                # Calculate new score
                new_score = mention_score * recency_factor * relation_factor
                new_score = max(0.1, min(1.0, new_score))
                
                if abs(new_score - old_score) > 0.01:
                    entity.importance_score = new_score
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
            now = datetime.now(UTC)
            time_windows = {
                "last_hour": now - timedelta(hours=1),
                "last_day": now - timedelta(days=1),
                "last_week": now - timedelta(weeks=1),
                "last_month": now - timedelta(days=30)
            }
            
            analysis = {
                "passage_creation_rate": {},
                "entity_discovery_rate": {},
                "top_entities": [],
                "relationship_growth": {},
                "memory_health": {}
            }
            
            # Passage creation rate
            for window_name, cutoff in time_windows.items():
                count = session.query(MemoryPassage).filter(
                    MemoryPassage.created_at > cutoff
                ).count()
                analysis["passage_creation_rate"][window_name] = count
            
            # Entity discovery rate
            for window_name, cutoff in time_windows.items():
                count = session.query(MemoryEntity).filter(
                    MemoryEntity.first_seen > cutoff
                ).count()
                analysis["entity_discovery_rate"][window_name] = count
            
            # Top entities by importance and mentions
            top_entities = session.query(MemoryEntity).order_by(
                MemoryEntity.importance_score.desc(),
                MemoryEntity.mention_count.desc()
            ).limit(10).all()
            
            analysis["top_entities"] = [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "importance": e.importance_score,
                    "mentions": e.mention_count
                }
                for e in top_entities
            ]
            
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
                "entities_without_relationships": session.query(MemoryEntity).filter(
                    ~exists().where(
                        or_(
                            MemoryRelation.subject_id == MemoryEntity.id,
                            MemoryRelation.object_id == MemoryEntity.id
                        )
                    )
                ).count()
            }
            
            return analysis