"""
Manager for knowledge graph entities and relationships.
"""

import uuid
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, UTC
from collections import Counter

from sqlalchemy import and_, or_

from lt_memory.models.base import MemoryEntity, MemoryRelation, MemoryPassage
from errors import error_context, ErrorCode

logger = logging.getLogger(__name__)


class EntityManager:
    """
    Manages knowledge graph entities and relationships.
    
    Handles entity extraction, relationship inference, and
    knowledge graph operations.
    """
    
    def __init__(self, memory_manager):
        """
        Initialize entity manager.
        
        Args:
            memory_manager: Parent MemoryManager instance
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
        
        # Entity patterns for extraction
        self.entity_patterns = {
            "person": [
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Simple two-word names
            ],
            "organization": [
                r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Inc|LLC|Corp|Corporation|Company|Ltd)\b',
                r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:University|College|School)\b'
            ],
            "location": [
                r'\b(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            ],
            "date": [
                r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b'
            ]
        }
    
    def extract_entities(self, text: str, passage_id: str) -> List[str]:
        """
        Extract entities from text and update database.
        
        Args:
            text: Text to extract entities from
            passage_id: Associated passage ID
            
        Returns:
            List of entity IDs
        """
        if not self.memory_manager.config.memory.entity_extraction_enabled:
            return []
        
        entities = []
        entity_ids = []
        
        # Extract entities by type
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Clean up the match
                    entity_name = match.strip() if isinstance(match, str) else match[0].strip()
                    if len(entity_name) > 2:  # Skip very short matches
                        entities.append((entity_name, entity_type))
        
        # Also extract capitalized sequences as potential entities
        cap_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        cap_matches = re.findall(cap_pattern, text)
        for match in cap_matches:
            if len(match) > 3 and not any(match == e[0] for e in entities):
                entities.append((match, "unknown"))
        
        # Update database
        with self.memory_manager.get_session() as session:
            for entity_name, entity_type in entities:
                # Check if entity exists
                entity = session.query(MemoryEntity).filter_by(
                    name=entity_name
                ).first()
                
                if entity:
                    # Update existing entity
                    entity.mention_count += 1
                    entity.last_seen = datetime.now(UTC)
                    
                    # Update importance based on frequency
                    entity.importance_score = min(
                        1.0, 
                        0.3 + (entity.mention_count * 0.05)
                    )
                else:
                    # Create new entity
                    entity = MemoryEntity(
                        name=entity_name,
                        entity_type=entity_type,
                        importance_score=0.5
                    )
                    session.add(entity)
                
                entity_ids.append(str(entity.id))
            
            session.commit()
        
        if entity_ids:
            self.logger.info(
                f"Extracted {len(entity_ids)} entities from passage {passage_id}"
            )
        
        return entity_ids
    
    def infer_relationships(self, passage_id: str, 
                           entity_ids: List[str]) -> List[str]:
        """
        Infer relationships between entities in a passage.
        
        Args:
            passage_id: Passage containing the entities
            entity_ids: List of entity IDs found in passage
            
        Returns:
            List of created relationship IDs
        """
        if not self.memory_manager.config.memory.relationship_inference_enabled:
            return []
        
        if len(entity_ids) < 2:
            return []
        
        relationships = []
        
        with self.memory_manager.get_session() as session:
            # Get the passage text for context
            passage = session.query(MemoryPassage).filter_by(
                id=passage_id
            ).first()
            
            if not passage:
                return []
            
            passage_text = passage.text.lower()
            
            # Get entity details
            entities = {}
            for entity_id in entity_ids:
                entity = session.query(MemoryEntity).filter_by(
                    id=entity_id
                ).first()
                if entity:
                    entities[entity_id] = entity
            
            # Infer relationships between each pair
            for i, entity1_id in enumerate(entity_ids):
                for entity2_id in entity_ids[i+1:]:
                    if entity1_id not in entities or entity2_id not in entities:
                        continue
                    
                    entity1 = entities[entity1_id]
                    entity2 = entities[entity2_id]
                    
                    # Infer relationship type based on context
                    predicate, confidence = self._infer_predicate(
                        entity1, entity2, passage_text
                    )
                    
                    if predicate:
                        # Check if relationship already exists
                        existing = session.query(MemoryRelation).filter(
                            or_(
                                and_(
                                    MemoryRelation.subject_id == entity1_id,
                                    MemoryRelation.object_id == entity2_id,
                                    MemoryRelation.predicate == predicate
                                ),
                                and_(
                                    MemoryRelation.subject_id == entity2_id,
                                    MemoryRelation.object_id == entity1_id,
                                    MemoryRelation.predicate == predicate
                                )
                            )
                        ).first()
                        
                        if existing:
                            # Strengthen existing relationship
                            existing.confidence = min(
                                1.0, 
                                existing.confidence + 0.1
                            )
                            if passage_id not in existing.evidence_ids:
                                evidence = existing.evidence_ids
                                evidence.append(passage_id)
                                existing.evidence_ids = evidence
                        else:
                            # Create new relationship
                            relation = MemoryRelation(
                                subject_id=entity1_id,
                                predicate=predicate,
                                object_id=entity2_id,
                                confidence=confidence,
                                evidence_ids=[passage_id]
                            )
                            session.add(relation)
                            relationships.append(str(relation.id))
            
            session.commit()
        
        if relationships:
            self.logger.info(
                f"Inferred {len(relationships)} relationships from passage {passage_id}"
            )
        
        return relationships
    
    def _infer_predicate(self, entity1: MemoryEntity, entity2: MemoryEntity, 
                        context: str) -> Tuple[Optional[str], float]:
        """
        Infer the relationship predicate between two entities.
        
        Args:
            entity1: First entity
            entity2: Second entity
            context: Text context containing both entities
            
        Returns:
            Tuple of (predicate, confidence) or (None, 0)
        """
        e1_name = entity1.name.lower()
        e2_name = entity2.name.lower()
        
        # Check for explicit relationship patterns
        patterns = [
            (r'{} works (?:at|for|with) {}'.format(re.escape(e1_name), re.escape(e2_name)), 
             "works_at", 0.9),
            (r'{} (?:is|was) (?:a|an|the) .{{0,20}} (?:at|of|for) {}'.format(
                re.escape(e1_name), re.escape(e2_name)), "affiliated_with", 0.8),
            (r'{} (?:knows|met|meets|talked to|spoke with) {}'.format(
                re.escape(e1_name), re.escape(e2_name)), "knows", 0.8),
            (r'{} (?:lives|resides|located) (?:in|at) {}'.format(
                re.escape(e1_name), re.escape(e2_name)), "located_in", 0.9),
            (r'{} (?:owns|has|possesses) {}'.format(
                re.escape(e1_name), re.escape(e2_name)), "owns", 0.8),
            (r'{} (?:created|built|developed|made) {}'.format(
                re.escape(e1_name), re.escape(e2_name)), "created", 0.8),
        ]
        
        for pattern, predicate, confidence in patterns:
            if re.search(pattern, context):
                return predicate, confidence
        
        # Check proximity-based relationships
        # Find positions of entities in text
        e1_pos = context.find(e1_name)
        e2_pos = context.find(e2_name)
        
        if e1_pos >= 0 and e2_pos >= 0:
            distance = abs(e1_pos - e2_pos)
            
            # Very close entities are likely related
            if distance < 50:
                # Check entity types for appropriate predicate
                if entity1.entity_type == "person" and entity2.entity_type == "organization":
                    return "associated_with", 0.6
                elif entity1.entity_type == "person" and entity2.entity_type == "location":
                    return "located_near", 0.5
                else:
                    return "related_to", 0.5
        
        return None, 0
    
    def get_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Get entity details by name.
        
        Args:
            entity_name: Entity name to look up
            
        Returns:
            Entity data or None
        """
        with self.memory_manager.get_session() as session:
            entity = session.query(MemoryEntity).filter_by(
                name=entity_name
            ).first()
            
            if entity:
                return {
                    "id": str(entity.id),
                    "name": entity.name,
                    "type": entity.entity_type,
                    "attributes": entity.attributes,
                    "importance": entity.importance_score,
                    "first_seen": entity.first_seen.isoformat(),
                    "last_seen": entity.last_seen.isoformat(),
                    "mention_count": entity.mention_count
                }
        
        return None
    
    def get_entity_graph(self, entity_name: str, 
                        depth: int = 2) -> Optional[Dict[str, Any]]:
        """
        Get entity and its relationships up to specified depth.
        
        Args:
            entity_name: Starting entity name
            depth: How many relationship hops to include
            
        Returns:
            Graph data structure or None
        """
        with self.memory_manager.get_session() as session:
            # Find the entity
            entity = session.query(MemoryEntity).filter_by(
                name=entity_name
            ).first()
            
            if not entity:
                return None
            
            # Build graph using breadth-first traversal
            graph = {
                "center": {
                    "id": str(entity.id),
                    "name": entity.name,
                    "type": entity.entity_type,
                    "importance": entity.importance_score
                },
                "nodes": {},
                "edges": []
            }
            
            visited = {str(entity.id)}
            current_level = [entity]
            
            for level in range(depth):
                next_level = []
                
                for current_entity in current_level:
                    # Get all relationships for this entity
                    relations = session.query(MemoryRelation).filter(
                        or_(
                            MemoryRelation.subject_id == current_entity.id,
                            MemoryRelation.object_id == current_entity.id
                        )
                    ).all()
                    
                    for relation in relations:
                        # Determine the other entity
                        if str(relation.subject_id) == str(current_entity.id):
                            other_id = str(relation.object_id)
                            direction = "outgoing"
                        else:
                            other_id = str(relation.subject_id)
                            direction = "incoming"
                        
                        # Skip if already visited
                        if other_id in visited:
                            # Still add the edge
                            graph["edges"].append({
                                "from": str(current_entity.id),
                                "to": other_id,
                                "predicate": relation.predicate,
                                "confidence": relation.confidence,
                                "direction": direction
                            })
                            continue
                        
                        # Get the other entity
                        other_entity = session.query(MemoryEntity).filter_by(
                            id=other_id
                        ).first()
                        
                        if other_entity:
                            # Add to graph
                            graph["nodes"][other_id] = {
                                "id": other_id,
                                "name": other_entity.name,
                                "type": other_entity.entity_type,
                                "importance": other_entity.importance_score,
                                "level": level + 1
                            }
                            
                            graph["edges"].append({
                                "from": str(current_entity.id),
                                "to": other_id,
                                "predicate": relation.predicate,
                                "confidence": relation.confidence,
                                "direction": direction
                            })
                            
                            visited.add(other_id)
                            next_level.append(other_entity)
                
                current_level = next_level
                if not current_level:
                    break
            
            return graph
    
    def search_entities(self, query: str, entity_type: Optional[str] = None,
                       limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for entities by name or type.
        
        Args:
            query: Search query (partial name match)
            entity_type: Optional entity type filter
            limit: Maximum results
            
        Returns:
            List of matching entities
        """
        with self.memory_manager.get_session() as session:
            query_obj = session.query(MemoryEntity)
            
            # Apply filters
            if query:
                query_obj = query_obj.filter(
                    MemoryEntity.name.ilike(f"%{query}%")
                )
            
            if entity_type:
                query_obj = query_obj.filter_by(entity_type=entity_type)
            
            # Order by importance and recency
            entities = query_obj.order_by(
                MemoryEntity.importance_score.desc(),
                MemoryEntity.last_seen.desc()
            ).limit(limit).all()
            
            return [
                {
                    "id": str(e.id),
                    "name": e.name,
                    "type": e.entity_type,
                    "importance": e.importance_score,
                    "mentions": e.mention_count,
                    "last_seen": e.last_seen.isoformat()
                }
                for e in entities
            ]
    
    def merge_entities(self, entity1_name: str, entity2_name: str) -> bool:
        """
        Merge two entities that refer to the same thing.
        
        Args:
            entity1_name: Name of entity to keep
            entity2_name: Name of entity to merge into entity1
            
        Returns:
            True if merge successful
        """
        with self.memory_manager.get_session() as session:
            entity1 = session.query(MemoryEntity).filter_by(
                name=entity1_name
            ).first()
            entity2 = session.query(MemoryEntity).filter_by(
                name=entity2_name
            ).first()
            
            if not entity1 or not entity2:
                return False
            
            # Update all relationships
            relations = session.query(MemoryRelation).filter(
                or_(
                    MemoryRelation.subject_id == entity2.id,
                    MemoryRelation.object_id == entity2.id
                )
            ).all()
            
            for relation in relations:
                if relation.subject_id == entity2.id:
                    relation.subject_id = entity1.id
                if relation.object_id == entity2.id:
                    relation.object_id = entity1.id
            
            # Merge attributes
            if entity2.attributes:
                entity1_attrs = entity1.attributes or {}
                entity1_attrs.update(entity2.attributes)
                entity1.attributes = entity1_attrs
            
            # Update entity1 stats
            entity1.mention_count += entity2.mention_count
            entity1.importance_score = max(
                entity1.importance_score, 
                entity2.importance_score
            )
            entity1.first_seen = min(entity1.first_seen, entity2.first_seen)
            entity1.last_seen = max(entity1.last_seen, entity2.last_seen)
            
            # Delete entity2
            session.delete(entity2)
            session.commit()
            
            self.logger.info(f"Merged entity '{entity2_name}' into '{entity1_name}'")
            return True