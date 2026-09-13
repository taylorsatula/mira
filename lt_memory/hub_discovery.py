"""
Hub discovery service for entity-driven memory retrieval.

Uses extracted entities as stable anchors to find related memories.
Entities mentioned in conversation lead directly to memories - no intermediate
hub filtering step that would second-guess the LLM's entity extraction.

Flow:
1. Match extracted entity names to database entities via pg_trgm fuzzy matching
2. Get memories linked to matched entities (capped per entity)
3. Score memories by expansion similarity (ranking, not gating)
4. Return top N memories

Key insight: The LLM already decided these entities are relevant when it
extracted them. We trust that decision and let expansion similarity
rank the memories, rather than filtering at the hub level.
"""
import logging
from typing import List, Set
from uuid import UUID

import numpy as np

from lt_memory.db_access import LTMemoryDB
from utils.user_context import get_current_user_id
from lt_memory.models import Memory

logger = logging.getLogger(__name__)


class HubDiscoveryService:
    """
    Discovers memories through entity-based hub navigation.

    Uses extracted entities as anchors to navigate the memory graph. Entities
    mentioned in conversation lead to memories linked to those entities.
    """

    def __init__(
        self,
        db: LTMemoryDB,
    ):
        """
        Initialize hub discovery service.

        Args:
            db: Database access layer
        """
        self.db = db

    def discover_hub_memories(
        self,
        extracted_entities: List[str],
        expansion_embedding: np.ndarray,
        limit_per_entity: int = 15,
        max_matched_entities: int = 5,
        entity_similarity_threshold: float = 0.3,
        top_n: int = 20
    ) -> List[Memory]:
        """
        Discover memories through entity-based navigation.

        Trust the LLM's entity extraction, then rank memories by expansion
        similarity rather than filtering hubs.

        Args:
            extracted_entities: Entity names from subcortical layer
            expansion_embedding: 768d embedding for memory ranking
            limit_per_entity: Max memories to collect per matched entity
            max_matched_entities: Max entities to match
            entity_similarity_threshold: Minimum trigram similarity threshold
            top_n: Final number of memories to return after ranking

        Returns:
            List of Memory objects ranked by expansion similarity
        """

        if not extracted_entities:
            logger.debug("No extracted entities, skipping hub discovery")
            return []

        # Step 1: Match entity names to database entities via pg_trgm fuzzy matching
        matched_entity_ids = self._match_entities(
            extracted_entities,
            similarity_threshold=entity_similarity_threshold
        )

        if not matched_entity_ids:
            logger.debug("No matching entities found in database")
            return []

        # Cap matched entities to prevent explosion
        if len(matched_entity_ids) > max_matched_entities:
            matched_entity_ids = set(list(matched_entity_ids)[:max_matched_entities])
            logger.debug(f"Capped matched entities to {max_matched_entities}")

        logger.debug(f"Matched {len(matched_entity_ids)} entities from {len(extracted_entities)} extracted")

        # Step 2: Get memories linked to matched entities (capped per entity)
        candidate_memories = self._get_entity_linked_memories(
            matched_entity_ids,
            limit_per_entity=limit_per_entity
        )

        if not candidate_memories:
            logger.debug("No memories linked to matched entities")
            return []

        logger.debug(f"Found {len(candidate_memories)} candidate memories from matched entities")

        # Step 3: Score memories by expansion similarity and return top N
        ranked_memories = self._rank_memories(
            candidate_memories,
            expansion_embedding,
            top_n=top_n
        )

        logger.info(
            f"Hub discovery: {len(extracted_entities)} entities → "
            f"{len(matched_entity_ids)} matched → {len(candidate_memories)} candidates → "
            f"{len(ranked_memories)} returned"
        )

        return ranked_memories

    def _match_entities(
        self,
        entity_names: List[str],
        similarity_threshold: float
    ) -> Set[UUID]:
        """
        Match extracted entity names to database entities using pg_trgm fuzzy matching.

        Tries exact match first (fast path), then trigram similarity fallback.

        Args:
            entity_names: List of entity names from subcortical extraction
            similarity_threshold: Minimum trigram similarity (0.0-1.0)

        Returns:
            Set of matched entity UUIDs
        """
        matched_ids: Set[UUID] = set()

        for name in entity_names:
            with self.db.session_manager.get_session(get_current_user_id()) as session:
                # Exact match first (fast path)
                exact = session.execute_single(
                    "SELECT id FROM entities WHERE name = %(name)s AND is_archived = FALSE LIMIT 1",
                    {'name': name}
                )
                if exact:
                    matched_ids.add(exact['id'])
                    logger.debug(f"Exact matched entity '{name}'")
                    continue

                # Trigram similarity fallback
                fuzzy = session.execute_single("""
                    SELECT id, name, similarity(name, %(name)s) AS sim_score
                    FROM entities
                    WHERE similarity(name, %(name)s) > %(threshold)s
                      AND is_archived = FALSE
                    ORDER BY sim_score DESC
                    LIMIT 1
                """, {
                    'name': name,
                    'threshold': similarity_threshold
                })

                if fuzzy:
                    matched_ids.add(fuzzy['id'])
                    logger.debug(
                        f"Fuzzy matched '{name}' → '{fuzzy['name']}' "
                        f"(similarity: {fuzzy['sim_score']:.3f})"
                    )

        return matched_ids

    def _get_entity_linked_memories(
        self,
        entity_ids: Set[UUID],
        limit_per_entity: int
    ) -> List[Memory]:
        """
        Get memories linked to matched entities with per-entity cap.

        Prevents explosion when an entity has many linked memories.
        Takes first N per entity.

        Args:
            entity_ids: Set of entity UUIDs
            limit_per_entity: Max memories to collect per entity

        Returns:
            Deduplicated list of Memory objects
        """
        memories = []
        seen_ids: Set[UUID] = set()

        for entity_id in entity_ids:
            entity_memories = self.db.get_memories_for_entity(entity_id)
            # Cap per entity to prevent explosion
            capped_memories = entity_memories[:limit_per_entity]

            for memory in capped_memories:
                if memory.id not in seen_ids:
                    memories.append(memory)
                    seen_ids.add(memory.id)

        return memories

    def _rank_memories(
        self,
        memories: List[Memory],
        expansion_embedding: np.ndarray,
        top_n: int
    ) -> List[Memory]:
        """
        Score and rank memories by expansion similarity, return top N.

        Expansion similarity is used for RANKING (which memories to return)
        rather than GATING. Low scores compete rather than get rejected.

        Args:
            memories: Candidate memories to rank
            expansion_embedding: 768d query embedding
            top_n: Number of memories to return

        Returns:
            Top N memories sorted by expansion similarity (descending)
        """
        scored_memories = []

        for memory in memories:
            if memory.embedding is None:
                logger.debug(f"Memory {memory.id} has no embedding, skipping")
                continue

            # Cosine similarity between memory and expansion
            mem_embedding = np.array(memory.embedding)
            dot_product = np.dot(expansion_embedding, mem_embedding)
            norm_product = np.linalg.norm(expansion_embedding) * np.linalg.norm(mem_embedding)
            similarity = float(dot_product / (norm_product + 1e-9))

            # Store similarity for debugging/tracing
            memory._hub_similarity = similarity
            scored_memories.append((memory, similarity))

        # Sort by similarity descending, take top N
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = [m for m, _ in scored_memories[:top_n]]

        if scored_memories:
            logger.debug(
                f"Ranked {len(scored_memories)} memories by expansion, "
                f"top score: {scored_memories[0][1]:.3f}, "
                f"bottom score: {scored_memories[-1][1]:.3f}"
            )

        return top_memories
