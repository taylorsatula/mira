"""
Memory Relevance Service - CNS Integration Point for LT_Memory

Provides the primary interface for the CNS orchestrator to interact with
the long-term memory system. Wraps ProactiveService from lt_memory.

CNS Integration Points:
- get_relevant_memories(query_expansion, expansion_embedding, extracted_entities) -> List[Dict]
- Uses pre-computed embeddings (no redundant embedding generation)
- Supports hub-based discovery via extracted entities
- Returns hierarchical memory structures with link metadata
"""
import logging
from typing import Optional
import numpy as np

from lt_memory.models import MemoryDict
from lt_memory.proactive import ProactiveService

logger = logging.getLogger(__name__)


class MemoryRelevanceService:
    """
    CNS service for memory relevance scoring.

    Wraps the lt_memory ProactiveService to provide memory surfacing for continuums.
    Uses pre-computed expansion embeddings from CNS.
    """

    def __init__(self, proactive_service: ProactiveService):
        """
        Initialize memory relevance service.

        Args:
            proactive_service: lt_memory ProactiveService instance (from factory)
        """
        self.proactive = proactive_service
        logger.info("MemoryRelevanceService initialized with ProactiveService")

    def get_relevant_memories(
        self,
        query_expansion: str,
        expansion_embedding: np.ndarray,
        limit: int = 10,
        extracted_entities: Optional[list[str]] = None
    ) -> list[MemoryDict]:
        """
        Get memories relevant to the query expansion using hybrid retrieval.

        Combines two retrieval paths:
        1. Similarity Pool: Hybrid search (BM25 + vector similarity)
        2. Hub-Derived Pool: Entity-driven discovery via hub navigation

        Args:
            query_expansion: Expanded retrieval-optimized query
            expansion_embedding: Pre-computed embedding of query expansion
            limit: Maximum memories to return (default: 10)
            extracted_entities: Entity names for hub-based discovery (optional)

        Returns:
            List of memory dicts with hierarchical structure:
            [
                {
                    "id": "uuid",
                    "text": "memory text",
                    "importance_score": 0.85,
                    "similarity_score": 0.82,
                    "created_at": "iso-timestamp",
                    "entity_links": [...],
                    "linked_memories": [...]
                }
            ]

        Raises:
            ValueError: If expansion embedding validation fails
            RuntimeError: If memory service infrastructure fails
        """
        # Validate embedding
        if expansion_embedding is None:
            raise ValueError("expansion_embedding is required")

        # Delegate to ProactiveService with extracted entities for hub discovery
        memories = self.proactive.search_with_embedding(
            embedding=expansion_embedding,
            query_expansion=query_expansion,
            limit=limit,
            extracted_entities=extracted_entities
        )

        if memories:
            entity_info = f" (entities: {len(extracted_entities)})" if extracted_entities else ""
            logger.info(f"Surfaced {len(memories)} relevant memories{entity_info}")
        else:
            logger.debug("No relevant memories found")

        return memories
