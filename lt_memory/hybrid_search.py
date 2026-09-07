"""
Hybrid search implementation combining BM25 text search with vector similarity.

This module provides hybrid retrieval that leverages both lexical matching
(for exact phrases) and semantic similarity (for related concepts).

Note: Entity-based retrieval is now handled by HubDiscoveryService in proactive.py,
which provides a proper retrieval path (entity → memories → hubs) rather than
just boosting existing similarity results.
"""
import logging
import math
from typing import List, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from lt_memory.db_access import LTMemoryDB
    from lt_memory.models import Memory

logger = logging.getLogger(__name__)

# Search defaults
HYBRID_DEFAULT_LIMIT = 20
HYBRID_DEFAULT_SIMILARITY_THRESHOLD = 0.5
HYBRID_DEFAULT_MIN_IMPORTANCE = 0.1
HYBRID_OVERSAMPLE_MULTIPLIER = 2

# Intent-based weights: (bm25_weight, vector_weight)
INTENT_RECALL_BM25 = 0.6
INTENT_RECALL_VECTOR = 0.4
INTENT_EXPLORE_BM25 = 0.3
INTENT_EXPLORE_VECTOR = 0.7
INTENT_EXACT_BM25 = 0.8
INTENT_EXACT_VECTOR = 0.2
INTENT_GENERAL_BM25 = 0.4
INTENT_GENERAL_VECTOR = 0.6

# Reciprocal Rank Fusion
RRF_K = 60


class HybridSearcher:
    """
    Combines BM25 text search with vector similarity for optimal retrieval.

    Uses Reciprocal Rank Fusion (RRF) to combine results from both methods,
    with intent-aware weighting to optimize for different query types.

    Note: Entity-based retrieval is handled separately by HubDiscoveryService,
    which provides a proper retrieval path rather than score boosting.
    """

    def __init__(self, db_access: 'LTMemoryDB'):
        self.db = db_access

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        search_intent: str = "general",
        limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        min_importance: Optional[float] = None
    ) -> List['Memory']:
        """
        Perform hybrid search combining BM25 and vector similarity.

        Args:
            query_text: Text query for BM25 search
            query_embedding: Embedding for vector search
            search_intent: Intent type (recall/explore/exact/general)
            limit: Maximum results to return (default from config)
            similarity_threshold: Minimum similarity for vector search (default from config)
            min_importance: Minimum importance score (default from config)

        Returns:
            List of Memory objects ranked by hybrid score
        """
        limit = limit if limit is not None else HYBRID_DEFAULT_LIMIT
        similarity_threshold = similarity_threshold if similarity_threshold is not None else HYBRID_DEFAULT_SIMILARITY_THRESHOLD
        min_importance = min_importance if min_importance is not None else HYBRID_DEFAULT_MIN_IMPORTANCE
        oversample = HYBRID_OVERSAMPLE_MULTIPLIER

        # Run searches in parallel (would be async in production)
        bm25_results = self._bm25_search(
            query_text,
            limit=limit * oversample,
            min_importance=min_importance
        )

        vector_results = self._vector_search(
            query_embedding,
            limit=limit * oversample,
            similarity_threshold=similarity_threshold,
            min_importance=min_importance
        )

        weights = {
            "recall": (INTENT_RECALL_BM25, INTENT_RECALL_VECTOR),
            "explore": (INTENT_EXPLORE_BM25, INTENT_EXPLORE_VECTOR),
            "exact": (INTENT_EXACT_BM25, INTENT_EXACT_VECTOR),
            "general": (INTENT_GENERAL_BM25, INTENT_GENERAL_VECTOR),
        }

        bm25_weight, vector_weight = weights.get(search_intent, weights["general"])

        # Combine using Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            bm25_weight,
            vector_weight,
            limit
        )

        logger.info(
            f"Hybrid search: {len(bm25_results)} BM25 + {len(vector_results)} vector "
            f"-> {len(fused_results)} fused results (intent: {search_intent})"
        )

        return fused_results

    def _bm25_search(
        self,
        query_text: str,
        limit: int,
        min_importance: float
    ) -> List[Tuple['Memory', float]]:
        """
        Perform BM25 text search using PostgreSQL full-text search.

        Searches both personal memories (RLS-filtered) and global memories (no RLS)
        via UNION. Results are tagged with source='personal' or source='global'.

        Returns list of (Memory, score) tuples.
        """
        resolved_user_id = self.db._resolve_user_id()

        with self.db.session_manager.get_session(resolved_user_id) as session:
            query = """
            (
                SELECT m.id, m.user_id, m.text, m.embedding, m.importance_score,
                       m.created_at, m.updated_at, m.expires_at, m.access_count,
                       m.mention_count, m.last_accessed, m.happens_at,
                       m.inbound_links, m.outbound_links, m.entity_links,
                       m.is_archived, m.archived_at,
                       m.activity_days_at_creation, m.activity_days_at_last_access,
                       m.annotations, m.source_segment_id,
                       ts_rank(m.search_vector, plainto_tsquery('english', %(query)s)) as rank,
                       'personal' as source
                FROM memories m
                WHERE m.search_vector @@ plainto_tsquery('english', %(query)s)
                  AND m.importance_score >= %(min_importance)s
                  AND (m.expires_at IS NULL OR m.expires_at > NOW())
                  AND m.is_archived = FALSE
            )
            UNION ALL
            (
                SELECT gm.id, NULL::uuid as user_id, gm.text, gm.embedding, gm.importance_score,
                       gm.created_at, gm.updated_at, NULL::timestamptz as expires_at, 0 as access_count,
                       0 as mention_count, NULL::timestamptz as last_accessed, gm.happens_at,
                       gm.inbound_links, gm.outbound_links, gm.entity_links,
                       gm.is_archived, gm.archived_at,
                       NULL::int as activity_days_at_creation, NULL::int as activity_days_at_last_access,
                       '[]'::jsonb as annotations,
                       NULL::uuid as source_segment_id,
                       ts_rank(gm.search_vector, plainto_tsquery('english', %(query)s)) as rank,
                       'global' as source
                FROM global_memories_runtime gm
                WHERE gm.search_vector @@ plainto_tsquery('english', %(query)s)
                  AND gm.is_archived = FALSE
            )
            ORDER BY rank DESC
            LIMIT %(limit)s
            """

            results = session.execute_query(query, {
                'query': query_text,
                'limit': limit,
                'min_importance': min_importance
            })

            # Convert to Memory objects with scores, preserving source
            from lt_memory.models import Memory  # Runtime import to avoid circular
            tuples = []
            for row in results:
                source = row.pop('source', 'personal')
                memory = Memory(**row)
                memory.source = source
                tuples.append((memory, row['rank']))
            return tuples

    def _vector_search(
        self,
        query_embedding: List[float],
        limit: int,
        similarity_threshold: float,
        min_importance: float
    ) -> List[Tuple['Memory', float]]:
        """
        Perform vector similarity search.

        Returns list of (Memory, score) tuples.
        """
        # Reuse existing vector search
        memories = self.db.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            min_importance=min_importance
        )

        # Use similarity scores calculated by database
        results = []
        for memory in memories:
            if memory.similarity_score is None:
                raise RuntimeError(
                    f"Memory {memory.id} missing similarity_score - "
                    f"this indicates db.search_similar() did not populate the transient field"
                )
            results.append((memory, memory.similarity_score))

        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple['Memory', float]],
        vector_results: List[Tuple['Memory', float]],
        bm25_weight: float,
        vector_weight: float,
        limit: int
    ) -> List['Memory']:
        """
        Combine results using Reciprocal Rank Fusion (RRF) with sigmoid normalization.

        RRF formula: score(d) = Σ(1 / (k + rank(d)))
        where k is a constant (typically 60) that determines how quickly scores decay.

        Raw RRF scores are compressed into ~0.007-0.016 range which provides poor
        discrimination. We apply sigmoid transformation to spread scores into
        a useful 0-1 range for meaningful thresholding and interpretability.
        """
        k = RRF_K

        # Calculate raw RRF scores
        rrf_scores = defaultdict(float)
        memory_map = {}

        # Process BM25 results
        for rank, (memory, _) in enumerate(bm25_results, 1):
            rrf_scores[memory.id] += bm25_weight * (1.0 / (k + rank))
            memory_map[memory.id] = memory

        # Process vector results - preserve original cosine similarity
        for rank, (memory, cosine_sim) in enumerate(vector_results, 1):
            rrf_scores[memory.id] += vector_weight * (1.0 / (k + rank))
            memory._vector_similarity = cosine_sim  # Preserve for logging
            memory_map[memory.id] = memory

        # Sort by combined RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Apply sigmoid transformation to spread scores into useful 0-1 range
        # Raw RRF scores cluster around 0.007-0.016; sigmoid with k=1000 and
        # midpoint=0.009 spreads this to ~0.1-0.85 for meaningful discrimination
        def sigmoid_normalize(raw_score: float) -> float:
            # Sigmoid: 1 / (1 + exp(-k * (x - midpoint)))
            # k=1000 provides good spread, midpoint=0.009 centers on typical RRF range
            return 1.0 / (1.0 + math.exp(-1000 * (raw_score - 0.009)))

        # Return top memories with normalized scores
        results = []
        for memory_id, raw_rrf_score in sorted_ids[:limit]:
            memory = memory_map[memory_id]
            # Store sigmoid-normalized score for interpretable thresholding
            memory.similarity_score = sigmoid_normalize(raw_rrf_score)
            memory._raw_rrf_score = raw_rrf_score  # Preserve raw for debugging
            results.append(memory)
        return results

