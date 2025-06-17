"""
Stateless proactive memory functions for LT_Memory system.

Provides functions to find and format relevant memories based on conversation context
using semantic similarity search.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import config directly for proactive memory settings
try:
    from config import config
    pm_config = config.proactive_memory
except:
    # Fallback if config isn't available
    pm_config = None

logger = logging.getLogger(__name__)


async def get_relevant_memories_async(
    messages: List[Any],
    memory_manager,
    max_memories: int = None,
    similarity_threshold: float = None,
    min_importance: float = None,
    context_window: int = None,
    use_reranker: bool = None,
    initial_candidates: int = None,
    rerank_candidates: int = None
) -> List[Dict[str, Any]]:
    """
    Find memories relevant to recent conversation context asynchronously.
    
    Multi-stage filtering process:
    1. Retrieve initial_candidates (25) memories by embedding similarity
    2. Take top rerank_candidates (15) for reranking
    3. Rerank those candidates with cross-encoder model
    4. Select up to max_memories (5) that meet importance threshold
    
    Args:
        messages: List of conversation messages
        memory_manager: LT_Memory MemoryManager instance
        max_memories: Maximum number of memories to return
        similarity_threshold: Minimum similarity score for relevance
        min_importance: Minimum importance score for inclusion
        context_window: Number of recent messages to consider
        use_reranker: Whether to use reranking if available
        initial_candidates: Number of memories to retrieve initially
        rerank_candidates: Number of top-ranked memories to rerank
        
    Returns:
        List of relevant memory passages
    """
    # Use config values if available, otherwise use hardcoded defaults
    if pm_config:
        max_memories = max_memories if max_memories is not None else pm_config.max_memories
        similarity_threshold = similarity_threshold if similarity_threshold is not None else pm_config.similarity_threshold
        min_importance = min_importance if min_importance is not None else pm_config.min_importance
        context_window = context_window if context_window is not None else pm_config.context_window
        use_reranker = use_reranker if use_reranker is not None else pm_config.use_reranker
        initial_candidates = initial_candidates if initial_candidates is not None else pm_config.initial_candidates
        rerank_candidates = rerank_candidates if rerank_candidates is not None else pm_config.rerank_candidates
    else:
        # Hardcoded defaults if config not available
        max_memories = max_memories if max_memories is not None else 5
        similarity_threshold = similarity_threshold if similarity_threshold is not None else 0.6
        min_importance = min_importance if min_importance is not None else 0.3
        context_window = context_window if context_window is not None else 6
        use_reranker = use_reranker if use_reranker is not None else True
        initial_candidates = initial_candidates if initial_candidates is not None else 25
        rerank_candidates = rerank_candidates if rerank_candidates is not None else 15
    
    if not messages:
        return []
    
    try:
        # Build weighted context from recent messages
        context_string = build_weighted_memory_context(messages, context_window)
        
        if not context_string:
            return []
        
        # Get embedding asynchronously
        context_embedding = await memory_manager.generate_embedding_async(context_string)
        
        if context_embedding is None:
            logger.warning("Failed to generate context embedding for memory search")
            return []
        
        # Search for similar memories
        results = memory_manager.passage_manager.search_passages_by_embedding(
            query_embedding=context_embedding,
            limit=max_memories * 4,  # Get extra for importance filtering and reranking
            filters={"min_similarity": similarity_threshold}
        )
        
        # First filter by importance score (fast operation)
        importance_filtered = []
        for result in results:
            if result.get("importance", 0) >= min_importance:
                importance_filtered.append(result)
        
        logger.debug(f"Filtered from {len(results)} to {len(importance_filtered)} memories by importance score")
        
        # Apply reranking if enabled and available (on the smaller filtered set)
        if use_reranker and importance_filtered and hasattr(memory_manager.embedding_model, 'rerank'):
            try:
                # Extract passage texts for reranking
                passage_texts = [result.get("text", "") for result in importance_filtered]
                
                # Rerank the filtered passages
                reranked_results = memory_manager.embedding_model.rerank(
                    query=context_string,
                    passages=passage_texts,
                    top_k=max_memories  # Only need top k results
                )
                
                # Build final results from reranked passages
                filtered_results = []
                for idx, rerank_score, _ in reranked_results:
                    passage = importance_filtered[idx].copy()
                    passage["rerank_score"] = rerank_score
                    passage["embedding_score"] = passage.get("similarity", 0)
                    # Use rerank score as primary similarity measure
                    passage["similarity"] = rerank_score
                    filtered_results.append(passage)
                
                logger.debug(f"Reranked {len(importance_filtered)} memories, selected top {len(filtered_results)}")
            except Exception as e:
                logger.warning(f"Reranking failed, using embedding scores: {e}")
                # Fall back to importance-filtered results with original scores
                filtered_results = importance_filtered[:max_memories]
        else:
            # No reranking available or disabled, just limit the importance-filtered results
            filtered_results = importance_filtered[:max_memories]
        
        return filtered_results
        
    except Exception as e:
        logger.error(f"Error finding relevant memories: {e}")
        return []


def get_relevant_memories(
    messages: List[Any],
    memory_manager,
    max_memories: int = None,
    similarity_threshold: float = None,
    min_importance: float = None,
    context_window: int = None,
    use_reranker: bool = None,
    initial_candidates: int = None,
    rerank_candidates: int = None
) -> List[Dict[str, Any]]:
    """
    Find memories relevant to recent conversation context with recency weighting.
    
    Multi-stage filtering process:
    1. Retrieve initial_candidates (25) memories by embedding similarity
    2. Take top rerank_candidates (15) for reranking
    3. Rerank those candidates with cross-encoder model
    4. Select up to max_memories (5) that meet importance threshold
    
    Args:
        messages: List of conversation messages
        memory_manager: LT_Memory MemoryManager instance
        max_memories: Maximum number of memories to return
        similarity_threshold: Minimum similarity score for relevance
        min_importance: Minimum importance score for inclusion
        context_window: Number of recent messages to consider
        use_reranker: Whether to use reranking if available
        initial_candidates: Number of memories to retrieve initially
        rerank_candidates: Number of top-ranked memories to rerank
        
    Returns:
        List of relevant memory passages
    """
    # Use config values if available, otherwise use hardcoded defaults
    if pm_config:
        max_memories = max_memories if max_memories is not None else pm_config.max_memories
        similarity_threshold = similarity_threshold if similarity_threshold is not None else pm_config.similarity_threshold
        min_importance = min_importance if min_importance is not None else pm_config.min_importance
        context_window = context_window if context_window is not None else pm_config.context_window
        use_reranker = use_reranker if use_reranker is not None else pm_config.use_reranker
        initial_candidates = initial_candidates if initial_candidates is not None else pm_config.initial_candidates
        rerank_candidates = rerank_candidates if rerank_candidates is not None else pm_config.rerank_candidates
    else:
        # Hardcoded defaults if config not available
        max_memories = max_memories if max_memories is not None else 5
        similarity_threshold = similarity_threshold if similarity_threshold is not None else 0.6
        min_importance = min_importance if min_importance is not None else 0.3
        context_window = context_window if context_window is not None else 6
        use_reranker = use_reranker if use_reranker is not None else True
        initial_candidates = initial_candidates if initial_candidates is not None else 25
        rerank_candidates = rerank_candidates if rerank_candidates is not None else 15
    
    if not messages:
        return []
    
    try:
        # Build weighted context from recent messages
        context_string = build_weighted_memory_context(messages, context_window)
        
        if not context_string:
            return []
        
        # Get embedding directly from memory manager
        context_embedding = memory_manager.generate_embedding(context_string)
        
        if context_embedding is None:
            logger.warning("Failed to generate context embedding for memory search")
            return []
        
        # Search for similar memories
        results = memory_manager.passage_manager.search_passages_by_embedding(
            query_embedding=context_embedding,
            limit=max_memories * 4,  # Get extra for importance filtering and reranking
            filters={"min_similarity": similarity_threshold}
        )
        
        # First filter by importance score (fast operation)
        importance_filtered = []
        for result in results:
            if result.get("importance", 0) >= min_importance:
                importance_filtered.append(result)
        
        logger.debug(f"Filtered from {len(results)} to {len(importance_filtered)} memories by importance score")
        
        # Apply reranking if enabled and available (on the smaller filtered set)
        if use_reranker and importance_filtered and hasattr(memory_manager.embedding_model, 'rerank'):
            try:
                # Extract passage texts for reranking
                passage_texts = [result.get("text", "") for result in importance_filtered]
                
                # Rerank the filtered passages
                reranked_results = memory_manager.embedding_model.rerank(
                    query=context_string,
                    passages=passage_texts,
                    top_k=max_memories  # Only need top k results
                )
                
                # Build final results from reranked passages
                filtered_results = []
                for idx, rerank_score, _ in reranked_results:
                    passage = importance_filtered[idx].copy()
                    passage["rerank_score"] = rerank_score
                    passage["embedding_score"] = passage.get("similarity", 0)
                    # Use rerank score as primary similarity measure
                    passage["similarity"] = rerank_score
                    filtered_results.append(passage)
                
                logger.debug(f"Reranked {len(importance_filtered)} memories, selected top {len(filtered_results)}")
            except Exception as e:
                logger.warning(f"Reranking failed, using embedding scores: {e}")
                # Fall back to importance-filtered results with original scores
                filtered_results = importance_filtered[:max_memories]
        else:
            # No reranking available or disabled, just limit the importance-filtered results
            filtered_results = importance_filtered[:max_memories]
        
        return filtered_results
        
    except Exception as e:
        logger.error(f"Error finding relevant memories: {e}")
        return []


def format_relevant_memories(memories: List[Dict[str, Any]]) -> str:
    """
    Format relevant memories for inclusion in system prompt.
    
    Args:
        memories: List of memory passages from get_relevant_memories
        
    Returns:
        Formatted memory content string
    """
    if not memories:
        return ""
        
    content_parts = ["# Relevant Context", ""]
    content_parts.append("Based on the current conversation, these memories may be relevant:")
    content_parts.append("")
    
    for memory in memories:
        # Extract key information
        summary = memory.get("summary", "")
        source = memory.get("source", "conversation")
        created_date = memory.get("created_at", "")
        similarity = memory.get("similarity", 0.0)
        
        # Format the memory entry
        if created_date:
            try:
                # Parse and format the date
                dt = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                date_str = dt.strftime("%B %d, %Y")
            except:
                date_str = "unknown date"
        else:
            date_str = "unknown date"
        
        # Add memory with relevance indicator
        relevance = "highly relevant" if similarity > 0.8 else "relevant"
        content_parts.append(f"- {summary} (from {source} on {date_str}, {relevance})")
    
    content_parts.append("")
    content_parts.append("*These memories were automatically surfaced based on conversation context.*")
    
    return "\n".join(content_parts)


def build_weighted_memory_context(messages: List[Any], context_window: int = 6) -> str:
    """
    Build weighted conversation context for memory search with recency weighting.
    
    Recent messages appear more frequently in the context to give them
    more weight in the embedding representation.
    
    Args:
        messages: List of conversation messages
        context_window: Number of recent messages to consider
        
    Returns:
        Formatted weighted context string
    """
    if not messages:
        return ""
    
    # Get recent messages within context window
    recent_messages = messages[-context_window:] if len(messages) > context_window else messages
    
    if not recent_messages:
        return ""
    
    # Build context with recency weighting through repetition
    context_parts = []
    
    # Filter to only user messages for memory context
    user_messages = []
    for message in recent_messages:
        role = getattr(message, 'role', 'unknown')
        content = getattr(message, 'content', '')
        
        # Only include user messages with string content
        if role == 'user' and isinstance(content, str) and content.strip():
            user_messages.append(message)
    
    if not user_messages:
        return ""
    
    for i, message in enumerate(user_messages):
        content = getattr(message, 'content', '')
        formatted_message = f"user: {content}"
        
        # Calculate how many times to include this message based on recency
        # Most recent messages get repeated more
        message_position = i + 1  # 1 = oldest, len = newest
        weight = _calculate_repetition_weight(message_position, len(user_messages))
        
        # Add the message the calculated number of times
        for _ in range(weight):
            context_parts.append(formatted_message)
    
    context_text = "\n".join(context_parts)
    logger.debug(f"Built weighted memory context: {len(context_parts)} total entries from {len(recent_messages)} messages")
    return context_text


def _calculate_repetition_weight(position: int, total_messages: int) -> int:
    """
    Calculate how many times to repeat a message based on its recency.
    
    Args:
        position: Position from oldest (1 = oldest, total_messages = newest)
        total_messages: Total number of messages
        
    Returns:
        Number of times to repeat the message (1-3)
    """
    if total_messages <= 1:
        return 1
    
    # Linear weighting: newest messages repeated most
    normalized_position = (position - 1) / (total_messages - 1)  # 0 = oldest, 1 = newest
    
    # Map to repetition count (1-3 times)
    if normalized_position >= 0.8:  # Most recent 20%
        return 3
    elif normalized_position >= 0.5:  # Middle 30%
        return 2
    else:  # Oldest 50%
        return 1


