"""
Standalone conversation context embedding utility for TOOL CLASSIFICATION ONLY.

Provides shared functionality for building conversation context strings
and caching embeddings for use by the tool relevance engine. Uses BGE 
embeddings (1024-dim) via the unified embeddings provider.

NOTE: Memory operations use the same embeddings provider with optional reranking.
Each system applies its own context building and weighting logic.
"""

import logging
import hashlib
from typing import List, Optional, Any, Dict
from collections import deque

from api.embeddings_provider import EmbeddingsProvider

logger = logging.getLogger(__name__)


class ConversationContextEmbedding:
    """
    Standalone utility for caching conversation context embeddings for TOOL CLASSIFICATION.
    
    This class provides embedding generation and caching for arbitrary
    context strings using BGE embeddings (1024-dim) via the unified provider.
    Used exclusively by the tool relevance engine.
    
    NOTE: Memory operations use the same embeddings provider with optional reranking.
    """
    
    def __init__(self, cache_size: int = 100, embeddings_provider: Optional[EmbeddingsProvider] = None):
        """
        Initialize conversation context embedding utility.
        
        Args:
            cache_size: Maximum number of context embeddings to cache
            embeddings_provider: Optional embeddings provider instance (creates one if not provided)
        """
        self.cache_size = cache_size
        self.logger = logging.getLogger(__name__)
        
        # Message history buffer for systems that need it
        self.message_history: deque = deque(maxlen=20)  # Reasonable default
        
        # Embedding cache for context strings
        self.context_cache: Dict[str, Any] = {}
        
        # Embeddings provider instance
        if embeddings_provider is not None:
            self.embeddings = embeddings_provider
        else:
            # Create a local BGE provider for tool classification
            self.embeddings = EmbeddingsProvider(
                provider_type="local",
                enable_reranker=False  # No reranker needed for tool classification
            )
        
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history buffer.
        
        This is a convenience method for systems that need to track
        conversation context. Each system can build its own context
        from this history.
        
        Args:
            role: Message role (user/assistant)
            content: Message content
        """
        # Only process string content
        if not isinstance(content, str) or not content.strip():
            return
            
        message = {
            "role": role,
            "content": content.strip()
        }
        
        self.message_history.append(message)
        self.logger.debug(f"Added {role} message to history buffer")
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get the current message history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return list(self.message_history)
    
    def flush_context(self) -> None:
        """
        Flush conversation context on topic changes.
        
        This clears the message history and context cache,
        typically called when a topic change is detected.
        """
        self.message_history.clear()
        self._trim_cache()  # Keep cache but trim if needed
        
        self.logger.info("Flushed conversation context due to topic change")
    
    def get_embedding(self, context_text: str) -> Optional[Any]:
        """
        Get embedding for arbitrary context text with caching.
        
        This is the main interface - systems pass their formatted
        context string and get back a cached or newly generated embedding.
        
        Args:
            context_text: Formatted context string to embed
            
        Returns:
            Context embedding vector or None if generation fails
        """
        if not context_text or not context_text.strip():
            return None
        
        try:
            # Generate hash for caching
            context_hash = self._hash_context(context_text)
            
            # Return cached embedding if available
            if context_hash in self.context_cache:
                self.logger.debug("Using cached context embedding")
                return self.context_cache[context_hash]
            
            # Generate new embedding
            embedding = self.embeddings.encode(context_text)
            
            if embedding is not None:
                # Cache the embedding (with size management)
                self.context_cache[context_hash] = embedding
                self._trim_cache()
                
                self.logger.debug(f"Generated and cached new context embedding")
                return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating context embedding: {e}")
        
        return None
    
    def _hash_context(self, context_text: str) -> str:
        """
        Generate a hash for context text for caching purposes.
        
        Args:
            context_text: Context string to hash
            
        Returns:
            SHA-256 hash of the context
        """
        return hashlib.sha256(context_text.encode('utf-8')).hexdigest()[:16]  # Shorter hash for efficiency
    
    def _trim_cache(self) -> None:
        """
        Trim cache to stay within size limits.
        
        Uses simple LRU-style removal (remove oldest entries).
        """
        while len(self.context_cache) > self.cache_size:
            # Remove one arbitrary entry (dict is insertion-ordered in Python 3.7+)
            oldest_key = next(iter(self.context_cache))
            del self.context_cache[oldest_key]
            self.logger.debug("Trimmed oldest entry from context cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the context cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self.context_cache),
            "max_cache_size": self.cache_size,
            "message_buffer_size": len(self.message_history),
            "cache_hit_rate": "not_tracked"  # Could add this later if needed
        }
    
    def clear_cache(self) -> None:
        """
        Clear the embedding cache but preserve message history.
        
        This is useful when you want to force regeneration of embeddings
        without losing conversation context.
        """
        cache_size = len(self.context_cache)
        self.context_cache.clear()
        
        self.logger.info(f"Cleared context embedding cache ({cache_size} entries)")


# Global instance for shared use across components
_global_context_embedding: Optional[ConversationContextEmbedding] = None


def get_conversation_context_embedding() -> ConversationContextEmbedding:
    """
    Get the global conversation context embedding instance for TOOL CLASSIFICATION.
    
    This provides a singleton pattern for sharing context for tool relevance engine.
    Uses ONNX embeddings (384-dim) for fast, local processing.
    
    NOTE: Memory operations use separate OpenAI embeddings (1024-dim) for higher quality.
    
    Returns:
        Global ConversationContextEmbedding instance for tool classification
    """
    global _global_context_embedding
    
    if _global_context_embedding is None:
        _global_context_embedding = ConversationContextEmbedding()
        logger.info("Initialized global conversation context embedding")
    
    return _global_context_embedding


def reset_conversation_context_embedding() -> None:
    """
    Reset the global conversation context embedding instance.
    
    This is primarily for testing or when you need to completely
    reset the conversation state.
    """
    global _global_context_embedding
    _global_context_embedding = None
    logger.info("Reset global conversation context embedding")