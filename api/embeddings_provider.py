"""
Unified embeddings provider for local and remote embedding models.

This module provides a single interface for generating text embeddings,
routing to either local (BGE) or remote (OpenAI) implementations based
on configuration. All embeddings are 1024-dimensional and L2-normalized.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np

from errors import APIError, ErrorCode, error_context
from config import config
from utils.timezone_utils import utc_now


class EmbeddingCompatibilityError(APIError):
    """Raised when stored embeddings don't match current provider configuration."""
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.EMBEDDING_COMPATIBILITY_ERROR)


class EmbeddingsProvider:
    """
    Unified provider for text embeddings.
    
    Provides a single interface for both local (BGE) and remote (OpenAI)
    embeddings, always returning 1024-dimensional normalized vectors.
    """
    
    MARKER_FILE = "data/.embedding_provider"
    
    def __init__(self,
                 provider_type: Optional[str] = None,
                 api_key: Optional[str] = None,
                 enable_reranker: bool = False,
                 cache_enabled: Optional[bool] = None):
        """
        Initialize embeddings provider.
        
        Args:
            provider_type: "local" or "remote" (defaults to config)
            api_key: API key for remote provider (defaults to env)
            enable_reranker: Enable reranking for local provider
            cache_enabled: Enable caching (defaults to config)
        """
        self.logger = logging.getLogger("embeddings_provider")
        
        # Set provider type from parameter or config
        self.provider_type = provider_type or config.embeddings.provider
        if self.provider_type not in ["local", "remote"]:
            raise APIError(
                f"Invalid provider type: {self.provider_type}. Must be 'local' or 'remote'",
                ErrorCode.CONFIGURATION_ERROR
            )
        
        # Set other parameters
        self.enable_reranker = enable_reranker
        self.cache_enabled = cache_enabled if cache_enabled is not None else config.embeddings.cache_enabled
        
        # Initialize implementation
        self._impl = None
        self._reranker = None
        self.model_name = None
        self._setup_implementation(api_key)
        
        # Check compatibility with stored embeddings
        self._ensure_compatibility()
        
        self.logger.info(
            f"Embeddings provider initialized: type={self.provider_type}, "
            f"model={self.model_name}, reranker={enable_reranker}"
        )
    
    def _setup_implementation(self, api_key: Optional[str] = None):
        """Load the appropriate embedding implementation."""
        with error_context(
            component_name="EmbeddingsProvider",
            operation="setup_implementation",
            error_class=APIError,
            error_code=ErrorCode.INITIALIZATION_ERROR,
            logger=self.logger
        ):
            if self.provider_type == "local":
                from utils.bge_embeddings import BGEEmbeddingModel, BGEReranker
                
                self._impl = BGEEmbeddingModel(
                    model_name="BAAI/bge-large-en-v1.5",
                    use_int8=config.embeddings.local.use_int8,
                    cache_dir=config.embeddings.local.cache_dir,
                    thread_limit=config.embeddings.local.thread_limit
                )
                self.model_name = "BAAI/bge-large-en-v1.5"
                
                # Initialize reranker if requested
                if self.enable_reranker:
                    self._reranker = BGEReranker(
                        model_name="BAAI/bge-reranker-base",
                        use_fp16=config.embeddings.local.reranker_use_fp16,
                        cache_dir=config.embeddings.local.cache_dir,
                        thread_limit=config.embeddings.local.thread_limit
                    )
                    
            elif self.provider_type == "remote":
                from utils.openai_embeddings import OpenAIEmbeddingModel
                
                # Use provided API key or get from config/env
                if not api_key:
                    api_key = config.embeddings_api_key
                
                self._impl = OpenAIEmbeddingModel(
                    api_key=api_key,
                    model=config.embeddings.remote.model
                )
                self.model_name = config.embeddings.remote.model
                
                # No reranker for remote provider
                if self.enable_reranker:
                    self.logger.warning(
                        "Reranker not available for remote provider. "
                        "Only local BGE provider supports reranking."
                    )
                    self.enable_reranker = False
    
    def _ensure_compatibility(self):
        """Check if current provider matches stored embeddings."""
        marker_path = Path(self.MARKER_FILE)
        current_id = f"{self.provider_type}:{self.model_name}"
        
        if marker_path.exists():
            try:
                stored = json.loads(marker_path.read_text())
                stored_id = f"{stored['provider']}:{stored['model']}"
                
                if stored_id != current_id:
                    raise EmbeddingCompatibilityError(
                        f"Embedding provider mismatch!\n"
                        f"  Stored embeddings: {stored_id}\n"
                        f"  Current config: {current_id}\n"
                        f"\n"
                        f"Options to resolve:\n"
                        f"  1. Change config.embeddings.provider back to '{stored['provider']}'\n"
                        f"  2. Delete {self.MARKER_FILE} and all stored embeddings\n"
                        f"  3. Run 'python scripts/migrate_embeddings.py' to re-embed with {current_id}"
                    )
                    
                self.logger.info(f"Embedding provider compatibility confirmed: {current_id}")
                
            except json.JSONDecodeError as e:
                raise APIError(
                    f"Corrupted embedding provider marker file: {self.MARKER_FILE}",
                    ErrorCode.CONFIGURATION_ERROR
                )
        else:
            # First time - create marker
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_data = {
                "provider": self.provider_type,
                "model": self.model_name,
                "dimension": 1024,
                "created_at": utc_now().isoformat(),
                "version": "v1"
            }
            marker_path.write_text(json.dumps(marker_data, indent=2))
            
            self.logger.info(f"Created embedding provider marker: {current_id}")
    
    def encode(self,
               texts: Union[str, List[str]],
               batch_size: Optional[int] = None,
               show_progress: bool = False,
               normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for processing (defaults to config)
            show_progress: Show progress bar for large batches
            normalize: L2-normalize embeddings (always True for consistency)
            
        Returns:
            numpy array of shape (n, 1024) with normalized embeddings
            
        Raises:
            APIError: If embedding generation fails
        """
        with error_context(
            component_name="EmbeddingsProvider",
            operation="encode",
            error_class=APIError,
            error_code=ErrorCode.EMBEDDING_GENERATION_ERROR,
            logger=self.logger
        ):
            # Use configured batch size if not provided
            if batch_size is None:
                batch_size = config.embeddings.batch_size
            
            # Ensure we always normalize for consistency
            if not normalize:
                self.logger.warning(
                    "normalize=False ignored. All embeddings are L2-normalized for consistency."
                )
            
            # Handle caching if enabled
            if self.cache_enabled:
                # Initialize cache if not already done
                if not hasattr(self, '_cache'):
                    from lt_memory.utils.embeddings import EmbeddingCache
                    cache_dir = config.paths.data_dir
                    self._cache = EmbeddingCache(cache_dir)
                
                # For single text, check cache and compute if needed
                if isinstance(texts, str):
                    cached_embedding = self._cache.get(texts)
                    if cached_embedding is not None:
                        return cached_embedding
                    
                    # Not in cache, compute and cache it
                    embedding = self._impl.encode(texts, batch_size=batch_size)
                    embedding = self._normalize_embedding(embedding)
                    self._cache.set(texts, embedding)
                    return embedding
                
                # For batch processing, check cache for each text
                embeddings = []
                uncached_texts = []
                uncached_indices = []
                
                for i, text in enumerate(texts):
                    cached_embedding = self._cache.get(text)
                    if cached_embedding is not None:
                        embeddings.append((i, cached_embedding))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                
                # Compute embeddings for uncached texts
                if uncached_texts:
                    new_embeddings = self._impl.encode(uncached_texts, batch_size=batch_size)
                    new_embeddings = self._normalize_embeddings(new_embeddings)
                    
                    # Cache new embeddings and add to results
                    for j, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                        self._cache.set(text, embedding)
                        embeddings.append((uncached_indices[j], embedding))
                
                # Sort by original order and extract embeddings
                embeddings.sort(key=lambda x: x[0])
                return np.array([emb for _, emb in embeddings])
            
            # Direct encoding without cache
            embeddings = self._impl.encode(texts, batch_size=batch_size)
            
            # Ensure normalization
            embeddings = self._normalize_embeddings(embeddings)
            
            return embeddings
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize a single embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize a batch of embedding vectors."""
        if len(embeddings.shape) == 1:
            # Single embedding
            return self._normalize_embedding(embeddings)
        else:
            # Batch of embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / (norms + 1e-10)
    
    def rerank(self,
               query: str,
               passages: List[str],
               top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        Rerank passages based on relevance to query.
        
        Only available for local provider with reranker enabled.
        
        Args:
            query: Query text
            passages: List of passages to rerank
            top_k: Number of top results to return
            
        Returns:
            List of (index, score, passage) tuples sorted by relevance
            
        Raises:
            APIError: If reranker not available
        """
        if not self.enable_reranker or not self._reranker:
            raise APIError(
                "Reranker not available. Enable with enable_reranker=True "
                "and use local provider.",
                ErrorCode.FEATURE_NOT_AVAILABLE
            )
        
        with error_context(
            component_name="EmbeddingsProvider",
            operation="rerank",
            error_class=APIError,
            error_code=ErrorCode.OPERATION_FAILED,
            logger=self.logger
        ):
            # Get reranking scores
            scores_with_indices = self._reranker.rerank(
                query, passages, return_scores=True
            )
            
            # Take top_k results
            top_results = scores_with_indices[:top_k]
            
            # Add passage text to results
            results = [
                (idx, score, passages[idx])
                for idx, score in top_results
            ]
            
            return results
    
    def search_and_rerank(self,
                          query: str,
                          passages: List[str],
                          passage_embeddings: Optional[np.ndarray] = None,
                          initial_top_k: int = 50,
                          final_top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        Two-stage retrieval: embedding search followed by reranking.
        
        Args:
            query: Query text
            passages: List of passages
            passage_embeddings: Pre-computed embeddings (will compute if None)
            initial_top_k: Number of candidates from embedding search
            final_top_k: Number of final results after reranking
            
        Returns:
            List of (index, score, passage) tuples
        """
        with error_context(
            component_name="EmbeddingsProvider",
            operation="search_and_rerank",
            error_class=APIError,
            error_code=ErrorCode.OPERATION_FAILED,
            logger=self.logger
        ):
            # Compute passage embeddings if not provided
            if passage_embeddings is None:
                passage_embeddings = self.encode(passages)
            
            # Encode query
            query_embedding = self.encode(query)
            
            # Compute similarities
            similarities = np.dot(passage_embeddings, query_embedding)
            
            # Get top candidates
            top_indices = np.argsort(similarities)[::-1][:initial_top_k]
            top_passages = [passages[i] for i in top_indices]
            
            # Rerank if available
            if self.enable_reranker and self._reranker:
                reranked = self.rerank(query, top_passages, top_k=final_top_k)
                # Map back to original indices
                results = [
                    (top_indices[idx], score, passage)
                    for idx, score, passage in reranked
                ]
                return results
            else:
                # Return embedding-based results
                results = [
                    (idx, float(similarities[idx]), passages[idx])
                    for idx in top_indices[:final_top_k]
                ]
                return results
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension (always 1024)."""
        return 1024
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get provider metadata."""
        return {
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "dimension": 1024,
            "reranker_enabled": self.enable_reranker,
            "cache_enabled": self.cache_enabled,
            "provider_id": f"{self.provider_type}:{self.model_name}"
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the provider connection and return diagnostics."""
        if hasattr(self._impl, 'test_connection'):
            return self._impl.test_connection()
        else:
            # Basic test
            try:
                test_embedding = self.encode("Hello, world!")
                return {
                    "status": "success",
                    "provider": self.provider_type,
                    "model": self.model_name,
                    "test_embedding_shape": test_embedding.shape,
                    "test_embedding_norm": float(np.linalg.norm(test_embedding))
                }
            except Exception as e:
                return {
                    "status": "error",
                    "provider": self.provider_type,
                    "model": self.model_name,
                    "error": str(e)
                }
    
    def close(self):
        """Clean up resources."""
        if hasattr(self._impl, 'close'):
            self._impl.close()
        if self._reranker and hasattr(self._reranker, 'close'):
            self._reranker.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()