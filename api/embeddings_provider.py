"""
Unified embeddings provider supporting local (BGE) and remote (OpenAI) models.

This module provides a single interface for generating text embeddings using either
local BAAI BGE models with quantization or remote OpenAI embeddings API.
"""
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from errors import APIError, ErrorCode, error_context
from config import config


class EmbeddingsProvider:
    """
    Unified provider for text embeddings.
    
    Supports:
    - Local: BGE-large-en-v1.5 with INT8 quantization for efficient inference
    - Remote: OpenAI text-embedding-3-small (1024 dimensions)
    """
    
    def __init__(self,
                 provider_type: Optional[str] = None,
                 api_endpoint: Optional[str] = None,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 max_retries: Optional[int] = None,
                 timeout: Optional[int] = None,
                 device: Optional[str] = None,
                 quantize_int8: bool = True,
                 enable_reranker: bool = True):
        """Initialize the embeddings provider.
        
        Args:
            provider_type: "local" for BGE models or "remote" for OpenAI
            api_endpoint: API endpoint URL (for remote provider)
            model: Model name override
            api_key: API key override (for remote provider)
            batch_size: Batch size for encoding
            max_retries: Max retries for API calls
            timeout: Timeout for API calls
            device: Device for local model ("cpu" or "cuda")
            quantize_int8: Whether to use INT8 quantization for local models
            enable_reranker: Whether to enable BGE reranker for local provider
        """
        self.logger = logging.getLogger("embeddings_provider")
        
        # Get configuration with overrides
        self.provider_type = provider_type if provider_type is not None else config.embeddings.provider
        self.batch_size = batch_size if batch_size is not None else config.embeddings.batch_size
        self.max_retries = max_retries if max_retries is not None else config.embeddings.max_retries
        self.timeout = timeout if timeout is not None else config.embeddings.timeout
        self.device = device if device is not None else config.embeddings.device
        self.quantize_int8 = quantize_int8
        
        # Provider-specific configuration
        self.enable_reranker = enable_reranker
        self._reranker = None
        
        if self.provider_type == "local":
            self.model = model if model is not None else config.embeddings.local_model
            self._local_model = None
            self._init_local_model()
        else:  # remote
            self.model = model if model is not None else config.embeddings.remote_model
            self.api_endpoint = api_endpoint if api_endpoint is not None else config.embeddings.api_endpoint
            self.api_key = api_key if api_key is not None else config.embeddings_api_key
            self._session = None
            self._request_lock = threading.Lock()
        
        # Initialize reranker if enabled (works with both local and remote embeddings)
        if self.enable_reranker:
            self._init_reranker()
        
        self.logger.info(
            f"Embeddings provider initialized: type={self.provider_type}, "
            f"model={self.model}, device={self.device if self.provider_type == 'local' else 'N/A'}"
        )
    
    def _init_local_model(self):
        """Initialize local BGE model with ONNX Runtime and INT8 quantization."""
        try:
            from utils.bge_embeddings import BGEEmbeddingModel
            
            # Initialize BGE model with ONNX Runtime
            self._local_model = BGEEmbeddingModel(
                model_name=self.model,
                use_int8=self.quantize_int8,
                cache_dir=None,  # Will use default cache dir
                thread_limit=4   # Reasonable default for CPU
            )
            
            self.logger.info(f"Local BGE ONNX model loaded: {self.model} (INT8={self.quantize_int8})")
            
        except ImportError as e:
            if "onnxruntime" in str(e):
                raise APIError(
                    "ONNX Runtime not installed. Run: pip install onnxruntime",
                    ErrorCode.DEPENDENCY_ERROR
                )
            raise
        except Exception as e:
            raise APIError(
                f"Failed to initialize local BGE model: {str(e)}",
                ErrorCode.INITIALIZATION_ERROR
            )
    
    def _init_reranker(self):
        """Initialize BGE reranker model (works with both local and remote embeddings)."""
        try:
            from utils.bge_embeddings import BGEReranker
            
            # Initialize BGE reranker with ONNX Runtime
            self._reranker = BGEReranker(
                model_name="BAAI/bge-reranker-base",
                use_fp16=True,  # FP16 for better performance
                thread_limit=4  # Reasonable default for CPU
            )
            
            self.logger.info("BGE reranker initialized with FP16 precision")
            
        except ImportError as e:
            if "onnxruntime" in str(e):
                raise APIError(
                    "ONNX Runtime not installed. Run: pip install onnxruntime",
                    ErrorCode.DEPENDENCY_ERROR
                )
            raise
        except Exception as e:
            raise APIError(
                f"Failed to initialize BGE reranker: {str(e)}",
                ErrorCode.INITIALIZATION_ERROR
            )
    
    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: Optional[int] = None,
               normalize: bool = True,
               show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for input texts.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Override batch size
            normalize: Whether to L2 normalize embeddings
            show_progress: Whether to show progress bar (local only)
            
        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        # Use provided batch_size or default
        batch_size = batch_size or self.batch_size
        
        with error_context(
            component_name="EmbeddingsProvider",
            operation="encode",
            error_class=APIError,
            error_code=ErrorCode.EMBEDDING_ERROR,
            logger=self.logger
        ):
            if self.provider_type == "local":
                embeddings = self._encode_local(texts, batch_size, normalize, show_progress)
            else:
                embeddings = self._encode_remote(texts, batch_size, normalize)
            
            # Return single embedding if single input
            if single_input:
                return embeddings[0]
            return embeddings
    
    def _encode_local(self, 
                      texts: List[str], 
                      batch_size: int,
                      normalize: bool,
                      show_progress: bool) -> np.ndarray:
        """Encode texts using local BGE model."""
        if self._local_model is None:
            raise APIError(
                "Local model not initialized",
                ErrorCode.INITIALIZATION_ERROR
            )
        
        # BGE ONNX model handles batching internally
        embeddings = self._local_model.encode(
            texts,
            batch_size=batch_size,
            normalize=normalize,
            show_progress=show_progress
        )
        
        return embeddings
    
    def _encode_remote(self,
                       texts: List[str],
                       batch_size: int,
                       normalize: bool) -> np.ndarray:
        """Encode texts using remote OpenAI API."""
        all_embeddings = []
        
        # Process in batches (OpenAI has a limit)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._call_openai_api(batch)
            all_embeddings.extend(batch_embeddings)
        
        # Convert to numpy array
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)
        
        return embeddings
    
    def _call_openai_api(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI embeddings API."""
        with self._request_lock:
            session = self._create_session()
            
            request_body = {
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
                "dimensions": 1024  # Match BGE dimensions for compatibility
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            try:
                response = session.post(
                    self.api_endpoint,
                    json=request_body,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                # Extract embeddings in order
                embeddings = [None] * len(texts)
                for item in data["data"]:
                    embeddings[item["index"]] = item["embedding"]
                
                return embeddings
                
            except requests.exceptions.HTTPError as e:
                self._handle_http_error(e)
            except requests.exceptions.RequestException as e:
                raise APIError(
                    f"Network error: {str(e)}",
                    ErrorCode.API_CONNECTION_ERROR
                )
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _handle_http_error(self, error: requests.exceptions.HTTPError):
        """Handle HTTP errors from API."""
        if error.response is None:
            raise APIError(
                "No response from API",
                ErrorCode.API_CONNECTION_ERROR
            )
        
        status_code = error.response.status_code
        
        try:
            error_data = error.response.json()
            error_message = error_data.get("error", {}).get("message", str(error))
        except:
            error_message = error.response.text or str(error)
        
        if status_code == 401:
            raise APIError(
                "Authentication failed. Check your API key.",
                ErrorCode.API_AUTHENTICATION_ERROR
            )
        elif status_code == 429:
            raise APIError(
                "Rate limit exceeded. Please try again later.",
                ErrorCode.API_RATE_LIMIT_ERROR
            )
        else:
            raise APIError(
                f"API error (status {status_code}): {error_message}",
                ErrorCode.API_RESPONSE_ERROR
            )
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        if self.provider_type == "local":
            return self._local_model.get_sentence_embedding_dimension()
        else:
            # OpenAI text-embedding-3-small produces 1024-dimensional embeddings
            return 1024
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to the embeddings provider."""
        try:
            # Test with a simple text
            test_text = "This is a test embedding."
            embedding = self.encode(test_text)
            
            return {
                "success": True,
                "provider": self.provider_type,
                "model": self.model,
                "embedding_dimension": len(embedding),
                "message": f"Successfully generated {len(embedding)}-dimensional embedding"
            }
        except Exception as e:
            return {
                "success": False,
                "provider": self.provider_type,
                "model": self.model,
                "error": str(e)
            }
    
    def compute_similarity(self, 
                          query_embedding: np.ndarray,
                          embeddings: np.ndarray,
                          metric: str = "cosine") -> np.ndarray:
        """
        Compute similarity between query embedding and a set of embeddings.
        
        Args:
            query_embedding: Query embedding vector
            embeddings: Matrix of embeddings to compare against
            metric: Similarity metric ("cosine" or "dot")
            
        Returns:
            Array of similarity scores
        """
        if metric == "cosine":
            # Ensure embeddings are normalized for cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
            return np.dot(embeddings_norm, query_norm)
        elif metric == "dot":
            return np.dot(embeddings, query_embedding)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def rerank(self,
               query: str,
               passages: List[str],
               top_k: Optional[int] = None,
               batch_size: Optional[int] = None) -> List[Tuple[int, float, str]]:
        """
        Rerank passages using BGE reranker for improved relevance.
        
        Args:
            query: Query text
            passages: List of passages to rerank
            top_k: Return only top K results (None for all)
            batch_size: Batch size for processing
            
        Returns:
            List of (index, score, passage) tuples sorted by relevance
            
        Raises:
            APIError: If reranker is not enabled
        """
        if not self.enable_reranker or self._reranker is None:
            raise APIError(
                "Reranker not enabled. Initialize with enable_reranker=True",
                ErrorCode.CONFIGURATION_ERROR
            )
        
        batch_size = batch_size or self.batch_size
        
        with error_context(
            component_name="EmbeddingsProvider",
            operation="rerank",
            error_class=APIError,
            error_code=ErrorCode.RERANKING_ERROR,
            logger=self.logger
        ):
            # Get reranked indices and scores
            reranked_results = self._reranker.rerank(
                query=query,
                passages=passages,
                batch_size=batch_size,
                return_scores=True
            )
            
            # Build results with passages
            results = [
                (idx, score, passages[idx])
                for idx, score in reranked_results
            ]
            
            # Apply top_k if specified
            if top_k is not None and top_k < len(results):
                results = results[:top_k]
            
            return results
    
    def search_and_rerank(self,
                          query: str,
                          passages: List[str],
                          passage_embeddings: Optional[np.ndarray] = None,
                          initial_top_k: int = 50,
                          final_top_k: int = 10,
                          batch_size: Optional[int] = None) -> List[Tuple[int, float, str]]:
        """
        Search using embeddings then rerank with BGE reranker.
        
        This implements the efficient two-stage retrieval:
        1. Fast embedding similarity search to get initial candidates
        2. Accurate reranking of top candidates
        
        Args:
            query: Query text
            passages: List of passages
            passage_embeddings: Pre-computed embeddings (will compute if None)
            initial_top_k: Number of candidates from embedding search
            final_top_k: Number of final results after reranking
            batch_size: Batch size for processing
            
        Returns:
            List of (index, score, passage) tuples sorted by relevance
        """
        batch_size = batch_size or self.batch_size
        
        with error_context(
            component_name="EmbeddingsProvider",
            operation="search_and_rerank",
            error_class=APIError,
            error_code=ErrorCode.SEARCH_ERROR,
            logger=self.logger
        ):
            # Compute passage embeddings if not provided
            if passage_embeddings is None:
                self.logger.info(f"Computing embeddings for {len(passages)} passages")
                passage_embeddings = self.encode(passages, batch_size=batch_size)
            
            # Compute query embedding
            query_embedding = self.encode(query)
            
            # Stage 1: Fast embedding similarity search
            similarities = self.compute_similarity(query_embedding, passage_embeddings)
            
            # Get top candidates
            top_indices = np.argsort(similarities)[::-1][:initial_top_k]
            top_passages = [passages[i] for i in top_indices]
            
            # Stage 2: Rerank if enabled
            if self.enable_reranker and self._reranker is not None:
                # Rerank the top candidates
                reranked_results = self.rerank(
                    query=query,
                    passages=top_passages,
                    top_k=final_top_k,
                    batch_size=batch_size
                )
                
                # Map back to original indices
                final_results = [
                    (top_indices[local_idx], score, passage)
                    for local_idx, score, passage in reranked_results
                ]
                
                return final_results
            else:
                # Return embedding-based results
                final_indices = top_indices[:final_top_k]
                return [
                    (idx, float(similarities[idx]), passages[idx])
                    for idx in final_indices
                ]