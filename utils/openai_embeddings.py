"""
OpenAI embeddings utility for high-quality, case-sensitive embedding generation.

This module provides access to OpenAI's text-embedding-3-small model for use cases
where case sensitivity and higher dimensional embeddings are critical for accuracy.
"""

import os
import logging
import numpy as np
import time
from typing import List, Union, Dict, Any
from pathlib import Path
import openai
from openai import OpenAI
from dotenv import load_dotenv
import httpx

from errors import error_context, ErrorCode, ToolError

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class OpenAIEmbeddingModel:
    """
    OpenAI text-embedding-3-small model wrapper for high-quality embeddings.
    
    Provides case-sensitive, 1024-dimensional embeddings suitable for memory systems
    where semantic precision is critical.
    """
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding model.
        
        Args:
            api_key: OpenAI API key. If None, loads from OAI_EMBEDDINGS_KEY env var
            model: OpenAI embedding model to use
        """
        with error_context("OpenAIEmbeddingModel", "initialization", error_code=ErrorCode.TOOL_INITIALIZATION_ERROR):
            # Get API key from parameter or environment
            if api_key is None:
                api_key = os.getenv("OAI_EMBEDDINGS_KEY")
                if not api_key:
                    raise ToolError(
                        "OpenAI API key not found. Set OAI_EMBEDDINGS_KEY environment variable or pass api_key parameter.",
                        code=ErrorCode.PARAMETER_MISSING,
                        permanent_failure=True
                    )
            
            self.api_key = api_key
            self.model = model
            
            # Initialize OpenAI client with optimized connection pooling
            try:
                # Create HTTP client with connection pooling and keepalive
                http_client = httpx.Client(
                    limits=httpx.Limits(
                        max_keepalive_connections=10,
                        max_connections=20,
                        keepalive_expiry=300  # 5 minutes
                    ),
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=60.0,
                        write=10.0,
                        pool=5.0
                    )
                )
                
                self.client = OpenAI(
                    api_key=self.api_key,
                    http_client=http_client
                )
            except Exception as e:
                raise ToolError(
                    f"Failed to initialize OpenAI client: {e}",
                    code=ErrorCode.TOOL_INITIALIZATION_ERROR,
                    permanent_failure=True
                )
            
            # Set embedding dimensions based on model
            self.embedding_dims = {
                "text-embedding-3-small": 1024,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1024
            }
            
            if model not in self.embedding_dims:
                raise ToolError(
                    f"Unsupported embedding model: {model}. Supported: {list(self.embedding_dims.keys())}",
                    code=ErrorCode.PARAMETER_INVALID,
                    permanent_failure=True
                )
            
            self.embedding_dim = self.embedding_dims[model]
            
            logger.info(f"Initialized OpenAI embedding model: {model} (dim={self.embedding_dim}) with connection pooling")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 100) -> np.ndarray:
        """
        Encode texts to embeddings using OpenAI API.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Maximum texts per API call (OpenAI limit is ~2048)
            
        Returns:
            Numpy array of embeddings (float32)
        """
        with error_context("OpenAIEmbeddingModel", "encoding", error_code=ErrorCode.MEMORY_EMBEDDING_ERROR):
            if isinstance(texts, str):
                texts = [texts]
                single_input = True
            else:
                single_input = False
            
            if not texts:
                raise ToolError(
                    "No texts provided for encoding",
                    code=ErrorCode.PARAMETER_MISSING,
                    permanent_failure=True
                )
            
            # Check for empty strings
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    raise ToolError(
                        f"Empty text at index {i}. OpenAI embedding API requires non-empty text.",
                        code=ErrorCode.PARAMETER_INVALID,
                        permanent_failure=True
                    )
            
            all_embeddings = []
            
            # Process in batches to respect API limits
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    # Log request start with stack trace info
                    start_time = time.time()
                    batch_info = f"batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
                    
                    # Get caller info for debugging
                    import traceback
                    caller_info = traceback.extract_stack()[-3:-1]  # Get 2 levels up
                    caller_desc = f"{caller_info[0].filename.split('/')[-1]}:{caller_info[0].lineno}"
                    
                    logger.info(f"OpenAI API request starting - {len(batch_texts)} texts, {batch_info} (called from {caller_desc})")
                    
                    # Call OpenAI embeddings API with dimensions parameter for 1024-dim output
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch_texts,
                        encoding_format="float",
                        dimensions=self.embedding_dim
                    )
                    
                    # Log response received with timing
                    end_time = time.time()
                    time_in_flight = (end_time - start_time) * 1000  # Convert to milliseconds
                    logger.info(f"OpenAI API response received - {len(response.data)} embeddings, {time_in_flight:.1f}ms time-in-flight")
                    
                    # Extract embeddings from response
                    batch_embeddings = []
                    for embedding_obj in response.data:
                        embedding = np.array(embedding_obj.embedding, dtype=np.float32)
                        
                        # Verify embedding dimension
                        if embedding.shape[0] != self.embedding_dim:
                            raise ToolError(
                                f"Unexpected embedding dimension: got {embedding.shape[0]}, expected {self.embedding_dim}",
                                code=ErrorCode.MEMORY_EMBEDDING_ERROR,
                                permanent_failure=False
                            )
                        
                        batch_embeddings.append(embedding)
                    
                    all_embeddings.extend(batch_embeddings)
                    
                except openai.RateLimitError as e:
                    raise ToolError(
                        f"OpenAI API rate limit exceeded: {e}",
                        code=ErrorCode.API_RATE_LIMIT_ERROR,
                        permanent_failure=False
                    )
                except openai.AuthenticationError as e:
                    raise ToolError(
                        f"OpenAI API authentication failed: {e}",
                        code=ErrorCode.API_AUTH_ERROR,
                        permanent_failure=True
                    )
                except openai.APIError as e:
                    raise ToolError(
                        f"OpenAI API error: {e}",
                        code=ErrorCode.API_RESPONSE_ERROR,
                        permanent_failure=False
                    )
                except Exception as e:
                    raise ToolError(
                        f"Unexpected error during embedding generation: {e}",
                        code=ErrorCode.MEMORY_EMBEDDING_ERROR,
                        permanent_failure=False
                    )
            
            # Convert to numpy array
            result = np.array(all_embeddings, dtype=np.float32)
            
            # Return single embedding if input was single text
            if single_input:
                return result[0]
            
            return result
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get the embedding dimension size."""
        return self.embedding_dim
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the OpenAI API connection and return diagnostics.
        
        Returns:
            Dictionary with connection test results
        """
        try:
            # Test with a simple embedding
            logger.info("Testing OpenAI API connection with test embedding")
            test_embedding = self.encode("Hello, world!")
            
            return {
                "status": "success",
                "model": self.model,
                "embedding_dim": self.embedding_dim,
                "test_embedding_shape": test_embedding.shape,
                "test_embedding_norm": float(np.linalg.norm(test_embedding)),
                "api_accessible": True
            }
        except Exception as e:
            return {
                "status": "error",
                "model": self.model,
                "error": str(e),
                "api_accessible": False
            }
    
    def close(self):
        """Close the HTTP client and clean up connections."""
        try:
            if hasattr(self.client, '_client') and hasattr(self.client._client, 'close'):
                self.client._client.close()
                logger.debug("Closed OpenAI HTTP client connections")
        except Exception as e:
            logger.warning(f"Error closing OpenAI client: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()