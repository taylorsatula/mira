"""
ONNX-optimized embedding model for efficient vector generation.

Uses ONNX Runtime for CPU-optimized inference with transformers models.
"""

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import logging
from typing import List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ONNXEmbeddingModel:
    """
    ONNX-optimized embedding model.
    
    Provides efficient embedding generation using ONNX Runtime with
    CPU-specific optimizations and batch processing support.
    """
    
    def __init__(self, model_path: str, tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize ONNX embedding model.
        
        Args:
            model_path: Path to ONNX model file
            tokenizer_name: HuggingFace tokenizer to use
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Create ONNX session with optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.inter_op_num_threads = 4
        sess_options.intra_op_num_threads = 4
        
        # Use CPU provider with optimizations
        providers = [
            ('CPUExecutionProvider', {
                'arena_extend_strategy': 'kSameAsRequested',
            })
        ]
        
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), 
                sess_options, 
                providers=providers
            )
            
            # Get input/output names
            self.input_names = [i.name for i in self.session.get_inputs()]
            self.output_name = self.session.get_outputs()[0].name
            
            # Get embedding dimension from model
            self.embedding_dim = self.session.get_outputs()[0].shape[-1]
            
            logger.info(f"Loaded ONNX model from {model_path} with dim={self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, 
               show_progress: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings using ONNX.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress (not implemented)
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np"
            )
            
            # Prepare inputs - handle different model architectures
            onnx_inputs = {}
            for name in self.input_names:
                if name in encoded:
                    onnx_inputs[name] = encoded[name].astype(np.int64)
                elif name == "token_type_ids" and "token_type_ids" not in encoded:
                    # Some models don't use token_type_ids
                    onnx_inputs[name] = np.zeros_like(encoded["input_ids"], dtype=np.int64)
            
            # Run inference
            outputs = self.session.run([self.output_name], onnx_inputs)
            
            # Extract embeddings - handle different output formats
            embeddings = outputs[0]
            
            # Apply pooling if needed (for token-level outputs)
            if len(embeddings.shape) == 3:  # [batch, sequence, hidden]
                # Mean pooling over sequence dimension
                attention_mask = encoded["attention_mask"]
                mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
                sum_embeddings = np.sum(embeddings * mask_expanded, axis=1)
                sum_mask = np.sum(mask_expanded, axis=1)
                embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)
            
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        result = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        
        # Return single embedding if input was single text
        if len(texts) == 1:
            return result[0]
        
        return result
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get the embedding dimension size."""
        return self.embedding_dim