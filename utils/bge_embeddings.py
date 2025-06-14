"""
BGE (BAAI General Embedding) model implementation with ONNX Runtime and INT8 quantization.

This module provides optimized BGE model inference using ONNX Runtime with support
for INT8 quantization on both CPU and GPU.
"""
import logging
import os
from typing import List, Union, Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import json
import requests
from tqdm import tqdm

from errors import APIError, ErrorCode, error_context


class BaseONNXModel:
    """Base class for ONNX models with common functionality."""
    
    def __init__(self, model_name: str, cache_dir: str, thread_limit: int):
        """Initialize base ONNX model."""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.thread_limit = thread_limit
        self.session = None
        self.tokenizer = None
    
    def _load_tokenizer(self):
        """Load tokenizer from local path or HuggingFace."""
        from transformers import AutoTokenizer
        
        tokenizer_path = os.path.dirname(self.model_path)
        tokenizer_config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
        
        if os.path.exists(tokenizer_config_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            # Save tokenizer for future use
            self.tokenizer.save_pretrained(tokenizer_path)
    
    def _create_onnx_session(self):
        """Create ONNX Runtime session with optimized settings."""
        import onnxruntime as ort
        
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.thread_limit
        sess_options.inter_op_num_threads = 1
        
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
    
    def _download_file(self, url: str, dest_path: str):
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


class BGEEmbeddingModel(BaseONNXModel):
    """
    BGE-large-en-v1.5 embedding model with ONNX Runtime and INT8 quantization support.
    
    This model produces 1024-dimensional embeddings optimized for both
    retrieval and similarity tasks.
    """
    
    def __init__(self,
                 model_name: str = "BAAI/bge-large-en-v1.5",
                 model_path: Optional[str] = None,
                 use_int8: bool = True,
                 cache_dir: Optional[str] = None,
                 thread_limit: Optional[int] = 4):
        """Initialize BGE model with ONNX Runtime.
        
        Args:
            model_name: HuggingFace model name (for downloading)
            model_path: Path to local ONNX model file
            use_int8: Whether to use INT8 quantization
            cache_dir: Directory to cache downloaded models
            thread_limit: Limit CPU threads for inference
        """
        self.logger = logging.getLogger("bge_embeddings")
        cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "bge_models")
        super().__init__(model_name, cache_dir, thread_limit)
        
        self.use_int8 = use_int8
        
        # Model paths
        if model_path:
            self.model_path = model_path
        else:
            # Determine model file name based on quantization
            model_file = "model_int8.onnx" if use_int8 else "model.onnx"
            self.model_path = os.path.join(self.cache_dir, model_name.replace("/", "_"), model_file)
        
        self.logger.info(f"Initializing BGE model with ONNX Runtime (INT8={use_int8})")
        self._load_model()
    
    def _load_model(self):
        """Load BGE model with ONNX Runtime."""
        try:
            import onnxruntime as ort
            
            # Download/convert model if not exists
            if not os.path.exists(self.model_path):
                self._convert_to_onnx()
            
            # Load tokenizer
            self._load_tokenizer()
            
            # Create ONNX session
            self._create_onnx_session()
            
            self.logger.info(f"Successfully loaded BGE ONNX model from {self.model_path}")
            
        except ImportError as e:
            missing_package = "onnxruntime"
            if "transformers" in str(e):
                missing_package = "transformers"
            raise APIError(
                f"Required package '{missing_package}' not installed. "
                f"Run: pip install {missing_package}",
                ErrorCode.DEPENDENCY_ERROR
            )
        except Exception as e:
            raise APIError(
                f"Failed to load BGE ONNX model: {str(e)}",
                ErrorCode.INITIALIZATION_ERROR
            )
    
    def _convert_to_onnx(self):
        """Convert PyTorch model to ONNX with optional INT8 quantization."""
        try:
            from transformers import AutoModel
            from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            import torch
            
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Load PyTorch model
            self.logger.info("Loading PyTorch model for conversion...")
            
            # Convert to ONNX using Optimum
            temp_dir = os.path.dirname(self.model_path)
            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                self.model_name,
                export=True,
                cache_dir=self.cache_dir
            )
            
            # Save initial ONNX model
            ort_model.save_pretrained(temp_dir)
            
            if self.use_int8:
                # Quantize to INT8
                self.logger.info("Quantizing model to INT8...")
                quantizer = ORTQuantizer.from_pretrained(temp_dir)
                
                # Configure dynamic quantization for CPU
                qconfig = AutoQuantizationConfig.avx512_vnni(
                    is_static=False,
                    per_channel=True
                )
                
                # Perform quantization
                quantizer.quantize(
                    save_dir=temp_dir,
                    quantization_config=qconfig
                )
                
                # Rename quantized model to expected name
                quantized_path = os.path.join(temp_dir, "model_quantized.onnx")
                if os.path.exists(quantized_path):
                    os.rename(quantized_path, self.model_path)
            else:
                # Rename to expected name
                original_path = os.path.join(temp_dir, "model.onnx")
                if original_path != self.model_path and os.path.exists(original_path):
                    os.rename(original_path, self.model_path)
            
            self.logger.info(f"Successfully converted model to ONNX format at {self.model_path}")
            
        except ImportError as e:
            raise APIError(
                "Required packages for ONNX conversion not installed. "
                "Run: pip install optimum[onnxruntime] torch transformers",
                ErrorCode.DEPENDENCY_ERROR
            )
        except Exception as e:
            raise APIError(
                f"Failed to convert model to ONNX: {str(e)}",
                ErrorCode.INITIALIZATION_ERROR
            )
    
    def encode(self,
               texts: Union[str, List[str]],
               batch_size: int = 32,
               normalize: bool = True,
               show_progress: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings using ONNX Runtime.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            Embeddings as numpy array
        """
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='np'
            )
            
            # Prepare inputs
            ort_inputs = {
                'input_ids': encoded_input['input_ids'],
                'attention_mask': encoded_input['attention_mask']
            }
            
            # Add token_type_ids if model expects it
            if 'token_type_ids' in [inp.name for inp in self.session.get_inputs()]:
                ort_inputs['token_type_ids'] = encoded_input.get(
                    'token_type_ids', 
                    np.zeros_like(encoded_input['input_ids'])
                )
            
            # Run inference
            outputs = self.session.run(None, ort_inputs)
            
            # Extract embeddings (last hidden state)
            last_hidden_state = outputs[0]
            
            # Mean pooling
            embeddings = self._mean_pooling(
                last_hidden_state,
                encoded_input['attention_mask']
            )
            
            # Normalize if requested
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-10)
            
            all_embeddings.append(embeddings)
            
            if show_progress and i + batch_size < len(texts):
                self.logger.info(f"Processed {i + batch_size}/{len(texts)} texts")
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
        # Return single embedding if single input
        if single_text:
            return embeddings[0]
        
        return embeddings
    
    def _mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Perform mean pooling on token embeddings."""
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.sum(input_mask_expanded, axis=1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return 1024  # BGE-large-en-v1.5 produces 1024-dim embeddings
    
    def precompute_embeddings(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        """
        Precompute embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (embeddings array, original texts list)
        """
        embeddings = self.encode(texts, batch_size=batch_size, normalize=True)
        return embeddings, texts


class BGEReranker(BaseONNXModel):
    """
    BGE-reranker-base model with ONNX Runtime and FP16 support for reranking search results.
    
    This model is specifically designed for reranking and provides more
    accurate relevance scores than embedding similarity alone.
    """
    
    def __init__(self,
                 model_name: str = "BAAI/bge-reranker-base",
                 model_path: Optional[str] = None,
                 use_fp16: bool = True,
                 cache_dir: Optional[str] = None,
                 thread_limit: Optional[int] = 4):
        """Initialize BGE reranker with ONNX Runtime.
        
        Args:
            model_name: HuggingFace model name (for downloading)
            model_path: Path to local ONNX model file
            use_fp16: Whether to use FP16 precision
            cache_dir: Directory to cache downloaded models
            thread_limit: Limit CPU threads for inference
        """
        self.logger = logging.getLogger("bge_reranker")
        cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "bge_models")
        super().__init__(model_name, cache_dir, thread_limit)
        
        self.use_fp16 = use_fp16
        
        # Model paths
        if model_path:
            self.model_path = model_path
        else:
            model_file = "reranker_fp16.onnx" if use_fp16 else "reranker.onnx"
            self.model_path = os.path.join(self.cache_dir, model_name.replace("/", "_"), model_file)
        
        self.logger.info(f"Initializing BGE reranker with ONNX Runtime (FP16={use_fp16})")
        self._load_model()
    
    def _load_model(self):
        """Load BGE reranker model with ONNX Runtime."""
        try:
            import onnxruntime as ort
            
            # Download/convert model if not exists
            if not os.path.exists(self.model_path):
                self._convert_to_onnx()
            
            # Load tokenizer
            self._load_tokenizer()
            
            # Create ONNX session
            self._create_onnx_session()
            
            self.logger.info(f"Successfully loaded BGE reranker ONNX model from {self.model_path}")
            
        except ImportError as e:
            missing_package = "onnxruntime"
            if "transformers" in str(e):
                missing_package = "transformers"
            raise APIError(
                f"Required package '{missing_package}' not installed. "
                f"Run: pip install {missing_package}",
                ErrorCode.DEPENDENCY_ERROR
            )
        except Exception as e:
            raise APIError(
                f"Failed to load BGE reranker ONNX model: {str(e)}",
                ErrorCode.INITIALIZATION_ERROR
            )
    
    def _convert_to_onnx(self):
        """Convert PyTorch reranker model to ONNX with optional FP16."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Load PyTorch model and tokenizer
            self.logger.info("Loading PyTorch reranker model for conversion...")
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Prepare dummy input
            dummy_input = tokenizer(
                [["query", "passage"]],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Export to ONNX
            input_names = ['input_ids', 'attention_mask']
            output_names = ['logits']
            dynamic_axes = {
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'}
            }
            
            # Add token_type_ids if present
            model_inputs = (dummy_input['input_ids'], dummy_input['attention_mask'])
            if 'token_type_ids' in dummy_input:
                model_inputs = model_inputs + (dummy_input['token_type_ids'],)
                input_names.append('token_type_ids')
                dynamic_axes['token_type_ids'] = {0: 'batch_size', 1: 'sequence'}
            
            torch.onnx.export(
                model,
                model_inputs,
                self.model_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )
            
            # Convert to FP16 if requested
            if self.use_fp16:
                self._convert_to_fp16()
            
            # Save tokenizer
            tokenizer.save_pretrained(os.path.dirname(self.model_path))
            
            self.logger.info(f"Successfully converted reranker to ONNX format at {self.model_path}")
            
        except ImportError as e:
            raise APIError(
                "Required packages for ONNX conversion not installed. "
                "Run: pip install torch transformers",
                ErrorCode.DEPENDENCY_ERROR
            )
        except Exception as e:
            raise APIError(
                f"Failed to convert reranker to ONNX: {str(e)}",
                ErrorCode.INITIALIZATION_ERROR
            )
    
    def _convert_to_fp16(self):
        """Convert ONNX model to FP16 precision."""
        try:
            import onnx
            from onnxconverter_common import float16
            
            model = onnx.load(self.model_path)
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, self.model_path)
            
            self.logger.info("Converted reranker model to FP16 precision")
            
        except ImportError:
            self.logger.warning("onnxconverter-common not installed, skipping FP16 conversion")
        except Exception as e:
            self.logger.warning(f"Failed to convert to FP16: {str(e)}")
    
    def rerank(self,
               query: str,
               passages: List[str],
               batch_size: int = 32,
               return_scores: bool = True) -> Union[List[int], List[tuple]]:
        """
        Rerank passages based on relevance to query using ONNX Runtime.
        
        Args:
            query: Query text
            passages: List of passages to rerank
            batch_size: Batch size for processing
            return_scores: Whether to return scores with indices
            
        Returns:
            If return_scores=False: List of indices sorted by relevance
            If return_scores=True: List of (index, score) tuples sorted by relevance
        """
        if not passages:
            return []
        
        all_scores = []
        
        # Process in batches
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i + batch_size]
            
            # Create query-passage pairs
            pairs = [[query, passage] for passage in batch_passages]
            
            # Tokenize
            encoded_input = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='np'
            )
            
            # Prepare inputs
            ort_inputs = {
                'input_ids': encoded_input['input_ids'],
                'attention_mask': encoded_input['attention_mask']
            }
            
            # Add token_type_ids if model expects it
            if 'token_type_ids' in [inp.name for inp in self.session.get_inputs()]:
                ort_inputs['token_type_ids'] = encoded_input.get(
                    'token_type_ids',
                    np.zeros_like(encoded_input['input_ids'])
                )
            
            # Run inference
            outputs = self.session.run(None, ort_inputs)
            
            # Get scores and apply sigmoid
            logits = outputs[0]
            scores = 1 / (1 + np.exp(-logits))  # Sigmoid
            
            # Handle both single and multi-dimensional outputs
            if len(scores.shape) > 1:
                scores = scores.squeeze(-1)
            
            all_scores.extend(scores.tolist())
        
        # Ensure all_scores is a flat list
        if isinstance(all_scores[0], list):
            all_scores = [score[0] if isinstance(score, list) else score for score in all_scores]
        
        # Create index-score pairs
        indexed_scores = [(idx, score) for idx, score in enumerate(all_scores)]
        
        # Sort by score (descending)
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        if return_scores:
            return indexed_scores
        else:
            return [idx for idx, _ in indexed_scores]
    
    def compute_relevance_scores(self,
                                 query: str,
                                 passages: List[str],
                                 batch_size: int = 32) -> np.ndarray:
        """
        Compute relevance scores for passages.
        
        Args:
            query: Query text
            passages: List of passages
            batch_size: Batch size for processing
            
        Returns:
            Array of relevance scores in original order
        """
        scores_with_indices = self.rerank(
            query, passages, batch_size=batch_size, return_scores=True
        )
        
        # Extract scores in original order
        scores = [0.0] * len(passages)
        for idx, score in scores_with_indices:
            scores[idx] = score
        
        return np.array(scores)