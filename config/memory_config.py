"""
Memory configuration for LT_Memory system.

This file contains the MemoryConfig class to be imported into config.py.
"""

import os
from typing import Dict, Optional
from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """LT_Memory configuration settings."""
    
    # PostgreSQL connection
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "LT_MEMORY_DATABASE_URL",
            "postgresql://mira:secure_password@localhost/lt_memory"
        ),
        description="PostgreSQL connection URL"
    )
    
    # Database pool settings
    db_pool_size: int = Field(
        default=10,
        description="Database connection pool size"
    )
    db_pool_max_overflow: int = Field(
        default=20,
        description="Maximum overflow connections"
    )
    
    # Core memory settings
    core_memory_blocks: Dict[str, int] = Field(
        default={"persona": 2048, "human": 2048, "system": 1024},
        description="Core memory blocks and their character limits"
    )
    
    # ONNX model settings
    onnx_model_path: str = Field(
        default_factory=lambda: os.getenv(
            "LT_MEMORY_ONNX_MODEL",
            "onnx/model.onnx"
        ),
        description="Path to ONNX model file"
    )
    onnx_tokenizer: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Tokenizer name for ONNX model"
    )
    embedding_dim: int = Field(
        default=384,
        description="Embedding dimension size"
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    
    # Search settings
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for retrieval"
    )
    max_search_results: int = Field(
        default=10,
        description="Maximum number of search results"
    )
    
    # Batch processing settings
    batch_process_hours: int = Field(
        default=1,
        description="Hours of conversations to process per batch"
    )
    
    # Consolidation settings
    consolidation_threshold: int = Field(
        default=1000,
        description="Message count before triggering consolidation"
    )
    max_memory_age_days: int = Field(
        default=90,
        description="Days before memory enters cold storage"
    )
    
    # Knowledge graph settings
    entity_extraction_enabled: bool = Field(
        default=True,
        description="Enable entity extraction and knowledge graph"
    )
    relationship_inference_enabled: bool = Field(
        default=True,
        description="Enable relationship inference between entities"
    )