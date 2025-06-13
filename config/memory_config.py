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
    
    # Embedding settings
    embedding_dim: int = Field(
        default=1024,
        description="Embedding dimension size (1024 for OpenAI text-embedding-3-small)"
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
    
    # Fact extraction settings
    fact_similarity_threshold: float = Field(
        default=0.85,
        description="Similarity threshold for fact deduplication"
    )
    auto_expire_enabled: bool = Field(
        default=True,
        description="Enable automatic expiration of old facts"
    )
    fact_extraction_batch_size: int = Field(
        default=10,
        description="Number of facts to extract per LLM call"
    )
    
