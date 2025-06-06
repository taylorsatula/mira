"""
Central memory manager orchestrating all LT_Memory operations.
"""

import uuid
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool
import numpy as np

from lt_memory.models.base import (
    Base, MemoryBlock, BlockHistory, MemoryPassage, MemorySnapshot, ArchivedConversation
)
from utils.onnx_embeddings import ONNXEmbeddingModel
from lt_memory.utils.embeddings import EmbeddingCache
from lt_memory.utils.pg_vector_store import PGVectorStore
from lt_memory.utils.summarization import SummarizationEngine
from errors import error_context, ErrorCode, ToolError
from utils.timezone_utils import utc_now, format_utc_iso

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Central manager for all LT_Memory operations.
    
    Coordinates between different memory components and provides
    a unified interface for memory operations.
    """
    
    def __init__(self, config, llm_bridge=None):
        """
        Initialize memory manager with configuration.
        
        Args:
            config: Application configuration object
            llm_bridge: LLM bridge for text generation (optional)
        """
        self.config = config
        self.llm_bridge = llm_bridge
        
        # Create database engine with connection pooling
        self.engine = self._create_engine()
        
        # Create all tables
        Base.metadata.create_all(self.engine)
        
        # Initialize embedding model with caching
        self.embedding_cache = EmbeddingCache(
            config.paths.data_dir,
            memory_cache_size=1000
        )
        self.embedding_model = ONNXEmbeddingModel(
            model_path=config.memory.onnx_model_path,
            thread_limit=4  # Use default thread limit
        )
        
        # Initialize vector store
        self.vector_store = PGVectorStore(
            self.engine,
            dimension=config.memory.embedding_dim
        )
        
        # Initialize summarization engine
        self.summarization_engine = SummarizationEngine(
            llm_bridge=self.llm_bridge
        )
        
        # Import component managers here to avoid circular imports
        from lt_memory.managers.block_manager import BlockManager
        from lt_memory.managers.passage_manager import PassageManager
        from lt_memory.managers.consolidation_engine import ConsolidationEngine
        from lt_memory.managers.conversation_archive import ConversationArchive
        
        # Initialize component managers
        self.block_manager = BlockManager(self)
        self.passage_manager = PassageManager(self)
        self.consolidation_engine = ConsolidationEngine(self)
        self.conversation_archive = ConversationArchive(self)
        
        # Initialize core memory blocks if not exist
        self._initialize_core_blocks()
        
        logger.info("Memory manager initialized successfully")
    
    def _create_engine(self):
        """Create PostgreSQL engine with connection pooling."""
        # Parse database URL to ensure it's PostgreSQL
        db_url = self.config.memory.database_url
        if not db_url.startswith("postgresql://"):
            raise ValueError("LT_Memory requires PostgreSQL database")
        
        # Create engine with connection pooling
        engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=self.config.memory.db_pool_size,
            max_overflow=self.config.memory.db_pool_max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections after 1 hour
            echo=False,          # Set to True for SQL debugging
            future=True          # Use SQLAlchemy 2.0 style
        )
        
        # Ensure required extensions
        with engine.connect() as conn:
            # Create extensions if they don't exist
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            conn.commit()
            
            # Verify pgvector version
            result = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()
            
            if result:
                logger.info(f"pgvector version: {result[0]}")
            
        return engine
    
    def _initialize_core_blocks(self):
        """Initialize default core memory blocks if they don't exist."""
        with self.get_session() as session:
            for label, limit in self.config.memory.core_memory_blocks.items():
                existing = session.query(MemoryBlock).filter_by(label=label).first()
                if not existing:
                    # Create with default content based on label
                    default_content = self._get_default_block_content(label)
                    
                    block = MemoryBlock(
                        label=label,
                        value=default_content,
                        character_limit=limit
                    )
                    session.add(block)
                    logger.info(f"Created core memory block: {label}")
            
            session.commit()
    
    def _get_default_block_content(self, label: str) -> str:
        """Get default content for a memory block label."""
        defaults = {
            "persona": "I am MIRA, an AI assistant with advanced memory capabilities.",
            "human": "The human I'm conversing with hasn't shared personal details yet.",
            "system": "System status: Memory system initialized and operational."
        }
        return defaults.get(label, "")
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Returns:
            SQLAlchemy session object
        """
        return Session(self.engine)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache first
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached
        
        # Generate new embedding
        with error_context("memory_manager", "generate_embedding", ToolError, ErrorCode.MEMORY_EMBEDDING_ERROR):
            embedding = self.embedding_model.encode(text)
            
            # Cache for future use
            self.embedding_cache.set(text, embedding)
            
            return embedding
    
    def create_snapshot(self, conversation_id: str, reason: str = "checkpoint") -> str:
        """
        Create a memory snapshot for recovery.
        
        Args:
            conversation_id: ID of the conversation triggering snapshot
            reason: Reason for creating snapshot
            
        Returns:
            Snapshot ID
        """
        with self.get_session() as session:
            # Get current block states
            blocks = session.query(MemoryBlock).all()
            blocks_data = {
                b.label: {
                    "value": b.value,
                    "version": b.version,
                    "updated_at": format_utc_iso(b.updated_at)
                } 
                for b in blocks
            }
            
            # Get counts for statistics
            passage_count = session.query(MemoryPassage).count()
            
            # Get recent activity metrics
            recent_cutoff = utc_now() - timedelta(hours=24)
            recent_passages = session.query(MemoryPassage).filter(
                MemoryPassage.created_at > recent_cutoff
            ).count()
            
            # Create snapshot
            snapshot = MemorySnapshot(
                conversation_id=conversation_id,
                blocks_snapshot=blocks_data,
                passage_count=passage_count,
                reason=reason,
                context={
                    "recent_passages_24h": recent_passages,
                    "cache_stats": self.embedding_cache.get_stats()
                }
            )
            session.add(snapshot)
            session.commit()
            
            logger.info(f"Created memory snapshot: {snapshot.id} (reason: {reason})")
            return str(snapshot.id)
    
    def restore_from_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore memory state from a snapshot.
        
        Args:
            snapshot_id: ID of snapshot to restore from
            
        Returns:
            True if successful
        """
        with self.get_session() as session:
            snapshot = session.query(MemorySnapshot).filter_by(
                id=snapshot_id
            ).first()
            
            if not snapshot:
                logger.error(f"Snapshot {snapshot_id} not found")
                return False
            
            # Restore core memory blocks
            for label, block_data in snapshot.blocks_snapshot.items():
                block = session.query(MemoryBlock).filter_by(label=label).first()
                if block:
                    # Create history entry for the restoration
                    history = BlockHistory(
                        block_id=block.id,
                        label=label,
                        old_value=block.value,
                        new_value=block_data["value"],
                        operation="restore",
                        actor="system"
                    )
                    session.add(history)
                    
                    # Update block
                    block.value = block_data["value"]
                    block.version += 1
                    block.updated_at = utc_now()
            
            session.commit()
            logger.info(f"Restored from snapshot {snapshot_id}")
            return True
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self.get_session() as session:
            stats = {
                "blocks": {
                    "count": session.query(MemoryBlock).count(),
                    "total_characters": sum(
                        len(b.value) for b in session.query(MemoryBlock).all()
                    )
                },
                "passages": {
                    "count": session.query(MemoryPassage).count(),
                    "avg_importance": session.query(
                        MemoryPassage.importance_score
                    ).scalar() or 0
                },
                "snapshots": {
                    "count": session.query(MemorySnapshot).count()
                },
                "embedding_cache": self.embedding_cache.get_stats()
            }
            
            
            return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on memory system."""
        health = {
            "status": "healthy",
            "components": {},
            "issues": []
        }
        
        # Check database connection
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            health["components"]["database"] = "ok"
        except Exception as e:
            health["components"]["database"] = "error"
            health["issues"].append(f"Database: {str(e)}")
            health["status"] = "unhealthy"
        
        # Check embedding model
        try:
            test_embedding = self.generate_embedding("test")
            if test_embedding.shape[0] == self.config.memory.embedding_dim:
                health["components"]["embedding_model"] = "ok"
            else:
                health["components"]["embedding_model"] = "dimension_mismatch"
                health["issues"].append("Embedding dimension mismatch")
                health["status"] = "degraded"
        except Exception as e:
            health["components"]["embedding_model"] = "error"
            health["issues"].append(f"Embedding model: {str(e)}")
            health["status"] = "unhealthy"
        
        # Check vector store
        try:
            # Try a dummy search
            test_vector = np.random.rand(self.config.memory.embedding_dim)
            self.vector_store.search(test_vector, k=1)
            health["components"]["vector_store"] = "ok"
        except Exception as e:
            health["components"]["vector_store"] = "error"
            health["issues"].append(f"Vector store: {str(e)}")
            health["status"] = "degraded"
        
        # Add statistics
        health["stats"] = self.get_memory_stats()
        
        return health