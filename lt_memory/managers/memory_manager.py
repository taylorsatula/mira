"""
Central memory manager orchestrating all LT_Memory operations.
"""

import uuid
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool
import numpy as np

from lt_memory.models.base import (
    Base, MemoryBlock, BlockHistory, MemoryPassage, MemorySnapshot, ArchivedConversation
)
from utils.openai_embeddings import OpenAIEmbeddingModel
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
    
    def __init__(self, config, llm_provider=None):
        """
        Initialize memory manager with configuration.
        
        Args:
            config: Application configuration object
            llm_provider: LLM provider for text generation (optional, will create default if None)
        """
        self.config = config
        
        # Initialize LLM provider if not provided
        if llm_provider is None:
            from api.llm_provider import LLMProvider
            self.llm_provider = LLMProvider()
        else:
            self.llm_provider = llm_provider
        
        # Create database engine with connection pooling
        self.engine = self._create_engine()
        
        # Create all tables
        Base.metadata.create_all(self.engine)
        
        # Initialize embedding model (no caching - contexts are always unique)
        self.embedding_model = OpenAIEmbeddingModel(
            model="text-embedding-3-small"  # High-quality, case-sensitive embeddings
        )
        
        # Initialize vector store
        self.vector_store = PGVectorStore(
            connection_string=config.memory.database_url,
            dimension=config.memory.embedding_dim,
            pool_size=config.memory.db_pool_size
        )
        
        # Initialize summarization engine
        self.summarization_engine = SummarizationEngine(
            llm_provider=self.llm_provider
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
        
        # Create engine with connection pooling and DOS protection
        engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=self.config.memory.db_pool_size,
            max_overflow=self.config.memory.db_pool_max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections after 1 hour
            pool_timeout=10,     # Max seconds to get connection from pool
            connect_args={
                "connect_timeout": 5,       # Max seconds to establish connection
            },
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
        Get a new database session with DOS protection.
        
        Returns:
            SQLAlchemy session object with query timeouts
        """
        session = Session(self.engine)
        
        # Set statement timeout to prevent DOS attacks
        # Any single query taking longer than 10 seconds will be killed
        session.execute(text("SET statement_timeout = '10s'"))
        
        return session
    
    async def generate_embedding_async(self, text: str) -> np.ndarray:
        """
        Generate embedding asynchronously (no caching - contexts are always unique).
        
        NOTE: This async version is not currently used. The synchronous generate_embedding()
        is used instead to ensure memory retrieval completes before LLM response generation,
        which is critical for safety (e.g., allergy information must be available before
        suggesting foods).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        with error_context("memory_manager", "generate_embedding_async", ToolError, ErrorCode.MEMORY_EMBEDDING_ERROR):
            import asyncio
            # Run the blocking embedding call in a thread pool
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.embedding_model.encode, text
            )
            return embedding

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding synchronously (for automations and non-async contexts).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        with error_context("memory_manager", "generate_embedding", ToolError, ErrorCode.MEMORY_EMBEDDING_ERROR):
            return self.embedding_model.encode(text)
    
    def _is_valid_embedding(self, embedding) -> bool:
        """
        Validate that an embedding is valid and not corrupted.
        
        Args:
            embedding: Embedding to validate
            
        Returns:
            True if embedding is valid
        """
        try:
            # Check if it's a numpy array
            if not isinstance(embedding, np.ndarray):
                return False
            
            # Check dimensions
            if embedding.shape != (self.config.memory.embedding_dim,):
                return False
            
            # Check data type
            if embedding.dtype != np.float32:
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                return False
            
            return True
            
        except Exception:
            # Any exception during validation means it's invalid
            return False
    
    def create_snapshot(self, conversation_id: str, reason: str = "checkpoint") -> str:
        """
        Create a memory block snapshot for conversation-level recovery.
        
        Snapshots capture the current state of core memory blocks (persona, human, system)
        for quick restoration during conversation recovery. Passages are not included
        as they are archival data better handled by database backups.
        
        Args:
            conversation_id: ID of the conversation triggering snapshot
            reason: Reason for creating snapshot
            
        Returns:
            Snapshot ID
        """
        # Validate input parameters
        if conversation_id is None:
            raise ValueError("conversation_id cannot be None")
        if not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a string")
        if conversation_id.strip() == "":
            raise ValueError("conversation_id cannot be empty")
        if reason is None:
            raise ValueError("reason cannot be None")
        if not isinstance(reason, str):
            raise ValueError("reason must be a string")
        
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
                    "recent_passages_24h": recent_passages
                }
            )
            session.add(snapshot)
            session.commit()
            
            logger.info(f"Created memory snapshot: {snapshot.id} (reason: {reason})")
            return str(snapshot.id)
    
    def restore_from_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore memory blocks from a snapshot.
        
        Restores core memory blocks (persona, human, system) to their state
        at snapshot time. Memory passages are not restored as they are
        archival data that should be recovered via database backups.
        
        Args:
            snapshot_id: ID of snapshot to restore from
            
        Returns:
            True if successful
            
        Raises:
            ToolError: If snapshot is corrupted or restoration fails
        """
        with error_context("memory_manager", "restore_from_snapshot", ToolError, ErrorCode.MEMORY_ERROR):
            with self.get_session() as session:
                snapshot = session.query(MemorySnapshot).filter_by(
                    id=snapshot_id
                ).first()
                
                if not snapshot:
                    logger.error(f"Snapshot {snapshot_id} not found")
                    return False
                
                # Validate snapshot data structure before attempting restore
                try:
                    self._validate_snapshot_structure(snapshot.blocks_snapshot)
                except Exception as e:
                    raise ToolError(
                        f"Snapshot {snapshot_id} is corrupted: {str(e)}",
                        ErrorCode.MEMORY_ERROR,
                        {"snapshot_id": snapshot_id, "corruption_details": str(e)}
                    )
                
                # Restore core memory blocks
                for label, block_data in snapshot.blocks_snapshot.items():
                    block = session.query(MemoryBlock).filter_by(label=label).first()
                    if block:
                        # Create history entry for the restoration
                        history = BlockHistory(
                            block_id=block.id,
                            label=label,
                            version=block.version + 1,
                            diff_data={
                                "operation_type": "restore",
                                "old_value": block.value,
                                "new_value": block_data["value"],
                                "snapshot_id": snapshot_id
                            },
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
    
    def _validate_snapshot_structure(self, blocks_snapshot: Dict[str, Any]) -> None:
        """
        Validate that snapshot data has the expected structure.
        
        Args:
            blocks_snapshot: The blocks snapshot data to validate
            
        Raises:
            ValueError: If snapshot structure is invalid
        """
        if not isinstance(blocks_snapshot, dict):
            raise ValueError("blocks_snapshot must be a dictionary")
        
        for label, block_data in blocks_snapshot.items():
            if not isinstance(block_data, dict):
                raise ValueError(f"Block data for '{label}' must be a dictionary")
            
            # Check required fields
            required_fields = ["value", "version", "updated_at"]
            missing_fields = [field for field in required_fields if field not in block_data]
            if missing_fields:
                raise ValueError(f"Block '{label}' missing required fields: {missing_fields}")
            
            # Validate field types
            if not isinstance(block_data["value"], str):
                raise ValueError(f"Block '{label}' value must be a string")
            if not isinstance(block_data["version"], int):
                raise ValueError(f"Block '{label}' version must be an integer")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with error_context("memory_manager", "get_memory_stats", ToolError, ErrorCode.MEMORY_ERROR):
            try:
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
                                func.avg(MemoryPassage.importance_score)
                            ).scalar() or 0
                        },
                        "snapshots": {
                            "count": session.query(MemorySnapshot).count()
                        }
                    }
                    
                    return stats
                    
            except Exception as e:
                # Database is unavailable - return partial stats with cache data only
                logger.warning(f"Database unavailable for memory stats: {str(e)}")
                return {
                    "blocks": {"count": 0, "total_characters": 0},
                    "passages": {"count": 0, "avg_importance": 0},
                    "snapshots": {"count": 0},
                    "database_error": "Database unavailable - showing partial stats"
                }
    
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