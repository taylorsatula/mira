"""
Database models for LT_Memory using PostgreSQL with pgvector.

All models use UUID primary keys and JSONB for metadata storage.
Timestamps are timezone-aware and default to UTC.
"""

from datetime import datetime, UTC
from sqlalchemy import Column, String, Integer, Float, DateTime, Index
from sqlalchemy import text as sql_text  # Rename to avoid shadowing by column names
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class MemoryBlock(Base):
    """
    Core memory block with versioning support.
    
    These blocks form the always-visible context that can be self-edited
    by the system through memory functions.
    """
    __tablename__ = 'memory_blocks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()"))
    label = Column(String, nullable=False, index=True)
    value = Column(String, nullable=False)
    character_limit = Column(Integer, default=2048)
    version = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    updated_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    context = Column(JSONB, default=dict)
    
    __table_args__ = (
        Index('idx_block_label_version', 'label', 'version'),
        Index('idx_block_updated', 'updated_at'),
    )


class BlockHistory(Base):
    """
    History tracking for memory blocks.
    
    Maintains a complete audit trail of all changes to memory blocks,
    enabling recovery and analysis of memory evolution.
    """
    __tablename__ = 'block_history'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()"))
    block_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    label = Column(String, nullable=False)
    old_value = Column(String)
    new_value = Column(String)
    operation = Column(String)  # append, replace, insert, rethink
    actor = Column(String)  # system, user, automation
    created_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    
    __table_args__ = (
        Index('idx_history_block_created', 'block_id', 'created_at'),
    )


class MemoryPassage(Base):
    """
    Archival memory with vector embeddings.
    
    Stores searchable long-term memories with embeddings for semantic search.
    Uses pgvector for efficient similarity operations.
    """
    __tablename__ = 'memory_passages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()"))
    text = Column(String, nullable=False)
    embedding = Column(Vector(384))  # Dimension matches all-MiniLM-L6-v2
    source = Column(String, nullable=False)  # conversation, document, automation
    source_id = Column(String, index=True)
    importance_score = Column(Float, default=0.5)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    context = Column(JSONB, default=dict)
    
    __table_args__ = (
        # IVFFlat index for similarity search
        Index('idx_passage_embedding', 'embedding', postgresql_using='ivfflat', postgresql_with={'lists': 100}),
        Index('idx_passage_importance', 'importance_score'),
        Index('idx_passage_created', 'created_at'),
        Index('idx_passage_source', 'source', 'source_id'),
    )


class MemoryEntity(Base):
    """
    Knowledge graph entities.
    
    Represents identified entities (people, places, concepts) extracted
    from conversations and documents.
    """
    __tablename__ = 'memory_entities'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()"))
    name = Column(String, nullable=False, index=True)
    entity_type = Column(String)  # person, place, organization, concept, etc.
    attributes = Column(JSONB, default=dict)
    importance_score = Column(Float, default=0.5)
    first_seen = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    last_seen = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    mention_count = Column(Integer, default=1)
    
    __table_args__ = (
        Index('idx_entity_type_importance', 'entity_type', 'importance_score'),
        Index('idx_entity_last_seen', 'last_seen'),
    )


class MemoryRelation(Base):
    """
    Relationships between entities.
    
    Forms the edges of the knowledge graph, connecting entities
    with typed relationships.
    """
    __tablename__ = 'memory_relations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()"))
    subject_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    predicate = Column(String, nullable=False)  # knows, works_at, located_in, etc.
    object_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    confidence = Column(Float, default=1.0)
    evidence_ids = Column(JSONB, default=list)  # passage IDs supporting this relation
    created_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    
    __table_args__ = (
        Index('idx_relation_subject_predicate', 'subject_id', 'predicate'),
        Index('idx_relation_object', 'object_id'),
    )


class MemorySnapshot(Base):
    """
    Point-in-time memory state snapshots.
    
    Enables recovery and auditing by capturing memory state
    at important moments.
    """
    __tablename__ = 'memory_snapshots'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()"))
    conversation_id = Column(String, index=True)
    blocks_snapshot = Column(JSONB)  # Core memory state
    entity_count = Column(Integer)
    passage_count = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    reason = Column(String)  # checkpoint, error_recovery, consolidation, etc.
    context = Column(JSONB, default=dict)