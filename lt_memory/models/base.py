"""
Database models for LT_Memory using PostgreSQL with pgvector.

All models use UUID primary keys and JSONB for metadata storage.
Timestamps are timezone-aware and default to UTC.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Index, Boolean
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
    user_id = Column(String, nullable=False, index=True)  # Multi-user support
    label = Column(String, nullable=False, index=True)
    value = Column(String, nullable=False)
    character_limit = Column(Integer, default=2048)
    version = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    updated_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    context = Column(JSONB, default=dict)
    
    __table_args__ = (
        Index('idx_block_user_label_version', 'user_id', 'label', 'version'),
        Index('idx_block_user_updated', 'user_id', 'updated_at'),
    )


class BlockHistory(Base):
    """
    History tracking for memory blocks using differential versioning.
    
    Stores only the changes (diffs) rather than full content copies,
    enabling efficient storage while maintaining complete audit trail.
    
    Storage format:
    - Version 1: Full content stored in diff_data['content']
    - Later versions: Only diffs stored in diff_data['operations']
    """
    __tablename__ = 'block_history'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()"))
    user_id = Column(String, nullable=False, index=True)  # Multi-user support
    block_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    label = Column(String, nullable=False)
    version = Column(Integer, nullable=False)  # Version this diff represents
    diff_data = Column(JSONB, nullable=False)  # Stores diff operations or base content
    operation = Column(String)  # append, replace, insert, rethink, base
    actor = Column(String)  # system, user, automation
    created_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    
    __table_args__ = (
        Index('idx_history_user_block_version', 'user_id', 'block_id', 'version'),
        Index('idx_history_user_block_created', 'user_id', 'block_id', 'created_at'),
    )


class MemoryPassage(Base):
    """
    Archival memory with vector embeddings.
    
    Stores searchable long-term memories with embeddings for semantic search.
    Uses pgvector for efficient similarity operations.
    """
    __tablename__ = 'memory_passages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()"))
    user_id = Column(String, nullable=False, index=True)  # Multi-user support
    text = Column(String, nullable=False)
    embedding = Column(Vector(1024))  # Dimension matches OpenAI text-embedding-3-small
    source = Column(String, nullable=False)  # conversation, document, automation
    source_id = Column(String, index=True)
    importance_score = Column(Float, default=0.5)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    expires_on = Column(DateTime(timezone=True), nullable=True)  # NULL means permanent fact
    human_verified = Column(Boolean, default=False)
    context = Column(JSONB, default=dict)
    
    __table_args__ = (
        # IVFFlat index for similarity search with user isolation
        Index('idx_passage_user_embedding', 'user_id', 'embedding', postgresql_using='ivfflat', postgresql_with={'lists': 100}),
        Index('idx_passage_user_importance', 'user_id', 'importance_score'),
        Index('idx_passage_user_created', 'user_id', 'created_at'),
        Index('idx_passage_user_source', 'user_id', 'source', 'source_id'),
        Index('idx_passage_user_expires', 'user_id', 'expires_on'),  # For expiration queries
    )




class ArchivedConversation(Base):
    """
    Archived conversation data with temporal indexing.
    
    Stores complete conversation history by date for retrieval
    and progressive summarization.
    """
    __tablename__ = 'archived_conversations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()"))
    user_id = Column(String, nullable=False, index=True)  # Multi-user support
    conversation_date = Column(DateTime(timezone=True), nullable=False, index=True)
    messages = Column(JSONB, nullable=False)  # Full message array
    message_count = Column(Integer, nullable=False)
    summary = Column(String, nullable=False)  # Pre-generated daily summary
    weekly_summary = Column(String, nullable=True)  # Rolling 7-day summary (created nightly)
    monthly_summary = Column(String, nullable=True)  # Monthly summary (created on 1st of month)
    archived_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    conversation_metadata = Column(JSONB, default=dict)  # Additional context
    
    __table_args__ = (
        Index('idx_archived_user_conversation_date', 'user_id', 'conversation_date'),
        Index('idx_archived_user_archived_at', 'user_id', 'archived_at'),
    )


class MemorySnapshot(Base):
    """
    Point-in-time memory state snapshots.
    
    Enables recovery and auditing by capturing memory state
    at important moments.
    """
    __tablename__ = 'memory_snapshots'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()"))
    user_id = Column(String, nullable=False, index=True)  # Multi-user support
    conversation_id = Column(String, index=True)
    blocks_snapshot = Column(JSONB)  # Core memory state
    passage_count = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    reason = Column(String)  # checkpoint, error_recovery, consolidation, etc.
    context = Column(JSONB, default=dict)
    
    __table_args__ = (
        Index('idx_snapshot_user_conversation', 'user_id', 'conversation_id'),
        Index('idx_snapshot_user_created', 'user_id', 'created_at'),
    )


class Customer(Base):
    """
    Customer model for storing customer data in PostgreSQL with multi-user support.
    
    Maps to the 'customers' table with columns that match Square's customer structure
    plus additional fields for geocoding and metadata.
    """
    __tablename__ = 'customers'
    
    # Primary key
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)  # Multi-user support
    
    # Basic contact info
    given_name = Column(String)
    family_name = Column(String)
    company_name = Column(String)
    email_address = Column(String)
    phone_number = Column(String)
    
    # Address fields
    address_line1 = Column(String)
    address_line2 = Column(String)
    city = Column(String)
    state = Column(String)
    postal_code = Column(String)
    country = Column(String)
    
    # Geocoding data
    latitude = Column(Float)
    longitude = Column(Float)
    geocoded_at = Column(DateTime(timezone=True))
    
    # Additional data stored as JSON
    additional_data = Column(JSONB)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    updated_at = Column(DateTime(timezone=True), server_default=sql_text("CURRENT_TIMESTAMP"))
    
    __table_args__ = (
        Index('idx_customer_user_email', 'user_id', 'email_address'),
        Index('idx_customer_user_phone', 'user_id', 'phone_number'),
        Index('idx_customer_user_name', 'user_id', 'given_name', 'family_name'),
        Index('idx_customer_user_location', 'user_id', 'latitude', 'longitude'),
        Index('idx_customer_user_created', 'user_id', 'created_at'),
    )