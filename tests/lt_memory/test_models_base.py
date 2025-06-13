"""
Tests for lt_memory/models/base.py - Database models with PostgreSQL and pgvector.

Tests the core contracts that memory system depends on for persistence and retrieval.
"""

import pytest
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from lt_memory.models.base import Base, MemoryBlock, MemoryPassage, BlockHistory, ArchivedConversation


@pytest.fixture(scope="session")
def test_database_engine():
    """Real PostgreSQL test database - same engine as production."""
    url = "postgresql://mira_app@localhost/lt_memory_test"
    engine = create_engine(url)
    
    # Create all tables (skip if they exist)
    Base.metadata.create_all(engine, checkfirst=True)
    
    yield engine
    
    # Skip cleanup to avoid permissions issues - test data will be cleaned per test


@pytest.fixture
def db_session(test_database_engine):
    """Clean database session for each test."""
    Session = sessionmaker(bind=test_database_engine)
    session = Session()
    
    yield session
    
    # Clean up test data
    session.rollback()
    # Truncate all tables to clean state
    session.execute(text("TRUNCATE memory_blocks, block_history, memory_passages, archived_conversations, memory_snapshots RESTART IDENTITY CASCADE"))
    session.commit()
    session.close()


class TestMemoryBlock:
    """Test MemoryBlock creation and retrieval - the core memory contract."""
    
    def test_memory_block_creation_and_retrieval(self, db_session):
        """
        Test that MemoryBlock can be created and retrieved correctly.
        
        REAL BUG THIS CATCHES: If the UUID generation, JSONB storage, or 
        timezone-aware timestamps fail, core memory blocks can't be saved 
        or retrieved, breaking the entire memory system that depends on 
        always-visible context.
        """
        # Create a memory block with all core fields
        block = MemoryBlock(
            label="user_preferences",
            value="User prefers concise responses and technical details",
            character_limit=1024,
            context={"source": "conversation", "importance": "high"}
        )
        
        # Save to real database
        db_session.add(block)
        db_session.commit()
        
        # Verify the contract promises are kept
        assert block.id is not None  # UUID was generated
        assert isinstance(block.created_at, datetime)  # Timestamp created
        assert block.created_at.tzinfo is not None  # Timezone-aware
        assert block.updated_at.tzinfo is not None  # Timezone-aware
        assert block.version == 1  # Default version set
        
        # Test retrieval contract - can we get it back?
        retrieved_block = db_session.query(MemoryBlock).filter_by(id=block.id).first()
        
        assert retrieved_block is not None
        assert retrieved_block.label == "user_preferences"
        assert retrieved_block.value == "User prefers concise responses and technical details"
        assert retrieved_block.character_limit == 1024
        assert retrieved_block.context == {"source": "conversation", "importance": "high"}
        assert retrieved_block.version == 1


class TestMemoryPassage:
    """Test MemoryPassage vector embedding storage - the semantic search contract."""
    
    def test_memory_passage_vector_storage_and_retrieval(self, db_session):
        """
        Test that MemoryPassage can store and retrieve vector embeddings correctly.
        
        REAL BUG THIS CATCHES: If pgvector integration fails or Vector(1024) column
        has issues with numpy array storage/retrieval, semantic search completely breaks
        because embeddings can't be saved or compared, making memory search unusable.
        """
        # Create a real 1024-dimensional embedding (matching OpenAI text-embedding-3-small)
        embedding_vector = np.random.rand(1024).astype(np.float32)
        
        passage = MemoryPassage(
            text="User mentioned they work in software engineering at a startup",
            embedding=embedding_vector,
            source="conversation",
            source_id="conv_123",
            importance_score=0.8,
            context={"topic": "work", "session": "2024-01-01"}
        )
        
        # Save to real PostgreSQL with pgvector
        db_session.add(passage)
        db_session.commit()
        
        # Verify core contracts
        assert passage.id is not None
        assert isinstance(passage.created_at, datetime)
        assert passage.access_count == 0  # Default value
        
        # Test vector storage contract - can we retrieve the exact embedding?
        retrieved_passage = db_session.query(MemoryPassage).filter_by(id=passage.id).first()
        
        assert retrieved_passage is not None
        assert retrieved_passage.text == "User mentioned they work in software engineering at a startup"
        assert retrieved_passage.source == "conversation"
        assert retrieved_passage.importance_score == 0.8
        
        # Critical: verify vector embedding was stored and retrieved correctly
        assert retrieved_passage.embedding is not None
        assert len(retrieved_passage.embedding) == 1024
        
        # Verify the embedding data integrity (numpy array comparison)
        np.testing.assert_array_almost_equal(
            retrieved_passage.embedding, 
            embedding_vector,
            decimal=6
        )


class TestBlockHistory:
    """Test BlockHistory differential versioning - the audit trail contract."""
    
    def test_block_history_diff_storage_and_versioning(self, db_session):
        """
        Test that BlockHistory can store version diffs and track changes correctly.
        
        REAL BUG THIS CATCHES: If JSONB diff_data storage fails or version tracking
        has bugs, memory block history gets lost or corrupted, breaking the entire
        audit trail and version recovery system that users depend on for rollbacks.
        """
        block_id = "550e8400-e29b-41d4-a716-446655440000"  # Valid UUID
        
        # Version 1: Base content (full content stored)
        version_1 = BlockHistory(
            block_id=block_id,
            label="user_preferences",
            version=1,
            diff_data={"content": "User prefers technical responses"},
            operation="base",
            actor="system"
        )
        
        # Version 2: Update (diff stored)
        version_2 = BlockHistory(
            block_id=block_id,
            label="user_preferences", 
            version=2,
            diff_data={
                "operations": [
                    {"op": "replace", "path": "/content", "value": "User prefers technical responses with examples"}
                ]
            },
            operation="replace",
            actor="user"
        )
        
        # Save both versions to real database
        db_session.add(version_1)
        db_session.add(version_2)
        db_session.commit()
        
        # Verify core contracts
        assert version_1.id is not None
        assert version_2.id is not None
        assert isinstance(version_1.created_at, datetime)
        assert isinstance(version_2.created_at, datetime)
        
        # Test version tracking contract - can we retrieve versions by block_id?
        history_records = db_session.query(BlockHistory).filter_by(block_id=block_id).order_by(BlockHistory.version).all()
        
        assert len(history_records) == 2
        
        # Verify version 1 (base content)
        v1 = history_records[0]
        assert v1.version == 1
        assert v1.operation == "base"
        assert v1.actor == "system"
        assert v1.diff_data["content"] == "User prefers technical responses"
        
        # Verify version 2 (diff operations) 
        v2 = history_records[1]
        assert v2.version == 2
        assert v2.operation == "replace"
        assert v2.actor == "user"
        assert "operations" in v2.diff_data
        assert v2.diff_data["operations"][0]["op"] == "replace"
        assert "examples" in v2.diff_data["operations"][0]["value"]


class TestConcurrentOperations:
    """Test concurrent database operations - the multi-user safety contract."""
    
    def test_memory_block_survives_extreme_concurrent_load(self, test_database_engine):
        """
        Test that MemoryBlocks can handle massive concurrent load without any corruption.
        
        REAL BUG THIS CATCHES: If PostgreSQL UUID generation, JSONB serialization, or 
        transaction handling has ANY race conditions, concurrent users will cause deadlocks,
        duplicate UUIDs, or data corruption. Since EVERY memory operation goes through base.py,
        even microscopic race conditions will break the entire system under real load.
        """
        def extreme_load_worker(worker_id):
            """Worker function that hammers the database with rapid operations."""
            Session = sessionmaker(bind=test_database_engine)
            session = Session()
            
            created_blocks = []
            
            try:
                # Each worker creates 20 blocks as fast as possible
                for block_num in range(20):  # 100 workers × 20 blocks = 2000 operations
                    # Complex JSONB data to stress serialization
                    complex_context = {
                        "worker_id": worker_id,
                        "block_num": block_num,
                        "load_test": True,
                        "timestamp": str(datetime.now()),
                        "unique_signature": f"LOAD_w{worker_id}_b{block_num}_{worker_id * 10000 + block_num}",
                        "nested_data": {
                            "level1": {"level2": {"level3": f"deep_data_{worker_id}_{block_num}"}},
                            "array_data": [worker_id, block_num, worker_id + block_num],
                            "boolean_flags": {"flag1": True, "flag2": False, "flag3": worker_id % 2 == 0}
                        },
                        "large_text": f"STRESS_TEST_DATA_" * 50 + f"_WORKER_{worker_id}_BLOCK_{block_num}"
                    }
                    
                    block = MemoryBlock(
                        label=f"LOAD_worker_{worker_id}_block_{block_num}", 
                        value=f"LOAD WORKER {worker_id} BLOCK {block_num}: " + "X" * 500,  # Large text
                        character_limit=4096,
                        version=1,
                        context=complex_context
                    )
                    
                    session.add(block)
                    session.commit()
                    
                    # Immediately verify - any corruption fails the test
                    retrieved = session.query(MemoryBlock).filter_by(id=block.id).first()
                    if not retrieved or retrieved.context.get("unique_signature") != complex_context["unique_signature"]:
                        return {"success": False, "error": f"CORRUPTION_w{worker_id}_b{block_num}"}
                    
                    created_blocks.append({
                        "id": str(block.id),
                        "worker_id": worker_id,
                        "block_num": block_num,
                        "signature": complex_context["unique_signature"],
                        "deep_data": complex_context["nested_data"]["level1"]["level2"]["level3"]
                    })
                
                return {"success": True, "blocks": created_blocks}
                    
            except Exception as e:
                session.rollback()
                return {"success": False, "error": str(e)}
            finally:
                session.close()
        
        # Launch 100 workers × 20 blocks = 2000 concurrent operations
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(extreme_load_worker, i) for i in range(100)]
            results = [f.result() for f in futures]
        
        # Count successes and failures
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        # ZERO TOLERANCE: ALL must succeed
        assert len(successful_results) == 100, f"Load test failed: {failed_results}"
        
        # Collect ALL 2000 blocks
        all_blocks = []
        for result in successful_results:
            all_blocks.extend(result["blocks"])
        
        assert len(all_blocks) == 2000, f"Missing blocks: Expected 2000, got {len(all_blocks)}"
        
        # Verify EVERY SINGLE operation was unique and perfect
        unique_ids = set(block["id"] for block in all_blocks)
        assert len(unique_ids) == 2000, f"UUID conflicts detected: {2000 - len(unique_ids)} duplicates"
        
        unique_signatures = set(block["signature"] for block in all_blocks)
        assert len(unique_signatures) == 2000, f"Data conflicts detected: {2000 - len(unique_signatures)} duplicates"
        
        # Verify EVERY block exists in database and is perfect
        Session = sessionmaker(bind=test_database_engine)
        session = Session()
        try:
            verified_count = 0
            for expected_block in all_blocks:
                db_block = session.query(MemoryBlock).filter_by(id=expected_block["id"]).first()
                assert db_block is not None, f"Missing block: {expected_block['id']}"
                
                # Verify complex JSONB data survived
                assert db_block.context["worker_id"] == expected_block["worker_id"]
                assert db_block.context["unique_signature"] == expected_block["signature"]
                assert db_block.context["nested_data"]["level1"]["level2"]["level3"] == expected_block["deep_data"]
                
                # Verify large text survived
                assert f"LOAD WORKER {expected_block['worker_id']} BLOCK {expected_block['block_num']}" in db_block.value
                assert "X" * 500 in db_block.value
                
                verified_count += 1
                
            # Final count - no extra blocks
            total_blocks = session.query(MemoryBlock).count()
            assert total_blocks == 2000, f"Unexpected block count: Found {total_blocks}, expected 2000"
            
        finally:
            session.close()


class TestEdgeCases:
    """Test edge cases that could cause production failures."""
    
    def test_massive_jsonb_storage_limits(self, db_session):
        """
        Test that PostgreSQL can handle realistically large JSONB objects.
        
        REAL BUG THIS CATCHES: If PostgreSQL hits JSONB size limits or performance 
        degrades severely with large JSON, the system fails when users have very long
        conversation histories, complex automation contexts, or large document summaries.
        Production conversations can easily reach thousands of messages.
        """
        # Generate a massive but realistic conversation history
        # Simulating 2000 messages (realistic for power users over months)
        massive_conversation = []
        
        for msg_id in range(2000):
            # Each message has realistic structure and size
            message = {
                "id": f"msg_{msg_id}",
                "timestamp": f"2024-01-{(msg_id % 30) + 1:02d}T{(msg_id % 24):02d}:00:00Z",
                "role": "user" if msg_id % 2 == 0 else "assistant",
                "content": f"This is message {msg_id} with substantial content. " * 20,  # ~1KB per message
                "metadata": {
                    "tokens": 150 + (msg_id % 100),
                    "model": "claude-3-sonnet",
                    "temperature": 0.7,
                    "context_used": f"context_block_{msg_id % 10}",
                    "tools_used": [f"tool_{i}" for i in range(msg_id % 5)],
                    "processing_time": 1.2 + (msg_id % 10) * 0.1
                },
                "context_blocks": [
                    {"block_id": f"block_{i}", "content": f"Context content {i} " * 10}
                    for i in range(msg_id % 3)
                ]
            }
            massive_conversation.append(message)
        
        # Also create massive conversation metadata
        massive_metadata = {
            "session_id": "ultra_long_session_2024",
            "user_preferences": {
                "communication_style": "detailed_technical",
                "expertise_level": "expert",
                "preferred_tools": ["automation", "calendar", "email", "maps"],
                "conversation_history": [f"topic_{i}" for i in range(500)],
                "learning_patterns": {f"pattern_{i}": f"data_{i}" * 50 for i in range(100)}
            },
            "automation_configs": [
                {
                    "name": f"automation_{i}",
                    "triggers": [f"trigger_{j}" for j in range(10)],
                    "actions": [f"action_{j}" for j in range(15)],
                    "config": {f"param_{k}": f"value_{k}" * 20 for k in range(20)}
                }
                for i in range(50)
            ],
            "analytics": {
                "daily_stats": [
                    {
                        "date": f"2024-01-{day:02d}",
                        "message_count": 50 + (day * 3),
                        "topics": [f"topic_{t}" for t in range(20)],
                        "tools_used": {f"tool_{t}": t * 5 for t in range(10)}
                    }
                    for day in range(1, 366)  # Full year of stats
                ]
            }
        }
        
        # Create ArchivedConversation with massive JSONB data
        archived_conversation = ArchivedConversation(
            conversation_date=datetime.now(),
            messages=massive_conversation,  # ~2MB of message data
            message_count=2000,
            summary="This is a comprehensive long-term conversation spanning multiple months with extensive technical discussions.",
            conversation_metadata=massive_metadata  # ~5MB of metadata
        )
        
        # Test storage of massive JSONB - this should not fail
        db_session.add(archived_conversation)
        db_session.commit()
        
        # Verify core contracts still work
        assert archived_conversation.id is not None
        assert isinstance(archived_conversation.archived_at, datetime)
        assert archived_conversation.message_count == 2000
        
        # Test retrieval of massive JSONB - this should be reasonably fast
        retrieved = db_session.query(ArchivedConversation).filter_by(id=archived_conversation.id).first()
        
        assert retrieved is not None
        assert retrieved.message_count == 2000
        assert len(retrieved.messages) == 2000
        
        # Verify complex nested data integrity
        assert retrieved.messages[1000]["id"] == "msg_1000"
        assert retrieved.messages[1000]["content"].startswith("This is message 1000")
        assert retrieved.messages[1000]["metadata"]["tokens"] == 150 + (1000 % 100)
        
        # Verify massive metadata survived
        assert "ultra_long_session_2024" in retrieved.conversation_metadata["session_id"]
        assert len(retrieved.conversation_metadata["automation_configs"]) == 50
        assert len(retrieved.conversation_metadata["analytics"]["daily_stats"]) == 365
        
        # Verify specific nested data deep in the structure
        assert retrieved.conversation_metadata["automation_configs"][25]["name"] == "automation_25"
        assert len(retrieved.conversation_metadata["automation_configs"][25]["triggers"]) == 10
        assert retrieved.conversation_metadata["analytics"]["daily_stats"][100]["date"] == "2024-01-101"