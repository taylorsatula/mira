"""
Tests for lt_memory.models - Pydantic data structures.

Tests all models for:
- Valid construction with required/optional fields
- Field validation rules
- Default values
- Edge cases and boundary conditions
- Factory methods where applicable
"""
import pytest
from datetime import datetime, timezone
from uuid import uuid4, UUID
from pydantic import ValidationError

from lt_memory.models import (
    Memory,
    ExtractedMemory,
    MemoryLink,
    Entity,
    ProcessingChunk,
    PostProcessingBatch,
    RefinementCandidate,
    ConsolidationCluster,
)


class TestMemory:
    """Tests for Memory model."""

    def test_memory_creation_minimal(self):
        """Memory can be created with minimal required fields."""
        memory_id = uuid4()
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        memory = Memory(
            id=memory_id,
            user_id=user_id,
            text="Test memory",
            importance_score=0.7,
            created_at=now,
        )

        assert memory.id == memory_id
        assert memory.user_id == user_id
        assert memory.text == "Test memory"
        assert memory.importance_score == 0.7
        assert memory.created_at == now
        assert memory.access_count == 0
        assert memory.confidence == 0.9  # Default
        assert memory.is_archived is False
        assert memory.is_refined is False

    def test_memory_with_all_fields(self):
        """Memory can be created with all fields populated."""
        memory_id = uuid4()
        user_id = uuid4()
        now = datetime.now(timezone.utc)
        embedding = [0.1] * 768

        memory = Memory(
            id=memory_id,
            user_id=user_id,
            text="Comprehensive test",
            embedding=embedding,
            importance_score=0.85,
            created_at=now,
            updated_at=now,
            expires_at=now,
            access_count=5,
            last_accessed=now,
            happens_at=now,
            inbound_links=[{"source_id": str(uuid4()), "link_type": "related"}],
            outbound_links=[{"target_id": str(uuid4()), "link_type": "supports"}],
            entity_links=[{"entity_id": str(uuid4()), "entity_name": "Test"}],
            confidence=0.95,
            is_archived=True,
            archived_at=now,
            is_refined=True,
            last_refined_at=now,
            refinement_rejection_count=2,
            activity_days_at_creation=100,
            activity_days_at_last_access=150,
            similarity_score=0.88,
        )

        assert memory.embedding == embedding
        assert memory.access_count == 5
        assert memory.is_archived is True
        assert memory.refinement_rejection_count == 2
        assert memory.similarity_score == 0.88

    def test_memory_importance_score_validation(self):
        """Memory importance_score must be between 0.0 and 1.0."""
        memory_id = uuid4()
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        # Valid boundary values
        Memory(id=memory_id, user_id=user_id, text="Test", importance_score=0.0, created_at=now)
        Memory(id=memory_id, user_id=user_id, text="Test", importance_score=1.0, created_at=now)

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            Memory(id=memory_id, user_id=user_id, text="Test", importance_score=-0.1, created_at=now)
        assert "importance_score" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Memory(id=memory_id, user_id=user_id, text="Test", importance_score=1.5, created_at=now)
        assert "importance_score" in str(exc_info.value)

    def test_memory_confidence_validation(self):
        """Memory confidence must be between 0.0 and 1.0."""
        memory_id = uuid4()
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        # Valid boundary values
        Memory(id=memory_id, user_id=user_id, text="Test", importance_score=0.5, created_at=now, confidence=0.0)
        Memory(id=memory_id, user_id=user_id, text="Test", importance_score=0.5, created_at=now, confidence=1.0)

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            Memory(id=memory_id, user_id=user_id, text="Test", importance_score=0.5, created_at=now, confidence=-0.1)
        assert "confidence" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Memory(id=memory_id, user_id=user_id, text="Test", importance_score=0.5, created_at=now, confidence=1.1)
        assert "confidence" in str(exc_info.value)

    def test_memory_transient_fields_excluded(self):
        """Transient fields (linked_memories, link_metadata) are excluded from dict."""
        memory_id = uuid4()
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        memory = Memory(
            id=memory_id,
            user_id=user_id,
            text="Test",
            importance_score=0.5,
            created_at=now,
            linked_memories=[{"id": str(uuid4())}],
            link_metadata={"traversal_depth": 1},
        )

        # Fields should be accessible
        assert memory.linked_memories is not None
        assert memory.link_metadata is not None

        # But excluded from dict export
        memory_dict = memory.model_dump()
        assert "linked_memories" not in memory_dict
        assert "link_metadata" not in memory_dict


class TestExtractedMemory:
    """Tests for ExtractedMemory model."""

    def test_extracted_memory_minimal(self):
        """ExtractedMemory can be created with just text."""
        extracted = ExtractedMemory(text="Extracted fact")

        assert extracted.text == "Extracted fact"
        assert extracted.importance_score == 0.5  # Default
        assert extracted.confidence == 0.9  # Default
        assert extracted.related_memory_ids == []
        assert extracted.consolidates_memory_ids == []

    def test_extracted_memory_with_relationships(self):
        """ExtractedMemory can include relationship information."""
        extracted = ExtractedMemory(
            text="Related fact",
            importance_score=0.8,
            relationship_type="supports",
            related_memory_ids=["mem1", "mem2"],
            consolidates_memory_ids=["mem3"],
        )

        assert extracted.relationship_type == "supports"
        assert extracted.related_memory_ids == ["mem1", "mem2"]
        assert extracted.consolidates_memory_ids == ["mem3"]

    def test_extracted_memory_score_validation(self):
        """ExtractedMemory validates importance_score and confidence ranges."""
        # Valid boundaries
        ExtractedMemory(text="Test", importance_score=0.0, confidence=0.0)
        ExtractedMemory(text="Test", importance_score=1.0, confidence=1.0)

        # Invalid importance_score
        with pytest.raises(ValidationError) as exc_info:
            ExtractedMemory(text="Test", importance_score=-0.1)
        assert "importance_score" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ExtractedMemory(text="Test", importance_score=1.5)
        assert "importance_score" in str(exc_info.value)

        # Invalid confidence
        with pytest.raises(ValidationError) as exc_info:
            ExtractedMemory(text="Test", confidence=-0.1)
        assert "confidence" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ExtractedMemory(text="Test", confidence=1.1)
        assert "confidence" in str(exc_info.value)

    def test_extracted_memory_temporal_fields(self):
        """ExtractedMemory supports temporal fields."""
        now = datetime.now(timezone.utc)
        extracted = ExtractedMemory(
            text="Future event",
            happens_at=now,
            expires_at=now,
        )

        assert extracted.happens_at == now
        assert extracted.expires_at == now


class TestMemoryLink:
    """Tests for MemoryLink model."""

    def test_memory_link_creation(self):
        """MemoryLink can be created with all required fields."""
        source_id = uuid4()
        target_id = uuid4()
        now = datetime.now(timezone.utc)

        link = MemoryLink(
            source_id=source_id,
            target_id=target_id,
            link_type="related",
            confidence=0.85,
            reasoning="Both discuss the same topic",
            created_at=now,
        )

        assert link.source_id == source_id
        assert link.target_id == target_id
        assert link.link_type == "related"
        assert link.confidence == 0.85
        assert link.reasoning == "Both discuss the same topic"

    def test_memory_link_type_validation(self):
        """MemoryLink validates link_type is one of allowed values."""
        source_id = uuid4()
        target_id = uuid4()
        now = datetime.now(timezone.utc)

        # Valid link types
        for link_type in ["related", "supports", "conflicts", "supersedes"]:
            MemoryLink(
                source_id=source_id,
                target_id=target_id,
                link_type=link_type,
                confidence=0.8,
                reasoning="Test",
                created_at=now,
            )

        # Invalid link type
        with pytest.raises(ValidationError) as exc_info:
            MemoryLink(
                source_id=source_id,
                target_id=target_id,
                link_type="invalid_type",
                confidence=0.8,
                reasoning="Test",
                created_at=now,
            )
        assert "link_type" in str(exc_info.value)
        assert "related" in str(exc_info.value)

    def test_memory_link_confidence_validation(self):
        """MemoryLink confidence must be between 0.0 and 1.0."""
        source_id = uuid4()
        target_id = uuid4()
        now = datetime.now(timezone.utc)

        # Valid boundaries
        MemoryLink(source_id=source_id, target_id=target_id, link_type="related",
                   confidence=0.0, reasoning="Test", created_at=now)
        MemoryLink(source_id=source_id, target_id=target_id, link_type="related",
                   confidence=1.0, reasoning="Test", created_at=now)

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            MemoryLink(source_id=source_id, target_id=target_id, link_type="related",
                      confidence=-0.1, reasoning="Test", created_at=now)
        assert "confidence" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            MemoryLink(source_id=source_id, target_id=target_id, link_type="related",
                      confidence=1.1, reasoning="Test", created_at=now)
        assert "confidence" in str(exc_info.value)


class TestEntity:
    """Tests for Entity model."""

    def test_entity_creation(self):
        """Entity can be created with required fields."""
        entity_id = uuid4()
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        entity = Entity(
            id=entity_id,
            user_id=user_id,
            name="Taylor",
            entity_type="PERSON",
            created_at=now,
        )

        assert entity.id == entity_id
        assert entity.user_id == user_id
        assert entity.name == "Taylor"
        assert entity.entity_type == "PERSON"
        assert entity.link_count == 0  # Default
        assert entity.is_archived is False

    def test_entity_with_embedding(self):
        """Entity can store spaCy word vector embedding."""
        entity_id = uuid4()
        user_id = uuid4()
        now = datetime.now(timezone.utc)
        embedding = [0.1] * 300  # spaCy en_core_web_lg 300d vectors

        entity = Entity(
            id=entity_id,
            user_id=user_id,
            name="Acme Corp",
            entity_type="ORG",
            embedding=embedding,
            created_at=now,
        )

        assert entity.embedding == embedding
        assert len(entity.embedding) == 300

    def test_entity_link_tracking(self):
        """Entity tracks link count and last linked timestamp."""
        entity_id = uuid4()
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        entity = Entity(
            id=entity_id,
            user_id=user_id,
            name="Python",
            entity_type="PRODUCT",
            link_count=15,
            last_linked_at=now,
            created_at=now,
        )

        assert entity.link_count == 15
        assert entity.last_linked_at == now

    def test_entity_archival(self):
        """Entity supports archival state."""
        entity_id = uuid4()
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        entity = Entity(
            id=entity_id,
            user_id=user_id,
            name="Deprecated API",
            entity_type="PRODUCT",
            is_archived=True,
            archived_at=now,
            created_at=now,
        )

        assert entity.is_archived is True
        assert entity.archived_at == now


class TestProcessingChunk:
    """Tests for ProcessingChunk model."""

    def test_processing_chunk_creation(self):
        """ProcessingChunk can be created with message list."""
        # Mock message objects
        class MockMessage:
            def __init__(self, created_at):
                self.created_at = created_at
                self.content = "Test message"

        now = datetime.now(timezone.utc)
        messages = [MockMessage(now), MockMessage(now)]

        chunk = ProcessingChunk(
            messages=messages,
            temporal_start=now,
            temporal_end=now,
            chunk_index=0,
        )

        assert len(chunk.messages) == 2
        assert chunk.chunk_index == 0
        assert chunk.memory_context_snapshot == {}

    def test_processing_chunk_rejects_empty_messages(self):
        """ProcessingChunk validates messages list is not empty."""
        now = datetime.now(timezone.utc)

        with pytest.raises(ValidationError) as exc_info:
            ProcessingChunk(
                messages=[],
                temporal_start=now,
                temporal_end=now,
                chunk_index=0,
            )
        assert "at least one message" in str(exc_info.value).lower()

    def test_processing_chunk_from_conversation_messages(self):
        """ProcessingChunk.from_conversation_messages factory method."""
        class MockMessage:
            def __init__(self, created_at):
                self.created_at = created_at
                self.content = "Test"

        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)
        messages = [MockMessage(start_time), MockMessage(end_time)]

        chunk = ProcessingChunk.from_conversation_messages(messages, chunk_index=2)

        assert len(chunk.messages) == 2
        assert chunk.temporal_start == start_time
        assert chunk.temporal_end == end_time
        assert chunk.chunk_index == 2
        assert chunk.memory_context_snapshot == {}

    def test_processing_chunk_from_conversation_rejects_empty(self):
        """ProcessingChunk.from_conversation_messages rejects empty list."""
        with pytest.raises(ValueError) as exc_info:
            ProcessingChunk.from_conversation_messages([], chunk_index=0)
        assert "empty" in str(exc_info.value).lower()


class TestPostProcessingBatch:
    """Tests for PostProcessingBatch model."""

    def test_post_processing_batch_creation(self):
        """PostProcessingBatch can be created with required fields."""
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        batch = PostProcessingBatch(
            batch_id="batch_rel_123",
            batch_type="relationship_classification",
            user_id=user_id,
            request_payload={"model": "claude-3-5-haiku-20241022"},
            input_data={"memory_pairs": []},
            status="submitted",
            items_submitted=10,
            created_at=now,
            submitted_at=now,
        )

        assert batch.batch_type == "relationship_classification"
        assert batch.items_submitted == 10
        assert batch.items_completed == 0
        assert batch.links_created == 0

    def test_post_processing_batch_type_validation(self):
        """PostProcessingBatch validates batch_type."""
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        # Valid batch types
        valid_types = ["relationship_classification", "consolidation", "consolidation_review"]
        for batch_type in valid_types:
            PostProcessingBatch(
                batch_id="batch_test",
                batch_type=batch_type,
                user_id=user_id,
                request_payload={},
                input_data={},
                status="submitted",
                items_submitted=5,
                created_at=now,
                submitted_at=now,
            )

        # Invalid batch type
        with pytest.raises(ValidationError) as exc_info:
            PostProcessingBatch(
                batch_id="batch_test",
                batch_type="invalid_type",
                user_id=user_id,
                request_payload={},
                input_data={},
                status="submitted",
                items_submitted=5,
                created_at=now,
                submitted_at=now,
            )
        assert "batch_type" in str(exc_info.value)

    def test_post_processing_batch_with_results(self):
        """PostProcessingBatch tracks completion metrics."""
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        batch = PostProcessingBatch(
            batch_id="batch_complete",
            batch_type="relationship_classification",
            user_id=user_id,
            request_payload={},
            input_data={},
            status="completed",
            items_submitted=20,
            items_completed=18,
            items_failed=2,
            links_created=25,
            conflicts_flagged=3,
            created_at=now,
            submitted_at=now,
            completed_at=now,
        )

        assert batch.items_completed == 18
        assert batch.items_failed == 2
        assert batch.links_created == 25
        assert batch.conflicts_flagged == 3


class TestRefinementCandidate:
    """Tests for RefinementCandidate model."""

    def test_refinement_candidate_creation(self):
        """RefinementCandidate can be created with required fields."""
        memory_id = uuid4()

        candidate = RefinementCandidate(
            memory_id=memory_id,
            reason="verbose",
            current_text="This is a very long memory that could be shortened...",
            char_count=150,
        )

        assert candidate.memory_id == memory_id
        assert candidate.reason == "verbose"
        assert candidate.char_count == 150
        assert candidate.target_memory_ids == []
        assert candidate.similarity_scores == []

    def test_refinement_candidate_reason_validation(self):
        """RefinementCandidate validates reason is one of allowed values."""
        memory_id = uuid4()

        # Valid reasons
        for reason in ["verbose", "consolidatable", "stale"]:
            RefinementCandidate(
                memory_id=memory_id,
                reason=reason,
                current_text="Test",
                char_count=50,
            )

        # Invalid reason
        with pytest.raises(ValidationError) as exc_info:
            RefinementCandidate(
                memory_id=memory_id,
                reason="invalid_reason",
                current_text="Test",
                char_count=50,
            )
        assert "reason" in str(exc_info.value)

    def test_refinement_candidate_with_consolidation_targets(self):
        """RefinementCandidate can specify consolidation targets."""
        memory_id = uuid4()
        target1 = uuid4()
        target2 = uuid4()

        candidate = RefinementCandidate(
            memory_id=memory_id,
            reason="consolidatable",
            current_text="Similar memory",
            char_count=50,
            target_memory_ids=[target1, target2],
            similarity_scores=[0.92, 0.88],
        )

        assert len(candidate.target_memory_ids) == 2
        assert candidate.similarity_scores == [0.92, 0.88]


class TestConsolidationCluster:
    """Tests for ConsolidationCluster model."""

    def test_consolidation_cluster_creation(self):
        """ConsolidationCluster can be created with valid data."""
        mem1 = uuid4()
        mem2 = uuid4()
        mem3 = uuid4()

        cluster = ConsolidationCluster(
            cluster_id="cluster_1",
            memory_ids=[mem1, mem2, mem3],
            memory_texts=["Memory 1", "Memory 2", "Memory 3"],
            similarity_scores=[0.95, 0.92, 0.90],
            avg_similarity=0.923,
            consolidation_confidence=0.85,
        )

        assert cluster.cluster_id == "cluster_1"
        assert len(cluster.memory_ids) == 3
        assert cluster.avg_similarity == 0.923
        assert cluster.consolidation_confidence == 0.85

    def test_consolidation_cluster_min_size_validation(self):
        """ConsolidationCluster requires at least 2 memories."""
        mem1 = uuid4()

        # Single memory should fail
        with pytest.raises(ValidationError) as exc_info:
            ConsolidationCluster(
                cluster_id="cluster_fail",
                memory_ids=[mem1],
                memory_texts=["Memory 1"],
                similarity_scores=[0.95],
                avg_similarity=0.95,
                consolidation_confidence=0.8,
            )
        assert "at least 2 memories" in str(exc_info.value).lower()

        # Two memories should succeed
        mem2 = uuid4()
        ConsolidationCluster(
            cluster_id="cluster_ok",
            memory_ids=[mem1, mem2],
            memory_texts=["Memory 1", "Memory 2"],
            similarity_scores=[0.95, 0.92],
            avg_similarity=0.935,
            consolidation_confidence=0.8,
        )

    def test_consolidation_cluster_confidence_validation(self):
        """ConsolidationCluster consolidation_confidence must be between 0.0 and 1.0."""
        mem1 = uuid4()
        mem2 = uuid4()

        # Valid boundaries
        ConsolidationCluster(
            cluster_id="cluster_low",
            memory_ids=[mem1, mem2],
            memory_texts=["M1", "M2"],
            similarity_scores=[0.9, 0.9],
            avg_similarity=0.9,
            consolidation_confidence=0.0,
        )
        ConsolidationCluster(
            cluster_id="cluster_high",
            memory_ids=[mem1, mem2],
            memory_texts=["M1", "M2"],
            similarity_scores=[0.9, 0.9],
            avg_similarity=0.9,
            consolidation_confidence=1.0,
        )

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            ConsolidationCluster(
                cluster_id="cluster_invalid",
                memory_ids=[mem1, mem2],
                memory_texts=["M1", "M2"],
                similarity_scores=[0.9, 0.9],
                avg_similarity=0.9,
                consolidation_confidence=-0.1,
            )
        assert "consolidation_confidence" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ConsolidationCluster(
                cluster_id="cluster_invalid",
                memory_ids=[mem1, mem2],
                memory_texts=["M1", "M2"],
                similarity_scores=[0.9, 0.9],
                avg_similarity=0.9,
                consolidation_confidence=1.5,
            )
        assert "consolidation_confidence" in str(exc_info.value)
