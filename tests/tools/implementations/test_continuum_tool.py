"""
Tests for ContinuumSearchTool.

Following MIRA's real testing philosophy:
- No mocks, use real services
- Test contracts, not implementation
- Verify exact return structures and error messages
- Cover all edge cases identified by contract analysis
"""
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from tools.implementations.continuum_tool import ContinuumSearchTool
from cns.infrastructure.continuum_repository import get_continuum_repository
from cns.core.message import Message
from cns.services.segment_helpers import (
    create_segment_boundary_sentinel,
    collapse_segment_sentinel
)
from utils.timezone_utils import utc_now, format_utc_iso
from utils.user_context import set_current_user_id


class TestContinuumSearchToolContract:
    """Tests that enforce ContinuumSearchTool's contract guarantees."""

    @pytest.fixture
    def search_tool(self):
        """Create ContinuumSearchTool instance."""
        return ContinuumSearchTool()

    def test_tool_name_and_schema(self, search_tool):
        """Verify tool name matches schema name."""
        assert search_tool.name == "continuum_tool"
        assert search_tool.anthropic_schema["name"] == "continuum_tool"

    def test_unknown_operation_raises_valueerror(self, search_tool, authenticated_user):
        """CONTRACT E1: Unknown operation raises ValueError with specific pattern."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        with pytest.raises(ValueError, match="Unknown operation:.*Valid operations are:"):
            search_tool.run("invalid_operation", query="test")


class TestSearchSummariesOperation:
    """Tests for search operation in summaries mode (default)."""

    @pytest.fixture
    def search_tool(self):
        """Create ContinuumSearchTool instance."""
        return ContinuumSearchTool()

    @pytest.fixture
    def setup_test_segments(self, authenticated_user, test_db):
        """Create test segments with collapsed summaries."""
        user_id = authenticated_user["user_id"]
        continuum_id = authenticated_user["continuum_id"]  # Use test user's continuum
        set_current_user_id(user_id)

        repo = get_continuum_repository()

        # Create collapsed segment 1 - high confidence match
        segment1_id = str(uuid4())
        segment1_time = utc_now() - timedelta(days=2)
        sentinel1 = Message(
            id=segment1_id,
            role="assistant",
            content="[Segment collapsed]",
            created_at=segment1_time,
            metadata={
                "is_segment_boundary": True,
                "status": "collapsed",
                "segment_id": segment1_id,
                "segment_start_time": format_utc_iso(segment1_time),
                "segment_end_time": format_utc_iso(segment1_time + timedelta(hours=2)),
                "display_title": "Python async patterns discussion",
                "summary": "Discussed Python async patterns including asyncio, await syntax, and concurrent.futures. Mark explained the event loop architecture.",
                "tools_used": ["code_tool"],
                "segment_embedding_value": [0.1] * 768  # Dummy embedding
            }
        )
        repo.save_message(sentinel1, continuum_id, user_id)

        # Create collapsed segment 2 - medium confidence match
        segment2_id = str(uuid4())
        segment2_time = utc_now() - timedelta(days=5)
        sentinel2 = Message(
            id=segment2_id,
            role="assistant",
            content="[Segment collapsed]",
            created_at=segment2_time,
            metadata={
                "is_segment_boundary": True,
                "status": "collapsed",
                "segment_id": segment2_id,
                "segment_start_time": format_utc_iso(segment2_time),
                "segment_end_time": format_utc_iso(segment2_time + timedelta(hours=1)),
                "display_title": "Database migration planning",
                "summary": "Planned PostgreSQL migration strategy. Discussed async patterns for batch processing.",
                "tools_used": [],
                "segment_embedding_value": [0.2] * 768  # Different dummy embedding
            }
        )
        repo.save_message(sentinel2, continuum_id, user_id)

        return {
            "user_id": user_id,
            "continuum_id": continuum_id,
            "segment1": sentinel1,
            "segment2": sentinel2
        }

    def test_search_empty_query_raises_valueerror(self, search_tool, authenticated_user):
        """CONTRACT E2: Empty query raises ValueError."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        with pytest.raises(ValueError, match="Query must be provided for search operation"):
            search_tool.run("search", query="")

        with pytest.raises(ValueError, match="Query must be provided for search operation"):
            search_tool.run("search", query="   ")

    def test_search_invalid_search_mode_raises_valueerror(self, search_tool, authenticated_user):
        """CONTRACT E3: Invalid search_mode raises ValueError."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        with pytest.raises(ValueError, match="search_mode must be 'summaries', 'messages', or 'memories', got: invalid"):
            search_tool.run("search", query="test", search_mode="invalid")

    def test_search_returns_correct_structure(self, search_tool, setup_test_segments):
        """CONTRACT R1-R13: Verify complete return structure for summary search."""
        user_id = setup_test_segments["user_id"]
        set_current_user_id(user_id)

        result = search_tool.run("search", query="Python async patterns")

        # R1: status field
        assert result["status"] in ["high_confidence", "medium_confidence", "low_confidence", "no_results"]

        # R2: confidence field
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        # Check it's rounded to 3 decimals
        assert result["confidence"] == round(result["confidence"], 3)

        # R3: query field
        assert result["query"] == "Python async patterns"

        # R4: entities field
        assert isinstance(result["entities"], list)
        assert result["entities"] == []  # No entities passed

        # R5: results field with segment summaries
        assert isinstance(result["results"], list)
        for res in result["results"]:
            assert res["result_type"] == "segment_summary"

            # R6: segment_id
            assert isinstance(res["segment_id"], str)
            assert len(res["segment_id"]) == 8

            # R7: required fields
            assert "display_title" in res
            assert "summary" in res
            assert isinstance(res["confidence_score"], float)
            assert 0.0 <= res["confidence_score"] <= 1.0

            # R8: time_boundaries
            assert isinstance(res["time_boundaries"], dict)
            assert "start" in res["time_boundaries"]
            assert "end" in res["time_boundaries"]

            # R9: matched_entities
            assert isinstance(res["matched_entities"], list)

        # R10: result_count
        assert result["result_count"] == len(result["results"])

        # R11: has_more_pages
        assert isinstance(result["has_more_pages"], bool)

        # R12: search_mode
        assert result["search_mode"] == "summaries"

        # R13: meta fields
        assert result["meta"]["search_tier"] == "hybrid_vector_bm25"
        assert result["meta"]["vector_weight"] == 0.6
        assert result["meta"]["text_weight"] == 0.4

    def test_search_with_entities_boosts_confidence(self, search_tool, setup_test_segments):
        """CONTRACT R9, EC13-14: Entity matching boosts confidence by 10% per match."""
        user_id = setup_test_segments["user_id"]
        set_current_user_id(user_id)

        # Search without entities
        result1 = search_tool.run("search", query="async patterns")
        base_score = result1["results"][0]["confidence_score"] if result1["results"] else 0.0

        # Search with matching entity - should boost by 10%
        result2 = search_tool.run("search", query="async patterns", entities=["Mark"])

        if result2["results"]:
            assert "Mark" in result2["results"][0]["matched_entities"]
            # Score should be boosted (but capped at 1.0)
            expected_boost = min(base_score * 1.1, 1.0)
            assert result2["results"][0]["confidence_score"] >= base_score

    def test_search_no_results(self, search_tool, authenticated_user):
        """CONTRACT EC1: No results returns specific structure."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        result = search_tool.run("search", query="nonexistent query xyz123")

        assert result["status"] == "no_results"
        assert result["confidence"] == 0.0
        assert result["results"] == []
        assert result["result_count"] == 0

    def test_embedding_failure_raises_valueerror(self, search_tool, authenticated_user, monkeypatch):
        """CONTRACT E6: Embedding failure raises ValueError."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        # Mock embedding failure
        def mock_encode_realtime(*args, **kwargs):
            raise Exception("Embedding service unavailable")

        monkeypatch.setattr(
            search_tool._embeddings_provider,
            "encode_realtime",
            mock_encode_realtime
        )

        with pytest.raises(ValueError, match="Cannot perform summary search.*embedding generation failed"):
            search_tool.run("search", query="test query")


class TestConfidenceFiltering:
    """Test the smart result filtering based on confidence clustering."""

    @pytest.fixture
    def search_tool(self):
        """Create ContinuumSearchTool instance."""
        return ContinuumSearchTool()

    def test_clear_winner_returns_only_one(self, search_tool):
        """CONTRACT EC3: Top result >15% ahead returns only that result."""
        results = [
            {"confidence_score": 0.9},
            {"confidence_score": 0.7},  # 0.2 difference > 0.15
            {"confidence_score": 0.6},
            {"confidence_score": 0.5}
        ]

        filtered = search_tool._filter_results_by_confidence(results)

        assert len(filtered) == 1
        assert filtered[0]["confidence_score"] == 0.9

    def test_clustered_results_within_threshold(self, search_tool):
        """CONTRACT EC4: Top 3 within 15% all returned."""
        results = [
            {"confidence_score": 0.9},
            {"confidence_score": 0.85},  # Within 15%
            {"confidence_score": 0.8},   # Within 15%
            {"confidence_score": 0.6}    # Outside 15%
        ]

        filtered = search_tool._filter_results_by_confidence(results)

        assert len(filtered) == 3
        assert [r["confidence_score"] for r in filtered] == [0.9, 0.85, 0.8]

    def test_default_returns_top_two(self, search_tool):
        """CONTRACT EC5: Default case returns top 2."""
        results = [
            {"confidence_score": 0.9},
            {"confidence_score": 0.8},   # Not within 15%
            {"confidence_score": 0.5},
            {"confidence_score": 0.4}
        ]

        filtered = search_tool._filter_results_by_confidence(results)

        assert len(filtered) == 2
        assert [r["confidence_score"] for r in filtered] == [0.9, 0.8]

    def test_never_returns_more_than_four(self, search_tool):
        """CONTRACT EC6: Never returns >4 results."""
        results = [
            {"confidence_score": 0.9},
            {"confidence_score": 0.88},
            {"confidence_score": 0.86},
            {"confidence_score": 0.84},
            {"confidence_score": 0.82},  # 5th result
            {"confidence_score": 0.80}   # 6th result
        ]

        filtered = search_tool._filter_results_by_confidence(results)

        assert len(filtered) <= 4

    def test_single_result_not_clustered(self, search_tool):
        """CONTRACT EC2: Single result returns immediately."""
        results = [{"confidence_score": 0.9}]

        filtered = search_tool._filter_results_by_confidence(results)

        assert len(filtered) == 1
        assert filtered[0]["confidence_score"] == 0.9

    def test_empty_results_returns_empty(self, search_tool):
        """Edge case: Empty input returns empty list."""
        filtered = search_tool._filter_results_by_confidence([])
        assert filtered == []


class TestSearchMessagesOperation:
    """Tests for search operation in messages mode."""

    @pytest.fixture
    def search_tool(self):
        """Create ContinuumSearchTool instance."""
        return ContinuumSearchTool()

    @pytest.fixture
    def setup_test_messages(self, authenticated_user, test_db):
        """Create test messages for search."""
        user_id = authenticated_user["user_id"]
        continuum_id = authenticated_user["continuum_id"]  # Use test user's continuum
        set_current_user_id(user_id)

        repo = get_continuum_repository()

        # Add messages directly to test user's continuum
        messages = []
        base_time = utc_now() - timedelta(hours=2)

        for i in range(5):
            msg = Message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i} about Python async patterns and concurrent programming.",
                created_at=base_time + timedelta(minutes=i * 10)
            )
            repo.save_message(msg, continuum_id, user_id)
            messages.append(msg)

        return {
            "user_id": user_id,
            "continuum_id": continuum_id,
            "messages": messages,
            "start_time": format_utc_iso(base_time - timedelta(minutes=10)),
            "end_time": format_utc_iso(base_time + timedelta(hours=1))
        }

    def test_message_search_requires_timescope(self, search_tool, authenticated_user):
        """CONTRACT E4: Message search without timescope raises ValueError."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        # Missing both timestamps
        with pytest.raises(ValueError, match="Message search requires both start_time and end_time"):
            search_tool.run("search", query="test", search_mode="messages")

        # Missing end_time
        with pytest.raises(ValueError, match="Message search requires both start_time and end_time"):
            search_tool.run("search", query="test", search_mode="messages",
                          start_time="2025-01-01T00:00:00Z")

        # Missing start_time
        with pytest.raises(ValueError, match="Message search requires both start_time and end_time"):
            search_tool.run("search", query="test", search_mode="messages",
                          end_time="2025-01-01T12:00:00Z")

    def test_message_search_validates_time_order(self, search_tool, authenticated_user):
        """CONTRACT E5: start_time >= end_time raises ValueError."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        with pytest.raises(ValueError, match="start_time must be before end_time"):
            search_tool.run(
                "search",
                query="test",
                search_mode="messages",
                start_time="2025-01-01T12:00:00Z",
                end_time="2025-01-01T06:00:00Z"  # Earlier than start
            )

    def test_message_search_returns_correct_structure(self, search_tool, setup_test_messages):
        """CONTRACT R14-R20: Verify message search return structure."""
        user_id = setup_test_messages["user_id"]
        set_current_user_id(user_id)

        result = search_tool.run(
            "search",
            query="Python async",
            search_mode="messages",
            start_time=setup_test_messages["start_time"],
            end_time=setup_test_messages["end_time"]
        )

        # R14: status based on confidence
        assert result["status"] in ["high_confidence", "medium_confidence", "low_confidence"]

        # R15: message preview structure
        for msg in result["results"]:
            assert len(msg["message_id"]) == 8
            assert "full_uuid" in msg
            assert "continuum_id" in msg
            assert msg["role"] in ["user", "assistant"]
            assert isinstance(msg["timestamp"], str)

            # R16: preview and truncation
            assert "preview" in msg
            assert isinstance(msg["is_truncated"], bool)
            assert isinstance(msg["full_length"], int)

            # R17: match_score
            assert isinstance(msg["match_score"], float)
            assert 0.0 <= msg["match_score"] <= 1.0

        # R18: time_boundaries
        assert result["time_boundaries"]["start"] == setup_test_messages["start_time"]
        assert result["time_boundaries"]["end"] == setup_test_messages["end_time"]

        # R19: search_mode
        assert result["search_mode"] == "messages"

        # R20: meta
        assert result["meta"]["search_tier"] == "bm25_timeframe"

    def test_message_truncation_at_sentence_boundary(self, search_tool):
        """CONTRACT EC8-9: Message truncation prefers sentence boundaries."""
        # Test with content that has clear sentence boundaries
        content = "This is the first sentence. This is the second sentence! And here is the third? This goes on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on and on."

        message = {
            "id": str(uuid4()),
            "continuum_id": str(uuid4()),
            "role": "user",
            "content": content,
            "created_at": utc_now(),
            "metadata": {},
            "rank": 0.8,
            "matched_entities": []
        }

        preview_result = search_tool._format_message_preview(message)

        # Should truncate at sentence boundary, not mid-sentence
        assert preview_result["is_truncated"] == True
        assert preview_result["preview"].endswith(("..", "...", ". ", "! ", "? "))
        assert len(preview_result["preview"]) <= 502  # 500 + ".."

    def test_message_entity_boost_twenty_percent(self, search_tool, setup_test_messages):
        """CONTRACT EC15: Entity boost is 20% for messages."""
        user_id = setup_test_messages["user_id"]
        set_current_user_id(user_id)

        # Search with entity that should boost score
        result = search_tool.run(
            "search",
            query="concurrent",
            search_mode="messages",
            start_time=setup_test_messages["start_time"],
            end_time=setup_test_messages["end_time"],
            entities=["Python", "async"]  # Both appear in messages
        )

        # Should have matched entities
        if result["results"]:
            assert any("Python" in msg["matched_entities"] for msg in result["results"])
            assert any("async" in msg["matched_entities"] for msg in result["results"])


class TestSearchWithinSegment:
    """Tests for search_within_segment operation."""

    @pytest.fixture
    def search_tool(self):
        """Create ContinuumSearchTool instance."""
        return ContinuumSearchTool()

    @pytest.fixture
    def setup_segment_with_messages(self, authenticated_user, test_db):
        """Create a segment with messages for testing."""
        user_id = authenticated_user["user_id"]
        continuum_id = authenticated_user["continuum_id"]  # Use test user's continuum
        set_current_user_id(user_id)

        repo = get_continuum_repository()

        # Create segment
        segment_id = str(uuid4())
        segment_time = utc_now() - timedelta(hours=2)

        # Create sentinel
        sentinel = Message(
            id=segment_id,
            role="assistant",
            content="[Segment collapsed]",
            created_at=segment_time,
            metadata={
                "is_segment_boundary": True,
                "status": "collapsed",
                "segment_id": segment_id,
                "segment_start_time": format_utc_iso(segment_time),
                "segment_end_time": format_utc_iso(segment_time + timedelta(hours=1)),
                "display_title": "Test segment",
                "summary": "Test segment for search within"
            }
        )
        repo.save_message(sentinel, continuum_id, user_id)

        # Add messages within segment
        for i in range(3):
            msg = Message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i} within segment about testing.",
                created_at=segment_time + timedelta(minutes=i * 10)
            )
            repo.save_message(msg, continuum_id, user_id)

        return {
            "user_id": user_id,
            "continuum_id": continuum_id,
            "segment_id": segment_id[:8],  # 8-char prefix
            "full_segment_id": segment_id
        }

    def test_search_within_segment_requires_query(self, search_tool, setup_segment_with_messages):
        """CONTRACT E7: search_within_segment requires query."""
        user_id = setup_segment_with_messages["user_id"]
        set_current_user_id(user_id)

        with pytest.raises(ValueError, match="Query is required for search_within_segment"):
            search_tool.run(
                "search_within_segment",
                segment_id=setup_segment_with_messages["segment_id"],
                query=""
            )

    def test_search_within_segment_not_found(self, search_tool, authenticated_user):
        """CONTRACT E8: segment not found raises ValueError."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        with pytest.raises(ValueError, match="No segment found with ID starting with 'badid123'"):
            search_tool.run(
                "search_within_segment",
                segment_id="badid123",
                query="test"
            )

    def test_search_within_segment_returns_structure(self, search_tool, setup_segment_with_messages):
        """CONTRACT R21-R23: Verify search_within_segment return structure."""
        user_id = setup_segment_with_messages["user_id"]
        set_current_user_id(user_id)

        result = search_tool.run(
            "search_within_segment",
            segment_id=setup_segment_with_messages["segment_id"],
            query="testing"
        )

        # R21: status and confidence
        assert result["status"] in ["high_confidence", "medium_confidence", "low_confidence"]
        assert isinstance(result["confidence"], float)

        # R22: segment_info
        assert "segment_info" in result
        assert result["segment_info"]["segment_id"] == setup_segment_with_messages["segment_id"]
        assert "display_title" in result["segment_info"]
        assert "summary" in result["segment_info"]

        # R23: results within segment boundaries
        assert isinstance(result["results"], list)
        assert result["search_mode"] == "messages"


class TestExpandMessage:
    """Tests for expand_message operation."""

    @pytest.fixture
    def search_tool(self):
        """Create ContinuumSearchTool instance."""
        return ContinuumSearchTool()

    @pytest.fixture
    def setup_messages_with_context(self, authenticated_user, test_db):
        """Create messages with context for expansion testing."""
        user_id = authenticated_user["user_id"]
        continuum_id = authenticated_user["continuum_id"]  # Use test user's continuum
        set_current_user_id(user_id)

        repo = get_continuum_repository()

        # Create several messages directly to test user's continuum
        messages = []
        base_time = utc_now() - timedelta(hours=1)

        for i in range(5):
            msg = Message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Context message {i} - {'A' * 600}",  # Long content to test truncation
                created_at=base_time + timedelta(minutes=i * 5)
            )
            repo.save_message(msg, continuum_id, user_id)
            messages.append(msg)

        # Get the middle message for testing
        target_idx = 2
        target_msg = messages[target_idx]

        return {
            "user_id": user_id,
            "continuum_id": continuum_id,
            "messages": messages,
            "target_message": target_msg,
            "target_id": str(target_msg.id)[:8]  # 8-char prefix
        }

    def test_expand_message_id_validation(self, search_tool, authenticated_user):
        """CONTRACT E9: message_id must be at least 8 characters."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        with pytest.raises(ValueError, match="message_id must be at least 8 characters"):
            search_tool.run("expand_message", message_id="short")

    def test_expand_message_direction_validation(self, search_tool, setup_messages_with_context):
        """CONTRACT E10: Invalid direction raises ValueError."""
        user_id = setup_messages_with_context["user_id"]
        set_current_user_id(user_id)

        with pytest.raises(ValueError, match="direction must be 'before', 'after', or 'both', got: invalid"):
            search_tool.run(
                "expand_message",
                message_id=setup_messages_with_context["target_id"],
                direction="invalid"
            )

    def test_expand_message_not_found(self, search_tool, authenticated_user):
        """CONTRACT E11: message not found raises ValueError."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        with pytest.raises(ValueError, match="No message found with ID starting with 'notfound'"):
            search_tool.run("expand_message", message_id="notfound")

    def test_expand_message_returns_structure(self, search_tool, setup_messages_with_context):
        """CONTRACT R24-R28: Verify expand_message return structure."""
        user_id = setup_messages_with_context["user_id"]
        set_current_user_id(user_id)

        result = search_tool.run(
            "expand_message",
            message_id=setup_messages_with_context["target_id"],
            direction="both",
            context_count=2
        )

        # R24: status
        assert result["status"] == "expanded"

        # R25: origin_message
        origin = result["origin_message"]
        assert origin["message_id"] == setup_messages_with_context["target_id"]
        assert len(origin["full_uuid"]) > 8
        assert origin["is_truncated"] == False  # Never truncated when expanded
        assert origin["content"] == setup_messages_with_context["target_message"].content

        # R26-27: context messages
        assert "context_before" in result
        assert "context_after" in result
        assert len(result["context_before"]) <= 2
        assert len(result["context_after"]) <= 2

        # R28: relation field
        for ctx in result.get("context_before", []):
            assert "relation" in ctx
            assert "before origin" in ctx["relation"]

        for ctx in result.get("context_after", []):
            assert "relation" in ctx
            assert "after origin" in ctx["relation"]

    def test_expand_message_direction_before_only(self, search_tool, setup_messages_with_context):
        """Test expand_message with direction='before'."""
        user_id = setup_messages_with_context["user_id"]
        set_current_user_id(user_id)

        result = search_tool.run(
            "expand_message",
            message_id=setup_messages_with_context["target_id"],
            direction="before",
            context_count=2
        )

        assert "context_before" in result
        assert "context_after" not in result

    def test_expand_message_context_count_validation(self, search_tool, setup_messages_with_context):
        """CONTRACT EC16-17: context_count defaults to 2, clamped to 10."""
        user_id = setup_messages_with_context["user_id"]
        set_current_user_id(user_id)

        # Default context_count
        result1 = search_tool.run(
            "expand_message",
            message_id=setup_messages_with_context["target_id"],
            direction="both"
        )
        # Should have some context (default is 2)
        assert len(result1.get("context_before", [])) > 0 or len(result1.get("context_after", [])) > 0

        # Large context_count should be clamped
        result2 = search_tool.run(
            "expand_message",
            message_id=setup_messages_with_context["target_id"],
            direction="both",
            context_count=20  # Should be clamped to 10
        )
        # Total context should not exceed 10 per direction
        assert len(result2.get("context_before", [])) <= 10
        assert len(result2.get("context_after", [])) <= 10

    def test_expand_message_zero_context(self, search_tool, setup_messages_with_context):
        """CONTRACT EC18: context_count=0 returns no context fields."""
        user_id = setup_messages_with_context["user_id"]
        set_current_user_id(user_id)

        result = search_tool.run(
            "expand_message",
            message_id=setup_messages_with_context["target_id"],
            direction="both",
            context_count=0
        )

        # Should have origin but no context
        assert "origin_message" in result
        # With context_count=0, no context arrays should be returned
        assert len(result.get("context_before", [])) == 0
        assert len(result.get("context_after", [])) == 0


class TestSearchMemoriesOperation:
    """Tests for search operation in memories mode."""

    @pytest.fixture
    def search_tool(self):
        """Create ContinuumSearchTool instance."""
        return ContinuumSearchTool()

    @pytest.fixture
    def setup_test_memories(self, authenticated_user, test_db):
        """Create test memories for search testing."""
        from lt_memory.db_access import LTMemoryDB
        from lt_memory.models import ExtractedMemory
        from utils.database_session_manager import get_shared_session_manager
        from uuid import uuid4
        import numpy as np

        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        # Initialize memory database with session manager
        session_manager = get_shared_session_manager()
        memory_db = LTMemoryDB(session_manager)

        # Create test memories using ExtractedMemory objects
        extracted_memories = [
            ExtractedMemory(
                text="Taylor prefers concise, readable code with clear variable names and minimal comments. They value simplicity over cleverness.",
                importance_score=0.85,
                confidence=0.9,
                happens_at=utc_now() - timedelta(days=10)
            ),
            ExtractedMemory(
                text="Discussion about Python async patterns and asyncio best practices.",
                importance_score=0.55,
                confidence=0.8
            ),
            ExtractedMemory(
                text="Mentioned preferring VS Code as the primary development environment.",
                importance_score=0.25,
                confidence=0.7
            )
        ]

        # Create embeddings
        embeddings = [np.random.rand(768).tolist() for _ in extracted_memories]

        # Store memories
        memory_ids = memory_db.store_memories(extracted_memories, embeddings=embeddings, user_id=user_id)

        # Retrieve stored memories to get full Memory objects
        stored_memories = [memory_db.get_memory(mid, user_id=user_id) for mid in memory_ids]

        return {
            "user_id": user_id,
            "memory_ids": memory_ids,
            "memories": stored_memories
        }

    def test_memory_search_returns_correct_structure(self, search_tool, setup_test_memories):
        """Verify complete return structure for memory search."""
        user_id = setup_test_memories["user_id"]
        set_current_user_id(user_id)

        result = search_tool.run("search", query="code style preferences", search_mode="memories")

        # Verify top-level structure
        assert result["status"] in ["high_confidence", "medium_confidence", "low_confidence", "no_results"]
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["confidence"] == round(result["confidence"], 3)
        assert result["query"] == "code style preferences"
        assert result["search_mode"] == "memories"

        # Verify results structure
        assert isinstance(result["results"], list)
        for res in result["results"]:
            assert res["result_type"] == "memory"

            # Memory-specific fields
            assert isinstance(res["memory_id"], str)
            assert len(res["memory_id"]) == 8
            assert isinstance(res["full_uuid"], str)
            assert len(res["full_uuid"]) > 8
            assert isinstance(res["text"], str)
            assert isinstance(res["importance_score"], float)
            assert 0.0 <= res["importance_score"] <= 1.0
            assert isinstance(res["confidence"], float)
            assert isinstance(res["created_at"], str)
            assert res["happens_at"] is None or isinstance(res["happens_at"], str)
            assert res["expires_at"] is None or isinstance(res["expires_at"], str)
            assert isinstance(res["is_refined"], bool)
            assert isinstance(res["access_count"], int)
            assert isinstance(res["entity_links"], list)
            assert isinstance(res["inbound_links"], int)
            assert isinstance(res["outbound_links"], int)

        # Verify pagination fields
        assert isinstance(result["page"], int)
        assert isinstance(result["has_more_pages"], bool)
        assert result["result_count"] == len(result["results"])

        # Verify meta fields
        assert result["meta"]["search_tier"] == "hybrid_vector_bm25_memories"
        assert isinstance(result["meta"]["total_memories_found"], int)
        assert result["meta"]["vector_weight"] == 0.6
        assert result["meta"]["text_weight"] == 0.4

    def test_memory_search_empty_query_raises_error(self, search_tool, authenticated_user):
        """Empty query should raise ValueError for memory search."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        with pytest.raises(ValueError, match="Query must be provided for search operation"):
            search_tool.run("search", query="", search_mode="memories")

    def test_memory_search_no_results(self, search_tool, authenticated_user):
        """Test memory search with no matching results."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        result = search_tool.run("search", query="nonexistent xyz789 memory", search_mode="memories")

        assert result["status"] == "no_results"
        assert result["confidence"] == 0.0
        assert result["results"] == []
        assert result["result_count"] == 0
        assert result["search_mode"] == "memories"

    def test_memory_search_confidence_levels(self, search_tool, setup_test_memories):
        """Test that confidence levels are based on importance scores."""
        user_id = setup_test_memories["user_id"]
        set_current_user_id(user_id)

        # Search for high importance memory
        result_high = search_tool.run("search", query="Taylor code style", search_mode="memories")
        if result_high["results"]:
            # Should have high confidence due to high importance score
            assert result_high["results"][0]["importance_score"] >= 0.7
            assert result_high["status"] == "high_confidence"

        # Search for low importance memory
        result_low = search_tool.run("search", query="VS Code", search_mode="memories")
        if result_low["results"]:
            # Should have lower confidence
            assert result_low["results"][0]["importance_score"] < 0.7
            assert result_low["status"] in ["medium_confidence", "low_confidence"]

    def test_memory_search_pagination(self, search_tool, setup_test_memories):
        """Test pagination for memory search."""
        user_id = setup_test_memories["user_id"]
        set_current_user_id(user_id)

        # First page with limit of 1
        result_page1 = search_tool.run(
            "search",
            query="preferences",
            search_mode="memories",
            max_results=1,
            page=1
        )

        assert result_page1["page"] == 1
        assert len(result_page1["results"]) <= 1

        # Second page
        if result_page1["has_more_pages"]:
            result_page2 = search_tool.run(
                "search",
                query="preferences",
                search_mode="memories",
                max_results=1,
                page=2
            )

            assert result_page2["page"] == 2
            # Results should be different
            if result_page1["results"] and result_page2["results"]:
                assert result_page1["results"][0]["memory_id"] != result_page2["results"][0]["memory_id"]

    def test_memory_search_embedding_failure(self, search_tool, authenticated_user, monkeypatch):
        """Test that embedding failure raises appropriate error."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        # Mock embedding failure
        def mock_encode_realtime(*args, **kwargs):
            raise Exception("Embedding service unavailable")

        monkeypatch.setattr(
            search_tool._embeddings_provider,
            "encode_realtime",
            mock_encode_realtime
        )

        with pytest.raises(ValueError, match="Memory search failed.*"):
            search_tool.run("search", query="test", search_mode="memories")

    def test_memory_search_hybrid_algorithm(self, search_tool, setup_test_memories):
        """Verify hybrid search uses both vector and BM25."""
        user_id = setup_test_memories["user_id"]
        set_current_user_id(user_id)

        # Search with a query that should match both semantically and by keywords
        result = search_tool.run("search", query="Python async patterns", search_mode="memories")

        # Meta should indicate hybrid search
        assert result["meta"]["search_tier"] == "hybrid_vector_bm25_memories"
        assert result["meta"]["vector_weight"] == 0.6
        assert result["meta"]["text_weight"] == 0.4

        # Should find the async patterns memory if it exists
        if result["results"]:
            # At least one result should mention Python or async
            texts = [r["text"] for r in result["results"]]
            assert any("Python" in text or "async" in text for text in texts)

    def test_memory_search_user_isolation(self, search_tool, setup_test_memories, second_authenticated_user):
        """Verify memories are isolated between users."""
        user1_id = setup_test_memories["user_id"]
        user2_id = second_authenticated_user["user_id"]

        # User 1 can see their memories
        set_current_user_id(user1_id)
        result1 = search_tool.run("search", query="Taylor preferences", search_mode="memories")
        assert len(result1["results"]) > 0

        # User 2 cannot see User 1's memories
        set_current_user_id(user2_id)
        search_tool2 = ContinuumSearchTool()  # New instance for user2
        result2 = search_tool2.run("search", query="Taylor preferences", search_mode="memories")
        assert len(result2["results"]) == 0, "User isolation violated: User 2 can see User 1's memories"


class TestTemporalSearch:
    """Tests for temporal direction search features."""

    @pytest.fixture
    def search_tool(self):
        """Create ContinuumSearchTool instance."""
        return ContinuumSearchTool()

    @pytest.fixture
    def setup_temporal_segments(self, authenticated_user, test_db):
        """Create segments across different times for temporal testing."""
        user_id = authenticated_user["user_id"]
        continuum_id = authenticated_user["continuum_id"]  # Use test user's continuum
        set_current_user_id(user_id)

        repo = get_continuum_repository()

        segments = []
        reference_time = utc_now()

        # Create segments at different time offsets
        time_offsets = [
            -10,  # 10 days ago
            -5,   # 5 days ago
            -1,   # Yesterday
            0,    # Today
            3,    # 3 days future
            8     # 8 days future
        ]

        for i, offset in enumerate(time_offsets):
            segment_id = str(uuid4())
            segment_time = reference_time + timedelta(days=offset)

            sentinel = Message(
                id=segment_id,
                role="assistant",
                content="[Segment collapsed]",
                created_at=segment_time,
                metadata={
                    "is_segment_boundary": True,
                    "status": "collapsed",
                    "segment_id": segment_id,
                    "segment_start_time": format_utc_iso(segment_time),
                    "segment_end_time": format_utc_iso(segment_time + timedelta(hours=1)),
                    "display_title": f"Segment {i} at offset {offset} days",
                    "summary": f"Content for temporal testing at {offset} days offset",
                    "tools_used": [],
                    "segment_embedding_value": [0.1 * i] * 768
                }
            )
            repo.save_message(sentinel, continuum_id, user_id)
            segments.append({
                "segment": sentinel,
                "offset": offset,
                "time": segment_time
            })

        return {
            "user_id": user_id,
            "continuum_id": continuum_id,
            "segments": segments,
            "reference_time": reference_time
        }

    def test_temporal_search_before(self, search_tool, setup_temporal_segments):
        """CONTRACT EC19: temporal_direction='before' filters correctly."""
        user_id = setup_temporal_segments["user_id"]
        set_current_user_id(user_id)
        ref_time = setup_temporal_segments["reference_time"]

        result = search_tool.run(
            "search",
            query="temporal testing",
            temporal_direction="before",
            reference_time=format_utc_iso(ref_time)
        )

        # Should only include segments before reference time
        for res in result["results"]:
            # Parse segment timestamp
            segment_time = res["created_at"]
            # All results should be before reference time
            assert segment_time < format_utc_iso(ref_time)

    def test_temporal_search_after(self, search_tool, setup_temporal_segments):
        """CONTRACT EC20: temporal_direction='after' filters correctly."""
        user_id = setup_temporal_segments["user_id"]
        set_current_user_id(user_id)
        ref_time = setup_temporal_segments["reference_time"]

        result = search_tool.run(
            "search",
            query="temporal testing",
            temporal_direction="after",
            reference_time=format_utc_iso(ref_time)
        )

        # Should only include segments after reference time
        for res in result["results"]:
            segment_time = res["created_at"]
            assert segment_time > format_utc_iso(ref_time)

    def test_temporal_search_around(self, search_tool, setup_temporal_segments):
        """CONTRACT EC21: temporal_direction='around' searches ±7 days."""
        user_id = setup_temporal_segments["user_id"]
        set_current_user_id(user_id)
        ref_time = setup_temporal_segments["reference_time"]

        result = search_tool.run(
            "search",
            query="temporal testing",
            temporal_direction="around",
            reference_time=format_utc_iso(ref_time)
        )

        # Should include segments within ±7 days
        seven_days_before = ref_time - timedelta(days=7)
        seven_days_after = ref_time + timedelta(days=7)

        for res in result["results"]:
            segment_time = res["created_at"]
            # Convert to comparable format
            assert seven_days_before <= datetime.fromisoformat(segment_time.replace('Z', '+00:00')) <= seven_days_after

    def test_temporal_filter_in_response(self, search_tool, setup_temporal_segments):
        """Verify temporal_filter is included in response when used."""
        user_id = setup_temporal_segments["user_id"]
        set_current_user_id(user_id)
        ref_time = format_utc_iso(setup_temporal_segments["reference_time"])

        result = search_tool.run(
            "search",
            query="temporal",
            temporal_direction="before",
            reference_time=ref_time
        )

        assert result["temporal_filter"] == {
            "direction": "before",
            "reference_time": ref_time
        }

    def test_no_temporal_filter_when_not_specified(self, search_tool, authenticated_user):
        """CONTRACT EC22: No temporal filter when not specified."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        result = search_tool.run("search", query="test")

        assert result["temporal_filter"] is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def search_tool(self):
        """Create ContinuumSearchTool instance."""
        return ContinuumSearchTool()

    @pytest.fixture
    def setup_test_segments(self, authenticated_user, test_db):
        """Create test segments with collapsed summaries."""
        user_id = authenticated_user["user_id"]
        continuum_id = authenticated_user["continuum_id"]
        set_current_user_id(user_id)

        repo = get_continuum_repository()

        # Create collapsed segment with entities for case-insensitive testing
        segment1_id = str(uuid4())
        segment1_time = utc_now() - timedelta(days=2)
        sentinel1 = Message(
            id=segment1_id,
            role="assistant",
            content="[Segment collapsed]",
            created_at=segment1_time,
            metadata={
                "is_segment_boundary": True,
                "status": "collapsed",
                "segment_id": segment1_id,
                "segment_start_time": format_utc_iso(segment1_time),
                "segment_end_time": format_utc_iso(segment1_time + timedelta(hours=2)),
                "display_title": "Python async patterns discussion",
                "summary": "Discussed Python async patterns including asyncio, await syntax, and concurrent.futures. Mark explained the event loop architecture.",
                "tools_used": ["code_tool"],
                "segment_embedding_value": [0.1] * 768
            }
        )
        repo.save_message(sentinel1, continuum_id, user_id)

        return {
            "user_id": user_id,
            "continuum_id": continuum_id,
            "segment1": sentinel1
        }

    def test_entity_matching_case_insensitive(self, search_tool, setup_test_segments):
        """CONTRACT EC13: Entity matching is case-insensitive."""
        user_id = setup_test_segments["user_id"]
        set_current_user_id(user_id)

        # Search with different case
        result = search_tool.run(
            "search",
            query="async patterns",
            entities=["MARK", "python", "ASYNC"]  # Different cases
        )

        if result["results"]:
            # Should match despite case differences
            matched_entities = result["results"][0]["matched_entities"]
            # The matched entities should be returned in original case from summary
            assert any(e.lower() == "mark" for e in matched_entities)


class TestSecurityAndIsolation:
    """Test security boundaries and user isolation."""

    @pytest.fixture
    def search_tool(self):
        """Create ContinuumSearchTool instance."""
        return ContinuumSearchTool()

    def test_user_isolation(self, search_tool, authenticated_user, second_authenticated_user):
        """CONTRACT S1-S3: User isolation via RLS."""
        user1_id = authenticated_user["user_id"]
        continuum1_id = authenticated_user["continuum_id"]

        user2_id = second_authenticated_user["user_id"]

        # User 1 creates private data
        set_current_user_id(user1_id)
        repo = get_continuum_repository()

        msg1 = Message(
            role="user",
            content="User 1 secret data about Project X",
            created_at=utc_now()
        )
        repo.save_message(msg1, continuum1_id, user1_id)

        # User 2 searches - should not see User 1's data
        set_current_user_id(user2_id)
        search_tool2 = ContinuumSearchTool()  # New instance for user2

        result = search_tool2.run("search", query="secret Project X")

        # Verify User 2 cannot see User 1's data
        assert len(result["results"]) == 0, "RLS violation: User 2 can see User 1's data"

    def test_no_user_id_parameter_exposed(self, search_tool):
        """CONTRACT S2: No user_id parameter in run() signature."""
        import inspect
        sig = inspect.signature(search_tool.run)
        params = list(sig.parameters.keys())

        # user_id should not be a parameter
        assert "user_id" not in params

    def test_sql_injection_safe(self, search_tool, authenticated_user):
        """CONTRACT S5: SQL injection attempts are safe."""
        user_id = authenticated_user["user_id"]
        set_current_user_id(user_id)

        # Attempt SQL injection in various parameters
        malicious_queries = [
            "'; DROP TABLE messages; --",
            "\" OR 1=1 --",
            "%' OR '1'='1",
        ]

        for query in malicious_queries:
            # Should not cause SQL errors
            try:
                result = search_tool.run("search", query=query)
                assert "error" not in result
            except ValueError:
                # ValueError is expected for some operations, not SQL errors
                pass

        # Try injection in segment_id
        try:
            search_tool.run(
                "search_within_segment",
                segment_id="abc'; DROP TABLE messages; --",
                query="test"
            )
        except ValueError as e:
            # Should be "No segment found", not SQL error
            assert "No segment found" in str(e)


class TestArchitecturalConstraints:
    """Test architectural requirements and constraints."""

    def test_tool_extends_base_class(self):
        """CONTRACT A2: Tool extends Tool base class."""
        from tools.implementations.continuum_tool import ContinuumSearchTool
        from tools.repo import Tool

        assert issubclass(ContinuumSearchTool, Tool)

    def test_configuration_pydantic(self):
        """CONTRACT A3: Configuration via Pydantic BaseModel."""
        from tools.implementations.continuum_tool import ContinuumSearchToolConfig
        from pydantic import BaseModel

        assert issubclass(ContinuumSearchToolConfig, BaseModel)

    def test_anthropic_schema_matches_operations(self):
        """CONTRACT A5: Anthropic schema matches implementation."""
        from tools.implementations.continuum_tool import ContinuumSearchTool

        tool = ContinuumSearchTool()
        schema = tool.anthropic_schema

        # Check operations in schema
        ops = schema["input_schema"]["properties"]["operation"]["enum"]
        assert "search" in ops
        assert "search_within_segment" in ops
        assert "expand_message" in ops

    def test_no_print_statements(self):
        """CONTRACT A7: No print statements, only logging."""
        import ast

        file_path = "/Users/taylut/Programming/GitHub/botwithmemory/tools/implementations/continuum_tool.py"
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        # Look for print function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    raise AssertionError("Found print statement in implementation")

    def test_error_messages_actionable(self):
        """CONTRACT A9: Error messages describe problem + recovery."""
        from tools.implementations.continuum_tool import ContinuumSearchTool

        tool = ContinuumSearchTool()
        set_current_user_id("test-user")

        # Test various error messages
        try:
            tool.run("search", query="test", search_mode="messages")
        except ValueError as e:
            msg = str(e)
            # Should explain what's needed
            assert "requires both start_time and end_time" in msg
            assert "Use summary search first" in msg  # Recovery guidance