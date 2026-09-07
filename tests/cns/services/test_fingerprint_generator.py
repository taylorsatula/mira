"""
Tests for fingerprint_generator.py - Memory fingerprint and retention parsing.

Focus: Testing the parsing, formatting, and filtering logic without LLM calls.
"""
import pytest

from cns.core.message import Message


class TestResponseParsing:
    """Tests for _parse_response() method - extracting fingerprint and retention."""

    def test_parses_fingerprint_from_tags(self):
        """CONTRACT: Extracts fingerprint content from <fingerprint> tags."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        response = """<fingerprint>
Taylor's PostgreSQL connection pooling configuration for production
</fingerprint>

<memory_retention>
[x] Taylor prefers PgBouncer
</memory_retention>"""

        previous_memories = [{"text": "Taylor prefers PgBouncer", "id": "1"}]

        fingerprint, retained = FingerprintGenerator._parse_response(
            None, response, previous_memories
        )

        assert fingerprint == "Taylor's PostgreSQL connection pooling configuration for production"

    def test_parses_retention_decisions(self):
        """CONTRACT: Extracts [x] marked texts as retained, ignores [ ] marked."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        response = """<fingerprint>query</fingerprint>

<memory_retention>
[x] Memory to keep
[ ] Memory to drop
[x] Another keeper
</memory_retention>"""

        previous_memories = [
            {"text": "Memory to keep", "id": "1"},
            {"text": "Memory to drop", "id": "2"},
            {"text": "Another keeper", "id": "3"},
        ]

        fingerprint, retained = FingerprintGenerator._parse_response(
            None, response, previous_memories
        )

        assert "Memory to keep" in retained
        assert "Another keeper" in retained
        assert "Memory to drop" not in retained
        assert len(retained) == 2

    def test_handles_response_without_fingerprint_tags(self):
        """CONTRACT: Falls back to entire response as fingerprint when no tags."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        response = "Simple expanded query without tags"

        fingerprint, retained = FingerprintGenerator._parse_response(
            None, response, None
        )

        assert fingerprint == "Simple expanded query without tags"
        assert retained == set()

    def test_fallback_keeps_all_memories_when_no_retention_block(self):
        """CONTRACT: Conservative fallback - keeps all memories when parse fails."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        response = """<fingerprint>expanded query</fingerprint>
No retention block here"""

        previous_memories = [
            {"text": "Memory A", "id": "1"},
            {"text": "Memory B", "id": "2"},
        ]

        fingerprint, retained = FingerprintGenerator._parse_response(
            None, response, previous_memories
        )

        assert fingerprint == "expanded query"
        assert retained == {"Memory A", "Memory B"}

    def test_handles_empty_previous_memories(self):
        """CONTRACT: Returns empty set when no previous memories provided."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        response = """<fingerprint>query</fingerprint>

<memory_retention>
[x] Some memory
</memory_retention>"""

        fingerprint, retained = FingerprintGenerator._parse_response(
            None, response, None
        )

        assert fingerprint == "query"
        assert retained == set()

    def test_strips_whitespace_from_parsed_content(self):
        """CONTRACT: Fingerprint and retained texts are stripped of whitespace."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        response = """<fingerprint>
   query with whitespace
</fingerprint>

<memory_retention>
[x]   Memory with spaces
</memory_retention>"""

        previous_memories = [{"text": "Memory with spaces", "id": "1"}]

        fingerprint, retained = FingerprintGenerator._parse_response(
            None, response, previous_memories
        )

        assert fingerprint == "query with whitespace"
        assert "Memory with spaces" in retained


class TestMemoryFormatting:
    """Tests for _format_previous_memories() method."""

    def test_formats_memories_into_xml_block(self):
        """CONTRACT: Creates <previous_memories> block containing memory texts."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        memories = [
            {"text": "Taylor likes coffee", "id": "1"},
            {"text": "Taylor works on MIRA", "id": "2"},
        ]

        result = FingerprintGenerator._format_previous_memories(None, memories)

        assert "<previous_memories>" in result
        assert "</previous_memories>" in result
        assert "Taylor likes coffee" in result
        assert "Taylor works on MIRA" in result

    def test_returns_empty_string_for_no_memories(self):
        """CONTRACT: Empty or None memories returns empty string."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        assert FingerprintGenerator._format_previous_memories(None, None) == ""
        assert FingerprintGenerator._format_previous_memories(None, []) == ""

    def test_skips_memories_with_empty_text(self):
        """CONTRACT: Only includes memories that have actual text content."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        memories = [
            {"text": "Valid memory", "id": "1"},
            {"text": "", "id": "2"},
            {"text": "Another valid", "id": "3"},
            {"id": "4"},  # Missing text key
        ]

        result = FingerprintGenerator._format_previous_memories(None, memories)

        assert "Valid memory" in result
        assert "Another valid" in result
        # Empty text shouldn't create blank lines between tags
        lines = [l for l in result.split('\n') if l.strip()]
        # Should have: <previous_memories>, Valid memory, Another valid, </previous_memories>
        assert len(lines) == 4


class TestSegmentSummaryFiltering:
    """Tests for _is_segment_summary() method - detecting collapsed segments."""

    def test_detects_segment_summary_correctly(self):
        """CONTRACT: Returns True for messages with is_segment_boundary=True and status='collapsed'."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        collapsed_msg = Message(
            content="[SEGMENT SUMMARY] Previous conversation about...",
            role="assistant",
            metadata={"is_segment_boundary": True, "status": "collapsed"}
        )

        assert FingerprintGenerator._is_segment_summary(None, collapsed_msg) is True

    def test_non_boundary_messages_not_filtered(self):
        """CONTRACT: Regular messages without is_segment_boundary return False."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        regular_msg = Message(
            content="Hello! How can I help?",
            role="assistant",
            metadata={}
        )

        assert FingerprintGenerator._is_segment_summary(None, regular_msg) is False

    def test_active_segment_boundary_not_filtered(self):
        """CONTRACT: Active segment boundaries (status='active') return False."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        active_segment = Message(
            content="[SEGMENT BOUNDARY]",
            role="assistant",
            metadata={"is_segment_boundary": True, "status": "active"}
        )

        assert FingerprintGenerator._is_segment_summary(None, active_segment) is False

    def test_handles_missing_metadata(self):
        """CONTRACT: Messages without metadata return False gracefully."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        # Message with no metadata attribute
        class BareMessage:
            content = "test"
            role = "assistant"

        msg = BareMessage()

        assert FingerprintGenerator._is_segment_summary(None, msg) is False

    def test_handles_none_metadata(self):
        """CONTRACT: Messages with None metadata return False."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        # Message with None metadata passed in constructor
        msg = Message(content="test", role="assistant", metadata=None)

        assert FingerprintGenerator._is_segment_summary(None, msg) is False


class TestMultilineRetention:
    """Tests for retention parsing with multi-line memory texts."""

    def test_parses_single_line_memories(self):
        """CONTRACT: Standard single-line memories are parsed correctly."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        response = """<fingerprint>query</fingerprint>

<memory_retention>
[x] Single line memory
[ ] Another single line
</memory_retention>"""

        previous = [
            {"text": "Single line memory", "id": "1"},
            {"text": "Another single line", "id": "2"},
        ]

        _, retained = FingerprintGenerator._parse_response(None, response, previous)

        assert "Single line memory" in retained
        assert "Another single line" not in retained

    def test_handles_special_characters_in_memory_text(self):
        """CONTRACT: Memory text with special chars (quotes, brackets) parsed correctly."""
        from cns.services.fingerprint_generator import FingerprintGenerator

        response = """<fingerprint>query</fingerprint>

<memory_retention>
[x] Taylor said "hello world" yesterday
[ ] Function call: foo(bar)
</memory_retention>"""

        previous = [
            {"text": 'Taylor said "hello world" yesterday', "id": "1"},
            {"text": "Function call: foo(bar)", "id": "2"},
        ]

        _, retained = FingerprintGenerator._parse_response(None, response, previous)

        assert 'Taylor said "hello world" yesterday' in retained
        assert "Function call: foo(bar)" not in retained
