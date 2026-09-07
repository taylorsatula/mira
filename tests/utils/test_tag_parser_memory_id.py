"""Tests for memory ID formatting utilities in tag_parser.py."""
import pytest

from utils.tag_parser import (
    format_memory_id,
    parse_memory_id,
    match_memory_id,
    MEMORY_ID_PREFIX,
    MEMORY_ID_LENGTH,
    TagParser
)


class TestFormatMemoryId:
    """Tests for format_memory_id() - UUID to mem_XXXXXXXX conversion."""

    def test_formats_uuid_with_prefix(self):
        """CONTRACT: Full UUID formats to mem_ prefix + first 8 chars."""
        result = format_memory_id("550e8400-e29b-41d4-a716-446655440001")
        assert result == "mem_550e8400"

    def test_preserves_mixed_case(self):
        """CONTRACT: Mixed case in UUID is preserved in output."""
        result = format_memory_id("5E9a8D3c-xxxx-yyyy-zzzz-xxxxxxxxxxxx")
        assert result == "mem_5E9a8D3c"

    def test_handles_uuid_without_dashes(self):
        """CONTRACT: UUID without dashes works correctly."""
        result = format_memory_id("550e8400e29b41d4a716446655440001")
        assert result == "mem_550e8400"

    def test_returns_empty_for_empty_input(self):
        """CONTRACT: Empty or None input returns empty string."""
        assert format_memory_id("") == ""
        assert format_memory_id(None) == ""

    def test_truncates_to_8_chars(self):
        """CONTRACT: Output is always mem_ prefix + exactly 8 chars."""
        result = format_memory_id("abcdefghijklmnop")
        assert result == "mem_abcdefgh"
        assert len(result) == len(MEMORY_ID_PREFIX) + MEMORY_ID_LENGTH


class TestParseMemoryId:
    """Tests for parse_memory_id() - extracting 8-char portion."""

    def test_strips_prefix(self):
        """CONTRACT: mem_ prefix is stripped from input."""
        result = parse_memory_id("mem_550e8400")
        assert result == "550e8400"

    def test_passthrough_raw_id(self):
        """CONTRACT: Raw 8-char ID without prefix passes through unchanged."""
        result = parse_memory_id("550e8400")
        assert result == "550e8400"

    def test_preserves_case(self):
        """CONTRACT: Case is preserved in output."""
        assert parse_memory_id("mem_5E9a8D3c") == "5E9a8D3c"
        assert parse_memory_id("5E9a8D3c") == "5E9a8D3c"

    def test_returns_empty_for_empty_input(self):
        """CONTRACT: Empty or None input returns empty string."""
        assert parse_memory_id("") == ""
        assert parse_memory_id(None) == ""


class TestMatchMemoryId:
    """Tests for match_memory_id() - UUID prefix matching."""

    def test_matches_with_prefix(self):
        """CONTRACT: Prefixed ID matches against full UUID."""
        assert match_memory_id(
            "550e8400-e29b-41d4-a716-446655440001",
            "mem_550e8400"
        )

    def test_matches_without_prefix(self):
        """CONTRACT: Raw 8-char ID matches against full UUID."""
        assert match_memory_id(
            "550e8400-e29b-41d4-a716-446655440001",
            "550e8400"
        )

    def test_case_insensitive_matching(self):
        """CONTRACT: Matching is case-insensitive."""
        assert match_memory_id(
            "550E8400-e29b-41d4-a716-446655440001",
            "mem_550e8400"
        )
        assert match_memory_id(
            "550e8400-e29b-41d4-a716-446655440001",
            "mem_550E8400"
        )

    def test_non_matching_ids(self):
        """CONTRACT: Non-matching IDs return False."""
        assert not match_memory_id(
            "550e8400-e29b-41d4-a716-446655440001",
            "mem_660e8400"
        )

    def test_returns_false_for_empty_inputs(self):
        """CONTRACT: Empty inputs return False."""
        assert not match_memory_id("", "mem_550e8400")
        assert not match_memory_id("550e8400-e29b-41d4-a716-446655440001", "")
        assert not match_memory_id("", "")
        assert not match_memory_id(None, "mem_550e8400")
        assert not match_memory_id("550e8400-e29b-41d4-a716-446655440001", None)


class TestTagParserMemoryRefs:
    """Tests for TagParser memory reference extraction."""

    def test_extracts_memory_refs_from_block(self):
        """CONTRACT: Extracts memory IDs from <mira:memory_refs> block."""
        parser = TagParser()
        response = """Some response text here.
<mira:memory_refs>mem_5E9a8d3C, mem_a1B2c3D4</mira:memory_refs>
<mira:my_emotion>😊</mira:my_emotion>"""

        parsed = parser.parse_response(response)

        assert len(parsed['referenced_memories']) == 2
        assert '5E9a8d3C' in parsed['referenced_memories']
        assert 'a1B2c3D4' in parsed['referenced_memories']

    def test_extracts_single_memory_ref(self):
        """CONTRACT: Works with single memory reference."""
        parser = TagParser()
        response = """Response.
<mira:memory_refs>mem_12345678</mira:memory_refs>"""

        parsed = parser.parse_response(response)

        assert len(parsed['referenced_memories']) == 1
        assert '12345678' in parsed['referenced_memories']

    def test_returns_empty_when_no_refs(self):
        """CONTRACT: Returns empty list when no memory refs present."""
        parser = TagParser()
        response = "Just a response without memory references."

        parsed = parser.parse_response(response)

        assert parsed['referenced_memories'] == []

    def test_handles_multiline_refs_block(self):
        """CONTRACT: Handles memory IDs on separate lines."""
        parser = TagParser()
        response = """Response.
<mira:memory_refs>
mem_aabbccdd
mem_11223344
</mira:memory_refs>"""

        parsed = parser.parse_response(response)

        assert len(parsed['referenced_memories']) == 2

    def test_ignores_invalid_format_ids(self):
        """CONTRACT: Only extracts properly formatted mem_XXXXXXXX hex IDs."""
        parser = TagParser()
        response = """Response.
<mira:memory_refs>mem_a1b2c3d4, invalid, mem_short, mem_e5f6a7b8</mira:memory_refs>"""

        parsed = parser.parse_response(response)

        # Only valid 8-char hex IDs after mem_ prefix
        assert 'a1b2c3d4' in parsed['referenced_memories']
        assert 'e5f6a7b8' in parsed['referenced_memories']
        assert len(parsed['referenced_memories']) == 2

    def test_removes_memory_refs_from_clean_text(self):
        """CONTRACT: memory_refs block is removed from clean_text output."""
        parser = TagParser()
        response = """Here is my response.
<mira:memory_refs>mem_12345678</mira:memory_refs>
<mira:my_emotion>😊</mira:my_emotion>"""

        parsed = parser.parse_response(response)

        assert "memory_refs" not in parsed['clean_text']
        assert "mem_12345678" not in parsed['clean_text']
        assert "Here is my response." in parsed['clean_text']
