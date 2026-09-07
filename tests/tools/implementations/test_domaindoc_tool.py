"""
Tests for DomaindocTool.

DomaindocTool provides section-aware editing for domain knowledge documents,
using SQLite storage via UserDataManager with section-level expand/collapse
and one-level subsection nesting.

Following MIRA's real testing philosophy:
- No mocks, use real SQLite with test user databases
- Test contracts, not implementation
- Verify exact return structures and error messages
"""
import pytest

from tools.implementations.domaindoc_tool import DomaindocTool
from utils.user_context import set_current_user_id, get_current_user_id
from utils.userdata_manager import get_user_data_manager, clear_manager_cache
from utils.timezone_utils import utc_now, format_utc_iso


@pytest.fixture(autouse=True)
def cleanup_manager_cache():
    """Clear UserDataManager cache after each test to prevent connection leakage."""
    yield
    clear_manager_cache()


@pytest.fixture
def domaindoc_tool():
    """Create DomaindocTool instance."""
    return DomaindocTool()


@pytest.fixture
def db(authenticated_user):
    """Get UserDataManager for test user."""
    user_id = authenticated_user["user_id"]
    return get_user_data_manager(user_id)


@pytest.fixture
def clean_domaindocs(db):
    """Ensure clean state before each test."""
    # Clean up any existing domaindocs for this user
    db.execute("DELETE FROM domaindoc_versions")
    db.execute("DELETE FROM domaindoc_sections")
    db.execute("DELETE FROM domaindocs")
    yield
    # Cleanup after test
    db.execute("DELETE FROM domaindoc_versions")
    db.execute("DELETE FROM domaindoc_sections")
    db.execute("DELETE FROM domaindocs")


@pytest.fixture
def sample_domaindoc(db, clean_domaindocs):
    """
    Create a sample domaindoc with sections for testing.

    Structure:
    - garden (enabled)
      - OVERVIEW section (sort_order=0, cannot be collapsed)
      - Plants section (sort_order=1)
      - Pests section (sort_order=2)
    """
    now = format_utc_iso(utc_now())

    # Create domaindoc
    doc_id = db.insert("domaindocs", {
        "label": "garden",
        "encrypted__description": "Track plants, pests, and suppliers",
        "enabled": True,
        "created_at": now,
        "updated_at": now
    })

    # Create sections
    db.insert("domaindoc_sections", {
        "domaindoc_id": doc_id,
        "header": "OVERVIEW",
        "encrypted__content": "This is my garden journal.",
        "sort_order": 0,
        "collapsed": False,
        "created_at": now,
        "updated_at": now
    })

    db.insert("domaindoc_sections", {
        "domaindoc_id": doc_id,
        "header": "Plants",
        "encrypted__content": "Tomatoes, peppers, carrots",
        "sort_order": 1,
        "collapsed": True,
        "created_at": now,
        "updated_at": now
    })

    db.insert("domaindoc_sections", {
        "domaindoc_id": doc_id,
        "header": "Pests",
        "encrypted__content": "Aphids, slugs",
        "sort_order": 2,
        "collapsed": True,
        "created_at": now,
        "updated_at": now
    })

    return {"doc_id": doc_id, "label": "garden"}


@pytest.fixture
def disabled_domaindoc(db, clean_domaindocs):
    """Create a disabled domaindoc."""
    now = format_utc_iso(utc_now())

    doc_id = db.insert("domaindocs", {
        "label": "work",
        "encrypted__description": "Work-related notes",
        "enabled": False,
        "created_at": now,
        "updated_at": now
    })

    return {"doc_id": doc_id, "label": "work"}


class TestDomaindocToolContract:
    """Tests that enforce DomaindocTool's contract guarantees."""

    def test_tool_name_and_schema(self, domaindoc_tool):
        """Verify tool name matches schema name."""
        assert domaindoc_tool.name == "domaindoc_tool"
        assert domaindoc_tool.anthropic_schema["name"] == "domaindoc_tool"

    def test_schema_operations(self, domaindoc_tool):
        """Verify schema includes all section-aware operations."""
        schema = domaindoc_tool.anthropic_schema
        operations = schema["input_schema"]["properties"]["operation"]["enum"]

        # Section management
        assert "expand" in operations
        assert "collapse" in operations
        assert "create_section" in operations
        assert "rename_section" in operations
        assert "delete_section" in operations
        assert "reorder_sections" in operations

        # Content editing
        assert "append" in operations
        assert "sed" in operations
        assert "sed_all" in operations
        assert "replace_section" in operations


class TestIsAvailable:
    """Tests for is_available() gated tool behavior."""

    def test_unavailable_when_no_domaindocs(self, domaindoc_tool, db, clean_domaindocs, authenticated_user):
        """is_available() returns False when no domaindocs exist."""
        assert domaindoc_tool.is_available() is False

    def test_unavailable_when_no_enabled_domains(self, domaindoc_tool, disabled_domaindoc, authenticated_user):
        """is_available() returns False when all domains are disabled."""
        assert domaindoc_tool.is_available() is False

    def test_available_when_domain_enabled(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """is_available() returns True when at least one domain is enabled."""
        assert domaindoc_tool.is_available() is True


class TestExpandCollapseOperations:
    """Tests for expand/collapse section operations."""

    def test_expand_section(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """expand sets collapsed=False on target section."""
        result = domaindoc_tool.run(
            operation="expand",
            label="garden",
            section="Plants"
        )

        assert result["success"] is True
        assert "Plants" in result["expanded"]

        # Verify in database (SQLite stores booleans as 0/1)
        section = db.fetchone(
            "SELECT collapsed FROM domaindoc_sections WHERE header = :header",
            {"header": "Plants"}
        )
        assert not section["collapsed"]  # 0 is falsy

    def test_expand_multiple_sections(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """expand can target multiple sections at once."""
        result = domaindoc_tool.run(
            operation="expand",
            label="garden",
            sections=["Plants", "Pests"]
        )

        assert result["success"] is True
        assert "Plants" in result["expanded"]
        assert "Pests" in result["expanded"]

    def test_collapse_section(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """collapse sets collapsed=True on target section."""
        # First expand Plants
        domaindoc_tool.run(operation="expand", label="garden", section="Plants")

        # Then collapse it
        result = domaindoc_tool.run(
            operation="collapse",
            label="garden",
            section="Plants"
        )

        assert result["success"] is True
        assert "Plants" in result["collapsed"]

    def test_collapse_first_section_skipped(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """First section (OVERVIEW) cannot be collapsed."""
        result = domaindoc_tool.run(
            operation="collapse",
            label="garden",
            section="OVERVIEW"
        )

        assert result["success"] is True
        assert "OVERVIEW" in result.get("skipped", [])
        assert "overview" in result.get("note", "").lower()


class TestCreateSectionOperation:
    """Tests for create_section operation."""

    def test_create_section_at_end(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """create_section adds new section at end by default."""
        result = domaindoc_tool.run(
            operation="create_section",
            label="garden",
            section="Suppliers",
            content="Local nursery, hardware store"
        )

        assert result["success"] is True
        assert result["created"] == "Suppliers"

        # Verify in database
        section = db.fetchone(
            "SELECT * FROM domaindoc_sections WHERE header = :header",
            {"header": "Suppliers"}
        )
        assert section is not None
        assert section["sort_order"] == 3  # After existing 0, 1, 2

    def test_create_section_after_specific(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """create_section can insert after a specific section."""
        result = domaindoc_tool.run(
            operation="create_section",
            label="garden",
            section="Watering",
            content="Water schedule",
            after="Plants"
        )

        assert result["success"] is True

        # Verify ordering
        sections = db.fetchall(
            "SELECT header, sort_order FROM domaindoc_sections ORDER BY sort_order"
        )
        headers = [s["header"] for s in sections]
        assert headers.index("Watering") == headers.index("Plants") + 1

    def test_create_section_requires_content(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """create_section raises ValueError when content not provided."""
        with pytest.raises(ValueError, match="requires 'content' parameter"):
            domaindoc_tool.run(
                operation="create_section",
                label="garden",
                section="NoContent"
            )


class TestSubsectionOperations:
    """Tests for subsection (nested section) operations."""

    def test_create_subsection(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """create_section with parent creates a subsection."""
        result = domaindoc_tool.run(
            operation="create_section",
            label="garden",
            section="Tomatoes",
            content="Roma, Cherry, Beefsteak",
            parent="Plants"
        )

        assert result["success"] is True
        assert result["parent"] == "Plants"

        # Verify parent_section_id is set
        parent = db.fetchone(
            "SELECT id FROM domaindoc_sections WHERE header = :header AND parent_section_id IS NULL",
            {"header": "Plants"}
        )
        subsection = db.fetchone(
            "SELECT * FROM domaindoc_sections WHERE header = :header",
            {"header": "Tomatoes"}
        )
        assert subsection["parent_section_id"] == parent["id"]

    def test_cannot_nest_deeper_than_one_level(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """Subsections cannot have children (depth limit is 1)."""
        # First create a subsection
        domaindoc_tool.run(
            operation="create_section",
            label="garden",
            section="Tomatoes",
            content="Roma, Cherry",
            parent="Plants"
        )

        # Try to create sub-subsection
        with pytest.raises(ValueError, match="Maximum nesting depth is 1"):
            domaindoc_tool.run(
                operation="create_section",
                label="garden",
                section="Roma",
                content="Red variety",
                parent="Tomatoes"
            )

    def test_cannot_add_subsections_to_overview(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """OVERVIEW section cannot have subsections."""
        with pytest.raises(ValueError, match="Cannot add subsections to the overview"):
            domaindoc_tool.run(
                operation="create_section",
                label="garden",
                section="Intro",
                content="Introduction",
                parent="OVERVIEW"
            )


class TestRenameSectionOperation:
    """Tests for rename_section operation."""

    def test_rename_section(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """rename_section changes section header."""
        result = domaindoc_tool.run(
            operation="rename_section",
            label="garden",
            section="Plants",
            new_name="Vegetables"
        )

        assert result["success"] is True
        assert result["renamed"] == "Plants"
        assert result["to"] == "Vegetables"

        # Verify in database
        old = db.fetchone(
            "SELECT * FROM domaindoc_sections WHERE header = :header",
            {"header": "Plants"}
        )
        new = db.fetchone(
            "SELECT * FROM domaindoc_sections WHERE header = :header",
            {"header": "Vegetables"}
        )
        assert old is None
        assert new is not None

    def test_rename_requires_new_name(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """rename_section raises ValueError when new_name not provided."""
        with pytest.raises(ValueError, match="requires 'new_name' parameter"):
            domaindoc_tool.run(
                operation="rename_section",
                label="garden",
                section="Plants"
            )


class TestDeleteSectionOperation:
    """Tests for delete_section operation."""

    def test_delete_section(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """delete_section removes section (must be expanded first)."""
        # Expand section first
        domaindoc_tool.run(operation="expand", label="garden", section="Pests")

        result = domaindoc_tool.run(
            operation="delete_section",
            label="garden",
            section="Pests"
        )

        assert result["success"] is True
        assert result["deleted"] == "Pests"

        # Verify removed from database
        section = db.fetchone(
            "SELECT * FROM domaindoc_sections WHERE header = :header",
            {"header": "Pests"}
        )
        assert section is None

    def test_cannot_delete_collapsed_section(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """delete_section requires section to be expanded first."""
        with pytest.raises(ValueError, match="expand.*before deleting"):
            domaindoc_tool.run(
                operation="delete_section",
                label="garden",
                section="Plants"  # This is collapsed by default
            )

    def test_cannot_delete_first_section(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """First section (overview) cannot be deleted."""
        with pytest.raises(ValueError, match="Cannot delete the first section"):
            domaindoc_tool.run(
                operation="delete_section",
                label="garden",
                section="OVERVIEW"
            )


class TestReorderSectionsOperation:
    """Tests for reorder_sections operation."""

    def test_reorder_sections(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """reorder_sections changes section order."""
        result = domaindoc_tool.run(
            operation="reorder_sections",
            label="garden",
            order=["OVERVIEW", "Pests", "Plants"]  # Swap Plants and Pests
        )

        assert result["success"] is True

        # Verify new order
        sections = db.fetchall(
            "SELECT header, sort_order FROM domaindoc_sections WHERE parent_section_id IS NULL ORDER BY sort_order"
        )
        headers = [s["header"] for s in sections]
        assert headers == ["OVERVIEW", "Pests", "Plants"]

    def test_reorder_requires_all_sections(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """reorder_sections fails if not all sections are listed."""
        with pytest.raises(ValueError, match="missing sections"):
            domaindoc_tool.run(
                operation="reorder_sections",
                label="garden",
                order=["OVERVIEW", "Plants"]  # Missing Pests
            )


class TestAppendOperation:
    """Tests for append content operation."""

    def test_append_adds_content_to_section(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """append adds content to end of section."""
        result = domaindoc_tool.run(
            operation="append",
            label="garden",
            section="Plants",
            content="- Cucumbers added"
        )

        assert result["success"] is True
        assert result["section"] == "Plants"

        # Verify content
        section = db._decrypt_dict(db.fetchone(
            "SELECT * FROM domaindoc_sections WHERE header = :header",
            {"header": "Plants"}
        ))
        assert section["encrypted__content"].endswith("- Cucumbers added")

    def test_append_adds_newline_if_missing(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """append adds newline before content if needed."""
        # Original content is "Tomatoes, peppers, carrots" (no trailing newline)
        domaindoc_tool.run(
            operation="append",
            label="garden",
            section="Plants",
            content="New line"
        )

        section = db._decrypt_dict(db.fetchone(
            "SELECT * FROM domaindoc_sections WHERE header = :header",
            {"header": "Plants"}
        ))
        assert "carrots\nNew line" in section["encrypted__content"]

    def test_append_requires_content(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """append raises ValueError when content not provided."""
        with pytest.raises(ValueError, match="requires 'content' parameter"):
            domaindoc_tool.run(operation="append", label="garden", section="Plants")


class TestSedOperation:
    """Tests for sed (find/replace) operation."""

    def test_sed_replaces_first_occurrence(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """sed replaces only first occurrence."""
        # Content is "Tomatoes, peppers, carrots"
        result = domaindoc_tool.run(
            operation="sed",
            label="garden",
            section="Plants",
            find="peppers",
            replace="bell peppers"
        )

        assert result["success"] is True
        assert result["replacements"] == 1

        section = db._decrypt_dict(db.fetchone(
            "SELECT * FROM domaindoc_sections WHERE header = :header",
            {"header": "Plants"}
        ))
        assert "bell peppers" in section["encrypted__content"]

    def test_sed_returns_not_found(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """sed returns success=False when pattern not found."""
        result = domaindoc_tool.run(
            operation="sed",
            label="garden",
            section="Plants",
            find="nonexistent",
            replace="replacement"
        )

        assert result["success"] is False
        assert "not found" in result["message"]

    def test_sed_requires_find(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """sed raises ValueError when find not provided."""
        with pytest.raises(ValueError, match="requires 'find' parameter"):
            domaindoc_tool.run(
                operation="sed",
                label="garden",
                section="Plants",
                replace="new"
            )

    def test_sed_requires_replace(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """sed raises ValueError when replace not provided."""
        with pytest.raises(ValueError, match="requires 'replace' parameter"):
            domaindoc_tool.run(
                operation="sed",
                label="garden",
                section="Plants",
                find="old"
            )


class TestSedAllOperation:
    """Tests for sed_all (global replace) operation."""

    def test_sed_all_replaces_all_occurrences(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """sed_all replaces all occurrences."""
        # Update content to have repeated pattern
        db.execute(
            "UPDATE domaindoc_sections SET encrypted__content = :content WHERE header = :header",
            {"content": "plant tomatoes, plant peppers, plant carrots", "header": "Plants"}
        )

        result = domaindoc_tool.run(
            operation="sed_all",
            label="garden",
            section="Plants",
            find="plant",
            replace="harvest"
        )

        assert result["success"] is True
        assert result["replacements"] == 3

        section = db._decrypt_dict(db.fetchone(
            "SELECT * FROM domaindoc_sections WHERE header = :header",
            {"header": "Plants"}
        ))
        assert section["encrypted__content"] == "harvest tomatoes, harvest peppers, harvest carrots"


class TestReplaceSectionOperation:
    """Tests for replace_section (full content replacement) operation."""

    def test_replace_section_replaces_entire_content(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """replace_section replaces entire section content."""
        result = domaindoc_tool.run(
            operation="replace_section",
            label="garden",
            section="Plants",
            content="Completely new content"
        )

        assert result["success"] is True
        assert result["section"] == "Plants"

        section = db._decrypt_dict(db.fetchone(
            "SELECT * FROM domaindoc_sections WHERE header = :header",
            {"header": "Plants"}
        ))
        assert section["encrypted__content"] == "Completely new content"

    def test_replace_section_allows_empty(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """replace_section allows setting content to empty string."""
        result = domaindoc_tool.run(
            operation="replace_section",
            label="garden",
            section="Plants",
            content=""
        )

        assert result["success"] is True

        section = db._decrypt_dict(db.fetchone(
            "SELECT * FROM domaindoc_sections WHERE header = :header",
            {"header": "Plants"}
        ))
        assert section["encrypted__content"] == ""


class TestErrorHandling:
    """Tests for error conditions."""

    def test_unknown_operation_raises_valueerror(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """Unknown operation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown operation"):
            domaindoc_tool.run(operation="invalid", label="garden")

    def test_nonexistent_domain_raises_valueerror(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """Non-existent domain raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            domaindoc_tool.run(
                operation="append",
                label="nonexistent",
                section="Plants",
                content="test"
            )

    def test_disabled_domain_raises_valueerror(self, domaindoc_tool, sample_domaindoc, disabled_domaindoc, authenticated_user):
        """Disabled domain raises ValueError."""
        with pytest.raises(ValueError, match="not enabled"):
            domaindoc_tool.run(
                operation="expand",
                label="work",
                section="Test"
            )

    def test_nonexistent_section_raises_valueerror(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """Non-existent section raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            domaindoc_tool.run(
                operation="append",
                label="garden",
                section="NonexistentSection",
                content="test"
            )


class TestUserIsolation:
    """Tests for user data isolation."""

    def test_tool_reads_current_user_context(self, domaindoc_tool, sample_domaindoc, authenticated_user):
        """Verify tool respects user context."""
        # With authenticated_user context and sample_domaindoc, tool should be available
        assert domaindoc_tool.is_available() is True

        # Switch to a user who has no domaindocs
        set_current_user_id("nonexistent-user-for-testing")

        # Tool should not find domaindocs for the other user
        assert domaindoc_tool.is_available() is False

        # Restore original user context for cleanup
        set_current_user_id(authenticated_user["user_id"])


class TestVersionHistory:
    """Tests for version history recording."""

    def test_operations_create_versions(self, domaindoc_tool, sample_domaindoc, authenticated_user, db):
        """Verify operations record version history."""
        doc_id = sample_domaindoc["doc_id"]

        # Perform an operation
        domaindoc_tool.run(
            operation="append",
            label="garden",
            section="Plants",
            content="New content"
        )

        # Check version was recorded
        versions = db.fetchall(
            "SELECT * FROM domaindoc_versions WHERE domaindoc_id = :doc_id",
            {"doc_id": doc_id}
        )
        assert len(versions) > 0
        assert versions[-1]["operation"] == "append"
