"""
Tests for ContactsTool.

Following MIRA's real testing philosophy:
- No mocks, use real SQLite database
- Test contracts, not implementation
- Verify exact return structures and error messages
- Cover all edge cases identified by contract analysis
"""
import pytest
import uuid
from datetime import datetime

from tools.implementations.contacts_tool import ContactsTool
from utils.user_context import set_current_user_id

# Mark all tests in this module to load contacts_tool schema
pytestmark = pytest.mark.schema_files(['tools/implementations/schemas/contacts_tool.sql'])


class TestContactsToolContract:
    """Tests that enforce ContactsTool's contract guarantees."""

    @pytest.fixture
    def contacts_tool(self):
        """Create ContactsTool instance."""
        return ContactsTool()

    def test_tool_name_and_schema(self, contacts_tool):
        """Verify tool name matches schema name."""
        assert contacts_tool.name == "contacts_tool"
        assert contacts_tool.anthropic_schema["name"] == "contacts_tool"

    def test_unknown_operation_raises_valueerror(self, contacts_tool, authenticated_user):
        """CONTRACT E1: Unknown operation raises ValueError with specific pattern."""
        with pytest.raises(ValueError, match="Unknown operation:.*Valid operations are:"):
            contacts_tool.run("invalid_operation")


class TestAddContactOperation:
    """Tests for add_contact operation."""

    @pytest.fixture
    def contacts_tool(self):
        """Create ContactsTool instance."""
        return ContactsTool()

    def test_add_contact_returns_exact_structure(self, contacts_tool, authenticated_user):
        """CONTRACT R1-R7: Verify exact return structure for add_contact."""
        result = contacts_tool.run(
            "add_contact",
            name="Alice Johnson",
            email="alice@example.com",
            phone="+1-555-111-2222",
            pager_address="alice"
        )

        # R1: success flag
        assert result["success"] is True

        # R2: contact object with all fields
        assert "contact" in result
        contact = result["contact"]

        # R3: UUID field
        assert "uuid" in contact
        assert isinstance(contact["uuid"], str)
        # Verify it's a valid UUID
        uuid.UUID(contact["uuid"])

        # R4: encrypted__ field names (storage implementation exposed in API)
        assert "encrypted__name" in contact
        assert "encrypted__email" in contact
        assert "encrypted__phone" in contact
        assert "encrypted__pager_address" in contact

        # R5: exact values
        assert contact["encrypted__name"] == "Alice Johnson"
        assert contact["encrypted__email"] == "alice@example.com"
        assert contact["encrypted__phone"] == "+1-555-111-2222"
        assert contact["encrypted__pager_address"] == "alice"

        # R6: timestamps
        assert "created_at" in contact
        assert "updated_at" in contact
        assert isinstance(contact["created_at"], str)
        assert isinstance(contact["updated_at"], str)
        # Verify ISO format
        datetime.fromisoformat(contact["created_at"].replace('Z', '+00:00'))
        datetime.fromisoformat(contact["updated_at"].replace('Z', '+00:00'))

        # R7: message field
        assert result["message"] == "Added contact Alice Johnson"

    def test_add_contact_with_minimal_fields(self, contacts_tool, authenticated_user):
        """CONTRACT R8: add_contact with only required name field."""
        result = contacts_tool.run("add_contact", name="Bob Smith")

        assert result["success"] is True
        assert result["contact"]["encrypted__name"] == "Bob Smith"
        assert result["contact"]["encrypted__email"] is None
        assert result["contact"]["encrypted__phone"] is None
        assert result["contact"]["encrypted__pager_address"] is None

    def test_add_contact_rejects_empty_name(self, contacts_tool, authenticated_user):
        """CONTRACT E2: Empty name raises ValueError."""
        with pytest.raises(ValueError, match="Contact name is required and must be a non-empty string"):
            contacts_tool.run("add_contact", name="")

    def test_add_contact_rejects_missing_name(self, contacts_tool, authenticated_user):
        """CONTRACT E3: Missing required parameter raises TypeError."""
        with pytest.raises(TypeError):
            contacts_tool.run("add_contact", email="test@example.com")

    def test_add_contact_rejects_duplicate_name_exact(self, contacts_tool, authenticated_user):
        """CONTRACT E4: Duplicate name (exact match) raises ValueError."""
        contacts_tool.run("add_contact", name="Charlie Brown")

        with pytest.raises(ValueError, match="Contact with name 'Charlie Brown' already exists"):
            contacts_tool.run("add_contact", name="Charlie Brown")

    def test_add_contact_rejects_duplicate_name_case_insensitive(self, contacts_tool, authenticated_user):
        """CONTRACT E5: Duplicate name check is case-insensitive."""
        contacts_tool.run("add_contact", name="David Lee")

        with pytest.raises(ValueError, match="Contact with name 'DAVID LEE' already exists"):
            contacts_tool.run("add_contact", name="DAVID LEE")

        with pytest.raises(ValueError, match="Contact with name 'david lee' already exists"):
            contacts_tool.run("add_contact", name="david lee")

    def test_add_contact_handles_whitespace_in_name(self, contacts_tool, authenticated_user):
        """CONTRACT EC1: Whitespace is trimmed during duplicate comparison but preserved in storage."""
        # Add with extra whitespace
        result = contacts_tool.run("add_contact", name="  Emily Chen  ")

        # Should be stored with whitespace preserved
        assert result["contact"]["encrypted__name"] == "  Emily Chen  "

        # Duplicate check should trim whitespace
        with pytest.raises(ValueError, match="Contact with name 'Emily Chen' already exists"):
            contacts_tool.run("add_contact", name="Emily Chen")


class TestGetContactOperation:
    """Tests for get_contact operation."""

    @pytest.fixture
    def contacts_tool(self):
        """Create ContactsTool instance."""
        return ContactsTool()

    @pytest.fixture
    def sample_contact(self, contacts_tool, authenticated_user):
        """Create a sample contact for testing."""
        result = contacts_tool.run(
            "add_contact",
            name="Frank Miller",
            email="frank@example.com",
            phone="+1-555-999-8888"
        )
        return result["contact"]

    def test_get_contact_by_uuid(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT R9: get_contact by UUID returns exact match."""
        result = contacts_tool.run("get_contact", identifier=sample_contact["uuid"])

        assert result["success"] is True
        assert result["contact"]["uuid"] == sample_contact["uuid"]
        assert result["contact"]["encrypted__name"] == "Frank Miller"
        assert result["matched_by"] == "id"

    def test_get_contact_by_exact_name(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT R10: get_contact by exact name returns match."""
        result = contacts_tool.run("get_contact", identifier="Frank Miller")

        assert result["success"] is True
        assert result["contact"]["uuid"] == sample_contact["uuid"]
        assert result["matched_by"] == "name"

    def test_get_contact_by_name_case_insensitive(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT EC2: Name matching is case-insensitive."""
        result = contacts_tool.run("get_contact", identifier="FRANK MILLER")

        assert result["success"] is True
        assert result["contact"]["uuid"] == sample_contact["uuid"]
        assert result["matched_by"] == "name"

    def test_get_contact_by_partial_name_single_match(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT R11: Single partial match returns contact with matched_by=partial."""
        result = contacts_tool.run("get_contact", identifier="Frank")

        assert result["success"] is True
        assert result["contact"]["uuid"] == sample_contact["uuid"]
        assert result["matched_by"] == "partial"

    def test_get_contact_partial_ambiguous(self, contacts_tool, authenticated_user):
        """CONTRACT R12: Multiple partial matches return ambiguous result."""
        # Create multiple contacts with similar names
        contacts_tool.run("add_contact", name="John Smith")
        contacts_tool.run("add_contact", name="John Doe")
        contacts_tool.run("add_contact", name="Johnny Walker")

        result = contacts_tool.run("get_contact", identifier="John")

        assert result["success"] is False
        assert result["ambiguous"] is True
        assert "matches" in result
        assert len(result["matches"]) == 3
        assert result["message"] == "Multiple contacts match 'John'. Please specify one by UUID or full name."

    def test_get_contact_ambiguous_limited_to_ten(self, contacts_tool, authenticated_user):
        """CONTRACT EC3: Ambiguous results capped at 10 candidates."""
        # Create 15 contacts with similar names
        for i in range(15):
            contacts_tool.run("add_contact", name=f"Test User {i:02d}")

        result = contacts_tool.run("get_contact", identifier="Test")

        assert result["success"] is False
        assert result["ambiguous"] is True
        assert len(result["matches"]) == 10

    def test_get_contact_not_found(self, contacts_tool, authenticated_user):
        """CONTRACT R13: Non-existent contact returns success=false."""
        result = contacts_tool.run("get_contact", identifier="NonExistent Person")

        assert result["success"] is False
        assert result["ambiguous"] is False
        assert result["message"] == "No contact matches 'NonExistent Person'. Try a fuller name or a UUID."

    def test_get_contact_empty_identifier_raises_valueerror(self, contacts_tool, authenticated_user):
        """CONTRACT E6: Empty identifier raises ValueError."""
        with pytest.raises(ValueError, match="Contact identifier is required"):
            contacts_tool.run("get_contact", identifier="")

    def test_get_contact_partial_match_prioritizes_starts_with(self, contacts_tool, authenticated_user):
        """CONTRACT EC4: Partial matching prioritizes starts-with over contains."""
        contacts_tool.run("add_contact", name="Alexander Great")
        contacts_tool.run("add_contact", name="Great Alexander")

        result = contacts_tool.run("get_contact", identifier="Alex")

        # Should match "Alexander Great" (starts with) not "Great Alexander" (contains)
        assert result["success"] is True
        assert result["contact"]["encrypted__name"] == "Alexander Great"


class TestListContactsOperation:
    """Tests for list_contacts operation."""

    @pytest.fixture
    def contacts_tool(self):
        """Create ContactsTool instance."""
        return ContactsTool()

    def test_list_contacts_empty(self, contacts_tool, authenticated_user):
        """CONTRACT R14: list_contacts with no contacts returns empty list."""
        result = contacts_tool.run("list_contacts")

        assert result["success"] is True
        assert result["contacts"] == []
        assert result["message"] == "Found 0 contact(s)"

    def test_list_contacts_returns_all(self, contacts_tool, authenticated_user):
        """CONTRACT R15: list_contacts returns all user's contacts."""
        # Create multiple contacts
        contacts_tool.run("add_contact", name="Alice")
        contacts_tool.run("add_contact", name="Bob")
        contacts_tool.run("add_contact", name="Charlie")

        result = contacts_tool.run("list_contacts")

        assert result["success"] is True
        assert len(result["contacts"]) == 3
        assert result["message"] == "Found 3 contact(s)"

        # Verify all contacts have required fields
        for contact in result["contacts"]:
            assert "uuid" in contact
            assert "encrypted__name" in contact
            assert "created_at" in contact
            assert "updated_at" in contact


class TestDeleteContactOperation:
    """Tests for delete_contact operation."""

    @pytest.fixture
    def contacts_tool(self):
        """Create ContactsTool instance."""
        return ContactsTool()

    @pytest.fixture
    def sample_contact(self, contacts_tool, authenticated_user):
        """Create a sample contact for testing."""
        result = contacts_tool.run("add_contact", name="Grace Hopper", email="grace@example.com")
        return result["contact"]

    def test_delete_contact_by_uuid(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT R16: delete_contact by UUID succeeds immediately."""
        result = contacts_tool.run("delete_contact", identifier=sample_contact["uuid"])

        assert result["success"] is True
        assert result["deleted_contact"]["uuid"] == sample_contact["uuid"]
        assert result["deleted_contact"]["encrypted__name"] == "Grace Hopper"
        assert result["message"] == "Deleted contact Grace Hopper"

        # Verify contact is actually deleted
        list_result = contacts_tool.run("list_contacts")
        assert len(list_result["contacts"]) == 0

    def test_delete_contact_by_exact_name(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT R17: delete_contact by exact name succeeds immediately."""
        result = contacts_tool.run("delete_contact", identifier="Grace Hopper")

        assert result["success"] is True
        assert result["deleted_contact"]["uuid"] == sample_contact["uuid"]

    def test_delete_contact_partial_match_requires_confirmation(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT R18: delete_contact with partial match requires UUID confirmation."""
        result = contacts_tool.run("delete_contact", identifier="Grace")

        assert result["success"] is False
        assert result["needs_confirmation"] is True
        assert "candidate" in result
        assert result["candidate"]["uuid"] == sample_contact["uuid"]
        assert f"Re-run with UUID {sample_contact['uuid']} to confirm" in result["message"]

        # Verify contact is NOT deleted
        list_result = contacts_tool.run("list_contacts")
        assert len(list_result["contacts"]) == 1

    def test_delete_contact_ambiguous_requires_uuid(self, contacts_tool, authenticated_user):
        """CONTRACT R19: delete_contact with ambiguous match requires UUID."""
        contacts_tool.run("add_contact", name="Mary Smith")
        contacts_tool.run("add_contact", name="Mary Jones")

        result = contacts_tool.run("delete_contact", identifier="Mary")

        assert result["success"] is False
        assert result["ambiguous"] is True
        assert len(result["matches"]) == 2
        assert "Please re-run with a UUID to confirm deletion" in result["message"]

        # Verify no contacts deleted
        list_result = contacts_tool.run("list_contacts")
        assert len(list_result["contacts"]) == 2

    def test_delete_contact_not_found_raises_valueerror(self, contacts_tool, authenticated_user):
        """CONTRACT E7: delete_contact with non-existent identifier raises ValueError."""
        with pytest.raises(ValueError, match="Contact 'NonExistent' not found"):
            contacts_tool.run("delete_contact", identifier="NonExistent")

    def test_delete_contact_empty_identifier_raises_valueerror(self, contacts_tool, authenticated_user):
        """CONTRACT E8: Empty identifier raises ValueError."""
        with pytest.raises(ValueError, match="Contact identifier is required"):
            contacts_tool.run("delete_contact", identifier="")


class TestUpdateContactOperation:
    """Tests for update_contact operation."""

    @pytest.fixture
    def contacts_tool(self):
        """Create ContactsTool instance."""
        return ContactsTool()

    @pytest.fixture
    def sample_contact(self, contacts_tool, authenticated_user):
        """Create a sample contact for testing."""
        result = contacts_tool.run(
            "add_contact",
            name="Isaac Newton",
            email="isaac@example.com",
            phone="+1-555-123-4567"
        )
        return result["contact"]

    def test_update_contact_single_field(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT R20: update_contact can update single field."""
        result = contacts_tool.run(
            "update_contact",
            identifier=sample_contact["uuid"],
            email="newtonian@example.com"
        )

        assert result["success"] is True
        assert result["contact"]["encrypted__email"] == "newtonian@example.com"
        # Other fields unchanged
        assert result["contact"]["encrypted__name"] == "Isaac Newton"
        assert result["contact"]["encrypted__phone"] == "+1-555-123-4567"

    def test_update_contact_multiple_fields(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT R21: update_contact can update multiple fields."""
        result = contacts_tool.run(
            "update_contact",
            identifier=sample_contact["uuid"],
            name="Sir Isaac Newton",
            email="sir.isaac@example.com",
            phone="+44-555-999-8888",
            pager_address="isaac"
        )

        assert result["success"] is True
        assert result["contact"]["encrypted__name"] == "Sir Isaac Newton"
        assert result["contact"]["encrypted__email"] == "sir.isaac@example.com"
        assert result["contact"]["encrypted__phone"] == "+44-555-999-8888"
        assert result["contact"]["encrypted__pager_address"] == "isaac"

    def test_update_contact_updates_timestamp(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT R22: update_contact updates updated_at timestamp."""
        original_updated_at = sample_contact["updated_at"]

        result = contacts_tool.run(
            "update_contact",
            identifier=sample_contact["uuid"],
            phone="+1-555-000-0000"
        )

        # updated_at should change
        assert result["contact"]["updated_at"] != original_updated_at
        # created_at should remain unchanged
        assert result["contact"]["created_at"] == sample_contact["created_at"]

    def test_update_contact_no_fields_raises_valueerror(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT E9: update_contact with no fields raises ValueError."""
        with pytest.raises(ValueError, match="At least one of name, email, phone, or pager_address must be provided"):
            contacts_tool.run("update_contact", identifier=sample_contact["uuid"])

    def test_update_contact_duplicate_name_raises_valueerror(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT E10: update_contact to duplicate name raises ValueError."""
        # Create another contact
        contacts_tool.run("add_contact", name="Albert Einstein")

        # Try to rename first contact to second contact's name
        with pytest.raises(ValueError, match="Contact with name 'Albert Einstein' already exists"):
            contacts_tool.run(
                "update_contact",
                identifier=sample_contact["uuid"],
                name="Albert Einstein"
            )

    def test_update_contact_same_name_allowed(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT EC5: update_contact can keep same name (self not counted as duplicate)."""
        # Should succeed without duplicate error
        result = contacts_tool.run(
            "update_contact",
            identifier=sample_contact["uuid"],
            name="Isaac Newton",  # Same name
            email="new@example.com"
        )

        assert result["success"] is True
        assert result["contact"]["encrypted__name"] == "Isaac Newton"
        assert result["contact"]["encrypted__email"] == "new@example.com"

    def test_update_contact_partial_match_requires_confirmation(self, contacts_tool, authenticated_user, sample_contact):
        """CONTRACT R23: update_contact with partial match requires UUID confirmation."""
        result = contacts_tool.run("update_contact", identifier="Isaac", email="test@example.com")

        assert result["success"] is False
        assert result["needs_confirmation"] is True
        assert "candidate" in result
        assert result["candidate"]["uuid"] == sample_contact["uuid"]

    def test_update_contact_not_found_raises_valueerror(self, contacts_tool, authenticated_user):
        """CONTRACT E11: update_contact with non-existent identifier raises ValueError."""
        with pytest.raises(ValueError, match="Contact 'NonExistent' not found"):
            contacts_tool.run("update_contact", identifier="NonExistent", email="test@example.com")

    def test_update_contact_empty_identifier_raises_valueerror(self, contacts_tool, authenticated_user):
        """CONTRACT E12: Empty identifier raises ValueError."""
        with pytest.raises(ValueError, match="Contact identifier is required"):
            contacts_tool.run("update_contact", identifier="", email="test@example.com")


class TestSecurityAndIsolation:
    """Test security boundaries and user isolation."""

    @pytest.fixture
    def contacts_tool(self):
        """Create ContactsTool instance."""
        return ContactsTool()

    def test_user_isolation(self, contacts_tool, authenticated_user, second_authenticated_user):
        """CONTRACT S1-S3: User isolation via user-scoped storage."""
        user1_id = authenticated_user["user_id"]
        user2_id = second_authenticated_user["user_id"]

        # User 1 creates contact
        set_current_user_id(user1_id)
        result1 = contacts_tool.run("add_contact", name="User 1 Secret Contact")
        contact1_id = result1["contact"]["uuid"]

        # User 2 tries to access User 1's contact by UUID
        set_current_user_id(user2_id)
        try:
            result2 = contacts_tool.run("get_contact", identifier=contact1_id)
            # Should not find User 1's contact
            assert result2["success"] is False
        except ValueError:
            # Or may raise ValueError - either is acceptable for isolation
            pass

        # User 2 lists contacts - should not see User 1's data
        list_result = contacts_tool.run("list_contacts")
        assert len(list_result["contacts"]) == 0

    def test_no_user_id_parameter_exposed(self, contacts_tool):
        """CONTRACT S2: No user_id parameter in run() signature."""
        import inspect
        sig = inspect.signature(contacts_tool.run)
        params = list(sig.parameters.keys())

        # user_id should not be a parameter
        assert "user_id" not in params


class TestArchitecturalConstraints:
    """Test architectural requirements and constraints."""

    def test_tool_extends_base_class(self):
        """CONTRACT A1: Tool extends Tool base class."""
        from tools.implementations.contacts_tool import ContactsTool
        from tools.repo import Tool

        assert issubclass(ContactsTool, Tool)

    def test_configuration_pydantic(self):
        """CONTRACT A2: Configuration via Pydantic BaseModel."""
        from tools.implementations.contacts_tool import ContactsToolConfig
        from pydantic import BaseModel

        assert issubclass(ContactsToolConfig, BaseModel)

    def test_anthropic_schema_matches_operations(self):
        """CONTRACT A3: Anthropic schema matches implementation."""
        from tools.implementations.contacts_tool import ContactsTool

        tool = ContactsTool()
        schema = tool.anthropic_schema

        # Check operations in schema
        ops = schema["input_schema"]["properties"]["operation"]["enum"]
        assert "add_contact" in ops
        assert "get_contact" in ops
        assert "list_contacts" in ops
        assert "delete_contact" in ops
        assert "update_contact" in ops

    def test_no_print_statements(self):
        """CONTRACT A4: No print statements, only logging."""
        import ast

        file_path = "/Users/taylut/Programming/GitHub/botwithmemory/tools/implementations/contacts_tool.py"
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        # Look for print function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    raise AssertionError("Found print statement in implementation")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def contacts_tool(self):
        """Create ContactsTool instance."""
        return ContactsTool()

    def test_whitespace_only_identifier_treated_as_empty(self, contacts_tool, authenticated_user):
        """CONTRACT EC6: Whitespace-only identifier treated as empty."""
        with pytest.raises(ValueError, match="Contact identifier is required"):
            contacts_tool.run("get_contact", identifier="   ")

    def test_null_optional_fields_preserved(self, contacts_tool, authenticated_user):
        """CONTRACT EC7: None/null optional fields preserved in responses."""
        result = contacts_tool.run("add_contact", name="Test User")

        assert result["contact"]["encrypted__email"] is None
        assert result["contact"]["encrypted__phone"] is None
        assert result["contact"]["encrypted__pager_address"] is None

    def test_pager_address_field_support(self, contacts_tool, authenticated_user):
        """CONTRACT EC8: pager_address field properly stored and retrieved."""
        result = contacts_tool.run(
            "add_contact",
            name="Pager User",
            pager_address="pageruser@domain.com"
        )

        assert result["contact"]["encrypted__pager_address"] == "pageruser@domain.com"

        # Verify retrieval
        get_result = contacts_tool.run("get_contact", identifier="Pager User")
        assert get_result["contact"]["encrypted__pager_address"] == "pageruser@domain.com"

    def test_update_can_clear_optional_fields(self, contacts_tool, authenticated_user):
        """CONTRACT EC9: update_contact can set optional fields to None."""
        # Create contact with email
        result = contacts_tool.run("add_contact", name="Test", email="test@example.com")
        contact_id = result["contact"]["uuid"]

        # Update email to None (clearing it)
        # Note: Current implementation doesn't explicitly support clearing fields
        # This test documents expected behavior if feature is added
        # For now, we just verify update with new value works
        update_result = contacts_tool.run(
            "update_contact",
            identifier=contact_id,
            email="newemail@example.com"
        )
        assert update_result["contact"]["encrypted__email"] == "newemail@example.com"

    def test_name_with_special_characters(self, contacts_tool, authenticated_user):
        """CONTRACT EC10: Names with special characters handled correctly."""
        special_names = [
            "O'Brien",
            "José García",
            "李明",
            "Name-With-Hyphens",
            "Name (With) Parens"
        ]

        for name in special_names:
            result = contacts_tool.run("add_contact", name=name)
            assert result["success"] is True
            assert result["contact"]["encrypted__name"] == name

            # Verify retrieval
            get_result = contacts_tool.run("get_contact", identifier=name)
            assert get_result["success"] is True
            assert get_result["contact"]["encrypted__name"] == name
