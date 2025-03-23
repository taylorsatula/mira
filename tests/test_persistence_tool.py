"""
Tests for the persistence tool.
"""
import os
import json
import pytest
from pathlib import Path
from typing import Dict, Any

from tools.persistence_tool import PersistenceTool
from errors import ToolError


@pytest.fixture
def persistence_tool():
    """Create a PersistenceTool instance for testing."""
    return PersistenceTool()


@pytest.fixture
def test_file(persistence_tool, tmp_path):
    """Create a test file with sample data."""
    # Override the base_dir for testing
    persistence_tool.base_dir = tmp_path
    
    # Test file path
    file_path = tmp_path / "test_data.json"
    
    # Sample data
    test_data = {
        "name": "Test User",
        "preferences": {
            "theme": "dark",
            "language": "en"
        },
        "history": [1, 2, 3]
    }
    
    # Write sample data to file
    with open(file_path, 'w') as f:
        json.dump(test_data, f)
    
    return "test_data.json"


def test_get_value(persistence_tool, test_file):
    """Test retrieving a value from a file."""
    # Get existing value
    result = persistence_tool.run(
        filename=test_file,
        operation="get",
        key="name"
    )
    assert result["value"] == "Test User"
    
    # Get nested value
    result = persistence_tool.run(
        filename=test_file,
        operation="get",
        key="preferences"
    )
    assert result["value"]["theme"] == "dark"
    
    # Get non-existent key
    result = persistence_tool.run(
        filename=test_file,
        operation="get",
        key="non_existent"
    )
    assert result["value"] is None
    assert "Key not found" in result["message"]


def test_set_value(persistence_tool, test_file):
    """Test setting a value in a file."""
    # Set new value
    result = persistence_tool.run(
        filename=test_file,
        operation="set",
        key="new_key",
        value="new_value"
    )
    assert result["success"] is True
    
    # Verify value was set
    result = persistence_tool.run(
        filename=test_file,
        operation="get",
        key="new_key"
    )
    assert result["value"] == "new_value"
    
    # Update existing value
    result = persistence_tool.run(
        filename=test_file,
        operation="set",
        key="name",
        value="Updated Name"
    )
    assert result["success"] is True
    
    # Verify value was updated
    result = persistence_tool.run(
        filename=test_file,
        operation="get",
        key="name"
    )
    assert result["value"] == "Updated Name"
    
    # Set complex value
    complex_value = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    result = persistence_tool.run(
        filename=test_file,
        operation="set",
        key="complex",
        value=complex_value
    )
    assert result["success"] is True
    
    # Verify complex value
    result = persistence_tool.run(
        filename=test_file,
        operation="get",
        key="complex"
    )
    assert result["value"] == complex_value


def test_delete_value(persistence_tool, test_file):
    """Test deleting a value from a file."""
    # Delete existing value
    result = persistence_tool.run(
        filename=test_file,
        operation="delete",
        key="name"
    )
    assert result["success"] is True
    
    # Verify deletion
    result = persistence_tool.run(
        filename=test_file,
        operation="get",
        key="name"
    )
    assert result["value"] is None
    
    # Delete non-existent key
    result = persistence_tool.run(
        filename=test_file,
        operation="delete",
        key="non_existent"
    )
    assert result["success"] is False
    assert "Key not found" in result["message"]


def test_list_keys(persistence_tool, test_file):
    """Test listing all keys in a file."""
    result = persistence_tool.run(
        filename=test_file,
        operation="list"
    )
    assert "keys" in result
    assert set(result["keys"]) == {"name", "preferences", "history"}


def test_new_file_creation(persistence_tool, tmp_path):
    """Test creating a new file when setting a value."""
    # Override base_dir for testing
    persistence_tool.base_dir = tmp_path
    
    # New filename
    new_file = "new_file.json"
    
    # Set value in non-existent file
    result = persistence_tool.run(
        filename=new_file,
        operation="set",
        key="first_key",
        value="first_value"
    )
    assert result["success"] is True
    
    # Verify file was created
    file_path = tmp_path / new_file
    assert file_path.exists()
    
    # Verify content
    with open(file_path, 'r') as f:
        data = json.load(f)
    assert data["first_key"] == "first_value"


def test_invalid_operation(persistence_tool):
    """Test providing an invalid operation."""
    with pytest.raises(ToolError) as e:
        persistence_tool.run(
            filename="test.json",
            operation="invalid_op",
            key="test"
        )
    assert "Invalid operation" in str(e.value)


def test_missing_key(persistence_tool):
    """Test missing key parameter for operations that require it."""
    with pytest.raises(ToolError) as e:
        persistence_tool.run(
            filename="test.json",
            operation="get"
        )
    assert "Key parameter is required" in str(e.value)
    
    with pytest.raises(ToolError) as e:
        persistence_tool.run(
            filename="test.json",
            operation="set",
            value="test"
        )
    assert "Key parameter is required" in str(e.value)


def test_missing_value(persistence_tool):
    """Test missing value parameter for set operation."""
    with pytest.raises(ToolError) as e:
        persistence_tool.run(
            filename="test.json",
            operation="set",
            key="test"
        )
    assert "Value parameter is required" in str(e.value)


def test_file_extension(persistence_tool, tmp_path):
    """Test automatic addition of .json extension."""
    # Override base_dir for testing
    persistence_tool.base_dir = tmp_path
    
    # Set value with and without extension
    persistence_tool.run(
        filename="test1",  # No extension
        operation="set",
        key="key1",
        value="value1"
    )
    
    persistence_tool.run(
        filename="test2.json",  # With extension
        operation="set",
        key="key2",
        value="value2"
    )
    
    # Verify both files exist with .json extension
    assert (tmp_path / "test1.json").exists()
    assert (tmp_path / "test2.json").exists()
    
    # Verify retrieval works with both forms
    result1 = persistence_tool.run(
        filename="test1",  # No extension
        operation="get",
        key="key1"
    )
    assert result1["value"] == "value1"
    
    result2 = persistence_tool.run(
        filename="test2.json",  # With extension
        operation="get",
        key="key2"
    )
    assert result2["value"] == "value2"