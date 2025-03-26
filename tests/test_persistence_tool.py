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
    persistence_tool.async_results_dir = tmp_path / "async_results"
    persistence_tool.async_results_dir.mkdir(exist_ok=True)
    
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


# ----- LEGACY API TESTS -----

def test_legacy_get_value(persistence_tool, test_file):
    """Test retrieving a value from a file using legacy API."""
    # Get existing value
    result = persistence_tool.run(
        operation="get",
        filename=test_file,
        key="name"
    )
    assert result["value"] == "Test User"
    
    # Get nested value
    result = persistence_tool.run(
        operation="get",
        filename=test_file,
        key="preferences"
    )
    assert result["value"]["theme"] == "dark"
    
    # Get non-existent key
    result = persistence_tool.run(
        operation="get",
        filename=test_file,
        key="non_existent"
    )
    assert result["value"] is None
    assert "Key not found" in result["message"]


def test_legacy_set_value(persistence_tool, test_file):
    """Test setting a value in a file using legacy API."""
    # Set new value
    result = persistence_tool.run(
        operation="set",
        filename=test_file,
        key="new_key",
        value="new_value"
    )
    assert result["success"] is True
    
    # Verify value was set
    result = persistence_tool.run(
        operation="get",
        filename=test_file,
        key="new_key"
    )
    assert result["value"] == "new_value"


def test_legacy_delete_value(persistence_tool, test_file):
    """Test deleting a value from a file using legacy API."""
    # Delete existing value
    result = persistence_tool.run(
        operation="delete",
        filename=test_file,
        key="name"
    )
    assert result["success"] is True
    
    # Verify deletion
    result = persistence_tool.run(
        operation="get",
        filename=test_file,
        key="name"
    )
    assert result["value"] is None


def test_legacy_list_keys(persistence_tool, test_file):
    """Test listing all keys in a file using legacy API."""
    result = persistence_tool.run(
        operation="list",
        filename=test_file
    )
    assert "keys" in result
    assert set(result["keys"]) == {"name", "preferences", "history"}


# ----- NEW API TESTS -----

def test_get_data(persistence_tool, test_file):
    """Test retrieving data with new API."""
    # Get existing value
    result = persistence_tool.run(
        operation="get_data",
        location=test_file,
        key="name"
    )
    assert result["value"] == "Test User"
    
    # Get nested value
    result = persistence_tool.run(
        operation="get_data",
        location=test_file,
        key="preferences"
    )
    assert result["value"]["theme"] == "dark"


def test_set_data(persistence_tool, test_file):
    """Test setting data with new API."""
    # Set new value
    result = persistence_tool.run(
        operation="set_data",
        location=test_file,
        key="api_key",
        value="new_api_value"
    )
    assert result["success"] is True
    
    # Verify value was set
    result = persistence_tool.run(
        operation="get_data",
        location=test_file,
        key="api_key"
    )
    assert result["value"] == "new_api_value"
    
    # Set complex value
    complex_value = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    result = persistence_tool.run(
        operation="set_data",
        location=test_file,
        key="complex",
        value=complex_value
    )
    assert result["success"] is True
    
    # Verify complex value
    result = persistence_tool.run(
        operation="get_data",
        location=test_file,
        key="complex"
    )
    assert result["value"] == complex_value


def test_delete_data(persistence_tool, test_file):
    """Test deleting data with new API."""
    # Delete existing value
    result = persistence_tool.run(
        operation="delete_data",
        location=test_file,
        key="name"
    )
    assert result["success"] is True
    
    # Verify deletion
    result = persistence_tool.run(
        operation="get_data",
        location=test_file,
        key="name"
    )
    assert result["value"] is None
    assert "Key not found" in result["message"]


def test_list_keys_new_api(persistence_tool, test_file):
    """Test listing keys with new API."""
    result = persistence_tool.run(
        operation="list_keys",
        location=test_file
    )
    assert "keys" in result
    assert set(result["keys"]) == {"name", "preferences", "history"}


def test_get_file(persistence_tool, test_file):
    """Test getting entire file contents."""
    result = persistence_tool.run(
        operation="get_file",
        location=test_file
    )
    assert "value" in result
    assert result["value"]["name"] == "Test User"
    assert "preferences" in result["value"]
    assert "history" in result["value"]


def test_set_file(persistence_tool, tmp_path):
    """Test setting entire file contents."""
    # Override base_dir for testing
    persistence_tool.base_dir = tmp_path
    
    # Prepare test data
    test_data = {
        "version": "1.0",
        "settings": {
            "mode": "advanced",
            "notifications": True
        }
    }
    
    # Set file contents
    result = persistence_tool.run(
        operation="set_file",
        location="settings.json",
        data=test_data
    )
    assert result["success"] is True
    
    # Verify file was created
    file_path = tmp_path / "settings.json"
    assert file_path.exists()
    
    # Verify contents
    with open(file_path, 'r') as f:
        data = json.load(f)
    assert data == test_data


def test_list_files(persistence_tool, tmp_path):
    """Test listing all files."""
    # Override base_dir for testing
    persistence_tool.base_dir = tmp_path
    persistence_tool.async_results_dir = tmp_path / "async_results"
    persistence_tool.async_results_dir.mkdir(exist_ok=True)
    
    # Create some test files
    with open(tmp_path / "file1.json", 'w') as f:
        json.dump({"key": "value1"}, f)
    
    with open(tmp_path / "file2.json", 'w') as f:
        json.dump({"key": "value2"}, f)
    
    with open(tmp_path / "async_results" / "task1.json", 'w') as f:
        json.dump({"task_id": "task1", "result": "done"}, f)
    
    # List files
    result = persistence_tool.run(
        operation="list_files"
    )
    
    # Verify results
    assert "files" in result
    assert "regular_files" in result
    assert "async_files" in result
    assert len(result["regular_files"]) == 2
    assert len(result["async_files"]) == 1
    assert len(result["files"]) == 3


# ----- ASYNC RESULT TESTS -----

def test_async_result_operations(persistence_tool, tmp_path):
    """Test async result operations."""
    # Override base_dir for testing
    persistence_tool.base_dir = tmp_path
    persistence_tool.async_results_dir = tmp_path / "async_results"
    persistence_tool.async_results_dir.mkdir(exist_ok=True)
    
    # Test saving async result
    task_id = "test-task-123"
    task_result = {
        "status": "completed",
        "output": "Task completed successfully",
        "data": [1, 2, 3]
    }
    
    result = persistence_tool.run(
        operation="save_async_result",
        task_id=task_id,
        result=task_result
    )
    assert result["success"] is True
    
    # Verify file was created
    file_path = persistence_tool.async_results_dir / f"{task_id}.json"
    assert file_path.exists()
    
    # Test getting async result
    result = persistence_tool.run(
        operation="get_async_result",
        task_id=task_id
    )
    assert "value" in result
    assert result["value"]["status"] == "completed"
    assert result["value"]["output"] == "Task completed successfully"
    assert result["value"]["data"] == [1, 2, 3]
    
    # Test getting non-existent async result
    result = persistence_tool.run(
        operation="get_async_result",
        task_id="non-existent-task"
    )
    assert result["value"] is None
    assert "No results found" in result["message"]
    
    # Test listing async results
    result = persistence_tool.run(
        operation="list_async_results"
    )
    assert "task_ids" in result
    assert task_id in result["task_ids"]


# ----- ERROR HANDLING TESTS -----

def test_invalid_operation(persistence_tool):
    """Test providing an invalid operation."""
    with pytest.raises(ToolError) as e:
        persistence_tool.run(
            operation="invalid_op",
            location="test.json",
            key="test"
        )
    assert "Invalid operation" in str(e.value)


def test_missing_parameters(persistence_tool):
    """Test missing required parameters."""
    # Missing location for get_data
    with pytest.raises(ToolError) as e:
        persistence_tool.run(
            operation="get_data",
            key="test"
        )
    
    # Missing key for set_data
    with pytest.raises(ToolError) as e:
        persistence_tool.run(
            operation="set_data",
            location="test.json",
            value="test_value"
        )
    
    # Missing value for set_data
    with pytest.raises(ToolError) as e:
        persistence_tool.run(
            operation="set_data",
            location="test.json",
            key="test"
        )


def test_path_resolution(persistence_tool, tmp_path):
    """Test path resolution with different formats."""
    # Override base_dir for testing
    persistence_tool.base_dir = tmp_path
    persistence_tool.async_results_dir = tmp_path / "async_results"
    persistence_tool.async_results_dir.mkdir(exist_ok=True)
    
    # Test data
    test_data = {"key": "value"}
    
    # Test different path formats
    paths = [
        "test_file",                  # No extension
        "test_file.json",             # With extension
        "persistent/test_file",       # With persistent prefix
        "persistent/test_file.json",  # With persistent prefix and extension
    ]
    
    for i, path in enumerate(paths):
        # Set data
        persistence_tool.run(
            operation="set_data",
            location=path,
            key=f"key{i}",
            value=f"value{i}"
        )
        
        # Verify we can retrieve it
        result = persistence_tool.run(
            operation="get_data",
            location=path,
            key=f"key{i}"
        )
        assert result["value"] == f"value{i}"
    
    # Verify only one file was created (all paths resolved to the same file)
    assert (tmp_path / "test_file.json").exists()
    
    # Test async_results path format
    async_path = "async_results/task-xyz"
    
    # Set data
    persistence_tool.run(
        operation="set_data",
        location=async_path,
        key="status",
        value="done"
    )
    
    # Verify we can retrieve it
    result = persistence_tool.run(
        operation="get_data",
        location=async_path,
        key="status"
    )
    assert result["value"] == "done"
    
    # Verify file was created in async_results directory
    assert (persistence_tool.async_results_dir / "task-xyz.json").exists()