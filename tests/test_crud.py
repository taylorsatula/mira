"""
Tests for the file operations (CRUD) module.

This module tests the FileOperations class, including file creation,
reading, updating, deletion, and error handling.
"""
import pytest
import os
import json
from pathlib import Path

from crud import FileOperations
from errors import FileOperationError, ErrorCode


def test_file_operations_init(temp_dir):
    """Test initializing FileOperations and creating data directory."""
    # Create FileOperations with a new directory
    file_ops = FileOperations(temp_dir / "new_dir")
    
    # Check that the directory was created
    assert (temp_dir / "new_dir").exists()
    assert (temp_dir / "new_dir").is_dir()


def test_file_path_generation(file_ops):
    """Test getting file paths with and without extensions."""
    # Without .json extension
    path = file_ops._get_file_path("test_file")
    assert path.name == "test_file.json"
    
    # With .json extension
    path = file_ops._get_file_path("test_file.json")
    assert path.name == "test_file.json"


def test_file_create(file_ops):
    """Test creating a new file."""
    # Test data
    data = {"key": "value"}
    
    # Create a file
    file_ops.create("test_file", data)
    
    # Check that the file exists
    file_path = file_ops._get_file_path("test_file")
    assert file_path.exists()
    
    # Check the file contents
    with open(file_path, "r") as f:
        assert json.load(f) == data
    
    # Creating an existing file should raise an error
    with pytest.raises(FileOperationError) as excinfo:
        file_ops.create("test_file", data)
    assert excinfo.value.code == ErrorCode.FILE_WRITE_ERROR


def test_file_read(file_ops):
    """Test reading a file."""
    # Test data
    data = {"key": "value"}
    
    # Create a file
    file_ops.create("test_file", data)
    
    # Read the file
    read_data = file_ops.read("test_file")
    assert read_data == data
    
    # Reading a non-existent file should raise an error
    with pytest.raises(FileOperationError) as excinfo:
        file_ops.read("non_existent_file")
    assert excinfo.value.code == ErrorCode.FILE_NOT_FOUND


def test_file_update(file_ops):
    """Test updating a file."""
    # Test data
    initial_data = {"key": "value"}
    
    # Create a file
    file_ops.create("test_file", initial_data)
    
    # Update function
    def update_func(data):
        data["key"] = "new_value"
        data["new_key"] = "added_value"
        return data
    
    # Update the file
    updated_data = file_ops.update("test_file", update_func)
    
    # Check the updated data
    assert updated_data["key"] == "new_value"
    assert updated_data["new_key"] == "added_value"
    
    # Check that the file was updated
    read_data = file_ops.read("test_file")
    assert read_data == updated_data


def test_file_write(file_ops):
    """Test writing to a file."""
    # Test data
    data = {"key": "value"}
    
    # Write to a new file
    file_ops.write("test_file", data)
    
    # Check that the file exists and contains the data
    assert file_ops.read("test_file") == data
    
    # Write to an existing file
    new_data = {"key": "new_value"}
    file_ops.write("test_file", new_data)
    
    # Check that the file was updated
    assert file_ops.read("test_file") == new_data


def test_file_delete(file_ops):
    """Test deleting a file."""
    # Create a file
    file_ops.create("test_file", {"key": "value"})
    
    # Check that the file exists
    file_path = file_ops._get_file_path("test_file")
    assert file_path.exists()
    
    # Delete the file
    file_ops.delete("test_file")
    
    # Check that the file was deleted
    assert not file_path.exists()
    
    # Deleting a non-existent file should raise an error
    with pytest.raises(FileOperationError) as excinfo:
        file_ops.delete("non_existent_file")
    assert excinfo.value.code == ErrorCode.FILE_NOT_FOUND


def test_file_list(file_ops):
    """Test listing files."""
    # Create some files
    file_ops.create("file1", {"key": "value1"})
    file_ops.create("file2", {"key": "value2"})
    file_ops.create("other_file", {"key": "value3"})
    
    # List all files
    files = file_ops.list_files()
    assert sorted(files) == ["file1", "file2", "other_file"]
    
    # List files with a pattern
    files = file_ops.list_files("file*")
    assert sorted(files) == ["file1", "file2"]


def test_invalid_json(file_ops, temp_dir):
    """Test handling invalid JSON."""
    # Create a file with invalid JSON
    file_path = file_ops._get_file_path("invalid_file")
    with open(file_path, "w") as f:
        f.write("{ invalid json ")
    
    # Reading the file should raise an error
    with pytest.raises(FileOperationError) as excinfo:
        file_ops.read("invalid_file")
    assert excinfo.value.code == ErrorCode.INVALID_JSON


def test_file_permission_error(monkeypatch, file_ops):
    """Test handling file permission errors."""
    # Mock os.remove to raise a permission error
    def mock_remove(path):
        raise PermissionError("Permission denied")
    
    monkeypatch.setattr(os, "remove", mock_remove)
    
    # Create a file
    file_ops.create("test_file", {"key": "value"})
    
    # Deleting the file should raise a FileOperationError
    with pytest.raises(FileOperationError) as excinfo:
        file_ops.delete("test_file")
    assert excinfo.value.code == ErrorCode.FILE_PERMISSION_ERROR