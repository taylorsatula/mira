"""
File operations (Create, Read, Update, Delete) for the AI agent system.

This module provides utilities for working with JSON files, including
path management, file handling, error handling, and data validation.
"""
import json
import os
from pathlib import Path
from typing import List, Any, Union, Optional, Callable

from errors import FileOperationError, ErrorCode


class FileOperations:
    """
    Handles file operations for the AI agent system.

    Provides methods for creating, reading, updating, and deleting
    JSON files, along with path management and error handling.
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the file operations handler.

        Args:
            data_dir: Directory for storing data files
        """
        self.data_dir = Path(data_dir)
        self._ensure_data_dir()

    def _ensure_data_dir(self) -> None:
        """
        Ensure the data directory exists.

        Creates the directory if it doesn't exist.

        Raises:
            FileOperationError: If the directory cannot be created
        """
        try:
            os.makedirs(self.data_dir, exist_ok=True)
        except Exception as e:
            raise FileOperationError(
                f"Failed to create data directory: {e}",
                ErrorCode.FILE_PERMISSION_ERROR
            )

    def _get_file_path(self, file_name: str) -> Path:
        """
        Get the full path for a file.

        Args:
            file_name: Name of the file

        Returns:
            Path object for the file
        """
        # Ensure the file name has a .json extension
        if not file_name.endswith('.json'):
            file_name = f"{file_name}.json"

        return self.data_dir / file_name

    def create(self, file_name: str, data: Any) -> None:
        """
        Create a new JSON file with the provided data.

        Args:
            file_name: Name of the file to create
            data: Data to write to the file

        Raises:
            FileOperationError: If the file already exists or cannot be written
        """
        file_path = self._get_file_path(file_name)

        if file_path.exists():
            raise FileOperationError(
                f"File already exists: {file_path}",
                ErrorCode.FILE_WRITE_ERROR
            )

        self.write(file_name, data)

    def read(self, file_name: str) -> Any:
        """
        Read data from a JSON file.

        Args:
            file_name: Name of the file to read

        Returns:
            Data from the file

        Raises:
            FileOperationError: If the file doesn't exist or cannot be read
        """
        file_path = self._get_file_path(file_name)

        if not file_path.exists():
            raise FileOperationError(
                f"File not found: {file_path}",
                ErrorCode.FILE_NOT_FOUND
            )

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise FileOperationError(
                f"Invalid JSON in file {file_path}: {e}",
                ErrorCode.INVALID_JSON
            )
        except Exception as e:
            raise FileOperationError(
                f"Error reading file {file_path}: {e}",
                ErrorCode.FILE_READ_ERROR
            )

    def write(self, file_name: str, data: Any) -> None:
        """
        Write data to a JSON file.

        Args:
            file_name: Name of the file to write
            data: Data to write to the file

        Raises:
            FileOperationError: If the file cannot be written
        """
        file_path = self._get_file_path(file_name)

        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise FileOperationError(
                f"Error writing to file {file_path}: {e}",
                ErrorCode.FILE_WRITE_ERROR
            )

    def update(self, file_name: str, update_func: Callable[[Any], Any]) -> Any:
        """
        Update a JSON file using a function.

        Args:
            file_name: Name of the file to update
            update_func: Function that takes the current data and returns updated data

        Returns:
            Updated data

        Raises:
            FileOperationError: If the file doesn't exist or cannot be updated
        """
        data = self.read(file_name)
        updated_data = update_func(data)
        self.write(file_name, updated_data)
        return updated_data

    def delete(self, file_name: str) -> None:
        """
        Delete a JSON file.

        Args:
            file_name: Name of the file to delete

        Raises:
            FileOperationError: If the file doesn't exist or cannot be deleted
        """
        file_path = self._get_file_path(file_name)

        if not file_path.exists():
            raise FileOperationError(
                f"File not found: {file_path}",
                ErrorCode.FILE_NOT_FOUND
            )

        try:
            os.remove(file_path)
        except Exception as e:
            raise FileOperationError(
                f"Error deleting file {file_path}: {e}",
                ErrorCode.FILE_PERMISSION_ERROR
            )

    def list_files(self, pattern: Optional[str] = None) -> List[str]:
        """
        List JSON files in the data directory.

        Args:
            pattern: Optional glob pattern to filter files

        Returns:
            List of file names (without .json extension)
        """
        if pattern:
            if not pattern.endswith('.json'):
                pattern = f"{pattern}.json"
            files = list(self.data_dir.glob(pattern))
        else:
            files = list(self.data_dir.glob("*.json"))

        # Return file names without .json extension
        return [file.stem for file in files]
