"""
Persistence tool for storing and retrieving data in JSON files.

This tool provides a simple interface for storing, updating, and retrieving
information in JSON files within the persistent/ directory.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

from tools.repo import Tool
from errors import ToolError, ErrorCode, FileOperationError


class PersistenceTool(Tool):
    """
    Tool for storing and retrieving data in JSON files.

    Provides operations to save, update, and remove information
    in JSON files stored in the persistent/ directory.
    """

    name = "persistence"
    description = "Store, update, and retrieve information in JSON files"
    usage_examples = [
        {
            "input": {"filename": "user_info.json", "operation": "get", "key": "name"},
            "output": {"value": "John Doe"}
        },
        {
            "input": {"filename": "user_info.json", "operation": "set", "key": "preferences", "value": {"theme": "dark"}},
            "output": {"success": True, "message": "Data stored successfully"}
        }
    ]

    def __init__(self):
        """Initialize the persistence tool."""
        super().__init__()
        self.base_dir = Path(__file__).parent.parent / "persistent"
        self.base_dir.mkdir(exist_ok=True)

        # Create async_results directory if it doesn't exist
        async_results_dir = self.base_dir / "async_results"
        async_results_dir.mkdir(exist_ok=True)
        self.logger.info(f"Persistence tool initialized with base dir: {self.base_dir}")

    def run(
        self,
        filename: str,
        operation: str,
        key: Optional[str] = None,
        value: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Perform operations on persistent JSON data.

        Args:
            filename: Name of the JSON file to operate on (with or without .json extension)
            operation: Operation to perform (get, set, delete, list)
            key: Key to operate on (required for get, set, delete operations)
            value: Value to store (required for set operation)

        Returns:
            Dictionary with operation result

        Raises:
            ToolError: If the operation is invalid or fails
        """
        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        # Remove 'persistent/' prefix if present to prevent double nesting
        if filename.startswith('persistent/'):
            filename = filename[len('persistent/'):]

        # Validate operation
        valid_operations = ["get", "set", "delete", "list"]
        if operation not in valid_operations:
            raise ToolError(
                f"Invalid operation: {operation}. Must be one of {valid_operations}",
                ErrorCode.TOOL_INVALID_INPUT
            )

        # Validate key parameter for operations that require it
        if operation in ["get", "set", "delete"] and key is None:
            raise ToolError(
                f"Key parameter is required for '{operation}' operation",
                ErrorCode.TOOL_INVALID_INPUT
            )

        # Validate value parameter for set operation
        if operation == "set" and value is None:
            raise ToolError(
                "Value parameter is required for 'set' operation",
                ErrorCode.TOOL_INVALID_INPUT
            )

        file_path = self.base_dir / filename

        try:
            # Perform the requested operation
            if operation == "get":
                return self._get_value(file_path, key)
            elif operation == "set":
                return self._set_value(file_path, key, value)
            elif operation == "delete":
                return self._delete_value(file_path, key)
            elif operation == "list":
                return self._list_keys(file_path)

        except FileNotFoundError:
            if operation == "get":
                raise ToolError(
                    f"File not found: {filename}",
                    ErrorCode.FILE_NOT_FOUND
                )
            # For other operations, we'll create the file if needed
            return {"success": False, "message": f"File not found: {filename}"}

        except json.JSONDecodeError:
            raise ToolError(
                f"Invalid JSON in file: {filename}",
                ErrorCode.INVALID_JSON
            )

        except Exception as e:
            raise ToolError(
                f"Error during {operation} operation: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )

    def _load_data(self, file_path: Path) -> Dict[str, Any]:
        """
        Load data from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Dictionary with file contents

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r') as f:
            return json.load(f)

    def _save_data(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Save data to a JSON file.

        Args:
            file_path: Path to the JSON file
            data: Data to save

        Raises:
            FileOperationError: If writing to the file fails
        """
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write data to file with pretty printing
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True)

        except Exception as e:
            raise FileOperationError(
                f"Failed to write to file: {e}",
                ErrorCode.FILE_WRITE_ERROR
            )

    def _get_value(self, file_path: Path, key: str) -> Dict[str, Any]:
        """
        Get a value from a JSON file.

        Args:
            file_path: Path to the JSON file
            key: Key to retrieve

        Returns:
            Dictionary with the retrieved value
        """
        data = self._load_data(file_path)

        if key in data:
            return {"value": data[key]}
        else:
            return {"value": None, "message": f"Key not found: {key}"}

    def _set_value(self, file_path: Path, key: str, value: Any) -> Dict[str, Any]:
        """
        Set a value in a JSON file.

        Args:
            file_path: Path to the JSON file
            key: Key to set
            value: Value to store

        Returns:
            Dictionary with operation result
        """
        try:
            # Log the file path being used
            self.logger.info(f"Setting value in file: {file_path} with key: {key}")

            # Load existing data or create empty dict if file doesn't exist
            data = {}
            if file_path.exists():
                data = self._load_data(file_path)

            # Update data
            data[key] = value

            # Save updated data
            self._save_data(file_path, data)

            self.logger.info(f"Successfully saved data to {file_path}")
            return {"success": True, "message": "Data stored successfully"}

        except Exception as e:
            self.logger.error(f"Failed to store data in {file_path}: {str(e)}")
            return {"success": False, "message": f"Failed to store data: {str(e)}"}

    def _delete_value(self, file_path: Path, key: str) -> Dict[str, Any]:
        """
        Delete a value from a JSON file.

        Args:
            file_path: Path to the JSON file
            key: Key to delete

        Returns:
            Dictionary with operation result
        """
        # Load existing data
        data = self._load_data(file_path)

        if key in data:
            # Remove the key
            del data[key]

            # Save updated data
            self._save_data(file_path, data)

            return {"success": True, "message": f"Key '{key}' deleted successfully"}
        else:
            return {"success": False, "message": f"Key not found: {key}"}

    def _list_keys(self, file_path: Path) -> Dict[str, Any]:
        """
        List all keys in a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Dictionary with the list of keys
        """
        if not file_path.exists():
            return {"keys": []}

        data = self._load_data(file_path)
        return {"keys": list(data.keys())}
