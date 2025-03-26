"""
Persistence tool for storing and retrieving data in JSON files.

This module provides a storage system with clear separation between data storage
and retrieval operations.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

from tools.repo import Tool
from errors import ToolError, ErrorCode, FileOperationError, error_context


class PersistenceTool(Tool):
    """
    Tool for storing and retrieving data in JSON files.

    Provides clearly separated getter and setter operations for working with
    persistent data stored in JSON files.
    """

    name = "persistence"
    description = "Store and retrieve data with dedicated getter and setter operations"
    usage_examples = [
        {
            "input": {"operation": "get_data", "location": "user_info.json", "key": "name"},
            "output": {"value": "John Doe"}
        },
        {
            "input": {"operation": "set_data", "location": "user_info.json", "key": "preferences", "value": {"theme": "dark"}},
            "output": {"success": True, "message": "Data stored successfully"}
        },
        {
            "input": {"operation": "get_file", "location": "async_results/123e4567-e89b-12d3-a456-426614174000.json"},
            "output": {"value": {"task_id": "123e4567-e89b-12d3-a456-426614174000", "status": "completed", "result": "Task output"}}
        },
        {
            "input": {"operation": "set_file", "location": "async_results/task-123.json", "data": {"status": "completed", "result": "Analysis complete"}},
            "output": {"success": True, "message": "File saved successfully"}
        }
    ]

    def __init__(self):
        """Initialize the persistence tool."""
        super().__init__()
        self.base_dir = Path(__file__).parent.parent / "persistent"
        self.base_dir.mkdir(exist_ok=True)

        # Create async_results directory if it doesn't exist
        self.async_results_dir = self.base_dir / "async_results"
        self.async_results_dir.mkdir(exist_ok=True)
        self.logger.info(f"Persistence tool initialized with base dir: {self.base_dir}")

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Route to the appropriate operation method based on the operation name.

        Args:
            operation: The operation to perform (get_data, set_data, etc.)
            **kwargs: Arguments specific to each operation

        Returns:
            Dictionary with operation result

        Raises:
            ToolError: If the operation is invalid or parameters are missing
        """
        # Define valid operations and required parameters
        operations = {
            # Data operations
            "get_data": self._get_data,
            "set_data": self._set_data,
            "delete_data": self._delete_data,
            "list_keys": self._list_keys,
            
            # File operations
            "get_file": self._get_file,
            "set_file": self._set_file,
            "list_files": self._list_files
        }
        
        # Check if the operation is valid
        if operation not in operations:
            valid_ops = list(operations.keys())
            raise ToolError(
                f"Invalid operation: {operation}. Must be one of {valid_ops}",
                ErrorCode.TOOL_INVALID_INPUT
            )
        
        # Execute the operation
        try:
            return operations[operation](**kwargs)
        except TypeError as e:
            # Handle missing required parameters
            self.logger.error(f"Parameter error in {operation}: {str(e)}")
            raise ToolError(
                f"Missing or invalid parameters for {operation}: {str(e)}",
                ErrorCode.TOOL_INVALID_INPUT
            )
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            with error_context(
                component_name=self.name,
                operation=operation,
                error_class=ToolError,
                error_code=ErrorCode.INVALID_JSON,
                logger=self.logger
            ):
                raise
        except Exception as e:
            # Handle other exceptions
            self.logger.error(f"Error in {operation}: {str(e)}")
            raise ToolError(
                f"Error in {operation}: {str(e)}",
                ErrorCode.TOOL_EXECUTION_ERROR
            )

    # ----- DATA OPERATIONS -----
    
    def _get_data(self, location: str, key: str) -> Dict[str, Any]:
        """
        Get a specific value from a JSON file.
        
        Args:
            location: Path to the JSON file (relative to persistent/)
            key: Key to retrieve
            
        Returns:
            Dictionary with the retrieved value
        """
        file_path = self._resolve_path(location)
        
        try:
            # Log the operation
            self.logger.info(f"Getting value from file: {file_path} with key: {key}")
            
            # Load data
            data = self._load_data(file_path)
            
            # Return the value if it exists
            if key in data:
                self.logger.info(f"Successfully retrieved value for key: {key} from {file_path}")
                return {"value": data[key]}
            else:
                self.logger.info(f"Key not found: {key} in {file_path}")
                return {"value": None, "message": f"Key not found: {key}"}
                
        except FileNotFoundError:
            self.logger.warning(f"File not found when retrieving data: {location}")
            return {"value": None, "message": f"File not found: {location}"}

    def _set_data(self, location: str, key: str, value: Any) -> Dict[str, Any]:
        """
        Set a specific value in a JSON file.
        
        Args:
            location: Path to the JSON file (relative to persistent/)
            key: Key to set
            value: Value to store
            
        Returns:
            Dictionary with operation result
        """
        file_path = self._resolve_path(location)
        
        try:
            # Log operation
            self.logger.info(f"Setting value in file: {file_path} with key: {key}")
            
            # Load existing data or create empty dict
            data = {}
            if file_path.exists():
                try:
                    data = self._load_data(file_path)
                except (json.JSONDecodeError, FileNotFoundError):
                    # Start fresh if file is invalid
                    self.logger.warning(f"Invalid JSON in {file_path}, creating new file")
                    data = {}
            
            # Update data and save
            data[key] = value
            self._save_data(file_path, data)
            
            self.logger.info(f"Successfully saved data to {file_path}")
            return {"success": True, "message": "Data stored successfully"}
            
        except Exception as e:
            self.logger.error(f"Failed to store data in {file_path}: {str(e)}")
            return {"success": False, "message": f"Failed to store data: {str(e)}"}

    def _delete_data(self, location: str, key: str) -> Dict[str, Any]:
        """
        Delete a specific key from a JSON file.
        
        Args:
            location: Path to the JSON file (relative to persistent/)
            key: Key to delete
            
        Returns:
            Dictionary with operation result
        """
        file_path = self._resolve_path(location)
        
        try:
            # Check if file exists
            if not file_path.exists():
                return {"success": False, "message": f"File not found: {location}"}
                
            # Load data
            data = self._load_data(file_path)
            
            # Delete the key if it exists
            if key in data:
                del data[key]
                self._save_data(file_path, data)
                return {"success": True, "message": f"Key '{key}' deleted successfully"}
            else:
                return {"success": False, "message": f"Key not found: {key}"}
                
        except Exception as e:
            self.logger.error(f"Failed to delete key from {file_path}: {str(e)}")
            return {"success": False, "message": f"Failed to delete key: {str(e)}"}

    def _list_keys(self, location: str) -> Dict[str, Any]:
        """
        List all keys in a JSON file.
        
        Args:
            location: Path to the JSON file (relative to persistent/)
            
        Returns:
            Dictionary with the list of keys
        """
        file_path = self._resolve_path(location)
        
        try:
            # Check if file exists
            if not file_path.exists():
                return {"keys": [], "message": f"File not found: {location}"}
                
            # Load data and return keys
            data = self._load_data(file_path)
            return {"keys": list(data.keys())}
            
        except Exception as e:
            self.logger.error(f"Failed to list keys in {file_path}: {str(e)}")
            return {"keys": [], "message": f"Failed to list keys: {str(e)}"}

    # ----- FILE OPERATIONS -----
    
    def _get_file(self, location: str) -> Dict[str, Any]:
        """
        Get the entire contents of a JSON file.
        
        Args:
            location: Path to the JSON file (relative to persistent/)
            
        Returns:
            Dictionary with the file contents
        """
        file_path = self._resolve_path(location)
        
        try:
            # Log the operation
            self.logger.info(f"Getting entire file: {file_path}")
            
            # Check if file exists
            if not file_path.exists():
                self.logger.warning(f"File not found: {location}")
                return {"value": None, "message": f"File not found: {location}"}
                
            # Load and return all data
            data = self._load_data(file_path)
            self.logger.info(f"Successfully retrieved file: {file_path}")
            return {"value": data}
            
        except Exception as e:
            self.logger.error(f"Failed to get file {file_path}: {str(e)}")
            return {"value": None, "message": f"Failed to get file: {str(e)}"}

    def _set_file(self, location: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set the entire contents of a JSON file.
        
        Args:
            location: Path to the JSON file (relative to persistent/)
            data: Data to store in the file
            
        Returns:
            Dictionary with operation result
        """
        file_path = self._resolve_path(location)
        
        try:
            # Log operation
            self.logger.info(f"Setting entire file: {file_path}")
            
            # Save data
            self._save_data(file_path, data)
            
            self.logger.info(f"Successfully saved file to {file_path}")
            return {"success": True, "message": "File saved successfully"}
            
        except Exception as e:
            self.logger.error(f"Failed to save file {file_path}: {str(e)}")
            return {"success": False, "message": f"Failed to save file: {str(e)}"}

    def _list_files(self) -> Dict[str, Any]:
        """
        List all JSON files in the persistent directory.
        
        Returns:
            Dictionary with lists of files
        """
        try:
            # Get regular files
            regular_files = [f.name for f in self.base_dir.glob("*.json")]
            
            # Get async result files
            async_files = [f"async_results/{f.name}" for f in self.async_results_dir.glob("*.json")]
            
            # Combine and return
            all_files = regular_files + async_files
            
            return {
                "files": all_files,
                "regular_files": regular_files,
                "async_files": async_files,
                "count": len(all_files)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to list files: {str(e)}")
            return {"files": [], "message": f"Failed to list files: {str(e)}"}

    # No async-specific operations needed - using the general file operations instead

    # Legacy operations removed - fully committed to new API
    
    # ----- HELPER METHODS -----
    
    def _resolve_path(self, location: str) -> Path:
        """
        Resolve a location string to a file path.
        
        Args:
            location: Location string (can be a filename or path)
            
        Returns:
            Absolute Path object
        """
        # Ensure .json extension
        if not location.endswith('.json'):
            location = f"{location}.json"
            
        # Remove persistent/ prefix if present
        if location.startswith('persistent/'):
            location = location[len('persistent/'):]
            
        # Handle async_results directory
        if location.startswith('async_results/'):
            # Extract the task ID
            parts = location.split('/')
            if len(parts) >= 2:
                task_id = parts[1]
                if task_id.endswith('.json'):
                    task_id = task_id[:-5]  # Remove .json extension
                return self.async_results_dir / f"{task_id}.json"
                
        # Return path in base directory
        return self.base_dir / location

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
