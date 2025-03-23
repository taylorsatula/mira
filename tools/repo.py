"""
Tool discovery, registration, and invocation for the AI agent system.

This module handles the interface definition for all tools, tool
discovery and registration, and tool invocation and response processing.
"""
import importlib
import inspect
import logging
import os
import pkgutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Type, Union, get_type_hints

from errors import ToolError, ErrorCode


class Tool:
    """
    Base class for all tools in the system.
    
    Defines the interface that all tools must implement and
    provides common functionality.
    """
    
    # Class properties for tool definition
    name: str = "base_tool"  # Name of the tool (must be set by subclasses)
    description: str = "Base tool class"  # Description of what the tool does
    usage_examples: List[Dict[str, Any]] = []  # Examples of how to use the tool
    
    def __init__(self):
        """Initialize the tool."""
        self.logger = logging.getLogger(f"tool.{self.name}")
    
    def run(self, **kwargs) -> Any:
        """
        Execute the tool with the provided parameters.
        
        This method must be implemented by all tool subclasses.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Tool execution result
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        raise NotImplementedError("Tool subclasses must implement 'run' method")
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """
        Get the parameter schema for the tool.
        
        This method uses type hints from the run method to generate
        a schema describing the expected parameters.
        
        Returns:
            Parameter schema as a dictionary
        """
        schema = {"type": "object", "properties": {}, "required": []}
        
        # Get type hints for the run method
        try:
            hints = get_type_hints(cls.run)
            # Remove 'return' from hints
            if "return" in hints:
                del hints["return"]
            
            # Get parameter details from docstring if available
            param_docs = {}
            if cls.run.__doc__:
                for line in cls.run.__doc__.split("\n"):
                    line = line.strip()
                    if line.startswith(":param ") or line.startswith("@param "):
                        parts = line.split(":", 2) if line.startswith(":param ") else line.split("@param ", 1)[1].split(":", 1)
                        if len(parts) >= 2:
                            param_name = parts[0].strip().replace("param ", "")
                            param_desc = parts[1].strip()
                            param_docs[param_name] = param_desc
            
            # Build schema from type hints
            for param_name, param_type in hints.items():
                param_info = {"description": param_docs.get(param_name, f"Parameter: {param_name}")}
                
                # Map Python types to JSON schema types
                if param_type == str:
                    param_info["type"] = "string"
                elif param_type == int:
                    param_info["type"] = "integer"
                elif param_type == float:
                    param_info["type"] = "number"
                elif param_type == bool:
                    param_info["type"] = "boolean"
                elif param_type == list or getattr(param_type, "__origin__", None) == list:
                    param_info["type"] = "array"
                elif param_type == dict or getattr(param_type, "__origin__", None) == dict:
                    param_info["type"] = "object"
                else:
                    # Default to string for complex types
                    param_info["type"] = "string"
                
                schema["properties"][param_name] = param_info
                
                # Mark as required if the parameter doesn't have a default value
                sig = inspect.signature(cls.run)
                if param_name in sig.parameters:
                    param = sig.parameters[param_name]
                    if param.default == inspect.Parameter.empty:
                        schema["required"].append(param_name)
        
        except Exception as e:
            logging.warning(f"Error generating parameter schema for {cls.name}: {e}")
            # Return a minimal schema if there was an error
            return {"type": "object", "properties": {}, "required": []}
        
        return schema
    
    @classmethod
    def get_tool_definition(cls) -> Dict[str, Any]:
        """
        Get the complete tool definition for Anthropic API integration.
        
        Returns:
            Tool definition dictionary following Anthropic's format
        """
        return {
            "name": cls.name,
            "description": cls.description,
            "input_schema": cls.get_parameter_schema()
        }


class ToolRepository:
    """
    Central registry and manager for all tools in the system.
    
    Handles tool discovery, registration, and invocation.
    """
    
    def __init__(self, tools_package: str = "tools"):
        """
        Initialize the tool repository.
        
        Args:
            tools_package: Python package where tools are located
        """
        self.logger = logging.getLogger("tool_repo")
        self.tools_package = tools_package
        # Store both tool classes and instances
        self.tools: Dict[str, Union[Type[Tool], Tool]] = {}
        
        # Track whether each entry is an instance or a class
        self.tool_instances: Dict[str, bool] = {}
        
        # Discover and register tools
        self.discover_tools()
        
        self.logger.info(f"Tool repository initialized with {len(self.tools)} tools")
    
    def discover_tools(self) -> None:
        """
        Discover and register all available tools.
        
        Searches the tools package for classes that inherit from Tool
        and registers them in the tool registry.
        """
        try:
            # Get the tools package
            package = importlib.import_module(self.tools_package)
            package_path = getattr(package, "__path__", [None])[0]
            
            if not package_path:
                self.logger.warning(f"Could not find tools package path for {self.tools_package}")
                return
            
            # Iterate through all modules in the package
            for _, module_name, _ in pkgutil.iter_modules([package_path]):
                # Skip the current module to avoid circular imports
                if module_name == "repo":
                    continue
                
                try:
                    # Import the module
                    module = importlib.import_module(f"{self.tools_package}.{module_name}")
                    
                    # Find all Tool subclasses in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, Tool) and obj != Tool:
                            # Register the tool
                            self.register_tool(obj)
                except Exception as e:
                    self.logger.error(f"Error loading tool module {module_name}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error discovering tools: {e}")
    
    def register_tool(self, tool: Union[Type[Tool], Tool]) -> None:
        """
        Register a tool class or instance in the repository.
        
        Args:
            tool: Tool class or instance to register
        """
        # Check if it's an instance or a class
        is_instance = not inspect.isclass(tool)
        
        # Get the tool name
        tool_name = getattr(tool, "name", None)
        
        # Validate the tool
        if not tool_name:
            class_name = tool.__class__.__name__ if is_instance else tool.__name__
            self.logger.warning(f"Skipping tool {class_name} with missing name")
            return
        
        if tool_name in self.tools:
            self.logger.warning(f"Tool with name '{tool_name}' already registered, overwriting")
        
        # Register the tool
        self.tools[tool_name] = tool
        self.tool_instances[tool_name] = is_instance
        self.logger.debug(f"Registered tool: {tool_name} ({'instance' if is_instance else 'class'})")
    
    def get_tool(self, tool_name: str) -> Optional[Union[Type[Tool], Tool]]:
        """
        Get a tool class or instance by name.
        
        Args:
            tool_name: Name of the tool to get
            
        Returns:
            Tool class, tool instance, or None if not found
        """
        return self.tools.get(tool_name)
    
    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get definitions for all registered tools.
        
        Returns:
            List of tool definitions
        """
        definitions = []
        for tool_name, tool in self.tools.items():
            # Handle both classes and instances
            is_instance = self.tool_instances.get(tool_name, False)
            if is_instance:
                # Get definition from instance method
                definitions.append(tool.get_tool_definition())
            else:
                # Get definition from class method
                definitions.append(tool.get_tool_definition())
        return definitions
    
    def invoke_tool(self, tool_name: str, tool_params: Dict[str, Any]) -> Any:
        """
        Invoke a tool by name with the provided parameters.
        
        Args:
            tool_name: Name of the tool to invoke
            tool_params: Parameters to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ToolError: If the tool is not found or execution fails
        """
        # Get the tool class or instance
        tool = self.get_tool(tool_name)
        if not tool:
            raise ToolError(
                f"Tool not found: {tool_name}",
                ErrorCode.TOOL_NOT_FOUND
            )
        
        try:
            # Determine if this is an instance or class
            is_instance = self.tool_instances.get(tool_name, False)
            
            if is_instance:
                # If it's already an instance, use it directly
                tool_instance = tool
            else:
                # If it's a class, create an instance
                tool_instance = tool()
            
            # Run the tool with the provided parameters
            result = tool_instance.run(**tool_params)
            
            return result
        
        except Exception as e:
            # Handle tool execution errors
            if isinstance(e, ToolError):
                raise
            else:
                raise ToolError(
                    f"Error executing tool {tool_name}: {e}",
                    ErrorCode.TOOL_EXECUTION_ERROR,
                    {"original_error": str(e)}
                )
    
    def __len__(self) -> int:
        """
        Get the number of registered tools.
        
        Returns:
            Number of tools
        """
        return len(self.tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool is registered, False otherwise
        """
        return tool_name in self.tools