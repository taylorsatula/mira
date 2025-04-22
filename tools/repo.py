"""
Tool repository for the botwithmemory system.

This module provides the base Tool class and ToolRepository for managing, 
discovering, and using tools within the conversation system.
"""
import inspect
import importlib
import json
import logging
import os
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Type, Callable, Union

from errors import ToolError, ErrorCode, error_context
from config import config


class Tool(ABC):
    """
    Base class for all tools in the botwithmemory system.
    
    This class defines the standard interface and behavior that all tools
    should implement. It includes metadata, parameter handling, and execution logic.
    
    Class Attributes:
        name (str): The unique name of the tool.
        description (str): A human-readable description of the tool's purpose.
        usage_examples (List[Dict]): Example usage of the tool.
    """
    
    name = "base_tool"
    description = "Base class for all tools"
    usage_examples: List[Dict[str, Any]] = []
    
    def __init__(self):
        """Initialize a new tool instance."""
        self.logger = logging.getLogger(f"tools.{self.name}")
    
    @abstractmethod
    def run(self, **params) -> Dict[str, Any]:
        """
        Execute the tool with the provided parameters.
        
        This is the main method that should be overridden by tool implementations.
        
        Args:
            **params: Keyword arguments containing the tool's parameters.
            
        Returns:
            A dictionary containing the tool's response.
            
        Raises:
            ToolError: If tool execution fails.
        """
        raise NotImplementedError("Tool subclasses must implement the run method")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this tool.
        
        Returns:
            A dictionary containing tool metadata.
        """
        # Get signature for the run method
        sig = inspect.signature(self.run)
        parameters = {}
        required_parameters = []
        
        # Process parameters
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
                
            param_info = {
                "type": "any",
                "description": f"Parameter: {name}"
            }
            
            # Check if parameter is required
            if param.default is inspect.Parameter.empty:
                required_parameters.append(name)
            
            # Add annotation type if available
            if param.annotation is not inspect.Parameter.empty:
                param_info["type"] = str(param.annotation).replace("<class '", "").replace("'>", "")
            
            parameters[name] = param_info
        
        # Get docstring information
        if self.run.__doc__:
            doc_content = inspect.getdoc(self.run)
            if doc_content is not None:
                doc_lines = doc_content.split('\n')
            
            # Extract parameter descriptions from docstring
            param_section = False
            current_param = None
            
            for line in doc_lines:
                line = line.strip()
                
                # Check for Args: section
                if line.lower().startswith('args:'):
                    param_section = True
                    continue
                
                # Check for end of Args section (blank line or next section)
                if param_section and (not line or line.lower().startswith(('returns:', 'raises:'))):
                    param_section = False
                    current_param = None
                    continue
                
                # Process parameter descriptions
                if param_section:
                    # Check for parameter definition (name: description)
                    import re
                    param_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:(.*)$', line)
                    
                    if param_match:
                        current_param = param_match.group(1).strip()
                        description = param_match.group(2).strip()
                        
                        if current_param in parameters:
                            parameters[current_param]["description"] = description
                    
                    # Continue description for current parameter
                    elif current_param and current_param in parameters:
                        parameters[current_param]["description"] += " " + line
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
            "required_parameters": required_parameters,
            "usage_examples": self.usage_examples
        }
    
    def get_dependencies(self) -> List[str]:
        """
        Get the list of tool dependencies for this tool.
        
        Returns:
            A list of tool names that this tool depends on.
        """
        # By default, no dependencies
        return []
    
    def get_formatted_description(self) -> str:
        """
        Get a human-readable formatted description of this tool.
        
        Returns:
            A formatted string containing the tool's name, description, and parameters.
        """
        metadata = self.get_metadata()
        
        result = f"{metadata['name']}: {metadata['description']}\n"
        
        if metadata['parameters']:
            result += "Parameters:\n"
            for param_name, param_spec in metadata['parameters'].items():
                required = " (required)" if param_name in metadata['required_parameters'] else ""
                param_desc = param_spec.get("description", "No description")
                result += f"  - {param_name}{required}: {param_desc}\n"
        
        if metadata['usage_examples']:
            result += "\nExample usage:\n"
            for example in metadata['usage_examples']:
                result += f"  Input: {json.dumps(example.get('input', {}))}\n"
                result += f"  Output: {json.dumps(example.get('output', {}))}\n"
        
        return result


class ToolRepository:
    """
    Repository for managing and accessing tools.
    
    This class is responsible for registering, discovering, and resolving
    dependencies between tools.
    
    Attributes:
        tools (Dict[str, Tool]): Dictionary mapping tool names to tool instances.
        enabled_tools (Set[str]): Set of names of currently enabled tools.
    """
    
    def __init__(self):
        """Initialize a new tool repository."""
        self.logger = logging.getLogger("tool_repository")
        self.tools: Dict[str, Tool] = {}
        self.enabled_tools: Set[str] = set()
        self.tool_list_path: str = os.path.join(config.paths.data_dir, "tools", "tool_list.json")
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool in the repository.
        
        Args:
            tool: The tool instance to register.
            
        Raises:
            ToolError: If a tool with the same name is already registered.
        """
        with error_context(
            component_name="ToolRepository",
            operation=f"registering tool {tool.name}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_REGISTRATION_ERROR,
            logger=self.logger
        ):
            if tool.name in self.tools:
                raise ToolError(
                    f"Tool with name '{tool.name}' is already registered",
                    ErrorCode.TOOL_DUPLICATE_NAME
                )
                
            self.tools[tool.name] = tool
            self.logger.info(f"Registered tool: {tool.name}")
            
            # Update tool list file
            self._update_tool_list_file()
    
    def enable_tool(self, name: str) -> None:
        """
        Enable a tool for use in the conversation.
        
        Args:
            name: The name of the tool to enable.
            
        Raises:
            ToolError: If the tool is not registered.
        """
        with error_context(
            component_name="ToolRepository",
            operation=f"enabling tool {name}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_NOT_FOUND,
            logger=self.logger
        ):
            if name not in self.tools:
                raise ToolError(
                    f"Cannot enable tool '{name}': Tool not found",
                    ErrorCode.TOOL_NOT_FOUND
                )
            
            # Resolve dependencies first
            dependencies = self.resolve_dependencies(name)
            for dep_name in dependencies:
                if dep_name not in self.enabled_tools:
                    self.enable_tool(dep_name)
            
            # Now enable this tool
            self.enabled_tools.add(name)
            self.logger.info(f"Enabled tool: {name}")
    
    def disable_tool(self, name: str) -> None:
        """
        Disable a tool from use in the conversation.
        
        Args:
            name: The name of the tool to disable.
            
        Raises:
            ToolError: If the tool is not registered.
        """
        with error_context(
            component_name="ToolRepository",
            operation=f"disabling tool {name}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_NOT_FOUND,
            logger=self.logger
        ):
            if name not in self.tools:
                raise ToolError(
                    f"Cannot disable tool '{name}': Tool not found",
                    ErrorCode.TOOL_NOT_FOUND
                )
            
            if name in self.enabled_tools:
                self.enabled_tools.remove(name)
                self.logger.info(f"Disabled tool: {name}")
            else:
                self.logger.debug(f"Tool '{name}' was already disabled")
    
    def get_tool(self, name: str) -> Tool:
        """
        Get a tool instance by name.
        
        Args:
            name: The name of the tool to get.
            
        Returns:
            The tool instance.
            
        Raises:
            ToolError: If the tool is not registered.
        """
        if name not in self.tools:
            raise ToolError(
                f"Tool not found: {name}",
                ErrorCode.TOOL_NOT_FOUND
            )
            
        return self.tools[name]
    
    def invoke_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a tool by name with the given parameters.
        
        Args:
            name: The name of the tool to invoke.
            params: Dictionary of parameters to pass to the tool.
            
        Returns:
            The tool's response.
            
        Raises:
            ToolError: If the tool is not registered or not enabled.
        """
        with error_context(
            component_name="ToolRepository",
            operation=f"invoking tool {name}",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            if name not in self.tools:
                raise ToolError(
                    f"Cannot invoke tool '{name}': Tool not found",
                    ErrorCode.TOOL_NOT_FOUND
                )
                
            if name not in self.enabled_tools:
                raise ToolError(
                    f"Cannot invoke tool '{name}': Tool is not enabled",
                    ErrorCode.TOOL_NOT_ENABLED
                )
                
            tool = self.tools[name]
            self.logger.debug(f"Invoking tool: {name} with params: {params}")
            
            
            # Execute the tool
            try:
                result = tool.run(**params)
                return result
            except TypeError as e:
                # Handle parameter errors
                raise ToolError(
                    f"Invalid parameters for tool '{name}': {str(e)}",
                    ErrorCode.TOOL_INVALID_PARAMETERS
                )
    
    def list_all_tools(self) -> List[str]:
        """
        List the names of all registered tools.
        
        Returns:
            A list of tool names.
        """
        return list(self.tools.keys())
    
    def list_enabled_tools(self) -> List[str]:
        """
        List the names of all enabled tools.
        
        Returns:
            A list of tool names.
        """
        return list(self.enabled_tools)
        
    def get_enabled_tools(self) -> List[str]:
        """
        Get the names of all currently enabled tools.
        
        Returns:
            A list of enabled tool names.
        """
        return list(self.enabled_tools)
    
    def is_tool_enabled(self, name: str) -> bool:
        """
        Check if a tool is currently enabled.
        
        Args:
            name: The name of the tool to check.
            
        Returns:
            True if the tool is enabled, False otherwise.
        """
        return name in self.enabled_tools
    
    def get_tool_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific tool.
        
        Args:
            name: The name of the tool to get metadata for.
            
        Returns:
            A dictionary containing the tool's metadata.
            
        Raises:
            ToolError: If the tool is not registered.
        """
        if name not in self.tools:
            raise ToolError(
                f"Cannot get metadata for tool '{name}': Tool not found",
                ErrorCode.TOOL_NOT_FOUND
            )
            
        return self.tools[name].get_metadata()
    
    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get definitions for all enabled tools.
        
        Returns:
            A list of tool definition dictionaries suitable for Anthropic's Claude API.
        """
        definitions = []
        
        for name in self.enabled_tools:
            tool = self.tools[name]
            metadata = tool.get_metadata()
            
            # Format metadata into Anthropic's tool definition format
            definition = {
                "name": metadata["name"],
                "description": metadata["description"],
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": metadata["required_parameters"]
                }
            }
            
            # Add properties
            for param_name, param_info in metadata["parameters"].items():
                param_type = param_info.get("type", "string")
                
                # Map Python types to JSON Schema types
                if param_type in ("int", "float", "complex", "number"):
                    json_type = "number"
                elif param_type in ("bool", "boolean"):
                    json_type = "boolean"
                elif param_type in ("dict", "Dict", "dictionary", "object"):
                    json_type = "object"
                elif param_type in ("list", "List", "array"):
                    json_type = "array"
                else:
                    json_type = "string"
                
                # Create property definition
                prop_def = {
                    "type": json_type,
                    "description": param_info.get("description", f"Parameter: {param_name}")
                }
                
                # Include any additional schema properties if available
                for schema_key, schema_value in param_info.items():
                    if schema_key not in ["type", "description"]:
                        prop_def[schema_key] = schema_value
                
                # Add to input_schema properties
                definition["input_schema"]["properties"][param_name] = prop_def
            
            definitions.append(definition)
        
        return definitions
    
    def resolve_dependencies(self, tool_name: str) -> List[str]:
        """
        Resolve all dependencies for a given tool.
        
        Args:
            tool_name: The name of the tool to resolve dependencies for.
            
        Returns:
            A list of tool names that the requested tool depends on.
            
        Raises:
            ToolError: If the tool or one of its dependencies is not found.
            ToolError: If a circular dependency is detected.
        """
        visited = set()
        result = []
        
        def dfs(name):
            if name in visited:
                raise ToolError(
                    f"Circular dependency detected involving tool '{name}'",
                    ErrorCode.TOOL_CIRCULAR_DEPENDENCY
                )
                
            visited.add(name)
            
            if name not in self.tools:
                raise ToolError(
                    f"Dependency '{name}' not found",
                    ErrorCode.TOOL_NOT_FOUND
                )
                
            tool = self.tools[name]
            dependencies = tool.get_dependencies()
            
            for dep_name in dependencies:
                if dep_name not in visited:
                    dfs(dep_name)
                    result.append(dep_name)
        
        dfs(tool_name)
        return result
    
    def discover_tools(self, package_path: str = "tools") -> None:
        """
        Discover and register tools from Python modules.
        
        This method scans the given package for Python modules and attempts
        to find and register Tool subclasses defined in them.
        
        Args:
            package_path: Path to the package containing tool modules.
        """
        self.logger.info(f"Discovering tools in package: {package_path}")
        
        try:
            # Import the package to get its loader
            package = importlib.import_module(package_path)
            
            # Use pkgutil to safely iterate through modules
            for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
                module_name = module_info.name.split('.')[-1]
                
                # Skip special modules
                if module_name.startswith('_') or module_name == 'repo':
                    continue
                
                # Process the module
                self._process_module(module_info.name)
                
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Error accessing package {package_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error discovering tools: {e}")
            
    def _process_module(self, module_path: str) -> None:
        """
        Process a module to find and register Tool subclasses.
        
        Args:
            module_path: Full import path to the module
        """
        try:
            self.logger.debug(f"Importing module: {module_path}")
            module = importlib.import_module(module_path)
            
            # Find Tool subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if (inspect.isclass(attr) and 
                    issubclass(attr, Tool) and 
                    attr is not Tool and
                    attr.__module__ == module.__name__):
                    
                    self.logger.debug(f"Found Tool subclass: {attr_name}")
                    
                    # Skip registration if the class doesn't have a name
                    if not hasattr(attr, 'name') or not attr.name:
                        self.logger.warning(f"Skipping Tool class without name: {attr_name}")
                        continue
                    
                    # Instantiate and register the tool
                    try:
                        # Check if we need to inject dependencies
                        dependencies = {}
                        sig = inspect.signature(attr.__init__)
                        
                        for param_name, param in sig.parameters.items():
                            if param_name != 'self' and param.default is inspect.Parameter.empty:
                                param_type = param.annotation
                                
                                # Only handle dependencies we know how to inject
                                # Add more dependency types as needed
                                if param_type.__name__ == 'LLMBridge':
                                    from api.llm_bridge import LLMBridge
                                    # Create LLMBridge instance
                                    dependencies[param_name] = LLMBridge()
                                elif param_type.__name__ == 'ToolRepository':
                                    # Pass self reference as ToolRepository
                                    dependencies[param_name] = self  # type: ignore
                        
                        # Instantiate with dependencies
                        tool_instance = attr(**dependencies)
                        self.register_tool(tool_instance)
                        
                    except Exception as e:
                        self.logger.error(f"Error instantiating tool {attr_name}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error processing module {module_path}: {e}")
    
    def enable_tools_from_config(self) -> None:
        """
        Enable tools based on configuration.
        
        If auto_discovery is ON, enables all discovered tools.
        If auto_discovery is OFF, enables only the essential tools.
        """
        auto_discovery = config.tools.auto_discovery
        
        if auto_discovery:
            # Auto-discovery ON: Enable all tools
            self.logger.info("Auto-discovery ON: Enabling all discovered tools")
            self.enable_all_tools()
        else:
            # Auto-discovery OFF: Enable only essential tools
            essential_tools = config.tools.essential_tools
            self.logger.info(f"Auto-discovery OFF: Enabling only essential tools: {essential_tools}")
            
            for name in essential_tools:
                try:
                    self.enable_tool(name)
                except ToolError as e:
                    self.logger.error(f"Error enabling tool {name}: {e}")
    
    def enable_all_tools(self) -> None:
        """
        Enable all registered tools.
        
        This method enables all registered tools at once, which can be useful
        for initialization or testing purposes.
        """
        self.logger.info("Enabling all registered tools")
        
        for name in self.tools:
            try:
                if name not in self.enabled_tools:
                    self.enable_tool(name)
            except ToolError as e:
                self.logger.error(f"Error enabling tool {name}: {e}")
    
    def _update_tool_list_file(self) -> None:
        """
        Update the tool list file with current registered tools.
        
        This method writes the current list of registered tools to a JSON file
        for reference and tracking.
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.tool_list_path), exist_ok=True)
            
            # Create the tool list data
            tool_list = []
            
            for name, tool in self.tools.items():
                metadata = tool.get_metadata()
                tool_list.append({
                    "name": name,
                    "description": metadata["description"],
                    "parameters": metadata["parameters"],
                    "required_parameters": metadata["required_parameters"],
                    "dependencies": tool.get_dependencies()
                })
            
            # Write to file
            with open(self.tool_list_path, 'w') as f:
                json.dump(tool_list, indent=2, sort_keys=True, default=str, fp=f)
                
            self.logger.debug(f"Updated tool list file: {self.tool_list_path}")
            
        except Exception as e:
            self.logger.error(f"Error updating tool list file: {e}")