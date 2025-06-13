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
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Type, Callable, Union

from pydantic import BaseModel, create_model
from errors import ToolError, ErrorCode, error_context

# Import the registry (which is initialized before any tools)
from config.registry import registry

# Lock for thread safety
_repo_lock = threading.Lock()

# Deferred import for config
def get_config():
    """Get the config singleton, importing it only when needed."""
    from config import config
    return config


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
    
    @classmethod
    def register_config(cls, config_class: Type[BaseModel]) -> None:
        """
        Register a tool configuration class.
        
        This method should be called by tool implementations to register
        their configuration class with the configuration registry. It can
        be called in the class body for immediate registration.
        
        Args:
            config_class: The Pydantic model class for this tool's configuration
        """
        registry.register(cls.name, config_class)
    
    def __init__(self):
        """Initialize a new tool instance with automatic config registration."""
        self.logger = logging.getLogger(f"tools.{self.name}")
        
        # Auto-register a default config if this tool doesn't have one
        if self.name not in registry._registry:
            self.logger.debug(f"Auto-registering default config for tool: {self.name}")
            
            # Generate appropriate class name
            class_name = f"{self.name.capitalize()}Config"
            if self.name.endswith('_tool'):
                # Convert snake_case to CamelCase
                parts = self.name.split('_')
                class_name = ''.join(part.capitalize() for part in parts[:-1]) + 'ToolConfig'
            
            # Create the model
            default_config = create_model(
                class_name,
                __base__=BaseModel,
                enabled=(bool, True),
                __doc__=f"Default configuration for {self.name}"
            )
            
            # Register the config class
            self.__class__.register_config(default_config)
    
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
        working_memory (Optional[WorkingMemory]): WorkingMemory instance for managing tool info in system prompts.
    """

    def __init__(self, working_memory=None):
        """
        Initialize a new tool repository.

        Args:
            working_memory: Optional WorkingMemory instance for managing tool info in system prompts.
        """
        self.logger = logging.getLogger("tool_repository")
        self.tools: Dict[str, Tool] = {}
        self.enabled_tools: Set[str] = set()
        self.working_memory = working_memory
        self._tool_guidance_id = None
        config = get_config()
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
            
            # Create the tool's data directory in data/tools/{tool_name}/
            config = get_config()
            tool_data_dir = os.path.join(config.paths.data_dir, "tools", tool.name)
            try:
                os.makedirs(tool_data_dir, exist_ok=True)
                self.logger.debug(f"Created or verified tool data directory: {tool_data_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to create tool data directory for {tool.name}: {e}")
            
            # Update tool list file
            self._update_tool_list_file()

            # Update tool guidance in working memory
            self._update_tool_guidance()
    
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

            # Update tool guidance in working memory
            self._update_tool_guidance()
    
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

                # Update tool guidance in working memory
                self._update_tool_guidance()
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
        Get OpenAI-format definitions for all enabled tools.

        Returns:
            A list of tool definition dictionaries in OpenAI format.
            Tools must provide an 'openai_schema' attribute.
        """
        # Always use OpenAI format for the unified provider
        # Tools should provide OpenAI-compatible schemas

        definitions = []

        for name in self.enabled_tools:
            tool = self.tools[name]
            
            # For unified provider, always use OpenAI format
            # Tools should provide an openai_schema attribute
            if hasattr(tool, 'openai_schema'):
                definitions.append(tool.openai_schema)
            else:
                # Log warning if tool doesn't have OpenAI schema
                self.logger.warning(f"Tool '{name}' does not have an openai_schema attribute")

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
                    attr.__module__ == module.__name__ and
                    not getattr(attr, '_is_abstract_base_class', False)):
                    
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
                                if param_type.__name__ == 'LLMBridge' or param_type.__name__ == 'LLMProvider':
                                    from api.llm_provider import LLMProvider
                                    # Create LLMProvider instance
                                    dependencies[param_name] = LLMProvider()
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
        config = get_config()
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
    
    def update_working_memory(self) -> None:
        """
        Update tool-related content in working memory.

        This method is called by the working memory system to refresh tool-related
        content. It delegates to _update_tool_guidance for implementation.
        """
        self._update_tool_guidance()

    def _update_tool_guidance(self) -> None:
        """
        Update tool guidance in working memory based on enabled tools.

        This method is called automatically when tools are enabled or disabled.
        It updates the tool guidance content in working memory.
        """
        # If working_memory is not available, skip
        if not self.working_memory:
            return

        # Remove existing tool guidance if present
        if self._tool_guidance_id:
            self.working_memory.remove(self._tool_guidance_id)
            self._tool_guidance_id = None

        # Get enabled tools
        enabled_tools = self.get_enabled_tools()

        # If fewer than 2 tools are enabled, no guidance needed
        if len(enabled_tools) <= 1:
            return

        # Create tool guidance text
        tool_list = ", ".join([t.replace("_tool", "") for t in enabled_tools])
        tool_guidance = f"# Available Tools\n"
        tool_guidance += f"Multiple tools are currently available: {tool_list}.\n"
        tool_guidance += "If the user's request is ambiguous about which tool to use, ask for clarification."

        # Add to working memory
        self._tool_guidance_id = self.working_memory.add(
            content=tool_guidance,
            category="tool_guidance"
        )

        self.logger.debug(f"Updated tool guidance for {len(enabled_tools)} tools in working memory")

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