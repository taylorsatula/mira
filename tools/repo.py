"""
Tool discovery, registration, and invocation for the AI agent system.

This module handles the interface definition for all tools, tool
discovery and registration, and tool invocation and response processing.
"""
import importlib
import inspect
import logging
import pkgutil
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Type, Union, get_type_hints

from errors import ToolError, ErrorCode, error_context


class Tool:
    """
    Base class for all tools in the system.

    Defines the interface that all tools must implement and
    provides common functionality including standardized error handling.
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
        
    # Tool error handling should use the centralized error_context function from errors.py

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

    Handles tool discovery, registration, invocation, and dependency injection.
    """

    def __init__(self, tools_package: str = "tools", initial_dependencies: Optional[Dict[str, Any]] = None):
        """
        Initialize the tool repository.

        Args:
            tools_package: Python package where tools are located
            initial_dependencies: Dictionary of dependencies available at initialization
        """
        self.logger = logging.getLogger("tool_repo")
        self.tools_package = tools_package

        # Store both tool classes and instances
        self.tools: Dict[str, Union[Type[Tool], Tool]] = {}
        
        # Initialize tool selection components
        self.selector = None
        self.metrics = None

        # Track whether each entry is an instance or a class
        self.tool_instances: Dict[str, bool] = {}

        # Dependency injection support
        self.dependencies: Dict[str, Any] = {}
        self.tool_requirements: Dict[str, List[str]] = {}
        self.initialization_status: Dict[str, str] = {}  # "pending", "initialized", "failed"

        # Discover and register tools as classes first
        self.discover_tools()
        
        # Initialize tool selector and metrics
        self._discover_and_initialize_selection_components()
        
        # Load optimized tool descriptions if available
        self._load_optimized_descriptions()

        # Register any initial dependencies after discovery
        if initial_dependencies:
            for name, dependency in initial_dependencies.items():
                self.register_dependency(name, dependency)

        self.logger.info(f"Tool repository initialized with {len(self.tools)} tools")

    def discover_tools(self) -> None:
        """
        Discover and register all available tools.

        Searches the tools package for classes that inherit from Tool,
        analyzes their dependency requirements, and registers them
        in the tool registry as either classes (if they have dependencies)
        or instances (if they can be instantiated without dependencies).
        """
        try:
            # Get the tools package
            package = importlib.import_module(self.tools_package)
            package_path = getattr(package, "__path__", [None])[0]

            if not package_path:
                self.logger.warning(f"Could not find tools package path for {self.tools_package}")
                return

            pending_count = 0
            total_count = 0

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
                            # Register tool class and analyze its dependencies
                            deps = self._register_tool_class(obj)
                            if deps:
                                pending_count += 1
                            total_count += 1
                except Exception as e:
                    self.logger.error(f"Error loading tool module {module_name}: {e}")

            if pending_count > 0:
                self.logger.info(f"Discovered {total_count} tools, {pending_count} pending dependency injection")

            # After all tools are discovered, try to initialize those that don't need dependencies
            self._initialize_pending_tools()

        except Exception as e:
            self.logger.error(f"Error discovering tools: {e}")

    def _register_tool_class(self, tool_class: Type[Tool]) -> List[str]:
        """
        Register a tool class and analyze its dependency requirements.

        Args:
            tool_class: Tool class to register

        Returns:
            List of required dependencies
        """
        # Get the tool name
        tool_name = getattr(tool_class, "name", None)

        # Validate the tool
        if not tool_name:
            self.logger.warning(f"Skipping tool {tool_class.__name__} with missing name")
            return []

        # Register the tool class
        self.tools[tool_name] = tool_class
        self.tool_instances[tool_name] = False  # It's a class, not an instance

        # Analyze the tool's constructor requirements
        required_deps = self._analyze_tool_requirements(tool_class)
        self.tool_requirements[tool_name] = required_deps

        if required_deps:
            # Tool has dependencies, mark as pending initialization
            self.initialization_status[tool_name] = "pending"
            self.logger.debug(f"Registered tool class: {tool_name} with dependencies: {required_deps}")
        else:
            # No dependencies required, can initialize immediately
            self.initialization_status[tool_name] = "ready"
            self.logger.debug(f"Registered tool class: {tool_name} (no dependencies required)")

        return required_deps

    def _analyze_tool_requirements(self, tool_class: Type[Tool]) -> List[str]:
        """
        Determine what dependencies a tool needs by examining its __init__ signature.

        Args:
            tool_class: Tool class to analyze

        Returns:
            List of required parameter names (excluding self and parameters with defaults)
        """
        try:
            required_deps = []
            signature = inspect.signature(tool_class.__init__)

            # Skip self parameter, check remaining parameters
            for param_name, param in list(signature.parameters.items())[1:]:
                if param.default == inspect.Parameter.empty:
                    required_deps.append(param_name)

            return required_deps
        except Exception as e:
            self.logger.error(f"Error analyzing requirements for {tool_class.__name__}: {e}")
            return []

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
            # Check if the tool is a class that needs initialization
            is_instance = self.tool_instances.get(tool_name, False)

            if not is_instance:
                # Tool is still a class, check initialization status
                status = self.initialization_status.get(tool_name)

                if status == "failed":
                    # Tool failed to initialize previously
                    required_deps = self.tool_requirements.get(tool_name, [])
                    missing_deps = [dep for dep in required_deps if dep not in self.dependencies]

                    if missing_deps:
                        raise ToolError(
                            f"Tool {tool_name} is missing required dependencies: {missing_deps}",
                            ErrorCode.TOOL_INITIALIZATION_ERROR
                        )
                    else:
                        raise ToolError(
                            f"Tool {tool_name} failed to initialize with available dependencies",
                            ErrorCode.TOOL_INITIALIZATION_ERROR
                        )

                # Try initializing one more time in case dependencies were added
                if not self._try_initialize_tool(tool_name):
                    # Still couldn't initialize
                    required_deps = self.tool_requirements.get(tool_name, [])
                    if required_deps:
                        raise ToolError(
                            f"Tool {tool_name} requires dependencies: {required_deps}",
                            ErrorCode.TOOL_INITIALIZATION_ERROR
                        )
                    else:
                        raise ToolError(
                            f"Failed to initialize tool {tool_name}",
                            ErrorCode.TOOL_INITIALIZATION_ERROR
                        )

                # Successfully initialized, get the instance
                tool_instance = self.tools[tool_name]
            else:
                # Already an instance
                tool_instance = tool

            # Log the tool invocation
            self.logger.info(f"Invoking tool: {tool_name} with parameters: {tool_params}")
            
            # Check if we have a conversation context with the last user message
            # This will be used later for learning from tool usage patterns
            last_user_message = None
            
            try:
                # Try to get the user message if available
                from conversation import Conversation
                import inspect
                
                # Look for conversation in the call stack frames
                for frame_info in inspect.stack():
                    frame = frame_info.frame
                    for var_name, var_val in frame.f_locals.items():
                        if isinstance(var_val, Conversation) and hasattr(var_val, 'messages'):
                            # Found a conversation, look for the last user message
                            for msg in reversed(var_val.messages):
                                if msg.role == 'user' and isinstance(msg.content, str):
                                    last_user_message = msg.content
                                    break
                            if last_user_message:
                                break
                    if last_user_message:
                        break
                        
                if last_user_message:
                    self.logger.info(f"Found last user message for learning: {last_user_message[:50]}...")
            except Exception as e:
                self.logger.warning(f"Could not extract user message for learning: {e}")
            
            # Run the tool with the provided parameters
            result = tool_instance.run(**tool_params)
            
            # Record tool usage for metrics and learning
            self.record_tool_usage(tool_name, conversation_id=None, message=last_user_message)
            
            # Log successful execution
            self.logger.info(f"Tool {tool_name} executed successfully")
            
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

    def register_dependency(self, name: str, dependency: Any) -> None:
        """
        Register a dependency that can be injected into tools.

        Args:
            name: Name of the dependency
            dependency: The dependency object
        """
        self.dependencies[name] = dependency
        self.logger.debug(f"Registered dependency: {name}")

        # Try to initialize pending tools with the new dependency
        self._initialize_pending_tools()

    def _initialize_pending_tools(self) -> None:
        """
        Try to initialize any pending tools whose dependencies are now available.
        """
        initialized_count = 0

        # Get all tools that are pending initialization
        pending_tools = [
            name for name, status in self.initialization_status.items()
            if status == "pending" or status == "ready"
        ]

        if pending_tools:
            self.logger.debug(f"Attempting to initialize pending tools: {pending_tools}")

        for tool_name in pending_tools:
            if self._try_initialize_tool(tool_name):
                initialized_count += 1

        if initialized_count > 0:
            self.logger.info(f"Successfully initialized {initialized_count} tools with dependencies")

    def _try_initialize_tool(self, tool_name: str) -> bool:
        """
        Try to initialize a tool if all its dependencies are available.

        Args:
            tool_name: Name of the tool to try initializing

        Returns:
            True if successfully initialized, False otherwise
        """
        if tool_name not in self.tools:
            return False

        # Skip if already an instance
        if self.tool_instances.get(tool_name, False):
            return False

        # Get required dependencies for this tool
        required_deps = self.tool_requirements.get(tool_name, [])

        # If no dependencies are required, initialize directly
        if not required_deps:
            try:
                tool_class = self.tools[tool_name]
                tool_instance = tool_class()
                self.tools[tool_name] = tool_instance
                self.tool_instances[tool_name] = True
                self.initialization_status[tool_name] = "initialized"
                self.logger.debug(f"Initialized tool with no dependencies: {tool_name}")
                return True
            except Exception as e:
                self.logger.error(f"Error initializing tool {tool_name}: {e}")
                self.initialization_status[tool_name] = "failed"
                return False

        # Check if all dependencies are available
        missing_deps = [dep for dep in required_deps if dep not in self.dependencies]

        if missing_deps:
            self.logger.debug(f"Tool {tool_name} still missing dependencies: {missing_deps}")
            return False

        # All dependencies are available, try to initialize
        try:
            tool_class = self.tools[tool_name]

            # Gather the required dependencies
            deps_to_inject = {
                name: self.dependencies[name]
                for name in required_deps
            }

            # Create the instance with dependencies
            tool_instance = tool_class(**deps_to_inject)

            # Update registry
            self.tools[tool_name] = tool_instance
            self.tool_instances[tool_name] = True
            self.initialization_status[tool_name] = "initialized"

            self.logger.info(f"Successfully initialized tool {tool_name} with dependencies")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize tool {tool_name}: {e}")
            self.initialization_status[tool_name] = "failed"
            return False
            
    def _discover_and_initialize_selection_components(self) -> None:
        """
        Initialize tool selection components.
        
        This sets up the selector, metrics, and tool_finder tools
        that enable the Just-in-Time tool selection system.
        """
        try:
            # Import selection components
            from tools.selector import ToolSelector
            from tools.metrics import ToolMetrics
            from tools.tool_finder import ToolFinderTool
            
            # Initialize the selector
            self.selector = ToolSelector()
            self.logger.info("Tool selector initialized")
            
            # Initialize metrics collector
            self.metrics = ToolMetrics()
            self.logger.info("Tool metrics collector initialized")
            
            # Initialize and register the tool finder
            tool_finder = ToolFinderTool(self)
            self.register_tool(tool_finder)
            self.logger.info("Tool finder registered")
            
        except Exception as e:
            self.logger.error(f"Error initializing tool selection components: {e}")
            # Fallback to standard operation without selection
            self.selector = None
            self.metrics = None
    
    def select_tools_for_message(
        self, 
        message: str, 
        min_tools: Optional[int] = None, 
        max_tools: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Select the most relevant tools for a user message.
        
        Args:
            message: User message text
            min_tools: Minimum number of tools to select (defaults to config value)
            max_tools: Maximum number of tools to select (defaults to config value)
            
        Returns:
            List of selected tool definitions
            
        Note:
            Falls back to all tools if selector is not available or selection is disabled
        """
        # Get all tool definitions
        all_tools = self.get_all_tool_definitions()
        
        try:
            # Get config values
            from config import config
            
            # Check if tool selection is enabled
            if not config.tools.selection_enabled:
                return all_tools
                
            # Get min/max values from config if not provided
            if min_tools is None:
                min_tools = config.tools.min_tools
            if max_tools is None:
                max_tools = config.tools.max_tools
            
            # If selector is not initialized, return all tools
            if not self.selector:
                return all_tools
                
            # Get essential tools that should always be included
            from config import config
            essential_tool_names = config.tools.essential_tools
            
            # Collect essential tools
            essential_tools = []
            for tool_name in essential_tool_names:
                tool = self.get_tool(tool_name)
                if tool:
                    # Get tool definition
                    if self.tool_instances.get(tool_name, False):
                        # It's an instance
                        tool_def = tool.get_tool_definition()
                    else:
                        # It's a class
                        tool_def = tool.get_tool_definition()
                    essential_tools.append(tool_def)
            
            # Let selector choose additional tools
            remaining_slots = max_tools - len(essential_tools)
            if remaining_slots <= 0:
                # If essential tools fill all slots, just use those
                selected_tools = essential_tools
            else:
                # Filter out essential tools from candidates
                essential_names = [t["name"] for t in essential_tools]
                candidate_tools = [t for t in all_tools if t["name"] not in essential_names]
                
                # Use selector to choose additional tools
                if candidate_tools:
                    additional_tools = self.selector.select_tools(
                        message=message,
                        all_tools=candidate_tools,
                        min_tools=min(min_tools, remaining_slots),
                        max_tools=remaining_slots
                    )
                    
                    # Combine essential and selected tools
                    selected_tools = essential_tools + additional_tools
                else:
                    selected_tools = essential_tools
            
            # Record selection metrics if metrics collector is available
            if self.metrics:
                self.metrics.record_tool_selection(message, selected_tools, all_tools)
                
            return selected_tools
            
        except Exception as e:
            # Log error and fall back to returning all tools
            self.logger.error(f"Error in tool selection: {e}")
            return all_tools
            
    def record_tool_usage(self, tool_name: str, conversation_id: str = None, message: str = None) -> None:
        """
        Record tool usage for metrics and learning.
        
        Args:
            tool_name: Name of the tool that was used
            conversation_id: Optional conversation ID for tracking
            message: Optional user message that led to this tool use
        """
        # Skip if selector or metrics are not available
        if not self.selector and not self.metrics:
            self.logger.warning("Neither selector nor metrics available for tool usage recording")
            return
            
        try:
            self.logger.info(f"Recording usage of tool: {tool_name}")
            was_in_selection = True
            
            # Check if metrics is available to determine if tool was in selection
            if self.metrics:
                was_in_selection = self.metrics.is_tool_in_current_selection(tool_name)
                self.logger.info(f"Tool '{tool_name}' was in selection: {was_in_selection}")
                
                # Record the usage with sequence tracking
                self.metrics.record_tool_usage(tool_name, was_in_selection, conversation_id)
                
                # Force save metrics after each tool usage to ensure sequence data persists
                if hasattr(self.metrics, '_save_data'):
                    self.logger.info("Forcing metrics save to persist sequence data")
                    self.metrics._save_data()
            
            # Update selector learning data
            if self.selector:
                self.logger.info(f"Updating selector data for tool: {tool_name}")
                self.selector.record_usage(tool_name, was_in_selection, message=message)
                
                # Ensure keyword mapping is added for this tool
                if not was_in_selection:
                    # Get the tool description to extract keywords
                    tool = self.get_tool(tool_name)
                    if tool and hasattr(tool, 'description'):
                        description = tool.description
                        import re
                        # Extract words from description
                        words = re.findall(r'\b\w+\b', description.lower())
                        for word in words:
                            if len(word) >= 5:  # Only use longer words
                                self.logger.info(f"Adding keyword mapping: {word} -> {tool_name}")
                                self.selector.add_keyword_mapping(word, tool_name)
                
        except Exception as e:
            self.logger.error(f"Error recording tool usage: {e}")
            import traceback
            self.logger.error(f"Detailed error: {traceback.format_exc()}")
            # Continue normal operation even if metrics fail
            
    def reset_sequence_tracking(self) -> None:
        """
        Reset the tool sequence tracking for a new conversation turn.
        """
        if self.metrics:
            self.metrics.reset_sequence_tracking()
            
    def get_likely_next_tools(self, tool_name: str, limit: int = 3) -> List[str]:
        """
        Get the most likely tools to be used after a specific tool.
        
        Args:
            tool_name: Name of the tool to get likely next tools for
            limit: Maximum number of tools to return
            
        Returns:
            List of tool names sorted by likelihood
        """
        if self.metrics:
            return self.metrics.get_likely_next_tools(tool_name, limit)
        return []
            
    def get_selection_metrics_report(self) -> Dict[str, Any]:
        """
        Generate a report of tool selection metrics.
        
        Returns:
            Dictionary with metrics data, or empty dict if metrics not available
        """
        if not self.metrics:
            return {"error": "Metrics collector not available"}
            
        try:
            return self.metrics.generate_report()
        except Exception as e:
            self.logger.error(f"Error generating metrics report: {e}")
            return {"error": f"Failed to generate metrics report: {e}"}
            
    def _load_optimized_descriptions(self) -> None:
        """
        Load optimized tool descriptions from file and apply them to tools.
        
        This replaces the default descriptions with optimized ones that follow
        Anthropic's guidelines for better tool understanding.
        """
        from config import config
        import json
        import os
        
        # Path to optimized descriptions file
        descriptions_path = os.path.join(config.paths.persistent_dir, "optimized_tool_descriptions.json")
        
        if not os.path.exists(descriptions_path):
            self.logger.debug("No optimized tool descriptions file found")
            return
            
        try:
            # Load optimized descriptions
            with open(descriptions_path, 'r') as f:
                optimized_descriptions = json.load(f)
                
            # Count of applied descriptions
            applied_count = 0
            
            # Apply descriptions to tools
            for tool_name, description in optimized_descriptions.items():
                tool = self.get_tool(tool_name)
                if tool:
                    # Update the description at the class level
                    if self.tool_instances.get(tool_name, False):
                        # It's an instance, get its class
                        tool_class = tool.__class__
                    else:
                        # It's already a class
                        tool_class = tool
                        
                    # Replace the description
                    tool_class.description = description
                    applied_count += 1
            
            if applied_count > 0:
                self.logger.info(f"Applied {applied_count} optimized tool descriptions")
                
        except Exception as e:
            self.logger.error(f"Error loading optimized tool descriptions: {e}")
            # Continue without optimized descriptions
