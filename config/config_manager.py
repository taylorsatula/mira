"""
Main configuration module for the application.

Provides centralized configuration management with validation, loading from
multiple sources, and a clean access interface.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from config.config import (
    ApiConfig,
    ApiServerConfig,
    PathConfig,
    ConversationConfig,
    ToolConfig,
    SystemConfig,
    EmbeddingsConfig,
    # Tool-specific configs are now defined in their respective tool files
    DatabaseConfig,
    ToolRelevanceConfig,
    OnloadCheckerConfig,
    MemoryConfig,
)
from errors import ConfigError, ErrorCode

# Import the registry (which was initialized in config/__init__.py)
from config.registry import registry


class AppConfig(BaseModel):
    """
    Central configuration manager for the application.
    
    Handles loading from multiple sources:
    1. JSON configuration files
    2. Environment variables
    
    Provides validated access to all application settings.
    Supports dynamic tool configuration through the configuration registry.
    """
    
    # Core configurations defined explicitly
    api: ApiConfig = Field(default_factory=ApiConfig)
    api_server: ApiServerConfig = Field(default_factory=ApiServerConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    # Tool-specific configs moved to their respective tool files
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    tool_relevance: ToolRelevanceConfig = Field(default_factory=ToolRelevanceConfig)
    onload_checker: OnloadCheckerConfig = Field(default_factory=OnloadCheckerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    
    # Cache for system prompts (non-model field)
    prompt_cache: Dict[str, str] = Field(default_factory=dict, exclude=True)
    
    # Cache for tool configs (non-model field, excluded from serialization)
    tool_configs: Dict[str, BaseModel] = Field(default_factory=dict, exclude=True)

    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> "AppConfig":
        """
        Load configuration from various sources and create a config instance.
        
        Args:
            config_path: Optional path to a JSON configuration file
            
        Returns:
            Validated AppConfig instance
            
        Raises:
            ConfigError: If configuration is invalid or required values are missing
        """
        # Load environment variables
        load_dotenv()
        
        # Start with empty config
        config_data: Dict[str, Any] = {}
        
        # Load from file if provided
        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise ConfigError(
                    f"Configuration file not found: {path}",
                    ErrorCode.CONFIG_NOT_FOUND
                )
            
            try:
                with open(path, "r") as f:
                    try:
                        file_config = json.load(f)
                        config_data.update(file_config)
                    except json.JSONDecodeError as e:
                        raise ConfigError(
                            f"Invalid JSON in configuration file: {e}",
                            ErrorCode.INVALID_CONFIG
                        )
            except Exception as e:
                if not isinstance(e, ConfigError):
                    raise ConfigError(
                        f"Error loading configuration file: {e}",
                        ErrorCode.INVALID_CONFIG
                    )
                raise
        
        
        # Create the config instance (uses default values for missing fields)
        try:
            # First create an instance with default values
            instance = cls()
            
            # Then update with the config data
            for section_name, section_data in config_data.items():
                if hasattr(instance, section_name):
                    section = getattr(instance, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
            
            # Create directories
            cls._ensure_directories(instance)
            
            return instance
        except Exception as e:
            raise ConfigError(
                f"Error initializing configuration: {e}",
                ErrorCode.INVALID_CONFIG
            )
    
    @classmethod
    def _load_from_env(cls) -> Dict[str, Dict[str, Any]]:
        """
        Load configuration from environment variables.
        Nested keys are separated by '__'.
        
        Examples:
            API_KEY=skBEEPBOOP_HEYKID+IMMAC0MPUTER
            EMAIL_PASSWORD=stopA11TheDownloadin'!
            
        Returns:
            Nested dictionary of configuration values from environment
        """
        config: Dict[str, Dict[str, Any]] = {}
        
        for key, value in os.environ.items():
            if key.startswith("AGENT_"):
                # Remove prefix and convert to lowercase
                config_key = key[6:].lower()
                
                # Handle nested keys (separated by __)
                if "__" in config_key:
                    parts = config_key.split("__")
                    section = parts[0]
                    setting = parts[1]
                    
                    if section not in config:
                        config[section] = {}
                    
                    # Try to parse value as JSON for complex types #ANNOTATION are there complex types? What is the usecase for this?
                    try:
                        parsed_value = json.loads(value)
                        config[section][setting] = parsed_value
                    except json.JSONDecodeError:
                        # Use as string if not valid JSON
                        config[section][setting] = value
                else:
                    # Top-level settings not supported, must use section
                    pass
        
        return config
    
    @classmethod
    def _ensure_directories(cls, config: "AppConfig") -> None:
        """
        Ensure required directories exist based on configuration.
        
        Args:
            config: Configuration instance
        """
        # Create data directory
        Path(config.paths.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Create persistent directory
        Path(config.paths.persistent_dir).mkdir(parents=True, exist_ok=True)
        
        # Create conversation history directory
        Path(config.paths.conversation_history_dir).mkdir(parents=True, exist_ok=True)
        
        # Create async_results directory
        Path(config.paths.async_results_dir).mkdir(parents=True, exist_ok=True)
        
        # Create prompts directory
        Path(config.paths.prompts_dir).mkdir(parents=True, exist_ok=True)
        
        # Create LT_Memory directories
        Path(config.paths.data_dir, "lt_memory").mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., "api.model")
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split(".")
        
        if len(parts) == 1:
            # Top-level attribute
            return getattr(self, parts[0], default)
        
        if len(parts) == 2:
            # Nested attribute
            section = getattr(self, parts[0], None)
            if section is None:
                return default
            return getattr(section, parts[1], default)
        
        # Unsupported nesting level
        return default
    
    def require(self, key: str) -> Any:
        """
        Get a required configuration value.
        
        Args:
            key: Configuration key in dot notation (e.g., "api.model")
            
        Returns:
            Configuration value
            
        Raises:
            ConfigError: If the key is not found
        """
        value = self.get(key)
        if value is None:
            raise ConfigError(
                f"Required configuration key not found: {key}",
                ErrorCode.MISSING_ENV_VAR
            )
        return value
    
    @property
    def api_key(self) -> str:
        """
        Get the LLM provider API key from environment variable.
        
        For local providers, returns empty string.
        For remote providers, looks for LLM_PROVIDER_API_KEY environment variable.
        
        Returns:
            API key string (empty for local providers)
            
        Raises:
            ConfigError: If the API key is not set for remote providers
        """
        # Local providers don't need API keys
        if self.api.provider == "local":
            return ""
            
        # Remote providers need LLM_PROVIDER_API_KEY
        api_key = os.getenv("LLM_PROVIDER_API_KEY")
        if not api_key:
            raise ConfigError(
                "LLM provider API key not found. Set LLM_PROVIDER_API_KEY environment variable.",
                ErrorCode.MISSING_ENV_VAR
            )
        return api_key
        
    @property
    def square_api_key(self) -> str:
        """
        Get the Square API key.
        
        Returns:
            Square API key string
            
        Raises:
            ConfigError: If the API key is not set
        """
        api_key = os.getenv("SQUARE_API_KEY")
        if not api_key:
            raise ConfigError(
                "Square API key not found. Set SQUARE_API_KEY environment variable.",
                ErrorCode.MISSING_ENV_VAR
            )
        return api_key
        
    @property
    def google_maps_api_key(self) -> str:
        """
        Get the Google Maps API key.
        
        Returns:
            Google Maps API key string
            
        Raises:
            ConfigError: If the API key is not set
        """
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise ConfigError(
                "Google Maps API key not found. Set GOOGLE_MAPS_API_KEY environment variable.",
                ErrorCode.MISSING_ENV_VAR
            )
        return api_key
        
    @property
    def email_password(self) -> str:
        """
        Get the email account password.
        
        Returns:
            Email password string
            
        Raises:
            ConfigError: If the password is not set
        """
        password = os.getenv("EMAIL_PASSWORD")
        if not password:
            raise ConfigError(
                "Email password not found. Set EMAIL_PASSWORD environment variable.",
                ErrorCode.MISSING_ENV_VAR
            )
        return password
    
    @property
    def embeddings_api_key(self) -> str:
        """
        Get the embeddings API key from environment variable.
        
        For local providers, returns empty string.
        For remote providers, looks for OAI_EMBEDDINGS_KEY environment variable.
        
        Returns:
            API key string (empty for local providers)
            
        Raises:
            ConfigError: If the API key is not set for remote providers
        """
        # Local providers don't need API keys
        if self.embeddings.provider == "local":
            return ""
            
        # Remote providers need OAI_EMBEDDINGS_KEY
        api_key = os.getenv("OAI_EMBEDDINGS_KEY")
        if not api_key:
            raise ConfigError(
                "OpenAI embeddings API key not found. Set OAI_EMBEDDINGS_KEY environment variable.",
                ErrorCode.MISSING_ENV_VAR
            )
        return api_key
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get the full configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.model_dump(exclude={"prompt_cache"})
    
    def get_system_prompt(
            self,
            prompt_name: str,
            replacements: Optional[Dict[str, str]] = None,
            reload: bool = False) -> str:
        """
        Get a system prompt by name.
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            replacements: Optional dictionary of placeholder replacements
            reload: Whether to reload the prompt from disk even if cached
            
        Returns:
            The prompt text with any replacements applied
            
        Raises:
            ConfigError: If the prompt file is not found
        """
        # Check cache first (unless reload is True)
        if not reload and prompt_name in self.prompt_cache:
            prompt_text = self.prompt_cache[prompt_name]
        else:
            # Construct file path
            file_path = Path(self.paths.prompts_dir) / f"{prompt_name}.txt"
            
            # Check if file exists
            if not file_path.exists():
                raise ConfigError(
                    f"Prompt file not found: {file_path}",
                    ErrorCode.CONFIG_NOT_FOUND
                )
            
            try:
                # Load prompt from file
                with open(file_path, 'r') as f:
                    prompt_text = f.read()
                
                # Cache the prompt
                self.prompt_cache[prompt_name] = prompt_text
                
            except Exception as e:
                raise ConfigError(
                    f"Error loading prompt file {file_path}: {e}",
                    ErrorCode.INVALID_CONFIG
                )
        
        # Apply replacements if provided
        if replacements:
            for placeholder, value in replacements.items():
                prompt_text = prompt_text.replace(placeholder, value)
        
        return prompt_text
    
    def reload_system_prompts(self) -> None:
        """
        Reload all system prompts from disk, refreshing the cache.
        """
        self.prompt_cache.clear()
        
    def __getattr__(self, name: str) -> Any:
        """
        Dynamic attribute access for tool configurations.
        
        This method is called when an attribute is not found through normal
        attribute access. It attempts to retrieve a tool configuration from
        the registry if the attribute name matches a tool name.
        
        Args:
            name: The attribute name to access
            
        Returns:
            Tool configuration instance if it's a tool name, otherwise raises AttributeError
            
        Raises:
            AttributeError: If the attribute is not a valid tool configuration
        """
        # Check if this might be a tool configuration
        if name.endswith('_tool') or name in self.tool_configs:
            # Get the tool configuration
            return self.get_tool_config(name)
            
        # Not a tool config, raise normal attribute error
        raise AttributeError(f"'AppConfig' object has no attribute '{name}'")
    
    def get_tool_config(self, tool_name: str) -> BaseModel:
        """
        Get a tool's configuration, creating it if needed.
        
        This method retrieves a tool configuration from the cache or creates
        a new one using the registry.
        
        Args:
            tool_name: The name of the tool
            
        Returns:
            Configuration instance for the tool
            
        Raises:
            ConfigError: If the tool configuration cannot be created
        """
        # Check if we already have this tool config cached
        if tool_name not in self.tool_configs:
            try:
                # Get the config class for this tool (or create a default)
                config_class = registry.get_or_create(tool_name)
                
                # Create an instance of the config class
                config_instance = config_class()
                
                # Store in cache for future access
                self.tool_configs[tool_name] = config_instance
                
                logging.debug(f"Created tool config for: {tool_name}")
                
            except Exception as e:
                raise ConfigError(
                    f"Error creating tool configuration for '{tool_name}': {e}",
                    ErrorCode.INVALID_CONFIG
                )
                
        # Return the cached config
        return self.tool_configs[tool_name]
    
    # We don't need a discover_tools method anymore.
    # Tools register themselves when they're imported naturally by the application.
            
    def list_available_tool_configs(self) -> List[str]:
        """
        List all available tool configurations.
        
        Returns:
            List of tool configuration names
        """
        # Return configs we've already loaded
        cached_configs = list(self.tool_configs.keys())
        
        # Add configs from registry
        registry_configs = list(registry._registry.keys())
        
        # Combine both lists, eliminating duplicates
        return list(set(cached_configs + registry_configs))


# Initialize configuration
def initialize_config() -> AppConfig:
    """
    Initialize the configuration.
    
    The registry is already initialized at this point (in config/__init__.py),
    so we just need to load the core configuration.
    
    Returns:
        Initialized AppConfig instance
    """
    try:
        # Load core configuration
        config_instance = AppConfig.load()
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, config_instance.system.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        
        logging.info("Configuration loaded successfully")
        logging.info(f"Registry initialized with: {list(registry._registry.keys())}")
        
        return config_instance
        
    except Exception as e:
        error_msg = f"Error initializing configuration: {e}"
        # Try to log the error, but we might not have logging configured yet
        try:
            logging.error(error_msg)
        except:
            pass
        
        raise ConfigError(
            error_msg,
            ErrorCode.INVALID_CONFIG
        )

# Create the global configuration instance
config = initialize_config()