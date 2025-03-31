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
    PathConfig,
    ConversationConfig,
    ToolConfig,
    SystemConfig,
)
from errors import ConfigError, ErrorCode


class AppConfig(BaseModel):
    """
    Central configuration manager for the application.
    
    Handles loading from multiple sources:
    1. JSON configuration files
    2. Environment variables
    
    Provides validated access to all application settings.
    """
    
    api: ApiConfig = Field(default_factory=ApiConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    
    # Cache for system prompts (non-model field)
    prompt_cache: Dict[str, str] = Field(default_factory=dict, exclude=True)
    
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
        
        # Load from environment variables
        env_config = cls._load_from_env()
        for section, settings in env_config.items():
            if section not in config_data:
                config_data[section] = {}
            config_data[section].update(settings)
        
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
        
        Environment variables should be prefixed with 'AGENT_'.
        Nested keys are separated by '__'.
        
        Examples:
            AGENT_SYSTEM__LOG_LEVEL=DEBUG
            AGENT_API__MODEL=claude-3-7-sonnet-20250219
            
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
                    
                    # Try to parse value as JSON for complex types
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
        
        # Create async results directory
        Path(config.paths.async_results_dir).mkdir(parents=True, exist_ok=True)
        
        # Create prompts directory
        Path(config.paths.prompts_dir).mkdir(parents=True, exist_ok=True)
    
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
        Get the Anthropic API key.
        
        Returns:
            API key string
            
        Raises:
            ConfigError: If the API key is not set
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.",
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
            replacements: Optional[Dict[str, str]] = None) -> str:
        """
        Get a system prompt by name.
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            replacements: Optional dictionary of placeholder replacements
            
        Returns:
            The prompt text with any replacements applied
            
        Raises:
            ConfigError: If the prompt file is not found
        """
        # Check cache first
        if prompt_name in self.prompt_cache:
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


# Create the global configuration instance
try:
    # Create a config instance with default values, overridden by environment variables
    config = AppConfig.load()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.system.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
except Exception as e:
    raise ConfigError(
        f"Error initializing configuration: {e}",
        ErrorCode.INVALID_CONFIG
    )