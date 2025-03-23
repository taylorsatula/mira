"""
Configuration management for the AI agent system.

This module handles loading configuration from various sources,
defines default settings, and provides centralized access to all
system parameters.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

from errors import ConfigError, ErrorCode


class Config:
    """
    Configuration manager for the AI agent system.
    
    Handles loading configuration from:
    1. Default settings
    2. Configuration files
    3. Environment variables
    
    Provides centralized access to all system parameters.
    """
    
    DEFAULT_CONFIG = {
        "log_level": "INFO",
        "data_dir": "data",
        "api": {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 1000,
            "temperature": 0.7,
            "max_retries": 3,
            "timeout": 60,
            "rate_limit_rpm": 10
        },
        "conversation": {
            "max_history": 10,
            "max_context_tokens": 100000
        },
        "tools": {
            "enabled": True,
            "timeout": 30
        }
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file (optional)
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize with default configuration
        self._config = self.DEFAULT_CONFIG.copy()
        
        # Load configuration from file if provided
        if config_path:
            self._load_from_file(config_path)
        
        # Override with environment variables
        self._load_from_env()
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self._config["log_level"]),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("config")
        self.logger.debug("Configuration initialized")
    
    def _load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            ConfigError: If the file cannot be read or contains invalid JSON
        """
        path = Path(config_path)
        try:
            if not path.exists():
                raise ConfigError(
                    f"Configuration file not found: {path}",
                    ErrorCode.CONFIG_NOT_FOUND
                )
            
            with open(path, "r") as f:
                try:
                    file_config = json.load(f)
                    self._update_config(file_config)
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
    
    def _load_from_env(self) -> None:
        """
        Override configuration with environment variables.
        
        Environment variables should be prefixed with 'AGENT_'.
        Nested configuration keys are separated by '__'.
        
        Examples:
            AGENT_LOG_LEVEL=DEBUG
            AGENT_API__MODEL=claude-3-7-sonnet-20250219
            AGENT_API__TEMPERATURE=0.8
        """
        for key, value in os.environ.items():
            if key.startswith("AGENT_"):
                # Remove prefix and convert to lowercase
                config_key = key[6:].lower()
                
                # Handle nested keys (separated by __)
                if "__" in config_key:
                    parts = config_key.split("__")
                    self._set_nested_config(parts, value)
                else:
                    # Try to parse as JSON for complex types
                    try:
                        parsed_value = json.loads(value)
                        self._config[config_key] = parsed_value
                    except json.JSONDecodeError:
                        # Use as string if not valid JSON
                        self._config[config_key] = value
    
    def _set_nested_config(self, key_parts: list, value: str) -> None:
        """
        Set a nested configuration value from parts.
        
        Args:
            key_parts: List of nested key parts
            value: String value to set
        """
        current = self._config
        for part in key_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Try to parse the value as JSON
        try:
            parsed_value = json.loads(value)
            current[key_parts[-1]] = parsed_value
        except json.JSONDecodeError:
            # Use as string if not valid JSON
            current[key_parts[-1]] = value
    
    def _update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Recursively update configuration dictionary.
        
        Args:
            new_config: New configuration dictionary to merge
        """
        for key, value in new_config.items():
            if key in self._config and isinstance(self._config[key], dict) and isinstance(value, dict):
                self._update_config(value)
            else:
                self._config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (dot notation for nested keys)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split(".")
        value = self._config
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def require(self, key: str) -> Any:
        """
        Get a required configuration value.
        
        Args:
            key: Configuration key (dot notation for nested keys)
            
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
    
    def __getitem__(self, key: str) -> Any:
        """
        Access configuration values using dictionary syntax.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
        """
        return self.get(key)
    
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
        # Create a copy to prevent modification
        return self._config.copy()


# Global configuration instance
# This can be imported and used throughout the application
config = Config()