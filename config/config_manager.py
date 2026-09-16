"""
Main configuration module for the application.

Provides centralized configuration management with validation, loading from
multiple sources, and a clean access interface.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from config.config import (
    ApiConfig,
    ApiServerConfig,
    AuthConfig,
    CacheConfig,
    DatabaseConfig,
    LtMemoryConfig,
    SystemConfig,
    ScheduledJobsConfig,
    WorkerPoolsConfig,
    MemoryCuratorConfig,
    LatticeConfig,
    SidebarDispatcherConfig,
    HeartbeatConfig,
)

# Import the registry from tools package
from tools.registry import registry


# Registry of cognitive feature switches: environment variable name -> SystemConfig field name.
# Adding a flag is a one-line entry here plus the matching SystemConfig field and factory/init
# omission point. Do not pre-register a flag for a subsystem that does not exist yet; add it with
# the subsystem (MIRA_PERSONA_ENABLED landed with Persona in WP5).
SYSTEM_FEATURE_FLAG_ENVIRONMENT_FIELDS: dict[str, str] = {
    "MIRA_SUBCORTICAL_ENABLED": "subcortical_enabled",
    "MIRA_PEANUTGALLERY_ENABLED": "peanutgallery_enabled",
    "MIRA_PERSONA_ENABLED": "persona_enabled",
}

# Registry of string-valued SystemConfig overrides: environment variable name -> field name.
# Deploy writes MIRA_TIMEZONE into the service unit so an install's timezone is an explicit
# operator choice rather than the hardcoded field default.
SYSTEM_STRING_ENVIRONMENT_FIELDS: dict[str, str] = {
    "MIRA_TIMEZONE": "timezone",
}


def _load_system_feature_flag_overrides() -> dict[str, bool]:
    """Load strict non-secret feature switches from the process environment.

    Parsing is intentionally strict: a value must be exactly "0" or "1". Anything else
    ("true", "yes", "") raises at config load rather than silently coercing to a meaning
    the operator did not intend. An unset variable (None) is omitted so the field default
    applies.
    """
    overrides: dict[str, bool] = {}
    for environment_name, field_name in SYSTEM_FEATURE_FLAG_ENVIRONMENT_FIELDS.items():
        raw_value = os.getenv(environment_name)
        if raw_value is None:
            continue
        if raw_value not in {"0", "1"}:
            raise ValueError(f"{environment_name} must be exactly 0 or 1")
        overrides[field_name] = raw_value == "1"
    return overrides


def _load_system_string_overrides() -> dict[str, str]:
    """Load strict string SystemConfig overrides from the process environment.

    Unset variables are omitted so the field default applies. Values are
    validated at the SystemConfig field validator (fail fast at boot), so
    this loader only requires presence and non-emptiness.
    """
    overrides: dict[str, str] = {}
    for environment_name, field_name in SYSTEM_STRING_ENVIRONMENT_FIELDS.items():
        raw_value = os.getenv(environment_name)
        if raw_value is None or raw_value == "":
            continue
        overrides[field_name] = raw_value
    return overrides


class AppConfig(BaseModel):
    """Configuration manager with Vault integration and dynamic tool configuration via registry."""
    
    api: ApiConfig = Field(default_factory=ApiConfig)
    api_server: ApiServerConfig = Field(default_factory=ApiServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    lt_memory: LtMemoryConfig = Field(default_factory=LtMemoryConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    scheduled_jobs: ScheduledJobsConfig = Field(default_factory=ScheduledJobsConfig)
    worker_pools: WorkerPoolsConfig = Field(default_factory=WorkerPoolsConfig)
    lattice: LatticeConfig = Field(default_factory=LatticeConfig)
    sidebar_dispatcher: SidebarDispatcherConfig = Field(default_factory=SidebarDispatcherConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    memory_curator: MemoryCuratorConfig = Field(default_factory=MemoryCuratorConfig)


    # System prompt loaded once at startup
    system_prompt_text: str = Field(default="", exclude=True)
    
    # Cache for tool configs (non-model field, excluded from serialization)
    tool_configs: Dict[str, BaseModel] = Field(default_factory=dict, exclude=True)

    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration with defaults and system prompt."""
        logger = logging.getLogger(__name__)
        
        try:
            instance = cls(
                system=SystemConfig(
                    **_load_system_feature_flag_overrides(),
                    **_load_system_string_overrides(),
                ),
            )
            instance._load_system_prompt()
            logger.info("Configuration initialized successfully")
            return instance
        except Exception as e:
            logger.error(f"Configuration initialization failed: {e}")
            raise ValueError(f"Error initializing configuration: {e}")
    
    
    
    def get(self, key: str, default: Any = None) -> Any:
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
        value = self.get(key)
        if value is None:
            raise KeyError(f"Required configuration key not found: {key}")
        return value

    def as_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude={"prompt_cache"})
    
    def _load_system_prompt(self) -> None:
        """Load system prompt once at startup."""
        _CONFIG_DIR = Path(__file__).parent.resolve()
        path = (_CONFIG_DIR / "system_prompt.txt").resolve()

        if not path.is_relative_to(_CONFIG_DIR):
            raise ValueError(
                f"System prompt path resolves outside config/: {path}"
            )

        if not path.suffix == ".txt":
            raise ValueError(
                f"System prompt must be a .txt file, "
                f"got {path.suffix!r}"
            )

        if not path.exists():
            raise FileNotFoundError(
                f"System prompt not found at {path}. "
                f"Prompts are system configuration, not optional features."
            )

        content = path.read_text(encoding="utf-8").strip()

        if not content:
            raise FileNotFoundError(
                f"System prompt file is empty: {path}. "
                f"Prompt files must contain non-whitespace content."
            )

        self._system_prompt = content
        logging.getLogger(__name__).debug(
            "Loaded system prompt (%d chars)", len(content)
        )
    
    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return self._system_prompt
    
        
    def __getattr__(self, name: str) -> Any:
        """Dynamic tool configuration access via registry - enables config.tool_name syntax."""
        # Check if this might be a tool configuration
        if name.endswith('_tool') or name in self.tool_configs:
            # Get the tool configuration
            return self.get_tool_config(name)
            
        # Not a tool config, raise normal attribute error
        raise AttributeError(f"'AppConfig' object has no attribute '{name}'")
    
    def get_tool_config(self, tool_name: str) -> BaseModel:
        """Return the current user's validated tool config or the global default."""
        if tool_name not in self.tool_configs:
            try:
                config_class = registry.get_or_create(tool_name)
                config_instance = config_class()
                self.tool_configs[tool_name] = config_instance
                
                logging.debug(f"Tool config created: {tool_name}")
                
            except Exception as e:
                logging.error(f"Tool config creation failed for {tool_name}: {e}")
                raise ValueError(f"Error creating tool configuration for '{tool_name}': {e}")
                
        default_config = self.tool_configs[tool_name]
        try:
            from utils.user_context import get_current_user_id

            get_current_user_id()
        except RuntimeError:
            return default_config

        from utils.tool_config_store import load_user_tool_config

        user_config = load_user_tool_config(tool_name, hydrate_secrets=True)
        if user_config is None:
            return default_config

        config_class = type(default_config)
        return config_class(**{**default_config.model_dump(), **user_config})
    
    # We don't need a discover_tools method anymore.
    # Tools register themselves when they're imported naturally by the application.
            
    def list_available_tool_configs(self) -> List[str]:
        cached_configs = list(self.tool_configs.keys())
        registry_configs = list(registry._registry.keys())
        return list(set(cached_configs + registry_configs))


# Initialize configuration
def initialize_config() -> AppConfig:
    """Initialize configuration and logging."""
    try:
        config_instance = AppConfig.load()
        
        logging.basicConfig(
            level=getattr(logging, config_instance.system.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        
        logging.info("Configuration loaded successfully")
        logging.info(f"Registry initialized with: {list(registry._registry.keys())}")
        
        return config_instance
        
    except Exception as e:
        error_msg = f"Error initializing configuration: {e}"
        try:
            logging.error(error_msg)
        except:
            pass
        
        raise RuntimeError(error_msg)

# Create the global configuration instance
config = initialize_config()
