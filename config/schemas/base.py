"""
Base schema definitions for configuration models.

These are the core schema models used throughout the application to
ensure type safety and validation of configuration values.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ApiConfig(BaseModel):
    """API configuration settings."""
    
    model: str = Field(
        default="claude-3-7-sonnet-20250219",
        description="LLM model to use for API requests"
    )
    max_tokens: int = Field(
        default=1000,
        description="Maximum number of tokens to generate in responses"
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature setting for response generation"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed API requests"
    )
    timeout: int = Field(
        default=60,
        description="Timeout in seconds for API requests"
    )
    rate_limit_rpm: int = Field(
        default=10,
        description="Maximum API requests per minute"
    )


class PathConfig(BaseModel):
    """Path configuration settings."""
    
    data_dir: str = Field(
        default="data",
        description="Directory for data storage"
    )
    persistent_dir: str = Field(
        default="persistent",
        description="Directory for persistent storage"
    )
    async_results_dir: str = Field(
        default="persistent/async_results",
        description="Directory for async operation results"
    )
    prompts_dir: str = Field(
        default="config/prompts",
        description="Directory containing prompt templates"
    )
    selector_data_path: str = Field(
        default="persistent/tool_selector_data.json",
        description="Path to store tool selector data"
    )
    metrics_data_path: str = Field(
        default="persistent/tool_metrics.json",
        description="Path to store tool metrics data"
    )


class ConversationConfig(BaseModel):
    """Conversation and history management settings."""
    
    max_history: int = Field(
        default=10,
        description="Maximum number of conversation turns to keep in active memory"
    )
    max_context_tokens: int = Field(
        default=100000,
        description="Maximum number of tokens to include in conversation context"
    )
    max_tool_iterations: int = Field(
        default=5,
        description="Maximum number of tool iterations for a single request"
    )


class ToolConfig(BaseModel):
    """Tool-related configuration settings."""
    
    enabled: bool = Field(
        default=True,
        description="Whether tools are enabled"
    )
    timeout: int = Field(
        default=30,
        description="Default timeout in seconds for tool operations"
    )
    # Tool selection settings
    selection_enabled: bool = Field(
        default=True,
        description="Whether Just-in-Time tool selection is enabled"
    )
    min_tools: int = Field(
        default=3,
        description="Minimum number of tools to include in selection"
    )
    max_tools: int = Field(
        default=7, 
        description="Maximum number of tools to include in selection"
    )
    essential_tools: List[str] = Field(
        default=["tool_finder"],
        description="Tools that are always included in the selection"
    )
    # Extraction tool settings
    extraction_temperature: float = Field(
        default=0.3,
        description="Temperature setting for extraction operations"
    )
    extraction_max_tokens: int = Field(
        default=500,
        description="Maximum tokens for extraction operations"
    )
    extraction_templates: Dict[str, str] = Field(
        default={
            "general": "Extract the following information from the message: {target}. Return ONLY the extracted information, nothing else.",
            "personal_info": "Extract any personal information from the message such as:\n- Name\n- Age\n- Location\n- Preferences\n- Goals\nReturn only the extracted information in JSON format with appropriate keys.",
            "keywords": "Extract the main keywords from the message. Return only a list of keywords, separated by commas.",
            "question": "Is there a question in this message? If so, extract it. Return only the question itself or 'No question found'.",
            "sentiment": "Analyze the sentiment of this message. Return only one word: positive, negative, or neutral.",
            "entities": "Extract named entities (people, places, organizations, products) from the message. Return only the entities in JSON format with appropriate type labels.",
        },
        description="Templates used for information extraction"
    )
    max_background_iterations: int = Field(
        default=10,
        description="Maximum iterations for background LLM tasks"
    )
    max_async_age_hours: int = Field(
        default=24,
        description="Maximum age in hours for async results before cleanup"
    )
    async_join_timeout: int = Field(
        default=5,
        description="Timeout in seconds for async thread joining"
    )


class SystemConfig(BaseModel):
    """System-level configuration settings."""
    
    log_level: str = Field(
        default="ERROR",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    streaming: bool = Field(
        default=True,
        description="Whether to stream responses from the API"
    )
    json_indent: int = Field(
        default=2,
        description="Indentation level for JSON output"
    )
