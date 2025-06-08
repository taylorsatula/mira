"""
Base schema definitions for configuration models.

These are the core schema models used throughout the application to
ensure type safety and validation of configuration values.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from config.memory_config import MemoryConfig

class ApiConfig(BaseModel):
    """API configuration settings."""

    # Provider configuration
    provider: str = Field(
        default="local",
        description="LLM provider type: 'local' for local servers or 'remote' for cloud APIs"
    )
    
    # API endpoint configuration
    api_endpoint: str = Field(
        default="http://localhost:11434/v1/chat/completions",
        description="Full API endpoint URL including the path (e.g., 'http://localhost:11434/v1/chat/completions' for local Ollama)"
    )
    
    # Model configuration
    model: str = Field(
        default="hermes3:8b",
        description="Model to use for API requests (e.g., 'gpt-4', 'llama2', 'mixtral')"
    )

    # Common settings for all providers
    max_tokens: int = Field(
        default=1000,
        description="Maximum number of tokens to generate in responses"
    )
    temperature: float = Field(
        default=0.4,
        description="Temperature setting for response generation"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed API requests"
    )
    timeout: int = Field(
        default=60,
        description="Request timeout in seconds"
    )

class ApiServerConfig(BaseModel):
    """FastAPI server configuration settings."""
    
    host: str = Field(
        default="0.0.0.0",
        description="Host address for the FastAPI server"
    )
    port: int = Field(
        default=8000,
        description="Port for the FastAPI server"
    )
    workers: int = Field(
        default=1,
        description="Number of uvicorn workers"
    )
    log_level: str = Field(
        default="info",
        description="Log level for uvicorn server"
    )
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS middleware"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    request_timeout: int = Field(
        default=300,
        description="Request timeout in seconds"
    )
    rate_limit_rpm: int = Field(
        default=10,
        description="Maximum API requests per minute"
    )
    burst_limit: int = Field(
        default=5,
        description="Maximum number of requests allowed in a burst"
    )
    extended_thinking: bool = Field(
        default=False,
        description="Whether to enable extended thinking capability"
    )
    extended_thinking_budget: int = Field(
        default=4096,
        description="Token budget for extended thinking when enabled (min: 1024)"
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
    conversation_history_dir: str = Field(
        default="persistent/conversation_history",
        description="Directory for storing conversation history"
    )
    prompts_dir: str = Field(
        default="config/prompts",
        description="Directory containing prompt templates"
    )
    async_results_dir: str = Field(
        default="persistent/async_results",
        description="Directory for storing asynchronous operation results"
    )


class ConversationConfig(BaseModel):
    """Conversation and history management settings."""
    
    max_history: int = Field(
        default=20,
        description="Maximum number of conversation turns to keep in active memory"
    )
    max_context_tokens: int = Field(
        default=100000,
        description="Maximum number of tokens to include in conversation context"
    )
    max_tool_iterations: int = Field(
        default=10,
        description="Maximum number of tool iterations for a single request"
    )

class ToolConfig(BaseModel):
    """Tool-related configuration settings."""
    
    enabled: bool = Field(
        default=True,
        description="Whether tools are enabled"
    )
    auto_discovery: bool = Field(
        default=False,
        description="Whether automatic tool discovery is enabled"
    )
    timeout: int = Field(
        default=30,
        description="Default timeout in seconds for tool operations"
    )
    essential_tools: List[str] = Field(
        default=[],
        description="List of essential tools to always load"
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
    # Synthetic data generator settings
    synthetic_data_analysis_model: str = Field(
        default="claude-3-7-sonnet-20250219",
        description="LLM model to use for code analysis and example review in synthetic data analysis"
    )
    synthetic_data_generation_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="LLM model to use for example generation in synthetic data generation"
    )
    synthetic_data_embedding_model: str = Field(
        default="all-MiniLM-L12-v2",
        description="Embedding model to use for synthetic data deduplication"
    )


# Email configuration should be moved to tools/email_tool.py


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    
    uri: str = Field(
        default="sqlite:///data/app.db",
        description="Database connection URI"
    )
    echo: bool = Field(
        default=False,
        description="Echo SQL commands for debugging"
    )
    pool_size: int = Field(
        default=5,
        description="Connection pool size"
    )
    pool_timeout: int = Field(
        default=30,
        description="Connection pool timeout in seconds"
    )
    pool_recycle: int = Field(
        default=3600,
        description="Connection recycle time in seconds"
    )


class ToolRelevanceConfig(BaseModel):
    """Configuration for the ToolRelevanceEngine."""
    
    primary_threshold: float = Field(
        default=0.4,  # Lower threshold for better recall
        description="Threshold for primary tool selection (0.0-1.0)"
    )
    secondary_threshold: float = Field(
        default=0.3,  # Lower threshold for better recall
        description="Threshold for secondary tool selection (0.0-1.0)"
    )
    max_tools: int = Field(
        default=3,
        description="Maximum number of tools to enable at once"
    )
    thread_limit: int = Field(
        default=2,
        description="Maximum number of threads to use for embedding calculations"
    )
    drastic_difference_threshold: float = Field(
        default=1.3,  # Lower factor for more balanced tool selection
        description="If top tool score exceeds second tool score by this factor, only enable the top tool"
    )
    context_window_size: int = Field(
        default=3,
        description="Number of previous messages to consider for context"
    )
    topic_coherence_threshold: float = Field(
        default=0.7,
        description="Minimum similarity to consider messages related"
    )
    tool_persistence_messages: int = Field(
        default=2,
        description="Minimum number of messages to keep a tool enabled after activation"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L12-v2",
        description="Sentence transformer model to use for embeddings"
    )


class OnloadCheckerConfig(BaseModel):
    """Configuration for the OnloadChecker."""
    
    reminder_lookahead_days: int = Field(
        default=3,
        description="Number of days to look ahead for upcoming reminders"
    )


class SystemConfig(BaseModel):
    """System-level configuration settings."""
    
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    timezone: str = Field(
        default="America/Chicago",
        description="Default timezone for date/time operations (must be a valid IANA timezone name like 'America/New_York', 'Europe/London')"
    )
    streaming: bool = Field(
        default=True,
        description="Whether to stream responses from the API"
    )
    json_indent: int = Field(
        default=2,
        description="Indentation level for JSON output"
    )
    tokenizers_parallelism: bool = Field(
        default=False,
        description="Whether to enable tokenizers parallelism for Hugging Face libraries"
    )
