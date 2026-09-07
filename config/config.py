"""
Centralized configuration models — operational and infrastructure settings only.

Algorithm tuning constants live inline in their consumer modules.
Only values that operators change without code changes belong here:
feature flags, infrastructure coordinates, scheduling cadences, deployment settings.
"""

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator


class ApiConfig(BaseModel):
    """LLM API and provider dialect configuration."""

    # Feature flags
    subcortical_prefill_warmup: bool = Field(default=False, description="Pre-warm subcortical KV cache after each turn (vLLM prefix-cache deployments only — wastes billed tokens on cloud providers)")
    show_openai_compat_thinking: bool = Field(default=True, description="Show thinking blocks from OpenAI-compatible dialects to end user")

    # Infrastructure coordinates
    api_key_name: str = Field(default="anthropic_key", description="Vault key name for Anthropic API key")

    # Operational limits
    timeout: int = Field(default=180, description="Max seconds an LLM provider HTTP request may run before it is aborted.")
    provider_response_timeout: int = Field(default=180, description="Max seconds an LLM provider call may run before the lifecycle aborts it.")
    async_work_barrier_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Max seconds a new chat turn waits for previous-turn background cache work.",
    )

    # Request sizing
    context_window_tokens: int = Field(default=200000, description="Total context window size in tokens")
    temperature: float = Field(default=1.0, description="Temperature for response generation (Anthropic default: 1.0)")
    compaction_trigger_tokens: int = Field(
        default=40_000,
        ge=1,
        description="Estimated input-token count at which to trigger live context compaction.",
    )
    compaction_raw_user_turns_to_preserve: int = Field(default=10, description="Number of recent user turns to preserve in raw format without compaction.")

    @model_validator(mode="after")
    def validate_compaction_trigger_tokens(self) -> "ApiConfig":
        if self.compaction_trigger_tokens >= self.context_window_tokens:
            raise ValueError(
                "compaction_trigger_tokens must be lower than context_window_tokens"
            )
        return self

    def validate_compaction_budget(self, primary_max_tokens: int) -> None:
        """Validate compaction against the database-owned primary output reserve."""
        available_input_tokens = self.context_window_tokens - primary_max_tokens
        if self.compaction_trigger_tokens > available_input_tokens:
            raise ValueError(
                "compaction_trigger_tokens must not exceed the primary model input budget "
                f"of {available_input_tokens} tokens"
            )


class ApiServerConfig(BaseModel):
    """FastAPI server deployment configuration."""

    # Infrastructure
    host: str = Field(default="0.0.0.0", description="Host address for the FastAPI server")
    port: int = Field(default=1993, description="Port for the FastAPI server")
    workers: int = Field(default=1, description="Number of uvicorn workers")

    # CORS
    enable_cors: bool = Field(default=True, description="Enable CORS middleware")
    cors_origins: List[str] = Field(
        default=["https://miraos.org", "http://localhost:1993", "http://127.0.0.1:1993"],
        description="Allowed CORS origins"
    )

    # Operational
    log_level: str = Field(default="warning", description="Log level for uvicorn server")
    extended_thinking: bool = Field(default=False, description="Enable extended thinking capability")
    extended_thinking_budget: int = Field(default=1024, description="Token budget for extended thinking (min: 1024)")


class SystemConfig(BaseModel):
    """System-level settings and feature flags."""

    # Feature flags
    subcortical_enabled: bool = Field(default=True, description="Enable subcortical memory surfacing and complexity assessment")
    peanutgallery_enabled: bool = Field(default=True, description="Enable peanut gallery metacognitive observer")
    persona_enabled: bool = Field(default=True, description="Enable Persona evaluation, refinement, and prompt injection")

    # Operational
    log_level: str = Field(default="WARNING", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    timezone: str = Field(default="America/Chicago", description="Default timezone (IANA name)")
    segment_timeout: int = Field(default=60, description="Segment collapse timeout in minutes")


class ScheduledJobsConfig(BaseModel):
    """Background job scheduling cadences — operational knobs for when jobs fire."""

    extraction_retry_hours: int = Field(
        default=6,
        description="Hours between failed extraction retries"
    )
    job_timeout_seconds: int = Field(
        default=120,
        description="Timeout for scheduled job monitors (ScheduledTaskMonitor wrappers)"
    )
    temporal_score_recalc_use_days: int = Field(
        default=1,
        description="Use-days between temporal score recalculations"
    )
    bulk_score_recalc_use_days: int = Field(
        default=1,
        description="Use-days between bulk score recalculations"
    )
    portrait_synthesis_use_days: int = Field(
        default=10,
        description="Use-days between portrait synthesis (runs in segment collapse chain)"
    )
    entity_merge_use_days: int = Field(
        default=7,
        ge=1,
        description="Use-day cadence for background entity dedup/merge (pg_trgm candidates → LLM judge)"
    )


class MemoryCuratorConfig(BaseModel):
    """Memory-graph curation agent (integration + floor modes).

    The MemoryCuratorAgent tends new memories at segment collapse (integration)
    and triages a random sample of low-value unseen memories on a use-day
    cadence (floor). Floor sampling is deterministic SQL heuristics over
    importance_score + last_tended_at staleness; the agent makes all judgment
    decisions (link / merge / archive / salvage).
    """

    enabled: bool = Field(
        default=True,
        description="Enable the memory curator (integration spawn + floor trigger)"
    )
    floor_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="importance_score strictly below this value makes a memory floor-eligible"
    )
    floor_unseen_days: int = Field(
        default=14,
        ge=1,
        description="Wall-clock days a memory must be un-tended before it is floor-eligible "
                    "(applied to last_tended_at, or to created_at when last_tended_at is NULL)"
    )
    floor_sample_size: int = Field(
        default=8,
        ge=1,
        description="Maximum memories sampled per floor cycle (bounds agent cost)"
    )
    floor_use_days: int = Field(
        default=7,
        ge=1,
        description="Use-day cadence for the floor trigger (get_users_due_for_job interval)"
    )


class LatticeConfig(BaseModel):
    """Lattice federation service configuration."""

    service_url: str = Field(default="http://localhost:1113", description="URL of the Lattice discovery service")
    timeout: int = Field(default=30, description="HTTP request timeout in seconds")


class SidebarDispatcherConfig(BaseModel):
    """Sidebar agent dispatcher configuration."""

    enabled: bool = Field(default=True, description="Enable the sidebar dispatcher polling loop")
    poll_interval_minutes: int = Field(default=1, description="Minutes between dispatcher poll cycles")
    max_concurrent_agents: int = Field(default=3, ge=1, description="Maximum sidebar agent threads running simultaneously")


class InboxToolConfig(BaseModel):
    """Configuration for the inbox_tool."""

    enabled: bool = Field(default=False, description="Whether this tool is enabled by default")
    inbox_path: str = Field(
        default="/tmp/mira-dropbox",
        description=(
            "Absolute filesystem path for the local file drop-off folder used by inbox_tool. "
            "Choose a narrowly scoped directory you intentionally want MIRA to inspect — "
            "do not point this at a broad or sensitive location unless that access is explicitly desired."
        ),
    )
    archive_subdir: str = Field(
        default="archive",
        description="Subdirectory inside inbox_path where archived files land.",
    )
    max_read_file_size_mb: int = Field(
        default=10,
        ge=1,
        description="Maximum file size in MB that inbox_tool will attempt to read before rejecting the request.",
    )
    max_read_chars: int = Field(
        default=20000,
        ge=1000,
        description="Maximum character count inbox_tool will return from a single read operation.",
    )

    @field_validator("inbox_path")
    @classmethod
    def validate_inbox_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("inbox_path must be an absolute path")
        return str(path)
