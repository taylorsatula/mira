"""
Centralized configuration models — operational and infrastructure settings only.

Algorithm tuning constants live inline in their consumer modules.
Only values that operators change without code changes belong here:
feature flags, infrastructure coordinates, scheduling cadences, deployment settings.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from utils.timezone_utils import get_default_timezone, validate_timezone


class ApiConfig(BaseModel):
    """LLM API and provider dialect configuration."""

    # Feature flags
    subcortical_prefill_warmup: bool = Field(default=False, description="Pre-warm subcortical KV cache after each turn (vLLM prefix-cache deployments only — wastes billed tokens on cloud providers)")
    show_openai_compat_thinking: bool = Field(default=True, description="Show thinking blocks from OpenAI-compatible dialects to end user")

    # Operational limits
    # Both bounds sit at 15 minutes: a large non-streaming completion (segment
    # collapse summaries over a long segment, batch passes) can legitimately
    # take many minutes on a loaded gateway, and a bound that fires first
    # turns a slow success into a failed collapse — observed live killing
    # every large-segment collapse on 2026-10-08 (jammed the sweep for hours).
    timeout: int = Field(default=900, description="Max seconds an LLM provider HTTP request (socket/connect/read) may run before it is aborted. Distinct from provider_response_timeout, which bounds no-progress stalls. The two move together: an HTTP bound below the stall bound makes the stall bound moot for non-streaming calls.")
    provider_response_timeout: int = Field(default=900, description="Max seconds an LLM provider call may show no progress before the lifecycle aborts it (stall detection). Distinct from timeout, which bounds the HTTP request itself. The two move together: an HTTP bound below this makes this moot for non-streaming calls.")
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

    # Provider failover (diverter): one global backup target for all five routes.
    # While tripped, ModelResolver diverts every route to this target carrying the
    # original route name (pricing continuity); the recovery probe job flips back
    # after consecutive passing probes. See clients/llm/diverter.py.
    failover_enabled: bool = Field(default=False, description="Enable the provider failover diverter: after N consecutive retryable provider failures, route all LLM calls to the backup target below until the recovery probe confirms the primary recovered.")
    failover_endpoint_url: Optional[str] = Field(default=None, description="Backup provider endpoint (OpenAI-compatible or the dialect's native shape). Required when failover_enabled.")
    failover_model: Optional[str] = Field(default=None, description="Backup provider model name. Required when failover_enabled.")
    failover_dialect_name: Optional[str] = Field(default=None, description="Dialect for the backup provider: openai, anthropic, openrouter, or groq. Required when failover_enabled.")
    failover_api_key_name: Optional[str] = Field(default=None, description="Vault key name holding the backup provider credential; empty/None for an unauthenticated endpoint (e.g. local llama-server).")
    failover_max_tokens_cap: Optional[int] = Field(default=None, description="Ceiling applied to each route's max_tokens while diverted (the backup may accept fewer output tokens than the primary route allows); None keeps each route's own ceiling.")
    failover_recovery_probe_seconds: int = Field(default=120, ge=30, description="Interval of the recovery probe job while the diverter is tripped: probe every primary target with a first-token reachability probe; flip back after failover_flipback_consecutive_passes consecutive full passes. No probes are billed while healthy.")
    failover_trip_consecutive_failures: int = Field(default=3, ge=1, description="Consecutive retryable provider failures (ProviderRetryableError, including stalls) from the same primary endpoint that trip the diverter. A single success resets the counter; auth and context-overflow errors never count.")
    failover_flipback_consecutive_passes: int = Field(default=2, ge=1, description="Consecutive passing primary probes required before traffic returns to the primary; one lucky probe cannot cause an outage-flap loop.")

    @model_validator(mode="after")
    def validate_failover_block(self) -> "ApiConfig":
        """When the diverter is enabled, its transport coordinates are required."""
        if not self.failover_enabled:
            return self
        missing = [
            name for name, value in (
                ("failover_endpoint_url", self.failover_endpoint_url),
                ("failover_model", self.failover_model),
                ("failover_dialect_name", self.failover_dialect_name),
            ) if not value or not str(value).strip()
        ]
        if missing:
            raise ValueError(
                "failover_enabled requires " + ", ".join(missing) +
                " (the backup provider's transport coordinates)"
            )
        if self.failover_max_tokens_cap is not None and self.failover_max_tokens_cap <= 0:
            raise ValueError("failover_max_tokens_cap must be a positive integer or None")
        return self

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
    sync_endpoint_thread_limit: int = Field(
        default=100, ge=1,
        description="Max concurrent threads FastAPI uses for synchronous endpoints (per worker process). Lower it on single-user installs; raise it for high-concurrency multi-user deployments"
    )

    # CORS
    enable_cors: bool = Field(default=True, description="Enable CORS middleware")
    cors_origins: List[str] = Field(
        default=["https://miraos.org", "http://localhost:1993", "http://127.0.0.1:1993"],
        description="Allowed CORS origins"
    )

    # Operational
    log_level: str = Field(default="warning", description="Log level for uvicorn server")


class SystemConfig(BaseModel):
    """System-level settings and feature flags."""

    # Feature flags
    subcortical_enabled: bool = Field(default=True, description="Enable subcortical memory surfacing and complexity assessment")
    peanutgallery_enabled: bool = Field(default=True, description="Enable peanut gallery metacognitive observer")
    persona_enabled: bool = Field(default=True, description="Enable Persona evaluation, refinement, and prompt injection")
    injection_screen_enabled: bool = Field(
        default=True,
        description="Enable the semantic prompt-injection screen (screen_untrusted). Disabled mode wraps external content without judging it — never passes it raw. Overridable via MIRA_INJECTION_SCREEN_ENABLED (strict 0/1)",
    )
    mcp_enabled: bool = Field(
        default=False,
        description="Enable the MCP endpoint at /v0/mcp exposing one tool (check_in: a complete chat turn via the same path as POST /v0/api/chat). Disabled constructs nothing — the route is not mounted and the mcp SDK is never imported. Overridable via MIRA_MCP_ENABLED (strict 0/1)",
    )

    # Operational
    log_level: str = Field(default="WARNING", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    anthropic_sdk_content_logging: bool = Field(
        default=False,
        description="Enable Anthropic SDK message-content logging to logs/anthropic_sdk.log (persists user/assistant conversation text and SDK request bodies; privacy-sensitive, off by default)"
    )
    timezone: str = Field(
        default_factory=get_default_timezone,
        description="Default timezone (IANA name); defaults to the host system timezone, overridable via MIRA_TIMEZONE"
    )
    local_session_first_name: str = Field(
        default="Friend",
        description="First name for the single-mode local account (auth/service.py create_local_session) and its orientation content; deploy collects it and writes MIRA_LOCAL_SESSION_FIRST_NAME, and the app-side default is the historical 'Friend'"
    )

    @field_validator("timezone")
    @classmethod
    def validate_timezone_name(cls, value: str) -> str:
        """Fail fast on a non-IANA timezone instead of surfacing ZoneInfo errors at runtime."""
        return validate_timezone(value)
    segment_timeout: int = Field(
        default=120,
        description="Segment collapse timeout in minutes — staleness at which an active segment collapses, evaluated in the segment owner's local time-of-day windows"
    )
    segment_timeout_morning: Optional[int] = Field(
        default=None,
        description="Optional override for the 06:00-09:00 window of the segment owner's local time; falls back to segment_timeout when unset"
    )
    segment_timeout_late_night: Optional[int] = Field(
        default=None,
        description="Optional override for the 23:00-06:00 window of the segment owner's local time; falls back to segment_timeout when unset"
    )


class ScheduledJobsConfig(BaseModel):
    """Background job scheduling cadences — operational knobs for when jobs fire."""

    extraction_retry_hours: int = Field(
        default=6,
        description="Hours between failed extraction retries"
    )
    job_timeout_seconds: int = Field(
        default=3600,
        description=(
            "Timeout in seconds for ScheduledTaskMonitor-wrapped jobs. The "
            "segment timeout job runs the whole collapse chain synchronously "
            "(summary, Persona, feedback passes — several batch LLM calls, each "
            "bounded by api.provider_response_timeout), so the ceiling must sit "
            "above that chain's worst case or every slow-but-legitimate collapse "
            "gets orphaned mid-flight and the next run re-claims it. Overlapping "
            "runs are prevented by the atomic collapse claim, not by this "
            "ceiling; a run that exceeds it is wedged, not slow"
        )
    )
    temporal_score_recalc_use_days: int = Field(
        default=1,
        ge=1,
        description="Use-days between temporal score recalculations"
    )
    bulk_score_recalc_use_days: int = Field(
        default=1,
        ge=1,
        description="Use-days between bulk score recalculations"
    )
    portrait_synthesis_use_days: int = Field(
        default=10,
        ge=1,
        description="Use-days between portrait synthesis (runs in segment collapse chain)"
    )
    entity_merge_use_days: int = Field(
        default=7,
        ge=1,
        description="Use-day cadence for background entity dedup/merge (pg_trgm candidates → LLM judge)"
    )


class DatabaseConfig(BaseModel):
    """PostgreSQL connection-pool sizing and query guards."""

    pool_min: int = Field(default=3, ge=1, description="Minimum connections kept warm in mira_service Postgres pools")
    pool_max: int = Field(default=30, ge=1, description="Maximum connections per mira_service Postgres pool under load")
    session_pool_min: int = Field(default=2, ge=1, description="Minimum connections in the LTMemory session-manager pools")
    session_pool_max: int = Field(default=15, ge=1, description="Maximum connections in the LTMemory session-manager pools")
    statement_timeout_ms: int = Field(default=300000, ge=1, description="Postgres statement_timeout in milliseconds — any single SQL query is aborted after this long")


class AuthConfig(BaseModel):
    """Authentication policy limits."""

    max_api_tokens_per_user: int = Field(default=50, ge=1, description="Maximum API tokens a single user account may hold")


class WorkerPoolsConfig(BaseModel):
    """Background executor thread-pool sizes."""

    peanutgallery_workers: int = Field(default=1, ge=1, description="Peanut-gallery commentary executor workers")
    tool_result_summarizer_workers: int = Field(default=2, ge=1, description="Tool-result summarization executor workers")
    orchestrator_encode_workers: int = Field(default=2, ge=1, description="Orchestrator dual-embedding encode executor workers")
    repulsion_rewriter_workers: int = Field(default=4, ge=1, description="Repulsion-rewrite executor workers")


class CacheConfig(BaseModel):
    """Valkey connection settings."""

    max_connections: int = Field(default=20, ge=1, description="Maximum simultaneous connections to the Valkey server")


class LtMemoryConfig(BaseModel):
    """LT_Memory ML-tuning knobs (batch sizes and worker counts)."""

    embeddings_batch_size: int = Field(default=32, ge=1, description="Texts per embedding batch: one SentenceTransformer encode batch (local provider) or one POST /v1/embeddings request (remote provider)")
    proactive_search_workers: int = Field(default=2, ge=1, description="Parallel proactive-memory search fan-out workers")
    entity_merge_candidate_limit: int = Field(default=500, ge=1, description="Maximum pg_trgm duplicate-entity candidate pairs returned per entity-merge sweep")


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

    enabled: bool = Field(
        default=False,
        description="Enable Lattice federation (cross-server pager messaging). Requires the optional lattice package; startup fails fast when enabled and unavailable"
    )
    service_url: str = Field(default="http://localhost:1113", description="URL of the Lattice discovery service")
    timeout: int = Field(default=30, description="HTTP request timeout in seconds")


class SystemOneConfig(BaseModel):
    """System One decision-model endpoint (TypeSafe `/v1/systemone` wire format).

    Any server speaking that contract works: hosted Jev, the lunaroute
    gateway's djev, or a self-hosted Kev. Consumer: clients/systemone_client.py.

    `provider` picks the credential model: `remote` is a hosted gateway
    whose bearer token is read from Vault under `api_key_name`; `local` is
    a self-hosted (Kev) endpoint that takes no Authorization header at all
    — `api_key_name` is ignored and no Vault field is required.
    """

    provider: Literal["remote", "local"] = Field(
        default="remote",
        description="remote = hosted gateway, bearer token from Vault (api_key_name); local = self-hosted endpoint, no Authorization header",
    )
    endpoint_url: str = Field(
        default="https://gw.lunaroute.com/v1/systemone",
        description="Full URL of the POST /v1/systemone endpoint",
    )
    model: str = Field(default="djev", description="Model identifier sent in every request")
    api_key_name: str = Field(
        default="systemone_key",
        description="Vault key name under mira/api_keys holding the endpoint's bearer token",
    )
    timeout: int = Field(
        default=10,
        gt=0,
        description="HTTP request timeout in seconds. Observed djev latency: median 0.44 s, max 1.18 s over 46 requests at 8-way concurrency (2026-09-26)",
    )
    max_concurrent_requests: int = Field(
        default=11,
        gt=0,
        description="Requests one process keeps in flight to the endpoint; the lunaroute key allows 11 and answers HTTP 429 beyond it. The cap is per process: N server workers can reach N times this",
    )


class SidebarDispatcherConfig(BaseModel):
    """Sidebar agent dispatcher configuration."""

    enabled: bool = Field(default=True, description="Enable the sidebar dispatcher polling loop")
    poll_interval_minutes: int = Field(default=1, description="Minutes between dispatcher poll cycles")
    max_concurrent_agents: int = Field(default=3, ge=1, description="Maximum sidebar agent threads running simultaneously")
    agent_timeout_seconds: int = Field(default=120, ge=1, description="Wall-clock seconds before a running sidebar agent is considered timed out")
    agent_iteration_timeout_seconds: int = Field(
        default=300, ge=1,
        description=(
            "Seconds a single sidebar-agent LLM iteration may run, evaluated between "
            "iterations. Only agents with wall-clock overrides this large can reach it "
            "(forage/wtcha); default agents hit agent_timeout_seconds first. Calibrated for "
            "local-model deployments where a thinking-model iteration plus web fetches "
            "routinely exceeds a minute — 45s aborted every forage iteration on a Q8 27B "
            "llama-server, failing the agent before its second iteration could start."
        )
    )
    agent_timeout_overrides: Dict[str, int] = Field(
        default={"forage": 600, "memorycurator": 480, "whilethecatsaway": 14400},
        description="Per-agent wall-clock timeout overrides keyed by agent class name lowercased with the 'Agent' suffix stripped (e.g. ForageAgent -> 'forage')"
    )
    blocking_agent_timeout_seconds: int = Field(
        default=120, ge=1,
        description=(
            "Wall-clock bound for blocking-mode agent runs (SidebarAgent.run_blocking): "
            "a blocking run executes inline on the caller's thread inside a turn, so "
            "this, not agent_timeout_overrides, is what bounds how long a blocking "
            "call can stretch the calling turn. The caller's own turn deadline "
            "(api turn cancel event) still applies on top"
        )
    )


class DevicePowerBindingConfig(BaseModel):
    """Optional binding site connecting the heartbeat wake cycle to the
    physical device's power management. The dispatcher publishes the earliest
    next wake obligation here; a deployer-supplied shim bridges to the C++
    layer that actually enters low power. See utils/device_binding.py."""

    module_path: Optional[str] = Field(
        default=None,
        description=(
            "Filesystem path to a deployer-supplied Python module exposing "
            "arm_low_power(wake_at_utc: str, wake_lead_seconds: int, "
            "metadata: HeartbeatSleepMetadata) -> None; see "
            "utils/device_binding.py for the contract. None (the default) "
            "leaves the device at full power — the unbound behavior"
        )
    )
    wake_lead_seconds: int = Field(
        default=30, ge=0,
        description=(
            "Seconds before the published wake time the device should be back "
            "at full power so the dispatcher tick fires on time"
        )
    )


class HeartbeatConfig(BaseModel):
    """Heartbeat (briefs) configuration.

    A brief is a persisted wake-up carrying a mission, a condition to
    check at wake, and (one-offs) a bail_if early-exit. The HeartbeatTrigger
    surfaces due armed briefs to the HeartbeatAgent on the sidebar
    dispatcher's poll cadence; the agent evaluates each brief and reports
    its outcome via heartbeat_tool. See tools/implementations/heartbeat_tool.py
    and agents/triggers/heartbeat_trigger.py.
    """

    enabled: bool = Field(default=True, description="Enable the heartbeat brief system")
    default_interval_seconds: int = Field(
        default=600, ge=60,
        description=(
            "Default re-arm cadence in seconds when a brief's condition is false at wake"
        ),
    )
    max_interval_seconds: int = Field(
        default=4800, ge=60,
        description=(
            "Ceiling for decayed false re-arms and failure backoff: interval * "
            "2^count is clamped to this"
        ),
    )
    max_failures: int = Field(
        default=3, ge=1,
        description=(
            "Consecutive failed agent runs (no valid report) before a brief goes "
            "failed_permanently with an exit note. Deliberate exemption from the "
            "infrastructure-failure exclusion rule: a brief whose runs keep dying "
            "has no separate signal to distinguish provider outage from a "
            "mission that can never succeed, and an underivable distinction "
            "must fail closed — the brief parks itself with an exit note and "
            "the failure history in last_result rather than waking forever. A "
            "healthy install's briefs re-arm fine after transient blips "
            "because the backoff (interval x 2^failures) spaces attempts out."
        ),
    )
    device_power_binding: DevicePowerBindingConfig = Field(
        default_factory=DevicePowerBindingConfig,
        description=(
            "Low-power device binding: publishes the earliest armed brief's next "
            "wake to a deployer-supplied shim so the metal can sleep between "
            "wakes (utils/device_binding.py)"
        ),
    )
