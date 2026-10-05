"""
SidebarAgent -- Base class for autonomous sidebar agents.

Encapsulates LLM-in-a-loop mechanics so implementations only define
domain logic: prompt, initial message, and tool list.

Loop terminates when the LLM calls sidebar_tool.complete_task().
The base class intercepts that call, enriches it with work item metadata,
executes the tool (which writes the activity record to SQLite), publishes
an UpdateTrinketEvent to refresh the agent's completion trinket
(AsyncActivityTrinket by default; subclasses override
_get_completion_trinket() / _build_completion_context() to target a
different trinket), and exits.
"""
import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING
from typing_extensions import NotRequired, TypedDict

from clients.llm.tool_messages import (
    append_tool_result_messages,
    assistant_message_from_result,
)
from clients.llm.types import ToolCall, ToolResult
from clients.llm_provider import LLMProvider, get_llm_provider
from utils.timezone_utils import utc_now, format_utc_iso
from utils.userdata_manager import (
    ACTIVITY_INDEX_DDL,
    ACTIVITY_TABLE_DDL,
    SCRATCHPAD_INDEX_DDL,
    SCRATCHPAD_TABLE_DDL,
)

from config import config

_OVERWATCH_SYSTEM_PROMPT = (
    "You produce one-sentence research progress log entries. "
    "Each entry reports what an agent found or attempted in a single "
    "iteration. Lead with findings, not process. Be specific — name "
    "topics, counts, sources. The research query is already shown in "
    "context; don't echo it.\n/nothink"
)

if TYPE_CHECKING:
    from agents.sidebar import WorkItem
    from tools.repo import ToolRepository
    from cns.integration.event_bus import EventBus

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Prompt loading
# -----------------------------------------------------------------------

def load_agent_prompt(filename: str) -> str:
    """Load a prompt file from config/prompts/agents/."""
    from config.prompts.loader import load_prompt
    return load_prompt(f"agents/{filename}")


def ensure_activity_schema(db) -> None:
    """Create sidebar_activity + scratchpad tables.

    Safe to call repeatedly -- CREATE IF NOT EXISTS handles existing tables.
    The DDL is owned by utils/userdata_manager.py with the other per-user
    SQLite schemas.
    """
    db.execute(ACTIVITY_TABLE_DDL)
    db.execute(ACTIVITY_INDEX_DDL)
    db.execute(SCRATCHPAD_TABLE_DDL)
    db.execute(SCRATCHPAD_INDEX_DDL)


def upsert_activity_record(
    db,
    *,
    interface_name: str,
    thread_id: str,
    agent_id: str,
    summary: str,
    status: str,
    escalation_reason: str | None = None,
    run_count: int = 1,
) -> None:
    """Write one activity-feed item, replacing the same (interface_name, thread_id).

    Every column is overwritten on conflict, escalation_reason included. Callers:
    sidebar_tool.complete_task and utils/self_edit.deliver_result.
    """
    db.execute(
        "INSERT INTO sidebar_activity "
        "(interface_name, thread_id, agent_id, summary, status, "
        "escalation_reason, run_count, updated_at) "
        "VALUES (:interface_name, :thread_id, :agent_id, :summary, "
        ":status, :escalation_reason, :run_count, datetime('now')) "
        "ON CONFLICT(interface_name, thread_id) DO UPDATE SET "
        "agent_id = excluded.agent_id, summary = excluded.summary, "
        "status = excluded.status, escalation_reason = excluded.escalation_reason, "
        "run_count = excluded.run_count, updated_at = datetime('now')",
        {
            'interface_name': interface_name,
            'thread_id': thread_id,
            'agent_id': agent_id,
            'summary': summary,
            'status': status,
            'escalation_reason': escalation_reason,
            'run_count': run_count,
        },
    )


# -----------------------------------------------------------------------
# Structured types for trace data
# -----------------------------------------------------------------------

class ToolCallTrace(TypedDict):
    tool_name: str
    input: dict[str, Any]
    result: Any
    is_error: bool


class IterationTrace(TypedDict):
    iteration: int
    assistant_text: str
    tool_calls: list[ToolCallTrace]


class SentryTrace(TypedDict):
    model: str | None
    decision: Literal['proceed', 'skip']
    reason: str


class AgentTrace(TypedDict):
    agent_id: str
    work_item_id: str
    interface_name: str
    model: str | None
    started_at: str
    iterations: list[IterationTrace]
    status: str | None
    completed_at: NotRequired[str]
    total_iterations: NotRequired[int]
    sentry: NotRequired[SentryTrace]


# -----------------------------------------------------------------------
# SidebarAgent ABC
# -----------------------------------------------------------------------

class SidebarAgent(ABC):
    """
    Abstract base class for autonomous sidebar agents.

    Implementations define:
        agent_id         -- unique identifier (e.g. "forage")
        model_config_name -- one of the five model_configs routes:
                            primary, fast, batch, assessment, other
        available_tools  -- domain-specific tool names from registry
                            (sidebar_tool is always included automatically)
        get_agent_prompt()          -> str  -- agent-specific rubric
        build_initial_message(item) -> str  -- first user message

    The base class owns: LLM init, tool schema assembly (always includes
    sidebar_tool + implementation tools), message loop with heartbeat,
    tool execution, complete_task detection, trace capture, and activity
    publishing.

    Optional features (set the relevant attribute to activate):
        sentry_model_config_name -- cheap pre-filter; see build_sentry_message()
        overwatch_model_config_name -- passive iteration observer
        max_retries              -- dispatcher retry-on-failure threshold

    Injection gate (fail closed, ON by default):
        sanitize_untrusted_input -- screens work_item.context['raw_content']
            through utils.untrusted_content.screen_untrusted before the LLM
            loop. Override to False ONLY for agents whose work items carry
            no external/untrusted content.

    Completion publishing:
        Override _get_completion_trinket() and _build_completion_context()
        to publish to a non-default trinket.
    """

    # Required -- subclasses must define
    agent_id: str
    model_config_name: str
    available_tools: list[str]

    def __init__(self, tool_repo: 'ToolRepository'):
        self._tool_repo = tool_repo
        self._trace: AgentTrace | None = None
        self._overwatch_history: list[str] = []
        self._work_item: 'WorkItem | None' = None
        self._event_bus: 'EventBus | None' = None

        # Wall-clock timeouts resolved from config: per-agent override keyed by
        # class name (ForageAgent -> 'forage'), falling back to the shared default.
        dispatcher_config = config.sidebar_dispatcher
        agent_key = type(self).__name__.removesuffix("Agent").lower()
        self.timeout_seconds = dispatcher_config.agent_timeout_overrides.get(
            agent_key, dispatcher_config.agent_timeout_seconds
        )
        self.iteration_timeout_seconds = dispatcher_config.agent_iteration_timeout_seconds
        self.blocking_timeout_seconds = dispatcher_config.blocking_agent_timeout_seconds
        # Blocking-mode state: _blocking is only ever true for the duration
        # of one run_blocking() call; _completion_record is written by _exit
        # so run_blocking can hand the terminal status + summary back to the
        # caller (background runs leave it unread).
        self._blocking = False
        self._completion_record: dict[str, Any] | None = None

    # Optional -- subclasses override as needed
    inherit_base_prompt: bool = True
    max_iterations: int = 5
    # Injection gate is ON by default (fail closed): any work item carrying
    # untrusted 'raw_content' must pass semantic screening before the LLM
    # loop; a rejection exits 'rejected', a screen outage fails the run — it
    # is never passed through unscreened. Subagents with only trusted
    # internal data opt out explicitly (see MemoryCuratorAgent).
    sanitize_untrusted_input: bool = True
    max_retries: int = 0  # 0 = fire-and-forget, no retry on failure

    # Per-tool schema overrides. Maps tool name → custom tool_schema.
    # Used to restrict tool capabilities for sidebar agents (e.g.
    # email_tool → reply_to_email only).
    tool_schema_overrides: dict[str, dict[str, Any]] = {}

    # Blocking mode -- opt-in synchronous execution via run_blocking().
    # A blocking run executes the full loop on the caller's thread with a
    # tight wall-clock bound (blocking_agent_timeout_seconds), suppresses
    # every trinket publication (completion event and overwatch -- the
    # result travels entirely through run_blocking()'s return value), and
    # still writes the sidebar_activity record, so the dedup/retry contract
    # with the dispatcher is unchanged. Off by default; subclasses that can
    # run inside a bounded turn set it to True.
    blocking_supported: bool = False

    # Sentry gate -- opt-in cheap pre-filter before the main loop.
    # Set sentry_model_config_name to activate. The gate exists to discard
    # work items cheaply, so it belongs on `fast` like the other high-volume
    # auxiliary judgments (subcortical, tool-result summary, entity merge,
    # injection defense); sentry_max_tokens stays a per-request ceiling so a
    # route change cannot widen it.
    sentry_model_config_name: str | None = None
    sentry_max_tokens: int = 150

    # Overwatch -- opt-in passive observer that summarizes each iteration
    # via a cheap one-shot LLM call in a background thread. The agent
    # loop is unaware of the observer. Set overwatch_model_config_name to activate.
    # overwatch_max_tokens is load-bearing: the observer's route row ceiling
    # is generous (4096 on `fast`, 16000 on `batch`/`primary`), and a per-request
    # override is the only thing keeping the observer's output to one summary
    # line.
    overwatch_model_config_name: str | None = None
    overwatch_max_tokens: int = 80

    @abstractmethod
    def get_agent_prompt(self, work_item: 'WorkItem') -> str:
        """Return the agent-specific system prompt / rubric."""
        ...

    @abstractmethod
    def build_initial_message(self, work_item: 'WorkItem') -> str:
        """Build the initial user message from the work item context."""
        ...

    def get_heartbeat(self, iteration: int) -> str:
        """Heartbeat injected as user message between iterations."""
        return "Continue."

    # ------------------------------------------------------------------
    # Overwatch -- passive iteration observer
    # ------------------------------------------------------------------

    def get_overwatch_context(self, work_item: 'WorkItem') -> str:
        """One-line task context for the overwatch observer.

        Override to provide agent-specific context (e.g. the forage query).
        """
        return f"Task: {work_item.interface_name}"

    def on_overwatch_update(
        self,
        event_bus: 'EventBus',
        work_item: 'WorkItem',
        iteration: int,
        summary: str,
    ) -> None:
        """Called with the overwatch one-sentence summary.

        Override to publish progress to the appropriate trinket.
        Default is no-op — agents that don't set overwatch_model_config_name
        never reach this.
        """

    def _fire_overwatch(
        self,
        event_bus: 'EventBus',
        work_item: 'WorkItem',
        iteration: int,
        assistant_text: str,
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
    ) -> None:
        """Spawn background thread for non-blocking overwatch LLM call."""
        if not self.overwatch_model_config_name:
            return
        if self._blocking:
            # Blocking runs publish no trinket events at all — an in_progress
            # overwatch entry with no terminal event following would strand
            # the trinket state machine in a non-terminal state.
            return

        import contextvars
        from threading import Thread

        # Snapshot prior log entries for continuity — the overwatch
        # thread appends its result after completion, so the next
        # iteration's snapshot will include it (best-effort ordering).
        prior_entries = list(self._overwatch_history)

        ctx = contextvars.copy_context()
        thread = Thread(
            target=ctx.run,
            args=(
                self._run_overwatch, event_bus, work_item, iteration,
                assistant_text, tool_calls, tool_results, prior_entries,
            ),
            daemon=True,
        )
        thread.start()

    def _run_overwatch(
        self,
        event_bus: 'EventBus',
        work_item: 'WorkItem',
        iteration: int,
        assistant_text: str,
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
        prior_entries: list[str],
    ) -> None:
        """Overwatch thread body — one-shot LLM call, then publish summary."""
        try:
            assert self.overwatch_model_config_name is not None
            llm = get_llm_provider()
            task_context = self.get_overwatch_context(work_item)
            prompt = _build_overwatch_prompt(
                task_context, iteration, self.max_iterations,
                assistant_text, tool_calls, tool_results,
                prior_entries,
            )

            response = llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model_config=self.overwatch_model_config_name,
                max_tokens=self.overwatch_max_tokens,
                system_prompt=_OVERWATCH_SYSTEM_PROMPT,
            )

            summary = llm.extract_text_content(response).strip()
            if summary:
                self._overwatch_history.append(summary)
                self.on_overwatch_update(
                    event_bus, work_item, iteration, summary
                )

        except Exception:
            logger.debug("%s: Overwatch failed", self.agent_id, exc_info=True)

    def build_recovery_context(self, prior_run: dict) -> str | None:
        """Build recovery context from a prior failed run.

        Called by the base class before the LLM loop when retrying a
        previously failed item. Override in subclasses to provide
        failure-aware initial context. The prior_run dict carries ONLY
        the dispatcher-selected keys 'status' and 'run_count' — it is
        not a copy of the sidebar_activity row; reading other keys
        (e.g. prior_run['summary']) raises KeyError inside run()'s
        try block and consumes the retry.

        Returns text prepended to the initial message, or None (default).
        """
        return None

    # ------------------------------------------------------------------
    # Sentry gate -- cheap one-shot LLM pre-filter
    # ------------------------------------------------------------------

    def build_sentry_message(self, work_item: 'WorkItem') -> str:
        """Build the sentry evaluation prompt.

        Only called when sentry_model_config_name is set. Must return a single
        user-role message asking the cheap model to evaluate whether
        the agent should proceed.

        Response format expected by default parse_sentry_response():
            <decision>proceed|skip</decision>
            <reason>Brief explanation</reason>
        """
        raise NotImplementedError(
            f"{self.agent_id} has sentry_model_config_name set but no "
            "build_sentry_message() implementation"
        )

    def parse_sentry_response(
        self, response_text: str
    ) -> tuple[bool, str]:
        """Parse sentry LLM response into (should_proceed, reason).

        Default: XML parser for <decision>proceed|skip</decision>
        and <reason>...</reason>. Fails open on parse failure.
        Override for custom response formats.
        """
        decision_match = re.search(
            r'<decision>\s*(proceed|skip)\s*</decision>', response_text
        )
        reason_match = re.search(
            r'<reason>\s*(.*?)\s*</reason>', response_text, re.DOTALL
        )

        if not decision_match:
            logger.warning(
                "%s: Sentry response missing <decision> tag, proceeding",
                self.agent_id,
            )
            return (True, "sentry parse failed — proceeding")

        should_proceed = decision_match.group(1) == 'proceed'
        reason = (
            reason_match.group(1) if reason_match else 'no reason given'
        )
        return (should_proceed, reason)

    def _run_sentry(
        self,
        work_item: 'WorkItem',
        trace: AgentTrace,
        event_bus: 'EventBus',
    ) -> bool:
        """Run the sentry gate. Returns True to proceed, False to skip.

        On skip: routes through _exit() with activity_status='dismissed'.
        On error: logs warning and returns True (fail open).
        """
        assert trace is not None
        assert event_bus is not None
        try:
            from utils.user_context import get_model_config

            assert self.sentry_model_config_name is not None
            sentry_cfg = get_model_config(self.sentry_model_config_name)
            llm = get_llm_provider()
            message = self.build_sentry_message(work_item)
            response = llm.generate_response(
                messages=[{"role": "user", "content": message}],
                model_config=self.sentry_model_config_name,
                max_tokens=self.sentry_max_tokens,
            )

            response_text = llm.extract_text_content(response)
            should_proceed, reason = self.parse_sentry_response(
                response_text
            )

            trace['sentry'] = SentryTrace(
                model=sentry_cfg.model,
                decision='proceed' if should_proceed else 'skip',
                reason=reason,
            )

            if not should_proceed:
                self._exit('skipped', reason, activity_status='dismissed')
                return False

            return True

        except Exception as e:
            logger.warning(
                "%s: Sentry failed, proceeding", self.agent_id, exc_info=True
            )
            trace['sentry'] = SentryTrace(
                model=None,
                decision='proceed',
                reason=f'sentry error: {e}',
            )
            return True

    # ------------------------------------------------------------------
    # Completion publication -- override to target a different trinket
    # ------------------------------------------------------------------

    def _get_completion_trinket(self) -> str:
        """Return the trinket name to publish completion to.
        
        Override in subclasses that publish to a non-default trinket.
        """
        return 'AsyncActivityTrinket'

    def _build_completion_context(
        self,
        status: str,
        summary: str,
        work_item: 'WorkItem',
    ) -> dict[str, Any]:
        """Build the context dict for the completion event.
        
        Override in subclasses that need additional fields (e.g. ForageAgent
        adds iterations, error_type, result/error).
        """
        return {
            'task_id': work_item.item_id,
            'status': status,
        }

    def on_completion(
        self,
        event_bus: 'EventBus',
        work_item: 'WorkItem',
        status: str,
        summary: str,
    ) -> None:
        """Called after agent completes (success or failure).

        Default publishes to AsyncActivityTrinket. Override _get_completion_trinket()
        and _build_completion_context() for custom trinkets. Blocking runs
        (run_blocking) publish nothing — the result is the caller's return
        value — so the guard here is the single suppression point.
        """
        if self._blocking:
            return
        from cns.core.events import UpdateTrinketEvent
        event_bus.publish(UpdateTrinketEvent.create(
            continuum_id='sidebar',
            target_trinket=self._get_completion_trinket(),
            context=self._build_completion_context(status, summary, work_item),
        ))

    # ------------------------------------------------------------------
    # System prompt assembly
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self, work_item: 'WorkItem', iteration: int = 1
    ) -> str:
        agent_prompt = self.get_agent_prompt(work_item)
        parts: list[str] = []
        if self.inherit_base_prompt:
            parts.append(load_agent_prompt("base_system.txt"))
        parts.append(agent_prompt)
        # A matched user trigger rule's prompt (attached by
        # rule_filter.apply_trigger_rules as context['rule_prompt']) is
        # appended here — the single consumption point, so every agent
        # honors rule prompts without per-agent wiring.
        rule_prompt = work_item.context.get("rule_prompt")
        if rule_prompt:
            parts.append(rule_prompt)
        status = self._iteration_status(iteration)
        if status:
            parts.append(status)
        return "\n\n".join(parts)

    def _iteration_status(self, iteration: int) -> str | None:
        """Optional per-iteration system-prompt addendum. Default: none."""
        return None

    # ------------------------------------------------------------------
    # Input sanitization
    # ------------------------------------------------------------------

    def _sanitize_work_item(self, work_item: 'WorkItem') -> None:
        """Screen raw_content, write the wrapped sanitized_content back.

        Called when sanitize_untrusted_input is True, before the LLM loop.
        raw_content is truncated to 8000 chars first, so the screen judges
        exactly what reaches the agent and the wrapper's closing tag survives.
        A rejection raises InjectionRejected so the agent exits through
        _exit('rejected'); every other error (screen outage, missing
        credential) propagates to run() and exits 'failed' — never passed
        through unscreened.
        """
        from utils.untrusted_content import screen_untrusted

        raw = work_item.context.get("raw_content", "")
        if len(raw) > 8000:
            raw = raw[:8000] + "\n[truncated]"
        work_item.context["sanitized_content"] = screen_untrusted(raw, work_item.interface_name)

    # ------------------------------------------------------------------
    # Tool schema assembly
    # ------------------------------------------------------------------

    def _get_all_tool_names(self) -> list[str]:
        """sidebar_tool is always included; subclass tools are appended."""
        names = ['sidebar_tool']
        for name in self.available_tools:
            if name not in names:
                names.append(name)
        return names

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        work_item: 'WorkItem',
        event_bus: 'EventBus',
        *,
        timeout_seconds: int | None = None,
    ) -> None:
        """Execute the agent loop. Implementations should not override this.

        timeout_seconds overrides the instance's background wall-clock bound
        for this run only — run_blocking() passes blocking_timeout_seconds so
        a blocking call is bounded tightly even when the class's background
        override (agent_timeout_overrides) is minutes long.
        """
        self._work_item = work_item
        self._event_bus = event_bus
        start_time = utc_now()
        self._trace = _init_trace(self.agent_id, work_item, start_time)
        self._overwatch_history = []

        try:
            tool_schemas = self._build_tool_schemas()
            if not tool_schemas:
                self._exit('failed', 'No tools available')
                return

            llm, llm_cfg = self._resolve_llm()

            if self._run_injection_gate(work_item) is False:
                return

            if self._run_sentry_gate(work_item) is False:
                return

            initial_message = self.build_initial_message(work_item)

            prior_run = work_item.context.get('prior_run')
            if prior_run:
                recovery = self.build_recovery_context(prior_run)
                if recovery:
                    initial_message = f"{recovery}\n\n{initial_message}"

            messages: list[dict[str, Any]] = [{
                "role": "user",
                "content": initial_message,
            }]

            prev_iteration_start: Any = None
            for iteration in range(1, self.max_iterations + 2):
                iteration_start = utc_now()

                elapsed = (iteration_start - start_time).total_seconds()
                wall_limit = (
                    timeout_seconds if timeout_seconds is not None
                    else self.timeout_seconds
                )
                if elapsed > wall_limit:
                    self._exit('timeout', f'Agent timed out after {elapsed:.0f}s')
                    return

                if iteration > 1:
                    iter_elapsed = (iteration_start - prev_iteration_start).total_seconds()
                    if iter_elapsed > self.iteration_timeout_seconds:
                        self._exit('timeout', f'Iteration timed out after {iter_elapsed:.0f}s')
                        return

                prev_iteration_start = iteration_start

                system_prompt = self._build_system_prompt(work_item, iteration)
                if self._run_iteration(iteration, messages, llm, tool_schemas, llm_cfg, system_prompt):
                    return

            self._exit('failed', 'Agent hit iteration cap without completing')

        except Exception as e:
            logger.exception("%s: Failed", self.agent_id)
            self._exit('failed', f'Agent error: {e}')

        finally:
            self._finalize_trace()

    def run_blocking(
        self,
        work_item: 'WorkItem',
        event_bus: 'EventBus',
    ) -> dict[str, Any]:
        """Run the agent loop synchronously on the caller's thread.

        Returns the completion record {'status', 'summary'} — 'success'
        carries the agent's briefing in 'summary'; every other terminal
        status ('timeout', 'failed', 'rejected', 'skipped') carries the
        failure explanation there instead. Raises only when the class opts
        out (blocking_supported false) or the base run/exit contract is
        broken; a failing agent run is a returned record, not a raise.

        Differences from run(): wall-clock bound is
        blocking_timeout_seconds (tight, turn-scale) instead of the class's
        background override; no trinket events are published at all
        (on_completion and overwatch both check _blocking) — the result
        travels entirely through this return value. The sidebar_activity
        record still lands via _exit, so dispatcher dedup/retry is unchanged.
        The caller's thread provides the user contextvar; no context copy
        is made, no thread is spawned.
        """
        if not self.blocking_supported:
            raise ValueError(
                f"{type(self).__name__} does not support blocking execution"
            )
        self._blocking = True
        try:
            self.run(
                work_item, event_bus,
                timeout_seconds=self.blocking_timeout_seconds,
            )
        finally:
            self._blocking = False
        if self._completion_record is None:
            # run() routes every termination through _exit, which always
            # writes the record; None here means the base contract broke —
            # fail loud rather than hand back an invented result.
            raise RuntimeError(
                f"{self.agent_id}: blocking run ended without a completion record"
            )
        return self._completion_record

    def _exit(
        self,
        status: str,
        summary: str,
        *,
        activity_status: str = 'failed',
    ) -> None:
        """Shared exit path for all termination cases."""
        assert self._trace is not None
        assert self._work_item is not None
        assert self._event_bus is not None
        self._trace['status'] = status
        # Terminal record for run_blocking(): status + summary as the caller
        # receives them. Written unconditionally; background runs simply
        # never have it read back.
        self._completion_record = {'status': status, 'summary': summary}
        _write_activity_record(
            self._work_item, self.agent_id, summary,
            run_count=self._work_item.context.get('run_count', 1),
            status=activity_status,
        )
        self.on_completion(self._event_bus, self._work_item, status, summary)

    def _finalize_trace(self) -> None:
        """Save trace after agent terminates."""
        assert self._trace is not None
        assert self._work_item is not None
        self._trace['completed_at'] = format_utc_iso(utc_now())
        self._trace['total_iterations'] = len(self._trace['iterations'])
        _save_trace(self.agent_id, self._work_item.item_id, self._trace)

    def _resolve_llm(self) -> tuple['LLMProvider', Any]:
        """Resolve the agent's fixed model route."""
        from utils.user_context import get_model_config

        assert self._trace is not None
        llm_cfg = get_model_config(self.model_config_name)
        llm = get_llm_provider()
        self._trace['model'] = llm_cfg.model
        return llm, llm_cfg

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        """Collect all tool schemas for this agent."""
        all_tool_names = self._get_all_tool_names()
        return _get_tool_schemas(
            self._tool_repo, all_tool_names, self.agent_id,
            self.tool_schema_overrides,
        )

    def _run_injection_gate(self, work_item: 'WorkItem') -> bool:
        """Run injection defense if enabled. Returns True to proceed, False to exit."""
        if not self.sanitize_untrusted_input:
            return True
        from utils.untrusted_content import InjectionRejected

        try:
            self._sanitize_work_item(work_item)
            return True
        except InjectionRejected as e:
            logger.warning("%s: Input rejected: %s", self.agent_id, e)
            self._exit('rejected', f'Input rejected: {e}')
            return False

    def _run_sentry_gate(self, work_item: 'WorkItem') -> bool:
        """Run sentry gate if configured. Returns True to proceed, False to exit."""
        if self.sentry_model_config_name is None:
            return True
        assert self._trace is not None
        assert self._event_bus is not None
        return self._run_sentry(
            work_item, self._trace, self._event_bus
        )

    def _run_iteration(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        llm: 'LLMProvider',
        tool_schemas: list[dict[str, Any]],
        llm_cfg: Any,
        system_prompt: str,
    ) -> bool:
        """Execute a single iteration. Returns True if agent completed."""
        assert self._trace is not None

        response = llm.generate_response(
            messages=messages,
            tools=tool_schemas,
            model_config=llm_cfg.name,
            system_prompt=system_prompt,
        )

        tool_calls = llm.extract_tool_calls(response)
        assistant_text = llm.extract_text_content(response)

        if not tool_calls:
            messages.append(assistant_message_from_result(response))
            self._trace['iterations'].append(IterationTrace(
                iteration=iteration,
                assistant_text=assistant_text,
                tool_calls=[],
            ))
            messages.append({
                "role": "user",
                "content": (
                    "You must call sidebar_tool complete_task to "
                    "finish your work."
                ),
            })
            return False

        completed, tool_results = self._execute_tool_calls(tool_calls)

        self._trace['iterations'].append(IterationTrace(
            iteration=iteration,
            assistant_text=assistant_text,
            tool_calls=[
                ToolCallTrace(
                    tool_name=tc.tool_name,
                    input=dict(tc.input),
                    result=_parse_result(tr.content),
                    is_error=tr.is_error,
                )
                for tc, tr in zip(tool_calls, tool_results)
            ],
        ))

        if completed:
            complete_result = _parse_result(tool_results[-1].content)
            summary = (
                complete_result.get('summary', '')
                if isinstance(complete_result, dict) else ''
            )
            # complete_task is terminal: the break in _execute_tool_calls
            # guarantees tool_results[-1] is its envelope. Route its own
            # status through to _exit -- the activity UPSERT default
            # ('failed') would otherwise re-dispatch completed items.
            activity_status = (
                complete_result.get('status', 'handled')
                if isinstance(complete_result, dict) else 'handled'
            )
            self._exit('success', summary, activity_status=activity_status)
            return True

        assert self._event_bus is not None
        assert self._work_item is not None
        self._fire_overwatch(
            self._event_bus, self._work_item, iteration,
            assistant_text, tool_calls, tool_results,
        )

        heartbeat = (
            f"You have hit the {self.max_iterations}-iteration "
            "limit imposed on this agent. Call sidebar_tool "
            "complete_task now with whatever you have. Summarize "
            "what you accomplished and what remains."
            if iteration == self.max_iterations
            else self.get_heartbeat(iteration)
        )

        messages[:] = append_tool_result_messages(messages, response, tuple(tool_results))
        messages.append({"role": "user", "content": heartbeat})
        return False

    def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
    ) -> tuple[bool, list[ToolResult]]:
        """Execute tool calls, inject identity into sidebar_tool, detect completion."""
        assert self._work_item is not None
        tool_results: list[ToolResult] = []
        completed = False

        for tc in tool_calls:
            is_sidebar = tc.tool_name == 'sidebar_tool'
            is_complete = (
                is_sidebar
                and dict(tc.input).get('operation') == 'complete_task'
            )
            if is_sidebar:
                # Mutate a copy of the frozen input mapping for sidebar identity injection
                mutable_input = dict(tc.input)
                mutable_input['thread_id'] = self._work_item.item_id
                if is_complete:
                    mutable_input['interface_name'] = self._work_item.interface_name
                    mutable_input['agent_id'] = self.agent_id
                    mutable_input['run_count'] = self._work_item.context.get('run_count', 1)
                tc = ToolCall(id=tc.id, tool_name=tc.tool_name, input=mutable_input)

            result = _execute_tool_call(
                self._tool_repo, tc, self.agent_id
            )
            tool_results.append(result)

            if is_complete and not result.is_error:
                completed = True
                break

        return completed, tool_results


# -----------------------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------------------

def _init_trace(
    agent_id: str, work_item: 'WorkItem', start_time: Any
) -> AgentTrace:
    """Initialize trace dict. completed_at and total_iterations set in finally."""
    return AgentTrace(
        agent_id=agent_id,
        work_item_id=work_item.item_id,
        interface_name=work_item.interface_name,
        model=None,
        started_at=format_utc_iso(start_time),
        iterations=[],
        status=None,
    )


def _get_tool_schemas(
    tool_repo: 'ToolRepository',
    tool_names: list[str],
    agent_id: str,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    schemas = []
    for tool_name in tool_names:
        if overrides and tool_name in overrides:
            schemas.append(overrides[tool_name])
            continue
        try:
            tool = tool_repo.get_tool(tool_name)
            if tool:
                schemas.append(tool.tool_schema)
        except Exception:
            logger.debug("%s: Tool %s unavailable", agent_id, tool_name, exc_info=True)
    return schemas


def _execute_tool_call(
    tool_repo: 'ToolRepository',
    tc: ToolCall,
    agent_id: str,
) -> ToolResult:
    try:
        if tc.invalid_reason:
            raise ValueError(
                f"Invalid tool call arguments for '{tc.tool_name}': {tc.invalid_reason}"
            )
        tool = tool_repo.get_tool(tc.tool_name)
        result = tool.run(**dict(tc.input))
        return ToolResult(
            tool_call_id=tc.id,
            content=json.dumps(result, default=str),
        )
    except Exception as e:
        logger.warning("%s: Tool %s failed", agent_id, tc.tool_name, exc_info=True)
        return ToolResult(
            tool_call_id=tc.id,
            content=json.dumps({"error": str(e)}),
            is_error=True,
        )


def _parse_result(content: str) -> Any:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content


def _write_activity_record(
    work_item: 'WorkItem',
    agent_id: str,
    summary: str,
    run_count: int = 1,
    status: str = 'failed',
) -> None:
    """Write an activity record to SQLite via UPSERT."""
    from utils.userdata_manager import get_user_data_manager
    from utils.user_context import get_current_user_id

    db = get_user_data_manager(get_current_user_id())
    ensure_activity_schema(db)
    db.execute(
        "INSERT INTO sidebar_activity "
        "(interface_name, thread_id, agent_id, summary, status, "
        "run_count, updated_at) "
        "VALUES (:interface_name, :thread_id, :agent_id, :summary, "
        ":status, :run_count, datetime('now')) "
        "ON CONFLICT(interface_name, thread_id) DO UPDATE SET "
        "agent_id = excluded.agent_id, summary = excluded.summary, "
        "status = excluded.status, run_count = excluded.run_count, "
        "updated_at = datetime('now')",
        {
            'interface_name': work_item.interface_name,
            'thread_id': work_item.item_id,
            'agent_id': agent_id,
            'summary': summary,
            'status': status,
            'run_count': run_count,
        },
    )


def _publish_trinket_refresh(event_bus: 'EventBus') -> None:
    """Publish UpdateTrinketEvent so AsyncActivityTrinket re-renders."""
    try:
        from cns.core.events import UpdateTrinketEvent
        event_bus.publish(UpdateTrinketEvent.create(
            continuum_id='sidebar',
            target_trinket='AsyncActivityTrinket',
            context={'action': 'refresh'},
        ))
    except Exception:
        logger.exception("Failed to publish trinket refresh")


def _build_overwatch_prompt(
    task_context: str,
    iteration: int,
    max_iterations: int,
    assistant_text: str,
    tool_calls: list[ToolCall],
    tool_results: list[ToolResult],
    prior_entries: list[str],
) -> str:
    """Build compact prompt for the overwatch observer model."""
    header = (
        f"Iteration {iteration} (grace)"
        if iteration > max_iterations
        else f"Iteration {iteration}/{max_iterations}"
    )
    parts = [f"{task_context}\n{header}"]

    # Prior log entries for continuity — the observer sees the arc
    if prior_entries:
        log_lines = [f"[{i+1}] {entry}" for i, entry in enumerate(prior_entries)]
        parts.append("Log so far:\n" + "\n".join(log_lines))

    # Tools first — most concrete data about what happened
    tool_lines = []
    for tc, tr in zip(tool_calls, tool_results):
        name = tc.tool_name
        if name == 'sidebar_tool':
            continue
        result_str = (tr.content[:200] if isinstance(tr.content, str) else json.dumps(tr.content)[:200])
        tool_lines.append(f"- {name} → {result_str}")
    if tool_lines:
        parts.append("This iteration's tools:\n" + "\n".join(tool_lines))

    if assistant_text:
        text = assistant_text[:300]
        if len(assistant_text) > 300:
            text += '…'
        parts.append(f"Agent notes:\n{text}")

    return "\n\n".join(parts)


def _save_trace(
    agent_id: str, item_id: str, trace: AgentTrace
) -> None:
    try:
        from utils.userdata_manager import get_user_data_manager
        from utils.user_context import get_current_user_id

        db = get_user_data_manager(get_current_user_id())
        trace_dir = Path(db.get_tool_data_dir('sidebar_agent'))
        path = trace_dir / f"{item_id}.json"
        with open(path, 'w') as f:
            json.dump(trace, f, indent=2, default=str)
    except Exception:
        logger.warning("%s: Failed to save trace", agent_id, exc_info=True)
