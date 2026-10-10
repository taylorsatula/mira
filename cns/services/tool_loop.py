"""Local tool execution support for orchestrator-owned LLM tool loops."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
from collections.abc import Generator, Iterator
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Any

from clients.llm.events import (
    GenerationCancelled,
    StreamEvent,
    ToolCompletedEvent,
    ToolErrorEvent,
    ToolExecutingEvent,
)
from clients.llm.types import ToolCall, ToolResult
from tools.repo import ParameterError
from utils.user_context import check_cancelled, get_cancel_event

logger = logging.getLogger(__name__)


class ToolReportedError(RuntimeError):
    """A tool ran successfully but reported that its requested operation failed."""

    @classmethod
    def from_result(cls, tool_name: str, result: dict[str, Any]) -> "ToolReportedError":
        raw_error = result.get("error")
        if isinstance(raw_error, dict):
            code = raw_error.get("code") or "TOOL_ERROR"
            message = raw_error.get("message")
        else:
            code = raw_error or "TOOL_ERROR"
            message = result.get("message")
        recovery = result.get("recovery")

        parts = [f"{tool_name} reported {code}"]
        if isinstance(message, str) and message.strip():
            parts.append(message.strip())
        if isinstance(recovery, str) and recovery.strip():
            parts.append(f"Recovery: {recovery.strip()}")
        return cls(" | ".join(parts))


@dataclass
class ToolExecution:
    """Record of a local tool execution used for loop detection."""

    tool_name: str
    result_hash: str | None
    input_hash: str
    error: Exception | None


@dataclass(frozen=True)
class ToolExecutionResult:
    """Result of invoking one local tool call."""

    tool_call: ToolCall
    result_content: str | list[dict[str, Any]]
    raw_result: Any
    hash_material: Any
    error: Exception | None


@dataclass
class CircuitBreaker:
    """Stops local tool chains when the same tool with the same arguments fails twice."""

    tool_results: list[ToolExecution] = field(default_factory=list)

    def record_execution(
        self,
        tool_name: str,
        result: Any,
        input_hash: str,
        error: Exception | None = None,
    ) -> None:
        serialized_result = json.dumps(result, sort_keys=True, default=str) if error is None else ""
        self.tool_results.append(
            ToolExecution(
                tool_name=tool_name,
                result_hash=None if error else hashlib.sha256(serialized_result.encode()).hexdigest(),
                input_hash=input_hash,
                error=error,
            )
        )

    def should_continue(self) -> tuple[bool, str]:
        if not self.tool_results:
            return True, "First tool"
        last = self.tool_results[-1]
        if last.error is not None:
            prior_failures = [
                execution
                for execution in self.tool_results[:-1]
                if execution.tool_name == last.tool_name
                and execution.input_hash == last.input_hash
                and execution.error is not None
            ]
            if prior_failures:
                return (
                    False,
                    f"Tool '{last.tool_name}' failed on repeated attempt with same arguments: {last.error}",
                )
        return True, "Continue"


class ToolLoopExecutor:
    """Executes local tool calls for the orchestrator's model-turn loop."""

    def __init__(self, tool_repo: Any) -> None:
        self.tool_repo = tool_repo

    def execute_tools(
        self,
        tool_calls: list[ToolCall],
        breaker: CircuitBreaker,
    ) -> Generator[StreamEvent, None, tuple[ToolResult, ...]]:
        sequential = []
        parallel = []
        for tool_call in tool_calls:
            tool_class = self.tool_repo.tool_classes.get(tool_call.tool_name)
            if tool_class and not tool_class.is_call_parallel_safe(dict(tool_call.input)):
                sequential.append(tool_call)
            else:
                parallel.append(tool_call)

        results: list[ToolResult] = []
        for tool_call in sequential:
            check_cancelled()
            yield ToolExecutingEvent(
                tool_name=tool_call.tool_name,
                tool_id=tool_call.id,
                arguments=dict(tool_call.input),
            )
            execution = self._execute_tool(tool_call)
            self._emit_tool_result(execution, breaker, results)
            yield from self._events_for_tool_execution(execution)
        if sequential:
            check_cancelled()

        if parallel:
            check_cancelled()
            context = copy_context()
            deferred_cancel: GenerationCancelled | None = None
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {}
                for tool_call in parallel:
                    # A halt must stop later calls from starting, but never
                    # abandons one that already started: its terminal event is
                    # still owed to the provider conversation.
                    cancel_event = get_cancel_event()
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    future = executor.submit(context.copy().run, self._execute_tool, tool_call)
                    futures[future] = tool_call
                    yield ToolExecutingEvent(
                        tool_name=tool_call.tool_name,
                        tool_id=tool_call.id,
                        arguments=dict(tool_call.input),
                    )
                for future in concurrent.futures.as_completed(futures):
                    try:
                        execution = future.result()
                    except GenerationCancelled as cancelled:
                        # A sibling tool observed the cancel signal and
                        # aborted. Every other started tool still owes its
                        # terminal event, so the remaining futures are drained
                        # and their events yielded before the cancellation is
                        # re-raised after the batch (the aborted tool has no
                        # result of its own to report).
                        deferred_cancel = cancelled
                        continue
                    self._emit_tool_result(execution, breaker, results)
                    yield from self._events_for_tool_execution(execution)
            if deferred_cancel is not None:
                raise deferred_cancel
            check_cancelled()

        return tuple(results)

    def _execute_tool(self, tool_call: ToolCall) -> ToolExecutionResult:
        if tool_call.invalid_reason:
            error = ValueError(
                f"Invalid tool call arguments for '{tool_call.tool_name}': "
                f"{tool_call.invalid_reason}"
            )
            logger.warning("Provider returned invalid tool call for %s: %s", tool_call.tool_name, error)
            hint = self._schema_hint(
                tool_call.tool_name, error, invalid_reason=tool_call.invalid_reason
            )
            result_content = f"Error: {error}{hint}"
            return ToolExecutionResult(tool_call, result_content, None, None, error)

        try:
            raw_result = self.tool_repo.invoke_tool(tool_call.tool_name, dict(tool_call.input))
            if isinstance(raw_result, list):
                # Tools returning content blocks directly (e.g. imagegen_tool)
                result_content = raw_result
            elif isinstance(raw_result, dict):
                result_content = json.dumps(raw_result)
            else:
                result_content = str(raw_result)
            if isinstance(raw_result, dict) and raw_result.get("success") is False:
                # A structured rejection is an operation failure the model can
                # correct, not a success. Without this the input-aware circuit
                # breaker sees no error and lets the model repeat the identical
                # failing call forever.
                error = ToolReportedError.from_result(tool_call.tool_name, raw_result)
                logger.warning("Tool reported an operation failure: %s", error)
                return ToolExecutionResult(
                    tool_call,
                    result_content,
                    raw_result,
                    None,
                    error,
                )
            return ToolExecutionResult(
                tool_call,
                result_content,
                raw_result,
                self._tool_result_hash_material(raw_result, result_content),
                None,
            )
        except GenerationCancelled:
            # A cancellation signal raised inside a tool must reach the
            # orchestrator's stopped-turn semantics. The broad handler
            # below would otherwise convert it into model-facing error
            # feedback and the turn would continue as if the tool had
            # merely failed — the cancel signal downgraded to content.
            raise
        except Exception as error:
            logger.error("Tool execution failed for %s: %s", tool_call.tool_name, error, exc_info=True)
            result_content = f"Error: {error}{self._schema_hint(tool_call.tool_name, error)}"
            return ToolExecutionResult(tool_call, result_content, None, None, error)

    def _tool_result_hash_material(
        self,
        raw_result: Any,
        result_content: str | list[dict[str, Any]],
    ) -> Any:
        return raw_result

    def _schema_hint(self, tool_name: str, error: Exception, *, invalid_reason: str | None = None) -> str:
        # Match by type, never by message substring: only genuine argument
        # errors (ParameterError from tools.repo, or a provider-rejected
        # call flagged with invalid_reason) get the CORRECT PARAMETERS hint.
        # Operational failures (connection errors, tool-body TypeErrors,
        # arbitrary ValueErrors) must not misdirect the model's recovery.
        is_parameter_error = isinstance(error, ParameterError) or invalid_reason is not None
        if not is_parameter_error:
            return ""
        try:
            definition = self.tool_repo.get_tool_definition(tool_name)
            properties = dict(definition.input_schema.get("properties", {}))
        except (AttributeError, KeyError):
            return ""
        return f"\n\nCORRECT PARAMETERS:\n{json.dumps(properties, indent=2)}"

    def _emit_tool_result(
        self,
        execution: ToolExecutionResult,
        breaker: CircuitBreaker,
        results: list[ToolResult],
    ) -> None:
        input_hash = hashlib.sha256(
            json.dumps(dict(execution.tool_call.input), sort_keys=True, default=str).encode()
        ).hexdigest()
        breaker.record_execution(
            execution.tool_call.tool_name,
            execution.hash_material,
            input_hash,
            execution.error,
        )
        results.append(ToolResult(
            tool_call_id=execution.tool_call.id,
            content=execution.result_content,
            is_error=execution.error is not None,
        ))

    def _events_for_tool_execution(
        self,
        execution: ToolExecutionResult,
    ) -> Iterator[StreamEvent]:
        if execution.error is not None:
            yield ToolErrorEvent(
                tool_name=execution.tool_call.tool_name,
                tool_id=execution.tool_call.id,
                error=str(execution.error),
                result=execution.result_content,
            )
            return
        yield ToolCompletedEvent(
            tool_name=execution.tool_call.tool_name,
            tool_id=execution.tool_call.id,
            result=execution.result_content,
        )
