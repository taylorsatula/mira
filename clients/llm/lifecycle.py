"""Shared fail-loud lifecycle for fixed-route LLM calls."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

from config import config
from clients.llm.events import CompleteEvent, GenerationCancelled, StreamEvent
from clients.llm.dialects.base import Dialect, ProviderProtocolError, ProviderStallError
from clients.llm.dialects.openai_chat_base import ToolNotLoadedError
from clients.llm.types import ProviderMetadata, Request, Result, ToolCall
from utils.user_context import check_cancelled, get_cancel_event

logger = logging.getLogger(__name__)


class LLMLifecycle:
    """Run one provider request with stall detection and no model fallback."""

    def __init__(self, *, response_timeout_seconds: int | None = None) -> None:
        self.response_timeout_seconds = (
            response_timeout_seconds
            if response_timeout_seconds is not None
            else config.api.provider_response_timeout
        )

    def complete(self, request: Request, dialect: Dialect) -> Result:
        final: Result | None = None
        for event in self._run(request, dialect, stream=False):
            if isinstance(event, CompleteEvent):
                final = event.response
        if final is None:
            raise ProviderProtocolError(
                self._endpoint(dialect),
                "non-streaming",
                "Lifecycle ended without completion event",
            )
        return final

    def stream(self, request: Request, dialect: Dialect) -> Iterator[StreamEvent]:
        yield from self._run(request, dialect, stream=True)

    def _run(
        self,
        request: Request,
        dialect: Dialect,
        *,
        stream: bool,
    ) -> Iterator[StreamEvent]:
        try:
            check_cancelled()
            result: Result | None = None
            events = dialect.stream(request) if stream else self._complete_as_events(dialect, request)
            event_iterator = iter(events)

            while True:
                try:
                    event = self._run_with_response_timeout(
                        lambda: next(event_iterator),
                        endpoint=self._endpoint(dialect),
                        mode="streaming" if stream else "non-streaming",
                    )
                except StopIteration:
                    break

                cancel_event = get_cancel_event()
                if cancel_event is not None and cancel_event.is_set():
                    raise GenerationCancelled()

                if isinstance(event, CompleteEvent):
                    result = event.response
                else:
                    yield event

            if result is None:
                raise ProviderProtocolError(
                    self._endpoint(dialect),
                    "streaming" if stream else "non-streaming",
                    "Provider ended without a complete response",
                )

            yield CompleteEvent(
                response=self._with_transport_metadata(result, dialect, request)
            )

        except ToolNotLoadedError as error:
            # Provider attempted to call a tool that was not in the request.
            # Recover by synthesizing an invokeother_tool call so the orchestrator's
            # tool loop loads the tool and re-invokes the provider with it available.
            logger.info(
                "Provider %s called unloaded tool '%s'; recovering via invokeother_tool",
                self._endpoint(dialect),
                error.tool_name,
            )
            synthetic = Result(
                text="",
                tool_calls=(
                    ToolCall(
                        id=f"toolu_{uuid.uuid4().hex[:24]}",
                        tool_name="invokeother_tool",
                        # Shape fixed by invokeother_tool's own schema:
                        # load is an array of tool names; the schema forbids
                        # additional properties, so mode/query would not run.
                        input={"load": [error.tool_name]},
                    ),
                ),
                reasoning=None,
                usage=None,
                stop_reason="tool_use",
                provider_metadata=ProviderMetadata(
                    dialect_name=dialect.dialect_name,
                    model=request.model,
                    endpoint_url=request.metadata.endpoint_url,
                    model_config_name=request.metadata.model_config_name,
                ),
            )
            yield CompleteEvent(response=synthetic)

    @staticmethod
    def _complete_as_events(dialect: Dialect, request: Request) -> Iterator[CompleteEvent]:
        yield CompleteEvent(response=dialect.complete(request))

    def _run_with_response_timeout(
        self,
        operation: Callable[[], Any],
        *,
        endpoint: str,
        mode: str,
    ) -> Any:
        result: list[Any] = []
        errors: list[BaseException] = []

        def invoke() -> None:
            try:
                result.append(operation())
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        worker.join(timeout=self.response_timeout_seconds)
        if worker.is_alive():
            raise ProviderStallError(self._endpoint_label(endpoint), self.response_timeout_seconds, mode)
        if errors:
            raise errors[0]
        return result[0]

    @staticmethod
    def _endpoint(dialect: Dialect) -> str:
        return getattr(dialect, "endpoint_url", dialect.dialect_name)

    @staticmethod
    def _endpoint_label(endpoint: str) -> str:
        return endpoint

    def _with_transport_metadata(
        self,
        result: Result,
        dialect: Dialect,
        request: Request,
    ) -> Result:
        endpoint_url = request.metadata.endpoint_url
        if endpoint_url is None and hasattr(dialect, "endpoint_url"):
            endpoint_url = getattr(dialect, "endpoint_url")
        final = result.with_provider_metadata(
            ProviderMetadata(
                dialect_name=dialect.dialect_name,
                model=request.model,
                endpoint_url=endpoint_url,
                model_config_name=request.metadata.model_config_name,
            )
        )
        self._record_cost(final)
        return final

    @staticmethod
    def _record_cost(result: Result) -> None:
        """Feed the per-request cost summary (D5) from one completed result.

        `cost_accumulator.record()` derives tokens and route name from the
        Result and is a no-op unless a request handler started an accumulator.
        """
        from utils import cost_accumulator

        cost_accumulator.record(result)
