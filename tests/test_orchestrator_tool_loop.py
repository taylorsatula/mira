from collections.abc import Iterator

from clients.llm.events import (
    CircuitBreakerEvent,
    CompleteEvent,
    ModelStepCompletedEvent,
    ToolCompletedEvent,
    ToolErrorEvent,
    ToolExecutingEvent,
)
from clients.llm.types import Result, ToolCall, ToolDefinition
from cns.services.orchestrator import ContinuumOrchestrator, TurnAccumulator


class SequentialTool:
    @staticmethod
    def is_call_parallel_safe(arguments: dict[str, object]) -> bool:
        return False


class FailingToolRepository:
    def __init__(self) -> None:
        self.tool_classes = {"invokeother_tool": SequentialTool}
        self.invocations: list[tuple[str, dict[str, object]]] = []
        self.definition = ToolDefinition(
            name="invokeother_tool",
            description="Load another tool.",
            input_schema={
                "type": "object",
                "properties": {"tool_name": {"type": "string"}},
                "required": ["tool_name"],
            },
        )

    def get_all_tool_definitions(self) -> list[ToolDefinition]:
        return [self.definition]

    def invoke_tool(self, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        self.invocations.append((tool_name, arguments))
        return {
            "success": False,
            "error": {
                "code": "TOOL_NOT_FOUND",
                "message": f"{arguments['tool_name']} not found",
            },
        }


class ToolCallingProvider:
    def __init__(self) -> None:
        self.requested_tools: list[tuple[str, ...]] = []
        self.results = iter(
            Result(
                tool_calls=(
                    ToolCall(
                        id=f"call-{index}",
                        tool_name="invokeother_tool",
                        input={"tool_name": "crm_clients_tool"},
                    ),
                ),
                stop_reason="tool_use",
            )
            for index in range(1, 4)
        )

    def stream_events(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> Iterator[CompleteEvent]:
        self.requested_tools.append(tuple(tool.name for tool in tools))
        yield CompleteEvent(response=next(self.results))


def test_circuit_breaker_remains_latched_after_final_no_tools_pass() -> None:
    repository = FailingToolRepository()
    provider = ToolCallingProvider()
    orchestrator = object.__new__(ContinuumOrchestrator)
    orchestrator.tool_repo = repository
    orchestrator.llm_provider = provider

    events = list(orchestrator._stream_model_tool_loop(
        base_messages=[{"role": "user", "content": "test"}],
        compose_messages=lambda: [{"role": "user", "content": "test"}],
        llm_kwargs={},
    ))

    assert repository.invocations == [
        ("invokeother_tool", {"tool_name": "crm_clients_tool"}),
        ("invokeother_tool", {"tool_name": "crm_clients_tool"}),
    ]
    assert provider.requested_tools == [
        ("invokeother_tool",),
        ("invokeother_tool",),
        (),
    ]
    assert len([event for event in events if isinstance(event, ModelStepCompletedEvent)]) == 2
    assert len([event for event in events if isinstance(event, ToolExecutingEvent)]) == 2
    assert len([event for event in events if isinstance(event, CircuitBreakerEvent)]) == 2
    refused_events = [
        event
        for event in events
        if isinstance(event, ToolErrorEvent) and event.tool_id == "call-3"
    ]
    assert len(refused_events) == 1
    assert "not executed" in refused_events[0].result

    final_event = events[-1]
    assert isinstance(final_event, CompleteEvent)
    assert final_event.response.stop_reason == "end_turn"
    assert "stopped the tool sequence" in final_event.response.text


def test_failed_tool_loader_does_not_trigger_auto_continuation(monkeypatch) -> None:
    monkeypatch.setattr("clients.valkey_client.get_valkey", lambda: object())
    orchestrator = object.__new__(ContinuumOrchestrator)
    accumulator = TurnAccumulator()

    orchestrator._consume_stream(
        iter((
            ToolExecutingEvent(
                tool_name="invokeother_tool",
                tool_id="call-1",
                arguments={"load": ["crm_clients_tool"]},
            ),
            ToolErrorEvent(
                tool_name="invokeother_tool",
                tool_id="call-1",
                error="crm_clients_tool not found",
                result='{"success": false}',
            ),
        )),
        accumulator,
        continuum_id="continuum-1",
        stream=False,
        stream_callback=None,
        show_thinking_stream=False,
    )

    assert accumulator.invoked_tool_loader is False


def test_successful_tool_loader_triggers_auto_continuation(monkeypatch) -> None:
    monkeypatch.setattr("clients.valkey_client.get_valkey", lambda: object())
    orchestrator = object.__new__(ContinuumOrchestrator)
    accumulator = TurnAccumulator()

    orchestrator._consume_stream(
        iter((
            ToolExecutingEvent(
                tool_name="invokeother_tool",
                tool_id="call-1",
                arguments={"load": ["weather_tool"]},
            ),
            ToolCompletedEvent(
                tool_name="invokeother_tool",
                tool_id="call-1",
                result='{"success": true, "loaded": ["weather_tool"]}',
            ),
        )),
        accumulator,
        continuum_id="continuum-1",
        stream=False,
        stream_callback=None,
        show_thinking_stream=False,
    )

    assert accumulator.invoked_tool_loader is True
