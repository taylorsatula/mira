"""Ordered provider-step persistence and durable terminal behavior."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from clients.llm.events import GenerationCancelled, ToolCompletedEvent, ToolErrorEvent, ToolExecutingEvent
from clients.llm.types import Result, ToolCall
from cns.core.continuum import Continuum
from cns.infrastructure.continuum_pool import UnitOfWork
from cns.services.orchestrator import (
    AssistantStep,
    ContinuumOrchestrator,
    ToolInteraction,
    TurnAccumulator,
)
from cns.services.tool_loop import CircuitBreaker, ToolLoopExecutor
from utils.user_context import set_cancel_event, set_cancel_reason


def test_text_tool_text_persists_each_provider_step_once_in_stream_order() -> None:
    first_id = uuid4()
    final_id = uuid4()
    tool_call = ToolCall(id="call-1", tool_name="viewcard_tool", input={"card_type": "notification"})
    accumulator = TurnAccumulator(
        assistant_steps=[
            AssistantStep(
                entry_id=first_id,
                text="I’ll check that.",
                result=Result(
                    text="I’ll check that.",
                    tool_calls=(tool_call,),
                    stop_reason="tool_use",
                ),
            ),
            AssistantStep(
                entry_id=final_id,
                text="The appointment is confirmed.",
                result=Result(text="The appointment is confirmed."),
            ),
        ],
        tool_interactions=[ToolInteraction(
            tool_name="viewcard_tool",
            tool_id="call-1",
            arguments={"card_type": "notification"},
            result='{"success":true}',
            completed=True,
        )],
    )
    turn_id = uuid4()
    segment_id = str(uuid4())
    base_time = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)

    orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)
    messages = orchestrator._build_turn_messages(
        accumulator,
        {"tools_used": ["viewcard_tool"]},
        turn_id=turn_id,
        segment_id=segment_id,
        base_time=base_time,
    )

    assert [message.role for message in messages] == ["assistant", "tool", "assistant"]
    assert messages[0].id == first_id
    assert messages[2].id == final_id
    assert messages[0].content == [
        {"type": "text", "text": "I’ll check that."},
        {
            "type": "tool_call",
            "id": "call-1",
            "name": "viewcard_tool",
            "input": {"card_type": "notification"},
        },
    ]
    assert messages[2].content == "The appointment is confirmed."
    assert all(message.metadata["turn_id"] == str(turn_id) for message in messages)
    assert all(message.metadata["segment_id"] == segment_id for message in messages)
    assert [message.created_at for message in messages] == sorted(
        message.created_at for message in messages
    )
    assert len({message.created_at for message in messages}) == len(messages)


def test_invalid_provider_tool_call_is_not_persisted_as_tool_pair() -> None:
    invalid_call = ToolCall(
        id="call-invalid",
        tool_name="jobs_tool",
        input={},
        invalid_reason="missing required fields: ['operation']",
    )
    final_id = uuid4()
    accumulator = TurnAccumulator(
        assistant_steps=[
            AssistantStep(
                entry_id=uuid4(),
                result=Result(
                    text="",
                    tool_calls=(invalid_call,),
                    stop_reason="tool_use",
                ),
            ),
            AssistantStep(
                entry_id=final_id,
                text="I could not run that CRM lookup because the tool call was malformed.",
                result=Result(text="I could not run that CRM lookup because the tool call was malformed."),
            ),
        ],
        tool_interactions=[ToolInteraction(
            tool_name="jobs_tool",
            tool_id="call-invalid",
            arguments={},
            result="Error: missing required fields: ['operation']",
            completed=True,
            is_error=True,
        )],
    )
    orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)

    messages = orchestrator._build_turn_messages(
        accumulator,
        {"tools_used": ["jobs_tool"]},
        turn_id=uuid4(),
        segment_id=str(uuid4()),
        base_time=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
    )

    assert len(messages) == 1
    assert messages[0].id == final_id
    assert messages[0].role == "assistant"
    assert messages[0].content == "I could not run that CRM lookup because the tool call was malformed."


def test_halt_marks_only_the_current_partial_assistant_step() -> None:
    partial_id = uuid4()
    accumulator = TurnAccumulator(assistant_steps=[AssistantStep(
        entry_id=partial_id,
        text="Partial answer",
        partial=True,
    )])
    orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)

    messages = orchestrator._build_turn_messages(
        accumulator,
        {},
        turn_id=uuid4(),
        segment_id=str(uuid4()),
        base_time=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
        stop_reason="halt",
    )

    assert len(messages) == 1
    assert messages[0].id == partial_id
    assert messages[0].metadata["partial_response"] is True
    assert messages[0].metadata["stop_reason"] == "halt"


def test_unit_of_work_callbacks_run_only_after_database_and_cache_commit() -> None:
    operations: list[str] = []
    continuum = Continuum.create_new("user-1")
    message, _ = continuum.add_user_message("hello")
    pool = SimpleNamespace(
        repository=SimpleNamespace(
            save_messages_batch=lambda *_args: operations.append("database"),
            update_continuum_metadata=lambda *_args: operations.append("metadata"),
        ),
        valkey_cache=SimpleNamespace(
            set_continuum=lambda *_args: operations.append("cache"),
        ),
    )
    unit_of_work = UnitOfWork(continuum, pool)
    unit_of_work.add_messages(message)
    unit_of_work.mark_metadata_updated()
    unit_of_work.add_post_commit_callback(lambda: operations.append("event"))

    unit_of_work.commit()

    assert operations == ["database", "cache", "metadata", "event"]


def test_internal_continuation_prompt_is_cache_only_and_narrowly_removable() -> None:
    continuum = Continuum.create_new("user-1")
    durable, _ = continuum.add_user_message("accepted user message")
    scaffold_id = uuid4()
    continuum.add_user_message(
        "<system-scaffold>continue</system-scaffold>",
        message_id=scaffold_id,
        metadata={"transient_system_scaffold": True},
    )
    assistant, _ = continuum.add_assistant_message("continued response")

    assert continuum.discard_transient_user_message(scaffold_id) is True
    assert [message.id for message in continuum.messages] == [durable.id, assistant.id]
    assert continuum.discard_transient_user_message(scaffold_id) is False
    with pytest.raises(ValueError, match="Only a transient"):
        continuum.discard_transient_user_message(durable.id)


def test_processing_failure_commits_the_already_accepted_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import clients.vault_client as vault_client

    monkeypatch.setattr(vault_client, "get_database_url", lambda _name: "postgresql://test")
    monkeypatch.setattr(vault_client, "get_service_config", lambda field: {
        "email_gateway_url": "https://mail.example.test",
        "email_gateway_api_key": "test-key",
        "email_gateway_hmac_secret": "test-secret",
        "valkey_url": "valkey://localhost:6379",
        "app_url": "https://mira.example.test",
        "crm_base_url": "https://crm.example.test",
        "crm_lifecycle_service_secret": "test-lifecycle-secret",
    }[field])
    websocket_module = importlib.import_module("cns.api.websocket_chat")
    operations: list[str] = []

    class FailingOrchestrator:
        def process_message(self, *_args: object, unit_of_work: object, **_kwargs: object) -> None:
            unit_of_work.pending_messages.append("accepted-user-message")
            raise RuntimeError("provider failed")

    class FakeUnitOfWork:
        def __init__(self) -> None:
            self.pending_messages: list[str] = []

        def commit(self) -> None:
            operations.extend(self.pending_messages)

    handler = websocket_module.WebSocketChatHandler.__new__(websocket_module.WebSocketChatHandler)
    handler.orchestrator = FailingOrchestrator()
    handler.continuum_pool = SimpleNamespace(begin_work=lambda _continuum: FakeUnitOfWork())

    with pytest.raises(RuntimeError, match="provider failed"):
        handler._process_with_orchestrator(
            Continuum.create_new("user-1"),
            "hello",
            None,
            None,
            lambda _frame: None,
            1,
            UUID("00000000-0000-0000-0000-000000000001"),
            UUID("00000000-0000-0000-0000-000000000002"),
        )

    assert operations == ["accepted-user-message"]


def test_halt_drains_every_parallel_tool_that_already_started() -> None:
    invoked: list[str] = []

    class ToolRepository:
        tool_classes: dict[str, object] = {}

        @staticmethod
        def invoke_tool(tool_name: str, _arguments: dict[str, object]) -> dict[str, str]:
            invoked.append(tool_name)
            return {"completed": tool_name}

    cancel_event = threading.Event()
    set_cancel_event(cancel_event)
    executor = ToolLoopExecutor(ToolRepository())
    stream = executor.execute_tools(
        [
            ToolCall(id="parallel-1", tool_name="first_tool", input={}),
            ToolCall(id="parallel-2", tool_name="second_tool", input={}),
        ],
        CircuitBreaker(),
    )

    assert isinstance(next(stream), ToolExecutingEvent)
    assert isinstance(next(stream), ToolExecutingEvent)
    set_cancel_reason("halt")
    cancel_event.set()
    terminal_events = [next(stream), next(stream)]

    assert all(isinstance(event, ToolCompletedEvent) for event in terminal_events)
    assert {event.tool_id for event in terminal_events} == {"parallel-1", "parallel-2"}
    assert set(invoked) == {"first_tool", "second_tool"}
    with pytest.raises(GenerationCancelled):
        next(stream)


def test_halt_does_not_start_a_later_parallel_tool() -> None:
    invoked: list[str] = []

    class ToolRepository:
        tool_classes: dict[str, object] = {}

        @staticmethod
        def invoke_tool(tool_name: str, _arguments: dict[str, object]) -> dict[str, str]:
            invoked.append(tool_name)
            return {"completed": tool_name}

    cancel_event = threading.Event()
    set_cancel_event(cancel_event)
    executor = ToolLoopExecutor(ToolRepository())
    stream = executor.execute_tools(
        [
            ToolCall(id="parallel-1", tool_name="first_tool", input={}),
            ToolCall(id="parallel-2", tool_name="second_tool", input={}),
        ],
        CircuitBreaker(),
    )

    first_event = next(stream)
    assert isinstance(first_event, ToolExecutingEvent)
    set_cancel_reason("halt")
    cancel_event.set()
    terminal_event = next(stream)

    assert isinstance(terminal_event, ToolCompletedEvent)
    assert terminal_event.tool_id == "parallel-1"
    assert invoked == ["first_tool"]
    with pytest.raises(GenerationCancelled):
        next(stream)


def test_invalid_provider_tool_arguments_emit_tool_error_without_invocation(caplog: pytest.LogCaptureFixture) -> None:
    invoked: list[str] = []

    class ToolRepository:
        tool_classes: dict[str, object] = {}

        @staticmethod
        def invoke_tool(tool_name: str, _arguments: dict[str, object]) -> dict[str, str]:
            invoked.append(tool_name)
            return {"completed": tool_name}

        @staticmethod
        def get_tool_definition(tool_name: str):
            return SimpleNamespace(input_schema={"properties": {"operation": {"type": "string"}}})

    set_cancel_event(threading.Event())
    caplog.set_level(logging.WARNING, logger="cns.services.tool_loop")
    executor = ToolLoopExecutor(ToolRepository())
    stream = executor.execute_tools(
        [
            ToolCall(
                id="call-invalid",
                tool_name="catalog_tool",
                input={},
                invalid_reason="missing required fields: ['operation']",
            ),
        ],
        CircuitBreaker(),
    )

    executing = next(stream)
    error = next(stream)

    assert isinstance(executing, ToolExecutingEvent)
    assert isinstance(error, ToolErrorEvent)
    assert error.tool_id == "call-invalid"
    assert "missing required fields: ['operation']" in error.result
    assert "CORRECT PARAMETERS" in error.result
    assert invoked == []
    assert "Provider returned invalid tool call for catalog_tool" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
