from types import SimpleNamespace

from agents.base import SidebarAgent, _init_trace
from agents.sidebar import WorkItem
from clients.llm.types import Result, ToolCall
from utils.timezone_utils import utc_now


class DummyAgent(SidebarAgent):
    agent_id = "dummy"
    model_config_name = "batch"
    available_tools = ["memory_tool"]

    def get_agent_prompt(self, work_item: WorkItem) -> str:
        return "Test prompt"

    def build_initial_message(self, work_item: WorkItem) -> str:
        return "Test task"


class UnusedToolRepo:
    called = False

    def get_tool(self, tool_name: str) -> object:
        self.called = True
        raise AssertionError("schema-invalid tool calls must not execute tools")


class StaticLLM:
    def __init__(self, response: Result):
        self.response = response

    def generate_response(self, **kwargs: object) -> Result:
        return self.response

    def extract_tool_calls(self, response: Result) -> list[ToolCall]:
        return list(response.tool_calls)

    def extract_text_content(self, response: Result) -> str:
        return response.text


def test_sidebar_agent_invalid_tool_call_becomes_repair_feedback() -> None:
    tool_repo = UnusedToolRepo()
    agent = DummyAgent(tool_repo)
    work_item = WorkItem(item_id="item-1", interface_name="test", context={})
    agent._work_item = work_item
    agent._event_bus = object()
    agent._trace = _init_trace(agent.agent_id, work_item, utc_now())

    response = Result(
        tool_calls=(
            ToolCall(
                id="call_missing_operation",
                tool_name="memory_tool",
                input={},
                invalid_reason="missing required fields: ['operation']",
            ),
        ),
        stop_reason="tool_use",
    )
    messages = [{"role": "user", "content": "Start"}]

    completed = agent._run_iteration(
        1,
        messages,
        StaticLLM(response),
        [],
        SimpleNamespace(name="batch"),
        "system",
    )

    assert completed is False
    assert tool_repo.called is False
    assert not any(message.get("role") == "assistant" and message.get("tool_calls") for message in messages)
    assert messages[-2]["role"] == "user"
    assert "memory_tool tool call was rejected before execution" in messages[-2]["content"]
    assert messages[-1] == {"role": "user", "content": "Continue."}
