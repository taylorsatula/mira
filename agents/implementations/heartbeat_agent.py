"""
HeartbeatAgent -- executes due briefs (wake-ups with missions).

Dispatched by the HeartbeatTrigger with every due armed brief for the
user batched into one work item. The agent is a focused get-in-and-get-out
worker: for each brief, evaluate bail_if first (true -> report 'bailed',
skip the mission), then the condition_check with tools (true -> execute
the mission, report 'fired'; false -> report 'false'), then complete.

Runs on the 'fast' route: check-report-act loops are that route's job.
Ephemeral by construction -- nothing touches the main conversation; the
trace lands in the agent trace store and the outcome in the brief row.

Failure accounting: on_completion advances the failure state machine for
every carried brief that received no valid report (the run failed, timed
out, or hit the iteration cap), via heartbeat_tool.apply_run_failure.
"""
import logging
from typing import Any, TYPE_CHECKING

from agents.base import SidebarAgent, load_agent_prompt
from tools.implementations.heartbeat_tool import HEARTBEAT_AGENT_SCHEMA
from tools.implementations.memory_tool import HEARTBEAT_MEMORY_SCHEMA

if TYPE_CHECKING:
    from agents.sidebar import WorkItem
    from tools.repo import ToolRepository

logger = logging.getLogger(__name__)


class HeartbeatAgent(SidebarAgent):
    agent_id = "heartbeat"
    model_config_name = "fast"
    available_tools = ["heartbeat_tool", "web_tool", "memory_tool"]
    inherit_base_prompt = False
    max_iterations = 8
    max_retries = 0  # failures are accounted per-brief, not re-dispatched

    # The agent's heartbeat_tool surface is report-only: the restricted
    # schema exported from the tool's own module (tool owns the contract).
    tool_schema_overrides = {
        'heartbeat_tool': HEARTBEAT_AGENT_SCHEMA,
        'memory_tool': HEARTBEAT_MEMORY_SCHEMA,
    }

    # Brief payloads are authored by the user or by MIRA herself -- no
    # external/untrusted content rides the work item, so the injection gate
    # is opted out at class level (mirrors MemoryCuratorAgent's reasoning).
    sanitize_untrusted_input = False

    def __init__(self, tool_repo: 'ToolRepository'):
        super().__init__(tool_repo)

    def get_agent_prompt(self, work_item: 'WorkItem') -> str:
        return load_agent_prompt("heartbeat_agent_system.txt")

    def build_initial_message(self, work_item: 'WorkItem') -> str:
        briefs = work_item.context.get('briefs', [])
        lines = ["Briefs due for evaluation:", ""]
        for b in briefs:
            lines.append(f"## Brief {b.get('id')} ({b.get('type')})")
            lines.append(f"- mission: {b.get('mission')}")
            lines.append(f"- condition_check: {b.get('condition_check')}")
            if b.get('bail_if'):
                lines.append(f"- bail_if: {b.get('bail_if')}")
            if b.get('last_result'):
                lines.append(f"- last_result: {b.get('last_result')}")
            lines.append("")
        lines.append(
            "Evaluate every brief per your rubric and report each one via "
            "heartbeat_tool (operation 'report') before completing."
        )
        return "\n".join(lines)

    def _build_completion_context(
        self,
        status: str,
        summary: str,
        work_item: 'WorkItem',
    ) -> dict[str, Any]:
        context: dict = {
            'task_id': work_item.item_id,
            'status': status,
        }
        if status == 'success':
            context['result'] = summary
        elif status == 'skipped':
            context['skipped_reason'] = summary
        else:  # failed | timeout | rejected
            context['error'] = summary
            context['error_type'] = 'AgentFailure'
        return context

    def on_completion(self, event_bus, work_item: 'WorkItem', status: str, summary: str) -> None:
        """Publish the trinket, then account for briefs that never received a
        valid report because the run died (failed, timed out, capped).

        The trigger stamps every dispatched brief's last_wake_at with the
        dispatch time and carries that stamp in the work item context; every
        report outcome overwrites last_wake_at a moment later. So an armed
        brief whose last_wake_at still equals the dispatch stamp is exactly
        one that was carried but never reported.
        """
        super().on_completion(event_bus, work_item, status, summary)
        # No success early-return: a run that completed but under-reported (the
        # model called complete_task before reporting every carried brief)
        # still needs its stragglers accounted for, or they stay armed with a
        # terminal dispatch-window record forever. The stamp comparison below
        # excludes every properly reported brief, so accounting on success
        # only ever touches genuine stragglers.
        try:
            from tools.implementations.heartbeat_tool import apply_run_failure
            from utils.userdata_manager import get_user_data_manager
            from utils.user_context import get_current_user_id

            db = get_user_data_manager(str(get_current_user_id()))
            unreported = []
            for b in work_item.context.get('briefs', []):
                bid = b.get('id')
                stamp = b.get('dispatch_stamp')
                if not bid or not stamp:
                    continue
                row = db.fetchone(
                    "SELECT last_wake_at, state FROM briefs WHERE id = :id",
                    {"id": bid},
                )
                if row is None:
                    continue
                if row.get('state') == 'armed' and row.get('last_wake_at') == stamp:
                    unreported.append(bid)
            if unreported:
                apply_run_failure(unreported, f"Agent run {status}: {summary[:300]}")
        except Exception:
            # Failure accounting is a side effect; it must never mask the
            # terminal trinket publication that super() already performed.
            # Consequence of this swallow: any briefs still carrying the
            # dispatch stamp stay armed with a past next_wake_at and a
            # terminal dispatch-window record (max_retries=0), so they stay
            # wedged until a state change un-blocks the window — surfaced
            # here rather than silently degraded.
            logger.exception(
                "HeartbeatAgent: failure accounting raised; unreported "
                "briefs remain armed behind their dispatch window"
            )
