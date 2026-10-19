"""
HeartbeatAgent -- the brief gate: evaluates due briefs and reports.

Dispatched by the HeartbeatTrigger with every due armed brief for the user
batched into one work item, whose context holds that briefs list and nothing
else. Separation of duties: this agent is a gate, not a worker. It evaluates
each brief (bail_if first, then condition_check) and reports the outcome.

The rubric asks the gate to dispatch a mission that needs real work to a
specialized agent. This surface cannot: `available_tools` carries no dispatch
tool, and no work-item key names a target agent. So a `fired` report records
this agent's own say-so, not an observed dispatch -- read one as a claim to
verify, never as evidence that work started.

Runs on the 'fast' route: gate decisions are that route's job. Ephemeral by
construction -- nothing touches the main conversation; the trace lands in the
agent trace store and the outcome in the brief row. The one tool is
heartbeat_tool narrowed to its report-only schema via `tool_schema_overrides`,
so the gate cannot schedule or cancel its own wake schedule.

Failure accounting: `on_completion` advances the failure state machine for
every carried brief that received no valid report (the run failed, timed out,
or hit the iteration cap), via heartbeat_tool.apply_run_failure. The tell is
`state == 'armed'` with `last_wake_at` still equal to the carried
`dispatch_stamp`: the trigger stamps every brief it hands over, and every
report overwrites that stamp a moment later. `max_retries = 0` is the
matching half -- a dispatcher retry would re-run a stale due-window payload
whose briefs the per-brief accounting has already re-armed.
"""
import logging
from typing import Any, TYPE_CHECKING

from agents.base import SidebarAgent, load_agent_prompt
from tools.implementations.heartbeat_tool import HEARTBEAT_AGENT_SCHEMA

if TYPE_CHECKING:
    from agents.sidebar import WorkItem
    from tools.repo import ToolRepository

logger = logging.getLogger(__name__)


class HeartbeatAgent(SidebarAgent):
    agent_id = "heartbeat"
    model_config_name = "fast"
    available_tools = ["heartbeat_tool"]
    inherit_base_prompt = False
    max_iterations = 8
    max_retries = 0  # failures are accounted per-brief, not re-dispatched

    # heartbeat_tool is report-only (restricted schema from the tool's own
    # module). The heartbeat is a gate, not a worker: conditions requiring
    # tools or a specialized agent are dispatched, not performed inline.
    tool_schema_overrides = {
        'heartbeat_tool': HEARTBEAT_AGENT_SCHEMA,
    }

    # Brief payloads are authored by the user or by MIRA herself -- no
    # external/untrusted content rides the work item.
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
        # Runs on every terminal status: an under-reported run's stragglers
        # (still armed, last_wake_at == dispatch stamp) would otherwise stay
        # wedged behind a terminal dispatch-window record.
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
            # Must not mask the trinket publication super() performed; the
            # exception log names the wedge state unreported briefs are left in.
            logger.exception(
                "HeartbeatAgent: failure accounting raised; unreported "
                "briefs remain armed behind their dispatch window"
            )
