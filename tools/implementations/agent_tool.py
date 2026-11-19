"""
agent_tool — create and run persistent agents.

One surface for the user-authored agent lifecycle: create an agent from a
template (investigator / implementer / watcher) with a standing brief, list
and inspect what exists, pause/resume/retire, recall archived results, and
trigger an immediate run. Scheduled runs are not this tool's job — a brief
with `agent_id` (heartbeat_tool) is what wakes an agent on a cadence.

Templates:
  investigator — read-only research; reports with sources and a verdict.
  implementer  — makes changes; verifies before claiming done.
  watcher      — recurring check; reports deltas against its own memory.

Agents never get heartbeat_tool or agent_tool: they cannot re-arm their own
schedule and cannot mint further agents.
"""
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry

if TYPE_CHECKING:
    from tools.repo import ToolRepository
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)


class AgentToolConfig(BaseModel):
    """Configuration for agent_tool."""

    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled",
    )


registry.register("agent_tool", AgentToolConfig)

_TEMPLATES = ("investigator", "implementer", "watcher")


class AgentTool(Tool):
    """Create, inspect, and run persistent user-authored agents."""

    name = "agent_tool"

    # Read-only operations (list, inspect, recall-listing) are parallel-safe;
    # everything that mutates definitions, the ledger, or spawns runs is not.
    _SEQUENTIAL_OPERATIONS = frozenset(
        {"create", "pause", "resume", "retire", "recall", "run_now"}
    )

    @classmethod
    def is_call_parallel_safe(cls, tool_input: Dict[str, Any]) -> bool:
        return tool_input.get("operation") not in cls._SEQUENTIAL_OPERATIONS

    simple_description = """
    Create and manage persistent agents: named, standing-brief agents that
    run on demand (run_now) or on a heartbeat schedule, keep their own
    memory, and leave results in the results ledger.
    """

    tool_schema = {
        "name": "agent_tool",
        "description": (
            "Create and manage persistent agents — named agents with a standing "
            "brief that outlive a single turn.\n\n"
            "  create (name, template, tophat, tools): define an agent. "
            "template is one of investigator (read-only research, reports with "
            "sources), implementer (makes changes, verifies, reports diffs), "
            "watcher (recurring check, reports only deltas). tophat is the "
            "standing brief — everything the agent needs to know, since it "
            "starts fresh each run. tools is the allowlist of tool names the "
            "agent may call; heartbeat_tool and agent_tool are never grantable. "
            "Optional: display_name, model_route (primary|fast|batch|assessment|"
            "other), max_iterations (1-50), workdir (advisory), min_interval_sec "
            "(dispatch floor, default 3600).\n"
            "  list: every agent with state, run count, last run, failures.\n"
            "  inspect (agent): one agent's definition, recent results, and "
            "counts. inspect (run_id): one result in full.\n"
            "  pause / resume / retire (agent): stop, re-enable, or permanently "
            "retire an agent. Retired agents never run again.\n"
            "  recall (run_id): unarchive an archived result so it shows again. "
            "recall (agent): list that agent's archived results.\n"
            "  run_now (agent, note?): dispatch the agent immediately, bypassing "
            "its min_interval (an explicit request), but not the already-running "
            "guard. note becomes the task for this run.\n\n"
            "To run an agent on a schedule, create a heartbeat brief with "
            "heartbeat_tool and set its agent_id to this agent's name or id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "create", "list", "inspect",
                        "pause", "resume", "retire",
                        "recall", "run_now",
                    ],
                    "description": "The action to perform.",
                },
                "name": {
                    "type": "string",
                    "description": "create only: the agent's slug (lowercase letters, digits, underscores; starts with a letter). Unique per user.",
                },
                "template": {
                    "type": "string",
                    "enum": list(_TEMPLATES),
                    "description": "create only: which behavioral template the agent runs.",
                },
                "tophat": {
                    "type": "string",
                    "description": "create only: the standing brief — the agent's whole job, written for a reader with no other context. Max 8000 chars.",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "create only: tool names the agent may call. heartbeat_tool and agent_tool are never grantable.",
                },
                "display_name": {
                    "type": "string",
                    "description": "create only: human-facing name. Defaults to the slug.",
                },
                "model_route": {
                    "type": "string",
                    "enum": ["primary", "fast", "batch", "assessment", "other"],
                    "description": "create only: LLM route. Defaults per template (investigator=batch, implementer=primary, watcher=fast).",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "create only: iteration cap, 1-50. Default 12.",
                },
                "workdir": {
                    "type": "string",
                    "description": "create only: advisory working directory hint. Never enforced.",
                },
                "min_interval_sec": {
                    "type": "integer",
                    "description": "create only: minimum seconds between dispatches (60-604800). Default 3600.",
                },
                "agent": {
                    "type": "string",
                    "description": "inspect/pause/resume/retire/recall/run_now: the agent's name or agent_id.",
                },
                "run_id": {
                    "type": "string",
                    "description": "inspect/recall: a specific result's run_id (from a <result> block or a previous recall listing).",
                },
                "note": {
                    "type": "string",
                    "description": "run_now only: the task for this run. Defaults to the standing brief.",
                },
            },
            "required": ["operation"],
        },
    }

    def __init__(self, tool_repo: 'ToolRepository', working_memory: 'WorkingMemory'):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.tool_repo = tool_repo
        self.working_memory = working_memory
        self.event_bus = working_memory.event_bus if working_memory else None

    def run(self, operation: str = None, **kwargs) -> Dict[str, Any]:
        handlers = {
            "create": self._create,
            "list": self._list,
            "inspect": self._inspect,
            "pause": self._set_state,
            "resume": self._set_state,
            "retire": self._set_state,
            "recall": self._recall,
            "run_now": self._run_now,
        }
        handler = handlers.get(operation or "")
        if handler is None:
            raise ValueError(
                f"Unknown operation: {operation!r}. Valid: {', '.join(sorted(handlers))}"
            )
        if operation in ("pause", "resume", "retire"):
            return handler(kwargs, {"pause": "paused", "resume": "active", "retire": "retired"}[operation])
        return handler(kwargs)

    # -- helpers -----------------------------------------------------------

    def _tool_names(self) -> set[str]:
        return set(registry.list_registered())

    @staticmethod
    def _public_result(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "run_id": row.get("run_id"),
            "agent": row.get("agent"),
            "status": row.get("status"),
            "subject": row.get("subject"),
            "summary": row.get("encrypted__summary"),
            "payload": row.get("encrypted__payload"),
            "created_at": row.get("created_at"),
            "seen_at": row.get("seen_at"),
            "archived_at": row.get("archived_at"),
        }

    @staticmethod
    def _public(defn: Dict[str, Any]) -> Dict[str, Any]:
        from agents.implementations.templated_agent import TEMPLATE_DEFAULT_ROUTES

        return {
            "agent_id": defn["agent_id"],
            "name": defn["name"],
            "display_name": defn.get("display_name"),
            "template": defn["template"],
            "state": defn["state"],
            "tools": defn.get("tools"),
            "model_route": defn.get("model_route")
            or TEMPLATE_DEFAULT_ROUTES.get(defn["template"]),
            "max_iterations": defn.get("max_iterations"),
            "workdir": defn.get("workdir"),
            "min_interval_sec": defn.get("min_interval_sec"),
            "run_count": defn.get("run_count"),
            "consecutive_failures": defn.get("consecutive_failures"),
            "last_run_at": defn.get("last_run_at"),
            "created_at": defn.get("created_at"),
        }

    def _resolve(self, key: Optional[str]) -> Dict[str, Any]:
        from agents.definitions import get_definition

        if not key or not str(key).strip():
            raise ValueError("agent is required (name or agent_id)")
        defn = get_definition(self.db, str(key).strip())
        if defn is None:
            raise ValueError(f"no agent named {key!r}")
        return defn

    # -- operations --------------------------------------------------------

    def _create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from agents.definitions import create_definition

        defn = create_definition(
            self.db,
            name=params.get("name"),
            template=params.get("template"),
            tophat=params.get("tophat") or "",
            tools=params.get("tools") or [],
            tool_names=self._tool_names(),
            model_route=params.get("model_route"),
            max_iterations=params.get("max_iterations"),
            workdir=params.get("workdir"),
            min_interval_sec=params.get("min_interval_sec", 3600),
            display_name=params.get("display_name"),
        )
        return {
            "success": True,
            "agent": self._public(defn),
            "message": (
                f"Agent {defn['name']!r} created. It runs on demand via "
                f"agent_tool run_now, or on a schedule via a heartbeat brief "
                f"with agent_id={defn['agent_id']!r} (or the name)."
            ),
        }

    def _list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from agents.definitions import list_definitions

        defs = list_definitions(self.db)
        return {
            "success": True,
            "agents": [self._public(d) for d in defs],
            "count": len(defs),
        }

    def _inspect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from agents import results_store

        run_id = params.get("run_id")
        if run_id:
            row = results_store.recall(self.db, str(run_id))
            if row is None:
                raise ValueError(f"no result with run_id {run_id!r}")
            return {"success": True, "result": self._public_result(row)}

        defn = self._resolve(params.get("agent"))
        agent_key = f"agent:{defn['name']}"
        recent = [self._public_result(r) for r in results_store.recent_for_agent(self.db, agent_key, 5)]
        counts = results_store.counts_by_agent(self.db).get(agent_key, {"unarchived": 0, "archived": 0})
        return {
            "success": True,
            "agent": self._public(defn),
            "tophat": defn.get("prompt_tophat"),
            "recent_results": recent,
            "results": counts,
        }

    def _set_state(self, params: Dict[str, Any], state: str) -> Dict[str, Any]:
        from agents.definitions import set_state

        defn = self._resolve(params.get("agent"))
        updated = set_state(self.db, defn["agent_id"], state)
        return {
            "success": True,
            "agent": self._public(updated),
            "message": f"Agent {updated['name']!r} is now {state}.",
        }

    def _recall(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from agents import results_store

        run_id = params.get("run_id")
        if run_id:
            row = results_store.recall(self.db, str(run_id))
            if row is None:
                raise ValueError(f"no result with run_id {run_id!r}")
            results_store.unarchive(self.db, str(run_id))
            row = results_store.recall(self.db, str(run_id))
            return {"success": True, "result": self._public_result(row)}

        defn = self._resolve(params.get("agent"))
        archived = results_store.list_archived(self.db, agent=f"agent:{defn['name']}", limit=20)
        return {
            "success": True,
            "agent": defn["name"],
            "archived": [self._public_result(r) for r in archived],
            "count": len(archived),
            "message": "Pass a run_id to recall to unarchive one of these.",
        }

    def _run_now(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from agents.dispatch import dispatch_brief_agent

        defn = self._resolve(params.get("agent"))
        if defn["state"] != "active":
            return {
                "success": False,
                "message": (
                    f"Agent {defn['name']!r} is {defn['state']}; "
                    f"resume it before running it."
                ),
            }
        note = (params.get("note") or "").strip()
        subject = note or f"Manual run of {defn['name']}"
        outcome = dispatch_brief_agent(
            db=self.db,
            event_bus=self.event_bus,
            tool_repo=self.tool_repo,
            brief={"id": None, "agent_id": defn["agent_id"], "dispatch_stamp": None},
            mission=subject,
            condition_check=None,
            last_result=None,
            bypass_min_interval=True,
            trigger="manual",
        )
        if not outcome.get("dispatched"):
            return {"success": False, "message": f"Not dispatched: {outcome.get('reason')}", **outcome}
        return {
            "success": True,
            "message": f"Agent {defn['name']!r} dispatched. Its result will land in the results feed.",
            **outcome,
        }

