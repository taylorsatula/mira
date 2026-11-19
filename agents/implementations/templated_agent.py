"""TemplatedAgent — the runtime for user-authored agent definitions.

A row in `agent_definitions` plus one of four templates is a complete
agent: no per-agent code. The template supplies the behavioral contract and
the default route; the row supplies the name, the mission (`prompt_tophat`),
the tool allowlist, and optionally a route, an iteration cap, and a workdir
hint (advisory only — never enforced).

The same class serves interactive dispatch (via `agent_tool run_now`) and
scheduled dispatch from the heartbeat gate. In both cases the work item
context carries `subject` (the mission) and optionally `detail` and
`last_result`; everything else is read from the table at run time, so
`pause`/`retire` take effect on the next run without a restart.

Memory: `agent_memory` rows are keyed by `agent:<name>`, so a persistent
agent accumulates its own memory across runs and the existing
`agent_memory` tooling works unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from agents.base import SidebarAgent, load_agent_prompt

if TYPE_CHECKING:
    from agents.sidebar import WorkItem
    from tools.repo import ToolRepository

logger = logging.getLogger(__name__)

TEMPLATE_PROMPTS = {
    "investigator": "template_investigator.txt",
    "implementer": "template_implementer.txt",
    "watcher": "template_watcher.txt",
    "general": "template_general.txt",
}

TEMPLATE_DEFAULT_ROUTES = {
    "investigator": "batch",
    "implementer": "primary",
    "watcher": "fast",
    "general": "primary",
}


class TemplatedAgent(SidebarAgent):
    """Runtime for `agent_definitions` rows. One instance per definition."""

    inherit_base_prompt = True
    max_iterations = 12
    sanitize_untrusted_input = True

    def __init__(self, tool_repo: 'ToolRepository', definition: dict):
        self.definition = definition
        self.template = definition["template"]
        self.definition_name = definition["name"]
        # Per-instance identity: the base class reads these at run time.
        self.agent_id = f"agent:{self.definition_name}"
        self.model_config_name = (
            definition.get("model_route")
            or TEMPLATE_DEFAULT_ROUTES.get(self.template, "batch")
        )
        tools = list(definition.get("tools") or [])
        if "sidebar_tool" not in tools:
            tools = ["sidebar_tool", *tools]
        self.available_tools = tools
        if definition.get("max_iterations"):
            self.max_iterations = int(definition["max_iterations"])
        super().__init__(tool_repo)

    def get_agent_prompt(self, work_item: 'WorkItem') -> str:
        contract = load_agent_prompt(TEMPLATE_PROMPTS[self.template])
        tophat = (self.definition.get("prompt_tophat") or "").strip()
        workdir = self.definition.get("workdir")
        workdir_line = (
            f"\nWorkdir (advisory, not a sandbox): {workdir}\n" if workdir else ""
        )
        display = self.definition.get("display_name") or self.definition_name
        return (
            f"# Agent: {display}\n"
            f"{contract}\n\n"
            f"# Standing brief\n{tophat}\n"
            f"{workdir_line}"
        )

    def build_initial_message(self, work_item: 'WorkItem') -> str:
        ctx = work_item.context or {}
        subject = (ctx.get("subject") or ctx.get("mission") or "").strip()
        detail = (ctx.get("detail") or ctx.get("context") or "").strip()
        last = (ctx.get("last_result") or "").strip()
        lines = [f"Task: {subject}" if subject else "Task: (see standing brief)"]
        if detail:
            lines.append(f"\nDetails:\n{detail}")
        if last:
            lines.append(f"\nYour last recorded result, for reference:\n{last}")
        lines.append(
            "\nDo the work, then call sidebar_tool complete_task with the report."
        )
        return "\n".join(lines)

    def _result_agent_key(self) -> str:
        return self.agent_id

    def _result_subject(self, work_item: 'WorkItem'):
        ctx = work_item.context or {}
        subject = ctx.get("subject") or ctx.get("mission")
        if subject:
            return str(subject)[:500]
        return self.definition.get("display_name") or self.definition_name

    def on_completion(self, event_bus, work_item: 'WorkItem', status: str, summary: str):
        # Account the run against the definition (counters, auto-pause),
        # then let the base path publish and write the ledger row.
        try:
            from agents.definitions import record_run_outcome
            from utils.userdata_manager import get_user_data_manager
            from utils.user_context import get_current_user_id

            db = get_user_data_manager(get_current_user_id())
            outcome = record_run_outcome(
                db,
                self.definition["agent_id"],
                success=(status == "success"),
            )
            if outcome and outcome.get("auto_paused"):
                logger.warning(
                    "agent %s auto-paused after %s consecutive failures",
                    self.definition_name, outcome["consecutive_failures"],
                )
        except Exception:
            logger.exception("failed to record outcome for %s", self.definition_name)
        super().on_completion(event_bus, work_item, status, summary)


def definition_spec(definition: dict):
    """Build the spawn spec for a definition.

    The factory closes over the row so each spawn constructs a fresh
    TemplatedAgent bound to it; `agent_class_path` is documentation for
    the spawn seam (which prefers `factory` when present).
    """
    from agents.specs import SubagentSpec

    name = definition["name"]
    route = (
        definition.get("model_route")
        or TEMPLATE_DEFAULT_ROUTES.get(definition["template"], "batch")
    )
    return SubagentSpec(
        agent_type=f"agent:{name}",
        interface_name="agent",
        model_route=route,
        invocable=False,
        blocking_only=False,
        continuable=False,
        publishes_pending=True,
        dismissable=True,
        result_ttl_turns=None,
        error_ttl_turns=8,
        agent_class_path="agents.implementations.templated_agent:TemplatedAgent",
        factory=lambda tool_repo: TemplatedAgent(tool_repo, definition),
    )
