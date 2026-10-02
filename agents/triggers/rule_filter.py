"""Dispatch-time consumer of trigger_rules rows.

The trigger-rules API (cns/api/trigger_rules.py) writes per-user filter
rules; this module is their only reader at dispatch time. Each trigger
applies its own rules as the final step of discovery (see
agents/triggers/AGENTS.md): this is discovery shaping, not a dispatch
judgment — dedup, retry, and terminal-status decisions stay with the
dispatcher.

Semantics:
- No enabled rules for the trigger → every discovered item passes
  (filtering is opt-in; a fresh install dispatches exactly as before).
- With rules present, an item passes only when an enabled rule's pattern
  regex-matches (case-insensitive) the item's context value for the rule's
  field. Non-string context values never match.
- A matching rule's prompt, when set, is attached as
  context['rule_prompt']; SidebarAgent._build_system_prompt (agents/base.py)
  appends it to the system prompt after the agent rubric.
- The rule's scope column is trigger-specific metadata; no registered
  trigger has a scoped surface, so it is stored and returned but not
  matched here.
"""
import logging
import re

from agents.sidebar import WorkItem

logger = logging.getLogger(__name__)


def apply_trigger_rules(
    user_id: str,
    trigger_id: str,
    items: list[WorkItem],
) -> list[WorkItem]:
    """Filter work-items through the user's enabled rules for this trigger.

    Invalid stored state fails loud: a rule row whose pattern does not
    compile raises re.error out of check_for_new_items, where the
    dispatcher's per-user handler logs it — the API validates patterns at
    write time, so an uncompilable stored pattern means corrupted state,
    not user error.
    """
    if not items:
        return items

    from utils.userdata_manager import get_user_data_manager

    db = get_user_data_manager(user_id)
    rules = db.select(
        "trigger_rules",
        where="trigger_id = :trigger_id AND enabled = 1",
        params={"trigger_id": trigger_id},
    )
    if not rules:
        return items

    kept: list[WorkItem] = []
    for item in items:
        for rule in rules:
            value = item.context.get(rule["field"])
            if not isinstance(value, str):
                continue
            if re.search(rule["pattern"], value, re.IGNORECASE) is None:
                continue
            prompt = rule.get("prompt")
            if prompt:
                item.context["rule_prompt"] = prompt
            kept.append(item)
            break

    dropped = len(items) - len(kept)
    if dropped:
        logger.info(
            "rule_filter: dropped %d/%d items for trigger %s (user rules)",
            dropped, len(items), trigger_id,
        )
    return kept
