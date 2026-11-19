---
name: dispatching-agents
description: Dispatching persistent agents — creating them, binding them to heartbeat briefs, and reading what they report. Use when the user asks for recurring or standing work, when a brief should wake a named agent, or when a persistent agent's result needs to be found, recalled, or retired.
---

# Dispatching agents

Persistent agents are named, standing-brief agents that outlive a turn. They are not subagents: a subagent runs once and returns; a persistent agent runs on demand or on a schedule, keeps its own memory, and leaves every run in a durable ledger.

## When to create one

- The same job will recur — a watch, a sweep, a standing check. One-off work stays with the one-shot mechanisms (forage, whilethecatsaway, phoneafriend).
- The user asks for something "every morning", "whenever X changes", "keep an eye on".
- You notice yourself re-briefing the same work a second time.

Do not create one for a single task. Do not create one "just in case".

## Creating

`agent_tool create` with:

- `template`: `investigator` (read-only research, sources required), `implementer` (makes changes, verifies, reports diffs), `watcher` (reports deltas only, against its own memory), or `general` (the catch-all — full tool access, minimal contract). Pick the narrowest template that fits; `general` is for work that fits none of the three.
- `name`: a short slug (`payment_rail_watcher`). Lowercase, unique per user.
- `tophat`: the standing brief. Write it for a reader with no other context — every run starts fresh. State the job, the standard, what to report, and what to stay silent about.
- `tools`: the allowlist. For `general`, omit it to grant every grantable tool. `heartbeat_tool` and `agent_tool` are never grantable — an agent can never re-arm its own schedule or mint other agents.

## Running one

- On demand: `agent_tool run_now` (optionally with a `note` as this run's task). Bypasses the min-interval floor, never the already-running guard.
- On a schedule: create a heartbeat brief whose `agent_id` is the agent's name or `agt_…` id. When the gate reports the brief `fired`, the system dispatches the agent — you do not dispatch it yourself, and the gate never runs the work.
- `min_interval_sec` is the floor between dispatches (default 3600). A brief that fires more often than that skips the dispatch and says so on the brief.

## Reading results

Every run lands in the results ledger. The results feed shows the five newest per agent; older ones archive after 14 days and are never deleted. `agent_tool inspect` shows an agent's recent results; `agent_tool recall` lists or unarchives archived ones. Results are samples, not answers — verify what you will act on.

## Lifecycle

- `pause` silences an agent without forgetting it; `resume` re-enables it.
- `retire` ends it for good.
- Three consecutive failed runs auto-pause the agent. That is a signal to read the failure, not to resume blindly.
- Renaming is not supported; a new name is a new agent with a fresh memory namespace.

## Failure semantics

A run that fails, times out, or hits its iteration cap records a failed ledger row and counts toward auto-pause. A brief whose gate never reports leaves the brief armed with a failure count and never dispatches. A brief that fires but whose agent is paused, missing, or already running records the skip reason on the brief — read it with `heartbeat_tool list` before assuming the agent ran.
