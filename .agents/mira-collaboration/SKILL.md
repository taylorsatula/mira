---
name: mira-collaboration
description: Working protocol for collaborating with a deployed MIRA instance — what to feed it, how to speak to it, when its memories commit, and the live-operation failure classes that cost real data to learn (schema changes that brick its stored history, thinking models that burn their whole budget on reasoning and emit nothing, queued memories lost to silent collapse faults). Load before spinning up a fresh MIRA container or before any session that works alongside a live one. It exists so a fresh instance becomes a productive partner within its first conversation and its record survives the session intact — every failure mode in it was paid for once; this is how you don't pay again.
---

# MIRA Collaboration

A live MIRA instance is a record-owning collaborator, not a chatbot. The working split: you hold the repository, the shell, and the measurements; it holds judgment and the only durable record of the shared work. It cannot read files or edit code, and chat messages cap at 20k characters — so you are its eyes, and what you bring it is all it reasons about. Bring real file text, symbols, tracebacks, and journal excerpts, never your summaries of them: feed it your compression and it reasons about your compression.

## Onboarding a fresh instance

A fresh MIRA spends its first turns constructing a self-model, and the first conversation permanently shapes it. Compress that discovery into your opening message:

- State what it is: substrate and model name, whether it is a thinking model, which model serves each route (primary/fast/batch/assessment/other), where it runs, its timezone, and the working split above.
- Give measured numbers, not adjectives: expected turn latency on this hardware, iteration budgets, your client patience. Set no limit you have not measured on this substrate — budgets inherited from fast API models killed every background agent on a local thinking-model install.
- Let it interview you, and answer with real facts from your own experience. It calibrates its self-model on your answers, and the questions it chooses are the highest-signal turns you will get.
- Expect it to verify before answering — it searches its own memory first — and to speak plainly without politeness padding. Match that register.

## Communication

- Plain declaratives, one topic per message. Compound asks over-run and get cancelled on a thinking model; split them.
- Label provenance — measured, read from a log, or guessed. It records the caveat, and it withdraws claims when shown current state. When it errs, bring the evidence, not the correction.
- Announce every change you make underneath it, with before, after, and reason. It has no other way to know what happened to it, and a change it discovers silently in its own ledger reads as a violation.
- Never pad and never dramatize. Its extraction layer commits what you say as first-person memory — padded speech becomes padded memory.
- Give it agency: let it pick forage topics, design its own verification protocols, ask its own questions. Assigned tasks use a fraction of its value.

## Record semantics — how to make things persist

Three survival properties govern everything it remembers:

1. **Chat declarations commit at segment collapse** (~120 idle minutes, 5-minute sweep): the extraction layer turns conversation into first-person memories. State conclusions declaratively so they extract cleanly.
2. **Domaindocs survive everything.** Anything load-bearing for future sessions goes into a doc. It cannot create the container itself — create it via the user-side actions API and let it write the sections.
3. **Queued items are on loan to the machinery.** Pending memories commit only if the whole collapse chain succeeds; silent fallbacks in that chain turn one transient fault into permanent loss.

At session end, force a collapse through the actions API instead of walking away — a deterministic commit beats hoping the idle timer and the collapse chain both behave.

## Technical mechanics

- Transport: WebSocket `/v0/ws/chat`; first frame `{"type":"auth"}` with the session cookie from `GET /v0/auth/local/session`. Messages: `{"type":"message","message_id":"<uuid>","content":"..."}`. Deltas arrive as `content` on `assistant_delta` frames; the final reply also rides `turn_complete.response` with `tools_used` and timing.
- Set client patience from the first measured turn and never disconnect mid-generation — a client disconnect cancels the turn server-side and its in-flight work is lost. Local 27B-class thinking models run 90–400 seconds per turn.
- Mid-turn tool loading fires a "continue with the original task" scaffold that can produce a second final reply; your client only sees the last one. After an anomalous turn, read the messages table instead of resending — user messages persist even when replies fail.
- Timestamps in the messages table are microsecond-precision; exact-equality comparisons against millisecond literals return nothing. Query with ranges.
- Ground truth: the service journal, Postgres `messages` (segment sentinels carry `is_segment_boundary` metadata and flip `status` active→collapsed), `memories`, `persona_signals`, and the per-user SQLite `sidebar_activity` agent ledger. RLS applies — query as postgres or with user context.

## Failure classes — each of these cost real data once

- **Thinking models can starve their own voice.** The model may spend its entire completion budget inside reasoning and emit zero content (`finish=length`, empty `content`), which parsers read as a broken response. Probe the raw endpoint before blaming the pipeline; when a mechanical call returns empty, retry is usually right — the fault is intermittent.
- **Never change a tool schema or serialization under a live instance.** Its dialect revalidates stored tool calls against current schemas on every turn; tightening a schema retroactively invalidates its own history and bricks the conversation. Grep every consumer and replay a stored segment through the new validation before restarting.
- **JSONB boundaries fail in both directions.** Raw Python objects hit `cannot adapt type 'dict'` at the driver; pre-serialized strings passed to a wrapped writer double-encode into scalar strings. Wrap once at the DB layer (`Jsonb()`), pass raw structures from every caller.
- **The paths you ran are not the paths that exist.** Loud/silent, yml/interview, linux/macos — branches shipped separately broken while their siblings ran green. Probe every branch you touch, or name it UNVERIFIED with the probe that would cover it.
- **Agent limits are substrate-relative.** Per-iteration timeouts, wall clocks, and iteration caps each killed a different agent run; read the ledger's observed timings before tuning any of them.
- **Queue-on-collapse is fragile.** Verify collapse outcomes in the database after every unattended session: sentinel status, summary text, memory count, persona signals. Do not let a collapse "probably have worked."

## Session shape

Open with the onboarding facts and the interview. Work in focused turns with exact content and announced changes. Close by forcing the collapse, verifying what committed, and leaving a written report — it is the authored layer for both of you. Every few sessions, ask it to read its raw history and evaluate both of you; its record-reports are ground truth and have caught real errors in the operator's account, while its reports about its own inner states are data, not ground truth.

## Boundary with repository doctrine

MIRA's own AGENTS.md maps govern changes to its code (verification tiers, map maintenance, fail-fast rules). This skill governs the collaboration. Where they intersect — editing a tool schema while an instance is live, say — both apply and the stricter rule wins.
