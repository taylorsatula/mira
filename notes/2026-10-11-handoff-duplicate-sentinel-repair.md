# HANDOFF — 2026-10-11 — mira duplicate-sentinel lockout: repair + root fix mid-probes

Read first: `notes/2026-10-10-turn-setup-failed-duplicate-sentinel-lockout.md` (incident
record), `notes/2026-10-10-turn-processing-failed-429.md` (same-day gateway context).

```yaml
objective_and_constraints:
  objective: >
    Repair the 2026-10-10 TURN_SETUP_FAILED chat lockout (duplicate live segment
    sentinel) on the deployed local instance, append the approved synthetic
    rewind pair to the real segment so the user can type "continue" and resume
    MIRA's interrupted heartbeat-feature work, AND land the approved 4-part
    root fix in the mira repo (~/Programming/Scratch/mira, branch main).
    Report when all work is complete.
  goal_deltas:
    - "User discarded heartbeat messages: delete orphan rows entirely, no reassignment."
    - "Scrollback unavailable (no tmux/screen, no TUI transcript persistence, bash_tool
       logs only guardrail_bypass.log) → rewind uses the honest synthetic pair, not
       reconstructed tool rows. User accepted by saying 'begin the work'."
    - "User confirmed scope is small: 4 sites, ~10 behavioral lines. Do not gold-plate."
    - "Probe scope = Tier 2 per tests/AGENTS.md (schema + failure-behavior changes)."
  constraints:
    - "MIRA repo AGENTS.md: commit only when the user explicitly asks. Changes sit uncommitted."
    - "tests/AGENTS.md NO MOCKS: live probes only. writing-probes skill NOT INSTALLED
       on this machine — follow the doctrine text directly; note the gap in the report."
    - "deploy/AGENTS.md: schema change = breaking. BREAKING.md at repo root carries the
       manual ALTER; mira update refuses the release. Do NOT add migration DDL to the
       schema file. Do NOT apply the ALTER to the live DB until the new code deploys
       (old deployed INSERT has ON CONFLICT ... WHERE status='active' which cannot
       infer an arbiter on the new index and would error)."
    - "Map updates are part of the changeset (cns/infrastructure/AGENTS.md, cns/api/AGENTS.md,
       deploy/AGENTS.md, docs/SEGMENT_SYSTEM.md). Run path-anchor + reciprocity audits
       per docs/AGENTS_MAP_SPEC.md. NOT DONE YET."

world_state_delta:
  - "DB surgery DONE and verified (11:21 CDT 2026-10-11, /tmp/mira_repair.sql):
     13 orphan rows (segment a7c09008 sentinel + heartbeat msgs) deleted; synthetic
     pair inserted at tail of b2b817f8 (user where-you-were note + assistant 'Yes.',
     one shared turn_id, metadata {turn_id, segment_id, content_type:text, tools_used:[]});
     b2b817f8 sentinel: still PAUSED, segment_turn_count left at 28 (counter counts
     user-initiated arrivals only — heartbeat user rows never increment it; user's
     next real message arrives as segment turn 29), segment_end_time updated to
     synthetic tail, last_turn_at untouched. Post-checks all green."
  - "Valkey keys continuum:<user>:messages (+epoch, manifest) DELeted post-surgery;
     pending_memories/trinkets preserved."
  - "mira.service RESTARTED but PARKED: boot gate (power_on_self_test) fails
     llm_provider_reachability — route primary (openai:glm-5.3@gw.lunaroute.com)
     ProviderStallError 15s no output, 3 attempts, process parked by design, never
     binds port 1993. Verified: /v1/models answers 0.17s; POST /v1/chat/completions
     glm-5.3 hangs >30s (re-probed at ~11:50, still hanging); other routes passed.
     No local bench/openclaw processes running. NOT fixable from here — user's org
     gateway. systemctl restart mira once glm-5.3 completions answer again."
  - "mira.service stop timed out and was SIGKILLed (graceful shutdown hung) — its own
     issue, flag in report. Because of the SIGKILL the shutdown Valkey flush did not
     run; explicit DEL covered it."
  - "Code fix written in ~/Programming/Scratch/mira (uncommitted, main):
     cns/infrastructure/continuum_repository.py — _ensure_active_segment now matches
     status IN ('active','paused'), reuses found sentinel + aligns
     set_current_segment_id when contextvar differs (plain-dict rows:
     row['metadata'], NOT .metadata — that was a caught bug); ON CONFLICT target
     widened to ('active','paused'); comment reference renamed to
     idx_messages_live_segment_unique. increment_segment_turn: single-row UPDATE
     pinned by subquery (ORDER BY created_at DESC, id DESC LIMIT 1) + explanatory
     comment. cns/api/actions.py — pause_session AND resume_session raise
     ValidationError when _user_request_lock.is_locked(user_id) (same probe as
     collapse action; import from cns.api.chat inside the branch). Comments carry
     the incident rationale. deploy/mira_service_schema.sql — index replaced:
     idx_messages_live_segment_unique WHERE status IN ('active','paused').
     BREAKING.md created at repo root with the manual ALTER + duplicate-cleanup note."
  - "py_compile + pyflakes clean on both .py files."

negative_knowledge:
  - "Bash tool does NOT persist command outputs anywhere (only guardrail_bypass.log
     under ~/.mira_logs). No firehose in the deployed build (grep confirmed)."
  - "Valkey cache is written only at commit; the failed turn's 72 minutes of content
     exists nowhere server-side. TUI scrollback is the only surviving transcript."
  - "Postgres local auth is peer-only; postgres OS user cannot read /home/taylor or
     /opt/mira/app (0700). System python3 has NO psycopg. The app venv has psycopg
     3.3.6 but /opt/mira/app is 0700 → sudo -u postgres cannot use it."
  - "sentinel created_at is backfilled (first-message-time − 1µs), NOT insert time;
     sentinel row ids are uuid4 (no timestamp). Sentinels minted for the same
     conversation position tie on created_at — hence the id DESC tiebreak added
     in every ORDER BY."
  - "segment_turn_count ≠ count of user rows (28 vs 59): heartbeat tick rows carry
     role='user' but never increment the counter. Do not 'fix' this discrepancy."

operational_state:
  current_step: "COMPLETE — probes executed 2026-10-11 ~11:58, all 10 checks PASS."
  probe_result: >
    /tmp/mira_sentinel_probe.py (disposable, one run): schema index present/old gone;
    RLS scoped-visible + contextless fail-closed (witnesses the Vault credentials are
    not BYPASSRLS); save-into-paused reuses sentinel + aligns stale contextvar;
    second live INSERT → UniqueViolation; production-shaped ON CONFLICT infers widened
    index and no-ops; increment resumes paused in place (count 5→6, paused_at cleared);
    legacy two-live (index dropped): increment pins newest only, index recreation
    refused over corrupt state, accepted after cleanup; pause_session AND
    resume_session raise ValidationError under the real cns.api.chat lock on live
    Valkey. One probe-caught bug fixed in the code fix itself: _ensure_active_segment
    still referenced the old variable name active_segment_query after the rename to
    live_segment_query (NameError at runtime; py_compile cannot catch it) — fixed.
    Probe assertion lessons: save_message does NOT stamp segment_id on the message
    (the orchestrator stamps from the aligned contextvar); after a resume the legacy
    check's older sentinel is 'active', not 'paused'.
    Probe plumbing resolution (supersedes the dead-end notes below): the app venv at
    /opt/mira/app is taylor-owned (0700 blocks postgres, not taylor) — the probe ran
    under /opt/mira/app/venv/bin/python3 with the scratch repo first on sys.path. A
    local peer-auth superuser role 'taylor' was created (sudo -u postgres CREATE ROLE
    taylor LOGIN SUPERUSER) so scratch_database()'s admin connection works as written.
    The user-scoped client subclasses PostgresClient (conninfo = Vault's mira_service
    credentials retargeted to the scratch DB), so RLS applies exactly as production
    (mira_dbuser); the fail-closed check is the witness the role is not BYPASSRLS."
  next_commands: >
    Probe plan (not yet written/execute): one disposable probe under /tmp (tests/tmp
    policy is autodelete-on-sight; exemplary shape: tests/tmp/jsonb_roundtrip_probe.py).
    Needs: tests/fixtures/live_infra.require('postgres'), scratch_database() (applies
    the NEW schema incl. new index), then a repo-subclassed ContinuumRepository
    pointing at the scratch DB. Key plumbing fact just discovered:
    PostgresClient(database_name, user_id=..., admin=False) builds conninfo via
    clients.vault_client.get_database_url() from VAULT — probes must instead subclass
    or monkeypatch get_user_db_client to construct PostgresClient directly with
    database_name=<scratch>. PostgresClient sets app.current_user_id per connection
    (set_config); scratch DB users row needed for RLS (users.user_policy) — seed a
    user + continuum row via the fixture's admin connection (admin kwargs are
    conn_kwargs, peer auth as taylor's OS user → connect as postgres over unix
    socket: psycopg.connect(dbname=..., user='postgres') works as the taylor OS
    user via peer). Probe checks: (1) save-with-paused-sentinel reuses it, no new
    sentinel, messages land under its segment_id, contextvar aligned;
    (2) second live sentinel INSERT is rejected by idx_messages_live_segment_unique
    / ON CONFLICT DO NOTHING no-ops; (3) increment_segment_turn on planted
    active+paused pair (plant via direct SQL with index dropped? NO — index blocks
    planting; instead verify single-row pick determinism via ORDER BY, and verify
    resume-from-paused works and returns one row); (4) pause_session refusal under
    held lock — actions.py is an HTTP handler class (ActionsEndpoint?) — may need
    direct call or... check how pause branch is reached (it's inside a big dispatch;
    simplest: probe is_locked refusal logic via the module-level lock object, or
    call the actions class with a stubbed request — decide when writing; do NOT
    mock what can be real: Valkey is live on 6379, UserRequestLock works against
    it). Also run the schema apply once to prove the new index DDL is valid.
  vcs_state: "~/Programming/Scratch/mira on main, f93a7d4, changes uncommitted:
    3 files modified + BREAKING.md new + notes/ untracked (2 files).
    Do NOT touch SEGMENT_TIMEOUT_DEFERRAL_PLAN.md (another session's artifact)."
  todo_state: "#1,#2 done. #3 (code fix) essentially done pending probes. #4 probes
    IN PROGRESS. #5 map updates NOT started. #6 final report NOT started."

open_questions:
  - "How to probe the pause/resume refusal without mocking: UserRequestLock is real
     and Valkey is live, so acquire a real lock for a synthetic user id and call the
     real dispatch if reachable; fall back to asserting is_locked contract + the
     branch's import path. Current bet: reach the ActionsEndpoint dispatch directly
     or extract the check into a small helper if the handler is un-callable —
     prefer NOT refactoring just for the probe; read actions.py's class shape first."
  - "Whether deploy/AGENTS.md map needs a BREAKING.md mention (check its ## Files
     section; likely cites BREAKING.md already)."
  - "Gateway glm-5.3 stall: recheck before final report; if recovered, restart mira
     and verify bind on 1993 + clean journal."

pointers:
  - "Incident causal chain + fix rationale: notes/2026-10-10-turn-setup-failed-duplicate-sentinel-lockout.md"
  - "Fixture contracts: tests/fixtures/AGENTS.md, scratch_db.py (scratch_database,
     _embedding_args via describe_for_installer), live_infra.require, harness.check/finish."
  - "Probe shape exemplar: tests/tmp/jsonb_roundtrip_probe.py + tests/tmp/AGENTS.md
     (autodelete-on-sight policy — disposable probes go in /tmp)."
  - "Sentinel lifecycle contract: cns/infrastructure/AGENTS.md lines ~34-41 (must be
     updated for the new index name + paused-in-_ensure)."
  - "Row shapes for messages inserts (verified): user {turn_id, segment_id,
     content_type}; assistant adds tools_used; tool rows add tool_name,
     tool_arguments, tool_call_id column; content_type 'json' for block lists."
```

## Checkpoint timeline

1. Principal pasted TUI errors (TURN_PROCESSING_FAILED then 2× TURN_SETUP_FAILED) → agent investigated.
2. Agent located codes in cns/api/websocket_chat.py; journal showed provider "scheduler admission timeout" (22:08:24) then UniqueViolation idx_messages_active_segment_unique (22:12:26/32).
3. Agent found DB state: two live sentinels b2b817f8 (paused, 28 turns) + a7c09008 (active, orphan) — root chain: pause landed mid-turn (actions pause has no lock probe) → save hit _ensure_active_segment which counts only 'active' → minted fresh sentinel via stale-contextvar collision branch → ON CONFLICT (active-only) no-op'd → increment's multi-row UPDATE resumed both → index violation → permanent TURN_SETUP_FAILED.
4. Agent filed incident note notes/2026-10-10-turn-setup-failed-duplicate-sentinel-lockout.md.
5. Agent explained design (sentinel-as-message-row, 1µs trick, dual-commit orphan window, lock as serializer); principal asked design questions; agent explained.
6. Principal ruled: continuation semantics matter (mid-task turn was a segment continuation); approved repair + "fix the db rows" + rewind so recent rows are the lost tool calls, then synthetic user/assistant pair ("You just hit a system error…" / "Yes."), then user types "continue" in TUI.
7. Agent verified "Sorry to interrupt" 20:54 turn IS committed (12 rows, halt-path partial commit) — durable, untouched by surgery.
8. Principal ruled heartbeat messages worthless — discard. Agent simplified repair to delete.
9. Agent probed recovery sources: no tmux/screen, no TUI transcript file, no firehose, bash_tool logs guardrail only → scrollback-only; principal said begin → synthetic fallback confirmed.
10. Agent ran surgery 11:21 CDT: stopped mira.service (SIGKILLed after shutdown hang), deleted 13 orphan rows, inserted synthetic pair (uuid casts fixed after two type errors), updated segment_end_time, verified all post-checks, DELeted Valkey continuum keys, restarted service.
11. Discovered segment_turn_count=28 vs 59 user rows → counter counts user-initiated arrivals only → left counter at 28 (corrects agent's earlier "turn 29/30" claim; user's next message = segment turn 29).
12. Service boot parked: llm_provider_reachability fails — glm-5.3 completions hang >30s (models endpoint fine, other routes fine). Agent cannot fix; must recheck before report.
13. Agent wrote the 4-part code fix in the repo (uncommitted) + BREAKING.md; py_compile/pyflakes clean; caught own dict-vs-attribute bug in the reuse branch by re-reading execute_query's contract.
14. Principal asked "more complex than we discussed?" → agent confirmed scope unchanged; ruled: do NOT apply ALTER to live DB until new code deploys (old ON CONFLICT arbiter incompatibility); noted parked service.
15. Agent began probe work; read tests/AGENTS.md doctrine (no writing-probes skill installed), fixtures maps, exemplar probe, PostgresClient/Vault plumbing; resolved scratch-DB admin-auth constraints (peer auth, 0700 dirs).
16. Principal announced compaction → agent wrote this handoff.
