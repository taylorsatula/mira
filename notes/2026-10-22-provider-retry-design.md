# Provider retry with fallback — design (2026-10-22)

Status: DECIDED, not yet implemented. Operator decisions recorded by
questionnaire this date. Supersedes the "fail the turn, divert the next"
decision of notes/2026-10-11-llm-failover-design.md (that note's diverter
machinery is unchanged; only the per-call policy above it changes).

## The four decisions (operator, 2026-10-22)

1. Schedule: retries 3 (so 4 attempts total); backoff doubles from a
   configurable base, default 30s, no single sleep over 180s
   (30/60/120 with defaults). Worst-case added wait 3.5 min.
2. Scope: all ProviderRetryableError, ProviderStallError and HTTP
   timeouts included — one failure class, no dialect changes. Accepted
   worst case: a stalling primary holds a turn roughly
   4 x provider_response_timeout + backoff before the fallback attempt.
3. Fallback: single attempt on the backup after exhaustion; its failure
   raises (one diversion level). While the diverter is tripped, calls
   start on the backup and get the retry budget; their exhaustion
   crashes out.
4. Notice: WARNING journal line per failed attempt plus a client-visible
   in-turn notice frame through the existing stream pipeline; both
   clients render it (twin rule).

## Contract (planned)

- Retry loop lives in clients/llm_provider.py generate_response and
  stream_events — the census-verified choke point for every text call.
  The lifecycle is untouched: it bounds one pull; retry is
  provider-level policy.
- stream_events carries a first-event gate: once provider content (text,
  thinking, tool detection) has been yielded, no retry — propagation as
  today, because a retry would duplicate user-visible output. Notice
  events do not trip the gate. The 2026-10-21 22:47-era 503-on-initial-
  response case (the 23:13 incident) retries cleanly; a mid-reply
  disconnect still fails honestly.
- Backoff sleep waits on the turn's cancel event when present, then
  check_cancelled(); GenerationCancelled ends the retry immediately.
  Background callers without a cancel context sleep plainly, bounded.
- Exhaustion: if the current selection is not the backup and
  failover_enabled plus the failover_* block is configured, build the
  backup selection via clients/llm/resolver.py's per-call divert builder
  (route name and failover_max_tokens_cap preserved) and run one
  attempt under the same gate; otherwise raise. "Crash out" is the
  existing turn-failure path (terminal error frame), unchanged.
- Diverter accounting keeps the call as its unit: one record_failure per
  call that exhausts its retries (NOT per attempt — the trip threshold's
  meaning is unchanged), record_success on any completion. After
  roughly failover_trip_consecutive_failures exhausted calls the
  diverter trips and subsequent calls start on the backup without paying
  retry latency; the recovery probe flips back as built.
- New config rows (config.py ApiConfig): provider_retry_max_attempts=3,
  provider_retry_backoff_base_seconds=30,
  provider_retry_backoff_max_seconds=180; validator max >= base,
  attempts >= 0. Retry is independent of failover_enabled; the fallback
  fires only when the failover block is configured.
- New StreamEvent member (clients/llm/events.py) carries the notice
  (reason, attempt, retry-in seconds, using-fallback flag); the frame
  flows llm_provider -> orchestrator pass-through ->
  websocket_chat._server_frame_for_event -> both clients' frame tables.
  Unmapped events are silently dropped server-side, so the frame-table
  entry is required. Non-streaming callers get journal lines only.
- Turn lock safe: TTL is (MAX_LOCAL_TOOL_CALLS_PER_TURN+1) x
  max(api.timeout, provider_response_timeout) x 2 and refreshed — a
  crash backstop (cns/api/chat.py), not a turn-length estimate.
- Maps in the same commit: root AGENTS.md "LLM stalls propagate ...
  no fallback route" Cross-Cutting bullet rewritten;
  clients/llm/AGENTS.md (retry policy), config/AGENTS.md (rows),
  cns/api/AGENTS.md and ui maps (frame), llm_provider module docstring.

## Probe plan (Tier 2, at implementation)

- Refused local port as primary with shrunk backoff base: assert 4
  attempts with expected deltas, WARNING lines, fallback call to the
  live gateway (billed), completion; failover-disabled variant raises.
- Scratch in-process SSE endpoint emitting one delta then aborting:
  assert single attempt, propagation (first-event gate).
- Cancel set during backoff: GenerationCancelled, no further attempts.
- Config mutability for the probe's shrunk backoff is the one
  implementation-time unknown (frozen pydantic config -> patch class
  default before instantiation).
- Boot gate POST is the out-of-funnel caller and is unaffected.

## Excluded by decision

- Embeddings retry (advisory logging only, per 2026-10-11 design).
- Second fallback level; retry budget on the backup after exhaustion.
- Diverter trip/flip-back/probe machinery changes.
- ProviderRetryableError rename.

## Implementation record (2026-10-22, same day)

Branch feat/provider-retry. Tier 2 verification: five-case live probe
(/tmp/provider-retry-probe.py, throwaway) — dead primary + configured
fallback completes on the live gateway (3 retry warnings + 1 fallback,
billed round trip); failover-disabled crash-out raises after 3 retries;
streaming case shows notices (1,30,F)(2,60,F)(3,120,F)(4,0,True) and
completes; first-event gate case (scratch SSE endpoint, one delta then
abrupt mid-body close) retries nothing and propagates; cancel case wakes
the 30s backoff at 1.2s and raises GenerationCancelled. Second-agent
re-derivation found three defects, all fixed before commit: the
server-side ServerFrame union rejected provider_notice (every notice
would have aborted the turn at ChatConnection.send's validation — the
probe had not covered the frame gate), exhaustion without fallback was
unlogged, and tools/HOW_TO_BUILD_A_TOOL.md still said "no fallback
route".

Deployed to /opt/mira/app the same night (operator request: resilience
during live use before PR review). The deployed tree is v2026.10.16-2.1;
its five target files differ from main by docstrings only except
config.py (v0 MCP string) and orchestrator.py (v0 file link), so the LLM
files were copied wholesale and config/orchestrator patched in place;
the v0→v1 wire promotion on main deliberately did NOT ride along.
Pre-patch backups: /opt/mira/provider-retry-backup-20261022/. Post-
restart POST passed, zero tracebacks; probe suite re-run green against
the deployed tree. Deployed maps were not updated (the release deploy
syncs them); the PR carries the map edits.
