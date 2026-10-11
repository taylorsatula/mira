# Incident: TURN_SETUP_FAILED chat lockout from a duplicate live segment sentinel (2026-10-10 evening)

Author: pi session on lunaroute-taylor-work. Instance: local `mira.service`
(PID 2314423, `/opt/mira/app`, journal via `journalctl -u mira` or
`journalctl _PID=2314423`). Companion incident to
`2026-10-10-turn-processing-failed-429.md` (same gateway saturation day;
the 22:08 provider failure was "scheduler admission timeout", the
same-family failure mode of the morning's 429s). Not yet distilled
into topical docs.

## User-visible symptoms

Long tool-call turn (MIRA driving claw-bench investigation in
mira-heartbeat via bash_tool, one turn, ~72 min) died with
`TURN_PROCESSING_FAILED` and the copy "The request took too long to
process. Please try again with a simpler message." Every subsequent
message then failed instantly with `TURN_SETUP_FAILED`
("I encountered an unexpected error while processing your message") —
a hard lockout: no chat entry point works until the DB is repaired.

## Verified causal chain (journal + mira_service DB)

All times CDT 2026-10-10. User 36dade70-d672-4506-a702-e3f020ffb86b,
continuum 642e8957, segment sentinels b2b817f8 (paused, 28 turns) and
a7c09008 (active, 1 turn).

1. **20:56:39** — user's long turn starts on segment b2b817f8 (turn 28
   increment stamps `last_turn_at`), holds the per-user Valkey request
   lock (deployed lock TTL ~100h, renewal active). Heartbeat ticks are
   lock-blocked for the whole turn.
2. **21:07:26 / 21:36:42 / 22:02:44** — actions-API pause/resume calls
   land (ActionsEndpoint warnings: "already active", "already paused",
   "already paused" + "already active" 103ms apart). Successful
   pause/resume calls are silent (no INFO in journal). Net verified
   result: b2b817f8's status was active → paused at 22:02:44.752
   (`paused_at` stamp). Inference, not verified: the caller was the
   user testing the TUI pause feature being developed in another
   session (per the 20:56 message content "A TUI change is being made
   in another session"). **The pause succeeded while the 66-minute
   turn was still in flight on that same segment.**
   `cns/api/actions.py` `pause_session` has no request-lock probe
   (the `collapse_session` branch right above it demonstrates the
   refusal pattern it should follow).
3. **22:08:24** — the long turn dies: provider streaming failure
   `gw.lunaroute.com ... openai provider API error 200: scheduler
   admission timeout` (gateway saturation, external). TURN_PROCESSING_FAILED;
   `get_friendly_error_message` matched "timeout" in the error text and
   sold it as "try again with a simpler message" — misleading copy;
   message size was irrelevant. All of the turn's 72 minutes of
   assistant/tool messages were lost (nothing commits mid-turn; the
   staged user message from 20:56:39 is all that persists).
4. **[22:02:44.752, 22:09:09.465)** — a save carrying the b2b817f8
   segment contextvar hit `_ensure_active_segment`
   (`cns/infrastructure/continuum_repository.py` ~line 291). That method
   queries `status = 'active'` ONLY — a paused segment reads as "no
   segment" — so it proceeded to mint a sentinel: contextvar id
   b2b817f8 collided with the existing b2b817f8 sentinel (the
   "stale contextvar" re-check branch allocates a fresh `uuid4`),
   INSERT with `ON CONFLICT ... WHERE status = 'active' DO NOTHING` —
   no conflict, because the only live sentinel (b2b817f8) is paused
   and outside the partial index. Result: **a second live sentinel,
   a7c09008, active, autocommitted**. The message batch behind it
   rolled back (zero messages under a7c09008 before 22:09:09) — the
   documented "orphan-sentinel window" in the save_messages_batch
   docstring, now instantiated. Exact caller not identifiable:
   INFO-level logs are suppressed in this journal, so the sentinel
   INSERT logged silently; candidates (dying-turn cleanup save,
   lock-blocked heartbeat turn from 22:02:09 unblocking at 22:08:24)
   all route through the same branch.
5. **22:09:09** — first post-lock heartbeat tick hb_3ed2866023cc.
   `find_active_segment` (matches active+paused, ORDER BY created_at
   DESC LIMIT 1) hit the tie — both sentinels carry the identical
   backfilled created_at 2026-10-09 17:57:51.657204 — and returned
   a7c09008 (active), so the paused-check did not skip; the heartbeat
   ran its turn on a7c09008 (staged message committed 22:09:09.489;
   turn halted at the 120s deadline 22:11:09, halt 22:11:28).
6. **22:12:26, 22:12:32** — the user's messages reached
   `increment_segment_turn` (continuum_repository.py ~line 897).
   Its UPDATE matches `status IN ('active', 'paused')` — both rows
   match — and sets both to 'active', which the partial unique index
   `idx_messages_active_segment_unique` (messages(continuum_id) WHERE
   boundary AND status='active') rejects:
   `psycopg.errors.UniqueViolation`. segment_id is still None at that
   point → TURN_SETUP_FAILED, generic copy (error text matches none of
   the friendly-mapper keywords). **Instant, permanent, every message.**

## Root cause, one sentence

The invariant "one live (active-or-paused) sentinel per continuum"
exists only as convention: `_ensure_active_segment` counts only
'active', `find_active_segment`/`increment_segment_turn` count
'active'+'paused', and the unique index enforces only 'active' — so a
save that runs while the real segment is paused mints a second live
sentinel, and once two exist, every chat entry UPDATEs both to active
and the index turns each message into an unrecoverable TURN_SETUP_FAILED.

## Proposed root fix (agreed direction, implementation pending)

1. `_ensure_active_segment`: match `status IN ('active','paused')`
   (paused is a state of the live segment, not its absence); reuse the
   found sentinel and `set_current_segment_id(sentinel.segment_id)` so
   the batch's messages land under it. Closes the mint at every writer.
2. Schema: widen the partial unique index to
   `status IN ('active','paused')` (drop
   `idx_messages_active_segment_unique`, create
   `idx_messages_live_segment_unique` in
   `deploy/mira_service_schema.sql` + migration); update the ON CONFLICT
   target in `_ensure_active_segment` to the new predicate. Guard lives
   in the mutation: two live sentinels become unrepresentable. Migration
   fails loudly on pre-corrupted rows — pair with the cleanup.
3. `actions.py` pause_session/resume_session: probe the per-user request
   lock (same mechanism as the timeout service's lock probe,
   `cns/services/segment_timeout_service.py`); refuse while a turn is
   in flight (same pattern as collapse's "still being processed").
4. `increment_segment_turn`: UPDATE exactly one row — pin the target
   sentinel (SELECT ... ORDER BY created_at DESC, id LIMIT 1; UPDATE
   WHERE id = %s), assert one affected row, raise a named diagnostic
   (not a raw UniqueViolation surfaced as "unexpected error") if the
   invariant is ever violated again.
5. Repair the live DB (user decision pending, instance still locked
   out at time of writing): a7c09008 holds 5 rows — the sentinel, one
   staged heartbeat user message (hb_3ed2866023cc, 22:09:09), and the
   completed hb_cae5fbea605 heartbeat exchange (user/tool/2x assistant,
   22:17:09, keepsleeping). Options: (a) reassign those 4 messages'
   metadata segment_id to b2b817f8 and delete the a7c09008 sentinel
   (keeps the history in the real segment); (b) mark a7c09008 collapsed
   (leaves an uncollapsed-processed stray segment). b2b817f8 is the
   real 28-turn session and stays paused-until-next-message either way.

Second-order (flagged to user, design changes, not in root fix):
- Long turns lose ALL work on mid-stream provider failure (72 min of
  tool calls left no server-side record). The halt path already retains
  partial content; incremental tool-loop checkpointing would make a
  provider death lose one round, not the turn. Changes UnitOfWork
  per-turn atomicity — deliberate decision needed.
- `get_friendly_error_message` maps provider "timeout" text to
  "try again with a simpler message" — misdiagnosis copy.

## Method / reuse

- Journal: `journalctl _PID=2314423 --no-pager --since ... --until ...`;
  INFO is suppressed in this journal, WARNING/ERROR only — successful
  actions calls and `_ensure_active_segment`'s "Created segment boundary
  sentinel" INFO leave no trace; absence of log lines is not absence of
  events.
- DB: `echo '4231' | sudo -S -u postgres psql -d mira_service` (sudo
  to postgres bypasses RLS). messages.metadata is jsonb; sentinel
  created_at is backfilled from first-message-time, not insert time;
  sentinel row ids are uuid4 (no embedded timestamp) — insert time of a
  sentinel row is not recoverable from the DB itself.
- Sentry-id queries: filter `metadata->>'is_segment_boundary' = 'true'`.

## Open questions

- Exact caller of the orphan-minting save (dying-turn cleanup vs
  unblocked heartbeat) — unresolvable from this journal (INFO
  suppressed); the fix does not depend on it.
- Connection to the morning note's lock finding: the deployed chat lock
  TTL ~100h is why a 72-minute turn holding it blocked every heartbeat
  tick silently; interplay already flagged there.
- Both sentinels share created_at 17:57:51.657204 because
  `_ensure_active_segment` backfills first_message_time after the last
  collapsed segment — ORDER BY created_at DESC cannot order them
  (nondeterministic tie), which is how find_active_segment handed the
  heartbeat the orphan.
