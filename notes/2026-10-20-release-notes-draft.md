# MIRA v2026.10.20-2.1.1

This is the largest release since 2.0. MIRA now has a browser chat client,
updates itself in place — database included, with your consent, a backup
first, and a full rollback if anything fails — and its long-running
subsystems (heartbeat, subagents, context compaction) were redesigned from
the ground up.

Run `mira update` as usual. The update flow itself is new in this release:
it shows you what it will do to the database and asks once before touching
anything.

**2.1.1 corrects the 2.1 build.** The 2.1 TUI refused to start: `mira`
exited with `ValueError: Invalid key: c-enter` before drawing anything,
because the Ctrl+Enter binding named a key prompt_toolkit does not have.
No terminal can deliver Ctrl+Enter as its own key anyway, so the
interrupt-and-send action moved to Alt+Enter. If you installed 2.1 and
never got a prompt, this release is the fix — nothing in your data was
affected.

---

## What's new

### A browser chat client

Open `http://localhost:1993/` on the machine running MIRA (or any device
that can reach it) and you land directly in a full web chat UI: chat,
settings, memories, and an Automations tab. On a single-user install the
visitor is signed in automatically. The TUI (`mira`) still works exactly
as before — the two clients share the same server, the same turn protocol,
and the same memory.

### Updates that migrate the database in place

`mira update` no longer refuses releases that change the database:

- Before anything is touched, the updater shows the plan — which
  database steps will run and what each does — and asks once.
- It takes a backup first (code, venvs, database) and puts everything
  back if any step fails.
- New commands: `mira backup` (pull a full backup any time, without
  touching the install), `mira restore <STAMP>`, `mira migrate`,
  `mira status`.
- A boot guard now refuses to serve a database that doesn't match the
  installed code's structure — a failed update can't silently run
  against half-migrated data.
- `mira update --nightly` installs the latest main-branch code between
  releases.
- Fresh installs can adopt an existing old MIRA database in place
  (with the same consent + backup + rollback flow) instead of the old
  rename-aside dance.

The three database steps shipped with this release are applied
automatically (details under "For upgraders" below).

### Heartbeat 2.0: briefs instead of blind wake-ups

The old wake cycle woke the main conversation with no payload and could
grind forever on an unreachable goal — it was observed retrying one
instruction 15 times overnight. Wake-ups are now **briefs**: persisted
wake-ups with a mission, a condition to check, a bail-out condition on
one-off briefs, and an exit state for every loop. Runs that fail back off
and terminate permanently after three failures instead of looping
forever. Briefs run on the fast model as an ephemeral agent and never
touch your conversation.

### One unified subagent system

The four separate subagent tools (`phoneafriend_tool`, `forage_tool`,
`whilethecatsaway_tool`, `sidebaragents_tool`) are now one
`subagent_tool` with activate / continue / dismiss / inspect / resolve,
one task-id family, and one results view. Phone-a-friend consults now run
synchronously — you get the answer in the turn instead of waiting on a
background agent. Every agent spawn now respects a global concurrency
cap (8) with a truthful capacity message when full.

### Shell tool: safer, more capable, cancellable

- **Operator blocklist** — config-level patterns that refuse commands no
  mode can bypass, for personal tripwires that outrank model judgment.
- **Skip mode** — with `dangerous_skip_permissions_enabled` on, a call
  carrying `skip_permissions: true` may take ordinarily-refused but
  *recoverable* actions after an explicit check-in, while the
  catastrophic core (disk-level destruction, container deletion,
  audit-log erasure, persistence planting — sudoers, SSH keys) stays
  enforced in every mode. Every bypass is logged to
  `guardrail_bypass.log`.
  - The catastrophic core was also hardened in *default* mode this
    release: multi-line command chains, unrecognized in-place edit
    spellings (`perl -pi`, `sed -Ei`, `awk -i inplace`), and
    authorized_keys copy/link planting are now refused.
- **`restart_mira`** — the sanctioned way for MIRA to restart itself
  after a change that needs a reboot; the restart waits for every
  in-flight turn (including yours) to finish, then hands off to the
  service supervisor.
- **`kill_background`** — background jobs are recorded per user and can
  be stopped by id, killing the whole process group even after the
  launching shell exited.
- **Halt works during tool calls** — a Ctrl+C / halt now interrupts a
  running shell command within a second instead of waiting out its
  timeout (previously a heartbeat turn ran 29 minutes past its deadline
  inside a foreground command, bouncing your messages with "MIRA is
  busy").
- Shell results now arrive with the directory's AGENTS.md context maps
  attached when relevant, so the model knows the local contracts of the
  tree it's working in.

### TUI overhaul

- Turns render as open boxes in a reading-width band; MIRA is bright
  green on the left, you're magenta on the right. Markdown (headings,
  bold, italic, code) renders in replies.
- Tool calls render **live** — name, arguments, elapsed timer, and
  errors — instead of a bare "used <tool>" row.
- Slash commands: `/pause`, `/resume`, `/status`, `/collapse` now work
  in the TUI (plus `/exit`).
- Startup replays the previous session's summary as a dim context block,
  so a restart no longer starts blind.
- Ctrl+C now clears the input box first; quitting requires a double tap
  within 3 seconds.
- Alt+Enter sends your message now, interrupting the reply in progress;
  `\` then Enter or Ctrl+J starts a new line.
- Resizing mid-turn no longer leaves stray bar fragments in
  scrollback; the window title is now `:mira:`.
- `mira update --nightly` supported.

### Provider resilience

- **Failover (opt-in)** — configure a backup endpoint and MIRA diverts
  all LLM calls after 3 consecutive provider failures, then probes the
  primary every 2 minutes and flips back automatically. Off by default;
  nothing changes until configured.
- **Boot gate fixed for thinking models** — the startup reachability
  probe now succeeds on the first streamed token instead of a full
  completion, so a slow-to-reason provider (which previously parked the
  whole server at boot) passes honestly.
- **Timeout bounds raised** — provider timeouts went from 180s to 900s:
  large segment-collapse summaries on a loaded gateway legitimately take
  minutes and were being killed, jamming the collapse sweep for hours.
- **Halts abort stalled pulls** — a user halt during a hung provider
  response is observed within half a second and cancels the request
  instead of being classified as a stall at the deadline.

---

## For upgraders

**How to update:** `mira update`, as always. It asks once before any
database change and backs up first.

### Applied automatically by the update's database steps

1. **Segment sentinel uniqueness** — a pause landing mid-turn used to be
   able to create a duplicate live segment marker, permanently locking
   that conversation with TURN_SETUP_FAILED on every message. The index
   now enforces one live segment per conversation, and legacy duplicate
   pairs are pinned non-bricking (newest wins).
   ⚠️ If a conversation is *already* in the corrupt state, the update
   stops before changing anything and tells you which conversation needs
   its duplicate removed (ask MIRA to remove it, then re-run
   `mira update`).
2. Removes a keepsleeping turn index retired with the old heartbeat.
3. Adds the heartbeat briefs coordination table.

### Defaults that changed under you

- **Compaction fires much later.** Context-window defaults were retuned
  from 200k window / 40k trigger to 256k window / 225k trigger; new
  installs resolve both from the chat endpoint at install time instead
  of guessing. Long conversations stay full-fidelity much longer.
- **Timeout ceilings up:** provider response timeout 180s → 900s,
  scheduled-job timeout 240s → 3600s (details in the timeout section
  above).
- **Per-turn tool-call ceiling raised 50 → 1000** — long autonomous
  turns no longer get cut off and answered from partial information.
  The trade-off: a pathological turn can now run a long time before the
  circuit breaker; halt it with Ctrl+C (which now also works inside
  tool calls).
- **New installs default to emb-qwen3 embeddings** (the previous
  default's endpoint rejected inputs over 512 tokens, aborting whole
  turns). Existing installs are untouched: embedding models can't be
  swapped after the fact, and your install keeps its configured model.
- **Fresh installs use PostgreSQL 17** (existing installs keep the
  server they run on; the update doesn't touch it).

### Reliability fixes worth knowing

- **Single-mode browser sessions pin to the original account.** If a QA
  campaign or browser visit ever provisioned a second empty
  `user@localhost` fixture account beside your real one, a browser
  session could land in the empty one ("a MIRA that knew nothing about
  me"). Sessions now always resolve to the oldest member account — the
  one your history lives on.
- **No more "MIRA is busy" right after a reply** — the per-user turn
  lock is now released before the terminal frame is sent, so a
  follow-up message you send the instant a reply completes goes through.
  (Remaining source: a heartbeat turn holding the lock — use
  `heartbeat.wake_mode "pregated"`, or the new `/pause`, if you hit it.)
- **Segments are no longer collapsed out from under running turns.** A
  multi-hour tool-loop turn used to read as "inactive" to the timeout
  sweep (turns commit atomically, so nothing was visible until the
  end), and its whole content could vanish from every digest. The sweep
  now checks whether a turn is in flight before publishing a timeout.
- **One rolling compaction report per segment** replaces the old
  compaction trinket: the covered history is replaced by a single
  updated brief that carries forward saved memories and reminders, and
  restarting the server no longer resends the full conversation.
- **Cost accounting fixed for cached tokens** — OpenAI-compatible routes
  were billed twice for cached input tokens; `Usage.input_tokens` now
  means the same thing in every dialect.
- **Dead tool-result compaction removed** — the summarizer disabled in
  September is deleted along with everything that existed only to serve
  it, including per-turn waits that always returned immediately.

---

## For operators and developers

- Tool results stored from old turns keep their existing truncation;
  nothing is reprocessed at update.
- Per-user tool config overrides merge fresh on every access, as before.
- `mira update` diffs your install against the last provenance commit to
  detect operator-made changes; this release records provenance for
  installs, updates, nightlies, and restores.

## Verification

- Platform batteries (Ubuntu, Fedora, Debian, Arch) green across the
  update/rollback/power-cut scenario matrix; container scenario green;
  macOS native gauntlet green in both passes (real install upgraded in
  place with memories surviving every hop, rollback witnessed under
  launchd, restore round trip).
- Release gate rows d & e (skip-chain; restore round trip): exercised
  during the update-mechanism verification campaign against this code —
  a genuine pre-mechanism v2026.10.06-2.0 install updated in a single
  hop to this tree (row d), and failed-step rollback plus restore round
  trip witnessed on the platform batteries and under launchd on macOS
  (row e). Not re-run against the tagged asset at cut time.
## Rev 2026-10-16 (post-publish): custom-build export classification

The tag now points at 702a2de (asset rebuilt and re-attached; same
version, no database steps changed). RESTORE_MY_CUSTOM_BUILD.txt now
tags every changed or added file against the release being installed
(absorbed / clean-apply / both-changed / upstream-deleted), marks
config/ and tools/ files operator-only (MIRA's guardrail refuses
agent edits there, so those restores end with the operator), and
states that collaborating with the operator on ambiguous restores is
expected, not a fallback — the restoring agent may lack the
conversation that explains the release's redesign, and forcing an old
version back over it is worse than no restore. Gate rows a-c
re-verified at build time; rows d and e unchanged (no migration or
updater-flow code changed).

- Known residuals: the bash-tool battery still carries the 8
  pre-existing interpreter-payload escapes (unchanged from the last
  release); terminals that derive their title from the running command
  may override the `:mira:` window title.

## Rev 2026-10-16 (post-publish): sidebar agent memory

The tag now points at 0bd8c36 (asset rebuilt and re-attached; same
version, no database steps changed). Adds a coarse MEMORY.md-style agent
memory: a single per-user agent_memory pool tagged by agent_id, push-loaded
into sidebar agent prompts, written only at complete_task with a stored
justification, plus main-context inspect/update/deprecate operations and a
configurable too-many-memories alert in the subagent results trinket.

- Known residuals: unchanged.
