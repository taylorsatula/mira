# Conditional Ctrl+C in the TUI (2026-10-11)

Status: IMPLEMENTED 2026-10-11, UNVERIFIED — no deployed instance reachable
from this checkout (store's `localhost:1993` not listening; no Vault AppRole
env, so the server cannot boot here per root map). Covering probe below.

## The behavior (principal's spec, verbatim intent)

`tui/chat.py:_on_interrupt` resolves one Ctrl+C against the box first, then
the turn, then the session:

1. Text in the input box → clear the box, nothing else. A draft being typed
   during a turn never halts the turn (old behavior: Ctrl+C halted the turn
   regardless of box content — the bug this change fixes).
2. Empty box + turn in flight → halt the turn. A press while a halt is
   already pending is a no-op.
3. Idle + empty box → arm `QUIT_DOUBLE_TAP_SECONDS` (3 s) and emit
   "press Ctrl+C again within 3s to exit"; a second Ctrl+C inside the window
   quits (exit 0). Every non-idle handling disarms the window (disarm happens
   at the top of the handler, so a draft typed between taps or any turn in
   between breaks the chain).

## Removed quit paths (deliberate)

The old code had two force-quit shortcuts removed by this change, because
the spec's quit condition ("no turn active, no input") excludes both states:

- Second Ctrl+C while `turn.halt_requested` used to quit mid-stop.
- Second Ctrl+C while `_halt_on_start` (send awaiting `turn_started`) used
  to quit.

Now both are no-ops; quitting is only `/exit`, Ctrl+D on an empty box, or the
double-tap window. If a stuck halt ever needs a force-exit, that is a new
decision for the principal, not preserved behavior.

## Implementation notes

- Box state is read live at handler time via `Screen.clear_input()`
  (compare-and-clear returns False on an empty box), not captured at
  keypress time — the single-consumer inbox may lag the keyboard.
- `QUIT_DOUBLE_TAP_SECONDS = 3.0` lives with the other chat.py timing
  constants; arming timestamp `self._quit_armed_at` (monotonic) on
  `ChatSession`.
- Kept as-is: empty-box Ctrl+C during a TURN_BUSY retry wait cancels the
  send (restores to box), during `connecting` cancels the connect. But note
  the new trap: with text in the box, the first Ctrl+C now clears the draft
  instead of cancelling the TURN_BUSY wait — that is the spec's letter
  (input wins) applied to a state the spec didn't name.
- User-facing strings updated to match: status hints now read
  "Ctrl+C clears input / stops" (chat.py `_status`), banner reads
  "Ctrl+C clear/stop · Ctrl+C ×2 quit" (transcript.py `banner`),
  `__main__.py` exit-code docstring names the window.
- Maps updated same-change: `tui/AGENTS.md` Wiring (new conditional-Ctrl+C
  bullet + rewritten Quit bullet), TURN_BUSY clause made empty-box-precise,
  screen.py bullet points at `_on_interrupt` as the policy owner.
- `tui/FRONTEND_PLAN.md` still documents the old table (line ~388) —
  historical record, code is truth per its own header; left alone.

## Covering probe (unrun)

Pty-driven TUI session against a deployed instance (`--config PATH` exists
for exactly this, per `__main__.py` docstring):

1. Mid-turn: type a draft, Ctrl+C → box clears, turn continues streaming.
2. Mid-turn, empty box: Ctrl+C → halt sent, `stopped` row; second press is
   a no-op (session alive).
3. Idle, empty box: Ctrl+C → exit notice; second within 3 s → process exits
   0; second after 3 s → another notice, session alive.
