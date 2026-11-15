# MIRA v2026.11.14-2.3.1

This is a point release. It ships the two-minute follow-up nudge — merged
after the v2026.11.13-2.3.0 tag, so it never reached an install — and a fix
for installs on Homebrew macOS. No database steps and no dependency
changes, so an update has nothing to consent to and no virtualenv to
rebuild.

## Amendment (2026-11-15): tag moved to ship the boot-gate fix

The tag `v2026.11.14-2.3.1` was moved from `b50749e` to `cf43eed` and the
asset was rebuilt and re-uploaded under the same name and release id. The
release id, `VERSION`, and the asset filename are unchanged; what changed is
what the tag points at. `cf43eed` adds one fix on top of the original tag:
the pre-server boot gate now waits for Postgres, Valkey and Vault to answer
before running its verification rounds (`MIRA_DEPENDENCY_WAIT_SECONDS`,
default 180 seconds, `0` disables), so a slow-starting database no longer
consumes the round budget and parks a healthy-but-slow boot. If the wait
expires, the gate takes the park/exit action without running the rounds, so
the billed provider probes never run against a boot whose infrastructure
never came up. No database steps, no dependency changes, no `VERSION`
change. Gate rows a–c were re-verified at rebuild time (lint, tag/version/
asset identity, `RELEASE_COMMIT` = `cf43eed`); rows d and e remain as
recorded below. Anyone who downloaded the asset before this amendment holds
the pre-fix build; re-download to get the fix.

## A quiet user gets one nudge, not a stream

After a completed turn, MIRA now arms a one-shot two-minute timer. If the
user replies before it fires, the timer is cancelled and nothing happens.
If the timer fires, MIRA makes one `primary`-route call with the recent
exchange and no tools; the model answers with a single JSON object deciding
whether to speak. If it says yes, one short message is persisted and pushed
to open clients as a normal assistant message, stamped `follow_up: true`. If
it says no, or the answer is anything other than the exact JSON shape,
nothing is sent.

At most one follow-up is armed per turn and the model call happens at most
once per armed timer, so the feature cannot nag a second time off the same
turn. Pause is honored: if the segment is paused at fire time, the nudge is
dropped. Cancelling on the next user message is unconditional, so an
ordinary reply always wins the race. A follow-up is not a user turn and
does not advance the segment clock.

On by default. To turn it off, set `MIRA_FOLLOW_UP_ENABLED=0`. The model
decides whether to speak at all, and the prompt biases it hard toward
silence; the flag only controls whether the question is asked.

This is deliberately narrow. It is not a general proactive-messaging
system, has no new tool, does not act on the user's behalf, and adds no
new persisted state — the `follow_up` metadata flag on the message row is
the entire record. Verified so far by observed one-shot firing on a live
instance; the reply-inside-the-window and silence paths are not yet
exercised under test.

## Homebrew macOS installs can build their virtualenv again

`venv_base_python` appended `/bin/python3` to the venv's `sys.base_prefix`.
Homebrew's `python@3.x` keg ships only the versioned `python3.X` in its
Framework bin — no unversioned `python3` — so that path does not exist and
`write_venv_stamp` died in `python_version_of` with "No such file or
directory". The resolver now probes `base_prefix` for the interpreter that
actually exists, preferring the unversioned `python3` (Linux and
python.org framework builds, unchanged behavior) and falling back to
`python3.X`, then `sys._base_executable`.

## For upgraders

`mira update`, as always, from v2026.10.16-2.1 or later. Nothing to consent
to, no dependency rebuild, no database steps. The follow-up nudge is on
unless `MIRA_FOLLOW_UP_ENABLED=0` is set.

## Verification

- Gate rows a–c verified at build time against the tagged asset:
  `build_release_artifact v2026.11.14-2.3.1` passed `scripts.migrate lint`
  (3 versioned and 1 repeatable step follow the rules), and
  `verify_release_identity` confirmed tag = v + VERSION, the asset's VERSION
  matches the repo, and the asset's RELEASE_COMMIT equals the tagged commit
  (b50749e at cut time; cf43eed after the 2026-11-15 amendment, re-verified
  at rebuild).
- Row c (convergence) ran on the build host against a local PostgreSQL 17
  superuser. Every origin and revision reached the fresh-install structure;
  the command rewrote `deploy/migrations/BASELINE` and
  `HEAD_FINGERPRINT.json` with byte-identical content, so the checked-in
  generated files still extend correctly.
- Gate rows d (skip-chain) and e (restore round trip): not run at cut time.
  This release ships no database steps, but it does change
  `deploy/lib/venv_stamp.sh` (the Homebrew base-interpreter resolution),
  which the update path exercises. The build host holds no VM substrate, so
  the rows carry their v2026.10.21-2.1.2 results unchanged with that caveat
  recorded here; the first `mira update` onto a Homebrew macOS install is
  the witness for the venv_stamp change.
- The pinned requirement set installed from scratch into a clean virtualenv
  on the build host at cut time.
