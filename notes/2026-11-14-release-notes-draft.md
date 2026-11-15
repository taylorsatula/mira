# MIRA v2026.11.14-2.3.1

This is a point release. It ships the two-minute follow-up nudge — merged
after the v2026.11.13-2.3.0 tag, so it never reached an install — and a fix
for installs on Homebrew macOS. No database steps and no dependency
changes, so an update has nothing to consent to and no virtualenv to
rebuild.

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

Gate results are recorded in a follow-up commit after the tag is built;
this release ships no database steps. Note that `deploy/lib/venv_stamp.sh`
changes in this release, so gate rows d and e cannot be carried forward
unexamined.
