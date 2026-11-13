# MIRA v2026.11.13-2.3.0

This is a feature release. Its defining changes are a provider failure no
longer costing the turn, and an update that preserves the operator's own
`config/config.py` values. It also carries background job completion
notices, tool-call descriptions in the browser client, a lite memory
surfacing mode for power-constrained installs, and four corrections to the
bash tool's guardrails. No database steps ship and the dependency set is
unchanged, so an update has nothing to consent to and no virtualenv to
rebuild.

## A provider blip no longer costs the turn

One HTTP 503 mid-stream used to lose the whole turn when no failover block
was configured, and even with one configured the failing call still died:
only calls after the diverter's consecutive-failure trip reached the
backup.

Every text call now retries a retryable provider failure, stalls included,
up to `provider_retry_max_attempts` (default 3), with a doubling backoff
from `provider_retry_backoff_base_seconds` (30) clamped at
`provider_retry_backoff_max_seconds` (180). When the budget is spent it
tries the failover target once and then raises. Backoff waits wake on the
turn's cancel event, so a halt from the user ends the retry immediately.

A streaming call retries only before the first content event reaches the
client, because a retry after that would duplicate output the user has
already seen. Between attempts the provider can still say what it is
doing: that status arrives as a `provider_notice` frame, rendered by both
the browser and the terminal client, and it does not count as content.

Two accounting facts an operator tuning failover should know. The trip
counter still counts failed calls rather than attempts, so a call that
recovers on a retry records nothing and the trip latency is bounded by the
retry budget times the threshold. And a failure recorded against the
failover endpoint never advances the counter, so a call that flips back to
the primary mid-retry cannot poison it.

The default budget is visible in the worst case it permits: about an hour
held on one call against a stalling primary (four attempts at
`provider_response_timeout` 900s plus 210s of backoff). Lowering
`provider_retry_max_attempts` or `provider_response_timeout` shortens it,
and the field descriptions in `config/config.py` say so.

## Your `config/config.py` values survive `mira update`

`config/config.py` is where an operator changes MIRA's behaviour, and
`mira update` used to replace it wholesale. Every knob the release notes
recommended setting was therefore unactionable on a deployed install: the
edit survived until the next update and no further.

The updater now reconciles the file instead. `deploy/lib/config_reconcile.py`
performs a three-way merge over the literal `Field(default=...)` values,
keyed by class and field: the release's baseline, your file, and the
release's new file. The release schema wins, meaning fields, validators,
descriptions and structure; your literal values carry forward. A field
that both sides changed keeps your value and logs a warning. A field the
release removed is dropped with a warning, a field it added arrives with
its release default, and defaults computed by a factory stay release-owned.

Container literals carry too, so a customised `cors_origins` allowlist or
`agent_timeout_overrides` map is not silently reverted. The merge is
generated from parsed Python, never by patching text, and a file that does
not parse aborts the update before any code is swapped. The pristine
release copy your edits are measured against is stored at
`data/config_release_baseline.py` and rotates after each successful update.
The first update under this scheme has no such copy yet, so it carries any
value that differs from the release and warns that it cannot tell an
operator edit from a stale release default.

## A background command announces its exit

`bash_tool` run in the background returns before the command finishes and
nothing watched it afterwards. A long command that completed while the
conversation was idle produced no signal anywhere.

The job wrapper now records the shell's exit status next to the log, and a
sweep every 30 seconds turns that record into one message in the owning
conversation, opening a conversation if none is active. The message is one
provider call with no tools and no history, so a completion arriving
mid-task reads as a notice rather than an invitation to resume unrelated
work. A provider outage cannot withhold it: past 5 failed attempts or 600
seconds the notice is delivered from a fixed template built from the job
record. A job whose process is gone with no completion record is dropped
rather than reported.

## Tool calls describe themselves

A tool row used to show raw arguments, so a long piped command read as its
own syntax instead of its intent. The fast route now writes one sentence
describing a call, and the browser swaps it into the settled tool card in
place. The description is display only: it is never stored and never
enters model context, so a replayed conversation shows the raw arguments.
Trivial calls are screened out by argument length and cost no provider
call, and a slow or failed description leaves the raw arguments on screen.
The terminal client is append only and keeps showing arguments.

## Lite memory surfacing for small installs

`MIRA_SUBCORTICAL_ENABLED=0` used to switch memory off entirely: no stored
memory reached the primary model, so long-term memory became write-only on
an install too small to run the fast-route call. Setting
`subcortical_enabled=False` now selects a lite mode that retrieves by
keyword search over the current message plus graph traversal, with no
embeddings, no vector search and no model call. A turn whose analysis
fails lands in the same text-only path instead of surfacing nothing.

The mode that served each turn is recorded with the message as
`memory_surfacing_mode`, one of `normal`, `lite`, `degraded` or `none`, so
an operator can tell which path ran.

## Bash tool corrections

Four reportable defects, all found in the instance journal or a run
census:

- `git reset --hard` was not refused when another flag sat between the
  subcommand and the discard option. A census of 2,255 calls found nine
  such invocations that passed the guard. The validator now parses the
  argument vector, recognizes global options that precede the subcommand,
  and skips the value of `--pathspec-from-file`.
- `cp /dev/null <file>` and `install /dev/null <file>` were refused, which
  blocked emptying a file through the bit bucket. The exemption is scoped
  to the source position and to those two verbs; `/dev/null` as a
  destination, `mv`, and `ln` remain refused.
- A `cwd` or `log_dir` under `/opt/mira/app` was refused by the tool's root
  gate even though the guardrail treats that tree as editable, so the
  app-tree exemption was unreachable.
- The caller-supplied `run` timeout ceiling moved from 1800 to 3601
  seconds. The default stays 120 seconds and background runs are
  unaffected.

Known residues, unchanged: wrapper forms that consume the command word
(`sudo -n git reset --hard`, `env -i git reset --hard`) still pass, as do
abbreviated long options such as `--har`, and eight interpreter-payload
bypasses in the guardrail battery predate this release.

## Also in this release

- Sidebar triggers are a base class with a no-op acknowledgement hook, and
  each trigger's discovery runs inside its own exception boundary. Before,
  the heartbeat trigger raised on every dispatch after starting its agent
  thread, and the trigger registry's order was load-bearing.
- The memory tool's schema names each operation's required arguments in
  both the base schema and the curator override, which removes a class of
  refused calls that each cost a retry.
- Root and directory orientation maps were revised, including a naming
  section and a diagnose-before-you-add principle.

## For upgraders

`mira update`, as always, from v2026.10.16-2.1 or later. Nothing to consent
to, no dependency rebuild, no database steps. Expect the update to report
that it carried your `config/config.py` values forward; if it instead
reports that reconciliation failed, nothing has been changed and the file
needs to parse as Python before the update is re-run.
