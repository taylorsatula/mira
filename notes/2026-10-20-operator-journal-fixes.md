# 2026-10-20 — operator journal fixes: /dev/null guardrail, app-tree cwd, memory_tool miscalls

Session asked to fix three items surfaced by a `journalctl -u mira` review
(2026-10-14 → 2026-10-20). Branch `fix/operator-journal-issues` (off `main`
@ bf20852), worked in worktree `.worktrees/operator-journal-fixes`. No commit
yet per instruction. Deployed instance `/opt/mira/app` NOT touched.

## Worktree collision (process note)

The scratch clone `~/Programming/Scratch/mira` was held by another live session
(HEAD moved to `heartbeat-every-hours` @ 5506928, committed 11:45 today, while
this session worked). This session created branch + worktree
`.worktrees/operator-journal-fixes` with lock file, copied its two modified
files in, and restored the shared checkout's files to `heartbeat-every-hours`
HEAD before leaving. Lesson: claim/create the worktree FIRST, before editing
files in a shared checkout.

## Fix 1 — bash_tool /dev/null blocks

Journal correction, recorded because the earlier chat summary got it wrong:
the Oct 16 blocks (`rm /dev/null`, `perl /dev/null` [in-place edit], `cp
/dev/null`) are NOT all false positives. `rm /dev/null` (pattern layer + argument
layer) and an in-place edit of the node are correctly refused — deleting or
rewriting the node breaks every process that redirects to it. Redirects
(`> /dev/null`) were already exempted in `_check_redirect_targets`. The one real
false positive: `/dev/null` as a COPY SOURCE (`cp /dev/null target`, `install
/dev/null target`) — the idiomatic truncate — was refused because `_WRITE_VERBS`
checks every operand identically.

Change: `_check_path_verbs` WRITE branch now exempts a `/dev/null` source
operand for `cp`/`install` only (mv excluded — moving consumes the source;
ln excluded — out of scope). Destination detection via new `_copy_destination`
(last operand, or the `-t`/`--target-directory[=]` value; with `-t` every path
operand is a source). The destination operand keeps its full check in all
forms, so the exemption never relaxes where the copy lands.

Battery: added Z76/Z77/Z78 (benign, allowed) and P29/P30 (both-modes refuse:
`cp /tmp/evil /dev/null`, `cp -t /etc /dev/null`) to
`tests/protected/bash_guardrail_probe.py`. Run with
`/opt/mira/app/venv/bin/python3 tests/protected/bash_guardrail_probe.py`:
201 destructive / 193 blocked / 8 BYPASSED — the 8 are the documented
pre-existing N01–N08 interpreter-payload red (see
notes/2026-10-11-bash-restart-mira-param.md:81, same numbers), benign 78/78
with 0 false positives, skip 95/95 + 29/29, blocklist 8/8, both-modes 44/44.
Exit 1 comes from the known N-red only.

## Fix 2 — `_resolve` root gate vs the app tree

Root cause of the "Path '/opt/mira/app' resolves to ... outside the permitted
root '/home/taylor'" refusals: `_resolve` (bash_tool, the tool-parameter gate
for `cwd`/`log_dir`) confined everything to `cfg.root`, while the guardrail's
app-tree machinery (`_classify_app_tree_path`, git-as-undo) already intends the
code tree to be editable. The two gates contradicted: the app-tree exemption
was unreachable because the root gate refused the tree first. (The deployed
install IS a git repo — `/opt/mira/app/.git/info/exclude` written by
`deploy/lib/state_preserve.sh:write_install_exclude`; an earlier check against
`/opt/mira` instead of `/opt/mira/app` wrongly reported "not a git repository".)

Change: `_resolve` now accepts paths at/under `_APP_TREE[0]` when `_APP_TREE`
is active. Destructive actions inside the tree stay governed by the guardrail
(tracked = editable, untracked exclude names = protected). `_APP_TREE=None`
(non-git install) keeps the single-root rule exactly as before.

Verified by direct snippet (monkeypatched `_APP_TREE` to the deployed layout):
tree root/deep path/untracked-name cwd allowed; `/opt/mira`, `/opt/mira/appx`,
`/etc/passwd` refused; `_APP_TREE=None` restores old behavior. Worktree
limitation, by design: a worktree's `.git` is a gitfile, so `_load_app_tree`
returns None there — the exemption is inert in worktrees, active in real
installs.

## Fix 5 — memory_tool miscalls (diagnosis + fix)

Failure classes from the journal, all on the `memory_curator` agent (route
`batch` = `glm-5.3-flash-background`, non-streaming):

1. `link_memories` missing `link_type` — 8× (2026-10-19 03:26–03:31), incl. a
   burst of 7 tool calls in one assistant turn, all missing the field.
2. `merge_memories` missing `consolidated_text` — 1×.
3. Provider-level: tool call emitted with NO `operation` at all — 4×
   (2026-10-17 18:02, 2026-10-19 03:24); mira's dialect validation
   (`openai_chat_base.py:_coerce_tool_input`) rejects it pre-tool, converted to
   an agent-recoverable `ToolResult(is_error)`. Loop recovers; warnings are the
   noise, not data loss.

Root cause: `memory_tool.tool_schema` sets `required: ["operation"]` only —
per-operation requireds are enforced by the `_OPERATION_REQUIRED_FIELDS`
presence gate in `run()` because conditional requirements are not portable
across providers (recorded constraint in the schema comment). The model
therefore has to infer each operation's field set from property descriptions —
and two of the three link fields it actually dropped (`source_memory_id`,
`link_type`) had NO "Required for" tag in their descriptions, while every field
the model never dropped (target_memory_id, reasoning, memory_ids, ...) carries
the tag. `consolidated_text` has the tag and was still dropped once — that
residue is flash-model flakiness, not a mira defect.

Fix: added "Required for 'link_memories'" to `source_memory_id` and `link_type`
descriptions, and put an explicit per-operation required-arguments mapping into
BOTH the base `tool_schema` description and `CURATOR_MEMORY_SCHEMA`'s
description (the curator's override replaces the whole schema dict, so the base
mapping alone would not reach the curator's model). Verified: jsonschema
Draft202012 behavior unchanged (valid call valid, op-less call still invalid at
the dialect layer), descriptions reach the provider verbatim via
`_convert_tools` (`parameters: dict(tool.input_schema)`).

Not mira defects, left alone: `Target memory 'dd7ba782-0000-0000-...-0000' not
found` — the model fabricated a zero-padded UUID; blank-assistant-response
rejections and the 1500s bash kills are guards working as designed.
`SegmentTimeoutService` max-instances skips are the segment-timeout-stuck-thread
problem — the OTHER session's untracked `SEGMENT_TIMEOUT_DEFERRAL_PLAN.md` in
the shared checkout is already aimed there; not touched here.

## Deployment state

`model_configs` primary switched glm-5.3 → glm-5.3-flash in the DB this
session (needs mira restart to load; not restarted). The three code fixes live
only in the worktree until committed, merged, and deployed — the running
service still has all three behaviors.
