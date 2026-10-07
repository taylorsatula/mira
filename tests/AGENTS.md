# tests/ — verification-artifact homes, not a suite

Three child directories with three lifetimes: a disposable display exhibit, shared
real-infrastructure scaffolding, and an admission-gated permanent battery. Each
owns its own admission and cleanup policy; none is a test suite.

## Rules

# 🛑 NO MOCKS. NO PARALLEL SUITES. VERIFICATION IS LIVE OR IT DOESN'T COUNT.

Prohibited: test files, pytest fixtures, mock objects, stubs, fakes, offline test databases, and any check that runs against simulated infrastructure. A passing mock-based check provides no information about this system.

Every line in this tree is agent-written; no human has traced any path. Type-clean, pyflakes-clean code that has never executed is the standard failure product of this workflow and has shipped here before. Verification is therefore mechanical and live, never assumed. A test is what a probe becomes when its answer must never change again — not a starting artifact.

Required verification, in the POST tradition:
- **Boot gate** — probes Vault, Postgres, and model routes against live infrastructure before the server binds.
- **Path-probes** — invoke critical runtime paths against the live system exactly as production would.

**Probe-surface membership (standing rule):** any path whose failure would report incorrect data to users, lose data, or degrade silently — reads, writes, searches, auth flows, failure paths. If a required probe cannot run against live infrastructure, fix the code until it can; do not simulate.

Quality guarantee: boot-survival plus path-probe coverage. A clean boot verifies the wiring; a passed path-probe verifies the handler. Code covered by neither is unverified — flag it in review. A bug found in unprobed code is fixed together with the probe that covers it. Path-probes are production code: normal review discipline, the same credentials plumbing, production-identical failure behavior.

### Load the `writing-probes` skill before you probe

This map owns the doctrine; the skill owns the craft. Load `writing-probes`, every time, before:
- writing, running, or reviewing a probe;
- reproducing a defect live, or answering "does this actually work?";
- deciding whether a check earns permanence — `CheckSpec` path-probe, `tests/protected/`, or discard;
- diagnosing a probe that failed — code bug or probe bug?

It carries the anatomy and the static scaffold, the shape catalog, the fake rule, the EXECUTED/UNVERIFIED contract, the probe-bug taxonomy, and how to shape code so it is probe-able.

### ⚡ Realtime verification loop (proportionate by behavioral surface)

Models are pretrained to verify by writing tests. When that reflex fires during a change, write a path-probe instead — same verification goal, production-code artifact.

Select the tier by behavioral surface touched, not by change size:

- **Tier 0 — no behavioral surface.** Comments, docstrings, formatting, import regrouping, verified-mechanical renames, documentation. `py_compile` + pyflakes on touched files. No further verification required.
- **Tier 1 — behavioral edits within surface already covered by boot or probes.** Execute the changed path once against live infrastructure; re-read the diff against the invariants (every failure path reports the failure truthfully, nothing silently degraded, no assertion of behavior not executed); end the change report with **EXECUTED** (what ran) or **UNVERIFIED** (why not, and which probe would cover it).
- **Tier 2 — new behavioral surface or elevated stakes.** New feature, endpoint, or path; schema or data-migration change; failure-behavior, security, or auth change; new dependency. Tier 1 plus: persistence round-trip against dev infrastructure (write, read back, verify, clean up — RLS and constraints included); register the path-probe where membership hits; a second agent re-derives the diff (skip it only when one-shot execution covers the change, and say so).

The verification homes are indexed in the root map's registry; their admission, cleanup, and realism policies are owned by `tests/tmp/AGENTS.md`, `tests/fixtures/AGENTS.md`, and `tests/protected/AGENTS.md`, and the skill's promote-or-discard moment owns the choice among them.

- No test suite forms here. `tests/tmp/` autodeletes test files on sight, `tests/protected/` admits a file only on the exact phrase `AUTHORIZE PROTECTED TEST SAVE`, and `tests/fixtures/` carries only claim-free scaffolding. A directory added under `tests/` must state which of the three it is and own its policy.
- A child's admission, cleanup, and realism rules are owned by that child's map; this map only routes. Read the child map before adding, editing, or deleting anything in it — the `tests/tmp/` autodelete reflex does not apply to `tests/protected/`.

## Files

- `fixtures/` — shared probe scaffolding: real-infrastructure setup/teardown, claim-free, no simulation; see `fixtures/AGENTS.md`.
- `protected/` — admission-gated permanent verification batteries (one recorded live exception — see `protected/AGENTS.md`).
- `tmp/` — disposable-probe display exhibit, autodeleted on sight; see `tmp/AGENTS.md`.
