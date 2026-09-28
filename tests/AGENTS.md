# tests/ — verification-artifact homes, not a suite

Three child directories with three lifetimes: a disposable display exhibit, shared
real-infrastructure scaffolding, and an admission-gated permanent battery. Each
owns its own admission and cleanup policy; none is a test suite.

## Rules

- No test suite forms here. `tests/tmp/` autodeletes test files on sight, `tests/protected/` admits a file only on the exact phrase `AUTHORIZE PROTECTED TEST SAVE`, and `tests/fixtures/` carries only claim-free scaffolding. A directory added under `tests/` must state which of the three it is and own its policy.
- A child's admission, cleanup, and realism rules are owned by that child's map; this map only routes. Read the child map before adding, editing, or deleting anything in it — the `tests/tmp/` autodelete reflex does not apply to `tests/protected/`.

## Files

- `fixtures/` — shared probe scaffolding: real-infrastructure setup/teardown, claim-free, no simulation; see `fixtures/AGENTS.md`.
- `protected/` — admission-gated permanent offline verification batteries; see `protected/AGENTS.md`.
- `tmp/` — disposable-probe display exhibit, autodeleted on sight; see `tmp/AGENTS.md`.
