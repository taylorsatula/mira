# Mapping Pass Jumpstart (one-off)

Kickoff brief for the agent swarm that performs a project's **first mapping
pass** under `AGENTS_MAP_SPEC.md`. Read it once at pass start; the spec is
the standing authority and this file is not — once the pass closes, this
document is a historical record with no further role.

Distilled from the 2026-09 MIRA-OSS pass: the shape-spec regeneration
(`4c85255`), the adversarial cross-verify pass it turned out to require
(`79ae2e4`), and the growth-gate pass (`f1f0853`), plus a hindsight review
of the mature corpus. Every recommendation below was either exercised there
or paid for there.

## What the source pass cost, so this one doesn't

- The first pass regenerated all maps from per-file full reads — and still
  shipped drifted wiring claims, because editors wrote from remembered or
  excerpted code. Correcting them took an entire second pass across all 28
  maps. **Verifiers belong inside the pass, not after it**: each map's
  claims are re-derived by a read-only agent that did not write them,
  before the wave merges.
- Stale-claim density in the pre-pass corpus (16) exceeded latent-code-bug
  density (4). Paraphrase decays faster than code; cite grep-able anchors,
  never paraphrase another map's contract.

## Swarm shape (validated)

A funnel, one wave per batch of directories:

1. **Editors** — one agent per map, restricted to producing exactly its own
   map file, dispatched bottom-up (children before parents, so parents shed
   migrated detail in the same changeset). The full-authoring verification
   protocol applies: `wc -l` first, every file read in full, segmented past
   truncation limits, per-file read accounting in the report.
2. **Verifiers** — read-only, one per map, adversarially re-deriving every
   Rule/Files/Wiring claim from full source reads. They do not edit.
3. **Arbiters** — edit authority for disputes only, settling each one with
   line-level evidence; no editor or verifier self-certifies a correction.
4. **Merge + audits** — run the spec's full audit battery (path anchors,
   Files bullets, line budget, citations, density) per wave before merging;
   a flagged anchor is a skipped trigger, not a style note.
5. **Assembly** — build the root map's map index from the merged corpus;
   select voice exemplars from the pass's own output into the spec's
   exemplar slot; apply editor parent-migration notes to the ancestor maps.

## Verifier briefing: the failure classes the source pass caught

Hunt for these by name; each had real instances:

- **False universals** — a true insight wrapped in a false universal
  ("the only surface", "exactly two"). Re-derive every count and scope
  claim from source.
- **False mechanisms** — a real constraint explained by an invented
  mechanism (an imagined import cycle) when the real reason was mundane
  (deferring heavy client initialization). Verify the *why*, not just the
  *what*.
- **Invisible contracts** — a cross-directory constraint documented in a
  map outside the ancestor chain of the file it binds. A claim is
  load-bearing only where its reader actually loads it.
- **Paraphrase drift** — a contract restated where it should be cited, so
  two co-loading maps can contradict each other (the source pass found two
  maps asserting different slot orderings for one contract).
- **One-sided references** — citing a root-level file the root map does not
  own (roughly 24 such edges were closed in the source pass).
- **Silent-drift contracts** — display prose, status vocabularies, schema
  domains, event names: none has a crash dialect, so the map layer is their
  only regression guard. Verify these against source character by
  character.

## Byproduct: latent bugs

Reading everything surfaces latent code defects — the source pass found six
(an unassigned attribute, a dead event target, a dead result contract,
unwired constants contradicting documented ports, a CLI flag no script
accepts, dead assets). Expect this. Policy: record every suspected defect in
the editor/verifier report with file, symbol, and symptom — do not fix it
mid-pass, and do not let the map editorialize around it (a latent bug
belongs in the owning bullet as a gotcha, per the spec).

## Gates that ended arguments in the source pass

- Requirement 5 (earns its tokens) is a deletion criterion during writing,
  not a later cleanup: if a competent agent reading the source would reach
  the same decision without the bullet, it is not written.
- Line budgets are enforced at merge time (60 target / 80 ceiling; root
  map exempt); density flags are recorded as debt for the first maintenance
  pass, not treated as blocking.
- Every full-authoring job reports: status, line counts, unverified claims,
  per-file read accounting, friction log, spec revisions, and
  parent-migration notes. Gaps in read accounting downgrade the job to
  Partial — treat a Partial as unfinished work, not a warning.

## After the pass

The maintenance trigger table in the spec takes over. This file carries no
standing authority, and no maintenance trigger fires on it — it exists so
the next project's pass starts from evidence instead of paying for the
lessons again.
