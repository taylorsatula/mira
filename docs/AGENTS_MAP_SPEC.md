# AGENTS.md Map Specification

Canonical shape for every AGENTS.md map in this repository. The compact
version lives in the root `AGENTS.md` map, which also carries the map index
this spec's trigger table references; this file adds the annotated detail and
is the sole authority. Follow it exactly.

The spec serves two regimes: **maintenance edits** — the default case, what
the trigger table fires — and **full (re)authoring jobs** (a first mapping
pass, a new map, regeneration of one, consolidation pass).

Voice: maps are model-read text — read the host repository's writing
register before writing or editing one.

## Scope constraints (every map job)

- Touch EXACTLY the map file(s) the change assigns to you; no other files, and
  never an ancestor map's content. For a full-authoring job, note in the
  report what the ancestor map should shed when this map lands.
- Map maintenance rides the triggering code commit, never a separate docs
  commit (see the trigger table).
- No test files, no mocks. Read-only exploration except your output file(s).

## Definitions

- Ancestor map of `<dir>/`: the `AGENTS.md` of any directory above it, up to
  repo root. The harness loads the full ancestor chain whenever a file in
  the directory is read, so ancestor maps are in context together with this
  map.

## Section order (fixed)

    # <dir>/ — <role, one clause, no terminal punctuation>
    [orientation: 0-2 sentences, only when the identity clause cannot carry
     runtime context]
    ## Rules
    ## Files
    ## Wiring          <- omit entirely if the directory has no cross-file
                          edges; do not pad
    ## <Deep-dive titled with a domain term>   <- optional, max one

Use bullets. Prose paragraphs are allowed only inside `###` subsections
(titled with a domain term, for contract chains that do not compress into a
bullet) and inside the deep-dive; `###` subsections may live inside ## Rules
and do not consume the deep-dive slot. Tables are allowed in the deep-dive
for matrix-shaped content.

## ## Rules — each bullet must satisfy all five requirements

1. Describes a constraint that is not visible from reading one file in this
   directory.
2. Does not restate global doctrine from root AGENTS.md. Ancestor maps are
   additional context, not repetition of root doctrine.
3. Is understandable using only this map and its ancestor maps.
4. Names the enforcing symbols or files in backticks. State the rule's
   content here; use the anchor to locate the enforcement. Do not summarize
   another map's rule — cite the map or symbol that owns it.
5. Earns its tokens: deleting the bullet must change agent behavior in a
   foreseeable case. If a competent agent reading the source would reach the
   same decision without the bullet, delete it instead of writing it.

## Cross-map rules

- Each fact is documented in exactly one of the maps that load together. A
  constraint that applies at both parent and child level may appear in both;
  the non-owning map states it in one line and cites the owning map.
- Adding a member to a registry, enum, or contract family requires
  coordinated edits in several files. Document the full list once, in order,
  with the file anchor and the consequence of skipping each step.
- `## Wiring` documents edges where this directory is an endpoint, plus
  ordering constraints within this directory (what flows, in what order,
  what breaks if reordered). Data flows owned by ancestor code are cited by
  owner, not restated.
- The citation formula is `owned by `<dir>/AGENTS.md` (`owning symbol`)` —
  one line in the non-owning map, owning statement in the owner. Variants in
  the corpus (`cited, not restated`, `see `<dir>/AGENTS.md``, `Map:
  `<dir>/AGENTS.md``) are equivalent; pick one per map and stay consistent.
  Every map-path citation must resolve to a map that exists (audit below).

## Recurring patterns (extend the existing instance, never a new form)

These shapes converged across the corpus and proved their worth in use. A map
update that touches one of them extends the existing instance rather than
inventing a parallel one.

- **Add-one recipe.** Whenever the directory is an extension point (a
  registry, plugin, handler, command, config flag — whatever the project's
  extension points are), the map carries the ordered checklist for adding one
  member: every file to touch, each step anchored, and the concrete failure
  each skipped step produces ("fails at startup", "silently disappears from
  the list callers see", "the new member cannot be constructed"). The map
  owning the registration mechanism owns the recipe; child maps carry only
  child-specific steps and cite it.
- **Consequence clause.** A constraint is half stated until its violation is:
  "a caller that skips the token release leaves the lock held until TTL
  expiry." Prefer the concrete failure over "may cause issues".
- **Twin contracts.** Some contracts exist in two files by necessity (an auth
  check duplicated where the framework cannot share it across transports, a
  client renderer and its terminal mirror, a base class and its authoring
  guide). The map names both ends, requires the same-commit change, and
  states the drift symptom ("one accepts identities the other rejects").
  Never paper over a twin with an abstraction suggestion — the duplication
  is the design.
- **Two-endpoint protocols.** A protocol spoken between two directories
  (WebSocket frames, request envelopes) is documented once per side: each
  side's map owns its side's frames and semantics and cites the counterpart
  for the other. This is the sanctioned exception to the one-home rule.
- **Named flows.** In `## Wiring`, each flow gets a label, an arrow chain
  (`A → B → C`), the ordering invariants, and what breaks if reordered.
  Ordering-sensitive initialization belongs here even when the code is a
  plain sequence of calls.
- **Gated wiring.** When construction is conditional on a config flag, the
  map states the gate, the construction site, and the consumers' duty to
  tolerate absence.
- **Verified-on stamps.** Sparse, and only where the date itself is the fact
  (a flow exercised end-to-end on a date, a failure mode learned the hard way
  on a date); never as change history.
- **Reverse-edge sweep.** `Consumers:`/Wiring sets are written from a grep, not
  recall: enumerate the module's importers, published/subscribed event names, and
  public symbols repo-wide, then write the set from the sweep output. Recall omits
  exactly the edges that matter — an unrepresented concurrent writer is invisible
  to a claim-by-claim read of the map. (Observed 2026-10-06: in the two-wave sweep,
  every missing-wiring finding came from the code→map direction; none from rereading
  the map.)

## ## Files — one bullet per file

Schema: `<name>.py` — what it owns. Entry points (the method/function names
an agent would search for). Non-obvious gotchas, if any. `Consumers:`
trailing field only for files used across directory boundaries; the anchor
may name the ultimate caller through an intermediate layer (state the path).
Frontend maps use `Calls:` as the reciprocal field for backend endpoints and
files called; both fields belong in ## Files only — outside that section,
write the anchor in prose. DOC FILES get: what they cover plus trust status; a stale doc
gets a staleness Rule in ## Rules as **debt to be repaid** — fix the doc or
remove it; staleness Rules are not permanent furniture. `__init__.py` always gets
an honest one-liner ("docstring only, no re-exports; registration happens in
<anchor>" is real content). Subdirectories get one bullet pointing to their
own map when one exists.

Density: a Files bullet is an invariant bundle; the corpus convention runs to
about 400–500 characters. Past ~1200 characters a bullet has become a
sub-article — break its sub-facts into sub-bullets or promote the
generalizing invariant into ## Rules. Line budgets measure the map, not the
bullet; a 60-line map of oversized bullets fails the intent while passing
the audit (density audit below).

Dead and vestigial code gets its own convention, because a future session's
default reflex is to "fix" it: state that it is dead, why it is kept, what
would re-activate it, and "do not build on it." An orphaned module no page
loads, an endpoint no caller reaches, an always-empty hook, a dangling
reference to an absent file — each gets this treatment in its owning bullet
or a Rule when the deadness is intentional; if nothing keeps it, delete the
code instead.

Dates: a validated-on or learned-on date is allowed when the date itself is
the fact; change history ("changed from X to Y on <date>") is not — that
belongs in the commit message.

## Deep-dive

Include only when deriving the content from source requires reading many
files or reversing a non-obvious design (state machine, protocol,
translation matrix, lifecycle ordering). Title with a domain term, never
"Notes" / "Details" / "Misc".

## Do not include

Restatements of root AGENTS.md global doctrine; change history ("we used to
X"); content derivable from a file's own docstring or signature; unspecific
summaries; filler to lengthen a map. (Root itself may briefly restate
cross-cutting patterns that maps also carry — the one-home rule applies
between maps, not between root and a map.)


Latent runtime bugs (undefined attributes, dead code paths, unreachable
branches) belong in the owning file's ## Files bullet as a gotcha, with the
verified symptom and the working alternative if one exists. If the bug
affects how code must be written across the directory, it becomes a Rule
instead.

Unqualified universals — `all`, `only`, `every`, `never`, `nothing else` —
that do not survive a repo-wide grep of the named symbol or property. The
highest-yield inaccuracy class (observed 2026-10-06): enumerations drift as
code grows — "all routers registered" missing the router added last month,
"the only DELETE trigger" missing the failure-branch delete, "every payload"
missing the one writing a different key. Name the exception list and keep it
current, or drop the quantifier.

## Anchors

Path anchors are repo-root-relative (`clients/llm/types.py`). Bare file
names in ## Files resolve against the map's own directory; that is the
intended form, not an audit failure. Do not cite line numbers; they change
on the next edit of the target file.

## Coverage gate

A directory gets its own map only with >=2 source files, or an invariant not
documented in an ancestor map. Otherwise its content folds into the parent's
## Files section.

**First mapping pass** (a project with no maps yet): map every directory
meeting the coverage gate; every job is a full-authoring job and follows the
verification protocol and report format in full; establish the root map's
map index as part of the pass; fill the voice-exemplar slot below from the
pass's own output.

## Verification protocol

Two regimes. A **maintenance edit** (the trigger table firing) verifies only
what it touches: grep the map for the changed symbols, read the current
source of every statement being updated or deleted, confirm each surviving
claim against it, and run the anchor audits below. A full re-read of the
directory is not required and slows the same-commit obligation.

A **full (re)authoring job** — new map, regeneration of one, consolidation
pass — must re-derive every claim from source; coverage is auditable, not
assumed:

1. Run `wc -l` on every source file in the directory before reading.
2. Read every source file in full. If a file is over 1900 lines or over
   45KB, read it in segments (the read tool truncates silently at 2000
   lines / 50KB).
3. If any file ends up only partially read or only grep-verified, list every
   claim resting on it in your report with the grep pattern used.
4. The report must include a per-file read account (full / partial with line
   ranges / grep-only) for every file in the directory. A map with gaps in
   this account is a Partial result.

**Two-wave refinement sweep** (established 2026-10-06, corpus-wide).
Wave 1, refinement: one agent per map, primed by mission only — refinement,
not replacement; null result permitted — reading every source file in its
directory end-to-end, editing only its own map. Wave 2, blind verification:
read-only agents with no knowledge of wave 1 verify every claim
section-by-section and trace wiring BOTH directions — map→code (each claimed
edge traced to its endpoint) and code→map (the reverse-edge sweep).
Reconciliation is single-writer: re-check each finding's quoted map text
against the current file before applying (verifier quotes can go stale),
apply factual corrections and wiring an agent would act on, record marginalia
as reported-not-applied.

## Anchor audits (run before finishing)

    cd <repo-root>
    for m in <MAP>; do
      d=$(dirname "$m")
      # path anchors (repo-root-relative), from everything EXCEPT ## Files —
      # Files names resolve against the map's directory, including
      # subdirectory-qualified names like assets/js/util.js:
      awk '/^## Files/{s=1;next} /^## /{s=0} !s' "$m" \
        | grep -ohE '`[^`]*\.(py|txt|sql|md|js|sh|hcl|html|css|json)(:[A-Za-z_][A-Za-z0-9_]*)?`' \
        | sed 's/`//g; s/:[A-Za-z_][A-Za-z0-9_]*$//' | grep / | grep -v '^/' | grep -v '\\*' | grep -v ' ' | sort -u \
        | while read p; do [ -e "$p" ] || echo "MISSING-PATH ($m): $p"; done
      # (absolute paths are out-of-tree references; the audit skips them)
      # bare file names from ## Files only, sub-bullets included:
      awk '/^## Files/{f=1;next} /^## /{f=0} f' "$m" \
        | grep -ohE '^[[:space:]]*- `[^`]*`' | sed 's/^[[:space:]]*- `//; s/`$//' | sort -u \
        | while read p; do [ -e "$d/$p" ] || echo "MISSING-FILE ($m): $p"; done
    done
    # budget audit: over-target obligates consolidation this commit;
    # over-ceiling is a defect (root AGENTS.md is exempt — the doctrine layer,
    # pruned separately)
    for m in <MAP>; do
      case "$m" in AGENTS.md) continue;; esac
      n=$(wc -l < "$m")
      if [ "$n" -gt 80 ]; then echo "OVER-CEILING ($m): $n lines"
      elif [ "$n" -gt 60 ]; then echo "OVER-TARGET ($m): $n lines"
      fi
    done
    # citation audit: every cited map path must resolve to an existing map
    # (relative citations resolve against the citing map's directory;
    # absolute paths are out-of-tree references and are skipped)
    for m in <MAP>; do
      d=$(dirname "$m")
      grep -ohE '`[A-Za-z0-9_./-]*AGENTS\.md`' "$m" | tr -d '`' | grep -v '^/' | sort -u \
        | while read p; do [ -e "$d/$p" ] || [ -e "$p" ] || echo "MISSING-MAP ($m): $p"; done
    done
    # density audit: a ## Files bullet over ~1200 characters has become a
    # sub-article (see ## Files — density); flags are debt, same as
    # OVER-TARGET, not necessarily this-commit defects
    for m in <MAP>; do
      awk -v m="$m" '/^## Files/{f=1;next} /^## /{f=0} f && /^[[:space:]]*- `/ && length($0) > 1200 {print "LONG-BULLET (" m "): line " NR}' "$m"
    done
    # (sed, not `tr -d '`- '` — tr strips all hyphens and corrupts hyphenated
    #  names)
    # Extensions: sh is included — shell scripts are first-class anchors.
    # Tune the extension list to the host project's file types.
    # Do not cite maps that do not exist yet; put planned parent changes in
    # the report's parent-migration notes instead.

## Maintenance trigger table (full version; root AGENTS.md mirrors the high-frequency triggers and points here)

Map updates are part of the same commit as the triggering change. Every
event has a deterministic action:

| Event | Required action |
|---|---|
| New file in a mapped directory | Add its ## Files bullet |
| Deleted or renamed file | Update or remove its ## Files bullet; fix any anchor citing the old path |
| Behavior or contract change in a file | Grep that directory's map for the changed symbol; update the Rule, Files gotcha, or Wiring edge stating the old behavior — or **delete it if the behavior is gone**; never soften or annotate a statement of dead behavior |
| Behavior removed from a file, or a Rule/gotcha whose enforcing anchor no longer exists | Delete the statement in the same commit; grep all maps for its symbols and delete every citation. A Rule citing a dead anchor is a defect on par with a missing Rule |
| Code becomes dead, orphaned, or unwired | In the same commit, either delete it or document it as dead in the owning bullet — dead, why kept, what re-activates it, do-not-build-on-it (see ## Files) |
| New member of a registry, enum, or contract family | Update the blast-radius Rule or the add-one recipe in the owning map |
| Directory reaches >=2 source files, or gains an invariant its parent's map does not own | Create its AGENTS.md (shape above) |
| File moved between directories | Update both maps' ## Files sections |
| New or removed cross-directory edge (import, call site, event publish/subscribe, registration site) | Update the callee's map `Consumers:`/Wiring in the same commit; the caller's map cites it when load-bearing |
| New map created or removed | Update the parent map's ## Files pointer and the root map's registry table |
| A map crosses 60 lines | Consolidation pass in the same commit (merge bullets, cite owners, delete decoration); at 80, split with subdirectory maps instead |

When in doubt, update the map. Additions and deletions are symmetric: every
deletion is recoverable from git history, while every unnecessary line costs
a slice of every future session's attention, forever — bloat is the expensive
option, deletion the cheap one. The anchor audits are the check that no
trigger was missed: a clean audit means no map action was skipped; a flagged
anchor is a skipped trigger. For changes the audit cannot express, the
grep-the-map step is the check.

## Voice calibration exemplars

Exemplars are local, not portable — name them here at the end of the first
mapping pass so later jobs copy a local instance instead of re-deriving the
register. Select one map per shape: rules as direct instructions and
contracts; named flows and a lifecycle deep-dive; an add-one recipe with
per-step skip-consequences; a dead-code inventory. Rules read as direct
instructions and contracts. State what to do and what not to do; omit
rhetorical framing.

## Report format (full-authoring jobs)

A maintenance edit reports inline in its change report: which statements were
updated or deleted, and the audit result. Full-authoring jobs return:

1. STATUS: Done / Partial / Blocked (Partial if read accounting has gaps).
2. Line count and section inventory per map produced.
3. Unverified claims, explicitly listed.
4. Per-file read accounting (the gate for Done vs Partial).
5. Friction log: where the spec was ambiguous, contradictory, or forced
   workarounds; which requirements rejected valuable content or admitted
   noise.
6. Recommended spec revisions.
7. Parent-migration notes: what the ancestor map should shed when this map
   lands.
