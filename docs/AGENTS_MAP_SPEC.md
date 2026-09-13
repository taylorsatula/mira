# AGENTS.md Map Specification (pass working spec)

Canonical shape for every AGENTS.md map in the MIRA repo during the
regeneration pass. The compact version lives in root AGENTS.md; this file
adds the annotated detail. Follow it exactly.

## Scope constraints (every map job)

- Produce EXACTLY the map file(s) assigned to you. Do not modify any other
  file; in particular do not touch ancestor maps. Note in your report what
  the ancestor map should shed when this map lands.
- No test files, no mocks. Read-only exploration except your output file(s).

## Definitions

- Ancestor map of `<dir>/`: the `AGENTS.md` of any directory above it, up to
  repo root. The harness loads the full ancestor chain whenever a file in
  the directory is read, so ancestor maps are in context together with this
  map.
- The `.mdbak` file in your directory (if present) is the previous map,
  retained as a lead source. Read it, verify every claim against source,
  and copy nothing blindly. Facts that still hold are re-expressed in this
  spec's form; facts that fail verification are dropped and reported.

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

## ## Rules — each bullet must satisfy all four requirements

1. Describes a constraint that is not visible from reading one file in this
   directory.
2. Does not restate global doctrine from root AGENTS.md. Ancestor maps are
   additional context, not repetition of root doctrine.
3. Is understandable using only this map and its ancestor maps.
4. Names the enforcing symbols or files in backticks. State the rule's
   content here; use the anchor to locate the enforcement. Do not summarize
   another map's rule — cite the map or symbol that owns it.

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

## ## Files — one bullet per file

Schema: `<name>.py` — what it owns. Entry points (the method/function names
an agent would search for). Non-obvious gotchas, if any. `Consumers:`
trailing field only for files used across directory boundaries; the anchor
may name the ultimate caller through an intermediate layer (state the path).
Frontend maps use `Calls:` as the reciprocal field for backend endpoints and
files called; both fields belong in ## Files only — outside that section,
write the anchor in prose. DOC FILES get: what they cover plus trust status; a stale doc
gets a corresponding staleness Rule in ## Rules. `__init__.py` always gets
an honest one-liner ("docstring only, no re-exports; registration happens in
<anchor>" is real content). Subdirectories get one bullet pointing to their
own map when one exists.

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

## Anchors

Path anchors are repo-root-relative (`clients/llm/types.py`). Bare file
names in ## Files resolve against the map's own directory; that is the
intended form, not an audit failure. Do not cite line numbers; they change
on the next edit of the target file.

## Coverage gate

A directory gets its own map only with >=2 source files, or an invariant not
documented in an ancestor map. Otherwise its content folds into the parent's
## Files section.

## Verification protocol (required; coverage is auditable, not assumed)

1. Run `wc -l` on every source file in the directory before reading.
2. Read every source file in full. If a file is over 1900 lines or over
   45KB, read it in segments (the read tool truncates silently at 2000
   lines / 50KB).
3. If any file ends up only partially read or only grep-verified, list every
   claim resting on it in your report with the grep pattern used.
4. The report must include a per-file read account (full / partial with line
   ranges / grep-only) for every file in the directory. A map with gaps in
   this account is a Partial result.

## Anchor audits (run before finishing)

    cd /Users/taylut/Programming/GitHub/mira-OSS
    for m in <MAP>; do
      d=$(dirname "$m")
      # path anchors (repo-root-relative), from everything EXCEPT ## Files —
      # Files names resolve against the map's directory, including
      # subdirectory-qualified names like agents/base_system.txt:
      awk '/^## Files/{s=1;next} /^## /{s=0} !s' "$m" \
        | grep -ohE '`[^`]*\.(py|txt|sql|md|js|sh|hcl|html|css|json)(:[A-Za-z_][A-Za-z0-9_]*)?`' \
        | sed 's/`//g; s/:[A-Za-z_][A-Za-z0-9_]*$//' | grep / | grep -v '^/' | grep -v '\\*' | grep -v ' ' | sort -u \
        | while read p; do [ -e "$p" ] || echo "MISSING-PATH ($m): $p"; done
      # (absolute paths like /opt/vault/... refer to the deploy host, not the
      #  repo; the audit skips them)
      # bare file names from ## Files only, sub-bullets included:
      awk '/^## Files/{f=1;next} /^## /{f=0} f' "$m" \
        | grep -ohE '^[[:space:]]*- `[^`]*`' | sed 's/^[[:space:]]*- `//; s/`$//' | sort -u \
        | while read p; do [ -e "$d/$p" ] || echo "MISSING-FILE ($m): $p"; done
    done
    # (sed, not `tr -d '`- '` — tr strips all hyphens and corrupts names
    #  like api-client.js)
    # Extensions: sh is included — shell scripts are first-class anchors.
    # Do not cite maps that do not exist yet; put planned parent changes in
    # the report's parent-migration notes instead.

## Maintenance trigger table (mirrors root AGENTS.md)

Map updates are part of the same commit as the triggering change. Every
event has a deterministic action:

| Event | Required action |
|---|---|
| New file in a mapped directory | Add its ## Files bullet |
| Deleted or renamed file | Update or remove its ## Files bullet; fix any anchor citing the old path |
| Behavior or contract change in a file | Grep that directory's map for the changed symbol; update the Rule, Files gotcha, or Wiring edge stating the old behavior |
| New member of a registry, enum, or contract family | Update the blast-radius Rule in the owning map |
| Directory reaches >=2 source files, or gains an invariant its parent's map does not own | Create its AGENTS.md (shape above) |
| File moved between directories | Update both maps' ## Files sections |
| New map created or removed | Update the parent map's ## Files pointer |

When in doubt, update the map. The anchor audits are the check that no
trigger was missed: a clean audit means no map action was skipped; a
flagged anchor is a skipped trigger. For changes the audit cannot express,
the grep-the-map step is the check.

## Voice calibration exemplars

`working_memory/AGENTS.md` (pre-flush: `working_memory/AGENTS.mdbak`),
`tools/implementations/AGENTS.md` (pre-flush: `.mdbak`),
`working_memory/trinkets/AGENTS.md` (new, batch 1). Rules read as direct
instructions and contracts. State what to do and what not to do; omit
rhetorical framing.

## Report format (every job)

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
