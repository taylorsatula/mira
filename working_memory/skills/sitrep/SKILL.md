---
name: sitrep
description: Session status report. Use when the user asks for a sitrep, recap, brief, status, or summary of the session's changes or where things stand. Runs against the session's diff, articulating what was changed, where, why, and whether each change follows codebase-native patterns or breaks uncharted ground — with the design decisions and uncharted territory up front — so the developer can sign off or redirect from the report alone.
---

# Sitrep

## What a sitrep is

Agents produce a lot of code across many files in one session. The developer's options are to read every line or to trust that the model made good design choices. A sitrep is the deliberate middle that leans hard toward the first option: a report on the diff that stands in for the line-by-line read.

Imagine the founder who wrote the codebase sitting across the table and you sliding the report over. They know this code better than anyone. They want to know what changed, where, why, and whether the changes follow codebase-native patterns or break uncharted ground. The sitrep is the moment agent-produced work crosses into the founder's codebase: reading it is the review, and the sign-off is the adoption.

## When to use

- The user asks for a summary of the session: "sitrep", "recap", "brief me", "where are we", "summarize the changes", "what did we do".

## Gather — never narrate from memory

1. `git log --oneline` for the session's commits (since session start or the last sitrep).
2. `git diff --cached` and `git diff` — read the actual diffs, in full.
3. `git status --short` — separate session work from pre-existing unrelated changes.
4. The session conversation: explicit decisions, rejected alternatives, and any point where the user overruled the agent.

The diff is the source of truth for *what* and *where*. The conversation is the source for *why*. Memory of a change set drifts over a long session — reread both.

## Rules

- Write to a competent programmer who knows this codebase. No definitional scaffolding, no jargon translation.
- No preamble, no celebration, no filler, no invented next steps.
- **Do not guess or infer.** Every claim about the code is traceable to a diff hunk. Every "why" comes from an explicit decision, requirement, or correction in the session. When a design choice is not grounded in either, say so — label it as inference rather than presenting it as rationale.
- Length follows the work. A mechanical change gets one line; a design choice gets as many sentences as its reasoning deserves. Cut padding, never substance.
- When the user overruled a decision during the session, record the final state as fact — one line, no re-litigation.

## Structure

1. **What** — one to two sentences: what now exists and the behavior it changes.
2. **Decisions and uncharted ground** — the review core, up front. One entry per design-level choice: what was chosen, the real rejected alternative, the why, and the pattern fit — native (naming the precedent) or uncharted (stating what it establishes). This is where the founder's attention goes; mechanical changes do not belong here.
3. **Change inventory** — the complete accounting. One entry per change, grouped by file or subsystem: what and where; for design-level changes, a pointer to section 2 instead of a restatement; for mechanical changes, one line plus the native-fit note. Every hunk of the session's diff is accounted for; nothing is silently dropped.
4. **Findings** — real issues found while verifying, severity-ordered, each with its smallest fix: producer/consumer shape drift, missing version bumps, dead code, convention violations. If none were found, say so in one line — do not drop the section.
5. **Validation** — what was run (type checks, smoke tests, suites) with the outcome, and what was *not* run and why. State the gaps honestly.
6. **Repo state and sign-off** — what is committed (hashes and subjects), what is staged or unstaged, what is untracked and unrelated, and what remains pending on the user. Close at the point where the next word is "ship it" or "change X."

## Verification duties

- The sitrep is a review artifact, not a self-praise artifact. Writing it is the agent reviewing its own work to the founder's standard: if a design choice cannot be articulated from the diff and an explicit decision, that is a finding, not a gap to paper over.
- Pattern-fit claims must be grounded: to call a change "native", name the precedent it follows. If no precedent can be pointed to, the change is uncharted — say so. Do not claim nativeness from a feeling.
- For every consumer built in the session (endpoint, tool result, event frame, API payload), verify the shape it assumes against what the producer actually emits.
- Distinguish the agent's changes from the user's changes made during the session. Review the user's changes on their merits — findings apply equally to both.
- Findings are severity-ordered. If verification surfaces more findings than the work can responsibly absorb in one sign-off, say so instead of merely listing them.
