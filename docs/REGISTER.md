# REGISTER.md — house register for agent-written text

Applies to skills, agent definitions, `AGENTS.md` maps, docs, commit messages, and prompts written
for a model to read. Repo-agnostic in content; lives here for convenience.

This file is written in the register it describes. Edit it in register.

## Name

**Cablese directive** — procedural directive prose written for a capacity-metered channel: grammar
shed to the recoverable minimum, while every rule, reason, count, negation, and literal is retained.
Condensation, never abridgment.

*Cablese* is attested: telegraph and newsroom compression under a per-word tariff. The motive
matches — context windows and human attention are metered, and the text says so when it matters.
*Directive* supplies the genre, which cablese alone does not.

The last four words of that definition carry the whole risk. "Cablese" connotes a short message;
retrieved cold, the name licenses making things shorter. It does not.

## The one rule

**Delete only what the reader can reconstruct. Keep everything the reader cannot.**

Recoverable, therefore deletable: articles, auxiliaries, complementizers, relative pronouns,
pronouns whose referent is unambiguous, copula and cleft frames, metadiscourse.

Not recoverable, therefore kept: numerals, literals, flags, symbols, defined terms, negations that
draw a boundary, causal claims, the reason a rule exists.

Length is not the target; information per word is. The witness: across one compression pass the
longest sentence that survived byte-identical ran 53 words, while short low-information sentences
beside it were rebuilt or cut. A length-driven rule would have cut the wrong one.

**The item-count test.** Count bullets, numbered items, and table rows before and after an edit.
Condensation leaves the count flat and drops words per item. A fallen count means you abridged —
you removed content, not grammar. This is the only objective check available; use it.

## Delete

Each with the operation that removes it.

| Target | Operation | Pair |
|---|---|---|
| Copula and cleft frames | drop `is what`, `there is` | `Merging is what keeps load flat` → `Merging keeps load flat` |
| Complementizers, relative pronouns | whiz-deletion | `a row that would require a file` → `a row requiring a file` |
| Finite clauses with the reader as actor | convert to non-finite | `a context that also holds queue position` → `a context also holding queue position` |
| Metadiscourse, throat-clearing | delete outright | `Why this shape:` · `Two collisions worth knowing:` → `Two collisions:` |
| Weak modality | harden | `may be scripted with gate:` → `script the fix wave with gate:` |
| Passives where the reader acts | active, second person | `is rejected here: it puts` → `rejected: it puts` |
| Third-person self-reference | `you` | `the orchestrator's job is` → `your job is` |
| Rhetorical amplification | keep the distinct claim, drop the restatement | a tricolon whose third term repeats the first |
| Coordinating conjunctions | clause-juncture punctuation | `and` → em-dash, semicolon, colon, or `→` |

Punctuation carries the load subordination used to. Expect one juncture mark per ~30 words. Give
each mark one job per file: an em-dash serving appositive, contrast, rationale, and parenthetical
in the same document is four relations wearing one shape.

## Keep

These look like fat and are not.

- **The reason a rule exists** — one clause, attached to the rule. A bare imperative is followed
  literally where the situation matches and abandoned where it does not; a rule carrying its
  mechanism gets re-derived correctly in a case the text never anticipated. That is the difference
  between a lookup table and a policy.
- **Every measured quantity.** Counts, thresholds, observed sizes, limits. They read as illustration
  and act as calibration. Tag their evidential status: `Observed:` · `measured` · `witnessed`.
- **Irreducible literals.** Flags, symbol names, file paths, ID shapes, exact quoted strings. Deleting
  one changes behavior.
- **Contrastive pairs that are the specification.** `attribute the judgment — "the pass classified
  these as per-invocation", not "these are per-invocation"`. A compressor sees a redundant example;
  the pair is the only thing that fixes the rule's boundary.
- **Long sentences whose words are all irrecoverable.** Compress the clause, not the content.

## Three layers, three floors

| Layer | Shape | Floor |
|---|---|---|
| Maxim | self-contained gnomic statement, timeless present | ~13 words. Already minimal — compressing further breaks it. `Refutations are samples too.` `Grep locates; reading concludes.` |
| Rule | bold label, directive, consequence clause | the directive and its reason both survive; only the grammar compresses |
| Template notation | fenced block, field labels, `<placeholders>`, pipe alternatives | verbatim. Never prose-ified, never paraphrased into a description of itself |

Most damage in a second compression pass lands on the maxim layer, because it looks like the least
informative text in the file. It is the most.

## Conventions

- **Person.** Address the reader as `you`. Use a role noun only to contrast roles, or where the
  statement is about the role rather than the reader's next action — `never an agent's judgment,
  never the orchestrator's` is about the role; `read it yourself` is about the reader.
- **Obligation.** Prohibitive and imperative carry most of it: `Never narrate dispatch.` Convert a
  gnomic statement to `must` where a fresh reader could take it for an observation — `A bucket must
  be answerable as one decision`, not `A bucket is answerable`. Leave definitional statements
  gnomic. Do not adopt RFC 2119 wholesale; capitalized MUST everywhere costs more attention than it
  disambiguates.
- **Defined terms.** One concept, one spelling, one casing, everywhere. `FLUSH block` is an artifact
  and never `flush block`; `flush` as a verb is fine. Case drift on a defined term is a defect, and
  it is greppable.
- **Coinage.** Naming a concept compresses more than deleting words does: one term replaces a
  sentence every time it recurs. Coin deliberately, define once at first use, then use the term
  without re-explaining it. Prefer a new term over overloading one already in use.
- **Absolutes.** A directive may be absolute — `never commit` is the point of the absolute. A claim
  about the world may not. `A clean tree at your exit is a hard requirement` is false whenever other
  work is in flight, and when reality disagrees the reader must choose between obeying and being
  accurate. State what is checkable instead: `your own contribution to the diff is one new file`.
  Never assert a mechanism no file mandates.

## Two settings

Chosen by what the reader does with the sentence, not by taste.

- **Condensed** — gates, rules, checklists, tables, field templates. The reader acts on it. Block
  language is correct here; full finite prose is noise.
- **Loose** — explanation of a causal chain or a state model to a reader who lacks the context. The
  reader must reconstruct a mechanism, and reconstruction needs finite clauses with their connectives
  intact. A handoff document written for a reader who just lost its context belongs here.

A file may hold both. What it may not do is drift between them without reason: if a passage runs
looser than its neighbours, the sentence should say what the reader needs it for.

## Failure modes

- **Abridgment dressed as condensation.** Item count fell. The test catches it.
- **Compressing a maxim.** It was already at the floor.
- **Cutting the mechanism clause.** The rule survives and stops transferring.
- **Restating a rule in new words.** Two phrasings of one constraint, with nothing to say which
  governs. Fluency hides this; the item-count test does not.
- **Absolutism about the world.** See Conventions.
- **Drama.** Not merely filler. It blurs traced and imagined for the writer, and for every agent
  seeded from the prose.

## Before saving

- [ ] Item count unchanged, or the change is an intentional addition or removal you can name
- [ ] Every rule still carries its reason
- [ ] Every count, literal, flag, and defined term survived
- [ ] No maxim shortened
- [ ] No absolute claim about the world; no asserted mechanism without a witness
- [ ] Defined terms: one spelling, one casing
- [ ] Read it cold — could a reader with no other context act on every line?
