# config/prompts/ — the model-facing prompt corpus and the single loader that serves it

## Files

**The loader**

- `__init__.py` — package marker re-exporting `load_prompt`; reads no template. Nothing to open here.
- `loader.py` — the one sanctioned template read. Open it before making new code load a prompt, or when a template seems to be missing.

**Memory extraction and curation**

- `memory_extraction_system.txt` — the live extraction contract. Open it before changing what becomes a durable memory.
- `memory_extraction_user.txt` — the transcript half of that pair. Open it when the extractor's input format changes.
- `memory_extraction_system_qwen_v2.txt` — dead: no call site loads it, and its head marker says so.
- `memory_extraction_system_qwen_experimental.txt` — dead: a second unused extraction variant, marked at its head too.
- `pending_memory_entities_system.txt` — the entity-tagging contract for memories already stored. Open it before changing how a stored memory earns its retrieval anchors.
- `pending_memory_entities_user.txt` — the batch of memories to tag. Open it when that input format changes.
- `entity_merge_system.txt` — the trigram-match dedup judge. Open it before changing when two entities count as one.
- `entity_merge_user.txt` — the grouped-candidate half. Open it when the group format changes.

**Segment summary, compaction, retrieval**

- `segment_summary_system.txt` — the contract for what survives a segment collapse. Open it before changing what a collapsed segment keeps.
- `segment_summary_user.txt` — the segment half of that pair. Open it when the summarizer's input format changes.
- `synthesis_summary_system.txt` — folds a long segment's partial traces into one summary. Open it before changing that fold.
- `synthesis_summary_user.txt` — the chunk half of that pair. Open it when the chunking input format changes.
- `live_context_compaction_system.txt` — the rolling continuation brief: one per session, each pass updating and extending the last. Open it before changing what a compacted range must carry forward.
- `live_context_compaction_user.txt` — the newly compacted range. Open it when the compaction input format changes.
- `subcortical_system.txt` — the per-turn retrieval filter. Open it before changing which memories and entities a turn sees.
- `subcortical_user.txt` — the turn half of that pair.
- `behavioral_primer.txt` — a rehearsal correction dialogue spliced into the session cache. Open it before editing its turns; its shape is a parser contract, see `## Rules`.

**User model: assess, synthesize, validate, revise**

- `assessment_extraction_system.txt` — the third-party judge of one segment against the behavioral contract. Open it before changing what counts as alignment.
- `assessment_extraction_user.txt` — the judge's input: the segment, the anonymized system prompt, and the current user model's observations.
- `thinking_block_instructions.txt` — the weighing rules for thinking traces, spliced into that judge's system prompt.
- `user_model_synthesis_system.txt` — the synthesizer contract. Open it before changing how signals become stored observations; two of its examples are filled from the live section vocabulary.
- `user_model_synthesis_user.txt` — its inputs: valid anchors, current model, unsynthesized signals.
- `user_model_critic_system.txt` — the quality gate every candidate model passes before storage. Two flows load it, so a wording change reaches both.
- `user_model_critic_user.txt` — the candidate under review. Open it before changing what the critic is shown.
- `lora_refinement_system.txt` — the user-feedback reviser of a stored model; its user message is code-built, so it has no user twin. Open it before changing what a user correction may edit.

**Persona and portrait**

- `persona_evaluation_system.txt` — the segment judge of MIRA's behavior against the contract and the current Persona. Open it before changing what a persona signal means.
- `persona_evaluation_user.txt` — its three inputs.
- `persona_refinement_system.txt` — the Persona editor, and the system prompt of the manual refinement path too. Open it before changing how a Persona directive is written.
- `persona_refinement_user.txt` — the evidence-driven revision inputs.
- `persona_manual_refinement_user.txt` — the user-instructed revision; no system twin of its own.
- `persona_critic_system.txt` — the validator over a candidate Persona. Open it before changing what a directive may contain.
- `persona_critic_user.txt` — the candidate under review.
- `portrait_synthesis_system.txt` — the User Context portrait contract. Open it before changing how the user is described in the system prompt.
- `portrait_synthesis_user.txt` — the segment summaries it profiles from.
- `portrait_refinement_system.txt` — the portrait reviser on user feedback; its user message is code-built, so it has no user twin.

**Response shaping**

- `repulsion_rewriter_system.txt` — the register-conditional de-telling rewriter. Open it before changing what counts as assistant-shaped writing.
- `repulsion_rewriter_user.txt` — the message pair plus the flagged tells.
- `peanutgallery_system.txt` — the Tuning Fork observer contract. Open it before changing when MIRA is told to correct itself mid-conversation.
- `peanutgallery_user.txt` — the history and execution trace it judges.
- `domaindoc_summary_system.txt` — the table-of-contents sentence contract; open it before changing what a section's index line says.
- `domaindoc_summary_user.txt` — the section header and content it summarizes.

**Untrusted content**

- `untrusted_content_directive.txt` — the treat-as-data sentence placed inside every untrusted-content wrapper. Open it before changing what a model is told about wrapped content.

Tier-1 questions — one file per name in `SCREEN_QUESTIONS`, and the name is also the answer key in every screen response. Open one before reworking the question it asks.

- `injection_screen/overrides_instructions.txt` — instruction and identity hijacking.
- `injection_screen/requests_action.txt` — orders to act on the user's behalf.
- `injection_screen/requests_disclosure.txt` — extraction of prompt, memories, or user detail.
- `injection_screen/requests_deception.txt` — steering the assistant to mislead its own user.
- `injection_screen/claims_false_authority.txt` — forged authority over the assistant.
- `injection_screen/exfiltration_channel.txt` — data leaving through a link, image, or address.
- `injection_screen/escalation_system.txt` — the tier-2 judge contract for content tier 1 could not place.
- `injection_screen/escalation_user.txt` — the per-question scores and the content under review.
- `injection_screen/revealed_text_header.txt` — the line placed above decoded hidden text.

**Sidebar agent rubrics**

- `agents/base_system.txt` — the sidebar loop contract; open it before changing what an agent tells itself about its own loop.
- `agents/forage_system.txt` — the research protocol; open it before changing how a sidebar research agent is told to get past the first page of results.
- `agents/heartbeat_agent_system.txt` — the brief gate: bail, check, or dispatch, one report per brief.
- `agents/phoneafriend_system.txt` — the outside voice, whose completion summary is the only channel back to the caller.
- `agents/whilethecatsaway_system.txt` — idle-time curiosity research that stores memories as it goes.
- `agents/memory_curator_integration.txt` — curator INTEGRATION mode, after a segment collapse.
- `agents/memory_curator_floor.txt` — curator FLOOR mode, over memories already scoring low.

## Wiring

- Loader boundary: `load_prompt` → each consumer's own load site → the LLM call that load feeds. Every consumer reads through it, the full set by `rg 'load_prompt\('`; loading happens at construction or first call and always before the call it feeds, so a load moved after the call sends a request with no system prompt. Counterpart: `cns/services/AGENTS.md`.
- Section vocabulary: `config.system_prompt` → `get_assessable_section_ids` → the `{example_section_a}`/`{example_section_b}` fill in `lora_refinement_system.txt` and `user_model_synthesis_system.txt` → `validate_section_anchors` at the save paths. The vocabulary must be one live read at fill time and at validation time; renaming a section between them strands every observation built in that window, because validation compares against the prompt as it is now. Counterpart: `config/AGENTS.md`.
- Screen chain: the reveal header → the six tier-1 questions → band edges → the escalation pair → `wrap_untrusted` with `untrusted_content_directive.txt`. Wording and band edges are calibrated to each other against one battery, so a reworded question moves both bands and the template cannot show which way: either benign content starts being refused, or an attack starts passing. Counterparts: `utils/AGENTS.md`, `tests/protected/AGENTS.md`.
- Extraction edge: `memory_extraction_*` crosses into `lt_memory/processing/AGENTS.md`, which owns parsing, id remap, and temporal handling; this map owns only what the template says. Counterpart: `lt_memory/processing/AGENTS.md`.
- Pending-entity edge: `pending_memory_entities_*` is loaded and its reply parsed in `cns/services/segment_collapse_handler.py`, which owns the pending_id echo keyed by identity. An output-shape change to that template and the parser move in the same commit. Counterpart: `cns/services/AGENTS.md`.
- Sidebar rubrics: `load_agent_prompt` → `load_prompt("agents/<name>")`. A rubric loads at the agent's first prompt fetch, not at import, so a `py_compile`-clean agent can still carry a missing rubric. Counterpart: `agents/AGENTS.md`.
- Gated edge — `config/prompts/agents/base_system.txt`: reached only when `BaseAgent.inherit_base_prompt` is left at its default `True`, and every concrete agent sets it `False`, so the file currently reaches no live prompt. A new agent leaving the default re-activates it; the loop contract it states is enforced by `agents/base.py` either way. Counterpart: `agents/AGENTS.md`.

## Rules

- Every `{var}` in a template pairs with exactly one fill at its call site, and the two fill mechanisms fail differently. `.format()` is used where every token is a call-site key: it raises `KeyError` on a placeholder nobody fills, but silently ignores a key the template no longer carries, so a call passing a stale key looks fine. `.replace()` is used where the fill value is generated in code and must reach the model verbatim — the pending-entity format example, the escalation verdict format, the thinking-weight block — and leaves the unfilled `{var}` in front of the model as literal text with no error anywhere. Template edit and call site move in the same commit.
- A template's output shape and its parser are one contract, so change both in the same commit. `<mira:complexity>` accepts exactly `0.5|1|2|3` (`utils/tag_parser.py`), so any other value a prompt offers degrades silently to the default score; an omitted persona tag raises `PersonaParseError`; an unparseable critic verdict fails closed rather than auto-publishing. Where a template's format token is code-built — `{OUTPUT_FORMAT_EXAMPLE}` in `pending_memory_entities_system.txt`, `{verdict_format}` in `config/prompts/injection_screen/escalation_system.txt` — edit the generating code, never the template text.
- Exactly three templates load with `required=False`: `thinking_block_instructions.txt`, `synthesis_summary_system.txt`, `synthesis_summary_user.txt`. Every other file here is load-bearing, and where the load sits decides the blast radius: `untrusted_content_directive.txt` loads at `utils/untrusted_content.py` module import, so an emptied copy fails the boot; a template loaded at module or service construction also fails the boot; one loaded inside a call-scope function survives boot and fails the first call of that feature. Emptying a template never disables a feature — it moves the failure. Absence is not always a degraded mode: the optional synthesis pair absent raises in `summary_generator._synthesize_chunk_summaries` and re-surfaces as a `ValueError` from `_generate_chunked_summary`, so the first segment too large for the context window cannot be collapsed at all.
- `behavioral_primer.txt` is a wire format, not prose: `cns/core/segment_cache_loader.py` splits the file on `---`, reads the bracketed first line of each block as the role, and takes the remainder as content. The roles it can emit are the three `Message` accepts, and a block with an unrecognized role or an empty remainder raises `ValueError` from `Message.__post_init__` while the session cache loads — so an edit that drops a bracket or a line fails the first session load rather than degrading the primer.

Elsewhere: this map does not own which pipeline fires a template, how its reply is parsed, or what a template ought to say — firing and reply parsing belong to the invoking service and its own map, cited per consumer under `## Wiring`; the escalation verdict's parse contract is the docstring of `utils/untrusted_content.py`; why a prompt reads the way it does is `git log`. Every other destination this directory reaches is an edge — see `## Wiring` above.

## The assessment loop, one segment to the next turn's system prompt

The user-model templates are one loop, not eight separate files, and it exercises three of the Rules above.

1. A segment collapses. `cns/services/assessment_extractor.py` fills `assessment_extraction_user.txt` with the anonymized system prompt, the current model's observations — this loop's own earlier output — the section list, and the segment as typed turns. Whether thinking traces are rendered at all is decided there, against `THINKING_INCLUSION_THRESHOLD`; `thinking_block_instructions.txt` only supplies how to weigh them, so with the optional file absent the traces are still read, unweighed.
2. The reply's `<mira:assessment>` signals carry, per section, an alignment verdict and its cited evidence. Misalignment outweighs alignment, a weighting both the extraction user template and the synthesis system template state.
3. Synthesis folds every unsynthesized signal plus the current model into a candidate `<mira:user_model>`. There is no signal window: the repository selects all unsynthesized signals, and the run cadence is a separate use-day counter.
4. The candidate passes three gates in order — deterministic anchor validation in code, then the LLM critic against its three failure modes, then `CRITIC_MAX_ATTEMPTS` exhaustion, which raises rather than publishing. The critic is the only quality gate before auto-publish.
5. The stored model is served into every later turn's system prompt by `working_memory/trinkets/lora_trinket.py` — the same observations step 1 then reads back as context. Explicit user feedback short-circuits steps 1 to 3: the refinement template revises the stored model from the user's words, then the same anchors and critic apply.

The ordering invariant that spans the loop: the anchor vocabulary filled into the templates at step 1 and step 3 and the vocabulary validation checks at step 4 must be the same live read of `config.system_prompt`. The persona family runs the same evaluate, revise, critic shape with regex parsing in place of anchor validation.
