# MIRA glossary

Codebase-specific terms, one sentence each. Each entry names the map or file that owns the detail.

## Conversation and memory

- **Segment**: a span of conversation between idle boundaries; one segment is active at a time and it collapses into a summary when idle (`docs/SEGMENT_SYSTEM.md`).
- **Collapse**: turning an idle segment into its summary sentinel, which changes stored history (`cns/services/AGENTS.md`).
- **Sentinel**: the stored message that replaces a collapsed segment and carries its summary fields and `segment_embedding` (`cns/infrastructure/AGENTS.md`).
- **Live context compaction**: shrinking the provider request during an active conversation with a rolling brief, leaving the PostgreSQL history intact (`docs/SEGMENT_SYSTEM.md`).
- **Continuum**: the aggregate object that holds one user's conversation (`cns/core/AGENTS.md`).
- **Short ID (`mem_XXXXXXXX`)**: an irreversible prefix of a memory's UUID, used on surfaces the LLM reads; persistence uses the full UUID (root `AGENTS.md`).
- **Domaindoc**: a section-aware knowledge document stored in the user's SQLite database (`tools/implementations/AGENTS.md`).
- **Persona**: user-directed behavior guidance stored as append-only revisions with per-segment evaluation signals (`cns/infrastructure/AGENTS.md`).
- **Portrait**: a synthesized description of the user, stored on `users.portrait` (`cns/services/AGENTS.md`).
- **Behavioral directives**: the user-model trinket slot whose content is synthesized from conversation history (`cns/services/AGENTS.md`).

## Turn pipeline and auxiliary models

- **Subcortical**: the auxiliary LLM stage that runs before the main model call, handling entity extraction, memory filtering, query expansion, and complexity classification (`config/prompts/AGENTS.md`).
- **Peanut gallery (Tuning Fork)**: a metacognitive observer that emits at most one out-of-band concern or coaching signal (`config/prompts/AGENTS.md`).
- **Trinket**: a per-user slot in the system prompt with one `variable_name`, updated by events (`working_memory/AGENTS.md`).
- **Compose (`ComposeSystemPromptEvent`)**: the event round-trip in which trinkets build the system prompt before a turn runs (`cns/services/AGENTS.md`).
- **Route (`primary`, `fast`, `batch`, `assessment`, `other`)**: a named LLM configuration row in `model_configs`; code calls routes by name and no route falls back to another (`clients/llm/AGENTS.md`).
- **Dialect**: a per-provider class that translates the neutral request and result contracts into that provider's wire format (`clients/llm/dialects/AGENTS.md`).
- **Preview-before-save**: a revision held in Valkey under an opaque `preview_id` with a 600-second TTL and consumed on read (`cns/services/AGENTS.md`).

## Background work

- **Heartbeat**: the scheduled wake cycle in which Mira acts without a user message (`cns/services/AGENTS.md`).
- **Use-day**: an activity-day counter that gates jobs, so jobs fire on days the user is active rather than on calendar days (`utils/AGENTS.md`).
- **Sidebar agent**: an autonomous background agent with its own loop that reports through a trinket (`agents/AGENTS.md`).
- **Forage, While-the-cats-away, Memory curator**: the three concrete sidebar agents in `agents/implementations/`; the curator tends existing memories and never creates them (`agents/implementations/AGENTS.md`).
- **Overwatch**: an observer that runs in a daemon thread beside a sidebar agent, and whose failures are logged at debug and dropped (`agents/AGENTS.md`).
- **Rubric**: the self-contained prompt file under `config/prompts/agents/` that defines one sidebar agent's loop and completion behavior (`config/prompts/AGENTS.md`).

## Security and external content

- **Injection screen**: the check on external content before it enters any model context (`utils/AGENTS.md`).
- **System One / `djev`**: the model service behind the injection screen; `djev` is the gateway name the installer uses for it (`deploy/AGENTS.md`).
- **`encrypted__` prefix**: the column-name prefix that makes SQLite columns Fernet-encrypted transparently (`utils/AGENTS.md`).

## Deployment and infrastructure

- **Lunaroute**: the default hosted LLM gateway the installer uses for chat, subcortical, and embedding routes (`deploy/AGENTS.md`).
- **Sarcophagus**: a sealed dump of a running instance, restored onto a dev VM with `pg_restore` (`deploy/vm/AGENTS.md`).
- **POST gate**: the check in `main.py` that rejects POST requests before the server binds, held closed until a deploy has real configuration (`deploy/vm/AGENTS.md`).
- **Lattice**: the federation subsystem, mounted only when `config.lattice.enabled` is set (`clients/AGENTS.md`).
- **Firehose**: a debug mode that writes every LLM API call to `firehose_output.json`, toggled by `SIGUSR1` (`main.py`).

## Verification

- **Probe**: a live check against real infrastructure, which is the required form of verification in this repo (`tests/AGENTS.md`).
- **BLOCKED**: a probe verdict meaning a required service (PostgreSQL, Valkey, or Vault) is not listening, so nothing was checked (root `AGENTS.md`).
- **UNVERIFIED**: a probe verdict meaning the path could not be checked on this machine, usually because Vault AppRole credentials are missing (root `AGENTS.md`).
- **Admission-gated battery**: a permanent test suite under `tests/protected/` that runs only after an exact-phrase authorization (`tests/protected/AGENTS.md`).
