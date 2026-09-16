# How to Build a Tool

*Technical guide based on successful tool development patterns*

## Contribute It Back

Every line in this tree is agent-written, and this repository is public. When you build a tool that is **broadly useful** — not a one-off shaped around one person's data, endpoints, or habits — propose opening a pull request against `origin/main` so every other MIRA installation gets it too. MIRA gets better for everyone when the tools built on one instance flow back into the shared trunk.

**Propose, never assume.** Ask the user first. Do not create a branch, commit, push, or open a PR without an explicit yes — root `AGENTS.md` Git Workflow requires it.

### When to propose

| Propose a PR | Keep it local |
|---|---|
| Solves a problem any installer would eventually hit | Encodes one user's personal workflow, naming, or data shape |
| Integrates a public API or service others would use | Wraps a private/self-hosted endpoint nobody else has |
| Fills an obvious gap in the tool set | Duplicates what an existing tool already does |
| Fixes a bug or contract defect you found while building | Is scaffolding for an unfinished local feature |
| — | Would contain secrets, user rows, or `data/users/**` |

When in doubt, ask. "This feels generalizable — want me to open a PR upstream?" costs one sentence and is never the wrong question.

### The process

1. **Finish and verify locally first.** Boot gate clean, the new path executed against live infrastructure, `tools/implementations/AGENTS.md` bullet written. Never propose a PR for code that has only been type-checked — see Verification below.
2. **Ask the user.** Say what the tool does, why it is generally useful rather than personal, and exactly which files the PR would touch.
3. **On yes, branch fresh off upstream.** Never PR from `main`, and never carry unrelated local changes along.
   ```bash
   git fetch origin
   git checkout -b feat/my-tool origin/main
   ```
   If you cannot push to `origin` (you cloned someone else's fork, or the repo is read-only to you), fork `taylorsatula/mira-OSS` first, add it as a remote, and target the public `main`.
4. **Stage explicit paths only.** `git add tools/implementations/my_tool.py tools/implementations/AGENTS.md`. Never `git add -A` or `git add .` — a working checkout holds `data/`, scratch notes, local config, and possibly secrets.
5. **Read what you staged** with `git diff --cached --stat` and then `git diff --cached`. Hunt specifically for credentials, personal paths, usernames, and debugging leftovers.
6. **Commit** with a semantic prefix (`feat:`, `fix:`, `refactor:`) and, for non-trivial changes, `ROOT CAUSE` and `SOLUTION RATIONALE` body sections. Report the hash and file summary to the user.
7. **Push and open the PR** against `main`. Short title in the project's own voice, body covering what it does, why it belongs upstream, and how it was verified. One concern per PR — an unrelated fix goes on its own branch.
8. **Return the user to the branch they were on.** Leave their checkout as you found it.

If your harness provides a PR-writing or git-workflow skill, load it before step 6.

### Before it ships

- [ ] No secrets, tokens, API keys, or real credential values
- [ ] No `data/users/**`, SQLite files, traces, scratch notes, or local config
- [ ] No user-specific hardcoding — paths, usernames, IDs, device names, timezone assumptions
- [ ] `tools/implementations/AGENTS.md` `## Files` bullet in the same commit
- [ ] Boot gate passes from a clean checkout: `python -m utils.power_on_self_test pre-server`
- [ ] The new path was executed live and the result reported
- [ ] New external dependency justified in the PR body, or none added
- [ ] Disabled-by-default (`enabled: bool = Field(default=False)`) unless the user and upstream both want it on — a new tool should not silently enter anyone's startup set
- [ ] Not added to `ESSENTIAL_TOOLS`; that is an upstream decision, not a contributor one

This repository is AGPL-3.0; contributions land under that license.

## 🚀 START HERE

Tools in MIRA follow consistent patterns. This guide shows you where to find each pattern in existing, battle-tested implementations. Use the **Pattern Index** below to jump directly to the code that demonstrates what you need.

**Learning approach:**
1. Scan the Pattern Index to see what's available
2. Find the pattern you need in an existing tool
3. Read that implementation — anchors are `file.py:symbol`, not line ranges
4. Copy the pattern and adapt it to your use case

**Doctrine owners:** `tools/AGENTS.md` (base class, repository, four-piece registration, param filtering) and `tools/implementations/AGENTS.md` (per-tool contracts, credential and parallelism rules). This guide is the walkthrough; those maps are the contract. Where they disagree, the maps and the code win.

## Core Concepts

A tool is a **function the model calls**. Its caller is a language model that infers behavior from names and descriptions, and its runtime is a per-user-scoped process that constructs a fresh instance on every call. Both facts shape the work:

| Because the caller is a model | Because the runtime is per-user |
|---|---|
| Parameter descriptions are interface contracts, not docs | `self.db` and `self.user_data_path` are already scoped — never filter by `user_id` manually |
| Undeclared params are unreachable; undeclared behavior is guessed | `__init__` runs per call — keep it cheap, defer table creation |
| Ambiguity becomes hallucination | User context comes from a contextvar, not from arguments |

Three component types, three guides:

| You want to… | Build | Guide |
|---|---|---|
| Execute an operation the model asks for | **tool** | this document |
| Reflect state in the system prompt | **trinket** | `working_memory/trinkets/HOW_TO_BUILD_A_TRINKET.md` |
| Do autonomous multi-step work with no user present | **sidebar agent** | `agents/HOW_TO_BUILD_AN_AGENT.md` |

When the ask is "and MIRA should *do something* about it unprompted," remember the boundary: a tool's job is to **surface** the state and let the model act conversationally — an overdue flag on a list result, a `gone_too_long` hint the model relays or turns into an internal reminder. That composition (tool flags, model decides) covers most "bug me about it" asks without a scheduler; a sidebar agent is for work that must happen when the user isn't there at all.

## 📋 Pattern Index

### Essential Patterns

Every tool needs these core patterns. Anchors name a file and a symbol; line numbers are deliberately absent because they go stale on the next edit.

| Pattern | Where to Find | What It Shows |
|---------|---------------|---------------|
| **Tool Base Class** | `tools/repo.py:Tool` | Properties (`user_id`, `user_data_path`, `db`), file helpers, abstract `run()` |
| **Tool Metadata** | `reminder_tool.py:ReminderTool` | `name`, `simple_description`, `tool_schema` |
| **Configuration** | `reminder_tool.py:ReminderToolConfig` | Pydantic config + module-level `registry.register()` |
| | `contacts_tool.py:ContactsToolConfig` | Config with `Field` descriptions |
| **Deferred Initialization** | `reminder_tool.py:ReminderTool.__init__` | `has_user_context()` guard around table creation |
| **Database Schema (inline)** | `reminder_tool.py:_ensure_reminders_table` | `self.db.create_table` + indexes owned by the tool |
| **Database Schema (central)** | `utils/userdata_manager.py:_init_contacts_schema` | Table created at user-DB init; DDL text in `tools/implementations/schemas/contacts_tool.sql` |
| **Operation Routing** | `reminder_tool.py:ReminderTool.run` | Operation dispatch with signature-filtered kwargs |
| | `contacts_tool.py:ContactsTool.run` | Clean routing, one `_handler` per operation |
| **Input Validation** | `contacts_tool.py:_add_contact` | Collect all errors before raising |
| **CRUD Operations** | `contacts_tool.py:_add_contact` / `_get_contact` / `_update_contact` / `_delete_contact` | Full CRUD with UUID generation |
| | `reminder_tool.py:_add_reminder` | CRUD with cross-tool linking (`contact_uuid`) |
| **Model-Supplied Timestamps** | `reminder_tool.py:_parse_date` | User-local wall time → UTC, DST-strict |
| | `punchclock_tool.py:_parse_time_input` | Offsets (`-10m`) plus absolute wall times |
| **Display Formatting** | `reminder_tool.py:_format_reminder_for_display` | UTC → user timezone, strips `encrypted__` prefixes |
| **Encryption** | `utils/userdata_manager.py:_encrypt_dict` / `_decrypt_dict` | How the `encrypted__` prefix works internally |
| | `pager_tool.py:_decrypt_row_safely` | Never double-decrypt an already-decrypted row |
| **Fuzzy Name Matching** | `contacts_tool.py:_find_by_identifier` | UUID → exact → starts-with → contains, ambiguity surfaced not guessed |
| | `homeassistant_tool.py:_resolve_entity` | Same ladder over a cached registry |
| **Credential Storage** | `utils/user_credentials.py:UserCredentialService` | Per-user API keys |
| **Credential Injection** | `web_tool.py:_get_credential` / `_http` | LLM names a credential it never sees |
| **File Operations** | `tools/repo.py:Tool.make_dir` / `get_file_path` / `open_file` / `file_exists` | User-scoped file helpers |
| **Response Formatting** | `contacts_tool.py:_format_contact` | `{"success": bool, "message": str, ...}` envelope |
| **Error Handling** | `reminder_tool.py:_get_reminder_not_found_error` | Helpful errors listing available options |
| **LLM Calls From a Tool** | `pager_tool.py` (`model_config="fast"`) | Route by name via `get_llm_provider()` |
| | `phoneafriend_tool.py:OUTSIDE_MODEL_CONFIG` | Constructor-injected `LLMProvider` |
| **Trinket Refresh From a Tool** | `forage_tool.py:_publish_event` | `event_bus.publish(UpdateTrinketEvent.create(...))` |
| | `sidebaragents_tool.py` | `agents/base.py:_publish_trinket_refresh` |

### Specialized Patterns

| Pattern | Where to Find | When You Need It |
|---------|---------------|------------------|
| **Batch Operations** | `contacts_tool.py:_batch_add_contacts` | Bulk imports with per-item error tracking |
| **Gated Tools** | `tools/repo.py:ToolRepository.register_gated_tool` + `is_available()` | Availability decided per call, not by config |
| **Tool Dependencies** | `tools/repo.py:Tool.get_dependencies` / `resolve_dependencies` | Tools that require other tools |
| **Dependency Injection** | `tools/repo.py:ToolRepository.get_tool` | `LLMProvider`/`LLMBridge`, `ToolRepository`, `WorkingMemory` injected by constructor signature |
| **Dynamic tool_schema** | `domaindoc_tool.py:tool_schema` (`@property`) | Enum values that change as user data changes |
| **Tool Discovery** | `tools/repo.py:ToolRepository.discover_tools` / `_process_module` | Auto-registration from `tools/implementations/` |
| **Param Filtering/Coercion** | `tools/repo.py:ToolRepository.invoke_tool` / `_coerce_to_schema_type` | What reaches `run()` and what is dropped |
| **Abstract Base in implementations/** | `tools/repo.py` (`_is_abstract_base_class`) | Shared base that must not appear in the catalog |
| **Natural Language Dates** | `reminder_tool.py:_parse_date` | Phrase ladder: explicit today/tomorrow/yesterday → regex relatives ("in 3 weeks") → lenient fallback. `parse_time_string` does ISO/date-only/time-only **only** — it raises on "tomorrow". Weekday phrases ("last Tuesday") are in no existing ladder — expect to extend it with `dateutil.relativedelta(weekday=...)` |
| **UUID Cross-Tool Linking** | `reminder_tool.py:_lookup_contact` / `_get_contact_by_uuid` | Linking rows across tools in the same user store — note the policy split: reminder links silently on substring match (read-path leniency), while contacts' mutating ops demand exact-or-UUID and return `needs_confirmation`/`ambiguous` instead of guessing. **For bindings that touch money or permanence, follow the stricter contacts policy** |
| **Duplicate Detection** | `reminder_tool.py:_check_duplicate_reminder` | Idempotent retries within a time window — the *time-window* shape. The exact-key shape (same logical item → return existing) is a `UNIQUE` constraint in the table DDL plus check-before-insert; both are valid, pick the one that matches your semantics |
| **Config Validation** | `tools/repo.py:Tool.validate_config`, `email_tool.py:validate_config` | Live connection test + folder auto-discovery for the validate endpoint |
| **Manager Caching** | `utils/userdata_manager.py:get_user_data_manager` | Per-user `UserDataManager` cache |
| **Response Sanitization** | `web_tool.py:_SENSITIVE_RESPONSE_HEADERS` | Strip credentials from outbound responses |
| **SSRF-Safe Requests** | `web_tool.py:_request_with_validated_redirects` | Per-hop validation + IP pinning |
| **Content-Block Results** | `imagegen_tool.py` | Returning a list of provider content blocks instead of a dict |
| **Restricted Agent Schema** | `memory_tool.py:CURATOR_MEMORY_SCHEMA` | Exporting a narrowed schema for sidebar agents |
| **Spawning an Agent** | `forage_tool.py:_dispatch` / `_run_agent_thread` | Daemon thread under `copy_context()` |

## Architecture Deep Dive

### Tool Base Class (`tools/repo.py:Tool`)

The `Tool` base class provides automatic user scoping and file operations:

```python
class Tool(ABC):
    name = "base_tool"
    description = "Base class for all tools"
    parallel_safe: bool = True  # Set to False for tools that mutate shared state

    @property
    def user_id(self) -> str:
        """Current user from context."""
        return get_current_user_id()

    @property
    def user_data_path(self) -> Path:
        """User-specific directory for this tool."""
        user_data = get_user_data_manager(self.user_id)
        return user_data.get_tool_data_dir(self.name)

    @property
    def db(self):
        """Lazy-loaded, user-scoped UserDataManager."""
        current_user_id = self.user_id
        if not self._db or self._db.user_id != current_user_id:
            self._db = get_user_data_manager(current_user_id)
        return self._db

    # File operations - user-scoped automatically
    def make_dir(self, path: str) -> Path: ...
    def get_file_path(self, filename: str) -> Path: ...
    def open_file(self, filename: str, mode: str = 'r'): ...
    def file_exists(self, filename: str) -> bool: ...
```

**Key Points:**
- **Automatic User Scoping**: `self.db` is always scoped to the current user - no manual filtering needed
- **Lazy Initialization**: Database connection created on first access
- **File Isolation**: `self.user_data_path` returns `data/users/{user_id}/tools/{tool_name}/`
- **Parallel Safety**: Set `parallel_safe = False` for tools that mutate shared state where operation order matters (e.g., create-then-edit). Sequential tools execute first, then parallel-safe tools run concurrently. **`parallel_safe = False` alone also serializes your reads** — mixed read/write tools pair it with the `is_call_parallel_safe(cls, tool_input)` override returning True for the read operations only (the Quick Start template shows the two pieces combined; the override fully replaces the base `return cls.parallel_safe`).

### Database Operations (utils/userdata_manager.py)

UserDataManager provides a simple API for SQLite operations with automatic encryption:

```python
# Create table with schema. The schema string is passed through to SQLite
# verbatim ("CREATE TABLE IF NOT EXISTS {name} ({schema})"), so constraints
# and multi-column UNIQUE() work. insert() surfaces a bare
# sqlite3.IntegrityError on constraint violation — check-before-insert if
# you want a helpful message instead. SQLite treats NULLs as DISTINCT in
# UNIQUE, so a nullable column in your dedup key silently stops deduping
# the NULL rows — use an explicit "IS :value" check for those cases.
self.db.create_table('my_items', """
    id TEXT PRIMARY KEY,
    encrypted__title TEXT NOT NULL,
    encrypted__notes TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(id, created_at)
""")

# Create indexes
self.db.execute("CREATE INDEX IF NOT EXISTS idx_items_date ON my_items(created_at)")

# CRUD operations - encryption is automatic
self.db.insert('my_items', {
    'id': item_id,
    'encrypted__title': 'Secret Title',  # Will be encrypted
    'created_at': format_utc_iso(utc_now())
})

items = self.db.select('my_items')  # Returns decrypted data
# Note: On read, encrypted__ prefix is KEPT in the field name but value is decrypted

self.db.update('my_items',
    {'encrypted__title': 'New Title'},  # Will be encrypted
    'id = :id',
    {'id': item_id}
)

self.db.delete('my_items', 'id = :id', {'id': item_id})
```

**Critical Encryption Details (`utils/userdata_manager.py:_encrypt_dict` / `_decrypt_dict`):**
- Fields prefixed with `encrypted__` are automatically encrypted on write
- On read, values are automatically decrypted but **the prefix is kept in the field name**
- Access decrypted values as `item['encrypted__title']`, not `item['title']`
- Encryption key is derived deterministically from user_id (persistent across sessions)

### Connection Management (`utils/userdata_manager.py:UserDataManager.connection`)

```python
@property
def connection(self) -> sqlite3.Connection:
    """Lazy persistent connection (thread-safe for cross-thread reuse)."""
    if self._conn is None:
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False  # WAL mode handles concurrency
        )
        self._conn.row_factory = sqlite3.Row
    return self._conn
```

- **Lazy Creation**: Connection created on first database access
- **Thread-Safe**: `check_same_thread=False` allows cross-ThreadPoolExecutor usage
- **Cached Per-User**: `get_user_data_manager()` returns cached instances
- **Automatic Cleanup**: Connections closed on session collapse via event subscription

## Development Process

### Phase 1: Requirements Discovery

Initial descriptions often use metaphors or analogies. Extract concrete requirements:

```
Example: "Like 90s pagers"
Extracted requirements:
- High urgency messaging only
- Minimal UI complexity
- Respects user attention
- No feature creep
```

Essential questions:
- "Can you walk me through a typical usage scenario?"
- "What problem does this solve that existing tools don't?"
- "What should this tool explicitly NOT do?"
- "What would indicate success for users?"

The **rejections** in an ask are requirements too. "i'm not a library, i don't need due dates" and "every calorie app turns it into homework" are the strongest spec you will get — a feature the user explicitly refused is a defect if you build it. When no human is available to clarify, extract the rejections first, then fill the remaining gaps with the smallest defensible assumption and record what you assumed.

### Phase 2: Specification Analysis

Detailed specifications often contain both explicit features and implicit design philosophy. Minor details frequently encode critical constraints.

Example: If a spec mentions "no notification fatigue," this implies rate limiting, priority systems, or other attention-management features.

### Phase 3: Codebase Pattern Study

**Use the Pattern Index above** to find exactly what you need. Each entry names the file and the symbol that demonstrates the pattern.

**Recommended reading order:**
1. **Start simple**: `reminder_tool.py` - clean CRUD, deferred schema, the two-clocks timezone handling, signature-filtered dispatch
2. **Add complexity**: `contacts_tool.py` - fuzzy resolution ladder, encryption, batch operations
3. **Learn infrastructure**: `tools/repo.py` - base class, DI, discovery, `invoke_tool` filtering
4. **Understand data**: `utils/userdata_manager.py` - database API, encryption internals
5. **Read the maps**: `tools/AGENTS.md` then `tools/implementations/AGENTS.md` - the contracts this guide walks through

**Infrastructure references:**
```text
tools/repo.py                   # Tool base class, ToolRepository, DI, ESSENTIAL_TOOLS
tools/registry.py               # ConfigRegistry: name -> Pydantic config class
utils/userdata_manager.py       # self.db API (create_table/insert/select/update/delete), encryption
utils/timezone_utils.py         # utc_now, format_utc_iso, parse_time_string, parse_utc_time_string,
                                # normalize_exact_local_wall_time, local_datetime_to_utc_iso
utils/user_context.py           # get_current_user_id, get_user_preferences, has_user_context
utils/user_credentials.py       # UserCredentialService (per-user secrets)
clients/vault_client.py         # system-level secrets
clients/llm_provider.py         # get_llm_provider() -- call with model_config='<route>'
config/config_manager.py        # config.<tool>_tool per-user config access
cns/core/events.py              # UpdateTrinketEvent and the rest of the event taxonomy
utils/power_on_self_test.py     # _check_tools: the boot gate your schema must pass
```

**Critical:** Deviating from established patterns causes integration issues and maintenance debt. Always check the Pattern Index first.

### Phase 4: Incremental Implementation

Build order matters:
1. **Config + registration** - `XxxToolConfig(BaseModel)` and module-level `registry.register("xxx_tool", XxxToolConfig)` (see `reminder_tool.py:ReminderToolConfig`)
2. **Tool structure** - `name`, `simple_description`, `tool_schema` (see `reminder_tool.py:ReminderTool`)
3. **Deferred init** - `has_user_context()` guard before table creation (see `reminder_tool.py:ReminderTool.__init__`)
4. **Database schema** - tables with indexes (see `reminder_tool.py:_ensure_reminders_table`)
5. **Basic CRUD** - one `_handler` per operation, dispatched from `run()` (see `contacts_tool.py:ContactsTool.run`)
6. **Advanced features** - search, batch, export as needed
7. **Verification** - execute the changed path against live infrastructure (see Verification below)

**Key implementation principles:**

- **Track progress**: Break a multi-operation tool into a task list and work it in order — schema before CRUD before advanced features
- **Deferred table creation**: Only create tables when user context exists (prevents startup failures)
- **Validate inputs**: Collect ALL errors before raising (see `contacts_tool.py:_add_contact`)
- **Log before raising**: Always log errors before propagating (see `reminder_tool.py:run`)
- **Consistent responses**: Use `{"success": bool, "message": str, ...}` format — include `"success": True` explicitly on success paths. (Some older tools, `reminder_tool` included, omit it on success; the explicit flag is the rule new tools follow, and `"success": False` is load-bearing — the tool loop treats it as a reported error)
- **Timezone everywhere**: Use `utc_now()` and `format_utc_iso()` for all timestamps
- **Encrypted fields**: Prefix sensitive data with `encrypted__` - access them the same way on read
- **Helpful errors**: Include suggestions and available options (see `reminder_tool.py:_get_reminder_not_found_error`)

### Phase 5: Resolving Design Decisions

Requirements arrive as metaphors. Every metaphor hides a set of decisions that must
be made explicitly before code. The pager tool is the worked example — the brief was
"like 90s pagers: when it goes off it matters, no spam, no noise", and each decision
below is visible in `tools/implementations/pager_tool.py`.

| Decision | Naive option | Why rejected | What shipped |
|---|---|---|---|
| Who may page you | Upfront allowlist | Painful to manage before you know who matters | Retroactive: first message from a new device lands in a review queue, approve/block once (`register_device`, `revoke_trust`) |
| Abuse friction | Captcha before send | External service dependency; an emergency shouldn't require solving a puzzle | Trust-on-first-use, no friction for trusted devices |
| Abuse friction (2) | Delay delivery 30s | Doesn't filter importance, just annoys; real emergencies need immediacy | Nothing — the review queue is the filter |
| Changed device | Silently block | People upgrade phones constantly; silent blocking looks like a bug | Fingerprint mismatch raises `PermissionError` and marks the trust row conflicted, so the surface can explain itself |
| Message length | Hard 300-char reject | Users have legitimate reasons to exceed it; rejection punishes verbosity | `max_message_length = 300`, over-length messages are AI-distilled on the `fast` route (`ai_distillation_enabled`, truncate fallback) — verbose is translated to concise, not blocked |
| Retention | Scheduled cleanup job | A scheduler for one table is more machinery than the problem needs | `cleanup_expired` operation; expiry checked on the paths that read/write |
| Status modelling | Severity levels | Binary worked/failed was sufficient; extra levels nobody acts on | Status enum only |

**How to run this phase:**

- Reject unsound proposals directly and say why — "that won't work, you'd need an external captcha service" is more useful than a diplomatic hedge. Follow the rejection with the working alternative.
- State the constraint that kills an option, not a preference. "A delay doesn't filter importance" ends the thread; "I'd rather not" doesn't.
- Uncertainty is fine and should be surfaced ("how short are we talking?"). Invented numbers are not — `300` came from the human, and it is a config field with a description, not a magic constant.
- Let the schema fall out of the decisions. Operations here are `send`/`approve`/`block` because the trust model is retroactive; a status enum and `device_fingerprint` exist because device change had to be explainable.
- Converge on the design neither party opened with. The brief said "pager"; the shipped tool is a TOFU trust system with distillation.

### Phase 6: Handling Mid-Implementation Feedback

Interruptions during tool use are course corrections, not annoyances:

```
[Request interrupted by user for tool use]
"You're setting the default expiry to 48 hours but that's too long
for a pager metaphor. These should be ephemeral - 24 hours max."
```

Parse for three things, in order:

| Signal | In the example | Action |
|---|---|---|
| Specific parameter correction | 48h → 24h | Change the value |
| Underlying philosophy mismatch | "ephemeral" is the design intent, not just a number | Re-check every other default against it (retention, queue depth, rate limits) |
| Missing requirement | Nothing stated expiry semantics for read vs unread | Ask before assuming |

A correction that only changes the named number, when the stated reason implies a
principle, will be corrected again later. Apply the principle.

## Technical Requirements

### User Scoping

**Every tool automatically gets user-scoped access via `self.db`** - no manual filtering needed.

**See `reminder_tool.py:_ensure_reminders_table`** for complete database patterns including:
- Table creation with proper schema
- Encrypted fields (use `encrypted__` prefix)
- Indexes for performance
- CRUD operations

**Key Points:**

1. **Automatic User Scoping**: All `self.db` operations are scoped to the current user
2. **Automatic Encryption**: Fields prefixed with `encrypted__` are encrypted on write, decrypted on read
3. **Prefix Retained on Read**: Access decrypted fields as `item['encrypted__title']` (prefix is kept)

**Example:**
```python
# Creating a table with encrypted fields
schema = """
    id TEXT PRIMARY KEY,
    encrypted__title TEXT NOT NULL,
    encrypted__notes TEXT,
    created_at TEXT NOT NULL
"""
self.db.create_table('my_items', schema)

# Insert - encryption happens automatically
self.db.insert('my_items', {
    'id': item_id,
    'encrypted__title': 'Secret Meeting',  # Will be encrypted
    'encrypted__notes': 'Confidential',     # Will be encrypted
    'created_at': timestamp
})

# Select - decryption happens automatically, prefix is KEPT
items = self.db.select('my_items')
# Returns: [{'id': '...', 'encrypted__title': 'Secret Meeting', 'encrypted__notes': 'Confidential', ...}]
#          ^^^^ Note: 'encrypted__title' NOT 'title'
```

**Encrypted columns are filter-opaque — you cannot `WHERE` on them.** `select(table, where, params)` interpolates filter params raw, but the stored value is Fernet ciphertext (randomized per write — the same plaintext never encrypts to the same bytes). `WHERE encrypted__url = :url` therefore matches nothing, ever. Filtering on an encrypted column means selecting broadly and filtering in Python after decryption, or storing a non-encrypted lookup twin (a plain hash column) alongside the ciphertext. Indexes on `encrypted__` columns are equally useless for the same reason — plan your queries around the plaintext columns you keep.

### Deferred Table Creation

Tools should only create tables when user context exists. This prevents startup failures during tool discovery:

```python
def __init__(self):
    super().__init__()
    self.logger = logging.getLogger(__name__)

    # Only create tables if user context is available (not during startup/discovery)
    from utils.user_context import has_user_context
    if has_user_context():
        self._ensure_tables()

def _ensure_tables(self):
    """Create tables if they don't exist."""
    schema = """
        id TEXT PRIMARY KEY,
        encrypted__data TEXT,
        created_at TEXT NOT NULL
    """
    self.db.create_table('my_data', schema)
```

Alternatively, ensure tables exist on first use in the `run()` method:

```python
def run(self, operation: str, **kwargs) -> Dict[str, Any]:
    # Ensure tables exist on first use
    self._ensure_tables()
    # ... rest of operation routing
```

### Credential Storage

Two sources, and picking the wrong one is a security defect:

| Credential belongs to | Source | Example |
|---|---|---|
| **The system** (one shared key MIRA owns) | Vault via `clients/vault_client.get_api_key("...")` | `web_tool`'s Kagi search key |
| **The user** (their own account/token) | `utils/user_credentials.py:UserCredentialService` | `imagegen_tool` (`api_key`/`google_genai`), `homeassistant_tool` (`api_key`/`home_assistant`) |

```python
from utils.user_credentials import UserCredentialService

cred_service = UserCredentialService()          # user from the contextvar
value = cred_service.get_credential('api_key', 'my_service')   # -> Optional[str]
if value is None:
    raise ValueError(
        "API key 'my_service' not found. Add it in Settings > API Credentials."
    )

cred_service.store_credential('api_key', 'my_service', secret)
meta = cred_service.get_credential_metadata('api_key', 'my_service')  # no value
```

`get_credential` returns `None` when absent — that is a genuine "not configured", not an infrastructure failure. Convert it to a raise with user-facing setup guidance; **never** fall back to an env var, a default, or a shared key. `email_tool` is the documented exception to the storage mechanism: it loads through `utils/tool_config_store.load_user_tool_config("email_tool", hydrate_secrets=True)`.

Never hardcode a secret and never read one from `os.environ`.

### Credential Injection (LLM-Invisible Authentication)

**See `web_tool.py:_get_credential` and `web_tool.py:_http`** for the reference implementation.

When your tool needs to make authenticated HTTP requests, you have two approaches:

#### Option 1: Use web_tool's Credential Injection (Recommended)

If your tool calls external APIs, leverage `web_tool`'s built-in credential injection. The LLM specifies credentials **by name only**—it never sees the actual values.

```python
# Declare the repository as a required constructor param; ToolRepository's
# DI injects itself by annotation when it instantiates your tool.
def __init__(self, tool_repo: ToolRepository):
    super().__init__()
    self.tool_repo = tool_repo

def _call_external_api(self, endpoint: str, credential_name: str) -> Dict[str, Any]:
    """Make authenticated API call using stored credential."""
    web_tool = self.tool_repo.get_tool("web_tool")

    return web_tool.run(
        operation="http",
        method="GET",
        url=f"https://api.example.com/{endpoint}",
        credential_name=credential_name,      # Name only - value retrieved server-side
        credential_header="Authorization",     # Which header to inject into
        credential_prefix="Bearer "            # Prefix for the value
    )
```

**Security benefits:**
- LLM sees: `credential_name="github_api"` (safe)
- LLM never sees: `ghp_xxxxxxxxxxxx` (the actual token)
- Response headers are sanitized—even if the API echoes credentials back, they're stripped before returning to the LLM

#### Option 2: Direct Credential Access (For Internal Use Only)

If your tool needs the credential value directly (e.g., for SDK initialization), retrieve it server-side:

```python
from utils.user_credentials import UserCredentialService

def _get_api_client(self):
    """Initialize API client with stored credential."""
    cred_service = UserCredentialService()
    api_key = cred_service.get_credential(
        credential_type="http_credential",
        service_name="my_service"
    )

    if api_key is None:
        raise ValueError(
            "API key 'my_service' not found. "
            "Add it in Settings > API Credentials."
        )

    return SomeAPIClient(api_key=api_key)
```

**CRITICAL:** Never return credential values to the LLM. If you use Option 2, ensure the credential value stays server-side and is never included in your tool's response dict.

#### Credential Type Conventions

| credential_type | Use Case |
|-----------------|----------|
| `http_credential` | Generic API keys for HTTP requests (used by web_tool) |
| `oauth_token` | OAuth access tokens |
| `oauth_refresh_token` | OAuth refresh tokens |
| `tool_config` | Tool-specific configuration blobs |

#### User Storage

Users store credentials via **Settings > API Credentials** in the web UI. The credentials are:
- Encrypted at rest (Fernet encryption, key derived from user_id)
- Stored in per-user SQLite databases
- Never exposed to the LLM—only referenced by name

### File Operations

**See the file helpers on `tools/repo.py:Tool`** (`make_dir`, `get_file_path`, `open_file`, `file_exists`).

Tools get automatic file methods that are user-scoped:
- `self.open_file(filename, mode)` - Open file in tool's user directory
- `self.get_file_path(filename)` - Get full path to file
- `self.file_exists(filename)` - Check if file exists
- `self.make_dir(path)` - Create subdirectory

**Example:**
```python
# Export data to JSON. The tool data directory is auto-created by
# get_tool_data_dir() (mkdir parents=True) — no defensive make_dir needed
# for files at the root of it.
filename = f"export_{utc_now().strftime('%Y%m%d_%H%M%S')}.json"
with self.open_file(filename, 'w') as f:
    json.dump(data, f, indent=2)

full_path = self.get_file_path(filename)
return {"success": True, "file_path": str(full_path)}
```

### Timezone Handling

**See `reminder_tool.py:_parse_date`** for model-supplied wall-time parsing and **`reminder_tool.py:_format_reminder_for_display`** for display conversion.

**Two different clocks, two different parsers.** Getting this wrong shifts every non-UTC user's intent by their whole offset.

| The timestamp came from | It means | Parse with |
|---|---|---|
| Your own storage | UTC | `parse_utc_time_string(s)` |
| **The model** (a tool parameter) | **The user's local wall time** | `normalize_exact_local_wall_time(s)` → `local_datetime_to_utc_iso(s, tz)`, or `parse_time_string(s, tz_name=...)` for relative phrases |

The only clock the model sees is the `<current_datetime>` trinket, rendered in the user's timezone. A model-supplied `2026-03-08T02:30:00` with no offset is the user naming a local moment, never UTC. Values carrying `Z` or an explicit offset are honoured as given.

**Store UTC, display local:**

```python
from utils.timezone_utils import (
    utc_now, format_utc_iso, parse_utc_time_string, convert_from_utc,
    format_datetime, parse_time_string, normalize_exact_local_wall_time,
    local_datetime_to_utc_iso, ensure_utc,
)
from utils.user_context import get_user_preferences

user_tz = get_user_preferences().timezone

# Store as UTC ISO strings (ALWAYS)
self.db.insert('items', {'created_at': format_utc_iso(utc_now())})

# Parse a MODEL-SUPPLIED exact wall time -- DST-strict, raises on ambiguity
wall = normalize_exact_local_wall_time(date_str)
if wall is not None:
    stored = parse_utc_time_string(local_datetime_to_utc_iso(wall, user_tz))

# Parse a MODEL-SUPPLIED relative phrase ("tomorrow", "in 3 weeks")
lenient = ensure_utc(parse_time_string(date_str, tz_name=user_tz))

# Parse your OWN stored value, convert to local for display only
stored_dt = parse_utc_time_string(item['created_at'])
display = format_datetime(convert_from_utc(stored_dt, user_tz), "date_time_short")
```

**Keep the DST-strict resolution outside the lenient fallback ladder.** `parse_time_string`'s `replace(tzinfo=...)` silently picks the first of two ambiguous instants and shifts nonexistent ones by an hour, and its broad `except` would replace the ambiguity message with a generic "Invalid date format" — the ambiguity message is exactly what the model needs in order to ask the user which time they meant. `reminder_tool.py:_parse_date` documents this ordering inline.

**One canonical stored format.** Compare timestamp strings in SQL only when every writer used the same serializer — store exclusively via `format_utc_iso()` so lexicographic comparisons (`due_at < :now`) hold. Mixed shapes (with/without milliseconds, with/without offset) compare incorrectly and fail silently, not loudly.

**Periods and ranges — the gap the instant parsers leave.** Neither `parse_time_string` nor dateutil resolves "last month", "September", or "2025-09" — month/period resolution is yours to build, in the **user's calendar**: take the month boundary in the user's timezone, then convert to UTC bounds. Conventions that hold up: a date-only end bound includes that whole day (exclusive next-midnight); a time-bearing end bound is the exact cutoff; dedupe invoices/periods on the *resolved UTC bounds*, not on the input string — "September", "2025-09", and "last month"-said-in-September must all resolve to the same key.

`get_user_preferences()` raises without user context. Catch `RuntimeError` → `"UTC"` only where a wrong label is cosmetic; never on a path that stores or compares an instant.

**Available timezone utilities (utils/timezone_utils.py):**
| Function | Use for |
|---|---|
| `utc_now()` | Current time. Never `datetime.now()` / `datetime.utcnow()`. |
| `format_utc_iso(dt)` | Storing a timestamp as an ISO 8601 string |
| `parse_utc_time_string(s)` | Parsing a timestamp **you** stored (UTC) |
| `parse_time_string(s, tz_name=...)` | Parsing a timestamp **the model** supplied (user-local wall time) |
| `normalize_exact_local_wall_time(s)` | DST-strict check on an exact wall time — ambiguous/nonexistent raises |
| `local_datetime_to_utc_iso(s, tz)` | Converting a validated local wall time to stored UTC |
| `ensure_utc(dt)` | Making a datetime UTC-aware |
| `convert_from_utc(dt, to_tz)` | UTC → user timezone, for display only |
| `convert_to_timezone(dt, tz)` | Converting between arbitrary timezones |
| `format_datetime(dt, style)` | Rendering for the user (`"date_time_short"`, `"time_short"`) |
| `format_relative_time(dt)` | "5 hours ago" phrasing |

### Code Organization

Tools don't require specific section markers, but consistency helps. Look at existing tools for organization patterns:

```text
# Standard tool structure
import statements
logging setup
configuration class (if needed)
tool class with:
    - metadata (name, descriptions)
    - tool_schema
    - __init__
    - run() method
    - operation handlers (_add_item, _get_items, etc.)
    - helper methods
```

### The Four-Piece Registration Flow

Adding a tool is exactly four coordinated pieces, in order. `tools/AGENTS.md` owns this contract; skipping a piece fails as shown.

| # | Piece | Where | Skipping it means |
|---|---|---|---|
| 1 | `XxxToolConfig(BaseModel)` with an `enabled` field, registered via `registry.register("xxx_tool", XxxToolConfig)` **at module level** | the implementation file | Your custom config fields silently do not exist — `Tool.__init__` fabricates an `enabled`-only default, and **that default is `True`: an unregistered tool is silently auto-ENABLED**, violating disabled-by-default. Skipping registration doesn't just lose fields; it ships the tool on |
| 2 | `Tool` subclass with `name`, `simple_description`, `tool_schema` class attributes | the implementation file | No `simple_description` drops the tool from the `invokeother_tool` catalog and enum, so it is unloadable at runtime unless it is in `ESSENTIAL_TOOLS` |
| 3 | `run()` dispatching on the `operation` param | the implementation file | — |
| 4 | The file inside `tools/implementations/` | — | Discovery never imports it; the class does not exist |

Then decide the tool's availability tier:

| Tier | How | Effect |
|---|---|---|
| Essential | Add `name` to `ESSENTIAL_TOOLS` in `tools/repo.py` | Always loaded at startup. This is an edit to `repo.py`, not to `implementations/` |
| Enabled by default | `enabled: bool = Field(default=True)` in the config | Loaded at startup via `enable_tools_from_config` |
| Opt-in | `enabled: bool = Field(default=False)` | Discovered but not loaded; surfaces through `invokeother_tool` (`load` = this turn only, `load_for_rest_of_session` = pinned). Not in `get_all_tool_definitions` until then — your probe script must `enable_tool(name)` or load via `invokeother_tool` first, or the tool is invisible to the LLM |
| Gated | `register_gated_tool(name)` in `cns/integration/factory.py:_register_gated_tools` + an `is_available()` method | Availability decided per call; `enable_tool()` raises on it |

`_register_gated_tools()` is currently an empty `pass` — no live gated tools. It is the hook, not a suggestion to call `register_gated_tool` from your own module.

Per-user config access is `config.<tool>_tool`, which merges the user's override fresh on every access over the global default; secret fields round-trip through a redaction sentinel (`config/config_manager.py`, `utils/tool_config_store.py`). Read it per call, never cache it at `__init__`.

### Tool Registration and Auto-Discovery

**See `reminder_tool.py:ReminderToolConfig`** for configuration examples.

**Auto-Discovery**: Place your tool in `tools/implementations/` and restart MIRA. `ToolRepository.discover_tools("tools.implementations")` imports every module in the package (skipping `_`-prefixed and `repo`) via `pkgutil.iter_modules()` and registers every concrete `Tool` subclass it finds (`_process_module`). A subclass in a module never imported there does not exist. Shared abstract bases inside `implementations/` must set `_is_abstract_base_class = True` to stay out of the catalog.

**Configuration (Optional)**:
```python
from pydantic import BaseModel, Field
from tools.registry import registry

class MyToolConfig(BaseModel):
    # Disabled by default — new tools opt in (see Contribute It Back).
    enabled: bool = Field(default=False, description="Whether enabled by default")
    max_items: int = Field(default=10, description="Max items to return")

# Register if you have custom config beyond 'enabled'
registry.register("my_tool", MyToolConfig)
```

If you don't register a config, `Tool.__init__` (tools/repo.py) builds one inline — it constructs a pydantic `create_model` class with `enabled: bool = True` and registers it directly via `registry.register` — so an unregistered tool is **auto-enabled**, the opposite of what a new tool wants. (`registry.create_default` exists but is only reached through `registry.get_or_create`, not through tool instantiation.) **A tool with custom config fields must register explicitly** or those fields silently do not exist — and the fabrication only wins the race if your module was imported before the config read. The reliable contract is always register-your-own, at module level. When probing the registry yourself: `get_or_create` returns the config **class**, not an instance — call it before instantiating.

### Tool Descriptions

**See `reminder_tool.py:ReminderTool`** for complete metadata examples.

Two required fields:
- `simple_description`: Ultra-concise action phrase (used by invokeother_tool for discovery)
- `tool_schema`: Full schema with `name`, `description`, and `input_schema`

### Tool Schema

**See `reminder_tool.py:ReminderTool.tool_schema`** for a complete schema with operations.

**Critical points:**
- Set `"additionalProperties": false` - prevents unexpected params
- Use `enum` for fixed options
- Clear descriptions - the model uses these to decide how to call the tool
- Mark required fields in `"required"` array
- Every static `tool_schema` is checked at boot by `utils/power_on_self_test.py:_check_tools` via `ToolDefinition.from_mapping()` — but that check validates the **envelope only** (top-level keys, non-empty name and description). `input_schema` internals — enums, `required`, `additionalProperties`, nested objects — pass through unvalidated and surface as model-side tool-call failures, not boot failures. A schema that boots is not thereby a correct schema
- A schema whose enums change with user data must be a `@property`, not a class dict — `domaindoc_tool.py:tool_schema` rebuilds a live label catalog at schema-read time

### The run() Contract

`ToolRepository.invoke_tool` filters and coerces the model's params **before** `run()` sees them. `tools/AGENTS.md` owns this contract; these are the consequences you must code against.

| Rule | Consequence for your `run()` |
|---|---|
| Keys not in `tool_schema.input_schema.properties` are dropped (debug-logged) | A param you forgot to declare is silently unreachable — not a runtime error, just never passed |
| Values are coerced to each property's declared type by `_coerce_to_schema_type` | `"10"` → `10` and `[10]` → `10` apply with a logged warning; irreparable values raise `ValueError` before your code runs |
| Schema-declared params reach **every** operation of the tool | Each handler must tolerate or filter extra kwargs. `reminder_tool.run` filters through `inspect.signature`; `continuum_tool` uses a `_SEARCH_PARAMS` allowlist |
| A `params` string is JSON-decoded, falling back to `{"query": ...}` | Don't name a param `params` unless you mean this |
| Identity params injected by callers (`thread_id`, `interface_name`, `agent_id`, `run_count` for `sidebar_tool`) are **not** in the schema | Never expose system-injected values as LLM-fillable params |

### The Return Envelope

What `run()` returns is interpreted by `cns/services/tool_loop.py`, not passed through blindly:

| You return | What happens |
|---|---|
| `dict` | `json.dumps`-ed into the tool result |
| `dict` with `"success": False` | Additionally treated as a **reported error** by the circuit breaker |
| `list` | Passed through verbatim as provider content blocks — only `imagegen_tool` does this (image + text blocks with a compressed replay copy) |
| `dict` with `_image_artifact` = `{file_id, alt_text}` | Consumed by `cns/services/orchestrator.py` to emit `![alt](/v0/api/images/{file_id})` into the user stream |
| raised exception | Propagates to the tool loop; log before raising so the cause is in server logs |

Do not invent other underscore-prefixed result keys without a consumer — they are a contract with `cns/services/`, not a naming convention.

Strip storage internals before returning: `encrypted__` field names are an at-rest detail, and `_format_contact` / `_format_reminder_for_display` exist to present clean field names to the model.

## Writing the Top-Level Tool Description

The `description` field in `tool_schema` is a tool-selection label. Its only job: the LLM glances at it and knows whether to pick up this tool. It is not an operation manifest, not defensive documentation, not a place for parameter guards or disambiguation infrastructure.

### What it is

Plain functional language. What does the tool do, stated simply.

**Good examples:**
- `"Search the web, fetch and extract webpage content, make HTTP requests to APIs."`
- `"Send and receive short messages between virtual pager devices."`
- `"Search past conversation history."`
- `"Search and manage long-term memories."`
- `"Create and manage scheduled reminders with contact linking."`

### What it is not

- Operation-by-operation documentation (the enum handles this)
- Parameter name corrections (parameter descriptions handle this)
- Disambiguation from other tools (system prompt handles this)
- Behavioral coaching or sequencing instructions (system prompt handles this)
- Consequence framing or failure mode prevention (parameter descriptions handle this)

### The one exception

If the tool has a behavioral constraint that is genuinely part of what the tool *does* — not a guardrail, but a core design property invisible from the enum and parameters alone — include it. Example:

> `"Generate and refine AI images. Results return to you only — the user sees the image when you publish it."`

The visibility model here isn't defensive documentation. It's what the tool does. Without it, the caller's mental model of the tool is fundamentally wrong.

### Rule of thumb

If you can describe what the tool does in under 15 words, do that. If you need more, ask whether the extra words describe what the tool *does* or what the caller *should do* — the latter belongs elsewhere.

---

## Writing tool_schema Parameter Descriptions

Parameter descriptions in `tool_schema` are interface contracts, not documentation. The reader is a language model that will infer behavior from your word choices — imprecise language causes real tool-call failures downstream. Every description must constrain behavior, not merely describe it.

### The Standard

Six attributes define a well-written parameter description:

**Precision-oriented compression.** Pack maximum behavioral information into minimum tokens. Word choice is functional, not aesthetic. "Literal string to match" prevents a regex assumption. "Exact" disambiguates matching behavior. Each word is load-bearing — if you can remove a word without losing a constraint, remove it.

**Caller-model empathy.** Before writing, ask: what will the model infer or assume from this description? What can go wrong if it reads this wrong? Drop internal jargon the caller has no context for. A description referencing "segment collapse" or "the overview section" means nothing to an LLM that has never seen your codebase — it will hallucinate what those mean and act on that hallucination.

**Implementation-grounded.** Read the code that consumes the parameter before writing its description. `str.replace()` means literal matching, not regex. A set equality check means exhaustive lists, not partial. A `max(0, min(value, 10))` clamp means the valid range is 0-10 regardless of what the caller passes. Describe actual code behavior, not intended behavior.

**Zero hedging.** State facts flatly. "Longer values are rejected." Not "may be rejected" or "could cause issues." Confidence is earned by reading the code — if you know what it does, say it.

**Terse failure-mode annotations.** When reviewing an existing description, the note identifies the specific failure the current wording enables. "A caller might assume truncation — code raises ValueError on overflow." Not an explanation of why bad descriptions are bad. Point at the gap, state the fix, move on.

**Contrastive proposals.** Generate at least three options, varying them along a real axis — specificity vs. brevity, implementation detail vs. caller-facing behavior. Description quality is empirical. You cannot know which phrasing an LLM will act on correctly without seeing alternatives side by side.

---

### Practical Standards

#### Parameter Naming

Names must match what the parameter actually contains. Every mismatch is cognitive overhead for the model — a reach toward a name that feels right is the root of a hallucination.

- **IDs:** Use `xxx_XXXXXXXX` format — 3-letter prefix + 8 hex chars. The prefix encodes entity type unambiguously: `mem_a1b2c3d4`, `rem_a1b2c3d4`, `msg_a1b2c3d4`. Never use dash-separated formats like `PAGER-XXXX` or bare hex strings.
- **Timestamps:** Suffix `_at` for moments in time (`happens_at`, `expires_at`, `created_at`). Suffix `_time` for search window boundaries (`start_time`, `end_time`). Never use bare `date` for a datetime field.
- **LLM instruction parameters:** Name them `instructions`, not `prompt`. "Instructions" signals directives to a downstream model. "Prompt" is ambiguous — the caller may assume it refers to the user's message.
- **Boolean filters:** Names should encode the filtering direction clearly. `unread_only` not `filter_unread`. The name should make the default state obvious.

#### Timestamps

MIRA operates in user-local time — UTC is a server implementation detail invisible at the tool interface. All timestamp descriptions and examples should use ISO 8601 in user-local time:

```
ISO 8601 (YYYY-MM-DDTHH:MM:SS)
```

Do not include the Z suffix. Do not say "UTC." The server handles timezone conversion internally.

For parameters that accept relative language alongside absolute timestamps, state both forms:

```
Accepts ISO 8601 (YYYY-MM-DDTHH:MM:SS) or relative phrases like 'tomorrow', 'in 2 days', 'next week'
```

Past-relative phrases apply to event occurrence timestamps (`happens_at`, `'6 months ago'`, `'last Tuesday'`). Future-relative phrases apply to expiry timestamps (`expires_at`, `'in 3 months'`). Reminder dates accept both directions naturally.

#### What to Omit

Remove parameters that:
- Are configurability-for-configurability's-sake — the default is always correct in practice
- Duplicate another parameter's function under a different name
- Expose implementation details the caller should never act on
- Can be handled automatically by the server (e.g., auth header names and prefixes when credential injection is already handling auth)

Every parameter you remove is one the model cannot get wrong.

#### Mutual Exclusivity vs. Co-dependency

**Mutually exclusive parameters** (providing both is an error) must be enforced in code — raise on both-provided — and declared in each description: "Mutually exclusive with X — providing both raises an error."

**Co-dependent parameters** (neither works without the other) must be declared in the schema using `dependentRequired`:

```json
"dependentRequired": {
  "temporal_direction": ["reference_time"],
  "reference_time": ["temporal_direction"]
}
```

The model reads `dependentRequired` as a schema-level constraint. Reinforce it in the description prose as well: "Ignored unless reference_time is also set."

#### Operation Scoping

When a parameter applies only to certain operations, say so explicitly at the end of the description: "Only for 'expand_message'." "Used by send_message and send_location." Omit this only when the parameter applies uniformly to all operations. A caller that doesn't know a parameter is scoped will set it on the wrong operation and expect it to work.

#### The One-Sentence Rule

Each description is one sentence. If you cannot fit the behavioral constraint, required co-parameters, valid range, and operation scope into one sentence — compress harder. Cut hedges. Cut filler. Cut historical context. Keep the constraint. If the sentence still won't close, the parameter is probably doing too much and should be split.

---

### Review Checklist

Before finalizing any parameter description:

- [ ] Read the code — does the description match actual validation, defaults, clamps, and rejection vs. truncation behavior?
- [ ] Would an LLM caller know exactly what value to pass here?
- [ ] Are co-dependencies named inline or enforced via `dependentRequired`?
- [ ] Are valid ranges, defaults, and format constraints stated?
- [ ] Is internal jargon absent?
- [ ] Is operation scope stated if the parameter is not universal?
- [ ] Does the name match what the parameter actually contains?
- [ ] Can any word be removed without losing a behavioral constraint?

---

### Database Operations Quick Reference

**See `utils/userdata_manager.py`** (`create_table`, `insert`, `select`, `update`, `delete`, `execute`, `fetchone`) for the full implementation.

```python
# Create table
schema = """
    id TEXT PRIMARY KEY,
    encrypted__title TEXT NOT NULL,
    created_at TEXT NOT NULL
"""
self.db.create_table('items', schema)

# Create indexes
self.db.execute("CREATE INDEX IF NOT EXISTS idx_items_created ON items(created_at)")

# CRUD operations
self.db.insert('items', {'id': id, 'encrypted__title': title, 'created_at': timestamp})
items = self.db.select('items', 'status = :status', {'status': 'active'})
self.db.update('items', {'encrypted__title': new_title}, 'id = :id', {'id': id})
self.db.delete('items', 'id = :id', {'id': id})

# Raw SQL for complex queries
results = self.db.execute("SELECT * FROM items WHERE created_at > :date", {'date': cutoff})
single = self.db.fetchone("SELECT * FROM items WHERE id = :id", {'id': id})
```

**Decryption split — the classic silent bug:** only `select()` decrypts `encrypted__` columns. `execute()` / `fetchone()` / `fetchall()` return rows verbatim, so any `encrypted__` field comes back as a Fernet ciphertext blob. Prefer `select()`; for raw SQL over encrypted tables, route rows through `db._decrypt_dict(row)` (the helper the tools themselves use). A ciphertext blob in a tool result fails loudly at the model — nothing warns you at fetch time.

### Quick Start Template

Start with this minimal structure, then add patterns from the index as needed:

```python
import inspect
import logging
import uuid
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import utc_now, format_utc_iso

logger = logging.getLogger(__name__)


class MyToolConfig(BaseModel):
    # New tools start DISABLED -- a contributor's tool must not enter
    # anyone's startup set uninvited (see Contribute It Back).
    enabled: bool = Field(default=False, description="Whether enabled by default")
    max_items: int = Field(default=50, description="Ceiling on items returned per call")


# Module level, not inside the class -- discovery imports the module and the
# config must be registered before anything reads it.
registry.register("my_tool", MyToolConfig)


class MyTool(Tool):
    name = "my_tool"
    simple_description = "does something useful"
    parallel_safe = True  # Set False if tool has ordering dependencies (create-then-edit)
    # Mutating tools with safe read ops use BOTH pieces together:
    #   parallel_safe = False  (writes stay sequential)
    #   _parallel_safe_operations = frozenset({"get_items"})
    #   @classmethod
    #   def is_call_parallel_safe(cls, tool_input):
    #       return tool_input.get("operation") in cls._parallel_safe_operations
    # The override fully replaces the base's `return cls.parallel_safe` —
    # without it, parallel_safe=False makes even the reads sequential.
    # For mixed read/write tools, override is_call_parallel_safe instead:
    # _parallel_safe_operations = frozenset({"search", "list"})
    # @classmethod
    # def is_call_parallel_safe(cls, tool_input):
    #     return tool_input.get("operation") in cls._parallel_safe_operations

    tool_schema = {
        "name": "my_tool",
        "description": "What this tool does and when to use it",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "get", "delete"],
                    "description": "Operation to perform."
                },
                "title": {
                    "type": "string",
                    "description": "Item title, 1-200 characters; longer values are rejected. Only for 'add'."
                },
                "notes": {
                    "type": "string",
                    "description": "Optional free-text detail stored encrypted at rest. Only for 'add'."
                },
                "item_id": {
                    "type": "string",
                    "description": "Item ID in itm_XXXXXXXX form as returned by 'add'. Omit on 'get' to list all items. Used by 'get' and 'delete'."
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    }

    def __init__(self, tool_repo: ToolRepository):
        # ToolRepository's DI injects itself by annotation when instantiating
        # the tool — no lookup call needed.
        super().__init__()
        self.tool_repo = tool_repo
        self.logger = logging.getLogger(__name__)

        # Only create tables if user context is available -- __init__ also runs
        # during discovery, where there is no user and self.db would raise.
        from utils.user_context import has_user_context
        if has_user_context():
            self._ensure_tables()

    def _ensure_tables(self):
        schema = """
            id TEXT PRIMARY KEY,
            encrypted__title TEXT NOT NULL,
            encrypted__notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        """
        self.db.create_table('my_items', schema)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_my_items_created ON my_items(created_at)")

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        try:
            # Ensure tables exist on first use
            self._ensure_tables()

            handlers = {
                "add": self._add,
                "get": self._get,
                "delete": self._delete,
            }
            method = handlers.get(operation)
            if method is None:
                raise ValueError(
                    f"Unknown operation: {operation}. Valid: {', '.join(handlers)}"
                )

            # invoke_tool passes EVERY schema-declared param to EVERY operation,
            # so a 'get' call can arrive carrying item_id AND title. Filter to
            # what the handler actually accepts (reminder_tool.run does this).
            accepted = set(inspect.signature(method).parameters)
            return method(**{k: v for k, v in kwargs.items() if k in accepted})
        except Exception as e:
            self.logger.error(f"Error in {operation}: {e}")
            raise

    def _add(self, title: str, notes: Optional[str] = None) -> Dict[str, Any]:
        if not title or not title.strip():
            raise ValueError("title is required for 'add' and must not be blank")
        if len(title) > 200:
            # The schema description promises rejection, not truncation -- do
            # what the description says or fix the description.
            raise ValueError(f"title exceeds 200 characters (got {len(title)})")

        item_id = f"itm_{uuid.uuid4().hex[:8]}"
        timestamp = format_utc_iso(utc_now())

        self.db.insert('my_items', {
            'id': item_id,
            'encrypted__title': title,
            'encrypted__notes': notes,
            'created_at': timestamp,
            'updated_at': timestamp
        })

        return {
            "success": True,
            "item": self._format_item({'id': item_id, 'encrypted__title': title,
                                       'encrypted__notes': notes,
                                       'created_at': timestamp}),
            "message": f"Created item: {title}"
        }

    def _get(self, item_id: Optional[str] = None) -> Dict[str, Any]:
        if item_id:
            items = self.db.select('my_items', 'id = :id', {'id': item_id})
            if not items:
                return {"success": False,
                        "message": self._not_found_error(item_id)}
            return {"success": True, "item": self._format_item(items[0])}

        items = self.db.select('my_items', order_by='created_at DESC')
        # Per-user config: read per call, never cached on self -- the merge in
        # config_manager is fresh on every access, and an instance attribute
        # would freeze the first user's override.
        from config.config_manager import config
        max_items = config.my_tool.max_items
        truncated = len(items) > max_items
        items = items[:max_items]
        return {
            "success": True,
            "items": [self._format_item(i) for i in items],
            "count": len(items),
            "truncated": truncated,
        }

    def _delete(self, item_id: str) -> Dict[str, Any]:
        if not item_id:
            raise ValueError("item_id is required for delete")

        items = self.db.select('my_items', 'id = :id', {'id': item_id})
        if not items:
            return {"success": False,
                    "message": self._not_found_error(item_id)}

        self.db.delete('my_items', 'id = :id', {'id': item_id})
        return {
            "success": True,
            "message": f"Deleted item: {items[0]['encrypted__title']}"
        }

    @staticmethod
    def _format_item(row: Dict[str, Any]) -> Dict[str, Any]:
        """Strip the at-rest encrypted__ prefixes before the model sees the row.

        Rows from self.db.select() are already DECRYPTED but keep the prefix in
        the field name. The prefix is a storage detail; the model should read
        'title', not 'encrypted__title'.
        """
        return {
            (k[len('encrypted__'):] if k.startswith('encrypted__') else k): v
            for k, v in row.items()
            if v is not None
        }

    def _not_found_error(self, item_id: str) -> str:
        """Recovery guidance, not just a failure -- the model can act on this."""
        items = self.db.select('my_items', order_by='created_at DESC')
        if not items:
            return f"Item '{item_id}' not found. No items exist yet."

        available = "\n".join(
            f"  - {i['id']}: {i['encrypted__title']}" for i in items[:5]
        )
        more = f"\n  ... and {len(items) - 5} more" if len(items) > 5 else ""
        return (f"Item '{item_id}' not found.\n\n"
                f"Available items ({len(items)} total):\n{available}{more}")
```

### API-Calling Tool Template

For tools that call external APIs with stored credentials:

```python
import logging
from typing import Dict, Any, Optional

from tools.repo import Tool, ToolRepository
from tools.registry import registry
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MyAPIToolConfig(BaseModel):
    # Disabled by default -- same rule as every new tool.
    enabled: bool = Field(default=False, description="Whether enabled by default")

registry.register("my_api_tool", MyAPIToolConfig)


class MyAPITool(Tool):
    """Tool that calls an external API using stored credentials."""

    name = "my_api_tool"
    simple_description = "fetches data from external API"

    tool_schema = {
        "name": "my_api_tool",
        "description": "Fetch data from ExampleAPI. Requires 'example_api' credential to be stored in Settings > API Credentials.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["get_user", "list_items"],
                    "description": "The operation to perform"
                },
                "user_id": {
                    "type": "string",
                    "description": "User ID for get_user operation"
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    }

    # Name of the credential users must store (documented in description)
    CREDENTIAL_NAME = "example_api"

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        if operation == "get_user":
            return self._get_user(kwargs.get("user_id"))
        elif operation == "list_items":
            return self._list_items()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _call_api(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make authenticated API call using web_tool's credential injection.

        The credential value is NEVER visible to the LLM - only referenced by name.
        """
        web_tool = self.tool_repo.get_tool("web_tool")

        return web_tool.run(
            operation="http",
            method=method,
            url=f"https://api.example.com/v1/{endpoint}",
            credential_name=self.CREDENTIAL_NAME,  # LLM sees this name only
            credential_header="Authorization",
            credential_prefix="Bearer ",
            **kwargs
        )

    def _get_user(self, user_id: Optional[str]) -> Dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required for get_user operation")

        result = self._call_api("GET", f"users/{user_id}")

        if not result.get("success"):
            return {
                "success": False,
                "message": f"API call failed: {result.get('message', 'Unknown error')}"
            }

        return {
            "success": True,
            "user": result.get("data"),
            "message": f"Retrieved user {user_id}"
        }

    def _list_items(self) -> Dict[str, Any]:
        result = self._call_api("GET", "items")

        if not result.get("success"):
            return {
                "success": False,
                "message": f"API call failed: {result.get('message', 'Unknown error')}"
            }

        items = result.get("data", [])
        return {
            "success": True,
            "items": items,
            "count": len(items),
            "message": f"Retrieved {len(items)} items"
        }
```

**Key points:**
- `CREDENTIAL_NAME` documents which credential users need to store
- `_call_api()` wraps web_tool with credential injection
- The LLM never sees the actual API key—only `credential_name="example_api"`
- Error handling passes through web_tool's response structure

### Error Handling

**See `reminder_tool.py:_get_reminder_not_found_error`** for helpful error patterns.

**Key principles:**
- Always log before raising: `self.logger.error(f"Error: {e}")` then `raise`
- Use `ValueError` for invalid input, `RuntimeError` for operation failures
- Provide helpful messages with suggestions
- Show available options when something isn't found

```python
def _get_item_not_found_error(self, item_id: str) -> str:
    """Generate helpful error message with available items."""
    items = self.db.select('my_items')

    error_msg = f"Item '{item_id}' not found."

    if not items:
        error_msg += " No items available."
    else:
        available = [f"  - {i['id']}: {i['encrypted__title']}" for i in items[:5]]
        error_msg += f"\n\nAvailable items ({len(items)} total):\n"
        error_msg += "\n".join(available)
        if len(items) > 5:
            error_msg += f"\n  ... and {len(items) - 5} more"

    return error_msg
```

### Dependency Injection

**See `tools/repo.py:ToolRepository.get_tool`** for how injection works, and `forage_tool.py` / `sidebaragents_tool.py` for live consumers.

Injection is **signature-driven and required-params-only**: `get_tool()` inspects `__init__` and fills a parameter only when it has **no default** and its annotation name is one of the three below.

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from working_memory.core import WorkingMemory
    from tools.repo import ToolRepository

class MyTool(Tool):
    def __init__(self, tool_repo: 'ToolRepository', working_memory: 'WorkingMemory'):
        super().__init__()
        self.tool_repo = tool_repo
        self.working_memory = working_memory
        self.event_bus = working_memory.event_bus if working_memory else None
```

| Annotation name | Injected value | Use for |
|---|---|---|
| `LLMProvider` or `LLMBridge` | `get_llm_provider()` | LLM calls on a named route |
| `ToolRepository` | the repository itself | Invoking other tools |
| `WorkingMemory` | the repo's `working_memory` (may be absent) | Publishing trinket updates |

**The `Optional[X] = None` form is NOT injected.** `phoneafriend_tool.__init__(self, llm_provider: LLMProvider | None = None)` has a default, so DI skips it and the tool falls back to `get_llm_provider()` at call time. Writing `working_memory: Optional["WorkingMemory"] = None` and expecting injection gives you a permanently `None` attribute and a tool that silently never refreshes its trinket. Use a required param when you need the dependency; use a defaulted one only when you have a real fallback.

Tool instances are **never cached** — `get_tool()` constructs a fresh instance per call under the current user context, so `__init__` runs often. Keep it cheap: no schema creation, no network, no LLM calls. Deferred table creation exists for exactly this reason.

If the repository was built without `working_memory`, a required `WorkingMemory` param cannot be satisfied and construction raises `TypeError`. The application factory always supplies it; ad-hoc `ToolRepository()` construction (as in a probe script) does not.

### LLM Calls From a Tool

For distillation, classification, or synthesis inside an operation (`pager_tool` distilling over-length messages, `web_tool` synthesizing long fetches, `phoneafriend_tool` consulting an outside model). Route by NAME on one of the five `model_configs` routes — never a model string, endpoint, or API key.

```python
from clients.llm_provider import get_llm_provider

# __init__: hold the provider (or use required-param DI — see above)
self.llm = get_llm_provider()

# inside an operation:
response = self.llm.generate_response(
    messages=[{"role": "user", "content": prompt}],
    model_config="fast",            # REQUIRED keyword — route name
    max_tokens=200,                  # per-request ceiling; set one for mechanical work
    # system_prompt=...,             # optional
)
text = self.llm.extract_text_content(response).strip()
if not text:
    raise RuntimeError("model returned an empty response for <operation>")
```

| Contract point | Reality |
|---|---|
| Import | `clients.llm_provider.get_llm_provider()` — or a **required** `LLMProvider` constructor param for DI |
| Call | `generate_response(messages, *, model_config, ...)` — `model_config` is a required keyword; a missing route name raises at `_resolve_selection` |
| Return | a `Result` object, **not** a dict — text comes out via `llm.extract_text_content(response).strip()` |
| Empty response | `extract_text_content` can return empty — check and raise (see `phoneafriend_tool`) |
| Failure | **propagates.** `LLMLifecycle` enforces provider timeouts and raises `ProviderStallError`; there is no fallback route. Wrap in `try` only to add context before re-raising, never to return a default |
| `ContextOverflowError` | raised on context overflow if you need a specific message (`clients.llm_provider`) |
| Route choice | high-frequency mechanical judgments on `fast`; semantic work stays on `primary`. Never `get_model_config()` by hand unless you need route metadata like the model name |

**References:** `pager_tool.py` (init-hold + `try`/extract/strip), `phoneafriend_tool.py` (defaulted DI param + `or get_llm_provider()` fallback — the sanctioned use of the Optional form), `web_tool.py` (synthesis with `focus` handling).

**Ordering when the LLM output is a garnish on a completed write.** Two sanctioned shapes, pick by what failure means. Pure distillation of a value you already have (pager's over-length message): wrap in `try` and fall back to truncation — the distilled text is cosmetic. Optional enrichment of a **persisted** write (an invoice note after the invoice is saved): persist the write first, then call; on LLM failure, raise an honest error stating the write succeeded and the garnish failed — never swallow, and never persist-half-then-rollback to "keep them atomic" when the write is the valuable part and the garnish is not.

The response content is model output — treat it as untrusted at any boundary that persists or renders it elsewhere.

### Gated Tools

**See `tools/repo.py:ToolRepository.register_gated_tool`** for gated tool registration.

Gated tools self-determine their availability via `is_available()` method:

```python
class MyGatedTool(Tool):
    name = "my_gated_tool"

    def is_available(self) -> bool:
        """Check if tool should appear in tool list."""
        # Example: only available if config file exists
        return self.file_exists("config.json")
```

Register in `cns/integration/factory.py:_register_gated_tools()` — that method is the single registration site and is currently an empty `pass` (no live gated tools):

```python
def _register_gated_tools(self) -> None:
    """Register tools that self-determine their availability via is_available()."""
    self._tool_repo.register_gated_tool("my_gated_tool")
```

Do not call `register_gated_tool` from your tool module — discovery imports the module before the repository exists.

Unlike regular enabled tools, gated tools:
- Cannot be enabled/disabled via `enable_tool()`/`disable_tool()` — `enable_tool()` raises
- Are checked at invocation time via `is_available()` (`invoke_tool`) and at listing time (`get_all_tool_definitions`)
- Automatically appear/disappear from the tool list based on state

Gated is for availability that genuinely varies at runtime. A tool that is simply off by default wants `enabled: bool = Field(default=False)` and `invokeother_tool`, not a gate.

### Config Validation (Optional)

**See `tools/repo.py:Tool.validate_config`** for the base pattern; `email_tool.py:validate_config` is the live override.

Tools that need custom validation (connection tests, auto-discovery) can override the `validate_config` classmethod. This is called by the `/actions/tools/{tool}/validate` API endpoint.

```python
@classmethod
def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and return discovered data.

    Args:
        config: The configuration dict to validate

    Returns:
        Dict with discovered data (e.g., {"folders": [...], "connection_test": "success"})

    Raises:
        ValueError: If validation fails (e.g., bad credentials, unreachable server)
    """
    server = config.get("server")
    password = config.get("password")

    if not server or not password:
        raise ValueError("Missing required fields: server, password")

    try:
        connection = connect_to_server(server, password)
        discovered_settings = connection.discover()
        return {
            "connection_test": "success",
            "discovered": discovered_settings
        }
    except ConnectionError as e:
        raise ValueError(f"Connection failed: {e}")
```

**Key points:**
- Use `@classmethod` - validation happens before tool instantiation
- Return discovered data that the frontend can use (folders, capabilities, etc.)
- Raise `ValueError` with helpful messages on failure - these are shown to users
- The base `Tool` class returns `{}` by default (no validation needed)

## Verification

🛑 **No mocks, no test files, no fixtures.** Verification is live or it does not count (root `AGENTS.md`). Type-clean code that has never executed is the standard failure product of this workflow and has shipped here before.

Pick the tier by behavioral surface touched, not by change size:

| Tier | When | Required |
|---|---|---|
| **0** | Docstrings, comments, formatting, verified-mechanical renames | `py_compile` + pyflakes on touched files |
| **1** | Bug fix, small logic change, contract-preserving refactor inside surface already covered | Execute the changed path once against live infrastructure; re-read the diff for silent degradation; end the report with EXECUTED / UNVERIFIED |
| **2** | New tool, new operation, new endpoint, schema change, failure-behavior or auth change, new dependency | Tier 1, plus round-trip persistence, plus a path-probe where the membership rule hits, plus an adversarial re-derivation for new-surface changes |

**Probe-surface membership (standing rule):** any path whose failure would report incorrect data to users, lose data, or degrade silently — reads, writes, searches, auth flows, failure paths. If a required probe cannot run against live infrastructure, fix the code until it can; do not simulate.

**The offline tier — what executes without the live stack.** Some build environments have no Vault, database, or model routes, and several real contracts route through all three (`config.<tool>_tool` reads, `get_user_preferences()`, and every `encrypted__` round-trip each depend on Vault). When live execution is unavailable, this is the floor — all of it real, none of it simulated:

- `py_compile` + pyflakes on every touched file
- Import your module directly (`ToolRepository._process_module('tools.implementations.my_tool')`) when full `discover_tools()` dies on an *unrelated* tool's missing optional dependency
- `ToolDefinition.from_mapping(Tool.schema)` — envelope validation, the boot gate's own check
- Registration confirmation via `registry.get('my_tool')` (note: `get_or_create` returns the config **class**, not an instance)
- Pure logic executed for real — parse/scale/settle/compute functions run against direct inputs; this is where offline probes earn the most (one portfolio tool caught a sign error in its settlement algorithm exactly this way)
- Every helper contract read from source, not assumed

Then report **UNVERIFIED** for everything above the floor and name the live probe that covers it. Offline-executed is not live-verified; conflating the two is the exact failure this section exists to prevent.

### What runs at boot

`utils/power_on_self_test.py` is the pre-server gate. `_check_tools` already covers every tool you add:

- `discover_tools()` imports your module — a syntax error or bad import fails the gate
- `ESSENTIAL_TOOLS` membership is verified in `tool_classes`, config-enabled, and present in `enabled_tools`
- Every **static** `tool_schema` is validated through `ToolDefinition.from_mapping()`; a malformed schema raises `RuntimeError` and the server never binds
- `@property` schemas are reported as `dynamic_schema_tools` and skipped — **a dynamic schema is not boot-verified**, so it needs an explicit live read

The gate runs in a subprocess and parks (sleeps forever) rather than exiting after `PRE_SERVER_GATE_ATTEMPTS` failed rounds, so a failing tool cannot become an unbounded loop of billed LLM probes.

### Executing your tool once

```bash
python3 -c "
from utils.user_context import set_current_user_id
from tools.repo import ToolRepository
set_current_user_id('<real-user-uuid>')
repo = ToolRepository(); repo.discover_tools(); repo.enable_all_tools()
print(repo.invoke_tool('my_tool', {'operation': 'add', 'title': 'probe'}))
print(repo.invoke_tool('my_tool', {'operation': 'get'}))
"
```

Going through `invoke_tool` rather than calling `run()` directly matters: it exercises the param filtering, type coercion, availability tier, and schema validation that production uses. Calling `run()` proves less than you think.

For persistence changes, round-trip against dev infrastructure — write, read back, verify, clean up. RLS and database constraints are part of the check.

### Re-read the diff against these invariants

- Does every failure path report the failure truthfully?
- Is any failure silently degraded (`except: return []` / `return None`)?
- Does the diff assert behavior it has not executed?
- Does a `[]` return mean "no data found" rather than "query failed"?

End the change report with a verification state: **EXECUTED** (what ran) or **UNVERIFIED** (why not, and which probe would cover it). A bug found in unprobed code is fixed together with the probe that covers it.

## Complete Checklist

**Registration (four pieces, in order)**
- [ ] `XxxToolConfig(BaseModel)` with `enabled`, registered at module level via `registry.register("xxx_tool", XxxToolConfig)`
- [ ] `Tool` subclass with `name`, `simple_description`, `tool_schema`
- [ ] `run()` dispatching on `operation`, tolerating extra kwargs
- [ ] File in `tools/implementations/`
- [ ] Availability tier chosen: `ESSENTIAL_TOOLS` / `enabled=True` / `enabled=False` / gated

**Schema and interface**
- [ ] `"additionalProperties": false`, `enum` for fixed options, `required` filled in
- [ ] Parameter descriptions written to the standard below (one sentence, implementation-grounded, no hedging, operation scope stated)
- [ ] Top-level `description` under ~15 words, describing what the tool *does*
- [ ] Co-dependent params declared via `dependentRequired`; mutually exclusive ones enforced in code *and* described
- [ ] Timestamp params described as user-local ISO 8601, no `Z`, no "UTC"
- [ ] IDs use the `xxx_XXXXXXXX` shape; `_at` for moments, `_time` for window bounds

**Data and scoping**
- [ ] All access via `self.db` / `self.user_data_path` — no manual `user_id` filtering
- [ ] Table creation behind `has_user_context()` or on first use in `run()`
- [ ] Sensitive columns declared with the `encrypted__` prefix; read back **with** the prefix
- [ ] No double-decryption of already-decrypted rows
- [ ] Model-supplied timestamps parsed as user-local wall time, DST-strict; stored as UTC
- [ ] `parallel_safe` / `is_call_parallel_safe` set deliberately for mutating operations

**Security**
- [ ] Credentials via `UserCredentialService` or `web_tool` `credential_name` injection — never env vars, never defaults, never returned to the model
- [ ] Missing credentials raise with user-facing setup guidance
- [ ] Untrusted content passed through `utils.prompt_injection_defense.wrap_untrusted(content, source)` before entering any return envelope; arguments JSON-Schema-validated
- [ ] Response headers/payloads sanitized of anything secret

**Failure behavior**
- [ ] Required-infrastructure failures propagate; no `except: return []`
- [ ] Log before raising; `ValueError` for bad input, `RuntimeError` for operation failure
- [ ] Not-found errors list available options
- [ ] Return envelope matches what `cns/services/tool_loop.py` expects

**Maps and verification**
- [ ] `tools/implementations/AGENTS.md` `## Files` bullet added in the same commit (or, when the work rides a branch, written in the branch and landing with the merge — a bullet in a worktree nobody merges is a bullet that never existed)
- [ ] `tools/AGENTS.md` updated if the change touched the framework contract
- [ ] Verified live per the tier above; report ends EXECUTED or UNVERIFIED

**Upstream (see Contribute It Back)**
- [ ] Broadly-useful judgment made explicitly, not skipped by default
- [ ] If generalizable: proposed a PR to the user and waited for an explicit yes
- [ ] If yes: fresh branch off `origin/main`, explicit paths staged, staged diff read for secrets and user data, `enabled=False` by default, not added to `ESSENTIAL_TOOLS`
- [ ] If local-only: said so in the change report and why

## Common Failure Patterns

| Pattern | Indicator | Resolution |
|---------|-----------|------------|
| Building without understanding | No clarifying questions asked | Stop coding, extract concrete requirements first (Phase 1) |
| Feature creep | Adding unrequested functionality | Return to core requirements |
| Over-engineering | Severity levels where binary worked/failed suffices | If you can't explain why it's needed, it isn't |
| Config fields that don't exist | Custom config without module-level `registry.register()` | Register explicitly; the fabricated default has only `enabled` |
| Tool missing from the catalog | No `simple_description` | Add it, or the tool is unloadable via `invokeother_tool` |
| Table creation at startup | Errors during tool discovery | `has_user_context()` guard, or create on first use in `run()` |
| Assuming prefix stripping | `item['title']` on an encrypted column | `item['encrypted__title']` — the prefix survives decryption |
| Double-decrypting | Garbled or raised on read of an already-decrypted row | `pager_tool._decrypt_row_safely` pattern |
| Wrong clock | Every non-UTC user's time shifted by their offset | Model input is user-local wall time — `parse_time_string(tz_name=...)`, not `parse_utc_time_string` |
| Silently picking a DST side | Ambiguous wall time resolved without complaint | `normalize_exact_local_wall_time()` outside the lenient ladder |
| Exposing credentials to the model | API key present in a tool response | `credential_name` reference; the value stays server-side |
| Handler crashes on a sibling op's param | `TypeError: unexpected keyword argument` | `invoke_tool` passes every schema-declared param to every operation — filter or tolerate |
| Infrastructure hedging | `try: db.query() except: return []` | Fail-fast; `[]` must mean "no data", never "query failed" |
| Unverified code | Clean types, never executed | Boot gate + live execution, per the tier table |
| Dead trinket target | Refresh published, nothing renders | `target_trinket` must be a registered class name (`punchclock_tool` used to publish to a trinket that did not exist; the dead refresh has since been removed) |

## Best Practices

1. **Survey before inventing** — models tend to add endpoints or patterns that already exist. Read the Pattern Index and the neighbouring tools first.
2. **Question assumptions** — initial requirements are rarely complete; extract concrete constraints from the metaphor.
3. **Reject unsound proposals directly** — then supply the working alternative.
4. **Integrate rather than invent** — use DI, validation, the event bus, the existing credential service.
5. **Simple solutions first** — the small fix, never at the cost of correctness. No "just in case" parameters.
6. **Defer table creation** — `has_user_context()` in `__init__`, or create on first use.
7. **Keep the encrypted prefix** — `item['encrypted__field']` on read.
8. **Never expose credentials to the model** — reference by name; the value stays server-side.
9. **Fix the caller, not the interface** — when code misuses a tool contract, the caller is wrong.
10. **Update the map in the same commit** — an unnecessary edit costs a line; a missed one misleads every later session.
11. **Verify live** — boot gate plus one real execution through `invoke_tool`.

## Summary

Tool building is a discovery process: the initial vision and the shipped tool differ because the decisions in Phase 5 were made explicitly rather than assumed. `pager_tool.py` is the record of one such pass — a "90s pager" brief that became a TOFU trust system with length distillation.

The mechanics are not the hard part. The four-piece registration flow, the schema contract, and the user-scoped `self.db` are all deterministic and documented above. What determines whether the tool works is (a) whether the design decisions were resolved before coding, (b) whether the parameter descriptions constrain the calling model precisely, and (c) whether the code was actually executed against live infrastructure before you called it done.

Related guides: `agents/HOW_TO_BUILD_AN_AGENT.md` (autonomous multi-step work), `working_memory/trinkets/HOW_TO_BUILD_A_TRINKET.md` (reflecting state in the system prompt).
