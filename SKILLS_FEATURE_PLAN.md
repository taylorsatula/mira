# Agent Skills Feature — Blind-Derivation Implementation Plan

**Provenance.** This plan was derived by a fresh subagent with no knowledge of the
in-session design work: it was given only the feature sketch (catalog in the stable
system prompt, an `invoke_skill` tool, activated skill bodies rendered via a trinket),
the four settled constraints (per-user storage under `data/users/{user_id}/`,
in-memory activation flushed on segment collapse, domaindoc-class trust for bodies,
no new dependencies), and the repo's own orientation material (root AGENTS.md, the
two HOW_TO guides). Every mechanism below was independently derived from source.
A second, earlier derivation reached the same design; the convergences and the
resolutions of the two divergences are recorded at the bottom.

---

## 0. Survey summary (what the codebase gives us)

- System prompt composition is trinket-driven: `cns/services/orchestrator.py:_compose_llm_messages()` publishes `ComposeSystemPromptEvent` every turn; `working_memory/core.py:_handle_compose_prompt()` broadcasts per-trinket updates; `working_memory/composer.py:compose()` routes by `SECTION_LAYOUT`. A trinket that re-reads its source on every `generate_content()` automatically reflects on-disk changes with no restart — the exact precedent is `asyncactivity_trinket.py`, which queries SQLite per render and holds no state.
- Placement slots: `composer.py:SECTION_LAYOUT` currently has a reserved-but-unexercised `tool_availability` slot in `PLACEMENT_SYSTEM`; `PLACEMENT_POST_HISTORY` holds only `domaindoc`.
- Tools: four-piece registration (`tools/HOW_TO_BUILD_A_TOOL.md`), auto-discovery via `tools/repo.py:ToolRepository.discover_tools`, DI of `WorkingMemory` into a tool constructor by annotation (`tools/repo.py`, the `annotation_name == 'WorkingMemory'` branch in `get_tool()`). Producer→trinket update precedent: `forage_tool.py:_publish_event` / Pattern 10 shape A (`working_memory.publish_trinket_update(target_trinket=<class name>, context=...)`).
- Per-user data root: `utils/userdata_manager.py:UserDataManager.base_dir` resolves `project_root/data/users/{user_id}` — the exact tree the settled constraint names. Note `get_user_data_manager(user_id: UUID)` takes a UUID; `domaindoc_trinket.py:generate_content` passes the contextvar value straight in, so that is the sanctioned call shape.
- Turn-scoped in-memory state that dies on segment collapse: `working_memory/trinkets/base.py:StatefulTrinket`, with `_clear_all_state()` invoked by `working_memory/core.py:_flush_stateful_trinkets()` on `SegmentCollapsedEvent`. This is precisely the required activation lifecycle — no custom event needed.
- YAML: **no YAML/frontmatter parsing exists anywhere in the tree** (grep found only a scratch file and a protected probe). PyYAML 6.0.2 is importable in the venv but is *not declared in requirements.txt* — it is transitive. Declaring it would arguably be a new dependency; see decision D1.

## 1. New files

### 1.1 `utils/skill_files.py` — shared skill-file access (no state)

Pure functions over `data/users/{user_id}/skills/*/SKILL.md`. Used by both the tool and the catalog trinket, so it lives in cross-cutting `utils/`, not in either consumer.

```python
SkillInfo = TypedDict("SkillInfo", {"name": str, "description": str})  # or a small pydantic model; TypedDict suffices (flat data, no validation surface beyond parse)

SKILLS_SUBDIR = "skills"  # under UserDataManager.base_dir
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-_]*$")

def parse_skill_md(text: str) -> SkillInfo:
    """Parse frontmatter. Raises ValueError with diagnostics on malformed input."""

def list_user_skills(user_id) -> dict[str, SkillInfo]:
    """Scan {base_dir}/skills/*/SKILL.md. Malformed SKILL.md files are skipped
    with logger.warning naming the directory; filesystem errors propagate."""

def load_skill(user_id, skillname: str) -> tuple[SkillInfo, str]:
    """Resolve skillname -> its directory via the scan (never string
    concatenation), return (frontmatter, markdown body). Raises ValueError
    listing available skill names when not found."""
```

How it works:

- `parse_skill_md` is a **hand-rolled flat-frontmatter parser**: file must start with `---\n`; a line `---` closes the frontmatter; within the block only `key: value` lines are accepted, `name` and `description` required, values optionally single- or double-quoted (quotes stripped), everything after the closing `---` is the body. `name` must match `NAME_RE`. Anything else → `ValueError` with the offending line quoted. Decision **D1**: hand-rolled over PyYAML because PyYAML is undeclared in `requirements.txt` and the settled constraint forbids new dependencies; the Agent-Skills frontmatter we must consume is this flat subset, and a strict parser makes malformed files loud rather than silently mis-parsed (Fail-Fast, Fail-Loud). Full YAML (multiline `>-`, anchors) is explicitly out of scope — a file using it raises and is skipped.
- `list_user_skills` iterates `sorted(base_dir/skills.iterdir())` for directories containing `SKILL.md`; resolution of a name to a directory happens through this scan result, so `invoke_skill("../foo")` can never match anything and `NAME_RE` rejects it before the lookup anyway — path traversal is structurally impossible, not filtered.
- `load_skill` re-reads the file at call time (fresh instance per tool call; no cache). Catalog freshness at compose time comes from `list_user_skills` being called per render.

### 1.2 `tools/implementations/skills_tool.py` — the `invoke_skill` tool

```python
class SkillsToolConfig(BaseModel):
    enabled: bool = Field(default=True, description=...)  # core feature; see note below
registry.register("skills_tool", SkillsToolConfig)

class SkillsTool(Tool):
    name = "skills_tool"
    simple_description = "Load an Agent Skill's instructions for the current task."

    def __init__(self, working_memory: "WorkingMemory"):   # REQUIRED param -> DI injects it (tools/repo.py get_tool)
        ...

    @property
    def tool_schema(self): ...   # dynamic enum of current skill names — domaindoc_tool.py:tool_schema precedent

    def run(self, operation: str, **kwargs) -> Dict[str, Any]: ...
    def _invoke_skill(self, skillname: str) -> Dict[str, Any]: ...
```

- `tool_schema`: one operation, `"invoke_skill"`, param `skillname` (string; enum = current skill names, rebuilt at schema-read time). `additionalProperties: false`. Parameter description states exact matching against the catalog names in `<skills_catalog>`.
- `_invoke_skill`: `load_skill(user_id, skillname)` → publish activation → return envelope:
  ```python
  self.working_memory.publish_trinket_update(
      target_trinket="ActiveSkillsTrinket",        # registered class name — forage_tool precedent
      context={"action": "activate", "skillname": name, "body": body},
  )
  return {"success": True, "message": f"Skill '{name}' loaded and active for this segment.",
          "skillname": name, "skill_body": body}
  ```
  The body is returned to the model (requirement: load AND return) *and* persisted in the prompt via the trinket, so it survives tool-result scroll-away.
- Not-found: `{"success": False, "message": ...}` listing available names (`reminder_tool.py:_get_reminder_not_found_error` precedent) — the circuit breaker treats `success: False` as a reported error, which is correct here: the model misnamed something that exists in its prompt.
- `working_memory` is a **required** constructor param — a defaulted `Optional` is deliberately skipped by DI (`tools/HOW_TO_BUILD_A_TOOL.md` Pattern 10; `forage_tool.py`'s `working_memory=None` is the shape that does *not* get injected).
- `enabled` default `True`: the feature is part of the product ask (catalog + tool are one surface); the disabled-by-default checklist item in `HOW_TO_BUILD_A_TOOL.md` is aimed at contributed/optional tools. If Taylor wants upstream-cautious defaults, flip to `False` and the catalog still renders (the trinket is not config-gated) — noted as the one open default.
- No SQLite table, no `_ensure_*_schema`, no `parallel_safe` override (activation is idempotent and read-only w.r.t. shared state).

### 1.3 `working_memory/trinkets/skills_catalog_trinket.py`

```python
class SkillsCatalogTrinket(EventAwareTrinket):
    variable_name = "skills_catalog"
    cache_policy = True

    def generate_content(self, context) -> str:
        skills = list_user_skills(get_current_user_id())   # live read per compose — asyncactivity precedent
        if not skills:
            return ""            # legitimately empty, logger.debug
        # render:
        # <skills_catalog>
        #   <skill name="..." description="..."/>   (html.escape both — user-controlled text into XML)
        # </skills_catalog>
        # plus one static instruction line: "When a task matches a skill's
        # description, call skills_tool invoke_skill with its exact name; the
        # skill's full instructions then persist in your system prompt."
```

- No state, no `handle_update_request` override: the compose broadcast drives it, and because it re-scans every compose, files added/removed while the server runs appear on the next turn with no restart and no cache-invalidation machinery.
- `cache_policy = True` (`domaindoc_trinket.py` precedent: user-curated, session-stable-in-practice content in `PLACEMENT_SYSTEM`). A file changed mid-session busts the provider prompt cache once — same trade domaindoc already makes.
- This content is model-facing prose: the instruction line is written to the LLM-caller-interface standard (literal, exact-name wording), per root AGENTS.md "LLM Caller Interface Design".

### 1.4 `working_memory/trinkets/active_skills_trinket.py`

```python
class ActiveSkillsTrinket(StatefulTrinket):
    variable_name = "active_skills"
    cache_policy = False

    def __init__(self, event_bus, working_memory):
        super().__init__(event_bus, working_memory)
        self._active: dict[str, dict[str, str]] = {}   # user_id -> skillname -> body

    def handle_update_request(self, event) -> None:
        if event.context.get("action") == "activate":
            self._active.setdefault(get_current_user_id(), {})[event.context["skillname"]] = event.context["body"]
        return super().handle_update_request(event)     # context FIRST, then super — Pattern 3

    def _expire_items(self) -> bool:
        return False        # no TTL — settled constraint; re-activation of the same name overwrites

    def _clear_all_state(self) -> None:
        self._active.clear()                          # segment collapse flush — the settled lifecycle

    def generate_content(self, context) -> str:
        bodies = self._active.get(get_current_user_id(), {})
        if not bodies:
            return ""
        # <active_skills>
        #   <skill name="...">html.escape(body)</skill>
        # </active_skills>
```

- `StatefulTrinket` gives the collapse flush for free: `working_memory/core.py:_flush_stateful_trinkets()` calls `_clear_all_state()` on `SegmentCollapsedEvent` — exactly the required "in-memory, per-user, flushed on collapse, no persistence, no TTL, no deactivate" semantics. `_expire_items` returning `False` encodes the no-TTL decision.
- The **body snapshot is taken at activation** (stored in the dict), not re-read per render. Rationale: `invoke_skill` means "load now"; a later file edit or delete mid-segment silently changing or blanking the prompt section would be drift, and re-reading raises the malformed-file-at-render-time failure class for no benefit.
- Bodies are `html.escape`-d at interpolation (user-curated trusted directives, same trust class and same escaping as `domaindoc_trinket.py` — **not** `wrap_untrusted`, per settled constraint).
- Placement: `PLACEMENT_SYSTEM`, `cache_policy=False` → lands in `non_cached_content`, so a mid-conversation activation changes only the uncached tail. Chosen over `PLACEMENT_POST_HISTORY` because activated skills are directives (persona-class), not reference material; domaindoc's post-history slot is for lookup content.

## 2. Existing files to touch

| File | Change |
|---|---|
| `working_memory/composer.py` | Add `'skills_catalog'` and `'active_skills'` to the `PLACEMENT_SYSTEM` list in `SECTION_LAYOUT` (after `tool_availability`, before `location_context` — stable orientation content). Without this both sections render at system-end with a warning (`working_memory/AGENTS.md` Rules). |
| `cns/integration/factory.py` | In `_get_working_memory()`: import and instantiate `SkillsCatalogTrinket(event_bus, self._working_memory)` and `ActiveSkillsTrinket(event_bus, self._working_memory)` alongside `DomaindocTrinket`. Construction self-registers; no attributes needed. |
| `tools/implementations/AGENTS.md` | `## Files` bullet for `skills_tool.py` (map maintenance rule, same commit). |
| `working_memory/trinkets/AGENTS.md` | `## Files` bullets for both trinkets. |
| `utils/AGENTS.md` | `## Files` bullet for `utils/skill_files.py`. |

No changes needed to: `tools/repo.py` (discovery is automatic; not adding to `ESSENTIAL_TOOLS` — config-default enablement is the right tier), `utils/power_on_self_test.py` (`_check_tools` picks up the new schema automatically), the base prompt (`config/system_prompt.txt` — catalog instructions live in the trinket content, not the base prompt), any schema/DDL (no SQLite table).

## 3. End-to-end data flow

**Catalog:** user drops `data/users/{uid}/skills/pdf-wrangling/SKILL.md` on disk → next turn, orchestrator composes the prompt → `ComposeSystemPromptEvent` → broadcast → `SkillsCatalogTrinket.generate_content()` → `utils/skill_files.list_user_skills(uid)` scans the directory live → catalog XML section → `composer.compose()` routes it into `cached_content` via `SECTION_LAYOUT` → `SystemPromptComposedEvent` → prompt sent. Files added/removed between turns appear on the next compose; no restart, no cache object, no event needed.

**Activation:** model reads `<skills_catalog>`, decides `pdf-wrangling` matches → tool call `skills_tool.invoke_skill(skillname="pdf-wrangling")` → `ToolRepository.invoke_tool` filters/coerces params → `SkillsTool.run` → `_invoke_skill` → `load_skill(uid, name)` resolves name via the directory scan, parses frontmatter, splits body → `publish_trinket_update(target_trinket="ActiveSkillsTrinket", context={action: activate, skillname, body})` → `ActiveSkillsTrinket.handle_update_request` stores body in `_active[uid]` **then** calls `super()` (render + `TrinketContentEvent` publish + Valkey persist) → section lands in `non_cached_content` from the next compose on → `run()` returns the envelope with `skill_body` to the model in the tool result this turn. On segment collapse: `SegmentCollapsedEvent` → `_flush_stateful_trinkets()` → `_clear_all_state()` → `_active` empty → `generate_content` returns `""` → `base.py:_clear_from_valkey` clears the stale Valkey field.

## 4. Failure / correctness surfaces

| Surface | Handling |
|---|---|
| Per-user isolation | All reads go through `UserDataManager.base_dir` for the contextvar user (`utils/userdata_manager.py:base_dir`); trinket activation state is a per-user dict keyed by `get_current_user_id()` (Pattern 4; cross-user leak is the documented singleton pitfall). No cross-user sharing — settled. |
| Malformed frontmatter | `parse_skill_md` raises `ValueError` with the offending line; `list_user_skills` catches **only that**, `logger.warning`s the directory, and skips the file — one bad skill never blanks the catalog or kills a turn. `load_skill` raises through the tool → `success: False` envelope listing valid names. Failures propagate to the tool loop; nothing returns `[]`/defaults silently. |
| Path traversal | `skillname` validated against `NAME_RE`; name→directory resolution happens through the filesystem scan, never string concatenation into a path; scan is rooted at `base_dir/skills`. A traversal-shaped argument simply matches nothing. |
| Empty catalog | `SkillsCatalogTrinket.generate_content` returns `""` with `logger.debug` — legitimately empty, composer strips it, no `<skills_catalog>` element appears. The model never sees a dangling instruction with no skills. |
| Segment collapse | `StatefulTrinket._clear_all_state()` via the existing central flush — activation dies with the segment, next segment re-activates by tool call. No persistence anywhere. |
| Missing user context in trinket | `get_current_user_id()` in `generate_content` — raise, per Pattern 7 (wiring bug, not empty state). |
| Skill file deleted while active | Body snapshot at activation → prompt section stays stable; catalog drops the entry on next compose. Deliberate. |
| Filesystem error mid-scan | `OSError`/`PermissionError` propagate from `list_user_skills` (infrastructure failure → `core.py:_handle_update_trinket` classifies and logs); not caught into `""`. |
| Config fabrication trap | `SkillsToolConfig` registered at module level — an unregistered tool is auto-*enabled* with a fabricated config (`tools/repo.py:Tool.__init__`), the documented trap. |
| `success: False` semantics | Only the not-found / malformed-file path uses it, so the circuit breaker sees genuine failures, not successes. |

## 5. Verification plan (Tier 2 — new behavioral surface, no mocks)

Gate everything with `tests/fixtures/live_infra.py:require` (or run on the dev instance at 192.168.1.9). No pytest suite, no mocks — probes in the `tests/tmp/` disposable pattern, promoted per the writing-probes skill:

1. **Boot gate:** `python -m utils.power_on_self_test pre-server` — validates the new `tool_schema` envelope via `_check_tools`.
2. **Catalog probe (live):** as a probe user, create `data/users/{uid}/skills/<name>/SKILL.md` (valid + one malformed file alongside), boot, run one real turn via the WebSocket/REST turn path, read back `working_memory.get_trinket_state("skills_catalog")` (Valkey round-trip is the HOW_TO Tier-2 requirement) and the composed prompt: valid skill listed, malformed file absent from catalog, `warning` in logs naming its directory.
3. **Invoke probe:** drive a turn whose message induces `skills_tool.invoke_skill` (or invoke the tool directly via `ToolRepository` with user context set — same live path): confirm envelope contains the body, `get_trinket_state("active_skills")` contains it, and the *next* turn's composed prompt carries `<active_skills>`.
4. **Live-mutation probe:** while the server is running, add a second skill directory and delete the first; next-turn catalog reflects both changes with no restart.
5. **Collapse probe:** trigger segment collapse (the collapse path in `cns/services/`) → confirm `_active` is empty and the Valkey `trinkets:{uid}` field for `active_skills` is cleared.
6. **Isolation probe:** second probe user with a different skill set — catalogs don't cross, activation of user A never renders for user B.
7. **Traversal probe:** `invoke_skill(skillname="../userdata")` → `success: False` listing real names; no file read outside the skills root.
8. Per Tier-2 doctrine: a second agent re-derives the diff before promotion.

## 6. What the codebase made easier / harder

**Easier:** the entire lifecycle fell out of existing machinery — `StatefulTrinket` *is* the "in-memory, per-user, flushed-on-collapse" activation store, verbatim; `asyncactivity_trinket.py`'s read-per-render pattern answers the no-restart catalog requirement with zero cache/invalidation code; `UserDataManager.base_dir` already names exactly the required storage tree; DI injection of `WorkingMemory` into the tool plus `publish_trinket_update` is a documented two-line producer shape with a working precedent in `forage_tool.py`. The reserved `tool_availability` slot in `SECTION_LAYOUT` shows system-placement catalog slots are an anticipated pattern.

**Harder:** (a) **no YAML parsing exists in the tree**, and PyYAML's presence is transitive/undeclared — the no-new-dependencies constraint forced a hand-rolled frontmatter parser, which is the one piece of genuinely new mechanism in this design (mitigated by keeping it a strict, loud, flat-subset parser); (b) the DI trap is real but asymmetric — `working_memory` must be a *required* constructor param or the tool silently gets `None` and every publish is a no-op (the `forage_tool.py` optional-param shape would have been the wrong copy); (c) cache placement required a deliberate choice: putting the catalog in cached content is the domaindoc precedent, but runtime file mutation means the "stable" prefix can change mid-session — accepted as a one-time cache bust, same trade domaindoc already makes; (d) the compose path degrades silently by design (`working_memory/AGENTS.md`) — a broken catalog trinket would ship a prompt with no skills section and only a log line, which is why the verification plan reads back both the Valkey field and the composed prompt rather than trusting absence of errors.

---

## Post-derivation status (added by the orchestrating session — not part of the blind output)

The earlier in-session derivation reached the same design independently. Two divergences were re-derived from source and resolved:

1. **Tool schema shape — resolved by Taylor's sketch.** This plan proposes `skills_tool` with an `operation` enum and a dynamic enum of skill names (the `domaindoc_tool` precedent). The orchestrating session recommended and Taylor's original sketch supports a **flat tool named `invoke_skill` with a free-string `skill_name` parameter** (the `feedback_tool` precedent): `input_schema` internals (enums) are not runtime-validated, a `@property` dynamic schema is not boot-verified, and the catalog section already carries the names in every prompt. Not-found errors list valid names for recovery. **Implement flat free-string** unless Taylor rules otherwise after reading this file.
2. **Active trinket `cache_policy` — this plan's choice stands.** `cache_policy=False` for `active_skills` is correct (adopted): `cache_policy` only routes `PLACEMENT_SYSTEM` sections, and a section that mutates on every activation belongs in `non_cached_content` so an invoke never busts the cacheable prefix.

Taylor has additionally ruled on catalog mutability: skills are rarely added/removed mid-run, and in the unlikely event one is, blowing the stable prefix is acceptable — so the catalog keeps `cache_policy=True` and no cache-invalidation machinery is warranted.

The full plan above is otherwise agreed and implementation-ready; no files have been created or modified for this feature yet.
