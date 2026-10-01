"""
Skill file access — the single sanctioned path for reading skills.

Two sources, two freshness doctrines:

- Global skills live in ``working_memory/skills/<name>/SKILL.md`` (repo
  content, deployed with the code). They are collected once at boot —
  frontmatter AND body — and served from that snapshot for process life:
  a global file edited or deleted while the server runs keeps serving what
  boot collected until the next restart, by design.
- Per-user skills live in ``data/users/{user_id}/skills/<name>/SKILL.md``
  (user-authored content). They are re-scanned on every catalog render, so
  adds and removes appear on the next turn with no restart — with one
  deliberate exception, the empty-directory negative cache in
  ``user_catalog()``.

A skill file is a flat YAML frontmatter block (``name``, ``description``)
followed by a markdown body. The frontmatter subset is parsed by hand —
deliberately strict and loud; files using YAML features beyond flat
``key: value`` pairs raise rather than silently mis-parse. Skill names are
resolved through scan results, never by building a path from caller-supplied
input, so path traversal is structurally impossible.
"""

import logging
import re
import shutil
from pathlib import Path
from typing import NamedTuple

from utils.userdata_manager import get_user_data_manager

logger = logging.getLogger(__name__)

# Agent-Skills name convention: lowercase slug, starts alphanumeric.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-_]*$")
NAME_MAX_LEN = 64

# Frontmatter keys we consume; others are parsed (flat value required) and ignored.
REQUIRED_KEYS = ("name", "description")
FRONTMATTER_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_REPO_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_SKILLS_DIR = _REPO_ROOT / "working_memory" / "skills"


class SkillRecord(NamedTuple):
    """One available skill."""
    name: str
    description: str
    path: Path       # the SKILL.md file the record came from
    source: str      # "user" (per-user dir) or "global" (boot-cached repo content)
    body: str = ""   # boot snapshot for global skills; user records re-read at load


class SkillNotFoundError(ValueError):
    """Raised when a skill name matches nothing in the user's catalog."""


class GlobalSkillReadOnlyError(SkillNotFoundError):
    """Raised when an action targets a global skill as if it were user-managed."""


# ── Process-life caches ────────────────────────────────────────────────────
# Both are module state by design: the global catalog is deploy content
# collected once at boot; the negative cache is a filesystem-cost
# optimization. Neither is conversation state and neither is ever cleared.

_global_records: list[SkillRecord] | None = None
_empty_user_dirs: set[str] = set()


def _strip_quotes(value: str) -> str:
    """Strip one layer of matching surrounding quotes, if present."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def parse_skill_md(text: str) -> tuple[dict[str, str], str]:
    """
    Parse a SKILL.md: flat frontmatter block, then the markdown body.

    Supported frontmatter is exactly flat ``key: value`` lines between two
    ``---`` delimiter lines. Nesting, lists, and empty values (``key:`` with
    nothing after the colon) are rejected — the Agent Skills files we consume
    need only this flat subset, and anything richer should fail loudly.

    Returns:
        (frontmatter dict, markdown body after the closing delimiter)

    Raises:
        ValueError: with the offending line, on any malformed input.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing opening '---' frontmatter delimiter on line 1")

    fields: dict[str, str] = {}
    closing = None
    for i, line in enumerate(lines[1:], start=2):
        if line.strip() == "---":
            closing = i
            break
        if not line.strip():
            raise ValueError(f"blank line inside frontmatter (line {i})")
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"line {i}: expected 'key: value', got: {line.strip()!r}")
        key = key.strip()
        if not FRONTMATTER_KEY_RE.match(key):
            raise ValueError(f"line {i}: invalid frontmatter key {key!r}")
        value = value.strip()
        if not value:
            # In real YAML this would open a nested block; we support only flat values.
            raise ValueError(f"line {i}: key {key!r} has no value (nested YAML is not supported)")
        fields[key] = _strip_quotes(value)

    if closing is None:
        raise ValueError("missing closing '---' frontmatter delimiter")

    for key in REQUIRED_KEYS:
        if key not in fields:
            raise ValueError(f"missing required frontmatter key {key!r}")
    if len(fields["name"]) > NAME_MAX_LEN:
        raise ValueError(f"name exceeds {NAME_MAX_LEN} characters")

    body = "\n".join(lines[closing:]).lstrip("\n")
    return fields, body


def _scan_skills_dir(skills_dir: Path, *, source: str) -> list[SkillRecord]:
    """
    Scan one skills directory: every subdirectory containing a SKILL.md.

    `source` ("user"/"global") is the record's identity, set at construction.
    Only global scans keep the parsed body — the boot snapshot is the global
    freshness doctrine; user bodies are re-read from disk at load time.

    Malformed SKILL.md files are skipped with a warning naming the directory —
    one bad file never blanks the catalog. Filesystem errors propagate.
    """
    records: list[SkillRecord] = []
    seen_names: set[str] = set()
    if not skills_dir.is_dir():
        return records

    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.is_file():
            continue
        try:
            fields, body = parse_skill_md(skill_md.read_text(encoding="utf-8"))
        except ValueError as e:
            logger.warning("Skipping malformed skill %s: %s", entry.name, e)
            continue
        name = fields["name"]
        if not NAME_RE.match(name):
            logger.warning("Skipping skill %s: name %r fails the name pattern", entry.name, name)
            continue
        if name in seen_names:
            logger.warning("Skipping skill %s: duplicate name %r (already listed)", entry.name, name)
            continue
        seen_names.add(name)
        records.append(SkillRecord(
            name=name,
            description=fields["description"],
            path=skill_md,
            source=source,
            body=body if source == "global" else "",
        ))

    return records


def load_global_catalog() -> list[SkillRecord]:
    """
    Collect the global skills catalog (``working_memory/skills/``) — once.

    Called at boot (factory wiring) and idempotent afterwards: the first call
    scans and snapshots frontmatter AND bodies for process life; later calls
    return the snapshot unchanged. A global skill file edited or deleted
    while the server runs keeps serving what boot collected until restart —
    global skills are deploy content, refreshed by redeploying, never live.
    """
    global _global_records
    if _global_records is not None:
        return _global_records
    _global_records = _scan_skills_dir(GLOBAL_SKILLS_DIR, source="global")
    logger.info("Global skills catalog collected at boot: %s",
                [r.name for r in _global_records] or "none")
    return _global_records


def user_catalog(user_id: str) -> list[SkillRecord]:
    """
    The user's own skills (bodies not read; the catalog is frontmatter only).

    Deliberate negative cache — and yes, it looks like a bug, so read this:
    a user whose skills directory is EMPTY on its first post-boot check is
    never re-checked for the life of the process. Most users will never
    author a skill, and without this cache every one of them pays a
    directory walk on every single compose so a minority can add skills
    live. The trade: a user who adds their FIRST skill after their first
    post-boot check will not see it until the server restarts. Skills are
    authored rarely and usually before a session, so that cost is accepted
    deliberately — do not "fix" this by re-checking empty directories. A
    directory that has EVER shown content is re-scanned every catalog
    render and stays fully live (adds/removes appear next turn).
    """
    if user_id in _empty_user_dirs:
        return []
    records = _scan_skills_dir(_user_skills_dir(user_id), source="user")
    if not records:
        _empty_user_dirs.add(user_id)
    return records


def _user_skills_dir(user_id: str) -> Path:
    manager = get_user_data_manager(user_id)
    return manager.get_skills_dir()


def catalog_for(user_id: str) -> list[SkillRecord]:
    """
    Merged catalog for a user: their own skills plus the boot-cached global
    catalog, user records winning on name collision (user-level customization
    overrides deploy defaults), sorted by name for prompt stability.
    """
    user_records = user_catalog(user_id)
    user_names = {r.name for r in user_records}
    merged = user_records + [r for r in load_global_catalog() if r.name not in user_names]
    return sorted(merged, key=lambda r: r.name)


def load_skill(user_id: str, skill_name: str) -> tuple[SkillRecord, str]:
    """
    Load a skill's record and body. Names resolve through scan results —
    a traversal-shaped argument simply matches nothing.

    User skills are re-read from disk at call time (live); global skills
    serve the boot snapshot (deploy-stable). Raises SkillNotFoundError
    naming the available skills when nothing matches.
    """
    for record in user_catalog(user_id):
        if record.name == skill_name:
            _, body = parse_skill_md(record.path.read_text(encoding="utf-8"))
            return record, body
    for record in load_global_catalog():
        if record.name == skill_name:
            return record, record.body
    available = ", ".join(r.name for r in catalog_for(user_id)) or "none"
    raise SkillNotFoundError(
        f"No skill named {skill_name!r}. Available skills: {available}."
    )


# ── Per-user write path (the API surface for the future skills UI) ─────────


def write_user_skill(user_id: str, name: str, description: str, body: str) -> SkillRecord:
    """
    Create a user skill: ``data/users/{user_id}/skills/<name>/SKILL.md``.

    Refuses a name that already exists as a USER skill (create semantics —
    shadowing a same-named global skill is legal and intended). The name is
    validated against NAME_RE before any path is built. A description
    containing newlines is refused — the flat frontmatter format cannot
    carry it, and writing a file we would then skip as malformed would be a
    silent write failure.

    Raises:
        ValueError: invalid name, multi-line description, or duplicate user skill.
    """
    name = name.strip()
    if not NAME_RE.match(name):
        raise ValueError(
            f"Invalid skill name {name!r}: lowercase slug required "
            f"(letters, digits, hyphens, underscores; starts alphanumeric)."
        )
    if len(name) > NAME_MAX_LEN:
        raise ValueError(f"Skill name exceeds {NAME_MAX_LEN} characters")
    if not description or not description.strip():
        raise ValueError("Skill description is required")
    if "\n" in description or "\r" in description:
        raise ValueError("Skill description must be a single line (flat frontmatter format)")

    if any(r.name == name for r in user_catalog(user_id)):
        raise ValueError(f"A user skill named {name!r} already exists")

    skills_dir = _user_skills_dir(user_id)
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: {description.strip()}\n---\n\n{body}",
        encoding="utf-8",
    )

    # The write may be this user's FIRST skill — their empty-dir negative cache
    # entry (user_catalog) is now stale and would hide the new skill until
    # restart. Discard it only after the write succeeded, so a failed write
    # leaves cache state truthful.
    _empty_user_dirs.discard(user_id)

    logger.info("User skill written: %s (user=%s)", name, user_id)
    return SkillRecord(name=name, description=description.strip(), path=skill_md, source="user")


def delete_user_skill(user_id: str, name: str) -> Path:
    """
    Delete a user's skill directory (``SKILL.md`` and any siblings).

    Raises:
        SkillNotFoundError: no such user skill — with a distinct message when
        the name belongs to a global skill (read-only repo content, deletable
        only by changing the repo, never through this path).
    """
    for record in user_catalog(user_id):
        if record.name == name:
            skill_dir = record.path.parent
            shutil.rmtree(skill_dir)
            logger.info("User skill deleted: %s (user=%s)", name, user_id)
            return skill_dir
    if any(r.name == name for r in load_global_catalog()):
        raise GlobalSkillReadOnlyError(
            f"'{name}' is a global skill (read-only repo content); only user skills can be deleted"
        )
    raise SkillNotFoundError(f"No user skill named {name!r}")
