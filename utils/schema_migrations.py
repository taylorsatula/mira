"""
Database step runner for ``deploy/migrations/``: discovery, ordering, the
ledger, structural fingerprints, validation, lint, status, and the consent
plan.

Every entry function that touches a database takes ``conninfo`` as a
required parameter. Only ``scripts/migrate.py`` resolves it (from Vault, as
``mira_admin``); probes pass a scratch database's conninfo. Nothing here
reads credentials.

Two vocabularies, never mixed: operator strings (``OPERATOR_MESSAGES``,
``Refusal.message``, ``render_plan``, ``render_drift_report``) are plain
English, each carrying a runnable command wherever one exists, and never
containing the words ledger, revision, checksum, migration, schema, fingerprint,
adoption, or baseline — an operator who reads jargon has no next step. The
technical detail behind each (step numbers, checksums, table names, hashes) goes
to the log line ``refusal()`` writes beside it.

``superuser_prefix()`` is this file's half of the superuser seam; the shell half
is ``deploy/lib/services.sh:seam_mode`` and the two must stay in step, or the
runner elevates one way while the deploy scripts elevate another.

A backup is complete when its ``manifest.json`` validates as a
``BackupManifest`` — that predicate is the completeness signal ``restore``,
``list_backups``, ``apply_retention`` and the residue sweeper all read, which is
why new manifest fields must be optional/defaulted and the manifest is written
last and fsynced.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal, TypedDict

import psycopg
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "deploy" / "migrations"
LEDGER_TABLE = "schema_migrations"
# libpq connect_timeout for a local server: 10 s is generous headroom — a
# server that cannot accept a local connection in 10 s is down, and callers
# refuse as blocked (exit 3) rather than hang.
CONNECT_TIMEOUT_SECONDS = 10

StepKind = Literal["versioned", "repeatable"]
StepStore = Literal["postgres", "vault", "env", "valkey"]
STEP_STORES: tuple[str, ...] = ("postgres", "vault", "env", "valkey")
Elevation = Literal["passwordless", "interactive", "none", "unavailable"]

_VERSIONED_RE = re.compile(r"V(\d{4})__([a-z0-9_]+)\.(sql|py)")
_REPEATABLE_RE = re.compile(r"R__([a-z0-9_]+)\.(sql|py)")
_SKIPPED_NAMES = frozenset({"BASELINE", "HEAD_FINGERPRINT.json", "AGENTS.md", "__pycache__"})
_SQL_HEADER_RE = re.compile(r"^--\s*(store|notice|REQUIRES_OPERATOR_INPUT|dialect-twins|expand-contract):\s*(.*)$")
_DIALECT_TWINS = ("DialectName", "DIALECT_NAMES")
_CONTRACT_RE = re.compile(r"^contract-for V(\d{4})$")


# Boot-refusal heads, shared by the wrapper-era ids and their -MANUAL twins
# (different remedy tail, same sentence): one home for the head so the two
# variants cannot drift apart. utils/power_on_self_test.py:_boot_refusal picks
# the id by wrapper presence — operator_message cannot know which install
# shape is reading it.
_BOOT_HEADS = {
    "S-BOOT-UNPREPARED": "MIRA's database has not been prepared for this version yet.",
    "S-BOOT-BEHIND": "MIRA's database is at step {ledger_head} and this version needs step {tree_head}.",
    "S-BOOT-EMPTY": (
        "MIRA's database is empty — it holds none of MIRA's tables (an update interrupted while "
        "putting your data back, a lost database volume, or a setup that never finished leaves it "
        "this way)"
    ),
}
_BOOT_MANUAL_MIGRATE_REMEDY = (
    "PYTHONPATH=. venv/bin/python -P -m scripts.migrate migrate --backups-dir ./backups --env-file ./systemone.env"
)
_BOOT_MANUAL_RESTORE_REMEDY = "PYTHONPATH=. venv/bin/python -P -m scripts.migrate restore --backups-dir ./backups"



OPERATOR_MESSAGES: dict[str, str] = {
    "S-NOTHING": "Nothing to do — your database is already ready for this version.",
    "S-STEP-DEFAULT": "This step changes MIRA's database structure. Nothing in your data is removed by it.",
    "S-PLAN-FIRST": (
        "This update includes a one-time upgrade to how MIRA stores its data. "
        "Your messages, memories, and settings are untouched."
    ),
    "S-ELEV-INTERACTIVE": "Your admin password may be asked for, once.",
    "S-ELEV-PASSWORDLESS": "No password is needed.",
    "S-ELEV-NONE": "No password is needed on this machine.",
    "S-PROTECT": (
        "MIRA backs up your data before it starts and puts your install back exactly as it was "
        "if anything goes wrong."
    ),
    "S-DOWNGRADE": (
        "Your database has already been upgraded past this version of MIRA (it is at step {ledger_head}; "
        "this version knows steps up to {tree_head}). MIRA will not move your data backwards, and nothing "
        "was changed. Install the newer version again: mira update"
    ),
    "S-CHECKSUM": (
        "Step {revision} ({slug}) was changed after MIRA ran it on your database, so MIRA cannot tell what "
        "state that part of your data is in. Nothing was changed. This is a fault in the release, not in "
        "your data — report it, then re-run: mira update"
    ),
    "S-OUTOFORDER": (
        "Step {revision} would have to run before steps MIRA has already run (your database is at step "
        "{ledger_head}). MIRA never runs steps out of order, and nothing was changed. Re-run: mira update"
    ),
    "S-DUPLICATE": (
        "Two steps in this release claim the same number ({revision}). This is a fault in the release; "
        "nothing was changed. Report it, then run: mira update"
    ),
    "S-GAP": (
        "This release has no step numbered {revision}. This is a fault in the release; nothing was changed. "
        "Report it, then run: mira update"
    ),
    "S-LINT": (
        "A step in this release breaks MIRA's rules for safe upgrades ({file}). This is a fault in the "
        "release; nothing was changed. Report it, then run: mira update"
    ),
    "S-UNREADABLE": (
        "A step in this release cannot be read ({file}). This is a fault in the release; nothing was "
        "changed. Report it, then run: mira update"
    ),
    "S-PAYLOAD": (
        "This release's own database-step files are malformed, so MIRA did not touch your database. "
        "This is a fault in the release, not your database; nothing was changed. Report it, then run: mira update"
    ),
    "S-PAYLOAD-POST": (
        "MIRA ran this release's steps, but could not run its final structure check because the "
        "release's own files are malformed. This is a fault in the release; the steps themselves "
        "completed. Report it"
    ),
    "S-DRIFT-LEAD": "Here is everything that is different about your database's structure:",
    "S-DRIFT": (
        "Your database's structure is not one this version of MIRA recognizes — every difference is listed "
        "above. Nothing was changed. If you made that change yourself and want to continue anyway, run "
        "exactly: mira update --adopt-drifted — on MIRA versions older than 2026.10.20-2.1 the typed "
        "flag does not reach the updater, so run the update script you just used with --adopt-drifted on "
        "the end instead"
    ),
    "S-DRIFT-ACCEPTED": (
        "As you asked, MIRA will keep any changes of yours that are listed above, record that you chose to "
        "continue, and upgrade around them. After the upgrade, your own changes remain."
    ),
    "S-BOOT-UNPREPARED": _BOOT_HEADS["S-BOOT-UNPREPARED"] + " Run: mira update",
    "S-BOOT-BEHIND": _BOOT_HEADS["S-BOOT-BEHIND"] + " Run: mira migrate from a terminal",
    "S-BOOT-EMPTY": _BOOT_HEADS["S-BOOT-EMPTY"] + ". Run: mira restore to put back a backup if one exists",
    # The -MANUAL twins: the remedy for an install without the mira wrapper
    # (a manual install, a bare checkout) — this tree's own scripts.migrate
    # spelling instead of a command-not-found.
    "S-BOOT-UNPREPARED-MANUAL": _BOOT_HEADS["S-BOOT-UNPREPARED"] + " Run: " + _BOOT_MANUAL_MIGRATE_REMEDY,
    "S-BOOT-BEHIND-MANUAL": _BOOT_HEADS["S-BOOT-BEHIND"] + " Run: " + _BOOT_MANUAL_MIGRATE_REMEDY,
    "S-BOOT-EMPTY-MANUAL": _BOOT_HEADS["S-BOOT-EMPTY"] + ". Run: " + _BOOT_MANUAL_RESTORE_REMEDY,
    "S-BOOT-AHEAD": (
        "MIRA's database is at step {ledger_head}, which is newer than this version of MIRA knows about "
        "(step {tree_head}). MIRA will not start on a database it would have to move backwards. Install the "
        "newer version: mira update"
    ),
    "S-BOOT-MISSING": (
        "This copy of MIRA is missing step {missing}, so it is incomplete. MIRA will not start on it. "
        "Reinstall the complete version: mira update"
    ),
    "S-NOT-INSTALLED": (
        "This database is empty — MIRA has not been installed into it yet. Nothing was changed. "
        "Install MIRA first: curl -fsSL https://raw.githubusercontent.com/{repo}/main/install.sh | bash"
    ),
}


def operator_message(message_id: str, **values: object) -> str:
    """The plain-English string for ``message_id`` with its placeholders filled (``{repo}`` from ``deploy/GITHUB_REPO``)."""
    from utils.release_identity import GITHUB_REPO

    return OPERATOR_MESSAGES[message_id].format(repo=GITHUB_REPO, **values)




@dataclass(frozen=True)
class Step:
    """One file in ``deploy/migrations/``, with its declared header."""

    revision: str
    slug: str
    kind: StepKind
    store: StepStore
    path: Path
    checksum: str
    requires_operator_input: str | None
    dialect_twins: tuple[str, ...]
    notice: str | None
    contract_for: str | None

    @property
    def ledger_key(self) -> str:
        """The ledger's ``revision`` value: the 4-digit number, or ``R__<slug>`` for a repeatable."""
        return self.revision if self.kind == "versioned" else f"R__{self.slug}"

    @property
    def number(self) -> int:
        if self.kind != "versioned":
            raise ValueError(f"repeatable step {self.path.name} has no number")
        return int(self.revision)


class StepOrderError(ValueError):
    """The chain's numbering is broken: a duplicate number or a gap."""

    def __init__(self, code: Literal["DUPLICATE_REVISION", "GAP"], revision: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.revision = revision


class StepFileError(ValueError):
    """A step file this release ships cannot be read: its name fails the grammar, its header is
    malformed, or its body does not parse. Raised at discovery, before any database is touched."""

    def __init__(self, file: str, detail: str):
        super().__init__(f"{file}: {detail}")
        self.file = file


@dataclass(frozen=True)
class OrderedChain:
    """The tree's steps: versioned in ascending number, repeatables by slug."""

    versioned: tuple[Step, ...]
    repeatable: tuple[Step, ...]

    @property
    def head(self) -> int:
        return self.versioned[-1].number if self.versioned else 0

    def by_revision(self) -> dict[str, Step]:
        return {step.revision: step for step in self.versioned}


def _sql_header(text: str, path: Path) -> dict[str, str]:
    """Header declarations from the leading ``--`` comment block; ``notice`` joins its continuation lines."""
    header: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("--"):
            break
        match = _SQL_HEADER_RE.match(stripped)
        if match:
            key, value = match.group(1), match.group(2).strip()
            if key in header:
                raise StepFileError(path.name, f"header field '{key}' declared twice")
            header[key] = value
            current = key
        elif current == "notice":
            continuation = stripped[2:].strip()
            if continuation:
                header["notice"] = f"{header['notice']} {continuation}"
        else:
            current = None
    return header


_PY_HEADER_NAMES = {
    "STORE": "store",
    "NOTICE": "notice",
    "REQUIRES_OPERATOR_INPUT": "REQUIRES_OPERATOR_INPUT",
    "DIALECT_TWINS": "dialect-twins",
    "EXPAND_CONTRACT": "expand-contract",
}


def _py_header(text: str, path: Path) -> dict[str, str]:
    """Header declarations from module-level literal assignments — the module is never imported here."""
    try:
        body = ast.parse(text, filename=str(path)).body
    except SyntaxError as error:
        raise StepFileError(path.name, f"does not parse ({error.msg}, line {error.lineno})") from error
    header: dict[str, str] = {}
    for node in body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name not in _PY_HEADER_NAMES:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError) as error:
            raise StepFileError(path.name, f"{name} is not a literal value ({error})") from error
        if isinstance(value, tuple):
            value = ", ".join(str(part) for part in value)
        if not isinstance(value, str):
            raise StepFileError(path.name, f"{name} must be a string literal")
        header[_PY_HEADER_NAMES[name]] = value
    return header


def _step_from_file(path: Path) -> Step | None:
    """A Step for a well-formed step file; None for a non-step file; StepFileError for a malformed one."""
    name = path.name
    if name in _SKIPPED_NAMES or not path.is_file():
        return None
    if not (name.startswith(("V", "R")) and name.endswith((".sql", ".py"))):
        return None
    versioned = _VERSIONED_RE.fullmatch(name)
    repeatable = _REPEATABLE_RE.fullmatch(name)
    if versioned:
        revision, slug, ext = versioned.group(1), versioned.group(2), versioned.group(3)
        kind: StepKind = "versioned"
    elif repeatable:
        revision, slug, ext = "", repeatable.group(1), repeatable.group(2)
        kind = "repeatable"
    else:
        raise StepFileError(
            name,
            "not a step file name (expected V<NNNN>__<slug>.sql|.py or R__<slug>.sql|.py, "
            "slug in [a-z0-9_])",
        )
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StepFileError(name, f"is not valid UTF-8 ({error.reason}, byte {error.start})") from error
    header = _sql_header(text, path) if ext == "sql" else _py_header(text, path)
    store = header.get("store")
    if store is None:
        raise StepFileError(name, "no store declared (`-- store: postgres` / `STORE = \"postgres\"`)")
    if store not in STEP_STORES:
        raise StepFileError(name, f"store '{store}' is not one of {', '.join(STEP_STORES)}")
    twins = tuple(part.strip() for part in header["dialect-twins"].split(",")) if "dialect-twins" in header else ()
    contract_for = None
    if "expand-contract" in header:
        match = _CONTRACT_RE.match(header["expand-contract"])
        if not match:
            raise StepFileError(name, "expand-contract must read 'contract-for V<NNNN>'")
        contract_for = match.group(1)
    return Step(
        revision=revision,
        slug=slug,
        kind=kind,
        store=store,  # type: ignore[arg-type]
        path=path,
        checksum=hashlib.sha256(raw).hexdigest(),
        requires_operator_input=header.get("REQUIRES_OPERATOR_INPUT"),
        dialect_twins=twins,
        notice=header.get("notice"),
        contract_for=contract_for,
    )


def discover_steps(migrations_dir: Path = MIGRATIONS_DIR) -> list[Step]:
    """
    Every step file in ``migrations_dir``.

    Raises:
        FileNotFoundError: The directory does not exist.
        StepFileError: A ``V``/``R`` ``.sql``/``.py`` file that fails the name grammar, or a step
            whose header is missing its store or malformed.
    """
    if not migrations_dir.is_dir():
        raise FileNotFoundError(f"migrations directory not found: {migrations_dir}")
    steps = []
    for path in sorted(migrations_dir.iterdir()):
        step = _step_from_file(path)
        if step is not None:
            steps.append(step)
    return steps


def order_steps(steps: list[Step]) -> OrderedChain:
    """
    Versioned steps in ascending order (dense from 0001) and repeatables by slug.

    Raises:
        StepOrderError: A duplicate number, a gap, or a duplicate repeatable slug.
    """
    versioned = sorted((s for s in steps if s.kind == "versioned"), key=lambda s: s.number)
    seen: dict[int, Step] = {}
    for step in versioned:
        if step.number in seen:
            raise StepOrderError(
                "DUPLICATE_REVISION", step.revision,
                f"steps {seen[step.number].path.name} and {step.path.name} share number {step.revision}",
            )
        seen[step.number] = step
    for expected, step in enumerate(versioned, start=1):
        if step.number != expected:
            raise StepOrderError("GAP", f"{expected:04d}", f"no step numbered {expected:04d} before {step.path.name}")
    repeatable = sorted((s for s in steps if s.kind == "repeatable"), key=lambda s: s.slug)
    slugs: dict[str, Step] = {}
    for step in repeatable:
        if step.slug in slugs:
            raise StepOrderError(
                "DUPLICATE_REVISION", step.ledger_key,
                f"repeatable steps {slugs[step.slug].path.name} and {step.path.name} share slug {step.slug}",
            )
        slugs[step.slug] = step
    return OrderedChain(versioned=tuple(versioned), repeatable=tuple(repeatable))




@dataclass(frozen=True)
class Origin:
    """A supported pre-mechanism release whose structure adopts at step 0000."""

    tag: str
    fingerprint_sha256: str


@dataclass(frozen=True)
class BaselineRow:
    """The structure after step ``revision`` has been applied."""

    revision: str
    fingerprint_sha256: str


@dataclass(frozen=True)
class Baseline:
    """``deploy/migrations/BASELINE``: origin fingerprints plus one row per step 0001..head."""

    origins: tuple[Origin, ...]
    rows: tuple[BaselineRow, ...]

    def match(self, fingerprint_sha256: str) -> str | None:
        """``"0000"`` for an origin, the revision of a matching row, None for an unknown structure."""
        if any(origin.fingerprint_sha256 == fingerprint_sha256 for origin in self.origins):
            return "0000"
        for row in reversed(self.rows):
            if row.fingerprint_sha256 == fingerprint_sha256:
                return row.revision
        return None


_BASELINE_ORIGIN_RE = re.compile(r"origin=(\S+) fingerprint_sha256=([0-9a-f]{64})")
_BASELINE_ROW_RE = re.compile(r"revision=(\d{4}) fingerprint_sha256=([0-9a-f]{64})")


def read_baseline(migrations_dir: Path = MIGRATIONS_DIR, chain_head: int | None = None) -> Baseline:
    """
    Parse ``BASELINE``.

    Raises:
        FileNotFoundError: No BASELINE file.
        ValueError: A malformed line, rows that are not dense from 0001, or a
            last row that is not ``chain_head`` (when given).
    """
    path = migrations_dir / "BASELINE"
    origins: list[Origin] = []
    rows: list[BaselineRow] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        origin = _BASELINE_ORIGIN_RE.fullmatch(line.strip())
        row = _BASELINE_ROW_RE.fullmatch(line.strip())
        if origin and not rows:
            origins.append(Origin(tag=origin.group(1), fingerprint_sha256=origin.group(2)))
        elif row:
            expected = f"{len(rows) + 1:04d}"
            if row.group(1) != expected:
                raise ValueError(f"BASELINE line {number}: revision {row.group(1)}, expected {expected}")
            rows.append(BaselineRow(revision=row.group(1), fingerprint_sha256=row.group(2)))
        else:
            raise ValueError(f"BASELINE line {number}: not 'origin=<tag> fingerprint_sha256=<hex>' "
                             f"(before rows) or 'revision=<NNNN> fingerprint_sha256=<hex>': {line!r}")
    last = int(rows[-1].revision) if rows else 0
    if chain_head is not None and last != chain_head:
        raise ValueError(f"BASELINE ends at revision {last:04d} but the chain head is {chain_head:04d}")
    return Baseline(origins=tuple(origins), rows=tuple(rows))


def read_head_fingerprint(migrations_dir: Path = MIGRATIONS_DIR, baseline: Baseline | None = None) -> dict:
    """
    The full structure at head from ``HEAD_FINGERPRINT.json`` (what drift reports diff against).

    Raises:
        FileNotFoundError: The file is absent.
        ValueError: Its hash disagrees with its content or with BASELINE's head row.
    """
    payload = json.loads((migrations_dir / "HEAD_FINGERPRINT.json").read_text())
    fingerprint = payload["fingerprint"]
    recorded = payload["fingerprint_sha256"]
    if fingerprint_sha256(fingerprint) != recorded:
        raise ValueError("HEAD_FINGERPRINT.json: fingerprint_sha256 does not match its fingerprint")
    baseline = baseline if baseline is not None else read_baseline(migrations_dir)
    head = baseline.rows[-1].fingerprint_sha256 if baseline.rows else None
    if head is not None and head != recorded:
        raise ValueError(f"HEAD_FINGERPRINT.json ({recorded[:12]}) differs from BASELINE's head row ({head[:12]})")
    return fingerprint


# The releases that predate the step chain and adopt at 0000 — a closed set: every
# later release carries its ledger, so a new origin is never added.
SCHEMA_ORIGINS = ("v2026.10.03-2.0", "v2026.10.06-2.0")
LEDGER_SENTINEL_BEGIN = "-- BEGIN GENERATED LEDGER DDL (utils/schema_migrations.py:ledger_ddl)"
LEDGER_SENTINEL_END = "-- END GENERATED LEDGER DDL"
SCHEMA_FILE = REPO_ROOT / "deploy" / "mira_service_schema.sql"
BOOTSTRAP_FILE = REPO_ROOT / "deploy" / "bootstrap_adopt.sql"
PSQL_TIMEOUT_SECONDS = 300


def ledger_ddl(if_not_exists: bool) -> str:
    """The ledger table and its runtime grant — the schema creates it, the bootstrap creates it when absent."""
    create = "CREATE TABLE IF NOT EXISTS" if if_not_exists else "CREATE TABLE"
    return (
        f"{create} {LEDGER_TABLE} (\n"
        "    revision    TEXT PRIMARY KEY,\n"
        "    name        TEXT NOT NULL,\n"
        "    kind        TEXT NOT NULL CHECK (kind IN ('versioned', 'repeatable')),\n"
        f"    store       TEXT NOT NULL CHECK (store IN ({', '.join(repr(s) for s in STEP_STORES)})),\n"
        "    checksum    TEXT NOT NULL,\n"
        "    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),\n"
        "    applied_by  TEXT NOT NULL,\n"
        "    duration_ms INTEGER NOT NULL DEFAULT 0\n"
        ");\n"
        f"GRANT SELECT ON {LEDGER_TABLE} TO mira_dbuser;\n"
    )


def _replace_sentinel_block(text: str, body: str, path: Path) -> str:
    begin, end = text.find(LEDGER_SENTINEL_BEGIN), text.find(LEDGER_SENTINEL_END)
    if begin < 0 or end < begin:
        raise ValueError(f"{path}: generated ledger sentinels missing or out of order")
    return text[:begin] + LEDGER_SENTINEL_BEGIN + "\n" + body + text[end:]


def write_generated_artifacts(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Rewrite the ledger sentinel blocks in the schema and the bootstrap; returns the files whose bytes changed."""
    changed: list[Path] = []
    for path, if_not_exists in ((repo_root / "deploy" / "mira_service_schema.sql", False),
                                (repo_root / "deploy" / "bootstrap_adopt.sql", True)):
        text = path.read_text()
        updated = _replace_sentinel_block(text, ledger_ddl(if_not_exists), path)
        if updated != text:
            path.write_text(updated)
            changed.append(path)
    return changed


def _psql_apply(conninfo: str, sql_file: Path, schema_args: list[str]) -> None:
    psql = shutil.which("psql")
    if psql is None:
        raise FileNotFoundError("psql is not on PATH (prepend deploy/lib/services.sh:pg_bindir)")
    result = subprocess.run([psql, "-X", "-q", "-v", "ON_ERROR_STOP=1", *schema_args, "-d", conninfo, "-f", str(sql_file)],
                            capture_output=True, text=True, timeout=PSQL_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise RuntimeError(f"psql -f {sql_file} exit {result.returncode}: {result.stderr.strip()[-2000:]}")


@contextmanager
def _installed_database(superuser: dict[str, str], schema_file: Path, schema_args: list[str]) -> Iterator[str]:
    """A throwaway database installed the way every installer does (schema, then bootstrap); yields mira_admin's conninfo."""
    name = f"mira_baseline_{os.getpid()}_{time.monotonic_ns()}"
    admin = psycopg.conninfo.make_conninfo(dbname="postgres", **superuser)
    created: list[str] = []
    with psycopg.connect(admin, autocommit=True) as conn:
        present = {row[0] for row in conn.execute("SELECT rolname FROM pg_roles WHERE rolname IN ('mira_admin', 'mira_dbuser')")}
        # The roles deploy/postgresql.sh Step 13 creates, minus passwords: a maintainer's server may have neither.
        for role, options in (("mira_admin", "LOGIN BYPASSRLS"), ("mira_dbuser", "LOGIN")):
            if role not in present:
                conn.execute(f"CREATE ROLE {role} {options}")
                created.append(role)
        conn.execute(f'CREATE DATABASE "{name}" OWNER mira_admin')
    try:
        target = psycopg.conninfo.make_conninfo(admin, dbname=name)
        _psql_apply(target, schema_file, schema_args)
        _psql_apply(target, BOOTSTRAP_FILE, [])
        yield psycopg.conninfo.make_conninfo(target, user="mira_admin")
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            for role in reversed(created):
                conn.execute(f"DROP ROLE IF EXISTS {role}")


class ConvergenceError(RuntimeError):
    """An origin plus the chain does not reach the fresh-install structure; ``report`` lists every difference."""

    def __init__(self, origin: str, report: str):
        super().__init__(f"{origin} + chain differs from the fresh-install structure:\n{report}")
        self.origin = origin
        self.report = report


@dataclass(frozen=True)
class GeneratedBaseline:
    text: str
    head_fingerprint: dict


def baseline_fingerprints(
    superuser: dict[str, str],
    schema_args: list[str],
    repo_root: Path = REPO_ROOT,
    origins: tuple[str, ...] = SCHEMA_ORIGINS,
    migrations_dir: Path = MIGRATIONS_DIR,
) -> GeneratedBaseline:
    """
    ``BASELINE`` and the head structure, built on real databases: each origin's
    tagged schema + today's bootstrap is fingerprinted (ledger excluded) and
    walked through every versioned step, fingerprinting after each; rows come
    from the newest origin's walk.

    Raises:
        ConvergenceError: Any origin + chain differs from the current schema +
            bootstrap — the release must not ship.
        ValueError: The regenerated text does not extend the checked-in BASELINE.
    """
    chain = order_steps(discover_steps(migrations_dir))
    with _installed_database(superuser, repo_root / "deploy" / "mira_service_schema.sql", schema_args) as current:
        expected = structural_fingerprint(current, include_ledger=False)
    origin_lines: list[str] = []
    rows: list[str] = []
    with tempfile.TemporaryDirectory() as work:
        for tag in origins:
            schema = Path(work) / f"{tag}.sql"
            schema.write_text(subprocess.run(["git", "-C", str(repo_root), "show", f"{tag}:deploy/mira_service_schema.sql"],
                                             capture_output=True, text=True, check=True).stdout)
            with _installed_database(superuser, schema, schema_args) as db:
                origin_lines.append(f"origin={tag} fingerprint_sha256={fingerprint_sha256(structural_fingerprint(db, include_ledger=False))}")
                walk: list[str] = []
                with connect(db) as conn:
                    ctx = StepContext(pg=conn, vault=VaultAdapter(), env=EnvFileAdapter(Path(work) / "systemone.env"),
                                      operator_input=_generator_input)
                    for step in chain.versioned:
                        # Only a postgres step changes structure; a vault/env/valkey step's row
                        # repeats the previous fingerprint and must not touch this machine's stores.
                        if step.store == "postgres":
                            with conn.transaction():
                                run_step(ctx, step)
                        walk.append(f"revision={step.revision} fingerprint_sha256="
                                    f"{fingerprint_sha256(_fingerprint_from(conn, include_ledger=False))}")
                    observed = _fingerprint_from(conn, include_ledger=False)
                if fingerprint_sha256(observed) != fingerprint_sha256(expected):
                    raise ConvergenceError(tag, "\n".join(_drift_items(observed, expected)))
                rows = walk
    existing = migrations_dir / "BASELINE"
    if existing.is_file():
        old_lines = existing.read_text().splitlines()
        old_rows = [line for line in old_lines if line.startswith("revision=")]
        old_origins = {line for line in old_lines if line.startswith("origin=")}
        if not old_origins <= set(origin_lines) or old_rows != rows[:len(old_rows)]:
            raise ValueError("regenerated BASELINE does not extend the checked-in file (rows are append-only)")
    return GeneratedBaseline(text="\n".join(origin_lines + rows) + "\n", head_fingerprint=expected)


def _generator_input(prompt: str) -> str:
    raise OperatorInputRequired(prompt)


def write_baseline(generated: GeneratedBaseline, migrations_dir: Path = MIGRATIONS_DIR) -> None:
    (migrations_dir / "BASELINE").write_text(generated.text)
    (migrations_dir / "HEAD_FINGERPRINT.json").write_text(json.dumps(
        {"fingerprint_sha256": fingerprint_sha256(generated.head_fingerprint), "fingerprint": generated.head_fingerprint},
        indent=1, sort_keys=True) + "\n")




class LedgerRow(TypedDict):
    revision: str
    name: str
    kind: str
    store: str
    checksum: str
    applied_at: str
    applied_by: str
    duration_ms: int


class DatabaseUnreachable(RuntimeError):
    """The database refused or timed out the connection (the caller exits 3, nothing changed)."""


def connect(conninfo: str, *, autocommit: bool = True) -> psycopg.Connection:
    """
    A connection with the runner's connect bound; the caller closes it.

    Raises:
        DatabaseUnreachable: The server is down, refuses, or does not answer
            within CONNECT_TIMEOUT_SECONDS.
    """
    try:
        return psycopg.connect(conninfo, autocommit=autocommit, connect_timeout=CONNECT_TIMEOUT_SECONDS)
    except psycopg.OperationalError as error:
        raise DatabaseUnreachable(str(error).strip()) from error


def ledger_present(conn: psycopg.Connection) -> bool:
    return conn.execute("SELECT to_regclass('public.schema_migrations') IS NOT NULL").fetchone()[0]


def user_tables_present(conn: psycopg.Connection) -> bool:
    """Whether the public schema holds any base table other than the ledger."""
    return conn.execute(
        """SELECT EXISTS (SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE' AND table_name <> %s)""",
        (LEDGER_TABLE,),
    ).fetchone()[0]


def read_ledger_rows(conn: psycopg.Connection) -> list[LedgerRow]:
    """
    Every ledger row, ordered by revision.

    Raises:
        psycopg.errors.UndefinedTable: The ledger does not exist.
    """
    cursor = conn.execute(
        """SELECT revision, name, kind, store, checksum, applied_at::text, applied_by, duration_ms
             FROM schema_migrations ORDER BY revision"""
    )
    keys = ("revision", "name", "kind", "store", "checksum", "applied_at", "applied_by", "duration_ms")
    return [LedgerRow(**dict(zip(keys, row))) for row in cursor.fetchall()]  # type: ignore[typeddict-item]


def ledger_head(rows: list[LedgerRow]) -> int:
    """The highest applied versioned revision (0 for none)."""
    return max((int(row["revision"]) for row in rows if row["kind"] == "versioned"), default=0)


def repeatable_due(rows: list[LedgerRow], steps: list[Step] | tuple[Step, ...]) -> list[Step]:
    """Repeatable steps whose file bytes differ from the ledger's record, or that have never run."""
    recorded = {row["revision"]: row["checksum"] for row in rows if row["kind"] == "repeatable"}
    return [s for s in steps if s.kind == "repeatable" and recorded.get(s.ledger_key) != s.checksum]


def pending(chain: OrderedChain, rows: list[LedgerRow]) -> list[Step]:
    """Versioned steps above the ledger head, then due repeatables."""
    head = ledger_head(rows)
    return [s for s in chain.versioned if s.number > head] + repeatable_due(rows, chain.repeatable)



# Grants are fingerprinted only for the roles MIRA defines; the superuser that
# created an older install's objects is named differently per platform
# (postgres, the Homebrew user) and must not make two identical structures differ.
_FINGERPRINT_GRANTEES = ("mira_admin", "mira_dbuser", "PUBLIC")
_VECTOR_DIM_RE = re.compile(r"vector\(\d+\)")


_CAST_RE = re.compile(r'::(?:"[^"]+"|[a-z_][a-z0-9_]*(?: [a-z_][a-z0-9_]*)*)(?:\(\d+(?:,\s*\d+)?\))?(?:\[\])?')


def _norm(text: str | None) -> str | None:
    if text is None:
        return None
    return _VECTOR_DIM_RE.sub("vector(<DIM>)", " ".join(text.split()))


def _norm_expr(text: str | None) -> str | None:
    """
    An expression's canonical form: casts, grouping parentheses and spaces removed.

    The server re-deparses some expressions differently after a dump/restore
    (``= ANY ((ARRAY['a'::character varying])::text[])`` comes back as
    ``= ANY (ARRAY[('a'::character varying)::text])``); without this, a restored
    install reads as structurally different. Names, literals, operators and
    their order still distinguish real changes.
    """
    if text is None:
        return None
    return re.sub(r"[()\s]", "", _CAST_RE.sub("", _norm(text)))


def _fingerprint_from(conn: psycopg.Connection, include_ledger: bool) -> dict:
    skip = () if include_ledger else (LEDGER_TABLE,)
    not_extension = """NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid = '{catalog}'::regclass
                                   AND d.objid = {oid} AND d.deptype = 'e')"""

    relations = conn.execute(
        f"""SELECT c.oid, c.relname, c.relkind, c.relrowsecurity, c.relforcerowsecurity
              FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'v', 'm', 'S')
               AND {not_extension.format(catalog='pg_class', oid='c.oid')}
             ORDER BY c.relname"""
    ).fetchall()
    relations = [r for r in relations if r[1] not in skip]
    table_oids = [r[0] for r in relations if r[2] in ("r", "p")]

    tables: dict[str, dict] = {}
    for oid, name, kind, rls, force_rls in relations:
        if kind not in ("r", "p"):
            continue
        columns = conn.execute(
            """SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull,
                      pg_get_expr(ad.adbin, ad.adrelid)
                 FROM pg_attribute a LEFT JOIN pg_attrdef ad ON ad.adrelid = a.attrelid AND ad.adnum = a.attnum
                WHERE a.attrelid = %s AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attname""",
            (oid,),
        ).fetchall()
        tables[name] = {
            "row_security": [rls, force_rls],
            "columns": {c[0]: [_norm(c[1]), c[2], _norm_expr(c[3])] for c in columns},
        }

    views = {
        r[1]: _norm_expr(conn.execute("SELECT pg_get_viewdef(%s, true)", (r[0],)).fetchone()[0])
        for r in relations if r[2] in ("v", "m")
    }
    sequences = sorted(r[1] for r in relations if r[2] == "S")
    # A sequence's structural attributes (pg_sequence: increment, min/max value, cache,
    # cycle; ownership as the OWNED BY table.column link) — the name list above reads as
    # "no drift" after ALTER SEQUENCE. No OIDs and no owning role: like the grants below,
    # the role varies per platform while the structure does not. Any change to what this
    # function records changes every computed fingerprint, so deploy/migrations/BASELINE
    # and HEAD_FINGERPRINT.json must be regenerated (scripts.migrate generate) in the same
    # release cut that ships the change.
    sequence_attributes = {
        row[0]: [int(row[1]), int(row[2]), int(row[3]), int(row[4]), row[5], row[6]]
        for row in conn.execute(
            """SELECT c.relname, s.seqincrement, s.seqmin, s.seqmax, s.seqcache, s.seqcycle,
                      -- the format literals double their percent signs: psycopg treats a bare one as a placeholder
                      coalesce(format('%%s.%%s', t.relname, a.attname), '')
                 FROM pg_sequence s
                      JOIN pg_class c ON c.oid = s.seqrelid
                      LEFT JOIN pg_depend d ON d.classid = 'pg_class'::regclass AND d.objid = s.seqrelid
                              AND d.refclassid = 'pg_class'::regclass AND d.refobjsubid > 0
                              AND d.deptype IN ('a', 'i')
                      LEFT JOIN pg_class t ON t.oid = d.refobjid
                      LEFT JOIN pg_attribute a ON a.attrelid = d.refobjid AND a.attnum = d.refobjsubid
                WHERE s.seqrelid = ANY(%s)
                ORDER BY c.relname""",
            ([r[0] for r in relations if r[2] == "S"],),
        ).fetchall()
    }

    indexes = {
        row[0]: _norm_expr(row[1])
        for row in conn.execute(
            """SELECT i.relname, pg_get_indexdef(i.oid)
                 FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid
                WHERE x.indrelid = ANY(%s) ORDER BY i.relname""",
            (table_oids,),
        ).fetchall()
    }
    # contype 'n' (NOT NULL) is a PG18+ catalog row PG17 does not write; nullability is
    # already recorded per column (a.attnotnull), so including 'n' rows would make the
    # same structure fingerprint differently across server versions.
    constraints = {
        f"{row[0]}.{row[1]}": _norm_expr(row[2])
        for row in conn.execute(
            """SELECT r.relname, c.conname, pg_get_constraintdef(c.oid)
                 FROM pg_constraint c JOIN pg_class r ON r.oid = c.conrelid
                WHERE c.conrelid = ANY(%s) AND c.contype <> 'n' ORDER BY r.relname, c.conname""",
            (table_oids,),
        ).fetchall()
    }
    policies = {
        f"{row[0]}.{row[1]}": [row[2], row[3], _norm_expr(row[4]), _norm_expr(row[5])]
        for row in conn.execute(
            """SELECT r.relname, p.polname, p.polcmd::text,
                      ARRAY(SELECT CASE WHEN o = 0 THEN 'PUBLIC' ELSE pg_get_userbyid(o) END
                              FROM unnest(p.polroles) o ORDER BY 1),
                      pg_get_expr(p.polqual, p.polrelid), pg_get_expr(p.polwithcheck, p.polrelid)
                 FROM pg_policy p JOIN pg_class r ON r.oid = p.polrelid
                WHERE p.polrelid = ANY(%s) ORDER BY r.relname, p.polname""",
            (table_oids,),
        ).fetchall()
    }
    triggers = {
        f"{row[0]}.{row[1]}": _norm_expr(row[2])
        for row in conn.execute(
            """SELECT r.relname, t.tgname, pg_get_triggerdef(t.oid)
                 FROM pg_trigger t JOIN pg_class r ON r.oid = t.tgrelid
                WHERE t.tgrelid = ANY(%s) AND NOT t.tgisinternal ORDER BY r.relname, t.tgname""",
            (table_oids,),
        ).fetchall()
    }
    functions = {
        f"{row[0]}({row[1]})": [row[2], row[3], row[4], row[5]]
        for row in conn.execute(
            f"""SELECT p.proname, pg_get_function_identity_arguments(p.oid), pg_get_function_result(p.oid),
                       p.prosecdef, md5(p.prosrc), coalesce(array_to_string(p.proconfig, ','), '')
                  FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
                 WHERE n.nspname = 'public' AND {not_extension.format(catalog='pg_proc', oid='p.oid')}
                 ORDER BY 1, 2"""
        ).fetchall()
    }

    # An object owner's privileges are recorded as mira_admin's: bootstrap_adopt.sql makes
    # mira_admin the owner of everything, and ALTER ... OWNER moves exactly those privileges,
    # so a pre-adoption database and the same structure after adoption compare equal.
    grants: dict[str, list[str]] = {}
    for rel_name, grantee, privilege in conn.execute(
        """SELECT c.relname, CASE WHEN a.grantee = 0 THEN 'PUBLIC'
                                  WHEN a.grantee = c.relowner THEN 'mira_admin'
                                  ELSE pg_get_userbyid(a.grantee) END,
                  a.privilege_type
             FROM pg_class c, aclexplode(coalesce(c.relacl, acldefault(
                      CASE WHEN c.relkind = 'S' THEN 's' ELSE 'r' END::"char", c.relowner))) a
            WHERE c.oid = ANY(%s)""",
        ([r[0] for r in relations],),
    ).fetchall():
        if grantee in _FINGERPRINT_GRANTEES:
            grants.setdefault(f"{rel_name} {grantee}", []).append(privilege)
    for rel_name, column, grantee, privilege in conn.execute(
        """SELECT c.relname, at.attname, CASE WHEN a.grantee = 0 THEN 'PUBLIC' ELSE pg_get_userbyid(a.grantee) END,
                  a.privilege_type
             FROM pg_attribute at JOIN pg_class c ON c.oid = at.attrelid, aclexplode(at.attacl) a
            WHERE at.attrelid = ANY(%s) AND at.attacl IS NOT NULL""",
        (table_oids,),
    ).fetchall():
        if grantee in _FINGERPRINT_GRANTEES:
            grants.setdefault(f"{rel_name}.{column} {grantee}", []).append(privilege)
    for signature, grantee, privilege in conn.execute(
        f"""SELECT p.proname || '(' || pg_get_function_identity_arguments(p.oid) || ')',
                   CASE WHEN a.grantee = 0 THEN 'PUBLIC'
                        WHEN a.grantee = p.proowner THEN 'mira_admin'
                        ELSE pg_get_userbyid(a.grantee) END, a.privilege_type
              FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace,
                   aclexplode(coalesce(p.proacl, acldefault('f', p.proowner))) a
             WHERE n.nspname = 'public' AND {not_extension.format(catalog='pg_proc', oid='p.oid')}"""
    ).fetchall():
        if grantee in _FINGERPRINT_GRANTEES:
            grants.setdefault(f"{signature} {grantee}", []).append(privilege)
    grants = {key: sorted(set(value)) for key, value in sorted(grants.items())}

    schema_privileges = sorted(
        f"{grantee} {privilege}"
        for grantee, privilege in conn.execute(
            """SELECT CASE WHEN a.grantee = 0 THEN 'PUBLIC' ELSE pg_get_userbyid(a.grantee) END, a.privilege_type
                 FROM pg_namespace n, aclexplode(coalesce(n.nspacl, acldefault('n', n.nspowner))) a
                WHERE n.nspname = 'public'"""
        ).fetchall()
        # Only mira_dbuser's schema rights are install state: the installer's own
        # GRANT USAGE (deploy/mira_service_schema.sql). PUBLIC's schema rights are
        # cluster-template state (a PG15+ fresh initdb revokes CREATE on public; a
        # cluster carried forward from before keeps it) that no release step or
        # installer sets, so recording them would make the same install fingerprint
        # differently across clusters; mira_admin's follow from database ownership.
        if grantee == "mira_dbuser"
    )
    extensions = sorted(row[0] for row in conn.execute("SELECT extname FROM pg_extension WHERE extname <> 'plpgsql'"))

    return {
        "tables": tables,
        "views": views,
        "sequences": sequences,
        "sequence_attributes": sequence_attributes,
        "indexes": indexes,
        "constraints": constraints,
        "policies": policies,
        "triggers": triggers,
        "functions": functions,
        "grants": grants,
        "schema_privileges": schema_privileges,
        "extensions": extensions,
    }


def structural_fingerprint(conninfo: str, *, include_ledger: bool) -> dict:
    """
    The database's structure, read from ``pg_catalog`` in a fixed key order.

    Covers tables (columns with type, nullability, default; row security),
    views, sequences (names plus each one's pg_sequence attributes: increment,
    min/max value, cache, cycle, OWNED BY link), index and constraint definitions, policies, triggers,
    functions (identity args, result, security definer, body hash, settings),
    grants to MIRA's roles and PUBLIC (tables, columns, functions, sequences)
    and on the public schema, and extensions. Normalized so that two
    structurally identical databases agree: no OIDs, no owners, no grantors,
    no superuser grants, ``vector(N)`` written ``vector(<DIM>)``, whitespace
    collapsed.
    """
    with connect(conninfo) as conn:
        return _fingerprint_from(conn, include_ledger)


def fingerprint_sha256(fingerprint: dict) -> str:
    """sha256 of the fingerprint's canonical JSON."""
    return hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


_DRIFT_SECTION_WORDS = {
    "tables": "table",
    "views": "view",
    "sequences": "sequence",
    "sequence_attributes": "sequence attribute",
    "indexes": "index",
    "constraints": "rule",
    "policies": "access rule",
    "triggers": "trigger",
    "functions": "function",
    "grants": "permission",
    "schema_privileges": "database-wide permission",
    "extensions": "extension",
}


def _drift_items(observed: dict, expected: dict) -> list[str]:
    lines: list[str] = []
    for section, word in _DRIFT_SECTION_WORDS.items():
        seen, want = observed.get(section), expected.get(section)
        if isinstance(seen, list) or isinstance(want, list):
            seen_set, want_set = set(seen or []), set(want or [])
            lines += [f"  • extra {word}: {item}" for item in sorted(seen_set - want_set)]
            lines += [f"  • missing {word}: {item}" for item in sorted(want_set - seen_set)]
            continue
        seen, want = seen or {}, want or {}
        for key in sorted(set(seen) - set(want)):
            lines.append(f"  • extra {word}: {key}")
        for key in sorted(set(want) - set(seen)):
            lines.append(f"  • missing {word}: {key}")
        for key in sorted(set(seen) & set(want)):
            if seen[key] == want[key]:
                continue
            if section == "tables":
                cols_seen, cols_want = seen[key]["columns"], want[key]["columns"]
                for col in sorted(set(cols_seen) - set(cols_want)):
                    lines.append(f"  • extra column: {key}.{col} ({cols_seen[col][0]})")
                for col in sorted(set(cols_want) - set(cols_seen)):
                    lines.append(f"  • missing column: {key}.{col} ({cols_want[col][0]})")
                for col in sorted(set(cols_seen) & set(cols_want)):
                    if cols_seen[col] != cols_want[col]:
                        lines.append(f"  • changed column: {key}.{col} (is {cols_seen[col]}, expected {cols_want[col]})")
                if seen[key]["row_security"] != want[key]["row_security"]:
                    lines.append(f"  • changed row security on table: {key}")
            else:
                lines.append(f"  • changed {word}: {key}")
                logger.info("drift detail %s %s: observed=%r expected=%r", section, key, seen[key], want[key])
    return lines


def render_drift_report(observed: dict, expected: dict) -> str:
    """S-DRIFT-LEAD followed by one line per added, missing, or changed object."""
    items = _drift_items(observed, expected)
    return "\n".join([operator_message("S-DRIFT-LEAD"), *(items or ["  • (no differences)"])])




@dataclass(frozen=True)
class LintFinding:
    file: str
    rule: str


_DOLLAR_BODY_RE = re.compile(r"(\$[A-Za-z_]*\$).*?\1", re.DOTALL)
_SQL_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_TXN_CONTROL_RE = re.compile(r"(?im)^\s*(BEGIN|START\s+TRANSACTION|COMMIT|ROLLBACK|SAVEPOINT)\b\s*(;|$|WORK|TRANSACTION)")
_CLOCK_SQL_RE = re.compile(r"(?i)(?<!DEFAULT\s)\b(NOW\(\)|CURRENT_DATE|CURRENT_TIMESTAMP|clock_timestamp\(\)|statement_timestamp\(\))")
_CLOCK_PY_RE = re.compile(r"\b(time\.time\(|datetime\.now\(|datetime\.utcnow\(|utc_now\(|date\.today\()")
_VERSION_PY_RE = re.compile(r"(release_identity|get_current_version|['\"]VERSION['\"]|/\s*['\"]VERSION)")
_ROUTE_FIELDS_RE = re.compile(r"(?is)UPDATE\s+model_configs\s+SET\b(.*?)(;|$)")
_ROUTE_FIELDS = ("model", "dialect_name", "endpoint_url", "api_key_name", "effort", "max_tokens")
_VECTOR_LITERAL_RE = re.compile(r"(?i)\bvector\(\s*\d+\s*\)")
_DIALECT_CHECK_RE = re.compile(r"(?is)dialect_name\b.*?\bCHECK\b|\bCHECK\b.*?dialect_name")
_DESTRUCTIVE_RE = re.compile(r"(?is)\bDROP\s+TABLE\b|\bDROP\s+COLUMN\b|\bALTER\s+TABLE\b[^;]*\bDROP\s+(?!CONSTRAINT|DEFAULT|NOT\s+NULL)\w|\bALTER\s+\w+\s+[^;]*\bRENAME\b")


def _sql_body(text: str) -> str:
    return _SQL_LINE_COMMENT_RE.sub("", text)


def lint_steps(chain: OrderedChain) -> list[LintFinding]:
    """
    Authoring-rule violations in the chain — the list below is the whole rule set:
    transaction control inside a step; reading the release VERSION or branching on the
    clock; any reference to ``embedding_config``; ``UPDATE model_configs SET`` of a route
    field; a literal ``vector(<digits>)``; a ``dialect_name`` CHECK without the
    dialect-twins marker; ``DROP TABLE``/``DROP COLUMN``/``RENAME`` without the
    expand-contract marker.
    """
    findings: list[LintFinding] = []
    for step in (*chain.versioned, *chain.repeatable):
        text = step.path.read_text()
        name = step.path.name
        is_sql = step.path.suffix == ".sql"
        body = _sql_body(text) if is_sql else text
        if is_sql and _TXN_CONTROL_RE.search(_DOLLAR_BODY_RE.sub("", body)):
            findings.append(LintFinding(name, "transaction control (BEGIN/COMMIT/ROLLBACK) — the runner owns the transaction"))
        clock = _CLOCK_SQL_RE if is_sql else _CLOCK_PY_RE
        if clock.search(body):
            findings.append(LintFinding(name, "reads the clock — a step must do the same thing on every install"))
        if not is_sql and _VERSION_PY_RE.search(body):
            findings.append(LintFinding(name, "reads the release VERSION — a step must not branch on the release"))
        if re.search(r"\bembedding_config\b", body):
            findings.append(LintFinding(name, "references embedding_config — the model and dimensions are fixed at install"))
        for match in _ROUTE_FIELDS_RE.finditer(body):
            touched = [f for f in _ROUTE_FIELDS if re.search(rf"\b{f}\b\s*=", match.group(1))]
            if touched:
                findings.append(LintFinding(name, f"rewrites operator route settings ({', '.join(touched)}) in model_configs"))
        if _VECTOR_LITERAL_RE.search(body):
            findings.append(LintFinding(name, "literal vector(<digits>) — use ctx.embedding_dimensions()"))
        if _DIALECT_CHECK_RE.search(body) and step.dialect_twins != _DIALECT_TWINS:
            findings.append(LintFinding(name, "changes the dialect_name CHECK without '-- dialect-twins: DialectName, DIALECT_NAMES'"))
        if _DESTRUCTIVE_RE.search(body) and step.contract_for is None:
            findings.append(LintFinding(name, "drops or renames a table/column without '-- expand-contract: contract-for V<N>'"))
    return findings



RefusalCode = Literal[
    "DOWNGRADE", "CHECKSUM_MISMATCH", "OUT_OF_ORDER",
    "DUPLICATE_REVISION", "GAP", "LINT", "UNREADABLE", "ADOPTION_DRIFT", "NOT_INSTALLED", "PAYLOAD",
]
REFUSAL_MESSAGE_IDS: dict[str, str] = {
    "DOWNGRADE": "S-DOWNGRADE",
    "CHECKSUM_MISMATCH": "S-CHECKSUM",
    "OUT_OF_ORDER": "S-OUTOFORDER",
    "DUPLICATE_REVISION": "S-DUPLICATE",
    "GAP": "S-GAP",
    "LINT": "S-LINT",
    "UNREADABLE": "S-UNREADABLE",
    "ADOPTION_DRIFT": "S-DRIFT",
    "NOT_INSTALLED": "S-NOT-INSTALLED",
    "PAYLOAD": "S-PAYLOAD",
}


class Refusal(BaseModel):
    """A reason the runner will not change the database; ``message`` is the operator string."""

    code: RefusalCode
    message: str = Field(description="Plain-English operator string, carrying a runnable command")


def refusal(code: RefusalCode, detail: str, **values: object) -> Refusal:
    """The Refusal for ``code`` with its operator string; ``detail`` goes to the log only."""
    logger.warning("refusal %s: %s", code, detail)
    return Refusal(code=code, message=operator_message(REFUSAL_MESSAGE_IDS[code], **values))


def _payload_refusal(error: Exception, file: str) -> Refusal:
    """
    A malformed BASELINE or HEAD_FINGERPRINT.json refuses as a first-class refusal, never a
    traceback: the defect is in the release's own files, and the readers below must not fall
    back to treating the file as absent (that path drift-adopts) or let update.sh blame the
    database with its generic plan-failure text. The file name goes to the log line only —
    operator strings never name it (the banned-words rule in utils/AGENTS.md).
    """
    logger.warning("release payload refused: %s: %s: %s", file, type(error).__name__, error)
    return refusal("PAYLOAD", f"{file}: {type(error).__name__}: {error}")


@dataclass
class Validation:
    refusals: list[Refusal] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refusals


def load_chain(migrations_dir: Path) -> tuple[OrderedChain | None, list[Refusal]]:
    """
    The ordered chain, or None with the refusal that broke it: a duplicate number or gap in the
    numbering, or a step file that cannot be read (its name fails the grammar, its header is
    malformed, or it does not parse).
    """
    try:
        return order_steps(discover_steps(migrations_dir)), []
    except StepOrderError as error:
        return None, [refusal(error.code, str(error), revision=error.revision)]
    except StepFileError as error:
        return None, [refusal("UNREADABLE", str(error), file=error.file)]
    except (ValueError, SyntaxError, FileNotFoundError) as error:
        return None, [refusal("UNREADABLE", f"{migrations_dir}: {type(error).__name__}: {error}",
                              file=migrations_dir.name)]


def validate(chain: OrderedChain, ledger: list[LedgerRow]) -> Validation:
    """
    Every reason the ledger and the tree disagree, plus lint findings; nothing is mutated.

    A recorded step whose file the tree lacks is always one of these: above the
    tree head (DOWNGRADE), a hole in the tree's numbering (GAP, from
    ``load_chain``), or a different file under its number (CHECKSUM_MISMATCH).
    """
    result = Validation()
    tree = chain.by_revision()
    head = chain.head
    applied = [row for row in ledger if row["kind"] == "versioned"]
    for row in applied:
        number = int(row["revision"])
        if number > head:
            result.refusals.append(refusal(
                "DOWNGRADE", f"ledger revision {row['revision']} above tree head {head:04d}",
                ledger_head=f"{ledger_head(ledger):04d}", tree_head=f"{head:04d}"))
            break
    for row in applied:
        step = tree.get(row["revision"])
        if step is not None and step.checksum != row["checksum"]:
            result.refusals.append(refusal(
                "CHECKSUM_MISMATCH",
                f"{step.path.name}: ledger checksum {row['checksum'][:12]}, file {step.checksum[:12]}",
                revision=step.revision, slug=step.slug))
    done = {row["revision"] for row in applied}
    current_head = ledger_head(ledger)
    downgrade = current_head > head
    for step in () if downgrade else chain.versioned:
        if step.revision not in done and step.number < current_head:
            result.refusals.append(refusal(
                "OUT_OF_ORDER", f"{step.path.name} is pending below ledger head {current_head:04d}",
                revision=step.revision, ledger_head=f"{current_head:04d}"))
    for finding in lint_steps(chain):
        result.refusals.append(refusal("LINT", f"{finding.file}: {finding.rule}", file=finding.file))
    return result



AdoptionState = Literal["ledger_absent", "ledger_empty", "ledger_rows", "not_installed"]


@dataclass(frozen=True)
class AdoptionView:
    """What the database is, before any step runs."""

    state: AdoptionState
    matched: str | None
    fingerprint_sha256: str | None

    @property
    def adoption_needed(self) -> bool:
        return self.state in ("ledger_absent", "ledger_empty")


def classify_database(conn: psycopg.Connection, baseline: Baseline | None) -> AdoptionView:
    """Ledger state plus, when the ledger holds no rows, the BASELINE match of the structure (ledger excluded)."""
    has_tables = user_tables_present(conn)
    if not ledger_present(conn):
        state: AdoptionState = "ledger_absent" if has_tables else "not_installed"
    elif read_ledger_rows(conn):
        return AdoptionView("ledger_rows", None, None)
    else:
        state = "ledger_empty" if has_tables else "not_installed"
    if state == "not_installed":
        return AdoptionView(state, None, None)
    if baseline is None:
        raise FileNotFoundError("deploy/migrations/BASELINE is required to adopt an installed database")
    digest = fingerprint_sha256(_fingerprint_from(conn, include_ledger=False))
    return AdoptionView(state, baseline.match(digest), digest)




class StatusReport(BaseModel):
    """`mira migrate status`: where the database stands against this tree."""

    ledger_present: bool
    ledger_head: int = Field(description="Highest applied versioned step (0 for none)")
    tree_head: int = Field(description="Highest versioned step in this tree")
    pending: list[str] = Field(description="File names of steps that would run")
    repeatable_due: list[str]
    refusals: list[Refusal]
    fingerprint_drift: bool = Field(description="At head, the live structure differs from HEAD_FINGERPRINT.json")
    adopted_with_drift_override: bool = Field(description="Any ledger row was written under --adopt-drifted")


class PlanReport(BaseModel):
    """What `mira update` will ask consent for, computed read-only before anything changes."""

    adoption_needed: bool
    pending: list[str]
    repeatable_due: list[str]
    operator_input: list[str] = Field(description="Prompts of pending steps that need a typed value")
    notices: dict[str, str] = Field(description="Step file name → plain-English line shown before consent")
    refusals: list[Refusal]
    drift_report: str | None = Field(default=None, description="S-DRIFT-LEAD report when the structure matches no known release")
    drift_accepted: bool = Field(default=False, description="Drift present and --adopt-drifted given: proceeds with consent")
    needs_consent: bool


def _baseline_if_present(migrations_dir: Path, chain: OrderedChain) -> Baseline | None:
    if not (migrations_dir / "BASELINE").is_file():
        return None
    return read_baseline(migrations_dir, chain_head=chain.head)


def status(conninfo: str, migrations_dir: Path = MIGRATIONS_DIR) -> StatusReport:
    """Read-only report of the ledger against the tree (see ``StatusReport``)."""
    chain, refusals = load_chain(migrations_dir)
    with connect(conninfo) as conn:
        present = ledger_present(conn)
        rows = read_ledger_rows(conn) if present else []
        if chain is None:
            return StatusReport(ledger_present=present, ledger_head=ledger_head(rows), tree_head=0, pending=[],
                                repeatable_due=[], refusals=refusals, fingerprint_drift=False,
                                adopted_with_drift_override=any(r["applied_by"].endswith("+adopt-drifted") for r in rows))
        # Classify first and derive the pending list from it, as plan() does: an installed
        # database with an empty ledger gets its steps stamped from the BASELINE match, so
        # naming every versioned step "waiting to run" contradicts what the runner will do.
        try:
            baseline = _baseline_if_present(migrations_dir, chain)
        except (ValueError, KeyError, TypeError) as error:
            return StatusReport(ledger_present=present, ledger_head=ledger_head(rows), tree_head=chain.head,
                                pending=[], repeatable_due=[],
                                refusals=[*refusals, _payload_refusal(error, "BASELINE")], fingerprint_drift=False,
                                adopted_with_drift_override=any(r["applied_by"].endswith("+adopt-drifted") for r in rows))
        view = classify_database(conn, baseline)
        validation = validate(chain, rows)
        status_refusals = validation.refusals
        steps = steps_to_run(chain, rows, view)
        if view.state == "not_installed":
            status_refusals = [refusal("NOT_INSTALLED", "no MIRA tables in the database"), *status_refusals]
            steps = []
        drift = False
        if present and rows and ledger_head(rows) == chain.head and validation.ok:
            try:
                head_fp = read_head_fingerprint(migrations_dir, baseline)
            except (ValueError, KeyError, TypeError) as error:
                status_refusals.append(_payload_refusal(error, "HEAD_FINGERPRINT.json"))
            else:
                drift = fingerprint_sha256(_fingerprint_from(conn, include_ledger=False)) != fingerprint_sha256(head_fp)
        return StatusReport(
            ledger_present=present,
            ledger_head=ledger_head(rows),
            tree_head=chain.head,
            pending=[s.path.name for s in steps],
            repeatable_due=[s.path.name for s in steps if s.kind == "repeatable"],
            refusals=status_refusals,
            fingerprint_drift=drift,
            adopted_with_drift_override=any(r["applied_by"].endswith("+adopt-drifted") for r in rows),
        )


def steps_to_run(chain: OrderedChain, rows: list[LedgerRow], view: AdoptionView) -> list[Step]:
    """
    The steps a migrate run would execute: for an adoption, every step above
    the BASELINE match (origin → all); otherwise everything pending.
    """
    if view.adoption_needed:
        start = int(view.matched) if view.matched is not None else 0
        return [s for s in chain.versioned if s.number > start] + list(chain.repeatable)
    return pending(chain, rows)


def plan(conninfo: str, migrations_dir: Path = MIGRATIONS_DIR, adopt_drifted: bool = False) -> PlanReport:
    """
    Read-only consent plan: what will run, what needs typed input, and every refusal.
    ``adopt_drifted`` turns an adoption-drift refusal into a consent line (the
    drift report still shown), exactly as ``migrate(adopt_drifted=True)`` will act.
    """
    chain, refusals = load_chain(migrations_dir)
    if chain is None:
        return PlanReport(adoption_needed=False, pending=[], repeatable_due=[], operator_input=[], notices={},
                          refusals=refusals, needs_consent=False)
    try:
        baseline = _baseline_if_present(migrations_dir, chain)
    except (ValueError, KeyError, TypeError) as error:
        refusals.append(_payload_refusal(error, "BASELINE"))
        return PlanReport(adoption_needed=False, pending=[], repeatable_due=[], operator_input=[], notices={},
                          refusals=refusals, needs_consent=False)
    with connect(conninfo) as conn:
        view = classify_database(conn, baseline)
        rows = read_ledger_rows(conn) if view.state == "ledger_rows" else []
    if view.state == "not_installed":
        refusals.append(refusal("NOT_INSTALLED", "no MIRA tables in the database"))
    drift_report = None
    drift_accepted = False
    if view.adoption_needed and view.matched is None:
        if adopt_drifted:
            drift_accepted = True
        else:
            refusals.append(refusal("ADOPTION_DRIFT", f"structure {view.fingerprint_sha256} matches no BASELINE row"))
        try:
            head_fp = read_head_fingerprint(migrations_dir, baseline)
        except (ValueError, KeyError, TypeError) as error:
            refusals.append(_payload_refusal(error, "HEAD_FINGERPRINT.json"))
        else:
            drift_report = render_drift_report(
                structural_fingerprint(conninfo, include_ledger=False),
                head_fp,
            )
    refusals.extend(validate(chain, rows).refusals)
    steps = steps_to_run(chain, rows, view) if not refusals else []
    versioned = [s for s in steps if s.kind == "versioned"]
    repeatables = [s for s in steps if s.kind == "repeatable"]
    return PlanReport(
        adoption_needed=view.adoption_needed,
        pending=[s.path.name for s in versioned],
        repeatable_due=[s.path.name for s in repeatables],
        operator_input=[s.requires_operator_input for s in steps if s.requires_operator_input],
        notices={s.path.name: s.notice or operator_message("S-STEP-DEFAULT") for s in versioned},
        refusals=refusals,
        drift_report=drift_report,
        drift_accepted=drift_accepted,
        needs_consent=bool(not refusals and (view.adoption_needed or steps)),
    )


_ELEVATION_MESSAGE_IDS = {
    "interactive": "S-ELEV-INTERACTIVE",
    "passwordless": "S-ELEV-PASSWORDLESS",
    "none": "S-ELEV-NONE",
}


def render_plan(report: PlanReport, elevation: Elevation) -> str:
    """
    The consent text: one line per pending step (its NOTICE or S-STEP-DEFAULT),
    one per due repeatable, S-PLAN-FIRST on adoption, the elevation sentence,
    each operator-input prompt, then S-PROTECT. S-NOTHING when consent is not needed.
    """
    if not report.needs_consent:
        return operator_message("S-NOTHING")
    lines: list[str] = []
    if report.drift_accepted and report.drift_report:
        lines += [report.drift_report, "", operator_message("S-DRIFT-ACCEPTED"), ""]
    if report.adoption_needed:
        lines.append(operator_message("S-PLAN-FIRST"))
        lines.append("")
    for name in report.pending:
        lines.append(f"  • {report.notices[name]}")
    for name in report.repeatable_due:
        lines.append("  • Refreshes reference data MIRA ships with. Nothing in your own data changes.")
    if elevation in _ELEVATION_MESSAGE_IDS:
        lines.append(f"  • {operator_message(_ELEVATION_MESSAGE_IDS[elevation])}")
    for prompt in report.operator_input:
        lines.append(f"  • You will be asked: {prompt}")
    lines.append("")
    lines.append(operator_message("S-PROTECT"))
    return "\n".join(lines)


MIGRATION_ADVISORY_KEY = 1296650817  # b"MIRA" read as a big-endian int; every runner process takes this one key
# A second update waits this long for the first to finish before giving up (exit 3).
# Must cover a first run's backup + steps to be useful — pg_dump of a large
# database alone can take minutes — but a first run that outruns it (pg_dump's
# own bound is an hour, below) makes the waiter refuse honestly (S-LOCKED, a
# retry); the first run is unaffected.
MIGRATION_LOCK_TIMEOUT_SECONDS = 900
# A step blocked on a table lock fails (and is restored) rather than hanging
# the update. A lock held past 5 min — typically by a live server's open
# transaction, since a standalone migrate runs against the running install —
# is treated as pathological: fail, roll back, restore.
STEP_LOCK_TIMEOUT_SECONDS = 300
CLIENT_CHECK_INTERVAL_SECONDS = 5
# An hour of headroom over an install-scale database's bulk copy (one
# operator's account per install — root AGENTS.md): the bound exists so a
# hung dump or restore cannot hang the update forever; it does not calibrate
# an expected duration.
PG_DUMP_TIMEOUT_SECONDS = 3600
PG_RESTORE_TIMEOUT_SECONDS = 3600
# Bounds the quick superuser commands — the sudo probe, pg_restore --list,
# dropdb, createdb, the bootstrap psql — each seconds-fast against a local
# server; the bulk copies carry their own hour bounds above.
SEAM_TIMEOUT_SECONDS = 120
BACKUP_RETENTION = 3
DEFAULT_BACKUPS_DIR = Path("/opt/mira/backups")
DEFAULT_ENV_FILE = Path("/opt/mira/systemone.env")
SUPERVISOR_FILES = (
    Path("/etc/systemd/system/mira.service"),
    Path.home() / "Library" / "LaunchAgents" / "com.mira.app.plist",
)
BACKUP_VAULT_PATHS = ("mira/api_keys", "mira/database", "mira/services")
REQUIRED_EXTENSIONS = ("pgcrypto", "pg_trgm", "vector")

OPERATOR_MESSAGES.update({
    # Platform-aware: the two hosts reach this message for different reasons.
    # On macOS the seam is the current user with no elevation prefix
    # (superuser_prefix), so reaching this means the account running MIRA
    # cannot act as the database's superuser — a terminal and a password
    # cannot change that, so the remedy names the account instead. Linux
    # keeps its accurate password wording.
    "S-NOSEAM": (
        "MIRA could not reach the database as its administrator, so it cannot upgrade your database's "
        "structure. Nothing was changed. "
        + (
            "The account running MIRA is not this database's superuser. Run: mira update again from "
            "the account that administers this database (its superuser)."
            if sys.platform == "darwin"
            else "Run: mira update from a terminal, and type your admin password when asked."
        )
    ),
    "S-OPINPUT": (
        "MIRA needs you to type a value to finish this upgrade, and this session has no terminal. The "
        "upgrade stopped before that step. Run: mira migrate from a terminal."
    ),
    "S-DISK": (
        "There is not enough free disk space to back up your data before upgrading (needs {need}, has "
        "{have}). Your data is untouched and the upgrade did not start. Free up space, then run: mira update"
    ),
    "S-PGTOOLS": (
        "MIRA needs the PostgreSQL {tool} program, version {major} or newer, to back up your data, and "
        "could not find it ({found}). Your data is untouched and the upgrade did not start. Install it, "
        "then run: mira update"
    ),
    "S-BACKUP-FAILED": (
        "MIRA could not make a complete backup of your data, so it did not start the upgrade. Your data "
        "is untouched. Run: mira update to try again."
    ),
    "S-BACKUP-ACL": (
        "MIRA could not make its backup readable by the database superuser, so it did not start the "
        "upgrade. Your data is untouched. Install your system's acl package (for example: sudo apt install "
        "acl) and check that /opt/mira is on a filesystem with POSIX ACL support, then run: mira update"
    ),
    "S-LOCKED": (
        "Another MIRA update is already changing this database. Nothing was changed by this one. Wait for "
        "it to finish, then run: mira update"
    ),
    "S-BACKUP-PARTIAL": (
        "MIRA backed up your database to {target} — that part is complete, and you can put it back with: "
        "mira restore {stamp} --backups-dir {parent}. But MIRA could not finish copying the rest of your "
        "install (the program files or the per-user tool data), so those parts of the archive are missing. "
        "Nothing on your install was changed. Run the command again to retry"
    ),
    "S-BACKUP-OK": (
        "MIRA wrote a complete backup of your install to {target}. It holds your database, your per-user "
        "tool data, your settings, your service configuration, and every credential MIRA stores in Vault — "
        "treat the directory as a password. Put your install's database back from it with: mira restore "
        "{stamp} --backups-dir {parent}"
    ),
    "S-STEP-FAILED": (
        "MIRA could not finish upgrading your database (it stopped while: {what}). The backup taken "
        "before this upgrade is at {backup}. If this was part of mira update, your install is being put "
        "back now; otherwise restore it with: mira restore {stamp}"
    ),
    "S-ADOPT-IGNORED": (
        "Your database is already set up for in-place upgrades, so the --adopt-drifted option was not "
        "needed and was ignored."
    ),
    "S-POSTDRIFT": (
        "Your database has changes MIRA did not make. MIRA left them exactly as they are and will keep "
        "working. They are listed below; if you did not make them, report them. Run: mira migrate to see "
        "this list again."
    ),
    "S-RESTORE-NOBACKUP": (
        "There is no backup named {stamp} in {backups_dir}. Nothing was changed. Backups there: {available}. "
        "Run: mira restore <one of those>"
    ),
    "S-RESTORE-NOBACKUP-NONE": (
        "There is no backup named {stamp} in {backups_dir}, and there are no other backups "
        "there. Nothing was changed. If you expected a backup to exist, check {backups_dir}."
    ),
    "S-PARK-DAMAGED": (
        "The backup at {stamp} is damaged — MIRA could not read it back, and running the usual "
        "restore commands against it would destroy your database. Do not restore this backup. "
        "Other backups there: {available}. Restore one of those instead: mira restore <name>"
    ),
    "S-PARK-DAMAGED-ALONE": (
        "The backup at {stamp} is damaged — MIRA could not read it back, and running the usual "
        "restore commands against it would destroy your database. MIRA has no other backup it "
        "can offer: leave the database as it is and report it — "
        "the damaged backup and the update log show what happened"
    ),
    "S-RESTORE-UNREADABLE": (
        "MIRA cannot read the backup at {stamp} — its manifest.json is damaged or unreadable, so "
        "MIRA will not guess at what it contains (a power loss during that backup can do this). "
        "Nothing was changed. The backup's data files may still be recoverable — do not try to "
        "restore it yourself, and copy the backup directory somewhere outside {backups_dir} if "
        "you want to keep it: the next successful backup will clean it up as residue. Other "
        "backups there: {available}. Restore one of those instead: mira restore <name>"
    ),
    "S-RESTORE-UNREADABLE-ALONE": (
        "MIRA cannot read the backup at {stamp} — its manifest.json is damaged or unreadable, so "
        "MIRA will not guess at what it contains (a power loss during that backup can do this). "
        "Nothing was changed. The backup's data files may still be recoverable — do not try to "
        "restore it yourself, and copy the backup directory somewhere outside {backups_dir} if "
        "you want to keep it: the next successful backup will clean it up as residue. MIRA has "
        "no other backup it can read: leave the database as it is and report it — the backup "
        "directory and the update log show what happened"
    ),
    "S-RESTORE-OK": "Your data is back exactly as it was before the upgrade (backup {stamp}).",
})


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _manifest_parses(manifest: Path) -> bool:
    # KVM-13: a power cut can leave a 0-byte manifest.json; is_file() alone counted it complete.
    # One predicate at every consumer (listing, retention, sweeper, restore's refusal):
    # a manifest that is JSON but not a valid BackupManifest must not be advertised either.
    try:
        BackupManifest.model_validate_json(manifest.read_text())
        return True
    except (OSError, ValueError):
        return False


class SeamUnavailable(RuntimeError):
    """No non-interactive way to run a command as the PostgreSQL superuser."""


class OperatorInputRequired(RuntimeError):
    """A step needs a typed value and stdin is not a terminal."""

    def __init__(self, prompt: str):
        super().__init__(prompt)
        self.prompt = prompt


class BackupBlocked(RuntimeError):
    """The pre-mutation backup could not be taken; ``message`` is the operator string."""

    def __init__(self, message: str, detail: str):
        super().__init__(detail)
        self.message = message


def _grant_backup_read_access(backups_dir: Path, target: Path, dump: Path) -> None:
    """
    Give the postgres user exactly what it needs to read the backup back: traverse on
    the backups dir and the stamp dir, read on the dump. Idempotent, owner-level, no
    elevation. Shared by take_backup (fail-closed: a backup born unreadable parks every
    rollback on an empty database — KVM-2) and restore (best-effort: a restore of a
    pre-ACL-era stamp must fix the access itself, or the park's own remedy is not
    runnable — Q13-final's recovery-step finding).
    """
    setfacl = shutil.which("setfacl")
    if setfacl is None:
        raise BackupBlocked(operator_message("S-BACKUP-ACL"), "setfacl not on PATH")
    for argv in (
        [setfacl, "-m", "u:postgres:--x", str(backups_dir)],
        [setfacl, "-m", "u:postgres:--x", str(target)],
        [setfacl, "-m", "u:postgres:r", str(dump)],
    ):
        result = subprocess.run(argv, capture_output=True, text=True, timeout=SEAM_TIMEOUT_SECONDS)
        if result.returncode != 0:
            raise BackupBlocked(operator_message("S-BACKUP-ACL"),
                                f"{' '.join(argv)} exit {result.returncode}: {result.stderr.strip()}")


class MigrationLocked(RuntimeError):
    """Another runner held the advisory lock past MIGRATION_LOCK_TIMEOUT_SECONDS."""


Emit = Callable[[str], None]


@contextmanager
def migration_lock(conn: psycopg.Connection) -> Iterator[None]:
    """
    Hold the runner's session advisory lock; steps then run under STEP_LOCK_TIMEOUT_SECONDS.

    The lock dies with the session, so a killed runner (SIGKILL, power loss)
    never leaves it held.

    Raises:
        MigrationLocked: Another runner kept it past MIGRATION_LOCK_TIMEOUT_SECONDS.
    """
    # A runner killed mid-step leaves its backend running the step until the next write to
    # the dead socket; with this check the server aborts the transaction (and drops the
    # lock) within seconds instead. Unsupported platforms keep the slower behavior.
    try:
        conn.execute(f"SET client_connection_check_interval = '{CLIENT_CHECK_INTERVAL_SECONDS}s'")
    except psycopg.Error as error:
        logger.info("client_connection_check_interval not available here: %s", error)
    conn.execute(f"SET lock_timeout = '{MIGRATION_LOCK_TIMEOUT_SECONDS}s'")
    try:
        conn.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_ADVISORY_KEY,))
    except psycopg.errors.LockNotAvailable as error:
        raise MigrationLocked(str(error).strip()) from error
    try:
        conn.execute(f"SET lock_timeout = '{STEP_LOCK_TIMEOUT_SECONDS}s'")
        yield
    finally:
        if not conn.closed and not conn.broken:
            conn.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_ADVISORY_KEY,))


class VaultAdapter:
    """``ctx.vault``: KV v2 through the app's own ``VaultClient`` (AppRole), constructed on first use."""

    def __init__(self) -> None:
        self._vault = None

    def _client(self):
        if self._vault is None:
            from clients.vault_client import VaultClient

            self._vault = VaultClient()
        return self._vault

    def read(self, path: str) -> dict:
        return dict(self._client()._read_secret_version(path)["data"]["data"])

    def patch(self, path: str, fields: dict[str, str]) -> list[str]:
        """Write only the fields whose value differs; returns the changed field names."""
        current = self.read(path)
        changed = {key: value for key, value in fields.items() if current.get(key) != value}
        if changed:
            self._client().client.secrets.kv.v2.patch(path=path, secret=changed)
        return sorted(changed)

    def delete_fields(self, path: str, fields: list[str], *, confirmed_new: list[str]) -> None:
        """
        Remove ``fields`` from ``path`` only once every ``confirmed_new`` field is present.

        Raises:
            ValueError: A confirmed_new field is missing — the old value is still the only copy.
        """
        current = self.read(path)
        missing = [name for name in confirmed_new if name not in current]
        if missing:
            raise ValueError(f"refusing to delete {fields} from {path}: replacement fields {missing} are not written yet")
        remaining = {key: value for key, value in current.items() if key not in fields}
        self._client().client.secrets.kv.v2.create_or_update_secret(path=path, secret=remaining)


class EnvFileAdapter:
    """``ctx.env``: ``NAME=value`` lines of the settings file, rewritten atomically at mode 0600."""

    def __init__(self, env_file: Path) -> None:
        self.env_file = env_file

    def lines(self) -> list[str]:
        return self.env_file.read_text().splitlines() if self.env_file.exists() else []

    def values(self) -> dict[str, str]:
        found = {}
        for line in self.lines():
            name, sep, value = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                found[name.strip()] = value
        return found

    def write_lines(self, lines: list[str]) -> None:
        self.env_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.env_file.parent, prefix=f".{self.env_file.name}.")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write("".join(f"{line}\n" for line in lines))
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.env_file)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def set_var(self, name: str, value: str) -> None:
        lines = [l for l in self.lines() if l.partition("=")[0].strip() != name]
        self.write_lines(lines + [f"{name}={value}"])

    def unset_var(self, name: str) -> None:
        self.write_lines([l for l in self.lines() if l.partition("=")[0].strip() != name])


def _operator_input(prompt: str) -> str:
    if not sys.stdin.isatty():
        raise OperatorInputRequired(prompt)
    return input(f"{prompt}: ")


@dataclass(frozen=True)
class StepContext:
    """What a ``.py`` step's ``up(ctx)``/``verify(ctx)`` receive. Steps never commit."""

    pg: psycopg.Connection
    vault: VaultAdapter
    env: EnvFileAdapter
    operator_input: Callable[[str], str] = _operator_input

    def invalidate_valkey(self) -> int:
        """Flush every Valkey key except the boot-preserved prefixes; returns the number flushed."""
        from clients.valkey_client import BOOT_FLUSH_PRESERVE_PREFIXES, get_valkey_client

        return get_valkey_client().flush_except_whitelist(list(BOOT_FLUSH_PRESERVE_PREFIXES))

    def embedding_dimensions(self) -> int:
        """The install's vector length, for interpolation with ``psycopg.sql`` only."""
        return int(self.pg.execute("SELECT dimensions FROM embedding_config LIMIT 1").fetchone()[0])


def run_step(ctx: StepContext, step: Step) -> None:
    """
    Execute one step inside the caller's transaction.

    Raises:
        ValueError: A ``.py`` step without a callable ``up``.
        Exception: Whatever the step raises — the caller rolls back and reports it.
    """
    if step.path.suffix == ".sql":
        ctx.pg.execute(step.path.read_text())
        return
    spec = importlib.util.spec_from_file_location(f"mira_step_{step.ledger_key}", step.path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{step.path.name}: cannot load as a Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    up = getattr(module, "up", None)
    if not callable(up):
        raise ValueError(f"{step.path.name}: defines no up(ctx)")
    up(ctx)
    verify = getattr(module, "verify", None)
    if callable(verify):
        verify(ctx)


def _record(conn: psycopg.Connection, step: Step, applied_by: str, duration_ms: int, name: str | None = None) -> None:
    conn.execute(
        """INSERT INTO schema_migrations (revision, name, kind, store, checksum, applied_by, duration_ms)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (revision) DO UPDATE SET name = EXCLUDED.name, checksum = EXCLUDED.checksum,
               applied_at = NOW(), applied_by = EXCLUDED.applied_by, duration_ms = EXCLUDED.duration_ms""",
        (step.ledger_key, name or step.slug, step.kind, step.store, step.checksum, applied_by, duration_ms),
    )


def _conn_params(conninfo: str) -> dict[str, str]:
    return psycopg.conninfo.conninfo_to_dict(conninfo)


def superuser_prefix() -> list[str]:
    """
    How this host runs a command as the PostgreSQL superuser: macOS — the
    current (Homebrew) user, directly; a container's root — ``sudo -u
    postgres``; Linux with an askpass helper (deploy/update.sh after consent)
    — ``sudo -A -u postgres``; otherwise ``sudo -n -u postgres``.
    """
    if sys.platform == "darwin":
        return []
    if os.geteuid() == 0:
        return ["sudo", "-u", "postgres"]
    if os.environ.get("SUDO_ASKPASS"):
        return ["sudo", "-A", "-u", "postgres"]
    return ["sudo", "-n", "-u", "postgres"]


def run_as_superuser(argv: list[str], timeout: int, stdin_text: str | None = None) -> subprocess.CompletedProcess:
    """
    The superuser seam — the only place the runner acts as the PostgreSQL superuser.

    Raises:
        SeamUnavailable: The elevation itself fails (no ticket, no askpass, no sudo).
        subprocess.TimeoutExpired: The command exceeded ``timeout``.
    """
    prefix = superuser_prefix()
    if prefix:
        try:
            probe = subprocess.run(prefix + ["true"], capture_output=True, text=True, timeout=SEAM_TIMEOUT_SECONDS)
        except FileNotFoundError as error:
            raise SeamUnavailable(f"{prefix[0]} not found") from error
        if probe.returncode != 0:
            raise SeamUnavailable(f"{' '.join(prefix)} failed: {probe.stderr.strip()}")
    return subprocess.run(prefix + argv, capture_output=True, text=True, timeout=timeout, input=stdin_text)


def pg_tool(name: str, server_major: int) -> str:
    """
    Absolute path of a PostgreSQL client tool whose major is at least the server's.

    Raises:
        BackupBlocked: Missing, or older than the server (pg_dump refuses a newer server).
    """
    path = shutil.which(name)
    found = "not on PATH"
    if path is not None:
        version = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=SEAM_TIMEOUT_SECONDS).stdout
        match = re.search(r"(\d+)(\.\d+)*", version)
        major = int(match.group(1)) if match else 0
        if major >= server_major:
            return path
        found = f"{path} is version {major}"
    raise BackupBlocked(operator_message("S-PGTOOLS", tool=name, major=server_major, found=found),
                        f"{name}: {found}, server major {server_major}")


def _server_major(conn: psycopg.Connection) -> int:
    return int(conn.execute("SHOW server_version_num").fetchone()[0]) // 10000


def _human_bytes(count: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if count < 1024 or unit == "TB":
            return f"{count:.0f} {unit}" if unit == "bytes" else f"{count:.1f} {unit}"
        count /= 1024
    return f"{count} TB"


class BackupManifest(BaseModel):
    """``manifest.json``: written last and fsynced, so its presence means the backup is complete and durable.

    New fields must be optional/defaulted, never required: _manifest_parses validates
    the full model, so a future required field would make every older stamp count as
    residue at the next backup's post-durability sweep."""

    stamp: str
    database: str
    port: str
    dump_file: str
    table_data_entries: int
    vault_field_counts: dict[str, int]
    env_file: str
    env_file_present: bool
    supervisor_file: str
    supervisor_file_present: bool
    database_size_bytes: int
    version: str
    ledger_head: int
    fingerprint_sha256: str = Field(description="Whole-database structure (ledger included) at dump time")


def take_backup(
    conninfo: str,
    backups_dir: Path,
    stamp: str,
    *,
    vault: VaultAdapter,
    env_file: Path,
    version: str,
    sweep: bool = True,
    grant: bool = True,
) -> BackupManifest:
    """
    Back up everything an upgrade can change into ``<backups_dir>/<stamp>/``:
    ``pg_dump -Fc`` (as mira_admin over TCP, verified with ``pg_restore
    --list``), the three MIRA Vault secrets (``vault.json``, mode 0600, never
    printed), the settings file and supervisor file copies, then
    ``manifest.json`` last.

    ``sweep=False, grant=False`` is the user-pull mode (``user_backup``): the
    destination is a directory the operator chose, so the residue sweeper must
    never touch it and no ACL grant is written into it — the dump is verified
    with ``pg_restore --list`` as the current user, and a later ``mira restore``
    re-grants best-effort itself.

    Raises:
        BackupBlocked: Disk short, client tools missing/old, or any step failed —
            the partial directory is removed and nothing in the database changed.
    """
    target = backups_dir / stamp
    params = _conn_params(conninfo)
    # Reclaim the residue of runs interrupted mid-backup (SIGKILL, power loss, Ctrl-C)
    # before creating this run's stamp directory — see sweep_incomplete_backups.
    if sweep:
        sweep_incomplete_backups(backups_dir, stamp, unreadable_is_residue=False)
    try:
        if not backups_dir.is_dir():
            # Mode only at creation: a chmod on every run re-collapses the
            # parent's ACL mask (mask::---), revoking the postgres traverse
            # grant every prior stamp needs — and a run that then fails
            # before its own grant phase left them that way (Q13-final's
            # mask-collapse witness).
            backups_dir.mkdir(parents=True, mode=0o700)
        target.mkdir(mode=0o700)
    except OSError as error:
        raise BackupBlocked(operator_message("S-BACKUP-FAILED"), f"cannot create {target}: {error}") from error
    try:
        with connect(conninfo) as conn:
            size = int(conn.execute("SELECT pg_database_size(current_database())").fetchone()[0])
            server_major = _server_major(conn)
            dbname = conn.execute("SELECT current_database()").fetchone()[0]
            rows = read_ledger_rows(conn) if ledger_present(conn) else []
            fingerprint = fingerprint_sha256(_fingerprint_from(conn, include_ledger=True))
        free = shutil.disk_usage(backups_dir).free
        if free < 2 * size:
            raise BackupBlocked(operator_message("S-DISK", need=_human_bytes(2 * size), have=_human_bytes(free)),
                                f"free {free} < 2 * pg_database_size {size}")
        pg_dump = pg_tool("pg_dump", server_major)
        pg_restore = pg_tool("pg_restore", server_major)
        dump = target / f"{dbname}.dump"
        env = dict(os.environ)
        password = params.pop("password", None)
        if password is not None:
            env["PGPASSWORD"] = password
        dsn = psycopg.conninfo.make_conninfo(**params)
        try:
            result = subprocess.run([pg_dump, "-Fc", "--file", str(dump), "--dbname", dsn],
                                    capture_output=True, text=True, timeout=PG_DUMP_TIMEOUT_SECONDS, env=env)
        except subprocess.TimeoutExpired as error:
            raise BackupBlocked(operator_message("S-BACKUP-FAILED"), f"pg_dump exceeded {PG_DUMP_TIMEOUT_SECONDS}s") from error
        if result.returncode != 0:
            raise BackupBlocked(operator_message("S-BACKUP-FAILED"), f"pg_dump exit {result.returncode}: {result.stderr.strip()}")
        listing = subprocess.run([pg_restore, "--list", str(dump)], capture_output=True, text=True,
                                 timeout=SEAM_TIMEOUT_SECONDS)
        if listing.returncode != 0:
            raise BackupBlocked(operator_message("S-BACKUP-FAILED"), f"pg_restore --list exit {listing.returncode}: {listing.stderr.strip()}")
        table_data = sum(1 for line in listing.stdout.splitlines() if " TABLE DATA " in line)

        # The restorer reads the dump as the database superuser (`restore`'s
        # run_as_superuser), which cannot traverse the 0700 backups tree this
        # function creates — grant exactly that user read access here, verified
        # from the restorer's side, or every rollback ends parked with an empty
        # database (campaign finding KVM-2).
        if grant and superuser_prefix():
            _grant_backup_read_access(backups_dir, target, dump)
            try:
                superuser_listing = run_as_superuser([pg_restore, "--list", str(dump)], SEAM_TIMEOUT_SECONDS)
            except SeamUnavailable as error:
                # The elevation itself failed — not an acl problem; S-NOSEAM's
                # terminal/password remedy is the one that can fix this arm.
                raise BackupBlocked(operator_message("S-NOSEAM"), f"superuser seam unavailable: {error}") from error
            if superuser_listing.returncode != 0:
                raise BackupBlocked(operator_message("S-BACKUP-FAILED"),
                                    f"superuser pg_restore --list exit {superuser_listing.returncode}: {superuser_listing.stderr.strip()}")

        # KVM-13 (live power-cut witness): these hand-written files were page-cache-only —
        # manifest.json and vault.json came back 0 bytes while pg_dump's fsynced dump
        # survived. The stamp is durable before its manifest exists.
        secrets = {path: vault.read(path) for path in BACKUP_VAULT_PATHS}
        vault_file = target / "vault.json"
        fd = os.open(vault_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(secrets, handle)
            handle.flush()
            os.fsync(handle.fileno())

        env_present = env_file.is_file()
        if env_present:
            shutil.copy2(env_file, target / "systemone.env")
            os.chmod(target / "systemone.env", 0o600)
            _fsync_path(target / "systemone.env")
        supervisor = next((p for p in SUPERVISOR_FILES if p.is_file()), SUPERVISOR_FILES[0])
        supervisor_present = supervisor.is_file()
        if supervisor_present:
            shutil.copy2(supervisor, target / supervisor.name)
            os.chmod(target / supervisor.name, 0o600)
            _fsync_path(target / supervisor.name)

        manifest = BackupManifest(
            stamp=stamp,
            database=dbname,
            port=str(params.get("port", "5432")),
            dump_file=dump.name,
            table_data_entries=table_data,
            vault_field_counts={path: len(data) for path, data in secrets.items()},
            env_file=str(env_file),
            env_file_present=env_present,
            supervisor_file=str(supervisor),
            supervisor_file_present=supervisor_present,
            database_size_bytes=size,
            version=version,
            ledger_head=ledger_head(rows),
            fingerprint_sha256=fingerprint,
        )
        tmp = target / "manifest.json.tmp"
        tmp.write_text(manifest.model_dump_json(indent=2))
        _fsync_path(tmp)
        os.replace(tmp, target / "manifest.json")
        _fsync_path(target)
    except BackupBlocked:
        shutil.rmtree(target, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(target, ignore_errors=True)
        raise BackupBlocked(operator_message("S-BACKUP-FAILED"), f"{type(error).__name__}: {error}") from error
    # Outside the try above: its handlers rmtree this run's target, so the
    # post-durability sweep must not be able to invert into deleting the
    # completed stamp (round-9 F-1).
    if sweep:
        sweep_incomplete_backups(backups_dir, stamp, unreadable_is_residue=True)
    logger.info("backup %s complete: %s bytes, %s TABLE DATA entries", target, size, table_data)
    return manifest


def sweep_incomplete_backups(backups_dir: Path, stamp: str, *, unreadable_is_residue: bool = False) -> list[Path]:
    """
    Delete incomplete backup directories older than ``stamp``: the residue of a run
    interrupted mid-backup (SIGKILL, power loss, Ctrl-C), which nothing else cleans —
    retention and update.sh count only manifest-complete directories, so without this
    each interrupt leaks one directory, sometimes still holding ``vault.json``.

    Two phases, both keyed on the manifest being unreadable-or-absent and never
    touching a manifest-complete directory (retention's completeness signal):
    pre-backup (``unreadable_is_residue=False``) reclaims only manifest-absent
    directories — no backup ever completed there, and the space may be needed for
    this run's dump; post-backup (``True``) also reclaims manifest-unreadable
    directories — damaged stamps whose dump and vault may be hand-recoverable, so
    they are deleted only after the replacement stamp is complete, never before
    (a failed backup must not leave the install with no backup at all).
    Directories sorting at or after this run's ``stamp`` are skipped, so a concurrent
    backup that started later is never swept; one that started earlier on this
    database cannot be mid-backup now, because the caller holds the migration
    advisory lock that run would need.
    """
    if not backups_dir.is_dir():
        return []
    removed = []
    for path in sorted(backups_dir.iterdir()):
        if not path.is_dir() or path.name >= stamp or _manifest_parses(path / "manifest.json"):
            continue
        if not unreadable_is_residue and (path / "manifest.json").is_file():
            continue
        try:
            shutil.rmtree(path)
            removed.append(path)
            logger.info("swept incomplete backup directory %s (manifest unreadable or absent)", path)
        except OSError as error:
            logger.warning("cannot sweep incomplete backup directory %s: %s", path, error)
    return removed


def apply_retention(backups_dir: Path, keep: int = BACKUP_RETENTION) -> list[Path]:
    """Delete complete backups beyond the newest ``keep`` (by stamp); returns the deleted paths."""
    complete = sorted((p for p in backups_dir.iterdir() if _manifest_parses(p / "manifest.json")), key=lambda p: p.name)
    doomed = complete[:-keep] if len(complete) > keep else []
    for path in doomed:
        shutil.rmtree(path)
    return doomed


def _archive_keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    # The sarcophagus shape: everything the install ships except venvs (rebuildable),
    # bytecode caches, and the per-user data tree (its own archive member).
    parts = Path(info.name).parts
    if "venv" in parts or "__pycache__" in parts or "data" in parts or info.name.endswith(".pyc"):
        return None
    return info


def _write_code_archive(target: Path) -> None:
    """The program files as ``app-code.tar.gz``: the install tree (with its code
    history) and the sibling logs directory, no venvs, no per-user data, no
    bytecode caches."""
    tar_path = target / "app-code.tar.gz"
    with tarfile.open(tar_path, "x:gz") as tar:
        tar.add(REPO_ROOT, arcname="app", filter=_archive_keep)
        logs = REPO_ROOT.parent / "logs"
        if logs.is_dir():
            tar.add(logs, arcname="logs")
    _fsync_path(tar_path)


def _write_user_data_archive(target: Path) -> None:
    """The per-user tool data as ``data-users.tar.gz``: a copy of the install's
    data directory in which every SQLite database is a consistent snapshot taken
    from the live file through the read-only backup API — the live files are
    never opened for writing and the install is not touched."""
    src_root = REPO_ROOT / "data"
    if not src_root.is_dir():
        return
    tar_path = target / "data-users.tar.gz"
    with tempfile.TemporaryDirectory(prefix="mira-userdata-") as tmp:
        copy_root = Path(tmp) / "data"
        shutil.copytree(src_root, copy_root)
        for copied in copy_root.rglob("*.db"):
            live = src_root / copied.relative_to(copy_root)
            source = sqlite3.connect(f"file:{live}?mode=ro", uri=True)
            snapshot = sqlite3.connect(copied)
            source.backup(snapshot)
            snapshot.close()
            source.close()
            for sidecar in (f"{copied}-wal", f"{copied}-shm"):
                Path(sidecar).unlink(missing_ok=True)
        with tarfile.open(tar_path, "x:gz") as tar:
            tar.add(copy_root, arcname="data")
    _fsync_path(tar_path)


def user_backup(
    conninfo: str,
    output_dir: Path,
    *,
    emit: Emit,
    vault: VaultAdapter | None = None,
    env_file: Path = DEFAULT_ENV_FILE,
    version: str = "",
) -> int:
    """
    Write a full backup of the install into a directory the operator chose,
    without touching the install: the standard stamp members — restorable with
    ``mira restore <stamp> --backups-dir <dir>`` — plus the two archive members
    the update's own stamps never carry (program files, per-user tool data).
    pg_dump is transactional, the SQLite snapshots are point-in-time, and the
    destination is never swept or retention-counted (only ``/opt/mira/backups``
    carries that machinery).

    Returns the CLI exit code (0 written, 1 could not).
    """
    vault = vault if vault is not None else VaultAdapter()
    output_dir = output_dir.expanduser().resolve()
    try:
        if not output_dir.is_dir():
            # 0700 from creation — the archive carries every Vault credential.
            # An existing directory is never chmod'd; it is the operator's.
            output_dir.mkdir(parents=True, mode=0o700)
    except OSError as error:
        logger.error("cannot create %s: %s", output_dir, error)
        emit(operator_message("S-BACKUP-FAILED"))
        return 1
    stamp = default_stamp(output_dir)
    target = output_dir / stamp
    try:
        take_backup(conninfo, output_dir, stamp, vault=vault, env_file=env_file,
                    version=version, sweep=False, grant=False)
    except BackupBlocked as error:
        logger.error("user backup blocked: %s", error)
        emit(error.message)
        return 1
    try:
        _write_code_archive(target)
        _write_user_data_archive(target)
        _fsync_path(target)
    except (OSError, sqlite3.Error, tarfile.TarError, ValueError) as error:
        # The database stamp is complete and restorable; only the archive
        # members failed. Remove the partial member files rather than leave a
        # torn archive beside a valid stamp — the detail goes to the log.
        for name in ("app-code.tar.gz", "data-users.tar.gz"):
            (target / name).unlink(missing_ok=True)
        logger.error("user backup %s: archive member failed: %s: %s", target, type(error).__name__, error)
        emit(operator_message("S-BACKUP-PARTIAL", target=target, stamp=stamp, parent=output_dir))
        return 1
    logger.info("user backup %s complete (program files and per-user data included)", target)
    emit(operator_message("S-BACKUP-OK", target=target, stamp=stamp, parent=output_dir))
    return 0


def bootstrap_ownership(conninfo: str) -> None:
    """
    Run ``deploy/bootstrap_adopt.sql`` through the superuser seam (fed on stdin,
    so the superuser needs no read access to the install tree).

    Raises:
        SeamUnavailable: No way to act as the superuser.
        RuntimeError: psql failed.
    """
    params = _conn_params(conninfo)
    psql = shutil.which("psql")
    if psql is None:
        raise SeamUnavailable("psql not on PATH")
    argv = [psql, "-X", "-v", "ON_ERROR_STOP=1", "-d", params["dbname"], "-f", "-"]
    if params.get("port"):
        argv[1:1] = ["-p", str(params["port"])]
    result = run_as_superuser(argv, SEAM_TIMEOUT_SECONDS, stdin_text=(REPO_ROOT / "deploy" / "bootstrap_adopt.sql").read_text())
    if result.returncode != 0:
        raise RuntimeError(f"bootstrap_adopt.sql failed (exit {result.returncode}): {result.stderr.strip()}")


def objects_not_owned_by_admin(conn: psycopg.Connection) -> int:
    """Non-extension relations and functions in public that mira_admin does not own."""
    return int(conn.execute(
        """SELECT (SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public' AND c.relkind IN ('r','p','v','m','S','f')
                      AND c.relowner <> 'mira_admin'::regrole
                      AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid = 'pg_class'::regclass
                                      AND d.objid = c.oid AND d.deptype = 'e'))
                + (SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
                    WHERE n.nspname = 'public' AND p.proowner <> 'mira_admin'::regrole
                      AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid = 'pg_proc'::regclass
                                      AND d.objid = p.oid AND d.deptype = 'e'))"""
    ).fetchone()[0])


def default_stamp(backups_dir: Path) -> str:
    """A backup name in deploy/update.sh's STAMP format, suffixed -2, -3, … when that second is taken."""
    from utils.timezone_utils import utc_now

    base = utc_now().strftime("%Y%m%dT%H%M%SZ")
    candidate, n = base, 1
    while (backups_dir / candidate).exists():
        n += 1
        candidate = f"{base}-{n}"
    return candidate


def migrate(
    conninfo: str,
    *,
    emit: Emit,
    migrations_dir: Path = MIGRATIONS_DIR,
    backups_dir: Path = DEFAULT_BACKUPS_DIR,
    stamp: str,
    adopt_drifted: bool,
    applied_by: str,
    env_file: Path = DEFAULT_ENV_FILE,
    vault: VaultAdapter | None = None,
    operator_input: Callable[[str], str] = _operator_input,
) -> int:
    """
    Bring the database to the tree's head; returns the CLI exit code
    (0 ok, 1 a step failed after the backup, 2 refusal, 3 blocked).

    Order: lock → classify (read-only) → refusals → ownership bootstrap (ledger
    absent) → backup (when anything will change) → stamp the matched BASELINE
    rows → each versioned step in its own transaction with its ledger row →
    due repeatables → retention → structure check against HEAD_FINGERPRINT.
    """
    vault = vault if vault is not None else VaultAdapter()
    chain, refusals = load_chain(migrations_dir)
    if chain is None:
        for item in refusals:
            emit(item.message)
        return 2
    with connect(conninfo) as conn:
        try:
            with migration_lock(conn):
                return _migrate_locked(conn, conninfo, chain, emit=emit, migrations_dir=migrations_dir,
                                       backups_dir=backups_dir, stamp=stamp, adopt_drifted=adopt_drifted,
                                       applied_by=applied_by, env_file=env_file, vault=vault,
                                       operator_input=operator_input)
        except MigrationLocked as error:
            logger.warning("advisory lock wait exceeded: %s", error)
            emit(operator_message("S-LOCKED"))
            return 3


def _migrate_locked(conn, conninfo, chain, *, emit, migrations_dir, backups_dir, stamp, adopt_drifted,
                    applied_by, env_file, vault, operator_input) -> int:
    try:
        baseline = _baseline_if_present(migrations_dir, chain)
    except (ValueError, KeyError, TypeError) as error:
        emit(_payload_refusal(error, "BASELINE").message)
        return 2
    view = classify_database(conn, baseline)
    if view.state == "not_installed":
        emit(refusal("NOT_INSTALLED", "no MIRA tables in the database").message)
        return 2
    if view.state == "ledger_rows" and adopt_drifted:
        emit(operator_message("S-ADOPT-IGNORED"))
        adopt_drifted = False
    drifted = view.adoption_needed and view.matched is None
    if drifted and not adopt_drifted:
        try:
            head_fp = read_head_fingerprint(migrations_dir, baseline)
        except (ValueError, KeyError, TypeError) as error:
            emit(_payload_refusal(error, "HEAD_FINGERPRINT.json").message)
            return 2
        emit(render_drift_report(_fingerprint_from(conn, include_ledger=False), head_fp))
        emit(refusal("ADOPTION_DRIFT", f"structure {view.fingerprint_sha256} matches no BASELINE row").message)
        return 2
    rows = read_ledger_rows(conn) if view.state == "ledger_rows" else []
    code_version = applied_by
    validation = validate(chain, rows)
    if not validation.ok:
        for item in validation.refusals:
            emit(item.message)
        return 2

    to_run = steps_to_run(chain, rows, view)
    stamped = [s for s in chain.versioned if view.adoption_needed and not drifted
               and view.matched is not None and s.number <= int(view.matched)]
    if drifted:
        applied_by = f"{applied_by}+adopt-drifted"

    needs_typing = [s for s in to_run if s.requires_operator_input]
    if needs_typing and operator_input is _operator_input and not sys.stdin.isatty():
        for step in needs_typing:
            emit(f"  {step.requires_operator_input}")
        emit(operator_message("S-OPINPUT"))
        return 3
    # The ownership bootstrap runs before the backup whenever anything is not mira_admin's
    # (always on a pre-mechanism install; later, a table someone created as the superuser):
    # pg_dump as mira_admin cannot read such an object. Idempotent; the structure
    # (fingerprint) is unchanged by it.
    if (to_run or view.adoption_needed) and objects_not_owned_by_admin(conn):
        try:
            bootstrap_ownership(conninfo)
        except SeamUnavailable as error:
            logger.error("superuser seam unavailable: %s", error)
            emit(operator_message("S-NOSEAM"))
            print("ADOPT-PROBE-BLOCKED", file=sys.stderr)
            return 3
        except RuntimeError as error:
            logger.error("%s", error)
            emit(operator_message("S-NOSEAM"))
            return 3
    if to_run or view.adoption_needed:
        try:
            take_backup(conninfo, backups_dir, stamp, vault=vault, env_file=env_file, version=code_version)
        except BackupBlocked as error:
            logger.error("backup blocked: %s", error)
            emit(error.message)
            return 3
        emit(f"Backed up your data to {backups_dir / stamp}.")
    if stamped:
        with conn.transaction():
            for step in stamped:
                _record(conn, step, "baseline", 0)
        logger.info("stamped %s..%s from BASELINE match", stamped[0].revision, stamped[-1].revision)

    ctx = StepContext(pg=conn, vault=vault, env=EnvFileAdapter(env_file), operator_input=operator_input)
    for index, step in enumerate(to_run):
        name = step.slug
        if drifted and step.revision == "0001":
            name = f"{step.slug} adopt-drifted={(view.fingerprint_sha256 or '')[:12]}"
        what = step.notice or step.path.name
        emit(f"Running: {what}")
        started = time.monotonic()
        try:
            with conn.transaction():
                run_step(ctx, step)
                _record(conn, step, applied_by, int((time.monotonic() - started) * 1000), name)
        except OperatorInputRequired as error:
            emit(f"  {error.prompt}")
            emit(operator_message("S-OPINPUT"))
            return 3
        except Exception as error:
            logger.error("step %s (%s) failed: %s: %s", step.ledger_key, step.slug, type(error).__name__, error)
            emit(operator_message("S-STEP-FAILED", what=what, backup=backups_dir / stamp, stamp=stamp))
            return 1

    if to_run:
        for path in apply_retention(backups_dir):
            emit(f"Removed an older backup: {path}")
    final_rows = read_ledger_rows(conn)
    post_drift_reported = False
    payload_note = False
    if ledger_head(final_rows) == chain.head and (migrations_dir / "HEAD_FINGERPRINT.json").is_file():
        observed = _fingerprint_from(conn, include_ledger=False)
        try:
            expected = read_head_fingerprint(migrations_dir, baseline)
        except (ValueError, KeyError, TypeError) as error:
            # The steps themselves completed; only the belt-and-braces final check cannot run.
            # Exiting nonzero here would roll back a finished upgrade, and rerun forever.
            logger.warning("release payload refused: HEAD_FINGERPRINT.json: %s: %s", type(error).__name__, error)
            payload_note = True
            emit(operator_message("S-PAYLOAD-POST"))
        else:
            if fingerprint_sha256(observed) != fingerprint_sha256(expected):
                logger.warning("post-run structure %s differs from head %s",
                               fingerprint_sha256(observed), fingerprint_sha256(expected))
                post_drift_reported = True
                emit(operator_message("S-POSTDRIFT"))
                emit(render_drift_report(observed, expected))
    # S-NOTHING here would contradict the drift warning printed just above it.
    if not to_run and not stamped and not post_drift_reported and not payload_note:
        emit(operator_message("S-NOTHING"))
    return 0


def list_backups(backups_dir: Path) -> list[str]:
    if not backups_dir.is_dir():
        return []
    return sorted(p.name for p in backups_dir.iterdir() if _manifest_parses(p / "manifest.json"))


class DumpUnreadable(RuntimeError):
    """The backup's dump itself cannot be read back — the park story must not claim it is safe."""


def _restore_park(emit: Emit, manifest: BackupManifest, target: Path, *, dump_unreadable: bool = False) -> None:
    # Hand commands must run in a FRESH terminal, where this process's
    # SUDO_ASKPASS is unset and a cached sudo ticket cannot be assumed: the
    # askpass/-n forms superuser_prefix() would print abort there, so print
    # the plain interactive spelling a human can answer.
    prefix = "sudo -u postgres " if superuser_prefix() else ""
    port = f"-p {manifest.port} " if manifest.port != "5432" else ""
    if dump_unreadable:
        # The restore refused at the pg_restore --list precheck: the dump is
        # the reason. Running the dropdb sequence against it would destroy
        # the half-migrated database and leave it empty (campaign finding
        # KVM-9) — point at the older stamps instead, never at this one.
        others = [name for name in list_backups(target.parent) if name != target.name]
        if others:
            emit(operator_message("S-PARK-DAMAGED", stamp=target.name, available=", ".join(others)))
        else:
            emit(operator_message("S-PARK-DAMAGED-ALONE", stamp=target.name))
        return
    emit("The upgrade failed and MIRA could not put your data back automatically.")
    emit(f"Your backup is safe at: {target}")
    emit("These commands put your database back exactly like MIRA would have, whatever state it is in now — run them by hand:")
    emit(f"  {prefix}dropdb --force {port}{manifest.database}")
    emit(f"  {prefix}createdb -O mira_admin {port}{manifest.database}")
    emit(f"  {prefix}pg_restore {port}-d {manifest.database} {target / manifest.dump_file}")


def restore(
    conninfo: str,
    *,
    emit: Emit,
    backups_dir: Path = DEFAULT_BACKUPS_DIR,
    stamp: str,
    vault: VaultAdapter | None = None,
) -> int:
    """
    Put the database (and the Vault fields and settings file) back exactly as
    the backup recorded them; returns 0 restored, 1 failed (S-RESTORE-PARK
    printed — never retried, never reported as success).
    """
    vault = vault if vault is not None else VaultAdapter()
    target = backups_dir / stamp
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        available = list_backups(backups_dir)
        if available:
            emit(operator_message("S-RESTORE-NOBACKUP", stamp=stamp, backups_dir=backups_dir,
                                  available=", ".join(available)))
        else:
            emit(operator_message("S-RESTORE-NOBACKUP-NONE", stamp=stamp, backups_dir=backups_dir))
        return 1
    try:
        manifest = BackupManifest.model_validate_json(manifest_path.read_text())
    except (OSError, ValueError) as error:
        # KVM-13: an unreadable manifest refuses here, before anything changes — never a raw
        # pydantic traceback. The dump may still be intact, so this is not the park's
        # dump-damaged story: no destroy warning, no hand commands, do-not-guess guidance.
        logger.error("backup %s has an unreadable manifest: %s: %s", target, type(error).__name__, error)
        others = [name for name in list_backups(backups_dir) if name != stamp]
        if others:
            emit(operator_message("S-RESTORE-UNREADABLE", stamp=stamp, backups_dir=backups_dir,
                                  available=", ".join(others)))
        else:
            emit(operator_message("S-RESTORE-UNREADABLE-ALONE", stamp=stamp, backups_dir=backups_dir))
        return 1
    dump = target / manifest.dump_file
    try:
        with connect(psycopg.conninfo.make_conninfo(conninfo, dbname="postgres")) as conn:
            server_major = _server_major(conn)
        pg_restore = pg_tool("pg_restore", server_major)
        listing = subprocess.run([pg_restore, "--list", str(dump)], capture_output=True, text=True,
                                 timeout=SEAM_TIMEOUT_SECONDS)
        if listing.returncode != 0:
            raise DumpUnreadable(f"pg_restore --list {dump} exit {listing.returncode}: {listing.stderr.strip()}")
        if superuser_prefix():
            # Best-effort: a pre-ACL-era stamp carries no grants of its own, and the
            # park's remedy is `mira restore` — the restore must fix the access itself
            # or fail honestly at the seam below (never mid-dropdb).
            try:
                _grant_backup_read_access(target.parent, target, dump)
            except BackupBlocked as error:
                logger.warning("could not re-grant backup read access (%s); continuing", error)
        port = ["-p", manifest.port]
        for argv, timeout in (
            ([shutil.which("dropdb") or "dropdb", *port, "--force", manifest.database], SEAM_TIMEOUT_SECONDS),
            ([shutil.which("createdb") or "createdb", *port, "-O", "mira_admin", manifest.database], SEAM_TIMEOUT_SECONDS),
            ([pg_restore, *port, "--exit-on-error", "-d", manifest.database, str(dump)], PG_RESTORE_TIMEOUT_SECONDS),
        ):
            result = run_as_superuser(argv, timeout)
            if result.returncode != 0:
                raise RuntimeError(f"{os.path.basename(argv[0])} exit {result.returncode}: {result.stderr.strip()}")
        with connect(conninfo) as conn:
            present = {row[0] for row in conn.execute("SELECT extname FROM pg_extension")}
            missing = [ext for ext in REQUIRED_EXTENSIONS if ext not in present]
            if missing:
                raise RuntimeError(f"restored database lacks extensions {missing}")
        if manifest.env_file_present:
            env_target = Path(manifest.env_file)
            shutil.copy2(target / "systemone.env", env_target)
            os.chmod(env_target, 0o600)
        secrets = json.loads((target / "vault.json").read_text())
        for path, fields in secrets.items():
            changed = vault.patch(path, fields)
            if changed:
                logger.info("restored %d Vault fields in %s", len(changed), path)
        with connect(conninfo) as conn:
            restored = fingerprint_sha256(_fingerprint_from(conn, include_ledger=True))
        if restored != manifest.fingerprint_sha256:
            raise RuntimeError(f"restored structure {restored[:12]} differs from the backup's {manifest.fingerprint_sha256[:12]}")
    except (BackupBlocked, SeamUnavailable, RuntimeError, OSError, ValueError, subprocess.TimeoutExpired, DatabaseUnreachable) as error:
        logger.error("restore of %s failed: %s: %s", target, type(error).__name__, error)
        _restore_park(emit, manifest, target, dump_unreadable=isinstance(error, DumpUnreadable))
        return 1
    emit(operator_message("S-RESTORE-OK", stamp=stamp))
    return 0
