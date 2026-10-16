"""
Database step runner CLI — the operator and installer surface over
``utils/schema_migrations.py``. Argument parsing and rendering only; all
logic lives in the runner.

    python -m scripts.migrate migrate [--plan|--dry-run] [--report-marker] [--elevation E]
                                      [--stamp STAMP] [--adopt-drifted]
                                      [--migrations-dir DIR] [--backups-dir DIR]
    python -m scripts.migrate restore <STAMP> [--backups-dir DIR]
    python -m scripts.migrate status [--json] [--migrations-dir DIR]
    python -m scripts.migrate lint [--migrations-dir DIR]
    python -m scripts.migrate reconcile [--env-file FILE]   (drops MIRA_* settings nothing reads; no Vault)
    python -m scripts.migrate generate [--superuser NAME] [--host HOST] [--port PORT]

Run from the install root with ``PYTHONPATH`` set to it and the AppRole env
(``VAULT_ADDR``, ``VAULT_ROLE_ID``, ``VAULT_SECRET_ID``) exported — the
database login is ``mira_admin``'s URL from Vault. ``lint`` opens no
connection and reads no Vault. ``generate`` is the release-cut generator
(``deploy/RELEASE.md``): it rewrites the generated ledger DDL, then
``BASELINE`` + ``HEAD_FINGERPRINT.json`` from throwaway databases on a local
server it reaches as a PostgreSQL superuser — maintainer-only, never run on an
install, and it reads no Vault.

Exit codes: 0 ok / nothing to do; 1 a step failed after the backup was
taken; 2 validation, lint, or drift refusal (nothing changed); 3 blocked — no
superuser access, no terminal for a typed value, backup failed or disk
short, database or Vault unreachable.
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
from pathlib import Path
from typing import Callable

PLAN_BEGIN = "MIRA_PLAN_BEGIN"
PLAN_END = "MIRA_PLAN_END"
EXIT_OK, EXIT_STEP_FAILED, EXIT_REFUSED, EXIT_BLOCKED = 0, 1, 2, 3

VAULT_UNREADABLE = (
    "MIRA could not read its database settings from Vault, so it cannot check or upgrade your database. "
    "Nothing was changed. Make sure Vault is running and unsealed, then run: mira migrate"
)
RECONCILE_DONE = (
    "MIRA removed {count} setting(s) this version no longer uses from {env_file}. Everything else in "
    "that file is unchanged; the removed lines are in the update log. To see the file: cat {env_file}"
)
DATABASE_UNREACHABLE = (
    "MIRA could not reach its database. Nothing was changed. Make sure PostgreSQL is running, "
    "then run: mira migrate"
)


def vault_admin_conninfo() -> str:
    """mira_admin's conninfo from Vault, parsed by the app's one URL parser."""
    from clients.postgres_client import PostgresClient
    from clients.vault_client import get_database_url

    return PostgresClient._parse_database_url(get_database_url("mira_service", admin=True))


def _parser() -> argparse.ArgumentParser:
    from utils.schema_migrations import MIGRATIONS_DIR

    parser = argparse.ArgumentParser(prog="mira migrate", description="MIRA database step runner")
    sub = parser.add_subparsers(dest="command", required=True)

    migrate = sub.add_parser("migrate", help="apply pending database steps")
    mode = migrate.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="print what would run and whether consent is needed")
    mode.add_argument("--dry-run", action="store_true", help="print what would run; change nothing")
    migrate.add_argument("--report-marker", action="store_true", help="with --plan: append the JSON plan between markers")
    migrate.add_argument("--elevation", choices=("passwordless", "interactive", "none", "unavailable"), default="none")
    migrate.add_argument("--stamp", help="backup directory name (deploy/update.sh passes its run STAMP)")
    migrate.add_argument("--adopt-drifted", action="store_true",
                         help="adopt a database whose structure matches no known release (recorded in the ledger)")
    migrate.add_argument("--applied-by", help="label recorded for each step (default: this tree's VERSION; "
                         "deploy/update.sh passes nightly-<sha> for a nightly)")
    migrate.add_argument("--env-file", type=Path, default=Path("/opt/mira/systemone.env"))
    migrate.add_argument("--migrations-dir", type=Path, default=MIGRATIONS_DIR)
    migrate.add_argument("--backups-dir", type=Path, default=Path("/opt/mira/backups"))

    restore = sub.add_parser("restore", help="restore a backup taken before an upgrade")
    restore.add_argument("stamp", nargs="?")
    restore.add_argument("--backups-dir", type=Path, default=Path("/opt/mira/backups"))

    backup = sub.add_parser("backup", help="write a full backup of the install to a directory you choose")
    backup.add_argument("--output", type=Path, default=Path.home() / "mira-backups",
                         help="destination directory, created 0700 if missing (default: %(default)s)")
    backup.add_argument("--env-file", type=Path, default=Path("/opt/mira/systemone.env"))

    status = sub.add_parser("status", help="report where the database stands")
    status.add_argument("--json", action="store_true")
    status.add_argument("--migrations-dir", type=Path, default=MIGRATIONS_DIR)

    lint = sub.add_parser("lint", help="check the step files against the authoring rules")
    lint.add_argument("--migrations-dir", type=Path, default=MIGRATIONS_DIR)

    reconcile = sub.add_parser("reconcile", help="remove settings this version no longer uses from the settings file")
    reconcile.add_argument("--env-file", type=Path, default=Path("/opt/mira/systemone.env"))

    generate = sub.add_parser("generate", help="release cut: regenerate the ledger DDL, BASELINE, HEAD_FINGERPRINT.json")
    generate.add_argument("--superuser", default=getpass.getuser(), help="a PostgreSQL superuser on the local server")
    generate.add_argument("--host", default="localhost")
    generate.add_argument("--port", default="5432")
    return parser


def _print_refusals(refusals, drift_report: str | None = None) -> None:
    if drift_report:
        print(drift_report)
    for refusal in refusals:
        print(refusal.message)


def _cmd_lint(args) -> int:
    from utils.schema_migrations import discover_steps, lint_steps, operator_message, order_steps, StepOrderError

    try:
        chain = order_steps(discover_steps(args.migrations_dir))
    except (StepOrderError, ValueError) as error:
        print(f"lint: {error}")
        return EXIT_REFUSED
    findings = lint_steps(chain)
    for finding in findings:
        print(operator_message("S-LINT", file=finding.file))
        print(f"  rule: {finding.rule}")
    if not findings:
        print(f"lint: {len(chain.versioned)} versioned and {len(chain.repeatable)} repeatable steps follow the rules.")
    return EXIT_REFUSED if findings else EXIT_OK


def _cmd_status(args, conninfo: str) -> int:
    from utils.schema_migrations import status

    report = status(conninfo, args.migrations_dir)
    if args.json:
        print(report.model_dump_json())
    else:
        print(f"Database steps applied: {report.ledger_head:04d} of {report.tree_head:04d} this version knows.")
        if report.pending or report.repeatable_due:
            print("Waiting to run: " + ", ".join(report.pending + report.repeatable_due) + " — run: mira update")
        if report.fingerprint_drift:
            print("Your database has changes MIRA did not make. Run: mira migrate to see the list.")
        if report.adopted_with_drift_override:
            print("It was upgraded with its own changes kept (mira update --adopt-drifted).")
        _print_refusals(report.refusals)
    return EXIT_REFUSED if report.refusals else EXIT_OK


def _cmd_plan(args, conninfo: str) -> int:
    from utils.schema_migrations import plan, render_plan

    report = plan(conninfo, args.migrations_dir, adopt_drifted=args.adopt_drifted)
    if report.refusals:
        _print_refusals(report.refusals, report.drift_report)
    else:
        print(render_plan(report, args.elevation))
    if args.dry_run:
        print("This was a dry run; nothing was changed.")
    if args.report_marker:
        payload = {
            "needs_consent": report.needs_consent,
            "adoption_needed": report.adoption_needed,
            "pending_notices": [report.notices[name] for name in report.pending],
            "repeatable_due": report.repeatable_due,
            "operator_input": report.operator_input,
            "refusals": [r.model_dump() for r in report.refusals],
        }
        print(PLAN_BEGIN)
        print(json.dumps(payload))
        print(PLAN_END)
    return EXIT_REFUSED if report.refusals else EXIT_OK


def _cmd_migrate(args, conninfo: str) -> int:
    from utils.release_identity import get_current_version
    from utils.schema_migrations import default_stamp, migrate

    return migrate(
        conninfo,
        emit=print,
        migrations_dir=args.migrations_dir,
        backups_dir=args.backups_dir,
        stamp=args.stamp or default_stamp(args.backups_dir),
        adopt_drifted=args.adopt_drifted,
        applied_by=args.applied_by or get_current_version(),
        env_file=args.env_file,
    )


def _cmd_backup(args, conninfo: str) -> int:
    from utils.release_identity import get_current_version
    from utils.schema_migrations import user_backup

    return user_backup(conninfo, args.output, emit=print, env_file=args.env_file,
                       version=get_current_version())


def _cmd_reconcile(args) -> int:
    from utils.env_reconcile import declared_environment_variables, reconcile_env_file
    from utils.schema_migrations import REPO_ROOT, EnvFileAdapter

    env = EnvFileAdapter(args.env_file)
    if not args.env_file.exists():
        print(f"No settings file at {args.env_file}; nothing to update.")
        return EXIT_OK
    result = reconcile_env_file(env.lines(), declared_environment_variables(REPO_ROOT))
    log = logging.getLogger("scripts.migrate")
    for item in result.dropped:
        log.info("settings file: dropped %s (nothing in this version reads it): %r", item.name, item.line)
    if result.dropped:
        env.write_lines(list(result.rendered))
        print(RECONCILE_DONE.format(count=len(result.dropped), env_file=args.env_file))
    return EXIT_OK


def _cmd_generate(args) -> int:
    from clients.embeddings_provider import describe_for_installer
    from utils.schema_migrations import (ConvergenceError, baseline_fingerprints, write_baseline,
                                         write_generated_artifacts)

    for path in write_generated_artifacts():
        print(f"rewrote the generated ledger DDL in {path}")
    model, dimensions = describe_for_installer(["describe", "local"]).split()
    schema_args = ["-v", "embedding_provider=local", "-v", f"embedding_model={model}", "-v", "embedding_endpoint_url=",
                   "-v", "embedding_api_key_name=", "-v", f"embedding_dimensions={dimensions}"]
    superuser = {"user": args.superuser, "host": args.host, "port": args.port}
    try:
        generated = baseline_fingerprints(superuser, schema_args)
    except ConvergenceError as error:
        print(f"DO NOT PUBLISH: {error.origin} plus the step chain does not reach the fresh-install structure:")
        print(error.report)
        return EXIT_REFUSED
    write_baseline(generated)
    print(generated.text, end="")
    print("wrote deploy/migrations/BASELINE and deploy/migrations/HEAD_FINGERPRINT.json")
    return EXIT_OK


class _DropVaultToast(logging.Filter):
    """Drop the Vault client's TOAST (60) lifecycle records from this runner's
    stderr — the operator transcript during an update's database steps; every
    other level (ERROR/WARNING/INFO) passes through."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != 60


def main(argv: list[str] | None = None, resolve_conninfo: Callable[[], str] = vault_admin_conninfo) -> int:
    """Parse ``argv`` and run one subcommand; ``resolve_conninfo`` is called only by commands that open the database."""
    # deploy/update.sh pipes this through tee: block-buffered stdout would hold
    # "Running: …" until a long step finished, and print it after the log lines.
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("clients.vault_client").addFilter(_DropVaultToast())
    args = _parser().parse_args(argv)
    if args.command == "lint":
        return _cmd_lint(args)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "reconcile":
        return _cmd_reconcile(args)
    if args.command == "restore" and not args.stamp:
        from utils.schema_migrations import list_backups

        available = ", ".join(list_backups(args.backups_dir)) or "none"
        print(f"Name the backup to restore. Backups in {args.backups_dir}: {available}. Run: mira restore <name>")
        return EXIT_STEP_FAILED

    from utils.schema_migrations import DatabaseUnreachable

    try:
        conninfo = resolve_conninfo()
    except Exception as error:
        logging.getLogger("scripts.migrate").error("Vault read failed: %s: %s", type(error).__name__, error)
        print(VAULT_UNREADABLE)
        return EXIT_BLOCKED
    try:
        if args.command == "status":
            return _cmd_status(args, conninfo)
        if args.command == "restore":
            from utils.schema_migrations import restore

            return restore(conninfo, emit=print, backups_dir=args.backups_dir, stamp=args.stamp)
        if args.command == "backup":
            return _cmd_backup(args, conninfo)
        if args.plan or args.dry_run:
            return _cmd_plan(args, conninfo)
        return _cmd_migrate(args, conninfo)
    except DatabaseUnreachable as error:
        logging.getLogger("scripts.migrate").error("database unreachable: %s", error)
        print(DATABASE_UNREACHABLE)
        return EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
