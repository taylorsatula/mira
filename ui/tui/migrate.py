"""
``mira status``, ``mira migrate``, ``mira restore`` — the install's history
and its database, from the terminal.

Runs on the TUI venv with the stdlib only (the TUI venv carries none of the
server's database or Vault dependencies). ``migrate``/``restore`` and the
database half of ``status`` run ``scripts/migrate.py`` on the install's
SERVER venv as a subprocess: arguments forwarded verbatim, exit code returned
unchanged. The child gets the AppRole the service uses (an operator's shell
carries none) and the PostgreSQL client tools of the server's major on PATH.

Environment: ``MIRA_APP_DIR`` — the install root (default ``/opt/mira/app``),
the same override ``ui/tui/update.py`` and ``deploy/update.sh`` honor; backups
and the settings file are its parent's ``backups/`` and ``systemone.env``.
The override scopes files, never the database (the AppRole and the connection
come from the host's Vault, not from the root): ``migrate``/``restore``/``backup`` refuse
with exit 3 and ``status`` skips its database half under a root that is not
the install root — the Vault-free ``lint``/``reconcile``/``generate`` commands
of the runner are not refused.

Exit codes: ``status`` 0 report printed, 1 not a MIRA install repository;
``migrate``/``restore`` — ``scripts/migrate.py``'s (0 ok, 1 step or restore
failed, 2 refused, 3 blocked).

Service management (canonical root only): ``mira restore`` stops the
install's running service first — a restore under a live server is a minute
of connection errors (M-4a) — and starts it again after success; a failed
restore leaves it stopped with the start command (the park doctrine,
``deploy/update.sh``'s twin). A successful ``migrate`` that ran while the
service was parked starts it (the boot gate re-verifies; ``--plan``,
``--dry-run``, and the Vault-free subcommands never do). The arms are
launchd and systemd; a supervisorless install gets the honest message and
the previous behavior.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

GIT_TIMEOUT_SECONDS = 30
STATUS_TIMEOUT_SECONDS = 120
# GIT_TIMEOUT bounds local git calls only (log/diff/ls-files on the install
# repo — never network): 30 s is generous headroom so a hung filesystem cannot
# hang `mira status`. STATUS_TIMEOUT bounds the runner child, whose own
# connect bound is 10 s (utils/schema_migrations.py:CONNECT_TIMEOUT_SECONDS)
# and whose slowest legal phase is the catalog fingerprint — 120 s is
# headroom over both, not a calibration.
VAULT_DIR = Path("/opt/vault")
DEFAULT_VAULT_ADDR = "http://127.0.0.1:8200"
VAULT_UNREADABLE = (
    "MIRA could not read its Vault login ({path}). Nothing was changed. Run this as the account "
    "MIRA was installed under, or check the file: sudo ls -l {path}"
)

# The one root whose database these commands may touch: the connection target
# is the host's Vault AppRole, which no MIRA_* override scopes — a root other
# than this one has no database of its own, and the one it would reach is the
# host install's. Twin of deploy/update.sh's CANONICAL gate.
CANONICAL_APP_DIR = Path("/opt/mira/app")
VAULT_FREE_COMMANDS = ("lint", "reconcile", "generate")
NON_CANONICAL_DATABASE_REFUSAL = (
    "MIRA_APP_DIR is set to {app_dir}, not the install root {canonical}. A throwaway root has no "
    "database of its own — these commands would reach the host install's database, not this "
    "tree's (MIRA_APP_DIR scopes files, never the database). Nothing was changed."
)

# Python twin of deploy/lib/custom_export.sh:MIRA_OFFICIAL_SUBJECT_ERE — the
# subjects deploy/lib/provenance.sh:official_commit_message writes, plus the
# pre-provenance `install <tag>[ (local tree)]` form. Change both together.
OFFICIAL_SUBJECT_RE = re.compile(
    r"^(install|update) (v[0-9]+(\.[0-9]+)*(-[0-9]+(\.[0-9]+)*)?|nightly-[0-9a-f]{12})"
    r"( @ [0-9a-f]{40}( local-dirty)?)?( \(local tree\))?$"
)


def _app_dir() -> Path:
    return Path(os.environ.get("MIRA_APP_DIR", "/opt/mira/app"))


def _git(app_dir: Path, *args: str) -> str:
    """
    Run git in ``app_dir`` and return stdout.

    Raises:
        subprocess.CalledProcessError: git exited nonzero.
        subprocess.TimeoutExpired: git exceeded ``GIT_TIMEOUT_SECONDS``.
    """
    result = subprocess.run(
        ["git", "-C", str(app_dir), *args],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=True,
    )
    return result.stdout


def official_baseline(app_dir: Path) -> tuple[str, str] | None:
    """The newest official install/update commit as ``(sha, subject)``; None when the history has none."""
    for line in _git(app_dir, "log", "--format=%H %s").splitlines():
        sha, _, subject = line.partition(" ")
        if OFFICIAL_SUBJECT_RE.match(subject):
            return sha, subject
    return None


def _indented(text: str, empty: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return f"  {empty}"
    return "\n".join(f"  {line}" for line in lines)


def runner_command(app_dir: Path, args: list[str]) -> tuple[list[str], dict[str, str]]:
    """
    argv and env for ``scripts/migrate.py`` on the install's server venv.

    Raises:
        OSError: The AppRole files are unreadable (message names the file).
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(app_dir)
    if not (env.get("VAULT_ROLE_ID") and env.get("VAULT_SECRET_ID")):
        for name, filename in (("VAULT_ROLE_ID", "role-id.txt"), ("VAULT_SECRET_ID", "secret-id.txt")):
            path = VAULT_DIR / filename
            try:
                env[name] = path.read_text().strip()
            except OSError as error:
                raise OSError(VAULT_UNREADABLE.format(path=path)) from error
    env.setdefault("VAULT_ADDR", DEFAULT_VAULT_ADDR)
    bindir = subprocess.run(
        ["bash", "-c", 'source deploy/lib/output.sh && source deploy/lib/services.sh && pg_bindir'],
        cwd=app_dir, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS,
    ).stdout.strip()
    if bindir:
        env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    return [str(app_dir / "venv" / "bin" / "python3"), "-P", "-m", "scripts.migrate", *args], env


def _install_paths(app_dir: Path, command: str, args: list[str]) -> list[str]:
    """The backups directory and settings file beside ``app_dir`` — where ``deploy/update.sh`` puts them — unless given."""
    given = {arg.split("=", 1)[0] for arg in args}
    added: list[str] = []
    if command in ("migrate", "restore") and "--backups-dir" not in given:
        added += ["--backups-dir", str(app_dir.parent / "backups")]
    if command in ("migrate", "backup") and "--env-file" not in given:
        added += ["--env-file", str(app_dir.parent / "systemone.env")]
    return added


SERVICE_PORT = 1993
# A port-up bound, not the full healthy-and-version poll — that is
# deploy/update.sh:wait_healthy's. 120 s matches the container image's
# HEALTHCHECK start-period (deploy/docker/Dockerfile), the slowest sanctioned
# first-boot allowance this repo names.
START_WAIT_SECONDS = 120
START_POLL_INTERVAL_SECONDS = 2
_LAUNCHD_PLIST = Path("Library/LaunchAgents/com.mira.app.plist")


def _port_answering() -> bool:
    """Whether the install's server is listening on its port (ground truth for stop/start — the same probe ``deploy/update.sh`` keys on)."""
    try:
        with socket.create_connection(("127.0.0.1", SERVICE_PORT), timeout=1):
            return True
    except OSError:
        return False


def _supervisor() -> str | None:
    """``"launchd"``, ``"systemd"``, or None — the arms ``deploy/update.sh``'s
    ``stop_mira_service``/``start_mira_service`` carry. A supervisorless
    install (a detached ``run.sh``) has no arm here."""
    if sys.platform == "darwin":
        return "launchd" if (Path.home() / _LAUNCHD_PLIST).is_file() else None
    probe = subprocess.run(["systemctl", "cat", "mira"], capture_output=True)
    return "systemd" if probe.returncode == 0 else None


def _start_hint() -> str:
    """The operator-runnable start command for this host's supervisor."""
    kind = _supervisor()
    if kind == "launchd":
        return "launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.mira.app.plist"
    if kind == "systemd":
        return "sudo systemctl start mira"
    return f"{CANONICAL_APP_DIR / 'run.sh'}"


def _stop_mira_service() -> str | None:
    """Stop the install's service; None once stopped, else the reason it was not."""
    if not _port_answering():
        return None
    kind = _supervisor()
    if kind == "launchd":
        # bootout, not kill: unloading the agent takes KeepAlive out of the
        # picture (the update.sh twin's reason). bootout of an unloaded agent
        # errors; the port re-check below is the ground truth either way.
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}/com.mira.app"],
            capture_output=True,
        )
    elif kind == "systemd":
        result = subprocess.run(["sudo", "-n", "systemctl", "stop", "mira"], capture_output=True, text=True)
        if result.returncode != 0:
            return "sudo needs a password — run: sudo systemctl stop mira, then re-run this command"
    else:
        return "no supervisor was found (no LaunchAgent, no systemd unit) — a MIRA started by hand cannot be stopped from here"
    if _port_answering():
        return f"the stop command ran but port {SERVICE_PORT} still answers"
    return None


def _start_mira_service() -> str | None:
    """Start the install's service; None once started, else the reason it was not."""
    kind = _supervisor()
    if kind == "launchd":
        result = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(Path.home() / _LAUNCHD_PLIST)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no detail"
            return f"launchctl failed: {detail}"
    elif kind == "systemd":
        result = subprocess.run(["sudo", "-n", "systemctl", "start", "mira"], capture_output=True, text=True)
        if result.returncode != 0:
            return "sudo needs a password — run: sudo systemctl start mira"
    else:
        return f"no supervisor was found — start it yourself: {_start_hint()}"
    return None


def _start_and_wait(lead: str) -> None:
    """Start the service, wait (bounded) for its port, and report honestly either way."""
    problem = _start_mira_service()
    if problem is None:
        deadline = time.monotonic() + START_WAIT_SECONDS
        while time.monotonic() < deadline:
            if _port_answering():
                print(f"{lead} MIRA is starting.")
                return
            time.sleep(START_POLL_INTERVAL_SECONDS)
        problem = f"it did not answer on port {SERVICE_PORT} within {START_WAIT_SECONDS} s — check its logs"
    print(f"{lead} Could not start MIRA ({problem}).")
    print(f"Start it yourself: {_start_hint()}")


def run_runner(command: str, args: list[str]) -> int:
    """``mira migrate …`` / ``mira restore …`` → ``scripts.migrate <command> …``, exit code unchanged.

    Under a ``MIRA_APP_DIR`` that is not the install root the database commands
    refuse (exit 3, nothing changed) before any child is started — the root
    they were pointed at has no database of its own. The runner's Vault-free
    commands (``lint``/``reconcile``/``generate``) are not refused.

    Under the install root the service is managed around the child: a running
    service is stopped before ``restore`` and started again after success
    (left stopped after failure), and a successful ``migrate``/``restore``
    that ran while the service was down starts it (never for ``--plan``,
    ``--dry-run``, or the Vault-free subcommands).
    """
    app_dir = _app_dir()
    subcommand = next((arg for arg in args if not arg.startswith("-")), "")
    if (
        command in ("migrate", "restore", "backup")
        and app_dir != CANONICAL_APP_DIR
        and subcommand not in VAULT_FREE_COMMANDS
    ):
        print(
            NON_CANONICAL_DATABASE_REFUSAL.format(app_dir=app_dir, canonical=CANONICAL_APP_DIR),
            file=sys.stderr,
        )
        return 3
    try:
        argv, env = runner_command(app_dir, [command, *args, *_install_paths(app_dir, command, args)])
    except OSError as error:
        print(str(error), file=sys.stderr)
        return 3
    # Service management applies only to the real install — the same
    # doctrine as deploy/update.sh's CANONICAL gate: a throwaway root's
    # commands never touch the host install's service.
    service_was_up = app_dir == CANONICAL_APP_DIR and _port_answering()
    stopped_for_restore = False
    if command == "restore" and service_was_up:
        reason = _stop_mira_service()
        if reason is None:
            stopped_for_restore = True
        else:
            # Today's behavior plus the truth about the window it leaves.
            print(f"mira restore: could not stop MIRA first ({reason}).")
            print("The restore still runs; MIRA will report errors until it finishes — restart it afterwards.")
    rc = subprocess.run(argv, env=env, cwd=app_dir).returncode
    if stopped_for_restore:
        if rc == 0:
            _start_and_wait("Your data is back.")
        else:
            # The park doctrine, twin of deploy/update.sh's failed-restore arm:
            # never restart onto a database the restore did not finish with.
            print("mira restore: MIRA is left stopped until your data is sorted out (the message above says where it stands).")
            print(f"Start it yourself when ready: {_start_hint()}")
    elif rc == 0 and not service_was_up and (
        command == "restore"
        or (
            command == "migrate"
            and subcommand not in VAULT_FREE_COMMANDS
            and not any(arg in ("--plan", "--dry-run") for arg in args)
        )
    ):
        # A parked service (the boot gate refused, or a reboot left it down)
        # stays parked after a successful maintenance run unless something
        # starts it. Start it; the boot gate re-verifies and stays the arbiter.
        _start_and_wait(f"The {command} ran while MIRA was not running.")
    return rc


def _database_status(app_dir: Path) -> str:
    """One plain-English paragraph on where the database stands (the runner's ``status --json``)."""
    try:
        argv, env = runner_command(app_dir, ["status", "--json"])
        result = subprocess.run(argv, env=env, cwd=app_dir, capture_output=True, text=True,
                                timeout=STATUS_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"Database: could not check ({error})."
    lines = result.stdout.strip().splitlines()
    try:
        report = json.loads(lines[-1])
    except (IndexError, ValueError):
        return "Database: could not check — " + " ".join(lines[-2:] or ["no answer"])
    text = [f"Database: {report['ledger_head']:04d} of {report['tree_head']:04d} upgrade steps applied."]
    if report["pending"] or report["repeatable_due"]:
        text.append("  Waiting to run (run: mira update): " + ", ".join(report["pending"] + report["repeatable_due"]))
    if report["fingerprint_drift"]:
        text.append("  Your database has changes MIRA did not make; MIRA keeps them (run: mira migrate to list them).")
    if report["adopted_with_drift_override"]:
        text.append("  It was upgraded with its own changes kept (mira update --adopt-drifted).")
    for refusal in report["refusals"]:
        text.append(f"  {refusal['message']}")
    return "\n".join(text)


def run_status() -> int:
    """Print the install's code history report; the exit code is 0 or 1 per the module docstring."""
    app_dir = _app_dir()
    try:
        _git(app_dir, "rev-parse", "--git-dir")
    except (subprocess.CalledProcessError, OSError):
        print(f"mira status: {app_dir} is not a MIRA install (no install history).", file=sys.stderr)
        return 1
    try:
        version = (app_dir / "VERSION").read_text().strip()
    except OSError as error:
        version = f"unreadable VERSION ({error})"

    baseline = official_baseline(app_dir)
    print(f"MIRA install: {app_dir} (version {version})")
    if baseline is None:
        print("Last official install or update: none recorded — this tree cannot be updated in place.")
        return 0
    sha, subject = baseline
    print(f"Last official install or update: {subject}")
    print(f"  commit {sha}")
    print("Changes to MIRA's files since then (committed and uncommitted):")
    print(_indented(_git(app_dir, "diff", "--stat", sha), "(none)"))
    print("Files added that git does not track:")
    print(_indented(_git(app_dir, "ls-files", "--others", "--exclude-standard"), "(none)"))
    print("Your own commits since then:")
    print(_indented(_git(app_dir, "log", "--oneline", f"{sha}..HEAD"), "(none)"))
    print("Stashed changes:")
    print(_indented(_git(app_dir, "stash", "list"), "(none)"))
    if app_dir == CANONICAL_APP_DIR:
        print(_database_status(app_dir))
    else:
        # The git half above is this tree's; the database half would read the
        # host install's database — MIRA_APP_DIR scopes files, never the
        # database — so under a non-install root it is skipped, not reported.
        print(
            f"Database: not checked — {app_dir} is not the install root, and MIRA_APP_DIR scopes "
            "files, never the database (the host install's database is not this tree's)."
        )
    return 0
