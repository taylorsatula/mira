"""
Guardrailed local shell execution on the machine MIRA runs on.

This tool runs shell commands LOCALLY on the machine MIRA runs on (a VM,
container, or host) via subprocess — no network transport. Commands execute as the service
user from a configured project root. Every command is validated against the
operator's blocklist and the destructive-command guardrail before anything
executes; a match is a hard refusal and the command never reaches the shell.

The operator blocklist (bash_tool config `blocked_patterns`) is matched first
and outranks every mode, including skip mode: entries are literal text except
`*`, which matches anything including slashes, spaces, quotes, and newlines.
It can make the tool refuse more, never less.

The two layers cover what the other cannot:

- **Pattern layer** — regexes over the raw command string. Because it sees the
  string before any quoting is removed, it catches destructive literals buried
  inside quotes or nested interpreters (`bash -c "rm -rf /"`,
  `python -c "os.system('rm -rf /')"`) that tokenization would hide.
- **Argument layer** — the command is split into the lines bash runs (heredoc
  bodies cut out as data), each line is tokenized and split into simple
  commands, and each command's verb and path operands are resolved against the
  effective working directory. This is what catches the operations no regex can
  express: relative paths (`rm -rf .` from the project root), quoted and globbed
  spellings of a protected path, shell expansions that cannot be resolved before
  the host shell sees them, respelled persistence targets (sudo config, SSH
  authorized_keys), and destructive verbs other than `rm`. Quoting it cannot
  balance is refused, never guessed at.

Both layers are a safety net, not a sandbox. They stop known-catastrophic
operations; they do not confine a determined command to the project tree, and
they cannot see through arbitrary obfuscation (base64 payloads, computed
strings). A refusal is a stop-work signal to the model, not a puzzle to route
around — every refusal message says so explicitly.

A human-gated escape hatch exists for the recoverable refusals. With the
operator's config switch `dangerous_skip_permissions_enabled` on, a call may
pass `skip_permissions=True` to run under the catastrophic core only: the
rules against system-ruining, unrecoverable, and audit-erasing actions stay,
while recoverable judgment calls (system-config overwrites, service control —
including restarting MIRA itself — package removal, the git revert rules) are
relaxed. The operator's blocklist is never relaxed, and neither are the
argument-layer checks that keep the core honest: unresolvable expansions,
container-emptying globs, persistence targets, the log trees, the bypass log
itself, and this file and MIRA's config directory. Every bypass is appended
to `guardrail_bypass.log` under the configured log_dir and marked in the
result, so it is visible in-transcript and on disk.
"""

import inspect
import logging
import os
import posixpath
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, NoReturn, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import utc_now
from utils.user_context import get_cancel_event, get_cancel_reason


# Home directory of the user MIRA runs as, whatever it is named. Path.home()
# raises when no home can be determined — a loud import failure, never a
# silently wrong working directory.
_SERVICE_HOME = str(Path.home())

# Foreground waits are sliced at this cadence so each slice re-checks the
# turn's cancel signal (the ``cancel_event`` contextvar). The tool loop
# honors cancellation only at tool boundaries, so a single
# communicate(timeout=effective) would make cancellation unreachable for
# up to max_timeout_seconds while the turn's deadline, the user's halt, and
# a client disconnect all wait. A slice of one second bounds the
# cancellation-to-kill latency to one slice plus the drain below, and the
# per-slice TimeoutExpired overhead is negligible against the ceiling.
_CANCEL_POLL_SECONDS = 1.0
# After a cancellation or timeout group kill, the pipes are drained within
# this budget; a drain that also expires means a descendant escaped the
# killed group and holds the inherited pipe ends, so the pipes are
# abandoned rather than blocking the caller on a read that can never EOF.
_CANCEL_DRAIN_SECONDS = 5.0

_BACKGROUND_JOBS_DDL = """
    pid TEXT PRIMARY KEY,
    pgid TEXT NOT NULL,
    log_path TEXT NOT NULL,
    command TEXT NOT NULL,
    started_at TEXT NOT NULL
"""


def ensure_background_jobs(db) -> None:
    """Create the background-jobs table if absent (idempotent).

    One row per run_background launch: the nohup'd child's pid, the process
    group that job and its descendants share, its log path, the command,
    and the start time. kill_background only ever stops pids present here,
    which is its whole safety property: the tool can only stop jobs it
    launched.
    """
    db.create_table("background_jobs", _BACKGROUND_JOBS_DDL)


class BashToolConfig(BaseModel):
    """Configuration for bash_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    root: str = Field(
        default=_SERVICE_HOME,
        description=(
            "Absolute local directory used as the default working directory for every command. "
            "Defaults to the home directory of the user MIRA runs as."
        ),
    )
    log_dir: str = Field(
        default=posixpath.join(_SERVICE_HOME, ".mira_logs"),
        description="Directory for run_background logs. Must resolve inside root.",
    )
    default_timeout_seconds: int = Field(default=120, description="Timeout applied when the caller gives none.")
    max_timeout_seconds: int = Field(default=1800, description="Ceiling a caller-supplied timeout is clamped to.")
    max_output_bytes: int = Field(default=100_000, description="Per-stream cap on returned stdout/stderr bytes.")
    background_start_timeout_seconds: int = Field(
        default=30,
        description=(
            "Budget for run_background's start sequence only (mkdir/cd/nohup launch). "
            "The detached job's own lifetime is never bounded by it."
        ),
    )
    dangerous_skip_permissions_enabled: bool = Field(
        default=False,
        description=(
            "Master gate for the skip_permissions parameter. Without it, any "
            "skip_permissions=True call errors. The operator's switch, not "
            "the model's."
        ),
    )
    blocked_patterns: List[str] = Field(
        default_factory=list,
        description=(
            "User-authored blocklist. Each entry is matched against the raw "
            "command string; all characters are literal except *, which matches "
            "anything including slashes, spaces, quotes, and newlines. Highest "
            "priority; not bypassable by skip mode."
        ),
    )

    @field_validator("blocked_patterns")
    @classmethod
    def _reject_blank_blocklist_entries(cls, value: List[str]) -> List[str]:
        """A blank entry compiles to a match-everything regex; refuse it at load."""
        for entry in value:
            if not entry.strip():
                raise ValueError("blocked_patterns entries must be non-empty strings")
        return value


registry.register("bash_tool", BashToolConfig)


# Every refusal the guardrail raises carries this text. It is addressed to the
# model: a blocked command means stop, not "find another spelling".
_ESCALATION = (
    "Do not try to circumvent this block. It was designed like this for a reason. "
    "If you cannot accomplish your goal without jumping this gate, stop work and "
    "wait for further direction from a human. No exceptions."
)


def _refuse(rule: str, reason: str, offending: str) -> NoReturn:
    """Raise the single refusal shape every guardrail path uses."""
    raise ValueError(
        f"Command blocked by the destructive-command guardrail ({rule}): {reason}. "
        f"Offending text: {offending!r}. {_ESCALATION}"
    )


def _compile_blocklist(
    patterns: Tuple[str, ...],
) -> Tuple[Tuple[str, "re.Pattern[str]"], ...]:
    """
    Compile operator blocklist entries to (original text, regex) pairs.

    Every character is literal except `*`, which is relaxed to `.*` so it
    crosses slashes, spaces, quotes, and newlines (re.DOTALL) — a blocklist
    over-matches rather than under-matches, so `systemctl restart *mira` still
    fires on a backslash-continued `systemctl restart \\<newline>mira`.
    Matching is re.search over the raw command string, the same surface the
    pattern layer sees, so it fires before quoting is removed. Compiled by the
    handlers (not the validator) and passed in as inert data; the protected
    battery's purity walk never sees a compiler.
    """
    compiled = []
    for raw in patterns:
        body = re.escape(raw).replace("\\*", ".*")
        compiled.append((raw, re.compile(body, re.DOTALL)))
    return tuple(compiled)


# -- pattern layer ------------------------------------------------------------
#
# Regexes over the raw command string. Each entry is (name, pattern, reason).
# Matching any of them is a hard refusal.
#
# _TRAILING is the set of characters that can legitimately follow a bare system
# directory in a destructive position. It includes quote characters so a path
# quoted inside a nested shell (`bash -c "rm -rf /"`) still matches.
_SYSTEM_DIRS = (
    "etc", "usr", "bin", "sbin", "lib", "lib64", "libx32", "boot", "dev",
    "proc", "sys", "var", "opt", "root", "srv", "run",
    # macOS top-level system directories: the same guardrail must protect
    # the platform's own FHS, or `rm -rf /System` sails through on the half
    # of the target matrix that does not use the Linux layout.
    "System", "Library", "Applications", "private", "Volumes", "cores",
)
_SYSTEM_DIR_ALT = "|".join(_SYSTEM_DIRS)
_SYSTEM_DIR_NAMES = frozenset(_SYSTEM_DIRS)
_TRAILING = r"""(?:/|\s|$|\*|&|\||;|\)|'|\")"""
_FLAGS_THEN = r"(?:-{1,2}[A-Za-z0-9-]+(?:[= ][^\s]*)?\s+)*"

_DESTRUCTIVE_PATTERNS: List[Tuple[str, "re.Pattern[str]", str]] = [
    (
        "no-preserve-root",
        re.compile(r"--no-preserve-root"),
        "recursive delete that disables the root-filesystem safety check",
    ),
    (
        "root-delete",
        re.compile(r"""\brm\s+(?:-[A-Za-z-]+\s+)*/(?:\.|\s|$|\*|&|\||;|\)|'|\"|/)"""),
        "deleting the filesystem root",
    ),
    (
        "system-path-delete",
        re.compile(r"\brm\s+(?:-[A-Za-z-]+\s+)*/(?:" + _SYSTEM_DIR_ALT + r")" + _TRAILING),
        "deleting a system directory",
    ),
    (
        "format-filesystem",
        re.compile(r"\b(?:mkfs(?:\.\w+)?|wipefs)\b"),
        "formatting or wiping a filesystem",
    ),
    (
        "raw-disk-write",
        re.compile(r"\bdd\b[^\n]*\bof=/(?:dev/|(?:" + _SYSTEM_DIR_ALT + r")/)"),
        "writing raw data onto a block device or system file",
    ),
    (
        "block-device-redirect",
        re.compile(r">\s*/dev/(?:sd|nvme|vd|hd|mmcblk|disk|mapper|disk\d)\w*"),
        "redirecting output onto a block device",
    ),
    (
        "kernel-interface-write",
        re.compile(r">\s*/(?:proc|sys)/"),
        "writing into procfs or sysfs, which can reboot or panic the host",
    ),
    (
        "partition-table-edit",
        re.compile(r"\b(?:fdisk|parted|sgdisk|gdisk|cfdisk|sfdisk)\b"),
        "editing a partition table",
    ),
    (
        "volume-metadata-destroy",
        re.compile(r"\b(?:blkdiscard|mdadm|pvcreate|vgcreate|lvremove|vgremove|pvremove|losetup)\b"),
        "destroying volume, RAID, or LVM metadata",
    ),
    (
        "runlevel-change",
        re.compile(r"\b(?:init|telinit)\s+[0-6]\b"),
        "changing the system runlevel",
    ),
    (
        "fork-bomb-colon",
        re.compile(r":\s*\(\s*\)\s*\{[^}]*\|[^}]*&[^}]*\}\s*;?\s*:"),
        "a shell fork bomb",
    ),
    (
        "fork-bomb-named",
        re.compile(r"(\w+)\s*\(\s*\)\s*\{[^}]*\1[^}]*\|[^}]*\1[^}]*&[^}]*\}\s*;"),
        "a shell fork bomb using a named function",
    ),
    (
        "kill-all-processes",
        re.compile(r"\b(?:kill|pkill|killall|killall5)\s+(?:-[A-Za-z0-9]+\s+)*-1\b"),
        "signalling every process on the host",
    ),
    (
        "overwrite-system-config",
        re.compile(r">\s*/(?:etc|boot|usr|bin|sbin|lib|lib64|var|opt|root|srv|run)/"),
        "overwriting a system file",
    ),
    (
        "ssh-key-plant",
        re.compile(r"(?:>{1,2}|\btee\b(?:\s+-\S+)*)\s*[^\n]*authorized_keys"),
        "planting an SSH key, which grants persistent access to the host",
    ),
    (
        "sudoers-write",
        re.compile(r"(?:/etc/sudoers|/etc/sudoers\.d|\bvisudo\b)"),
        "modifying sudo privileges",
    ),
    (
        "pipe-download-to-shell",
        re.compile(
            r"\b(?:curl|wget)\b[^;]*\|\s*(?:sudo\s+)?(?:ba|z|d|k)?sh\b"
            r"|\b(?:curl|wget)\b[^;]*\|\s*(?:sudo\s+)?(?:python[23.]*|perl|ruby|node|php)\b"
        ),
        "piping a downloaded script directly into an interpreter",
    ),
    (
        "decode-to-shell",
        re.compile(
            r"\b(?:base64|xxd|openssl)\b[^;]*(?:-d|--decode|-D)[^;]*\|\s*(?:sudo\s+)?(?:ba|z|d|k)?sh\b"
        ),
        "decoding a payload and piping it into a shell",
    ),
    (
        "service-control",
        re.compile(
            r"\bsystemctl\s+(?:--[^\s]+\s+)*(?:stop|restart|disable|mask|kill)\s+"
            r"(?:mira|ssh|sshd|libvirtd|docker|containerd|network|networking|dbus|systemd"
            r"|cron|crond|rsyslog|vault|postgresql|valkey|redis|polkit|getty)"
            r"|\bservice\s+\S+\s+(?:stop|restart|force-stop|force-reload)\b"
            r"|\binvoke-rc\.d\s+\S+\s+(?:stop|restart)\b"
        ),
        "stopping or disabling a host-critical service",
    ),
    (
        "package-removal",
        re.compile(
            r"\b(?:apt|apt-get|dnf|yum|zypper)\s+(?:-\S+\s+)*(?:remove|purge|autoremove|erase)\b"
            r"|\bdpkg\s+(?:-\S+\s+)*-r\b"
            r"|\brpm\s+(?:-\S+\s+)*-e\b"
            r"|\bsnap\s+(?:\S+\s+)*remove\b"
            r"|\bpacman\s+(?:-\S+\s+)*-R\b"
            r"|\bapk\s+(?:\S+\s+)*del\b"
            r"|\bflatpak\s+(?:\S+\s+)*uninstall\b"
        ),
        "removing system packages",
    ),
    (
        "firewall-flush",
        re.compile(
            r"\b(?:iptables|ip6tables)\s+(?:-\S+\s+)*-(?:F|X|P)\b"
            r"|\bnft\s+flush\s+ruleset\b"
            r"|\bip\s+route\s+flush\b"
        ),
        "flushing firewall rules or routes",
    ),
    (
        "network-down",
        re.compile(r"\bifdown\b|\bip\s+link\s+set\s+\S+\s+down\b|\bifconfig\s+\S+\s+down\b"),
        "bringing a network interface down",
    ),
    (
        "log-destruction",
        re.compile(r"\b(?:truncate|shred)\b[^\n]*/var/log/"),
        "destroying host logs, which removes the audit trail",
    ),
    # Git work-destruction. These are deliberately broader than the damage they
    # name: the user's standing rule is that `git checkout`/`git restore` are
    # never used to revert, so the whole verb is refused rather than only its
    # discarding forms. Do not narrow them without that direction.
    (
        "git-checkout-revert",
        re.compile(r"\bgit\b[^\n;|&]*?\bcheckout\b"),
        "git checkout discards uncommitted work with no recovery path; the user's standing rule is "
        "never to use it to revert a change (recover into a new path or make a new commit instead)",
    ),
    (
        "git-restore-revert",
        re.compile(r"\bgit\b[^\n;|&]*?\brestore\b"),
        "git restore discards uncommitted work with no recovery path; same standing rule as checkout",
    ),
    (
        "git-reset-hard",
        re.compile(r"\bgit\b[^\n;|&]*?\breset\s+(?:--hard|--merge|--keep)\b"),
        "git reset discarding the working tree loses uncommitted work irreversibly",
    ),
    (
        "git-clean-force",
        re.compile(r"\bgit\b[^\n;|&]*?\bclean\b[^\n;|&]*?\s-[A-Za-z]*f"),
        "git clean -f deletes untracked files irreversibly",
    ),
]

# Pattern rules enforced in every mode, including skip_permissions: the
# system-ruining, unrecoverable, or audit-erasing core. Every other rule in
# _DESTRUCTIVE_PATTERNS names a recoverable judgment call the operator may
# relax per call via skip_permissions. Note `losetup` sits inside
# volume-metadata-destroy and therefore stays refused in skip mode too.
_CATASTROPHIC_PATTERNS = frozenset({
    "no-preserve-root", "root-delete", "format-filesystem", "raw-disk-write",
    "block-device-redirect", "kernel-interface-write", "partition-table-edit",
    "volume-metadata-destroy", "fork-bomb-colon", "fork-bomb-named",
    "kill-all-processes", "ssh-key-plant", "sudoers-write",
    "pipe-download-to-shell", "decode-to-shell", "log-destruction",
})
# A misspelled name here would silently drop that rule from skip mode, so the
# subset relation is checked at import rather than trusted.
_UNKNOWN_CATASTROPHIC = _CATASTROPHIC_PATTERNS - {name for name, _, _ in _DESTRUCTIVE_PATTERNS}
if _UNKNOWN_CATASTROPHIC:
    raise RuntimeError(f"_CATASTROPHIC_PATTERNS names unknown rules: {sorted(_UNKNOWN_CATASTROPHIC)}")


# -- argument layer -----------------------------------------------------------
#
# Verbs whose arguments are resolved and checked against the protected-path
# classifier below.

_DELETION_VERBS = frozenset({"rm", "rmdir", "shred", "unlink", "srm"})
_RELOCATE_VERBS = frozenset({"mv"})
_PERMISSION_VERBS = frozenset({"chmod", "chown", "chgrp", "setfacl"})
_TRUNCATE_VERBS = frozenset({"truncate", "tee"})
_WRITE_VERBS = frozenset({"cp", "install", "ln"})
_INPLACE_EDIT_VERBS = frozenset({"sed", "perl", "gawk", "awk"})
# Copy verbs outside the classifier's reach: their operands are checked only
# for persistence planting and audit-log tampering (_check_planting).
_COPY_VERBS = frozenset({"rsync", "scp"})

# Never legitimate from this tool in default mode, in any argument position.
_PROHIBITED_VERBS = frozenset({
    "mkfs", "wipefs", "fdisk", "parted", "sgdisk", "gdisk", "cfdisk", "sfdisk",
    "blkdiscard", "mdadm", "pvcreate", "vgcreate", "lvremove", "vgremove",
    "pvremove", "killall5", "visudo",
    "shutdown", "reboot", "poweroff", "halt", "telinit", "swapoff", "ifdown",
    "userdel", "deluser", "groupdel", "chpasswd", "passwd", "chattr",
})
# The subset with no recovery path, refused in every mode. `visudo` belongs
# here: sudo changes survive the skip-mode check-in as persistent privilege.
# The remainder (power, account, password, attribute verbs) is what a
# competent operator with a backup habit recovers from, relaxed under skip.
_CATASTROPHIC_VERBS = frozenset({
    "mkfs", "wipefs", "fdisk", "parted", "sgdisk", "gdisk", "cfdisk", "sfdisk",
    "blkdiscard", "mdadm", "pvcreate", "vgcreate", "lvremove", "vgremove",
    "pvremove", "killall5", "visudo",
})
if not _CATASTROPHIC_VERBS <= _PROHIBITED_VERBS:
    raise RuntimeError(
        f"_CATASTROPHIC_VERBS names verbs outside _PROHIBITED_VERBS: "
        f"{sorted(_CATASTROPHIC_VERBS - _PROHIBITED_VERBS)}"
    )

# Command runners that stand in front of the real verb.
_WRAPPERS = frozenset({
    "sudo", "doas", "pkexec", "nohup", "env", "time", "nice", "ionice",
    "setsid", "command", "builtin", "stdbuf", "timeout", "script", "watch",
    "xargs", "eval", "exec", "ssh",
})
_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ksh", "ash", "csh", "tcsh"})

_SYSTEMCTL_POWER = frozenset({
    "poweroff", "reboot", "halt", "suspend", "hibernate", "hybrid-sleep",
    "suspend-then-hibernate", "exit", "isolate", "default", "emergency",
    "rescue", "kexec",
})
_SYSTEMCTL_STOP = frozenset({"stop", "restart", "kill", "disable", "mask"})
_CRITICAL_UNIT = re.compile(
    r"^(?:mira|ssh|sshd|libvirtd|docker|containerd|network|networking"
    r"|NetworkManager|systemd|dbus|cron|crond|rsyslog|vault|postgresql"
    r"|valkey|redis|polkit|getty|serial-getty|user)(?:[.@]|\d|$)"
)
_FIND_NARROWING = frozenset({
    "-name", "-iname", "-path", "-ipath", "-regex", "-iregex", "-type",
    "-mtime", "-atime", "-ctime", "-newer", "-size", "-empty", "-user",
    "-group", "-perm", "-mmin", "-amin", "-cmin", "-wholename",
})
_MATCH_ALL_REGEXES = frozenset({".", ".*", ".+", "^", "$", "^.*$", "^.*", ".*$", "[^]", "^^"})

# Pattern rules that anchor on an absolute path. For these, the match text
# consumes only the system directory ("rm -rf /opt/"), and the path the match
# points at is whatever continues at the match's anchored slash — see
# _match_in_editable_app_tree, which exempts matches pointing inside MIRA's
# own code tree; the argument layer still classifies every operand of the
# same command precisely.
_APP_TREE_PATH_RULES = frozenset({"system-path-delete", "overwrite-system-config"})

# Newlines never reach the tokenizer unquoted: _split_lines splits on them
# first, so each line is its own command chain.
_OPERATORS = frozenset({";", ";;", "&", "&&", "|", "||", "|&", "(", ")", "{", "}"})
_WRITE_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>", "<>"})
_SED_INPLACE = re.compile(r"^-[nrsuzE]*i")
# Perl switches that consume the rest of their cluster as a value.
_PERL_VALUE_SWITCHES = "MmIxFdDCV"
# Marks a working directory this guardrail cannot know (skip mode only: a `cd`
# with no target or into an expansion or glob, any `pushd`/`popd`). Relative
# operands against it are refused.
_UNKNOWN_CWD = ""
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_DURATION = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")
_INTEGER = re.compile(r"^[-+]?\d+$")
# User-home roots on both supported layouts: /home/<name> (Linux) and
# /Users/<name> (macOS). The classifier must know both shapes — protecting
# only the Linux one left cross-user home deletion unguarded on macOS.
_HOME_DIR = re.compile(r"^(?:/home|/Users)/[^/]+$")
_NESTING_CHARS = " \t\n;|&<>()$`"
_EXPANSION_CHARS = "$`~"
_GLOB_CHARS = "*?["
_MAX_NESTING = 6

# Protected as themselves; their contents stay reachable, so `rm -rf /tmp/scratch`
# is fine while `rm -rf /tmp` is not.
_SELF_ONLY_DIRS = frozenset({
    "/tmp", "/mnt", "/media", "/snap", "/lost+found",
    # Home containers: deleting the container itself is destructive, while its
    # contents stay reachable so per-user homes classify via _HOME_DIR below.
    "/home", "/Users",
})

# Skip mode relaxes system-directory contents, except these. On macOS `/etc`,
# `/var` and `/tmp` are symlinks into `/private`, so the `/private` spellings
# are the same containers. The log trees are the host's audit trail.
_SKIP_ALIAS_CONTAINERS = frozenset({"/private/etc", "/private/var", "/private/tmp"})
_LOG_TREES = ("/var/log", "/private/var/log")

# Persistence targets, refused in every mode (_planting_rule). Sudo config is
# any `/etc/sudo*` entry; the names here are what a glob is tested against.
_SUDO_CONFIG_NAMES = ("sudoers", "sudoers.d", "sudo.conf", "sudo_logsrvd.conf")
_SSH_KEY_NAMES = ("authorized_keys", "authorized_keys2")
_RULE_REASONS: Dict[str, str] = {
    "filesystem-root": "the command targets the filesystem root",
    "project-root-delete": (
        "the command targets the configured project root or one of its parent "
        "directories, which destroys the harness this tool exists to operate on"
    ),
    "system-path-delete": "the command targets a system directory or its contents",
    "system-dir-delete": "the command targets a host-shared directory",
    "user-home-delete": "the command targets a user's home directory",
    "vcs-history-delete": (
        "the command targets the project's git history, which cannot be recreated"
    ),
    "app-root-delete": (
        "the command targets MIRA's code tree as a whole; edit or delete the "
        "files inside it instead"
    ),
    "app-untracked-delete": (
        "the command targets a part of MIRA's code tree that git does not track "
        "(user data, the virtualenv, credentials, logs, or the git history "
        "itself), so nothing can restore it"
    ),
    "unverifiable-expansion": (
        "the path contains a shell expansion ($, `, or ~) that this guardrail "
        "cannot resolve before the host shell does, so it cannot prove the target "
        "is safe. Use an explicit literal path instead"
    ),
    "protected-glob": (
        "the glob can match every entry in a protected directory, which would "
        "empty it. Name the entries you mean, or glob inside a subdirectory"
    ),
    "unglobbable-parent": (
        "the glob expands inside a protected directory, so which entries it "
        "matches cannot be verified here"
    ),
    "unverifiable-cwd": (
        "an earlier `cd`, `pushd`, `popd`, or `env -C` moved into a directory "
        "this guardrail cannot resolve, so a relative path after it cannot be "
        "proven safe. Use an explicit literal absolute path instead"
    ),
    "unparseable-quoting": (
        "the command's quoting is unbalanced, so its structure cannot be "
        "verified. Balance the quotes"
    ),
    "log-destruction": "the command targets the host's log tree, which is the audit trail",
    "sudoers-write": "the command targets sudo's configuration, which grants persistent privilege",
    "ssh-key-plant": (
        "the command writes into an SSH directory or authorized_keys file, "
        "which grants persistent access to the host"
    ),
    "audit-log-tamper": (
        "the command targets guardrail_bypass.log (or a directory holding it), "
        "the operator's record of every skip_permissions call"
    ),
    "guardrail-self-edit": (
        "the command targets this guardrail's own source or MIRA's config "
        "directory, which define what this tool may do; changes there are the "
        "operator's to make"
    ),
}


def _normalize_absolute(path: str) -> str:
    """
    Normalize an absolute path, collapsing the POSIX `//` special case.

    `posixpath.normpath` deliberately preserves exactly two leading slashes, so
    `//` and `//etc` would otherwise slip past every comparison against `/`.
    """
    resolved = posixpath.normpath(path)
    if resolved.startswith("//") and not resolved.startswith("///"):
        resolved = resolved[1:]
    return resolved


def _protected_ancestors(root: str) -> FrozenSet[str]:
    """The project root and every directory above it, `/` included."""
    protected = set()
    node = _normalize_absolute(root)
    while True:
        protected.add(node)
        parent = posixpath.dirname(node)
        if parent == node:
            break
        node = parent
    return frozenset(protected)


def _classify_path(
    path: str,
    root: str,
    ancestors: FrozenSet[str],
    skip: bool = False,
) -> Optional[str]:
    """
    Name the protection rule a resolved absolute path violates.

    Returns None when the path is not protected. Deleting and modifying paths
    *inside* the project root stays legal; that is the tool's contract.

    Skip mode narrows system-path protection to the protected containers
    themselves: `/etc/nginx` is content an operator restores from a backup,
    while `/etc` is the container whose loss takes the host with it. The
    `/private/etc`, `/private/var` and `/private/tmp` spellings of the macOS
    containers count as containers too. Three trees keep their contents
    protected even under skip mode — `/dev` (block devices have no recovery
    path), `/Volumes` (a mounted volume is usually the backup itself), and the
    log trees (the audit trail). Everything else inside a system directory is
    skip-eligible recoverable content; the operator's blocklist is the
    tripwire for anything instance-specific.

    The guardrail's own source file (and the directories holding it) and
    MIRA's config directory are refused in every mode, ahead of the app-tree
    exemption that makes the rest of the code tree editable.
    """
    resolved = _normalize_absolute(path)
    if resolved == "/":
        return "filesystem-root"
    if resolved in ancestors:
        return "project-root-delete"
    if resolved in _GUARDRAIL_PATHS or any(
        _at_or_under(resolved, config_dir) for config_dir in _APP_CONFIG_DIRS
    ):
        return "guardrail-self-edit"
    app_rule = _classify_app_tree_path(resolved)
    if app_rule is not None:
        return app_rule or None
    head = resolved[1:].partition("/")[0]
    if head in _SYSTEM_DIR_NAMES:
        if skip:
            if resolved in _SKIP_ALIAS_CONTAINERS:
                return "system-path-delete"
            if any(_at_or_under(resolved, tree) for tree in _LOG_TREES):
                return "log-destruction"
            if resolved != "/" + head and head not in ("dev", "Volumes"):
                return None
        return "system-path-delete"
    if resolved in _SELF_ONLY_DIRS:
        return "system-dir-delete"
    if _HOME_DIR.match(resolved):
        return "user-home-delete"
    git_dir = posixpath.join(root, ".git")
    if resolved == git_dir or resolved.startswith(git_dir + "/"):
        return "vcs-history-delete"
    return None


# MIRA's own code tree: the root this package was loaded from. An install
# records the tree as a git repository (deploy/python.sh: one commit per
# install), and that history is the undo for anything MIRA changes in it —
# `git diff` shows the change, `git stash` sets it aside, one command reverts
# it. A tree with that undo is ordinary editable material, not a system path.
_APP_ROOT = Path(__file__).resolve().parents[2]


def _load_app_tree() -> Optional[Tuple[str, FrozenSet[str]]]:
    """
    (code tree root, top-level names git does not track), or None when the tree
    is not a git repository.

    Read once at import, outside the validator: the validator stays a pure
    decision function over this inert tuple
    (tests/protected/bash_guardrail_probe.py enforces that). The untracked names
    come from the installed `.git/info/exclude` — the generating source, never a
    transcribed list, so a code edit cannot widen what is deletable. Its
    state-carrying entries are slash-anchored (deploy/python.sh), so git ignores
    them at top level only, matching this top-level reading; an unanchored entry
    would also hide a nested same-named directory from git while leaving it
    deletable here. No exclude file means no git and no undo, so the tree gets
    no exemption and stays a system path.
    """
    exclude = _APP_ROOT / ".git" / "info" / "exclude"
    if not exclude.is_file():
        return None
    names = {".git"}
    for raw in exclude.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or any(ch in line for ch in "*?[!"):
            continue
        names.add(line.strip("/").split("/")[0])
    return str(_APP_ROOT), frozenset(names)


# Inert from import onwards: (app root, untracked top-level names) or None.
_APP_TREE: Optional[Tuple[str, FrozenSet[str]]] = _load_app_tree()


def _load_guardrail_paths() -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """
    (this file plus the directories between it and the code-tree root,
    MIRA's config directory) under both the symlink-resolved and the as-loaded
    spelling of the tree.

    These define what the tool may do — the rules, the skip gate's default, the
    operator's blocklist — so the app-tree exemption never covers them. The
    holding directories are included because moving `tools/` aside and putting
    a replacement in its place swaps the guardrail without touching this file's
    path. Read once at import, like _APP_TREE, so the validator sees inert
    strings.
    """
    guarded = set()
    config_dirs = set()
    for spelling in {str(Path(__file__).resolve()), os.path.abspath(__file__)}:
        implementations_dir = posixpath.dirname(spelling)
        tools_dir = posixpath.dirname(implementations_dir)
        guarded.update({spelling, implementations_dir, tools_dir})
        config_dirs.add(posixpath.join(posixpath.dirname(tools_dir), "config"))
    return frozenset(guarded), frozenset(config_dirs)


_GUARDRAIL_PATHS, _APP_CONFIG_DIRS = _load_guardrail_paths()


def _classify_app_tree_path(resolved: str) -> Optional[str]:
    """
    Classify a path against MIRA's own code tree.

    Returns None when the path is outside the tree (or the tree is not a git
    repository), "" when it is an editable file or directory inside the tree, or
    the rule name it violates. Tracked contents are editable because git is the
    undo for them. What git does not track — the names in `.git/info/exclude`,
    `.git` included — has no undo and stays protected, as does the tree root
    itself. Without a git tree there is no exemption at all: under `/opt` it
    stays a system path.
    """
    if _APP_TREE is None:
        return None
    app_root, protected = _APP_TREE
    if resolved == app_root:
        return "app-root-delete"
    if not resolved.startswith(app_root + "/"):
        return None
    if resolved[len(app_root) + 1:].partition("/")[0] in protected:
        return "app-untracked-delete"
    return ""


def _match_in_editable_app_tree(match: "re.Match[str]") -> bool:
    """True when the path a path-anchored pattern match points at lies inside
    MIRA's own code tree.

    Both patterns in _APP_TREE_PATH_RULES begin their path at the first slash of
    the match text ("rm -rf /opt/", "> /opt/"): the pattern consumes only the
    system directory, never the full path, so the path this match points at is
    whatever continues at that offset in the command. An anchored prefix test
    there is exact — it cannot exempt the tree root (refused as app-root-delete)
    nor a longer path merely sharing the root as a prefix — and needs no path
    extraction: no token regex, no normalization, no re-classification. The
    argument layer still resolves every operand of the same command (relative
    paths, quotes, globs), so the pattern layer only needs this coarse signal.
    """
    if _APP_TREE is None:
        return False
    anchor = match.start() + match.group(0).index("/")
    return match.string.startswith(_APP_TREE[0] + "/", anchor)


def _at_or_under(path: str, root: str) -> bool:
    """True when path is the project root or somewhere inside it."""
    return path == root or path.startswith(root + "/")


def _resolve_operand(operand: str, cwd: str) -> str:
    """Resolve one command operand to an absolute path against the effective cwd.

    A relative operand against _UNKNOWN_CWD is refused: there is no directory
    to resolve it against.
    """
    if posixpath.isabs(operand):
        return _normalize_absolute(operand)
    if cwd == _UNKNOWN_CWD:
        _refuse("unverifiable-cwd", _RULE_REASONS["unverifiable-cwd"], operand)
    return _normalize_absolute(posixpath.join(cwd, operand))


def _matches_everything(basename: str) -> bool:
    """True when a glob basename can match every entry in its directory."""
    collapsed = re.sub(r"\[[^\]]*\]", "?", basename)
    return bool(collapsed) and all(char in "*?." for char in collapsed)


def _glob_matches(component: str, name: str) -> bool:
    """
    True when one path component — literal or shell glob — can name `name`.

    Bash semantics without dotglob: a glob reaches a dot-name only when the
    component itself starts with a literal dot. A bracket expression this
    translation cannot compile counts as a match, so an unreadable glob is
    treated as dangerous rather than safe.
    """
    if not any(char in component for char in _GLOB_CHARS):
        return component == name
    if name.startswith(".") and not component.startswith("."):
        return False
    pattern = ""
    index = 0
    while index < len(component):
        char = component[index]
        if char == "*":
            pattern += ".*"
        elif char == "?":
            pattern += "."
        elif char == "[":
            start = index + 1
            if start < len(component) and component[start] in "!^":
                start += 1
            # A `]` right after `[` or `[!` is a literal member, not the close.
            close = component.find("]", start + 1)
            if close == -1:
                pattern += re.escape(char)
            else:
                members = component[index + 1:close]
                if members[:1] in ("!", "^"):
                    members = "^" + members[1:]
                pattern += "[" + members.replace("\\", "\\\\") + "]"
                index = close
        else:
            pattern += re.escape(char)
        index += 1
    try:
        return re.fullmatch(pattern, name) is not None
    except re.error:
        return True


def _glob_overlaps(path: str, target: str) -> bool:
    """
    True when a resolved, possibly globbed path can name `target`, a directory
    holding it, or something inside it — compared component by component, so
    `.mira_logs/*` and `.m*` both reach `.mira_logs/guardrail_bypass.log`.
    """
    parts = [part for part in path.split("/") if part]
    target_parts = [part for part in target.split("/") if part]
    return all(_glob_matches(part, name) for part, name in zip(parts, target_parts))


def _glob_split(resolved: str, skip: bool) -> Tuple[str, str, bool]:
    """
    (directory the glob expands in, the globbed component, whether that
    component is the last one) for a resolved path containing a glob.

    Default mode looks only at the final component. Skip mode, whose path
    classifier relaxes container contents, takes the FIRST globbed component:
    `/*/nginx` expands inside `/`, where every match is a protected container.
    """
    if not skip:
        return posixpath.dirname(resolved), posixpath.basename(resolved), True
    parts = resolved.split("/")
    for index, part in enumerate(parts):
        if any(char in part for char in _GLOB_CHARS):
            return "/".join(parts[:index]) or "/", part, index == len(parts) - 1
    return posixpath.dirname(resolved), posixpath.basename(resolved), True


def _planting_rule(resolved: str, ssh_keys: bool) -> Optional[str]:
    """
    Name the persistence rule a resolved (possibly globbed) path violates.

    Sudo configuration — any `/etc/sudo*` entry, `/private/etc` included — is
    refused for every checked verb, matching the sudoers-write pattern that
    refuses any literal mention. With ssh_keys, an SSH directory itself or an
    `authorized_keys*` file inside one is refused: callers set it for verbs
    that write (copy, link, move, truncate, in-place edit, dd, redirect), not
    for deletes or permission changes, which plant nothing.
    """
    parts = [part for part in resolved.split("/") if part]
    if parts[:1] == ["private"]:
        parts = parts[1:]
    if len(parts) >= 2 and _glob_matches(parts[0], "etc") and (
        parts[1].startswith("sudo")
        or any(_glob_matches(parts[1], name) for name in _SUDO_CONFIG_NAMES)
    ):
        return "sudoers-write"
    if not ssh_keys:
        return None
    for index, part in enumerate(parts):
        if not _glob_matches(part, ".ssh"):
            continue
        if index == len(parts) - 1:
            return "ssh-key-plant"
        child = parts[index + 1]
        if child.startswith("authorized_keys") or any(
            _glob_matches(child, name) for name in _SSH_KEY_NAMES
        ):
            return "ssh-key-plant"
    return None


def _check_planting(
    resolved: str,
    offending: str,
    root: str,
    audit_log: Optional[str],
    ssh_keys: bool,
) -> None:
    """Refuse a resolved operand that plants persistence or reaches the audit log.

    Enforced in every mode. The audit log is reached by naming it, a directory
    holding it below the project root, or a glob that can expand to either; the
    root and its ancestors belong to the path classifier, which refuses them
    for every destructive verb.
    """
    rule = _planting_rule(resolved, ssh_keys)
    if rule:
        _refuse(rule, _RULE_REASONS[rule], offending)
    if audit_log and not _at_or_under(root, resolved) and _glob_overlaps(resolved, audit_log):
        _refuse("audit-log-tamper", _RULE_REASONS["audit-log-tamper"], offending)


def _verb_name(token: str) -> str:
    """Basename of a command word, with the alias-defeating backslash removed."""
    return posixpath.basename(token.lstrip("\\"))


def _path_operands(args: Iterable[str]) -> List[str]:
    """Command arguments that are paths rather than flags.

    An empty string is never a path: shlex keeps BSD `sed -i ''`'s empty
    suffix as a token, and resolving it yields the cwd — a false positive
    against the project root — so empty operands are dropped here for every
    verb. An empty token is a flag suffix, not a path.
    """
    return [arg for arg in args if arg and not arg.startswith("-")]


def _check_operand(
    operand: str,
    verb: str,
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    skip: bool = False,
    audit_log: Optional[str] = None,
    ssh_keys: bool = False,
) -> None:
    """
    Refuse a destructive verb's path operand if it cannot be proven safe.

    Resolves relative operands against the effective cwd, refuses operands whose
    meaning depends on shell expansion (in every mode: `~`, `$HOME`, and
    `${X:-/etc}` all resolve where the host shell decides), classifies the
    resolved path, refuses persistence planting and audit-log tampering
    (_check_planting; ssh_keys for verbs that write), and refuses globs that
    could empty a protected directory or expand inside one. Skip mode narrows
    only the classification to containers (see _classify_path) and locates the
    glob at its first globbed component (see _glob_split).
    """
    offending = f"{verb} {operand}"
    if any(char in operand for char in _EXPANSION_CHARS):
        _refuse("unverifiable-expansion", _RULE_REASONS["unverifiable-expansion"], offending)

    resolved = _resolve_operand(operand, cwd)
    hit = _classify_path(resolved, root, ancestors, skip)
    if hit:
        _refuse(hit, _RULE_REASONS[hit], offending)

    if any(char in operand for char in _GLOB_CHARS):
        parent, component, final = _glob_split(resolved, skip)
        if _classify_path(parent, root, ancestors, skip):
            if final and _matches_everything(component):
                _refuse("protected-glob", _RULE_REASONS["protected-glob"], offending)
            if not _at_or_under(parent, root):
                _refuse("unglobbable-parent", _RULE_REASONS["unglobbable-parent"], offending)

    _check_planting(resolved, offending, root, audit_log, ssh_keys)


def _check_redirect_targets(
    tokens: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    skip: bool = False,
    audit_log: Optional[str] = None,
) -> None:
    """Refuse output redirections aimed at a protected path.

    Skip mode also refuses a target that depends on shell expansion, whose
    container-only classification cannot be applied to a path it cannot see.
    """
    for index, token in enumerate(tokens):
        if token not in _WRITE_REDIRECTS:
            continue
        if index + 1 >= len(tokens):
            continue
        target = tokens[index + 1]
        offending = f"{token} {target}"
        # `2>&1` and friends name a descriptor, not a path.
        if target.startswith("&") or _INTEGER.match(target):
            continue
        if skip and any(char in target for char in _EXPANSION_CHARS):
            _refuse("unverifiable-expansion", _RULE_REASONS["unverifiable-expansion"], offending)
        resolved = _resolve_operand(target, cwd)
        # /dev/null is the bit bucket: discarding output there is benign no
        # matter what protected path classification would otherwise say.
        if resolved == "/dev/null":
            continue
        hit = _classify_path(resolved, root, ancestors, skip)
        if hit:
            _refuse(hit, _RULE_REASONS[hit], offending)
        _check_planting(resolved, offending, root, audit_log, ssh_keys=True)


def _check_prohibited_verb(verb: str, skip: bool = False) -> None:
    """Refuse verbs that have no legitimate use from this tool.

    Skip mode keeps only the catastrophic subset — verbs that ruin the system
    with no recovery path. The recoverable remainder (power state, account and
    password changes) becomes a skip-mode judgment call.
    """
    forbidden = _CATASTROPHIC_VERBS if skip else _PROHIBITED_VERBS
    if verb in forbidden or verb.startswith("mkfs."):
        _refuse(
            "prohibited-command",
            "this command has no legitimate use against the remote host and "
            "damages the host or its data",
            verb,
        )


def _check_runlevel(args: List[str], skip: bool = False) -> None:
    """Refuse `init N`, which changes the system runlevel.

    Relaxed in skip mode alongside the runlevel-change pattern: a target or
    power change is recoverable, exactly the class a check-in legitimizes.
    """
    if skip:
        return
    operands = _path_operands(args)
    if operands and operands[0] in {"0", "1", "2", "3", "4", "5", "6"}:
        _refuse("runlevel-change", "changing the system runlevel", f"init {operands[0]}")


def _check_service_control(verb: str, args: List[str], skip: bool = False) -> None:
    """Refuse systemd and SysV operations that power off or disable the host.

    Relaxed in skip mode: delegating a service restart to the instance after
    a check-in — including restarting MIRA itself — is one of the workflow
    wins the operator enables the switch for.
    """
    if skip:
        return
    operands = _path_operands(args)
    if verb == "systemctl":
        if not operands:
            return
        subcommand, units = operands[0], operands[1:]
        if subcommand in _SYSTEMCTL_POWER:
            _refuse(
                "system-power",
                "changing the host's power state or runlevel target",
                f"systemctl {subcommand}",
            )
        if subcommand in _SYSTEMCTL_STOP and any(_CRITICAL_UNIT.match(unit) for unit in units):
            _refuse(
                "service-control",
                "stopping or disabling a host-critical service",
                " ".join([verb] + list(args)),
            )
    elif verb in ("service", "invoke-rc.d"):
        if len(operands) >= 2 and operands[1] in {"stop", "restart", "force-stop", "force-reload"}:
            if _CRITICAL_UNIT.match(operands[0]):
                _refuse(
                    "service-control",
                    "stopping or disabling a host-critical service",
                    " ".join([verb] + list(args)),
                )


def _check_pkill(args: List[str]) -> None:
    """Refuse a pkill/pgrep pattern that matches every process."""
    if "-f" not in args:
        return
    operands = _path_operands(args)
    for operand in operands:
        if operand.strip("'\"") in _MATCH_ALL_REGEXES:
            _refuse(
                "kill-all-processes",
                "a match-everything pattern signals every process on the host",
                f"pkill -f {operand}",
            )


def _check_crontab(args: List[str]) -> None:
    """Refuse `crontab -r`, which wipes every scheduled job for the user.

    Enforced in both modes (operator decision, 2026-10-03): scheduled jobs are
    standing automation the human owns, and wiping them has no recovery path
    short of the operator's memory of what was scheduled.
    """
    if any(arg in ("-r", "--remove") for arg in args):
        _refuse("cron-wipe", "removing every scheduled job for the user", "crontab -r")


def _check_mount(
    verb: str,
    args: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    skip: bool = False,
) -> None:
    """Refuse unmounting broadly or remounting a protected filesystem.

    Enforced in both modes (operator decision, 2026-10-03): `umount -a` takes
    every filesystem offline in one move.
    """
    if any(arg in ("-a", "--all", "-f", "--force") for arg in args):
        _refuse("mount-change", f"{verb} applied to every filesystem", " ".join([verb] + list(args)))
    for operand in _path_operands(args):
        if any(char in operand for char in _EXPANSION_CHARS):
            continue
        hit = _classify_path(_resolve_operand(operand, cwd), root, ancestors, skip)
        if hit:
            _refuse(hit, _RULE_REASONS[hit], f"{verb} {operand}")


def _check_dd(
    args: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    skip: bool = False,
    audit_log: Optional[str] = None,
) -> None:
    """Refuse a dd write target that is a device or a protected path."""
    for arg in args:
        if not arg.startswith("of="):
            continue
        target = arg[len("of="):]
        if any(char in target for char in _EXPANSION_CHARS):
            _refuse("unverifiable-expansion", _RULE_REASONS["unverifiable-expansion"], arg)
        resolved = _resolve_operand(target, cwd)
        hit = _classify_path(resolved, root, ancestors, skip)
        if hit:
            _refuse(hit, _RULE_REASONS[hit], arg)
        _check_planting(resolved, arg, root, audit_log, ssh_keys=True)


def _find_exec_deletes(args: List[str]) -> bool:
    """True when find runs a deletion verb through -exec/-execdir/-ok/-okdir."""
    for index, arg in enumerate(args):
        if arg not in ("-exec", "-execdir", "-ok", "-okdir"):
            continue
        for candidate in args[index + 1:index + 4]:
            if _verb_name(candidate) in _DELETION_VERBS:
                return True
    return False


def _find_narrows(args: List[str]) -> bool:
    """
    True when the find expression carries a predicate that provably narrows.

    The deletion carve-out needs a demonstrated constraint, not the mere
    presence of a predicate flag: `-name '*'` matches every entry, and a
    predicate with no value is unparseable. Anything this walk cannot prove
    narrows returns False, keeping the deletion on the guarded-refusal path.
    """
    narrowed = False
    match_all = False
    disjoined = False
    negated = False
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("!", "-not"):
            negated = True
        elif arg == ",":
            # The expression-list operator: `find . -name '*.pyc' , -delete`
            # unions the narrowed match with the un-narrowed walk, so the
            # delete applies to everything. Presence of a comma can never be
            # proven narrowing — refuse the carve-out.
            return False
        elif arg in ("(", ")", "-a", "-and"):
            pass
        elif arg in ("-o", "-or"):
            disjoined = True
        elif arg == "-empty":
            if not negated:
                narrowed = True
            negated = False
        elif arg in _FIND_NARROWING:
            if index + 1 >= len(args):
                return False
            value = args[index + 1]
            if value in _FIND_NARROWING or value in (
                "-delete", "-exec", "-execdir", "-ok", "-okdir", "(", ")", "!", ",",
            ):
                return False
            index += 1
            if arg in ("-name", "-iname", "-path", "-ipath", "-wholename"):
                pattern = value.strip("'\"")
                if "*" in pattern and set(pattern) <= {"*", "?"}:
                    if not negated:
                        match_all = True
                elif not negated:
                    narrowed = True
            elif arg in ("-regex", "-iregex"):
                if value.strip("'\"") in _MATCH_ALL_REGEXES:
                    if not negated:
                        match_all = True
                elif not negated:
                    narrowed = True
            elif not negated:
                narrowed = True
            negated = False
        index += 1
    if match_all and disjoined:
        return False
    return narrowed


def _check_find(
    args: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    skip: bool = False,
    audit_log: Optional[str] = None,
) -> None:
    """
    Refuse a find that deletes without a narrowing predicate over a safe tree.

    `find . -name '*.pyc' -delete` inside the project root is ordinary cleanup.
    `find . -delete` from the same directory removes the whole harness, and
    `find / ...` walks off the project tree entirely. A predicate only earns
    the carve-out when it demonstrably narrows: `-name '*'` matches everything
    and does not count, and neither does an unparseable expression. A walk
    over the audit log's directory additionally needs a `-name`/`-iname` that
    provably excludes the log (_find_spares).
    """
    if "-delete" not in args and not _find_exec_deletes(args):
        return
    starts: List[str] = []
    for arg in args:
        if arg.startswith("-") or arg in ("!", "(", ")", ","):
            break
        starts.append(arg)
    narrowed = _find_narrows(args)
    for start in starts or ["."]:
        offending = f"find {start}"
        if any(char in start for char in _EXPANSION_CHARS):
            _refuse("unverifiable-expansion", _RULE_REASONS["unverifiable-expansion"], offending)
        resolved = _resolve_operand(start, cwd)
        hit = _classify_path(resolved, root, ancestors, skip)
        if hit and not (narrowed and _at_or_under(resolved, root)):
            _refuse(hit, _RULE_REASONS[hit], offending)
        _check_planting(resolved, offending, root, None, ssh_keys=False)
        if audit_log and _glob_overlaps(resolved, audit_log) and not _find_spares(
            args, posixpath.basename(audit_log)
        ):
            _refuse("audit-log-tamper", _RULE_REASONS["audit-log-tamper"], offending)


def _find_spares(args: List[str], name: str) -> bool:
    """
    True when a find expression provably never matches a file named `name`:
    some un-negated `-name`/`-iname` pattern excludes it and no `-o`/`,`
    disjunction can bring it back.
    """
    if any(arg in ("-o", "-or", ",") for arg in args):
        return False
    for index, arg in enumerate(args[:-1]):
        if arg not in ("-name", "-iname") or (index and args[index - 1] in ("!", "-not")):
            continue
        pattern = args[index + 1]
        if arg == "-iname":
            if not _glob_matches(pattern.lower(), name.lower()):
                return True
        elif not _glob_matches(pattern, name):
            return True
    return False


def _check_path_verbs(
    verb: str,
    args: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    skip: bool = False,
    audit_log: Optional[str] = None,
) -> None:
    """Dispatch a path-taking verb to its operand check."""
    if verb in _DELETION_VERBS or verb in _PERMISSION_VERBS:
        for operand in _path_operands(args):
            _check_operand(operand, verb, cwd, root, ancestors, skip, audit_log)
    elif verb in _RELOCATE_VERBS or verb in _TRUNCATE_VERBS or verb in _WRITE_VERBS:
        for operand in _path_operands(args):
            _check_operand(operand, verb, cwd, root, ancestors, skip, audit_log, ssh_keys=True)
    elif verb in _INPLACE_EDIT_VERBS:
        for operand in _inplace_operands(verb, args):
            _check_operand(operand, verb, cwd, root, ancestors, skip, audit_log, ssh_keys=True)
    elif verb in _COPY_VERBS:
        for operand in _path_operands(args):
            offending = f"{verb} {operand}"
            _check_planting(_resolve_operand(operand, cwd), offending, root, audit_log, ssh_keys=True)
    elif verb == "dd":
        _check_dd(args, cwd, root, ancestors, skip, audit_log)
    elif verb == "find":
        _check_find(args, cwd, root, ancestors, skip, audit_log)


def _inplace_operands(verb: str, args: List[str]) -> List[str]:
    """
    The file operands an in-place edit rewrites, or [] when the call does not
    edit in place.

    sed: `-i`, `-i.bak`, `-Ei`-style clusters, or `--in-place[=SUFFIX]`; every
    non-flag argument counts, the script included. perl: a switch cluster
    reaching `i` (`-pi`, `-0pi`, `-i.bak`); `-e`/`-E` values and, without
    them, the program file are code, not targets. gawk/awk: `-i inplace`
    (or `--include=inplace`); the program text and flag values are skipped.
    """
    if verb == "sed":
        if any(arg.startswith("--in-place") or _SED_INPLACE.match(arg) for arg in args):
            return _path_operands(args)
        return []
    in_place = False
    program_given = False
    operands: List[str] = []
    if verb == "perl":
        takes_value = False
        for arg in args:
            if takes_value:
                takes_value = False
                continue
            if arg.startswith("-") and arg != "-" and not arg.startswith("--"):
                cluster = arg[1:]
                for position, letter in enumerate(cluster):
                    if letter == "i":
                        in_place = True
                        break
                    if letter in "eE":
                        program_given = True
                        takes_value = position == len(cluster) - 1
                        break
                    if letter in _PERL_VALUE_SWITCHES:
                        break
                continue
            if not arg or arg.startswith("-"):
                continue
            if not program_given:
                program_given = True
                continue
            operands.append(arg)
        return operands if in_place else []
    # gawk / awk
    value_of = ""
    for arg in args:
        if value_of:
            if value_of == "include" and arg in ("inplace", "inplace.awk"):
                in_place = True
            value_of = ""
            continue
        if arg in ("-i", "--include"):
            value_of = "include"
        elif arg in ("-iinplace", "-iinplace.awk", "--include=inplace", "--include=inplace.awk"):
            in_place = True
        elif arg in ("-f", "--file", "-e", "--source", "-E", "--exec"):
            program_given = True
            value_of = "program"
        elif arg in ("-v", "--assign", "-F", "--field-separator", "-l", "--load"):
            value_of = "value"
        elif arg.startswith(("-f", "-e", "-E", "--file=", "--source=", "--exec=")):
            program_given = True
        elif not arg or arg.startswith("-"):
            continue
        elif not program_given:
            program_given = True
        else:
            operands.append(arg)
    return operands if in_place else []


def _looks_nested(token: str) -> bool:
    """True when a token is itself a command line rather than a single word."""
    return any(char in token for char in _NESTING_CHARS)


def _dollar_bodies(text: str) -> List[str]:
    """Bodies of every `$( ... )` substitution in `text`, nested parens honored."""
    bodies: List[str] = []
    index = 0
    while True:
        start = text.find("$(", index)
        if start == -1:
            return bodies
        depth = 0
        end = -1
        for pos in range(start + 1, len(text)):
            if text[pos] == "(":
                depth += 1
            elif text[pos] == ")":
                depth -= 1
                if depth == 0:
                    end = pos
                    break
        if end == -1:
            # Unterminated inside these tokens. Unquoted `$(` is split by
            # shlex into separate `$` and `(` tokens and the plain segment
            # walk already analyzes the body, so no fallback is needed here.
            return bodies
        bodies.append(text[start + 2 : end])
        index = end + 1


def _backtick_bodies(text: str) -> List[str]:
    """
    Bodies of every backtick substitution in `text`.

    An unterminated trailing backtick is dropped: bash rejects that line
    outright, so nothing inside it executes.
    """
    parts = text.split("`")
    return [parts[i] for i in range(1, len(parts) - 1, 2)]


def _analyze_substitutions(
    tokens: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    depth: int,
    skip: bool = False,
    audit_log: Optional[str] = None,
) -> None:
    """
    Re-run the full argument layer over every command substitution body.

    shlex keeps a quoted substitution inside one token (`echo "$(rm -rf .)"`
    tokenizes to `['echo', '$(rm -rf .)']`), and backticks stay glued to the
    words they wrap (`` echo `rm -rf .` `` → `` ['echo', '`rm', '-rf', '.`'] ``),
    so the plain segment walk never sees the body as a command. Extract each
    body and analyze it as its own command line — benign substitutions
    (`wc -l $(find . -name '*.py')`) keep passing, and unquoted `$( ... )`,
    which the tokenizer splits at the parens, keeps its existing path.
    """
    if depth >= _MAX_NESTING:
        return
    joined = " ".join(tokens)
    if "$(" not in joined and "`" not in joined:
        return
    for body in _dollar_bodies(joined) + _backtick_bodies(joined):
        _analyze_command(body, cwd, root, ancestors, depth + 1, skip, audit_log)


def _strip_wrapper_args(verb: str, rest: List[str]) -> List[str]:
    """Drop the arguments a command runner consumes before the real command."""
    out = list(rest)
    while out and out[0].startswith("-"):
        # Flags that take a separate value.
        if out[0] in ("-n", "-u", "-k", "-s", "-i", "-p", "-K", "-c", "-l") and len(out) > 1:
            out = out[2:]
        else:
            out = out[1:]
    if verb == "env":
        while out and _ENV_ASSIGNMENT.match(out[0]):
            out = out[1:]
    if verb == "timeout" and out and _DURATION.match(out[0]):
        out = out[1:]
        if out and out[0].startswith("-"):
            out = out[1:]
    if verb == "ssh" and out:
        out = out[1:]
    if verb in ("nice", "ionice") and out and _INTEGER.match(out[0]):
        out = out[1:]
    return out


def _analyze_shell(
    rest: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    depth: int,
    skip: bool = False,
    audit_log: Optional[str] = None,
) -> None:
    """Recurse into a `sh -c '<command>'` style argument."""
    for index, token in enumerate(rest):
        if not token.startswith("-"):
            continue
        if "c" not in token and token != "--command":
            continue
        if index + 1 < len(rest):
            _analyze_command(rest[index + 1], cwd, root, ancestors, depth + 1, skip, audit_log)


def _analyze_wrapper(
    verb: str,
    rest: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    depth: int,
    skip: bool = False,
    audit_log: Optional[str] = None,
) -> None:
    """
    Analyze what a command runner will actually run.

    Handles both shapes: the real command as trailing arguments
    (`sudo rm -rf /`) and the real command as a single quoted string
    (`eval "rm -rf /"`, `ssh host reboot`). In skip mode `env -C DIR` is
    refused: the wrapped command runs in a directory this walk does not track.
    """
    if skip and verb == "env":
        for arg in rest:
            if arg.startswith(("-C", "--chdir")):
                _refuse("unverifiable-cwd", _RULE_REASONS["unverifiable-cwd"], f"env {arg}")
            if not arg.startswith("-") and not _ENV_ASSIGNMENT.match(arg):
                break
    inner = _strip_wrapper_args(verb, rest)
    _analyze_segment(inner, cwd, root, ancestors, depth + 1, skip, audit_log)
    for token in inner:
        if _looks_nested(token):
            _analyze_command(token, cwd, root, ancestors, depth + 1, skip, audit_log)
        elif _verb_name(token) in _PROHIBITED_VERBS:
            _check_prohibited_verb(_verb_name(token), skip)


def _analyze_segment(
    tokens: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    depth: int,
    skip: bool = False,
    audit_log: Optional[str] = None,
) -> str:
    """
    Analyze one simple command. Returns the cwd in effect for the next segment.
    """
    if depth > _MAX_NESTING or not tokens:
        return cwd

    # A substitution body is a command line in every token position — not
    # just under a wrapper verb — so analyze it before the verb-specific walk.
    _analyze_substitutions(tokens, cwd, root, ancestors, depth, skip, audit_log)

    index = 0
    while index < len(tokens) and _ENV_ASSIGNMENT.match(tokens[index]):
        index += 1
    if index >= len(tokens):
        return cwd

    verb = _verb_name(tokens[index])
    rest = tokens[index + 1:]

    if verb in _SHELLS:
        _analyze_shell(rest, cwd, root, ancestors, depth, skip, audit_log)
        return cwd
    if verb in _WRAPPERS:
        _analyze_wrapper(verb, rest, cwd, root, ancestors, depth, skip, audit_log)
        return cwd

    # `cd` changes what every later relative path in this command means. In
    # skip mode a target this walk cannot resolve — none (`cd` alone goes
    # home), an expansion, or a glob — makes the cwd unknown, so later
    # relative operands are refused instead of resolved against a guess.
    if verb == "cd":
        operands = _path_operands(rest)
        target = operands[0] if operands else ""
        if not target or any(char in target for char in _EXPANSION_CHARS):
            return _UNKNOWN_CWD if skip else cwd
        if skip and any(char in target for char in _GLOB_CHARS):
            return _UNKNOWN_CWD
        if cwd == _UNKNOWN_CWD and not posixpath.isabs(target):
            return _UNKNOWN_CWD
        return _resolve_operand(target, cwd)
    if skip and verb in ("pushd", "popd"):
        return _UNKNOWN_CWD

    _check_prohibited_verb(verb, skip)
    if verb == "init":
        _check_runlevel(rest, skip)
    _check_service_control(verb, rest, skip)
    if verb in ("pkill", "pgrep"):
        _check_pkill(rest)
    elif verb == "crontab":
        _check_crontab(rest)
    elif verb in ("umount", "mount"):
        _check_mount(verb, rest, cwd, root, ancestors, skip)
    else:
        _check_path_verbs(verb, rest, cwd, root, ancestors, skip, audit_log)
    _check_redirect_targets(tokens, cwd, root, ancestors, skip, audit_log)
    return cwd


def _tokenize(command: str) -> List[str]:
    """
    Split one line into shell tokens, keeping operators as separate items.

    `commenters` is cleared so a `#` mid-command does not silently hide the rest
    of the line from analysis (_split_lines has already neutralized quotes
    inside real comments). Unbalanced quoting is refused: any guessed split
    could hide a command boundary the host shell will honor.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        _refuse("unparseable-quoting", _RULE_REASONS["unparseable-quoting"], command)


def _split_segments(tokens: List[str]) -> List[List[str]]:
    """Group tokens into simple commands, splitting at shell operators."""
    segments: List[List[str]] = []
    current: List[str] = []
    for token in tokens:
        if token in _OPERATORS:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _heredoc_delimiter(command: str, start: int) -> Optional[Tuple[str, bool, bool, int]]:
    """
    Parse the heredoc word after a `<<` that ends at `start`.

    Returns (delimiter, strip_tabs, body_expands, index past the word), or None
    when no word follows or a quote in it never closes. Any quoting in the word
    makes the body literal; unquoted, the body undergoes command substitution.
    """
    index = start
    strip_tabs = command.startswith("-", index)
    if strip_tabs:
        index += 1
    while index < len(command) and command[index] in " \t":
        index += 1
    word_start = index
    word = ""
    quoted = False
    while index < len(command) and command[index] not in " \t\n;&|()<>":
        char = command[index]
        if char in "'\"":
            close = command.find(char, index + 1)
            if close == -1:
                return None
            word += command[index + 1:close]
            quoted = True
            index = close + 1
        elif char == "\\" and index + 1 < len(command):
            word += command[index + 1]
            quoted = True
            index += 2
        else:
            word += char
            index += 1
    if index == word_start:
        return None
    return word, strip_tabs, not quoted, index


def _split_lines(command: str) -> List[Tuple[str, bool]]:
    """
    Split a command into what bash runs line by line, in order.

    Returns (text, is_heredoc_body) pairs. A line is everything up to an
    unquoted newline; a newline inside quotes stays in its line. shlex treats
    newlines as plain whitespace, so without this split `ls\\nrm -rf .` would
    tokenize as one `ls` command and the second line would go unchecked.

    On the way the text is made shlex-safe without changing what it means to
    bash: backslash-newline continuations are joined, ANSI-C `$'...'` strings
    are rewritten so their escaped quotes do not unbalance shlex (the `$` is
    kept, so the expansion rules still see it), and quote characters inside
    comments are escaped (a comment's apostrophe would otherwise open a quote
    that swallows the following lines). Comment text itself stays in the line,
    so the argument layer still analyzes it conservatively.

    Heredoc bodies are data, not commands: they are cut out of the line stream.
    A body whose delimiter is unquoted is still returned, flagged, because its
    `$(...)` and backtick substitutions execute. `<<` inside `$((`, `((`, `${`,
    or `$[` is arithmetic or parameter syntax, not a heredoc, and never hides
    the lines after it.
    """
    items: List[Tuple[str, bool]] = []
    line: List[str] = []
    pending: List[Tuple[str, bool, bool]] = []
    nesting: List[str] = []
    quote = ""
    index = 0
    length = len(command)
    while index < length:
        char = command[index]
        if quote == "'":
            line.append(char)
            if char == "'":
                quote = ""
            index += 1
            continue
        if quote == "$'":
            if char == "\\" and index + 1 < length:
                escaped = command[index + 1]
                line.append("'\\''" if escaped == "'" else char + escaped)
                index += 2
                continue
            line.append(char)
            if char == "'":
                quote = ""
            index += 1
            continue
        if char == "\\" and index + 1 < length:
            if command[index + 1] != "\n":
                line.append(command[index:index + 2])
            index += 2
            continue
        if quote == '"':
            line.append(char)
            if char == '"':
                quote = ""
            index += 1
            continue
        if char == "\n":
            items.append(("".join(line), False))
            line = []
            index += 1
            for delimiter, strip_tabs, expands in pending:
                body: List[str] = []
                while index < length:
                    end = command.find("\n", index)
                    if end == -1:
                        end = length
                    raw = command[index:end]
                    index = end + 1
                    if (raw.lstrip("\t") if strip_tabs else raw) == delimiter:
                        break
                    body.append(raw)
                if expands and body:
                    items.append(("\n".join(body), True))
            pending = []
            continue
        if char in "'\"":
            quote = char
            line.append(char)
            index += 1
            continue
        if command.startswith("$'", index):
            quote = "$'"
            line.append("$'")
            index += 2
            continue
        if char == "#" and (index == 0 or command[index - 1] in " \t\n;&|()<>"):
            end = command.find("\n", index)
            if end == -1:
                end = length
            comment = command[index:end]
            line.append(comment.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"'))
            index = end
            continue
        opener = ""
        for candidate in ("$((", "((", "$(", "${", "$[", "("):
            if command.startswith(candidate, index):
                opener = candidate
                break
        if opener:
            nesting.append("((" if opener.endswith("((") else opener)
            line.append(opener)
            index += len(opener)
            continue
        if char == ")" and nesting:
            if nesting[-1] == "((" and command.startswith("))", index):
                nesting.pop()
                line.append("))")
                index += 2
                continue
            if nesting[-1] in ("(", "$("):
                nesting.pop()
        elif (char == "}" and nesting and nesting[-1] == "${") or (
            char == "]" and nesting and nesting[-1] == "$["
        ):
            nesting.pop()
        elif command.startswith("<<<", index):
            line.append("<<<")
            index += 3
            continue
        elif command.startswith("<<", index):
            parsed = None
            if not any(context in ("((", "${", "$[") for context in nesting):
                parsed = _heredoc_delimiter(command, index + 2)
            if parsed:
                delimiter, strip_tabs, expands, end = parsed
                pending.append((delimiter, strip_tabs, expands))
                line.append(command[index:end])
                index = end
                continue
            line.append("<<")
            index += 2
            continue
        line.append(char)
        index += 1
    if line:
        items.append(("".join(line), False))
    return items


def _analyze_command(
    command: str,
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    depth: int = 0,
    skip: bool = False,
    audit_log: Optional[str] = None,
) -> None:
    """
    Run the argument layer over a command, one line at a time.

    The effective cwd carries across lines exactly as it carries across `;`,
    so `cd /etc` on one line governs the relative paths on the next. An
    expanding heredoc body contributes only its substitution bodies.
    """
    effective_cwd = cwd
    for text, heredoc_body in _split_lines(command):
        if heredoc_body:
            if depth < _MAX_NESTING:
                for body in _dollar_bodies(text) + _backtick_bodies(text):
                    _analyze_command(body, effective_cwd, root, ancestors, depth + 1, skip, audit_log)
            continue
        for segment in _split_segments(_tokenize(text)):
            effective_cwd = _analyze_segment(
                segment, effective_cwd, root, ancestors, depth, skip, audit_log
            )


def _validate_command(
    command: str,
    root: str,
    cwd: Optional[str] = None,
    skip_permissions: bool = False,
    blocked: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (),
    audit_log: Optional[str] = None,
) -> None:
    """
    Refuse a command the destructive-command guardrail cannot prove safe.

    Args:
        command: The raw shell command as the model supplied it.
        root: The configured project root; it and its ancestors are protected.
        cwd: The directory the command will actually run in, used to resolve
            relative operands. Defaults to root.
        skip_permissions: Run under the catastrophic core only. The operator's
            config gate is enforced at the call site; this function itself only
            narrows which rules apply.
        blocked: The operator's compiled blocklist entries, checked before
            everything and never relaxed by skip mode.
        audit_log: Absolute path of guardrail_bypass.log. Writing, deleting,
            or truncating it (or a directory holding it) is refused in every
            mode. None when the caller has no audit log to protect.

    Raises:
        ValueError: Naming the matched rule and the offending text. A refused
            command is never sent to the host.
    """
    # 1. Operator blocklist — always, unbypassable. A match is traceable to
    # config, not code: the user's own pattern is quoted back.
    for raw, pattern in blocked:
        match = pattern.search(command)
        if match:
            _refuse("user-blocklist", f"blocked by your configured pattern {raw!r}", match.group(0))

    # 2. Pattern layer — default: all rules. Skip: catastrophic subset only.
    for name, pattern, reason in _DESTRUCTIVE_PATTERNS:
        if skip_permissions and name not in _CATASTROPHIC_PATTERNS:
            continue
        for match in pattern.finditer(command):
            if name in _APP_TREE_PATH_RULES and _match_in_editable_app_tree(match):
                continue
            _refuse(name, reason, match.group(0))

    # 3. Argument layer — default: full analysis. Skip: reduced analysis
    # (catastrophic verbs, match-everything kills, container-only paths);
    # expansions, protected globs, persistence targets, and the audit log
    # are checked in both.
    normalized_root = _normalize_absolute(root)
    _analyze_command(
        command,
        _normalize_absolute(cwd) if cwd else normalized_root,
        normalized_root,
        _protected_ancestors(normalized_root),
        skip=skip_permissions,
        audit_log=_normalize_absolute(audit_log) if audit_log else None,
    )


# Model-facing note on editing MIRA's own code, present only where the tree is a
# git repository: an install without that undo never advertises self-editing.
_SELF_EDIT_NOTE = (
    f" Files in MIRA's own code tree ({_APP_TREE[0]}) may be edited, created, and "
    "deleted; the parts git does not track (data, venv, .env, logs, .git) stay "
    "refused, and so does the tree as a whole. An edit changes nothing until MIRA "
    "is restarted, and restarting is the operator's step — ask for it. `git diff` "
    "shows the change and `git stash` sets it aside."
    if _APP_TREE is not None else ""
)


class BashTool(Tool):
    """Run guardrailed shell commands locally on the machine MIRA runs on."""

    name = "bash_tool"
    # Any shell command may mutate host state — no per-operation gating
    parallel_safe = False

    simple_description = (
        "Run shell commands locally on this machine; destructive patterns are "
        "refused. For precise file edits, invoke the 'precise-file-editing' skill."
    )

    tool_schema = {
        "name": "bash_tool",
        "description": (
            "Run shell commands locally on this machine; destructive patterns are "
            "refused. For fast, precise edits to files, invoke the "
            "'precise-file-editing' skill via invoke_skill_tool — it teaches the "
            "efficient one-liner patterns (perl/sed in-place edits, heredoc writes, "
            "targeted reads)."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["run", "run_background", "kill_background"],
                    "description": (
                        "run: execute and wait, returning exit code and output. "
                        "run_background: start detached with nohup and return immediately "
                        "with the process ID and log path. kill_background: stop a "
                        "background job this tool previously launched, by its pid "
                        "(the command field is ignored for this operation)."
                    ),
                },
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command executed with `bash -lc` locally on this machine as the service "
                        "user, from cwd (default the project root). Validated against a "
                        "destructive-command guardrail first; a blocked command is refused "
                        "with the matched rule and nothing runs. A refusal means stop: do not "
                        "rephrase the command to slip past the guardrail, and if the goal "
                        "needs an operation the guardrail refuses, stop work and wait for a "
                        "human. Paths that depend on shell expansion ($VAR, ~, backticks) are "
                        "refused for destructive verbs — pass explicit literal paths. Not "
                        "sandboxed: it can read the whole machine's filesystem, but "
                        "system-destructive operations are refused. For precise edits to "
                        "files, the 'precise-file-editing' skill teaches the fastest "
                        "patterns (perl in-place substitution, heredoc writes, targeted "
                        "reads)." + _SELF_EDIT_NOTE
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Working directory for the command. Relative paths resolve against the "
                        "project root; absolute paths must stay inside it or they are rejected. "
                        "Defaults to the project root."
                    ),
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Required. Kill the command after this many seconds "
                        "(the entire process group is SIGKILLed), clamped to "
                        "the configured maximum. "
                        "Ignored by run_background, which uses its own start timeout — still "
                        "pass a value; it is not acted on there."
                    ),
                },
                "log_name": {
                    "type": "string",
                    "description": (
                        "For run_background: short label used in the generated log filename, "
                        "not a path. Defaults to 'job'. The full log path is returned."
                    ),
                },
                "pid": {
                    "type": "string",
                    "description": (
                        "For kill_background: the process ID of a background job this "
                        "tool launched (from the run_background result or the "
                        "conversation). Only pids this tool recorded can be stopped; "
                        "anything else is refused."
                    ),
                },
                "skip_permissions": {
                    "type": "boolean",
                    "description": (
                        "Runs this command with only the catastrophic-core guardrail: rules "
                        "against system-ruining, unrecoverable, or audit-erasing actions. "
                        "Everything else is permitted. Set true ONLY when the human has "
                        "already approved this exact command in this conversation, after you "
                        "stated the command and its purpose. If you have not asked, ask and "
                        "wait. A refusal is still a stop signal; skip_permissions is the "
                        "human's override, never yours."
                    ),
                },
            },
            "required": ["operation", "command", "timeout_seconds"],
        },
    }

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    # -- plumbing -----------------------------------------------------------

    def _resolve(self, cfg: BashToolConfig, path: str) -> str:
        """Normalize a path and confirm it stays inside the configured root."""
        if not path or not path.strip():
            raise ValueError("path is required and cannot be empty")
        if path.startswith("/"):
            resolved = posixpath.normpath(path)
        else:
            resolved = posixpath.normpath(posixpath.join(cfg.root, path))
        root = posixpath.normpath(cfg.root)
        if resolved != root and not resolved.startswith(root + "/"):
            raise ValueError(
                f"Path '{path}' resolves to '{resolved}', outside the permitted root '{root}'. "
                f"All paths must stay inside {root}."
            )
        return resolved

    def _execute_local(
        self,
        cfg: BashToolConfig,
        shell_command: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> subprocess.CompletedProcess:
        """Run a composed shell command locally via subprocess.

        The child is started as its own session leader, so its pid names
        the whole process group and any kill SIGKILLs every descendant —
        the escalation the GNU `timeout -k` wrapper used to provide. That
        binary is absent on stock macOS, so the wrapper must not appear in
        any composed command; the parent timeout bounds duration and this
        group kill supplies the enforcement.
        """
        effective = timeout_seconds or float(cfg.default_timeout_seconds)
        # The wait is sliced at _CANCEL_POLL_SECONDS so the turn's cancel
        # signal is re-checked every slice (see the constant's rationale);
        # the overall budget stays `effective`, enforced by the deadline
        # check in the slice loop.
        try:
            with subprocess.Popen(
                ["bash", "-c", shell_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            ) as proc:
                stdout: Optional[bytes] = None
                stderr: Optional[bytes] = None
                deadline = time.monotonic() + effective
                while stdout is None:
                    try:
                        stdout, stderr = proc.communicate(
                            timeout=min(_CANCEL_POLL_SECONDS, effective)
                        )
                    except subprocess.TimeoutExpired as exc:
                        cancel_event = get_cancel_event()
                        if cancel_event is not None and cancel_event.is_set():
                            cancelled_message = (
                                "Command was stopped by turn cancellation "
                                f"(reason: {get_cancel_reason()}) and its "
                                "process group was killed; partial output "
                                "was discarded."
                            )
                            self._kill_group(proc, cancelled_message)
                            self._drain_killed(proc, _CANCEL_DRAIN_SECONDS)
                            raise ValueError(cancelled_message) from exc
                        if time.monotonic() >= deadline:
                            timeout_message = (
                                f"Command exceeded {effective:.0f}s and its "
                                "process group was killed."
                            )
                            self._kill_group(proc, timeout_message)
                            self._drain_killed(proc, _CANCEL_DRAIN_SECONDS)
                            raise ValueError(timeout_message) from exc
                        # Slice elapsed with neither completion nor
                        # cancellation: keep waiting. The deadline check
                        # above bounds the total wait to one slice past
                        # `effective`.
                return subprocess.CompletedProcess(
                    ["bash", "-c", shell_command], proc.returncode, stdout, stderr
                )
        except OSError as exc:
            raise ValueError(f"Could not launch the local shell: {exc}") from exc

    def _kill_group(self, proc: subprocess.Popen, fail_message: str) -> None:
        """SIGKILL the child's process group.

        ProcessLookupError means the group is already gone (an exited
        leader), the normal no-op. Any other signalling failure raises the
        caller's timeout/cancel-shaped message here, so the outer OSError
        handler cannot relabel a kill failure as a launch failure.
        """
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as kill_exc:
            raise ValueError(fail_message) from kill_exc

    def _drain_killed(
        self, proc: subprocess.Popen, drain_seconds: float
    ) -> None:
        """After the group kill, drain the pipes within drain_seconds.

        A drain that also expires means a descendant escaped the killed
        process group (setsid or double-fork) and holds the inherited pipe
        ends, so the drain can never reach EOF — abandon it: close both
        pipes instead of blocking forever.
        """
        try:
            proc.communicate(timeout=drain_seconds)
        except subprocess.TimeoutExpired:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

    @staticmethod
    def _decode(raw: bytes) -> str:
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _truncate(raw: bytes, limit: int) -> Tuple[str, bool]:
        if len(raw) <= limit:
            return raw.decode("utf-8", errors="replace"), False
        return raw[:limit].decode("utf-8", errors="replace"), True

    def _clamp_timeout(self, cfg: BashToolConfig, requested: Optional[int]) -> int:
        if requested is None:
            return cfg.default_timeout_seconds
        return max(1, min(int(requested), cfg.max_timeout_seconds))

    def _validate_for_run(
        self,
        cfg: BashToolConfig,
        command: str,
        workdir: str,
        skip_permissions: bool,
    ) -> None:
        """
        Config hard gate + the full guardrail, in one place for both operations.

        The gate is the enforcement backstop for the ask-first contract: with
        the operator's switch off, a skip_permissions call is an error, never
        a silent partial relaxation. The user blocklist is compiled from config
        here (outside the pure validator) and passed in as inert data.
        """
        if skip_permissions and not cfg.dangerous_skip_permissions_enabled:
            raise ValueError(
                "skip_permissions is disabled on this instance "
                "(dangerous_skip_permissions_enabled is false). Do not retry with it "
                "set; if this exact action is genuinely needed, ask the operator to "
                "enable the feature."
            )
        audit_log = self._audit_log_path(cfg)
        _validate_command(
            command,
            cfg.root,
            workdir,
            skip_permissions=skip_permissions,
            blocked=_compile_blocklist(tuple(cfg.blocked_patterns)),
            audit_log=audit_log,
        )
        if skip_permissions:
            self._audit_bypass(audit_log, workdir, command)

    def _audit_log_path(self, cfg: BashToolConfig) -> str:
        """guardrail_bypass.log under the configured log_dir (inside root)."""
        return posixpath.join(self._resolve(cfg, cfg.log_dir), "guardrail_bypass.log")

    def _audit_bypass(self, path: str, cwd: str, command: str) -> None:
        """Append one line per skip-mode invocation to guardrail_bypass.log.

        Detection rather than prevention, by design: the ask-first gate is
        instruction-following and the config gate is the hard backstop. This
        record is the operator's after-the-fact audit trail; the validator
        refuses commands that would write, delete, or truncate it. Both fields
        are repr-escaped so an embedded newline cannot forge a second line.
        """
        os.makedirs(posixpath.dirname(path), exist_ok=True)
        stamp = utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{stamp} cwd={cwd!r} command={command!r}\n")

    # -- run ----------------------------------------------------------------

    def run(self, operation: str, command: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a shell command locally on the machine MIRA runs on.

        Args:
            operation: "run" to execute and wait, "run_background" to detach with nohup.
            command: Shell command run with bash -lc on the local machine.
            **kwargs: cwd, timeout_seconds, log_name, skip_permissions as applicable.

        Returns:
            Dict with exit code and captured output (run) or process ID and log path
            (run_background).

        Raises:
            ValueError: If the operation is unknown, a path escapes the root, the
                command is refused by the destructive-command guardrail, or shell
                execution fails.
        """
        from config.config_manager import config

        cfg = config.bash_tool
        handlers = {
            "run": self._run,
            "run_background": self._run_background,
            "kill_background": self._kill_background,
        }
        handler = handlers.get(operation)
        if handler is None:
            raise ValueError(
                f"Unknown operation: {operation}. Valid operations are: " + ", ".join(handlers)
            )
        accepted = set(inspect.signature(handler).parameters)
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return handler(cfg, command, **filtered)

    # -- operations ---------------------------------------------------------

    def _run(
        self,
        cfg: BashToolConfig,
        command: str,
        cwd: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        skip_permissions: bool = False,
    ) -> Dict[str, Any]:
        # The working directory is resolved before validation so relative
        # operands in the command are checked against where they will really
        # land, not against the project root by assumption.
        workdir = self._resolve(cfg, cwd) if cwd else posixpath.normpath(cfg.root)
        self._validate_for_run(cfg, command, workdir, skip_permissions)
        timeout = self._clamp_timeout(cfg, timeout_seconds)
        composed = f"cd {shlex.quote(workdir)} && bash -lc {shlex.quote(command)}"
        result = self._execute_local(
            cfg, composed, timeout_seconds=float(timeout)
        )
        stdout, stdout_truncated = self._truncate(result.stdout, cfg.max_output_bytes)
        stderr, stderr_truncated = self._truncate(result.stderr, cfg.max_output_bytes)
        outcome = {
            "success": True,
            "exit_code": result.returncode,
            "cwd": workdir,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
        if skip_permissions:
            outcome["guardrail"] = "bypassed"
        return outcome

    def _run_background(
        self,
        cfg: BashToolConfig,
        command: str,
        cwd: Optional[str] = None,
        log_name: Optional[str] = None,
        skip_permissions: bool = False,
    ) -> Dict[str, Any]:
        workdir = self._resolve(cfg, cwd) if cwd else posixpath.normpath(cfg.root)
        # Start-failure site 1 — cwd not a directory: refuse it at resolve
        # time, before anything is composed. The check sits beside `_resolve`
        # rather than inside it because `_run` must keep its contract of
        # surfacing a bad cwd as the shell's nonzero exit code.
        if not os.path.isdir(workdir):
            raise ValueError(
                f"run_background failed to start: cwd '{workdir}' is not a directory"
            )
        self._validate_for_run(cfg, command, workdir, skip_permissions)
        log_dir = self._resolve(cfg, cfg.log_dir)
        stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", log_name or "job").strip("_")[:60] or "job"
        log_path = posixpath.join(log_dir, f"{stamp}-{slug}.log")
        # `mkdir` and `cd` run in the FOREGROUND, so a start failure
        # (log_dir not creatable at runtime, a cwd that vanished between the
        # check above and the cd) lands in this process's exit status instead
        # of dying unseen inside a backgrounded subshell. Only the nohup'd job
        # is backgrounded, and every one of its descriptors points at the log
        # or /dev/null — no long-lived process holds the inherited stdout
        # pipe, so `_execute_local` returns as soon as PID/LOG print and `$!`
        # is the survivable job process itself, not a wrapper subshell.
        # `echo "PGID:$$"` records the wrapper's pid: the wrapper is the
        # session leader `_execute_local` creates, so it is the group the
        # launched job and its descendants share — and that group outlives
        # the wrapper, because the nohup'd child stays a member. It is
        # recorded so `kill_background` can still address live descendants
        # after the job's own shell has exited.
        composed = (
            f"mkdir -p {shlex.quote(log_dir)} && cd {shlex.quote(workdir)} || exit 1; "
            f"echo \"PGID:$$\"; "
            f"nohup bash -lc {shlex.quote(command)} "
            f"> {shlex.quote(log_path)} 2>&1 </dev/null & "
            f"child=$!; echo \"PID:$child\"; echo \"LOG:{log_path}\""
        )
        result = self._execute_local(
            cfg,
            composed,
            timeout_seconds=float(cfg.background_start_timeout_seconds),
        )
        if result.returncode != 0:
            stderr = self._decode(result.stderr).strip()
            raise ValueError(
                f"run_background failed to start (exit {result.returncode}): "
                f"{stderr or '(no stderr output)'}"
            )
        stdout = self._decode(result.stdout)
        pid: Optional[str] = None
        pgid: Optional[str] = None
        log_from_host = log_path
        for line in stdout.splitlines():
            if line.startswith("PID:"):
                pid = line[len("PID:"):].strip()
            elif line.startswith("PGID:"):
                pgid = line[len("PGID:"):].strip()
            elif line.startswith("LOG:"):
                log_from_host = line[len("LOG:"):].strip()
        outcome = {
            "success": True,
            "pid": pid,
            "log_path": log_from_host,
            "cwd": workdir,
            "command": command,
        }
        if skip_permissions:
            outcome["guardrail"] = "bypassed"
        if pid is not None:
            # Record the launch so kill_background can stop it later: only
            # pids present in this table are stoppable from the
            # conversation, which is the operation's whole safety property.
            # The group is recorded because it outlives the job's own shell
            # (the nohup'd child and its descendants stay members), which is
            # what lets a job whose shell exited still be stopped.
            ensure_background_jobs(self.db)
            self.db.insert(
                "background_jobs",
                {
                    "pid": pid,
                    "pgid": pgid or "",
                    "log_path": log_from_host,
                    "command": command,
                    "started_at": utc_now().isoformat(),
                },
            )
        return outcome

    def _kill_background(
        self,
        cfg: BashToolConfig,
        command: str,
        pid: Optional[str] = None,
        skip_permissions: bool = False,
    ) -> Dict[str, Any]:
        """Stop a background job this tool launched. `command` is ignored
        here: it is required by the shared schema and run() always passes
        it, so the signature must accept it.

        Only pids recorded by run_background are stoppable. The group is
        resolved from the recorded child while the child is alive (os.getpgid,
        authoritative); when the child has already exited but descendants
        remain in the group, the pgid recorded at launch is addressed
        instead, after confirming that group still exists. The recorded
        group id can in principle be recycled onto an unrelated process once
        the whole group is gone; the existence probe narrows that window to
        the pid counter wrapping onto the recorded value, and the exposure is
        bounded to pids this tool itself recorded. A group kill reaches
        everything in the group; a descendant that left it with setsid is out
        of reach and is named in the result note.
        """
        if pid is None or not str(pid).strip():
            raise ValueError(
                "kill_background requires the pid of a background job this tool "
                "launched; find it in the run_background result or the "
                "conversation that started it."
            )
        pid_clean = str(pid).strip()
        # A kill on a store whose background_jobs table was never created
        # must hit the designed refusal below, not a raw sqlite error.
        ensure_background_jobs(self.db)
        row = self.db.fetchone(
            "SELECT pid, pgid, log_path, command, started_at FROM background_jobs "
            "WHERE pid = :pid",
            {"pid": pid_clean},
        )
        if row is None:
            raise ValueError(
                f"No background job recorded with pid {pid_clean}. kill_background "
                "only stops jobs this tool launched; check the conversation for the "
                "pid, or stop the process from an operator shell."
            )
        try:
            group = os.getpgid(int(pid_clean))
            anchor_alive = True
        except (ProcessLookupError, ValueError):
            # The job's own shell exited; its descendants may still be
            # running in the group recorded at launch. Fall back to that
            # group, addressing it only while it still exists.
            anchor_alive = False
            group = self._recorded_pgid(row)
            if group is None:
                self.db.delete("background_jobs", "pid = :pid", {"pid": pid_clean})
                return {
                    "success": True,
                    "pid": pid_clean,
                    "status": "already exited",
                    "command": row["command"],
                    "note": (
                        "No process group was recorded for this job and its shell "
                        "is gone; nothing was signalled. If descendants survived "
                        "it, stop them from an operator shell."
                    ),
                }
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                self.db.delete("background_jobs", "pid = :pid", {"pid": pid_clean})
                return {
                    "success": True,
                    "pid": pid_clean,
                    "status": "already exited",
                    "command": row["command"],
                }
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            status = "already exited"
        except OSError as exc:
            raise ValueError(
                f"Could not stop background job pid {pid_clean} (group {group}): {exc}"
            ) from exc
        else:
            status = "killed"
        self.db.delete("background_jobs", "pid = :pid", {"pid": pid_clean})
        result: Dict[str, Any] = {
            "success": True,
            "pid": pid_clean,
            "status": status,
            "command": row["command"],
            "log_path": row["log_path"],
        }
        if not anchor_alive:
            result["note"] = (
                "The job's own shell had already exited, so the process group "
                "recorded at launch was addressed directly. A descendant that "
                "left that group with setsid is not reachable and may still run."
            )
        return result

    @staticmethod
    def _recorded_pgid(row: Dict[str, Any]) -> Optional[int]:
        """The pgid recorded at launch, or None when absent or malformed."""
        try:
            return int(row.get("pgid") or "")
        except (TypeError, ValueError):
            return None
