"""
Guardrailed local shell execution on the machine MIRA runs on.

This tool runs shell commands LOCALLY on the machine MIRA runs on (a VM,
container, or host) via subprocess — no network transport. Commands execute as the service
user from a configured project root. Every command is validated against a
two-layer destructive-command guardrail before anything executes; a match is
a hard refusal and the command never reaches the shell.

The two layers cover what the other cannot:

- **Pattern layer** — regexes over the raw command string. Because it sees the
  string before any quoting is removed, it catches destructive literals buried
  inside quotes or nested interpreters (`bash -c "rm -rf /"`,
  `python -c "os.system('rm -rf /')"`) that tokenization would hide.
- **Argument layer** — the command is tokenized, split into simple commands, and
  each command's verb and path operands are resolved against the effective
  working directory. This is what catches the operations no regex can express:
  relative paths (`rm -rf .` from the project root), quoted and globbed spellings
  of a protected path, shell expansions that cannot be resolved before the host
  shell sees them, and destructive verbs other than `rm`.

Both layers are a safety net, not a sandbox. They stop known-catastrophic
operations; they do not confine a determined command to the project tree, and
they cannot see through arbitrary obfuscation (base64 payloads, computed
strings). A refusal is a stop-work signal to the model, not a puzzle to route
around — every refusal message says so explicitly.
"""

import inspect
import logging
import os
import posixpath
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, NoReturn, Optional, Tuple

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import utc_now


# Home directory of the user MIRA runs as, whatever it is named. Path.home()
# raises when no home can be determined — a loud import failure, never a
# silently wrong working directory.
_SERVICE_HOME = str(Path.home())


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


# -- argument layer -----------------------------------------------------------
#
# Verbs whose arguments are resolved and checked against the protected-path
# classifier below.

_DELETION_VERBS = frozenset({"rm", "rmdir", "shred", "unlink", "srm"})
_RELOCATE_VERBS = frozenset({"mv"})
_PERMISSION_VERBS = frozenset({"chmod", "chown", "chgrp", "setfacl"})
_TRUNCATE_VERBS = frozenset({"truncate", "tee"})
_WRITE_VERBS = frozenset({"cp", "install", "ln"})
_INPLACE_EDIT_VERBS = frozenset({"sed", "perl"})

# Never legitimate from this tool, in any argument position.
_PROHIBITED_VERBS = frozenset({
    "shutdown", "reboot", "poweroff", "halt", "telinit", "wipefs",
    "fdisk", "parted", "sgdisk", "gdisk", "cfdisk", "sfdisk",
    "blkdiscard", "swapoff", "ifdown", "killall5",
    "userdel", "deluser", "groupdel", "chpasswd", "passwd", "visudo",
    "mdadm", "pvcreate", "vgcreate", "lvremove", "vgremove", "pvremove",
    "chattr", "mkfs",
})

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

_OPERATORS = frozenset({";", ";;", "&", "&&", "|", "||", "|&", "(", ")", "{", "}", "\n"})
_WRITE_REDIRECTS = frozenset({">", ">>", "&>", "&>>", "1>", "2>", ">", "<>"})
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_DURATION = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")
_INTEGER = re.compile(r"^[-+]?\d+$")
_HOME_DIR = re.compile(r"^/home/[^/]+$")
_NESTING_CHARS = " \t\n;|&<>()$`"
_EXPANSION_CHARS = "$`~"
_GLOB_CHARS = "*?["
_MAX_NESTING = 6

# Protected as themselves; their contents stay reachable, so `rm -rf /tmp/scratch`
# is fine while `rm -rf /tmp` is not.
_SELF_ONLY_DIRS = frozenset({"/tmp", "/mnt", "/media", "/snap", "/lost+found"})

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


def _classify_path(path: str, root: str, ancestors: FrozenSet[str]) -> Optional[str]:
    """
    Name the protection rule a resolved absolute path violates.

    Returns None when the path is not protected. Deleting and modifying paths
    *inside* the project root stays legal; that is the tool's contract.
    """
    resolved = _normalize_absolute(path)
    if resolved == "/":
        return "filesystem-root"
    if resolved in ancestors:
        return "project-root-delete"
    head = resolved[1:].partition("/")[0]
    if head in _SYSTEM_DIR_NAMES:
        return "system-path-delete"
    if resolved in _SELF_ONLY_DIRS:
        return "system-dir-delete"
    if _HOME_DIR.match(resolved):
        return "user-home-delete"
    git_dir = posixpath.join(root, ".git")
    if resolved == git_dir or resolved.startswith(git_dir + "/"):
        return "vcs-history-delete"
    return None


def _at_or_under(path: str, root: str) -> bool:
    """True when path is the project root or somewhere inside it."""
    return path == root or path.startswith(root + "/")


def _resolve_operand(operand: str, cwd: str) -> str:
    """Resolve one command operand to an absolute path against the effective cwd."""
    if posixpath.isabs(operand):
        return _normalize_absolute(operand)
    return _normalize_absolute(posixpath.join(cwd, operand))


def _matches_everything(basename: str) -> bool:
    """True when a glob basename can match every entry in its directory."""
    collapsed = re.sub(r"\[[^\]]*\]", "?", basename)
    return bool(collapsed) and all(char in "*?." for char in collapsed)


def _verb_name(token: str) -> str:
    """Basename of a command word, with the alias-defeating backslash removed."""
    return posixpath.basename(token.lstrip("\\"))


def _path_operands(args: Iterable[str]) -> List[str]:
    """Command arguments that are paths rather than flags."""
    return [arg for arg in args if not arg.startswith("-")]


def _check_operand(
    operand: str,
    verb: str,
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
) -> None:
    """
    Refuse a destructive verb's path operand if it cannot be proven safe.

    Resolves relative operands against the effective cwd, refuses operands whose
    meaning depends on shell expansion, and refuses globs that would empty a
    protected directory.
    """
    if any(char in operand for char in _EXPANSION_CHARS):
        _refuse(
            "unverifiable-expansion",
            _RULE_REASONS["unverifiable-expansion"],
            f"{verb} {operand}",
        )

    resolved = _resolve_operand(operand, cwd)
    hit = _classify_path(resolved, root, ancestors)
    if hit:
        _refuse(hit, _RULE_REASONS[hit], f"{verb} {operand}")

    if not any(char in operand for char in _GLOB_CHARS):
        return

    parent = posixpath.dirname(resolved)
    parent_hit = _classify_path(parent, root, ancestors)
    if not parent_hit:
        return
    if _matches_everything(posixpath.basename(resolved)):
        _refuse(
            "protected-glob",
            _RULE_REASONS["protected-glob"],
            f"{verb} {operand}",
        )
    if not _at_or_under(parent, root):
        _refuse(
            "unglobbable-parent",
            _RULE_REASONS["unglobbable-parent"],
            f"{verb} {operand}",
        )


def _check_redirect_targets(
    tokens: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
) -> None:
    """Refuse output redirections aimed at a protected path."""
    for index, token in enumerate(tokens):
        if token not in _WRITE_REDIRECTS:
            continue
        if index + 1 >= len(tokens):
            continue
        target = tokens[index + 1]
        # `2>&1` and friends name a descriptor, not a path.
        if target.startswith("&") or _INTEGER.match(target):
            continue
        # /dev/null is the bit bucket: discarding output there is benign no
        # matter what protected path classification would otherwise say.
        if _resolve_operand(target, cwd) == "/dev/null":
            continue
        resolved = _resolve_operand(target, cwd)
        hit = _classify_path(resolved, root, ancestors)
        if hit:
            _refuse(hit, _RULE_REASONS[hit], f"{token} {target}")


def _check_prohibited_verb(verb: str) -> None:
    """Refuse verbs that have no legitimate use from this tool."""
    if verb in _PROHIBITED_VERBS or verb.startswith("mkfs."):
        _refuse(
            "prohibited-command",
            "this command has no legitimate use against the remote host and "
            "damages the host or its data",
            verb,
        )


def _check_runlevel(args: List[str]) -> None:
    """Refuse `init N`, which changes the system runlevel."""
    operands = _path_operands(args)
    if operands and operands[0] in {"0", "1", "2", "3", "4", "5", "6"}:
        _refuse("runlevel-change", "changing the system runlevel", f"init {operands[0]}")


def _check_service_control(verb: str, args: List[str]) -> None:
    """Refuse systemd and SysV operations that power off or disable the host."""
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
    """Refuse `crontab -r`, which wipes every scheduled job for the user."""
    if any(arg in ("-r", "--remove") for arg in args):
        _refuse("cron-wipe", "removing every scheduled job for the user", "crontab -r")


def _check_mount(
    verb: str,
    args: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
) -> None:
    """Refuse unmounting broadly or remounting a protected filesystem."""
    if any(arg in ("-a", "--all", "-f", "--force") for arg in args):
        _refuse("mount-change", f"{verb} applied to every filesystem", " ".join([verb] + list(args)))
    for operand in _path_operands(args):
        if any(char in operand for char in _EXPANSION_CHARS):
            continue
        hit = _classify_path(_resolve_operand(operand, cwd), root, ancestors)
        if hit:
            _refuse(hit, _RULE_REASONS[hit], f"{verb} {operand}")


def _check_dd(args: List[str], cwd: str, root: str, ancestors: FrozenSet[str]) -> None:
    """Refuse a dd write target that is a device or a protected path."""
    for arg in args:
        if not arg.startswith("of="):
            continue
        target = arg[len("of="):]
        if any(char in target for char in _EXPANSION_CHARS):
            _refuse("unverifiable-expansion", _RULE_REASONS["unverifiable-expansion"], arg)
        hit = _classify_path(_resolve_operand(target, cwd), root, ancestors)
        if hit:
            _refuse(hit, _RULE_REASONS[hit], arg)


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


def _check_find(args: List[str], cwd: str, root: str, ancestors: FrozenSet[str]) -> None:
    """
    Refuse a find that deletes without a narrowing predicate over a safe tree.

    `find . -name '*.pyc' -delete` inside the project root is ordinary cleanup.
    `find . -delete` from the same directory removes the whole harness, and
    `find / ...` walks off the project tree entirely. A predicate only earns
    the carve-out when it demonstrably narrows: `-name '*'` matches everything
    and does not count, and neither does an unparseable expression.
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
        if any(char in start for char in _EXPANSION_CHARS):
            _refuse("unverifiable-expansion", _RULE_REASONS["unverifiable-expansion"], f"find {start}")
        resolved = _resolve_operand(start, cwd)
        hit = _classify_path(resolved, root, ancestors)
        if hit and not (narrowed and _at_or_under(resolved, root)):
            _refuse(hit, _RULE_REASONS[hit], f"find {start}")


def _check_path_verbs(
    verb: str,
    args: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
) -> None:
    """Dispatch a path-taking verb to its operand check."""
    if verb in _DELETION_VERBS or verb in _RELOCATE_VERBS or verb in _PERMISSION_VERBS:
        for operand in _path_operands(args):
            _check_operand(operand, verb, cwd, root, ancestors)
    elif verb in _TRUNCATE_VERBS or verb in _WRITE_VERBS:
        for operand in _path_operands(args):
            _check_operand(operand, verb, cwd, root, ancestors)
    elif verb in _INPLACE_EDIT_VERBS:
        if any(arg == "-i" or arg.startswith("-i") for arg in args):
            for operand in _path_operands(args):
                _check_operand(operand, verb, cwd, root, ancestors)
    elif verb == "dd":
        _check_dd(args, cwd, root, ancestors)
    elif verb == "find":
        _check_find(args, cwd, root, ancestors)


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
        _analyze_command(body, cwd, root, ancestors, depth + 1)


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
) -> None:
    """Recurse into a `sh -c '<command>'` style argument."""
    for index, token in enumerate(rest):
        if not token.startswith("-"):
            continue
        if "c" not in token and token != "--command":
            continue
        if index + 1 < len(rest):
            _analyze_command(rest[index + 1], cwd, root, ancestors, depth + 1)


def _analyze_wrapper(
    verb: str,
    rest: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    depth: int,
) -> None:
    """
    Analyze what a command runner will actually run.

    Handles both shapes: the real command as trailing arguments
    (`sudo rm -rf /`) and the real command as a single quoted string
    (`eval "rm -rf /"`, `ssh host reboot`).
    """
    inner = _strip_wrapper_args(verb, rest)
    _analyze_segment(inner, cwd, root, ancestors, depth + 1)
    for token in inner:
        if _looks_nested(token):
            _analyze_command(token, cwd, root, ancestors, depth + 1)
        elif _verb_name(token) in _PROHIBITED_VERBS:
            _check_prohibited_verb(_verb_name(token))


def _analyze_segment(
    tokens: List[str],
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    depth: int,
) -> str:
    """
    Analyze one simple command. Returns the cwd in effect for the next segment.
    """
    if depth > _MAX_NESTING or not tokens:
        return cwd

    # A substitution body is a command line in every token position — not
    # just under a wrapper verb — so analyze it before the verb-specific walk.
    _analyze_substitutions(tokens, cwd, root, ancestors, depth)

    index = 0
    while index < len(tokens) and _ENV_ASSIGNMENT.match(tokens[index]):
        index += 1
    if index >= len(tokens):
        return cwd

    verb = _verb_name(tokens[index])
    rest = tokens[index + 1:]

    if verb in _SHELLS:
        _analyze_shell(rest, cwd, root, ancestors, depth)
        return cwd
    if verb in _WRAPPERS:
        _analyze_wrapper(verb, rest, cwd, root, ancestors, depth)
        return cwd

    # `cd` changes what every later relative path in this command means.
    if verb == "cd":
        operands = _path_operands(rest)
        if operands and not any(char in operands[0] for char in _EXPANSION_CHARS):
            return _resolve_operand(operands[0], cwd)
        return cwd

    _check_prohibited_verb(verb)
    if verb == "init":
        _check_runlevel(rest)
    _check_service_control(verb, rest)
    if verb in ("pkill", "pgrep"):
        _check_pkill(rest)
    elif verb == "crontab":
        _check_crontab(rest)
    elif verb in ("umount", "mount"):
        _check_mount(verb, rest, cwd, root, ancestors)
    else:
        _check_path_verbs(verb, rest, cwd, root, ancestors)
    _check_redirect_targets(tokens, cwd, root, ancestors)
    return cwd


def _tokenize(command: str) -> List[str]:
    """
    Split a command line into shell tokens, keeping operators as separate items.

    `commenters` is cleared so a `#` mid-command does not silently hide the rest
    of the line from analysis. On unbalanced quoting — where shlex is stricter
    than bash — falls back to a crude split so this layer still runs; the
    pattern layer has already examined the raw string either way.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return [token for token in re.split(r"[\s;|&()<>]+", command) if token]


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


def _analyze_command(
    command: str,
    cwd: str,
    root: str,
    ancestors: FrozenSet[str],
    depth: int = 0,
) -> None:
    """Run the argument layer over a command line."""
    effective_cwd = cwd
    for segment in _split_segments(_tokenize(command)):
        effective_cwd = _analyze_segment(segment, effective_cwd, root, ancestors, depth)


def _validate_command(command: str, root: str, cwd: Optional[str] = None) -> None:
    """
    Refuse a command the destructive-command guardrail cannot prove safe.

    Args:
        command: The raw shell command as the model supplied it.
        root: The configured project root; it and its ancestors are protected.
        cwd: The directory the command will actually run in, used to resolve
            relative operands. Defaults to root.

    Raises:
        ValueError: Naming the matched rule and the offending text. A refused
            command is never sent to the host.
    """
    for name, pattern, reason in _DESTRUCTIVE_PATTERNS:
        match = pattern.search(command)
        if match:
            _refuse(name, reason, match.group(0))

    normalized_root = _normalize_absolute(root)
    _analyze_command(
        command,
        _normalize_absolute(cwd) if cwd else normalized_root,
        normalized_root,
        _protected_ancestors(normalized_root),
    )


class BashTool(Tool):
    """Run guardrailed shell commands locally on the machine MIRA runs on."""

    name = "bash_tool"
    # Any shell command may mutate host state — no per-operation gating
    parallel_safe = False

    simple_description = "Run shell commands locally on this machine; destructive patterns are refused."

    tool_schema = {
        "name": "bash_tool",
        "description": "Run shell commands locally on this machine; destructive patterns are refused.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["run", "run_background"],
                    "description": (
                        "run: execute and wait, returning exit code and output. "
                        "run_background: start detached with nohup and return immediately "
                        "with the process ID and log path."
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
                        "system-destructive operations are refused."
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
                        "Required. Kill the command after this many seconds (SIGTERM, then "
                        "SIGKILL ten seconds later), clamped to the configured maximum. "
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
        """Run a composed shell command locally via subprocess."""
        effective = timeout_seconds or float(cfg.default_timeout_seconds)
        try:
            return subprocess.run(
                ["bash", "-c", shell_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=effective,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                f"Command exceeded {effective:.0f}s and was killed. "
                f"The process may still be running."
            ) from exc
        except OSError as exc:
            raise ValueError(f"Could not launch the local shell: {exc}") from exc

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

    # -- run ----------------------------------------------------------------

    def run(self, operation: str, command: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a shell command locally on the machine MIRA runs on.

        Args:
            operation: "run" to execute and wait, "run_background" to detach with nohup.
            command: Shell command run with bash -lc on the local machine.
            **kwargs: cwd, timeout_seconds, log_name as applicable.

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
    ) -> Dict[str, Any]:
        # The working directory is resolved before validation so relative
        # operands in the command are checked against where they will really
        # land, not against the project root by assumption.
        workdir = self._resolve(cfg, cwd) if cwd else posixpath.normpath(cfg.root)
        _validate_command(command, cfg.root, workdir)
        timeout = self._clamp_timeout(cfg, timeout_seconds)
        composed = (
            f"cd {shlex.quote(workdir)} && "
            f"timeout -k 10s {timeout}s bash -lc {shlex.quote(command)}"
        )
        result = self._execute_local(cfg, composed, timeout_seconds=timeout + 20)
        stdout, stdout_truncated = self._truncate(result.stdout, cfg.max_output_bytes)
        stderr, stderr_truncated = self._truncate(result.stderr, cfg.max_output_bytes)
        return {
            "success": True,
            "exit_code": result.returncode,
            "timed_out": result.returncode == 124,
            "cwd": workdir,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }

    def _run_background(
        self,
        cfg: BashToolConfig,
        command: str,
        cwd: Optional[str] = None,
        log_name: Optional[str] = None,
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
        _validate_command(command, cfg.root, workdir)
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
        composed = (
            f"mkdir -p {shlex.quote(log_dir)} && cd {shlex.quote(workdir)} || exit 1; "
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
        log_from_host = log_path
        for line in stdout.splitlines():
            if line.startswith("PID:"):
                pid = line[len("PID:"):].strip()
            elif line.startswith("LOG:"):
                log_from_host = line[len("LOG:"):].strip()
        return {
            "success": True,
            "pid": pid,
            "log_path": log_from_host,
            "cwd": workdir,
            "command": command,
        }
