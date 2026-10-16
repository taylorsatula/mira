"""
Settings-file reconciliation for ``mira update``: remove ``MIRA_*`` variables
from ``/opt/mira/systemone.env`` that nothing in the installed tree reads any
more, keeping every other line (comments, order, non-MIRA variables) exactly.

Stdlib only. The declared set is every ``MIRA_[A-Z0-9_]+`` token in the tree's
Python and shell sources — deliberately broad: a name kept by mistake costs
one line, a name dropped by mistake changes a running install.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_NAME_RE = re.compile(r"\bMIRA_[A-Z0-9_]+\b")
_SKIPPED_DIRS = frozenset({".git", "venv", "data", "logs", "scratch", "__pycache__", "node_modules"})
_SOURCE_SUFFIXES = (".py", ".sh")


def declared_environment_variables(repo_root: Path) -> dict[str, str]:
    """Every ``MIRA_*`` name the tree's sources mention → the first ``path:line`` mentioning it."""
    declared: dict[str, str] = {}
    for path in sorted(repo_root.rglob("*")):
        relative = path.relative_to(repo_root)
        if any(part in _SKIPPED_DIRS or part.startswith(("venv", "tui-venv")) for part in relative.parts):
            continue
        # s6 service scripts (deploy/docker/s6-rc.d/*/run, */up) carry no suffix.
        if not path.is_file() or not (path.suffix in _SOURCE_SUFFIXES or path.name in ("run", "up")):
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            for name in _NAME_RE.findall(line):
                declared.setdefault(name, f"{relative}:{number}")
    return declared


@dataclass(frozen=True)
class DroppedVariable:
    name: str
    line: str


@dataclass(frozen=True)
class EnvReconciliation:
    """``rendered`` is the file with each dropped line removed and every other line unchanged."""

    carried_forward: tuple[str, ...]
    dropped: tuple[DroppedVariable, ...]
    rendered: tuple[str, ...]


def reconcile_env_file(lines: list[str], declared: dict[str, str]) -> EnvReconciliation:
    """Pure: classify each ``MIRA_*`` assignment in ``lines`` against ``declared``."""
    kept: list[str] = []
    carried: list[str] = []
    dropped: list[DroppedVariable] = []
    for line in lines:
        name, sep, _ = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        if sep and not line.lstrip().startswith("#") and name.startswith("MIRA_"):
            if name in declared:
                carried.append(name)
            else:
                dropped.append(DroppedVariable(name=name, line=line))
                continue
        kept.append(line)
    return EnvReconciliation(carried_forward=tuple(carried), dropped=tuple(dropped), rendered=tuple(kept))
