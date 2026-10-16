"""
Release identity: what version this tree is, how release names order, how
the newest published release is resolved, and whether a built release
artifact agrees with the tag it claims.

Dependency-free on purpose — stdlib only — so both the server process (via
``utils/update_check.py``, which owns the 24 h check job and the cached
verdict) and the TUI's ``mira update`` subcommand (``ui/tui/update.py``, which
runs on the minimal TUI venv) share one implementation of each half. Neither
duplicates the other; this module is the single owner.

CLI (used by ``deploy/lib/provenance.sh:build_release_artifact``)::

    python3 -m utils.release_identity verify <repo-root> <tag> <artifact-root>

prints one line per check and exits 0 when every check holds, 1 otherwise.
"""
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
VERSION_FILE = REPO_ROOT / "VERSION"
GITHUB_REPO_FILE = REPO_ROOT / "deploy" / "GITHUB_REPO"

FETCH_TIMEOUT_SECONDS = 10
GIT_TIMEOUT_SECONDS = 30
RELEASE_COMMIT_FILE = "RELEASE_COMMIT"
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def _read_github_repo() -> str:
    """
    The repository spelling from ``deploy/GITHUB_REPO``.

    Raises:
        OSError: When the file is missing, unreadable, or empty.
    """
    try:
        repo = GITHUB_REPO_FILE.read_text().strip()
    except OSError as error:
        raise OSError(f"cannot read {GITHUB_REPO_FILE}: {error}") from error
    if not repo:
        raise OSError(f"{GITHUB_REPO_FILE} is empty")
    return repo


GITHUB_REPO = _read_github_repo()

# Newest *published* release — the endpoint install.sh resolves (REPO_URL/releases/latest).
RELEASES_LATEST_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# The built release asset (deploy/lib/provenance.sh:build_release_artifact) —
# the only payload installs and updates fetch. Format with the bare version
# (no leading ``v``).
RELEASE_ASSET_URL = (
    f"https://github.com/{GITHUB_REPO}/releases/download/v{{version}}/mira-{{version}}.tar.gz"
)

# `mira update --nightly` installs this repo's main HEAD as a
# ``nightly-<shortsha>`` identity, so patchfixes reach installs without
# cutting a release.
BRANCH_TARBALL_URL = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/main.tar.gz"
BRANCH_HEAD_URL = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"


def version_sort_key(raw: str) -> tuple[int, ...] | None:
    """
    Comparable sort key for MIRA's release identity scheme.

    Releases are named ``YYYY.MM.DD`` with an optional ``-N.N`` revision suffix
    (``2026.10.05``, ``2026.10.03-2.0``). The ``-N.N`` suffix is not PEP 440, so
    ``packaging.version.parse`` rejected every such string and comparison
    silently fell through to "no update" — including for installs that report
    the suffix while the newer release does not. Each numeric component becomes
    an int, so padded and unpadded forms compare equal (``10.03`` == ``10.3``)
    and a leading ``v`` (release tags carry one) is ignored. A trailing revision
    makes a version sort above the same date without one.

    Returns None when the input is not a MIRA release name, so the caller
    declines to compare rather than mis-ordering.
    """
    s = raw.strip()
    if s[:1] in ("v", "V"):
        s = s[1:]
    parts = s.replace("-", ".").split(".")
    if any(not (p.isascii() and p.isdigit()) for p in parts):
        return None
    return tuple(int(p) for p in parts)


def get_current_version() -> str:
    """
    This tree's release identity, read from the repo-root ``VERSION``.

    Raises:
        OSError: If VERSION is missing or unreadable (a broken install, not a
            transient condition — callers that must not fail are expected to
            translate it).
    """
    return VERSION_FILE.read_text().strip()


def fetch_latest_release_tag() -> str:
    """
    Newest published release name from GitHub, leading ``v`` stripped.

    Raises:
        urllib.error.URLError: On any transport failure, including the
            ``FETCH_TIMEOUT_SECONDS`` bound expiring.
        ValueError: When the response is not JSON, or carries no usable
            ``tag_name``.
    """
    request = urllib.request.Request(
        RELEASES_LATEST_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "mira-release-check",
        },
    )
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read())
    tag = payload.get("tag_name")
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError(f"release response carried no tag_name: {payload!r}")
    tag = tag.strip()
    return tag[1:] if tag[:1] in ("v", "V") else tag


def fetch_branch_head_sha() -> str:
    """
    Full SHA of the repo's ``main`` HEAD.

    Raises:
        urllib.error.URLError: On any transport failure, including the
            ``FETCH_TIMEOUT_SECONDS`` bound expiring.
        ValueError: When the response is not JSON, or carries no usable ``sha``.
    """
    request = urllib.request.Request(
        BRANCH_HEAD_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "mira-release-check",
        },
    )
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read())
    sha = payload.get("sha")
    if not isinstance(sha, str) or not sha.strip():
        raise ValueError(f"branch-head response carried no sha: {payload!r}")
    return sha.strip()


@dataclass(frozen=True)
class IdentityCheck:
    """One release-identity comparison: the two values it compared and whether they agree."""

    name: str
    expected: str
    observed: str

    @property
    def ok(self) -> bool:
        return self.expected == self.observed


@dataclass(frozen=True)
class IdentityReport:
    """The three checks a release artifact must pass before it is published."""

    tag_matches_version: IdentityCheck
    artifact_matches_repo: IdentityCheck
    release_commit_matches_tag: IdentityCheck

    @property
    def checks(self) -> tuple[IdentityCheck, ...]:
        return (
            self.tag_matches_version,
            self.artifact_matches_repo,
            self.release_commit_matches_tag,
        )

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def _read_stripped(path: Path) -> str:
    """File contents stripped, or an ``<unreadable …>`` marker that can never equal a real value."""
    try:
        return path.read_text().strip()
    except OSError as error:
        return f"<unreadable {path}: {error}>"


def _tag_commit(repo_root: Path, tag: str) -> str:
    """The commit ``tag`` points at in ``repo_root``, or an ``<unresolvable …>`` marker."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    sha = result.stdout.strip()
    if result.returncode != 0 or not _SHA_PATTERN.fullmatch(sha):
        return f"<unresolvable tag {tag} in {repo_root}>"
    return sha


def verify_release_identity(repo_root: Path, tag: str, artifact_root: Path) -> IdentityReport:
    """
    Compare a built release artifact against the repository and tag it claims.

    - ``tag_matches_version``: ``tag`` equals ``"v" + <repo VERSION>``.
    - ``artifact_matches_repo``: the artifact's ``VERSION`` equals the repo's.
    - ``release_commit_matches_tag``: the artifact's ``RELEASE_COMMIT`` equals
      the commit ``tag`` resolves to.

    An unreadable input fails its own check, naming the path; nothing raises.
    """
    repo_version = _read_stripped(repo_root / "VERSION")
    return IdentityReport(
        tag_matches_version=IdentityCheck(
            name="tag_matches_version",
            expected=tag,
            observed=f"v{repo_version}",
        ),
        artifact_matches_repo=IdentityCheck(
            name="artifact_matches_repo",
            expected=repo_version,
            observed=_read_stripped(artifact_root / "VERSION"),
        ),
        release_commit_matches_tag=IdentityCheck(
            name="release_commit_matches_tag",
            expected=_tag_commit(repo_root, tag),
            observed=_read_stripped(artifact_root / RELEASE_COMMIT_FILE),
        ),
    )


def _main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[0] != "verify":
        print(
            "usage: python3 -m utils.release_identity verify <repo-root> <tag> <artifact-root>",
            file=sys.stderr,
        )
        return 2
    report = verify_release_identity(Path(argv[1]), argv[2], Path(argv[3]))
    for check in report.checks:
        verdict = "ok" if check.ok else "MISMATCH"
        print(f"{check.name}: {verdict} (expected {check.expected!r}, observed {check.observed!r})")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
