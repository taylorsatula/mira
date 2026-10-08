"""
Release identity: what version this tree is, how release names order, and how
the newest published release is resolved.

Dependency-free on purpose — stdlib only — so both the server process (via
``utils/update_check.py``, which owns the 24 h check job and the cached
verdict) and the TUI's ``mira update`` subcommand (``tui/update.py``, which
runs on the minimal TUI venv) share one implementation of each half. Neither
duplicates the other; this module is the single owner.
"""
import json
import urllib.request
from pathlib import Path

# Repo root: this file lives in utils/.
VERSION_FILE = Path(__file__).parent.parent / "VERSION"

# The newest *published* release, which is also what install.sh resolves. The
# unauthenticated GitHub API allows 60 requests/hour/IP; the check job uses one
# per 24 h and `mira update` one per invocation.
RELEASES_LATEST_URL = "https://api.github.com/repos/taylorsatula/mira/releases/latest"
FETCH_TIMEOUT_SECONDS = 10

GITHUB_REPO = "taylorsatula/mira"
TAG_TARBALL_URL = f"https://github.com/{GITHUB_REPO}/archive/refs/tags/v{{tag}}.tar.gz"

# The OSS development repo: `mira update --nightly` installs its main HEAD
# as a ``nightly-<shortsha>`` identity, so patchfixes reach installs without
# cutting a release.
OSS_REPO = "taylorsatula/mira-OSS"
BRANCH_TARBALL_URL = f"https://github.com/{OSS_REPO}/archive/refs/heads/main.tar.gz"
BRANCH_HEAD_URL = f"https://api.github.com/repos/{OSS_REPO}/commits/main"


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
    Full SHA of the OSS repo's ``main`` HEAD.

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
