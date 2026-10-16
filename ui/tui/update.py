"""
``mira update`` — update a deployed MIRA install in place.

The decision half of the update: resolve the newest published release (one
mechanism with the server's daily check — ``utils/release_identity.py``),
compare it against the installed ``VERSION``, fetch the release asset (the
tarball ``deploy/lib/provenance.sh:build_release_artifact`` builds — never
GitHub's auto-archive). The machine half — consent for database changes,
service stop, venv rebuild, code swap, database steps with a backup first,
restart, health-poll, and putting everything back on failure — is
``deploy/update.sh`` in the *new* tree,
executed as a subprocess: like ``install.sh``, the newest release's own code
does the work, so conventions never lag behind the release being installed.

``mira update --nightly`` skips the release ladder entirely and installs the
HEAD of the repository's main branch, identified as ``nightly-<shortsha>``.
A successful update records the SHA in ``<MIRA_APP_DIR>/data/nightly_stamp``
(written by ``deploy/update.sh`` after the health poll passes), and the next
nightly run is a no-op while main has not moved.

``--rebuild-venvs`` (either mode) takes a dedicated fast path that only
rebuilds the venvs from the INSTALLED tree — no release download, no code
swap, no database steps, no consent gate — so the escape hatch works exactly
when the version is already current and a venv is wrong. ``--adopt-drifted``
lets the one-time database handover proceed on a database whose structure
matches no known release (the update refuses it otherwise and says exactly
that); the up-to-date no-ops do not swallow it.

Environment (both this module and ``deploy/update.sh`` honor the same names;
they exist so verification probes can run the real flow against a throwaway
install root):

- ``MIRA_APP_DIR``   the install root (default ``/opt/mira/app``)
- ``MIRA_TUI_VENV``  the TUI client venv (default ``/opt/mira/tui-venv``)

Exit codes (``deploy/update.sh``'s, forwarded unchanged): 0 up to date,
updated, or cancelled at the prompt; 1 failure (rolled back, or parked with
the exact commands); 3 the update needs a
terminal or the operator's decision and changed nothing. Nothing on the
install is touched before ``deploy/update.sh`` starts.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from utils.release_identity import (
    BRANCH_TARBALL_URL,
    RELEASE_ASSET_URL,
    fetch_branch_head_sha,
    fetch_latest_release_tag,
    get_current_version,
    version_sort_key,
)

DOWNLOAD_TIMEOUT_SECONDS = 120

# The nightly tarball, pinned to the resolved head SHA: GitHub serves any
# commit's archive at ``archive/<sha>.tar.gz``, so formatting this URL with
# the SHA makes the payload that lands on disk and the recorded identity
# (``NIGHTLY_SHA`` / ``data/nightly_stamp``) the same commit — the
# ``refs/heads/main`` archive would be a second, independent read of a
# moving ref.
NIGHTLY_TARBALL_URL = BRANCH_TARBALL_URL.replace("refs/heads/main", "{sha}")


def _fail(message: str) -> None:
    print(f"mira update: {message}", file=sys.stderr)


def _download(url: str, destination: Path, name: str) -> Path:
    """
    Download ``url`` into ``destination`` as ``name``.

    Raises:
        OSError: On any transport failure, including the download bound.
    """
    tarball = destination / name
    request = urllib.request.Request(
        url, headers={"User-Agent": "mira-update"}
    )
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        tarball.write_bytes(response.read())
    return tarball


def _download_release(version: str, destination: Path) -> Path:
    """
    Download the release asset for ``version`` (no leading ``v``) into
    ``destination``.

    Raises:
        OSError: On any transport failure, including the download bound; a
            release without the asset is reported by name.
    """
    url = RELEASE_ASSET_URL.format(version=version)
    name = f"mira-{version}.tar.gz"
    try:
        return _download(url, destination, name)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise OSError(f"release v{version} has no asset {name} ({url})") from error
        raise


def _extract_release(tarball: Path, destination: Path) -> Path:
    """
    Extract the tarball and return the release tree's root directory.

    GitHub tag archives contain exactly one top-level directory. The ``data``
    filter rejects path traversal, absolute paths, and special members — a
    tampered archive cannot write outside ``destination``.
    """
    with tarfile.open(tarball, mode="r:gz") as archive:
        archive.extractall(destination, filter="data")
    roots = [entry for entry in destination.iterdir() if entry.is_dir()]
    if len(roots) != 1:
        raise ValueError(
            f"release archive contained {len(roots)} top-level directories, expected 1"
        )
    return roots[0]


def _gate_and_hand_off(
    release_tree: Path,
    label: str,
    tag: str,
    extra_env: dict[str, str] | None = None,
    rebuild_venvs: bool = False,
    adopt_drifted: bool = False,
) -> int:
    """
    Check the tree carries ``deploy/update.sh``, then hand off to it with
    ``tag`` (and ``--rebuild-venvs`` / ``--adopt-drifted`` when asked).
    Nothing on the installed machine is touched before the handoff. The env
    carries the install roots (MIRA_APP_DIR / MIRA_TUI_VENV) so a non-default
    root set by the caller flows through unchanged, plus any ``extra_env``
    (the nightly mode adds NIGHTLY / NIGHTLY_SHA).
    """
    updater = release_tree / "deploy" / "update.sh"
    if not updater.is_file():
        _fail(f"release {label} carries no deploy/update.sh — it predates in-place update.")
        return 1
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    argv = ["bash", str(updater), str(release_tree), tag]
    if rebuild_venvs:
        argv.append("--rebuild-venvs")
    if adopt_drifted:
        argv.append("--adopt-drifted")
    result = subprocess.run(argv, env=env)
    return result.returncode


def _run_nightly_update(rebuild_venvs: bool, adopt_drifted: bool) -> int:
    """
    Update the install to the repository's ``main`` HEAD.

    Identity is the branch-head SHA (displayed as its first 12 chars), not
    the VERSION file: a nightly tree can carry the same VERSION as the
    installed tree while the code differs. The SHA is recorded in
    ``<MIRA_APP_DIR>/data/nightly_stamp`` by ``deploy/update.sh`` after the
    health poll passes, so a failed update does not advance the stamp.
    """
    app_dir = Path(os.environ.get("MIRA_APP_DIR", "/opt/mira/app"))
    stamp_path = app_dir / "data" / "nightly_stamp"
    try:
        sha = fetch_branch_head_sha()
    except Exception as error:
        _fail(f"could not resolve the main head ({type(error).__name__}: {error}).")
        return 1
    if stamp_path.is_file():
        try:
            stamped = stamp_path.read_text().strip()
        except OSError as error:
            _fail(f"cannot read the nightly stamp ({error}).")
            return 1
        if stamped == sha and not adopt_drifted:
            print(f"MIRA nightly {sha[:12]} already current.")
            return 0

    label = f"nightly-{sha[:12]}"
    print(f"MIRA {label} is available — updating in place.")
    with tempfile.TemporaryDirectory(prefix="mira-update-") as scratch:
        try:
            tarball = _download(NIGHTLY_TARBALL_URL.format(sha=sha), Path(scratch), f"{label}.tar.gz")
            release_tree = _extract_release(tarball, Path(scratch))
        except (OSError, ValueError, tarfile.TarError) as error:
            _fail(f"could not fetch the {label} tree ({type(error).__name__}: {error}).")
            return 1
        return _gate_and_hand_off(
            release_tree, label, label,
            extra_env={"NIGHTLY": "1", "NIGHTLY_SHA": sha},
            rebuild_venvs=rebuild_venvs,
            adopt_drifted=adopt_drifted,
        )


def _rebuild_venvs_only() -> int:
    """
    The dedicated ``--rebuild-venvs`` fast path: rebuild both venvs from the
    INSTALLED tree, nothing else — no release fetch, no code swap, no database
    steps, no consent gate. The installed tree's own ``deploy/update.sh``
    does the work in its ``--rebuild-venvs-only`` mode: extras re-detected
    from the old venvs, builds with the stamp written last, the aside/undo
    discipline around the swap, and a stop/start around it.
    """
    app_dir = Path(os.environ.get("MIRA_APP_DIR", "/opt/mira/app"))
    updater = app_dir / "deploy" / "update.sh"
    if not updater.is_file():
        _fail(f"{app_dir} carries no deploy/update.sh — it predates in-place update.")
        return 1
    try:
        version = (app_dir / "VERSION").read_text().strip()
    except OSError as error:
        _fail(f"cannot read the installed VERSION ({error}).")
        return 1
    argv = ["bash", str(updater), str(app_dir), f"v{version}", "--rebuild-venvs-only"]
    result = subprocess.run(argv, env=os.environ.copy(), check=False)
    return result.returncode


def run_update(
    nightly: bool = False,
    release_tree: Path | None = None,
    rebuild_venvs: bool = False,
    adopt_drifted: bool = False,
) -> int:
    """
    Resolve, gate, and hand off to the release's own ``deploy/update.sh``.

    Nothing on the installed machine is touched until ``deploy/update.sh``
    starts; any pre-handoff failure leaves the install exactly as it was.

    ``--rebuild-venvs`` never reaches this machinery: it dispatches to the
    dedicated fast path (``_rebuild_venvs_only``) before any version or
    nightly-stamp comparison, so no up-to-date short-circuit can swallow it.

    ``release_tree`` is the verification seam: a real, extracted release tree
    on disk, used to exercise the compare/gate/handoff flow against real
    files when the release in question is not published on GitHub yet (the
    live path resolves and downloads it). The tree's own ``VERSION`` is the
    release identity; every later step is identical.
    """
    if rebuild_venvs:
        return _rebuild_venvs_only()
    if nightly:
        return _run_nightly_update(rebuild_venvs, adopt_drifted)
    try:
        current = get_current_version()
    except OSError as error:
        _fail(f"cannot read the installed VERSION ({error}).")
        return 1

    if release_tree is None:
        try:
            latest = fetch_latest_release_tag()
        except Exception as error:
            _fail(f"could not resolve the newest published release ({type(error).__name__}: {error}).")
            return 1
    else:
        try:
            latest = (release_tree / "VERSION").read_text().strip()
        except OSError as error:
            _fail(f"cannot read the release tree's VERSION ({error}).")
            return 1

    current_key = version_sort_key(current)
    latest_key = version_sort_key(latest)
    if current_key is None or latest_key is None:
        _fail(f"unparseable release name (installed={current!r}, latest={latest!r}).")
        return 1
    if latest_key <= current_key and not adopt_drifted:
        print(f"MIRA {current} is up to date (newest published release: {latest}).")
        return 0

    tag = f"v{latest}"
    print(f"MIRA {latest} is available (installed: {current}) — updating in place.")
    with tempfile.TemporaryDirectory(prefix="mira-update-") as scratch:
        try:
            if release_tree is None:
                tarball = _download_release(latest, Path(scratch))
                release_tree = _extract_release(tarball, Path(scratch))
        except (OSError, ValueError, tarfile.TarError) as error:
            _fail(f"could not fetch the {latest} release ({type(error).__name__}: {error}).")
            return 1

        return _gate_and_hand_off(release_tree, latest, tag, rebuild_venvs=rebuild_venvs,
                                  adopt_drifted=adopt_drifted)
