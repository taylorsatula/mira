"""
``mira update`` — update a deployed MIRA install in place.

The decision half of the update: resolve the newest published release (one
mechanism with the server's daily check — ``utils/release_identity.py``),
compare it against the installed ``VERSION``, fetch the release tarball, and
gate on breaking releases. The machine half — service stop, venv rebuild,
code swap, restart, health-poll — is ``deploy/update.sh`` in the *new* tree,
executed as a subprocess: like ``install.sh``, the newest release's own code
does the work, so conventions never lag behind the release being installed.

A release is breaking when its tree carries ``BREAKING.md`` at the repo root
(the contract is documented in ``deploy/RELEASE.md``). Breaking releases are
never updated automatically: ``mira update`` prints the file and the manual
path (reinstall via ``install.sh`` — the old database is renamed aside and
Vault credentials are preserved — then ask MIRA to bring its history forward
itself via its bash tool).

Environment (both this module and ``deploy/update.sh`` honor the same names;
they exist so verification probes can run the real flow against a throwaway
install root):

- ``MIRA_APP_DIR``   the install root (default ``/opt/mira/app``)
- ``MIRA_TUI_VENV``  the TUI client venv (default ``/opt/mira/tui-venv``)

Exit codes: 0 up to date or updated; 1 any failure or a blocked breaking
release. Never touches the install on a blocked or failed update — the
destructive work is entirely inside ``deploy/update.sh``, which snapshots
everything it replaces.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from utils.release_identity import (
    TAG_TARBALL_URL,
    fetch_latest_release_tag,
    get_current_version,
    version_sort_key,
)

DOWNLOAD_TIMEOUT_SECONDS = 120


def _fail(message: str) -> None:
    print(f"mira update: {message}", file=sys.stderr)


def _download_release(tag: str, destination: Path) -> Path:
    """
    Download the release tarball for ``tag`` into ``destination``.

    Raises:
        OSError: On any transport failure, including the download bound.
    """
    url = TAG_TARBALL_URL.format(tag=tag)
    tarball = destination / f"{tag}.tar.gz"
    request = urllib.request.Request(
        url, headers={"User-Agent": "mira-update"}
    )
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        tarball.write_bytes(response.read())
    return tarball


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


def _print_breaking_block(tag: str, release_tree: Path) -> None:
    notes = (release_tree / "BREAKING.md").read_text()
    print(f"MIRA {tag} is a BREAKING release — the automatic in-place update is refused.")
    print()
    print(notes.strip())
    print()
    print("To take a breaking release manually:")
    print("  1. Re-run the installer for the new release (your old database is kept,")
    print("     renamed aside automatically, and Vault credentials are preserved):")
    print("       curl -fsSL https://raw.githubusercontent.com/taylorsatula/mira/main/install.sh | bash")
    print("  2. Ask MIRA to bring its history forward: it imports its own data from the")
    print("     kept database with its bash tool — see deploy/HOW_TO_MIGRATE_OLD_INSTALLS.txt")
    print("     in the new install.")


def run_update(release_tree: Path | None = None) -> int:
    """
    Resolve, gate, and hand off to the release's own ``deploy/update.sh``.

    Nothing on the installed machine is touched until ``deploy/update.sh``
    starts; a refused breaking release or any pre-handoff failure leaves the
    install exactly as it was.

    ``release_tree`` is the verification seam: a real, extracted release tree
    on disk, used to exercise the compare/gate/handoff flow against real
    files when the release in question is not published on GitHub yet (the
    live path resolves and downloads it). The tree's own ``VERSION`` is the
    release identity; every later step is identical.
    """
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
    if latest_key <= current_key:
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

        if (release_tree / "BREAKING.md").is_file():
            _print_breaking_block(latest, release_tree)
            return 1

        updater = release_tree / "deploy" / "update.sh"
        if not updater.is_file():
            _fail(f"release {latest} carries no deploy/update.sh — it predates in-place update.")
            return 1
        # The release's own updater runs the machine half; env carries the
        # install roots (MIRA_APP_DIR / MIRA_TUI_VENV) so a non-default root
        # set by the caller flows through unchanged.
        result = subprocess.run(
            ["bash", str(updater), str(release_tree), tag],
            env=os.environ.copy(),
        )
        return result.returncode
