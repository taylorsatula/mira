# deploy/lib/provenance.sh
# Release provenance: building the release artifact and naming the exact
# commit a tree came from.
# Source this file - do not execute directly.
#
# Requires: lib/services.sh (run_bounded) and lib/repo_identity.sh
# (mira_github_repo) sourced first

# Prints the 40-hex SHA recorded in <tree>/RELEASE_COMMIT. Returns 1 for a
# pre-provenance tree (file absent, empty, or not exactly 40 lowercase hex).
# Usage: read_release_commit <tree>
read_release_commit() {
    local file="$1/RELEASE_COMMIT" sha
    if [ ! -f "$file" ]; then
        echo "read_release_commit: pre-provenance tree (no ${file})" >&2
        return 1
    fi
    sha="$(tr -d '[:space:]' < "$file")"
    case "$sha" in
        *[!0-9a-f]*|"")
            echo "read_release_commit: pre-provenance tree (${file} is not a commit SHA)" >&2
            return 1 ;;
    esac
    if [ "${#sha}" -ne 40 ]; then
        echo "read_release_commit: pre-provenance tree (${file} holds ${#sha} hex digits, expected 40)" >&2
        return 1
    fi
    printf '%s\n' "$sha"
}

# Prints the commit a published tag points at, asked of GitHub. Release tags
# may be lightweight (no peeled ^{} entry) or annotated (the peeled entry is
# the commit); the peeled entry wins when both are listed. Used only for a
# tree that carries no RELEASE_COMMIT — one a pre-mechanism updater fetched
# from GitHub's auto-archive. Returns 1 when the tag is unknown or the remote
# is unreachable within the bound.
# Usage: resolve_release_sha <tag>
resolve_release_sha() {
    local tag="$1" repo listing sha
    repo="$(mira_github_repo)" || return 1
    if ! listing="$(run_bounded 30 git ls-remote "https://github.com/${repo}.git" \
            "refs/tags/${tag}" "refs/tags/${tag}^{}" 2>/dev/null)"; then
        echo "resolve_release_sha: could not reach github.com/${repo} to resolve ${tag}" >&2
        return 1
    fi
    sha="$(printf '%s\n' "$listing" | awk -v peeled="refs/tags/${tag}^{}" '$2 == peeled {print $1}')"
    [ -n "$sha" ] || sha="$(printf '%s\n' "$listing" | awk -v ref="refs/tags/${tag}" '$2 == ref {print $1}')"
    if [ -z "$sha" ]; then
        echo "resolve_release_sha: github.com/${repo} has no tag ${tag}" >&2
        return 1
    fi
    printf '%s\n' "$sha"
}

# Prints the checkout's HEAD SHA, suffixed " local-dirty" when the working
# tree has uncommitted or untracked changes (the installed bytes then differ
# from the named commit).
# Usage: resolve_local_sha <tree>
resolve_local_sha() {
    local tree="$1" sha
    sha="$(git -C "$tree" rev-parse HEAD)" || return 1
    if [ -n "$(git -C "$tree" status --porcelain --untracked-files=all)" ]; then
        printf '%s local-dirty\n' "$sha"
    else
        printf '%s\n' "$sha"
    fi
}

# Prints the install repo's commit subject for an official install or update.
# custom_export.sh:official_baseline_commit recognizes exactly these shapes
# (and ui/tui/migrate.py:OFFICIAL_SUBJECT_RE mirrors the pattern).
# Usage: official_commit_message <install|update> <label> <sha>
official_commit_message() {
    local kind="$1" label="$2" sha="$3"
    case "$kind" in
        install|update) ;;
        *) echo "official_commit_message: kind must be install or update, got '${kind}'" >&2; return 1 ;;
    esac
    printf '%s %s @ %s\n' "$kind" "$label" "$sha"
}

# Builds <dest-dir>/mira-<version>.tar.gz from <tag> in the repository this
# library lives in: git archive, RELEASE_COMMIT injected, the step-chain lint
# and the release-identity checks run against the extracted tree. Any failure
# produces no asset and returns 1 naming the failed check.
# Usage: build_release_artifact <tag> <dest-dir>
build_release_artifact() {
    local tag="$1" dest="$2" root version scratch tree sha repo installer_repo release_tag rc
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    version="${tag#v}"
    if [ "$version" = "$tag" ]; then
        echo "build_release_artifact: tag '${tag}' must start with v" >&2
        return 1
    fi
    if ! sha="$(git -C "$root" rev-parse --verify --quiet "${tag}^{commit}")"; then
        echo "build_release_artifact: ${root} has no tag ${tag}" >&2
        return 1
    fi
    if [ -n "$(git -C "$root" ls-tree --name-only "$tag" -- RELEASE_COMMIT)" ]; then
        echo "build_release_artifact: ${tag} tracks RELEASE_COMMIT — it is injected at build time and must never be committed" >&2
        return 1
    fi
    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
        echo "build_release_artifact: needs Python 3.12+ as python3 on PATH (found: $(python3 --version 2>&1))" >&2
        return 1
    fi
    mkdir -p "$dest" || return 1
    scratch="$(mktemp -d "${TMPDIR:-/tmp}/mira-release.XXXXXX")" || return 1
    tree="${scratch}/mira-${version}"

    rc=0
    git -C "$root" archive --format=tar.gz --prefix="mira-${version}/" "$tag" \
        > "${scratch}/archive.tar.gz" || rc=1
    [ "$rc" -eq 0 ] && { tar -xzf "${scratch}/archive.tar.gz" -C "$scratch" || rc=1; }
    [ "$rc" -eq 0 ] && { printf '%s\n' "$sha" > "${tree}/RELEASE_COMMIT" || rc=1; }
    if [ "$rc" -ne 0 ]; then
        echo "build_release_artifact: git archive of ${tag} failed" >&2
        rm -rf "$scratch"
        return 1
    fi

    # install.sh and deploy.sh's standalone bootstrap spell the repository
    # literally (each runs before any checkout exists); both must equal
    # deploy/GITHUB_REPO.
    repo="$(mira_github_repo)" || { rm -rf "$scratch"; return 1; }
    installer_repo="$(sed -n 's/^REPO="\(.*\)"$/\1/p' "${tree}/install.sh")"
    if [ "$installer_repo" != "$repo" ]; then
        echo "build_release_artifact: install.sh REPO='${installer_repo}' differs from deploy/GITHUB_REPO '${repo}'" >&2
        rm -rf "$scratch"
        return 1
    fi
    if ! grep -qF "https://github.com/${repo}.git \"\$CLONE_DIR\"" "${tree}/deploy/deploy.sh"; then
        echo "build_release_artifact: deploy/deploy.sh's bootstrap clone does not use github.com/${repo}.git" >&2
        rm -rf "$scratch"
        return 1
    fi
    # RELEASE_TAG picks the release fresh installs download (python.sh builds
    # the release-asset URL from it), so it must equal the tag being built —
    # a cut that moved VERSION but left this constant behind would otherwise
    # pass every gate and install the previous release under the new tag.
    release_tag="$(sed -n 's/^RELEASE_TAG="\(.*\)"$/\1/p' "${tree}/deploy/deploy.sh")"
    if [ "$release_tag" != "$tag" ]; then
        echo "build_release_artifact: deploy/deploy.sh RELEASE_TAG='${release_tag}' differs from the tag being built '${tag}'" >&2
        rm -rf "$scratch"
        return 1
    fi

    if [ -f "${tree}/scripts/migrate.py" ]; then
        if ! (cd "$tree" && PYTHONDONTWRITEBYTECODE=1 python3 -m scripts.migrate lint) >&2; then
            echo "build_release_artifact: the database step lint failed for ${tag} (scripts.migrate lint)" >&2
            rm -rf "$scratch"
            return 1
        fi
    fi

    if ! (cd "$tree" && PYTHONDONTWRITEBYTECODE=1 python3 -m utils.release_identity verify "$root" "$tag" "$tree") >&2; then
        echo "build_release_artifact: release identity check failed for ${tag} (see the MISMATCH line above)" >&2
        rm -rf "$scratch"
        return 1
    fi

    if ! tar -C "$scratch" -czf "${dest}/mira-${version}.tar.gz" "mira-${version}"; then
        echo "build_release_artifact: could not write ${dest}/mira-${version}.tar.gz" >&2
        rm -f "${dest}/mira-${version}.tar.gz"
        rm -rf "$scratch"
        return 1
    fi
    rm -rf "$scratch"
    printf '%s\n' "${dest}/mira-${version}.tar.gz"
}
