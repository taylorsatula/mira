# deploy/lib/repo_identity.sh
# The GitHub repository spelling every deploy fetch site uses.
# Source this file - do not execute directly. No dependencies.
#
# One reader, no fallback literal: deploy/GITHUB_REPO is resolved relative to
# this file (so a checkout, a release tree, and /opt/mira/app all work), and a
# missing or empty file returns 1 rather than printing a default — a fallback
# spelling would keep fetching from a repository the project no longer
# publishes under, and the failure would look like a network problem.
#
# utils/release_identity.py:GITHUB_REPO is the Python reader of the same file.
# Two sites spell the repository literally because each runs before any
# checkout exists: root install.sh's REPO= and deploy/deploy.sh's bootstrap
# clone; provenance.sh:build_release_artifact refuses a release where either
# differs from the file.

# Prints the owner/name from deploy/GITHUB_REPO (resolved relative to this
# file, so it works from a checkout, a release tree, and /opt/mira/app alike).
# Usage: mira_github_repo
mira_github_repo() {
    local file repo
    file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/GITHUB_REPO"
    if [ ! -f "$file" ]; then
        echo "mira_github_repo: ${file} is missing" >&2
        return 1
    fi
    repo="$(tr -d '[:space:]' < "$file")"
    if [ -z "$repo" ]; then
        echo "mira_github_repo: ${file} is empty" >&2
        return 1
    fi
    printf '%s\n' "$repo"
}
