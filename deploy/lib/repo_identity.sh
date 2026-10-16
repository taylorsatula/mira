# deploy/lib/repo_identity.sh
# The GitHub repository spelling every deploy fetch site uses.
# Source this file - do not execute directly. No dependencies.

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
