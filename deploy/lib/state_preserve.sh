# deploy/lib/state_preserve.sh
# The one definition of host-local state inside an install tree: what a code
# clear keeps, what git never tracks, and what a clear destroys.
# Source this file - do not execute directly. No dependencies.
#
# Every slash-anchored entry of write_install_exclude must name a member of
# MIRA_PRESERVE_NAMES: tools/implementations/bash_tool.py:_load_app_tree reads
# the exclude's literal entries as the names it refuses to delete, so the code
# clear must keep exactly those too.

# find(1) -name patterns, matched against top-level names only.
MIRA_PRESERVE_NAMES=(venv .env data logs scratch .git run.sh 'venv*' 'tui-venv*')

# Returns 0 when a top-level name is host-local state the clear keeps.
# Usage: is_preserved_name <name>
is_preserved_name() {
    local pattern
    for pattern in "${MIRA_PRESERVE_NAMES[@]}"; do
        # shellcheck disable=SC2254
        case "$1" in $pattern) return 0 ;; esac
    done
    return 1
}

# Writes <app-dir>/.git/info/exclude: everything git must never track or
# show as a change. State-carrying names are slash-anchored (top level only,
# matching bash_tool's top-level protection); regenerable junk is unanchored.
# Usage: write_install_exclude <app-dir>
write_install_exclude() {
    mkdir -p "$1/.git/info" || return 1
    cat > "$1/.git/info/exclude" <<'EXCLUDE'
# Written by deploy/lib/state_preserve.sh:write_install_exclude (install and
# every update). Never tracked, never deletable through bash_tool. Edit that
# function, not this file. Slash-anchored entries are protected at the TOP
# LEVEL only, matching bash_tool; unanchored entries are ignored at any depth.
/data/
/venv/
/.env
/logs/
/scratch/
/run.sh
/venv*/
/tui-venv*/
__pycache__/
*.pyc
.DS_Store
EXCLUDE
}

# Prints the non-preserved top-level names of <app-dir>, NUL-terminated.
_cleared_top_level_names() {
    local dir="$1" entry name
    find "$dir" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' entry; do
        name="${entry##*/}"
        is_preserved_name "$name" || printf '%s\0' "$name"
    done
}

# Deletes every top-level entry of <app-dir> except the preserve set.
# --sudo runs the deletion under sudo; --print writes the find argv
# NUL-separated to stdout and executes nothing.
# Usage: clear_code_preserving_state <app-dir> [--sudo] [--print]
clear_code_preserving_state() {
    local dir="$1" use_sudo="no" print_only="no" pattern
    shift
    while [ $# -gt 0 ]; do
        case "$1" in
            --sudo) use_sudo="yes" ;;
            --print) print_only="yes" ;;
            *) echo "clear_code_preserving_state: unknown flag $1" >&2; return 1 ;;
        esac
        shift
    done
    local argv=(find "$dir" -mindepth 1 -maxdepth 1)
    [ "$use_sudo" = "yes" ] && argv=(sudo "${argv[@]}")
    for pattern in "${MIRA_PRESERVE_NAMES[@]}"; do
        argv=("${argv[@]}" ! -name "$pattern")
    done
    argv=("${argv[@]}" -exec rm -rf {} +)
    if [ "$print_only" = "yes" ]; then
        printf '%s\0' "${argv[@]}"
        return 0
    fi
    "${argv[@]}"
}

# Moves every non-preserved top-level entry of <app-dir> into <aside-dir>
# (created; must not exist). Same-volume renames: instant, and lossless for
# anything a delete-then-restore could mangle (nested repositories, modes,
# special files). restore_code_aside undoes it exactly.
# Usage: move_code_aside <app-dir> <aside-dir>
move_code_aside() {
    local dir="$1" aside="$2" name
    if [ -e "$aside" ]; then
        echo "move_code_aside: ${aside} already exists" >&2
        return 1
    fi
    mkdir "$aside" || return 1
    _cleared_top_level_names "$dir" > "${aside}.names" || return 1
    while IFS= read -r -d '' name; do
        mv "${dir}/${name}" "${aside}/${name}" || return 1
    done < "${aside}.names"
    rm -f "${aside}.names"
}

# Moves every entry of <aside-dir> back into <app-dir> and removes the empty
# aside. --clear-first deletes the code now in <app-dir> (preserve set kept)
# before the move — required once a release has been overlaid; without it a
# partially completed move_code_aside is undone without touching the entries
# that never moved.
# Usage: restore_code_aside <app-dir> <aside-dir> [--clear-first]
restore_code_aside() {
    local dir="$1" aside="$2" entry
    if [ ! -d "$aside" ]; then
        echo "restore_code_aside: ${aside} does not exist" >&2
        return 1
    fi
    if [ "${3:-}" = "--clear-first" ]; then
        clear_code_preserving_state "$dir" || return 1
    fi
    find "$aside" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' entry; do
        mv "$entry" "${dir}/" || exit 1
    done || return 1
    rmdir "$aside" && rm -f "${aside}.names"
}

# Prints, NUL-separated and sorted, every regular file and symlink a code
# clear of <app-dir> would delete (paths relative to <app-dir>), git-ignored
# files included. Omits only regenerable output: __pycache__/ directories,
# *.pyc files, and .DS_Store. The single definition of "what the clear
# destroys" — the custom-build export reads it.
# Usage: list_cleared_files <app-dir>
list_cleared_files() {
    local dir="$1" names=() name path
    # "./" keeps a top-level name that starts with "-" from reading as a find
    # option; it is stripped again before printing.
    while IFS= read -r -d '' name; do
        names=(${names[@]+"${names[@]}"} "./${name}")
    done < <(_cleared_top_level_names "$dir")
    [ "${#names[@]}" -gt 0 ] || return 0
    (
        cd "$dir" || exit 1
        find "${names[@]}" \( -type d -name __pycache__ -prune \) -o \
            \( \( -type f -o -type l \) ! -name '*.pyc' ! -name .DS_Store -print0 \)
    ) | while IFS= read -r -d '' path; do
        printf '%s\0' "${path#./}"
    done | LC_ALL=C sort -z
}
