# deploy/lib/custom_export.sh
# Custom-build preservation: before a code clear, every file the operator (or
# MIRA's own bash tool) changed relative to the last official install/update
# commit is copied out of the install tree, with a patch and a catalog.
# Source this file - do not execute directly.
#
# Requires: lib/output.sh, lib/services.sh (file_sha256), and
# lib/state_preserve.sh (list_cleared_files, is_preserved_name) sourced first
#
# The delta is computed from list_cleared_files (what the clear destroys)
# against the baseline tree, never from git's tracked/untracked view —
# .gitignore must not hide a file from the save.
#
# export_custom_build must run before the code clear: it reads the operator's
# files out of the live tree, so after the clear there is nothing left to
# compute the delta from. A failed export removes its partial directory and
# returns 1, and the caller is expected to stop with the install unchanged.
#
# A caller that sets CUSTOM_EXPORT_NEW_TREE (the incoming release tree) gets
# each saved file classified against it: absorbed, clean-apply, both-changed,
# upstream-deleted, or not-classified when the variable is unset. Paths under
# config/ or tools/ are tagged operator-only, because MIRA's own guardrail
# refuses agent edits there, so those restores end with the operator.

# Subjects deploy/lib/provenance.sh:official_commit_message writes, plus the
# pre-provenance forms (`install <tag>`, `install <tag> (local tree)`) the
# v2026.10.06 installer wrote. ui/tui/migrate.py:OFFICIAL_SUBJECT_RE is the
# Python twin — change both together.
MIRA_TAB="$(printf '\t')"

MIRA_OFFICIAL_SUBJECT_ERE='^(install|update) (v[0-9]+(\.[0-9]+)*(-[0-9]+(\.[0-9]+)*)?|nightly-[0-9a-f]{12})( @ [0-9a-f]{40}( local-dirty)?)?( \(local tree\))?$'

# Prints the SHA of the newest commit whose subject is an official install or
# update record. Returns 1 when the history holds none.
# Usage: official_baseline_commit <tree>
official_baseline_commit() {
    local log sha subject line
    log="$(git -C "$1" log --format='%H %s')" || return 1
    while IFS= read -r line; do
        sha="${line%% *}"
        subject="${line#* }"
        if [[ "$subject" =~ $MIRA_OFFICIAL_SUBJECT_ERE ]]; then
            printf '%s\n' "$sha"
            return 0
        fi
    done <<< "$log"
    echo "official_baseline_commit: ${1} has no official install or update commit" >&2
    return 1
}

# Bytes of a regular file (0 for a symlink — it is copied as a link).
_file_bytes() {
    if [ -L "$1" ]; then
        echo 0
    else
        wc -c < "$1" | tr -d '[:space:]'
    fi
}

# Free KiB on the volume holding a path.
_free_kib() {
    df -Pk "$1" | awk 'NR == 2 {print $4}'
}

# Builds the delta index and lists. Writes into <work>:
#   index     — a git index: baseline + every changed/added file - deletions
#   nested    — NUL list of files inside nested git repositories (status A;
#               git cannot stage them, so they are copied but not patched)
#   nested_roots — NUL list of those repositories' directories
#   delta_git — NUL-separated "<status>\t<path>" records git computed (A, M, D)
#   delta     — delta_git plus an A record per nested file
# Usage: _build_delta <tree> <baseline> <work>
_build_delta() {
    local tree="$1" baseline="$2" work="$3" path root roots=() nested status top
    local listed="${work}/listed" addable="${work}/addable" removed="${work}/removed"
    : > "${work}/nested"; : > "$addable"; : > "$removed"; : > "${work}/delta_git"; : > "${work}/delta"
    : > "${work}/nested_roots"
    list_cleared_files "$tree" > "$listed" || return 1

    # Outermost nested repositories: any listed path with a .git component.
    local found known
    while IFS= read -r -d '' path; do
        case "$path" in
            */.git/*) found="${path%%/.git/*}" ;;
            */.git) found="${path%/.git}" ;;
            *) continue ;;
        esac
        known="no"
        for root in ${roots[@]+"${roots[@]}"}; do
            [ "$root" = "$found" ] && { known="yes"; break; }
        done
        [ "$known" = "yes" ] || roots=(${roots[@]+"${roots[@]}"} "$found")
    done < "$listed"

    while IFS= read -r -d '' path; do
        nested="no"
        for root in ${roots[@]+"${roots[@]}"}; do
            case "$path" in "$root"/*) nested="yes"; break ;; esac
        done
        if [ "$nested" = "yes" ]; then
            printf '%s\0' "$path" >> "${work}/nested"
        else
            printf '%s\0' "$path" >> "$addable"
        fi
    done < "$listed"

    # A baseline path is deleted when its top-level name is not preserved and
    # nothing exists at it any more.
    git -C "$tree" ls-tree -r -z --name-only "$baseline" > "${work}/baseline" || return 1
    while IFS= read -r -d '' path; do
        top="${path%%/*}"
        is_preserved_name "$top" && continue
        if [ ! -e "${tree}/${path}" ] && [ ! -L "${tree}/${path}" ]; then
            printf '%s\0' "$path" >> "$removed"
        fi
    done < "${work}/baseline"

    GIT_INDEX_FILE="${work}/index" git -C "$tree" read-tree "$baseline" || return 1
    if [ -s "$addable" ]; then
        GIT_LITERAL_PATHSPECS=1 GIT_INDEX_FILE="${work}/index" git -C "$tree" add -f \
            --pathspec-from-file="$addable" --pathspec-file-nul || return 1
    fi
    if [ -s "$removed" ]; then
        GIT_INDEX_FILE="${work}/index" git -C "$tree" update-index --force-remove -z --stdin \
            < "$removed" || return 1
    fi

    GIT_INDEX_FILE="${work}/index" git -C "$tree" diff --cached --no-renames \
        --name-status -z "$baseline" > "${work}/diff" || return 1
    status=""
    while IFS= read -r -d '' path; do
        if [ -z "$status" ]; then
            status="$path"
            continue
        fi
        case "$status" in
            A|M|D) printf '%s\t%s\0' "$status" "$path" >> "${work}/delta_git" ;;
            T) printf 'M\t%s\0' "$path" >> "${work}/delta_git" ;;
            *) echo "_build_delta: unexpected diff status '${status}' for ${path}" >&2; return 1 ;;
        esac
        status=""
    done < "${work}/diff"
    cat "${work}/delta_git" > "${work}/delta"
    while IFS= read -r -d '' path; do
        printf 'A\t%s\0' "$path" >> "${work}/delta"
    done < "${work}/nested"
    for root in ${roots[@]+"${roots[@]}"}; do
        printf '%s\0' "$root" >> "${work}/nested_roots"
    done
}

# Copies the custom delta out of <tree> into <dest-root>/mira-custom-<stamp>/:
# full copies of every A/M file (relative paths, modes, symlinks as links),
# custom.patch (baseline -> current, binary-safe), RESTORE_MY_CUSTOM_BUILD.txt.
# Sets CUSTOM_EXPORT_DIR and CUSTOM_EXPORT_COUNT (files the operator changed —
# a nested repository's own .git internals are copied but not counted; 0 and
# no directory when the delta is empty). Every failure removes the partial directory and returns 1.
# The caller sets CUSTOM_EXPORT_REMEDY to the remedy the free-space refusal
# prints after "Free up space, then ": the updater passes "run: mira update",
# the installer passes "re-run deploy.sh" — the lib has no caller of its own.
# Usage: export_custom_build <tree> <baseline> <tag> <stamp> <dest-root>
export_custom_build() {
    local tree="$1" baseline="$2" tag="$3" stamp="$4" dest_root="$5"
    local work exp record status path bytes=0 need_kib free_kib count=0 copied=0
    CUSTOM_EXPORT_DIR=""
    CUSTOM_EXPORT_COUNT=0
    if [ ! -d "$dest_root" ] || [ ! -w "$dest_root" ]; then
        print_error "MIRA could not save a copy of the files you changed: ${dest_root} is not a writable folder."
        echo "export_custom_build: ${dest_root} is not a writable directory" >&2
        return 1
    fi

    # The delta index writes blobs into the install's .git; bound it by the
    # size of everything the clear would delete.
    work="$(mktemp -d "${TMPDIR:-/tmp}/mira-export.XXXXXX")" || return 1
    list_cleared_files "$tree" > "${work}/all" || { rm -rf "$work"; return 1; }
    while IFS= read -r -d '' path; do
        bytes=$((bytes + $(_file_bytes "${tree}/${path}")))
    done < "${work}/all"
    need_kib=$((bytes / 1024 + 1024))
    free_kib="$(_free_kib "${tree}/.git")"
    if [ "$free_kib" -lt "$need_kib" ]; then
        print_error "There is not enough free disk space to save a copy of the files you changed (needs ${need_kib} KB, has ${free_kib} KB at ${tree}/.git). Nothing was changed. Free up space, then ${CUSTOM_EXPORT_REMEDY}"
        rm -rf "$work"
        return 1
    fi

    if ! _build_delta "$tree" "$baseline" "$work"; then
        print_error "MIRA could not work out which files you changed in ${tree}."
        rm -rf "$work"
        return 1
    fi
    if [ ! -s "${work}/delta" ]; then
        rm -rf "$work"
        return 0
    fi

    bytes=0
    local shown=0
    while IFS= read -r -d '' record; do
        status="${record%%"$MIRA_TAB"*}"
        path="${record#*"$MIRA_TAB"}"
        case "$status" in
            A|M)
                count=$((count + 1))
                bytes=$((bytes + $(_file_bytes "${tree}/${path}")))
                case "$path" in */.git/*|*/.git) ;; *) shown=$((shown + 1)) ;; esac
                ;;
        esac
    done < "${work}/delta"
    # Copies plus a binary patch of the same bytes (base85 inflates ~25%).
    need_kib=$((bytes * 3 / 1024 + 1024))
    free_kib="$(_free_kib "$dest_root")"
    if [ "$free_kib" -lt "$need_kib" ]; then
        print_error "There is not enough free disk space to save a copy of the files you changed (needs ${need_kib} KB, has ${free_kib} KB at ${dest_root}). Nothing was changed. Free up space, then ${CUSTOM_EXPORT_REMEDY}"
        rm -rf "$work"
        return 1
    fi

    exp="${dest_root}/mira-custom-${stamp}"
    if ! mkdir "$exp"; then
        print_error "MIRA could not create ${exp} to save the files you changed."
        rm -rf "$work"
        return 1
    fi
    if ! _export_into "$tree" "$baseline" "$tag" "$work" "$exp" "$count"; then
        print_error "MIRA could not save a complete copy of the files you changed (see the message above)."
        rm -rf "$exp" "$work"
        return 1
    fi
    rm -rf "$work"
    CUSTOM_EXPORT_DIR="$exp"
    CUSTOM_EXPORT_COUNT="$shown"
}

# --- Classification against the incoming release ---------------------------
# The caller sets CUSTOM_EXPORT_NEW_TREE to the release tree being installed
# (deploy/update.sh: its first argument; deploy/python.sh: the tree the script
# itself runs from). Unset or missing: every file is tagged not-classified and
# the restore guidance in the catalog still applies.
_blob_sha256() {
    # sha256 of a blob piped on stdin, same tool fallback as file_sha256
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    else
        shasum -a 256 | awk '{print $1}'
    fi
}

# MIRA's bash guardrail refuses edits under config/ and the tools/ tree in
# every mode (they define what the tool may do), so those restores end with
# the operator's hands, not the agent's.
_is_agent_refused_path() {
    case "$1" in
        config/*|tools/*) return 0 ;;
        *) return 1 ;;
    esac
}

# Prints one classification tag for a delta file, computed against the
# incoming release tree: absorbed (already byte-identical upstream), clean-apply
# (the release never touched the file), both-changed (yours and the release's
# edits meet; expect a 3-way conflict), upstream-deleted (the release removed or
# redesigned the path; the release wins by default). Symlinks compare by target.
# Usage: _classify_delta_file <exp> <tree> <baseline> <status> <path>
_classify_delta_file() {
    local exp="$1" tree="$2" baseline="$3" status="$4" path="$5"
    local new custom_sha new_sha base_sha
    if [ -z "${CUSTOM_EXPORT_NEW_TREE:-}" ] || [ ! -d "${CUSTOM_EXPORT_NEW_TREE}" ]; then
        echo "not-classified"
        return 0
    fi
    new="${CUSTOM_EXPORT_NEW_TREE}/${path}"
    if [ ! -e "$new" ] && [ ! -L "$new" ]; then
        # A pure addition of yours the release never carried applies clean;
        # a modified file the release no longer has was redesigned away.
        if [ "$status" = "A" ]; then echo "clean-apply"; else echo "upstream-deleted"; fi
        return 0
    fi
    if [ -L "${exp}/${path}" ] || [ -L "$new" ]; then
        if [ -L "${exp}/${path}" ] && [ -L "$new" ] \
            && [ "$(readlink "${exp}/${path}")" = "$(readlink "$new")" ]; then
            echo "absorbed"
        else
            echo "both-changed"
        fi
        return 0
    fi
    custom_sha="$(file_sha256 "${exp}/${path}")"
    new_sha="$(file_sha256 "$new")"
    if [ "$custom_sha" = "$new_sha" ]; then
        echo "absorbed"
        return 0
    fi
    if [ "$status" = "A" ]; then
        echo "both-changed"
        return 0
    fi
    base_sha="$(git -C "$tree" show "${baseline}:${path}" 2>/dev/null | _blob_sha256)"
    if [ "$new_sha" = "$base_sha" ]; then
        echo "clean-apply"
    else
        echo "both-changed"
    fi
}

# Copies, verifies, and catalogs into an existing empty export directory.
_export_into() {
    local tree="$1" baseline="$2" tag="$3" work="$4" exp="$5" expected="$6"
    local record status path cls op copied=0 catalog="${exp}/RESTORE_MY_CUSTOM_BUILD.txt"
    while IFS= read -r -d '' record; do
        status="${record%%"$MIRA_TAB"*}"
        path="${record#*"$MIRA_TAB"}"
        case "$status" in A|M) ;; *) continue ;; esac
        case "$path" in
            */*) mkdir -p "${exp}/${path%/*}" || return 1 ;;
        esac
        cp -RPp "${tree}/${path}" "${exp}/${path}" || return 1
        if [ -L "${tree}/${path}" ]; then
            [ -L "${exp}/${path}" ] || { echo "export: ${path} lost its symlink" >&2; return 1; }
            [ "$(readlink "${tree}/${path}")" = "$(readlink "${exp}/${path}")" ] \
                || { echo "export: ${path} symlink target differs" >&2; return 1; }
        else
            [ "$(file_sha256 "${tree}/${path}")" = "$(file_sha256 "${exp}/${path}")" ] \
                || { echo "export: ${path} copy differs from the original" >&2; return 1; }
        fi
        copied=$((copied + 1))
    done < "${work}/delta"
    if [ "$copied" -ne "$expected" ]; then
        echo "export: copied ${copied} files, expected ${expected}" >&2
        return 1
    fi

    GIT_INDEX_FILE="${work}/index" git -C "$tree" diff --cached --binary "$baseline" \
        > "${exp}/custom.patch" || return 1

    {
        echo "Your MIRA install had changes of your own when it was updated to ${tag}."
        echo "Every changed or added file is copied in this folder, at the same path it had"
        echo "inside the install (${tree}). The update replaced MIRA's program files and"
        echo "kept your data; nothing in this folder is ever deleted by MIRA."
        echo ""
        echo "Restoring (written for a coding agent — there is no built-in restore command):"
        echo "  Re-apply these changes onto the NEW release rather than copying old files"
        echo "  over it: the new release may have changed the same files. Start from"
        echo "  custom.patch (git apply --3way --binary custom.patch, run in the install"
        echo "  tree), resolve conflicts against the new code, and use the full copies here"
        echo "  as the reference for what each file looked like before the update."
        echo "  A folder marked 'not in custom.patch' is a git repository of its own and is"
        echo "  restored by copying the folder back."
        echo "  Each changed or added file below carries a classification computed against"
        echo "  THIS release (the update had the incoming tree in hand):"
        echo "    absorbed          already byte-identical in the new release; nothing to do"
        echo "    clean-apply       the release never touched the file; expect the patch to apply"
        echo "    both-changed      your edit and the release's meet in one file; resolve for the"
        echo "                      new code — where the release redesigned the area, the release wins"
        echo "    upstream-deleted  the release removed or redesigned this path; restore the change"
        echo "                      only if it still fits the new design, drop the rest"
        echo "  Files marked operator-only live under config/ or the tools/ tree: they are"
        echo "  MIRA's own guardrail and tool sources, and a MIRA agent cannot edit them in"
        echo "  any mode — plan to apply those by hand."
        echo "  The restore does not need to be one-shot. Collaborating with the operator"
        echo "  is a valid and expected part of it, not a fallback: the agent may lack the"
        echo "  conversation that explains why the release changed an area, and a redesign"
        echo "  can delete or replace files a custom change touched. Apply the clear cases"
        echo "  (absorbed: nothing to do; clean-apply: apply) and bring the operator the"
        echo "  ambiguous ones — both-changed conflicts that do not resolve from the full"
        echo "  copies, upstream-deleted files whose change may or may not still fit the"
        echo "  new design. A restore that forces an old version back over a deliberate"
        echo "  redesign is worse than no restore; the operator knows which it is."
        echo ""
        echo "Baseline (the last official MIRA install/update before your changes):"
        echo "  $(git -C "$tree" log -1 --format='%s' "$baseline")"
        echo "  commit ${baseline}"
        echo ""
        echo "Files (status: A added, M modified, D deleted; classification: absorbed, clean-apply, both-changed, upstream-deleted, not-classified):"
        while IFS= read -r -d '' record; do
            status="${record%%"$MIRA_TAB"*}"
            path="${record#*"$MIRA_TAB"}"
            case "$status" in
                A|M)
                    cls="$(_classify_delta_file "$exp" "$tree" "$baseline" "$status" "$path")"
                    op=""
                    if _is_agent_refused_path "$path"; then op="  operator-only"; fi
                    if [ -L "${exp}/${path}" ]; then
                        echo "  ${status}  ${path}  ${cls}${op}  (symlink -> $(readlink "${exp}/${path}"))"
                    else
                        echo "  ${status}  ${path}  ${cls}${op}  sha256 $(file_sha256 "${exp}/${path}")"
                    fi
                    ;;
            esac
        done < "${work}/delta_git"
        while IFS= read -r -d '' path; do
            echo "  A  ${path}/  (a git repository of its own: copied in full, not in custom.patch)"
        done < "${work}/nested_roots"
        echo ""
        echo "Deleted files (the update brings these back; to delete one again, run the rm shown):"
        local any_deleted="no"
        while IFS= read -r -d '' record; do
            status="${record%%"$MIRA_TAB"*}"
            path="${record#*"$MIRA_TAB"}"
            if [ "$status" = "D" ]; then
                any_deleted="yes"
                printf '  D  %s    rm -- %q\n' "$path" "${tree}/${path}"
            fi
        done < "${work}/delta"
        [ "$any_deleted" = "yes" ] || echo "  (none)"
        echo ""
        echo "Your own commits since the baseline:"
        if [ -n "$(git -C "$tree" log --format='%H %s' "${baseline}..HEAD")" ]; then
            git -C "$tree" log --format='  %H %s' "${baseline}..HEAD"
        else
            echo "  (none)"
        fi
    } > "$catalog" || return 1
}

# Prints the plain-English notice that a custom build was saved. Files the
# operator deleted are named explicitly: no copy of a deleted file exists
# (the update brings it back), so the "safe at" sentence covers only the
# changed and added files. The re-delete rm commands live in the catalog's
# machine-readable "Deleted files" section ("D  <path>    rm -- <install>"),
# the form MIRA's own bash tool reads to re-apply them after the update.
# Usage: print_custom_alert <export-dir> <count>
print_custom_alert() {
    local catalog="${1}/RESTORE_MY_CUSTOM_BUILD.txt" deleted names
    deleted="$(sed -n 's/^  D  \(.*\)    rm -- .*$/\1/p' "$catalog" 2>/dev/null)"
    names="${deleted//$'\n'/, }"
    echo ""
    if [ -n "$deleted" ] && [ "$2" -gt 0 ]; then
        print_warning "MIRA found $2 files you changed in your install, and files you deleted that this update brings back: ${names}. A copy of every changed or added file is safe at $1; the deleted files have no copy — RESTORE_MY_CUSTOM_BUILD.txt in that folder lists each one with the rm command that deletes it again. This update replaces MIRA's program files and keeps your data. To put your changes back, open that file, or ask MIRA to put them back."
    elif [ -n "$deleted" ]; then
        print_warning "MIRA found files you deleted in your install, which this update brings back: ${names}. Nothing was copied for them — RESTORE_MY_CUSTOM_BUILD.txt in the folder at $1 lists each one with the rm command that deletes it again. This update replaces MIRA's program files and keeps your data. To delete them again after the update, open that file, or ask MIRA to delete them again."
    else
        print_warning "MIRA found $2 files you changed in your install. A copy of every one is safe at $1. This update replaces MIRA's program files and keeps your data. To put your changes back, open RESTORE_MY_CUSTOM_BUILD.txt in that folder, or ask MIRA to put them back."
    fi
    echo ""
}

# Name<TAB>version of every distribution in a venv, normalized and sorted.
_venv_packages() {
    "$1/bin/python3" -c '
import importlib.metadata as metadata
for dist in metadata.distributions():
    name = (dist.metadata["Name"] or "").strip().lower().replace("_", "-").replace(".", "-")
    if name and name not in ("pip", "setuptools", "wheel"):
        print(f"{name}\t{dist.version}")
' | LC_ALL=C sort -u
}

# Appends a "## Venv package drift" section to <catalog> describing what
# replacing <live-venv> with <new-venv> changes: packages only the live venv
# has (installed by hand, lost by the swap) and packages whose version moves.
# Runs only for a venv the update will swap, before the swap. An empty drift
# still writes the section, saying so.
# Usage: record_venv_drift <label> <live-venv> <new-venv> <catalog>
record_venv_drift() {
    local label="$1" live="$2" new="$3" catalog="$4" work tab
    tab="$MIRA_TAB"
    work="$(mktemp -d "${TMPDIR:-/tmp}/mira-drift.XXXXXX")" || return 1
    if ! _venv_packages "$live" > "${work}/live" || ! _venv_packages "$new" > "${work}/new"; then
        rm -rf "$work"
        return 1
    fi
    LC_ALL=C join -t "$tab" -v 1 "${work}/live" "${work}/new" > "${work}/lost"
    LC_ALL=C join -t "$tab" "${work}/live" "${work}/new" \
        | awk -F '\t' '$2 != $3 {print $1 "\t" $2 "\t" $3}' > "${work}/moved"
    {
        echo ""
        echo "## Venv package drift (${label} venv: ${live})"
        if [ ! -s "${work}/lost" ] && [ ! -s "${work}/moved" ]; then
            echo "The rebuilt venv has exactly the packages the old one had."
        else
            if [ -s "${work}/lost" ]; then
                echo "Installed in the old venv but not in the new one (reinstall with pip if you need them):"
                awk -F '\t' '{print "  " $1 "==" $2}' "${work}/lost"
            fi
            if [ -s "${work}/moved" ]; then
                echo "Version changed (old -> new):"
                awk -F '\t' '{print "  " $1 " " $2 " -> " $3}' "${work}/moved"
            fi
        fi
    } >> "$catalog"
    local rc=$?
    VENV_DRIFT_LOST="$(wc -l < "${work}/lost" | tr -d '[:space:]')"
    rm -rf "$work"
    return "$rc"
}
