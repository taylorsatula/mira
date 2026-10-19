# deploy/lib/venv_stamp.sh
# Venv build stamps: a record of the inputs a venv was built from, so an
# update rebuilds a venv only when those inputs changed.
# Source this file - do not execute directly.
#
# Requires: lib/services.sh sourced first (file_sha256)
#
# A stamp describes build inputs, never live state: packages the operator
# pip-installed by hand are not in it, and a matching stamp keeps the venv
# as-is (those packages survive). `deploy/update.sh --rebuild-venvs` is the
# escape hatch when the live venv itself is wrong.
#
# write_venv_stamp must be the caller's last statement: a stamp written before
# the extras install claims extras the venv does not have, and every later
# update then matches it and skips the install that would have added them.

VENV_STAMP_FILE=".mira-build-stamp"

# Prints the extras CSV in canonical order (sorted, comma-joined, no spaces)
# from any order or spacing of torch / sentence-transformers / playwright.
# Usage: canonical_extras <csv>
canonical_extras() {
    local joined
    joined="$(printf '%s\n' "$1" | tr ',' '\n' | tr -d ' \t' | grep -v '^$' | LC_ALL=C sort -u | tr '\n' ',' || true)"
    printf '%s\n' "${joined%,}"
}

# Prints "<major>.<minor>.<micro>" of a Python interpreter.
# Usage: python_version_of <python>
python_version_of() {
    "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])'
}

# Prints the base interpreter a venv runs on (sys.base_prefix/bin/python3).
# Usage: venv_base_python <venv>
venv_base_python() {
    local base
    base="$("$1/bin/python3" -c 'import sys; print(sys.base_prefix)')" || return 1
    printf '%s/bin/python3\n' "$base"
}

# Writes <venv>/.mira-build-stamp (mode 0644).
# Usage: write_venv_stamp <venv> <reqfile> <base-python> <extras-csv>
write_venv_stamp() {
    local venv="$1" reqfile="$2" base="$3" extras req_sha base_version
    extras="$(canonical_extras "$4")"
    req_sha="$(file_sha256 "$reqfile")" || return 1
    base_version="$(python_version_of "$base")" || return 1
    {
        printf 'base_python=%s\n' "$base"
        printf 'base_python_version=%s\n' "$base_version"
        printf 'requirements_sha256=%s\n' "$req_sha"
        printf 'requirements_path=%s\n' "$reqfile"
        printf 'extras=%s\n' "$extras"
        printf 'built_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${venv}/${VENV_STAMP_FILE}" || return 1
    chmod 0644 "${venv}/${VENV_STAMP_FILE}"
}

# Reads one key from a stamp file.
_venv_stamp_value() {
    sed -n "s/^$2=//p" "$1" | head -1
}

# Compares a venv's stamp against the inputs a build would use now.
# Returns 0 on a match (base_python_version + requirements_sha256 + extras),
# 1 on a mismatch or when the venv's interpreter no longer runs, 2 when the
# stamp is absent. Sets VENV_STAMP_REASON to the operator-readable cause on
# every nonzero return.
# Usage: venv_stamp_matches <venv> <reqfile> <base-python> <extras-csv>
venv_stamp_matches() {
    local venv="$1" reqfile="$2" base="$3" extras stamp
    extras="$(canonical_extras "$4")"
    stamp="${venv}/${VENV_STAMP_FILE}"
    VENV_STAMP_REASON=""
    if [ ! -f "$stamp" ]; then
        VENV_STAMP_REASON="stamp absent"
        return 2
    fi
    if ! "${venv}/bin/python3" -c 'import sys' > /dev/null 2>&1; then
        VENV_STAMP_REASON="interpreter dead"
        return 1
    fi
    if [ "$(_venv_stamp_value "$stamp" requirements_sha256)" != "$(file_sha256 "$reqfile")" ]; then
        VENV_STAMP_REASON="requirements changed"
        return 1
    fi
    if [ "$(_venv_stamp_value "$stamp" extras)" != "$extras" ]; then
        VENV_STAMP_REASON="extras changed"
        return 1
    fi
    if [ "$(_venv_stamp_value "$stamp" base_python_version)" != "$(python_version_of "$base")" ]; then
        VENV_STAMP_REASON="base python changed"
        return 1
    fi
    return 0
}
