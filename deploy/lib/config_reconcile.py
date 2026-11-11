"""Reconcile an operator's config.py against a new release's config.py.

`mira update` replaces the code tree, but config.py is the operator's durable
configuration surface: the values an operator chose must survive the update
while the release's schema (fields, validators, descriptions, structure) wins.
This module performs that reconciliation as a three-way merge over the literal
`Field(default=...)` values, keyed by (class, field):

    base    the pristine release config.py the operator's edits were made
            against (the stored baseline)
    ours    the operator's current config.py
    theirs  the new release's config.py

Drift rules, per field:
    ours == base                     -> theirs (operator did not touch it)
    ours != base, theirs == base     -> ours   (operator edit, release unchanged)
    ours != base, theirs != base,
        ours != theirs               -> ours + WARNING (both changed; operator wins)
    field absent from theirs         -> dropped + WARNING
    field new in theirs              -> release default
    non-literal default (factory)    -> release-owned, never carried

Container literals (list/tuple/dict/set of literals, e.g. a CORS allowlist) are
carried like scalars; only non-literal defaults (names, calls, factories) are
release-owned.

With no baseline (an install's first update under this scheme) the merge is
two-way: any operator value differing from the release is carried, with one
warning that release-default drift cannot be distinguished. Structural edits
to config.py (anything but literal defaults) are release-owned and not carried.

The merge is generated from parsed AST sources, never string-patched. A file
that does not parse fails the whole reconciliation (exit 2) so the updater
aborts before swapping any code.

Stdlib only; requires Python 3.9+ for ast.unparse.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def _literal_value(node: ast.AST) -> object:
    """The Python value of a literal default node, or raise ValueError.

    Accepts anything ast.literal_eval accepts: scalars (int, float, str, bool,
    None, negative numbers) and container literals (list, tuple, dict, set) whose
    elements are themselves literals. Names, calls, and factories are not literal
    defaults and are release-owned.
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError) as error:
        raise ValueError("not a literal default") from error


def _default_node(value: ast.AST) -> ast.AST | None:
    """The default-value node of a field assignment, or None.

    Handles `name: T = Field(default=X, ...)` and a bare `name: T = X`.
    Returns the node whose source span a carried value replaces.
    """
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "Field":
            return None
        for keyword in value.keywords:
            if keyword.arg == "default":
                return keyword.value
        # Pydantic's Field also accepts the default as the first positional arg.
        return value.args[0] if value.args else None
    return value


def extract_defaults(source: str) -> dict[tuple[str, str], tuple[ast.AST, object]]:
    """Map (class, field) -> (default node, literal value) for literal defaults.

    Non-literal defaults are omitted: they are release-owned and never carried.
    """
    tree = ast.parse(source)
    defaults: dict[tuple[str, str], tuple[ast.AST, object]] = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for stmt in cls.body:
            if isinstance(stmt, ast.AnnAssign):
                target, value = stmt.target, stmt.value
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target, value = stmt.targets[0], stmt.value
            else:
                continue
            if not isinstance(target, ast.Name) or value is None:
                continue
            node = _default_node(value)
            if node is None:
                continue
            try:
                value_lit = _literal_value(node)
            except ValueError:
                continue
            defaults[(cls.name, target.id)] = (node, value_lit)
    return defaults


def _line_starts(data: bytes) -> list[int]:
    """Absolute byte offset of the start of each newline-delimited line.

    Scans only b'\n' (not str.splitlines, which also splits on \f, \u2028, etc.
    and would misalign with the AST's lineno). Assumes \n or \r\n line endings.
    """
    starts = [0]
    for index, byte in enumerate(data):
        if byte == 0x0A:
            starts.append(index + 1)
    return starts


def _span(offsets: list[int], node: ast.AST) -> tuple[int, int]:
    # ast col_offset/end_col_offset are UTF-8 BYTE offsets within the line, so
    # the splice must operate on bytes, not str characters: a non-ASCII char
    # earlier on the line shifts a character-based span and corrupts the source.
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def reconcile(
    base_source: str | None,
    ours_source: str,
    theirs_source: str,
) -> tuple[str, list[str]]:
    """Return (merged config.py source, warnings).

    Raises SyntaxError if any supplied file does not parse.
    """
    ours = extract_defaults(ours_source)
    theirs = extract_defaults(theirs_source)
    base = extract_defaults(base_source) if base_source is not None else None
    warnings: list[str] = []

    if base is None:
        warnings.append(
            "no baseline: carrying every operator value that differs from the "
            "release; release-default drift cannot be distinguished on this "
            "first update"
        )

    for key in ours:
        if key not in theirs:
            warnings.append(
                f"dropped operator value for {key[0]}.{key[1]}: the release "
                "removed this field"
            )

    theirs_bytes = theirs_source.encode("utf-8")
    offsets = _line_starts(theirs_bytes)
    replacements: list[tuple[int, int, bytes]] = []
    for key, (theirs_node, theirs_value) in theirs.items():
        if key not in ours:
            continue
        ours_node, ours_value = ours[key]
        if ours_value == theirs_value:
            continue
        if base is not None and key in base:
            base_value = base[key][1]
            if ours_value == base_value:
                continue  # operator did not touch it; release evolution wins
            if theirs_value != base_value:
                warnings.append(
                    f"{key[0]}.{key[1]}: operator and release both changed the "
                    f"default (operator={ours_value!r}, release={theirs_value!r}); "
                    "keeping the operator value"
                )
        elif base is not None:
            warnings.append(
                f"{key[0]}.{key[1]}: no baseline entry; keeping the operator "
                f"value {ours_value!r} over the release {theirs_value!r}"
            )
        start, end = _span(offsets, theirs_node)
        replacements.append((start, end, ast.unparse(ours_node).encode("utf-8")))

    merged = theirs_bytes
    for start, end, text in sorted(replacements, key=lambda r: r[0], reverse=True):
        merged = merged[:start] + text + merged[end:]
    return merged.decode("utf-8"), warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=None, help="pristine baseline config.py")
    parser.add_argument("--ours", type=Path, required=True, help="operator's current config.py")
    parser.add_argument("--theirs", type=Path, required=True, help="new release config.py")
    parser.add_argument("--out", type=Path, required=True, help="where to write the merged config.py")
    args = parser.parse_args(argv)

    try:
        ours_source = args.ours.read_text(encoding="utf-8")
        theirs_source = args.theirs.read_text(encoding="utf-8")
        base_source = (
            args.base.read_text(encoding="utf-8")
            if args.base is not None and args.base.exists()
            else None
        )
    except OSError as error:
        print(f"config reconcile: cannot read input: {error}", file=sys.stderr)
        return 2

    try:
        merged, warnings = reconcile(base_source, ours_source, theirs_source)
    except SyntaxError as error:
        print(f"config reconcile: a config.py does not parse: {error}", file=sys.stderr)
        return 2

    try:
        args.out.write_text(merged, encoding="utf-8")
    except OSError as error:
        print(f"config reconcile: cannot write output: {error}", file=sys.stderr)
        return 2

    for warning in warnings:
        print(f"config reconcile: WARNING: {warning}", file=sys.stderr)
    print(f"config reconcile: merged {args.out} ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())