"""Bench the persona generator: many samples per preset, each graded by the witness battery.

    python -m scripts.persona_bench --endpoint URL --model NAME [options]

Writes one report per run so a human can read candidates side by side and cherry-pick
the best parts of several into the authored preset prose. This is an authoring tool,
not a runtime path: nothing in the app imports it, and the install never calls a model
to produce a persona.

The battery below is the mechanical half of the craft contract in
``config/prompts/persona_voice_system.txt`` — the greppable invariants, nothing that
requires judgment. A candidate that passes is *shape*-valid, not good; the whitewashing
test, the deletion test, contract duplication, and announced interiority stay with the
reader. Two checks are heuristics and are named as such where they are implemented: the
base-form-verb opener is a curated blocklist and cannot be exhaustive, and "final word
carries content" is a function-word list.

    --check-only FILE      grade an existing block and print its failures, no network
    --preset NAME|all      bench named presets from config/persona_voices.py (default: all)
    --words "W1,W2,W3,W4"  bench an arbitrary four-word set instead of a preset
    --samples N            samples per set (default 4)
    --temperature T        sampling temperature, default 1.0; variety is the point here
    --seed-base N          first seed; sample i uses seed-base + i (servers may ignore it)
    --out DIR              where the report and per-sample blocks land (default: bench/<stamp>)
    --show                 also print every block to stdout as it arrives

Exit codes: 0 every requested sample returned a response; 1 a request failed; 2 bad arguments.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = _REPO_ROOT / "config" / "prompts" / "persona_voice_system.txt"

# A reasoning model spends budget before it emits anything: at 400 a thinking model
# returned finish_reason=length with empty content and the whole block still inside
# reasoning_content. The block itself is ~110 words; the rest is headroom to think.
_MAX_TOKENS = 2048
_TIMEOUT_SECONDS = 300  # overridable per run: a reasoning model at xhigh can take minutes

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PERSONA_BLOCK = re.compile(r"<mira:persona>(.*?)</mira:persona>", re.DOTALL)
_SECOND_PERSON = re.compile(r"\b(you|your|yours|you're|yourself)\b", re.IGNORECASE)
_THIRD_PERSON_OTHER = re.compile(r"\b(they|them|their|theirs|the user|your user)\b", re.IGNORECASE)

# Heuristic, documented as such: a curated blocklist of base-form verbs. It cannot be
# exhaustive, so a clean result means "no opener we know of", not "no imperative".
_BASE_FORM_OPENERS = frozenset(
    """
    keep say take lose use avoid never always make be do ask write put remember note ensure treat
    spend report stay meet follow reach hear read play build hold skip run try name greet mean stand
    carry bring catch talk let set give find choose lead watch commit answer show start stop give
    explain describe offer propose prefer tend work think begin continue leave enter open close
    """.split()
)

# Heuristic, documented as such: a function-word list, so "carries content" means "is not
# one of these", not a part-of-speech analysis.
_FUNCTION_WORDS = frozenset(
    """
    the of and to a an in for with it is are was were be been being at on or by as that than so no
    not you your yours yourself this these those there here they them their theirs we us our he she
    his her its but if then when while about into over under from up down out off again once very
    just only also too much more most some any all each every other such same own
    """.split()
)


def _normalise(value: str) -> str:
    """Casefold and treat a hyphen as a space so 'Absent Minded' matches 'absent-minded'."""
    return re.sub(r"[^a-z0-9 ]+", " ", value.casefold().replace("-", " ")).strip()


def sentences(block: str) -> list[str]:
    """Split a block into sentences. An abbreviation splits one; that is visible in the count."""
    return [part.strip() for part in _SENTENCE_SPLIT.split(block.strip()) if part.strip()]


def witness(block: str, attributes: list[str]) -> list[str]:
    """Return every mechanical invariant the block violates, in a stable order.

    An empty list means shape-valid. It does not mean good: the deletion test, the
    whitewashing test, contract duplication, and announced interiority are the reader's.
    """
    failures: list[str] = []
    parts = sentences(block)

    if len(parts) != 6:
        failures.append(f"sentences: {len(parts)}, want 6")
    if "\u2014" in block:
        failures.append("em dash present")
    if "\u2013" in block:
        failures.append("en dash present")
    if re.search(r"\bshould\b", block, re.IGNORECASE):
        failures.append('"should" present')
    if re.search(r"\bbecause\b", block, re.IGNORECASE):
        failures.append('"because" present')
    contrast = len(re.findall(r"\brather than\b", block, re.IGNORECASE))
    if contrast > 2:
        failures.append(f'"rather than" x{contrast}, max 2')
    if "{" in block or "}" in block:
        failures.append("curly braces present")

    normalised_block = _normalise(block)
    for attribute in attributes:
        needle = _normalise(attribute)
        if not needle:
            continue
        if needle not in normalised_block:
            failures.append(f"attribute missing: {attribute}")
            continue
        for index, sentence in enumerate(parts[:4], start=1):
            if needle in _normalise(sentence):
                failures.append(f"attribute {attribute!r} appears in sentence {index}, earliest 5")

    for index, sentence in enumerate(parts, start=1):
        if not _SECOND_PERSON.search(sentence):
            failures.append(f"sentence {index} has no second-person token")
        if _THIRD_PERSON_OTHER.search(sentence):
            failures.append(f"sentence {index} refers to the operator in third person")
        opener = re.sub(r"[^a-z]", "", sentence.split()[0].casefold()) if sentence.split() else ""
        if opener in _BASE_FORM_OPENERS:
            failures.append(f"sentence {index} opens with a base-form verb {opener!r}")

    if parts:
        tail = re.findall(r"[A-Za-z']+", parts[-1])
        if tail and tail[-1].casefold() in _FUNCTION_WORDS:
            failures.append(f"sentence 6 ends on function word {tail[-1]!r}")

    return failures


def user_message(attributes: list[tuple[str, str, str]]) -> str:
    """The operator's choice, as the generator receives it: axis, word, and what it must do."""
    lines = ["The operator chose these four attributes:", ""]
    for axis, word, gloss in attributes:
        lines.append(f"- {axis}: {word} — {gloss}")
    lines.append("")
    lines.append("Write the persona block.")
    return "\n".join(lines)


def request(
    endpoint: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user: str,
    temperature: float,
    seed: int | None,
    max_tokens: int,
    reasoning_effort: str | None,
    timeout: int,
) -> str:
    """One chat-completions call. Raises on transport or protocol failure, never invents a block."""
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    if reasoning_effort:
        # A reasoning model will otherwise spend the whole budget in reasoning_content
        # and return empty content with finish_reason=length. Not every endpoint honors
        # this field; one that ignores it falls back to starving, which the request
        # error reports by name.
        payload["reasoning_effort"] = reasoning_effort
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(body, timeout=timeout) as response:
        parsed = json.load(response)

    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"response carried no choices: {str(parsed)[:200]}")
    first = choices[0]
    content = first.get("message", {}).get("content") if isinstance(first, dict) else None
    if content is None and isinstance(first, dict):
        content = first.get("text")
    if not isinstance(content, str) or not content.strip():
        # Name the starvation case explicitly: an empty content with a full reasoning
        # payload means the budget ran out before the block was written, which is a
        # different fix (raise --max-tokens) from an endpoint returning garbage.
        reasoning = ""
        if isinstance(first, dict):
            reasoning = str(first.get("message", {}).get("reasoning_content") or "")
        finish = first.get("finish_reason") if isinstance(first, dict) else None
        raise ValueError(
            f"no content (finish_reason={finish!r}, {len(reasoning)} chars of reasoning); "
            "a reasoning model that ran out of budget looks like this — raise --max-tokens"
        )
    return content


def extract_block(raw: str) -> str:
    """Prefer the tagged block; fall back to the whole reply so a sloppy model is still graded."""
    match = _PERSONA_BLOCK.search(raw)
    return match.group(1).strip() if match else raw.strip()


def _write_report(out_dir: Path, rows: list[dict]) -> Path:
    report = out_dir / "report.md"
    lines = ["# Persona bench", ""]
    passed = sum(1 for row in rows if not row["failures"])
    lines.append(f"{passed}/{len(rows)} samples passed the battery.")
    lines.append("")
    for row in rows:
        verdict = "PASS" if not row["failures"] else "FAIL"
        lines.append(f"## {row['name']} · sample {row['sample']} · {verdict}")
        lines.append("")
        lines.append(f"attributes: {', '.join(row['attributes'])}  ·  seed: {row['seed']}")
        lines.append("")
        if row["failures"]:
            lines.extend(f"- {failure}" for failure in row["failures"])
            lines.append("")
        lines.append(row["block"] or "(no block)")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def load_presets():
    """Load ``config/persona_voices.py`` by path, without importing the ``config`` package.

    ``config/__init__.py`` imports the app's config singleton and the tool registry, which
    need the application venv. This tool is an authoring aid that must run from a bare
    interpreter against an endpoint, so it reads the one stdlib-only module directly.
    """
    import importlib.util

    path = _REPO_ROOT / "config" / "persona_voices.py"
    spec = importlib.util.spec_from_file_location("persona_voices", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load presets from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.persona_bench",
        description="Sample the persona generator per preset and grade each block.",
    )
    parser.add_argument("--endpoint", help="chat-completions URL, e.g. http://host:3090/v1/chat/completions")
    parser.add_argument("--model", help="model id as the endpoint knows it")
    parser.add_argument("--api-key", default="", help="bearer token; omit for an endpoint that takes none")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT, help=f"generator prompt (default: {DEFAULT_PROMPT.name})")
    parser.add_argument("--preset", action="append", default=[], help="preset name from config/persona_voices.py, or 'all'")
    parser.add_argument("--words", action="append", default=[], help=f"arbitrary four words, comma separated (samples named 'custom-N')")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=_MAX_TOKENS, help=f"generation budget (default {_MAX_TOKENS})")
    parser.add_argument("--reasoning-effort", default=None, help="none|minimal|low|medium|high|xhigh; a thinking model needs this lowered or it will starve its own output")
    parser.add_argument("--timeout", type=int, default=_TIMEOUT_SECONDS, help=f"per-request seconds (default {_TIMEOUT_SECONDS})")
    parser.add_argument("--combos", type=Path, default=None, help="file of 'name: Word, Word, Word, Word' lines to bench alongside presets")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--check-only", type=Path, default=None, help="grade an existing block, no network")
    args = parser.parse_args(argv)

    # Loaded by path, not imported: see load_presets.
    sys.path.insert(0, str(_REPO_ROOT))
    presets = load_presets()
    PERSONA_VOICES, get = presets.PERSONA_VOICES, presets.get

    if args.check_only is not None:
        block = extract_block(args.check_only.read_text(encoding="utf-8"))
        attributes = [word for row in (args.words or []) for word in (w.strip() for w in row.split(",")) if word]
        failures = witness(block, attributes)
        print(f"{len(sentences(block))} sentences, {len(block.split())} words")
        print("PASS" if not failures else "\n".join(f"FAIL {failure}" for failure in failures))
        return 0 if not failures else 1

    if not args.endpoint or not args.model:
        parser.error("--endpoint and --model are required unless --check-only is given")
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    try:
        system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    except OSError as error:
        parser.error(f"cannot read the generator prompt {args.prompt}: {error}")

    sets: list[tuple[str, list[tuple[str, str, str]]]] = []
    names = args.preset or ([] if (args.words or args.combos) else ["all"])
    if "all" in names:
        names = sorted(PERSONA_VOICES)
    for name in names:
        voice = get(name)
        sets.append((voice.name, [(a.axis, a.word, a.gloss) for a in voice.attributes]))
    if args.combos is not None:
        try:
            combo_lines = args.combos.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            parser.error(f"cannot read --combos {args.combos}: {error}")
        for line in combo_lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, word_list = line.partition(":")
            words = [word.strip() for word in word_list.split(",") if word.strip()]
            if len(words) != 4:
                parser.error(f"combos line needs four words after a colon: {line!r}")
            sets.append((name.strip() or f"combo-{len(sets)+1}",
                         [(f"Attribute {i}", word, "") for i, word in enumerate(words, start=1)]))
    for index, row in enumerate(args.words, start=1):
        words = [word.strip() for word in row.split(",") if word.strip()]
        if len(words) != 4:
            parser.error(f"--words needs exactly four comma-separated words, got {len(words)}")
        sets.append((f"custom-{index}", [(f"Attribute {i}", word, "") for i, word in enumerate(words, start=1)]))

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or (_REPO_ROOT / "bench" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    transport_failures = 0
    for name, attributes in sets:
        for sample in range(1, args.samples + 1):
            seed = None if args.seed_base is None else args.seed_base + sample - 1
            try:
                raw = request(
                    args.endpoint,
                    args.model,
                    args.api_key,
                    system_prompt,
                    user_message(attributes),
                    args.temperature,
                    seed,
                    args.max_tokens,
                    args.reasoning_effort,
                    args.timeout,
                )
            except (urllib.error.URLError, ValueError, json.JSONDecodeError, TimeoutError, OSError) as error:
                transport_failures += 1
                print(f"{name} #{sample}: request failed: {error}", file=sys.stderr)
                continue
            block = extract_block(raw)
            failures = witness(block, [word for _, word, _ in attributes])
            row = {
                "name": name,
                "sample": sample,
                "seed": seed,
                "attributes": [word for _, word, _ in attributes],
                "failures": failures,
                "block": block,
                "raw": raw,
            }
            rows.append(row)
            (out_dir / f"{name}--{sample:02d}.md").write_text(block + "\n", encoding="utf-8")
            (out_dir / f"{name}--{sample:02d}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            status = "PASS" if not failures else "FAIL: " + "; ".join(failures)
            print(f"{name} #{sample}: {status}")
            if args.show and not failures:
                print(f"\n{block}\n")

    if not rows:
        print("no samples returned", file=sys.stderr)
        return 1

    report = _write_report(out_dir, rows)
    passed = sum(1 for row in rows if not row["failures"])
    print(f"\n{passed}/{len(rows)} passed · report: {report}")
    if transport_failures:
        print(f"{transport_failures} request(s) failed", file=sys.stderr)
    return 0 if transport_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
