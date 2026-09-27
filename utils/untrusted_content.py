"""
Untrusted-content boundary: the sanctioned path for text sourced outside the
system (fetched pages, email, third-party API responses, federated messages,
user-supplied files) entering any model context.

Two entry points:

- `wrap_untrusted(content, source)` — structural boundary. Pure string
  transform, never raises, no network. Mandatory wherever external text
  enters a model context.
- `screen_untrusted(content, source)` — semantic screen, then the same wrap.
  Raises `InjectionRejected` (a `ValueError`) when the screen judges the
  content a manipulation attempt; infrastructure failures propagate
  (fail closed — unscreened content never passes as screened).

The screen runs these steps:

1. Reveal. Decode text the judge could not otherwise read — Unicode tag
   characters (invisible to people, readable by models), letter-spaced runs,
   base64, hex, percent-encoding, HTML numeric entities — and append it after
   the content under a header, so a payload cannot hide inside an encoding.
2. Judge (tier 1). Content plus revealed text, split into overlapping
   chunks; each chunk is judged against every question in
   `SCREEN_QUESTIONS` (question text: config/prompts/injection_screen/).
   The highest probability decides: at or below `PASS_CEILING` passes, at
   or above `REJECT_FLOOR` rejects, between escalates. The first chunk
   reaching `REJECT_FLOOR` ends the scan.
3. Escalate (tier 2, escalated content only). The chunk that scored highest
   goes to an LLM on `_ESCALATION_ROUTE` with the questions that made it
   uncertain; the LLM answers one verdict tag.
4. Wrap. `wrap_untrusted`.
"""

import base64
import binascii
import html
import logging
import re
import threading
import urllib.parse
from dataclasses import dataclass
from typing import Literal, Mapping

from clients.llm_provider import LLMProvider, get_llm_provider
from clients.systemone_client import NoulQuestion, SystemOneClient, get_systemone_client
from config.prompts import load_prompt

logger = logging.getLogger(__name__)

# One file per question in config/prompts/injection_screen/<name>.txt; the name
# is also the answer key in every System One response and verdict.
SCREEN_QUESTIONS: tuple[str, ...] = (
    "overrides_instructions",
    "requests_action",
    "requests_disclosure",
    "requests_deception",
    "claims_false_authority",
    "exfiltration_channel",
)

# Tier-1 band over the highest question probability. Measured on the corpus in
# tests/protected/injection_defense_probe.py (2026-09-26, final wording): the
# weakest attack scored 0.58 and the strongest benign case 0.91, so passing at
# <= 0.5 and rejecting at >= 0.95 decided no case wrongly; 18 of 110 cases fell
# between and escalate. Move either edge only with that battery's numbers.
PASS_CEILING = 0.5
REJECT_FLOOR = 0.95

# Route judging escalated content (model per deployment in model_configs).
_ESCALATION_ROUTE = "fast"

# Verdict format: the escalation prompt's answer instructions and the parser
# are both generated from these.
_VERDICT_REJECT = "reject"
_VERDICT_PASS = "pass"
_VERDICT_FORMAT = f"<verdict>{_VERDICT_REJECT}</verdict> or <verdict>{_VERDICT_PASS}</verdict>"
_VERDICT_RE = re.compile(rf"<verdict>\s*({_VERDICT_REJECT}|{_VERDICT_PASS})\s*</verdict>")

Disposition = Literal["pass", "escalate", "reject"]

# Chunk geometry. Witnessed: an attack sentence buried mid-way through ~7,100
# characters of filler scored 1.00 in a single djev request (2026-09-26), so an
# 8,000-character state does not dilute a single-sentence payload. The overlap
# keeps a payload split across a boundary whole in at least one chunk.
_CHUNK_CHARS = 8000
_CHUNK_OVERLAP_CHARS = 500

# Unicode tag block: renders as nothing, yet models read it as ASCII
# ("ASCII smuggling"). Code point minus 0xE0000 is the ASCII character.
_TAG_CHARS_RE = re.compile("[\U000E0000-\U000E007F]+")

# Four or more single letters/digits separated by single spaces:
# "i g n o r e". Word gaps in letter-spaced text are wider runs of spaces.
_LETTER_SPACED_RE = re.compile(r"(?<!\w)(?:[^\W_] ){3,}[^\W_](?!\w)")

_BASE64_RE = re.compile(r"[A-Za-z0-9+/_-]{16,}={0,2}")
_HEX_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}){8,}\b")
_ESCAPED_HEX_RE = re.compile(r"(?:\\x[0-9A-Fa-f]{2}){4,}")
_PERCENT_RE = re.compile(r"(?:%[0-9A-Fa-f]{2}){4,}")
_NUMERIC_ENTITY_RE = re.compile(r"(?:&#(?:[0-9]{1,7}|[xX][0-9A-Fa-f]{1,6});){4,}")

# Decoded bytes count as hidden text only when they read as text: at least
# this many characters, nearly all printable. Keys, hashes, and binary blobs
# decode to non-printable noise and are dropped.
_MIN_REVEALED_CHARS = 6
_MIN_PRINTABLE_RATIO = 0.9

# Loaded at import so a missing file fails at startup, keeping wrap_untrusted
# pure and non-raising at call time.
_WRAPPER_DIRECTIVE = load_prompt("untrusted_content_directive.txt")


@dataclass(frozen=True)
class ScreenVerdict:
    """Tier-1 outcome of one screen.

    `signals` holds, per question, the highest probability seen across the
    chunks screened; `decisive_chunk` is the judged chunk (content plus any
    revealed text) that produced the highest single probability — the text an
    escalation judges. `revealed` names the encodings whose decoded text was
    appended for the judge.
    """

    disposition: Disposition
    signals: Mapping[str, float]
    chunks_screened: int
    revealed: tuple[str, ...]
    decisive_chunk: str


class InjectionRejected(ValueError):
    """Content judged a manipulation attempt; carries the tier-1 verdict."""

    def __init__(self, source: str, verdict: ScreenVerdict, escalated: bool):
        suspected = ", ".join(
            f"{name}={p:.2f}" for name, p in verdict.signals.items() if p > PASS_CEILING
        )
        tier = "on escalation" if escalated else "by the fast screen"
        super().__init__(f"Content from {source} rejected {tier}: {suspected}")
        self.source = source
        self.verdict = verdict
        self.escalated = escalated


class EscalationResponseError(RuntimeError):
    """The escalation LLM did not answer with exactly one verdict tag."""


def parse_escalation_verdict(text: str) -> bool:
    """True when the escalation answer rejects; raises unless exactly one verdict tag."""
    verdicts = _VERDICT_RE.findall(text)
    if len(verdicts) != 1:
        raise EscalationResponseError(
            f"Escalation answer must hold exactly one verdict tag, found {len(verdicts)}: {text[:300]!r}"
        )
    return verdicts[0] == _VERDICT_REJECT


def _as_text(raw: bytes) -> str | None:
    """Decoded bytes as text, or None when they do not read as text."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if len(text) < _MIN_REVEALED_CHARS:
        return None
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return text if printable / len(text) >= _MIN_PRINTABLE_RATIO else None


def _decode_base64(token: str) -> str | None:
    token = token.rstrip("=").replace("-", "+").replace("_", "/")
    try:
        raw = base64.b64decode(token + "=" * (-len(token) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None
    return _as_text(raw)


def _reveal_hidden(content: str) -> list[tuple[str, str]]:
    """(encoding, decoded text) for every hidden or encoded run in content."""
    found: list[tuple[str, str]] = []

    for run in _TAG_CHARS_RE.findall(content):
        text = "".join(chr(ord(ch) - 0xE0000) for ch in run if 0xE0020 <= ord(ch) <= 0xE007E)
        if text.strip():
            found.append(("unicode_tag_characters", text))

    spaced = [run.replace(" ", "") for run in _LETTER_SPACED_RE.findall(content)]
    if spaced:
        found.append(("letter_spacing", " ".join(spaced)))

    for token in _HEX_RE.findall(content):
        text = _as_text(bytes.fromhex(token))
        if text:
            found.append(("hex", text))
    for run in _ESCAPED_HEX_RE.findall(content):
        text = _as_text(bytes.fromhex(run.replace("\\x", "")))
        if text:
            found.append(("hex", text))
    for token in _BASE64_RE.findall(content):
        text = _decode_base64(token)
        if text:
            found.append(("base64", text))
    for run in _PERCENT_RE.findall(content):
        text = _as_text(urllib.parse.unquote_to_bytes(run))
        if text:
            found.append(("percent_encoding", text))
    for run in _NUMERIC_ENTITY_RE.findall(content):
        text = html.unescape(run)
        if len(text) >= _MIN_REVEALED_CHARS:
            found.append(("html_entities", text))

    unique: list[tuple[str, str]] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique


def _split_chunks(text: str) -> list[str]:
    if len(text) <= _CHUNK_CHARS:
        return [text]
    chunks = []
    step = _CHUNK_CHARS - _CHUNK_OVERLAP_CHARS
    for start in range(0, len(text), step):
        chunks.append(text[start:start + _CHUNK_CHARS])
        if start + _CHUNK_CHARS >= len(text):
            break
    return chunks


class InjectionScreen:
    """Two-tier injection screen: System One band, LLM for the uncertain middle.

    Holds construction-time wiring only (clients, loaded prompt text), so the
    process-wide instance is shared safely across threads and users.
    """

    def __init__(self, systemone: SystemOneClient, llm: LLMProvider):
        self._systemone = systemone
        self._llm = llm
        self._questions = {
            name: NoulQuestion(load_prompt(f"injection_screen/{name}.txt"))
            for name in SCREEN_QUESTIONS
        }
        self._revealed_header = load_prompt("injection_screen/revealed_text_header.txt")
        self._escalation_system = load_prompt("injection_screen/escalation_system.txt").replace(
            "{verdict_format}", _VERDICT_FORMAT
        )
        self._escalation_user = load_prompt("injection_screen/escalation_user.txt")

    def assess(self, content: str) -> ScreenVerdict:
        """Tier-1 disposition; never calls the LLM. Infrastructure errors propagate."""
        if not content.strip():
            return ScreenVerdict(
                disposition="pass", signals={}, chunks_screened=0, revealed=(), decisive_chunk=""
            )

        revealed = _reveal_hidden(content)
        judged = content
        if revealed:
            lines = "\n".join(f"({encoding}) {text}" for encoding, text in revealed)
            judged = f"{content}\n\n{self._revealed_header}\n{lines}"

        signals = {name: 0.0 for name in SCREEN_QUESTIONS}
        chunks_screened = 0
        decisive_chunk, decisive_top = "", -1.0
        for chunk in _split_chunks(judged):
            answers = self._systemone.ask_nouls(chunk, self._questions)
            chunks_screened += 1
            for name, probability in answers.items():
                signals[name] = max(signals[name], probability)
            chunk_top = max(answers.values())
            if chunk_top > decisive_top:
                decisive_chunk, decisive_top = chunk, chunk_top
            if chunk_top >= REJECT_FLOOR:
                break

        top = max(signals.values())
        disposition: Disposition = (
            "reject" if top >= REJECT_FLOOR else "pass" if top <= PASS_CEILING else "escalate"
        )
        return ScreenVerdict(
            disposition=disposition,
            signals=signals,
            chunks_screened=chunks_screened,
            revealed=tuple(dict.fromkeys(encoding for encoding, _ in revealed)),
            decisive_chunk=decisive_chunk,
        )

    def escalation_messages(self, verdict: ScreenVerdict, source: str) -> list[dict[str, str]]:
        """System and user messages asking the LLM to settle an escalated verdict."""
        suspected = sorted(
            ((p, name) for name, p in verdict.signals.items() if p > PASS_CEILING), reverse=True
        )
        suspicions = "\n".join(f"- {p:.2f}: {self._questions[name].instructions}" for p, name in suspected)
        # Content is spliced last so text inside it can never reach another placeholder.
        user = self._escalation_user.replace("{suspicions}", suspicions).replace(
            "{content}", wrap_untrusted(verdict.decisive_chunk, source)
        )
        return [
            {"role": "system", "content": self._escalation_system},
            {"role": "user", "content": user},
        ]

    def screen(self, content: str, source: str) -> str:
        """Wrapped content, or InjectionRejected when either tier rejects it."""
        verdict = self.assess(content)
        escalated = verdict.disposition == "escalate"
        if escalated:
            response = self._llm.generate_response(
                messages=self.escalation_messages(verdict, source),
                model_config=_ESCALATION_ROUTE,
            )
            rejected = parse_escalation_verdict(self._llm.extract_text_content(response))
        else:
            rejected = verdict.disposition == "reject"
        if rejected:
            error = InjectionRejected(source, verdict, escalated)
            logger.warning("%s (revealed encodings: %s)", error, list(verdict.revealed) or "none")
            raise error
        return wrap_untrusted(content, source)


_screen: InjectionScreen | None = None
_screen_lock = threading.Lock()


def get_injection_screen() -> InjectionScreen:
    """Process-wide screen over `get_systemone_client()` and `get_llm_provider()`;
    construction failures propagate."""
    global _screen
    if _screen is None:
        with _screen_lock:
            if _screen is None:
                _screen = InjectionScreen(get_systemone_client(), get_llm_provider())
    return _screen


def screen_untrusted(content: str, source: str) -> str:
    """Screen external content, then wrap it. Raises InjectionRejected on rejection."""
    return get_injection_screen().screen(content, source)


def _escape_markup(text: str) -> str:
    # Ampersand first, so a pre-encoded "&lt;" stays literal instead of
    # unescaping back into a forged tag; quotes so no attribute can be broken.
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def wrap_untrusted(content: str, source: str) -> str:
    """Structural boundary for external text entering a model context.

    Removes Unicode tag characters (an invisible channel models still read),
    escapes markup so the content cannot forge or close the boundary or break
    the source label, and wraps it in a labeled `<untrusted_content>` region
    with a treat-as-data directive. Pure; never raises. Falsy content is
    returned unchanged.
    """
    if not content:
        return content
    escaped = _escape_markup(_TAG_CHARS_RE.sub("", content))
    return (
        f'<untrusted_content source="{_escape_markup(source)}">\n'
        f"{_WRAPPER_DIRECTIVE}\n"
        f"{escaped}\n"
        "</untrusted_content>"
    )
