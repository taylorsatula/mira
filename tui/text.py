"""Display text for the MIRA TUI: escape sanitizing and the streaming reply
filter.

Pure and synchronous: stdlib only, no I/O. ``assistant_delta`` text is raw
model output (think blocks, ``mira:`` namespaced tags, an ephemeral
``[5:47pm]`` timestamp); ``ReplyStream`` turns it into committed display
lines plus a display-safe in-progress tail.

Contract points the callers rely on:

- A committed line is never retracted, so blank lines are held back until a
  later non-blank line proves they are interior (no leading blanks, runs
  collapse to one, no trailing blanks).
- Text that could still turn out to be (part of) a tag is held back, never
  shown and corrected. ``flush`` / ``finish`` never reveal what streaming
  held back: an unclosed think or mira tag, everything after it in that
  entry, and a trailing partial tag fragment (including a lone ``<``) are
  dropped, because text inside an unclosed tag is metadata by construction.
  A reply held back entirely leaves ``has_text`` False, which is the
  caller's cue to fall back to the server's tag-parsed final response.
- ``feed`` sanitizes each delta on its own, so an escape sequence split
  across two deltas can leak its printable tail (``[2J``) but never ESC.
- Output is independent of how the text was split into deltas, except for
  that escape-split leak. ``display_lines`` is the one-shot path through the
  same machinery.

Tag literals are built by concatenation: literal think/mira tag strings
written through agent tool payloads get HTML-entity-mangled on disk.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# sanitize
# ---------------------------------------------------------------------------

_ESC_SEQUENCE_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # CSI
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC, BEL or ST terminated
    r"|\x1b[PX^_][^\x1b]*\x1b\\"  # DCS / SOS / PM / APC, ST terminated
    r"|\x1b[\s\S]?"  # any other ESC + one char, or a lone ESC
)
# C0 except \t \n, DEL, C1. ESC is included so nothing can survive the
# sequence pass; lone \r is dropped here (CRLF is normalized first).
_CONTROL_RE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    """Return ``text`` safe to print: CRLF normalized, every escape sequence
    and control character removed except ``\\n`` and ``\\t``. Never contains
    ESC."""
    text = text.replace("\r\n", "\n")
    text = _ESC_SEQUENCE_RE.sub("", text)
    return _CONTROL_RE.sub("", text)


# ---------------------------------------------------------------------------
# Tag filters
# ---------------------------------------------------------------------------

_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "</" + "think" + ">"
_MIRA_OPEN_PREFIX = "<" + "mira:"
_MIRA_CLOSE_PREFIX = "</" + "mira:"

_THINK_BLOCK_PAT = "<" + "think" + ">.*?</" + "think" + ">"
# Paired (name backreference, case-insensitive) or self-closing.
_MIRA_TAG_PAT = (
    "<" + r"mira:([^>/\s]+)(?:\s[^>]*)?>[\s\S]*?</" + r"mira:\1>"
    "|<" + r"mira:[^>]*/>"
)
# One pass over both kinds: spans found while streaming are exactly the
# spans the final filter removes, so chunked commits equal the one-shot.
_BLOCK_RE = re.compile(
    _THINK_BLOCK_PAT + "|" + _MIRA_TAG_PAT, re.IGNORECASE | re.DOTALL
)
# An opening tag that is not self-closing: unclosed when no block span
# starts at it.
_OPEN_RE = re.compile(
    "<" + "think" + ">|<" + r"mira:[^>/\s]+(?:\s[^>]*)?>", re.IGNORECASE
)

_TIMESTAMP_RE = re.compile(r"^\[\d{1,2}:\d{2}[ap]m\]\s*", re.IGNORECASE)
# Every proper prefix of a timestamp that could still complete.
_TIMESTAMP_PREFIX_RE = re.compile(
    r"^\[(?:\d{1,2}(?::(?:\d(?:\d(?:[ap]m?)?)?)?)?)?$", re.IGNORECASE
)


def _filter(text: str) -> str:
    return _BLOCK_RE.sub("", text)


def _covered(spans: list[tuple[int, int]], pos: int) -> bool:
    return any(start <= pos < end for start, end in spans)


def _is_fragment(tail: str) -> bool:
    """True when ``tail`` (starts with ``<``, contains no ``>``) is, or could
    grow into, a think tag or a mira tag."""
    tail = tail.lower()
    return (
        _THINK_OPEN.startswith(tail)
        or _THINK_CLOSE.startswith(tail)
        or _MIRA_OPEN_PREFIX.startswith(tail)
        or _MIRA_CLOSE_PREFIX.startswith(tail)
        or tail.startswith(_MIRA_OPEN_PREFIX)
        or tail.startswith(_MIRA_CLOSE_PREFIX)
    )


def _scan(raw: str) -> tuple[list[tuple[int, int]], int]:
    """Return the complete tag-block spans of ``raw`` and ``hold``, the index
    where text stops being safe to show (``len(raw)`` when nothing is held)."""
    spans = [m.span() for m in _BLOCK_RE.finditer(raw)]
    hold = len(raw)
    for match in _OPEN_RE.finditer(raw):
        if not _covered(spans, match.start()):
            hold = match.start()
            break
    # A trailing fragment can only start after the last ">" (it has none).
    pos = raw.find("<", raw.rfind(">") + 1)
    while pos != -1 and pos < hold:
        if _is_fragment(raw[pos:]):
            hold = pos
            break
        pos = raw.find("<", pos + 1)
    return spans, hold


# Leading-timestamp state of one entry.
_LEAD_UNDECIDED = 0  # no display text seen yet
_LEAD_STRIPPING = 1  # timestamp stripped; eating the whitespace after it
_LEAD_DONE = 2


class ReplyStream:
    """Streaming display filter for one reply (a turn or a proactive
    message). One instance per reply; not reusable after ``finish``."""

    def __init__(self) -> None:
        self._raw = ""  # uncommitted sanitized text of the current entry
        self._entry_id: str | None = None
        self._lead = _LEAD_UNDECIDED
        self._shown = False  # a non-blank line has been returned
        self._blank_pending = False  # one blank line owed before the next text
        self._finished = False

    @property
    def has_text(self) -> bool:
        return self._shown

    def feed(self, entry_id: str, delta: str) -> list[str]:
        """Append ``delta`` to the entry ``entry_id``; return the lines this
        newly committed. A different ``entry_id`` first flushes the current
        entry."""
        if self._finished:
            raise RuntimeError("ReplyStream.feed called after finish()")
        out: list[str] = []
        if self._entry_id is not None and entry_id != self._entry_id:
            out.extend(self.flush())
        self._entry_id = entry_id
        self._raw += sanitize(delta)
        spans, hold = _scan(self._raw)
        cut = self._raw.rfind("\n", 0, hold)
        while cut != -1 and _covered(spans, cut):
            cut = self._raw.rfind("\n", 0, cut)
        if cut != -1:
            chunk, self._raw = self._raw[: cut + 1], self._raw[cut + 1 :]
            self._commit(self._strip_lead(_filter(chunk)), out)
        return out

    def flush(self) -> list[str]:
        """End the current provider step: commit the remaining text up to
        the streaming hold-back point, drop what was held (unclosed tag,
        trailing tag fragment), and owe a paragraph break before the next
        text."""
        out: list[str] = []
        _, hold = _scan(self._raw)
        self._commit(self._strip_lead(_filter(self._raw[:hold])), out)
        self._reset_entry()
        return out

    def pending(self) -> str:
        """The in-progress tail of the current line: no newline, no ESC,
        never a tag fragment, never a partial leading timestamp."""
        _, hold = _scan(self._raw)
        text = _filter(self._raw[:hold])
        if self._lead == _LEAD_UNDECIDED:
            match = _TIMESTAMP_RE.match(text)
            if match:
                text = text[match.end() :]
            elif _TIMESTAMP_PREFIX_RE.match(text):
                text = ""
        elif self._lead == _LEAD_STRIPPING:
            text = text.lstrip()
        return text if text.strip() else ""

    def finish(self) -> list[str]:
        """``flush`` and end the stream; ``feed`` afterwards raises."""
        out = self.flush()
        self._finished = True
        return out

    def discard(self) -> None:
        """Drop the uncommitted text (server-side context reset). Committed
        lines stay; the next text starts a fresh entry after a paragraph
        break."""
        self._reset_entry()

    def _reset_entry(self) -> None:
        self._raw = ""
        self._lead = _LEAD_UNDECIDED
        self._blank_pending = self._shown

    def _strip_lead(self, text: str) -> str:
        if self._lead == _LEAD_UNDECIDED and text:
            match = _TIMESTAMP_RE.match(text)
            if match:
                text = text[match.end() :]
                self._lead = _LEAD_STRIPPING
            else:
                self._lead = _LEAD_DONE
        if self._lead == _LEAD_STRIPPING:
            text = text.lstrip()
            if text:
                self._lead = _LEAD_DONE
        return text

    def _commit(self, text: str, out: list[str]) -> None:
        lines = text.split("\n")
        if lines[-1] == "":
            lines.pop()
        for line in lines:
            line = line.rstrip()
            if not line:
                self._blank_pending = self._shown
                continue
            if self._blank_pending:
                out.append("")
                self._blank_pending = False
            out.append(line)
            self._shown = True


def display_lines(text: str) -> list[str]:
    """One-shot display lines for a complete text (the same path as
    streaming: one ``feed``, then ``finish``)."""
    stream = ReplyStream()
    lines = stream.feed("", text)
    lines.extend(stream.finish())
    return lines
