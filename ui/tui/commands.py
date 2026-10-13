"""Slash commands for the MIRA TUI: registry, parse, and result rendering.

Pure, stdlib-only (the discipline of ``text.py``): strings in, strings out;
no I/O, no Rich, no httpx, no asyncio. ``ChatSession`` owns policy — what a
command does to session state and how a result reaches the screen; this module
owns shape — what text is a command, and what a response dict says. The
registry is data: adding a command is one ``Command`` entry plus its renderer,
and a future prompt_toolkit completer iterates ``COMMANDS`` (``name`` +
``summary``) without touching dispatch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

# A command word: "/" then letters, digits, underscores, dashes, and nothing
# else. Only word-shaped first tokens are command attempts; any other
# leading-slash text ("/etc/hosts is the path", "/pause.") is a message to MIRA.
_WORD_RE = re.compile(r"/[A-Za-z][A-Za-z0-9_-]*\Z")


@dataclass(frozen=True)
class Command:
    """One registered slash command. ``name`` is canonical, without the slash."""

    name: str
    summary: str
    kind: Literal["quit", "continuum"]  # who executes it
    action: str  # continuum action name; empty iff kind == "quit"
    render: Callable[[dict[str, Any]], str] | None  # success payload -> notice text; None iff kind == "quit"


@dataclass(frozen=True)
class CommandHit:
    """The text names a registered command; ``args`` is everything after the word."""

    command: Command
    args: str  # the remainder of the text (this line and every later one), stripped


@dataclass(frozen=True)
class UnknownCommand:
    """A slash word that is not registered. Never sent to the model."""

    word: str  # the token as typed, e.g. "/puse"


@dataclass(frozen=True)
class PlainMessage:
    """Not command-shaped: deliver to the model unchanged."""

    text: str


ParseResult = CommandHit | UnknownCommand | PlainMessage


# Every renderer tolerates field drift: isinstance checks, never data["k"] —
# the endpoint also injects a "meta" dict into every payload, and a wording
# change on the server must degrade the notice, never crash the session.


def _payload_message(data: dict[str, Any], default: str) -> str:
    message = data.get("message")
    return message if isinstance(message, str) and message.strip() else default


def _render_pause(data: dict[str, Any]) -> str:
    return _payload_message(data, "session paused")


def _render_resume(data: dict[str, Any]) -> str:
    return _payload_message(data, "session resumed")


def _render_status(data: dict[str, Any]) -> str:
    if not data.get("has_active_segment"):
        return "no active segment — your next message starts a fresh one"
    if data.get("is_paused"):
        return "session is paused — it resumes automatically when you send your next message"
    at = data.get("collapse_at")
    timeout = data.get("timeout_minutes")
    when = f" and collapses at {at}" if isinstance(at, str) and at else ""
    mins = f" after {timeout} min of inactivity" if isinstance(timeout, int) else ""
    return f"active segment{when}{mins}"


def _render_collapse(data: dict[str, Any]) -> str:
    text = _payload_message(data, "segment collapsed")
    title = data.get("title")
    if isinstance(title, str) and title.strip():
        text += f' — "{title}"'
    return text


COMMANDS: dict[str, Command] = {
    "exit": Command("exit", "quit the client", "quit", "", None),
    "pause": Command("pause", "suspend this segment's collapse timer",
                     "continuum", "pause_session", _render_pause),
    "resume": Command("resume", "lift a pause without sending a message",
                      "continuum", "resume_session", _render_resume),
    "status": Command("status", "show this segment's state",
                      "continuum", "get_segment_status", _render_status),
    "collapse": Command("collapse", "summarize and close this segment now",
                        "continuum", "collapse_segment", _render_collapse),
}


def help_list() -> str:
    """The registered commands, for the unknown-command notice."""
    return " ".join(f"/{name}" for name in sorted(COMMANDS))


def parse(text: str) -> ParseResult:
    """Classify one stripped, non-empty submission.

    Command-shaped means the first whitespace token matches ``_WORD_RE``.
    Lookup is exact and case-sensitive, so "/Pause" is an unknown command
    naming "/Pause" — never a silent match, never a message. The remainder of
    the text is the command's args.
    """
    first = text.split(None, 1)[0]
    if not _WORD_RE.fullmatch(first):
        return PlainMessage(text)
    command = COMMANDS.get(first[1:])
    if command is None:
        return UnknownCommand(first)
    return CommandHit(command, text[len(first):].strip())
