"""
Deterministic preprocessing — runs *before* the LLM.

Why this matters
----------------
- The single-vs-multi-message routing decision is made here, in plain Python.
  We never ask the model "is this a single message thread?" — that's the kind
  of trivial classification an LLM can get wrong on edge cases (e.g. an inline
  reply containing the prior agent message). Code is deterministic.
- We strip quoted/forwarded blocks before counting the prospect's messages so
  a 1-message reply that includes the agent's prior email doesn't get mistaken
  for a multi-turn conversation.
- We produce a single canonical, sender-labelled transcript string for the LM
  so the Signature can stay model-agnostic.
"""

from __future__ import annotations

import re
from typing import Iterable

from app.schemas import Message, ThreadStats

# Quoted-reply markers we see in the wild. Conservative: we only strip lines
# that *clearly* belong to a quoted block, never anything ambiguous.
_QUOTE_LINE = re.compile(r"^\s*>")
_ON_WROTE = re.compile(
    r"^On\s+.+?\s+(?:wrote|sent):?\s*$", re.IGNORECASE
)
_FROM_HEADER = re.compile(r"^\s*From:\s+.+", re.IGNORECASE)
_OOO_MARKERS = re.compile(
    r"\b(out of office|on vacation|currently away|automatic reply|auto-?reply)\b",
    re.IGNORECASE,
)


def strip_quoted_reply(body: str) -> str:
    """Return the body with quoted/forwarded sections removed.

    Heuristic only — good enough for the common Gmail/Outlook patterns. We err
    on the side of keeping content; false positives here would drop real
    prospect text.
    """
    lines = body.splitlines()
    cleaned: list[str] = []
    in_quote_block = False
    for line in lines:
        if _ON_WROTE.match(line) or _FROM_HEADER.match(line):
            in_quote_block = True
            continue
        if in_quote_block:
            # Once we're in a quote block, blank lines + > lines stay in it.
            if not line.strip() or _QUOTE_LINE.match(line):
                continue
            # A non-blank, non-> line probably means the quote ended (rare).
            in_quote_block = False
        if _QUOTE_LINE.match(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def is_out_of_office(body: str) -> bool:
    return bool(_OOO_MARKERS.search(body or ""))


def split_by_sender(thread: Iterable[Message]) -> tuple[list[Message], list[Message]]:
    prospect = [m for m in thread if m.sender == "prospect"]
    agent = [m for m in thread if m.sender == "agent"]
    return prospect, agent


def compute_thread_stats(thread: list[Message]) -> ThreadStats:
    prospect, agent = split_by_sender(thread)
    # Count "real" prospect messages — ignore OOO autoresponders for the
    # routing decision because they aren't sentiment signal.
    real_prospect = [m for m in prospect if not is_out_of_office(m.body)]
    return ThreadStats(
        prospect_messages=len(real_prospect),
        agent_messages=len(agent),
        total_messages=len(thread),
        single_message=len(real_prospect) <= 1,
    )


def format_thread_for_llm(thread: list[Message], prospect_name: str) -> str:
    """Render the thread as a sender-labelled transcript.

    Quoted replies are stripped from each message body so the LM sees a clean
    sequence of newly-written content. Empty bodies (after stripping) are
    represented as ``[empty]`` so message count is preserved.
    """
    parts: list[str] = []
    for i, m in enumerate(thread, start=1):
        body = strip_quoted_reply(m.body) or "[empty]"
        sender_label = (
            f"PROSPECT ({prospect_name})" if m.sender == "prospect" else "AGENT"
        )
        ts = f"  [{m.timestamp.isoformat()}]" if m.timestamp else ""
        parts.append(f"--- Message {i} — {sender_label}{ts} ---\n{body}")
    return "\n\n".join(parts)
