"""
DSPy Signatures — the typed I/O contracts the LLM is asked to satisfy.

We deliberately maintain *two* Signatures:

  - SnapshotSentiment: one prospect message. No trend can be inferred.
  - TrendedSentiment:  multiple prospect messages. Trend is required.

Why two instead of one with optional fields:

  1. Each prompt stays focused on a single job — quality goes up. No "if there
     is only one message, leave trend null" hedge clauses to confuse the model.
  2. The DSPy optimizer (BootstrapFewShot / MIPROv2) can specialize each
     Signature with its own demonstrations. A single-message demo teaches
     nothing useful about trend detection and vice versa.
  3. Type safety: `sentiment_trend` is required for TrendedSentiment and
     simply absent from SnapshotSentiment. The model literally cannot forget
     it on multi-message threads.

The docstring of each Signature becomes part of the prompt and is also what
the optimizer rewrites during compilation. Keep them clear, behavioral, and
focused.
"""

from __future__ import annotations

from typing import Literal

import dspy
from pydantic import BaseModel, Field


# Output sub-type. DSPy parses lists of Pydantic models for free.
class Signal(BaseModel):
    quote: str = Field(
        description=(
            "Verbatim excerpt copied from the thread that supports the analysis. "
            "Must appear in the thread text exactly as quoted."
        )
    )
    type: Literal[
        "buying_signal",
        "objection",
        "concern",
        "delay",
        "ghosting",
        "positive_tone",
        "negative_tone",
        "question",
        "other",
    ]


class SnapshotSentiment(dspy.Signature):
    """Analyze a sales prospect's sentiment from a SINGLE message they have sent.

    The thread may contain prior agent messages for context, but only one
    prospect message exists. Do NOT infer a sentiment trajectory — there is no
    earlier prospect message to compare to. Focus on what this single reply
    tells the agent, what the prospect cares about, and what to do next.

    Be conservative: if the message is short, polite, or non-committal, prefer
    "neutral" over forcing positive/negative. Always ground key_signals in
    verbatim quotes from the thread.
    """

    thread: str = dspy.InputField(
        desc="Full sender-labelled transcript. Contains exactly one prospect message."
    )
    prospect_name: str = dspy.InputField()

    overall_sentiment: Literal["positive", "neutral", "negative"] = dspy.OutputField(
        desc="The prospect's sentiment in this single message."
    )
    engagement_level: Literal["low", "medium", "high"] = dspy.OutputField(
        desc="How engaged the prospect appears: question density, length, "
        "specificity, willingness to take next steps."
    )
    urgency: Literal["low", "medium", "high"] = dspy.OutputField(
        desc="How quickly the agent should respond. High when the prospect "
        "is actively asking, ready to buy, or about to disengage."
    )
    key_signals: list[Signal] = dspy.OutputField(
        desc="Up to 5 grounded signals. Each quote MUST be a verbatim substring "
        "of the thread. Do not paraphrase."
    )
    objections_or_concerns: list[str] = dspy.OutputField(
        desc="Concrete objections, blockers, or concerns the prospect raised. "
        "Empty list if none."
    )
    recommended_next_action: str = dspy.OutputField(
        desc="One short, specific action the sales agent should take next. "
        "Concrete (e.g. 'Send Friday calendar link and ROI one-pager'), "
        "not generic (e.g. 'Follow up')."
    )
    rationale: str = dspy.OutputField(
        desc="2-4 sentences explaining the analysis. Reference the quotes."
    )


class TrendedSentiment(dspy.Signature):
    """Analyze a sales prospect's sentiment trajectory across MULTIPLE messages.

    Compare the prospect's earlier messages to their later ones to determine
    whether they are warming, cooling, holding steady, or volatile (mixed
    signals). Weight recency: the most recent prospect message matters more
    than the first one for both `overall_sentiment` and `recommended_next_action`.

    Calibrate `trend_confidence`:
      - high (>=0.8) when direction is unambiguous across multiple messages
      - medium (~0.5) when there are mixed signals or only 2 prospect messages
      - low  (<=0.3) when most messages are short/non-committal

    Always ground key_signals in verbatim quotes from the thread.
    """

    thread: str = dspy.InputField(
        desc="Full sender-labelled transcript with multiple prospect messages."
    )
    prospect_name: str = dspy.InputField()

    overall_sentiment: Literal["positive", "neutral", "negative"] = dspy.OutputField(
        desc="Current sentiment, weighting the most recent prospect messages."
    )
    sentiment_trend: Literal["warming", "cooling", "steady", "volatile"] = (
        dspy.OutputField(
            desc="Direction across the prospect's messages over time. "
            "'volatile' when sentiment swings without a clear direction."
        )
    )
    trend_confidence: float = dspy.OutputField(
        desc="0.0-1.0 confidence in the chosen sentiment_trend."
    )
    engagement_level: Literal["low", "medium", "high"] = dspy.OutputField(
        desc="How engaged the prospect is *now*, not on average."
    )
    urgency: Literal["low", "medium", "high"] = dspy.OutputField(
        desc="How quickly the agent should respond. High when prospect is "
        "ready to buy or about to disengage."
    )
    key_signals: list[Signal] = dspy.OutputField(
        desc="Up to 6 grounded signals across the thread. Each quote MUST be "
        "a verbatim substring of the thread. Do not paraphrase."
    )
    objections_or_concerns: list[str] = dspy.OutputField(
        desc="Outstanding objections, blockers, or concerns. Empty list if none."
    )
    recommended_next_action: str = dspy.OutputField(
        desc="One short, specific action for the sales agent. Concrete, not generic."
    )
    rationale: str = dspy.OutputField(
        desc="2-4 sentences explaining state, trend, and reasoning. Reference quotes."
    )
