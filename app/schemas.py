"""
Pydantic request/response models for the /analyze-sentiment endpoint.

Design notes
------------
- Thread is a list of structured Message objects (not a raw blob). Real RevReply
  systems already store messages as records; structured input lets us reliably
  isolate the prospect's messages and avoids brittle email-header parsing in the
  LLM. We're rating the *prospect's* sentiment, not the agent's.
- The response makes the analysis_mode ("snapshot" vs "trended") explicit, so
  downstream consumers and the eval harness know exactly what was produced.
- key_signals carry verbatim quotes from the thread. This is what powers the
  cheapest hallucination guardrail we have (substring grounding check).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


SentimentLabel = Literal["positive", "neutral", "negative"]
TrendLabel = Literal["warming", "cooling", "steady", "volatile", "unknown"]
LevelLabel = Literal["low", "medium", "high", "unknown"]
SignalType = Literal[
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
AnalysisMode = Literal["snapshot", "trended"]


class Message(BaseModel):
    """One email in the thread."""

    model_config = ConfigDict(populate_by_name=True)

    sender: Literal["agent", "prospect"] = Field(
        description="Who wrote this message. Determined by the upstream system, "
        "not inferred from headers."
    )
    # `from` is reserved in Python; expose alias for callers.
    from_: Optional[str] = Field(default=None, alias="from")
    timestamp: Optional[datetime] = None
    body: str = Field(min_length=1)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "prospect_name": "Sara Kim",
                    "thread": [
                        {"sender": "agent", "from": "alex@revreply.com", "body": "Hi Sara, thought RevReply might be relevant for your SDR team. Worth 15 minutes?"},
                        {"sender": "prospect", "from": "sara@helio.io", "body": "Maybe. Send me a one-pager and I'll take a look."},
                        {"sender": "agent", "from": "alex@revreply.com", "body": "Attached. Happy to answer questions whenever."},
                        {"sender": "prospect", "from": "sara@helio.io", "body": "Yes, let's do it Friday afternoon. Include our RevOps lead Jamie too."},
                    ],
                }
            ]
        }
    )

    prospect_name: str = Field(min_length=1)
    thread: list[Message] = Field(min_length=1)

    @field_validator("thread")
    @classmethod
    def _must_have_prospect_message(cls, v: list[Message]) -> list[Message]:
        if not any(m.sender == "prospect" for m in v):
            raise ValueError(
                "Thread must contain at least one prospect message; nothing to analyze."
            )
        return v


class Signal(BaseModel):
    quote: str = Field(description="Verbatim excerpt from the thread.")
    type: SignalType


class ThreadStats(BaseModel):
    prospect_messages: int
    agent_messages: int
    total_messages: int
    single_message: bool = Field(
        description="True when there is <=1 prospect message — no trend can be inferred."
    )


class AnalyzeResponse(BaseModel):
    prospect_name: str
    analysis_mode: AnalysisMode

    overall_sentiment: SentimentLabel
    sentiment_trend: TrendLabel = Field(
        description="Direction across prospect messages. 'unknown' in snapshot mode."
    )
    trend_confidence: float = Field(
        ge=0.0, le=1.0, description="0.0 in snapshot mode (no trend can be measured)."
    )
    engagement_level: LevelLabel
    urgency: LevelLabel

    key_signals: list[Signal]
    objections_or_concerns: list[str]
    recommended_next_action: str
    rationale: str

    thread_stats: ThreadStats
    model_version: Optional[str] = None
    program_version: Optional[str] = None
