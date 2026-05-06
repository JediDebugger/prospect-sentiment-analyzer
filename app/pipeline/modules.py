"""
The DSPy Module that powers the endpoint.

`ProspectSentimentAnalyzer` is a tiny router:

    1. Compute thread stats deterministically (Python).
    2. Route to SnapshotSentiment (1 prospect msg) or TrendedSentiment (2+).
    3. Coerce the prediction into the API response schema.

The router lives in code, not in a prompt — that's the whole point of having
two specialized Signatures. The LM is never asked to figure out which mode it
is in; only to do its specific job once routed.

This Module is what gets passed to a DSPy optimizer (BootstrapFewShot,
MIPROv2). The optimizer sees the two sub-predictors as separate optimization
targets and can produce different prompts/demonstrations for each.
"""

from __future__ import annotations

from typing import Any

import dspy

from app.pipeline.preprocessing import compute_thread_stats, format_thread_for_llm
from app.pipeline.signatures import SnapshotSentiment, TrendedSentiment
from app.schemas import (
    AnalyzeResponse,
    LevelLabel,
    Message,
    SentimentLabel,
    Signal,
    ThreadStats,
    TrendLabel,
)


def _to_signal_list(raw: Any) -> list[Signal]:
    """Best-effort coercion of DSPy output (Pydantic objects or dicts) into Signals."""
    if not raw:
        return []
    out: list[Signal] = []
    for item in raw:
        if isinstance(item, Signal):
            out.append(item)
            continue
        if hasattr(item, "model_dump"):
            data = item.model_dump()
        elif isinstance(item, dict):
            data = item
        else:
            # Fallback: stringify; better than dropping.
            data = {"quote": str(item), "type": "other"}
        try:
            out.append(Signal(**data))
        except Exception:
            # Don't let one malformed signal kill the response.
            continue
    return out


class ProspectSentimentAnalyzer(dspy.Module):
    """Composite module: routes between snapshot and trended analysis."""

    def __init__(self) -> None:
        super().__init__()
        # ChainOfThought adds a hidden reasoning step before structured output.
        # For nuanced sales sentiment (sarcasm, polite-no, mixed signals) the
        # extra reasoning materially improves accuracy. Worth the tokens.
        self.snapshot = dspy.ChainOfThought(SnapshotSentiment)
        self.trended = dspy.ChainOfThought(TrendedSentiment)

    # NOTE: `forward` returns a dspy.Prediction so the standard DSPy evaluation
    # / optimization machinery works. The HTTP layer adapts it into AnalyzeResponse.
    def forward(
        self, thread: str, prospect_name: str, mode: str | None = None
    ) -> dspy.Prediction:
        if mode is None:
            # When called by DSPy optimizers (BootstrapFewShot, MIPROv2), mode
            # isn't passed — they only forward the Signature's input fields.
            # Infer it from the formatted transcript by counting prospect blocks.
            prospect_tag = f"PROSPECT ({prospect_name})"
            n_prospect = thread.count(prospect_tag)
            mode = "snapshot" if n_prospect <= 1 else "trended"
        if mode == "snapshot":
            return self.snapshot(thread=thread, prospect_name=prospect_name)
        return self.trended(thread=thread, prospect_name=prospect_name)


def analyze(
    analyzer: ProspectSentimentAnalyzer,
    thread: list[Message],
    prospect_name: str,
    *,
    model_version: str | None = None,
    program_version: str | None = None,
) -> AnalyzeResponse:
    """Public helper used by the FastAPI route + the eval harness.

    Centralizes:
      - thread stats / mode detection
      - LLM dispatch
      - response shaping (filling trend fields with 'unknown'/0.0 in snapshot mode)
    """
    stats: ThreadStats = compute_thread_stats(thread)
    mode = "snapshot" if stats.single_message else "trended"
    formatted = format_thread_for_llm(thread, prospect_name)

    pred = analyzer(thread=formatted, prospect_name=prospect_name, mode=mode)

    # Common fields produced by both Signatures.
    overall_sentiment: SentimentLabel = pred.overall_sentiment
    engagement_level: LevelLabel = pred.engagement_level
    urgency: LevelLabel = pred.urgency
    key_signals = _to_signal_list(pred.key_signals)
    objections = list(pred.objections_or_concerns or [])
    recommended_next_action = pred.recommended_next_action
    rationale = pred.rationale

    if mode == "snapshot":
        sentiment_trend: TrendLabel = "unknown"
        trend_confidence: float = 0.0
    else:
        sentiment_trend = pred.sentiment_trend
        trend_confidence = float(pred.trend_confidence or 0.0)
        # Defensive clamp.
        trend_confidence = max(0.0, min(1.0, trend_confidence))

    return AnalyzeResponse(
        prospect_name=prospect_name,
        analysis_mode=mode,
        overall_sentiment=overall_sentiment,
        sentiment_trend=sentiment_trend,
        trend_confidence=trend_confidence,
        engagement_level=engagement_level,
        urgency=urgency,
        key_signals=key_signals,
        objections_or_concerns=objections,
        recommended_next_action=recommended_next_action,
        rationale=rationale,
        thread_stats=stats,
        model_version=model_version,
        program_version=program_version,
    )
