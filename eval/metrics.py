"""
Composite metric for the sentiment analyzer.

Uses a weighted combination of sub-scores so the optimizer can't cheat
by nailing the sentiment label while producing garbage next-actions.
"""

from __future__ import annotations

from typing import Any

WEIGHTS = {
    "sentiment_match": 0.30,
    "trend_match": 0.20,
    "groundedness": 0.20,
    "structural": 0.10,
    "next_action_relevance": 0.10,
    "objection_match": 0.10,
}


def _get(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _extract_quotes(pred: Any) -> list[str]:
    raw = _get(pred, "key_signals", []) or []
    out: list[str] = []
    for s in raw:
        q = _get(s, "quote", None)
        if isinstance(q, str) and q.strip():
            out.append(q)
    return out


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def groundedness_score(thread_text: str, pred: Any) -> float:
    """Fraction of quoted signals that actually appear in the thread."""
    quotes = _extract_quotes(pred)
    if not quotes:
        return 1.0
    haystack = _normalize(thread_text)
    hits = sum(1 for q in quotes if _normalize(q) and _normalize(q) in haystack)
    return hits / len(quotes)


def structural_score(pred: Any, mode: str) -> float:
    required = [
        "overall_sentiment",
        "engagement_level",
        "urgency",
        "key_signals",
        "objections_or_concerns",
        "recommended_next_action",
        "rationale",
    ]
    if mode == "trended":
        required += ["sentiment_trend", "trend_confidence"]
    have = sum(1 for f in required if _get(pred, f, None) not in (None, ""))
    return have / len(required)


def next_action_relevance(pred: Any, must_mention: list[str]) -> float:
    if not must_mention:
        return 1.0
    text = _normalize(_get(pred, "recommended_next_action", "") or "")
    if not text:
        return 0.0
    for needle in must_mention:
        if _normalize(needle) in text:
            return 1.0
    return 0.0


def objection_match(pred: Any, expected_present: bool) -> float:
    objections = _get(pred, "objections_or_concerns", []) or []
    has = bool([o for o in objections if str(o).strip()])
    return 1.0 if has == expected_present else 0.0


def sentiment_match(pred: Any, expected: str) -> float:
    return 1.0 if _get(pred, "overall_sentiment", None) == expected else 0.0


def trend_match(pred: Any, expected: str | None, mode: str) -> float | None:
    if mode != "trended" or not expected:
        return None
    actual = _get(pred, "sentiment_trend", None)
    if actual == expected:
        return 1.0
    # partial credit for close-ish misses
    soft = {
        ("warming", "steady"): 0.5,
        ("cooling", "steady"): 0.5,
        ("steady", "warming"): 0.5,
        ("steady", "cooling"): 0.5,
        ("volatile", "warming"): 0.3,
        ("volatile", "cooling"): 0.3,
    }
    return soft.get((expected, actual), 0.0)


def composite_score(example: Any, pred: Any) -> dict:
    """Per-component scores + weighted total. Works for both modes."""
    mode = _get(example, "mode", "trended")
    thread_text = _get(example, "thread", "") or ""

    parts: dict[str, float] = {
        "sentiment_match": sentiment_match(pred, _get(example, "overall_sentiment")),
        "groundedness": groundedness_score(thread_text, pred),
        "structural": structural_score(pred, mode),
        "next_action_relevance": next_action_relevance(
            pred, _get(example, "next_action_must_mention", []) or []
        ),
        "objection_match": objection_match(
            pred, bool(_get(example, "objection_present", False))
        ),
    }

    tm = trend_match(pred, _get(example, "sentiment_trend", None), mode)
    if tm is not None:
        parts["trend_match"] = tm

    # renormalize weights for whatever components are active
    active_weights = {k: WEIGHTS[k] for k in parts}
    total_w = sum(active_weights.values())
    weighted = sum(parts[k] * active_weights[k] for k in parts) / total_w

    parts["_total"] = weighted
    parts["_mode"] = mode
    return parts


def composite_metric(example: Any, pred: Any, trace=None) -> float:
    """Single float for dspy.Evaluate / optimizers."""
    return composite_score(example, pred)["_total"]
