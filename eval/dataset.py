"""
Hand-labeled gold examples for evaluation and optimization.

Each example has inputs (thread, prospect_name) and gold labels the metric
scores against. In production these come from an agent feedback loop;
here they're hand-labeled from the test fixtures.
"""

from __future__ import annotations

import dspy

from app.pipeline.preprocessing import format_thread_for_llm
from tests.fixtures import threads as fixtures


def build_example(fixture: dict, **labels) -> dspy.Example:
    """Turn a fixture dict + gold labels into a dspy.Example."""
    formatted = format_thread_for_llm(fixture["thread"], fixture["prospect_name"])
    return dspy.Example(
        thread=formatted,
        prospect_name=fixture["prospect_name"],
        **labels,
    ).with_inputs("thread", "prospect_name")


# -- Snapshot (single prospect message) --

SNAPSHOT_GOLD: list[dspy.Example] = [
    build_example(
        fixtures.SINGLE_MESSAGE_POSITIVE,
        mode="snapshot",
        overall_sentiment="positive",
        engagement_level="high",
        urgency="high",
        objection_present=False,
        next_action_must_mention=["call", "Friday", "calendar", "demo"],
    ),
    build_example(
        fixtures.SINGLE_MESSAGE_POLITE_NO,
        mode="snapshot",
        overall_sentiment="negative",
        engagement_level="low",
        urgency="low",
        objection_present=True,
        next_action_must_mention=["nurture", "later", "quarter", "close", "do not"],
    ),
    build_example(
        fixtures.SINGLE_MESSAGE_NEUTRAL_QUESTION,
        mode="snapshot",
        overall_sentiment="neutral",
        engagement_level="medium",
        urgency="medium",
        objection_present=False,
        next_action_must_mention=["pricing", "HubSpot", "integration", "answer"],
    ),
]


# -- Trended (multiple prospect messages) --

TRENDED_GOLD: list[dspy.Example] = [
    build_example(
        fixtures.WARMING,
        mode="trended",
        overall_sentiment="positive",
        sentiment_trend="warming",
        engagement_level="high",
        urgency="high",
        objection_present=False,
        next_action_must_mention=["Thursday", "Friday", "Jamie", "calendar", "invite"],
    ),
    build_example(
        fixtures.COOLING,
        mode="trended",
        overall_sentiment="negative",
        sentiment_trend="cooling",
        engagement_level="low",
        urgency="low",
        objection_present=True,
        next_action_must_mention=["nurture", "next quarter", "deprioritize", "close"],
    ),
    build_example(
        fixtures.GHOSTING,
        mode="trended",
        overall_sentiment="neutral",
        sentiment_trend="cooling",
        engagement_level="low",
        urgency="medium",
        objection_present=False,
        next_action_must_mention=["break-up", "different angle", "stop", "pause", "value"],
    ),
    build_example(
        fixtures.OBJECTION_HEAVY,
        mode="trended",
        overall_sentiment="positive",
        sentiment_trend="warming",
        engagement_level="high",
        urgency="high",
        objection_present=True,
        next_action_must_mention=["ROI", "case study", "calculator", "next week"],
    ),
    build_example(
        fixtures.SARCASM,
        mode="trended",
        overall_sentiment="negative",
        sentiment_trend="cooling",
        engagement_level="low",
        urgency="low",
        objection_present=True,
        next_action_must_mention=["close", "stop", "do not", "disqualify"],
    ),
    build_example(
        fixtures.MIXED_VOLATILE,
        mode="trended",
        overall_sentiment="positive",
        sentiment_trend="volatile",
        engagement_level="medium",
        urgency="low",
        objection_present=True,
        next_action_must_mention=["renewal", "next year", "nurture", "roadmap"],
    ),
]


def all_examples() -> list[dspy.Example]:
    return SNAPSHOT_GOLD + TRENDED_GOLD


def split(examples: list[dspy.Example], dev_size: int = 3, seed: int = 7):
    """Deterministic train/dev split."""
    import random

    rng = random.Random(seed)
    pool = list(examples)
    rng.shuffle(pool)
    return pool[dev_size:], pool[:dev_size]
