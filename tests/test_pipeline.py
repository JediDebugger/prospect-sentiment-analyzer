"""
Pipeline tests that don't require an LLM.

These cover the deterministic plumbing: validation, preprocessing,
single-vs-multi routing, and metric computation. They run instantly in CI
without an API key.

The LLM-touching API tests live in test_api.py and are gated on OPENAI_API_KEY.
"""

from __future__ import annotations

import pytest

from app.pipeline.preprocessing import (
    compute_thread_stats,
    format_thread_for_llm,
    is_out_of_office,
    strip_quoted_reply,
)
from app.schemas import AnalyzeRequest, Message
from eval.metrics import composite_score, groundedness_score
from tests.fixtures import threads as F


def _model_for(fixture: dict) -> AnalyzeRequest:
    return AnalyzeRequest(
        prospect_name=fixture["prospect_name"],
        thread=fixture["thread"],
    )


# ---- Validation -----------------------------------------------------------

def test_request_rejects_thread_with_no_prospect_message():
    with pytest.raises(Exception):
        AnalyzeRequest(
            prospect_name="Nobody",
            thread=F.NO_PROSPECT_MSG["thread"],
        )


def test_request_rejects_empty_thread():
    with pytest.raises(Exception):
        AnalyzeRequest(prospect_name="x", thread=[])


def test_request_accepts_single_message_fixture():
    req = _model_for(F.SINGLE_MESSAGE_POSITIVE)
    assert any(m.sender == "prospect" for m in req.thread)


# ---- Routing / stats ------------------------------------------------------

def test_single_message_fixture_routes_to_snapshot():
    stats = compute_thread_stats(F.SINGLE_MESSAGE_POSITIVE["thread"])
    assert stats.single_message is True
    assert stats.prospect_messages == 1


def test_multi_message_fixture_routes_to_trended():
    stats = compute_thread_stats(F.WARMING["thread"])
    assert stats.single_message is False
    assert stats.prospect_messages >= 2


def test_ooo_only_thread_treated_as_no_real_prospect_msg():
    """OOO autoresponder is not a real reply; should route as snapshot
    with prospect_messages == 0 (the OOO is filtered out for the count)."""
    stats = compute_thread_stats(F.OOO_ONLY["thread"])
    assert stats.single_message is True
    assert stats.prospect_messages == 0


def test_one_prospect_many_agent_routes_to_snapshot():
    """Five agent messages, one prospect reply -> still snapshot mode.
    This is the case where total_messages > 1 but prospect_messages == 1."""
    thread = [
        *F.SINGLE_MESSAGE_POSITIVE["thread"][:1] * 4,  # multiple agent messages
        F.SINGLE_MESSAGE_POSITIVE["thread"][1],         # the one prospect reply
    ]
    stats = compute_thread_stats(thread)
    assert stats.single_message is True
    assert stats.prospect_messages == 1
    assert stats.agent_messages == 4


# ---- Quoted-reply stripping ----------------------------------------------

def test_strip_quoted_reply_removes_gmail_style_quote():
    body = (
        "Sounds great, let's set up a call.\n"
        "\n"
        "On Mon, May 4, 2026 at 9:13 AM Alex <alex@revreply.com> wrote:\n"
        "> Hi Jane, would you like a demo?\n"
        "> Cheers, Alex\n"
    )
    cleaned = strip_quoted_reply(body)
    assert "Sounds great" in cleaned
    assert "Hi Jane" not in cleaned
    assert "Cheers" not in cleaned


def test_is_out_of_office_detects_common_phrases():
    assert is_out_of_office("I am out of office until Friday.")
    assert is_out_of_office("Automatic reply: on vacation")
    assert not is_out_of_office("Sounds great, let's chat.")


# ---- Formatting -----------------------------------------------------------

def test_format_thread_for_llm_labels_senders():
    rendered = format_thread_for_llm(
        F.WARMING["thread"], F.WARMING["prospect_name"]
    )
    assert "PROSPECT (Sara Kim)" in rendered
    assert "AGENT" in rendered
    # Sequential message numbering preserved
    assert "Message 1" in rendered
    assert "Message 6" in rendered


# ---- Metrics --------------------------------------------------------------

class _StubPred:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_groundedness_perfect_when_quotes_present_in_thread():
    thread = "AGENT: hi\nPROSPECT: We need ROI before next week."
    pred = _StubPred(key_signals=[_StubPred(quote="ROI before next week", type="objection")])
    assert groundedness_score(thread, pred) == 1.0


def test_groundedness_punishes_hallucinated_quote():
    thread = "AGENT: hi\nPROSPECT: We need ROI before next week."
    pred = _StubPred(key_signals=[
        _StubPred(quote="I love your product", type="positive_tone"),
        _StubPred(quote="ROI before next week", type="objection"),
    ])
    assert groundedness_score(thread, pred) == 0.5


def test_composite_score_rewards_correct_sentiment():
    example = type("E", (), {})()
    example.mode = "snapshot"
    example.thread = "PROSPECT: this is great"
    example.overall_sentiment = "positive"
    example.objection_present = False
    example.next_action_must_mention = ["call"]

    pred = _StubPred(
        overall_sentiment="positive",
        engagement_level="high",
        urgency="high",
        key_signals=[_StubPred(quote="this is great", type="positive_tone")],
        objections_or_concerns=[],
        recommended_next_action="Send a calendar link to set up a call.",
        rationale="Looks good.",
    )
    score = composite_score(example, pred)
    assert score["_total"] >= 0.9
    assert score["sentiment_match"] == 1.0
    assert score["groundedness"] == 1.0


def test_composite_score_punishes_wrong_sentiment_and_missed_objection():
    example = type("E", (), {})()
    example.mode = "snapshot"
    example.thread = "PROSPECT: not interested, too expensive"
    example.overall_sentiment = "negative"
    example.objection_present = True
    example.next_action_must_mention = ["close"]

    pred = _StubPred(
        overall_sentiment="positive",            # wrong
        engagement_level="high",
        urgency="high",
        key_signals=[],
        objections_or_concerns=[],               # missed
        recommended_next_action="Schedule a meeting.",
        rationale="they seem keen",
    )
    score = composite_score(example, pred)
    assert score["sentiment_match"] == 0.0
    assert score["objection_match"] == 0.0
    assert score["_total"] < 0.6
