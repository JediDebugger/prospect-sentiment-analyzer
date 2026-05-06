"""
API tests for /analyze-sentiment.

Most assertions don't actually call the LLM — we monkey-patch the analyzer
with a stub so CI runs are deterministic and free. There's also one
opt-in live test (gated on `RUN_LIVE_TESTS=1`) that hits the real model.

What we're checking:
  - 422 when the thread has no prospect message
  - snapshot mode is selected for a single-message thread
  - trended mode is selected for a multi-message thread
  - response shape matches AnalyzeResponse for both modes
  - thread_stats reflect the input
  - in snapshot mode, sentiment_trend == 'unknown' and trend_confidence == 0.0
"""

from __future__ import annotations

import os
from typing import Any

import dspy
import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.pipeline import modules as modules_mod
from app.schemas import Message
from tests.fixtures import threads as F


# ---------------------------------------------------------------------------
# Stub analyzer so tests don't need an OPENAI_API_KEY
# ---------------------------------------------------------------------------

class _StubSignal:
    def __init__(self, quote: str, type_: str):
        self.quote = quote
        self.type = type_

    def model_dump(self):
        return {"quote": self.quote, "type": self.type}


class _StubPred:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class StubAnalyzer:
    """Drop-in replacement for ProspectSentimentAnalyzer.

    Returns deterministic, schema-valid predictions so the API layer can be
    exercised without the LLM. Quotes are pulled verbatim from the thread so
    groundedness checks pass.
    """

    def __init__(self):
        pass

    def __call__(self, thread: str, prospect_name: str, mode: str) -> Any:
        first_line = next(
            (ln.strip() for ln in thread.splitlines() if ln.strip() and "---" not in ln),
            "looks good",
        )
        # Pick a short snippet that is verbatim in `thread` so groundedness=1.
        snippet = first_line[:40]
        common = dict(
            overall_sentiment="positive",
            engagement_level="high",
            urgency="medium",
            key_signals=[_StubSignal(quote=snippet, type_="positive_tone")],
            objections_or_concerns=[],
            recommended_next_action="Send a calendar invite to schedule a 20-min call.",
            rationale=f"Stubbed prediction for {prospect_name} in {mode} mode.",
        )
        if mode == "trended":
            return _StubPred(
                **common,
                sentiment_trend="warming",
                trend_confidence=0.85,
            )
        return _StubPred(**common)


@pytest.fixture
def client(monkeypatch):
    # Replace the cached analyzer with our stub. Also short-circuit configure_lm.
    monkeypatch.setattr(app_main, "get_analyzer", lambda: StubAnalyzer())
    monkeypatch.setattr(app_main, "get_model_version", lambda: "stub-model")
    monkeypatch.setattr(app_main, "get_program_version", lambda: "stub-program")
    return TestClient(app_main.app)


# ---- Validation -----------------------------------------------------------

def test_rejects_thread_with_no_prospect_message(client):
    body = {
        "prospect_name": "Nobody",
        "thread": [{"sender": "agent", "body": "First touch"}],
    }
    r = client.post("/analyze-sentiment", json=body)
    assert r.status_code == 422


def test_rejects_empty_thread(client):
    r = client.post("/analyze-sentiment", json={"prospect_name": "x", "thread": []})
    assert r.status_code == 422


# ---- Snapshot mode --------------------------------------------------------

def _post_fixture(client, fixture):
    payload = {
        "prospect_name": fixture["prospect_name"],
        "thread": [
            {
                "sender": m.sender,
                "from": m.from_,
                "body": m.body,
                **({"timestamp": m.timestamp.isoformat()} if m.timestamp else {}),
            }
            for m in fixture["thread"]
        ],
    }
    return client.post("/analyze-sentiment", json=payload)


def test_single_message_returns_snapshot_mode(client):
    r = _post_fixture(client, F.SINGLE_MESSAGE_POSITIVE)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["analysis_mode"] == "snapshot"
    assert data["sentiment_trend"] == "unknown"
    assert data["trend_confidence"] == 0.0
    assert data["thread_stats"]["single_message"] is True
    assert data["thread_stats"]["prospect_messages"] == 1


def test_polite_no_single_message_still_snapshot(client):
    r = _post_fixture(client, F.SINGLE_MESSAGE_POLITE_NO)
    assert r.status_code == 200
    assert r.json()["analysis_mode"] == "snapshot"


# ---- Trended mode ---------------------------------------------------------

def test_multi_message_returns_trended_mode(client):
    r = _post_fixture(client, F.WARMING)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["analysis_mode"] == "trended"
    assert data["sentiment_trend"] in ("warming", "cooling", "steady", "volatile")
    assert 0.0 <= data["trend_confidence"] <= 1.0
    assert data["thread_stats"]["single_message"] is False
    assert data["thread_stats"]["prospect_messages"] >= 2


def test_response_shape_has_all_required_fields(client):
    r = _post_fixture(client, F.OBJECTION_HEAVY)
    assert r.status_code == 200
    data = r.json()
    for f in (
        "prospect_name",
        "analysis_mode",
        "overall_sentiment",
        "sentiment_trend",
        "trend_confidence",
        "engagement_level",
        "urgency",
        "key_signals",
        "objections_or_concerns",
        "recommended_next_action",
        "rationale",
        "thread_stats",
    ):
        assert f in data, f"missing field: {f}"


# ---- Health ---------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Optional live test against the real LLM. Run with:
#     RUN_LIVE_TESTS=1 pytest tests/test_api.py::test_live_warming -q
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Set RUN_LIVE_TESTS=1 and OPENAI_API_KEY to enable.",
)
def test_live_warming():
    from app.dspy_config import configure_lm, get_analyzer

    configure_lm()
    real_analyzer = get_analyzer(force_reload=True)

    # Replace stub with real analyzer for this single test.
    from app import main as app_main_mod

    def _real_get_analyzer():
        return real_analyzer

    app_main_mod.get_analyzer = _real_get_analyzer  # type: ignore
    client = TestClient(app_main_mod.app)

    r = _post_fixture(client, F.WARMING)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["analysis_mode"] == "trended"
    assert data["overall_sentiment"] in ("positive", "neutral")
    assert data["sentiment_trend"] in ("warming", "steady")
