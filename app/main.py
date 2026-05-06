"""
FastAPI app exposing POST /analyze-sentiment.

Run:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.dspy_config import get_analyzer, get_model_version, get_program_version
from app.pipeline.modules import analyze
from app.schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger("revreply.sentiment")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Eagerly load LM + analyzer (and compiled artifact if present) so the first
    # request doesn't pay the configuration cost.
    get_analyzer()
    logger.info(
        "Analyzer ready: model=%s program=%s",
        get_model_version(),
        get_program_version(),
    )
    yield


app = FastAPI(
    title="RevReply Prospect Sentiment API",
    description=(
        "Analyzes a sales prospect's sentiment from an email thread.\n\n"
        "Routes single-message threads to a snapshot analyzer (no trend) and "
        "multi-message threads to a trended analyzer. The DSPy program may be "
        "served as an optimized (compiled) artifact; see eval/optimize.py."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": get_model_version(),
        "program": get_program_version(),
    }


@app.post("/analyze-sentiment", response_model=AnalyzeResponse)
async def analyze_sentiment(req: AnalyzeRequest) -> AnalyzeResponse:
    started = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]
    try:
        analyzer = get_analyzer()
        # Run the (synchronous) DSPy call in a thread pool so we don't block
        # the event loop — important when multiple requests arrive concurrently.
        response = await asyncio.to_thread(
            analyze,
            analyzer,
            thread=req.thread,
            prospect_name=req.prospect_name,
            model_version=get_model_version(),
            program_version=get_program_version(),
        )
    except HTTPException:
        raise
    except Exception as e:
        # Structured log with request_id for traceability.
        logger.exception("analyze_sentiment failed: request_id=%s %s", request_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed. ref={request_id}",
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Lightweight prediction log — in production this is the seed for the
    # feedback loop (label later, re-compile weekly).
    logger.info(
        "predict mode=%s sentiment=%s trend=%s urgency=%s elapsed_ms=%.0f "
        "n_prospect=%d n_agent=%d",
        response.analysis_mode,
        response.overall_sentiment,
        response.sentiment_trend,
        response.urgency,
        elapsed_ms,
        response.thread_stats.prospect_messages,
        response.thread_stats.agent_messages,
    )
    return response
