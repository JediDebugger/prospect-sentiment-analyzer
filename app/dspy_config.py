"""
DSPy / LM configuration.

Single place where:
  - the global LM is configured (one call, all modules use it)
  - the analyzer is constructed
  - the compiled (optimized) artifact is loaded if present

Loading the compiled artifact at startup means the optimization step never
runs in the request path — predictable p95 latency.
"""

from __future__ import annotations

import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

from app.pipeline.modules import ProspectSentimentAnalyzer

load_dotenv(override=False)

DEFAULT_MODEL = os.getenv("REVREPLY_MODEL", "openai/gpt-4o-mini")
DEFAULT_COMPILED_PATH = os.getenv("REVREPLY_COMPILED_PATH", "compiled/analyzer.json")

_LM_CONFIGURED = False
_ANALYZER: ProspectSentimentAnalyzer | None = None
_PROGRAM_VERSION: str = "uncompiled-baseline"


def configure_lm(model: str | None = None, **lm_kwargs) -> str:
    """Configure DSPy's global LM. Idempotent.

    Returns the model identifier actually used.
    """
    global _LM_CONFIGURED
    chosen = model or DEFAULT_MODEL
    # temperature=0 because we want deterministic, repeatable classifications.
    # max_tokens controls *output* length; sentiment payload fits comfortably in 1024.
    defaults = {"temperature": 0, "max_tokens": 1024}
    defaults.update(lm_kwargs)
    lm = dspy.LM(chosen, **defaults)
    dspy.configure(lm=lm)
    _LM_CONFIGURED = True
    return chosen


def get_analyzer(
    *, compiled_path: str | None = None, force_reload: bool = False
) -> ProspectSentimentAnalyzer:
    """Return the (optionally compiled) analyzer instance.

    On first call:
      - configures LM if not already configured
      - constructs a fresh ProspectSentimentAnalyzer
      - if a compiled artifact exists at `compiled_path`, loads it
    """
    global _ANALYZER, _PROGRAM_VERSION
    if _ANALYZER is not None and not force_reload:
        return _ANALYZER

    if not _LM_CONFIGURED:
        configure_lm()

    analyzer = ProspectSentimentAnalyzer()
    path = Path(compiled_path or DEFAULT_COMPILED_PATH)
    if path.exists():
        try:
            analyzer.load(str(path))
            _PROGRAM_VERSION = f"compiled:{path.name}"
        except Exception as e:  # pragma: no cover - defensive
            # Fall back to uncompiled baseline if the artifact is incompatible.
            print(f"[dspy_config] Failed to load compiled artifact at {path}: {e}")
            _PROGRAM_VERSION = "uncompiled-baseline (load-failed)"
    else:
        _PROGRAM_VERSION = "uncompiled-baseline"

    _ANALYZER = analyzer
    return analyzer


def get_program_version() -> str:
    return _PROGRAM_VERSION


def get_model_version() -> str:
    return DEFAULT_MODEL
