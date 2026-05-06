"""
Run evaluation against the labeled gold set.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --variant zero-shot
    python -m eval.run_eval --compiled compiled/analyzer.json
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from typing import Any

import dspy

from app.dspy_config import configure_lm
from app.pipeline.modules import ProspectSentimentAnalyzer
from app.pipeline.signatures import SnapshotSentiment, TrendedSentiment
from eval.dataset import all_examples
from eval.metrics import composite_score


def build_analyzer(variant: str, compiled_path: str | None) -> ProspectSentimentAnalyzer:
    analyzer = ProspectSentimentAnalyzer()
    if variant == "zero-shot":
        analyzer.snapshot = dspy.Predict(SnapshotSentiment)
        analyzer.trended = dspy.Predict(TrendedSentiment)
    if compiled_path:
        analyzer.load(compiled_path)
    return analyzer


def predict_for(example: dspy.Example, analyzer: ProspectSentimentAnalyzer) -> Any:
    """Run through the module's forward() — same path the optimizer uses."""
    return analyzer(thread=example.thread, prospect_name=example.prospect_name)


def evaluate(
    analyzer: ProspectSentimentAnalyzer,
    examples: list[dspy.Example],
    *,
    verbose: bool = True,
) -> dict:
    per_example: list[dict] = []
    for i, ex in enumerate(examples, 1):
        if verbose:
            print(
                f"  [{i}/{len(examples)}] {ex.prospect_name} ({ex.mode})...",
                end=" ",
                flush=True,
            )
        try:
            pred = predict_for(ex, analyzer)
            scores = composite_score(ex, pred)
            scores["_error"] = None
        except Exception as e:
            scores = {
                "_total": 0.0,
                "_mode": ex.mode,
                "_error": f"{type(e).__name__}: {e}",
            }
            pred = None
        if verbose:
            err = scores.get("_error")
            status = f"score={scores['_total']:.3f}" if not err else f"ERR: {err}"
            print(status, flush=True)
        per_example.append({
            "prospect_name": ex.prospect_name,
            "mode": ex.mode,
            "expected_sentiment": ex.overall_sentiment,
            "predicted_sentiment": getattr(pred, "overall_sentiment", None),
            "expected_trend": getattr(ex, "sentiment_trend", None),
            "predicted_trend": getattr(pred, "sentiment_trend", None),
            "scores": scores,
        })

    totals = [r["scores"]["_total"] for r in per_example]
    component_keys = sorted({k for r in per_example for k in r["scores"] if not k.startswith("_")})
    component_avgs: dict[str, float] = {}
    for k in component_keys:
        vals = [r["scores"].get(k) for r in per_example if isinstance(r["scores"].get(k), (int, float))]
        if vals:
            component_avgs[k] = statistics.mean(vals)

    by_mode = defaultdict(list)
    for r in per_example:
        by_mode[r["mode"]].append(r["scores"]["_total"])
    mode_avgs = {m: statistics.mean(v) for m, v in by_mode.items() if v}

    return {
        "n": len(per_example),
        "aggregate": statistics.mean(totals) if totals else 0.0,
        "by_component": component_avgs,
        "by_mode": mode_avgs,
        "per_example": per_example,
    }


def print_report(label: str, report: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"n = {report['n']}    aggregate = {report['aggregate']:.3f}")
    print("\nBy component:")
    for k, v in sorted(report["by_component"].items()):
        print(f"  {k:<24s} {v:.3f}")
    print("\nBy mode:")
    for m, v in sorted(report["by_mode"].items()):
        print(f"  {m:<10s} {v:.3f}")
    print("\nPer example:")
    print(f"  {'prospect':<18s} {'mode':<10s} {'gold->pred sentiment':<28s} "
          f"{'gold->pred trend':<26s} {'score':>6s}")
    for r in report["per_example"]:
        sent = f"{r['expected_sentiment']}->{r['predicted_sentiment']}"
        trend = f"{r['expected_trend']}->{r['predicted_trend']}"
        score = r["scores"]["_total"]
        err = r["scores"].get("_error")
        line = (
            f"  {r['prospect_name'][:17]:<18s} {r['mode']:<10s} {sent:<28s} "
            f"{trend:<26s} {score:>6.3f}"
        )
        if err:
            line += f"  ERR: {err}"
        print(line)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["zero-shot", "cot"], default="cot")
    p.add_argument("--compiled", default=None, help="Path to compiled analyzer.json")
    p.add_argument("--model", default=None, help="Override REVREPLY_MODEL")
    args = p.parse_args(argv)

    configure_lm(model=args.model)
    analyzer = build_analyzer(args.variant, args.compiled)

    examples = all_examples()
    label_bits = [args.variant]
    if args.compiled:
        label_bits.append(f"compiled={args.compiled}")
    report = evaluate(analyzer, examples)
    print_report(" ".join(label_bits), report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
