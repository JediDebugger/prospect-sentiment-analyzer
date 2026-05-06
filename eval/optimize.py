"""
Compile (optimize) the analyzer with a DSPy optimizer.

This is the answer to the assignment's:
    "...how would you use DSPy's optimization capabilities to sharpen the
     prompts systematically rather than tweaking them by hand."

What it does
------------
1. Builds a baseline ProspectSentimentAnalyzer (ChainOfThought).
2. Runs `eval/run_eval.py` against the baseline → aggregate composite.
3. Runs `BootstrapFewShot` (cheap, fast) → measures the lift.
4. Runs `MIPROv2` (more expensive, also rewrites instructions) → measures the lift.
5. Saves the best compiled program to `compiled/analyzer.json` so the API
   loads it at startup.

Usage
-----
    python -m eval.optimize                       # bootstrap + miprov2
    python -m eval.optimize --skip-mipro          # bootstrap only (faster)
    python -m eval.optimize --output compiled/v1.json

Notes
-----
- The dataset is small on purpose for this assignment. In production you'd
  feed in hundreds-to-thousands of agent-labeled threads from the in-product
  feedback loop and re-run this nightly/weekly.
- Each optimizer call costs LLM tokens. Budget ~ a few cents on gpt-4o-mini
  for this set; a real run on a real dataset is dollars not pennies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dspy

from app.dspy_config import configure_lm
from app.pipeline.modules import ProspectSentimentAnalyzer
from eval.dataset import all_examples, split
from eval.metrics import composite_metric
from eval.run_eval import evaluate, print_report


def _avg(report: dict) -> float:
    return report["aggregate"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="compiled/analyzer.json")
    p.add_argument("--skip-mipro", action="store_true")
    p.add_argument("--model", default=None)
    args = p.parse_args(argv)

    configure_lm(model=args.model)

    examples = all_examples()
    trainset, devset = split(examples, dev_size=3, seed=7)
    print(f"Loaded {len(examples)} examples (train={len(trainset)}, dev={len(devset)})")

    # ---- 1. Baseline ------------------------------------------------------
    baseline = ProspectSentimentAnalyzer()
    baseline_report = evaluate(baseline, devset)
    print_report("baseline (ChainOfThought, zero-shot)", baseline_report)

    best = baseline
    best_score = _avg(baseline_report)
    best_label = "baseline"

    # ---- 2. BootstrapFewShot ---------------------------------------------
    print("\n>>> Compiling with BootstrapFewShot ...")
    try:
        bfs = dspy.BootstrapFewShot(
            metric=composite_metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=3,
            max_rounds=1,
        )
        bfs_compiled = bfs.compile(student=baseline.deepcopy(), trainset=trainset)
        bfs_report = evaluate(bfs_compiled, devset)
        print_report("BootstrapFewShot", bfs_report)
        if _avg(bfs_report) > best_score:
            best, best_score, best_label = bfs_compiled, _avg(bfs_report), "BootstrapFewShot"
    except Exception as e:
        print(f"BootstrapFewShot failed: {type(e).__name__}: {e}")

    # ---- 3. MIPROv2 -------------------------------------------------------
    if not args.skip_mipro:
        print("\n>>> Compiling with MIPROv2 (this rewrites instructions; takes a few minutes) ...")
        try:
            mipro = dspy.MIPROv2(
                metric=composite_metric,
                auto="light",          # 'light' is appropriate for this dataset size
                num_threads=4,
            )
            mipro_compiled = mipro.compile(
                student=baseline.deepcopy(),
                trainset=trainset,
            )
            mipro_report = evaluate(mipro_compiled, devset)
            print_report("MIPROv2", mipro_report)
            if _avg(mipro_report) > best_score:
                best, best_score, best_label = mipro_compiled, _avg(mipro_report), "MIPROv2"
        except Exception as e:
            print(f"MIPROv2 failed: {type(e).__name__}: {e}")

    # ---- 4. Save best -----------------------------------------------------
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    best.save(str(out))
    print(f"\n✓ Saved best program ({best_label}, score={best_score:.3f}) to {out}")
    print("Restart the API to pick it up (load happens at startup).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
