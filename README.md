# RevReply — Prospect Sentiment API

A FastAPI service that analyzes a sales prospect's sentiment from an email
thread and returns something a sales agent can actually act on. The reasoning
is powered by **DSPy Signatures + a `ChainOfThought` Module**, with an offline
optimization step (`BootstrapFewShot` and `MIPROv2`) that improves the prompt
*systematically with data and a metric* — not by hand-tweaking strings.

```
POST /analyze-sentiment
{
  "prospect_name": "Sara Kim",
  "thread": [
    {"sender": "agent",    "from": "alex@revreply.com", "body": "Hi Sara..."},
    {"sender": "prospect", "from": "sara@helio.io",     "body": "Maybe. Send a one-pager."},
    {"sender": "agent",    "from": "alex@revreply.com", "body": "Attached..."},
    {"sender": "prospect", "from": "sara@helio.io",     "body": "Yes, let's do it Friday."}
  ]
}
```
Returns `analysis_mode`, `overall_sentiment`, `sentiment_trend`,
`trend_confidence`, `engagement_level`, `urgency`, `key_signals` (with
verbatim quote evidence), `objections_or_concerns`, `recommended_next_action`,
`rationale`, and `thread_stats`. Full schema in [`app/schemas.py`](app/schemas.py).

---

## Setup

```bash
cd prospect-sentiment-analyzer
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env       # then put your OPENAI_API_KEY in .env
```

### Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Smoke test:

```bash
curl -s -X POST http://localhost:8000/analyze-sentiment \
  -H 'content-type: application/json' \
  -d @- <<'EOF' | jq .
{
  "prospect_name": "Sara Kim",
  "thread": [
    {"sender":"agent","from":"alex@revreply.com","body":"Hi Sara, would RevReply be a fit?"},
    {"sender":"prospect","from":"sara@helio.io","body":"Maybe — send a one-pager."},
    {"sender":"agent","from":"alex@revreply.com","body":"Attached. Happy to walk through on a call."},
    {"sender":"prospect","from":"sara@helio.io","body":"Yes, let's do Friday afternoon. Include our RevOps lead Jamie too."}
  ]
}
EOF
```

### Run tests

```bash
pytest -q                          # 21 tests; no API key needed (analyzer is stubbed)
RUN_LIVE_TESTS=1 pytest -q         # also runs the live LLM smoke test
```

### Run the eval harness

```bash
python -m eval.run_eval                    # baseline (ChainOfThought)
python -m eval.run_eval --variant zero-shot
python -m eval.run_eval --compiled compiled/analyzer.json
```

### Optimize the program (compile)

```bash
python -m eval.optimize                    # bootstrap + MIPROv2, saves compiled/analyzer.json
python -m eval.optimize --skip-mipro       # bootstrap only (faster, cheaper)
```
Restart `uvicorn` afterwards — the API loads the compiled artifact at startup.

---

## Architecture (5-second tour)

```
                                    ┌──────────────────────┐
   POST /analyze-sentiment ─────►   │  FastAPI route       │
                                    │  app/main.py         │
                                    └─────────┬────────────┘
                                              ▼
                                    ┌──────────────────────┐
                                    │  Pydantic validation │  ← rejects threads
                                    │  app/schemas.py      │     w/ no prospect msg
                                    └─────────┬────────────┘
                                              ▼
                                    ┌──────────────────────┐
                                    │ Deterministic prep:  │  ← the routing
                                    │ strip quotes, count, │     decision is
                                    │ format transcript    │     made HERE,
                                    │ app/pipeline/        │     not in a prompt
                                    │   preprocessing.py   │
                                    └─────────┬────────────┘
                                              ▼
                              ┌───────────────┴───────────────┐
                              │      ProspectSentimentAnalyzer │
                              │      (dspy.Module)             │
                              └───────────────┬───────────────┘
                                  single? │       │ multiple?
                                          ▼       ▼
                          ┌────────────────┐  ┌──────────────────┐
                          │ SnapshotSentiment│  │ TrendedSentiment │
                          │ ChainOfThought   │  │ ChainOfThought   │
                          │ (dspy.Signature) │  │ (dspy.Signature) │
                          └────────────────┘  └──────────────────┘
                                          │       │
                                          ▼       ▼
                                    ┌──────────────────────┐
                                    │  AnalyzeResponse     │
                                    └──────────────────────┘
```

```
prospect-sentiment-analyzer/
├── app/
│   ├── main.py                  FastAPI app, lifespan, /analyze-sentiment, /health
│   ├── schemas.py               Pydantic request + response models
│   ├── dspy_config.py           configures dspy.LM, loads compiled artifact
│   └── pipeline/
│       ├── preprocessing.py     deterministic: stats, format, quote stripping
│       ├── signatures.py        DSPy Signatures (Snapshot + Trended)
│       └── modules.py           ProspectSentimentAnalyzer + analyze()
├── eval/
│   ├── dataset.py               labeled gold examples (built from fixtures)
│   ├── metrics.py               composite metric (sentiment + trend + groundedness + ...)
│   ├── run_eval.py              evaluate any variant; per-component / per-mode slices
│   └── optimize.py              BootstrapFewShot + MIPROv2 → compiled/analyzer.json
├── tests/
│   ├── test_pipeline.py         deterministic tests (no LLM)
│   ├── test_api.py              FastAPI tests w/ stubbed analyzer + 1 live test
│   └── fixtures/threads.py      9 realistic threads incl. edge cases
└── compiled/
    └── analyzer.json            (generated) optimized program loaded at startup
```

---

## Design decisions

### Why two Signatures (`SnapshotSentiment` + `TrendedSentiment`)
The assignment specifically calls out *single-message vs multiple*. I treat
this as a **routing problem**, not a prompt problem:

- The router is deterministic Python in `preprocessing.compute_thread_stats`.
  The LLM is never asked "is this a single-message thread?" — code is.
- Each Signature gets a focused prompt with no hedge clauses ("if there's
  only one message, leave trend null"). Quality goes up.
- The DSPy optimizer specializes each independently — a single-message demo
  teaches nothing about trend detection and vice versa.
- Type safety: `sentiment_trend` is required for `TrendedSentiment` and
  simply absent from `SnapshotSentiment`. The model can't forget it.

### Why the response shape it is
| Field | Reason |
|---|---|
| `analysis_mode` (`snapshot`/`trended`) | Lets the frontend render different UI; lets the eval harness slice by mode. Honest about what we *can* tell the agent. |
| `overall_sentiment` + `sentiment_trend` | Separates *state* from *derivative*. Agents care about direction, not just snapshot. |
| `key_signals[].quote` (verbatim) | Forces grounding. Powers the cheapest hallucination guardrail we have (substring check). |
| `objections_or_concerns` | First-class field because product UX surfaces blockers separately from sentiment. |
| `recommended_next_action` | The only field a busy agent reads first. Required to be specific, not generic. |
| `urgency` | Drives queue prioritization. |
| `thread_stats.single_message` | Makes routing visible in the response. |

### Edge cases I designed for explicitly
- **No prospect message in thread** → 422 at the Pydantic layer. We don't ask the LLM to invent sentiment for a one-sided thread.
- **Out-of-office autoresponder** as the only "prospect message" → routed as snapshot with `prospect_messages = 0` (OOO is filtered for the count); the prompt's docstring also tells the model what to do.
- **One prospect message + many agent messages** → still snapshot. Routes on prospect-msg count, not total count.
- **Quoted/forwarded blocks** in a body → stripped in `preprocessing.strip_quoted_reply` before the model sees the thread, so an inline reply containing the agent's prior email isn't mistaken for two messages.
- **Sarcasm / polite-no** (positive surface tone, negative meaning) → covered by gold examples; the metric punishes "positive" labels here.
- **Mixed signals** (positive tone + blocking concern) → modeled with a `volatile` trend label; partial credit in `trend_match` for direction-correct misses.

### Tradeoffs
- **`ChainOfThought` over `Predict`** — costs more tokens but materially better on nuanced cases (sarcasm, mixed signals). For B2B sales one wrong call is expensive, so worth it.
- **Optimizer runs offline; we serve a compiled artifact** — predictable p95 latency. The optimizer never touches the request path.
- **Structured thread input vs raw paste** — chose structured. Real usage already has messages-as-records; cleaner attribution; reliable single-vs-multi routing.
- **Quote-grounding via substring match** — fast, deterministic, no extra LLM call. Imperfect (won't catch a paraphrased-but-true quote) but the cost/value is excellent.
- **Composite metric** instead of pure sentiment accuracy — if we only optimize for the label, MIPRO would produce a prompt great at the label and terrible at next-action wording. Composite keeps the optimizer honest across all the fields an agent reads.
- **Async endpoint with `asyncio.to_thread`** — the DSPy LLM call is synchronous, so we run it in a thread pool. This keeps the event loop free under concurrent load without adding full async-LLM complexity.
- **`forward()` infers `mode` when not passed** — DSPy optimizers only pass Signature input fields (`thread`, `prospect_name`), not custom routing args. Making `mode` optional with auto-inference from the formatted transcript means the module works identically in both the API path and the optimizer path.

---

## Evaluation & optimization (the part that matters)

### How I'd evaluate it
1. Hand-label a small **gold set** spanning the failure modes (`tests/fixtures/threads.py` + `eval/dataset.py`): single-positive, single-polite-no, single-neutral-question, warming, cooling, ghosting, objection-heavy, sarcasm, mixed/volatile.
2. **Composite metric** in [`eval/metrics.py`](eval/metrics.py):

   | Component | Weight | What it measures |
   |---|---|---|
   | `sentiment_match` | 0.30 | exact match on `overall_sentiment` |
   | `trend_match` | 0.20 | trended only — partial credit for direction-correct misses |
   | `groundedness` | 0.20 | every `key_signals.quote` is a verbatim substring of the thread |
   | `structural` | 0.10 | all required fields present |
   | `next_action_relevance` | 0.10 | recommended action mentions a known-good keyword |
   | `objection_match` | 0.10 | did we surface an objection when one exists? |

3. **Slice metrics** by mode (snapshot/trended) and by sentiment class.  Aggregate-only numbers hide the failures we most care about.
4. **Per-example printout** in `run_eval.py` so a regression is diagnosable, not just measurable.

### How I'd optimize it (DSPy)
The flow in [`eval/optimize.py`](eval/optimize.py):

```
ChainOfThought baseline                    ─►  measure
  │
  ├─► BootstrapFewShot (auto-pick demos)   ─►  measure
  │
  └─► MIPROv2 (rewrite instruction +       ─►  measure
              demos jointly, Bayesian search)
                                           ─►  save best to compiled/analyzer.json
```
- `BootstrapFewShot` is fast and cheap. Often gets you 80% of the lift.
- `MIPROv2` *also* rewrites the Signature's instruction string using an LLM as a "prompt engineer" and runs a Bayesian search over (instruction, demos). This is the actual answer to *"sharpen prompts systematically rather than tweaking by hand."*
- We keep the best variant by composite score and save it. The API loads it at startup. No optimization in the request path.

#### Measured results (9-example gold set, `gpt-4o-mini`)

| Variant | composite | sentiment_match | trend_match | groundedness | objection_match |
|---|---|---|---|---|---|
| `ChainOfThought` (baseline) | 0.798 | 0.667 | 0.883 | 1.000 | 0.778 |
| + `BootstrapFewShot` compiled | **0.890** | **0.889** | 0.883 | 1.000 | 0.778 |

The compiled artifact (`compiled/analyzer.json`) is loaded at startup. Key wins:
- Mark Chen "polite-no" (hardest snapshot): 0.500 → **0.875** — the bootstrapped demos taught the model what a polite rejection looks like.
- Priya Patel neutral-question: 0.625 → **1.000** — stops over-reading interest into a factual question.

> Rerun `python -m eval.optimize` on your own key to reproduce. MIPROv2 also runs (instruction rewriting + Bayesian search over 20 trials) but on this small dataset BootstrapFewShot captures most of the lift.

---

## How I'd make this genuinely accurate at scale

The 9-example gold set is the seed; the system gets accurate by closing the
loop with real agent feedback.

1. **In-product feedback loop.** Every prediction is logged with `model_version`, `program_version`, and a hash of the thread. Agents click 👍/👎 (or correct one field) inline. Those flagged predictions flow into a labeling queue and become new gold examples.
2. **Weekly automated re-compile.** A scheduled job (Airflow / cron) runs `MIPROv2` against the growing labeled set, evaluates the new artifact on a held-out dev split, and **A/B tests it behind a feature flag** before promoting to default. This is what makes "improvement" mean *measured improvement*, not vibes.
3. **Drift monitors.** Distribution of predicted classes per week (alert if `negative` rate doubles overnight), groundedness pass-rate (catch degradation in evidence-quoting), latency p95, refusal rate.
4. **Slice dashboards.** Track the composite metric by industry / seller / thread length. Catches *systematic* failures (e.g. "we're 15 points worse on healthcare prospects") that aggregate accuracy hides.
5. **Active labeling.** Surface low-confidence + high-value-account predictions for human labeling first — best label-spend ROI.
6. **Specialize as we learn.** As patterns emerge, factor out sub-modules (`objection_extractor`, `next_action_recommender`) into their own Signatures with their own metrics. DSPy's composability means each can be optimized independently.
7. **Multi-LLM evaluation.** Same Signatures → swap the underlying model (`gpt-4o-mini` → `claude-sonnet-4` → a fine-tuned local model) and pick the best by composite score on the dev set. The Signature is the API; the model is a hyperparameter.
8. **Distillation.** Once the system is stable, use `BootstrapFinetune` to distill into a smaller/cheaper model for the request path; keep the big model only in optimization runs.

---
