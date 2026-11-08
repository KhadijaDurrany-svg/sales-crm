# leadsift

**Offline lead prioritization and CRM record deduplication, with honest benchmarks.**

`leadsift` is a small, dependency-light engine for two everyday CRM-intelligence
jobs:

1. **Prioritize leads** — score and rank leads by a transparent *fit + intent*
   signal so a sales team works the most promising records first.
2. **Clean the data** — detect duplicate and dirty records (typo'd names,
   plus-tagged emails, reformatted phones, dropped company suffixes) with
   multi-signal fuzzy matching.

Everything runs **offline**: no network, no API keys, no model downloads. All
data is seeded synthetic, so every number below is reproducible from scratch,
and the benchmark reports **only measured values** with bootstrap confidence
intervals and a paired permutation test.

## How this differs from a typical "sales CRM agent"

Many sales-CRM projects are online LLM agents: they call vendor APIs, draft
outreach, and orchestrate multi-agent workflows behind an authenticated service.
`leadsift` is deliberately the opposite end of the stack — a small, transparent,
**offline analytics core**:

- No LLM, no agents, no network, no secrets. Pure, inspectable scoring and
  record-linkage logic you can read end to end.
- Every priority score is explainable component by component (`ScoreBreakdown`).
- The focus is **prioritization and data quality**, measured against baselines —
  not outreach generation, and **not** uplift / treatment-effect modelling (it
  ranks by fit and propensity; it does not estimate the causal effect of an
  intervention).

## Results

Measured by running `leadsift bench` on this repository. Configuration:
40 seeded batches, 200 leads per batch for ranking and 120 base records per
batch for dedup; bootstrap = 2000 resamples, permutation = 10,000.
Environment recorded in [`benchmarks/manifest.json`](benchmarks/manifest.json)
(Python 3.13, NumPy 2.5). Reproduce with `leadsift bench`.

### Lead ranking — relevance is `converted == True`, metrics @ k = 20

| Method | Precision@20 (95% CI) | NDCG@20 (95% CI) |
|---|---|---|
| **fit_intent** (this engine) | **0.672** [0.641, 0.704] | **0.699** [0.665, 0.733] |
| raw_score (unweighted events) | 0.486 [0.447, 0.525] | 0.500 [0.462, 0.537] |
| recency (last activity) | 0.444 [0.410, 0.478] | 0.457 [0.416, 0.496] |
| random | 0.338 [0.311, 0.364] | 0.334 [0.300, 0.370] |

The fit + intent engine beats the strongest baseline (`raw_score`) by
**+0.186 Precision@20** and **+0.199 NDCG@20**. Paired permutation test over the
40 batches: **0 of 10,000** sign-flips reached the observed difference
(p ≈ 1.0e-4) for both metrics.

### Record deduplication — duplicate-pair detection, per-batch mean

| Method | Precision | Recall | F1 (95% CI) |
|---|---|---|---|
| **fuzzy** (blocking + multi-signal) | 0.950 | 0.823 | **0.881** [0.869, 0.893] |
| exact_match (identical normalized email) | 0.989 | 0.599 | 0.744 [0.726, 0.760] |

Fuzzy matching recovers far more true duplicates (recall 0.823 vs 0.599) at a
small precision cost, for **+0.137 F1**. Paired permutation test over the
40 batches: **0 of 10,000** sign-flips reached the observed F1 gap (p ≈ 1.0e-4).

> These are synthetic-data results and a ceiling, not a product SLA. See
> **Known gaps** below.

## Install

Requires Python ≥ 3.10.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"      # dev extras: pytest, hypothesis, ruff
```

## Use

```bash
# Run the full benchmark; writes benchmarks/results.json + manifest.json
leadsift bench

# Same, but print only (no files written)
leadsift bench --no-write

# Inspect a scored/ranked batch and a dedup run
leadsift demo --seed 7 --n 200 --top 10
```

As a library:

```python
from leadsift.generate import generate_leads, generate_dedup_records
from leadsift.scoring import default_weights, score_lead
from leadsift.ranking import rank_leads
from leadsift.dedup import detect_duplicates

ds = generate_leads(seed=7, n=200)
weights = default_weights()

order = rank_leads(ds.leads, ds.horizon_day, weights)  # best-first indices
top = ds.leads[order[0]]
print(score_lead(top, weights, ds.horizon_day))  # explainable breakdown

dd = generate_dedup_records(seed=7, n_base=120)
pairs = detect_duplicates(dd.records)  # {(left_id, right_id), ...}
```

## Project layout

```
src/leadsift/
  schemas.py     Pydantic v2 models (Lead, ScoringWeights, ScoreBreakdown, ...)
  generate.py    Seeded synthetic leads + near-duplicate records; dataset checksum
  scoring.py     Transparent fit + intent scoring with per-component breakdown
  ranking.py     fit_intent ranker + random / recency / raw_score baselines
  dedup.py       Normalization, Levenshtein, blocking, multi-signal matcher, exact baseline
  metrics.py     precision@k, NDCG@k, P/R/F1, bootstrap CI, paired permutation test
  bench.py       Runs both suites, assembles the measured results dict
  cli.py         argparse CLI: `bench`, `demo`
tests/           65 tests (pytest + Hypothesis)
benchmarks/      results.json + manifest.json (written by `leadsift bench`)
docs/adr/        Architecture decision records
```

## How the numbers are produced (and kept honest)

- **Seeded synthetic data.** `numpy.random.default_rng(seed)` drives everything;
  conversion and duplicate labels come from the same generator, so ground truth
  always exists. A SHA-256 checksum of the generated data is written into the
  results and manifest.
- **Stochastic, mismatched by design.** Conversion is a noisy logistic function
  of latent fit and intent, and the scorer's weights are *not* fitted to the
  generator — so no ranker can be perfect and the baselines are real competitors.
- **Statistics.** Per-batch metrics feed a percentile bootstrap (95% CI) and a
  two-sided paired permutation test; p-values are add-one smoothed, so the
  smallest reportable value with 10,000 permutations is ≈ 1.0e-4.
- **Perfect-score guard.** A test (`test_fuzzy_is_not_perfect`) fails if dedup F1
  reaches ≥ 0.99, forcing the generator to stay non-trivial.

## Known gaps

- **Synthetic only.** Results characterize this generator's distribution, not any
  real CRM. Absolute numbers will differ on real data; treat them as relative
  comparisons between methods under a controlled, reproducible setup.
- **Fixed ICP and weights.** Scoring weights and the target-industry set are
  sensible defaults, not tuned per organization. They are inspectable and meant
  to be edited (`ScoringWeights`, `DedupConfig`).
- **Dedup is pairwise, not clustered.** The detector flags duplicate *pairs*; it
  does not resolve them into transitive clusters or pick a surviving golden
  record. Blocking can also miss a duplicate when every blocking field drifts at
  once (the "hard" duplicates the generator injects on purpose).
- **No learned ranker.** By choice (see ADR 0002) the ranker is a transparent
  rule set. A learned model could do better on real data; it is out of scope
  here and would mostly measure recovery of the generator's own weights.
- **Not causal.** This ranks by fit/propensity. It does not estimate treatment
  effects or optimize a budgeted intervention.

## Development

```bash
ruff check . && ruff format --check .   # lint + format
pytest -q                               # 65 tests
```

CI runs lint, the full test suite, and a benchmark smoke run on Python 3.10,
3.11, and 3.12.

## License

MIT — see [LICENSE](LICENSE).
