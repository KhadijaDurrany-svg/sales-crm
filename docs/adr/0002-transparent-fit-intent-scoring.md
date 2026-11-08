# ADR 0002: Transparent fit/intent scoring, not a learned ranker

## Status
Accepted.

## Context
Lead prioritization could be framed as a supervised learning-to-rank problem.
But in a CRM setting the score must be explainable to a sales team ("why is this
lead near the top?"), auditable by ops, and stable without a training pipeline.
A learned model trained on the same synthetic generator would also mostly
measure how well we can recover our own generative weights, which is not an
honest test of a business rule set.

## Decision
- The score is a transparent, hand-set blend of a **fit** sub-score (ICP match:
  target industry, ideal company-size band, title seniority) and an **intent**
  sub-score (recency-decayed, weighted engagement, squashed to `[0, 1]`).
- Every component is returned in `ScoreBreakdown`, so any ranking decision can
  be explained field by field.
- The scoring weights are deliberately *not* fitted to the generator. They are
  plausible business defaults that differ from the latent weights driving
  conversion, so the engine is a realistic, imperfect ranker rather than an
  oracle.

## Consequences
- Precision@k and NDCG land well above the baselines but clearly below 1.0,
  which is the honest ceiling given stochastic conversion and mismatched weights.
- Comparison baselines (`random`, `recency`, `raw_score`) isolate the value of
  the fit + intent blend over naive CRM-default orderings.
- This is explicitly *prioritization*, not uplift or treatment-effect modelling:
  we rank by predicted propensity/fit, we do not estimate the causal effect of an
  intervention.
