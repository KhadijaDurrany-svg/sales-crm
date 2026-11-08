"""Transparent fit + intent lead scoring.

The score has two inspectable halves:

* **Fit** -- how well the lead matches the ideal customer profile (industry in
  the target set, company size inside the ideal band, title seniority).
* **Intent** -- recency-decayed, weighted engagement signals, squashed to
  [0, 1] so a few hot signals do not dominate unboundedly.

Both halves are in [0, 1]; ``total`` blends them with ``fit_weight``. Every
component is returned in :class:`~leadsift.schemas.ScoreBreakdown` so a score
can always be explained field by field.
"""

from __future__ import annotations

import numpy as np

from leadsift.schemas import Lead, ScoreBreakdown, ScoringWeights


def default_weights() -> ScoringWeights:
    """Default ICP targeting the SaaS / Fintech / Healthcare segment."""
    return ScoringWeights(target_industries=("SaaS", "Fintech", "Healthcare"))


def _size_fit(employees: int, lo: int, hi: int) -> float:
    if employees <= 0:
        return 0.0
    if lo <= employees <= hi:
        return 1.0
    if employees < lo:
        return float(max(0.0, 1.0 - (np.log10(lo) - np.log10(employees))))
    return float(max(0.0, 1.0 - (np.log10(employees) - np.log10(hi))))


def fit_score(lead: Lead, weights: ScoringWeights) -> tuple[float, float, float, float]:
    """Return ``(fit, industry_fit, size_fit, title_fit)``, each in [0, 1]."""
    industry_fit = 1.0 if lead.industry in weights.target_industries else 0.0
    size = _size_fit(lead.employee_count, weights.ideal_min_employees, weights.ideal_max_employees)
    title_fit = float(weights.title_seniority.get(lead.title, 0.3))
    fit = weights.w_industry * industry_fit + weights.w_size * size + weights.w_title * title_fit
    return fit, industry_fit, size, title_fit


def intent_score(lead: Lead, weights: ScoringWeights, horizon_day: int) -> tuple[float, float]:
    """Return ``(intent, recency_factor)``.

    ``horizon_day`` is the "today" the batch was cut on; recency decays with a
    configurable half-life relative to the lead's last activity.
    """
    days_since = max(0, horizon_day - lead.last_activity_day)
    recency = float(0.5 ** (days_since / weights.recency_half_life))
    raw = (
        weights.w_open * lead.email_opens
        + weights.w_click * lead.email_clicks
        + weights.w_visit * lead.site_visits
        + weights.w_pricing * lead.pricing_views
        + weights.w_demo * (1.0 if lead.demo_requested else 0.0)
    )
    weighted = raw * recency
    intent = weighted / (weighted + weights.intent_saturation)
    return float(intent), recency


def score_lead(lead: Lead, weights: ScoringWeights, horizon_day: int) -> ScoreBreakdown:
    """Compute the full, explainable priority score for one lead."""
    fit, industry_fit, size_fit, title_fit = fit_score(lead, weights)
    intent, recency = intent_score(lead, weights, horizon_day)
    total = weights.fit_weight * fit + (1.0 - weights.fit_weight) * intent
    return ScoreBreakdown(
        fit=fit,
        intent=intent,
        total=total,
        industry_fit=industry_fit,
        size_fit=size_fit,
        title_fit=title_fit,
        recency_factor=recency,
    )


def score_many(leads: list[Lead], weights: ScoringWeights, horizon_day: int) -> np.ndarray:
    """Vector of ``total`` scores aligned with ``leads``."""
    return np.array([score_lead(lead, weights, horizon_day).total for lead in leads], dtype=float)
