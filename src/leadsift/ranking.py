"""Lead ranking strategies: the fit/intent engine plus comparison baselines.

Each ranker maps a batch of leads to an ordering (indices, best first). The
benchmark compares the fit/intent engine against:

* ``random``       -- a seeded shuffle (floor),
* ``recency``      -- most recently active first (a common CRM default view),
* ``raw_score``    -- unweighted total engagement count, ignoring fit and
  recency (the naive "lead score" many CRMs surface).
"""

from __future__ import annotations

import numpy as np

from leadsift.schemas import Lead, ScoringWeights
from leadsift.scoring import default_weights, score_many


def _order_desc(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stable descending order with random tie-breaks (so ties are not biased)."""
    jitter = rng.random(len(values)) * 1e-9
    return np.argsort(-(values + jitter), kind="stable")


def rank_leads(
    leads: list[Lead],
    horizon_day: int,
    weights: ScoringWeights | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Rank by the transparent fit/intent score (descending). Returns indices."""
    rng = rng if rng is not None else np.random.default_rng(0)
    weights = weights if weights is not None else default_weights()
    scores = score_many(leads, weights, horizon_day)
    return _order_desc(scores, rng)


def rank_random(leads: list[Lead], rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(len(leads))


def rank_recency(leads: list[Lead], rng: np.random.Generator) -> np.ndarray:
    values = np.array([lead.last_activity_day for lead in leads], dtype=float)
    return _order_desc(values, rng)


def rank_raw_score(leads: list[Lead], rng: np.random.Generator) -> np.ndarray:
    """Naive unweighted engagement count: no fit, no recency, no event weights."""
    values = np.array(
        [
            lead.email_opens + lead.email_clicks + lead.site_visits + lead.pricing_views + int(lead.demo_requested)
            for lead in leads
        ],
        dtype=float,
    )
    return _order_desc(values, rng)


RANKERS = {
    "fit_intent": "the transparent fit + intent engine",
    "raw_score": "unweighted engagement count",
    "recency": "most recently active first",
    "random": "seeded shuffle",
}
