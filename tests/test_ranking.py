from __future__ import annotations

import numpy as np

from leadsift.generate import generate_leads
from leadsift.metrics import ndcg_at_k, precision_at_k
from leadsift.ranking import (
    rank_leads,
    rank_random,
    rank_raw_score,
    rank_recency,
)


def test_rank_returns_permutation() -> None:
    leads = generate_leads(seed=1, n=40).leads
    order = rank_leads(leads, 180)
    assert sorted(order.tolist()) == list(range(40))


def test_recency_orders_by_last_activity() -> None:
    leads = generate_leads(seed=2, n=40).leads
    rng = np.random.default_rng(0)
    order = rank_recency(leads, rng)
    days = [leads[i].last_activity_day for i in order]
    assert days == sorted(days, reverse=True)


def test_raw_score_orders_by_total_events() -> None:
    leads = generate_leads(seed=3, n=40).leads
    rng = np.random.default_rng(0)
    order = rank_raw_score(leads, rng)

    def total(i: int) -> int:
        lead = leads[i]
        return lead.email_opens + lead.email_clicks + lead.site_visits + lead.pricing_views + int(lead.demo_requested)

    totals = [total(i) for i in order]
    assert totals == sorted(totals, reverse=True)


def test_random_is_seed_reproducible() -> None:
    leads = generate_leads(seed=4, n=40).leads
    a = rank_random(leads, np.random.default_rng(123))
    b = rank_random(leads, np.random.default_rng(123))
    assert a.tolist() == b.tolist()


def test_fit_intent_beats_random_on_held_out_batch() -> None:
    # Single-batch sanity check (the full statistical claim lives in the bench).
    leads = generate_leads(seed=11, n=300).leads
    labels = generate_leads(seed=11, n=300).labels
    rng = np.random.default_rng(0)
    fit_order = rank_leads(leads, 180, rng=rng)
    rand_order = rank_random(leads, rng)
    k = 30
    assert precision_at_k(labels[fit_order], k) > precision_at_k(labels[rand_order], k)
    assert ndcg_at_k(labels[fit_order], k) > ndcg_at_k(labels[rand_order], k)
