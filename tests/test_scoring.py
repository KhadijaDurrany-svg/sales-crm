from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from leadsift.generate import generate_leads
from leadsift.schemas import Lead
from leadsift.scoring import (
    default_weights,
    fit_score,
    intent_score,
    score_lead,
    score_many,
)


def _lead(**over: object) -> Lead:
    base = dict(
        lead_id="L1",
        first_name="A",
        last_name="B",
        email="a.b@acme.com",
        company="Acme Inc",
        phone="+1-555-123-4567",
        title="Director",
        industry="SaaS",
        employee_count=200,
        created_day=10,
        last_activity_day=170,
        email_opens=0,
        email_clicks=0,
        site_visits=0,
        pricing_views=0,
        demo_requested=False,
        converted=False,
    )
    base.update(over)
    return Lead(**base)  # type: ignore[arg-type]


def test_target_industry_scores_higher_fit() -> None:
    w = default_weights()
    on = fit_score(_lead(industry="SaaS"), w)[0]
    off = fit_score(_lead(industry="Retail"), w)[0]
    assert on > off


def test_ideal_size_scores_full_size_fit() -> None:
    w = default_weights()
    assert fit_score(_lead(employee_count=500), w)[2] == 1.0
    assert fit_score(_lead(employee_count=5), w)[2] < 1.0
    assert fit_score(_lead(employee_count=100000), w)[2] < 1.0


def test_demo_request_raises_intent() -> None:
    w = default_weights()
    base = _lead(last_activity_day=180)
    hi = intent_score(base.model_copy(update={"demo_requested": True}), w, 180)[0]
    lo = intent_score(base, w, 180)[0]
    assert hi > lo


def test_recency_decays_intent() -> None:
    w = default_weights()
    lead = _lead(email_clicks=5, pricing_views=2, last_activity_day=180)
    recent = intent_score(lead, w, 180)[0]
    stale = intent_score(lead.model_copy(update={"last_activity_day": 180 - 60}), w, 180)[0]
    assert recent > stale


def test_total_blends_fit_and_intent() -> None:
    w = default_weights()
    b = score_lead(_lead(industry="SaaS", employee_count=500, demo_requested=True), w, 180)
    expected = w.fit_weight * b.fit + (1 - w.fit_weight) * b.intent
    assert abs(b.total - expected) < 1e-9


def test_score_many_aligned() -> None:
    w = default_weights()
    leads = generate_leads(seed=5, n=25).leads
    scores = score_many(leads, w, 180)
    assert len(scores) == 25
    assert scores.min() >= 0.0


@given(
    opens=st.integers(min_value=0, max_value=100),
    clicks=st.integers(min_value=0, max_value=100),
    visits=st.integers(min_value=0, max_value=100),
    pricing=st.integers(min_value=0, max_value=100),
    demo=st.booleans(),
    days_ago=st.integers(min_value=0, max_value=180),
)
@settings(max_examples=80, deadline=None)
def test_all_components_bounded_unit_interval(
    opens: int, clicks: int, visits: int, pricing: int, demo: bool, days_ago: int
) -> None:
    w = default_weights()
    lead = _lead(
        email_opens=opens,
        email_clicks=clicks,
        site_visits=visits,
        pricing_views=pricing,
        demo_requested=demo,
        last_activity_day=180 - days_ago,
    )
    b = score_lead(lead, w, 180)
    for value in (b.fit, b.intent, b.total, b.industry_fit, b.size_fit, b.title_fit, b.recency_factor):
        assert 0.0 <= value <= 1.0 + 1e-9


@given(extra=st.integers(min_value=1, max_value=50))
@settings(max_examples=40, deadline=None)
def test_intent_monotonic_in_pricing_views(extra: int) -> None:
    w = default_weights()
    low = _lead(pricing_views=0, last_activity_day=180)
    high = _lead(pricing_views=extra, last_activity_day=180)
    assert intent_score(high, w, 180)[0] >= intent_score(low, w, 180)[0]
