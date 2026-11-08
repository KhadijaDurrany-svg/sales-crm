from __future__ import annotations

import pytest
from pydantic import ValidationError

from leadsift.schemas import Lead, RecordPair, ScoreBreakdown, ScoringWeights


def _lead(**over: object) -> Lead:
    base = dict(
        lead_id="L1",
        first_name="Aisha",
        last_name="Ahmed",
        email="aisha.ahmed@acme.com",
        company="Acme Inc",
        phone="+1-555-123-4567",
        title="Director",
        industry="SaaS",
        employee_count=200,
        created_day=10,
        last_activity_day=170,
        email_opens=3,
        email_clicks=1,
        site_visits=2,
        pricing_views=0,
        demo_requested=False,
        converted=True,
    )
    base.update(over)
    return Lead(**base)  # type: ignore[arg-type]


def test_lead_is_frozen() -> None:
    lead = _lead()
    with pytest.raises(ValidationError):
        lead.email_opens = 99  # type: ignore[misc]


def test_lead_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        _lead(unexpected="x")


def test_lead_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        _lead(email_opens=-1)


def test_lead_rejects_negative_employee_count() -> None:
    with pytest.raises(ValidationError):
        _lead(employee_count=-5)


def test_scoring_weights_defaults() -> None:
    w = ScoringWeights(target_industries=("SaaS",))
    assert w.ideal_min_employees == 50
    assert w.ideal_max_employees == 5000
    assert 0.0 < w.fit_weight < 1.0
    assert w.title_seniority["CEO"] == 1.0


def test_fit_component_weights_sum_to_one() -> None:
    w = ScoringWeights(target_industries=("SaaS",))
    assert abs((w.w_industry + w.w_size + w.w_title) - 1.0) < 1e-9


def test_score_breakdown_fields() -> None:
    b = ScoreBreakdown(
        fit=0.5,
        intent=0.4,
        total=0.45,
        industry_fit=1.0,
        size_fit=1.0,
        title_fit=0.7,
        recency_factor=0.9,
    )
    assert b.total == 0.45


def test_record_pair_roundtrip() -> None:
    p = RecordPair(left_id="A", right_id="B", is_duplicate=True)
    assert p.is_duplicate
    assert p.left_id == "A"
