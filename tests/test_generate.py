from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from leadsift.generate import (
    dataset_checksum,
    generate_dedup_records,
    generate_leads,
)


def test_generate_leads_count_and_fields() -> None:
    ds = generate_leads(seed=1, n=50)
    assert len(ds.leads) == 50
    assert ds.horizon_day == 180
    for lead in ds.leads:
        assert lead.last_activity_day <= ds.horizon_day
        assert lead.created_day <= lead.last_activity_day
        assert "@" in lead.email


def test_generate_leads_is_deterministic() -> None:
    a = generate_leads(seed=42, n=30).leads
    b = generate_leads(seed=42, n=30).leads
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]


def test_generate_leads_seed_changes_data() -> None:
    a = generate_leads(seed=1, n=30).leads
    b = generate_leads(seed=2, n=30).leads
    assert [x.model_dump() for x in a] != [x.model_dump() for x in b]


def test_labels_are_mixed_not_degenerate() -> None:
    labels = generate_leads(seed=7, n=400).labels
    rate = labels.mean()
    # Conversion must be a real signal: neither everyone nor no one converts.
    assert 0.05 < rate < 0.95


def test_dedup_dataset_has_true_duplicates() -> None:
    dd = generate_dedup_records(seed=3, n_base=80)
    assert len(dd.duplicate_pairs) > 0
    assert len(dd.records) > 80  # duplicates added
    ids = {r.lead_id for r in dd.records}
    for lo, hi in dd.duplicate_pairs:
        assert lo in ids and hi in ids
        assert lo < hi


def test_dedup_records_deterministic() -> None:
    a = generate_dedup_records(seed=9, n_base=60)
    b = generate_dedup_records(seed=9, n_base=60)
    assert [r.model_dump() for r in a.records] == [r.model_dump() for r in b.records]
    assert a.duplicate_pairs == b.duplicate_pairs


def test_checksum_stable_and_sensitive() -> None:
    c1 = dataset_checksum([1, 2], n_leads=20, n_base=20)
    c2 = dataset_checksum([1, 2], n_leads=20, n_base=20)
    c3 = dataset_checksum([1, 3], n_leads=20, n_base=20)
    assert c1 == c2
    assert c1 != c3
    assert len(c1) == 64


@given(seed=st.integers(min_value=0, max_value=5000), n=st.integers(min_value=1, max_value=40))
@settings(max_examples=40, deadline=None)
def test_generate_leads_invariants(seed: int, n: int) -> None:
    ds = generate_leads(seed=seed, n=n)
    assert len(ds.leads) == n
    for lead in ds.leads:
        assert lead.employee_count >= 0
        assert 0 <= lead.last_activity_day <= ds.horizon_day
        assert lead.email_opens >= 0
