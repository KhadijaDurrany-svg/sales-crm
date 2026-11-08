from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from leadsift.metrics import (
    bootstrap_ci,
    dcg_at_k,
    ndcg_at_k,
    paired_permutation_pvalue,
    precision_at_k,
    prf,
)


def test_precision_at_k_basic() -> None:
    labels = np.array([1, 1, 0, 1, 0], dtype=bool)
    assert precision_at_k(labels, 2) == 1.0
    assert precision_at_k(labels, 4) == 0.75


def test_precision_at_k_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        precision_at_k(np.array([1, 0]), 0)


def test_perfect_ranking_has_ndcg_one() -> None:
    labels = np.array([1, 1, 1, 0, 0], dtype=bool)
    assert ndcg_at_k(labels, 5) == 1.0


def test_ndcg_prefers_relevant_first() -> None:
    good = np.array([1, 0, 0, 0], dtype=bool)
    bad = np.array([0, 0, 0, 1], dtype=bool)
    assert ndcg_at_k(good, 4) > ndcg_at_k(bad, 4)


def test_ndcg_zero_when_no_relevant() -> None:
    assert ndcg_at_k(np.array([0, 0, 0], dtype=bool), 3) == 0.0


def test_dcg_discounts_later_positions() -> None:
    first = dcg_at_k(np.array([1, 0], dtype=bool), 2)
    second = dcg_at_k(np.array([0, 1], dtype=bool), 2)
    assert first > second


def test_prf_counts() -> None:
    pred = {("a", "b"), ("c", "d"), ("e", "f")}
    truth = {("a", "b"), ("c", "d"), ("g", "h")}
    m = prf(pred, truth)
    assert (m.tp, m.fp, m.fn) == (2, 1, 1)
    assert m.precision == 2 / 3
    assert m.recall == 2 / 3
    assert abs(m.f1 - 2 / 3) < 1e-9


def test_prf_empty_prediction() -> None:
    m = prf(set(), {("a", "b")})
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0


def test_bootstrap_ci_single_value_collapses() -> None:
    lo, hi = bootstrap_ci(np.array([0.5]))
    assert lo == 0.5 and hi == 0.5


def test_bootstrap_ci_brackets_mean() -> None:
    values = np.random.default_rng(0).normal(0.6, 0.05, size=200)
    lo, hi = bootstrap_ci(values, n_boot=1000, seed=1)
    assert lo < values.mean() < hi
    assert lo < hi


def test_permutation_significant_for_clear_difference() -> None:
    a = np.full(40, 0.7)
    b = np.full(40, 0.4)
    p = paired_permutation_pvalue(a, b, n_perm=5000, seed=0)
    assert p < 0.01


def test_permutation_nonsignificant_for_identical() -> None:
    a = np.array([0.5, 0.5, 0.5, 0.5])
    p = paired_permutation_pvalue(a, a.copy(), n_perm=2000, seed=0)
    assert p == 1.0


def test_permutation_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        paired_permutation_pvalue(np.array([1.0, 2.0]), np.array([1.0]))


@given(
    labels=st.lists(st.booleans(), min_size=1, max_size=50),
    k=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=100, deadline=None)
def test_metric_ranges(labels: list[bool], k: int) -> None:
    arr = np.array(labels, dtype=bool)
    assert 0.0 <= precision_at_k(arr, k) <= 1.0
    assert 0.0 <= ndcg_at_k(arr, k) <= 1.0 + 1e-9


@given(vals=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=2, max_size=60))
@settings(max_examples=60, deadline=None)
def test_bootstrap_ci_ordered_and_in_range(vals: list[float]) -> None:
    arr = np.array(vals)
    lo, hi = bootstrap_ci(arr, n_boot=500, seed=0)
    assert lo <= hi
    assert arr.min() - 1e-9 <= lo
    assert hi <= arr.max() + 1e-9


@given(p=st.integers(min_value=1, max_value=500))
@settings(max_examples=30, deadline=None)
def test_permutation_pvalue_in_unit_interval(p: int) -> None:
    rng = np.random.default_rng(p)
    a = rng.random(20)
    b = rng.random(20)
    pv = paired_permutation_pvalue(a, b, n_perm=p + 50, seed=0)
    assert 0.0 < pv <= 1.0
