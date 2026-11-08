from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from leadsift.dedup import (
    DedupConfig,
    candidate_pairs,
    detect_duplicates,
    exact_match_pairs,
    levenshtein,
    levenshtein_ratio,
    normalize_email,
    phone_digits,
    token_jaccard,
)
from leadsift.generate import generate_dedup_records
from leadsift.metrics import prf


def test_normalize_email_strips_plus_and_dots() -> None:
    assert normalize_email("A.B+crm@Acme.com") == "ab@acme.com"
    assert normalize_email("a.b@acme.com") == normalize_email("ab@acme.com")


def test_phone_digits_last_ten() -> None:
    assert phone_digits("+1-555-123-4567") == "5551234567"
    assert phone_digits("(555) 123-4567") == "5551234567"


def test_levenshtein_known_values() -> None:
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("", "abc") == 3


def test_levenshtein_ratio_bounds() -> None:
    assert levenshtein_ratio("abc", "abc") == 1.0
    assert 0.0 <= levenshtein_ratio("abc", "xyz") < 1.0


def test_token_jaccard() -> None:
    assert token_jaccard("acme corp", "acme corp") == 1.0
    assert token_jaccard("acme corp", "globex llc") == 0.0
    assert 0.0 < token_jaccard("acme corp", "acme llc") < 1.0


def test_exact_match_flags_identical_email() -> None:
    dd = generate_dedup_records(seed=4, n_base=60)
    flagged = exact_match_pairs(dd.records)
    truth = dd.duplicate_pairs
    m = prf(flagged, truth)
    # Exact match should be precise but miss drifted duplicates.
    assert m.precision > 0.9
    assert m.recall < 1.0


def test_fuzzy_beats_exact_on_f1() -> None:
    dd = generate_dedup_records(seed=5, n_base=120)
    truth = dd.duplicate_pairs
    fuzzy = prf(detect_duplicates(dd.records, DedupConfig()), truth)
    exact = prf(exact_match_pairs(dd.records), truth)
    assert fuzzy.recall > exact.recall
    assert fuzzy.f1 > exact.f1


def test_fuzzy_is_not_perfect() -> None:
    # Honesty guard: if dedup ever hits a perfect score the generator is too easy.
    dd = generate_dedup_records(seed=6, n_base=120)
    fuzzy = prf(detect_duplicates(dd.records, DedupConfig()), dd.duplicate_pairs)
    assert fuzzy.f1 < 0.99


def test_candidate_pairs_are_sorted_indices() -> None:
    dd = generate_dedup_records(seed=7, n_base=40)
    for i, j in candidate_pairs(dd.records):
        assert i < j


def test_detect_duplicates_threshold_monotonic_recall() -> None:
    dd = generate_dedup_records(seed=8, n_base=80)
    loose = detect_duplicates(dd.records, DedupConfig(threshold=0.5))
    strict = detect_duplicates(dd.records, DedupConfig(threshold=0.85))
    # A lower threshold cannot flag fewer pairs than a higher one.
    assert strict.issubset(loose)


@given(a=st.text(max_size=12), b=st.text(max_size=12))
@settings(max_examples=100, deadline=None)
def test_levenshtein_metric_properties(a: str, b: str) -> None:
    d_ab = levenshtein(a, b)
    assert d_ab == levenshtein(b, a)  # symmetry
    assert d_ab >= abs(len(a) - len(b))  # lower bound
    assert d_ab <= max(len(a), len(b))  # upper bound
    assert (d_ab == 0) == (a == b)  # identity


@given(local=st.text(alphabet="abcdefg.", min_size=1, max_size=8))
@settings(max_examples=50, deadline=None)
def test_normalize_email_idempotent(local: str) -> None:
    email = f"{local}@acme.com"
    once = normalize_email(email)
    assert normalize_email(once) == once
