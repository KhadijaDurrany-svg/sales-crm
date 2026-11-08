"""Evaluation metrics and statistics: ranking quality, set P/R/F1, and the
resampling tools (bootstrap CIs, paired permutation test) the benchmark reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def precision_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """Fraction of the top-``k`` ranked items that are relevant (converted)."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = ranked_labels[:k]
    if len(top) == 0:
        return 0.0
    return float(np.sum(top) / len(top))


def dcg_at_k(ranked_labels: np.ndarray, k: int) -> float:
    top = ranked_labels[:k].astype(float)
    if len(top) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    return float(np.sum(top * discounts))


def ndcg_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """Normalized DCG@k for binary relevance; 0.0 when no item is relevant."""
    ideal = np.sort(ranked_labels)[::-1]
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_labels, k) / idcg


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def prf(predicted: set, truth: set) -> PRF:
    """Precision / recall / F1 for two sets of items (e.g. duplicate pairs)."""
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return PRF(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    With a single observation the interval collapses to that value (there is
    nothing to resample), which the benchmark reports honestly.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided paired permutation test on the mean of per-item differences.

    Returns the fraction of sign-flip permutations whose absolute mean
    difference is at least the observed one (add-one smoothed). Requires the
    same number of paired observations in ``a`` and ``b``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    diff = a - b
    n = len(diff)
    if n == 0:
        return 1.0
    observed = abs(float(np.mean(diff)))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, n))
    perm_means = np.abs((signs * diff).mean(axis=1))
    count = int(np.sum(perm_means >= observed - 1e-12))
    return (count + 1) / (n_perm + 1)
