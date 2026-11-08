"""Run the end-to-end benchmark and assemble a results dictionary.

All numbers here are *measured* on freshly generated, seeded data at call time
-- nothing is hard-coded. The CLI serializes the returned dict to
``benchmarks/results.json`` alongside a reproducibility manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from leadsift import __version__
from leadsift.dedup import DedupConfig, detect_duplicates, exact_match_pairs
from leadsift.generate import (
    dataset_checksum,
    generate_dedup_records,
    generate_leads,
)
from leadsift.metrics import (
    bootstrap_ci,
    ndcg_at_k,
    paired_permutation_pvalue,
    precision_at_k,
    prf,
)
from leadsift.ranking import (
    rank_leads,
    rank_random,
    rank_raw_score,
    rank_recency,
)
from leadsift.scoring import default_weights


@dataclass(frozen=True)
class BenchConfig:
    n_batches: int = 40
    n_leads: int = 200
    k: int = 20
    n_base_records: int = 120
    base_seed: int = 7
    n_boot: int = 2000
    n_perm: int = 10000

    @property
    def seeds(self) -> list[int]:
        return [self.base_seed + i for i in range(self.n_batches)]


_RANK_METHODS = ("fit_intent", "raw_score", "recency", "random")


def _ranked_labels(order: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return labels[order]


def run_ranking(config: BenchConfig) -> dict:
    weights = default_weights()
    k = config.k
    prec: dict[str, list[float]] = {m: [] for m in _RANK_METHODS}
    ndcg: dict[str, list[float]] = {m: [] for m in _RANK_METHODS}

    for seed in config.seeds:
        ds = generate_leads(seed=seed, n=config.n_leads)
        labels = ds.labels
        # A dedicated, seed-derived RNG keeps tie-breaks / shuffles reproducible.
        rng = np.random.default_rng(seed + 900_000)
        orders = {
            "fit_intent": rank_leads(ds.leads, ds.horizon_day, weights, rng),
            "raw_score": rank_raw_score(ds.leads, rng),
            "recency": rank_recency(ds.leads, rng),
            "random": rank_random(ds.leads, rng),
        }
        for method, order in orders.items():
            rlabels = _ranked_labels(order, labels)
            prec[method].append(precision_at_k(rlabels, k))
            ndcg[method].append(ndcg_at_k(rlabels, k))

    methods_out: dict[str, dict] = {}
    for method in _RANK_METHODS:
        p = np.array(prec[method])
        g = np.array(ndcg[method])
        methods_out[method] = {
            "precision_at_k_mean": float(p.mean()),
            "precision_at_k_ci95": list(bootstrap_ci(p, config.n_boot, seed=1)),
            "ndcg_at_k_mean": float(g.mean()),
            "ndcg_at_k_ci95": list(bootstrap_ci(g, config.n_boot, seed=2)),
        }

    # Significance: fit_intent vs the strongest baseline, per metric.
    baselines = [m for m in _RANK_METHODS if m != "fit_intent"]
    best_ndcg_baseline = max(baselines, key=lambda m: methods_out[m]["ndcg_at_k_mean"])
    best_prec_baseline = max(baselines, key=lambda m: methods_out[m]["precision_at_k_mean"])

    sig = {
        "ndcg": {
            "comparison": f"fit_intent vs {best_ndcg_baseline}",
            "delta_mean": methods_out["fit_intent"]["ndcg_at_k_mean"]
            - methods_out[best_ndcg_baseline]["ndcg_at_k_mean"],
            "p_value": paired_permutation_pvalue(
                np.array(ndcg["fit_intent"]),
                np.array(ndcg[best_ndcg_baseline]),
                config.n_perm,
                seed=3,
            ),
        },
        "precision": {
            "comparison": f"fit_intent vs {best_prec_baseline}",
            "delta_mean": methods_out["fit_intent"]["precision_at_k_mean"]
            - methods_out[best_prec_baseline]["precision_at_k_mean"],
            "p_value": paired_permutation_pvalue(
                np.array(prec["fit_intent"]),
                np.array(prec[best_prec_baseline]),
                config.n_perm,
                seed=4,
            ),
        },
    }

    return {
        "k": k,
        "n_batches": config.n_batches,
        "n_leads_per_batch": config.n_leads,
        "relevance": "converted == True",
        "methods": methods_out,
        "significance": sig,
    }


def run_dedup(config: BenchConfig) -> dict:
    dedup_config = DedupConfig()
    f1 = {"fuzzy": [], "exact_match": []}
    precision = {"fuzzy": [], "exact_match": []}
    recall = {"fuzzy": [], "exact_match": []}

    for seed in config.seeds:
        dd = generate_dedup_records(seed=seed, n_base=config.n_base_records)
        truth = dd.duplicate_pairs
        fuzzy_pred = detect_duplicates(dd.records, dedup_config)
        exact_pred = exact_match_pairs(dd.records)
        for name, pred in (("fuzzy", fuzzy_pred), ("exact_match", exact_pred)):
            m = prf(pred, truth)
            precision[name].append(m.precision)
            recall[name].append(m.recall)
            f1[name].append(m.f1)

    methods_out: dict[str, dict] = {}
    for name in ("fuzzy", "exact_match"):
        f = np.array(f1[name])
        methods_out[name] = {
            "precision_mean": float(np.mean(precision[name])),
            "recall_mean": float(np.mean(recall[name])),
            "f1_mean": float(f.mean()),
            "f1_ci95": list(bootstrap_ci(f, config.n_boot, seed=5)),
        }

    sig = {
        "metric": "f1",
        "comparison": "fuzzy vs exact_match",
        "delta_mean": methods_out["fuzzy"]["f1_mean"] - methods_out["exact_match"]["f1_mean"],
        "p_value": paired_permutation_pvalue(np.array(f1["fuzzy"]), np.array(f1["exact_match"]), config.n_perm, seed=6),
    }

    return {
        "n_batches": config.n_batches,
        "n_base_records": config.n_base_records,
        "methods": methods_out,
        "significance": sig,
    }


def run_benchmark(config: BenchConfig | None = None) -> dict:
    """Run both suites and return a fully measured results dictionary."""
    config = config if config is not None else BenchConfig()
    checksum = dataset_checksum(config.seeds, config.n_leads, config.n_base_records)
    return {
        "package": "leadsift",
        "version": __version__,
        "config": asdict(config),
        "dataset_checksum_sha256": checksum,
        "ranking": run_ranking(config),
        "dedup": run_dedup(config),
    }
