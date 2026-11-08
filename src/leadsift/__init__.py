"""leadsift: offline lead prioritization and CRM record deduplication.

The package ships two cooperating engines with transparent, inspectable logic
and no network access or model downloads:

* ``scoring`` / ``ranking`` -- score and rank synthetic leads by a transparent
  fit-and-intent signal, and rank them against random / recency / raw-score
  baselines.
* ``dedup`` -- detect duplicate and dirty CRM records with multi-signal fuzzy
  matching, measured against an exact-match baseline.

Everything is driven by seeded synthetic data (see :mod:`leadsift.generate`) so
benchmark numbers (see :mod:`leadsift.bench`) are reproducible.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "Lead",
    "RecordPair",
    "ScoringWeights",
    "DedupConfig",
    "score_lead",
    "rank_leads",
    "detect_duplicates",
]

__version__ = "0.1.0"

from leadsift.dedup import DedupConfig, detect_duplicates
from leadsift.ranking import rank_leads
from leadsift.schemas import Lead, RecordPair, ScoringWeights
from leadsift.scoring import score_lead

# DedupConfig re-exported above; keep import near its siblings for clarity.
