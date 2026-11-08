"""Typed data models shared across the package (Pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Lead(BaseModel):
    """A single CRM lead record.

    ``converted`` is the ground-truth outcome label used *only* for evaluation.
    The scoring engine never reads it; it scores from the observable fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lead_id: str
    first_name: str
    last_name: str
    email: str
    company: str
    phone: str
    title: str
    industry: str
    employee_count: int = Field(ge=0)

    # Day indices (integer days since the synthetic epoch).
    created_day: int = Field(ge=0)
    last_activity_day: int = Field(ge=0)

    # Observable engagement signals.
    email_opens: int = Field(ge=0)
    email_clicks: int = Field(ge=0)
    site_visits: int = Field(ge=0)
    pricing_views: int = Field(ge=0)
    demo_requested: bool = False

    # Ground-truth evaluation label (hidden from the scorer).
    converted: bool = False


class ScoreBreakdown(BaseModel):
    """Transparent, per-component explanation of a lead's priority score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fit: float
    intent: float
    total: float
    industry_fit: float
    size_fit: float
    title_fit: float
    recency_factor: float


class ScoringWeights(BaseModel):
    """Inspectable weights for the fit/intent scoring engine.

    Weights are deliberately hand-set and transparent rather than fit to the
    data generator, so the engine is a realistic (imperfect) business rule set
    rather than an oracle. ``fit`` component weights sum to 1; ``intent`` event
    weights are relative.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_industries: tuple[str, ...]
    ideal_min_employees: int = 50
    ideal_max_employees: int = 5000

    # Fit sub-score component weights (sum to 1.0).
    w_industry: float = 0.40
    w_size: float = 0.25
    w_title: float = 0.35

    # Intent event weights (relative importance of each engagement signal).
    w_open: float = 1.0
    w_click: float = 2.5
    w_visit: float = 1.5
    w_pricing: float = 4.0
    w_demo: float = 8.0

    # Recency half-life in days for intent decay.
    recency_half_life: float = 14.0
    # Saturation constant for the intent squashing function.
    intent_saturation: float = 20.0
    # Blend between fit and intent for the total score (0..1 toward fit).
    fit_weight: float = 0.5

    # Title seniority map (title -> normalized seniority in [0, 1]).
    title_seniority: dict[str, float] = Field(
        default_factory=lambda: {
            "Analyst": 0.1,
            "Associate": 0.2,
            "Manager": 0.45,
            "Senior Manager": 0.55,
            "Director": 0.7,
            "VP": 0.85,
            "CTO": 1.0,
            "CEO": 1.0,
        }
    )


class RecordPair(BaseModel):
    """A candidate duplicate pair of lead ids with a ground-truth label."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    left_id: str
    right_id: str
    is_duplicate: bool
