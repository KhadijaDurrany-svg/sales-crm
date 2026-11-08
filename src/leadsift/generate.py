"""Seeded synthetic data generation for leads and duplicate records.

No network, no external files. Everything derives from a single integer seed
via :class:`numpy.random.Generator`, so a given seed reproduces byte-identical
data (see :func:`dataset_checksum`).

The generator is deliberately *noisy* and only partly aligned with the scoring
engine's weights. Conversion is a stochastic function of latent fit and intent,
so no ranker can be perfect; duplicates carry realistic typos and formatting
drift, so exact-match cannot catch them all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from leadsift.schemas import Lead

INDUSTRIES: tuple[str, ...] = (
    "SaaS",
    "Fintech",
    "Healthcare",
    "Manufacturing",
    "Retail",
    "Education",
    "Logistics",
    "Media",
)
# The ideal-customer-profile target set (drives true fit).
TARGET_INDUSTRIES: tuple[str, ...] = ("SaaS", "Fintech", "Healthcare")

TITLES: tuple[str, ...] = (
    "Analyst",
    "Associate",
    "Manager",
    "Senior Manager",
    "Director",
    "VP",
    "CTO",
    "CEO",
)
# True (hidden) seniority used by the generator; the scorer uses its own map.
_TRUE_SENIORITY = {
    "Analyst": 0.05,
    "Associate": 0.15,
    "Manager": 0.4,
    "Senior Manager": 0.5,
    "Director": 0.75,
    "VP": 0.9,
    "CTO": 1.0,
    "CEO": 1.0,
}

FIRST_NAMES: tuple[str, ...] = (
    "Aisha",
    "Ben",
    "Carla",
    "Diego",
    "Emma",
    "Farah",
    "Grace",
    "Hassan",
    "Iris",
    "Jamal",
    "Kira",
    "Liam",
    "Maya",
    "Noah",
    "Omar",
    "Priya",
    "Quinn",
    "Rosa",
    "Sven",
    "Tara",
    "Umar",
    "Vera",
    "Wes",
    "Xena",
)
LAST_NAMES: tuple[str, ...] = (
    "Ahmed",
    "Brooks",
    "Chen",
    "Dubois",
    "Evans",
    "Fischer",
    "Gupta",
    "Haddad",
    "Ibrahim",
    "Jones",
    "Kowalski",
    "Lopez",
    "Mehta",
    "Novak",
    "Owens",
    "Patel",
    "Quraishi",
    "Rossi",
    "Sato",
    "Turner",
    "Ueda",
    "Vargas",
    "Watson",
    "Xu",
)
COMPANY_STEMS: tuple[str, ...] = (
    "Northwind",
    "Acme",
    "Globex",
    "Initech",
    "Umbrella",
    "Soylent",
    "Hooli",
    "Vandelay",
    "Stark",
    "Wayne",
    "Wonka",
    "Tyrell",
    "Cyberdyne",
    "Massive",
)
COMPANY_SUFFIXES: tuple[str, ...] = ("Inc", "LLC", "Corp", "Group", "Labs", "Systems")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class LeadDataset:
    """A single scoring batch: the leads plus the horizon day they were cut on."""

    leads: list[Lead]
    horizon_day: int

    @property
    def labels(self) -> np.ndarray:
        return np.array([lead.converted for lead in self.leads], dtype=bool)


def _make_email(first: str, last: str, company: str, rng: np.random.Generator) -> str:
    stem = company.split()[0].lower()
    patterns = [
        f"{first.lower()}.{last.lower()}@{stem}.com",
        f"{first[0].lower()}{last.lower()}@{stem}.com",
        f"{first.lower()}{last[0].lower()}@{stem}.io",
    ]
    return patterns[int(rng.integers(len(patterns)))]


def _make_phone(rng: np.random.Generator) -> str:
    area = int(rng.integers(200, 999))
    mid = int(rng.integers(200, 999))
    end = int(rng.integers(0, 9999))
    return f"+1-{area}-{mid}-{end:04d}"


def generate_leads(seed: int, n: int = 200, horizon_day: int = 180) -> LeadDataset:
    """Generate ``n`` leads for one scoring batch.

    Conversion depends on latent fit and recency-weighted intent plus Gaussian
    noise, so the Bayes-optimal ranking is unattainable and all rankers land
    strictly below 1.0 on precision@k / NDCG.
    """

    rng = np.random.default_rng(seed)
    leads: list[Lead] = []

    for i in range(n):
        first = FIRST_NAMES[int(rng.integers(len(FIRST_NAMES)))]
        last = LAST_NAMES[int(rng.integers(len(LAST_NAMES)))]
        company_stem = COMPANY_STEMS[int(rng.integers(len(COMPANY_STEMS)))]
        company = f"{company_stem} {COMPANY_SUFFIXES[int(rng.integers(len(COMPANY_SUFFIXES)))]}"
        industry = INDUSTRIES[int(rng.integers(len(INDUSTRIES)))]
        title = TITLES[int(rng.integers(len(TITLES)))]

        # Company size: log-uniform roughly 10 .. 20000 employees.
        employee_count = int(10 ** rng.uniform(1.0, 4.3))

        # --- latent true fit (what actually drives conversion) ---
        is_target = 1.0 if industry in TARGET_INDUSTRIES else 0.0
        size_fit = _size_fit(employee_count, 50, 5000)
        seniority = _TRUE_SENIORITY[title]
        true_fit = 0.45 * is_target + 0.30 * size_fit + 0.25 * seniority

        # --- latent interest drives engagement volume and recency ---
        interest = float(rng.beta(1.6, 3.0))  # skewed toward low interest
        days_since_activity = int(rng.exponential(scale=18.0 * (1.2 - interest)))
        days_since_activity = min(days_since_activity, horizon_day - 1)
        last_activity_day = horizon_day - days_since_activity
        created_day = max(0, last_activity_day - int(rng.integers(1, 120)))

        rate = 0.5 + 6.0 * interest
        email_opens = int(rng.poisson(rate))
        email_clicks = int(rng.poisson(rate * 0.4 * interest))
        site_visits = int(rng.poisson(rate * 0.6))
        pricing_views = int(rng.poisson(rate * 0.25 * interest))
        demo_requested = bool(rng.random() < (0.02 + 0.35 * interest))

        # --- latent true intent (recency-decayed weighted engagement) ---
        recency = 0.5 ** (days_since_activity / 14.0)
        raw_intent = (
            1.0 * email_opens + 2.5 * email_clicks + 1.5 * site_visits + 4.0 * pricing_views + 8.0 * demo_requested
        )
        true_intent = (raw_intent * recency) / (raw_intent * recency + 18.0)

        # --- stochastic conversion ---
        noise = float(rng.normal(0.0, 0.6))
        logit = 3.0 * true_fit + 3.6 * true_intent - 3.3 + noise
        converted = bool(rng.random() < _sigmoid(logit))

        leads.append(
            Lead(
                lead_id=f"L{seed:04d}-{i:04d}",
                first_name=first,
                last_name=last,
                email=_make_email(first, last, company, rng),
                company=company,
                phone=_make_phone(rng),
                title=title,
                industry=industry,
                employee_count=employee_count,
                created_day=created_day,
                last_activity_day=last_activity_day,
                email_opens=email_opens,
                email_clicks=email_clicks,
                site_visits=site_visits,
                pricing_views=pricing_views,
                demo_requested=demo_requested,
                converted=converted,
            )
        )

    return LeadDataset(leads=leads, horizon_day=horizon_day)


def _size_fit(employees: int, lo: int, hi: int) -> float:
    """Triangular fit: 1.0 inside the ideal band, decaying (log) outside it."""
    if lo <= employees <= hi:
        return 1.0
    if employees < lo:
        return max(0.0, 1.0 - (np.log10(max(lo, 1)) - np.log10(max(employees, 1))))
    return max(0.0, 1.0 - (np.log10(employees) - np.log10(hi)))


# --------------------------------------------------------------------------- #
# Duplicate / dirty-record generation
# --------------------------------------------------------------------------- #


def _typo(text: str, rng: np.random.Generator) -> str:
    """Introduce a single realistic typo (swap / drop / duplicate a char)."""
    if len(text) < 3:
        return text
    kind = int(rng.integers(3))
    pos = int(rng.integers(1, len(text) - 1))
    chars = list(text)
    if kind == 0:  # transpose
        chars[pos], chars[pos - 1] = chars[pos - 1], chars[pos]
    elif kind == 1:  # drop
        del chars[pos]
    else:  # duplicate
        chars.insert(pos, chars[pos])
    return "".join(chars)


def _perturb_email(email: str, rng: np.random.Generator) -> str:
    local, _, domain = email.partition("@")
    choice = int(rng.integers(4))
    if choice == 0:
        return f"{local}+crm@{domain}"  # plus-addressing
    if choice == 1:
        return f"{local.replace('.', '')}@{domain}"  # dropped dots
    if choice == 2:
        return f"{local}@{domain}".upper()  # case drift
    return _typo(local, rng) + "@" + domain  # local typo


def _perturb_company(company: str, rng: np.random.Generator) -> str:
    stem, _, suffix = company.partition(" ")
    choice = int(rng.integers(4))
    if choice == 0:
        return stem  # suffix dropped
    if choice == 1:
        return f"{stem} {suffix}.".replace("Inc.", "Incorporated")
    if choice == 2:
        return company.lower()
    return _typo(stem, rng) + " " + suffix


_PERSONAL_DOMAINS: tuple[str, ...] = ("gmail.com", "outlook.com", "yahoo.com", "icloud.com")


def _personal_email(first: str, last: str, rng: np.random.Generator) -> str:
    domain = _PERSONAL_DOMAINS[int(rng.integers(len(_PERSONAL_DOMAINS)))]
    num = int(rng.integers(0, 99))
    patterns = [
        f"{first.lower()}{last.lower()}{num}@{domain}",
        f"{first.lower()}.{last.lower()}@{domain}",
        f"{first[0].lower()}{last.lower()}@{domain}",
    ]
    return patterns[int(rng.integers(len(patterns)))]


def _perturb_phone(phone: str, rng: np.random.Generator) -> str:
    digits = "".join(c for c in phone if c.isdigit())
    choice = int(rng.integers(3))
    if choice == 0:
        return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"  # US format
    if choice == 1:
        return digits  # bare digits
    return f"{digits[1:4]}.{digits[4:7]}.{digits[7:]}"  # dotted, no country code


@dataclass(frozen=True)
class DedupDataset:
    """A dedup batch: records plus the ground-truth duplicate pairs among them."""

    records: list[Lead]
    duplicate_pairs: set[tuple[str, str]]


def generate_dedup_records(
    seed: int,
    n_base: int = 120,
    dup_rate: float = 0.35,
) -> DedupDataset:
    """Generate base records plus near-duplicate copies with realistic drift.

    A fraction ``dup_rate`` of base records gets 1-2 duplicate copies whose
    name/email/company/phone are perturbed (typos, formatting, abbreviations).
    Ground-truth duplicate pairs are all copies sharing a base identity.
    """

    rng = np.random.default_rng(seed + 100_000)
    base = generate_leads(seed=seed + 50_000, n=n_base).leads

    records: list[Lead] = []
    groups: dict[str, list[str]] = {}

    counter = 0

    def _add(rec: Lead, group: str) -> None:
        nonlocal counter
        records.append(rec)
        groups.setdefault(group, []).append(rec.lead_id)
        counter += 1

    for b in base:
        gid = b.lead_id
        canonical = b.model_copy(update={"lead_id": f"R{seed:04d}-{counter:04d}"})
        _add(canonical, gid)

        if rng.random() < dup_rate:
            n_dupes = int(rng.integers(1, 3))
            for _ in range(n_dupes):
                hard = rng.random() < 0.22
                if hard:
                    # A genuinely hard duplicate: the contact re-entered under a
                    # personal email on a different provider with a new phone, so
                    # the strong email/phone signals are gone and only the name
                    # (and maybe company) ties the records together. Realistic
                    # fuzzy matching will miss some of these -> recall < 1.
                    personal = _personal_email(b.first_name, b.last_name, rng)
                    upd = {
                        "lead_id": f"R{seed:04d}-{counter:04d}",
                        "first_name": _typo(b.first_name, rng) if rng.random() < 0.3 else b.first_name,
                        "last_name": _typo(b.last_name, rng) if rng.random() < 0.3 else b.last_name,
                        "email": personal,
                        "company": _perturb_company(b.company, rng) if rng.random() < 0.5 else b.company,
                        "phone": _make_phone(rng),
                    }
                else:
                    upd = {
                        "lead_id": f"R{seed:04d}-{counter:04d}",
                        "first_name": _typo(b.first_name, rng) if rng.random() < 0.4 else b.first_name,
                        "last_name": _typo(b.last_name, rng) if rng.random() < 0.4 else b.last_name,
                        "email": _perturb_email(b.email, rng) if rng.random() < 0.7 else b.email,
                        "company": _perturb_company(b.company, rng) if rng.random() < 0.6 else b.company,
                        "phone": _perturb_phone(b.phone, rng) if rng.random() < 0.6 else b.phone,
                    }
                _add(b.model_copy(update=upd), gid)

    duplicate_pairs: set[tuple[str, str]] = set()
    for ids in groups.values():
        for a in range(len(ids)):
            for c in range(a + 1, len(ids)):
                lo, hi = sorted((ids[a], ids[c]))
                duplicate_pairs.add((lo, hi))

    # Shuffle so duplicates are not adjacent.
    order = rng.permutation(len(records))
    records = [records[i] for i in order]
    return DedupDataset(records=records, duplicate_pairs=duplicate_pairs)


def dataset_checksum(seeds: list[int], n_leads: int, n_base: int) -> str:
    """Stable SHA-256 over the generated data for a benchmark configuration.

    Captures lead + dedup content so the manifest can prove *which* data a
    result was measured on, independent of machine or run order.
    """

    hasher = hashlib.sha256()
    for seed in seeds:
        ds = generate_leads(seed=seed, n=n_leads)
        for lead in ds.leads:
            hasher.update(lead.model_dump_json().encode("utf-8"))
        dd = generate_dedup_records(seed=seed, n_base=n_base)
        for rec in dd.records:
            hasher.update(rec.model_dump_json().encode("utf-8"))
        hasher.update(json.dumps(sorted(dd.duplicate_pairs)).encode("utf-8"))
    return hasher.hexdigest()
