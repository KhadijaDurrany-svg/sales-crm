"""Multi-signal duplicate / dirty-record detection.

The fuzzy detector normalizes each field, generates candidate pairs with
light blocking (to avoid an all-pairs blow-up), then scores each candidate from
several independent similarity signals: email local-part / domain, name
Levenshtein ratio and token Jaccard, company token overlap, and phone digits.
A pair above ``threshold`` is flagged a duplicate.

The baseline, :func:`exact_match_pairs`, flags only records with an identical
normalized email -- strong precision, poor recall once emails drift.

All string logic is implemented here; there are no fuzzy-matching dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from leadsift.schemas import Lead

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PLUS_TAG = re.compile(r"\+[^@]*")


def normalize_text(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def normalize_email(email: str) -> str:
    """Lowercase, strip plus-addressing, drop dots in the local part."""
    local, _, domain = email.lower().partition("@")
    local = _PLUS_TAG.sub("", local).replace(".", "")
    return f"{local}@{domain}" if domain else local


def email_domain(email: str) -> str:
    return email.lower().partition("@")[2]


def phone_digits(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    return digits[-10:] if len(digits) >= 10 else digits


def levenshtein(a: str, b: str) -> int:
    """Classic iterative edit distance (insert/delete/substitute = 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    """1 - normalized edit distance; 1.0 is identical, 0.0 is maximally different."""
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / longest


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass(frozen=True)
class DedupConfig:
    """Weights and threshold for the fuzzy duplicate scorer."""

    w_email: float = 0.40
    w_name: float = 0.30
    w_company: float = 0.15
    w_phone: float = 0.15
    threshold: float = 0.62


def _full_name(lead: Lead) -> str:
    return normalize_text(f"{lead.first_name} {lead.last_name}")


def pair_similarity(a: Lead, b: Lead, config: DedupConfig) -> float:
    """Blended similarity in [0, 1] for a candidate pair."""
    ea, eb = normalize_email(a.email), normalize_email(b.email)
    if ea and ea == eb:
        email_sim = 1.0
    else:
        la, lb = ea.partition("@")[0], eb.partition("@")[0]
        email_sim = levenshtein_ratio(la, lb)
        if email_domain(a.email) != email_domain(b.email):
            email_sim *= 0.7  # different domains are weaker evidence

    na, nb = _full_name(a), _full_name(b)
    name_sim = max(levenshtein_ratio(na, nb), token_jaccard(na, nb))

    company_sim = token_jaccard(normalize_text(a.company), normalize_text(b.company))

    pa, pb = phone_digits(a.phone), phone_digits(b.phone)
    phone_sim = 1.0 if pa and pa == pb else 0.0

    return (
        config.w_email * email_sim
        + config.w_name * name_sim
        + config.w_company * company_sim
        + config.w_phone * phone_sim
    )


def _blocking_keys(lead: Lead) -> set[str]:
    """Cheap keys so true duplicates still collide despite one field drifting."""
    name = _full_name(lead)
    keys: set[str] = set()
    dom = email_domain(lead.email)
    if dom:
        keys.add(f"dom:{dom}")
    if name:
        keys.add(f"nm:{name[:4]}")
    last = normalize_text(lead.last_name)
    if last:
        keys.add(f"ln:{last[:4]}")
    digits = phone_digits(lead.phone)
    if digits:
        keys.add(f"ph:{digits[-4:]}")
    return keys


def candidate_pairs(records: list[Lead]) -> set[tuple[int, int]]:
    """Index pairs sharing at least one blocking key."""
    buckets: dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        for key in _blocking_keys(rec):
            buckets.setdefault(key, []).append(idx)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((min(members[i], members[j]), max(members[i], members[j])))
    return pairs


def detect_duplicates(
    records: list[Lead],
    config: DedupConfig | None = None,
) -> set[tuple[str, str]]:
    """Return the set of ``(left_id, right_id)`` pairs flagged as duplicates."""
    config = config if config is not None else DedupConfig()
    flagged: set[tuple[str, str]] = set()
    for i, j in candidate_pairs(records):
        if pair_similarity(records[i], records[j], config) >= config.threshold:
            lo, hi = sorted((records[i].lead_id, records[j].lead_id))
            flagged.add((lo, hi))
    return flagged


def exact_match_pairs(records: list[Lead]) -> set[tuple[str, str]]:
    """Baseline: flag only records that share an identical normalized email."""
    by_email: dict[str, list[str]] = {}
    for rec in records:
        by_email.setdefault(normalize_email(rec.email), []).append(rec.lead_id)
    flagged: set[tuple[str, str]] = set()
    for ids in by_email.values():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                lo, hi = sorted((ids[i], ids[j]))
                flagged.add((lo, hi))
    return flagged
