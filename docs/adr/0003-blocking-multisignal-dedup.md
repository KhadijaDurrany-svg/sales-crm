# ADR 0003: Blocking + multi-signal fuzzy dedup over exact match

## Status
Accepted.

## Context
CRM records for the same person or company drift: emails gain plus-tags or lose
dots, names pick up typos, companies lose their legal suffix, phones change
format. Exact-match deduplication (identical normalized email) is precise but
misses all of this drift. A full all-pairs fuzzy comparison is O(n^2) and does
not scale, and an opaque learned pair classifier is hard to audit and needs
labelled training data we do not have in production.

## Decision
- **Blocking** first: each record emits a few cheap keys (email domain, name
  prefix, last-name prefix, phone last-4). Only records sharing a key become
  candidate pairs, which keeps comparisons tractable while still colliding true
  duplicates when a single field drifts.
- **Multi-signal scoring** for each candidate: a weighted blend of email
  similarity (normalized, with a domain-mismatch penalty), name Levenshtein ratio
  and token Jaccard, company token overlap, and phone-digit equality. A pair at
  or above `threshold` is flagged. All string logic (including Levenshtein) is
  implemented in-package; there are no fuzzy-matching dependencies.
- **Exact match** is kept as the baseline so the recall gain is quantified.

## Consequences
- Fuzzy matching raises recall substantially over exact match at a small
  precision cost, giving a higher F1 (reported with bootstrap CIs and a paired
  permutation p-value).
- Blocking can miss a true duplicate when every blocking field drifts at once;
  the generator produces such "hard" duplicates on purpose, so recall is honestly
  below 1.0. The threshold and weights in `DedupConfig` are the main tuning knobs
  and trade precision against recall.
