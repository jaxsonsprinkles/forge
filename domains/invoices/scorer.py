"""Scorer for the invoices domain.

Dataset format (see dataset.jsonl): `input["invoice_text"]` is a plain-text
invoice; `expected` is a ground-truth dict with the fields:

    vendor          str
    invoice_number  str
    date            str, ISO "YYYY-MM-DD"
    line_items      list[{"description": str, "qty": number,
                          "unit_price": number, "amount": number}]
    subtotal        number
    tax             number (0.0 if the source document states no tax)
    total           number
    currency        str, ISO 4217 code (e.g. "USD")

`output` is whatever an agent under evaluation returns for a task: either a
dict already in this shape, or a JSON string that parses to one. Anything
else (unparseable string, list, None, ...) is treated as a fully-failed
submission rather than raising.

score(output, expected) -> (passed, details):

  - Each field is compared independently and given a score in [0, 1]
    (partial credit, not exact-match booleans) - see `_score_field` for the
    per-field-type rules.
  - The aggregate score is a weighted mean across fields, with `total`
    weighted twice as heavily as every other field (WEIGHTS below), since
    getting the bottom-line amount right matters most for a usable
    extraction.
  - `passed` is True iff the aggregate score is >= PASS_THRESHOLD (0.75):
    chosen so a submission needs to get nearly every field substantially
    right - and in particular can't pass by nailing `total` alone while
    botching everything else - while still tolerating minor formatting
    noise (e.g. a slightly-off vendor string) on to that gets close.
  - `details["fields"]` carries the per-field score and a short diagnostic
    for each field; `details["aggregate_score"]` carries the weighted mean.

Never raises: any failure (malformed output, wrong types, missing keys) is
reported as (False, details) with per-field scores of 0 for the fields that
couldn't be evaluated.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

PASS_THRESHOLD = 0.75

WEIGHTS: dict[str, float] = {
    "vendor": 1.0,
    "invoice_number": 1.0,
    "date": 1.0,
    "line_items": 1.0,
    "subtotal": 1.0,
    "tax": 1.0,
    "total": 2.0,
    "currency": 1.0,
}


def _coerce_output(output: Any) -> dict[str, Any] | None:
    """Best-effort coercion of an agent's raw output into a dict, or None."""
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.strip().lower().split())


def _score_string_similarity(candidate: Any, expected: Any) -> tuple[float, str]:
    exp_norm = _normalize_str(expected)
    cand_norm = _normalize_str(candidate)
    if exp_norm is None:
        return 0.0, "expected value is not a string; cannot score"
    if cand_norm is None:
        return 0.0, f"candidate value {candidate!r} is not a string"
    ratio = SequenceMatcher(None, exp_norm, cand_norm).ratio()
    return ratio, f"similarity ratio {ratio:.2f} between {candidate!r} and {expected!r}"


def _score_exact_string(candidate: Any, expected: Any) -> tuple[float, str]:
    exp_norm = _normalize_str(expected)
    cand_norm = _normalize_str(candidate)
    if exp_norm is None:
        return 0.0, "expected value is not a string; cannot score"
    if cand_norm is None:
        return 0.0, f"candidate value {candidate!r} is not a string"
    if cand_norm == exp_norm:
        return 1.0, "exact match"
    return 0.0, f"{candidate!r} != {expected!r}"


def _score_date(candidate: Any, expected: Any) -> tuple[float, str]:
    exp_norm = _normalize_str(expected)
    cand_norm = _normalize_str(candidate)
    if exp_norm is None or cand_norm is None:
        return 0.0, f"could not compare dates: {candidate!r} vs {expected!r}"
    if cand_norm == exp_norm:
        return 1.0, "exact match"
    exp_parts = exp_norm.split("-")
    cand_parts = cand_norm.split("-")
    if len(exp_parts) == 3 and len(cand_parts) == 3 and exp_parts[:2] == cand_parts[:2]:
        return 0.5, f"year/month match, day differs: {candidate!r} vs {expected!r}"
    return 0.0, f"{candidate!r} != {expected!r}"


def _score_number(candidate: Any, expected: Any) -> tuple[float, str]:
    if not isinstance(expected, (int, float)):
        return 0.0, "expected value is not numeric; cannot score"
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        return 0.0, f"candidate value {candidate!r} is not numeric"
    if candidate == expected:
        return 1.0, "exact match"
    denom = max(abs(expected), 1e-6)
    rel_error = abs(candidate - expected) / denom
    score = max(0.0, 1.0 - min(rel_error, 1.0))
    return score, f"relative error {rel_error:.3f} ({candidate!r} vs {expected!r})"


def _score_line_item_pair(candidate: dict[str, Any], expected: dict[str, Any]) -> float:
    """Average numeric closeness across qty/unit_price/amount for one matched pair."""
    sub_scores = []
    for key in ("qty", "unit_price", "amount"):
        score, _ = _score_number(candidate.get(key), expected.get(key))
        sub_scores.append(score)
    desc_score, _ = _score_string_similarity(candidate.get("description"), expected.get("description"))
    sub_scores.append(desc_score)
    return sum(sub_scores) / len(sub_scores)


def _score_line_items(candidate: Any, expected: Any) -> tuple[float, str]:
    if not isinstance(expected, list) or not expected:
        return 0.0, "expected line_items is missing or not a non-empty list"
    if not isinstance(candidate, list):
        return 0.0, f"candidate line_items is not a list: {candidate!r}"

    cand_pool = [item for item in candidate if isinstance(item, dict)]
    matched_scores = []
    used = [False] * len(cand_pool)
    for exp_item in expected:
        if not isinstance(exp_item, dict):
            matched_scores.append(0.0)
            continue
        best_idx, best_score = -1, -1.0
        for idx, cand_item in enumerate(cand_pool):
            if used[idx]:
                continue
            s = _score_line_item_pair(cand_item, exp_item)
            if s > best_score:
                best_idx, best_score = idx, s
        if best_idx == -1:
            matched_scores.append(0.0)
        else:
            used[best_idx] = True
            matched_scores.append(best_score)

    match_score = sum(matched_scores) / len(matched_scores)
    coverage = min(len(expected), len(cand_pool)) / max(len(expected), len(cand_pool))
    final = match_score * coverage
    return final, (
        f"matched {sum(used)}/{len(expected)} expected items, "
        f"avg match quality {match_score:.2f}, coverage {coverage:.2f} "
        f"(expected {len(expected)} items, candidate had {len(cand_pool)})"
    )


def _score_field(field: str, candidate: Any, expected: Any) -> tuple[float, str]:
    if field in ("vendor", "invoice_number"):
        return _score_string_similarity(candidate, expected)
    if field == "currency":
        return _score_exact_string(candidate, expected)
    if field == "date":
        return _score_date(candidate, expected)
    if field in ("subtotal", "tax", "total"):
        return _score_number(candidate, expected)
    if field == "line_items":
        return _score_line_items(candidate, expected)
    return 0.0, f"unknown field {field!r}"


def score(output: Any, expected: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Score a candidate invoice extraction against ground truth.

    Returns (passed, details). `details` always includes `fields` (a dict of
    per-field {"score": float, "note": str}) and `aggregate_score` (the
    weighted-mean score in [0, 1] used against PASS_THRESHOLD).
    """
    try:
        candidate = _coerce_output(output)
    except Exception as exc:  # never let scoring itself crash the caller
        return False, {
            "error": f"scorer crashed while coercing output: {exc!r}",
            "aggregate_score": 0.0,
            "fields": {},
        }

    if candidate is None:
        fields = {f: {"score": 0.0, "note": "output could not be parsed into a dict"} for f in WEIGHTS}
        return False, {"aggregate_score": 0.0, "fields": fields, "error": "unparseable output"}

    fields: dict[str, dict[str, Any]] = {}
    weighted_sum = 0.0
    weight_total = 0.0
    for field, weight in WEIGHTS.items():
        try:
            field_score, note = _score_field(field, candidate.get(field), expected.get(field))
        except Exception as exc:  # a single malformed field must not crash the whole score
            field_score, note = 0.0, f"scoring crashed for this field: {exc!r}"
        fields[field] = {"score": field_score, "note": note}
        weighted_sum += field_score * weight
        weight_total += weight

    aggregate_score = weighted_sum / weight_total if weight_total else 0.0
    passed = aggregate_score >= PASS_THRESHOLD
    return passed, {"aggregate_score": aggregate_score, "fields": fields}
