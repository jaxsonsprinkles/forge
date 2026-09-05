"""Scorer for the github_triage domain.

Dataset format (see dataset.jsonl): each record's `expected` is
{"labels": [...], "assignee": "<username>"|null, "priority":
"low"|"medium"|"high"|"critical"}. A candidate `output` is expected to be
a dict with the same three keys (an agent's run.py return shape isn't
dictated by this domain, but this scorer only knows how to grade this
shape; anything else - a string, None, a dict missing these keys -
scores 0 on the affected dimension(s) rather than raising).

Three independently-scored dimensions, combined into one `passed` bool:

  labels    - set F1 over label name strings (exact, case-sensitive
              match: this repo's real label set mixes casing
              conventions, e.g. "bug" alongside "Invalid"/"Stale", and
              getting the casing right is part of the convention being
              tested here, not incidental noise to normalize away).
  assignee  - exact match, case-insensitive (GitHub logins are
              case-insensitive) after stripping whitespace. None/null on
              both sides counts as a match - this repo leaves plenty of
              issues unassigned, and correctly predicting "no assignee"
              is itself a repo-specific behavior worth crediting, not a
              default to ignore.
  priority  - ordinal match over PRIORITY_LEVELS. Exact level = full
              credit (1.0), one level off = partial credit (0.5), two or
              more levels off = no credit (0.0). Priority here is
              inherently fuzzy (derived from close-time, not an explicit
              repo label - see generate_dataset.py's DERIVATION note), so
              a near-miss shouldn't score identically to a wildly wrong
              guess.

composite_score = 0.5*label_f1 + 0.3*assignee_score + 0.2*priority_score
passed = composite_score >= PASS_THRESHOLD (0.7)

Weights: labels carry the largest share because they're multi-valued (a
set-F1 in [0, 1] already reflects partial correctness on its own, unlike
the other two all-or-nothing/ordinal dimensions); assignee (a hard
binary signal) outweighs priority (a soft/ordinal one, which already
gets its own partial credit). PASS_THRESHOLD=0.7 was picked so that
acing any *one* dimension alone is never enough to pass (labels alone:
0.5; assignee alone: 0.3; priority alone: 0.2), but two of the three
being right (or close) is: perfect labels + right assignee = 0.8; right
labels + right priority = 0.7; right assignee + right priority alone
(wrong labels) = 0.5, still fails, since labels is where most of a
prediction's information content lives for this task.

Entirely offline: this module makes no network calls and never imports
the tools/ package - it only compares `output` to `expected`.
"""

from __future__ import annotations

from typing import Any

PRIORITY_LEVELS = ["low", "medium", "high", "critical"]
PASS_THRESHOLD = 0.7

LABEL_WEIGHT = 0.5
ASSIGNEE_WEIGHT = 0.3
PRIORITY_WEIGHT = 0.2


def _as_label_set(labels: Any) -> set[str]:
    if not isinstance(labels, (list, tuple, set)):
        return set()
    return {str(label).strip() for label in labels if str(label).strip()}


def _label_f1(predicted: Any, expected: Any) -> tuple[float, float, float]:
    """Precision/recall/F1 of `predicted` labels against `expected` labels.

    Both-empty counts as a perfect match (correctly predicting "no
    labels apply"); either-but-not-both empty scores 0 across the board.
    """
    pred_set = _as_label_set(predicted)
    exp_set = _as_label_set(expected)

    if not pred_set and not exp_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not exp_set:
        return 0.0, 0.0, 0.0

    true_positives = len(pred_set & exp_set)
    precision = true_positives / len(pred_set)
    recall = true_positives / len(exp_set)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _normalize_assignee(assignee: Any) -> str | None:
    if assignee is None:
        return None
    text = str(assignee).strip()
    return text.lower() if text else None


def _assignee_score(predicted: Any, expected: Any) -> float:
    return 1.0 if _normalize_assignee(predicted) == _normalize_assignee(expected) else 0.0


def _priority_score(predicted: Any, expected: Any) -> tuple[float, int | None]:
    try:
        pred_idx = PRIORITY_LEVELS.index(str(predicted).strip().lower())
    except (ValueError, AttributeError):
        pred_idx = None
    try:
        exp_idx = PRIORITY_LEVELS.index(str(expected).strip().lower())
    except (ValueError, AttributeError):
        exp_idx = None

    if pred_idx is None or exp_idx is None:
        return 0.0, None

    distance = abs(pred_idx - exp_idx)
    if distance == 0:
        return 1.0, distance
    if distance == 1:
        return 0.5, distance
    return 0.0, distance


def score(output: Any, expected: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Score a candidate's predicted labels/assignee/priority against `expected`.

    Never raises: a non-dict `output`/`expected`, or one missing the
    keys this scorer looks for, is treated as an empty/absent prediction
    on the affected dimension(s) rather than an error.
    """
    try:
        predicted = output if isinstance(output, dict) else {}
        exp = expected if isinstance(expected, dict) else {}

        precision, recall, label_f1 = _label_f1(predicted.get("labels"), exp.get("labels"))
        assignee_score = _assignee_score(predicted.get("assignee"), exp.get("assignee"))
        priority_score, priority_distance = _priority_score(predicted.get("priority"), exp.get("priority"))

        composite = LABEL_WEIGHT * label_f1 + ASSIGNEE_WEIGHT * assignee_score + PRIORITY_WEIGHT * priority_score
        passed = composite >= PASS_THRESHOLD

        return passed, {
            "label_precision": precision,
            "label_recall": recall,
            "label_f1": label_f1,
            "assignee_score": assignee_score,
            "priority_score": priority_score,
            "priority_distance": priority_distance,
            "composite_score": composite,
        }
    except Exception as exc:  # noqa: BLE001 - a malformed submission must never crash the scorer
        return False, {
            "label_precision": 0.0,
            "label_recall": 0.0,
            "label_f1": 0.0,
            "assignee_score": 0.0,
            "priority_score": 0.0,
            "priority_distance": None,
            "composite_score": 0.0,
            "error": f"scorer crashed: {exc!r}",
        }


def score_v1(output: Any, expected: dict[str, Any]) -> bool:
    """Bool-only wrapper of `score`, for use as this domain's scorer_id target.

    See domains/coderepair/scorer.py's identical score_v1 for why this
    wrapper is required: core.runner.run_agent does
    `bool(scorer_fn(output, expected))`, and a non-empty (passed, details)
    tuple is always truthy under bool() regardless of `passed`.
    """
    return score(output, expected)[0]
