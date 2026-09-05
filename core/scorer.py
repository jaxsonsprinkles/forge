"""Aggregates per-task RunResults into a ScoreCard for one split."""

from __future__ import annotations

from statistics import median

from core.types import RunResult, ScoreCard


def score_runs(results: list[RunResult], split: str) -> ScoreCard:
    """Aggregate a list of RunResults into a single ScoreCard.

    accuracy is the fraction with `passed is True`, reliability is the
    fraction with `error is None`, cost_per_task and p50_latency_ms are
    the mean cost and median latency across all results. An empty
    `results` list produces an all-zero ScoreCard rather than raising.
    """
    n = len(results)
    if n == 0:
        return ScoreCard(
            accuracy=0.0,
            reliability=0.0,
            cost_per_task=0.0,
            p50_latency_ms=0,
            n=0,
            split=split,
        )

    accuracy = sum(1 for r in results if r.passed is True) / n
    reliability = sum(1 for r in results if r.error is None) / n
    cost_per_task = sum(r.cost_usd for r in results) / n
    p50_latency_ms = int(median(r.latency_ms for r in results))

    return ScoreCard(
        accuracy=accuracy,
        reliability=reliability,
        cost_per_task=cost_per_task,
        p50_latency_ms=p50_latency_ms,
        n=n,
        split=split,
    )
