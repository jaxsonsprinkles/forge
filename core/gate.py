"""Decides which re-scored candidate, if any, becomes the next generation.

`select(candidates, baseline)` takes one generation's candidates - each a
mapping of `domain_id -> ScoreCard` re-scored on the train split - and the
current agent's own `domain_id -> ScoreCard` baseline on the same split,
and returns the winning candidate's id, or `None` if nothing qualifies.

A candidate is only eligible if BOTH hold, across every domain in
`baseline`:
  1. Accuracy floor: its accuracy is never more than
     `ACCURACY_FLOOR_TOLERANCE` (0.02) below baseline on any domain.
  2. Meaningful improvement: averaged across domains, at least one of
     accuracy/reliability/cost_per_task/p50_latency_ms improves past its
     threshold (`MIN_ACCURACY_GAIN`, `MIN_RELIABILITY_GAIN`,
     `MIN_COST_REDUCTION`, `MIN_LATENCY_REDUCTION` below - the spec left
     the exact numbers to us, chosen here and documented alongside each
     constant).

Every eligible candidate that isn't Pareto-dominated by another eligible
candidate (see `_dominates`) gets archived under `agents/archive/<id>/`.
The single best of that non-dominated set - by accuracy, then cost, then
latency, then id, all deterministic tie-breaks - is returned as the
winner.

Signature note: the spec's base signature is `select(candidates,
baseline)`. It's extended here with keyword-only parameters for the
regression guard it also asks for: `candidate_results` (per-candidate
`list[RunResult]`, since a watchlisted task_id's pass/fail isn't
recoverable from an aggregate ScoreCard alone) and `watchlist` (defaults
to reading `ledger/watchlist.json`). A candidate that fails any
watchlisted task_id is rejected outright, before the floor/improvement
checks run at all - see `_fails_watchlist`. `candidate_refs` is also
optional and only affects what gets written into the archive (a git
ref/branch string alongside the ScoreCards, when the caller - e.g.
core/executor_ao.dispatch()'s return value - has one).

`ledger/watchlist.json` format: a flat JSON array of task_id strings
pulled from previously-fixed FailureClusters' `example_task_ids` (see
core/analyzer.py). Once a task_id is on the list, no future candidate is
allowed to reintroduce a failure on it, no matter how good its aggregate
scores look. The seed file ships as `[]` - nothing is watched yet.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from core.types import RunResult, ScoreCard

logger = logging.getLogger(__name__)

# A candidate may lose up to this many accuracy points, per domain,
# relative to baseline, and still be eligible.
ACCURACY_FLOOR_TOLERANCE = 0.02

# Thresholds for "improves meaningfully" (checked on the across-domain
# mean of each metric). Chosen per the spec's examples (accuracy +1%,
# cost -10%, latency -10%); reliability is given the same +1% bar as
# accuracy since both are 0-1 pass-rate-shaped metrics.
MIN_ACCURACY_GAIN = 0.01
MIN_RELIABILITY_GAIN = 0.01
MIN_COST_REDUCTION = 0.10
MIN_LATENCY_REDUCTION = 0.10

DEFAULT_WATCHLIST_PATH = Path("ledger/watchlist.json")
DEFAULT_ARCHIVE_ROOT = Path("agents/archive")


def _load_watchlist(path: str | Path = DEFAULT_WATCHLIST_PATH) -> list[str]:
    """Load the flat JSON array of watched task_ids. Missing file -> []."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{p} must contain a JSON array of task_id strings")
    return list(data)


def _fails_watchlist(results: list[RunResult], watchlist: list[str]) -> bool:
    """True if any watchlisted task_id shows up failing in `results`."""
    if not watchlist:
        return False
    watched = set(watchlist)
    return any(r.task_id in watched and not r.passed for r in results)


def _mean(cards: dict[str, ScoreCard], domain_ids: list[str], attr: str) -> float:
    values = [getattr(cards[d], attr) for d in domain_ids if d in cards]
    return sum(values) / len(values) if values else 0.0


def _passes_accuracy_floor(
    cand: dict[str, ScoreCard], base: dict[str, ScoreCard], domain_ids: list[str]
) -> bool:
    for domain_id in domain_ids:
        base_card = base[domain_id]
        cand_card = cand.get(domain_id)
        if cand_card is None:
            return False
        if cand_card.accuracy < base_card.accuracy - ACCURACY_FLOOR_TOLERANCE:
            return False
    return True


def _improves_meaningfully(
    cand: dict[str, ScoreCard], base: dict[str, ScoreCard], domain_ids: list[str]
) -> bool:
    base_acc = _mean(base, domain_ids, "accuracy")
    cand_acc = _mean(cand, domain_ids, "accuracy")
    if cand_acc >= base_acc + MIN_ACCURACY_GAIN:
        return True

    base_rel = _mean(base, domain_ids, "reliability")
    cand_rel = _mean(cand, domain_ids, "reliability")
    if cand_rel >= base_rel + MIN_RELIABILITY_GAIN:
        return True

    base_cost = _mean(base, domain_ids, "cost_per_task")
    cand_cost = _mean(cand, domain_ids, "cost_per_task")
    if base_cost > 0 and cand_cost <= base_cost * (1 - MIN_COST_REDUCTION):
        return True

    base_lat = _mean(base, domain_ids, "p50_latency_ms")
    cand_lat = _mean(cand, domain_ids, "p50_latency_ms")
    if base_lat > 0 and cand_lat <= base_lat * (1 - MIN_LATENCY_REDUCTION):
        return True

    return False


def _metrics_vector(cards: dict[str, ScoreCard], domain_ids: list[str]) -> tuple[float, float, float, float]:
    """(accuracy, reliability, -cost_per_task, -p50_latency_ms) - all "higher is better"."""
    return (
        _mean(cards, domain_ids, "accuracy"),
        _mean(cards, domain_ids, "reliability"),
        -_mean(cards, domain_ids, "cost_per_task"),
        -_mean(cards, domain_ids, "p50_latency_ms"),
    )


def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """True if `a` Pareto-dominates `b`: >= on every metric, > on at least one."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _archive_candidate(
    candidate_id: str,
    cards: dict[str, ScoreCard],
    archive_root: str | Path,
    ref: str | None,
) -> None:
    dest = Path(archive_root) / candidate_id
    dest.mkdir(parents=True, exist_ok=True)
    scorecards = {domain_id: asdict(card) for domain_id, card in cards.items()}
    with open(dest / "scorecards.json", "w") as f:
        json.dump(scorecards, f, indent=2, sort_keys=True)
    if ref is not None:
        (dest / "ref.txt").write_text(ref)


def select(
    candidates: dict[str, dict[str, ScoreCard]],
    baseline: dict[str, ScoreCard],
    *,
    candidate_results: dict[str, list[RunResult]] | None = None,
    watchlist: list[str] | None = None,
    watchlist_path: str | Path = DEFAULT_WATCHLIST_PATH,
    candidate_refs: dict[str, str] | None = None,
    archive_root: str | Path = DEFAULT_ARCHIVE_ROOT,
) -> str | None:
    """Pick a generation's winner, if any, and archive the non-dominated set.

    See module docstring for the full acceptance rule and the signature
    extension over the spec's `select(candidates, baseline)`.
    """
    if watchlist is None:
        watchlist = _load_watchlist(watchlist_path)
    candidate_results = candidate_results or {}
    candidate_refs = candidate_refs or {}
    domain_ids = list(baseline.keys())

    eligible: list[str] = []
    for candidate_id, cards in candidates.items():
        results = candidate_results.get(candidate_id, [])
        if _fails_watchlist(results, watchlist):
            logger.info("candidate %s rejected: fails a watchlisted task_id", candidate_id)
            continue
        if not _passes_accuracy_floor(cards, baseline, domain_ids):
            logger.info("candidate %s rejected: breaches accuracy floor", candidate_id)
            continue
        if not _improves_meaningfully(cards, baseline, domain_ids):
            logger.info("candidate %s rejected: no meaningful improvement over baseline", candidate_id)
            continue
        eligible.append(candidate_id)

    if not eligible:
        return None

    vectors = {cid: _metrics_vector(candidates[cid], domain_ids) for cid in eligible}
    non_dominated = [
        cid
        for cid in eligible
        if not any(_dominates(vectors[other], vectors[cid]) for other in eligible if other != cid)
    ]

    for cid in non_dominated:
        _archive_candidate(cid, candidates[cid], archive_root, candidate_refs.get(cid))

    winner = min(
        non_dominated,
        key=lambda cid: (-vectors[cid][0], -vectors[cid][2], -vectors[cid][3], cid),
    )
    return winner
