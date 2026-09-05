from core.scorer import score_runs
from core.types import RunResult


def _result(**overrides):
    defaults = dict(
        task_id="t1",
        output=None,
        passed=True,
        error=None,
        cost_usd=0.01,
        latency_ms=100,
        trace_id=None,
    )
    defaults.update(overrides)
    return RunResult(**defaults)


def test_score_runs_accuracy_and_reliability():
    results = [
        _result(task_id="t1", passed=True, error=None),
        _result(task_id="t2", passed=False, error=None),
        _result(task_id="t3", passed=False, error="boom"),
        _result(task_id="t4", passed=True, error=None),
    ]

    card = score_runs(results, split="train")

    assert card.accuracy == 0.5
    assert card.reliability == 0.75
    assert card.n == 4
    assert card.split == "train"


def test_score_runs_cost_and_latency_aggregation():
    results = [
        _result(cost_usd=0.01, latency_ms=100),
        _result(cost_usd=0.03, latency_ms=300),
        _result(cost_usd=0.02, latency_ms=200),
    ]

    card = score_runs(results, split="holdout")

    assert card.cost_per_task == 0.02
    assert card.p50_latency_ms == 200


def test_score_runs_empty_results():
    card = score_runs([], split="train")

    assert card.n == 0
    assert card.accuracy == 0.0
    assert card.reliability == 0.0
    assert card.cost_per_task == 0.0
    assert card.p50_latency_ms == 0
    assert card.split == "train"
