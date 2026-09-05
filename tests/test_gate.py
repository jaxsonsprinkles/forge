import json

from core.gate import select
from core.types import RunResult, ScoreCard


def _card(**overrides) -> ScoreCard:
    defaults = dict(
        accuracy=0.80,
        reliability=0.95,
        cost_per_task=0.05,
        p50_latency_ms=1000,
        n=20,
        split="train",
    )
    defaults.update(overrides)
    return ScoreCard(**defaults)


def _baseline(**overrides) -> dict[str, ScoreCard]:
    card = _card(**overrides)
    return {"domain_a": card}


def _result(**overrides) -> RunResult:
    defaults = dict(
        task_id="t1",
        output=None,
        passed=True,
        error=None,
        cost_usd=0.0,
        latency_ms=10,
        trace_id=None,
    )
    defaults.update(overrides)
    return RunResult(**defaults)


def test_candidate_improving_accuracy_is_accepted(tmp_path):
    baseline = _baseline(accuracy=0.80)
    candidates = {"cand-1": {"domain_a": _card(accuracy=0.85)}}

    winner = select(
        candidates,
        baseline,
        watchlist=[],
        archive_root=tmp_path / "archive",
    )

    assert winner == "cand-1"
    assert (tmp_path / "archive" / "cand-1" / "scorecards.json").exists()


def test_candidate_degrading_accuracy_beyond_tolerance_is_rejected(tmp_path):
    baseline = _baseline(accuracy=0.80)
    # 0.80 - 0.02 = 0.78 floor; 0.70 breaches it even though cost improves a lot.
    candidates = {"cand-1": {"domain_a": _card(accuracy=0.70, cost_per_task=0.01)}}

    winner = select(
        candidates,
        baseline,
        watchlist=[],
        archive_root=tmp_path / "archive",
    )

    assert winner is None
    assert not (tmp_path / "archive" / "cand-1").exists()


def test_candidate_trading_small_accuracy_loss_for_big_cost_win_is_accepted(tmp_path):
    baseline = _baseline(accuracy=0.80, cost_per_task=0.10)
    # -1.5 accuracy points (within the 2-point tolerance), -40% cost.
    candidates = {"cand-1": {"domain_a": _card(accuracy=0.785, cost_per_task=0.06)}}

    winner = select(
        candidates,
        baseline,
        watchlist=[],
        archive_root=tmp_path / "archive",
    )

    assert winner == "cand-1"


def test_candidate_failing_watchlist_task_is_rejected_even_if_scores_look_good(tmp_path):
    baseline = _baseline(accuracy=0.80)
    candidates = {"cand-1": {"domain_a": _card(accuracy=0.95)}}
    candidate_results = {
        "cand-1": [
            _result(task_id="watched-1", passed=False),
            _result(task_id="other", passed=True),
        ]
    }

    winner = select(
        candidates,
        baseline,
        candidate_results=candidate_results,
        watchlist=["watched-1"],
        archive_root=tmp_path / "archive",
    )

    assert winner is None
    assert not (tmp_path / "archive" / "cand-1").exists()


def test_select_returns_none_when_no_candidates_qualify(tmp_path):
    baseline = _baseline(accuracy=0.80)
    candidates = {"cand-1": {"domain_a": _card(accuracy=0.80, cost_per_task=0.05)}}

    winner = select(
        candidates,
        baseline,
        watchlist=[],
        archive_root=tmp_path / "archive",
    )

    assert winner is None


def test_non_dominated_candidates_are_all_archived_dominated_one_is_not(tmp_path):
    baseline = _baseline(accuracy=0.80, cost_per_task=0.10, p50_latency_ms=1000)
    candidates = {
        # Best accuracy, worse cost than cand-2.
        "cand-1": {"domain_a": _card(accuracy=0.90, cost_per_task=0.09, p50_latency_ms=1000)},
        # Slightly lower accuracy but much cheaper - not dominated by cand-1.
        "cand-2": {"domain_a": _card(accuracy=0.85, cost_per_task=0.05, p50_latency_ms=1000)},
        # Strictly worse than cand-1 on every metric -> dominated, excluded from archive.
        "cand-3": {"domain_a": _card(accuracy=0.82, cost_per_task=0.095, p50_latency_ms=1000)},
    }

    winner = select(
        candidates,
        baseline,
        watchlist=[],
        archive_root=tmp_path / "archive",
    )

    assert winner == "cand-1"
    assert (tmp_path / "archive" / "cand-1").exists()
    assert (tmp_path / "archive" / "cand-2").exists()
    assert not (tmp_path / "archive" / "cand-3").exists()


def test_candidate_refs_are_written_alongside_scorecards(tmp_path):
    baseline = _baseline(accuracy=0.80)
    candidates = {"cand-1": {"domain_a": _card(accuracy=0.90)}}

    select(
        candidates,
        baseline,
        watchlist=[],
        candidate_refs={"cand-1": "forge/abc123/mut-00-prompt"},
        archive_root=tmp_path / "archive",
    )

    ref_file = tmp_path / "archive" / "cand-1" / "ref.txt"
    assert ref_file.read_text() == "forge/abc123/mut-00-prompt"


def test_watchlist_loaded_from_file_when_not_passed_explicitly(tmp_path):
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(json.dumps(["watched-1"]))

    baseline = _baseline(accuracy=0.80)
    candidates = {"cand-1": {"domain_a": _card(accuracy=0.95)}}
    candidate_results = {"cand-1": [_result(task_id="watched-1", passed=False)]}

    winner = select(
        candidates,
        baseline,
        candidate_results=candidate_results,
        watchlist_path=watchlist_path,
        archive_root=tmp_path / "archive",
    )

    assert winner is None
