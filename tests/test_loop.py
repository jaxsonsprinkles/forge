"""Tests for core.loop.run_loop(). Every pipeline step (score/analyze/
propose/dispatch/gate) is injected as a fast, deterministic, offline
stand-in - no real git worktrees, no `ao` CLI, no LLM calls."""

from core import ledger, llm
from core.loop import run_loop
from core.types import FailureCluster, Mutation, RunResult, ScoreCard, TaskSpec


def _task_spec(domain_id: str = "d1") -> TaskSpec:
    return TaskSpec(
        domain_id=domain_id,
        goal="goal",
        tools=[],
        dataset_path="dataset.jsonl",
        scorer_id="module:score",
        max_tasks=5,
    )


def _card(**overrides) -> ScoreCard:
    defaults = dict(accuracy=0.5, reliability=0.9, cost_per_task=0.01, p50_latency_ms=100, n=5, split="train")
    defaults.update(overrides)
    return ScoreCard(**defaults)


def _result(**overrides) -> RunResult:
    defaults = dict(task_id="t1", output=None, passed=False, error=None, cost_usd=0.0, latency_ms=10, trace_id=None)
    defaults.update(overrides)
    return RunResult(**defaults)


def _mutation(**overrides) -> Mutation:
    defaults = dict(id="mut-00-prompt", surface="prompt", rationale="r", instruction="i", target_files=["prompt.md"])
    defaults.update(overrides)
    return Mutation(**defaults)


def _make_pipeline(*, pick_winner: bool = True):
    """A one-mutation-per-generation pipeline that always finds one failing
    cluster, proposes one mutation, dispatches it to a fake branch named
    after the parent ref, and (optionally) always picks it as the winner
    with a higher accuracy than baseline."""
    score_calls: list[str] = []

    def score_fn(ref, task_specs):
        score_calls.append(ref)
        cards = {ts.domain_id: _card(accuracy=0.5) for ts in task_specs}
        results = {ts.domain_id: [_result(task_id=f"{ts.domain_id}-1")] for ts in task_specs}
        return cards, results

    def analyze_fn(results, agent_path):
        if not results:
            return []
        return [
            FailureCluster(
                label="fails",
                count=len(results),
                example_task_ids=[r.task_id for r in results],
                trace_ids=[],
                hypothesis="h",
            )
        ]

    def propose_fn(clusters, agent_path, n):
        return [_mutation()] if clusters else []

    def dispatch_fn(mutations, parent_sha, project=None):
        return {m.id: f"branch-of-{parent_sha}" for m in mutations}

    def score_candidate_fn(branch, task_specs):
        cards = {ts.domain_id: _card(accuracy=0.9) for ts in task_specs}
        results = {ts.domain_id: [_result(task_id=f"{ts.domain_id}-1", passed=True)] for ts in task_specs}
        return cards, results

    def select_fn(candidates, baseline, **kwargs):
        return next(iter(candidates)) if pick_winner and candidates else None

    fns = dict(
        score_fn=score_fn,
        analyze_fn=analyze_fn,
        propose_fn=propose_fn,
        dispatch_fn=dispatch_fn,
        score_candidate_fn=score_candidate_fn,
        select_fn=select_fn,
    )
    return fns, score_calls


def test_run_loop_records_n_generations_with_chained_parent_sha(tmp_path):
    ledger_path = tmp_path / "generations.jsonl"
    fns, _score_calls = _make_pipeline()
    task_specs = [_task_spec("d1")]

    result = run_loop(task_specs, 3, starting_sha="sha0", ledger_path=ledger_path, **fns)

    assert [g.gen_n for g in result] == [1, 2, 3]
    assert result[0].parent_sha == "sha0"
    assert result[0].winner_id == "mut-00-prompt"
    assert result[0].winner_sha == "branch-of-sha0"
    assert result[1].parent_sha == "branch-of-sha0"
    assert result[1].winner_sha == "branch-of-branch-of-sha0"
    assert result[2].parent_sha == "branch-of-branch-of-sha0"

    loaded = ledger.load_generations(ledger_path)
    assert len(loaded) == 3


def test_run_loop_returns_immediately_when_already_complete(tmp_path):
    ledger_path = tmp_path / "generations.jsonl"
    fns, score_calls = _make_pipeline()
    task_specs = [_task_spec("d1")]

    run_loop(task_specs, 2, starting_sha="sha0", ledger_path=ledger_path, **fns)
    calls_after_first = len(score_calls)

    result = run_loop(task_specs, 2, starting_sha="sha0", ledger_path=ledger_path, **fns)

    assert len(result) == 2
    assert len(score_calls) == calls_after_first  # no new work done


def test_run_loop_resumes_after_simulated_crash_without_duplicating(tmp_path):
    ledger_path = tmp_path / "generations.jsonl"
    fns, score_calls = _make_pipeline()
    task_specs = [_task_spec("d1")]

    first = run_loop(task_specs, 1, starting_sha="sha0", ledger_path=ledger_path, **fns)
    assert len(first) == 1
    calls_after_first = len(score_calls)

    second = run_loop(task_specs, 3, starting_sha="sha0", ledger_path=ledger_path, **fns)

    assert [g.gen_n for g in second] == [1, 2, 3]
    assert second[0] == first[0]  # gen 1's record is untouched, not redone
    # score_fn was only invoked twice more (gen 2 and gen 3), not for gen 1 again
    assert len(score_calls) == calls_after_first + 2

    loaded = ledger.load_generations(ledger_path)
    assert len(loaded) == 3
    assert loaded[0] == first[0]


def test_run_loop_records_noop_generation_when_no_failures(tmp_path):
    ledger_path = tmp_path / "generations.jsonl"

    def score_fn(ref, task_specs):
        cards = {ts.domain_id: _card(accuracy=0.95) for ts in task_specs}
        results = {ts.domain_id: [_result(task_id=f"{ts.domain_id}-1", passed=True)] for ts in task_specs}
        return cards, results

    def analyze_fn(results, agent_path):
        return []

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("should not be called when there are no failure clusters")

    task_specs = [_task_spec("d1")]
    result = run_loop(
        task_specs,
        1,
        starting_sha="sha0",
        ledger_path=ledger_path,
        score_fn=score_fn,
        analyze_fn=analyze_fn,
        propose_fn=must_not_be_called,
        dispatch_fn=must_not_be_called,
        score_candidate_fn=must_not_be_called,
        select_fn=must_not_be_called,
    )

    assert len(result) == 1
    gen = result[0]
    assert gen.mutations == []
    assert gen.results == {}
    assert gen.winner_id is None
    assert gen.winner_sha is None


def test_run_loop_no_winner_keeps_parent_sha_unchanged(tmp_path):
    ledger_path = tmp_path / "generations.jsonl"
    fns, _ = _make_pipeline(pick_winner=False)
    task_specs = [_task_spec("d1")]

    result = run_loop(task_specs, 2, starting_sha="sha0", ledger_path=ledger_path, **fns)

    assert result[0].winner_id is None
    assert result[0].winner_sha is None
    assert result[1].parent_sha == "sha0"  # unchanged - no winner in gen 1


def test_run_loop_spend_cap_exceeded_aborts_cleanly_without_corrupting_ledger(tmp_path):
    ledger_path = tmp_path / "generations.jsonl"
    fns, _ = _make_pipeline()
    real_score_fn = fns["score_fn"]

    calls = {"n": 0}

    def flaky_score_fn(ref, task_specs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise llm.SpendCeilingExceeded("boom")
        return real_score_fn(ref, task_specs)

    fns["score_fn"] = flaky_score_fn
    task_specs = [_task_spec("d1")]

    result = run_loop(task_specs, 3, starting_sha="sha0", ledger_path=ledger_path, **fns)

    assert len(result) == 1  # only gen 1 completed; gen 2 aborted before recording

    loaded = ledger.load_generations(ledger_path)
    assert len(loaded) == 1

    text = ledger_path.read_text()
    assert text.count("\n") == 1
    assert text.endswith("\n")
