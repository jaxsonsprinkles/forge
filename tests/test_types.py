from core.types import (
    FailureCluster,
    Generation,
    Mutation,
    RunResult,
    ScoreCard,
    TaskSpec,
)


def test_task_spec_defaults_max_tasks():
    spec = TaskSpec(
        domain_id="coderepair",
        goal="fix the failing test",
        tools=["read_file", "write_file"],
        dataset_path="domains/coderepair/dataset.jsonl",
        scorer_id="coderepair_v1",
    )
    assert spec.max_tasks == 20


def test_run_result_roundtrip():
    result = RunResult(
        task_id="t1",
        output={"diff": "..."},
        passed=True,
        error=None,
        cost_usd=0.01,
        latency_ms=120,
        trace_id="trace-1",
    )
    assert result.passed is True
    assert result.error is None


def test_score_card_fields():
    card = ScoreCard(
        accuracy=0.8,
        reliability=0.95,
        cost_per_task=0.02,
        p50_latency_ms=150,
        n=20,
        split="test",
    )
    assert card.n == 20


def test_failure_cluster_fields():
    cluster = FailureCluster(
        label="timeout on large diffs",
        count=3,
        example_task_ids=["t1", "t2", "t3"],
        trace_ids=["tr1", "tr2", "tr3"],
        hypothesis="agent runs out of tool budget on large repos",
    )
    assert cluster.count == len(cluster.example_task_ids)


def test_mutation_surface_literal():
    mutation = Mutation(
        id="m1",
        surface="prompt",
        rationale="agent ignores edge cases",
        instruction="add an explicit edge-case checklist",
        target_files=["agents/current/prompt.md"],
    )
    assert mutation.surface == "prompt"


def test_generation_composes_scorecards_and_mutations():
    scores_before = {
        "coderepair": ScoreCard(
            accuracy=0.5,
            reliability=0.9,
            cost_per_task=0.01,
            p50_latency_ms=100,
            n=20,
            split="test",
        )
    }
    mutation = Mutation(
        id="m1",
        surface="tool",
        rationale="tool times out",
        instruction="add a retry",
        target_files=["agents/current/tools.py"],
    )
    generation = Generation(
        gen_n=1,
        parent_sha="abc123",
        scores_before=scores_before,
        mutations=[mutation],
        results={"m1": scores_before},
        winner_id="m1",
        winner_sha="def456",
    )
    assert generation.winner_id == "m1"
    assert generation.results["m1"]["coderepair"].accuracy == 0.5
