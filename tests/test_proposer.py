from core.proposer import propose
from core.types import FailureCluster

_VALID_AGENT_FILES = {"prompt.md", "tools.py", "memory.py", "graph.yaml", "run.py"}


def _cluster(**overrides) -> FailureCluster:
    defaults = dict(
        label="generic failure",
        count=1,
        example_task_ids=["t1"],
        trace_ids=[],
        hypothesis="unspecified",
    )
    defaults.update(overrides)
    return FailureCluster(**defaults)


def _clusters_with_structural_case() -> list[FailureCluster]:
    return [
        _cluster(
            label="loses data on multi-page inputs",
            count=8,
            example_task_ids=["t1", "t2", "t3"],
            hypothesis="the agent skips verification step before final answer and drops line "
            "items from pages after the first",
        ),
        _cluster(
            label="wrong output format",
            count=5,
            example_task_ids=["t4", "t5"],
            hypothesis="the agent's answer doesn't follow the requested JSON schema wording",
        ),
        _cluster(
            label="tool call fails with invalid argument",
            count=3,
            example_task_ids=["t6"],
            hypothesis="the tool call passes a malformed argument and the api error is unhandled",
        ),
        _cluster(
            label="forgets earlier step result",
            count=2,
            example_task_ids=["t7"],
            hypothesis="the agent forgets context from an earlier step and does not remember it",
        ),
    ]


def test_propose_includes_orchestration_when_structural_cluster_present():
    clusters = _clusters_with_structural_case()

    mutations = propose(clusters, agent_path="agents/current", n=4)

    surfaces = [m.surface for m in mutations]
    assert "orchestration" in surfaces


def test_propose_spreads_across_surfaces():
    clusters = _clusters_with_structural_case()

    mutations = propose(clusters, agent_path="agents/current", n=4)

    surfaces = {m.surface for m in mutations}
    assert len(surfaces) > 1


def test_propose_target_files_are_valid_agent_files():
    clusters = _clusters_with_structural_case()

    mutations = propose(clusters, agent_path="agents/current", n=4)

    assert mutations
    for m in mutations:
        assert m.target_files, m
        assert set(m.target_files) <= _VALID_AGENT_FILES


def test_propose_honors_surface_filter():
    clusters = _clusters_with_structural_case()

    mutations = propose(clusters, agent_path="agents/current", n=4, surface_filter=["prompt"])

    assert mutations
    assert all(m.surface == "prompt" for m in mutations)
    assert all(m.target_files == ["prompt.md"] for m in mutations)


def test_propose_rationale_cites_cluster_label():
    clusters = _clusters_with_structural_case()

    mutations = propose(clusters, agent_path="agents/current", n=4)

    labels = {c.label for c in clusters}
    for m in mutations:
        assert any(label in m.rationale for label in labels)


def test_propose_instructions_name_their_target_file():
    clusters = _clusters_with_structural_case()

    mutations = propose(clusters, agent_path="agents/current", n=4)

    for m in mutations:
        assert m.target_files[0] in m.instruction


def test_propose_returns_empty_for_no_clusters():
    assert propose([], agent_path="agents/current", n=4) == []


def test_propose_does_not_force_orchestration_without_structural_signal():
    clusters = [
        _cluster(
            label="wrong output format",
            count=5,
            hypothesis="the agent's answer doesn't follow the requested JSON schema wording",
        ),
        _cluster(
            label="tool call fails with invalid argument",
            count=3,
            hypothesis="the tool call passes a malformed argument and the api error is unhandled",
        ),
    ]

    mutations = propose(clusters, agent_path="agents/current", n=4, surface_filter=["prompt", "tool"])

    assert all(m.surface in ("prompt", "tool") for m in mutations)
