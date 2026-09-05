from core import analyzer
from core.analyzer import MAX_CLUSTERS, analyze
from core.types import RunResult


def _result(**overrides) -> RunResult:
    defaults = dict(
        task_id="t1",
        output=None,
        passed=False,
        error=None,
        cost_usd=0.0,
        latency_ms=10,
        trace_id=None,
    )
    defaults.update(overrides)
    return RunResult(**defaults)


def test_analyze_returns_empty_list_when_no_failures():
    results = [_result(task_id="t1", passed=True), _result(task_id="t2", passed=True)]

    assert analyze(results, agent_path="agents/current") == []


def test_analyze_collapses_shared_error_pattern_into_one_cluster(monkeypatch):
    monkeypatch.setattr(analyzer, "_neatlogs_client", lambda: None)

    results = [
        _result(task_id="t1", error="IndexError: list index out of range: index 5"),
        _result(task_id="t2", error="IndexError: list index out of range: index 9"),
        _result(task_id="t3", error="IndexError: list index out of range: index 2"),
        _result(task_id="t4", error="IndexError: list index out of range: index 7"),
        # One-off failures that must NOT be folded into the pattern above.
        _result(task_id="t5", error="ValueError: could not convert string to float"),
        _result(task_id="t6", output={"unexpected": "shape"}, error=None),
    ]

    clusters = analyze(results, agent_path="agents/current")

    # The four IndexError failures collapse into a single cluster, not four
    # (or several near-duplicate clusters for the same underlying issue).
    index_error_clusters = [c for c in clusters if "IndexError" in c.label]
    assert len(index_error_clusters) == 1
    shared = index_error_clusters[0]
    assert shared.count == 4
    assert set(shared.example_task_ids) <= {"t1", "t2", "t3", "t4"}
    assert len(shared.example_task_ids) <= 3

    # The two one-off failures remain distinct from the shared cluster and
    # from each other.
    other_labels = {c.label for c in clusters if c is not shared}
    assert len(clusters) == 3
    assert all("IndexError" not in label for label in other_labels)


def test_analyze_sorts_clusters_by_count_descending(monkeypatch):
    monkeypatch.setattr(analyzer, "_neatlogs_client", lambda: None)

    results = (
        [_result(task_id=f"a{i}", error="TimeoutError: deadline exceeded") for i in range(2)]
        + [_result(task_id=f"b{i}", error="RuntimeError: boom") for i in range(5)]
        + [_result(task_id=f"c{i}", error="KeyError: 'missing'") for i in range(3)]
    )

    clusters = analyze(results, agent_path="agents/current")

    counts = [c.count for c in clusters]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 5


def test_analyze_caps_at_max_clusters_and_merges_overflow(monkeypatch):
    monkeypatch.setattr(analyzer, "_neatlogs_client", lambda: None)

    # Eight distinct one-off error patterns, sized so ordering is unambiguous.
    results = []
    for i in range(8):
        for j in range(8 - i):  # sizes: 8,7,6,5,4,3,2,1
            results.append(_result(task_id=f"g{i}-{j}", error=f"Error{i}: unique failure kind {i}"))

    clusters = analyze(results, agent_path="agents/current")

    assert len(clusters) <= MAX_CLUSTERS
    counts = [c.count for c in clusters]
    assert counts == sorted(counts, reverse=True)
    # Total task coverage across clusters must not silently drop failures.
    assert sum(counts) == len(results)


def test_analyze_uses_trace_step_to_group_failures_with_different_messages(monkeypatch):
    class FakeClient:
        def get_trace(self, trace_id):
            # Every trace diverges at the same step, "merge_pages", even
            # though the surface-level error text differs per task.
            return {"steps": [{"name": "extract_page", "error": None}, {"name": "merge_pages", "error": "boom"}]}

    monkeypatch.setattr(analyzer, "_neatlogs_client", lambda: FakeClient())

    results = [
        _result(task_id="p1", trace_id="tr-1", error="ValueError: page 2 missing"),
        _result(task_id="p2", trace_id="tr-2", error="ValueError: page 3 dropped"),
        _result(task_id="p3", trace_id="tr-3", error="ValueError: lost data on page 5"),
    ]

    clusters = analyze(results, agent_path="agents/current")

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.count == 3
    assert "merge_pages" in cluster.label
    assert set(cluster.trace_ids) == {"tr-1", "tr-2", "tr-3"}
    assert "merge_pages" in cluster.hypothesis


def test_analyze_falls_back_when_trace_fetch_fails(monkeypatch):
    class ExplodingClient:
        def get_trace(self, trace_id):
            raise ConnectionError("neatlogs unreachable")

    monkeypatch.setattr(analyzer, "_neatlogs_client", lambda: ExplodingClient())

    results = [
        _result(task_id="t1", trace_id="tr-1", error="RuntimeError: same failure"),
        _result(task_id="t2", trace_id="tr-2", error="RuntimeError: same failure"),
    ]

    clusters = analyze(results, agent_path="agents/current")

    assert len(clusters) == 1
    assert clusters[0].count == 2
    assert "low confidence" in clusters[0].hypothesis.lower()


def test_analyze_marks_no_trace_clusters_as_lower_confidence(monkeypatch):
    monkeypatch.setattr(analyzer, "_neatlogs_client", lambda: None)

    results = [
        _result(task_id="t1", error="RuntimeError: same failure"),
        _result(task_id="t2", error="RuntimeError: same failure"),
    ]

    clusters = analyze(results, agent_path="agents/current")

    assert len(clusters) == 1
    assert "low confidence" in clusters[0].hypothesis.lower()
