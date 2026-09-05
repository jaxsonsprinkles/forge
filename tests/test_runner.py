from pathlib import Path

import pytest

from core import llm, runner
from core.runner import run_agent
from core.types import TaskSpec

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATASET_PATH = str(FIXTURES_DIR / "dataset.jsonl")
SCORER_ID = "tests.fixtures.scorer:score_exact"
GOOD_AGENT = str(FIXTURES_DIR / "agents" / "good_agent")
BROKEN_AGENT = str(FIXTURES_DIR / "agents" / "broken_agent")


@pytest.fixture(autouse=True)
def _reset_spend_tracker():
    llm.reset_spend_tracker()
    yield
    llm.reset_spend_tracker()


def _task_spec(**overrides) -> TaskSpec:
    defaults = dict(
        domain_id="dummy",
        goal="add two numbers",
        tools=[],
        dataset_path=DATASET_PATH,
        scorer_id=SCORER_ID,
        max_tasks=20,
    )
    defaults.update(overrides)
    return TaskSpec(**defaults)


def test_run_agent_success_produces_passing_results():
    task_spec = _task_spec()

    results = run_agent(GOOD_AGENT, task_spec, split="train")

    assert len(results) == 3  # three "train" rows in the fixture dataset
    assert all(r.passed is True for r in results)
    assert all(r.error is None for r in results)
    assert all(r.latency_ms >= 0 for r in results)
    assert all(r.cost_usd >= 0 for r in results)


def test_run_agent_respects_split():
    task_spec = _task_spec()

    results = run_agent(GOOD_AGENT, task_spec, split="holdout")

    assert len(results) == 2  # two "holdout" rows in the fixture dataset
    assert {r.task_id for r in results} == {"t4", "t5"}


def test_run_agent_respects_max_tasks():
    task_spec = _task_spec(max_tasks=1)

    results = run_agent(GOOD_AGENT, task_spec, split="train")

    assert len(results) == 1


def test_run_agent_broken_agent_never_raises():
    task_spec = _task_spec()

    results = run_agent(BROKEN_AGENT, task_spec, split="train")

    assert len(results) == 3
    assert all(r.passed is False for r in results)
    assert all(r.error is not None for r in results)
    assert all("RuntimeError" in r.error for r in results)


def test_run_agent_broken_agent_produces_low_accuracy_scorecard():
    from core.scorer import score_runs

    task_spec = _task_spec()
    results = run_agent(BROKEN_AGENT, task_spec, split="train")
    card = score_runs(results, split="train")

    assert card.accuracy == 0.0
    assert card.reliability == 0.0
    assert card.n == 3


def test_trace_id_none_when_neatlogs_not_configured(monkeypatch):
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    task_spec = _task_spec(max_tasks=1)

    results = run_agent(GOOD_AGENT, task_spec, split="train")

    assert all(r.trace_id is None for r in results)


def test_trace_id_populated_when_neatlogs_configured(monkeypatch):
    class FakeSpan:
        def __init__(self, trace_id):
            self.id = trace_id

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    class FakeNeatlogsSDK:
        def __init__(self):
            self._n = 0

        def init(self, api_key):
            assert api_key == "fake-key"

        def trace(self, name, kind):
            self._n += 1
            return FakeSpan(trace_id=f"trace-{self._n}")

        def flush(self):
            pass

    monkeypatch.setattr(runner, "_neatlogs_sdk", FakeNeatlogsSDK())
    monkeypatch.setenv("NEATLOGS_API_KEY", "fake-key")

    task_spec = _task_spec()
    results = run_agent(GOOD_AGENT, task_spec, split="train")

    assert len(results) == 3
    assert all(r.trace_id is not None for r in results)
    # One "WORKFLOW" trace() call per task (the id-bearing one), plus a
    # nested "AGENT" and "GUARDRAIL" call each - so ids land 3 apart.
    assert {r.trace_id for r in results} == {"trace-1", "trace-4", "trace-7"}


def test_agent_current_opens_a_span_per_graph_step(monkeypatch):
    """Proves the trace_id/per-step-span code path is real, not just
    documented: with a fake SDK standing in for a configured Neatlogs key,
    running the actual agents/current graph interpreter (not the minimal
    good_agent fixture) must produce a non-null trace_id AND open a
    separate LLM-kind span per llm_call step, instead of collapsing the
    whole task into one "agent_run" span.
    """

    def fake_complete(messages, model, **params):
        return "final answer", 0.0, 1

    monkeypatch.setattr(llm, "complete", fake_complete)

    class FakeSpan:
        def __init__(self, trace_id):
            self.id = trace_id

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    class FakeNeatlogsSDK:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []

        def init(self, api_key):
            assert api_key == "fake-key"

        def trace(self, name, kind):
            self.calls.append((name, kind))
            return FakeSpan(trace_id=f"trace-{len(self.calls)}")

        def flush(self):
            pass

    fake_sdk = FakeNeatlogsSDK()
    monkeypatch.setattr(runner, "_neatlogs_sdk", fake_sdk)
    monkeypatch.setenv("NEATLOGS_API_KEY", "fake-key")

    task_spec = _task_spec(max_tasks=1)
    results = run_agent("agents/current", task_spec, split="train")

    assert len(results) == 1
    assert results[0].trace_id is not None

    kinds = [kind for _, kind in fake_sdk.calls]
    # Baseline graph.yaml has two llm_call steps (solve, finalize) and one
    # verify step (check_draft) - each must be its own span, not folded
    # into the single outer "agent_run" AGENT span.
    assert kinds.count("LLM") == 2
    assert "GUARDRAIL" in kinds
    assert kinds.count("AGENT") == 1


def test_neatlogs_trace_id_falls_back_to_otel_span_context():
    """The real neatlogs SDK's spans don't expose a top-level trace_id/id
    attribute (unlike the FakeSpan test doubles above) - they're
    OpenTelemetry spans, so the id lives on `get_span_context().trace_id`.
    Without this fallback, _neatlogs_trace_id would silently return None
    for every real, correctly-configured run.
    """

    class FakeSpanContext:
        trace_id = 0x1234ABCD1234ABCD1234ABCD1234ABCD

    class FakeOtelSpan:
        def get_span_context(self):
            return FakeSpanContext()

    assert runner._neatlogs_trace_id(FakeOtelSpan()) == format(FakeSpanContext.trace_id, "032x")


def test_trace_id_none_when_neatlogs_init_raises(monkeypatch):
    class ExplodingNeatlogsSDK:
        def init(self, api_key):
            raise ConnectionError("no network")

    monkeypatch.setattr(runner, "_neatlogs_sdk", ExplodingNeatlogsSDK())
    monkeypatch.setenv("NEATLOGS_API_KEY", "fake-key")

    task_spec = _task_spec(max_tasks=1)
    results = run_agent(GOOD_AGENT, task_spec, split="train")

    assert len(results) == 1
    assert results[0].trace_id is None
    assert results[0].passed is True
    assert results[0].error is None
