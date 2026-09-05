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
        def end(self):
            pass

    class FakeTrace:
        def __init__(self, trace_id):
            self.id = trace_id

        def start_span(self, name):
            return FakeSpan()

        def end(self):
            pass

    class FakeTracer:
        def __init__(self):
            self._n = 0

        def start_trace(self, name):
            self._n += 1
            return FakeTrace(trace_id=f"trace-{self._n}")

    class FakeNeatlogsSDK:
        def init(self, api_key):
            assert api_key == "fake-key"
            return FakeTracer()

    monkeypatch.setattr(runner, "_neatlogs_sdk", FakeNeatlogsSDK())
    monkeypatch.setenv("NEATLOGS_API_KEY", "fake-key")

    task_spec = _task_spec()
    results = run_agent(GOOD_AGENT, task_spec, split="train")

    assert len(results) == 3
    assert all(r.trace_id is not None for r in results)
    assert {r.trace_id for r in results} == {"trace-1", "trace-2", "trace-3"}


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
