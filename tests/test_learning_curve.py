"""Tests for evals/learning_curve.py: the memory-on-vs-off learning curve harness.

`core.llm.complete()` is monkeypatched throughout (the same seam
test_reflect.py and test_agent_current.py use) so these tests are fast,
offline, and deterministic - the agent under test here
(tests/fixtures/agents/good_agent) never calls it itself, only
core.reflect.reflect() does, when memory is on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import llm, memory, reflect
from core.memory import MemoryEntry
from evals import learning_curve

FIXTURES_DOMAINS_ROOT = str(Path(__file__).parent / "fixtures" / "domains")
GOOD_AGENT = str(Path(__file__).parent / "fixtures" / "agents" / "good_agent")
DOMAIN = "dummy"

# tests/fixtures/dataset.jsonl: t1,t2,t3 train (a+b matches expected); t4,t5 holdout.


@pytest.fixture(autouse=True)
def _reset_spend_tracker():
    llm.reset_spend_tracker()
    yield
    llm.reset_spend_tracker()


def _fake_reflect_complete():
    """Returns one brand-new, mutually dissimilar lesson candidate per call.

    Trigger/content use word tokens (never digits), so they never
    keyword-match tests/fixtures/dataset.jsonl's numeric task_input
    (e.g. {"a": 1, "b": 2}) - retrieve() naturally returns nothing for
    them, keeping existing_memory empty on every reflect() call, which
    means every candidate is treated as brand new (see reflect.py's
    _find_similar_entry) rather than folded into an update. That's what
    lets a plain call counter prove monotonic accumulation.
    """
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliett"]
    calls = {"n": 0}

    def fake(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
        word = words[calls["n"] % len(words)]
        calls["n"] += 1
        payload = [
            {
                "content": f"lesson {word} about numeric addition strategy",
                "trigger": f"topic {word}",
                "scope": "rule",
                "confidence": 0.6,
            }
        ]
        return json.dumps(payload), 0.0, 5

    return fake


def _fail_if_called(name: str):
    def fake(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{name} should not have been called")

    return fake


def test_memory_on_accumulates_entries_across_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "complete", _fake_reflect_complete())

    records = learning_curve.run_learning_curve(
        DOMAIN,
        passes=3,
        memory_mode="on",
        agent_path=GOOD_AGENT,
        domains_root=FIXTURES_DOMAINS_ROOT,
        memory_base_dir=tmp_path / "mem",
        output_dir=tmp_path / "out",
    )

    counts = [r["memory_entry_count"] for r in records]
    assert counts == sorted(counts), "memory_entry_count must be non-decreasing across passes"
    assert counts[-1] > counts[0] > 0, "memory should actually grow, not just stay flat"


def test_memory_off_never_calls_reflect_or_write(tmp_path, monkeypatch):
    monkeypatch.setattr(reflect, "reflect", _fail_if_called("reflect.reflect"))
    monkeypatch.setattr(memory, "write", _fail_if_called("memory.write"))
    monkeypatch.setattr(memory, "retrieve", _fail_if_called("memory.retrieve"))
    monkeypatch.setattr(memory, "reinforce", _fail_if_called("memory.reinforce"))

    records = learning_curve.run_learning_curve(
        DOMAIN,
        passes=3,
        memory_mode="off",
        agent_path=GOOD_AGENT,
        domains_root=FIXTURES_DOMAINS_ROOT,
        memory_base_dir=tmp_path / "mem",
        output_dir=tmp_path / "out",
    )

    assert len(records) == 3
    assert all(r["memory_entry_count"] == 0 for r in records)
    assert all(r["avg_entries_retrieved"] == 0.0 for r in records)


def test_holdout_never_writes_memory_even_when_it_retrieves(tmp_path, monkeypatch):
    """Holdout is read-only: it must retrieve accumulated memory (so the
    curve reflects what's been learned) but never call memory.write(),
    not even via retrieve()'s own times_retrieved side effect."""
    base_dir = tmp_path / "mem"
    domain_id = DOMAIN

    # Seed memory directly (not via retrieve/reflect) with an entry whose
    # trigger matches holdout task t4's input ({"a": 5, "b": 5}).
    seeded = MemoryEntry(
        id="mem-seeded",
        domain_id=domain_id,
        scope="rule",
        content="always add both operands directly",
        trigger="a value of 5",
        evidence_task_ids=[],
        source_run_id="seed",
        created_gen=0,
        confidence=0.9,
    )
    memory.write(seeded, base_dir=base_dir)

    received_inputs: list[dict[str, Any]] = []

    def recording_run_fn(task_input: dict[str, Any]) -> dict[str, Any]:
        received_inputs.append(task_input)
        return {"answer": task_input["a"] + task_input["b"]}

    def score_exact(output: Any, expected: Any) -> bool:
        return output == expected

    holdout_tasks = [
        {"task_id": "t4", "input": {"a": 5, "b": 5}, "expected": {"answer": 10}},
        {"task_id": "t5", "input": {"a": 7, "b": 1}, "expected": {"answer": 8}},
    ]

    write_calls = {"n": 0}
    real_write = memory.write

    def counting_write(*args: Any, **kwargs: Any) -> None:
        write_calls["n"] += 1
        return real_write(*args, **kwargs)

    monkeypatch.setattr(memory, "write", counting_write)

    results = learning_curve._run_holdout_pass(
        recording_run_fn,
        score_exact,
        holdout_tasks,
        domain_id=domain_id,
        memory_mode="on",
        k=5,
        base_dir=base_dir,
    )

    assert write_calls["n"] == 0, "holdout scoring must never call core.memory.write()"
    assert results[0].passed is True

    # The seeded lesson was still injected into the task whose input matched it.
    assert learning_curve._MEMORY_CONTEXT_KEY in received_inputs[0]
    assert received_inputs[0][learning_curve._MEMORY_CONTEXT_KEY][0]["content"] == seeded.content
    # times_retrieved was never bumped, since that would require a write.
    reloaded = memory.get_entry("mem-seeded", domain_id, base_dir=base_dir)
    assert reloaded.times_retrieved == 0


def test_output_jsonl_has_one_record_per_pass_with_documented_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "complete", _fake_reflect_complete())
    output_dir = tmp_path / "out"

    learning_curve.run_learning_curve(
        DOMAIN,
        passes=4,
        memory_mode="on",
        agent_path=GOOD_AGENT,
        domains_root=FIXTURES_DOMAINS_ROOT,
        memory_base_dir=tmp_path / "mem",
        output_dir=output_dir,
    )

    output_path = output_dir / f"{DOMAIN}_memory_on.jsonl"
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 4

    required_fields = {
        "pass",
        "holdout_scorecard",
        "memory_entry_count",
        "avg_entries_retrieved",
        "cost_per_task",
        "p50_latency_ms",
    }
    for i, line in enumerate(lines):
        record = json.loads(line)
        assert required_fields <= record.keys()
        assert record["pass"] == i


def test_memory_off_control_arm_is_deterministic_across_independent_runs(tmp_path):
    records_a = learning_curve.run_learning_curve(
        DOMAIN,
        passes=3,
        memory_mode="off",
        agent_path=GOOD_AGENT,
        domains_root=FIXTURES_DOMAINS_ROOT,
        memory_base_dir=tmp_path / "run_a" / "mem",
        output_dir=tmp_path / "run_a" / "out",
    )
    records_b = learning_curve.run_learning_curve(
        DOMAIN,
        passes=3,
        memory_mode="off",
        agent_path=GOOD_AGENT,
        domains_root=FIXTURES_DOMAINS_ROOT,
        memory_base_dir=tmp_path / "run_b" / "mem",
        output_dir=tmp_path / "run_b" / "out",
    )

    scorecards_a = [r["holdout_scorecard"] for r in records_a]
    scorecards_b = [r["holdout_scorecard"] for r in records_b]
    assert scorecards_a == scorecards_b
    # And flat across passes within a single run, per the control's contract.
    assert all(sc["accuracy"] == scorecards_a[0]["accuracy"] for sc in scorecards_a)


def test_augment_task_input_does_not_mutate_original_or_add_key_when_empty():
    original = {"a": 1}
    result = learning_curve._augment_task_input(original, [])
    assert result is original
    assert learning_curve._MEMORY_CONTEXT_KEY not in result

    entry = MemoryEntry(
        id="mem-1",
        domain_id=DOMAIN,
        scope="rule",
        content="c",
        trigger="t",
        evidence_task_ids=[],
        source_run_id="r",
        created_gen=0,
    )
    augmented = learning_curve._augment_task_input(original, [entry])
    assert augmented is not original
    assert learning_curve._MEMORY_CONTEXT_KEY not in original
    assert augmented[learning_curve._MEMORY_CONTEXT_KEY][0]["content"] == "c"


def test_peek_retrieve_never_calls_memory_write(tmp_path, monkeypatch):
    base_dir = tmp_path / "mem"
    entry = MemoryEntry(
        id="mem-1",
        domain_id=DOMAIN,
        scope="rule",
        content="c",
        trigger="foo",
        evidence_task_ids=[],
        source_run_id="r",
        created_gen=0,
        confidence=0.8,
    )
    memory.write(entry, base_dir=base_dir)

    monkeypatch.setattr(memory, "write", _fail_if_called("memory.write"))

    found = learning_curve._peek_retrieve({"question": "foo bar"}, DOMAIN, 5, base_dir)
    assert len(found) == 1
    assert found[0].id == "mem-1"


def test_cli_main_writes_same_records_it_prints(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(llm, "complete", _fake_reflect_complete())
    output_dir = tmp_path / "out"

    exit_code = learning_curve.main(
        [
            "--domain",
            DOMAIN,
            "--passes",
            "2",
            "--memory",
            "on",
            "--agent",
            GOOD_AGENT,
            "--domains-root",
            FIXTURES_DOMAINS_ROOT,
            "--memory-base-dir",
            str(tmp_path / "mem"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    on_disk = [
        json.loads(line)
        for line in (output_dir / f"{DOMAIN}_memory_on.jsonl").read_text().strip().splitlines()
    ]
    assert printed == on_disk
    assert len(printed) == 2
