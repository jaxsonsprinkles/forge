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

from core import architect, llm, memory, reflect
from core.memory import MemoryEntry
from core.runner import _load_agent_run_fn
from evals import learning_curve

FIXTURES_DOMAINS_ROOT = str(Path(__file__).parent / "fixtures" / "domains")
GOOD_AGENT = str(Path(__file__).parent / "fixtures" / "agents" / "good_agent")
BASELINE_AGENT = str(Path(__file__).parent.parent / "agents" / "current")
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


def _recording_fake_complete(
    replies: list[str],
) -> tuple[Any, list[list[dict[str, Any]]]]:
    """Like the fixtures above, but also records every call's `messages`
    list - so a test can assert on what context (e.g. recalled lessons) a
    step actually received."""
    calls: list[list[dict[str, Any]]] = []

    def fake(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
        calls.append(messages)
        return replies[len(calls) - 1], 0.0, 1

    return fake, calls


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


def _recall_then_solve_graph(base_dir: Path, domain_id: str) -> str:
    return (
        "steps:\n"
        "  - name: recall_step\n"
        "    type: recall\n"
        f"    domain_id: {domain_id}\n"
        "    output_key: lessons\n"
        "  - name: solve\n"
        "    type: llm_call\n"
        "    input_keys: [lessons]\n"
        "    output_key: draft\n"
    )


def test_holdout_never_writes_memory_even_when_it_retrieves(tmp_path, monkeypatch):
    """Holdout is read-only: a native recall step must still retrieve
    accumulated memory (so the curve reflects what's been learned - and
    the lesson text must actually reach the model's prompt) but never
    call memory.write(), not even via retrieve()'s own times_retrieved
    side effect. Recall now lives inside run() (see agents/current/
    run.py's _handle_recall), reached only via the FORGE_MEMORY_* env
    vars this harness sets - not via task_input - so this test drives it
    through the real _run_holdout_pass entrypoint instead of a hand-rolled
    run_fn."""
    base_dir = tmp_path / "mem"
    domain_id = DOMAIN

    agent_dir = architect.scaffold(tmp_path / "agent", source=BASELINE_AGENT)
    (agent_dir / "graph.yaml").write_text(_recall_then_solve_graph(base_dir, domain_id))

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

    fake, calls = _recording_fake_complete(["draft text"])
    monkeypatch.setattr(llm, "complete", fake)
    run_fn = _load_agent_run_fn(str(agent_dir))

    write_calls = {"n": 0}
    real_write = memory.write

    def counting_write(*args: Any, **kwargs: Any) -> None:
        write_calls["n"] += 1
        return real_write(*args, **kwargs)

    monkeypatch.setattr(memory, "write", counting_write)

    holdout_tasks = [
        {"task_id": "t4", "input": {"a": 5, "b": 5}, "expected": {"answer": 10}},
    ]

    results = learning_curve._run_holdout_pass(
        run_fn,
        lambda output, expected: output == expected,
        holdout_tasks,
        domain_id=domain_id,
        memory_mode="on",
        k=5,
        base_dir=base_dir,
    )

    assert write_calls["n"] == 0, "holdout scoring must never call core.memory.write()"
    assert results[0].error is None

    # The seeded lesson still reached the model's own prompt.
    user_message = calls[0][-1]["content"]
    assert seeded.content in user_message
    # times_retrieved was never bumped, since that would require a write.
    reloaded = memory.get_entry("mem-seeded", domain_id, base_dir=base_dir)
    assert reloaded.times_retrieved == 0


def test_native_recall_avoids_double_counting_times_retrieved(tmp_path, monkeypatch):
    """Regression test for the double-retrieval bug class (TASK 11/PR #24
    fixed one instance of it for the `reflect` step's dedup lookup; the
    CRITICAL CONFLICT this milestone flagged was a second instance: this
    harness's own retrieve() call for injection, alongside a native
    `recall` step's own retrieve() call, would double-bump
    `times_retrieved` for one real exposure.

    This harness no longer retrieves/injects itself at all (see module
    docstring) - retrieval happens exactly once, inside the run, via the
    agent's own `recall` step. So a lesson retrieved for N train tasks
    must end up with `times_retrieved == N`, never `2N`, by construction:
    there is only one call site left that can bump it.
    """
    base_dir = tmp_path / "mem"
    domain_id = DOMAIN

    agent_dir = architect.scaffold(tmp_path / "agent", source=BASELINE_AGENT)
    (agent_dir / "graph.yaml").write_text(_recall_then_solve_graph(base_dir, domain_id))

    seeded = MemoryEntry(
        id="mem-seeded",
        domain_id=domain_id,
        scope="rule",
        content="always add both operands directly",
        trigger="a b 1 2 3",
        evidence_task_ids=[],
        source_run_id="seed",
        created_gen=0,
        confidence=0.9,
    )
    memory.write(seeded, base_dir=base_dir)

    def fake_complete(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
        system = messages[0]["content"]
        if "reflection step" in system:
            return "[]", 0.0, 1  # core.reflect.reflect(): nothing new to learn
        return "some draft answer", 0.0, 1

    monkeypatch.setattr(llm, "complete", fake_complete)

    learning_curve.run_learning_curve(
        domain_id,
        passes=1,
        memory_mode="on",
        agent_path=str(agent_dir),
        domains_root=FIXTURES_DOMAINS_ROOT,
        memory_base_dir=base_dir,
        reset_memory=False,
        output_dir=tmp_path / "out",
    )

    # 3 train tasks (t1-t3) each bump times_retrieved exactly once, via
    # the graph's own recall step. Holdout (t4, t5) never bumps at all:
    # this harness sets FORGE_MEMORY_READ_ONLY for holdout, which makes
    # the same recall step use peek() instead.
    reloaded = memory.get_entry("mem-seeded", domain_id, base_dir=base_dir)
    assert reloaded.times_retrieved == 3


def test_reinforce_called_once_per_retrieved_entry_per_train_task(tmp_path, monkeypatch):
    """Acceptance: core.memory.reinforce() must be called exactly once per
    (train task, retrieved entry) pair, with the real pass/fail outcome -
    not the reflect step's local non-empty-output proxy."""
    base_dir = tmp_path / "mem"
    domain_id = DOMAIN

    agent_dir = architect.scaffold(tmp_path / "agent", source=BASELINE_AGENT)
    (agent_dir / "graph.yaml").write_text(_recall_then_solve_graph(base_dir, domain_id))

    seeded = MemoryEntry(
        id="mem-seeded",
        domain_id=domain_id,
        scope="rule",
        content="always add both operands directly",
        trigger="a b 1 2 3",
        evidence_task_ids=[],
        source_run_id="seed",
        created_gen=0,
        confidence=0.9,
    )
    memory.write(seeded, base_dir=base_dir)

    def fake_complete(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
        system = messages[0]["content"]
        if "reflection step" in system:
            return "[]", 0.0, 1
        return "some draft answer", 0.0, 1  # never matches the dict `expected` -> always fails scoring

    monkeypatch.setattr(llm, "complete", fake_complete)

    reinforce_calls: list[tuple[str, str, bool]] = []
    real_reinforce = memory.reinforce

    def recording_reinforce(entry_id: str, domain: str, helped: bool, base_dir: Any = memory.DEFAULT_MEMORY_ROOT) -> None:
        reinforce_calls.append((entry_id, domain, helped))
        real_reinforce(entry_id, domain, helped, base_dir=base_dir)

    monkeypatch.setattr(memory, "reinforce", recording_reinforce)

    learning_curve.run_learning_curve(
        domain_id,
        passes=1,
        memory_mode="on",
        agent_path=str(agent_dir),
        domains_root=FIXTURES_DOMAINS_ROOT,
        memory_base_dir=base_dir,
        reset_memory=False,
        output_dir=tmp_path / "out",
    )

    # 3 train tasks (t1-t3), each retrieving mem-seeded exactly once ->
    # exactly 3 reinforce() calls, none from holdout (t4, t5 never reinforce).
    assert reinforce_calls == [("mem-seeded", domain_id, False)] * 3


def test_seed_memory_prepopulates_store_before_pass_0_without_touching_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "complete", _fake_reflect_complete())

    records = learning_curve.run_learning_curve(
        DOMAIN,
        passes=1,
        memory_mode="on",
        agent_path=GOOD_AGENT,
        domains_root=FIXTURES_DOMAINS_ROOT,
        memory_base_dir=tmp_path / "mem",
        output_dir=tmp_path / "out",
        seed_memory=2,
    )

    # Seeding ran 2 extra train tasks (with reflection) before pass 0's own
    # 3 train tasks, so pass 0 already sees the seeded lessons - more
    # entries than a single pass of 3 train tasks alone could produce.
    assert records[0]["memory_entry_count"] >= 2

    baseline = learning_curve.run_learning_curve(
        DOMAIN,
        passes=1,
        memory_mode="on",
        agent_path=GOOD_AGENT,
        domains_root=FIXTURES_DOMAINS_ROOT,
        memory_base_dir=tmp_path / "mem_unseeded",
        output_dir=tmp_path / "out_unseeded",
        seed_memory=0,
    )
    assert records[0]["memory_entry_count"] > baseline[0]["memory_entry_count"]


def test_seed_memory_is_a_noop_when_memory_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "write", _fail_if_called("memory.write"))
    monkeypatch.setattr(reflect, "reflect", _fail_if_called("reflect.reflect"))

    records = learning_curve.run_learning_curve(
        DOMAIN,
        passes=1,
        memory_mode="off",
        agent_path=GOOD_AGENT,
        domains_root=FIXTURES_DOMAINS_ROOT,
        memory_base_dir=tmp_path / "mem",
        output_dir=tmp_path / "out",
        seed_memory=4,
    )

    assert records[0]["memory_entry_count"] == 0


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
