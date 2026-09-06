"""Tests for core/reflect.py: turning a completed run into memory lessons.

core.llm.complete() is monkeypatched throughout so these tests are fast,
offline, and deterministic - the same seam test_llm.py and
test_agent_current.py use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import yaml

from core import architect, llm, memory, reflect
from core.memory import MemoryEntry
from core.runner import _load_agent_run_fn
from core.types import RunResult

AGENT_DIR = Path("agents/current")


def _result(**overrides) -> RunResult:
    defaults = dict(
        task_id="task-042",
        output="the final answer",
        passed=False,
        error="ValueError: bad input",
        cost_usd=0.0,
        latency_ms=10,
        trace_id="trace-abc123",
    )
    defaults.update(overrides)
    return RunResult(**defaults)


def _entry(**overrides) -> MemoryEntry:
    defaults = dict(
        id="mem-existing",
        domain_id="coderepair",
        scope="rule",
        content="Always run the linter before submitting a patch.",
        trigger="submitting a patch to the repo",
        evidence_task_ids=["task-1"],
        source_run_id="run-1",
        created_gen=0,
        confidence=0.5,
        times_retrieved=0,
        times_helped=0,
        times_hurt=0,
        status="active",
    )
    defaults.update(overrides)
    return MemoryEntry(**defaults)


def _fake_complete(reply_text: str) -> Callable[..., tuple[str, float, int]]:
    def fake(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
        return reply_text, 0.0, 1

    return fake


def _fake_complete_sequence(replies: list[str]) -> Callable[..., tuple[str, float, int]]:
    calls = {"n": 0}

    def fake(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
        idx = min(calls["n"], len(replies) - 1)
        calls["n"] += 1
        return replies[idx], 0.0, 1

    return fake


def _lesson(**overrides) -> dict[str, Any]:
    defaults = dict(
        content="When a step retries at least once before verification passes, check whether the retried "
        "step actually received the corrected input rather than assuming the model just needs another try.",
        trigger="a step retried before its verify check passed",
        scope="rule",
        supersedes_id=None,
        confidence=0.6,
    )
    defaults.update(overrides)
    return defaults


def test_reflect_default_model_is_the_cheap_reflection_model():
    """reflect() defaults to a cheap model, distinct from the baseline
    agent's main-generation DEFAULT_MODEL in agents/current/run.py -
    reflection writes lessons, it doesn't produce the task answer."""
    assert reflect.DEFAULT_MODEL == "claude-haiku-4-5"


def test_reflect_returns_empty_list_when_model_finds_nothing_new(monkeypatch):
    monkeypatch.setattr(llm, "complete", _fake_complete("[]"))

    entries = reflect.reflect(
        _result(passed=True, error=None),
        {"question": "anything"},
        expected="anything",
        trace=None,
        existing_memory=[],
        domain_id="docqa",
    )

    assert entries == []


def test_reflect_returns_empty_list_on_malformed_model_output(monkeypatch):
    monkeypatch.setattr(llm, "complete", _fake_complete("not json at all"))

    entries = reflect.reflect(
        _result(), {"question": "anything"}, expected=None, trace=None, existing_memory=[], domain_id="docqa"
    )

    assert entries == []


def test_reflect_produces_a_general_lesson_from_a_failure(monkeypatch):
    monkeypatch.setattr(llm, "complete", _fake_complete(json.dumps([_lesson()])))

    entries = reflect.reflect(
        _result(task_id="task-042"),
        {"question": "anything"},
        expected="42",
        trace=[{"name": "solve", "error": "ValueError: bad input"}],
        existing_memory=[],
        domain_id="docqa",
        gen_n=3,
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry.domain_id == "docqa"
    assert entry.scope == "rule"
    assert entry.created_gen == 3
    assert entry.status == "active"
    assert entry.evidence_task_ids == ["task-042"]
    assert "task-042" not in entry.content


def test_reflect_reflects_on_successes_too(monkeypatch):
    """A passing run isn't exempt from producing a lesson - a non-obvious
    choice that worked is exactly as valid a thing to remember."""
    monkeypatch.setattr(
        llm,
        "complete",
        _fake_complete(json.dumps([_lesson(content="Trying the simpler regex-based parser first, before "
                                            "falling back to a full parser, resolved this class of input "
                                            "faster without losing correctness.")])),
    )

    entries = reflect.reflect(
        _result(passed=True, error=None),
        {"question": "anything"},
        expected="anything",
        trace=[{"name": "solve", "error": None}],
        existing_memory=[],
        domain_id="docqa",
    )

    assert len(entries) == 1


def test_reflect_rejects_candidate_naming_the_run_result_task_id(monkeypatch):
    bad_lesson = _lesson(
        content="task-042's invoice total is $530.10.",
        trigger="task-042",
    )
    monkeypatch.setattr(llm, "complete", _fake_complete(json.dumps([bad_lesson])))

    entries = reflect.reflect(
        _result(task_id="task-042"),
        {"invoice_text": "..."},
        expected="$530.10",
        trace=None,
        existing_memory=[],
        domain_id="invoices",
    )

    assert entries == []


def test_reflect_rejects_candidate_matching_a_generic_task_reference(monkeypatch):
    """Even when the real task_id is empty (e.g. the in-run reflect step,
    which never sees a dataset row's task_id - see agents/current/README.md),
    a model inventing its own task-reference-shaped text is still caught."""
    bad_lesson = _lesson(
        content="For task 17, the invoice total should always be doubled before reporting it.",
        trigger="task 17",
    )
    monkeypatch.setattr(llm, "complete", _fake_complete(json.dumps([bad_lesson])))

    entries = reflect.reflect(
        _result(task_id=""),
        {"invoice_text": "..."},
        expected=None,
        trace=None,
        existing_memory=[],
        domain_id="invoices",
    )

    assert entries == []


def test_reflect_rejects_candidate_restating_expected_verbatim(monkeypatch):
    bad_lesson = _lesson(
        content="The correct final answer for this kind of question is exactly 'the quick brown fox jumps'.",
        trigger="questions like this one",
    )
    monkeypatch.setattr(llm, "complete", _fake_complete(json.dumps([bad_lesson])))

    entries = reflect.reflect(
        _result(),
        {"question": "anything"},
        expected="the quick brown fox jumps",
        trace=None,
        existing_memory=[],
        domain_id="docqa",
    )

    assert entries == []


def test_reflect_returns_empty_list_when_lesson_already_known(monkeypatch):
    """A candidate that just restates an existing entry must not produce a
    near-duplicate - see core/reflect.py's dedup section."""
    existing = _entry(
        id="mem-existing",
        content="Always run the linter before submitting a patch.",
        trigger="submitting a patch to the repo",
    )
    restated = _lesson(
        content="Always run the linter before submitting any patch.",
        trigger="submitting a patch to the repo",
    )
    monkeypatch.setattr(llm, "complete", _fake_complete(json.dumps([restated])))

    entries = reflect.reflect(
        _result(),
        {"diff": "..."},
        expected=None,
        trace=None,
        existing_memory=[existing],
        domain_id="coderepair",
    )

    assert entries == []


def test_reflect_two_runs_with_same_underlying_cause_collapse_not_duplicate(monkeypatch):
    """Reflecting on run A produces one general rule. Reflecting on run B,
    whose failure shares the same underlying cause, must not add a second
    near-duplicate entry once existing_memory already contains run A's
    lesson - it should return an update/no-op instead."""
    lesson_json = json.dumps([_lesson(
        content="A KeyError on the 'total' field means the parser assumed a fixed schema; check for "
        "alternate field names before failing.",
        trigger="a KeyError on a field the parser assumes is always present",
    )])
    monkeypatch.setattr(llm, "complete", _fake_complete(lesson_json))

    run_a_entries = reflect.reflect(
        _result(task_id="task-A", error="KeyError: 'total'"),
        {"invoice_text": "doc A"},
        expected=None,
        trace=None,
        existing_memory=[],
        domain_id="invoices",
    )
    assert len(run_a_entries) == 1

    run_b_entries = reflect.reflect(
        _result(task_id="task-B", error="KeyError: 'total'"),
        {"invoice_text": "doc B"},
        expected=None,
        trace=None,
        existing_memory=run_a_entries,
        domain_id="invoices",
    )

    assert run_b_entries == []


def test_reflect_honors_explicit_supersedes_id_even_with_low_text_similarity(monkeypatch):
    """The model can explicitly flag a contradiction via supersedes_id;
    this must go through as an update even when the wording is quite
    different from the old entry (retiring a stale rule is desirable)."""
    stale = _entry(
        id="mem-stale",
        content="Never retry the solve step; retries always waste budget for no benefit.",
        trigger="the solve step's output looks wrong",
        evidence_task_ids=["task-old"],
        confidence=0.4,
    )
    contradiction = _lesson(
        content="Retrying the solve step once, with the verifier's failure reason appended to the prompt, "
        "fixes most malformed-output cases without materially increasing cost.",
        trigger="verification fails on the first attempt",
        supersedes_id="mem-stale",
        confidence=0.7,
    )
    monkeypatch.setattr(llm, "complete", _fake_complete(json.dumps([contradiction])))

    entries = reflect.reflect(
        _result(task_id="task-new"),
        {"question": "anything"},
        expected=None,
        trace=None,
        existing_memory=[stale],
        domain_id="docqa",
    )

    assert len(entries) == 1
    updated = entries[0]
    assert updated.id == "mem-stale"
    assert updated.content == contradiction["content"]
    assert updated.confidence == 0.7
    assert set(updated.evidence_task_ids) == {"task-old", "task-new"}


def test_reflect_new_entries_get_a_deterministic_id(monkeypatch):
    """Two independent reflect() calls landing on the identical lesson
    (e.g. existing_memory wasn't threaded through correctly) get the same
    id, so core.memory.write() folds them into one entry instead of
    accumulating copies - a second, independent safety net against
    duplicates beyond the similarity check."""
    monkeypatch.setattr(llm, "complete", _fake_complete(json.dumps([_lesson()])))

    first = reflect.reflect(
        _result(task_id="task-1"), {"question": "a"}, expected=None, trace=None, existing_memory=[], domain_id="docqa"
    )
    second = reflect.reflect(
        _result(task_id="task-2"), {"question": "b"}, expected=None, trace=None, existing_memory=[], domain_id="docqa"
    )

    assert len(first) == len(second) == 1
    assert first[0].id == second[0].id


def test_reflect_new_entry_records_evidence_and_source_run(monkeypatch):
    monkeypatch.setattr(llm, "complete", _fake_complete(json.dumps([_lesson()])))

    entries = reflect.reflect(
        _result(task_id="task-99", trace_id="trace-xyz"),
        {"question": "anything"},
        expected=None,
        trace=None,
        existing_memory=[],
        domain_id="docqa",
    )

    assert entries[0].evidence_task_ids == ["task-99"]
    assert entries[0].source_run_id == "trace-xyz"


def test_reflect_ignores_malformed_candidate_entries(monkeypatch):
    """A candidate missing required fields is dropped rather than raising."""
    monkeypatch.setattr(llm, "complete", _fake_complete(json.dumps([{"content": ""}, {"not": "a lesson"}])))

    entries = reflect.reflect(
        _result(), {"question": "anything"}, expected=None, trace=None, existing_memory=[], domain_id="docqa"
    )

    assert entries == []


def test_reflect_defaults_unknown_scope_to_rule(monkeypatch):
    monkeypatch.setattr(llm, "complete", _fake_complete(json.dumps([_lesson(scope="not-a-real-scope")])))

    entries = reflect.reflect(
        _result(), {"question": "anything"}, expected=None, trace=None, existing_memory=[], domain_id="docqa"
    )

    assert entries[0].scope == "rule"


# --- agents/current/run.py's "reflect" step type -----------------------------


def test_reflect_step_writes_entries_via_core_memory(tmp_path, monkeypatch):
    agent_dir = architect.scaffold(tmp_path / "agent", source=AGENT_DIR)
    base_dir = tmp_path / "memstore"
    graph = {
        "steps": [
            {"name": "solve", "type": "llm_call", "output_key": "draft"},
            {
                "name": "learn",
                "type": "reflect",
                "domain_id": "testdomain",
                "input_key": "draft",
                "base_dir": str(base_dir),
            },
        ]
    }
    (agent_dir / "graph.yaml").write_text(yaml.dump(graph, sort_keys=False))

    lesson_json = json.dumps([_lesson()])
    monkeypatch.setattr(llm, "complete", _fake_complete_sequence(["a draft answer", lesson_json]))
    run_fn = _load_agent_run_fn(str(agent_dir))

    run_fn({"question": "anything"})

    written = memory.load_all("testdomain", base_dir=base_dir)
    assert len(written) == 1
    assert written[0].content == _lesson()["content"]


def test_reflect_step_without_domain_id_is_a_no_op(tmp_path, monkeypatch):
    agent_dir = architect.scaffold(tmp_path / "agent", source=AGENT_DIR)
    graph = {
        "steps": [
            {"name": "solve", "type": "llm_call", "output_key": "draft"},
            {"name": "learn", "type": "reflect"},
        ]
    }
    (agent_dir / "graph.yaml").write_text(yaml.dump(graph, sort_keys=False))
    monkeypatch.setattr(llm, "complete", _fake_complete_sequence(["a draft answer"]))
    run_fn = _load_agent_run_fn(str(agent_dir))

    result = run_fn({"question": "anything"})

    assert result == "a draft answer"


def test_reflect_step_never_crashes_the_run_on_a_bad_model_reply(tmp_path, monkeypatch):
    agent_dir = architect.scaffold(tmp_path / "agent", source=AGENT_DIR)
    base_dir = tmp_path / "memstore"
    graph = {
        "steps": [
            {"name": "solve", "type": "llm_call", "output_key": "draft"},
            {
                "name": "learn",
                "type": "reflect",
                "domain_id": "testdomain",
                "input_key": "draft",
                "base_dir": str(base_dir),
            },
        ]
    }
    (agent_dir / "graph.yaml").write_text(yaml.dump(graph, sort_keys=False))
    monkeypatch.setattr(llm, "complete", _fake_complete_sequence(["a draft answer", "not json"]))
    run_fn = _load_agent_run_fn(str(agent_dir))

    result = run_fn({"question": "anything"})

    # No exception propagates, and no entries were written since the
    # model's reply couldn't be parsed as a lesson.
    assert result == []
    assert memory.load_all("testdomain", base_dir=base_dir) == []
