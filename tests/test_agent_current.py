"""Tests for the baseline agent at agents/current/: the graph.yaml step
interpreter in run.py, driven end-to-end with core.llm.complete()
monkeypatched so it's fast, offline, and deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from core import architect, llm, memory
from core.memory import MemoryEntry
from core.runner import _load_agent_run_fn

AGENT_DIR = Path("agents/current")
FIXTURES_DIR = Path("tests/fixtures")


def _recording_fake_complete(
    replies: list[str],
) -> tuple[Callable[..., tuple[str, float, int]], list[list[dict[str, Any]]]]:
    """Like _fake_complete, but also records every call's `messages` list -
    so a test can assert on what context (memory) a later step actually
    received, proving data flows by name rather than by step position."""
    calls: list[list[dict[str, Any]]] = []

    def fake(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
        calls.append(messages)
        return replies[len(calls) - 1], 0.0, 1

    return fake, calls


def _fake_complete(replies: list[str]) -> Callable[..., tuple[str, float, int]]:
    calls = {"n": 0}

    def fake(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
        idx = min(calls["n"], len(replies) - 1)
        calls["n"] += 1
        return replies[idx], 0.0, 1

    return fake


def test_graph_yaml_parses_with_expected_step_shape():
    graph = yaml.safe_load((AGENT_DIR / "graph.yaml").read_text())

    assert isinstance(graph["steps"], list)
    assert len(graph["steps"]) >= 1
    for step in graph["steps"]:
        assert "name" in step
        assert "type" in step
        assert step["type"] in {"llm_call", "tool_call", "verify"}


def test_run_happy_path_returns_final_llm_output(monkeypatch):
    monkeypatch.setattr(llm, "complete", _fake_complete(["draft answer", "def f(): return 1"]))
    run_fn = _load_agent_run_fn(str(AGENT_DIR))

    result = run_fn({"question": "anything"})

    assert result == "def f(): return 1"


def test_run_parses_json_final_output_into_a_dict(monkeypatch):
    monkeypatch.setattr(llm, "complete", _fake_complete(["draft", '{"answer": "42"}']))
    run_fn = _load_agent_run_fn(str(AGENT_DIR))

    result = run_fn({"question": "anything"})

    assert result == {"answer": "42"}


def test_run_retries_solve_step_when_draft_is_empty(monkeypatch):
    fake = _fake_complete(["", "non-empty draft", "final answer"])
    monkeypatch.setattr(llm, "complete", fake)
    run_fn = _load_agent_run_fn(str(AGENT_DIR))

    result = run_fn({"question": "anything"})

    assert result == "final answer"


def test_run_gives_up_retrying_after_max_retries(monkeypatch):
    # "solve" always returns empty; check_draft's max_retries: 1 caps the
    # loop, so execution must still reach "finalize" instead of hanging.
    monkeypatch.setattr(llm, "complete", _fake_complete(["", "", "still finalized"]))
    run_fn = _load_agent_run_fn(str(AGENT_DIR))

    result = run_fn({"question": "anything"})

    assert result == "still finalized"


def test_tool_call_step_invokes_a_real_tool(tmp_path, monkeypatch):
    agent_dir = architect.scaffold(tmp_path / "agent", source=AGENT_DIR)
    (agent_dir / "graph.yaml").write_text(
        "steps:\n"
        "  - name: run_snippet\n"
        "    type: tool_call\n"
        "    tool: run_python\n"
        "    args:\n"
        "      code: \"print(1 + 1)\"\n"
        "    output_key: tool_result\n"
    )
    monkeypatch.setattr(llm, "complete", _fake_complete(["unused"]))
    run_fn = _load_agent_run_fn(str(agent_dir))

    result = run_fn({})

    assert result["returncode"] == 0
    assert result["stdout"].strip() == "2"


def test_unknown_tool_name_is_reported_without_crashing(tmp_path, monkeypatch):
    agent_dir = architect.scaffold(tmp_path / "agent", source=AGENT_DIR)
    (agent_dir / "graph.yaml").write_text(
        "steps:\n  - name: bad_tool\n    type: tool_call\n    tool: does_not_exist\n    output_key: out\n"
    )
    monkeypatch.setattr(llm, "complete", _fake_complete(["unused"]))
    run_fn = _load_agent_run_fn(str(agent_dir))

    result = run_fn({})

    assert "error" in result


def test_unknown_step_type_raises(tmp_path, monkeypatch):
    agent_dir = architect.scaffold(tmp_path / "agent", source=AGENT_DIR)
    (agent_dir / "graph.yaml").write_text("steps:\n  - name: mystery\n    type: teleport\n")
    monkeypatch.setattr(llm, "complete", _fake_complete(["unused"]))
    run_fn = _load_agent_run_fn(str(agent_dir))

    with pytest.raises(ValueError, match="teleport"):
        run_fn({})


def test_tools_module_exposes_working_run_python_and_search_text():
    agent_dir_module = _load_agent_run_fn(str(AGENT_DIR)).__globals__["tools"]

    result = agent_dir_module.TOOLS["run_python"](code="print('hi')")
    assert result["returncode"] == 0
    assert "hi" in result["stdout"]

    matches = agent_dir_module.TOOLS["search_text"](text="alpha\nbeta\ngamma", query="beta")
    assert matches == ["beta"]


def test_two_step_fixture_graph_runs_through_unmodified_run_py(tmp_path, monkeypatch):
    """Proves run.py has no hardcoded step count: a 2-step graph.yaml runs
    correctly through the exact same run.py copied unmodified from
    agents/current/ (see the 5-step counterpart below)."""
    agent_dir = architect.scaffold(tmp_path / "agent", source=AGENT_DIR)
    (agent_dir / "graph.yaml").write_text((FIXTURES_DIR / "graph_2_steps.yaml").read_text())
    fake, calls = _recording_fake_complete(["draft-text", "final-text"])
    monkeypatch.setattr(llm, "complete", fake)
    run_fn = _load_agent_run_fn(str(agent_dir))

    result = run_fn({"question": "anything"})

    assert result == "final-text"
    assert len(calls) == 2
    # step "wrap_up"'s input_keys: [draft] must carry step "only_step"'s
    # output forward, purely by name - proving order/data-flow is driven
    # by the graph, not by any fixed step count or position in run.py.
    wrap_up_prompt = calls[1][-1]["content"]
    assert "draft-text" in wrap_up_prompt


def test_five_step_fixture_graph_runs_through_unmodified_run_py(tmp_path, monkeypatch):
    """Proves run.py has no hardcoded step count/sequence: a longer,
    differently-shaped 5-step graph.yaml mixing all three step types runs
    correctly through the exact same, byte-identical run.py as the 2-step
    fixture above - see test_two_step_fixture_graph_runs_through_unmodified_run_py."""
    agent_dir = architect.scaffold(tmp_path / "agent", source=AGENT_DIR)
    (agent_dir / "graph.yaml").write_text((FIXTURES_DIR / "graph_5_steps.yaml").read_text())
    fake, calls = _recording_fake_complete(["draft-one", "combined-answer", "FINAL-RESULT"])
    monkeypatch.setattr(llm, "complete", fake)
    run_fn = _load_agent_run_fn(str(agent_dir))

    result = run_fn({"question": "anything"})

    assert result == "FINAL-RESULT"
    # Exactly 3 llm_call steps (step_one, step_four, step_five); step_two
    # is a tool_call and step_three is a verify - neither calls the model.
    assert len(calls) == 3

    # step_four's input_keys: [out_one, out_two] must carry both step_one's
    # llm output AND step_two's tool_call output forward by name, proving
    # a tool_call's result flows into a later llm_call just like any other
    # step's output would, regardless of graph shape or step count.
    step_four_prompt = calls[1][-1]["content"]
    assert "draft-one" in step_four_prompt
    assert "hello-from-tool" in step_four_prompt

    # step_five's input_keys: [out_four] must carry step_four's output forward.
    step_five_prompt = calls[2][-1]["content"]
    assert "combined-answer" in step_five_prompt


def test_reflect_step_dedup_lookup_never_bumps_times_retrieved(tmp_path, monkeypatch):
    """The reflect step's `existing` entries are only ever handed to
    core.reflect.reflect() as dedup context - they're never written back
    into ctx.memory, so the agent's own prompt never sees them. That
    lookup must use core.memory.peek(), not retrieve(): using retrieve()
    would durably bump times_retrieved for an entry the agent was never
    actually shown, and would double-count it whenever some other caller
    (e.g. evals/learning_curve.py) also retrieves for the same task."""
    agent_dir = architect.scaffold(tmp_path / "agent", source=AGENT_DIR)
    base_dir = tmp_path / "mem"
    (agent_dir / "graph.yaml").write_text(
        "steps:\n"
        "  - name: solve\n"
        "    type: llm_call\n"
        "    output_key: draft\n"
        "  - name: reflect_step\n"
        "    type: reflect\n"
        "    domain_id: demo\n"
        "    input_key: draft\n"
        f"    base_dir: {base_dir}\n"
        "    k: 5\n"
    )

    entry = MemoryEntry(
        id="mem-1",
        domain_id="demo",
        scope="rule",
        content="a lesson about anything",
        trigger="question anything",
        evidence_task_ids=[],
        source_run_id="seed",
        created_gen=0,
        confidence=0.8,
    )
    memory.write(entry, base_dir=base_dir)

    def fake(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
        system = messages[0]["content"]
        if "reflection step" in system:
            return "[]", 0.0, 1  # core.reflect.reflect(): nothing new to learn
        return "draft text", 0.0, 1

    monkeypatch.setattr(llm, "complete", fake)
    run_fn = _load_agent_run_fn(str(agent_dir))

    run_fn({"question": "anything"})

    # The entry was relevant enough to be looked up (proving the lookup
    # actually ran and matched), but that lookup must never have counted
    # as a real retrieval.
    reloaded = memory.get_entry("mem-1", "demo", base_dir=base_dir)
    assert reloaded.times_retrieved == 0
