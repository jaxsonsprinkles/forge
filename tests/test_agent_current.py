"""Tests for the baseline agent at agents/current/: the graph.yaml step
interpreter in run.py, driven end-to-end with core.llm.complete()
monkeypatched so it's fast, offline, and deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from core import architect, llm
from core.runner import _load_agent_run_fn

AGENT_DIR = Path("agents/current")


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
