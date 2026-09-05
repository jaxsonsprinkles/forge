"""Tests for core.architect: validating, scaffolding, and building agent directories."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from core import architect, llm
from core.architect import REQUIRED_FILES, build_agent, scaffold, validate
from core.runner import run_agent
from core.scorer import score_runs
from core.types import TaskSpec

REAL_AGENT_DIR = Path("agents/current")


def _copy_real_agent(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name in (*REQUIRED_FILES, "README.md"):
        shutil.copy2(REAL_AGENT_DIR / name, dest / name)
    return dest


def test_real_agents_current_is_valid():
    result = validate(REAL_AGENT_DIR)
    assert result.ok is True, result.errors
    assert result.errors == []


def test_missing_required_file_is_flagged(tmp_path):
    agent_dir = _copy_real_agent(tmp_path / "agent")
    (agent_dir / "tools.py").unlink()

    result = validate(agent_dir)

    assert result.ok is False
    assert any("tools.py" in e for e in result.errors)


def test_missing_directory_is_flagged(tmp_path):
    result = validate(tmp_path / "does-not-exist")

    assert result.ok is False
    assert result.errors


def test_unexpected_extra_file_is_flagged(tmp_path):
    agent_dir = _copy_real_agent(tmp_path / "agent")
    (agent_dir / "scratch.txt").write_text("not part of the contract")

    result = validate(agent_dir)

    assert result.ok is False
    assert any("scratch.txt" in e for e in result.errors)


def test_malformed_graph_yaml_is_flagged(tmp_path):
    agent_dir = _copy_real_agent(tmp_path / "agent")
    (agent_dir / "graph.yaml").write_text("not: valid: yaml: [")

    result = validate(agent_dir)

    assert result.ok is False
    assert any("graph.yaml" in e for e in result.errors)


def test_graph_yaml_missing_steps_is_flagged(tmp_path):
    agent_dir = _copy_real_agent(tmp_path / "agent")
    (agent_dir / "graph.yaml").write_text("not_steps: []\n")

    result = validate(agent_dir)

    assert result.ok is False
    assert any("steps" in e for e in result.errors)


def test_graph_yaml_step_missing_name_or_type_is_flagged(tmp_path):
    agent_dir = _copy_real_agent(tmp_path / "agent")
    (agent_dir / "graph.yaml").write_text("steps:\n  - name: only_a_name\n")

    result = validate(agent_dir)

    assert result.ok is False
    assert any("step 0" in e for e in result.errors)


def test_run_py_without_run_function_is_flagged(tmp_path):
    agent_dir = _copy_real_agent(tmp_path / "agent")
    (agent_dir / "run.py").write_text("def not_run(task_input):\n    return {}\n")

    result = validate(agent_dir)

    assert result.ok is False
    assert any("run(task_input)" in e for e in result.errors)


def test_run_py_that_fails_to_import_is_flagged(tmp_path):
    agent_dir = _copy_real_agent(tmp_path / "agent")
    (agent_dir / "run.py").write_text("this is not valid python(((\n")

    result = validate(agent_dir)

    assert result.ok is False
    assert any("run.py failed to import" in e for e in result.errors)


def test_scaffold_copies_a_valid_agent(tmp_path):
    dest = scaffold(tmp_path / "scaffolded", source=REAL_AGENT_DIR)

    for name in (*REQUIRED_FILES, "README.md"):
        assert (dest / name).is_file()

    result = validate(dest)
    assert result.ok is True, result.errors


def test_scaffold_rejects_an_invalid_source(tmp_path):
    bad_source = tmp_path / "bad_source"
    bad_source.mkdir()
    (bad_source / "prompt.md").write_text("only one of the five files")

    with pytest.raises(FileNotFoundError):
        scaffold(tmp_path / "dest", source=bad_source)


# --- build_agent() -----------------------------------------------------

_FAKE_BUILD_RESPONSE = json.dumps(
    {
        "prompt_md": "You are a careful assistant that solves one task at a time.",
        "graph_steps": [
            {
                "name": "solve",
                "type": "llm_call",
                "instruction": "Produce your best answer to the task input above.",
                "output_key": "final",
            }
        ],
    }
)


def _fake_build_complete(text: str) -> Callable[..., tuple[str, float, int]]:
    def fake(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
        return text, 0.0, 1

    return fake


def _dummy_task_spec(**overrides: Any) -> TaskSpec:
    defaults: dict[str, Any] = {
        "domain_id": "dummy",
        "goal": "Answer the question in task_input.",
        "tools": [],
        "dataset_path": "unused.jsonl",
        "scorer_id": "unused:unused",
    }
    defaults.update(overrides)
    return TaskSpec(**defaults)


def test_build_agent_writes_fixed_scaffolding_plus_generated_files(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "complete", _fake_build_complete(_FAKE_BUILD_RESPONSE))

    agent_dir = build_agent(_dummy_task_spec(), tmp_path / "built")

    result = validate(agent_dir)
    assert result.ok is True, result.errors
    for name in REQUIRED_FILES:
        assert (agent_dir / name).is_file()

    # run.py/memory.py/tools.py are fixed scaffolding: copied byte-for-byte
    # from the baseline agent, never touched by the model call.
    for name in ("run.py", "memory.py", "tools.py"):
        assert (agent_dir / name).read_text() == (REAL_AGENT_DIR / name).read_text()

    # prompt.md and graph.yaml reflect the (mocked) model's response.
    assert "careful assistant" in (agent_dir / "prompt.md").read_text()
    graph = yaml.safe_load((agent_dir / "graph.yaml").read_text())
    assert graph["steps"] == [
        {
            "name": "solve",
            "type": "llm_call",
            "instruction": "Produce your best answer to the task input above.",
            "output_key": "final",
        }
    ]


def test_build_agent_tolerates_a_markdown_code_fence(tmp_path, monkeypatch):
    fenced = f"```json\n{_FAKE_BUILD_RESPONSE}\n```"
    monkeypatch.setattr(llm, "complete", _fake_build_complete(fenced))

    agent_dir = build_agent(_dummy_task_spec(), tmp_path / "built")

    assert validate(agent_dir).ok is True


def test_build_agent_rejects_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "complete", _fake_build_complete("not json at all"))

    with pytest.raises(ValueError, match="not valid JSON"):
        build_agent(_dummy_task_spec(), tmp_path / "built")


def test_build_agent_rejects_missing_prompt_md(tmp_path, monkeypatch):
    bad_response = json.dumps({"graph_steps": [{"name": "solve", "type": "llm_call"}]})
    monkeypatch.setattr(llm, "complete", _fake_build_complete(bad_response))

    with pytest.raises(ValueError, match="prompt_md"):
        build_agent(_dummy_task_spec(), tmp_path / "built")


def test_build_agent_rejects_empty_graph_steps(tmp_path, monkeypatch):
    bad_response = json.dumps({"prompt_md": "hello", "graph_steps": []})
    monkeypatch.setattr(llm, "complete", _fake_build_complete(bad_response))

    with pytest.raises(ValueError, match="graph_steps"):
        build_agent(_dummy_task_spec(), tmp_path / "built")


def test_build_agent_rejects_a_step_missing_name_or_type(tmp_path, monkeypatch):
    bad_response = json.dumps({"prompt_md": "hello", "graph_steps": [{"name": "solve"}]})
    monkeypatch.setattr(llm, "complete", _fake_build_complete(bad_response))

    with pytest.raises(ValueError, match="malformed graph_steps"):
        build_agent(_dummy_task_spec(), tmp_path / "built")


def test_build_agent_makes_exactly_one_model_call(tmp_path, monkeypatch):
    call_count = {"n": 0}

    def counting_fake(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
        call_count["n"] += 1
        return _FAKE_BUILD_RESPONSE, 0.0, 1

    monkeypatch.setattr(llm, "complete", counting_fake)

    build_agent(_dummy_task_spec(), tmp_path / "built")

    assert call_count["n"] == 1


def test_build_agent_on_coderepair_scores_nonzero_but_imperfect(tmp_path, monkeypatch):
    """Integration test: build_agent() on the real coderepair task_spec,
    then run the built agent through core.runner.run_agent against
    coderepair's real train split. Both the build call and every per-task
    run call go through a mocked core.llm.complete() so this stays fast,
    offline, and deterministic while still exercising build_agent() end to
    end against a real domain and real scorer.
    """
    task_spec = TaskSpec(
        domain_id="coderepair",
        goal="Fix a buggy Python function so it passes its hidden tests.",
        tools=["run_python"],
        dataset_path="domains/coderepair/dataset.jsonl",
        scorer_id="domains.coderepair.scorer:score_v1",
        max_tasks=20,
    )
    build_response = json.dumps(
        {
            "prompt_md": "You are a careful Python engineer who fixes one buggy function at a time.",
            "graph_steps": [
                {
                    "name": "solve",
                    "type": "llm_call",
                    "instruction": (
                        "Fix the bug in broken_code so it satisfies its hidden tests. Respond with "
                        "ONLY the corrected Python function source, no commentary."
                    ),
                    "output_key": "final",
                }
            ],
        }
    )
    monkeypatch.setattr(llm, "complete", _fake_build_complete(build_response))
    agent_dir = build_agent(task_spec, tmp_path / "built_coderepair")
    assert validate(agent_dir).ok is True

    # A deliberately weak, deterministic "solver": correctly fixes a known
    # subset of the train split's functions and echoes the (still-broken)
    # original source back for the rest - guaranteeing a score strictly
    # between 0 and 1, the way a real weak first-draft agent would score.
    correct_fixes = {
        "is_even": "def is_even(n):\n    return n % 2 == 0\n",
        "gcd": "def gcd(a, b):\n    while b != 0:\n        a, b = b, a % b\n    return a\n",
        "capitalize_words": (
            'def capitalize_words(s):\n    return " ".join(word[0].upper() + word[1:] for word in s.split(" "))\n'
        ),
        "is_leap_year": (
            "def is_leap_year(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n"
        ),
        "parse_int": (
            'def parse_int(s):\n    if s.startswith("-"):\n        return -int(s[1:])\n    return int(s)\n'
        ),
        "clamp": (
            "def clamp(value, low, high):\n"
            "    if value < low:\n"
            "        return low\n"
            "    if value > high:\n"
            "        return high\n"
            "    return value\n"
        ),
    }

    def fake_solve(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
        user_content = messages[-1]["content"]
        json_blob = user_content[len("Task input:\n") :].split("\n\n", 1)[0]
        task_input = json.loads(json_blob)
        fixed = correct_fixes.get(task_input["function_name"])
        return fixed if fixed is not None else task_input["broken_code"], 0.0, 1

    monkeypatch.setattr(llm, "complete", fake_solve)

    results = run_agent(str(agent_dir), task_spec, split="train")
    score = score_runs(results, "train")

    assert score.n == 20
    assert 0.0 < score.accuracy < 1.0
