"""Tests for core.architect: validating and scaffolding agent directories."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core.architect import REQUIRED_FILES, scaffold, validate

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
