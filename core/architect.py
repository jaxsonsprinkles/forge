"""Validates and scaffolds an agents/current/-shaped agent directory.

See AGENTS.md: the agent is always exactly five plain files - prompt.md,
tools.py, memory.py, graph.yaml, run.py - never a class, package, or
config object. This module is the one place that knows that fixed
five-file contract well enough to (a) check a directory actually
satisfies it, before core/runner.py tries to load it, and (b) scaffold a
fresh copy of the current baseline into a new directory - e.g. so a
mutation's working copy, or an archived generation under
agents/archive/<id>/, starts from a known-good copy rather than an
agent-shaped directory hand-assembled from scratch.

Neither core/gate.py nor core/executor_ao.py import this module today:
executor_ao dispatches a plain text instruction to an AO worker that
edits an existing checkout in place, and gate only persists ScoreCards
and a branch ref (not agent files) into agents/archive/<id>/ (see their
docstrings). So the minimal real interface needed right now is
validation (any candidate agent directory should be checkable before
it's trusted) plus scaffolding (something has to produce the first copy
of those five files) - nothing more speculative than that.
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED_FILES: tuple[str, ...] = ("prompt.md", "tools.py", "memory.py", "graph.yaml", "run.py")

# Not a mutation surface, so allowed alongside the five required files
# without tripping the "exactly five files" check.
_ALSO_ALLOWED = {"README.md", "__pycache__"}

BASELINE_AGENT_DIR = Path(__file__).resolve().parent.parent / "agents" / "current"


@dataclass
class ValidationResult:
    """The outcome of validating one agents/current/-shaped directory."""

    ok: bool
    errors: list[str]


def _check_required_files(agent_dir: Path) -> list[str]:
    return [f"missing required file: {name}" for name in REQUIRED_FILES if not (agent_dir / name).is_file()]


def _check_extra_files(agent_dir: Path) -> list[str]:
    """Flag anything at the top level of agent_dir beyond the five
    required files - AGENTS.md requires the agent be EXACTLY these five
    files, never extras that fall outside the four mutation surfaces."""
    allowed = set(REQUIRED_FILES) | _ALSO_ALLOWED
    return [f"unexpected file/dir in agent directory: {p.name}" for p in agent_dir.iterdir() if p.name not in allowed]


def _check_graph_yaml(agent_dir: Path) -> list[str]:
    graph_path = agent_dir / "graph.yaml"
    if not graph_path.is_file():
        return []
    try:
        graph = yaml.safe_load(graph_path.read_text())
    except yaml.YAMLError as exc:
        return [f"graph.yaml is not valid YAML: {exc}"]

    if not isinstance(graph, dict) or not isinstance(graph.get("steps"), list) or not graph["steps"]:
        return ["graph.yaml must define a non-empty top-level 'steps' list"]

    return [
        f"graph.yaml step {i} must be a mapping with 'name' and 'type'"
        for i, step in enumerate(graph["steps"])
        if not isinstance(step, dict) or "name" not in step or "type" not in step
    ]


def _check_run_py(agent_dir: Path) -> list[str]:
    run_path = agent_dir / "run.py"
    if not run_path.is_file():
        return []
    try:
        spec = importlib.util.spec_from_file_location("forge_architect_validate_run", run_path)
        if spec is None or spec.loader is None:
            return ["run.py could not be loaded as a module"]
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - any load-time failure is a validation error, not a crash
        return [f"run.py failed to import: {type(exc).__name__}: {exc}"]

    if not callable(getattr(module, "run", None)):
        return ["run.py has no callable run(task_input) entrypoint"]
    return []


def validate(agent_path: str | Path) -> ValidationResult:
    """Check that `agent_path` is a well-formed agents/current/-shaped directory.

    Checks: exactly the five required files are present (plus, optionally,
    README.md), graph.yaml parses as YAML with a non-empty 'steps' list of
    {name, type} mappings, and run.py imports cleanly and exposes a
    callable `run(task_input)`. Never raises - every failure becomes an
    entry in the returned ValidationResult.errors instead.
    """
    agent_dir = Path(agent_path)
    if not agent_dir.is_dir():
        return ValidationResult(ok=False, errors=[f"{agent_dir} is not a directory"])

    errors = [
        *_check_required_files(agent_dir),
        *_check_extra_files(agent_dir),
        *_check_graph_yaml(agent_dir),
        *_check_run_py(agent_dir),
    ]
    return ValidationResult(ok=not errors, errors=errors)


def scaffold(dest: str | Path, source: str | Path = BASELINE_AGENT_DIR) -> Path:
    """Copy an agents/current/-shaped directory's five files into `dest`.

    Used to seed a fresh agent directory (e.g. a mutation's working copy,
    or an agents/archive/<id>/ entry) from a known-good source, defaulting
    to this repo's own agents/current/. Only copies the five required
    files (plus README.md, if present) - never anything else that might
    be sitting in `source`. Raises FileNotFoundError up front if `source`
    isn't itself a valid agent directory.
    """
    source_dir = Path(source)
    result = validate(source_dir)
    if not result.ok:
        raise FileNotFoundError(f"{source_dir} is not a valid agent directory: {result.errors}")

    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in (*REQUIRED_FILES, "README.md"):
        src_file = source_dir / name
        if src_file.is_file():
            shutil.copy2(src_file, dest_dir / name)
    return dest_dir
