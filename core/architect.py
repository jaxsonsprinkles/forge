"""Validates, scaffolds, and builds an agents/current/-shaped agent directory.

See AGENTS.md: the agent is always exactly five plain files - prompt.md,
tools.py, memory.py, graph.yaml, run.py - never a class, package, or
config object. This module is the one place that knows that fixed
five-file contract well enough to (a) check a directory actually
satisfies it, before core/runner.py tries to load it, (b) scaffold a
fresh copy of the current baseline into a new directory - e.g. so a
mutation's working copy, or an archived generation under
agents/archive/<id>/, starts from a known-good copy rather than an
agent-shaped directory hand-assembled from scratch - and (c) build a
brand-new agent from a TaskSpec via `build_agent()`.

Neither core/gate.py nor core/executor_ao.py import this module today:
executor_ao dispatches a plain text instruction to an AO worker that
edits an existing checkout in place, and gate only persists ScoreCards
and a branch ref (not agent files) into agents/archive/<id>/ (see their
docstrings). So the minimal real interface needed right now is
validation (any candidate agent directory should be checkable before
it's trusted) plus scaffolding/building (something has to produce the
first copy of those five files) - nothing more speculative than that.

## What `build_agent()` generates vs. what's fixed scaffolding

`build_agent(task_spec, dest)` makes exactly ONE `core.llm.complete()`
call, not five. The five files split into two groups:

- **Fixed scaffolding - `run.py`, `memory.py`, `tools.py`.** Copied
  byte-for-byte from `BASELINE_AGENT_DIR` (this repo's own
  `agents/current/`). `run.py` is the generic graph interpreter itself -
  it has no task-specific knowledge by construction, so there is nothing
  for a model to usefully generate there, only room to introduce bugs.
  `memory.py` is an equally generic scratchpad. `tools.py` ships
  `run_python` and `search_text`, which already cover every tool name any
  shipped domain's `task_spec.tools` names (see `domains/*/task_spec.json`);
  a tool implementation is real, testable Python that benefits far more
  from being hand-written once than regenerated per domain.
- **Model-generated - `prompt.md`, `graph.yaml`.** The one thing that
  actually varies per domain is *what the agent should do*: its system
  instructions and its step sequence. One `complete()` call, prompted
  with `task_spec.goal` and the tool names available in the fixed
  `tools.py`, returns a JSON object with `prompt_md` (the system prompt
  text) and `graph_steps` (a list of step dicts, serialized to YAML here
  rather than asked of the model directly - a model asked for raw YAML
  text can produce subtly invalid indentation, but a model asked for a
  JSON list of step objects only has to get JSON right, and this module
  turns that into well-formed YAML deterministically via `yaml.dump`).

The prompt explicitly asks for a **weak** first draft (typically a single
`llm_call` step) - see AGENTS.md and the task-10 spec: a strong first
draft would leave Forge's outer mutation loop (core/proposer.py, which
edits exactly these five files) nothing to improve. `build_agent()`
validates its own output with `validate()` before returning and raises
`ValueError` if the model's response doesn't parse into a well-formed
agent, since a build that produces a broken agent should fail loudly
rather than hand `core/runner.py` something that will blow up later.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core import llm
from core.types import TaskSpec

REQUIRED_FILES: tuple[str, ...] = ("prompt.md", "tools.py", "memory.py", "graph.yaml", "run.py")

# The generic interpreter and its scratchpad/tool implementations never
# depend on task_spec - build_agent() copies these verbatim rather than
# asking a model to regenerate them (see module docstring).
_FIXED_SCAFFOLD_FILES: tuple[str, ...] = ("run.py", "memory.py", "tools.py")

# The only two files build_agent()'s single model call actually produces.
_LLM_GENERATED_FILES: tuple[str, ...] = ("prompt.md", "graph.yaml")

assert set(_FIXED_SCAFFOLD_FILES) | set(_LLM_GENERATED_FILES) == set(REQUIRED_FILES)

DEFAULT_BUILD_MODEL = "claude-sonnet-5"

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


def _load_tool_names(source_dir: Path) -> list[str]:
    """Return the sorted tool names `source_dir`'s tools.py exposes via TOOLS.

    Used only to tell build_agent()'s prompt which tool names a
    `tool_call` step is actually allowed to reference - never raises;
    an unloadable tools.py just means no tool names are offered.
    """
    tools_path = source_dir / "tools.py"
    try:
        spec = importlib.util.spec_from_file_location("forge_architect_build_tools", tools_path)
        if spec is None or spec.loader is None:
            return []
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - tool-name discovery is best-effort
        return []
    return sorted(getattr(module, "TOOLS", {}).keys())


def _build_messages(task_spec: TaskSpec, available_tools: list[str]) -> list[dict[str, str]]:
    """Build the single prompt used to generate prompt.md + graph.yaml.

    Deliberately asks for a *weak* first draft: build_agent() is Forge's
    architect for a brand-new agent, not its optimizer. See module
    docstring for why a strong first draft is the wrong goal.
    """
    tool_list = ", ".join(available_tools) if available_tools else "(none available - do not emit any tool_call steps)"
    system_prompt = (
        "You are Forge's architect. Forge is a meta-system that builds and iteratively improves "
        "AI agents. Your job here is to design the FIRST DRAFT of a brand-new agent for one task "
        "domain - a deliberately weak, simple baseline, not a well-tuned solution. A later "
        "automated mutation loop will improve this agent over many generations by editing its "
        "prompt, tools, memory, and step graph one change at a time; a strong first draft would "
        "leave that loop nothing to measurably improve. Prefer the simplest graph that could "
        "plausibly attempt the task - usually a single llm_call step that reads the task input "
        "and produces a final answer directly, with no verification or multi-step reasoning.\n\n"
        "Respond with ONLY a single JSON object - no markdown code fences, no commentary before or "
        "after it - with exactly two keys:\n"
        '  "prompt_md": a string of markdown system instructions for the agent to follow on every '
        "task in this domain.\n"
        '  "graph_steps": a JSON list of one or more step objects, in execution order. Each step '
        'is an object with a unique "name" and a "type" of "llm_call", "tool_call", or "verify", '
        "plus that type's other fields:\n"
        '    llm_call: "instruction" (str), "input_keys" (list of earlier step names to include as '
        'context, optional), "output_key" (str, optional, defaults to the step name).\n'
        '    tool_call: "tool" (str, must be one of the available tool names below), "args" (static '
        'dict of keyword arguments, optional), "output_key" (str, optional).\n'
        '    verify: "input_key" (str, required), "check" ("non_empty" or "is_json"), "on_fail" '
        '(name of an earlier step to retry from, optional), "max_retries" (int, optional).\n'
        "The interpreter that runs these steps is fixed and completely generic - it has no "
        "knowledge of this task domain, so graph_steps is the only place task-specific step logic "
        "can live."
    )
    user_prompt = (
        f"Task domain goal: {task_spec.goal}\n"
        f"Tool names available to tool_call steps: {tool_list}\n\n"
        "Respond with the JSON object described above, for this domain."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _strip_code_fence(text: str) -> str:
    """Strip a single leading/trailing markdown code fence, if present.

    Models asked for "ONLY JSON" sometimes wrap it in ```json ... ```
    anyway; tolerate that rather than failing the whole build over it.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_build_response(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse build_agent()'s model response into (prompt_md, graph_steps).

    Raises ValueError with the raw response attached on any malformed
    shape, since a malformed response means build_agent() cannot produce
    a trustworthy agent and must fail loudly rather than guess.
    """
    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"build_agent's model response was not valid JSON: {exc}\nRaw response:\n{text}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"build_agent's model response must be a JSON object, got: {data!r}")

    prompt_md = data.get("prompt_md")
    graph_steps = data.get("graph_steps")

    if not isinstance(prompt_md, str) or not prompt_md.strip():
        raise ValueError(f"build_agent's model response is missing a non-empty 'prompt_md' string: {data!r}")
    if not isinstance(graph_steps, list) or not graph_steps:
        raise ValueError(f"build_agent's model response is missing a non-empty 'graph_steps' list: {data!r}")
    if not all(isinstance(step, dict) and "name" in step and "type" in step for step in graph_steps):
        raise ValueError(f"build_agent's model response has a malformed graph_steps entry: {graph_steps!r}")

    return prompt_md, graph_steps


def build_agent(
    task_spec: TaskSpec,
    dest: str | Path,
    *,
    model: str = DEFAULT_BUILD_MODEL,
    source: str | Path = BASELINE_AGENT_DIR,
) -> Path:
    """Build a brand-new agents/current/-shaped agent for `task_spec` at `dest`.

    Writes all five required files: run.py, memory.py, and tools.py are
    copied verbatim from `source` (the fixed, generic scaffolding - see
    module docstring), while prompt.md and graph.yaml are generated by
    exactly ONE `core.llm.complete()` call tailored to `task_spec.goal`
    and the tool names `source`'s tools.py exposes.

    Raises ValueError if the model's response doesn't parse into a
    well-formed agent (see `_parse_build_response`) or if the resulting
    directory fails `validate()`. Never silently produces a broken agent.
    """
    source_dir = Path(source)
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for name in _FIXED_SCAFFOLD_FILES:
        shutil.copy2(source_dir / name, dest_dir / name)

    available_tools = _load_tool_names(source_dir)
    messages = _build_messages(task_spec, available_tools)
    text, _cost_usd, _latency_ms = llm.complete(messages, model=model)
    prompt_md, graph_steps = _parse_build_response(text)

    (dest_dir / "prompt.md").write_text(prompt_md)
    (dest_dir / "graph.yaml").write_text(yaml.dump({"steps": graph_steps}, sort_keys=False))

    result = validate(dest_dir)
    if not result.ok:
        raise ValueError(f"build_agent produced an invalid agent at {dest_dir}: {result.errors}")
    return dest_dir
