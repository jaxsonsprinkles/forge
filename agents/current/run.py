"""Baseline general-purpose agent entrypoint: run(task_input: dict) -> Any.

A generic interpreter: reads sibling graph.yaml for step order/control
flow, prompt.md for system instructions, tools.py for available tool
functions, and drives core.llm.complete() through the steps, using
memory.py's scratchpad to pass state between them. See README.md in this
directory for the full file/schema contract that core/proposer.py's
mutations edit.

Sibling files (tools.py, memory.py) are loaded by file path via
importlib, the same way core.runner._load_agent_run_fn loads this very
file - so this works whether agents/current/ is imported as a package,
loaded from an arbitrary checkout (a mutation's worktree, an archived
generation), or copied elsewhere entirely.

Each step opens its own best-effort Neatlogs span via
core.runner.open_step_span (LLM for llm_call, TOOL for tool_call), so a
task's trace shows every model/tool call individually instead of one
span for the whole run. Tracing is entirely optional here: outside of a
core.runner.run_agent() call (e.g. a test that calls run() directly)
open_step_span degrades to a no-op, same as tracing being unconfigured.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from core import llm
from core import runner as _runner

AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "claude-sonnet-5"


def _load_sibling_module(name: str) -> ModuleType:
    """Import a sibling file (e.g. tools.py) by path, relative to this
    file's own location rather than sys.path or package context."""
    module_path = AGENT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"forge_agent_current_{name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load sibling module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tools = _load_sibling_module("tools")
memory_mod = _load_sibling_module("memory")


def _load_graph() -> list[dict[str, Any]]:
    graph_path = AGENT_DIR / "graph.yaml"
    graph = yaml.safe_load(graph_path.read_text())
    steps = graph.get("steps") if isinstance(graph, dict) else None
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{graph_path} must define a non-empty top-level 'steps' list")
    return steps


def _load_system_prompt() -> str:
    return (AGENT_DIR / "prompt.md").read_text()


def _run_llm_call(step: dict[str, Any], system_prompt: str, task_input: dict[str, Any], memory: dict[str, Any]) -> str:
    instruction = step.get("instruction", "Produce your best answer for the task input below.")
    user_message = f"Task input:\n{json.dumps(task_input, indent=2, default=str)}\n\n{instruction}"

    input_keys = step.get("input_keys") or []
    if input_keys:
        context = {k: memory.get(k) for k in input_keys}
        user_message += f"\n\nContext from earlier steps:\n{json.dumps(context, indent=2, default=str)}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    model = step.get("model", DEFAULT_MODEL)
    text, _cost_usd, _latency_ms = llm.complete(messages, model=model)
    return text


def _run_tool_call(step: dict[str, Any], memory: dict[str, Any]) -> Any:
    tool_name = step.get("tool")
    tool_fn = tools.TOOLS.get(tool_name)
    if tool_fn is None:
        return {"error": f"unknown tool {tool_name!r}"}

    if "args_from" in step:
        args = memory.get(step["args_from"], {})
    else:
        args = step.get("args", {})
    if not isinstance(args, dict):
        args = {}

    try:
        return tool_fn(**args)
    except Exception as exc:  # noqa: BLE001 - a tool's crash must never kill the run
        return {"error": f"{type(exc).__name__}: {exc}"}


def _check_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _check_is_json(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


_CHECKS = {"non_empty": _check_non_empty, "is_json": _check_is_json}


def _run_verify(
    step: dict[str, Any],
    memory: dict[str, Any],
    step_index_by_name: dict[str, int],
    retry_counts: dict[str, int],
) -> int | None:
    """Run a verify step's check. Returns a step index to jump back to on
    failure (bounded by max_retries), or None to continue in order."""
    check_fn = _CHECKS.get(step.get("check", "non_empty"), _check_non_empty)
    value = memory.get(step.get("input_key"))
    if check_fn(value):
        return None

    on_fail = step.get("on_fail")
    if on_fail is None or on_fail not in step_index_by_name:
        return None

    max_retries = step.get("max_retries", 1)
    step_name = step["name"]
    retries_so_far = retry_counts.get(step_name, 0)
    if retries_so_far >= max_retries:
        return None

    retry_counts[step_name] = retries_so_far + 1
    return step_index_by_name[on_fail]


def _finalize_output(memory: dict[str, Any], output_key: str | None) -> Any:
    """Return the last step's raw output, parsed as JSON when it looks
    like one - so a JSON-object answer (invoices, docqa) comes back as a
    dict, while a plain-text answer (code repair's raw source) comes back
    unmodified, matching each domain's scorer without this file needing
    to know which domain it's running in."""
    if output_key is None:
        return {}
    value = memory.get(output_key)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                return value
    return value


def run(task_input: dict) -> Any:
    """Drive this agent's graph.yaml step order to answer one task.

    See README.md for the full contract. Nominally returns a dict per
    AGENTS.md's `run(task_input: dict) -> dict` entrypoint shape; in
    practice returns whatever the graph's last step produced, parsed as
    JSON when possible (see _finalize_output).
    """
    steps = _load_graph()
    system_prompt = _load_system_prompt()
    memory = memory_mod.create(task_input)

    step_index_by_name = {step["name"]: i for i, step in enumerate(steps)}
    retry_counts: dict[str, int] = {}

    last_output_key: str | None = None
    i = 0
    guard = 0
    max_iterations = len(steps) * 4 + 4  # bounds retry loops so a bad graph can't hang a run
    while i < len(steps) and guard < max_iterations:
        guard += 1
        step = steps[i]
        step_type = step.get("type")

        if step_type == "llm_call":
            with _runner.open_step_span(f"llm_call:{step['name']}", "LLM"):
                output = _run_llm_call(step, system_prompt, task_input, memory)
            output_key = step.get("output_key", step["name"])
            memory_mod.record(memory, output_key, output)
            last_output_key = output_key
            i += 1

        elif step_type == "tool_call":
            with _runner.open_step_span(f"tool_call:{step['name']}", "TOOL"):
                output = _run_tool_call(step, memory)
            output_key = step.get("output_key", step["name"])
            memory_mod.record(memory, output_key, output)
            last_output_key = output_key
            i += 1

        elif step_type == "verify":
            with _runner.open_step_span(f"verify:{step['name']}", "GUARDRAIL"):
                jump_to = _run_verify(step, memory, step_index_by_name, retry_counts)
            i = jump_to if jump_to is not None else i + 1

        else:
            raise ValueError(f"graph.yaml step {step.get('name')!r} has unknown type {step_type!r}")

    return _finalize_output(memory, last_output_key)
