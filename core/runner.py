"""Runs an agent over a domain's dataset split, producing per-task RunResults.

Loads the agent's `run.py` (an `agents/current/`-shaped directory: prompt.md,
tools.py, memory.py, graph.yaml, run.py) and calls its `run(task_input: dict)
-> dict` entrypoint for every task in the requested split, up to
`task_spec.max_tasks`. A task's dataset row is scored by the domain's scorer,
addressed by `task_spec.scorer_id` in `"module.path:function_name"` form
(resolved via `importlib`), e.g. `"domains.coderepair.scorer:score_v1"`.

Every task is wrapped so no single task's crash can kill the run: exceptions
are caught and recorded on `RunResult.error`, never re-raised.

Neatlogs tracing is best-effort. If the SDK isn't installed, no API key is
configured, or any call into it fails, tracing silently degrades to
`trace_id=None` (with a logged warning) - it must never break a run.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from core import llm
from core.types import RunResult, TaskSpec

logger = logging.getLogger(__name__)

try:
    import neatlogs as _neatlogs_sdk
except ImportError:
    _neatlogs_sdk = None


def _load_agent_run_fn(agent_path: str | Path) -> Callable[[dict], dict]:
    """Load the `run(task_input: dict) -> dict` entrypoint from an agent's run.py."""
    run_py = Path(agent_path) / "run.py"
    spec = importlib.util.spec_from_file_location(f"forge_agent_{Path(agent_path).name}", run_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent run.py from {run_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_fn = getattr(module, "run", None)
    if run_fn is None:
        raise AttributeError(f"Agent at {agent_path} has no run(task_input) entrypoint in run.py")
    return run_fn


def _load_scorer(scorer_id: str) -> Callable[[Any, Any], bool]:
    """Resolve a scorer_id of the form 'module.path:function_name' to a callable."""
    module_name, _, attr = scorer_id.partition(":")
    if not attr:
        raise ValueError(f"scorer_id {scorer_id!r} must be in 'module.path:function_name' form")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _load_dataset(dataset_path: str, split: str, max_tasks: int) -> list[dict]:
    """Load rows for `split` from a JSONL dataset, capped at max_tasks rows."""
    tasks: list[dict] = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split", "train") != split:
                continue
            tasks.append(row)
            if len(tasks) >= max_tasks:
                break
    return tasks


def _neatlogs_init() -> Any | None:
    """Best-effort Neatlogs tracer init. Never raises; returns None on any failure."""
    if _neatlogs_sdk is None:
        logger.warning("neatlogs package not installed; tracing disabled for this run")
        return None
    api_key = os.environ.get("NEATLOGS_API_KEY")
    if not api_key:
        logger.warning("NEATLOGS_API_KEY not set; tracing disabled for this run")
        return None
    try:
        return _neatlogs_sdk.init(api_key=api_key)
    except Exception:
        logger.warning("neatlogs.init() failed; tracing disabled for this run", exc_info=True)
        return None


def _neatlogs_start_trace(tracer: Any, task_id: str) -> tuple[Any, str | None]:
    """Start a per-task trace. Returns (trace_or_None, trace_id_or_None)."""
    if tracer is None:
        return None, None
    try:
        trace = tracer.start_trace(name=f"task:{task_id}")
        trace_id = getattr(trace, "id", None) or getattr(trace, "trace_id", None)
        return trace, trace_id
    except Exception:
        logger.warning("neatlogs start_trace failed for task %s", task_id, exc_info=True)
        return None, None


def _neatlogs_end_trace(trace: Any) -> None:
    if trace is None:
        return
    try:
        trace.end()
    except Exception:
        logger.warning("neatlogs end_trace failed", exc_info=True)


def _neatlogs_start_span(trace: Any, name: str) -> Any | None:
    """Start a named span (e.g. an agent step or tool call) within a trace."""
    if trace is None:
        return None
    try:
        return trace.start_span(name)
    except Exception:
        logger.warning("neatlogs start_span(%r) failed", name, exc_info=True)
        return None


def _neatlogs_end_span(span: Any | None) -> None:
    if span is None:
        return
    try:
        span.end()
    except Exception:
        logger.warning("neatlogs end_span failed", exc_info=True)


def run_agent(agent_path: str, task_spec: TaskSpec, split: str) -> list[RunResult]:
    """Run the agent at `agent_path` over `task_spec`'s dataset split.

    Never raises on a per-task basis: any exception raised while running or
    scoring a task is caught and recorded on that task's `RunResult.error`,
    so one broken task never aborts the rest of the run.
    """
    run_fn = _load_agent_run_fn(agent_path)
    scorer_fn = _load_scorer(task_spec.scorer_id)
    tasks = _load_dataset(task_spec.dataset_path, split, task_spec.max_tasks)

    tracer = _neatlogs_init()

    results: list[RunResult] = []
    for task in tasks:
        task_id = task.get("task_id", "")
        task_input = task.get("input", {})
        expected = task.get("expected")

        trace, trace_id = _neatlogs_start_trace(tracer, task_id)
        spend_before = llm._cumulative_spend_usd
        start = monotonic()

        output: Any = None
        error: str | None = None
        passed = False
        try:
            span = _neatlogs_start_span(trace, "agent_run")
            try:
                output = run_fn(task_input)
            finally:
                _neatlogs_end_span(span)

            span = _neatlogs_start_span(trace, "score")
            try:
                passed = bool(scorer_fn(output, expected))
            finally:
                _neatlogs_end_span(span)
        except Exception as exc:  # noqa: BLE001 - a task's crash must never kill the run
            error = f"{type(exc).__name__}: {exc}"
        finally:
            latency_ms = int((monotonic() - start) * 1000)
            cost_usd = llm._cumulative_spend_usd - spend_before
            _neatlogs_end_trace(trace)

        results.append(
            RunResult(
                task_id=task_id,
                output=output,
                passed=passed,
                error=error,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                trace_id=trace_id,
            )
        )

    return results
