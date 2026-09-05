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

Tracing follows the documented SDK surface at docs.neatlogs.com/sdk/python:
`neatlogs.init(api_key=...)` to configure the client, and
`with neatlogs.trace(name=..., kind=...) as span:` context managers for
spans (kinds are the documented WORKFLOW/AGENT/LLM/TOOL/.../GUARDRAIL
values). `_neatlogs_trace_id` best-effort-probes common attribute names
(`trace_id`, `id`) first, then falls back to the OpenTelemetry span
context (`span.get_span_context().trace_id`) the real SDK's spans expose,
and degrades to `None` if none of those are present.

Only `run_agent()` opens the outer per-task/per-run spans (WORKFLOW,
outer AGENT, GUARDRAIL). The `agents/current/run.py` graph interpreter
opens one span per graph step (LLM for `llm_call`, TOOL for `tool_call`)
via `open_step_span()` below, so tool calls and model calls are visible
as their own spans within a task's trace rather than collapsed into a
single "agent_run" span - without threading the SDK through the fixed
`run(task_input) -> dict` entrypoint contract shared by every agent.
"""

from __future__ import annotations

import contextvars
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
    """Best-effort `neatlogs.init(api_key=...)`. Never raises; returns the SDK
    module on success (there to call `.trace()`/`.flush()` on), else None."""
    if _neatlogs_sdk is None:
        logger.warning("neatlogs package not installed; tracing disabled for this run")
        return None
    api_key = os.environ.get("NEATLOGS_API_KEY")
    if not api_key:
        logger.warning("NEATLOGS_API_KEY not set; tracing disabled for this run")
        return None
    try:
        _neatlogs_sdk.init(api_key=api_key)
    except Exception:
        logger.warning("neatlogs.init() failed; tracing disabled for this run", exc_info=True)
        return None
    return _neatlogs_sdk


class _NoopSpan:
    """No-op stand-in for `neatlogs.trace(...)` when tracing is inactive."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _SafeSpan:
    """Wraps a real `neatlogs.trace(...)` context manager so that a failure to
    start or stop tracing degrades silently instead of affecting the run."""

    def __init__(self, inner: Any, label: str) -> None:
        self._inner = inner
        self._label = label
        self._active = False

    def __enter__(self) -> Any:
        try:
            value = self._inner.__enter__()
        except Exception:
            logger.warning("neatlogs span %r failed to start", self._label, exc_info=True)
            return None
        self._active = True
        return value

    def __exit__(self, *exc_info: object) -> bool:
        if not self._active:
            return False
        try:
            return bool(self._inner.__exit__(*exc_info))
        except Exception:
            logger.warning("neatlogs span %r failed to stop", self._label, exc_info=True)
            return False


def _neatlogs_span(sdk: Any | None, name: str, kind: str) -> _NoopSpan | _SafeSpan:
    """Best-effort `with neatlogs.trace(name=..., kind=...) as span:` per the
    documented span kinds (WORKFLOW, AGENT, CHAIN, TOOL, RETRIEVER, EMBEDDING,
    GUARDRAIL, MCP_TOOL). Degrades to a no-op span on any failure."""
    if sdk is None:
        return _NoopSpan()
    try:
        inner = sdk.trace(name=name, kind=kind)
    except Exception:
        logger.warning("neatlogs.trace(name=%r, kind=%r) failed", name, kind, exc_info=True)
        return _NoopSpan()
    return _SafeSpan(inner, name)


def _neatlogs_trace_id(span: Any | None) -> str | None:
    """Best-effort trace id extraction.

    Probes common id attribute names first (what a fake/test SDK is
    likely to expose), then falls back to the OpenTelemetry span context
    the real neatlogs SDK's spans expose (`get_span_context().trace_id`,
    a 128-bit int formatted as hex) - covering both cases without
    depending on either.
    """
    if span is None:
        return None

    direct = getattr(span, "trace_id", None) or getattr(span, "id", None)
    if direct:
        return direct

    get_span_context = getattr(span, "get_span_context", None)
    if callable(get_span_context):
        try:
            context = get_span_context()
        except Exception:
            return None
        trace_id_int = getattr(context, "trace_id", None)
        if trace_id_int:
            return format(trace_id_int, "032x")

    return None


_active_sdk: contextvars.ContextVar[Any | None] = contextvars.ContextVar("_active_sdk", default=None)


def open_step_span(name: str, kind: str) -> _NoopSpan | _SafeSpan:
    """Best-effort span for one graph step, for use by an agent's own step
    interpreter (e.g. `agents/current/run.py`) while it runs inside
    `run_agent()`'s "agent_run" span.

    Reads the in-flight run's neatlogs SDK (or lack thereof) off a
    contextvar set for the duration of the `run_fn(task_input)` call, so
    an agent can open one span per step without the SDK needing to be
    threaded through the fixed `run(task_input) -> dict` entrypoint
    contract every agent implements. Outside of a `run_agent()` call (e.g.
    a unit test that calls `run_fn` directly) this degrades to a no-op
    span, exactly like tracing being unconfigured.
    """
    return _neatlogs_span(_active_sdk.get(), name, kind)


def _neatlogs_flush(sdk: Any | None) -> None:
    if sdk is None:
        return
    try:
        sdk.flush()
    except Exception:
        logger.warning("neatlogs.flush() failed", exc_info=True)


def run_agent(agent_path: str, task_spec: TaskSpec, split: str) -> list[RunResult]:
    """Run the agent at `agent_path` over `task_spec`'s dataset split.

    Never raises on a per-task basis: any exception raised while running or
    scoring a task is caught and recorded on that task's `RunResult.error`,
    so one broken task never aborts the rest of the run.
    """
    run_fn = _load_agent_run_fn(agent_path)
    scorer_fn = _load_scorer(task_spec.scorer_id)
    tasks = _load_dataset(task_spec.dataset_path, split, task_spec.max_tasks)

    sdk = _neatlogs_init()

    results: list[RunResult] = []
    for task in tasks:
        task_id = task.get("task_id", "")
        task_input = task.get("input", {})
        expected = task.get("expected")

        spend_before = llm._cumulative_spend_usd
        start = monotonic()

        output: Any = None
        error: str | None = None
        passed = False
        trace_id: str | None = None
        try:
            with _neatlogs_span(sdk, f"task:{task_id}", "WORKFLOW") as trace:
                trace_id = _neatlogs_trace_id(trace)

                with _neatlogs_span(sdk, "agent_run", "AGENT"):
                    token = _active_sdk.set(sdk)
                    try:
                        output = run_fn(task_input)
                    finally:
                        _active_sdk.reset(token)

                with _neatlogs_span(sdk, "score", "GUARDRAIL"):
                    passed = bool(scorer_fn(output, expected))
        except Exception as exc:  # noqa: BLE001 - a task's crash must never kill the run
            error = f"{type(exc).__name__}: {exc}"
        finally:
            latency_ms = int((monotonic() - start) * 1000)
            cost_usd = llm._cumulative_spend_usd - spend_before

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

    _neatlogs_flush(sdk)

    return results
