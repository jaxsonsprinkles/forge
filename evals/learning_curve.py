"""The headline harness: prove memory makes the SAME agent code better.

    python evals/learning_curve.py --domain X --passes 5 --memory on|off

Runs one agent (unmodified across every pass - `agents/current/run.py`, or
whatever `--agent` points at) over a domain's TRAIN split `--passes` times,
with `core/memory.py` persisting across passes and `core/reflect.py` active,
then scores the same agent against the HOLDOUT split after every pass. The
`--memory off` run is the control: identical code, dataset, and pass count,
but memory retrieval and reflection are switched off entirely. If `on`
improves holdout accuracy pass over pass while `off` stays flat, that
separation - not any single number - is this milestone's actual claim.

Recall is native now - this harness no longer injects memory itself
---------------------------------------------------------------------
Earlier versions of this harness reached into a run by copying
`task_input` and adding retrieved lessons under a namespaced key
(`_augment_task_input`), because at the time nothing in `run(task_input)
-> Any`'s fixed entrypoint ever called `core.memory.retrieve()`. That is
no longer true: `agents/current/run.py` now has a native `recall` step
type (`_handle_recall`) that graph.yaml can declare, which calls
`core.memory.retrieve()`/`peek()` itself and exposes lessons to later
`llm_call` steps via the scratchpad, not via `task_input`.

Keeping this harness's own retrieval *and* the agent's native `recall`
step would retrieve the same lesson twice per task - once each - which is
exactly the double-counting bug class TASK 11/PR #24 fixed for the
`reflect` step's dedup lookup, reintroduced via a new path. So this
harness no longer calls `core.memory.retrieve()`/`peek()` or augments
`task_input` at all; retrieval happens exactly once, inside the run, via
whatever `recall` step(s) the agent's own graph.yaml declares (zero, for
an agent with no memory support - `retrieve()` is then never called for
that agent, same as `--memory off`).

Controlling a native recall step from outside graph.yaml
------------------------------------------------------------
`_handle_recall` reads three env vars for exactly this purpose (see its
docstring in agents/current/run.py): `FORGE_MEMORY_MODE` ("off" makes
recall skip retrieval entirely - this is how `--memory off` reaches a
native recall step without touching graph.yaml or task_input, keeping
task_input byte-identical for caching - see "On the control arm being
flat" below), `FORGE_MEMORY_BASE_DIR` (redirects reads to this harness's
own scratch store), and `FORGE_MEMORY_READ_ONLY` (forces `peek()` over
`retrieve()` for holdout - see "Holdout must never write" below). This
harness sets all three around every `run_fn()` call via `_memory_env()`,
and restores whatever was there before once the call returns.

Getting the real outcome back out: the `_trace` kwarg
-----------------------------------------------------------
`run(task_input) -> Any` never receives the scorer's verdict - reinforcing
with the real pass/fail outcome (rather than a run-local proxy) has to
happen in a caller that has both. `agents/current/run.py`'s `run()` now
accepts an optional `_trace: dict | None` that a `recall` step populates
with what it retrieved (domain_id/base_dir/entry ids). This harness passes
`_trace={}` into any `run_fn` whose signature accepts it (checked once via
`_supports_trace_kwarg`, so agents without a `_trace` parameter - e.g.
`tests/fixtures/agents/good_agent`, or any other agent's fixed two-arg
`run(task_input)` - are called exactly as before, unaffected). After
scoring, every entry id the trace reports gets `core.memory.reinforce
(entry_id, helped=<task passed>)`, and the same ids (resolved back to
`MemoryEntry` objects via `core.memory.get_entry()`, never a second
`retrieve()`/`peek()` call) are handed to `core.reflect.reflect()` as
`existing_memory` dedup context - see "Reinforcement and reflection
wiring" below. `core/runner.py`'s `run_agent()` was checked too: it calls
`run_fn(task_input)` positionally with no `_trace`, so agents evaluated
through it (the outer loop, `evals/run_eval.py`) get native recall's
lesson injection but not reinforcement - extending that is a natural
follow-up, but `core/runner.py` is out of this change's scope.

Holdout must never write - so it never uses retrieve()
-----------------------------------------------------------
`core.memory.retrieve()` is not read-only: it durably bumps every returned
entry's `times_retrieved` via its own internal `write()` call (see that
module's docstring). Holdout scoring must stay a pure read of "what has
memory learned so far" with zero side effects - it is never a source of
new lessons, confidence changes, or even retrieval-count bookkeeping, so
that a later run's holdout results can never be traced back to something
holdout itself did. `core.memory.peek()` is `retrieve()`'s read-only
twin - identical ranking, no write-back. This harness never calls either
itself anymore (see above); instead it sets `FORGE_MEMORY_READ_ONLY=1`
around holdout's `run_fn()` calls, so a native `recall` step uses
`peek()` for holdout and `retrieve()` for train - the module invariant
"holdout never calls core.memory.write()" still holds by construction.

Reinforcement and reflection wiring
--------------------------------------
Neither `core/memory.py` nor `core/reflect.py` calls `reinforce()`
themselves (both modules say so in their own docstrings/PRs), and
`core.reflect.reflect()` is a pure function that never calls
`core.memory.write()` either. This harness is the caller that closes both
loops for a train task: after scoring, every memory entry a native
`recall` step retrieved gets `reinforce(helped=<task passed>)` - a crude
but directionally-correct proxy for "did this lesson correlate with
success" - and `reflect.reflect()` is called with the real `RunResult`/
`expected` (ground truth this harness has and a graph-native reflect step
never would, per agents/current/README.md's `reflect` step limitations)
plus those same retrieved entries as dedup context, its output written via
`memory.write()`. An agent whose graph.yaml has no `recall` step simply
contributes an empty `existing_memory` list to `reflect()` - reflection
and reinforcement don't require retrieval to have happened.

Fresh memory per invocation
-----------------------------
`core/memory.py`'s functions all take an explicit `base_dir` parameter
rather than reading a module-level global, so this harness never needs to
touch `core/memory.py` to get isolation: each `(domain, memory_mode)`
combination gets its own scratch directory under
`memory/_learning_curve/{domain}_memory_{on|off}/`, wiped at the start of
every run unless `--resume` is passed. Skipping this reset would silently
mix an earlier invocation's accumulated lessons into what's supposed to be
a fresh N-pass experiment, invalidating the whole curve.

On the control arm being flat
-------------------------------
`--memory off` sets `FORGE_MEMORY_MODE=off` for every `run_fn()` call, so
a native `recall` step never calls `core.memory.retrieve()`/`peek()` at
all, and this harness's own post-scoring reflect/reinforce step is
skipped outright (`memory_mode != "on"`). The agent still runs the
identical train and holdout tasks every pass (for cost/latency parity
with the `on` arm), through the identical unmodified code and byte-
identical `task_input` (never augmented, on or off). If the agent's own
`llm_call` steps go through `core.llm.complete()` (real provider calls,
not this file's own mocked tests), those calls are cached on disk keyed
by `(model, messages, params)`; since `task_input` never changes pass to
pass under `--memory off`, every prompt after pass 1 is a byte-identical
cache hit, and the holdout ScoreCard is expected to be *exactly* flat
rather than "flat with some noise." That is the correct, expected
behavior of this harness, not a bug to chase - it's documented here so a
bit-identical control curve isn't mistaken for the harness failing to
actually run anything.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import logging
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import Any, Iterator, Literal

from core import llm, memory, reflect
from core.memory import MemoryEntry
from core.runner import _load_agent_run_fn, _load_dataset, _load_scorer
from core.scorer import score_runs
from core.types import RunResult
from evals.run_eval import load_task_spec

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("ledger/learning_curves")
DEFAULT_MEMORY_SCRATCH_ROOT = Path("memory/_learning_curve")

# Env vars agents/current/run.py's `_handle_recall` reads to let a caller
# control a native `recall` step from outside graph.yaml/task_input - see
# that function's docstring and this module's docstring above.
_ENV_MEMORY_MODE = "FORGE_MEMORY_MODE"
_ENV_MEMORY_BASE_DIR = "FORGE_MEMORY_BASE_DIR"
_ENV_MEMORY_READ_ONLY = "FORGE_MEMORY_READ_ONLY"
_ENV_MEMORY_K = "FORGE_MEMORY_K"
_RECALL_ENV_VARS = (_ENV_MEMORY_MODE, _ENV_MEMORY_BASE_DIR, _ENV_MEMORY_READ_ONLY, _ENV_MEMORY_K)


def _memory_base_dir_for(domain_id: str, memory_mode: str, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override)
    return DEFAULT_MEMORY_SCRATCH_ROOT / f"{domain_id}_memory_{memory_mode}"


def _reset_memory_store(domain_id: str, base_dir: str | Path) -> None:
    """Wipe this experiment's scratch memory store so the run starts empty.

    See module docstring's "Fresh memory per invocation" - a stale store
    silently polluting what's supposed to be a from-scratch curve would
    invalidate the whole result.
    """
    shutil.rmtree(Path(base_dir) / domain_id, ignore_errors=True)


def _supports_trace_kwarg(run_fn: Any) -> bool:
    """Whether `run_fn` accepts a `_trace` keyword argument.

    See module docstring's "Getting the real outcome back out" - only
    `agents/current/run.py`'s `run()` currently declares this parameter.
    An agent without it (e.g. `tests/fixtures/agents/good_agent`'s bare
    `run(task_input)`) is called exactly as it always was, with no trace
    threaded through - this harness never assumes every run_fn supports it.
    """
    try:
        params = inspect.signature(run_fn).parameters
    except (TypeError, ValueError):
        return False
    return "_trace" in params


@contextmanager
def _memory_env(memory_mode: Literal["on", "off"], base_dir: str | Path, k: int, read_only: bool) -> Iterator[None]:
    """Point a native `recall` step (agents/current/run.py's
    `_handle_recall`) at this harness's scratch store for one `run_fn()`
    call, then restore whatever env vars were there before.

    See module docstring's "Controlling a native recall step from outside
    graph.yaml" - this is the whole mechanism, no graph.yaml or
    task_input change required.
    """
    saved = {name: os.environ.get(name) for name in _RECALL_ENV_VARS}
    os.environ[_ENV_MEMORY_MODE] = memory_mode
    os.environ[_ENV_MEMORY_BASE_DIR] = str(base_dir)
    os.environ[_ENV_MEMORY_K] = str(k)
    if read_only:
        os.environ[_ENV_MEMORY_READ_ONLY] = "1"
    else:
        os.environ.pop(_ENV_MEMORY_READ_ONLY, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_one_task(
    run_fn: Any,
    scorer_fn: Any,
    task: dict[str, Any],
    *,
    domain_id: str,
    memory_mode: Literal["on", "off"],
    k: int,
    base_dir: str | Path,
    read_only: bool,
) -> tuple[RunResult, dict[str, Any] | None, dict[str, Any], Any]:
    """Run and score one dataset row.

    No memory injection happens here anymore - see module docstring. If
    `run_fn` supports it, a `_trace` dict is passed in and returned
    alongside the result; a native `recall` step (if the agent's
    graph.yaml has one) populates its `"recall"` key with what it
    retrieved, which the caller uses to reflect/reinforce with the real
    outcome. `read_only=True` (holdout) makes a native recall step use
    `peek()` instead of `retrieve()` via `FORGE_MEMORY_READ_ONLY`.
    """
    task_id = task.get("task_id", "")
    task_input = task.get("input", {})
    expected = task.get("expected")

    trace: dict[str, Any] | None = {} if _supports_trace_kwarg(run_fn) else None

    spend_before = llm._cumulative_spend_usd
    start = monotonic()
    output: Any = None
    error: str | None = None
    passed = False
    try:
        with _memory_env(memory_mode, base_dir, k, read_only):
            output = run_fn(task_input, _trace=trace) if trace is not None else run_fn(task_input)
        passed = bool(scorer_fn(output, expected))
    except Exception as exc:  # noqa: BLE001 - one task's crash must never kill the pass
        error = f"{type(exc).__name__}: {exc}"
    finally:
        latency_ms = int((monotonic() - start) * 1000)
        cost_usd = llm._cumulative_spend_usd - spend_before

    result = RunResult(
        task_id=task_id,
        output=output,
        passed=passed,
        error=error,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        trace_id=None,
    )
    return result, trace, task_input, expected


def _recall_records(trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    return (trace or {}).get("recall", [])


def _resolve_retrieved_entries(recall_records: list[dict[str, Any]]) -> list[MemoryEntry]:
    """Resolve a trace's retrieved entry ids back to `MemoryEntry` objects
    via `core.memory.get_entry()` - never a second `retrieve()`/`peek()`
    call, which is exactly the double-retrieval this harness now avoids
    (see module docstring)."""
    entries: list[MemoryEntry] = []
    for record in recall_records:
        for entry_id in record["entry_ids"]:
            entry = memory.get_entry(entry_id, record["domain_id"], base_dir=record["base_dir"])
            if entry is not None:
                entries.append(entry)
    return entries


def _run_train_pass(
    run_fn: Any,
    scorer_fn: Any,
    train_tasks: list[dict[str, Any]],
    *,
    domain_id: str,
    memory_mode: Literal["on", "off"],
    k: int,
    base_dir: str | Path,
    gen_n: int,
    reflect_model: str,
) -> float:
    """Run every train task once, reflecting and reinforcing when memory is on.

    Returns the average number of entries retrieved per train task (0.0
    when memory is off, or when the agent's graph has no `recall` step).
    """
    retrieved_counts: list[int] = []

    for task in train_tasks:
        result, trace, task_input, expected = _run_one_task(
            run_fn,
            scorer_fn,
            task,
            domain_id=domain_id,
            memory_mode=memory_mode,
            k=k,
            base_dir=base_dir,
            read_only=False,
        )
        recall_records = _recall_records(trace)
        retrieved_counts.append(sum(len(record["entry_ids"]) for record in recall_records))

        if memory_mode != "on":
            continue

        existing_memory = _resolve_retrieved_entries(recall_records)
        new_entries = reflect.reflect(
            result,
            task_input,
            expected,
            None,
            existing_memory,
            domain_id=domain_id,
            gen_n=gen_n,
            model=reflect_model,
        )
        for entry in new_entries:
            memory.write(entry, base_dir=base_dir)

        for record in recall_records:
            for entry_id in record["entry_ids"]:
                try:
                    memory.reinforce(entry_id, record["domain_id"], helped=result.passed, base_dir=record["base_dir"])
                except KeyError:
                    logger.warning(
                        "reinforce(): entry %r for domain %r no longer exists", entry_id, record["domain_id"]
                    )

    if not retrieved_counts:
        return 0.0
    return sum(retrieved_counts) / len(retrieved_counts)


def _run_holdout_pass(
    run_fn: Any,
    scorer_fn: Any,
    holdout_tasks: list[dict[str, Any]],
    *,
    domain_id: str,
    memory_mode: Literal["on", "off"],
    k: int,
    base_dir: str | Path,
) -> list[RunResult]:
    """Score the agent against holdout. Never reflects, writes, or reinforces."""
    results = []
    for task in holdout_tasks:
        result, _, _, _ = _run_one_task(
            run_fn,
            scorer_fn,
            task,
            domain_id=domain_id,
            memory_mode=memory_mode,
            k=k,
            base_dir=base_dir,
            read_only=True,
        )
        results.append(result)
    return results


def _truncate_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def _append_record(record: dict[str, Any], path: Path) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_learning_curve(
    domain: str,
    passes: int,
    memory_mode: Literal["on", "off"],
    *,
    agent_path: str | Path = "agents/current",
    domains_root: str | Path = "domains",
    k: int = 5,
    memory_base_dir: str | Path | None = None,
    reset_memory: bool = True,
    reflect_model: str = reflect.DEFAULT_MODEL,
    gen_start: int = 0,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed_memory: int = 0,
) -> list[dict[str, Any]]:
    """Run the full N-pass learning curve for one (domain, memory_mode).

    Writes one JSON record per pass to
    `{output_dir}/{domain}_memory_{on|off}.jsonl` (truncated at the start
    of this call - each invocation produces a complete, self-contained
    curve, not an append to a previous one) and returns the same records.

    `seed_memory`, when > 0 and `memory_mode == "on"`, runs that many
    TRAIN tasks (a prefix of the train split, via `_run_train_pass` -
    same reflect+reinforce wiring as every other pass) before pass 0, to
    pre-populate the store rather than starting every curve from a
    literally empty one. Holdout is never touched by this - it only ever
    runs after a pass's own train tasks, same as always. A no-op under
    `--memory off` (there is nothing to seed).
    """
    task_spec = load_task_spec(domain, domains_root)
    run_fn = _load_agent_run_fn(agent_path)
    scorer_fn = _load_scorer(task_spec.scorer_id)
    train_tasks = _load_dataset(task_spec.dataset_path, "train", task_spec.max_tasks)
    holdout_tasks = _load_dataset(task_spec.dataset_path, "holdout", task_spec.max_tasks)

    base_dir = _memory_base_dir_for(domain, memory_mode, memory_base_dir)
    if reset_memory:
        _reset_memory_store(domain, base_dir)

    if seed_memory > 0 and memory_mode == "on":
        _run_train_pass(
            run_fn,
            scorer_fn,
            train_tasks[:seed_memory],
            domain_id=domain,
            memory_mode=memory_mode,
            k=k,
            base_dir=base_dir,
            gen_n=gen_start - 1,
            reflect_model=reflect_model,
        )

    output_path = Path(output_dir) / f"{domain}_memory_{memory_mode}.jsonl"
    _truncate_output(output_path)

    records: list[dict[str, Any]] = []
    for pass_idx in range(passes):
        gen_n = gen_start + pass_idx

        avg_retrieved = _run_train_pass(
            run_fn,
            scorer_fn,
            train_tasks,
            domain_id=domain,
            memory_mode=memory_mode,
            k=k,
            base_dir=base_dir,
            gen_n=gen_n,
            reflect_model=reflect_model,
        )

        holdout_results = _run_holdout_pass(
            run_fn,
            scorer_fn,
            holdout_tasks,
            domain_id=domain,
            memory_mode=memory_mode,
            k=k,
            base_dir=base_dir,
        )
        holdout_scorecard = score_runs(holdout_results, "holdout")

        memory_entry_count = (
            len(memory.load_all(domain, base_dir, statuses=None)) if memory_mode == "on" else 0
        )

        record = {
            "pass": pass_idx,
            "domain": domain,
            "memory_mode": memory_mode,
            "holdout_scorecard": dataclasses.asdict(holdout_scorecard),
            "memory_entry_count": memory_entry_count,
            "avg_entries_retrieved": avg_retrieved,
            "cost_per_task": holdout_scorecard.cost_per_task,
            "p50_latency_ms": holdout_scorecard.p50_latency_ms,
        }
        _append_record(record, output_path)
        records.append(record)

    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same unmodified agent over a domain's train split for N passes, "
            "with memory persisting across passes, scoring against holdout after each."
        )
    )
    parser.add_argument("--domain", required=True, help="Domain id, e.g. 'github_triage'.")
    parser.add_argument("--passes", type=int, default=5, help="Number of train-then-holdout passes.")
    parser.add_argument("--memory", required=True, choices=["on", "off"], help="Memory/reflection arm.")
    parser.add_argument("--agent", default="agents/current", help="Path to the agent directory.")
    parser.add_argument("--domains-root", default="domains")
    parser.add_argument("--k", type=int, default=5, help="Max memory entries retrieved per task.")
    parser.add_argument(
        "--memory-base-dir",
        default=None,
        help="Override the scratch memory root (default: memory/_learning_curve/{domain}_memory_{mode}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Do not reset the memory store before running (default: always start from empty memory).",
    )
    parser.add_argument("--reflect-model", default=reflect.DEFAULT_MODEL)
    parser.add_argument("--gen-start", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--seed-memory",
        type=int,
        default=0,
        help=(
            "Run N train tasks with reflection on before pass 0, to pre-populate the "
            "memory store (never touches holdout). Default 0 (no seeding)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    args = build_arg_parser().parse_args(argv)

    records = run_learning_curve(
        domain=args.domain,
        passes=args.passes,
        memory_mode=args.memory,
        agent_path=args.agent,
        domains_root=args.domains_root,
        k=args.k,
        memory_base_dir=args.memory_base_dir,
        reset_memory=not args.resume,
        reflect_model=args.reflect_model,
        gen_start=args.gen_start,
        output_dir=args.output_dir,
        seed_memory=args.seed_memory,
    )
    for record in records:
        print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
