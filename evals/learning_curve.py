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

Injecting memory without touching run.py
-----------------------------------------
`core/runner.py`'s `run_agent()` and `agents/current/run.py`'s graph
interpreter both take the fixed entrypoint `run(task_input: dict) -> Any`;
`run()` builds its own per-run scratchpad internally (`memory_mod.create
(task_input)`) and nothing external can reach into it - the only channel a
caller has into a run at all is the `task_input` dict itself. Every
`llm_call` step already serializes the *whole* `task_input` dict into its
user message (see `run.py`'s `_run_llm_call`), so the least invasive
injection point - no change to run.py, graph.yaml, prompt.md, or tools.py
needed, exactly one caller-side function (`_augment_task_input`) - is to
retrieve relevant `MemoryEntry` lessons before the call and add them to
`task_input` under one extra, namespaced key. Every step of the graph that
already dumps `task_input` into its prompt sees the lessons for free.

Holdout must never write - so it doesn't call retrieve()
-----------------------------------------------------------
`core.memory.retrieve()` is not read-only: it durably bumps every returned
entry's `times_retrieved` via its own internal `write()` call (see that
module's docstring). Holdout scoring must stay a pure read of "what has
memory learned so far" with zero side effects - it is never a source of
new lessons, confidence changes, or even retrieval-count bookkeeping, so
that a later run's holdout results can never be traced back to something
holdout itself did. `_peek_retrieve()` below reimplements `retrieve()`'s
exact ranking (same private helpers, same tie-breaks) without the
write-back, so holdout tasks still see accumulated memory but the module
invariant "holdout never calls core.memory.write()" holds by construction,
not by convention.

Reinforcement wiring
---------------------
Neither `core/memory.py` nor `core/reflect.py` calls `reinforce()`
themselves (both modules say so in their own docstrings/PRs). This harness
is the caller that closes the loop: after a train task finishes, every
memory entry that was retrieved for it gets `reinforce(helped=<task
passed>)` - a crude but directionally-correct proxy for "did this lesson
correlate with success," which is exactly the "confidence moves based on
outcome" mechanic TASK 11 introduced but nothing yet drove.

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
`--memory off` skips retrieval and reflection outright - it never calls
`core.memory.retrieve`, `.write`, or `.reinforce`, and never augments
`task_input`. The agent still runs the identical train and holdout tasks
every pass (for cost/latency parity with the `on` arm), through the
identical unmodified code. If the agent's own `llm_call` steps go through
`core.llm.complete()` (real provider calls, not this file's own mocked
tests), those calls are cached on disk keyed by `(model, messages,
params)`; since `task_input` never changes pass to pass under `--memory
off`, every prompt after pass 1 is a byte-identical cache hit, and the
holdout ScoreCard is expected to be *exactly* flat rather than "flat with
some noise." That is the correct, expected behavior of this harness, not
a bug to chase - it's documented here so a bit-identical control curve
isn't mistaken for the harness failing to actually run anything.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from core import llm, memory, reflect
from core.memory import MemoryEntry
from core.runner import _load_agent_run_fn, _load_dataset, _load_scorer
from core.scorer import score_runs
from core.types import RunResult
from evals.run_eval import load_task_spec

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("ledger/learning_curves")
DEFAULT_MEMORY_SCRATCH_ROOT = Path("memory/_learning_curve")

# Key added to a copy of task_input to carry retrieved lessons into the
# agent's run - see module docstring's "Injecting memory" section. Kept
# off the real task_input dict entirely when there's nothing to inject
# (memory=off, or no entries matched), so a memory-off run's task_input
# is byte-identical every pass (see the caching note above).
_MEMORY_CONTEXT_KEY = "retrieved_memory_lessons"


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


def _augment_task_input(task_input: dict[str, Any], retrieved: list[MemoryEntry]) -> dict[str, Any]:
    """Copy `task_input`, adding retrieved lessons under a namespaced key.

    Never mutates the caller's dict. Returns `task_input` itself, unchanged,
    when there's nothing to add - see the caching note in the module
    docstring for why that matters to the control arm.
    """
    if not retrieved:
        return task_input
    augmented = dict(task_input)
    augmented[_MEMORY_CONTEXT_KEY] = [
        {"trigger": entry.trigger, "content": entry.content, "confidence": entry.confidence}
        for entry in retrieved
    ]
    return augmented


def _peek_retrieve(task_input: dict[str, Any], domain_id: str, k: int, base_dir: str | Path) -> list[MemoryEntry]:
    """Read-only equivalent of `core.memory.retrieve()`.

    Same ranking (identical private helpers, identical tie-break order)
    but never calls `memory.write()` to bump `times_retrieved` - see module
    docstring's "Holdout must never write" section. This is the only thing
    standing between holdout scoring and a hidden write path.
    """
    active = memory.load_all(domain_id, base_dir, statuses={"active"})
    input_tokens = memory._task_input_tokens(task_input)

    scored = [
        (memory._relevance(entry.trigger, input_tokens) * entry.confidence, entry)
        for entry in active
    ]
    scored = [(score, entry) for score, entry in scored if score > 0]
    scored.sort(key=lambda pair: (-pair[0], -pair[1].confidence, pair[1].id))
    return [entry for _, entry in scored[:k]]


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
) -> tuple[RunResult, list[MemoryEntry], dict[str, Any], Any]:
    """Run and score one dataset row, injecting memory first if enabled.

    `read_only=True` (holdout) uses `_peek_retrieve`; `read_only=False`
    (train) uses the real `memory.retrieve()`, which durably bumps
    `times_retrieved`. `memory_mode="off"` skips retrieval entirely in
    either case - no `core.memory` call is made at all, not even a
    read-only one.
    """
    task_id = task.get("task_id", "")
    task_input = task.get("input", {})
    expected = task.get("expected")

    retrieved: list[MemoryEntry] = []
    if memory_mode == "on":
        retrieved = (
            _peek_retrieve(task_input, domain_id, k, base_dir)
            if read_only
            else memory.retrieve(task_input, domain_id, k, base_dir=base_dir)
        )

    augmented_input = _augment_task_input(task_input, retrieved)

    spend_before = llm._cumulative_spend_usd
    start = monotonic()
    output: Any = None
    error: str | None = None
    passed = False
    try:
        output = run_fn(augmented_input)
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
    return result, retrieved, task_input, expected


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
    when memory is off, since retrieval never runs at all).
    """
    retrieved_counts: list[int] = []

    for task in train_tasks:
        result, retrieved, task_input, expected = _run_one_task(
            run_fn,
            scorer_fn,
            task,
            domain_id=domain_id,
            memory_mode=memory_mode,
            k=k,
            base_dir=base_dir,
            read_only=False,
        )
        retrieved_counts.append(len(retrieved))

        if memory_mode != "on":
            continue

        new_entries = reflect.reflect(
            result,
            task_input,
            expected,
            None,
            retrieved,
            domain_id=domain_id,
            gen_n=gen_n,
            model=reflect_model,
        )
        for entry in new_entries:
            memory.write(entry, base_dir=base_dir)

        for entry in retrieved:
            try:
                memory.reinforce(entry.id, domain_id, helped=result.passed, base_dir=base_dir)
            except KeyError:
                logger.warning(
                    "reinforce(): entry %r for domain %r no longer exists", entry.id, domain_id
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
) -> list[dict[str, Any]]:
    """Run the full N-pass learning curve for one (domain, memory_mode).

    Writes one JSON record per pass to
    `{output_dir}/{domain}_memory_{on|off}.jsonl` (truncated at the start
    of this call - each invocation produces a complete, self-contained
    curve, not an append to a previous one) and returns the same records.
    """
    task_spec = load_task_spec(domain, domains_root)
    run_fn = _load_agent_run_fn(agent_path)
    scorer_fn = _load_scorer(task_spec.scorer_id)
    train_tasks = _load_dataset(task_spec.dataset_path, "train", task_spec.max_tasks)
    holdout_tasks = _load_dataset(task_spec.dataset_path, "holdout", task_spec.max_tasks)

    base_dir = _memory_base_dir_for(domain, memory_mode, memory_base_dir)
    if reset_memory:
        _reset_memory_store(domain, base_dir)

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
    )
    for record in records:
        print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
