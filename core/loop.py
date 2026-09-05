"""Runs Forge's improvement loop: score -> analyze -> propose -> dispatch ->
gate -> record, once per generation, for a caller-chosen number of
generations.

`run_loop(task_specs, generations)` is the entry point. Each generation:

  1. score: `score_fn(parent_sha, task_specs)` evaluates the agent as it
     exists at `parent_sha` against every TaskSpec's train split (see
     core.runner.run_agent / core.scorer.score_runs), returning both the
     ScoreCards (the generation's baseline) and the raw RunResults (fed to
     analyze).
  2. analyze: `analyze_fn` (core.analyzer.analyze) clusters each domain's
     failing RunResults into FailureClusters.
  3. propose: `propose_fn` (core.proposer.propose) turns the combined
     clusters into up to `n_mutations` Mutations. No clusters (or no
     mutations) ends the generation early as a no-op record - nothing to
     dispatch, nothing to gate, `winner_id`/`winner_sha` stay `None`.
  4. dispatch: `dispatch_fn` (core.executor_ao.dispatch) fans the
     mutations out to their own branch off `parent_sha`, returning
     `{mutation_id: branch}` for whichever ones produced a commit.
  5. gate: each candidate branch is re-scored with `score_candidate_fn`
     (same shape as `score_fn`) and handed to `select_fn` (core.gate.select)
     along with the generation's baseline, to pick a winner or `None`.
  6. record: the generation (see core.types.Generation) is appended to the
     ledger via core.ledger.append_generation - durably, before the next
     generation starts (see core/ledger.py).

Every step function is a keyword parameter with a real default, so a
caller never has to override anything to run the loop for real, and tests
can swap in fast, offline stand-ins for all five without monkeypatching.

RESUMABILITY: at the start of `run_loop`, the ledger is read back (via
`core.ledger.load_generations`) to find the last recorded generation.
Already-recorded generations are never redone. The next generation's
`parent_sha` is the last recorded generation's `winner_sha` if it has one,
else that generation's own `parent_sha` (i.e. a generation that produced
no winner leaves the lineage's parent unchanged for the next attempt).
With no generations recorded yet, `parent_sha` is `starting_sha` (or, if
that's not given, the calling repo's current `HEAD`).

SPEND CAP: `core.llm`'s `FORGE_MAX_SPEND_USD` ceiling is a *cumulative,
per-process* limit (see core/llm.py). This loop turns it into a
*per-generation* cap by calling `llm.reset_spend_tracker()` at the start
of every generation - so each generation gets a fresh budget against the
same env var, rather than sharing one budget across the whole run. This
only guards spend made directly by this process (i.e. by `score_fn`/
`score_candidate_fn` calling into an agent that calls `core.llm.complete`);
it has no visibility into a dispatched AO worker's own spend, which is a
separate process. If a generation's work raises `llm.SpendCeilingExceeded`,
the loop logs a warning, does NOT append a partial/incomplete record for
that generation (the ledger file is never touched for it), and returns
whatever generations completed before it - cleanly, without raising.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from core import analyzer, executor_ao, gate, ledger, llm, proposer
from core.runner import run_agent
from core.scorer import score_runs
from core.types import FailureCluster, Generation, Mutation, RunResult, ScoreCard, TaskSpec

logger = logging.getLogger(__name__)

DEFAULT_AGENT_SUBPATH = "agents/current"

ScoreFn = Callable[[str, list[TaskSpec]], tuple[dict[str, "ScoreCard"], dict[str, list[RunResult]]]]


def _git_head_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


@contextmanager
def _worktree(ref: str) -> Iterator[Path]:
    """Check `ref` out into a scratch git worktree, cleaned up on exit."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="forge-eval-"))
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", "--force", str(tmp_dir), ref],
            capture_output=True,
            text=True,
            check=True,
        )
        yield tmp_dir
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(tmp_dir)], capture_output=True, text=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _evaluate_agent_at(
    ref: str,
    task_specs: list[TaskSpec],
    agent_subpath: str = DEFAULT_AGENT_SUBPATH,
) -> tuple[dict[str, ScoreCard], dict[str, list[RunResult]]]:
    """Default `score_fn`/`score_candidate_fn`.

    Checks `ref` (a sha or branch name) out into a scratch git worktree
    and runs+scores the agent found at `<worktree>/<agent_subpath>` against
    every TaskSpec's train split, so evaluating a candidate branch never
    disturbs this process's own working tree or requires it to already be
    checked out anywhere.
    """
    scores: dict[str, ScoreCard] = {}
    results: dict[str, list[RunResult]] = {}
    with _worktree(ref) as workdir:
        agent_path = workdir / agent_subpath
        for task_spec in task_specs:
            run_results = run_agent(str(agent_path), task_spec, "train")
            results[task_spec.domain_id] = run_results
            scores[task_spec.domain_id] = score_runs(run_results, "train")
    return scores, results


def _resume_state(recorded: list[Generation], starting_sha: str | None) -> tuple[int, str]:
    """Return (next_gen_n, parent_ref) to resume from, given what's recorded."""
    if not recorded:
        return 1, starting_sha or _git_head_sha()
    last = recorded[-1]
    return last.gen_n + 1, (last.winner_sha or last.parent_sha)


def _run_one_generation(
    *,
    gen_n: int,
    parent_sha: str,
    task_specs: list[TaskSpec],
    project: str | None,
    n_mutations: int,
    watchlist_path: str | Path,
    archive_root: str | Path,
    score_fn: ScoreFn,
    analyze_fn: Callable[[list[RunResult], str], list[FailureCluster]],
    propose_fn: Callable[[list[FailureCluster], str, int], list[Mutation]],
    dispatch_fn: Callable[..., dict[str, str]],
    select_fn: Callable[..., str | None],
    score_candidate_fn: ScoreFn,
) -> Generation:
    scores_before, results_before = score_fn(parent_sha, task_specs)

    clusters: list[FailureCluster] = []
    for task_spec in task_specs:
        clusters.extend(analyze_fn(results_before.get(task_spec.domain_id, []), DEFAULT_AGENT_SUBPATH))

    mutations: list[Mutation] = propose_fn(clusters, DEFAULT_AGENT_SUBPATH, n_mutations) if clusters else []

    if not mutations:
        return Generation(
            gen_n=gen_n,
            parent_sha=parent_sha,
            scores_before=scores_before,
            mutations=[],
            results={},
            winner_id=None,
            winner_sha=None,
        )

    branches = dispatch_fn(mutations, parent_sha, project=project)

    candidates: dict[str, dict[str, ScoreCard]] = {}
    candidate_results: dict[str, list[RunResult]] = {}
    for mutation in mutations:
        branch = branches.get(mutation.id)
        if branch is None:
            continue
        cand_scores, cand_results = score_candidate_fn(branch, task_specs)
        candidates[mutation.id] = cand_scores
        candidate_results[mutation.id] = [r for results in cand_results.values() for r in results]

    winner_id: str | None = None
    if candidates:
        winner_id = select_fn(
            candidates,
            scores_before,
            candidate_results=candidate_results,
            watchlist_path=watchlist_path,
            candidate_refs=branches,
            archive_root=archive_root,
        )

    winner_sha = branches.get(winner_id) if winner_id is not None else None

    return Generation(
        gen_n=gen_n,
        parent_sha=parent_sha,
        scores_before=scores_before,
        mutations=mutations,
        results=candidates,
        winner_id=winner_id,
        winner_sha=winner_sha,
    )


def run_loop(
    task_specs: list[TaskSpec],
    generations: int,
    *,
    starting_sha: str | None = None,
    project: str | None = None,
    n_mutations: int = 4,
    ledger_path: str | Path = ledger.DEFAULT_LEDGER_PATH,
    watchlist_path: str | Path = gate.DEFAULT_WATCHLIST_PATH,
    archive_root: str | Path = gate.DEFAULT_ARCHIVE_ROOT,
    score_fn: ScoreFn = _evaluate_agent_at,
    analyze_fn: Callable[[list[RunResult], str], list[FailureCluster]] = analyzer.analyze,
    propose_fn: Callable[[list[FailureCluster], str, int], list[Mutation]] = proposer.propose,
    dispatch_fn: Callable[..., dict[str, str]] = executor_ao.dispatch,
    select_fn: Callable[..., str | None] = gate.select,
    score_candidate_fn: ScoreFn | None = None,
) -> list[Generation]:
    """Run the improvement loop up to `generations` total recorded generations.

    Resumable: generations already present in `ledger_path` are not redone
    (see module docstring). Returns the full list of Generation records
    (previously recorded plus newly run) in gen_n order. If `generations`
    are already recorded, returns immediately without doing any work.

    `score_candidate_fn` defaults to `score_fn` itself (both have the same
    `(ref, task_specs) -> (scores, results)` shape; a caller only needs to
    pass one override if candidates should be evaluated differently from
    the generation's own baseline, e.g. a smaller eval budget).
    """
    score_candidate_fn = score_candidate_fn or score_fn

    recorded = ledger.load_generations(ledger_path)
    start_gen_n, current_ref = _resume_state(recorded, starting_sha)

    remaining = generations - len(recorded)
    if remaining <= 0:
        return recorded

    new_generations: list[Generation] = []
    for offset in range(remaining):
        gen_n = start_gen_n + offset
        llm.reset_spend_tracker()
        try:
            gen = _run_one_generation(
                gen_n=gen_n,
                parent_sha=current_ref,
                task_specs=task_specs,
                project=project,
                n_mutations=n_mutations,
                watchlist_path=watchlist_path,
                archive_root=archive_root,
                score_fn=score_fn,
                analyze_fn=analyze_fn,
                propose_fn=propose_fn,
                dispatch_fn=dispatch_fn,
                select_fn=select_fn,
                score_candidate_fn=score_candidate_fn,
            )
        except llm.SpendCeilingExceeded as exc:
            logger.warning(
                "generation %d aborted: per-generation spend cap exceeded (%s); "
                "not recording a partial generation, stopping the loop",
                gen_n,
                exc,
            )
            break

        ledger.append_generation(gen, ledger_path)
        new_generations.append(gen)
        current_ref = gen.winner_sha or current_ref

    return recorded + new_generations
