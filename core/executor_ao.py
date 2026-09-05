"""Dispatches a generation's Mutations to AO worker sessions, in parallel.

`dispatch(mutations, parent_sha) -> dict[mutation_id, branch]` gives each
Mutation (see core/types.py / core/proposer.py) its own git branch off
`parent_sha` and its own AO worker session (via the `ao` CLI) to carry
out `mutation.instruction`, then waits for every session to finish and
returns the branches that actually produced a new commit.

SPIKE FINDINGS (about 30 of the allotted 90 minutes; see the PR
description for the full narrative and transcript). AO sessions ARE
programmatically launchable and pollable from a plain Python subprocess,
with no human or watching agent in the loop:

  - `ao spawn --project <p> --name <n> --branch <b> --prompt <text>`
    creates a worker in a fresh git worktree and returns immediately
    (the agent then runs in the background). Exit code is 0 on success;
    non-zero with a parseable error line on failure (confirmed by
    spawning against a nonexistent project: exit 1,
    "Unknown project (PROJECT_NOT_FOUND) [...]" on stderr). On success,
    stdout is one line: "spawned session <id> (idle) [...]" - the
    session id is pulled out with a regex.
  - Passing `--branch <name>` for a branch that already exists (created
    beforehand with plain `git branch -f <name> <sha>`) makes AO check
    that branch out as-is, HEAD and all - confirmed by branching off an
    old commit several merges behind main and spawning against it: the
    resulting worktree's `git log` showed that old commit, not main's
    tip. This is what lets `dispatch()` fan mutations out from an
    arbitrary `parent_sha` instead of always off the project's default
    branch. Because git worktrees share one object store, a branch
    created from *this* worktree is immediately visible to `ao spawn`
    regardless of which worktree runs the git command (confirmed
    directly).
  - `ao session get <id> --json` returns structured JSON with a
    `status` field (`working`, `idle`, `merged`, `terminated`, ...).
    Plain polling in a loop is enough to detect completion - no
    interactive/attached harness is required.
  - This project's AO config (`ao project get forge --json`) sets
    `agentConfig.permissions: "bypass-permissions"`, so a spawned worker
    doesn't stall on a tool-permission prompt nobody is present to
    answer. Without that setting, "idle" would be ambiguous between
    "finished" and "stuck waiting on a prompt."
  - Caveat this module defends against: `idle` only means "the agent
    stopped producing activity." It does NOT mean the mutation
    succeeded, or even that a commit happened (e.g. the agent could get
    confused, ask a question into the void, and go idle without ever
    touching the branch). So `dispatch()` never trusts status alone -
    once a session goes idle (or times out) it reads the branch's
    actual `git rev-parse` and only reports it as a result if the HEAD
    moved past `parent_sha`.
  - `ao session kill <id>` tears its worktree down; `dispatch()` kills
    every session it spawned once that session's outcome (success,
    no-op, or timeout) has been recorded.

Conclusion: the AO path was viable well inside the 90-minute budget, so
this module implements it directly - no core/executor_local.py fallback
was needed for this task.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from core.types import Mutation

logger = logging.getLogger(__name__)

_MUTATION_WORKER_PROMPT_PATH = Path(__file__).parent / "prompts" / "mutation_worker.md"

_SPAWNED_SESSION_RE = re.compile(r"spawned session (\S+)")

# Session statuses that mean "this worker has stopped acting," whether or
# not it actually accomplished anything. See module docstring: this is
# read as "the mutation attempt is over," never as "it succeeded."
_TERMINAL_STATUSES = {"idle", "merged", "terminated"}

_DEFAULT_POLL_INTERVAL_S = 5.0
_DEFAULT_TIMEOUT_S = 1800.0  # 30 minutes per mutation before giving up on it


def _branch_name(mutation_id: str, parent_sha: str) -> str:
    return f"forge/{parent_sha[:8]}/{mutation_id}"


def _mutation_prompt(mutation: Mutation) -> str:
    """Build the prompt handed to a mutation worker session.

    Prepends the shared instructions in core/prompts/mutation_worker.md
    (framing, and the "don't hardcode eval answers" constraint) to this
    mutation's specific surface/target_files/rationale/instruction.
    """
    base_prompt = _MUTATION_WORKER_PROMPT_PATH.read_text()
    files = ", ".join(mutation.target_files)
    return (
        f"{base_prompt}\n\n"
        f"Surface: {mutation.surface}\n"
        f"Target file(s): {files}\n\n"
        f"Rationale: {mutation.rationale}\n\n"
        f"Instruction: {mutation.instruction}"
    )


def _create_branch(branch: str, parent_sha: str) -> None:
    subprocess.run(
        ["git", "branch", "-f", branch, parent_sha],
        capture_output=True,
        text=True,
        check=True,
    )


def _spawn_worker(project: str, name: str, branch: str, prompt: str) -> str:
    """Spawn an AO worker session and return its session id.

    Raises RuntimeError if `ao spawn` exits non-zero or its stdout doesn't
    contain a parseable session id.
    """
    result = subprocess.run(
        [
            "ao",
            "spawn",
            "--project",
            project,
            "--name",
            name,
            "--branch",
            branch,
            "--prompt",
            prompt,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ao spawn failed for branch {branch!r}: {result.stderr.strip() or result.stdout.strip()}"
        )
    match = _SPAWNED_SESSION_RE.search(result.stdout)
    if not match:
        raise RuntimeError(f"could not parse a session id from `ao spawn` output: {result.stdout!r}")
    return match.group(1)


def _session_status(session_id: str, project: str) -> str | None:
    """Best-effort `ao session get --json` status read; None on any failure."""
    result = subprocess.run(
        ["ao", "session", "get", session_id, "--json", "--project", project],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return data.get("session", {}).get("status")


def _kill_session(session_id: str, project: str) -> None:
    subprocess.run(
        ["ao", "session", "kill", session_id, "--project", project],
        capture_output=True,
        text=True,
    )


def _branch_head_sha(branch: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", branch],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def dispatch(
    mutations: list[Mutation],
    parent_sha: str,
    *,
    project: str | None = None,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, str]:
    """Fan `mutations` out to one AO worker session each, off `parent_sha`.

    Every mutation gets its own branch (named from `parent_sha` and the
    mutation id, so re-running a generation is idempotent) and its own AO
    worker session; all sessions are spawned before any polling starts,
    so they run concurrently. `project` defaults to the `AO_PROJECT_ID`
    environment variable (set for any code running inside an AO session).

    Returns `{mutation.id: branch}` only for mutations whose session
    produced a new commit on their branch; a mutation that failed to
    spawn, timed out, or went idle without committing is omitted -
    callers should treat a missing id as "no candidate from this
    mutation," not as an error.
    """
    resolved_project = project or os.environ.get("AO_PROJECT_ID")
    if not resolved_project:
        raise RuntimeError("dispatch() needs an AO project id (pass project= or set AO_PROJECT_ID)")

    pending: dict[str, tuple[str, str]] = {}  # mutation.id -> (session_id, branch)
    for mutation in mutations:
        branch = _branch_name(mutation.id, parent_sha)
        try:
            _create_branch(branch, parent_sha)
            session_id = _spawn_worker(
                project=resolved_project,
                name=mutation.id[:20],
                branch=branch,
                prompt=_mutation_prompt(mutation),
            )
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            logger.warning("failed to dispatch mutation %s: %s", mutation.id, exc)
            continue
        pending[mutation.id] = (session_id, branch)

    results: dict[str, str] = {}
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        finished = [
            mutation_id
            for mutation_id, (session_id, _branch) in pending.items()
            if _session_status(session_id, resolved_project) in _TERMINAL_STATUSES
        ]
        for mutation_id in finished:
            session_id, branch = pending.pop(mutation_id)
            head = _branch_head_sha(branch)
            if head and head != parent_sha:
                results[mutation_id] = branch
            else:
                logger.warning(
                    "mutation %s session %s went idle with no new commit on %s",
                    mutation_id,
                    session_id,
                    branch,
                )
            _kill_session(session_id, resolved_project)
        if pending:
            time.sleep(poll_interval_s)

    for mutation_id, (session_id, branch) in pending.items():
        logger.warning(
            "mutation %s session %s timed out after %.0fs", mutation_id, session_id, timeout_s
        )
        _kill_session(session_id, resolved_project)

    return results
