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

from core.types import Mutation


def dispatch(mutations: list[Mutation], parent_sha: str) -> dict[str, str]:
    """Not yet implemented - see the spike findings above and the
    follow-up commit that adds the real AO-backed implementation."""
    raise NotImplementedError
