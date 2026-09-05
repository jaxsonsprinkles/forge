PROJECT CONTEXT

What this repo is
Forge is a meta-system that builds and improves AI agents automatically. Given a task spec (a goal, a set of tools, and a scorable dataset), Forge writes an agent, runs it against test tasks, analyzes why it failed, proposes changes, tests those changes in parallel, and keeps whichever change measurably helped. It repeats this for many generations.

Two levels, never confuse them
Forge is the meta-system. It lives in core/. It is normal software; it is not being optimized.
The agent is what Forge builds and improves. It lives in agents/current/ as plain files. It IS the artifact under optimization.
When a task says "the agent," it means the thing in agents/current/, never Forge itself.

The agent must be files, not code objects
agents/current/ contains exactly five files, one per mutation surface:
agents/current/
    prompt.md     # system instructions        (prompt mutations)
    tools.py      # tool implementations       (tool mutations)
    memory.py     # what persists across steps (memory mutations)
    graph.yaml    # step order and control flow (orchestration mutations)
    run.py        # entrypoint: run(task_input: dict) -> dict
Never refactor the agent into a class, a package, or a config object. Coding agents must be able to edit these as ordinary files, and git must record each change as a diff.

Rules for every task
Python 3.10-3.13 only. Neatlogs requires this range.
Scorers must be deterministic and offline. No network calls, no LLM-as-judge. Same input always produces the same score.
All model calls go through core/llm.py. Nothing else may call a model provider directly.
Every model call is cached to disk, keyed by a hash of (model, messages, params). Re-running unchanged code must cost nothing.
Respect max_tasks in a TaskSpec. Never evaluate more tasks than it specifies.
Type-hint public functions. Docstrings on anything in core/.
Do not modify files outside the ones your task names.
Commit in small logical units with clear messages.
If a requirement is ambiguous, implement the simplest thing that satisfies the acceptance criteria and note the assumption in your PR description. Do not expand scope.

Definition of done
A task is done when its acceptance criteria pass from a clean checkout, and python -m pytest is green.
