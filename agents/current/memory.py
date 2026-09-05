"""Scratchpad that persists across steps within one run(task_input) call.

A plain dict, not a class, so that memory mutations (see AGENTS.md) are
ordinary diffs to this file rather than changes to a schema/class shape.
"""

from __future__ import annotations

from typing import Any


def create(task_input: dict[str, Any]) -> dict[str, Any]:
    """Build a fresh scratchpad for one run(), seeded with the task input."""
    return {"task_input": task_input, "steps_run": []}


def record(memory: dict[str, Any], key: str, value: Any) -> None:
    """Persist a step's output under `key` and note that the step ran."""
    memory[key] = value
    memory["steps_run"].append(key)
