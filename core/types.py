"""Shared data types passed between Forge's core modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class TaskSpec:
    """A domain's goal, tools, dataset, and scorer, plus an eval budget."""

    domain_id: str
    goal: str
    tools: list[str]
    dataset_path: str
    scorer_id: str
    max_tasks: int = 20


@dataclass
class RunResult:
    """The outcome of running the agent on a single task."""

    task_id: str
    output: Any
    passed: bool
    error: str | None
    cost_usd: float
    latency_ms: int
    trace_id: str | None


@dataclass
class ScoreCard:
    """Aggregate metrics for a set of RunResults on one split."""

    accuracy: float
    reliability: float
    cost_per_task: float
    p50_latency_ms: int
    n: int
    split: str


@dataclass
class FailureCluster:
    """A group of similar task failures with a proposed explanation."""

    label: str
    count: int
    example_task_ids: list[str]
    trace_ids: list[str]
    hypothesis: str


@dataclass
class Mutation:
    """A single proposed change to one mutation surface of the agent."""

    id: str
    surface: Literal["prompt", "tool", "memory", "orchestration"]
    rationale: str
    instruction: str
    target_files: list[str]


@dataclass
class Generation:
    """One generation of the improvement loop: mutations tried and their results."""

    gen_n: int
    parent_sha: str
    scores_before: dict[str, ScoreCard]
    mutations: list[Mutation]
    results: dict[str, dict[str, ScoreCard]]
    winner_id: str | None
    winner_sha: str | None
