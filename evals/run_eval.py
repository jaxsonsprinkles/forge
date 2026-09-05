"""CLI entrypoint: run an agent against a domain's eval split, print a ScoreCard as JSON.

    python evals/run_eval.py --domain X --agent PATH --split train|holdout

A domain is looked up by convention: `<domains_root>/<domain>/task_spec.json`,
a JSON object with the same fields as `core.types.TaskSpec`. This is what CI
calls, so stdout carries only the ScoreCard JSON; everything else (warnings,
tracing fallbacks) goes to stderr via the standard `logging` module.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from core.runner import run_agent
from core.scorer import score_runs
from core.types import TaskSpec

logger = logging.getLogger(__name__)


def load_task_spec(domain_id: str, domains_root: str | Path = "domains") -> TaskSpec:
    """Load a domain's TaskSpec from `<domains_root>/<domain_id>/task_spec.json`."""
    spec_path = Path(domains_root) / domain_id / "task_spec.json"
    with spec_path.open() as f:
        data = json.load(f)
    data.setdefault("domain_id", domain_id)
    return TaskSpec(**data)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an agent against a domain's eval split.")
    parser.add_argument("--domain", required=True, help="Domain id, e.g. 'coderepair'.")
    parser.add_argument("--agent", required=True, help="Path to the agent directory (contains run.py).")
    parser.add_argument("--split", required=True, choices=["train", "holdout"])
    parser.add_argument(
        "--domains-root",
        default="domains",
        help="Root directory domains are looked up under (default: 'domains').",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    args = build_arg_parser().parse_args(argv)

    task_spec = load_task_spec(args.domain, args.domains_root)
    results = run_agent(args.agent, task_spec, args.split)
    score_card = score_runs(results, args.split)

    print(json.dumps(dataclasses.asdict(score_card)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
