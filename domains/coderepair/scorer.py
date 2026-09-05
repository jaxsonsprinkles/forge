"""Scorer for the coderepair domain.

Dataset format (see dataset.jsonl): each record's `expected["tests"]` is a
string of plain Python `assert` statements that call the function named by
`input["function_name"]`. To score a candidate fix, this module writes the
candidate code followed by the test code to a temp file and runs it as a
standalone script in a subprocess (never `exec()`-ed in-process), so a
crash, infinite loop, or malicious candidate can't affect this process.

score(output, expected) -> (passed, details) where `output` is a string of
candidate Python source (expected to define the target function) and
`expected` is the dict {"tests": "<assert statements>"}. Never raises: any
failure (bad syntax, timeout, non-zero exit, unexpected exception) is
reported as (False, details).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TIMEOUT_SECONDS = 5


def score(output: str, expected: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Run the hidden tests against candidate code in a subprocess.

    Returns (passed, details). `details` always includes `timed_out`,
    `returncode`, `stdout`, and `stderr` (or `error` if the harness itself
    failed before/without running a subprocess, e.g. bad input types).
    """
    try:
        tests = expected["tests"]
        script = f"{output}\n\n{tests}"
    except Exception as exc:  # malformed `output`/`expected` shape
        return False, {
            "timed_out": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": f"failed to build script: {exc!r}",
        }

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = Path(tmp_dir) / "candidate.py"
            script_path.write_text(script)

            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                return False, {
                    "timed_out": True,
                    "returncode": None,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                }

            passed = result.returncode == 0
            return passed, {
                "timed_out": False,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
    except Exception as exc:  # never let scoring itself crash the caller
        return False, {
            "timed_out": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": f"scorer crashed: {exc!r}",
        }


def score_v1(output: str, expected: dict[str, Any]) -> bool:
    """Bool-only wrapper of `score`, for use as a TaskSpec.scorer_id target.

    `core.runner.run_agent` calls `bool(scorer_fn(output, expected))` on
    whatever the resolved scorer_id returns. Since `score` returns a
    `(passed, details)` tuple, a non-empty tuple is always truthy under
    `bool()` regardless of `passed` - wiring `score` itself up as a
    task_spec.json's scorer_id would make every task register as passed.
    This wrapper works around that, mirroring domains/docqa/scorer.py's
    identical fix: a coderepair task_spec.json should point scorer_id at
    "domains.coderepair.scorer:score_v1" instead of at `score` directly.
    """
    return score(output, expected)[0]
