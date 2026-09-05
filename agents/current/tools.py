"""Tool implementations available to this agent's graph.yaml steps.

Each tool is a plain function; TOOLS maps the name graph.yaml (or a
future tool_call step) references to the callable. This is the "tool
mutations" surface (see AGENTS.md) - keep it a small number of real,
working tools rather than many stubs.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

RUN_PYTHON_TIMEOUT_SECONDS = 5


def run_python(code: str) -> dict[str, Any]:
    """Run a Python snippet in a subprocess and report what happened.

    Never raises: a syntax error, exception, infinite loop, or non-zero
    exit is reported in the returned dict rather than propagated, since a
    tool's failure must never crash the agent run (see run.py's caller).
    Runs in its own subprocess, never exec()-ed in-process, so a broken or
    malicious snippet can't affect this process - the same approach
    domains/coderepair/scorer.py uses to run candidate fixes.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = Path(tmp_dir) / "snippet.py"
            script_path.write_text(code)
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=RUN_PYTHON_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                return {
                    "timed_out": True,
                    "returncode": None,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                }
            return {
                "timed_out": False,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
    except Exception as exc:  # noqa: BLE001 - a tool's crash must never kill the run
        return {
            "timed_out": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def search_text(text: str, query: str, max_results: int = 5) -> list[str]:
    """Return up to `max_results` lines from `text` containing `query` (case-insensitive).

    Useful for grounding an answer in a supplied document/corpus without
    the model having to re-read the whole thing verbatim.
    """
    if not query:
        return []
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return [line for line in text.splitlines() if pattern.search(line)][:max_results]


TOOLS: dict[str, Any] = {
    "run_python": run_python,
    "search_text": search_text,
}
