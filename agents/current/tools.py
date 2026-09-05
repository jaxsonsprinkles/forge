"""Tool implementations available to this agent's graph.yaml steps.

Each tool is a plain function; TOOLS maps the name graph.yaml (or a
future tool_call step) references to the callable. This is the "tool
mutations" surface (see AGENTS.md) - keep it a small number of real,
working tools rather than many stubs.

The github_* tools below wrap domains/github_triage/tools' real GitHub
REST API functions (offline-reproducible via that package's on-disk
record/replay cache - see its _cache.py). They live here, in the one
tools.py every domain's graph.yaml shares, rather than in a
domain-specific tools.py, because this baseline agent has no per-domain
variant (core.architect.build_agent always copies this exact file - see
its docstring). Each wrapper accepts `**_ignored` and returns a
`{"skipped": ...}` placeholder instead of raising when the fields it
needs (e.g. "repo") aren't present in the caller's args, so a graph.yaml
step calling one of these is harmless noise on a non-github_triage task
rather than a crash.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from domains.github_triage.tools import get_contributor_activity as _github_get_contributor_activity
from domains.github_triage.tools import get_issue as _github_get_issue
from domains.github_triage.tools import list_labels as _github_list_labels
from domains.github_triage.tools import search_issues as _github_search_issues

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


def github_get_issue(repo: str | None = None, issue_number: int | None = None, **_ignored: Any) -> dict[str, Any]:
    """Fetch the incoming issue's title/body/reporter/comments for a github_triage task.

    Never includes labels, assignee, or state - see get_issue's own
    docstring for why (those are exactly what's being scored).
    """
    if repo is None or issue_number is None:
        return {"skipped": "no repo/issue_number available"}
    return _github_get_issue(repo, issue_number)


def github_list_labels(repo: str | None = None, **_ignored: Any) -> Any:
    """Return `repo`'s defined label vocabulary (name/description/color)."""
    if repo is None:
        return {"skipped": "no repo available"}
    return _github_list_labels(repo)


def github_search_similar_issues(repo: str | None = None, **_ignored: Any) -> Any:
    """Return `repo`'s recently closed issues with their labels/assignees -
    the historical pattern an agent can match the incoming issue against."""
    if repo is None:
        return {"skipped": "no repo available"}
    return _github_search_issues(repo, "", state="closed", per_page=10)


def github_get_reporter_activity(repo: str | None = None, reporter: str | None = None, **_ignored: Any) -> Any:
    """Summarize the incoming issue's reporter's own closed-issue history on `repo`."""
    if repo is None or reporter is None:
        return {"skipped": "no repo/reporter available"}
    return _github_get_contributor_activity(repo, reporter)


TOOLS: dict[str, Any] = {
    "run_python": run_python,
    "search_text": search_text,
    "github_get_issue": github_get_issue,
    "github_list_labels": github_list_labels,
    "github_search_similar_issues": github_search_similar_issues,
    "github_get_reporter_activity": github_get_reporter_activity,
}
