"""get_issue tool: fetch one GitHub issue's triage-time content.

Wraps `GET /repos/{repo}/issues/{issue_number}`. Deliberately strips
labels, assignee(s), state, closed_at, and state_reason from the raw API
response: this tool represents what a triager sees when an issue first
lands (title, body, reporter, created_at, comment count), not its
eventual resolution.

That's what keeps this domain from being solvable by simply reading the
answer back off the API - the labels/assignee/priority this domain
scores against (see dataset.jsonl's `expected`) are exactly the fields
this tool omits. Historical, already-resolved data on OTHER issues (via
search_issues, get_contributor_activity) is what actually carries a
repo's unwritten conventions, so only the incoming issue's own outcome
is hidden here.
"""

from __future__ import annotations

from typing import Any

from ._cache import cached_get


def get_issue(repo: str, issue_number: int) -> dict[str, Any]:
    """Return `issue_number`'s title/body/reporter/created_at/comment count.

    Never includes labels, assignee(s), state, or closed_at - see module
    docstring.
    """
    raw = cached_get(f"/repos/{repo}/issues/{issue_number}")
    user = raw.get("user") or {}
    return {
        "repo": repo,
        "number": raw.get("number"),
        "title": raw.get("title"),
        "body": raw.get("body"),
        "reporter": user.get("login"),
        "created_at": raw.get("created_at"),
        "comments": raw.get("comments"),
    }
