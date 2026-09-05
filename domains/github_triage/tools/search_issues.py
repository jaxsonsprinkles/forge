"""search_issues tool: search a repo's issues via the GitHub search API.

Wraps `GET /search/issues?q=repo:{repo} is:issue is:{state} {query}`.
Unlike get_issue, this intentionally surfaces full historical outcome
data (labels, assignees, state, close time) on the issues it returns -
searching how *similar past issues* were actually triaged is how an
agent infers a repo's unwritten label vocabulary, priority patterns, and
who-owns-what, without that data ever being available for the one
incoming issue actually being scored (see get_issue.py).
"""

from __future__ import annotations

from typing import Any

from ._cache import cached_get


def search_issues(repo: str, query: str, state: str = "closed", per_page: int = 10) -> list[dict[str, Any]]:
    """Search `repo`'s issues matching `query` (GitHub search qualifiers,
    e.g. "label:bug", free text, etc.), scoped to `state` ("open"/"closed"/"all").

    Returns each match's number, title, labels, assignees, state, and
    close time.
    """
    q = f"repo:{repo} is:issue is:{state} {query}".strip()
    raw = cached_get("/search/issues", {"q": q, "per_page": per_page})
    items = raw.get("items", []) if isinstance(raw, dict) else []
    return [
        {
            "number": item.get("number"),
            "title": item.get("title"),
            "labels": [label.get("name") for label in item.get("labels", []) if isinstance(label, dict)],
            "assignees": [a.get("login") for a in item.get("assignees", []) if isinstance(a, dict)],
            "state": item.get("state"),
            "created_at": item.get("created_at"),
            "closed_at": item.get("closed_at"),
        }
        for item in items
    ]
