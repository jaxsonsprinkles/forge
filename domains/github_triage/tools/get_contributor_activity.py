"""get_contributor_activity tool: what has a given user actually closed?

Wraps a search-issues query scoped to `assignee:{username}`, summarizing
which labels appear on issues that user has closed - so an agent can
check "does this person usually work on area X" before suggesting them
as the assignee for a new issue, instead of guessing from a username
alone.
"""

from __future__ import annotations

from typing import Any

from ._cache import cached_get


def get_contributor_activity(repo: str, username: str, per_page: int = 20) -> dict[str, Any]:
    """Summarize `username`'s closed-issue history on `repo`: how many
    issues they've closed, and a count of each label seen across them."""
    q = f"repo:{repo} is:issue is:closed assignee:{username}"
    raw = cached_get("/search/issues", {"q": q, "per_page": per_page})
    items = raw.get("items", []) if isinstance(raw, dict) else []

    label_counts: dict[str, int] = {}
    for item in items:
        for label in item.get("labels", []):
            name = label.get("name") if isinstance(label, dict) else None
            if name:
                label_counts[name] = label_counts.get(name, 0) + 1

    return {
        "username": username,
        "closed_issue_count": len(items),
        "label_counts": label_counts,
    }
