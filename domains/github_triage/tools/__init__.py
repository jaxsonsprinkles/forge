"""GitHub REST API tool functions for the github_triage domain.

Each function wraps one real GitHub REST endpoint and routes its request
through `_cache.cached_get` (see _cache.py for the record/replay
contract that makes every call reproducible offline). TOOLS mirrors the
{name: callable} convention agents/current/tools.py uses for its own
tool table.
"""

from __future__ import annotations

from typing import Any

from .get_contributor_activity import get_contributor_activity
from .get_file_owners import get_file_owners
from .get_issue import get_issue
from .list_labels import list_labels
from .search_issues import search_issues

TOOLS: dict[str, Any] = {
    "search_issues": search_issues,
    "get_issue": get_issue,
    "list_labels": list_labels,
    "get_contributor_activity": get_contributor_activity,
    "get_file_owners": get_file_owners,
}

__all__ = [
    "TOOLS",
    "get_contributor_activity",
    "get_file_owners",
    "get_issue",
    "list_labels",
    "search_issues",
]
