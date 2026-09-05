"""get_file_owners tool: who actually commits to a given file path?

GitHub's REST API has no direct CODEOWNERS/blame-ownership endpoint
usable without extra scopes, so this wraps `GET /repos/{repo}/commits`
scoped to `path` and ranks authors by recent commit frequency - the
closest genuine signal for "who owns this subsystem", useful for
suggesting an assignee based on the file/area an issue is actually about.
"""

from __future__ import annotations

from typing import Any

from ._cache import cached_get


def get_file_owners(repo: str, path: str, per_page: int = 20) -> list[dict[str, Any]]:
    """Rank the most frequent recent committers to `path` in `repo`."""
    raw = cached_get(f"/repos/{repo}/commits", {"path": path, "per_page": per_page})
    items = raw if isinstance(raw, list) else []

    counts: dict[str, int] = {}
    for item in items:
        author = (item.get("author") or {}).get("login") if isinstance(item, dict) else None
        if author:
            counts[author] = counts.get(author, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"username": login, "commit_count": count} for login, count in ranked]
