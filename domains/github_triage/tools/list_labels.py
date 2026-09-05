"""list_labels tool: list a repo's defined labels (name/description/color).

Wraps `GET /repos/{repo}/labels`. This is API-schema-adjacent (it says
which label *words* a repo has defined), not a learned convention by
itself - which of these labels a team actually reaches for, and for
what, only shows up in search_issues' historical usage. Still a useful
starting vocabulary an agent can narrow down with real examples.
"""

from __future__ import annotations

from typing import Any

from ._cache import cached_get


def list_labels(repo: str, per_page: int = 100) -> list[dict[str, Any]]:
    """Return every label defined on `repo` as {"name", "description", "color"}."""
    raw = cached_get(f"/repos/{repo}/labels", {"per_page": per_page})
    items = raw if isinstance(raw, list) else []
    return [
        {"name": label.get("name"), "description": label.get("description"), "color": label.get("color")}
        for label in items
    ]
