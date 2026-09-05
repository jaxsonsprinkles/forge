"""Generates dataset.jsonl, split.json, and cache/ for the github_triage domain.

Not run automatically by tests; documents/reproduces how this domain's
40-issue snapshot of eslint/eslint was recorded and turned into ground
truth. Every request this script makes goes through
`tools._cache.cached_get`/`seed_cache`, so re-running it makes zero new
network calls: it replays the same cache/ files it already wrote.

REPO: eslint/eslint was chosen after probing about a dozen well-known,
active public repos (see PR description) for two properties this domain
needs: (a) issues that actually carry assignees (most popular OSS repos
leave the vast majority of issues unassigned, which would make
"assignee" ungradeable), and (b) a real, opinionated label vocabulary
that only shows up in usage - e.g. "accepted" (a maintainer-decision
label, not a bug/feature-type label), "repro:yes"/"repro:needed" (an
eslint-specific reproduction-tracking convention), and inconsistent
capitalization ("bug" vs "Invalid" vs "Stale") that a naive agent
wouldn't guess from the label list alone. Of ~90 non-PR closed issues in
its 4 most recently-updated pages, 74 (80%) had a real assignee.

SELECTION: the 40 most recently-updated closed, non-pull-request issues
(GitHub's `issues?state=closed&sort=updated&direction=desc` order,
paginated 100/page across 4 pages, non-PR items only, first 40 in that
order). This is a deterministic, reproducible-from-cache rule with no
extra randomness - the recorded API pages are exactly the data this
selection is computed over.

PRIORITY DERIVATION: eslint/eslint has no explicit priority/severity
label, so priority is derived purely from close latency (time from
`created_at` to `closed_at`), on the theory that the team burns down
what it considers urgent fastest and lets low-stakes issues sit:

    <= 1 day   -> "critical"
    <= 7 days  -> "high"
    <= 30 days -> "medium"
    > 30 days  -> "low"

This is a heuristic proxy, not a ground truth the repo itself asserts -
see scorer.py's off-by-one partial credit, which exists specifically
because this label is fuzzier than labels/assignee.

ASSIGNEE: the first login in the issue's `assignees` list, or `null` if
none (an issue with no assignee is a real, correctly-scoreable outcome
for this repo - see scorer.py's None==None handling).

This is a REAL recorded snapshot (not synthetic/fabricated data): every
issue, label, commit, and search result in cache/ came from a live call
to the public GitHub REST API (unauthenticated, read-only), made once by
this script and committed to cache/ so no later run - eval, test, or
CI - ever needs network access again.
"""

from __future__ import annotations

import datetime
import json
import random
from pathlib import Path
from typing import Any

from domains.github_triage.tools._cache import cached_get, seed_cache

REPO = "eslint/eslint"
DOMAIN_DIR = Path(__file__).parent
SEED = 42
N_TRAIN = 26
N_HOLDOUT = 14

# A couple of real rule files, used to record example get_file_owners() calls.
EXAMPLE_FILE_PATHS = ["lib/rules/no-unused-vars.js", "lib/rules/no-undef.js"]
# A couple of real search queries, used to record example search_issues() calls.
EXAMPLE_SEARCH_QUERIES = ["label:bug", "label:enhancement"]


def _priority_from_close_latency(created_at: str, closed_at: str) -> str:
    created = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    closed = datetime.datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
    duration_days = (closed - created).total_seconds() / 86400
    if duration_days <= 1:
        return "critical"
    if duration_days <= 7:
        return "high"
    if duration_days <= 30:
        return "medium"
    return "low"


def _fetch_candidate_issues() -> list[dict[str, Any]]:
    """Fetch the 4 most recent pages of closed issues, replaying cache if present."""
    items: list[dict[str, Any]] = []
    for page in range(1, 5):
        page_items = cached_get(
            f"/repos/{REPO}/issues",
            {"state": "closed", "per_page": 100, "sort": "updated", "direction": "desc", "page": page},
        )
        items.extend(page_items)
    return [item for item in items if "pull_request" not in item]


def _to_expected(issue: dict[str, Any]) -> dict[str, Any]:
    labels = [label["name"] for label in issue.get("labels", [])]
    assignees = issue.get("assignees") or []
    assignee = assignees[0]["login"] if assignees else None
    priority = _priority_from_close_latency(issue["created_at"], issue["closed_at"])
    return {"labels": labels, "assignee": assignee, "priority": priority}


def _record_example_tool_calls(issues: list[dict[str, Any]]) -> None:
    """Record one real example call per tool (besides get_issue/search_issues'
    dataset-driving usage above) so cache/ demonstrates every tool works,
    and tests/test_github_triage.py has real cached data to assert against."""
    cached_get(f"/repos/{REPO}/labels", {"per_page": 100})

    for path in EXAMPLE_FILE_PATHS:
        cached_get(f"/repos/{REPO}/commits", {"path": path, "per_page": 20})

    for query in EXAMPLE_SEARCH_QUERIES:
        q = f"repo:{REPO} is:issue is:closed {query}"
        cached_get("/search/issues", {"q": q, "per_page": 10})

    example_assignees = sorted(
        {issue["assignees"][0]["login"] for issue in issues if issue.get("assignees")}
    )[:3]
    for username in example_assignees:
        q = f"repo:{REPO} is:issue is:closed assignee:{username}"
        cached_get("/search/issues", {"q": q, "per_page": 20})


def main() -> None:
    candidates = _fetch_candidate_issues()
    selected = candidates[:40]
    assert len(selected) == 40, f"expected 40 candidate issues, found {len(selected)}"

    _record_example_tool_calls(selected)

    task_ids = [f"gt_{i:03d}" for i in range(1, 41)]

    random.seed(SEED)
    shuffled = task_ids[:]
    random.shuffle(shuffled)
    train_ids = set(shuffled[:N_TRAIN])
    holdout_ids = set(shuffled[N_TRAIN:])
    assert len(train_ids) == N_TRAIN and len(holdout_ids) == N_HOLDOUT

    rows = []
    for task_id, issue in zip(task_ids, selected):
        # Seed get_issue()'s single-issue cache slot from this same (already
        # recorded) list-issues data, rather than making 40 separate live
        # requests - see _cache.seed_cache's docstring.
        seed_cache(f"/repos/{REPO}/issues/{issue['number']}", None, issue)

        rows.append(
            {
                "task_id": task_id,
                "input": {"repo": REPO, "issue_number": issue["number"]},
                "expected": _to_expected(issue),
                "split": "train" if task_id in train_ids else "holdout",
            }
        )

    with (DOMAIN_DIR / "dataset.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    split = {"train": sorted(train_ids), "holdout": sorted(holdout_ids)}
    with (DOMAIN_DIR / "split.json").open("w") as f:
        json.dump(split, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
