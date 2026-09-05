import json
from pathlib import Path

import pytest

from domains.github_triage.scorer import PRIORITY_LEVELS, score
from domains.github_triage.tools import TOOLS, get_contributor_activity, get_issue, list_labels, search_issues
from domains.github_triage.tools import _cache
from evals.run_eval import load_task_spec

DOMAIN_DIR = Path("domains/github_triage")


def _load_dataset() -> dict[str, dict]:
    tasks = {}
    with (DOMAIN_DIR / "dataset.jsonl").open() as f:
        for line in f:
            record = json.loads(line)
            tasks[record["task_id"]] = record
    return tasks


# --- dataset / split shape ---------------------------------------------------


def test_dataset_has_40_records_with_expected_shape():
    tasks = _load_dataset()
    assert len(tasks) == 40
    for task_id, record in tasks.items():
        assert record["task_id"] == task_id
        assert "repo" in record["input"]
        assert "issue_number" in record["input"]
        expected = record["expected"]
        assert isinstance(expected["labels"], list)
        assert expected["assignee"] is None or isinstance(expected["assignee"], str)
        assert expected["priority"] in PRIORITY_LEVELS
        assert record["split"] in ("train", "holdout")


def test_split_covers_dataset_without_overlap():
    tasks = _load_dataset()
    with (DOMAIN_DIR / "split.json").open() as f:
        split = json.load(f)

    train, holdout = set(split["train"]), set(split["holdout"])
    assert len(split["train"]) == 26
    assert len(split["holdout"]) == 14
    assert train.isdisjoint(holdout)
    assert train | holdout == set(tasks)

    for task_id, record in tasks.items():
        expected_split = "train" if task_id in train else "holdout"
        assert record["split"] == expected_split


def test_task_spec_loads_through_run_eval_without_network():
    task_spec = load_task_spec("github_triage")
    assert task_spec.domain_id == "github_triage"
    assert task_spec.scorer_id == "domains.github_triage.scorer:score_v1"
    assert set(task_spec.tools) == set(TOOLS)
    assert task_spec.dataset_path == "domains/github_triage/dataset.jsonl"


# --- scorer -------------------------------------------------------------------


def test_matching_submission_passes():
    tasks = _load_dataset()
    for task_id in ["gt_002", "gt_010", "gt_020"]:
        expected = tasks[task_id]["expected"]
        output = {"labels": list(expected["labels"]), "assignee": expected["assignee"], "priority": expected["priority"]}
        passed, details = score(output, expected)
        assert passed is True, (task_id, details)
        assert details["label_f1"] == 1.0
        assert details["assignee_score"] == 1.0
        assert details["priority_score"] == 1.0


def test_wrong_submission_fails_with_low_subscores():
    expected = {"labels": ["bug", "accepted", "repro:yes"], "assignee": "electrohyun", "priority": "critical"}
    output = {"labels": ["documentation"], "assignee": "someone-else", "priority": "low"}

    passed, details = score(output, expected)

    assert passed is False
    assert details["label_f1"] == 0.0
    assert details["assignee_score"] == 0.0
    assert details["priority_score"] == 0.0


def test_partial_label_overlap_gives_partial_not_zero_credit():
    expected = {"labels": ["bug", "accepted", "repro:yes"], "assignee": "electrohyun", "priority": "critical"}
    output = {"labels": ["bug", "accepted"], "assignee": "electrohyun", "priority": "critical"}

    passed, details = score(output, expected)

    assert 0.0 < details["label_f1"] < 1.0
    assert details["label_precision"] == 1.0
    assert details["label_recall"] == pytest.approx(2 / 3)
    assert passed is True  # assignee + priority correct, labels mostly right


def test_priority_off_by_one_gives_partial_credit_off_by_two_gives_none():
    expected = {"labels": [], "assignee": None, "priority": "high"}

    _, adjacent = score({"labels": [], "assignee": None, "priority": "medium"}, expected)
    assert adjacent["priority_score"] == 0.5
    assert adjacent["priority_distance"] == 1

    _, far = score({"labels": [], "assignee": None, "priority": "low"}, expected)
    assert far["priority_score"] == 0.0
    assert far["priority_distance"] == 2

    _, exact = score({"labels": [], "assignee": None, "priority": "high"}, expected)
    assert exact["priority_score"] == 1.0


def test_no_assignee_on_both_sides_counts_as_a_match():
    expected = {"labels": ["build"], "assignee": None, "priority": "critical"}
    output = {"labels": ["build"], "assignee": None, "priority": "critical"}

    passed, details = score(output, expected)

    assert details["assignee_score"] == 1.0
    assert passed is True


def test_assignee_match_is_case_insensitive():
    expected = {"labels": [], "assignee": "ElectroHyun", "priority": "low"}
    output = {"labels": [], "assignee": "electrohyun", "priority": "low"}

    _, details = score(output, expected)

    assert details["assignee_score"] == 1.0


def test_malformed_submission_does_not_crash_scorer():
    expected = {"labels": ["bug"], "assignee": "someone", "priority": "high"}

    for garbage in ["not a dict", None, 12345, ["a", "list"], {"labels": "bug-not-a-list"}]:
        passed, details = score(garbage, expected)
        assert passed is False
        assert details["composite_score"] == 0.0


def test_malformed_expected_does_not_crash_scorer():
    passed, details = score({"labels": ["bug"], "assignee": "x", "priority": "high"}, {})
    assert passed is False
    assert details["label_f1"] == 0.0


def test_no_network_imports_in_scorer():
    """The scorer must never be able to make a network call, even indirectly."""
    import ast

    banned_modules = {"socket", "urllib", "requests", "http", "http.client"}
    tree = ast.parse((DOMAIN_DIR / "scorer.py").read_text())

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not (imported_modules & banned_modules), imported_modules
    assert not any(m == "tools" or m.endswith(".tools") for m in imported_modules), imported_modules


# --- cache: record/replay -------------------------------------------------


def test_replay_makes_only_one_network_call_for_repeated_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)

    calls = []

    def fake_http_get_json(url):
        calls.append(url)
        return {"fake": "payload", "n_calls_so_far": len(calls)}

    monkeypatch.setattr(_cache, "_http_get_json", fake_http_get_json)

    first = _cache.cached_get("/repos/octocat/example/labels", {"per_page": 100})
    second = _cache.cached_get("/repos/octocat/example/labels", {"per_page": 100})

    assert first == second
    assert len(calls) == 1, "second call with an identical request must replay from cache, not hit the network"


def test_cached_get_writes_a_cache_file_replay_reads_back(tmp_path, monkeypatch):
    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_cache, "_http_get_json", lambda url: {"hello": "world"})

    _cache.cached_get("/repos/octocat/example/issues/1")

    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    assert json.loads(cache_files[0].read_text()) == {"hello": "world"}


def test_offline_env_var_blocks_uncached_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setenv(_cache.OFFLINE_ENV_VAR, "1")

    def boom(url):
        raise AssertionError("must not touch the network when offline env var is set")

    monkeypatch.setattr(_cache, "_http_get_json", boom)

    with pytest.raises(_cache.NetworkDisabledError):
        _cache.cached_get("/repos/octocat/example/labels")


def test_full_dataset_resolves_offline_from_the_committed_cache(monkeypatch):
    """Every dataset issue must be answerable by get_issue() with the network
    fully disabled - proving the shipped cache/ is a complete, reproducible
    snapshot and that a real eval run never needs live GitHub access."""

    def boom(url):
        raise AssertionError(f"unexpected network call for {url!r}: cache/ is incomplete for this request")

    monkeypatch.setattr(_cache, "_http_get_json", boom)
    monkeypatch.setenv(_cache.OFFLINE_ENV_VAR, "1")

    for record in _load_dataset().values():
        issue = get_issue(record["input"]["repo"], record["input"]["issue_number"])
        assert issue["number"] == record["input"]["issue_number"]
        assert "labels" not in issue
        assert "assignee" not in issue
        assert "state" not in issue


def test_get_issue_redacts_ground_truth_fields():
    tasks = _load_dataset()
    record = tasks["gt_002"]
    issue = get_issue(record["input"]["repo"], record["input"]["issue_number"])

    assert set(issue) == {"repo", "number", "title", "body", "reporter", "created_at", "comments"}


def test_search_issues_returns_historical_labels_and_assignees():
    results = search_issues("eslint/eslint", "label:bug", state="closed", per_page=10)
    assert len(results) > 0
    for item in results:
        assert "labels" in item
        assert "assignees" in item
        assert "state" in item


def test_list_labels_returns_repo_label_vocabulary():
    labels = list_labels("eslint/eslint")
    names = {label["name"] for label in labels}
    assert "accepted" in names
    assert "bug" in names


def test_get_contributor_activity_summarizes_label_counts():
    activity = get_contributor_activity("eslint/eslint", "8ell")
    assert activity["username"] == "8ell"
    assert isinstance(activity["label_counts"], dict)
    assert activity["closed_issue_count"] >= 1


def test_tools_dict_matches_task_spec():
    task_spec = load_task_spec("github_triage")
    assert set(TOOLS) == set(task_spec.tools)
    for name, fn in TOOLS.items():
        assert callable(fn), name
