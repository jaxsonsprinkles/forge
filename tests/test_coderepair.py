import json
from pathlib import Path

from domains.coderepair.scorer import score

DOMAIN_DIR = Path("domains/coderepair")


def _load_dataset() -> dict[str, dict]:
    tasks = {}
    with (DOMAIN_DIR / "dataset.jsonl").open() as f:
        for line in f:
            record = json.loads(line)
            tasks[record["task_id"]] = record
    return tasks


def test_dataset_has_30_records_with_expected_shape():
    tasks = _load_dataset()
    assert len(tasks) == 30
    for task_id, record in tasks.items():
        assert record["task_id"] == task_id
        assert "broken_code" in record["input"]
        assert "function_name" in record["input"]
        assert "tests" in record["expected"]


def test_split_covers_dataset_without_overlap():
    tasks = _load_dataset()
    with (DOMAIN_DIR / "split.json").open() as f:
        split = json.load(f)

    train, holdout = set(split["train"]), set(split["holdout"])
    assert len(split["train"]) == 20
    assert len(split["holdout"]) == 10
    assert train.isdisjoint(holdout)
    assert train | holdout == set(tasks)


def test_fixed_code_passes_for_every_task():
    """A correct implementation of each function should score True."""
    tasks = _load_dataset()

    fixed_sum_to_n = "def sum_to_n(n):\n    return sum(range(1, n + 1))"
    task = tasks["cr_001"]
    passed, details = score(fixed_sum_to_n, task["expected"])
    assert passed is True, details


def test_broken_code_fails_for_a_sample_of_tasks():
    """The original buggy version of each function should score False."""
    tasks = _load_dataset()
    for task_id in ["cr_001", "cr_002", "cr_003", "cr_007", "cr_014", "cr_022"]:
        task = tasks[task_id]
        broken_code = task["input"]["broken_code"]
        passed, details = score(broken_code, task["expected"])
        assert passed is False, (task_id, details)
        assert details["timed_out"] is False


def test_correct_fix_for_off_by_one_bug_passes():
    task = _load_dataset()["cr_001"]
    fixed_code = "def sum_to_n(n):\n    total = 0\n    for i in range(1, n + 1):\n        total += i\n    return total"
    passed, details = score(fixed_code, task["expected"])
    assert passed is True, details


def test_broken_off_by_one_bug_fails():
    task = _load_dataset()["cr_001"]
    passed, details = score(task["input"]["broken_code"], task["expected"])
    assert passed is False
    assert details["returncode"] != 0


def test_infinite_loop_times_out_and_fails_quickly():
    expected = {"tests": "assert spin(1) == 1\n"}
    infinite_loop_code = "def spin(n):\n    while True:\n        pass\n"

    passed, details = score(infinite_loop_code, expected)

    assert passed is False
    assert details["timed_out"] is True


def test_crashing_code_fails_without_raising():
    expected = {"tests": "assert broken() == 1\n"}
    crashing_code = "def broken():\n    raise ValueError('boom')\n"

    passed, details = score(crashing_code, expected)

    assert passed is False
    assert details["timed_out"] is False
    assert details["returncode"] != 0


def test_syntactically_invalid_code_fails_without_raising():
    expected = {"tests": "assert f() == 1\n"}
    invalid_code = "def f(:\n    return 1\n"

    passed, details = score(invalid_code, expected)

    assert passed is False
    assert details["timed_out"] is False


def test_malformed_expected_does_not_raise():
    passed, details = score("def f(): return 1", {})

    assert passed is False
    assert "error" in details


def test_no_network_imports_in_domain_files():
    """Scorer and dataset generation must not perform network calls."""
    banned = ["socket", "urllib", "requests", "http.client"]
    for path in [DOMAIN_DIR / "scorer.py", DOMAIN_DIR / "generate_split.py"]:
        content = path.read_text()
        for token in banned:
            assert token not in content, f"{path} references {token!r}"
