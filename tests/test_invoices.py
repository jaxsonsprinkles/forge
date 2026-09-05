import json
from pathlib import Path

from domains.invoices.scorer import PASS_THRESHOLD, WEIGHTS, score

DOMAIN_DIR = Path("domains/invoices")


def _load_dataset() -> dict[str, dict]:
    tasks = {}
    with (DOMAIN_DIR / "dataset.jsonl").open() as f:
        for line in f:
            record = json.loads(line)
            tasks[record["task_id"]] = record
    return tasks


def test_dataset_has_35_records_with_expected_shape():
    tasks = _load_dataset()
    assert len(tasks) == 35
    for task_id, record in tasks.items():
        assert record["task_id"] == task_id
        assert "invoice_text" in record["input"]
        expected = record["expected"]
        for field in WEIGHTS:
            assert field in expected, f"{task_id} missing expected field {field!r}"
        assert isinstance(expected["line_items"], list) and expected["line_items"]
        for item in expected["line_items"]:
            for key in ("description", "qty", "unit_price", "amount"):
                assert key in item


def test_split_covers_dataset_without_overlap():
    tasks = _load_dataset()
    with (DOMAIN_DIR / "split.json").open() as f:
        split = json.load(f)

    train, holdout = set(split["train"]), set(split["holdout"])
    assert len(split["train"]) == 23
    assert len(split["holdout"]) == 12
    assert train.isdisjoint(holdout)
    assert train | holdout == set(tasks)


def test_dataset_row_split_field_matches_split_json():
    """core.runner filters dataset rows by a per-row 'split' field; keep both in sync."""
    tasks = _load_dataset()
    with (DOMAIN_DIR / "split.json").open() as f:
        split = json.load(f)

    for task_id in split["train"]:
        assert tasks[task_id]["split"] == "train"
    for task_id in split["holdout"]:
        assert tasks[task_id]["split"] == "holdout"


def test_hard_cases_present():
    tasks = _load_dataset()
    for task_id in ("inv_005", "inv_012", "inv_019", "inv_026", "inv_033"):
        assert task_id in tasks


def test_missing_tax_case_has_zero_tax():
    task = _load_dataset()["inv_012"]
    assert task["expected"]["tax"] == 0.0
    text_lines = task["input"]["invoice_text"].lower().splitlines()
    assert not any(line.strip().startswith(("tax", "vat")) for line in text_lines)


def test_bad_arithmetic_case_line_items_dont_sum_to_total():
    task = _load_dataset()["inv_026"]
    expected = task["expected"]
    line_item_sum = round(sum(item["amount"] for item in expected["line_items"]), 2)
    assert line_item_sum != expected["subtotal"]


def test_credit_memo_case_has_negative_amounts():
    task = _load_dataset()["inv_033"]
    expected = task["expected"]
    assert expected["total"] < 0
    assert all(item["amount"] < 0 for item in expected["line_items"])


def test_fully_correct_submission_passes_with_high_score():
    task = _load_dataset()["inv_001"]
    output = dict(task["expected"])

    passed, details = score(output, task["expected"])

    assert passed is True
    assert details["aggregate_score"] >= PASS_THRESHOLD
    for field_result in details["fields"].values():
        assert field_result["score"] == 1.0


def test_fully_correct_submission_as_json_string_also_passes():
    task = _load_dataset()["inv_001"]
    output = json.dumps(task["expected"])

    passed, details = score(output, task["expected"])

    assert passed is True
    assert details["aggregate_score"] >= PASS_THRESHOLD


def test_wrong_total_scores_lower_and_reflects_double_weight():
    task = _load_dataset()["inv_001"]
    expected = task["expected"]

    correct_output = dict(expected)
    passed_correct, details_correct = score(correct_output, expected)

    wrong_total_output = dict(expected)
    wrong_total_output["total"] = expected["total"] * 2 + 100
    passed_wrong, details_wrong = score(wrong_total_output, expected)

    assert passed_correct is True
    assert details_correct["aggregate_score"] > details_wrong["aggregate_score"]
    assert details_wrong["fields"]["total"]["score"] < 1.0

    # A wrong `total` (weight 2) must cost more aggregate score than an
    # equally-wrong field with weight 1, all else held equal.
    wrong_vendor_output = dict(expected)
    wrong_vendor_output["vendor"] = "Completely Different Company Name"
    _, details_wrong_vendor = score(wrong_vendor_output, expected)

    total_penalty = details_correct["aggregate_score"] - details_wrong["aggregate_score"]
    vendor_penalty = details_correct["aggregate_score"] - details_wrong_vendor["aggregate_score"]
    assert total_penalty > vendor_penalty


def test_completely_wrong_total_fails_even_if_everything_else_right():
    task = _load_dataset()["inv_001"]
    expected = task["expected"]

    output = dict(expected)
    output["total"] = 0.01

    passed, details = score(output, expected)

    assert details["fields"]["total"]["score"] < 0.1
    assert details["aggregate_score"] < 1.0


def test_garbage_string_output_does_not_crash_and_fails():
    task = _load_dataset()["inv_001"]
    passed, details = score("not json at all {{{", task["expected"])
    assert passed is False
    assert details["aggregate_score"] == 0.0


def test_none_output_does_not_crash_and_fails():
    task = _load_dataset()["inv_001"]
    passed, details = score(None, task["expected"])
    assert passed is False


def test_list_output_does_not_crash_and_fails():
    task = _load_dataset()["inv_001"]
    passed, details = score([1, 2, 3], task["expected"])
    assert passed is False


def test_output_with_wrong_types_does_not_crash():
    task = _load_dataset()["inv_001"]
    output = {
        "vendor": 12345,
        "invoice_number": None,
        "date": ["not", "a", "string"],
        "line_items": "not a list",
        "subtotal": "abc",
        "tax": {"nested": "dict"},
        "total": [1, 2],
        "currency": 99,
    }
    passed, details = score(output, task["expected"])
    assert passed is False
    assert details["aggregate_score"] == 0.0


def test_empty_dict_output_does_not_crash_and_fails():
    task = _load_dataset()["inv_001"]
    passed, details = score({}, task["expected"])
    assert passed is False
    assert details["aggregate_score"] < PASS_THRESHOLD


def test_partial_line_items_get_partial_credit_not_zero():
    task = _load_dataset()["inv_005"]  # multi-page: 5 expected line items
    expected = task["expected"]
    output = dict(expected)
    output["line_items"] = expected["line_items"][:2]  # only 2 of 5 items reported

    passed, details = score(output, expected)

    line_items_score = details["fields"]["line_items"]["score"]
    assert 0.0 < line_items_score < 1.0


def test_no_network_imports_in_domain_files():
    """Scorer and dataset generation must not perform network calls."""
    banned = ["socket", "urllib", "requests", "http.client"]
    for path in [DOMAIN_DIR / "scorer.py", DOMAIN_DIR / "generate_dataset.py"]:
        content = path.read_text()
        for token in banned:
            assert token not in content, f"{path} references {token!r}"
