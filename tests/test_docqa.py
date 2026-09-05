import json
from pathlib import Path

from domains.docqa.scorer import score, score_v1

DOMAIN_DIR = Path("domains/docqa")


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
        assert "question" in record["input"]
        assert isinstance(record["input"]["question"], str) and record["input"]["question"]
        assert "answer" in record["expected"]
        assert isinstance(record["expected"]["answer"], str) and record["expected"]["answer"]
        source_docs = record["expected"]["source_docs"]
        assert len(source_docs) == 2
        assert source_docs[0] != source_docs[1]


def test_questions_require_exactly_two_distinct_corpus_documents():
    tasks = _load_dataset()
    corpus_files = {p.name for p in (DOMAIN_DIR / "corpus").glob("*.txt")}
    assert len(corpus_files) == 15
    for task_id, record in tasks.items():
        for doc in record["expected"]["source_docs"]:
            assert doc in corpus_files, f"{task_id} references missing corpus file {doc!r}"


def test_document_pair_coverage_is_not_lopsided():
    """Every one of the 15 corpus docs should be used by a handful of questions,
    not just one or two of them carrying the whole dataset."""
    tasks = _load_dataset()
    counts: dict[str, int] = {}
    for record in tasks.values():
        for doc in record["expected"]["source_docs"]:
            counts[doc] = counts.get(doc, 0) + 1

    assert len(counts) == 15
    assert min(counts.values()) >= 2
    assert max(counts.values()) <= 8


def test_split_covers_dataset_without_overlap():
    tasks = _load_dataset()
    with (DOMAIN_DIR / "split.json").open() as f:
        split = json.load(f)

    train, holdout = set(split["train"]), set(split["holdout"])
    assert len(split["train"]) == 20
    assert len(split["holdout"]) == 10
    assert train.isdisjoint(holdout)
    assert train | holdout == set(tasks)


def test_correct_answer_passes():
    tasks = _load_dataset()
    task = tasks["dq_004"]
    passed, details = score("Founders Street", task["expected"])
    assert passed is True, details
    assert details["similarity"] >= details["threshold"]


def test_correct_paraphrased_answer_passes():
    tasks = _load_dataset()
    task = tasks["dq_001"]
    candidate = "The bridge was funded by Rowan Thornwick, the son of Elias Thornwick who founded the town."
    passed, details = score(candidate, task["expected"])
    assert passed is True, details


def test_wrong_answer_fails():
    tasks = _load_dataset()
    task = tasks["dq_001"]
    passed, details = score("Agatha Voss built the bridge.", task["expected"])
    assert passed is False, details


def test_unrelated_answer_fails():
    tasks = _load_dataset()
    task = tasks["dq_022"]
    passed, details = score("The library was built on Founders Street.", task["expected"])
    assert passed is False, details


def test_garbage_and_empty_submissions_do_not_crash_the_scorer():
    tasks = _load_dataset()
    task = tasks["dq_013"]

    for garbage in ["", None, 12345, [], {}, {"answer": ""}, object()]:
        passed, details = score(garbage, task["expected"])
        assert passed is False
        assert isinstance(details, dict)


def test_malformed_expected_does_not_raise():
    passed, details = score("some answer", {})
    assert passed is False
    assert "error" in details

    passed, details = score("some answer", {"answer": 12345})
    assert passed is False
    assert "error" in details


def test_dict_output_with_answer_key_is_unwrapped():
    tasks = _load_dataset()
    task = tasks["dq_013"]
    passed, details = score({"answer": "71 years"}, task["expected"])
    assert passed is True, details


def test_score_v1_returns_plain_bool_for_runner_compatibility():
    """core.runner.run_agent does `bool(scorer_fn(output, expected))`. Since
    `score` returns a (passed, details) tuple, that call is always truthy
    regardless of `passed` - `score` itself must never be wired up as a
    task_spec.json scorer_id. `score_v1` is the bool-only entrypoint that
    works around this."""
    tasks = _load_dataset()
    task = tasks["dq_013"]

    assert score_v1("71 years", task["expected"]) is True
    assert score_v1("completely wrong", task["expected"]) is False

    # Demonstrates the bug `score_v1` exists to work around: a bare tuple,
    # even (False, {...}), is truthy.
    assert bool(score("completely wrong", task["expected"])) is True


def test_no_network_imports_in_domain_files():
    """Scorer and split generation must not perform network calls."""
    banned = ["socket", "urllib", "requests", "http.client"]
    for path in [DOMAIN_DIR / "scorer.py", DOMAIN_DIR / "generate_split.py"]:
        content = path.read_text()
        for token in banned:
            assert token not in content, f"{path} references {token!r}"


def test_run_eval_end_to_end_with_a_real_task_spec(tmp_path):
    """Verifies the acceptance criterion that this domain scores through
    evals/run_eval.py unchanged: build a task_spec.json pointing at the
    real dataset/scorer and run a fake agent through the real pipeline."""
    import dataclasses

    from core.runner import run_agent
    from core.scorer import score_runs
    from evals.run_eval import load_task_spec

    domains_root = tmp_path / "domains"
    (domains_root / "docqa").mkdir(parents=True)
    task_spec_path = domains_root / "docqa" / "task_spec.json"
    task_spec_path.write_text(
        json.dumps(
            {
                "goal": "answer questions from a small local corpus",
                "tools": [],
                "dataset_path": str(DOMAIN_DIR / "dataset.jsonl"),
                "scorer_id": "domains.docqa.scorer:score_v1",
                "max_tasks": 30,
            }
        )
    )

    agent_dir = tmp_path / "agents" / "oracle_agent"
    agent_dir.mkdir(parents=True)
    tasks = _load_dataset()
    answers_by_question = {t["input"]["question"]: t["expected"]["answer"] for t in tasks.values()}
    (agent_dir / "run.py").write_text(
        "ANSWERS = "
        + repr(answers_by_question)
        + "\n\ndef run(task_input: dict) -> dict:\n"
        "    return {'answer': ANSWERS.get(task_input['question'], 'I do not know')}\n"
    )

    task_spec = load_task_spec("docqa", domains_root=domains_root)
    results = run_agent(str(agent_dir), task_spec, split="train")
    score_card = score_runs(results, "train")

    assert score_card.n == 20
    assert score_card.accuracy == 1.0
    assert all(r.error is None for r in results)
    assert dataclasses.asdict(score_card)["split"] == "train"
