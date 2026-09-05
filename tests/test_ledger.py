import json

from core import ledger
from core.types import Generation, Mutation, ScoreCard


def _card(**overrides) -> ScoreCard:
    defaults = dict(
        accuracy=0.80,
        reliability=0.90,
        cost_per_task=0.02,
        p50_latency_ms=500,
        n=10,
        split="train",
    )
    defaults.update(overrides)
    return ScoreCard(**defaults)


def _mutation(**overrides) -> Mutation:
    defaults = dict(
        id="mut-00-prompt",
        surface="prompt",
        rationale="because",
        instruction="edit prompt.md",
        target_files=["prompt.md"],
    )
    defaults.update(overrides)
    return Mutation(**defaults)


def _generation(gen_n: int, **overrides) -> Generation:
    defaults = dict(
        gen_n=gen_n,
        parent_sha=f"sha{gen_n}",
        scores_before={"d1": _card()},
        mutations=[_mutation()],
        results={"mut-00-prompt": {"d1": _card(accuracy=0.85)}},
        winner_id="mut-00-prompt",
        winner_sha=f"sha{gen_n}-winner",
    )
    defaults.update(overrides)
    return Generation(**defaults)


def test_load_generations_missing_file_returns_empty(tmp_path):
    path = tmp_path / "generations.jsonl"

    assert ledger.load_generations(path) == []


def test_append_generation_is_immediately_readable(tmp_path):
    """Each append must be durable and readable right away, not buffered
    until the process ends - simulates writing N records one at a time."""
    path = tmp_path / "generations.jsonl"

    for gen_n in range(1, 4):
        ledger.append_generation(_generation(gen_n), path)
        loaded = ledger.load_generations(path)
        assert len(loaded) == gen_n
        assert loaded[-1] == _generation(gen_n)


def test_append_generation_preserves_nested_dataclasses(tmp_path):
    path = tmp_path / "generations.jsonl"
    gen = _generation(1, winner_id=None, winner_sha=None)

    ledger.append_generation(gen, path)
    loaded = ledger.load_generations(path)

    assert loaded == [gen]
    assert isinstance(loaded[0].scores_before["d1"], ScoreCard)
    assert isinstance(loaded[0].mutations[0], Mutation)
    assert isinstance(loaded[0].results["mut-00-prompt"]["d1"], ScoreCard)


def test_append_generation_creates_missing_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "generations.jsonl"

    ledger.append_generation(_generation(1), path)

    assert path.exists()
    assert len(ledger.load_generations(path)) == 1


def test_append_generation_writes_one_line_per_call(tmp_path):
    path = tmp_path / "generations.jsonl"

    ledger.append_generation(_generation(1), path)
    ledger.append_generation(_generation(2), path)

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["gen_n"] == 1
    assert json.loads(lines[1])["gen_n"] == 2


def test_load_generations_skips_truncated_last_line(tmp_path):
    """A crash mid-write shouldn't corrupt reading of prior complete lines."""
    path = tmp_path / "generations.jsonl"
    ledger.append_generation(_generation(1), path)
    ledger.append_generation(_generation(2), path)

    # Simulate a process dying mid-`write`: a truncated, invalid JSON
    # fragment with no trailing newline.
    with open(path, "a") as f:
        f.write('{"gen_n": 3, "parent_sha": "sha3", "scores_be')

    loaded = ledger.load_generations(path)

    assert [g.gen_n for g in loaded] == [1, 2]


def test_load_generations_skips_blank_lines(tmp_path):
    path = tmp_path / "generations.jsonl"
    ledger.append_generation(_generation(1), path)
    with open(path, "a") as f:
        f.write("\n")
    ledger.append_generation(_generation(2), path)

    loaded = ledger.load_generations(path)

    assert [g.gen_n for g in loaded] == [1, 2]
