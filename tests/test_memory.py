import json

import pytest

from core import memory
from core.memory import MemoryEntry


def _entry(**overrides) -> MemoryEntry:
    defaults = dict(
        id="mem-001",
        domain_id="coderepair",
        scope="rule",
        content="Always run the linter before submitting a patch.",
        trigger="submitting a patch to the repo",
        evidence_task_ids=["task-1"],
        source_run_id="run-1",
        created_gen=0,
        confidence=0.5,
        times_retrieved=0,
        times_helped=0,
        times_hurt=0,
        status="active",
    )
    defaults.update(overrides)
    return MemoryEntry(**defaults)


def test_entries_survive_a_process_restart(tmp_path):
    entry = _entry()
    memory.write(entry, base_dir=tmp_path)

    # Nothing in this module is cached in memory - every read goes back to
    # disk - so a second, independent load call is equivalent to a fresh
    # process reading the same file after a restart.
    reloaded = memory.load_all("coderepair", base_dir=tmp_path)

    assert reloaded == [entry]


def test_write_appends_one_line_per_call(tmp_path):
    memory.write(_entry(id="mem-001"), base_dir=tmp_path)
    memory.write(_entry(id="mem-002"), base_dir=tmp_path)

    lines = memory.entries_path("coderepair", tmp_path).read_text().splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "mem-001"
    assert json.loads(lines[1])["id"] == "mem-002"


def test_retrieve_ranks_relevant_entry_higher(tmp_path):
    relevant = _entry(
        id="mem-relevant",
        trigger="fixing a null pointer exception in java",
        confidence=0.5,
    )
    irrelevant = _entry(
        id="mem-irrelevant",
        trigger="formatting yaml indentation",
        confidence=0.5,
    )
    memory.write(relevant, base_dir=tmp_path)
    memory.write(irrelevant, base_dir=tmp_path)

    results = memory.retrieve(
        {"bug_report": "we are seeing a null pointer exception in java"},
        domain_id="coderepair",
        k=5,
        base_dir=tmp_path,
    )

    assert [e.id for e in results] == ["mem-relevant"]


def test_retrieve_excludes_retired_entries(tmp_path):
    retired = _entry(id="mem-retired", trigger="null pointer exception", status="retired")
    memory.write(retired, base_dir=tmp_path)

    results = memory.retrieve(
        {"bug_report": "null pointer exception"},
        domain_id="coderepair",
        k=5,
        base_dir=tmp_path,
    )

    assert results == []


def test_retrieve_increments_times_retrieved(tmp_path):
    memory.write(_entry(id="mem-001", trigger="null pointer exception"), base_dir=tmp_path)

    memory.retrieve({"bug": "null pointer exception"}, "coderepair", k=5, base_dir=tmp_path)
    results = memory.retrieve({"bug": "null pointer exception"}, "coderepair", k=5, base_dir=tmp_path)

    assert results[0].times_retrieved == 2


def test_peek_matches_retrieve_ranking_without_bumping_times_retrieved(tmp_path):
    relevant = _entry(id="mem-relevant", trigger="fixing a null pointer exception in java")
    irrelevant = _entry(id="mem-irrelevant", trigger="formatting yaml indentation")
    memory.write(relevant, base_dir=tmp_path)
    memory.write(irrelevant, base_dir=tmp_path)

    peeked = memory.peek(
        {"bug_report": "we are seeing a null pointer exception in java"},
        domain_id="coderepair",
        k=5,
        base_dir=tmp_path,
    )

    assert [e.id for e in peeked] == ["mem-relevant"]
    assert peeked[0].times_retrieved == 0

    reloaded = memory.get_entry("mem-relevant", "coderepair", base_dir=tmp_path)
    assert reloaded.times_retrieved == 0, "peek() must never durably bump times_retrieved"


def test_peek_never_calls_write(tmp_path, monkeypatch):
    memory.write(_entry(id="mem-001", trigger="null pointer exception"), base_dir=tmp_path)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("memory.write should not have been called")

    monkeypatch.setattr(memory, "write", _fail_if_called)

    found = memory.peek({"bug": "null pointer exception"}, "coderepair", k=5, base_dir=tmp_path)
    assert len(found) == 1
    assert found[0].id == "mem-001"


def test_reinforce_helped_raises_confidence(tmp_path):
    memory.write(_entry(confidence=0.5), base_dir=tmp_path)

    memory.reinforce("mem-001", "coderepair", helped=True, base_dir=tmp_path)

    entry = memory.get_entry("mem-001", "coderepair", base_dir=tmp_path)
    assert entry.confidence == pytest.approx(0.6)
    assert entry.times_helped == 1
    assert entry.times_hurt == 0


def test_reinforce_hurt_lowers_confidence(tmp_path):
    memory.write(_entry(confidence=0.5), base_dir=tmp_path)

    memory.reinforce("mem-001", "coderepair", helped=False, base_dir=tmp_path)

    entry = memory.get_entry("mem-001", "coderepair", base_dir=tmp_path)
    assert entry.confidence == pytest.approx(0.35)
    assert entry.times_hurt == 1
    assert entry.times_helped == 0


def test_reinforce_retires_entry_after_enough_negative_reinforcement(tmp_path):
    memory.write(_entry(confidence=0.5), base_dir=tmp_path)

    for _ in range(3):
        memory.reinforce("mem-001", "coderepair", helped=False, base_dir=tmp_path)

    entry = memory.get_entry("mem-001", "coderepair", base_dir=tmp_path)
    assert entry.status == "retired"
    assert entry.confidence < 0.2
    assert entry.times_hurt == 3


def test_reinforce_unknown_entry_raises(tmp_path):
    with pytest.raises(KeyError):
        memory.reinforce("does-not-exist", "coderepair", helped=True, base_dir=tmp_path)


def test_compact_merges_near_duplicates_and_unions_evidence(tmp_path):
    a = _entry(
        id="mem-a",
        content="Run the linter before submitting a patch.",
        trigger="submitting a patch to the repository",
        evidence_task_ids=["task-1", "task-2"],
        confidence=0.5,
    )
    b = _entry(
        id="mem-b",
        content="Run the linter before submitting any patch.",
        trigger="submitting a patch to the repository",
        evidence_task_ids=["task-3"],
        confidence=0.7,
    )
    memory.write(a, base_dir=tmp_path)
    memory.write(b, base_dir=tmp_path)

    memory.compact("coderepair", base_dir=tmp_path)

    all_entries = {e.id: e for e in memory.load_all("coderepair", base_dir=tmp_path)}
    survivor = all_entries["mem-a"]
    absorbed = all_entries["mem-b"]

    assert survivor.status == "active"
    assert set(survivor.evidence_task_ids) == {"task-1", "task-2", "task-3"}
    assert survivor.confidence == pytest.approx(0.7)
    assert absorbed.status == "merged"
    # The absorbed entry's own evidence is still visible on disk, just
    # consolidated under the survivor - nothing is deleted.
    assert absorbed.evidence_task_ids == ["task-3"]


def test_compact_leaves_dissimilar_entries_untouched(tmp_path):
    a = _entry(id="mem-a", trigger="null pointer exception in java", content="check for null before dereferencing")
    b = _entry(id="mem-b", trigger="yaml indentation error", content="use two spaces for yaml indentation")
    memory.write(a, base_dir=tmp_path)
    memory.write(b, base_dir=tmp_path)

    memory.compact("coderepair", base_dir=tmp_path)

    entries = {e.id: e for e in memory.load_all("coderepair", base_dir=tmp_path)}
    assert entries["mem-a"].status == "active"
    assert entries["mem-b"].status == "active"


def test_snapshot_matches_state_at_call_time(tmp_path):
    memory.write(_entry(id="mem-001"), base_dir=tmp_path)
    memory.write(_entry(id="mem-002"), base_dir=tmp_path)

    path = memory.snapshot("coderepair", gen_n=3, base_dir=tmp_path)

    assert path == tmp_path / "coderepair" / "snapshots" / "gen_3.jsonl"
    lines = path.read_text().splitlines()
    snapshotted_ids = sorted(json.loads(line)["id"] for line in lines)
    assert snapshotted_ids == ["mem-001", "mem-002"]

    # A later write shouldn't retroactively change an already-taken snapshot.
    memory.write(_entry(id="mem-003"), base_dir=tmp_path)
    lines_after = path.read_text().splitlines()
    assert len(lines_after) == 2


def test_load_all_status_filter_keeps_retired_queryable(tmp_path):
    memory.write(_entry(id="mem-001", status="active"), base_dir=tmp_path)
    memory.write(_entry(id="mem-002", status="retired"), base_dir=tmp_path)

    active_only = memory.load_all("coderepair", base_dir=tmp_path, statuses={"active"})
    everything = memory.load_all("coderepair", base_dir=tmp_path)

    assert [e.id for e in active_only] == ["mem-001"]
    assert {e.id for e in everything} == {"mem-001", "mem-002"}
