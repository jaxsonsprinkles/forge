"""Tests for core.executor_ao.dispatch(). No real `ao` CLI or subprocess
calls: every AO/git touchpoint (_create_branch, _spawn_worker,
_session_status, _kill_session, _branch_head_sha) is monkeypatched so
these run fast and offline.
"""

from core import executor_ao
from core.executor_ao import dispatch
from core.types import Mutation

PARENT_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


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


def _patch_common(monkeypatch, *, statuses=None, heads=None, spawn_ok=True):
    """statuses: dict[mutation_id -> status str] (default "idle").
    heads: dict[branch -> sha] (default a new sha, i.e. "committed")."""
    statuses = statuses or {}
    heads = heads or {}
    killed = []
    created_branches = []

    def fake_create_branch(branch, parent_sha):
        created_branches.append((branch, parent_sha))

    def fake_spawn_worker(project, name, branch, prompt):
        if not spawn_ok:
            raise RuntimeError("spawn failed")
        return f"session-for-{name}"

    def fake_session_status(session_id, project):
        name = session_id.replace("session-for-", "")
        return statuses.get(name, "idle")

    def fake_branch_head_sha(branch):
        return heads.get(branch, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

    def fake_kill_session(session_id, project):
        killed.append(session_id)

    monkeypatch.setattr(executor_ao, "_create_branch", fake_create_branch)
    monkeypatch.setattr(executor_ao, "_spawn_worker", fake_spawn_worker)
    monkeypatch.setattr(executor_ao, "_session_status", fake_session_status)
    monkeypatch.setattr(executor_ao, "_branch_head_sha", fake_branch_head_sha)
    monkeypatch.setattr(executor_ao, "_kill_session", fake_kill_session)
    return created_branches, killed


def test_dispatch_returns_branch_for_mutation_that_committed(monkeypatch):
    _patch_common(monkeypatch)
    mutation = _mutation(id="mut-00-prompt")

    result = dispatch([mutation], PARENT_SHA, project="forge")

    assert result == {"mut-00-prompt": executor_ao._branch_name("mut-00-prompt", PARENT_SHA)}


def test_dispatch_omits_mutation_with_no_new_commit(monkeypatch):
    branch = executor_ao._branch_name("mut-00-prompt", PARENT_SHA)
    _patch_common(monkeypatch, heads={branch: PARENT_SHA})
    mutation = _mutation(id="mut-00-prompt")

    result = dispatch([mutation], PARENT_SHA, project="forge")

    assert result == {}


def test_dispatch_omits_mutation_that_fails_to_spawn(monkeypatch):
    _patch_common(monkeypatch, spawn_ok=False)
    mutation = _mutation(id="mut-00-prompt")

    result = dispatch([mutation], PARENT_SHA, project="forge")

    assert result == {}


def test_dispatch_kills_every_spawned_session(monkeypatch):
    _created, killed = _patch_common(monkeypatch)
    mutations = [_mutation(id="mut-00-prompt"), _mutation(id="mut-01-tool", surface="tool", target_files=["tools.py"])]

    dispatch(mutations, PARENT_SHA, project="forge")

    assert len(killed) == 2


def test_dispatch_returns_dict_shape_for_multiple_mutations(monkeypatch):
    _patch_common(monkeypatch)
    mutations = [
        _mutation(id="mut-00-prompt"),
        _mutation(id="mut-01-tool", surface="tool", target_files=["tools.py"]),
    ]

    result = dispatch(mutations, PARENT_SHA, project="forge")

    assert set(result.keys()) == {"mut-00-prompt", "mut-01-tool"}
    assert all(isinstance(v, str) for v in result.values())


def test_dispatch_creates_one_branch_per_mutation_from_parent_sha(monkeypatch):
    created, _killed = _patch_common(monkeypatch)
    mutations = [
        _mutation(id="mut-00-prompt"),
        _mutation(id="mut-01-tool", surface="tool", target_files=["tools.py"]),
    ]

    dispatch(mutations, PARENT_SHA, project="forge")

    assert created == [
        (executor_ao._branch_name("mut-00-prompt", PARENT_SHA), PARENT_SHA),
        (executor_ao._branch_name("mut-01-tool", PARENT_SHA), PARENT_SHA),
    ]


def test_dispatch_times_out_when_session_never_goes_idle(monkeypatch):
    _patch_common(monkeypatch, statuses={"mut-00-prompt": "working"})
    mutation = _mutation(id="mut-00-prompt")

    result = dispatch(
        [mutation],
        PARENT_SHA,
        project="forge",
        timeout_s=0.05,
        poll_interval_s=0.01,
    )

    assert result == {}


def test_dispatch_raises_without_a_project_id(monkeypatch):
    monkeypatch.delenv("AO_PROJECT_ID", raising=False)
    mutation = _mutation(id="mut-00-prompt")

    try:
        dispatch([mutation], PARENT_SHA)
    except RuntimeError as exc:
        assert "project" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
