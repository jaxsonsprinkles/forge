import pytest

from core import llm
from core.llm import SpendCeilingExceeded, complete, estimate_cost_usd


@pytest.fixture(autouse=True)
def _reset_spend_tracker():
    llm.reset_spend_tracker()
    yield
    llm.reset_spend_tracker()


def _fake_seam(reply: str, input_tokens: int = 10, output_tokens: int = 5, calls: list | None = None):
    """Build a fake `_invoke_provider` seam that records how many times it's called."""
    call_log = calls if calls is not None else []

    def _seam(model, messages, params, config):
        call_log.append((model, messages, params))
        return reply, input_tokens, output_tokens

    return _seam, call_log


def test_complete_calls_seam_on_first_call(tmp_path, monkeypatch):
    seam, calls = _fake_seam("hi there")
    monkeypatch.setattr(llm, "_invoke_provider", seam)

    text, cost_usd, latency_ms = complete(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-opus-5",
        cache_dir=tmp_path,
    )

    assert text == "hi there"
    assert cost_usd > 0
    assert latency_ms >= 0
    assert len(calls) == 1


def test_complete_caches_identical_calls(tmp_path, monkeypatch):
    seam, calls = _fake_seam("hi there")
    monkeypatch.setattr(llm, "_invoke_provider", seam)

    args = dict(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-opus-5",
        cache_dir=tmp_path,
    )
    first = complete(**args)
    second = complete(**args)

    assert first == second
    # The seam must be hit exactly once; the second call is served from disk.
    assert len(calls) == 1


def test_complete_distinguishes_different_params(tmp_path, monkeypatch):
    seam, calls = _fake_seam("hi there")
    monkeypatch.setattr(llm, "_invoke_provider", seam)
    messages = [{"role": "user", "content": "hi"}]

    complete(messages=messages, model="claude-opus-5", cache_dir=tmp_path, max_tokens=100)
    complete(messages=messages, model="claude-opus-5", cache_dir=tmp_path, max_tokens=200)

    assert len(calls) == 2


def test_complete_distinguishes_different_messages(tmp_path, monkeypatch):
    seam, calls = _fake_seam("hi there")
    monkeypatch.setattr(llm, "_invoke_provider", seam)

    complete(messages=[{"role": "user", "content": "hi"}], model="claude-opus-5", cache_dir=tmp_path)
    complete(messages=[{"role": "user", "content": "bye"}], model="claude-opus-5", cache_dir=tmp_path)

    assert len(calls) == 2


def test_estimate_cost_usd_known_model():
    assert estimate_cost_usd("claude-opus-5", input_tokens=1_000_000, output_tokens=0) == 5.00


def test_estimate_cost_usd_unknown_model_is_free():
    assert estimate_cost_usd("some-unknown-model", input_tokens=1000, output_tokens=1000) == 0.0


def test_spend_ceiling_raises_before_calling_seam(tmp_path, monkeypatch):
    seam, calls = _fake_seam("hi there")
    monkeypatch.setattr(llm, "_invoke_provider", seam)
    monkeypatch.setenv("FORGE_MAX_SPEND_USD", "0.0000001")

    with pytest.raises(SpendCeilingExceeded):
        complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-opus-5",
            cache_dir=tmp_path,
        )

    # The ceiling check happens before the provider is ever invoked.
    assert len(calls) == 0


def test_spend_ceiling_allows_calls_under_the_limit(tmp_path, monkeypatch):
    seam, calls = _fake_seam("hi there")
    monkeypatch.setattr(llm, "_invoke_provider", seam)
    monkeypatch.setenv("FORGE_MAX_SPEND_USD", "1000")

    complete(messages=[{"role": "user", "content": "hi"}], model="claude-opus-5", cache_dir=tmp_path)

    assert len(calls) == 1


def test_spend_ceiling_accumulates_across_calls(tmp_path, monkeypatch):
    seam, calls = _fake_seam("hi there")
    monkeypatch.setattr(llm, "_invoke_provider", seam)

    # Same-length messages so their precall cost estimates are identical.
    messages1 = [{"role": "user", "content": "hi"}]
    messages2 = [{"role": "user", "content": "yo"}]
    ceiling = llm._estimate_precall_cost_usd("claude-opus-5", messages1, {})
    monkeypatch.setenv("FORGE_MAX_SPEND_USD", str(ceiling))

    # First call: projected spend == ceiling exactly, so it's allowed.
    complete(messages=messages1, model="claude-opus-5", cache_dir=tmp_path)
    # Second call: prior call's real cost pushes projected spend over the ceiling.
    with pytest.raises(SpendCeilingExceeded):
        complete(messages=messages2, model="claude-opus-5", cache_dir=tmp_path)

    # First call went through; second was blocked before hitting the seam.
    assert len(calls) == 1


def test_cache_hit_does_not_touch_spend_ceiling(tmp_path, monkeypatch):
    seam, calls = _fake_seam("hi there")
    monkeypatch.setattr(llm, "_invoke_provider", seam)

    args = dict(messages=[{"role": "user", "content": "hi"}], model="claude-opus-5", cache_dir=tmp_path)
    complete(**args)

    # Now set an impossibly low ceiling; a cache hit must still succeed since it's free.
    monkeypatch.setenv("FORGE_MAX_SPEND_USD", "0")
    text, cost_usd, latency_ms = complete(**args)

    assert text == "hi there"
    assert len(calls) == 1
