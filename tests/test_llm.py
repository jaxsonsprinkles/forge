import anthropic
import httpx2
import pytest

from core import llm
from core.llm import InfraError, SpendCeilingExceeded, complete, estimate_cost_usd


@pytest.fixture(autouse=True)
def _reset_spend_tracker():
    llm.reset_spend_tracker()
    yield
    llm.reset_spend_tracker()


def _api_error_response(status_code: int, error_type: str, message: str) -> httpx2.Response:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx2.Response(
        status_code,
        request=request,
        json={"error": {"type": error_type, "message": message}},
    )


def _auth_error(message: str = "invalid x-api-key") -> anthropic.AuthenticationError:
    response = _api_error_response(401, "authentication_error", message)
    return anthropic.AuthenticationError(message, response=response, body=response.json())


def _rate_limit_error(message: str = "rate limit exceeded") -> anthropic.RateLimitError:
    response = _api_error_response(429, "rate_limit_error", message)
    return anthropic.RateLimitError(message, response=response, body=response.json())


def _bad_request_error(message: str) -> anthropic.BadRequestError:
    response = _api_error_response(400, "invalid_request_error", message)
    return anthropic.BadRequestError(message, response=response, body=response.json())


class _RaisingMessages:
    def __init__(self, exc: Exception):
        self._exc = exc

    def create(self, **kwargs):
        raise self._exc


class _RaisingClient:
    def __init__(self, exc: Exception):
        self.messages = _RaisingMessages(exc)


def _fake_client_raising(exc: Exception):
    return lambda **kwargs: _RaisingClient(exc)


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


# --- InfraError: auth/credit/rate-limit failures must abort, not fail one call ---
#
# These tests exercise the real `_invoke_provider` (not the `_fake_seam`
# monkeypatch used above), since that's exactly where the anthropic SDK
# exceptions get caught and re-raised as InfraError. `anthropic.Anthropic`
# is monkeypatched to a fake client whose `.messages.create()` raises a
# real SDK exception instance - no network access involved.


def test_authentication_error_raises_infra_error(tmp_path, monkeypatch):
    monkeypatch.setattr(llm.anthropic, "Anthropic", _fake_client_raising(_auth_error()))

    with pytest.raises(InfraError, match="authentication"):
        complete(messages=[{"role": "user", "content": "hi"}], model="claude-opus-5", cache_dir=tmp_path)


def test_rate_limit_error_raises_infra_error(tmp_path, monkeypatch):
    monkeypatch.setattr(llm.anthropic, "Anthropic", _fake_client_raising(_rate_limit_error()))

    with pytest.raises(InfraError, match="rate limit"):
        complete(messages=[{"role": "user", "content": "hi"}], model="claude-opus-5", cache_dir=tmp_path)


def test_credit_balance_bad_request_raises_infra_error(tmp_path, monkeypatch):
    exc = _bad_request_error(
        "Your credit balance is too low to access the Anthropic API. "
        "Please go to Plans & Billing to upgrade or purchase credits."
    )
    monkeypatch.setattr(llm.anthropic, "Anthropic", _fake_client_raising(exc))

    with pytest.raises(InfraError, match="credit"):
        complete(messages=[{"role": "user", "content": "hi"}], model="claude-opus-5", cache_dir=tmp_path)


def test_non_credit_bad_request_error_propagates_unchanged(tmp_path, monkeypatch):
    """A real bad-request-shape bug (malformed params, not a credit issue)
    must NOT be reclassified as an InfraError - it's an agent/caller bug,
    not an infra failure, so it should surface as the original SDK
    exception."""
    exc = _bad_request_error("messages: at least one message is required")
    monkeypatch.setattr(llm.anthropic, "Anthropic", _fake_client_raising(exc))

    with pytest.raises(anthropic.BadRequestError) as excinfo:
        complete(messages=[{"role": "user", "content": "hi"}], model="claude-opus-5", cache_dir=tmp_path)

    assert not isinstance(excinfo.value, InfraError)
