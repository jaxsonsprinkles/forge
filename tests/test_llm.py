from dataclasses import dataclass, field
from typing import Any

from core.llm import complete, estimate_cost_usd


@dataclass
class _FakeBlock:
    type: str
    text: str = ""


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _FakeResponse:
    content: list[_FakeBlock]
    usage: _FakeUsage
    stop_reason: str = "end_turn"

    def to_dict(self) -> dict[str, Any]:
        return {"stop_reason": self.stop_reason}


class _FakeMessages:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.call_count = 0

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.call_count += 1
        return _FakeResponse(
            content=[_FakeBlock(type="text", text=self.reply)],
            usage=_FakeUsage(input_tokens=10, output_tokens=5),
        )


class _FakeClient:
    def __init__(self, reply: str = "hello") -> None:
        self.messages = _FakeMessages(reply)


def test_complete_calls_provider_on_first_call(tmp_path):
    client = _FakeClient(reply="hi there")
    response = complete(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        cache_dir=tmp_path,
        client=client,
    )
    assert response.content == "hi there"
    assert response.cached is False
    assert client.messages.call_count == 1


def test_complete_caches_identical_calls(tmp_path):
    client = _FakeClient(reply="hi there")
    args = dict(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        cache_dir=tmp_path,
        client=client,
    )
    first = complete(**args)
    second = complete(**args)

    assert first.cached is False
    assert second.cached is True
    assert second.content == first.content
    # The provider must be hit exactly once; the second call is served from disk.
    assert client.messages.call_count == 1


def test_complete_distinguishes_different_params(tmp_path):
    client = _FakeClient(reply="hi there")
    messages = [{"role": "user", "content": "hi"}]

    complete(model="claude-opus-5", messages=messages, cache_dir=tmp_path, client=client, max_tokens=100)
    complete(model="claude-opus-5", messages=messages, cache_dir=tmp_path, client=client, max_tokens=200)

    assert client.messages.call_count == 2


def test_estimate_cost_usd_known_model():
    cost = estimate_cost_usd("claude-opus-5", input_tokens=1_000_000, output_tokens=0)
    assert cost == 5.00


def test_estimate_cost_usd_unknown_model_is_free():
    assert estimate_cost_usd("some-unknown-model", input_tokens=1000, output_tokens=1000) == 0.0
