"""The single entry point for all model-provider calls.

Nothing outside this module may call a model provider directly (see
AGENTS.md). Every call is cached to disk, keyed by a hash of
(model, messages, params), so re-running unchanged code costs nothing.

The actual network call lives behind the `_invoke_provider` seam below,
which calls the Anthropic API. Tests monkeypatch that seam to prove
caching and spend-ceiling behavior with zero real network access.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import anthropic

DEFAULT_CACHE_DIR = Path(os.environ.get("FORGE_LLM_CACHE_DIR", "ledger/llm_cache"))

# USD per million tokens, as (input_rate, output_rate).
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Assumed output size when estimating cost *before* a call is made, since
# the real output length isn't known until the provider responds.
_DEFAULT_ASSUMED_OUTPUT_TOKENS = 1024

# Process-wide cumulative spend, checked against FORGE_MAX_SPEND_USD.
_cumulative_spend_usd = 0.0


class SpendCeilingExceeded(Exception):
    """Raised when a call would push cumulative spend past FORGE_MAX_SPEND_USD."""


class InfraError(Exception):
    """Raised for infrastructure-level failures - bad credentials, an
    exhausted credit balance, or a rate limit - rather than an agent bug.

    These conditions won't resolve themselves on the next task, so callers
    (`core/runner.py`, `evals/learning_curve.py`) must let this propagate
    instead of recording it as a per-task `RunResult.error`: continuing a
    run under one of these conditions just produces many more meaningless
    zero-cost failure records instead of surfacing the real problem.
    """


def _is_credit_balance_error(exc: "anthropic.BadRequestError") -> bool:
    """Whether a 400 from the API is actually a credit-balance error.

    Anthropic reports an exhausted credit balance as a 400
    `invalid_request_error` (the same status code as a real malformed-
    request bug), distinguishable only by message text - there is no
    dedicated exception type or status code for it.
    """
    body = getattr(exc, "body", None)
    message = ""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", ""))
    if not message:
        message = str(exc)
    return "credit balance" in message.lower()


@dataclass
class ProviderConfig:
    """Provider credentials/config, read from the environment."""

    api_key: str | None
    base_url: str | None


def reset_spend_tracker() -> None:
    """Reset the process-wide cumulative spend counter. Mainly for tests."""
    global _cumulative_spend_usd
    _cumulative_spend_usd = 0.0


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the dollar cost of a call from its token counts.

    Unknown models are priced at $0 rather than raising, since cost
    estimation should never be the reason a call fails.
    """
    input_rate, output_rate = PRICING_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _estimate_precall_cost_usd(model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> float:
    """Rough worst-case cost estimate used only to enforce the spend ceiling.

    Uses a ~4-chars-per-token heuristic for input size and `max_tokens`
    (or a conservative default) as the assumed output size. This is
    intentionally approximate: it only needs to guard against blowing the
    budget, not to bill accurately (the real cost, from actual token
    usage, is what gets cached and returned).
    """
    input_tokens = max(1, len(json.dumps(messages)) // 4)
    output_tokens = params.get("max_tokens", _DEFAULT_ASSUMED_OUTPUT_TOKENS)
    return estimate_cost_usd(model, input_tokens, output_tokens)


def _spend_ceiling_usd() -> float | None:
    raw = os.environ.get("FORGE_MAX_SPEND_USD")
    return float(raw) if raw is not None else None


def _load_provider_config() -> ProviderConfig:
    """Read provider credentials/config from the environment."""
    return ProviderConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )


def _invoke_provider(
    model: str,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    config: ProviderConfig,
) -> tuple[str, int, int]:
    """Make the real model call against the Anthropic API.

    This is the only function in the module that performs network I/O.
    Tests monkeypatch `core.llm._invoke_provider` to return
    (text, input_tokens, output_tokens) without touching the network.

    `messages` may include `role: "system"` entries (Anthropic takes the
    system prompt as a separate top-level parameter, not as a message).
    """
    client = anthropic.Anthropic(api_key=config.api_key, base_url=config.base_url)

    call_params = dict(params)
    max_tokens = call_params.pop("max_tokens", _DEFAULT_ASSUMED_OUTPUT_TOKENS)

    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    chat_messages = [m for m in messages if m.get("role") != "system"]

    if system_parts:
        call_params["system"] = "\n\n".join(system_parts)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=chat_messages,
            **call_params,
        )
    except anthropic.AuthenticationError as exc:
        raise InfraError(f"Anthropic authentication failed: {exc}") from exc
    except anthropic.RateLimitError as exc:
        raise InfraError(f"Anthropic rate limit exceeded: {exc}") from exc
    except anthropic.BadRequestError as exc:
        if _is_credit_balance_error(exc):
            raise InfraError(f"Anthropic credit balance exhausted: {exc}") from exc
        raise

    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.usage.input_tokens, response.usage.output_tokens


def _cache_key(model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
    """Compute a deterministic hash key for a (model, messages, params) triple."""
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _read_cache(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def _write_cache(cache_dir: Path, key: str, entry: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, key)
    # Write to a temp file first so a crash mid-write can't corrupt the cache entry.
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        json.dump(entry, f)
    tmp_path.replace(path)


def complete(messages: list[dict[str, Any]], model: str, **params: Any) -> tuple[str, float, int]:
    """Run a single model completion, returning (text, cost_usd, latency_ms).

    Cached on disk, keyed by a hash of (model, messages, params) - a
    repeat call with identical inputs is served from disk and never
    invokes the provider. Pass `cache_dir` in params to override where
    cache entries live (defaults to FORGE_LLM_CACHE_DIR or ledger/llm_cache).

    Enforces FORGE_MAX_SPEND_USD as a per-process cumulative spend
    ceiling: raises SpendCeilingExceeded *before* invoking the provider
    if the call's estimated cost would push cumulative spend past it.
    Cache hits never touch the ceiling, since they cost nothing.
    """
    global _cumulative_spend_usd

    params = dict(params)
    cache_dir = Path(params.pop("cache_dir", None) or DEFAULT_CACHE_DIR)
    key = _cache_key(model, messages, params)

    cached_entry = _read_cache(cache_dir, key)
    if cached_entry is not None:
        return cached_entry["text"], cached_entry["cost_usd"], cached_entry["latency_ms"]

    ceiling = _spend_ceiling_usd()
    if ceiling is not None:
        estimated_cost = _estimate_precall_cost_usd(model, messages, params)
        projected_spend = _cumulative_spend_usd + estimated_cost
        if projected_spend > ceiling:
            raise SpendCeilingExceeded(
                f"Call to model {model!r} would cost an estimated ${estimated_cost:.4f}, "
                f"bringing cumulative spend to ${projected_spend:.4f}, over the "
                f"FORGE_MAX_SPEND_USD ceiling of ${ceiling:.4f}."
            )

    config = _load_provider_config()
    start = monotonic()
    text, input_tokens, output_tokens = _invoke_provider(model, messages, params, config)
    latency_ms = int((monotonic() - start) * 1000)

    cost_usd = estimate_cost_usd(model, input_tokens, output_tokens)
    _cumulative_spend_usd += cost_usd

    _write_cache(cache_dir, key, {"text": text, "cost_usd": cost_usd, "latency_ms": latency_ms})

    return text, cost_usd, latency_ms
