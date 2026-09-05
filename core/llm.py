"""The single entry point for all model-provider calls.

Nothing outside this module may call a model provider directly (see
AGENTS.md). Every call is cached to disk, keyed by a hash of
(model, messages, params), so re-running unchanged code costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

DEFAULT_CACHE_DIR = Path(os.environ.get("FORGE_LLM_CACHE_DIR", "ledger/llm_cache"))

# USD per million tokens, as (input_rate, output_rate).
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class LLMResponse:
    """The result of a single (possibly cached) model call."""

    content: str
    model: str
    cost_usd: float
    latency_ms: int
    cached: bool
    stop_reason: str | None
    raw: dict[str, Any] = field(default_factory=dict)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the dollar cost of a call from its token counts.

    Unknown models are priced at $0 rather than raising, since cost
    estimation should never be the reason a call fails.
    """
    input_rate, output_rate = PRICING_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


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


def complete(
    model: str,
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
    max_tokens: int = 16000,
    cache_dir: Path | str | None = None,
    client: anthropic.Anthropic | None = None,
    **params: Any,
) -> LLMResponse:
    """Run a single model completion, transparently cached on disk.

    The cache key hashes (model, messages, system, max_tokens, and any
    extra params); a repeated call with identical inputs never hits the
    network. Set FORGE_LLM_CACHE_DIR to change where cache entries live.
    """
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    key_params = {"system": system, "max_tokens": max_tokens, **params}
    key = _cache_key(model, messages, key_params)

    cached_entry = _read_cache(resolved_cache_dir, key)
    if cached_entry is not None:
        return LLMResponse(
            content=cached_entry["content"],
            model=model,
            cost_usd=cached_entry["cost_usd"],
            latency_ms=cached_entry["latency_ms"],
            cached=True,
            stop_reason=cached_entry.get("stop_reason"),
            raw=cached_entry.get("raw", {}),
        )

    active_client = client or anthropic.Anthropic()
    request_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        **params,
    }
    if system is not None:
        request_kwargs["system"] = system

    start = time.monotonic()
    response = active_client.messages.create(**request_kwargs)
    latency_ms = int((time.monotonic() - start) * 1000)

    content = "".join(block.text for block in response.content if block.type == "text")
    cost_usd = estimate_cost_usd(model, response.usage.input_tokens, response.usage.output_tokens)

    entry = {
        "content": content,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "stop_reason": response.stop_reason,
        "raw": response.to_dict(),
    }
    _write_cache(resolved_cache_dir, key, entry)

    return LLMResponse(
        content=content,
        model=model,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cached=False,
        stop_reason=response.stop_reason,
        raw=entry["raw"],
    )
