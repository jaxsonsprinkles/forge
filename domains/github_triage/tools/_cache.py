"""Shared record/replay HTTP cache for the github_triage domain's tools.

Every tool function in this package (search_issues, get_issue,
list_labels, get_contributor_activity, get_file_owners) routes its
GitHub REST API call through `cached_get()` below rather than calling
`urllib` directly. The cache key is a hash of the full request URL
(method + path + sorted query params); the first call for a given key
makes the real HTTP request and writes its JSON response body to
`cache/<key>.json`, and every later call for that same request - whether
later in this process or in a brand-new one - replays that file and
never touches the network again.

This is what makes this domain's eval/test runs reproducible offline:
once cache/ is populated (see generate_dataset.py, which performed the
one real recording pass this domain ships with), a score can never
change because of a live API response, and CI/tests never need network
access or a GitHub token.

Set FORGE_GITHUB_TRIAGE_OFFLINE=1 to make an uncached request raise
NetworkDisabledError instead of silently falling through to a live call
- used by tests to prove a given code path never touches the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
GITHUB_API_BASE = "https://api.github.com"
USER_AGENT = "forge-github-triage-domain"
REQUEST_TIMEOUT_SECONDS = 15
OFFLINE_ENV_VAR = "FORGE_GITHUB_TRIAGE_OFFLINE"


class NetworkDisabledError(RuntimeError):
    """Raised when a request isn't cached and the offline env var forbids a live call."""


def _build_url(path: str, params: dict[str, Any] | None) -> str:
    if not params:
        return f"{GITHUB_API_BASE}{path}"
    query = urllib.parse.urlencode(sorted(params.items()))
    return f"{GITHUB_API_BASE}{path}?{query}"


def _cache_key(url: str) -> str:
    return hashlib.sha256(f"GET|{url}".encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _http_get_json(url: str) -> Any:
    """The single point where this module touches the network.

    Isolated in its own function (rather than inlined in cached_get) so
    tests can monkeypatch just this - to assert it's called exactly
    once across two identical requests (proving replay works), or to
    make it raise (simulating no network access) while a cached request
    still succeeds.
    """
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def cached_get(path: str, params: dict[str, Any] | None = None) -> Any:
    """GET `path` (a GitHub REST API path, e.g. "/repos/x/y/labels") through
    the on-disk record/replay cache. See module docstring for the contract."""
    url = _build_url(path, params)
    cache_file = _cache_path(_cache_key(url))

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    if os.environ.get(OFFLINE_ENV_VAR):
        raise NetworkDisabledError(f"no cache entry for {url!r} and {OFFLINE_ENV_VAR} is set")

    body = _http_get_json(url)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def seed_cache(path: str, params: dict[str, Any] | None, body: Any) -> None:
    """Write `body` directly to the cache slot for GET `path`+`params`,
    without making a network call.

    Used by generate_dataset.py to seed get_issue's per-issue cache
    entries from data already fetched via the bulk list-issues endpoint
    (the same underlying GitHub data - the list and single-issue
    endpoints return identical issue schemas - just fetched in bulk
    rather than one live request per dataset issue, to stay well under
    the unauthenticated API's 60 requests/hour limit).
    """
    url = _build_url(path, params)
    cache_file = _cache_path(_cache_key(url))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
