"""The agent's inner learning loop: a durable, inspectable store of lessons.

Memory is not a scratchpad and not conversation history. It's a persisted,
growing store of rules the agent writes about its own experience and reads
back on later runs of the *same, unchanged* agent code. Every entry is
traceable to the run and task(s) that produced it (`source_run_id`,
`evidence_task_ids`), so "show me a specific thing it learned and where it
came from" is a `load_all()` + filter away.

Storage: one JSONL file per domain, `memory/{domain_id}/entries.jsonl`,
opened in append mode only. There is no in-place edit and no database.

Append-only updates, explained
-------------------------------
`MemoryEntry.id` is stable for the life of an entry, but `confidence`,
`status`, and the `times_*` counters change over time (via `reinforce()`
and `compact()`). Since the file is append-only, an "update" is really:
append a brand new line with the same `id` and the new field values. A
read never trusts a single line in isolation - it folds every record for
a given `id` in file order and keeps the last one it sees. That fold is
the entire "mutation" mechanism: `for rec in records: latest[rec["id"]] =
rec`. This makes every write crash-safe (a torn write only ever affects
the newest, still-superseded-on-next-write record) and keeps the full
history of an entry's confidence trajectory on disk for free, without
ever needing a separate audit log.

Retrieval ranking: keyword overlap, not embeddings
---------------------------------------------------
`retrieve()` scores each active entry by what fraction of its `trigger`'s
tokens also appear in the current `task_input`, then weights that by the
entry's `confidence`. This is deterministic, requires no model call, and
needs zero extra dependencies - which matters here because retrieval must
be reproducible in tests and safe to run offline (see AGENTS.md: scorers
and their inputs must stay deterministic). Embeddings via `core/llm.py`
would add a network dependency (or another cache to reason about) for a
problem that, at this store's expected size (dozens to low hundreds of
entries per domain), keyword overlap already solves well. If retrieval
quality becomes the bottleneck at much larger scale, swapping `_relevance`
for an embedding-based scorer is a localized change - nothing else in
this module depends on how relevance is computed.

Confidence dynamics
--------------------
`reinforce(entry_id, domain_id, helped=True)` adds +0.10 to confidence;
`helped=False` subtracts 0.15. The penalty is steeper than the reward on
purpose: a lesson that actively misleads the agent should lose the
store's trust faster than a merely-unconfirmed one gains it, so bad
entries get retired quickly rather than lingering near the boundary.
Confidence is clamped to [0, 1]. Any entry whose confidence drops below
0.2 is marked `status="retired"` - never deleted, just excluded from
`retrieve()`'s results. Retired entries remain on disk and queryable via
`load_all(..., statuses=None)` so the store's growth (including its
dead ends) stays auditable.

Compaction
----------
`compact(domain_id)` finds pairs of *active* entries in the same `scope`
whose `trigger`+`content` token sets are highly similar (Jaccard >= 0.6)
and merges them: the survivor's `evidence_task_ids` becomes the union of
both, its `confidence` the max of both, and its `times_*` counters the
sum of both. The absorbed entry is not deleted - a new record for its
`id` is appended with `status="merged"`, so it stays visible in
`load_all()` and its evidence is never silently lost, just consolidated
under the survivor.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_ROOT = Path("memory")

# reinforce(): confidence deltas per call. Asymmetric on purpose - see
# module docstring ("Confidence dynamics").
REINFORCE_HELP_DELTA = 0.10
REINFORCE_HURT_DELTA = -0.15

# An entry's confidence falling below this retires it (excluded from
# retrieve(), kept on disk).
RETIREMENT_CONFIDENCE_THRESHOLD = 0.2

# compact(): minimum Jaccard similarity of (trigger + content) token sets,
# within the same domain and scope, for two active entries to be merged.
COMPACT_SIMILARITY_THRESHOLD = 0.6

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "if", "in",
        "into", "is", "it", "no", "not", "of", "on", "or", "such", "that",
        "the", "their", "then", "there", "these", "they", "this", "to",
        "was", "will", "with",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class MemoryEntry:
    """One lesson the agent has written about its own experience.

    `scope` is one of "rule" (a general behavioral instruction),
    "tool_fact" (a learned fact about a tool/API's real-world behavior),
    "exception" (a carve-out to an existing rule), or "preference" (a
    non-binding stylistic lesson). `trigger` is the plain-language
    condition under which this entry applies - it's what `retrieve()`
    matches against, not a machine-parseable predicate.
    """

    id: str
    domain_id: str
    scope: Literal["rule", "tool_fact", "exception", "preference"]
    content: str
    trigger: str
    evidence_task_ids: list[str]
    source_run_id: str
    created_gen: int
    confidence: float = 0.5
    times_retrieved: int = 0
    times_helped: int = 0
    times_hurt: int = 0
    status: Literal["active", "retired", "merged"] = "active"


def entries_path(domain_id: str, base_dir: str | Path = DEFAULT_MEMORY_ROOT) -> Path:
    """The append-only JSONL path for a domain's memory store."""
    return Path(base_dir) / domain_id / "entries.jsonl"


def snapshots_dir(domain_id: str, base_dir: str | Path = DEFAULT_MEMORY_ROOT) -> Path:
    return Path(base_dir) / domain_id / "snapshots"


def _load_records(path: Path) -> list[dict]:
    """Read every complete JSON record from `path`, in file order.

    Returns `[]` if the file doesn't exist yet. A line that fails to
    parse (e.g. a crash mid-write left a truncated trailing line) is
    logged and skipped rather than raising, so it never hides the
    complete records written before it.
    """
    if not path.exists():
        return []

    records: list[dict] = []
    with open(path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping malformed memory line %d in %s", line_no, path)
                continue
    return records


def _fold(records: list[dict]) -> dict[str, MemoryEntry]:
    """Reconstruct current state by keeping the latest record per id.

    See the module docstring's "Append-only updates" section: this fold
    is the whole update mechanism. `records` must already be in write
    order (i.e. straight from `_load_records`).
    """
    latest: dict[str, dict] = {}
    for rec in records:
        latest[rec["id"]] = rec
    return {entry_id: MemoryEntry(**rec) for entry_id, rec in latest.items()}


def write(entry: MemoryEntry, base_dir: str | Path = DEFAULT_MEMORY_ROOT) -> None:
    """Append one record for `entry` to its domain's entries.jsonl.

    This both creates a brand new entry and "updates" an existing one
    (by appending a new record under the same id - see module
    docstring). Flushes and fsyncs before returning so the write is
    durable the instant this call returns.
    """
    path = entries_path(entry.domain_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dataclasses.asdict(entry), sort_keys=True)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_all(
    domain_id: str,
    base_dir: str | Path = DEFAULT_MEMORY_ROOT,
    statuses: set[str] | None = None,
) -> list[MemoryEntry]:
    """Return current-state entries for a domain, folded from all records.

    `statuses`, when given, filters to entries whose current `status` is
    in the set (e.g. `{"active"}`). `None` (the default) returns every
    entry regardless of status - including retired and merged ones - so
    the store's full growth trajectory stays queryable.
    """
    records = _load_records(entries_path(domain_id, base_dir))
    entries = list(_fold(records).values())
    if statuses is not None:
        entries = [e for e in entries if e.status in statuses]
    return entries


def get_entry(
    entry_id: str,
    domain_id: str,
    base_dir: str | Path = DEFAULT_MEMORY_ROOT,
) -> MemoryEntry | None:
    """Look up one entry's current state by id, or None if it doesn't exist."""
    for entry in load_all(domain_id, base_dir):
        if entry.id == entry_id:
            return entry
    return None


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _task_input_tokens(task_input: dict) -> set[str]:
    return _tokenize(json.dumps(task_input, sort_keys=True, default=str))


def _relevance(trigger: str, input_tokens: set[str]) -> float:
    """Fraction of `trigger`'s tokens that also appear in the task input.

    Recall-oriented on purpose: a trigger applies when its key terms
    show up in the current task, regardless of how much *other* text the
    task input also contains.
    """
    trigger_tokens = _tokenize(trigger)
    if not trigger_tokens:
        return 0.0
    return len(trigger_tokens & input_tokens) / len(trigger_tokens)


def _rank(task_input: dict, domain_id: str, k: int, base_dir: str | Path) -> list[MemoryEntry]:
    """Shared ranking logic behind both `retrieve()` and `peek()`.

    Ranked by keyword-overlap relevance of `trigger` to `task_input`,
    weighted by `confidence` (see module docstring). Retired entries are
    never returned. Entries with zero relevance are excluded even if
    that means returning fewer than `k` results.
    """
    active = load_all(domain_id, base_dir, statuses={"active"})
    input_tokens = _task_input_tokens(task_input)

    scored = [
        (_relevance(entry.trigger, input_tokens) * entry.confidence, entry)
        for entry in active
    ]
    scored = [(score, entry) for score, entry in scored if score > 0]
    scored.sort(key=lambda pair: (-pair[0], -pair[1].confidence, pair[1].id))
    return [entry for _, entry in scored[:k]]


def retrieve(
    task_input: dict,
    domain_id: str,
    k: int,
    base_dir: str | Path = DEFAULT_MEMORY_ROOT,
) -> list[MemoryEntry]:
    """Return up to `k` active entries most relevant to `task_input`.

    Same ranking as `peek()` (see `_rank`), but each returned entry has
    its `times_retrieved` counter durably incremented as a side effect of
    being retrieved - this is the call that should represent a real,
    countable exposure of the agent to a lesson. A caller that only needs
    the ranked entries as read context - without that exposure actually
    happening - should use `peek()` instead, or `times_retrieved` will be
    inflated by lookups that were never shown to the agent.
    """
    bumped = []
    for entry in _rank(task_input, domain_id, k, base_dir):
        updated = dataclasses.replace(entry, times_retrieved=entry.times_retrieved + 1)
        write(updated, base_dir)
        bumped.append(updated)
    return bumped


def peek(
    task_input: dict,
    domain_id: str,
    k: int,
    base_dir: str | Path = DEFAULT_MEMORY_ROOT,
) -> list[MemoryEntry]:
    """Read-only equivalent of `retrieve()`: identical ranking, but never
    calls `write()` and never bumps `times_retrieved`.

    For callers that need to look at "what does memory currently say"
    without that lookup counting as a real retrieval event - e.g. holdout
    scoring (which must never mutate memory) or a step that only wants
    existing entries as dedup context for `core.reflect.reflect()` rather
    than something the agent's own prompt is built from.
    """
    return _rank(task_input, domain_id, k, base_dir)


def reinforce(
    entry_id: str,
    domain_id: str,
    helped: bool,
    base_dir: str | Path = DEFAULT_MEMORY_ROOT,
) -> None:
    """Record whether retrieving `entry_id` helped or hurt a run's outcome.

    `helped=True` raises confidence by REINFORCE_HELP_DELTA and
    increments `times_helped`; `helped=False` lowers it by
    REINFORCE_HURT_DELTA and increments `times_hurt`. Confidence is
    clamped to [0, 1]. If the new confidence drops below
    RETIREMENT_CONFIDENCE_THRESHOLD, the entry is retired (see module
    docstring's "Confidence dynamics").

    Raises KeyError if no entry with `entry_id` exists in `domain_id`.
    """
    entry = get_entry(entry_id, domain_id, base_dir)
    if entry is None:
        raise KeyError(f"no memory entry {entry_id!r} in domain {domain_id!r}")

    if helped:
        new_confidence = min(1.0, entry.confidence + REINFORCE_HELP_DELTA)
        times_helped, times_hurt = entry.times_helped + 1, entry.times_hurt
    else:
        new_confidence = max(0.0, entry.confidence + REINFORCE_HURT_DELTA)
        times_helped, times_hurt = entry.times_helped, entry.times_hurt + 1

    new_status = entry.status
    if new_status == "active" and new_confidence < RETIREMENT_CONFIDENCE_THRESHOLD:
        new_status = "retired"

    write(
        dataclasses.replace(
            entry,
            confidence=new_confidence,
            times_helped=times_helped,
            times_hurt=times_hurt,
            status=new_status,
        ),
        base_dir,
    )


def _similarity(a: MemoryEntry, b: MemoryEntry) -> float:
    """Jaccard similarity of two entries' (trigger + content) token sets."""
    tokens_a = _tokenize(a.trigger + " " + a.content)
    tokens_b = _tokenize(b.trigger + " " + b.content)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _merge(survivor: MemoryEntry, absorbed: MemoryEntry) -> MemoryEntry:
    """Fold `absorbed` into `survivor` without losing evidence or counts."""
    return dataclasses.replace(
        survivor,
        evidence_task_ids=sorted(set(survivor.evidence_task_ids) | set(absorbed.evidence_task_ids)),
        confidence=max(survivor.confidence, absorbed.confidence),
        times_retrieved=survivor.times_retrieved + absorbed.times_retrieved,
        times_helped=survivor.times_helped + absorbed.times_helped,
        times_hurt=survivor.times_hurt + absorbed.times_hurt,
        created_gen=min(survivor.created_gen, absorbed.created_gen),
    )


def compact(domain_id: str, base_dir: str | Path = DEFAULT_MEMORY_ROOT) -> None:
    """Merge near-duplicate active entries within a domain.

    Two active entries in the same `scope` are merged when their
    trigger+content token overlap is >= COMPACT_SIMILARITY_THRESHOLD.
    The survivor (lower `id`, for determinism) keeps its own id and
    content but absorbs the other's evidence_task_ids and counters (see
    `_merge`). The absorbed entry is never deleted: a new record for its
    id is appended with `status="merged"`, so `load_all(..., statuses=
    None)` still shows it and the evidence it originally carried.
    """
    entries = sorted(load_all(domain_id, base_dir, statuses={"active"}), key=lambda e: e.id)
    merged_away: set[str] = set()

    for i, anchor in enumerate(entries):
        if anchor.id in merged_away:
            continue
        survivor = anchor
        for candidate in entries[i + 1 :]:
            if candidate.id in merged_away or candidate.scope != survivor.scope:
                continue
            if _similarity(survivor, candidate) >= COMPACT_SIMILARITY_THRESHOLD:
                survivor = _merge(survivor, candidate)
                merged_away.add(candidate.id)
                write(dataclasses.replace(candidate, status="merged"), base_dir)
        if survivor is not anchor:
            write(survivor, base_dir)


def snapshot(domain_id: str, gen_n: int, base_dir: str | Path = DEFAULT_MEMORY_ROOT) -> Path:
    """Write a point-in-time copy of every current entry in `domain_id`.

    Written to memory/{domain_id}/snapshots/gen_{gen_n}.jsonl, one JSON
    object per line, sorted by id for a deterministic diff between
    snapshots. Includes entries of every status, so a later dashboard
    can chart the full composition (active/retired/merged) over time.
    Returns the path written.
    """
    entries = sorted(load_all(domain_id, base_dir), key=lambda e: e.id)
    out_dir = snapshots_dir(domain_id, base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gen_{gen_n}.jsonl"

    with open(out_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(dataclasses.asdict(entry), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())

    return out_path
