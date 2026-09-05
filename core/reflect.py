"""Turns one completed task run into zero or more durable memory lessons.

`reflect(run_result, task_input, expected, trace, existing_memory, ...)`
is the agent's inner learning loop (see core/memory.py's module
docstring for the store itself): it shows a model the full picture of
what happened on one task - not just pass/fail, but the Neatlogs trace
steps that led there - and asks it to extract a general, transferable
lesson, if there truly is one.

Pure function, no side effects
-------------------------------
`reflect()` only returns `MemoryEntry` objects; it never calls
`core.memory.write()`, `reinforce()`, or `compact()` itself. This is a
deliberate choice among the options the spec allowed ("you may need
reflect() to also call reinforce()/write() directly, or just return
entries and let the caller persist them"): keeping this function pure
makes it trivially testable with a monkeypatched `core.llm.complete()`
and no filesystem, and it cleanly separates "what did we learn" from
"how is it persisted" - the caller (e.g. `agents/current/run.py`'s
`reflect` step handler) decides when and where to call
`core.memory.write()`.

Updates instead of duplicates
------------------------------
`existing_memory` (the entries already known going into this run,
typically whatever `core.memory.retrieve()` returned for this task) is
shown to the model, which may reference an existing entry's `id` via a
candidate's `supersedes_id` to explicitly propose revising or
contradicting it. But this is not trusted blindly: every candidate is
also checked, independent of what the model said, against
`existing_memory` using the exact same Jaccard-similarity function
`core.memory.compact()` uses to decide two entries are "the same
lesson" (`core.memory._similarity`, at `core.memory.COMPACT_SIMILARITY_
THRESHOLD`). A candidate that matches:

- and restates the match almost verbatim (see
  `NEAR_DUPLICATE_NOOP_THRESHOLD`) contributes nothing - the run
  confirmed something already known, which is not new information.
- but says something meaningfully different produces an *update*: a
  `MemoryEntry` with the matched entry's `id`, its `evidence_task_ids`
  extended, and the revised `content`/`trigger`/`confidence`. Writing
  this via `core.memory.write()` appends a new record under the same
  id, which folds over the old content per that module's append-and-
  fold mechanism - exactly the "supersede" outcome the spec asks for.

This is the whole answer to "reflecting on the same underlying cause
twice must not produce near-duplicate entries": it doesn't depend on
the model perfectly tracking `supersedes_id` across calls, because the
similarity check runs regardless of what the model said.

Freshly-minted entries (no match in `existing_memory`) get a
deterministic id derived from `(domain_id, scope, trigger, content)`
(see `_derive_id`) rather than a random one. This is a second,
independent safety net against duplicates: two `reflect()` calls that
land on the literal same lesson - e.g. because a caller forgot to pass
`existing_memory` - produce the same id, so `core.memory.write()` folds
them into one entry instead of accumulating copies.

Rejecting task-specific "lessons"
-----------------------------------
The system prompt asks for general, transferable lessons and explicitly
shows a bad example ("task_42's invoice total is $530.10") and a good
one. That instruction is not trusted alone; `_is_task_specific()` is a
post-hoc backstop that rejects a candidate if its `content`/`trigger`:

- contains `run_result.task_id` or `run_result.trace_id` verbatim
  (catches the model directly naming the identifiers it was shown), or
- matches a generic `task[-_]?<digits>` / `task[-_ ]?id` pattern
  (catches the model inventing its own-looking task reference even when
  the real `task_id` is empty or differently formatted - the in-graph
  `reflect` step in `agents/current/run.py`, in particular, never has a
  real `task_id` to check against), or
- restates `expected` verbatim when `expected` is a non-trivial string
  (catches memorizing the graded answer rather than a pattern behind it).

None of these are perfect (a false negative is always possible for a
sufficiently creative bad lesson), but they catch the shapes of
memorization the spec calls out, cheaply and deterministically.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
from typing import Any

from core import llm
from core import memory
from core.memory import MemoryEntry
from core.types import RunResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"

# Above this Jaccard similarity (see core.memory._similarity) to an
# existing entry's content alone, a candidate is treated as restating
# something already known rather than adding new information.
NEAR_DUPLICATE_NOOP_THRESHOLD = 0.85

_ALLOWED_SCOPES = {"rule", "tool_fact", "exception", "preference"}

# Catches a model inventing its own task-reference-shaped text (e.g. the
# spec's bad example "task_42's invoice total...") even when the real
# task_id isn't available to check against verbatim.
_GENERIC_TASK_REF_RE = re.compile(r"\btask[-_ ]?(id)?[-_ ]?#?\d+\b", re.IGNORECASE)

_SYSTEM_PROMPT = """You are the reflection step of a learning coding agent.

You will be shown one completed task run: its input, what happened, and
(when available) a step-by-step trace and the graded expected answer.
Your job is to decide whether this run revealed a GENERAL, TRANSFERABLE
lesson worth remembering for future, DIFFERENT tasks of the same kind -
not just this one.

Reflect on successes as well as failures. A run that worked because of a
non-obvious choice is a lesson too, not just a mistake to avoid.

A lesson must generalize. Reject anything that only makes sense for this
one task's specific identifiers or exact values.

Bad (task-specific, memorization - reject this shape entirely):
  "task_42's invoice total is $530.10."
Good (general, transferable):
  "when an invoice shows two currencies, the total is usually stated in
  the currency listed last, not first."

You will also be shown memory entries already known before this run. Do
not restate them. If this run confirms one is still correct, say
nothing about it. If this run shows one is wrong, incomplete, or needs a
carve-out, propose a revision by setting "supersedes_id" to that entry's
id and writing the corrected "content"/"trigger".

Respond with ONLY a JSON array (no markdown fences, no commentary
outside the array). Each element is an object:
  {
    "content": "the lesson itself, as an instruction or fact",
    "trigger": "the general situation in which it applies",
    "scope": "rule" | "tool_fact" | "exception" | "preference",
    "supersedes_id": "<id of an existing entry this revises, or null>",
    "confidence": 0.0-1.0 (optional; how sure you are this generalizes)
  }

If there is nothing genuinely new to remember from this run, respond
with an empty array: []
"""


def _build_messages(
    run_result: RunResult,
    task_input: dict[str, Any],
    expected: Any,
    trace: list[dict[str, Any]] | None,
    existing_memory: list[MemoryEntry],
) -> list[dict[str, str]]:
    outcome = {
        "task_id": run_result.task_id,
        "passed": run_result.passed,
        "error": run_result.error,
        "output": run_result.output,
    }
    existing_summary = [
        {"id": e.id, "scope": e.scope, "trigger": e.trigger, "content": e.content, "confidence": e.confidence}
        for e in existing_memory
    ]
    user_content = (
        f"Task input:\n{json.dumps(task_input, indent=2, default=str)}\n\n"
        f"Expected answer (ground truth, for reflection only):\n{json.dumps(expected, indent=2, default=str)}\n\n"
        f"What happened (this run's outcome):\n{json.dumps(outcome, indent=2, default=str)}\n\n"
        f"Trace steps, in order (may be empty if no trace was available):\n"
        f"{json.dumps(trace or [], indent=2, default=str)}\n\n"
        f"Memory entries already known before this run:\n{json.dumps(existing_summary, indent=2)}\n\n"
        "Now decide what, if anything, is worth remembering from this run."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _try_parse_json_array(text: str) -> list[Any] | None:
    """Best-effort JSON-array parse, tolerating stray prose around the array."""
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, list) else None
    except (json.JSONDecodeError, ValueError):
        pass

    start, end = stripped.find("["), stripped.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(stripped[start : end + 1])
        return parsed if isinstance(parsed, list) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _normalize_candidate(raw: Any) -> tuple[str, str, str, str | None, float | None] | None:
    """Validate and coerce one raw candidate dict, or None if malformed."""
    if not isinstance(raw, dict):
        return None

    content = str(raw.get("content", "")).strip()
    trigger = str(raw.get("trigger", "")).strip()
    if not content or not trigger:
        return None

    scope = raw.get("scope")
    if scope not in _ALLOWED_SCOPES:
        scope = "rule"

    supersedes_id = raw.get("supersedes_id")
    supersedes_id = supersedes_id.strip() if isinstance(supersedes_id, str) and supersedes_id.strip() else None

    confidence_raw = raw.get("confidence")
    confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))

    return content, trigger, scope, supersedes_id, confidence


def _mentions_task_identity(text: str, run_result: RunResult) -> bool:
    lowered = text.lower()
    if run_result.task_id and len(run_result.task_id) >= 3 and run_result.task_id.lower() in lowered:
        return True
    if run_result.trace_id and len(run_result.trace_id) >= 6 and run_result.trace_id.lower() in lowered:
        return True
    return bool(_GENERIC_TASK_REF_RE.search(text))


def _restates_expected_verbatim(text: str, expected: Any) -> bool:
    if not isinstance(expected, str):
        return False
    stripped = expected.strip()
    if len(stripped) < 8:
        return False
    return stripped.lower() in text.lower()


def _is_task_specific(content: str, trigger: str, run_result: RunResult, expected: Any) -> bool:
    """Post-hoc backstop rejecting candidates shaped like memorization
    rather than a general lesson - see module docstring."""
    combined = f"{trigger} {content}"
    return _mentions_task_identity(combined, run_result) or _restates_expected_verbatim(combined, expected)


def _derive_id(domain_id: str, scope: str, trigger: str, content: str) -> str:
    """Deterministic id for a brand-new entry - see module docstring's
    'Updates instead of duplicates' section for why this isn't random."""
    digest = hashlib.sha256(f"{domain_id}\x1f{scope}\x1f{trigger}\x1f{content}".encode()).hexdigest()
    return f"mem-{digest[:16]}"


def _find_similar_entry(
    candidate_probe: MemoryEntry, existing_memory: list[MemoryEntry]
) -> MemoryEntry | None:
    """Best matching existing entry of the same scope, if similar enough
    to be considered the same underlying lesson (see COMPACT_SIMILARITY_
    THRESHOLD, reused from core.memory so "near-duplicate" means the same
    thing here as it does in memory.compact())."""
    same_scope = [e for e in existing_memory if e.scope == candidate_probe.scope]
    if not same_scope:
        return None
    best = max(same_scope, key=lambda e: memory._similarity(candidate_probe, e))
    if memory._similarity(candidate_probe, best) >= memory.COMPACT_SIMILARITY_THRESHOLD:
        return best
    return None


def _content_only_similarity(a: MemoryEntry, b: MemoryEntry) -> float:
    """Like core.memory._similarity, but ignoring `trigger` - used to tell
    "restates the same content" apart from "matched mostly on trigger
    text but says something new" (which should become an update, not a
    silent no-op)."""
    return memory._similarity(dataclasses.replace(a, trigger=""), dataclasses.replace(b, trigger=""))


def _resolve_entry(
    *,
    content: str,
    trigger: str,
    scope: str,
    supersedes_id: str | None,
    confidence: float | None,
    existing_by_id: dict[str, MemoryEntry],
    existing_memory: list[MemoryEntry],
    run_result: RunResult,
    domain_id: str,
    gen_n: int,
) -> MemoryEntry | None:
    candidate_probe = MemoryEntry(
        id="__candidate__",
        domain_id=domain_id,
        scope=scope,
        content=content,
        trigger=trigger,
        evidence_task_ids=[],
        source_run_id="",
        created_gen=gen_n,
    )

    match: MemoryEntry | None = None
    forced_update = False
    if supersedes_id and supersedes_id in existing_by_id:
        match = existing_by_id[supersedes_id]
        forced_update = True
    else:
        match = _find_similar_entry(candidate_probe, existing_memory)

    if match is not None:
        if not forced_update and _content_only_similarity(candidate_probe, match) >= NEAR_DUPLICATE_NOOP_THRESHOLD:
            # Nothing new: this run just confirmed what was already known.
            return None
        evidence = set(match.evidence_task_ids)
        if run_result.task_id:
            evidence.add(run_result.task_id)
        return dataclasses.replace(
            match,
            content=content,
            trigger=trigger,
            scope=scope,
            evidence_task_ids=sorted(evidence),
            confidence=confidence if confidence is not None else match.confidence,
        )

    source_run_id = run_result.trace_id or run_result.task_id or "unknown-run"
    evidence_task_ids = [run_result.task_id] if run_result.task_id else []
    return MemoryEntry(
        id=_derive_id(domain_id, scope, trigger, content),
        domain_id=domain_id,
        scope=scope,
        content=content,
        trigger=trigger,
        evidence_task_ids=evidence_task_ids,
        source_run_id=source_run_id,
        created_gen=gen_n,
        confidence=confidence if confidence is not None else 0.5,
    )


def reflect(
    run_result: RunResult,
    task_input: dict[str, Any],
    expected: Any,
    trace: list[dict[str, Any]] | None,
    existing_memory: list[MemoryEntry],
    *,
    domain_id: str,
    gen_n: int = 0,
    model: str = DEFAULT_MODEL,
) -> list[MemoryEntry]:
    """Reflect on one completed task run, returning new/updated lessons.

    `run_result` is the graded outcome of the run (see core.types.RunResult);
    `expected` is the dataset row's graded answer; `trace` is the Neatlogs
    trace step data for this run's task (the same shape core.analyzer
    fetches - a list of step dicts with at least "name"/"error" - or None
    if no trace is available); `existing_memory` is whatever
    core.memory.retrieve() returned for this task before it ran.

    `domain_id` and `gen_n` aren't part of the run's own data (RunResult
    has no domain or generation field) but are required to construct a
    valid MemoryEntry, so they're required keyword arguments here.

    Returns an empty list when there's nothing genuinely new to say -
    either because the model found nothing worth remembering, every
    candidate merely restated existing_memory, or every candidate was
    rejected as task-specific. Never raises for a malformed model
    response; degrades to an empty list instead (mirroring this
    codebase's other best-effort model-output parsing, e.g.
    core/architect.py's build_agent).
    """
    messages = _build_messages(run_result, task_input, expected, trace, existing_memory)
    text, _cost_usd, _latency_ms = llm.complete(messages, model=model)

    raw_candidates = _try_parse_json_array(text)
    if not raw_candidates:
        return []

    existing_by_id = {e.id: e for e in existing_memory}
    entries: list[MemoryEntry] = []
    seen_ids: set[str] = set()

    for raw in raw_candidates:
        normalized = _normalize_candidate(raw)
        if normalized is None:
            continue
        content, trigger, scope, supersedes_id, confidence = normalized

        if _is_task_specific(content, trigger, run_result, expected):
            logger.info("reflect(): rejected a task-specific candidate lesson for domain %r", domain_id)
            continue

        entry = _resolve_entry(
            content=content,
            trigger=trigger,
            scope=scope,
            supersedes_id=supersedes_id,
            confidence=confidence,
            existing_by_id=existing_by_id,
            existing_memory=existing_memory,
            run_result=run_result,
            domain_id=domain_id,
            gen_n=gen_n,
        )
        if entry is None or entry.id in seen_ids:
            continue
        seen_ids.add(entry.id)
        entries.append(entry)

    return entries
