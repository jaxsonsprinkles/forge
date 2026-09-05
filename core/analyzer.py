"""Groups a completed eval run's failures into human-readable patterns.

`analyze(results, agent_path)` looks at every failed `RunResult` (see
core/types.py), pulls its Neatlogs trace when one is available, and
clusters failures that share a root cause into a single `FailureCluster`
instead of reporting one entry per task.

Trace fetching reuses the same best-effort Neatlogs client that
core.runner sets up (`runner._neatlogs_init`): if the SDK isn't
installed, no API key is configured, or the fetch call fails for any
reason, this degrades to `None` and the affected failures are grouped
by error/output pattern alone. Neatlogs doesn't expose a documented
"fetch trace by id" call anywhere else in this codebase, so this
assumes a `client.get_trace(trace_id)` method returning an object (or
dict) with a `steps` sequence describing each agent step's
name/input/output/error - the same shape runner.py's spans would
produce. That assumption only matters when a real provider is wired
up; today (and in tests) it's exercised entirely through monkeypatched
seams, exactly like runner.py's own Neatlogs calls.

`FailureCluster` has no `confidence` field, and this module doesn't add
one: clusters built from trace data note that in their hypothesis, and
clusters that fall back to error/output-pattern grouping (no trace, or
trace fetch failed) say so explicitly in their hypothesis text instead.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core import runner as _runner
from core.types import FailureCluster, RunResult

logger = logging.getLogger(__name__)

# Cap on how many clusters analyze() returns; smaller clusters beyond this
# are merged into one trailing "miscellaneous" cluster rather than dropped,
# so no failing task silently disappears from the report.
MAX_CLUSTERS = 6

_LABEL_MAX_LEN = 70
_TEXT_BUCKET_LEN = 80


def _neatlogs_client() -> Any | None:
    """Best-effort Neatlogs client, reusing core.runner's init/degrade logic."""
    try:
        return _runner._neatlogs_init()
    except Exception:
        logger.warning("neatlogs client init failed during analysis", exc_info=True)
        return None


def _coerce_step(step: Any) -> dict[str, Any] | None:
    """Normalize one trace step (dict or SDK object) to a plain dict, or None."""
    if isinstance(step, dict):
        name = step.get("name")
        error = step.get("error")
    else:
        name = getattr(step, "name", None)
        error = getattr(step, "error", None)
    if name is None:
        return None
    return {"name": name, "error": error}


def _fetch_trace_steps(client: Any, trace_id: str) -> list[dict[str, Any]] | None:
    """Best-effort fetch of a trace's step data. Returns None on any failure."""
    try:
        trace = client.get_trace(trace_id)
    except Exception:
        logger.warning("neatlogs get_trace(%r) failed", trace_id, exc_info=True)
        return None

    steps = trace.get("steps") if isinstance(trace, dict) else getattr(trace, "steps", None)
    if not steps:
        return None

    coerced = [s for s in (_coerce_step(step) for step in steps) if s is not None]
    return coerced or None


def _normalize_text(text: Any, max_len: int = _TEXT_BUCKET_LEN) -> str:
    """Collapse a message/output into a rough similarity bucket.

    Digits are folded to '#' so messages that differ only by an index,
    id, or count (e.g. "index 5" vs "index 9") land in the same bucket.
    """
    normalized = re.sub(r"\d+", "#", str(text))
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized[:max_len]


def _parse_error(error: str) -> tuple[str, str]:
    """Split a runner-style '{ExcType}: {message}' string into its parts."""
    exc_type, sep, message = error.partition(":")
    if not sep:
        return "Error", error.strip()
    return exc_type.strip(), message.strip()


def _divergence_step(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the step where a trace first shows a problem, else the last step."""
    for step in steps:
        if step.get("error"):
            return step
    return steps[-1]


def _classify(result: RunResult) -> tuple[str, str]:
    """Return (kind, normalized_bucket) describing how a result failed."""
    if result.error:
        exc_type, message = _parse_error(result.error)
        return exc_type, _normalize_text(message)
    return "BadOutput", _normalize_text(repr(result.output))


def _signature(result: RunResult, steps: list[dict[str, Any]] | None) -> tuple[Any, ...]:
    """Grouping key for one failed result. Same signature -> same cluster."""
    kind, bucket = _classify(result)
    if steps:
        step = _divergence_step(steps)
        return ("step", step["name"], kind)
    return ("pattern", kind, bucket)


def _label_for(signature: tuple[Any, ...], kind: str, bucket: str) -> str:
    if signature[0] == "step":
        _, step_name, _kind = signature
        label = f"fails at '{step_name}' step ({kind})"
    else:
        label = f"{kind}: {bucket}" if bucket else kind
    return label[:_LABEL_MAX_LEN]


def _hypothesis_for(signature: tuple[Any, ...], count: int, kind: str, bucket: str) -> str:
    if signature[0] == "step":
        step_name = signature[1]
        return (
            f"{count} task(s) diverge at the '{step_name}' step with a {kind}-shaped failure, "
            "based on where the trace first shows a problem - likely a bug in that step's "
            "handling of this input pattern."
        )
    return (
        f"{count} task(s) fail with the same {kind} pattern ('{bucket}'), suggesting a shared "
        "root cause in the agent. No trace data was available to localize which step causes it "
        "(low confidence: grouped by error/output pattern only)."
    )


def _merge_overflow(clusters: list[FailureCluster]) -> list[FailureCluster]:
    """Keep the largest (MAX_CLUSTERS - 1) clusters, merge the rest into one."""
    kept, overflow = clusters[: MAX_CLUSTERS - 1], clusters[MAX_CLUSTERS - 1 :]
    merged = FailureCluster(
        label="other failures (miscellaneous, below clustering threshold)",
        count=sum(c.count for c in overflow),
        example_task_ids=[task_id for c in overflow for task_id in c.example_task_ids][:5],
        trace_ids=[trace_id for c in overflow for trace_id in c.trace_ids],
        hypothesis=(
            f"{len(overflow)} smaller failure pattern(s) merged here to stay within the "
            f"{MAX_CLUSTERS}-cluster cap; inspect example_task_ids individually for root causes."
        ),
    )
    merged_list = kept + [merged]
    merged_list.sort(key=lambda c: c.count, reverse=True)
    return merged_list


def analyze(results: list[RunResult], agent_path: str) -> list[FailureCluster]:
    """Cluster a run's failures into patterns, sorted by count descending.

    `agent_path` isn't needed to fetch traces (those are keyed by
    `trace_id`, already stored on each RunResult); it's kept for
    signature symmetry with `core.runner.run_agent` and as a hook for
    future agent-config-aware analysis.

    Failures (passed is False, or error is not None) that share a
    Neatlogs trace divergence point and failure kind collapse into one
    cluster; failures without usable trace data fall back to grouping
    by normalized error/output pattern, noted as lower confidence in
    the hypothesis text. Returns at most MAX_CLUSTERS clusters - beyond
    that, the smallest are merged into one trailing cluster rather than
    dropped.
    """
    del agent_path

    failed = [r for r in results if r.passed is False or r.error is not None]
    if not failed:
        return []

    client = _neatlogs_client()

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for result in failed:
        steps = _fetch_trace_steps(client, result.trace_id) if client is not None and result.trace_id else None
        kind, bucket = _classify(result)
        signature = _signature(result, steps)

        group = groups.setdefault(
            signature,
            {"results": [], "trace_ids": [], "kind": kind, "bucket": bucket},
        )
        group["results"].append(result)
        if result.trace_id:
            group["trace_ids"].append(result.trace_id)

    clusters = [
        FailureCluster(
            label=_label_for(signature, group["kind"], group["bucket"]),
            count=len(group["results"]),
            example_task_ids=[r.task_id for r in group["results"][:3]],
            trace_ids=group["trace_ids"],
            hypothesis=_hypothesis_for(signature, len(group["results"]), group["kind"], group["bucket"]),
        )
        for signature, group in groups.items()
    ]
    clusters.sort(key=lambda c: c.count, reverse=True)

    if len(clusters) > MAX_CLUSTERS:
        clusters = _merge_overflow(clusters)

    return clusters
