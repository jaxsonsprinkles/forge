"""Turns a run's failure clusters into concrete edits to `agents/current/`.

`propose(clusters, agent_path, n=4, surface_filter=None)` looks at each
`FailureCluster` (see core/analyzer.py / core/types.py) and produces
`Mutation`s that name an exact file under `agents/current/` (see
AGENTS.md: the agent is always the fixed five files prompt.md, tools.py,
memory.py, graph.yaml, run.py) and describe the exact edit to make there.

Classification is purely heuristic (keyword matching on each cluster's
label/hypothesis), not LLM-based: it needs to be deterministic so the
proposer can be exercised in tests and ablation runs without touching
core/llm.py's cache or spend ceiling. A cluster whose text implies a
structural problem (a skipped step, a missing verification pass, losing
state across an input that spans multiple steps, ...) is classified as
"orchestration"; everything else lands on "prompt", "tool", or "memory"
based on which failure mode it best matches, defaulting to "prompt".

Two things this module guarantees about its output:
  - Surface spread: unless `surface_filter` pins the output to a single
    surface, `propose()` never returns `n` mutations that all touch the
    same file. See `_build_surface_plan`.
  - Orchestration coverage: if any cluster classifies as structural and
    "orchestration" isn't excluded by `surface_filter`, at least one
    returned mutation has surface == "orchestration".
"""

from __future__ import annotations

from typing import Literal

from core.types import FailureCluster, Mutation

Surface = Literal["prompt", "tool", "memory", "orchestration"]

# Canonical surface order, also used as the padding order when a plan
# needs more distinct surfaces than the clusters naturally classify into.
SURFACE_ORDER: tuple[Surface, ...] = ("prompt", "tool", "memory", "orchestration")

# The agent is always exactly these five files (see AGENTS.md). A
# Mutation's target_files must never name anything outside this set.
TARGET_FILE_BY_SURFACE: dict[Surface, list[str]] = {
    "prompt": ["prompt.md"],
    "tool": ["tools.py"],
    "memory": ["memory.py"],
    "orchestration": ["graph.yaml"],
}

# Keyword buckets used to classify a cluster's label+hypothesis text.
# "orchestration" keywords are deliberately specific (not generic words
# like "step", which shows up in almost every analyzer.py cluster label)
# so only genuinely structural failures pull a proposal onto graph.yaml.
_ORCHESTRATION_KEYWORDS = (
    "skip", "skips", "skipped",
    "verification", "verify", "unverified",
    "multi-page", "multi page", "multipage", "pagination",
    "before final", "before the final",
    "loses data", "lose data", "losing data", "loses state", "lose state",
    "out of order", "wrong order", "reorder",
    "missing step", "extra step", "split into two steps",
    "structural", "sequence problem", "control flow",
)
_TOOL_KEYWORDS = (
    "tool call", "tool use", "wrong tool", "no tool", "tool returns",
    "api error", "api call", "invalid argument", "invalid parameter",
    "schema mismatch", "timeout calling", "function call", "malformed call",
)
_MEMORY_KEYWORDS = (
    "memory", "forgets", "forgot", "loses context", "lost context",
    "doesn't remember", "does not remember", "prior turn", "earlier step",
    "earlier answer", "context truncat", "state not persisted", "not carried over",
)


def _classify_surface(cluster: FailureCluster) -> Surface:
    """Heuristically map a cluster's text to the surface most likely to fix it."""
    text = f"{cluster.label} {cluster.hypothesis}".lower()
    if any(kw in text for kw in _ORCHESTRATION_KEYWORDS):
        return "orchestration"
    if any(kw in text for kw in _TOOL_KEYWORDS):
        return "tool"
    if any(kw in text for kw in _MEMORY_KEYWORDS):
        return "memory"
    return "prompt"


def _examples(cluster: FailureCluster, limit: int = 3) -> str:
    ids = cluster.example_task_ids[:limit]
    return ", ".join(ids) if ids else "the affected tasks"


def _rationale(cluster: FailureCluster) -> str:
    return f"Targets the '{cluster.label}' cluster ({cluster.count} failing task(s)): {cluster.hypothesis}"


def _prompt_instruction(cluster: FailureCluster) -> str:
    return (
        f"In prompt.md, add an explicit rule addressing the '{cluster.label}' failure pattern. "
        "Insert a short paragraph near the output-format / step-by-step instructions that tells "
        "the agent, in imperative language, how to avoid it - for example: "
        f'"Before producing the final answer, explicitly check for this case: {cluster.hypothesis}" '
        f"This should eliminate the {cluster.count} failure(s) seen in tasks such as {_examples(cluster)}."
    )


def _tool_instruction(cluster: FailureCluster) -> str:
    return (
        f"In tools.py, add input/output validation to the tool implementation most relevant to "
        f"the '{cluster.label}' failure pattern. Check inputs/outputs against the exact case "
        f"described here ({cluster.hypothesis}) and raise a clear, actionable error (or normalize "
        "the input) instead of letting the call fail silently or return a malformed result. This "
        f"should fix {cluster.count} failure(s), e.g. {_examples(cluster)}."
    )


def _memory_instruction(cluster: FailureCluster) -> str:
    return (
        f"In memory.py, persist the piece of state implicated by the '{cluster.label}' failure "
        f"pattern ({cluster.hypothesis}). Add a field to the memory structure that records it as "
        "soon as it is first produced, and read that field back in later steps instead of "
        f"re-deriving or losing it. This should fix {cluster.count} failure(s), e.g. {_examples(cluster)}."
    )


def _orchestration_instruction(cluster: FailureCluster) -> str:
    return (
        f"In graph.yaml, restructure the step sequence to address the '{cluster.label}' failure "
        f"pattern ({cluster.hypothesis}). Split the step where this happens into two explicit "
        "steps - one that performs the action and a separate verification step that checks the "
        "result before the pipeline is allowed to proceed - or reorder the existing steps so that "
        f"check runs before the final-answer step. This should fix {cluster.count} failure(s), "
        f"e.g. {_examples(cluster)}."
    )


_INSTRUCTION_BY_SURFACE = {
    "prompt": _prompt_instruction,
    "tool": _tool_instruction,
    "memory": _memory_instruction,
    "orchestration": _orchestration_instruction,
}


def _build_surface_plan(classified: list[tuple[FailureCluster, Surface]], allowed: list[Surface], n: int) -> list[Surface]:
    """Pick which surface each of the n output mutations should target.

    Orders candidate surfaces as: "orchestration" first if any cluster
    classified as structural (so it survives truncation to n), then the
    other surfaces clusters actually classified into (in cluster order),
    then any remaining allowed surfaces as padding - so that unless
    `allowed` has only one member, the plan always has >= 2 distinct
    surfaces to cycle through and a single dominant cluster type can't
    monopolize every mutation.
    """
    if len(allowed) <= 1:
        return [allowed[0]] * n if allowed else []

    plan_order: list[Surface] = []
    if any(s == "orchestration" for _, s in classified) and "orchestration" in allowed:
        plan_order.append("orchestration")
    for _, s in classified:
        if s in allowed and s not in plan_order:
            plan_order.append(s)
    for s in SURFACE_ORDER:
        if s in allowed and s not in plan_order:
            plan_order.append(s)

    return [plan_order[i % len(plan_order)] for i in range(n)]


def propose(
    clusters: list[FailureCluster],
    agent_path: str,
    n: int = 4,
    surface_filter: list[str] | None = None,
) -> list[Mutation]:
    """Turn failure clusters into up to `n` concrete Mutations.

    `agent_path` isn't read (the agent's file set is fixed - see
    AGENTS.md); it's kept for signature symmetry with
    `core.analyzer.analyze` and `core.runner.run_agent`.

    `surface_filter`, when given, restricts every returned Mutation's
    surface to that list (e.g. `["prompt"]` for a prompt-only ablation).
    Otherwise all four surfaces in core.types.Mutation.surface are
    eligible, and the result is spread across more than one of them
    whenever the clusters plausibly support it.
    """
    del agent_path

    if not clusters or n <= 0:
        return []

    allowed: list[Surface] = list(surface_filter) if surface_filter else list(SURFACE_ORDER)

    sorted_clusters = sorted(clusters, key=lambda c: c.count, reverse=True)
    classified = [(c, _classify_surface(c)) for c in sorted_clusters]

    surface_plan = _build_surface_plan(classified, allowed, n)

    clusters_by_surface: dict[Surface, list[FailureCluster]] = {}
    for cluster, surface in classified:
        clusters_by_surface.setdefault(surface, []).append(cluster)

    cursor: dict[Surface, int] = {}
    mutations: list[Mutation] = []
    for i, surface in enumerate(surface_plan):
        matches = clusters_by_surface.get(surface)
        if matches:
            idx = cursor.get(surface, 0)
            cluster = matches[idx % len(matches)]
            cursor[surface] = idx + 1
        else:
            # No cluster naturally classified onto this surface (e.g. all
            # clusters are prompt-shaped but we still need spread): fall
            # back to the top cluster with a surface-specific template.
            cluster = sorted_clusters[0]

        mutations.append(
            Mutation(
                id=f"mut-{i:02d}-{surface}",
                surface=surface,
                rationale=_rationale(cluster),
                instruction=_INSTRUCTION_BY_SURFACE[surface](cluster),
                target_files=TARGET_FILE_BY_SURFACE[surface],
            )
        )

    return mutations
