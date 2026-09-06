# forge

Forge builds and improves AI agents automatically. You give it a task spec
(a goal, some tools, a scorable dataset); it writes an agent, runs it,
looks at what failed, proposes a change, tests the change, and keeps it if
it measurably helped. Repeat for generations.

The one thing worth understanding before anything else: Forge has **two
learning loops**, and they change different things.

- **Outer loop** — between generations, Forge edits the agent's own
  source: its prompt, its tools, its memory policy, its orchestration
  graph. A failure analysis turns into a set of candidate diffs
  (`core/proposer.py`), each is dispatched to a parallel worker
  (`core/executor_ao.py`), re-scored, and the best one wins
  (`core/gate.py`). This changes the agent's *code*.
- **Inner loop** — while a single generation of the agent is running, it
  reflects on each completed task, writes durable lessons to a memory
  store (`core/memory.py`, `core/reflect.py`), and reads relevant lessons
  back on later tasks. This changes the agent's *knowledge*, with zero
  code edits.

The outer loop was built first and is the easier of the two to believe in
— diffing code and re-scoring it is a familiar idea. The inner loop is the
newer, harder claim: **the same agent code, run repeatedly on the same
kind of task, gets measurably better because its memory grew.** If
accuracy only ever moves when code changes, the inner loop doesn't exist.
See `AGENTS.md` for the full philosophy, including why memory has to be a
first-class, traceable artifact rather than a scratchpad, and why the
agent is required to learn a third party's *conventions* rather than its
schema.

Forge itself lives in `core/`. It's normal software and nobody is trying
to optimize it. The thing being optimized is always `agents/current/`.

## Setup

```
git clone https://github.com/jaxsonsprinkles/forge.git
cd forge
pip install -r requirements.txt
```

Needs Python 3.10–3.13 (see Rough Edges below for why). Nothing that
actually calls a model will run until you set:

- `ANTHROPIC_API_KEY` — required. `core/llm.py` reads it straight out of
  `os.environ` when it builds the provider client, and every model call
  in this repo goes through `core/llm.py` (that's a hard rule, not just
  the common path — see `AGENTS.md`). No key, no `llm_call` steps,
  no `evals/run_eval.py`, no `evals/learning_curve.py`.
- `NEATLOGS_API_KEY` — optional. It's only for tracing. `core/runner.py`
  checks for it, logs a warning, and degrades to `trace_id=None` if it's
  missing — a run works fine without it, you just don't get spans.

Confirm the install actually works before spending anything: run the test
suite. It's fully offline — LLM calls in tests are mocked or hit the
on-disk cache under `ledger/llm_cache/`, so this needs no API key and
costs nothing real.

```
python -m pytest
```

Green from a clean checkout is literally this project's own definition of
done (`AGENTS.md`). Once that passes and you've exported
`ANTHROPIC_API_KEY`, you're ready for a real (billed) eval run — see
"Running an eval" below.

## The five-file agent contract

`agents/current/` is exactly five plain files, one per mutation surface.
Not a class, not a package, not a config object — a coding agent (human
or AO worker) has to be able to edit these as ordinary files and have git
record each change as a normal diff.

```
agents/current/
    prompt.md     # system instructions         (prompt mutations)
    tools.py      # tool implementations         (tool mutations)
    memory.py     # per-run scratchpad           (memory mutations)
    graph.yaml    # step order and control flow  (orchestration mutations)
    run.py        # entrypoint: run(task_input: dict) -> dict
```

`run.py` is a generic graph interpreter — it reads `graph.yaml` for step
order, `prompt.md` for the system message, `tools.py` for callable tools,
and drives `core/llm.py` through whatever steps are listed. It never
assumes a step count or a fixed sequence, so an orchestration mutation can
insert, split, reorder, or delete steps in `graph.yaml` without ever
touching `run.py`. Step types today: `llm_call`, `tool_call`, `verify`
(bounded retry), `recall` (the inner loop's read path —
`core.memory.retrieve()`), and `reflect` (the inner loop's write path —
`core.reflect.reflect()` + `core.memory.write()`). `core/architect.py`
enforces the five-file shape and can scaffold a fresh baseline agent from
a `TaskSpec` with a single `llm.complete()` call.

Don't confuse `agents/current/memory.py` (a plain dict scratchpad, lives
and dies with one `run()` call) with `core/memory.py` (the persistent,
cross-run store the `recall`/`reflect` steps talk to). Same word, two very
different lifetimes — see `agents/current/README.md` for the full step
schema if you're writing a mutation.

## Running an eval

Score one agent against one domain's holdout split:

```
python evals/run_eval.py --domain github_triage --agent agents/current --split holdout
```

Prints a `ScoreCard` as JSON on stdout (accuracy, cost, latency,
reliability). This is what CI calls per candidate during the outer loop's
gate step.

The inner-loop harness is `evals/learning_curve.py` — it runs the *same*
agent code over a domain's train split `--passes` times, letting memory
persist and reflection fire between passes, then scores the identical
agent against holdout after every pass:

```
python evals/learning_curve.py --domain github_triage --passes 5 --memory on
python evals/learning_curve.py --domain github_triage --passes 5 --memory off
```

`--memory off` is the control arm: same code, same dataset, same pass
count, but the `recall`/`reflect` steps are switched off via
`FORGE_MEMORY_MODE=off` rather than by editing `graph.yaml`. Each
`(domain, mode)` pair gets its own scratch memory store under
`memory/_learning_curve/<domain>_memory_<mode>/`, wiped at the start of
the run unless you pass `--resume`. Output is one JSON line per pass,
appended to `ledger/learning_curves/<domain>_memory_<mode>.jsonl`. If `on`
climbs pass over pass while `off` stays flat, that gap is the actual
deliverable — not any single accuracy number.

## The dashboard

`dashboard/index.html` reads straight from `ledger/` and `memory/` and
renders learning curves, per-pass scorecards, and a searchable view of
every memory entry (content, trigger, source run, retrieval count). No
build step, no server — open the file directly in a browser. It tries
`fetch()` first and falls back to `XMLHttpRequest` for the case where
`file://` gets treated as an opaque origin and `fetch` is blocked by CORS.
If you'd rather serve it, any static file server pointed at the repo root
works too, but it isn't required.

## Does the memory loop actually work?

Short answer: on the one domain that's actually been run through a full
curve, yes, but the margin is small and the run is short enough that you
shouldn't read much into the shape of it yet.

`github_triage` is the only domain with a real 3-pass curve committed
(`ledger/learning_curves/github_triage_memory_{on,off}.jsonl`), against a
14-task holdout split — worth saying up front, because with 14 tasks
every point of accuracy is worth ~7 percentage points, so a two-task swing
looks like a big jump on a chart and isn't one.

| pass | memory ON — holdout acc. | memory ON — entries in store | memory OFF — holdout acc. |
|------|---------------------------|-------------------------------|-----------------------------|
| 0    | 28.6% (4/14)              | 44                             | 28.6% (4/14)                |
| 1    | 42.9% (6/14)              | 65                             | 28.6% (4/14)                |
| 2    | 35.7% (5/14)              | 80                             | 28.6% (4/14)                |

Memory-off is exactly flat across all three passes — not approximately,
exactly, down to the millisecond breakdown giving it away: p50 holdout
latency drops from 5929ms at pass 0 to 3ms at pass 1 and 2ms at pass 2.
That's `core/llm.py`'s on-disk call cache doing its job — with memory off,
`task_input` is byte-identical every pass, so every call after the first
is a cache hit, not a re-inference. That's expected behavior for the
control arm, documented in `evals/learning_curve.py`, and it's also a
decent sanity check that the harness is wired correctly.

Memory-on isn't flat: the store grows from 44 to 65 to 80 entries,
retrieval settles at the `k=5` cap after pass 0, and holdout accuracy goes
28.6% → 42.9% → 35.7% — up overall, but it gives a task back between pass
1 and pass 2 rather than climbing monotonically. Read that as "the
separation the milestone asked for is present" (on ends 7 points above a
perfectly flat off arm) and not as "memory makes this agent strictly
better every pass." Three passes on one domain is a demonstration that
the mechanism works, not a learning curve with a stable slope.

Two things not to trust yet:

- **Cost per task reads 0.0 for every pass, both arms.** This isn't the
  memory loop being free — it's a pricing bug. `core/llm.py`'s
  `PRICING_PER_MTOK` had no entry for `claude-sonnet-4-6` (the agent's
  `DEFAULT_MODEL`) at the time this ledger was generated, so
  `estimate_cost_usd()` silently priced every call at $0. The fix is
  written (`fix(llm): add missing pricing entry for claude-sonnet-4-6`,
  branch `ao/forge-38/fix-sonnet-4-6-pricing`) but hasn't landed on `main`
  as of this ledger, so the cost columns here predate it. Don't cite them
  as evidence of anything until the curve is re-run against the fixed
  pricing table.
- **`coderepair` has no curve yet.** `domains/coderepair/` has a real
  dataset, scorer, and task spec, but
  `ledger/learning_curves/coderepair_memory_{on,off}.jsonl` are both
  empty — nobody has run the full learning-curve harness against it. It
  exists as a second domain to prove the memory mechanism generalizes
  past github_triage, not as a second result.

## Repo layout

```
core/        Forge itself: llm.py (the only thing allowed to call a model
             provider, cached to disk by request hash), memory.py,
             reflect.py, runner.py, scorer.py, analyzer.py, proposer.py,
             gate.py, executor_ao.py, architect.py, loop.py, ledger.py
agents/      current/ is the live agent; archive/ holds past generations
domains/     one dir per task domain: task_spec.json, dataset.jsonl,
             split.json, scorer.py, plus any domain-specific tools
evals/       run_eval.py (score once) and learning_curve.py (the
             memory-on/off harness)
ledger/      append-only records: generations, learning curves, LLM cache
dashboard/   dashboard/index.html — open it, no build step
tests/       python -m pytest
```

## Rough edges

- Python 3.10–3.13 only (Neatlogs' range).
- The outer loop (`core/loop.py`) has no CLI wrapper yet — it's driven
  programmatically / by tests and AO orchestration, not a `forge run`
  command.
- The `recall`/`reflect` steps in the baseline `graph.yaml` are wired for
  `github_triage` specifically (hardcoded `domain_id`, github-shaped
  `fetch_*` tool calls). On other domains those tool calls no-op cleanly
  instead of doing anything useful — there's no per-domain graph variant
  yet.
- `evals/learning_curve.py`'s concurrency guard against two runs hitting
  the same memory store is an `flock`, so it's POSIX-only.
- Scorers must stay deterministic and offline; third-party API calls
  (e.g. the GitHub tools in `domains/github_triage/tools/`) go through a
  record/replay cache under `domains/github_triage/cache/` so a rerun
  never gets a different score because a live API answered differently.
