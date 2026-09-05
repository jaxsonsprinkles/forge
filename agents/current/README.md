# agents/current/

This directory is the agent Forge builds and improves - never Forge
itself (see AGENTS.md). It contains exactly five plain files, one per
mutation surface, plus this README (documentation, not a mutation
surface):

```
agents/current/
    prompt.md     # system instructions        (prompt mutations)
    tools.py      # tool implementations       (tool mutations)
    memory.py     # what persists across steps (memory mutations)
    graph.yaml    # step order and control flow (orchestration mutations)
    run.py        # entrypoint: run(task_input: dict) -> dict
```

`core/architect.validate()` enforces that this directory contains exactly
these files (plus this README) and that they're well-formed.

## How a fresh agent is built: `core.architect.build_agent()`

`build_agent(task_spec, dest)` produces a brand-new agent for a domain's
`TaskSpec` with exactly ONE `core.llm.complete()` call - not five. The
five files split into two groups:

- **Fixed scaffolding - copied byte-for-byte, never regenerated:**
  `run.py` (the generic graph interpreter below - it has no task-specific
  knowledge by construction, so a model has nothing useful to generate
  there), `memory.py` (an equally generic scratchpad), and `tools.py`
  (ships `run_python` and `search_text`, which already cover every tool
  name any shipped domain's `task_spec.tools` names).
- **Model-generated - the one thing that actually varies per domain:**
  `prompt.md` (system instructions) and `graph.yaml` (the step list). One
  `complete()` call, prompted with `task_spec.goal` and the tool names
  the fixed `tools.py` exposes, returns a JSON object with `prompt_md`
  and `graph_steps` (a list of step dicts); `build_agent()` serializes
  `graph_steps` to YAML itself via `yaml.dump` rather than asking the
  model for raw YAML text, since a model only has to get JSON right that
  way, not indentation.

The prompt explicitly asks for a **weak** first draft (typically a single
`llm_call` step): a strong first draft would leave Forge's outer mutation
loop (`core/proposer.py`, which edits exactly these five files) nothing
left to measurably improve. `build_agent()` validates its own output with
`validate()` and raises `ValueError` rather than ever handing back a
broken agent.

## The `run(task_input)` contract

- `task_input` is one dataset row's `"input"` dict (see a domain's
  `dataset.jsonl`, e.g. `{"broken_code": "...", "function_name": "..."}`
  for coderepair, `{"invoice_text": "..."}` for invoices,
  `{"question": "..."}` for docqa).
- The nominal return type is a `dict` (per AGENTS.md's
  `run(task_input: dict) -> dict`). In practice, `run()` returns whatever
  the graph's last step produced, parsed as JSON when the text looks like
  a JSON object/array, otherwise returned as plain text unchanged. This
  lets one baseline agent satisfy every domain's scorer without knowing
  which domain it's in: a JSON-object answer (invoices, docqa) comes back
  as a `dict`, while a plain-text answer (coderepair's raw Python source,
  which its scorer requires as a bare string) comes back as a `str`.
- `run()` never raises for a normal task failure (a bad LLM output, a
  failed tool call) - it just returns a worse answer. `core.runner.run_agent`
  is still the layer responsible for catching any actual exception a run
  raises and recording it on `RunResult.error`.

## graph.yaml step-type vocabulary

`graph.yaml` is a mapping with one key, `steps`: a list of step mappings,
executed in order (top to bottom) unless a `verify` step jumps backward.
Every step needs `name` (unique) and `type` (one of the three below).

`run.py`'s interpreter is fully generic over this list: it never assumes
a step count, a fixed sequence, or that a particular step name exists.
Internally it holds one dict, `_STEP_HANDLERS`, mapping each `type` string
to a handler function `(step, i, ctx) -> next_index`; the main loop just
looks up the current step's type and jumps to whatever index the handler
returns. This is why an orchestration mutation (`core/proposer.py`
inserting, splitting, reordering, or removing steps in this file) never
requires touching `run.py` - and why adding an entirely new step type
later (e.g. a future `reflect` type) only means writing one more handler
function and adding it to `_STEP_HANDLERS`, not restructuring the loop.

### `llm_call`

Sends `prompt.md` as the system message and a user message built from
`task_input` plus this step's `instruction` text (and, optionally,
`memory` context - see `input_keys`) to `core.llm.complete()`, then
stores the raw text response.

| field         | required | meaning                                                              |
|---------------|----------|-----------------------------------------------------------------------|
| `instruction` | no       | extra instruction text appended for this step                        |
| `input_keys`  | no       | list of memory keys whose values are included as JSON context        |
| `output_key`  | no       | memory key the response is stored under (defaults to the step `name`)|
| `model`       | no       | model id passed to `core.llm.complete` (defaults to `claude-sonnet-5`)|

### `tool_call`

Looks up `tool` by name in `tools.py`'s `TOOLS` dict and calls it.

| field        | required | meaning                                                          |
|--------------|----------|-------------------------------------------------------------------|
| `tool`       | yes      | key into `tools.TOOLS`                                            |
| `args`       | no       | static dict of keyword arguments                                  |
| `args_from`  | no       | memory key holding a dict of keyword arguments (overrides `args`)  |
| `output_key` | no       | memory key the tool's return value is stored under (defaults to `name`) |

A tool that raises has its exception caught and recorded as
`{"error": "..."}` in the output - a broken tool call never crashes the
run.

### `verify`

Checks one memory value and, on failure, optionally jumps back to an
earlier step for a bounded retry - the only branching construct in this
interpreter.

| field         | required | meaning                                                        |
|---------------|----------|-------------------------------------------------------------------|
| `check`       | no       | `"non_empty"` (default) or `"is_json"`                            |
| `input_key`   | yes      | memory key to check                                                |
| `on_fail`     | no       | name of an earlier step to jump back to when the check fails       |
| `max_retries` | no       | max times to take that jump for this step (default `1`)           |

If the check passes, or fails with no `on_fail` (or a retry budget
already spent), execution just continues to the next step in the list.

## memory.py

`create(task_input) -> dict` builds a fresh scratchpad once per
`run(task_input)` call; `record(memory, key, value)` is how `run.py`
persists a step's output into it. It is a plain dict, not a class, so
memory mutations are ordinary diffs to this file.

## tools.py

Exposes `TOOLS: dict[str, Callable]`. The baseline ships `run_python`
(runs a snippet in a subprocess, reporting stdout/stderr/returncode -
useful for self-checking a code-repair fix) and `search_text` (finds
matching lines in a block of text - useful for grounding a doc-QA
answer). The baseline `graph.yaml` doesn't call either by default; a
future orchestration mutation can add a `tool_call` step that does.
