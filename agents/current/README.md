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
