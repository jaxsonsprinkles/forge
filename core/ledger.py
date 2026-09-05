"""Appends and reads Generation records to/from ledger/generations.jsonl.

One JSON object per line (JSONL). `append_generation` is called once per
completed generation by core/loop.py and makes that generation's record
durable before returning: each call opens the file, writes the line,
flushes it, and fsyncs it to disk immediately - never buffered across
generations - so an overnight run that crashes mid-loop leaves every
already-completed generation intact and readable, even if the process
never gets to close the file handle cleanly.

`load_generations` is tolerant of a truncated last line (e.g. the process
died mid-`write`, or mid-`fsync`): any line that fails to parse as JSON is
logged and skipped rather than raising, so a corrupted trailing record
never hides the complete ones written before it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path

from core.types import Generation, Mutation, ScoreCard

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = Path("ledger/generations.jsonl")


def _scorecards_from_dict(raw: dict) -> dict[str, ScoreCard]:
    return {domain_id: ScoreCard(**card) for domain_id, card in raw.items()}


def _generation_from_dict(raw: dict) -> Generation:
    return Generation(
        gen_n=raw["gen_n"],
        parent_sha=raw["parent_sha"],
        scores_before=_scorecards_from_dict(raw["scores_before"]),
        mutations=[Mutation(**m) for m in raw["mutations"]],
        results={mid: _scorecards_from_dict(cards) for mid, cards in raw["results"].items()},
        winner_id=raw.get("winner_id"),
        winner_sha=raw.get("winner_sha"),
    )


def append_generation(gen: Generation, path: str | Path = DEFAULT_LEDGER_PATH) -> None:
    """Append one Generation record to the ledger as a single JSON line.

    Flushes and fsyncs before returning, so the record is durable on disk
    the instant this call returns rather than sitting in a buffer until
    the loop's next iteration or process exit. See module docstring.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dataclasses.asdict(gen), sort_keys=True)
    with open(p, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_generations(path: str | Path = DEFAULT_LEDGER_PATH) -> list[Generation]:
    """Load every complete Generation record from the ledger, in file order.

    Returns `[]` if the file doesn't exist yet (no generations recorded).
    A line that fails to parse as JSON (e.g. truncated by a crash mid-write)
    is logged and skipped; it never prevents earlier, complete lines from
    loading.
    """
    p = Path(path)
    if not p.exists():
        return []

    generations: list[Generation] = []
    with open(p) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed ledger line %d in %s", line_no, p)
                continue
            generations.append(_generation_from_dict(raw))
    return generations
