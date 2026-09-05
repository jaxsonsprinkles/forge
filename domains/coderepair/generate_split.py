"""Documents how split.json was produced. Not run automatically.

Random 20/10 train/holdout split over the 30 task_ids in dataset.jsonl,
using a fixed seed so the split is reproducible on demand:

    random.seed(42)
    shuffled = task_ids[:]
    random.shuffle(shuffled)
    train, holdout = sorted(shuffled[:20]), sorted(shuffled[20:])

core.runner's dataset loader filters on an inline "split" field on each
dataset.jsonl row (not on split.json, which it never reads), so this
also rewrites dataset.jsonl with that field set to match split.json -
the same convention domains/invoices and domains/github_triage use.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
DOMAIN_DIR = Path(__file__).parent


def main() -> None:
    with (DOMAIN_DIR / "dataset.jsonl").open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    task_ids = [row["task_id"] for row in rows]

    random.seed(SEED)
    shuffled = task_ids[:]
    random.shuffle(shuffled)

    train, holdout = sorted(shuffled[:20]), sorted(shuffled[20:])
    split_of = {tid: "train" for tid in train}
    split_of.update({tid: "holdout" for tid in holdout})

    with (DOMAIN_DIR / "dataset.jsonl").open("w") as f:
        for row in rows:
            row["split"] = split_of[row["task_id"]]
            f.write(json.dumps(row) + "\n")

    with (DOMAIN_DIR / "split.json").open("w") as f:
        json.dump({"train": train, "holdout": holdout}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
