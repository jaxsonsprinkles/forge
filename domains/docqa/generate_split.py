"""Documents how split.json was produced. Not run automatically.

Random 20/10 train/holdout split over the 30 task_ids in dataset.jsonl,
using a fixed seed so the split is reproducible on demand:

    random.seed(42)
    shuffled = task_ids[:]
    random.shuffle(shuffled)
    train, holdout = sorted(shuffled[:20]), sorted(shuffled[20:])
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
DOMAIN_DIR = Path(__file__).parent


def main() -> None:
    with (DOMAIN_DIR / "dataset.jsonl").open() as f:
        task_ids = [json.loads(line)["task_id"] for line in f]

    random.seed(SEED)
    shuffled = task_ids[:]
    random.shuffle(shuffled)

    split = {"train": sorted(shuffled[:20]), "holdout": sorted(shuffled[20:])}

    with (DOMAIN_DIR / "split.json").open("w") as f:
        json.dump(split, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
