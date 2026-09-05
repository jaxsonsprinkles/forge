"""A minimal deterministic scorer used by runner/scorer/run_eval tests."""

from __future__ import annotations

from typing import Any


def score_exact(output: Any, expected: Any) -> bool:
    return output == expected
