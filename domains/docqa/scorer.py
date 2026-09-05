"""Scorer for the docqa domain.

Dataset format (see dataset.jsonl): each record's `expected` is
{"answer": "<reference answer>", "source_docs": [...]} (source_docs is
metadata for humans/debugging only and is not used for scoring).

score(output, expected) -> (passed, details): `output` is the candidate's
answer. It is typically a free-form string, but since an agent's run.py
result shape isn't dictated by this domain, a dict with an "answer" key
(e.g. {"answer": "..."}) is also accepted and unwrapped; anything else is
coerced with str(). Matching is normalized fuzzy string comparison,
entirely offline and stdlib-only (no network calls, no LLM-as-judge):
both strings are lowercased, punctuation is stripped, and whitespace is
collapsed, then compared with difflib.SequenceMatcher.ratio(). `passed` is
True when the ratio clears SIMILARITY_THRESHOLD.

SIMILARITY_THRESHOLD = 0.5 was picked empirically (see tests/test_docqa.py):
the dataset's reference answers are short multi-clause sentences (e.g.
"Rowan Thornwick funded it; he was the son of founder Elias Thornwick."),
and a correct-but-differently-worded paraphrase of one typically lands
around 0.5-0.75 similarity, while an unrelated or wrong answer typically
lands below 0.3. 0.5 sits in the gap between those two clusters.

Known limitation: character-level fuzzy matching has no notion of
semantics, so on very short reference answers dominated by a single
token (e.g. "71 years"), a same-shape wrong answer (e.g. "42 years")
can score deceptively close to - or even above - a verbose-but-correct
paraphrase. This is an inherent tradeoff of offline string-similarity
scoring rather than something a threshold alone can fix; it is not
expected to bite in practice because a real candidate answer for this
dataset is either the short factual answer or a short sentence built
from the exact same wording as the source documents.

Never raises: any failure (missing/malformed `expected`, non-string
`output`, etc.) is reported as (False, details).
"""

from __future__ import annotations

import difflib
import re
from typing import Any

SIMILARITY_THRESHOLD = 0.5


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score(output: Any, expected: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Fuzzy-match a candidate answer against the expected reference answer."""
    try:
        if isinstance(output, dict) and "answer" in output:
            output = output["answer"]
        candidate = "" if output is None else output if isinstance(output, str) else str(output)
        expected_answer = expected["answer"]
        if not isinstance(expected_answer, str):
            raise TypeError(f"expected['answer'] must be a str, got {type(expected_answer).__name__}")
    except Exception as exc:  # malformed `output`/`expected` shape
        return False, {
            "similarity": 0.0,
            "threshold": SIMILARITY_THRESHOLD,
            "error": f"failed to read inputs: {exc!r}",
        }

    try:
        norm_candidate = _normalize(candidate)
        norm_expected = _normalize(expected_answer)
        similarity = difflib.SequenceMatcher(None, norm_candidate, norm_expected).ratio()
        passed = similarity >= SIMILARITY_THRESHOLD
        return passed, {"similarity": similarity, "threshold": SIMILARITY_THRESHOLD}
    except Exception as exc:  # never let scoring itself crash the caller
        return False, {
            "similarity": 0.0,
            "threshold": SIMILARITY_THRESHOLD,
            "error": f"scorer crashed: {exc!r}",
        }


def score_v1(output: Any, expected: dict[str, Any]) -> bool:
    """Bool-only wrapper of `score`, for use as a TaskSpec.scorer_id target.

    `core.runner.run_agent` calls `bool(scorer_fn(output, expected))` on
    whatever the resolved scorer_id returns. Since `score` returns a
    `(passed, details)` tuple, a non-empty tuple is always truthy under
    `bool()` regardless of `passed` - wiring `score` itself up as a
    task_spec.json's scorer_id would make every task register as passed.
    This wrapper works around that by returning just the boolean, so a
    docqa task_spec.json should point scorer_id at
    "domains.docqa.scorer:score_v1" instead of at `score` directly.
    """
    return score(output, expected)[0]
