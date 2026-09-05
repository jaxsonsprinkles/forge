import json
import subprocess
import sys
from pathlib import Path

import pytest

from core import llm
from evals.run_eval import load_task_spec, main

REPO_ROOT = Path(__file__).parent.parent
FIXTURES_DOMAINS_ROOT = str(Path(__file__).parent / "fixtures" / "domains")
GOOD_AGENT = str(Path(__file__).parent / "fixtures" / "agents" / "good_agent")
BROKEN_AGENT = str(Path(__file__).parent / "fixtures" / "agents" / "broken_agent")


@pytest.fixture(autouse=True)
def _reset_spend_tracker():
    llm.reset_spend_tracker()
    yield
    llm.reset_spend_tracker()


def test_load_task_spec_reads_json_by_convention():
    task_spec = load_task_spec("dummy", domains_root=FIXTURES_DOMAINS_ROOT)

    assert task_spec.domain_id == "dummy"
    assert task_spec.scorer_id == "tests.fixtures.scorer:score_exact"
    assert task_spec.max_tasks == 20


def test_main_prints_scorecard_json_for_good_agent(capsys, monkeypatch):
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)

    exit_code = main(
        [
            "--domain",
            "dummy",
            "--agent",
            GOOD_AGENT,
            "--split",
            "train",
            "--domains-root",
            FIXTURES_DOMAINS_ROOT,
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    card = json.loads(captured.out.strip())
    assert card["accuracy"] == 1.0
    assert card["n"] == 3
    assert card["split"] == "train"


def test_main_never_raises_for_broken_agent(capsys, monkeypatch):
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)

    exit_code = main(
        [
            "--domain",
            "dummy",
            "--agent",
            BROKEN_AGENT,
            "--split",
            "train",
            "--domains-root",
            FIXTURES_DOMAINS_ROOT,
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    card = json.loads(captured.out.strip())
    assert card["accuracy"] == 0.0
    assert card["reliability"] == 0.0


def test_cli_stdout_is_clean_json_only():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.run_eval",
            "--domain",
            "dummy",
            "--agent",
            GOOD_AGENT,
            "--split",
            "train",
            "--domains-root",
            FIXTURES_DOMAINS_ROOT,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    # stdout must be exactly one line of valid JSON - no logs, no warnings.
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    card = json.loads(lines[0])
    assert card["n"] == 3
