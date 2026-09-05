"""A deliberately broken fake agent used to prove run_agent never crashes the run."""


def run(task_input: dict) -> dict:
    raise RuntimeError("this agent always crashes")
