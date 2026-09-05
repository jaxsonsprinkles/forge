"""A minimal well-behaved fake agent used in tests: sums two numbers."""


def run(task_input: dict) -> dict:
    return {"answer": task_input["a"] + task_input["b"]}
