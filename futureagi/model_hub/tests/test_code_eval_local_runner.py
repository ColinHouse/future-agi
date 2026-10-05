"""
Real runs of the local RestrictedPython runner.

test_code_eval_executor.py covers when the local runner is used, with
subprocess.run stubbed out. These tests send code through the runner itself,
so they catch Python the runner can't execute.
"""

import pytest

from agentic_eval.core_evals.fi_utils import sandbox


@pytest.fixture
def local_runner(settings, monkeypatch):
    """Self-hosted install with no code executor and the local opt-in set."""
    settings.CLOUD_DEPLOYMENT = ""
    settings.CODE_EXECUTOR_LOCAL_FALLBACK = True
    monkeypatch.setattr(sandbox, "CODE_EXECUTOR_URL", "")


def test_tuple_unpacking_assignment_runs(local_runner):
    code = (
        "def evaluate(output, **kwargs):\n"
        "    first, second = output.split(',')\n"
        "    return {'score': 1.0, 'reason': second + first}\n"
    )

    result = sandbox.execute_sandboxed_python(code, {"output": "a,b"}, timeout=10)

    assert result == {"status": "success", "data": {"result": 1.0, "reason": "ba"}}
