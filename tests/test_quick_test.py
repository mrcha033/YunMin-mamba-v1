import subprocess
import types
import quick_test

class DummyCompleted:
    def __init__(self, returncode=0):
        self.returncode = returncode

def test_quick_test_runs(monkeypatch):
    def fake_run(cmd, timeout=None):
        return DummyCompleted(0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(quick_test, "time", types.SimpleNamespace(sleep=lambda x: None, time=lambda: 0))
    assert quick_test.quick_test("baseline") is True

