"""Executable provenance contracts for TUI/Desktop completion dispatch."""

import queue
import threading

import pytest

from tui_gateway import server
from tools.process_registry import process_registry


class _OneLivePoll:
    def __init__(self):
        self.checks = 0

    def is_set(self):
        self.checks += 1
        return self.checks > 1


def _session():
    return {
        "session_key": "completion-owner",
        "history_lock": threading.RLock(),
        "history": [],
        "running": False,
        "_finalized": False,
    }


@pytest.mark.parametrize("drain", [False, True], ids=["live", "shutdown-drain"])
@pytest.mark.parametrize(
    "event",
    [
        {
            "type": "completion",
            "session_id": "process-provenance",
            "command": "echo complete",
            "exit_code": 0,
            "output": "complete",
        },
        {
            "type": "async_delegation",
            "delegation_id": "single-provenance",
            "session_key": "completion-owner",
            "task": "one child",
            "result": "done",
            "status": "completed",
        },
        {
            "type": "async_delegation",
            "delegation_id": "batch-provenance",
            "session_key": "completion-owner",
            "results": [
                {"task": "first", "result": "done", "status": "completed"},
                {"task": "second", "result": "done", "status": "completed"},
            ],
        },
    ],
    ids=["process", "delegation-single", "delegation-batch"],
)
def test_poller_dispatches_every_completion_shape_with_explicit_provenance(
    monkeypatch, drain, event
):
    """Both poller branches explicitly type process/single/batch completion turns."""
    isolated = queue.Queue()
    isolated.put(dict(event))
    monkeypatch.setattr(process_registry, "completion_queue", isolated)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    calls = []

    def capture(_rid, _sid, session, text, **kwargs):
        calls.append((text, kwargs))
        session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", capture)
    session = _session()
    server._sessions["completion-tab"] = session
    stop = threading.Event() if drain else _OneLivePoll()
    if drain:
        stop.set()
    try:
        server._notification_poller_loop(stop, "completion-tab", session)
    finally:
        server._sessions.pop("completion-tab", None)
        while not isolated.empty():
            isolated.get_nowait()

    assert len(calls) == 1
    assert calls[0][1]["is_autonomous_completion"] is True
    if event["type"] == "async_delegation":
        assert calls[0][1]["display_kind"] == "async_delegation_complete"

