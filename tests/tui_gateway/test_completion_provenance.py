"""Executable provenance contracts for TUI/Desktop completion dispatch."""

import queue
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tui_gateway import server
from tools.process_registry import process_registry


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self.target, self.args, self.kwargs = target, args, kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


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


@pytest.mark.parametrize(
    ("event", "synthetic"),
    [
        ({"type": "completion", "session_id": "completion-owner"}, "[IMPORTANT: Background process done.]"),
        ({"type": "async_delegation", "delegation_id": "one", "session_key": "completion-owner"}, "[ASYNC DELEGATION COMPLETE child=one]"),
        ({"type": "async_delegation", "delegation_id": "batch", "session_key": "completion-owner", "results": [{"status": "completed"}]}, "[ASYNC DELEGATION BATCH COMPLETE children=one]"),
    ],
    ids=["process", "delegation-single", "delegation-batch"],
)
def test_run_prompt_submit_post_turn_drain_forwards_explicit_provenance(
    monkeypatch, tmp_path, event, synthetic
):
    """A completion arriving during a real submit is recursively typed true."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_args: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_session_owns_notification_event", lambda *_args: True)
    pending = []

    def drain(**_kwargs):
        return pending.pop() if pending else []

    monkeypatch.setattr(process_registry, "drain_notifications", drain)
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *_args: "claim")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *_args: None)
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.context_compressor import ContextCompressor

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("completion-owner", "tui", model="test/model")
    with patch("run_agent.get_tool_definitions", return_value=[]), patch(
        "run_agent.check_toolset_requirements", return_value={}
    ), patch("run_agent.OpenAI"):
        real_agent = AIAgent(
            api_key="test-key", base_url="https://example.invalid/v1",
            model="test/model", quiet_mode=True, session_db=db,
            session_id="completion-owner", skip_context_files=True, skip_memory=True,
        )
    real_agent.compression_enabled = False
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="completion handled", tool_calls=None,
            reasoning_content=None, reasoning=None,
        ), finish_reason="stop")],
        model="test/model", usage=None,
    )
    real_agent.client = MagicMock()
    real_agent.client.chat.completions.create.return_value = response
    seen = []

    def run_conversation(
        message, *, persist_user_is_autonomous_completion=False, **_kwargs
    ):
        seen.append((message, persist_user_is_autonomous_completion))
        if len(seen) == 1:
            pending.append([(event, synthetic)])
            return {"final_response": "done", "messages": []}
        return real_agent.run_conversation(
            message,
            conversation_history=[],
            persist_user_display_kind="internal_notification",
            persist_user_is_autonomous_completion=persist_user_is_autonomous_completion,
        )

    agent = types.SimpleNamespace(
        session_id="completion-owner", run_conversation=run_conversation,
        clear_interrupt=lambda: None, interim_assistant_callback=None,
    )
    session = {
        **_session(), "agent": agent, "session_key": "completion-owner",
        "history_version": 0, "attached_images": [], "image_counter": 0,
        "cols": 80, "slash_worker": None, "show_reasoning": False,
        "tool_progress_mode": "all", "inflight_turn": None, "running": True,
    }

    server._run_prompt_submit("rid", "sid", session, "first user turn")

    assert seen == [("first user turn", False), (synthetic, True)]
    replay = db.get_messages_as_conversation("completion-owner")
    assert ContextCompressor._has_autonomous_completion_chain(replay[:-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [True, False], ids=["completion", "ordinary-internal"])
async def test_gateway_explicit_event_bit_reaches_conversation_forwarder(explicit):
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner, _event_is_autonomous_completion

    event = MessageEvent(text="same internal wrapper", internal=True)
    event.autonomous_completion = explicit
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    captured = {}

    async def inner(*_args, **kwargs):
        captured.update(kwargs)
        return {}

    runner._run_agent_inner = inner
    await runner._run_agent(
        "message", "context", [], MagicMock(), "sid",
        persist_user_is_autonomous_completion=_event_is_autonomous_completion(event),
    )
    assert captured["persist_user_is_autonomous_completion"] is explicit


def test_cli_sentinel_not_wrapper_text_controls_conversation_provenance():
    from cli import _SyntheticCompletionInput, _unwrap_completion_input

    wrapper = "[ASYNC DELEGATION COMPLETE child=identical]"
    assert _unwrap_completion_input(_SyntheticCompletionInput(wrapper)) == (wrapper, True)
    assert _unwrap_completion_input(wrapper) == (wrapper, False)
