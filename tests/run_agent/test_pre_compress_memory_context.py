"""Behavior contracts for the pre-compression memory-context handoff."""

import copy
import threading
import time
from unittest.mock import MagicMock

import pytest


def _make_agent(memory_manager, compressor):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        provider="openrouter",
        api_mode="chat_completions",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        session_db=None,
        session_id="test-session",
        skip_context_files=True,
        skip_memory=True,
    )

    agent._memory_manager = memory_manager
    agent.context_compressor = compressor
    agent._compression_feasibility_checked = True
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda _message: "new-system-prompt"
    return agent


def _messages():
    return [{"role": "user", "content": f"message {i}"} for i in range(6)]


def _configure_engine_state(engine):
    engine.compression_count = 1
    engine.last_prompt_tokens = 0
    engine.last_completion_tokens = 0
    engine._last_summary_error = None
    engine._last_compress_aborted = False
    engine._last_aux_model_failure_model = None
    engine._last_aux_model_failure_error = None


@pytest.mark.parametrize("provider", ["openrouter", "anthropic"])
def test_non_wire_only_reclaim_is_rejected_by_provider_projection(
    monkeypatch, provider
):
    """Arbitrarily large private/display fields cannot earn admission."""
    compressor = MagicMock()
    original = _messages()
    for index, message in enumerate(original):
        message.update({
            "display_kind": "hidden",
            "display_metadata": {"padding": "x" * 100_000},
            "_row_id": index,
            "_private_provenance": "y" * 100_000,
        })
    compressor.compress.return_value = [
        {"role": row["role"], "content": row["content"]} for row in original
    ]
    _configure_engine_state(compressor)
    agent = _make_agent(None, compressor)
    agent.provider = provider
    agent._use_prompt_caching = False

    returned, _ = agent._compress_context(
        original, "sys", approx_tokens=100_000, force=True
    )

    assert returned is original
    assert agent._last_compression_outcome == "rejected_no_progress"
    assert original[0]["display_metadata"]["padding"] == "x" * 100_000


def test_provider_visible_reclaim_control_is_admitted(monkeypatch):
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "genuinely smaller provider input"}
    ]
    _configure_engine_state(compressor)
    agent = _make_agent(None, compressor)
    estimates = iter((100_000, 1_000))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )

    returned, _ = agent._compress_context(
        _messages(), "sys", approx_tokens=100_000, force=True
    )

    assert returned != _messages()
    assert agent._last_compression_outcome == "committed_in_memory"


def test_on_pre_compress_runs_after_engine_and_does_not_influence_summary(monkeypatch):
    manager = MagicMock()
    manager.on_pre_compress.return_value = "Checkpoint id: ctx-orchestrator"
    received = {}
    compressor = MagicMock()

    def capture_compress(
        incoming,
        current_tokens=None,
        focus_topic=None,
        force=False,
        memory_context="",
    ):
        received.update(
            current_tokens=current_tokens,
            focus_topic=focus_topic,
            force=force,
            memory_context=memory_context,
        )
        return [incoming[0], incoming[-1]]

    compressor.compress.side_effect = capture_compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)
    messages = _messages()
    estimates = iter((100_000, 1_000))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )

    agent._compress_context(
        messages,
        "sys",
        approx_tokens=100_000,
        focus_topic="checkpoint continuity",
        force=True,
    )

    manager.on_pre_compress.assert_called_once_with(messages)
    assert received == {
        "current_tokens": 100_000,
        "focus_topic": "checkpoint continuity",
        "force": True,
        "memory_context": "",
    }


def test_legacy_engine_receives_only_supported_compression_arguments(monkeypatch):
    manager = MagicMock()
    manager.on_pre_compress.return_value = "Checkpoint id: unsupported-by-legacy"
    calls = []

    class StrictLegacyEngine:
        def compress(self, messages, current_tokens=None):
            calls.append(current_tokens)
            return [messages[0], messages[-1]]

    engine = StrictLegacyEngine()
    _configure_engine_state(engine)
    agent = _make_agent(manager, engine)
    estimates = iter((100_000, 1_000))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )

    compressed, _prompt = agent._compress_context(
        _messages(),
        "sys",
        approx_tokens=100_000,
        focus_topic="unsupported focus",
        force=True,
    )

    assert len(compressed) == 2
    assert calls == [100_000]


def test_provider_context_is_strictly_sanitized_before_plugin_engine(monkeypatch):
    prefix_secret = "sk-" + "a" * 30
    query_secret = "opaque-query-secret"
    userinfo_value = "opaque-userinfo-value"
    fragment_secret = "FRAG_SECRET"
    relative_secret = "REL_SECRET"
    encoded_key_secret = "ENC_SECRET"
    hyphen_client_secret = "HYPHEN_CLIENT_SECRET"
    hyphen_access_secret = "HYPHEN_ACCESS_SECRET"
    hyphen_api_secret = "HYPHEN_API_SECRET"
    encoded_hyphen_secret = "ENCODED_HYPHEN_SECRET"
    network_userinfo_secret = "NET_SECRET"
    manager = MagicMock()
    manager.on_pre_compress.return_value = (
        f"api key: {prefix_secret}\n"
        f"callback: https://example.test/cb?access_token={query_secret}&state=ok\n"
        f"endpoint: https://user:{userinfo_value}@example.test/private\n"
        f"fragment: https://x.test/#access_token={fragment_secret}&view=public\n"
        f"relative: /resume?token={relative_secret}&view=public\n"
        f"encoded: https://x.test/cb?client%5Fsecret={encoded_key_secret}&view=public\n"
        f"hyphen-client: /resume?client-secret={hyphen_client_secret}&view=public\n"
        f"hyphen-access: /resume?Access-Token={hyphen_access_secret}&view=public\n"
        f"hyphen-api: /resume?api-key={hyphen_api_secret}&view=public\n"
        f"encoded-hyphen: /resume?client%2Dsecret={encoded_hyphen_secret}&view=public\n"
        f"network: //user:{network_userinfo_secret}@x.test/path"
    )
    received = []
    compressor = MagicMock()

    def capture_compress(messages, current_tokens=None, memory_context="", **_kwargs):
        received.append(memory_context)
        return [messages[0], messages[-1]]

    compressor.compress.side_effect = capture_compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)
    estimates = iter((100_000, 1_000))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )

    # Provider-to-engine handoff is an external-LLM egress boundary, so it
    # remains strict even when display/log redaction was explicitly disabled.
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    agent._compress_context(_messages(), "sys", approx_tokens=100_000)

    assert len(received) == 1
    assert received == [""]
    manager.on_pre_compress.assert_called_once()


def test_provider_context_is_bounded_before_plugin_engine(monkeypatch):
    manager = MagicMock()
    manager.on_pre_compress.return_value = "HEAD-SENTINEL" + "x" * 8_000 + "TAIL-SENTINEL"
    received = []
    compressor = MagicMock()

    def capture_compress(messages, current_tokens=None, memory_context="", **_kwargs):
        received.append(memory_context)
        return [messages[0], messages[-1]]

    compressor.compress.side_effect = capture_compress
    _configure_engine_state(compressor)
    agent = _make_agent(manager, compressor)
    estimates = iter((100_000, 1_000))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )

    agent._compress_context(_messages(), "sys", approx_tokens=100_000)

    assert len(received) == 1
    assert received == [""]
    manager.on_pre_compress.assert_called_once()


def test_internal_engine_type_error_propagates_after_one_call():
    manager = MagicMock()
    manager.on_pre_compress.return_value = "Checkpoint id: ctx-typeerror"
    calls = []

    class BrokenEngine:
        def compress(
            self,
            messages,
            current_tokens=None,
            focus_topic=None,
            force=False,
            memory_context="",
        ):
            calls.append(memory_context)
            raise TypeError("engine implementation bug")

    engine = BrokenEngine()
    _configure_engine_state(engine)
    agent = _make_agent(manager, engine)

    with pytest.raises(TypeError, match="engine implementation bug"):
        agent._compress_context(_messages(), "sys", approx_tokens=100_000)

    assert calls == [""]
    manager.on_pre_compress.assert_not_called()


class _BoundaryObserver:
    """Stateful observer: unlike a mock, it records immutable transcript values."""

    def __init__(self, calls):
        self.calls = calls

    def on_pre_compress(self, messages):
        self.calls.append(("on_pre_compress", copy.deepcopy(messages)))
        messages.clear()
        return "legacy return is observational only"

    def on_session_switch(self, new_session_id, **kwargs):
        self.calls.append(("on_session_switch", new_session_id, kwargs))


def test_successful_no_db_boundary_runs_memory_only_after_admission(monkeypatch):
    """No-DB admission orders observer then memory commit with frozen input."""
    calls = []
    observer = _BoundaryObserver(calls)
    compressor = MagicMock()
    compressor.compress.side_effect = lambda incoming, **_kwargs: [
        {"role": "user", "content": "small durable candidate"}
    ]
    _configure_engine_state(compressor)
    agent = _make_agent(observer, compressor)
    original = _messages()

    def commit(messages):
        calls.append(("commit_memory_session", copy.deepcopy(messages)))
        messages.clear()

    agent.commit_memory_session = commit
    estimates = iter((100_000, 1_000))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )

    returned, _ = agent._compress_context(
        original, "sys", approx_tokens=100_000, force=True
    )

    assert returned != original
    assert [entry[0] for entry in calls] == [
        "on_pre_compress", "commit_memory_session", "on_session_switch"
    ]
    assert calls[0][1] == _messages()
    assert calls[1][1] == _messages()


@pytest.mark.parametrize("out_tokens", [100_001, 100_000, 99_000])
def test_rejected_candidate_never_reaches_memory_observer(monkeypatch, out_tokens):
    """Grow, no-op and below-minimum admission failures are side-effect free."""
    calls = []
    observer = _BoundaryObserver(calls)
    compressor = MagicMock()
    compressor.compress.return_value = [{"role": "user", "content": "candidate"}]
    _configure_engine_state(compressor)
    agent = _make_agent(observer, compressor)
    agent.commit_memory_session = lambda messages: calls.append(
        ("commit_memory_session", copy.deepcopy(messages))
    )
    estimates = iter((100_000, out_tokens))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )

    original = _messages()
    returned, _ = agent._compress_context(original, "sys", approx_tokens=100_000)

    assert returned is original
    assert calls == []


def test_stateful_in_place_boundary_orders_durable_publish_before_memory(
    monkeypatch, tmp_path, request
):
    """The atomic DB boundary is authoritative before any observer is invoked."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    request.addfinalizer(db.close)
    sid = "ordered-memory-boundary"
    db.create_session(sid, "cli", model="test/model")
    for message in _messages():
        db.append_message(sid, message["role"], message["content"])
    calls = []
    observer = _BoundaryObserver(calls)
    compressor = MagicMock()
    compressor.compress.return_value = [{"role": "user", "content": "small candidate"}]
    _configure_engine_state(compressor)
    agent = _make_agent(observer, compressor)
    agent._session_db = db
    agent.session_id = sid
    agent.compression_in_place = True
    compressor.on_session_start.side_effect = lambda *_args, **kwargs: calls.append(
        ("boundary_callback", kwargs)
    )
    original = db.get_messages_as_conversation(sid)
    real_publish = db.archive_and_compact

    def publish(*args, **kwargs):
        result = real_publish(*args, **kwargs)
        calls.append(("atomic_db_publication", copy.deepcopy(db.get_messages_as_conversation(sid))))
        return result

    agent.commit_memory_session = lambda messages: calls.append(
        ("commit_memory_session", copy.deepcopy(messages))
    )
    monkeypatch.setattr(db, "archive_and_compact", publish)
    estimates = iter((100_000, 1_000))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )

    returned, _ = agent._compress_context(original, "sys", approx_tokens=100_000, force=True)

    assert returned is not original
    assert [item[0] for item in calls] == [
        "atomic_db_publication", "on_pre_compress", "commit_memory_session",
        "boundary_callback", "on_session_switch",
    ]
    assert calls[1][1] == original
    assert calls[2][1] == original
    assert len([item for item in calls if item[0] == "on_pre_compress"]) == 1
    assert len([item for item in calls if item[0] == "commit_memory_session"]) == 1


def test_stateful_in_place_persistence_failure_has_no_observer_calls(
    monkeypatch, tmp_path, request
):
    """An atomic publication failure cannot escape to boundary observers."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    request.addfinalizer(db.close)
    sid = "failed-memory-boundary"
    db.create_session(sid, "cli", model="test/model")
    db.append_message(sid, "user", "durable original")
    calls = []
    observer = _BoundaryObserver(calls)
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "must not become durable"}
    ]
    _configure_engine_state(compressor)
    agent = _make_agent(observer, compressor)
    agent._session_db = db
    agent.session_id = sid
    agent.compression_in_place = True
    agent.commit_memory_session = lambda messages: calls.append(
        ("commit_memory_session", copy.deepcopy(messages))
    )
    compressor.on_session_start.side_effect = lambda *_args, **_kwargs: calls.append(
        ("context_engine_boundary",)
    )
    monkeypatch.setattr(
        db,
        "archive_and_compact",
        MagicMock(side_effect=RuntimeError("durable publication failed")),
    )
    estimates = iter((100_000, 1_000))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )

    original = _messages()
    returned, _ = agent._compress_context(
        original, "sys", approx_tokens=100_000, force=True
    )

    assert returned is original
    assert calls == []
    assert agent._last_compression_outcome == "persistence_failure"
    assert db.get_messages_as_conversation(sid)[0]["content"] == "durable original"


def test_post_commit_memory_exception_keeps_compacted_db_and_committed_outcome(
    monkeypatch, tmp_path, request
):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    request.addfinalizer(db.close)
    sid = "memory-exception-after-commit"
    db.create_session(sid, "cli", model="test/model")
    db.append_message(sid, "user", "durable original")
    calls = []
    observer = _BoundaryObserver(calls)
    compressor = MagicMock()
    compressor.compress.return_value = [{"role": "user", "content": "durable compacted"}]
    _configure_engine_state(compressor)
    agent = _make_agent(observer, compressor)
    agent._session_db = db
    agent.session_id = sid
    agent.compression_in_place = True
    agent.commit_memory_session = lambda _messages: (_ for _ in ()).throw(RuntimeError("observer failed"))
    estimates = iter((100_000, 1_000))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )

    returned, _ = agent._compress_context(_messages(), "sys", approx_tokens=100_000, force=True)

    expected = "message 5\n\ndurable compacted"
    assert [row["content"] for row in returned] == [expected]
    assert [row["content"] for row in db.get_messages_as_conversation(sid)] == [
        expected,
    ]
    assert agent._last_compression_outcome == "committed_materially_shrunk"
    assert calls[0][0] == "on_pre_compress"


def test_post_commit_pre_compress_exception_does_not_skip_memory_commit(
    monkeypatch, tmp_path, request
):
    """Observer failure after publication cannot rewrite the committed result."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    request.addfinalizer(db.close)
    sid = "pre-compress-exception-after-commit"
    db.create_session(sid, "cli", model="test/model")
    db.append_message(sid, "user", "durable original")
    calls = []

    class RaisingObserver(_BoundaryObserver):
        def on_pre_compress(self, messages):
            calls.append(("on_pre_compress", copy.deepcopy(messages)))
            raise RuntimeError("observer failed")

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "durable compacted"}
    ]
    _configure_engine_state(compressor)
    agent = _make_agent(RaisingObserver(calls), compressor)
    agent._session_db = db
    agent.session_id = sid
    agent.compression_in_place = True
    agent.event_callback = MagicMock()
    agent.commit_memory_session = lambda messages: calls.append(
        ("commit_memory_session", copy.deepcopy(messages))
    )
    estimates = iter((100_000, 1_000))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )

    returned, _ = agent._compress_context(
        _messages(), "sys", approx_tokens=100_000, force=True
    )

    expected = "message 5\n\ndurable compacted"
    assert [row["content"] for row in returned] == [expected]
    assert [row["content"] for row in db.get_messages_as_conversation(sid)] == [
        expected,
    ]
    assert agent._last_compression_outcome == "committed_materially_shrunk"
    assert [entry[0] for entry in calls].count("on_pre_compress") == 1
    assert [entry[0] for entry in calls].count("commit_memory_session") == 1
    assert not any(entry[0] == "persistence_failure" for entry in calls)
    agent.event_callback.assert_called_once_with(
        "session:compress",
        {
            "platform": "",
            "session_id": sid,
            "old_session_id": "",
            "in_place": True,
            "compression_count": 1,
        },
    )


def test_blocked_postcommit_provider_is_bounded_and_releases_lease(
    monkeypatch, tmp_path, request, caplog
):
    from agent.conversation_compression import CompressionCommitFence
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    request.addfinalizer(db.close)
    sid = "blocked-postcommit-provider"
    db.create_session(sid, "cli", model="test/model")
    db.append_message(sid, "user", "durable original")
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    class BlockingObserver:
        def on_pre_compress(self, _messages):
            entered.set()
            release.wait()

        def on_session_switch(self, *_args, **_kwargs):
            completed.set()

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "durable compacted"}
    ]
    _configure_engine_state(compressor)
    agent = _make_agent(BlockingObserver(), compressor)
    agent._session_db = db
    agent.session_id = sid
    agent.compression_in_place = True
    agent.commit_memory_session = MagicMock()
    fence = CompressionCommitFence()
    estimates = iter((100_000, 1_000))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )
    monkeypatch.setattr(
        "agent.conversation_compression._POSTCOMMIT_CALLBACK_WAIT_SECONDS", 0.05
    )

    started = time.monotonic()
    returned, _ = agent._compress_context(
        _messages(), "sys", approx_tokens=100_000, force=True, commit_fence=fence
    )
    elapsed = time.monotonic() - started

    assert entered.is_set()
    assert elapsed < 1.0
    assert fence.commit_in_flight is False
    assert db.get_compression_lock_holder(sid) is None
    assert db.try_acquire_compression_lock(sid, "later-attempt", ttl_seconds=30)
    db.release_compression_lock(sid, "later-attempt")
    assert returned[0]["content"].endswith("durable compacted")
    assert db.get_messages_as_conversation(sid)[0]["content"].endswith(
        "durable compacted"
    )
    assert agent._last_compression_outcome == "committed_materially_shrunk"
    assert "postcommit compression callbacks exceeded" in caplog.text
    agent.commit_memory_session.assert_not_called()
    release.set()
    assert completed.wait(1.0)
    agent.commit_memory_session.assert_called_once()


def test_prepublication_prompt_work_remains_cancellable(
    monkeypatch, tmp_path, request, caplog
):
    from agent.conversation_compression import CompressionCommitFence
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    request.addfinalizer(db.close)
    sid = "cancel-prepublication-prompt"
    db.create_session(sid, "cli", model="test/model")
    db.append_message(sid, "user", "durable original")
    entered = threading.Event()
    release = threading.Event()
    compressor = MagicMock()
    compressor.compress.return_value = [{"role": "user", "content": "candidate"}]
    _configure_engine_state(compressor)
    agent = _make_agent(MagicMock(), compressor)
    agent._session_db = db
    agent.session_id = sid
    agent.compression_in_place = True
    agent.commit_memory_session = MagicMock()
    original = _messages()
    original_copy = copy.deepcopy(original)
    agent._cached_system_prompt = None
    agent._cached_system_prompt_static = "stable-prefix"
    agent._persist_user_message_idx = 3
    agent._persist_user_message_override = {"sentinel": [1, 2]}
    persistence_before = copy.deepcopy(agent._persist_user_message_override)
    compressor._summary_failure_cooldown_until = 12345.0
    cooldown_before = compressor._summary_failure_cooldown_until
    fence = CompressionCommitFence()

    def blocked_prompt(_message):
        entered.set()
        release.wait()
        return "candidate prompt"

    agent._build_system_prompt = blocked_prompt
    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            agent._compress_context(
                original,
                "sys",
                approx_tokens=100_000,
                force=True,
                commit_fence=fence,
            )
        )
    )
    worker.start()
    assert entered.wait(1.0)
    assert fence.commit_in_flight is False
    assert fence.cancel_before_commit() is True
    release.set()
    worker.join(2.0)

    assert not worker.is_alive()
    assert result[0][0] is original
    assert original == original_copy
    assert agent._last_compression_outcome == "cancelled_commit_fence"
    assert '"failure_class":"commit_fence_cancelled"' in caplog.text
    assert '"commit_status":"aborted"' in caplog.text
    assert compressor._summary_failure_cooldown_until == cooldown_before
    assert agent._cached_system_prompt is None
    assert agent._cached_system_prompt_static == "stable-prefix"
    assert agent._persist_user_message_idx == 3
    assert agent._persist_user_message_override == persistence_before
    assert db.get_messages_as_conversation(sid)[0]["content"] == "durable original"
    assert db.get_compression_lock_holder(sid) is None
    assert db.try_acquire_compression_lock(sid, "later-attempt", ttl_seconds=30)
    db.release_compression_lock(sid, "later-attempt")
    assert fence.commit_in_flight is False
    compressor.compress.assert_not_called()
    agent._memory_manager.on_pre_compress.assert_not_called()
    agent._memory_manager.on_session_switch.assert_not_called()
    agent.commit_memory_session.assert_not_called()


def test_materially_admitted_candidate_honors_precommit_fence_cancel(
    monkeypatch, tmp_path, request, caplog
):
    from agent.conversation_compression import CompressionCommitFence
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    request.addfinalizer(db.close)
    sid = "cancel-after-material-admission"
    db.create_session(sid, "cli", model="test/model")
    db.append_message(sid, "user", "durable original")
    manager = MagicMock()
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "materially smaller candidate"}
    ]
    _configure_engine_state(compressor)
    compressor._summary_failure_cooldown_until = 67890.0
    agent = _make_agent(manager, compressor)
    agent._session_db = db
    agent.session_id = sid
    agent.compression_in_place = True
    agent.commit_memory_session = MagicMock()
    original = _messages()
    original_copy = copy.deepcopy(original)
    fence = CompressionCommitFence()

    class CancelOnAdmission(int):
        def __sub__(self, other):
            result = int(self) - int(other)
            assert result >= 4_096
            assert fence.cancel_before_commit() is True
            return result

    estimates = iter((CancelOnAdmission(100_000), 1_000))
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda *_args, **_kwargs: next(estimates),
    )

    returned, _ = agent._compress_context(
        original,
        "sys",
        approx_tokens=100_000,
        force=True,
        commit_fence=fence,
    )

    assert returned is original
    assert original == original_copy
    assert agent._last_compression_outcome == "cancelled_commit_fence"
    assert '"failure_class":"commit_fence_cancelled"' in caplog.text
    assert compressor._summary_failure_cooldown_until == 67890.0
    assert db.get_messages_as_conversation(sid)[0]["content"] == "durable original"
    assert db.get_compression_lock_holder(sid) is None
    assert db.try_acquire_compression_lock(sid, "later-attempt", ttl_seconds=30)
    db.release_compression_lock(sid, "later-attempt")
    assert fence.commit_in_flight is False
    manager.on_pre_compress.assert_not_called()
    manager.on_session_switch.assert_not_called()
    agent.commit_memory_session.assert_not_called()
