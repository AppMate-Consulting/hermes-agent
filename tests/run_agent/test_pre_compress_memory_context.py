"""Behavior contracts for the pre-compression memory-context handoff."""

import copy
import threading
import time
from types import SimpleNamespace
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
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        lambda _payload: next(estimates),
    )

    returned, _ = agent._compress_context(
        _messages(), "sys", approx_tokens=100_000, force=True
    )

    assert returned != _messages()
    assert agent._last_compression_outcome == "committed_in_memory"


@pytest.mark.parametrize(
    ("finalized_sizes", "expected_outcome"),
    [
        ((100_000, 101_000), "rejected_would_grow"),
        ((100_000, 97_000), "rejected_below_minimum_reclaim"),
        ((100_000, 1_000), "committed_in_memory"),
    ],
)
def test_compression_admission_uses_complete_finalized_payload_sizes(
    monkeypatch, finalized_sizes, expected_outcome
):
    """Late provider fields, not obsolete transcript projections, decide admission."""
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "genuinely changed final body"}
    ]
    _configure_engine_state(compressor)
    agent = _make_agent(None, compressor)
    sizes = iter(finalized_sizes)
    estimated_payloads = []

    def estimate_complete_payload(payload):
        estimated_payloads.append(copy.deepcopy(payload))
        return next(sizes)

    monkeypatch.setattr(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        estimate_complete_payload,
    )
    original = _messages()
    returned, _ = agent._compress_context(
        original, "sys", approx_tokens=100_000, force=True
    )

    assert agent._last_compression_outcome == expected_outcome
    assert len(estimated_payloads) == 2
    assert all("messages" in payload for payload in estimated_payloads)
    if expected_outcome.startswith("rejected_"):
        assert returned is original
    else:
        assert returned is not original


def test_compression_rejects_middleware_only_finalized_request_growth(
    monkeypatch,
):
    """A smaller transcript cannot hide a provider field added after shaping."""
    from hermes_cli.middleware import RequestMiddlewareResult

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "small middleware trigger"}
    ]
    _configure_engine_state(compressor)
    agent = _make_agent(None, compressor)
    original = [{"role": "user", "content": "ordinary original body " * 100}]

    def grow_only_final_candidate(payload, **_context):
        original_payload = copy.deepcopy(payload)
        shaped = copy.deepcopy(payload)
        if "small middleware trigger" in str(payload):
            shaped["provider_preflight_extension"] = "late " * 50_000
        return RequestMiddlewareResult(
            payload=shaped,
            original_payload=original_payload,
            changed=shaped != original_payload,
            trace=[{"middleware": "late-growth"}],
        )

    monkeypatch.setattr(
        "hermes_cli.middleware.apply_llm_request_middleware",
        grow_only_final_candidate,
    )
    returned, _ = agent._compress_context(
        original, "sys", approx_tokens=100_000, force=True
    )

    assert returned is original
    assert agent._last_compression_outcome == "rejected_would_grow"


def test_compression_rejects_nondeterministic_before_middleware(monkeypatch):
    """The frozen before body must be replayed, not only the candidate body."""
    from hermes_cli.middleware import RequestMiddlewareResult

    compressor = MagicMock()
    compressor.compress.return_value = [{"role": "user", "content": "small"}]
    _configure_engine_state(compressor)
    agent = _make_agent(None, compressor)
    calls = 0

    def stateful(payload, **_context):
        nonlocal calls
        calls += 1
        shaped = copy.deepcopy(payload)
        if calls == 1:
            shaped["one_time_inflation"] = "x" * 100_000
        return RequestMiddlewareResult(
            payload=shaped,
            original_payload=copy.deepcopy(payload),
            changed=shaped != payload,
            trace=[],
        )

    monkeypatch.setattr("hermes_cli.middleware.apply_llm_request_middleware", stateful)
    original = _messages()
    returned, _ = agent._compress_context(
        original, "sys", approx_tokens=100_000, force=True
    )

    assert calls == 4
    assert returned is original
    assert agent._last_compression_outcome == "rejected_nondeterministic_middleware"


@pytest.mark.parametrize(
    ("sizes", "outcome"),
    [
        ((100_000, 100_000), "rejected_no_progress"),
        ((100_000, 101_000), "rejected_would_grow"),
        ((100_000, 97_000), "rejected_below_minimum_reclaim"),
    ],
)
def test_policy_rejection_restores_exact_compressor_state(monkeypatch, sizes, outcome):
    compressor = MagicMock()
    _configure_engine_state(compressor)
    compressor._previous_summary = "committed summary"
    compressor._fallback_compression_streak = 2
    compressor._ineffective_compression_count = 1
    compressor.summary_model = "stable/model"

    def mutate_then_compress(_messages, **_kwargs):
        compressor.compression_count = 99
        compressor._previous_summary = "speculative summary"
        compressor._fallback_compression_streak = 9
        compressor._ineffective_compression_count = 9
        compressor.summary_model = "speculative/model"
        return [{"role": "user", "content": "candidate"}]

    compressor.compress.side_effect = mutate_then_compress
    agent = _make_agent(None, compressor)
    estimates = iter(sizes)
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        lambda _payload: next(estimates),
    )
    original = _messages()
    returned, _ = agent._compress_context(original, "sys", force=True)

    assert returned is original
    assert agent._last_compression_outcome == outcome
    assert compressor.compression_count == 1
    assert compressor._previous_summary == "committed summary"
    assert compressor._fallback_compression_streak == 2
    assert compressor._ineffective_compression_count == 1
    assert compressor.summary_model == "stable/model"


@pytest.mark.parametrize(
    ("kind", "outcome"),
    [
        ("summary_failure", "summary_failure"),
        ("no_progress", "rejected_no_progress"),
        ("empty", "rejected_empty_transcript"),
    ],
)
def test_early_rejection_restores_exact_compressor_state(kind, outcome):
    compressor = MagicMock()
    _configure_engine_state(compressor)
    compressor._previous_summary = "committed summary"
    compressor._fallback_compression_streak = 2
    compressor._ineffective_compression_count = 1
    compressor.summary_model = "stable/model"
    original = _messages()

    def reject(_messages, **_kwargs):
        compressor.compression_count = 99
        compressor._previous_summary = "speculative summary"
        compressor._fallback_compression_streak = 9
        compressor._ineffective_compression_count = 9
        compressor.summary_model = "speculative/model"
        if kind == "summary_failure":
            compressor._last_compress_aborted = True
            compressor._last_summary_error = "summary unavailable"
        if kind == "empty":
            return []
        return copy.deepcopy(original)

    compressor.compress.side_effect = reject
    agent = _make_agent(None, compressor)
    returned, _ = agent._compress_context(original, "sys", force=True)

    assert returned is original
    assert agent._last_compression_outcome == outcome
    assert compressor.compression_count == 1
    assert compressor._previous_summary == "committed summary"
    assert compressor._fallback_compression_streak == 2
    assert compressor._ineffective_compression_count == 1
    assert compressor.summary_model == "stable/model"
    assert compressor._last_compress_aborted is False


def test_rebuilt_system_growth_outweighs_real_message_shrink():
    """Admission sizes the complete request, including prompt and tool schema."""
    compressor = MagicMock()
    original = [{"role": "user", "content": "old " * 20_000}]
    compressor.compress.return_value = [{"role": "user", "content": "small"}]
    _configure_engine_state(compressor)
    agent = _make_agent(MagicMock(), compressor)
    agent._cached_system_prompt = "old system"
    agent._cached_system_prompt_static = "old"
    agent._build_system_prompt = lambda _message: "grown " * 100_000
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "large_schema",
            "description": "schema " * 5_000,
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    tools_before = copy.deepcopy(agent.tools)

    returned, prompt = agent._compress_context(
        original, "sys", approx_tokens=100_000, force=True
    )

    assert returned is original
    assert prompt == "old system"
    assert agent._last_compression_outcome == "rejected_would_grow"
    assert agent._cached_system_prompt == "old system"
    assert agent._cached_system_prompt_static == "old"
    assert agent.tools == tools_before


def test_projection_preserves_current_turn_identity_after_interrupt_ghost():
    """Filtering a hidden interrupt row must not shift the caller's index."""
    from agent.conversation_loop import _project_provider_request

    compressor = MagicMock()
    _configure_engine_state(compressor)
    agent = _make_agent(None, compressor)
    rows = [
        {"role": "user", "content": "historical"},
        {
            "role": "assistant",
            "content": "[This response was interrupted by a user correction.]",
            "display_kind": "hidden",
        },
        {"role": "user", "content": "current"},
    ]

    projected = _project_provider_request(
        agent,
        rows,
        current_turn_user_idx=2,
        external_prefetch="fresh recall",
        apply_context_selection=False,
    )["messages"]

    assert not any(
        row.get("content") == "[This response was interrupted by a user correction.]"
        for row in projected
    )
    users = [row["content"] for row in projected if row["role"] == "user"]
    assert len(users) == 1
    merged = users[0]
    assert merged.index("historical") < merged.index("current") < merged.index(
        "fresh recall"
    )


@pytest.mark.parametrize("mode", ["openai-chat", "anthropic-native"])
def test_complete_projection_matches_live_send_and_is_non_mutating(
    monkeypatch, mode
):
    """run_conversation consumes the canonical complete request verbatim."""
    from agent.conversation_loop import finalize_provider_request

    compressor = MagicMock()
    _configure_engine_state(compressor)
    agent = _make_agent(None, compressor)
    agent.compression_enabled = False
    agent._cached_system_prompt = "stable system"
    agent.ephemeral_system_prompt = "ephemeral system"
    agent.prefill_messages = [{"role": "assistant", "content": " prefill "}]
    agent.provider = "anthropic" if mode == "anthropic-native" else "openrouter"
    agent.api_mode = (
        "anthropic_messages" if mode == "anthropic-native"
        else "chat_completions"
    )
    agent._use_prompt_caching = mode == "anthropic-native"
    agent._use_native_cache_layout = mode == "anthropic-native"
    agent._cache_ttl = "5m"
    agent.context_length = 200_000
    agent.context_compressor.context_length = 200_000
    agent._direct_native_anthropic_tool_cache_capability = (
        lambda: mode == "anthropic-native"
    )
    agent._should_sanitize_tool_calls = lambda: True
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "demo",
            "description": "immutable schema",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    history = [
        {"role": "tool", "tool_call_id": "orphan", "content": "drop orphan"},
        {"role": "user", "content": "display old", "api_content": " wire old ",
         "display_metadata": {"private": "never sent"}},
        {"role": "assistant", "content": "calling", "reasoning": "thought",
         "reasoning_details": [{"type": "text", "text": "detail"}],
         "tool_calls": [{"id": "c1", "type": "function", "function": {
             "name": "demo", "arguments": " { \"x\" : 1 } "}}]},
        {"role": "tool", "tool_call_id": "c1", "content": " result "},
    ]
    before = copy.deepcopy((
        history, agent.prefill_messages, agent.tools,
        agent._cached_system_prompt,
        getattr(agent, "_cached_system_prompt_static", None),
    ))
    sent = {}

    def create(**kwargs):
        sent.update(copy.deepcopy(kwargs))
        if mode == "anthropic-native":
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="done")],
                stop_reason="end_turn", model=agent.model, usage=None,
            )
        message = SimpleNamespace(
            content="done", tool_calls=None, reasoning_content=None, reasoning=None
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model=agent.model,
            usage=None,
        )

    live_rows = history + [{"role": "user", "content": " current "}]
    expected = finalize_provider_request(
        agent,
        live_rows,
        system_message="stable system",
        tools=agent.tools,
        current_turn_user_idx=len(history),
        incoming_message=live_rows[-1],
    )
    expected_payload = copy.deepcopy(expected["payload"])
    compressor.select_context.reset_mock()

    if mode == "anthropic-native":
        native_client = MagicMock()

        def stream(**kwargs):
            final_message = create(**kwargs)
            event = SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="done"),
            )
            context = MagicMock()
            context.__enter__.return_value = context
            context.__iter__.return_value = iter([event])
            context.get_final_message.return_value = final_message
            return context

        native_client.messages.stream.side_effect = stream
        monkeypatch.setattr(
            agent,
            "_create_request_anthropic_client",
            lambda **_kwargs: native_client,
        )
    else:
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = create
    agent.run_conversation(" current ", conversation_history=history)

    assert sent == expected_payload
    assert (
        history, agent.prefill_messages, agent.tools,
        agent._cached_system_prompt,
        getattr(agent, "_cached_system_prompt_static", None),
    ) == before
    assert compressor.select_context.call_count == 1


def test_codex_responses_finalized_request_matches_real_transport_dispatch(
    monkeypatch,
):
    """The real Responses transport preflights once, then dispatches verbatim."""
    from agent.conversation_loop import finalize_provider_request
    from hermes_cli import lifecycle
    from hermes_cli.middleware import RequestMiddlewareResult

    compressor = MagicMock()
    _configure_engine_state(compressor)
    compressor.select_context.side_effect = lambda rows, **_kwargs: [
        copy.deepcopy(rows[0]), copy.deepcopy(rows[-1])
    ]
    agent = _make_agent(None, compressor)
    agent.compression_enabled = False
    agent.provider = "openai-codex"
    agent.api_mode = "codex_responses"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent._base_url_lower = agent.base_url
    agent._base_url_hostname = "chatgpt.com"
    agent.model = "gpt-5-codex"
    agent._cached_system_prompt = "stable system"
    agent._cached_system_prompt_static = "stable system"
    agent._reapply_reasoning_echo_for_provider = lambda rows: rows[1].setdefault(
        "reasoning_content", "fallback reasoning"
    )
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "demo",
            "description": "immutable schema",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        },
    }]
    history = [
        {"role": "user", "content": "historical selection input"},
        {"role": "assistant", "content": "prior answer"},
    ]
    live_rows = history + [{"role": "user", "content": "current request"}]
    before = copy.deepcopy((history, agent.tools, agent._cached_system_prompt,
                            agent._cached_system_prompt_static))

    transport = agent._get_transport()
    real_preflight = transport.preflight_kwargs
    preflights = []
    real_build_kwargs = agent._build_api_kwargs
    shaping_calls = []

    def counted_build_kwargs(rows, tools_for_api=None):
        shaping_calls.append(copy.deepcopy((rows, tools_for_api)))
        return real_build_kwargs(rows, tools_for_api=tools_for_api)

    monkeypatch.setattr(agent, "_build_api_kwargs", counted_build_kwargs)

    def counted_preflight(self, payload, **kwargs):
        preflights.append(copy.deepcopy((payload, kwargs)))
        return real_preflight(payload, **kwargs)

    monkeypatch.setattr(type(transport), "preflight_kwargs", counted_preflight)

    def add_provider_field(payload, **_context):
        original = copy.deepcopy(payload)
        shaped = copy.deepcopy(payload)
        shaped["provider_extension"] = {"proof": "middleware"}
        return RequestMiddlewareResult(
            payload=shaped, original_payload=original, changed=True,
            trace=[{"middleware": "proof"}],
        )

    monkeypatch.setattr(
        "hermes_cli.middleware.apply_llm_request_middleware", add_provider_field
    )
    expected = finalize_provider_request(
        agent,
        live_rows,
        system_message="stable system",
        tools=agent.tools,
        current_turn_user_idx=len(history),
        incoming_message=live_rows[-1],
        middleware_context={
            "task_id": "default", "turn_id": "expected",
            "api_request_id": "expected", "session_id": agent.session_id,
            "platform": agent.platform or "", "model": agent.model,
            "provider": agent.provider, "base_url": agent.base_url,
            "api_mode": agent.api_mode, "api_call_count": 0,
        },
    )["payload"]
    preflights.clear()
    shaping_calls.clear()
    compressor.select_context.reset_mock()

    def mutate_every_legacy_alias(_name, **kwargs):
        for key in ("request", "request_messages", "conversation_history"):
            value = kwargs.get(key)
            if isinstance(value, dict):
                value.clear()
                value["mutated"] = True
            elif isinstance(value, list):
                value.clear()
                value.append({"role": "user", "content": "mutated"})
        kwargs["system_prompt"] = "mutated"

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "pre_api_request")
    monkeypatch.setattr(lifecycle, "invoke_hook", mutate_every_legacy_alias)
    dispatched = {}

    def final_dispatch(payload, *, client, on_first_delta=None):
        dispatched.update(copy.deepcopy(payload))
        return SimpleNamespace(
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="done")],
            )],
            usage=None, status="completed", model=agent.model,
        )

    monkeypatch.setattr(agent, "_run_codex_stream", final_dispatch)
    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **_kwargs: MagicMock()
    )
    agent.run_conversation("current request", conversation_history=history)

    # Request ids are middleware context, but this middleware is deliberately
    # context-independent; the complete provider body must therefore match.
    assert dispatched == expected
    assert len(preflights) == 1
    assert len(shaping_calls) == 1
    assert compressor.select_context.call_count == 1
    assert (history, agent.tools, agent._cached_system_prompt,
            agent._cached_system_prompt_static) == before


def test_in_memory_publication_cancel_and_claim_are_linearized(monkeypatch):
    """The production seam deterministically proves both sides of the race."""
    from agent.conversation_compression import CompressionCommitFence

    for cancellation_wins in (True, False):
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "user", "content": "small committed candidate"}
        ]
        _configure_engine_state(compressor)
        compressor._session_id = "test-session"
        compressor._proactive_prune_rearm_tokens = 77
        agent = _make_agent(None, compressor)
        agent._cached_system_prompt = "old prompt"
        agent._cached_system_prompt_static = "old static"
        agent._persist_user_message_idx = 4
        agent._persist_user_message_override = {"identity": [1]}
        agent.commit_memory_session = MagicMock()
        agent.event_callback = MagicMock()
        original = _messages()
        snapshot = copy.deepcopy(original)
        fence = CompressionCommitFence()
        barrier_entered = threading.Event()
        release = threading.Event()
        result = []

        def barrier():
            barrier_entered.set()
            assert release.wait(2)

        agent._before_in_memory_compression_publication = barrier
        estimates = iter((100_000, 1_000))
        monkeypatch.setattr(
            "agent.conversation_compression.estimate_request_tokens_rough",
            lambda *_a, **_k: next(estimates),
        )
        worker = threading.Thread(target=lambda: result.append(
            agent._compress_context(
                original, "sys", approx_tokens=100_000, force=True,
                commit_fence=fence,
            )
        ))
        worker.start()
        assert barrier_entered.wait(1)
        if cancellation_wins:
            assert fence.cancel_before_commit() is True
        else:
            assert fence.claim_caller_publication() is True
            assert fence.cancel_before_commit() is False
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert fence.commit_in_flight is False

        if cancellation_wins:
            assert result[0][0] is original and original == snapshot
            assert agent._last_compression_outcome == "cancelled_commit_fence"
            assert agent._cached_system_prompt == "old prompt"
            assert agent._cached_system_prompt_static == "old static"
            assert agent._persist_user_message_idx == 4
            assert agent._persist_user_message_override == {"identity": [1]}
            assert compressor._session_id == "test-session"
            assert compressor._proactive_prune_rearm_tokens == 77
            agent.commit_memory_session.assert_not_called()
            agent.event_callback.assert_not_called()
        else:
            committed = result[0][0]
            assert len(committed) == 1
            assert committed[0]["role"] == "user"
            merged = committed[0]["content"]
            assert merged.index("message 5") < merged.index(
                "small committed candidate"
            )
            assert agent._last_compression_outcome == "committed_in_memory"


def test_in_memory_exception_after_claim_restores_all_shared_state(monkeypatch):
    """A failure at the first claimed outcome publication is transactional."""
    import agent.conversation_compression as compression
    from agent.conversation_compression import CompressionCommitFence

    compressor = MagicMock()
    compressor.compress.return_value = [{"role": "user", "content": "small"}]
    _configure_engine_state(compressor)
    compressor._session_id = "test-session"
    compressor._proactive_prune_rearm_tokens = 91
    agent = _make_agent(None, compressor)
    agent._cached_system_prompt = "old prompt"
    agent._cached_system_prompt_static = "old static"
    agent._persist_user_message_idx = 5
    agent._persist_user_message_override = {"cursor": [2]}
    agent.commit_memory_session = MagicMock()
    agent.event_callback = MagicMock()
    original = _messages()
    original_copy = copy.deepcopy(original)
    fence = CompressionCommitFence()
    real_publish = compression._publish_compression_outcome

    def fail_first_publication(target, value, **kwargs):
        if value == "committed_in_memory":
            raise RuntimeError("injected after publication claim")
        return real_publish(target, value, **kwargs)

    monkeypatch.setattr(compression, "_publish_compression_outcome", fail_first_publication)
    estimates = iter((100_000, 1_000))
    monkeypatch.setattr(
        compression, "estimate_request_tokens_rough",
        lambda *_a, **_k: next(estimates),
    )

    with pytest.raises(RuntimeError, match="injected after publication claim"):
        agent._compress_context(
            original, "sys", approx_tokens=100_000, force=True, commit_fence=fence
        )

    assert original == original_copy
    assert agent._cached_system_prompt == "old prompt"
    assert agent._cached_system_prompt_static == "old static"
    assert agent._persist_user_message_idx == 5
    assert agent._persist_user_message_override == {"cursor": [2]}
    assert compressor._session_id == "test-session"
    assert compressor._proactive_prune_rearm_tokens == 91
    assert fence.commit_in_flight is False
    agent.commit_memory_session.assert_not_called()
    agent.event_callback.assert_not_called()


def test_permanently_blocked_event_callback_is_bounded_and_fifo(
    monkeypatch, tmp_path, request
):
    """A real durable publication returns while its event observer is wedged."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "blocked-event.db")
    request.addfinalizer(db.close)
    sid = "blocked-event-parent"
    db.create_session(sid, "cli", model="test/model")
    db.append_message(sid, "user", "earlier task")
    db.append_message(sid, "assistant", "large history " * 10_000)
    db.append_message(sid, "user", "latest short human task")
    compressor = MagicMock()
    _configure_engine_state(compressor)
    compressor.compress.side_effect = lambda *_a, **_k: [
        {"role": "user", "content": "small child"}
    ]
    agent = _make_agent(None, compressor)
    agent._session_db = db
    agent.session_id = sid
    agent.compression_in_place = False
    first_entered = threading.Event()
    release = threading.Event()
    request.addfinalizer(release.set)
    events = []

    def commit_memory(rows, *, old_session_id):
        events.append(("commit", old_session_id, copy.deepcopy(rows)))

    def callback(_name, payload):
        events.append(("enter", payload["old_session_id"], payload["session_id"]))
        if not first_entered.is_set():
            first_entered.set()
            assert release.wait(5)
        events.append(("exit", payload["old_session_id"], payload["session_id"]))

    agent.event_callback = callback
    agent.commit_memory_session = commit_memory
    monkeypatch.setattr(
        "agent.conversation_compression._POSTCOMMIT_CALLBACK_WAIT_SECONDS", 0.03
    )
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        lambda messages, **_k: sum(
            len(str(row.get("content", ""))) for row in messages
        ),
    )

    first, _ = agent._compress_context(
        db.get_messages_as_conversation(sid), "sys", approx_tokens=100_000,
        force=True
    )
    child_b = agent.session_id
    assert child_b != sid and first_entered.is_set()
    assert db.get_compression_lock_holder(sid) is None
    assert agent._last_compression_outcome == "committed_materially_shrunk"

    # Grow the authoritative child while leaving its latest human task short.
    db.append_message(child_b, "assistant", "large second history " * 10_000)
    db.append_message(child_b, "user", "second latest short human task")
    second_input = db.get_messages_as_conversation(child_b)
    second, _ = agent._compress_context(
        second_input, "sys", approx_tokens=100_000, force=True
    )
    child_c = agent.session_id
    assert child_c not in (sid, child_b)
    assert [entry[:2] for entry in events] == [
        ("commit", sid), ("enter", sid)
    ]
    frozen_parent_a = copy.deepcopy(events[0][2])
    assert db.get_compression_lock_holder(child_b) is None
    assert "small child" in second[0]["content"]
    assert "second latest short human task" in second[0]["content"]

    release.set()
    tail = agent._compression_observer_lane_tail
    assert tail.wait(2)
    assert [entry[:2] for entry in events] == [
        ("commit", sid), ("enter", sid), ("exit", sid),
        ("commit", child_b), ("enter", child_b), ("exit", child_b),
    ]
    assert events[0][2] == frozen_parent_a
    assert events[3][2] == second_input
    assert events[1][2] == child_b and events[4][2] == child_c
    assert agent.event_callback is callback


def test_real_compressions_deferred_notification_is_claimed_and_fifo_frozen(
    monkeypatch, tmp_path, request
):
    """Real SessionDB publications prove deferred FIFO lifecycle end to end."""
    from agent.conversation_compression import (
        _PENDING_CONTEXT_ENGINE_NOTIFICATION,
        finalize_context_engine_compression_notification,
    )
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "real-deferred-fifo.db")
    request.addfinalizer(db.close)
    parent = "fifo-parent"
    db.create_session(parent, "gateway-a", model="test/model")
    db.append_message(parent, "user", "first task")
    db.append_message(parent, "assistant", "large A " * 20_000)
    db.append_message(parent, "user", "latest A")

    compressor = MagicMock()
    _configure_engine_state(compressor)
    compressor.compression_count = 0

    def compress_candidate(*_args, **_kwargs):
        compressor.compression_count += 1
        return [{"role": "user", "content": "small committed summary"}]

    compressor.compress.side_effect = compress_candidate
    agent = _make_agent(None, compressor)
    agent._session_db = db
    agent.session_id = parent
    agent.platform = "gateway-a"
    agent._gateway_session_key = ("gateway-a", "conversation-a")
    agent.compression_in_place = False
    agent.commit_memory_session = lambda *_args, **_kwargs: None
    agent.event_callback = None

    a_entered = threading.Event()
    release_a = threading.Event()
    request.addfinalizer(release_a.set)
    delivered = threading.Event()
    order = []
    notifications = []

    def frozen_callback(new_session_id, **kwargs):
        record = (
            new_session_id,
            kwargs["old_session_id"],
            kwargs["platform"],
            copy.deepcopy(kwargs["conversation_id"]),
            threading.current_thread().name,
        )
        notifications.append(record)
        if len(notifications) == 1:
            order.append("a-enter")
            a_entered.set()
            assert release_a.wait(timeout=5)
            order.append("a-exit")
        elif len(notifications) == 2:
            order.append("b-notify")
            delivered.set()
        else:
            order.append("c-notify")

    compressor.on_session_start = frozen_callback

    def status_callback(_kind, _message):
        if "Session compressed 2 times" in _message:
            order.append("b-status")

    monkeypatch.setattr(
        "agent.conversation_compression._POSTCOMMIT_CALLBACK_WAIT_SECONDS", 0.02
    )
    monkeypatch.setattr(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        lambda payload: sum(len(str(value)) for value in payload.values()),
    )

    agent._compress_context(
        db.get_messages_as_conversation(parent), "sys",
        approx_tokens=100_000, force=True,
    )
    child_a = agent.session_id
    assert a_entered.is_set()
    agent.status_callback = status_callback

    db.append_message(child_a, "assistant", "large B " * 20_000)
    db.append_message(child_a, "user", "latest B")
    agent._compress_context(
        db.get_messages_as_conversation(child_a), "sys",
        approx_tokens=100_000, force=True,
        defer_context_engine_notification=True,
    )
    child_b = agent.session_id
    assert child_b not in (parent, child_a)
    assert not delivered.is_set()
    assert finalize_context_engine_compression_notification(agent, committed=True)
    assert getattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION) is None

    # Every value below changes after synchronous claim. Delivery must still
    # use the publication-time record and callable.
    compressor.on_session_start = lambda *_args, **_kwargs: order.append("mutated")
    agent.platform = "gateway-mutated"
    agent._gateway_session_key = ("mutated", "conversation")
    agent.session_id = "mutated-session-id"
    agent.status_callback = lambda *_args: order.append("mutated-status")

    release_a.set()
    assert delivered.wait(timeout=2)
    assert agent._compression_observer_lane_tail.wait(timeout=2)
    assert notifications[:2] == [
        (
            child_a, parent, "gateway-a", ("gateway-a", "conversation-a"),
            f"compression-postcommit-{child_a}",
        ),
        (
            child_b, child_a, "gateway-a", ("gateway-a", "conversation-a"),
            f"compression-postcommit-{child_b}",
        ),
    ]
    assert order[:4] == ["a-enter", "a-exit", "b-status", "b-notify"]
    assert notifications.count(notifications[1]) == 1

    # A normal later deferred boundary neither sees nor restores B's record.
    agent.session_id = child_b
    agent.platform = "gateway-c"
    agent._gateway_session_key = ("gateway-c", "conversation-c")
    compressor.on_session_start = frozen_callback
    db.append_message(child_b, "assistant", "large C " * 20_000)
    db.append_message(child_b, "user", "latest C")
    agent._compress_context(
        db.get_messages_as_conversation(child_b), "sys",
        approx_tokens=100_000, force=True,
        defer_context_engine_notification=True,
    )
    child_c = agent.session_id
    assert finalize_context_engine_compression_notification(agent, committed=True)
    assert agent._compression_observer_lane_tail.wait(timeout=2)
    assert getattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION) is None
    assert notifications[-1][:4] == (
        child_c, child_b, "gateway-c", ("gateway-c", "conversation-c")
    )
    assert order.count("c-notify") == 1


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
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        lambda _payload: next(estimates),
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
