"""Production-boundary ownership regressions for compression admission."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli.middleware import RequestMiddlewareResult
from run_agent import AIAgent


STRUCTURED = [
    {"type": "text", "text": "duplicate human content"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
]


def _response(*, tool=False, invalid=False):
    if invalid:
        return SimpleNamespace(choices=[], model="test/model", usage=None)
    calls = None
    if tool:
        calls = [SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="web_search", arguments='{"query":"x"}'),
        )]
    message = SimpleNamespace(
        content=None if tool else "done", tool_calls=calls,
        reasoning_content=None, reasoning=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=message, finish_reason="tool_calls" if tool else "stop"
        )],
        model="test/model", usage=None,
    )


def _agent():
    tool = {
        "type": "function",
        "function": {
            "name": "web_search", "description": "search",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    with (
        patch("run_agent.get_tool_definitions", return_value=[tool]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=6,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "system"
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent._api_max_retries = 3
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = True

    compressor = MagicMock()
    compressor.threshold_tokens = 10
    compressor.context_length = 100
    compressor.should_defer_preflight_to_real_usage.return_value = False
    compressor.get_active_compression_failure_cooldown.return_value = None
    compression_checks = []

    def should_compress(tokens):
        compression_checks.append(tokens)
        return len(compression_checks) == 1

    compressor.should_compress.side_effect = should_compress
    compressor.compression_checks = compression_checks
    compressor.should_compress_info.return_value = (False, None)
    compressor.protect_first_n = 0
    compressor.protect_last_n = 0
    compressor.compression_count = 0
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor.select_context.side_effect = lambda rows, **_kw: copy.deepcopy(rows)

    def deterministic_summary(rows, **_kwargs):
        # This is the sole mocked summary boundary.  The real
        # compress_context transaction, finalizer, policy and publication run.
        assert all("_compression_turn_anchor" not in row for row in rows)
        return [
            {"role": "assistant", "content": "compact summary"},
            {"role": "user", "content": copy.deepcopy(STRUCTURED)},
        ]

    compressor.compress.side_effect = deterministic_summary
    agent.context_compressor = compressor
    agent._compression_feasibility_checked = True
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda _message: "system"
    return agent, compressor


def _run(
    agent,
    responses,
    *,
    admitted,
    admitted_snapshots,
    middleware_calls,
    boundary_events=None,
):
    boundary_events = boundary_events if boundary_events is not None else []
    response_iter = iter(responses)

    def dispatch(**kwargs):
        boundary_events.append(("dispatch", copy.deepcopy(kwargs)))
        return next(response_iter)

    agent.client.chat.completions.create.side_effect = dispatch

    select_context = agent.context_compressor.select_context.side_effect

    def select(rows, **kwargs):
        boundary_events.append(("select", copy.deepcopy(rows)))
        return select_context(rows, **kwargs)

    agent.context_compressor.select_context.side_effect = select

    real_compress_context = agent._compress_context

    def capture_admission(*args, **kwargs):
        result = real_compress_context(*args, **kwargs)
        handoff = kwargs["live_request_context"]["admission_handoff"]
        request = handoff.get("request")
        if request is not None and not admitted:
            admitted.append(request)
            admitted_snapshots.append(copy.deepcopy(request))
        return result

    def middleware(payload, **context):
        shaped = copy.deepcopy(payload)
        shaped["extra_headers"] = {
            **shaped.get("extra_headers", {}),
            "x-proof-middleware": "applied",
        }
        middleware_calls.append((copy.deepcopy(shaped), copy.deepcopy(context)))
        boundary_events.append(("middleware", copy.deepcopy(shaped)))
        return RequestMiddlewareResult(
            payload=shaped,
            original_payload=copy.deepcopy(payload),
            changed=True,
            trace=[{"middleware": "proof"}],
        )

    with (
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=1),
        patch("agent.conversation_loop.estimate_messages_tokens_rough", return_value=50),
        patch(
            "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
            side_effect=lambda payload: 1_000 if "compact summary" in str(payload) else 100_000,
        ),
        patch("hermes_cli.middleware.apply_llm_request_middleware", side_effect=middleware),
        patch("agent.conversation_loop.jittered_backoff", return_value=0),
        patch.object(agent, "_compress_context", side_effect=capture_admission),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch(
            "run_agent.handle_function_call",
            lambda *_args, **_kwargs: json.dumps({"ok": True}),
        ),
    ):
        return agent.run_conversation(
            copy.deepcopy(STRUCTURED),
            conversation_history=[
                {"role": "user", "content": copy.deepcopy(STRUCTURED)},
                {"role": "assistant", "content": "older answer " * 20_000},
            ],
        )


def _assert_dispatch_uses_admission(dispatch, admitted_request, snapshot):
    payload = admitted_request["payload"]
    assert dispatch == snapshot["payload"]
    assert dispatch["messages"] is payload["messages"]
    assert dispatch["tools"] is payload["tools"]
    prepared = admitted_request.get("moa_prepared_request")
    if prepared is not None:
        assert dispatch["_moa_prepared_request"] is prepared


def test_real_compression_admission_dispatches_exact_finalized_payload():
    agent, compressor = _agent()
    admitted, admitted_snapshots, middleware_calls = [], [], []
    events = []

    result = _run(
        agent, [_response()], admitted=admitted,
        admitted_snapshots=admitted_snapshots,
        middleware_calls=middleware_calls, boundary_events=events,
    )

    assert result["completed"] is True
    assert compressor.compress.call_count == 1
    # Candidate construction and the compression handoff each select context.
    # Prove the finalized handoff then flows directly to dispatch without
    # depending on obsolete context-free projection totals.
    assert compressor.select_context.call_count == 2
    assert len(admitted) == 1
    dispatch_index = next(i for i, event in enumerate(events) if event[0] == "dispatch")
    admission_index = max(
        i for i, event in enumerate(events[:dispatch_index])
        if event[0] == "middleware"
        and event[1] == admitted_snapshots[0]["payload"]
    )
    assert [event[0] for event in events[admission_index:dispatch_index + 1]] == [
        "middleware", "dispatch",
    ]
    dispatch = agent.client.chat.completions.create.call_args.kwargs
    _assert_dispatch_uses_admission(dispatch, admitted[0], admitted_snapshots[0])
    assert dispatch["extra_headers"]["x-proof-middleware"] == "applied"
    assert sum("duplicate human content" in str(row) for row in dispatch["messages"]) == 1
    assert all("_compression_turn_anchor" not in row for row in result["messages"])
    assert "_admitted_provider_request" not in vars(agent)


def test_same_provider_retry_reuses_admitted_bytes_verbatim():
    agent, _ = _agent()
    admitted, admitted_snapshots, middleware_calls = [], [], []
    events = []

    result = _run(
        agent, [_response(invalid=True), _response()], admitted=admitted,
        admitted_snapshots=admitted_snapshots,
        middleware_calls=middleware_calls, boundary_events=events,
    )

    assert result["completed"] is True
    calls = agent.client.chat.completions.create.call_args_list
    assert len(calls) == 2
    _assert_dispatch_uses_admission(
        calls[0].kwargs, admitted[0], admitted_snapshots[0]
    )
    _assert_dispatch_uses_admission(
        calls[1].kwargs, admitted[0], admitted_snapshots[0]
    )
    assert calls[0].kwargs == calls[1].kwargs
    assert calls[0].kwargs["messages"] is calls[1].kwargs["messages"]
    assert calls[0].kwargs["tools"] is calls[1].kwargs["tools"]


def test_tool_iteration_cannot_consume_stale_admission():
    agent, _ = _agent()
    admitted, admitted_snapshots, middleware_calls = [], [], []

    result = _run(
        agent, [_response(tool=True), _response()], admitted=admitted,
        admitted_snapshots=admitted_snapshots, middleware_calls=middleware_calls,
    )

    assert result["completed"] is True
    calls = agent.client.chat.completions.create.call_args_list
    _assert_dispatch_uses_admission(
        calls[0].kwargs, admitted[0], admitted_snapshots[0]
    )
    assert calls[1].kwargs != admitted_snapshots[0]["payload"]
    assert calls[1].kwargs["messages"] is not admitted[0]["payload"]["messages"]
    assert calls[1].kwargs["tools"] is not admitted[0]["payload"]["tools"]
    assert calls[1].kwargs["messages"][-1]["role"] == "tool"
    assert admitted[0] == admitted_snapshots[0]
    assert len(middleware_calls) == 5
    assert agent.context_compressor.compression_checks == [63, 0]
    assert "_admitted_provider_request" not in vars(agent)
