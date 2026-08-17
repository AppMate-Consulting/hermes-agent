"""Production-boundary ownership regressions for compression admission."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli.middleware import RequestMiddlewareResult
from agent.conversation_compression import (
    estimate_finalized_payload_tokens_rough as _real_finalized_estimate,
)
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
    execution_callbacks=None,
    middleware_transform=None,
    finalized_estimator=None,
    conversation_history=None,
    ownership_handoffs=None,
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
        if ownership_handoffs is not None:
            ownership_handoffs.append({
                "request": request,
                "admitted": handoff.get("admitted"),
            })
        if (
            request is not None
            and handoff.get("admitted") is True
            and not admitted
        ):
            admitted.append(request)
            admitted_snapshots.append(copy.deepcopy(request))
        return result

    def middleware(payload, **context):
        shaped = copy.deepcopy(payload)
        shaped["extra_headers"] = {
            **shaped.get("extra_headers", {}),
            "x-proof-middleware": "applied",
        }
        if middleware_transform is not None:
            shaped = middleware_transform(shaped)
        middleware_calls.append((copy.deepcopy(shaped), copy.deepcopy(context)))
        boundary_events.append(("middleware", copy.deepcopy(shaped)))
        return RequestMiddlewareResult(
            payload=shaped,
            original_payload=copy.deepcopy(payload),
            changed=True,
            trace=[{"middleware": "proof"}],
        )

    if finalized_estimator is None:
        finalized_estimator = (
            lambda payload: 1_000
            if "compact summary" in str(payload) else 100_000
        )

    with (
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=1),
        patch("agent.conversation_loop.estimate_messages_tokens_rough", return_value=50),
        patch(
            "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
            side_effect=finalized_estimator,
        ),
        patch(
            "agent.conversation_loop.estimate_finalized_payload_tokens_rough",
            side_effect=finalized_estimator,
        ),
        patch("hermes_cli.middleware.apply_llm_request_middleware", side_effect=middleware),
        patch(
            "hermes_cli.middleware._get_middleware_callbacks",
            side_effect=lambda kind: list(execution_callbacks or [])
            if kind == "llm_execution" else [],
        ),
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
            conversation_history=conversation_history or [
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
    ownership = []

    result = _run(
        agent, [_response()], admitted=admitted,
        admitted_snapshots=admitted_snapshots,
        middleware_calls=middleware_calls, boundary_events=events,
        ownership_handoffs=ownership,
    )

    assert result["completed"] is True
    assert compressor.compress.call_count == 1
    # Candidate construction and the compression handoff each select context.
    # Prove the finalized handoff then flows directly to dispatch without
    # depending on obsolete context-free projection totals.
    assert compressor.select_context.call_count == 2
    assert len(admitted) == 1
    assert ownership == [{"request": admitted[0], "admitted": True}]
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


def test_finalized_middleware_pressure_triggers_one_old_one_candidate_handoff():
    agent, compressor = _agent()
    compressor.threshold_tokens = 1_000
    compressor.should_compress.side_effect = (
        lambda tokens: tokens >= compressor.threshold_tokens
    )
    observed_payloads = []

    def inflate_old_request(payload):
        if "older answer" not in str(payload.get("messages", [])):
            return payload
        payload["messages"].append(
            {"role": "system", "content": "middleware-visible " * 2_000}
        )
        payload["tools"] = [
            *payload.get("tools", []),
            {
                "type": "function",
                "function": {
                    "name": "middleware_tool",
                    "description": "provider-visible " * 2_000,
                    "parameters": {"type": "object"},
                },
            },
        ]
        payload["provider_options"] = {"routing": "middleware " * 2_000}
        return payload

    def pressure(payload):
        observed_payloads.append(copy.deepcopy(payload))
        return _real_finalized_estimate(payload)

    admitted, snapshots, middleware_calls = [], [], []
    result = _run(
        agent,
        [_response()],
        admitted=admitted,
        admitted_snapshots=snapshots,
        middleware_calls=middleware_calls,
        middleware_transform=inflate_old_request,
        finalized_estimator=pressure,
        conversation_history=[
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "older answer"},
        ],
    )

    assert result["completed"] is True
    assert compressor.compress.call_count == 1
    finalized_pressure = compressor.should_compress.call_args_list[0].args[0]
    assert finalized_pressure >= compressor.threshold_tokens
    pre_middleware = copy.deepcopy(observed_payloads[0])
    pre_middleware["messages"].pop()
    pre_middleware["tools"].pop()
    pre_middleware.pop("provider_options")
    assert _real_finalized_estimate(pre_middleware) < compressor.threshold_tokens
    assert observed_payloads[0]["messages"][-1]["content"].startswith(
        "middleware-visible"
    )
    assert observed_payloads[0]["tools"][-1]["function"]["name"] == (
        "middleware_tool"
    )
    assert "provider_options" in observed_payloads[0]
    # Exactly one ordinary old transition plus one admitted candidate.
    assert len(middleware_calls) == 2
    assert len(admitted) == 1
    dispatch = agent.client.chat.completions.create.call_args.kwargs
    _assert_dispatch_uses_admission(dispatch, admitted[0], snapshots[0])


def test_no_compression_finalized_request_invokes_middleware_once():
    agent, compressor = _agent()
    compressor.threshold_tokens = 1_000
    compressor.should_compress.side_effect = lambda tokens: tokens >= 1_000

    def add_visible_option(payload):
        payload["provider_options"] = {"preview": "small"}
        return payload

    admitted, snapshots, middleware_calls = [], [], []
    result = _run(
        agent,
        [_response()],
        admitted=admitted,
        admitted_snapshots=snapshots,
        middleware_calls=middleware_calls,
        middleware_transform=add_visible_option,
        finalized_estimator=lambda payload: 50,
    )

    assert result["completed"] is True
    assert compressor.compress.call_count == 0
    assert len(middleware_calls) == 1
    dispatch = agent.client.chat.completions.create.call_args.kwargs
    assert dispatch == middleware_calls[0][0]
    assert dispatch["provider_options"] == {"preview": "small"}


def test_no_compression_execution_middleware_may_replace_finalized_old_request():
    agent, compressor = _agent()
    compressor.threshold_tokens = 1_000
    compressor.should_compress.side_effect = lambda _tokens: False
    observed = {}

    def replace(request, next_call, **_context):
        replacement = copy.deepcopy(request)
        replacement["provider_options"] = {"execution": "replacement"}
        observed["old"] = request
        observed["replacement"] = replacement
        return next_call(replacement)

    admitted, snapshots, middleware_calls = [], [], []
    result = _run(
        agent,
        [_response()],
        admitted=admitted,
        admitted_snapshots=snapshots,
        middleware_calls=middleware_calls,
        execution_callbacks=[replace],
        finalized_estimator=lambda _payload: 50,
    )

    assert result["completed"] is True
    assert compressor.compress.call_count == 0
    assert len(middleware_calls) == 1
    assert admitted == []
    dispatch = agent.client.chat.completions.create.call_args.kwargs
    assert dispatch["messages"] is observed["replacement"]["messages"]
    assert dispatch["messages"] is not observed["old"]["messages"]
    assert dispatch["provider_options"] == {"execution": "replacement"}


def test_ordinary_old_retry_remains_non_admitted_and_refinalizes():
    agent, compressor = _agent()
    compressor.should_compress.side_effect = lambda _tokens: False
    execution_requests = []

    def replace(request, next_call, **_context):
        execution_requests.append(request)
        replacement = copy.deepcopy(request)
        replacement["provider_options"] = {"attempt": len(execution_requests)}
        return next_call(replacement)

    admitted, snapshots, middleware_calls = [], [], []
    result = _run(
        agent,
        [_response(invalid=True), _response()],
        admitted=admitted,
        admitted_snapshots=snapshots,
        middleware_calls=middleware_calls,
        execution_callbacks=[replace],
    )

    assert result["completed"] is True
    assert admitted == []
    assert len(middleware_calls) == 2
    assert len(execution_requests) == 2
    assert execution_requests[1] is not execution_requests[0]
    calls = agent.client.chat.completions.create.call_args_list
    assert calls[0].kwargs["provider_options"] == {"attempt": 1}
    assert calls[1].kwargs["provider_options"] == {"attempt": 2}


def test_rejected_compression_hands_off_old_as_mutable_without_middleware_replay():
    agent, compressor = _agent()
    ownership = []
    observed = {}

    def replace(request, next_call, **_context):
        replacement = copy.deepcopy(request)
        replacement["provider_options"] = {"rejected": "replacement"}
        observed["old"] = request
        observed["replacement"] = replacement
        return next_call(replacement)

    admitted, snapshots, middleware_calls = [], [], []
    result = _run(
        agent,
        [_response()],
        admitted=admitted,
        admitted_snapshots=snapshots,
        middleware_calls=middleware_calls,
        execution_callbacks=[replace],
        # Equal old/candidate pressure rejects the candidate for no reclaim.
        finalized_estimator=lambda _payload: 100_000,
        ownership_handoffs=ownership,
    )

    assert result["completed"] is True
    assert compressor.compress.call_count == 1
    assert len(ownership) == 1
    assert ownership[0]["admitted"] is False
    assert ownership[0]["request"]["payload"] is observed["old"]
    assert admitted == []
    # One old transition and one speculative candidate transition; dispatch
    # consumes the exact restored old object without a third request callback.
    assert len(middleware_calls) == 2
    dispatch = agent.client.chat.completions.create.call_args.kwargs
    assert dispatch["messages"] is observed["replacement"]["messages"]
    assert dispatch["messages"] is not observed["old"]["messages"]
    assert dispatch["provider_options"] == {"rejected": "replacement"}


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


def test_admitted_execution_replacement_fails_closed_before_dispatch():
    agent, _ = _agent()
    admitted, snapshots, middleware_calls = [], [], []

    def replace(request, next_call, **_context):
        replacement = copy.deepcopy(request)
        replacement["messages"].append({"role": "user", "content": "inflated"})
        replacement["tools"] = []
        replacement["input"] = [{"role": "user", "content": "changed"}]
        return next_call(replacement)

    result = _run(
        agent, [_response()], admitted=admitted, admitted_snapshots=snapshots,
        middleware_calls=middleware_calls, execution_callbacks=[replace],
    )

    assert result["failed"] is True
    assert "immutable admitted request" in result["error"]
    agent.client.chat.completions.create.assert_not_called()
    assert admitted[0] == snapshots[0]


def test_admitted_execution_in_place_mutation_fails_closed_before_dispatch():
    agent, _ = _agent()
    admitted, snapshots, middleware_calls = [], [], []
    identities = {}

    def mutate(request, next_call, **_context):
        identities["payload"] = request
        identities["messages"] = request["messages"]
        identities["message"] = request["messages"][0]
        identities["tools"] = request["tools"]
        identities["extra_headers"] = request["extra_headers"]
        request["messages"][0]["content"] = "mutated in place"
        request["messages"].append({"role": "user", "content": "injected"})
        request["tools"].clear()
        request["extra_headers"] = copy.deepcopy(request["extra_headers"])
        return next_call()

    result = _run(
        agent, [_response()], admitted=admitted, admitted_snapshots=snapshots,
        middleware_calls=middleware_calls, execution_callbacks=[mutate],
    )

    assert result["failed"] is True
    assert "mutated the immutable admitted request" in result["error"]
    agent.client.chat.completions.create.assert_not_called()
    assert admitted[0] == snapshots[0]
    payload = admitted[0]["payload"]
    assert payload is identities["payload"]
    assert payload["messages"] is identities["messages"]
    assert payload["messages"][0] is identities["message"]
    assert payload["tools"] is identities["tools"]
    assert payload["extra_headers"] is identities["extra_headers"]


def test_admitted_execution_mutate_then_raise_restores_without_dispatch():
    agent, _ = _agent()
    admitted, snapshots, middleware_calls = [], [], []
    identities = {}

    def mutate_then_raise(request, next_call, **_context):
        identities["payload"] = request
        identities["messages"] = request["messages"]
        identities["first"] = request["messages"][0]
        request["messages"][0]["content"] = "changed before exception"
        request["messages"].append({"role": "user", "content": "injected"})
        raise RuntimeError("wrapper failed")

    result = _run(
        agent, [_response()], admitted=admitted, admitted_snapshots=snapshots,
        middleware_calls=middleware_calls,
        execution_callbacks=[mutate_then_raise],
    )

    assert result["failed"] is True
    assert "mutated the immutable admitted request" in result["error"]
    agent.client.chat.completions.create.assert_not_called()
    assert admitted[0] == snapshots[0]
    payload = admitted[0]["payload"]
    assert payload is identities["payload"]
    assert payload["messages"] is identities["messages"]
    assert payload["messages"][0] is identities["first"]


def test_admitted_execution_mutate_then_raise_after_dispatch_never_duplicates():
    agent, _ = _agent()
    admitted, snapshots, middleware_calls = [], [], []

    def dispatch_then_mutate_then_raise(request, next_call, **_context):
        next_call()
        request["messages"][0]["content"] = "changed after dispatch"
        request["tools"].clear()
        raise RuntimeError("post-dispatch wrapper failed")

    result = _run(
        agent, [_response()], admitted=admitted, admitted_snapshots=snapshots,
        middleware_calls=middleware_calls,
        execution_callbacks=[dispatch_then_mutate_then_raise],
    )

    assert result["failed"] is True
    assert "mutated the immutable admitted request" in result["error"]
    assert agent.client.chat.completions.create.call_count == 1
    dispatched = agent.client.chat.completions.create.call_args.kwargs
    assert dispatched == snapshots[0]["payload"]
    assert admitted[0] == snapshots[0]


def test_admitted_execution_wrapper_observes_exact_object_and_wraps_call():
    agent, _ = _agent()
    admitted, snapshots, middleware_calls, events = [], [], [], []
    observed = []

    def wrapper(request, next_call, **_context):
        observed.append(request)
        events.append("before")
        response = next_call()
        events.append("after")
        return response

    result = _run(
        agent, [_response()], admitted=admitted, admitted_snapshots=snapshots,
        middleware_calls=middleware_calls, execution_callbacks=[wrapper],
    )

    assert result["completed"] is True
    assert events == ["before", "after"]
    assert observed == [admitted[0]["payload"]]
    assert observed[0] is admitted[0]["payload"]
    dispatch = agent.client.chat.completions.create.call_args.kwargs
    assert dispatch["messages"] is observed[0]["messages"]
    assert dispatch["tools"] is observed[0]["tools"]


def test_admitted_retry_wrapper_cannot_statefully_replace_payload():
    agent, _ = _agent()
    admitted, snapshots, middleware_calls = [], [], []
    attempts = []

    def stateful_wrapper(request, next_call, **_context):
        attempts.append(request)
        # Semantically equal replacements are ignored: every terminal call
        # must still receive the exact admitted object and nested objects.
        return next_call(copy.deepcopy(request))

    result = _run(
        agent, [_response(invalid=True), _response()], admitted=admitted,
        admitted_snapshots=snapshots, middleware_calls=middleware_calls,
        execution_callbacks=[stateful_wrapper],
    )

    assert result["completed"] is True
    assert attempts == [admitted[0]["payload"], admitted[0]["payload"]]
    assert attempts[0] is attempts[1]
    calls = agent.client.chat.completions.create.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["messages"] is calls[1].kwargs["messages"]
    assert calls[0].kwargs["tools"] is calls[1].kwargs["tools"]


def test_admitted_retry_refuses_statefully_changed_second_payload():
    agent, _ = _agent()
    admitted, snapshots, middleware_calls = [], [], []
    attempts = 0

    def stateful_wrapper(request, next_call, **_context):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return next_call()
        replacement = copy.deepcopy(request)
        replacement["messages"].append(
            {"role": "user", "content": f"retry rewrite {attempts}"}
        )
        return next_call(replacement)

    result = _run(
        agent, [_response(invalid=True), _response()], admitted=admitted,
        admitted_snapshots=snapshots, middleware_calls=middleware_calls,
        execution_callbacks=[stateful_wrapper],
    )

    assert result["failed"] is True
    assert "replace the immutable admitted request" in result["error"]
    assert attempts == 2
    # The first admitted attempt reached the provider.  The statefully changed
    # retry did not, so there is no publication/dispatch of altered bytes.
    calls = agent.client.chat.completions.create.call_args_list
    assert len(calls) == 1
    assert calls[0].kwargs["messages"] is admitted[0]["payload"]["messages"]
    assert calls[0].kwargs["tools"] is admitted[0]["payload"]["tools"]


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
    # Old and candidate admission each run once; the later tool iteration is
    # one ordinary provider request.  Determinism replay must not duplicate
    # side effects.
    assert len(middleware_calls) == 3
    assert len(agent.context_compressor.compression_checks) == 2
    assert (
        agent.context_compressor.compression_checks[0]
        > agent.context_compressor.compression_checks[1]
    )
    assert "_admitted_provider_request" not in vars(agent)


def test_tool_iteration_cannot_consume_stale_prepared_old_handoff():
    agent, compressor = _agent()
    compressor.should_compress.side_effect = lambda _tokens: False
    execution_requests = []

    def observe(request, next_call, **_context):
        execution_requests.append(request)
        return next_call()

    admitted, snapshots, middleware_calls = [], [], []
    result = _run(
        agent,
        [_response(tool=True), _response()],
        admitted=admitted,
        admitted_snapshots=snapshots,
        middleware_calls=middleware_calls,
        execution_callbacks=[observe],
    )

    assert result["completed"] is True
    assert admitted == []
    assert len(execution_requests) == 2
    first, second = execution_requests
    assert second is not first
    assert second["messages"] is not first["messages"]
    assert second["tools"] is not first["tools"]
    assert second["messages"][-1]["role"] == "tool"
    assert len(middleware_calls) == 2
