"""Production-loop ownership regressions for compression request admission."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _response(*, tool=False):
    call = None
    if tool:
        call = [SimpleNamespace(
            id="call-1", type="function",
            function=SimpleNamespace(name="web_search", arguments='{"query":"x"}'),
        )]
    message = SimpleNamespace(
        content=None if tool else "done", tool_calls=call,
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
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = True
    compressor = MagicMock()
    compressor.threshold_tokens = 10
    compressor.context_length = 100
    compressor.should_defer_preflight_to_real_usage.return_value = False
    compressor.get_active_compression_failure_cooldown.return_value = None
    compressor.should_compress.side_effect = [True, False, False]
    compressor.should_compress_info.return_value = (False, None)
    compressor.protect_first_n = 0
    compressor.protect_last_n = 0
    agent.context_compressor = compressor
    return agent


def _install_admitting_compressor(agent):
    admitted = {}
    calls = []

    def compress(messages, system_message, *, live_request_context, **_kwargs):
        calls.append(copy.deepcopy(live_request_context))
        compacted = [dict(row) for row in messages]
        for row in compacted:
            row.pop("_compression_turn_anchor", None)
        payload = {
            "model": "test/model",
            "messages": [{"role": "user", "content": "ADMITTED EXACT BYTES"}],
            "tools": copy.deepcopy(agent.tools),
        }
        frozen = {
            "payload": payload,
            "original_payload": copy.deepcopy(payload),
            "messages": copy.deepcopy(payload["messages"]),
            "tools": copy.deepcopy(payload["tools"]),
            "moa_prepared_request": None,
            "middleware_trace": [],
            "_consumes_user_initiator": False,
        }
        admitted.update(copy.deepcopy(frozen))
        live_request_context["admission_handoff"]["request"] = frozen
        return compacted, system_message

    agent._compress_context = compress
    return admitted, calls


def _run(agent, responses):
    agent.client.chat.completions.create.side_effect = responses
    with (
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=1),
        patch("agent.conversation_loop.estimate_messages_tokens_rough", return_value=50),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch(
            "run_agent.handle_function_call",
            lambda *_args, **_kwargs: json.dumps({"ok": True}),
        ),
    ):
        return agent.run_conversation(
            "same structured content",
            conversation_history=[
                {"role": "user", "content": "same structured content"},
                {"role": "assistant", "content": "older answer"},
            ],
        )


def test_successful_admission_is_dispatched_verbatim_in_same_iteration():
    agent = _agent()
    admitted, compression_calls = _install_admitting_compressor(agent)

    result = _run(agent, [_response()])

    assert result["completed"] is True
    assert len(compression_calls) == 1
    assert agent.client.chat.completions.create.call_args.kwargs == admitted["payload"]
    assert "_admitted_provider_request" not in vars(agent)
    assert all(
        "_compression_turn_anchor" not in row for row in result["messages"]
    )


def test_tool_iteration_cannot_consume_stale_admission():
    agent = _agent()
    admitted, _ = _install_admitting_compressor(agent)

    result = _run(agent, [_response(tool=True), _response()])

    assert result["completed"] is True
    calls = agent.client.chat.completions.create.call_args_list
    assert calls[0].kwargs == admitted["payload"]
    assert calls[1].kwargs != admitted["payload"]
    assert calls[1].kwargs["messages"][-1]["role"] == "tool"
    assert "_admitted_provider_request" not in vars(agent)
