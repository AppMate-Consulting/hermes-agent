"""Tests for gateway/wake.py — background wake delivery.

Two strategies:
* push-capable adapters keep the synthetic MessageEvent / handle_message path;
* the stateless API server (supports_async_delivery=False) self-POSTs
  /v1/chat/completions with the RAW session id in X-Hermes-Session-Id, so the
  wake turn resumes the REAL session instead of a parallel invisible one
  keyed by build_session_key().
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.wake import deliver_wake, adapter_supports_push


class PushAdapter:
    """Default adapter shape — no supports_async_delivery attribute."""

    def __init__(self):
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self, host="0.0.0.0", port=0, key="test-key", model="hermes"):
        self._host = host
        self._port = port
        self._api_key = key
        self._model_name = model
        self._internal_wake_token = "process-private-wake-token"

    async def handle_message(self, event):  # pragma: no cover — must NOT be hit
        raise AssertionError("non-push adapter must not receive handle_message wakes")


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
    )


def test_adapter_supports_push_default_true():
    assert adapter_supports_push(PushAdapter()) is True
    assert adapter_supports_push(ApiServerLikeAdapter()) is False


@pytest.mark.parametrize("completion_first", [False, True])
def test_deliver_wake_active_session_preserves_human_completion_fifo(completion_first):
    """A push wake queued beside human input remains its own durable turn."""
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner, _event_conversation_forwarding_metadata

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
        user_id="human-id",
        user_name="Human",
    )
    session_key = "active-session"
    runner = object.__new__(GatewayRunner)

    class ActiveAdapter:
        def __init__(self):
            self._pending_messages = {}

        async def handle_message(self, event):
            runner._queue_or_replace_pending_event(session_key, event)

    adapter = ActiveAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_states = {}
    human = MessageEvent(
        text="human stays human",
        source=source,
        user_id="human-id",
        user_name="Human",
    )

    async def exercise():
        if completion_first:
            await deliver_wake(adapter, text="completion", source=source)
            runner._queue_or_replace_pending_event(session_key, human)
        else:
            runner._queue_or_replace_pending_event(session_key, human)
            await deliver_wake(adapter, text="completion", source=source)

    asyncio.run(exercise())
    head = adapter._pending_messages[session_key]
    tail = runner._session_state(session_key).conversation.queued_events
    events = [head, *tail]
    assert [event.text for event in events] == (
        ["completion", "human stays human"]
        if completion_first else ["human stays human", "completion"]
    )
    assert sum(event.autonomous_completion for event in events) == 1
    human_event = next(event for event in events if event.text.startswith("human"))
    completion_event = next(event for event in events if event.text == "completion")
    assert human_event.internal is False
    assert _event_conversation_forwarding_metadata(human_event)[1] is False
    assert completion_event.internal is True
    assert completion_event.allow_gateway_control is True
    assert _event_conversation_forwarding_metadata(completion_event)[1] is True


async def _serve(handler):
    """Spin an in-process aiohttp server on an ephemeral loopback port."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def test_deliver_wake_non_push_self_posts_raw_session_id(monkeypatch):
    """The self-post carries the RAW session id header + bearer auth and a
    single user message with stream=false — the exact entry point real
    gateway turns use."""
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["session_id"] = request.headers.get("X-Hermes-Session-Id")
        seen["auth"] = request.headers.get("Authorization")
        seen["internal_wake"] = request.headers.get("X-Hermes-Internal-Wake")
        seen["body"] = await request.json()
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(host="0.0.0.0", port=port, key="sekrit")
            await deliver_wake(adapter, text="task done — wake", session_id="raw-sid-42")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["session_id"] == "raw-sid-42"
    assert seen["auth"] == "Bearer sekrit"
    assert seen["internal_wake"] == "process-private-wake-token"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [
        {"role": "user", "content": "task done — wake"}
    ]


def test_deliver_wake_retries_429_then_succeeds(monkeypatch):
    """HTTP 429 (max_concurrent_runs cap) is transient — retried with backoff."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.json_response({"error": "busy"}, status=429)
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            await deliver_wake(adapter, text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 2


def test_api_internal_wake_capability_is_end_to_end_and_unforgeable(monkeypatch):
    """Only the per-process capability minted by the real adapter types a turn."""
    from aiohttp import ClientSession, web
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sekrit"}))
    observed = []

    fake_agent = MagicMock()
    fake_agent.session_prompt_tokens = 0
    fake_agent.session_completion_tokens = 0
    fake_agent.session_total_tokens = 0
    fake_agent.session_id = "wake-capability-session"

    def run_conversation(user_message, **kwargs):
        observed.append((
            user_message,
            kwargs.get("persist_user_is_autonomous_completion", False),
        ))
        return {"final_response": "ok", "completed": True}

    fake_agent.run_conversation.side_effect = run_conversation
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: fake_agent)
    monkeypatch.setattr(adapter, "_ensure_session_db_async", AsyncMock(return_value=None))

    async def exercise():
        app = web.Application()
        app["api_server_adapter"] = adapter
        app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        adapter._host = "127.0.0.1"
        adapter._port = site._server.sockets[0].getsockname()[1]
        payload = {
            "model": "hermes-agent", "stream": False,
            "messages": [{"role": "user", "content": "identical completion text"}],
        }
        authenticated = {
            "Authorization": "Bearer sekrit",
            "X-Hermes-Session-Id": "wake-capability-session",
        }
        try:
            await deliver_wake(
                adapter, text="identical completion text",
                session_id="wake-capability-session",
            )
            async with ClientSession() as client:
                forged_body = {
                    **payload,
                    "autonomous_completion": True,
                    "persist_user_is_autonomous_completion": True,
                    "trusted_autonomous_completion": True,
                }
                cases = [
                    (dict(authenticated), forged_body),
                    (
                        {
                            **authenticated,
                            "X-Hermes-Internal-Wake": "wrong-capability",
                        },
                        payload,
                    ),
                    ({"Authorization": "Bearer sekrit"}, payload),
                    (
                        {
                            **authenticated,
                            "Origin": "https://external.example",
                            "X-Forwarded-For": "203.0.113.10",
                        },
                        payload,
                    ),
                ]
                for headers, request_payload in cases:
                    response = await client.post(
                        f"http://127.0.0.1:{adapter._port}/v1/chat/completions",
                        json=request_payload, headers=headers,
                    )
                    assert response.status == 200
                    await response.read()
                response = await client.post(
                    f"http://127.0.0.1:{adapter._port}/v1/chat/completions",
                    json=payload,
                    headers={"X-Hermes-Internal-Wake": "wrong-capability"},
                )
                assert response.status == 401
                await response.read()
        finally:
            await runner.cleanup()

    asyncio.run(exercise())
    assert observed == [
        ("identical completion text", True),
        ("identical completion text", False),
        ("identical completion text", False),
        ("identical completion text", False),
        ("identical completion text", False),
    ]
