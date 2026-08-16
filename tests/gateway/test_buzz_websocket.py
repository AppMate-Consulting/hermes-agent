"""Tests for the Buzz WebSocket transport (NIP-42) and Nostr signing module.

The signing module and WS transport were contributed in PR #73636 by
@ScaleLeanChris and consolidated onto the merged poll-based adapter; these
tests cover the crypto (against the official BIP-340 vector) and the WS
lifecycle as wired into BuzzAdapter.
"""

import asyncio
import hashlib
import json
import time

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz_mod = load_plugin_adapter("buzz")
BuzzAdapter = _buzz_mod.BuzzAdapter

import importlib.util as _ilu
from pathlib import Path as _Path

_auth_path = _Path(_buzz_mod.__file__).with_name("nostr_auth.py")
_spec = _ilu.spec_from_file_location("plugin_adapter_buzz_nostr_auth", _auth_path)
nostr_auth = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(nostr_auth)

SELF_PUBKEY = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
# BIP-340 test vector 0 private key
TEST_PRIVATE_KEY = "00" * 31 + "03"
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay", **(extra or {})})
    adapter = BuzzAdapter(cfg)
    adapter._self_pubkey = SELF_PUBKEY
    adapter._private_key = TEST_PRIVATE_KEY
    adapter._display_name = "Chip"
    return adapter


# ── nostr_auth: BIP-340 / NIP-42 ──────────────────────────────────────────


def test_schnorr_sign_matches_official_bip340_vector_zero():
    signature = nostr_auth.schnorr_sign(
        bytes(32), TEST_PRIVATE_KEY, auxiliary_randomness=bytes(32)
    )
    assert nostr_auth.public_key_hex(TEST_PRIVATE_KEY).upper() == (
        "F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9"
    )
    assert signature.hex().upper() == (
        "E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA8215"
        "25F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C0"
    )


def test_decode_private_key_rejects_bad_input():
    with pytest.raises(ValueError):
        nostr_auth.decode_private_key("not-a-key")
    with pytest.raises(ValueError):
        nostr_auth.decode_private_key("00" * 32)  # zero — outside range
    with pytest.raises(ValueError):
        nostr_auth.decode_private_key("nsec1qqqqqqqq")  # bad checksum/length


def test_build_auth_event_shape_and_owner_tag():
    tag = json.dumps(["auth", "b" * 64, "", "c" * 128])
    event = nostr_auth.build_auth_event(
        private_key=TEST_PRIVATE_KEY,
        challenge="challenge-1",
        relay_url="wss://relay.example",
        auth_tag_json=tag,
        created_at=1_700_000_000,
        auxiliary_randomness=bytes(32),
    )
    assert event["kind"] == 22242
    assert ["relay", "wss://relay.example"] in event["tags"]
    assert ["challenge", "challenge-1"] in event["tags"]
    assert ["auth", "b" * 64, "", "c" * 128] in event["tags"]
    assert len(bytes.fromhex(event["sig"])) == 64
    assert event["pubkey"] == nostr_auth.public_key_hex(TEST_PRIVATE_KEY)


def test_build_signed_typing_event_has_canonical_id_and_shape():
    event = nostr_auth.build_signed_event(
        private_key=TEST_PRIVATE_KEY,
        kind=20002,
        tags=[["h", CHANNEL]],
        content="",
        created_at=1_700_000_000,
        auxiliary_randomness=bytes(32),
    )
    serialized = json.dumps(
        [0, event["pubkey"], event["created_at"], 20002, [["h", CHANNEL]], ""],
        separators=(",", ":"),
    ).encode()
    assert event["id"] == hashlib.sha256(serialized).hexdigest()
    assert event["kind"] == 20002
    assert event["tags"] == [["h", CHANNEL]]
    assert event["content"] == ""
    assert len(bytes.fromhex(event["pubkey"])) == 32
    assert len(bytes.fromhex(event["sig"])) == 64
    assert TEST_PRIVATE_KEY not in json.dumps(event)


# ── Adapter WS wiring ─────────────────────────────────────────────────────


class _FakeWebSocket:
    """Replays a NIP-42 handshake: AUTH challenge, then OK for the reply."""

    def __init__(self):
        self.sent = []

    async def recv(self):
        if self.sent:
            auth_event = self.sent[0][1]
            return json.dumps(["OK", auth_event["id"], True, "authenticated"])
        return json.dumps(["AUTH", "relay-challenge"])

    async def send(self, raw):
        self.sent.append(json.loads(raw))


@pytest.mark.asyncio
async def test_websocket_auth_raises_on_rejection():
    adapter = _make_adapter()

    class RejectingWs(_FakeWebSocket):
        async def recv(self):
            if self.sent:
                auth_event = self.sent[0][1]
                return json.dumps(["OK", auth_event["id"], False, "denied"])
            return json.dumps(["AUTH", "relay-challenge"])

    with pytest.raises(ConnectionError):
        await adapter._authenticate_websocket(RejectingWs())


@pytest.mark.asyncio
async def test_send_typing_uses_active_socket_and_throttles_per_chat(monkeypatch):
    adapter = _make_adapter()
    websocket = _FakeWebSocket()
    websocket.sent.clear()
    adapter._ws_socket = websocket
    clock = iter((10.0, 10.0, 12.0, 13.1, 13.1, 13.2, 13.2))
    monkeypatch.setattr(_buzz_mod, "_monotonic", lambda: next(clock))

    await adapter.send_typing(CHANNEL)
    await adapter.send_typing(CHANNEL)
    await adapter.send_typing(CHANNEL)
    await adapter.send_typing("another-channel")

    assert len(websocket.sent) == 3
    first = websocket.sent[0]
    assert first[0] == "EVENT"
    assert first[1]["kind"] == 20002
    assert first[1]["content"] == ""
    assert first[1]["tags"] == [["h", CHANNEL]]
    assert TEST_PRIVATE_KEY not in json.dumps(first)
    assert websocket.sent[1][1]["tags"] == [["h", CHANNEL]]
    assert websocket.sent[2][1]["tags"] == [["h", "another-channel"]]


@pytest.mark.asyncio
async def test_send_typing_serializes_same_chat_throttle_check_and_send(monkeypatch):
    adapter = _make_adapter()
    first_send_started = asyncio.Event()
    release_first_send = asyncio.Event()

    class BlockingWs(_FakeWebSocket):
        async def send(self, raw):
            first_send_started.set()
            await release_first_send.wait()
            await super().send(raw)

    websocket = BlockingWs()
    adapter._ws_socket = websocket
    monkeypatch.setattr(_buzz_mod, "_monotonic", lambda: 10.0)

    first = asyncio.create_task(adapter.send_typing(CHANNEL))
    await first_send_started.wait()
    second = asyncio.create_task(adapter.send_typing(CHANNEL))
    await asyncio.sleep(0)

    assert not second.done()
    release_first_send.set()
    await asyncio.gather(first, second)

    assert len(websocket.sent) == 1
    assert adapter._typing_locks == {}
    assert adapter._typing_lock_users == {}


@pytest.mark.asyncio
async def test_send_typing_different_chats_do_not_share_locks(monkeypatch):
    adapter = _make_adapter()
    first_send_started = asyncio.Event()
    release_first_send = asyncio.Event()

    class BlockingFirstChatWs(_FakeWebSocket):
        async def send(self, raw):
            event = json.loads(raw)
            if event[1]["tags"] == [["h", CHANNEL]]:
                first_send_started.set()
                await release_first_send.wait()
            self.sent.append(event)

    websocket = BlockingFirstChatWs()
    adapter._ws_socket = websocket
    monkeypatch.setattr(_buzz_mod, "_monotonic", lambda: 10.0)

    first = asyncio.create_task(adapter.send_typing(CHANNEL))
    await first_send_started.wait()
    second = asyncio.create_task(adapter.send_typing("another-channel"))
    await second

    assert len(websocket.sent) == 1
    assert websocket.sent[0][1]["tags"] == [["h", "another-channel"]]
    release_first_send.set()
    await first
    assert adapter._typing_locks == {}


@pytest.mark.asyncio
async def test_send_typing_is_noop_without_socket_or_valid_chat_id():
    adapter = _make_adapter()
    await adapter.send_typing(CHANNEL)
    adapter._ws_socket = _FakeWebSocket()
    await adapter.send_typing("")
    await adapter.send_typing(None)
    assert adapter._ws_socket.sent == []


@pytest.mark.asyncio
async def test_send_typing_failures_do_not_escape_and_failed_send_is_retryable():
    adapter = _make_adapter()

    class FailingWs:
        def __init__(self):
            self.calls = 0

        async def send(self, raw):
            self.calls += 1
            raise ConnectionError("closed")

    websocket = FailingWs()
    adapter._ws_socket = websocket
    await adapter.send_typing(CHANNEL)
    await adapter.send_typing(CHANNEL)
    assert websocket.calls == 2

    adapter._private_key = "invalid"
    await adapter.send_typing("signing-failure")


@pytest.mark.asyncio
async def test_send_typing_preserves_cancellation():
    adapter = _make_adapter()

    class CancelledWs:
        async def send(self, raw):
            raise asyncio.CancelledError

    adapter._ws_socket = CancelledWs()
    with pytest.raises(asyncio.CancelledError):
        await adapter.send_typing(CHANNEL)
    assert CHANNEL not in adapter._typing_sent_at
    assert adapter._typing_locks == {}
    assert adapter._typing_lock_users == {}


@pytest.mark.asyncio
async def test_disconnect_clears_typing_socket():
    adapter = _make_adapter()
    adapter._ws_socket = _FakeWebSocket()
    await adapter.disconnect()
    assert adapter._ws_socket is None

