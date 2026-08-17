"""Executable acceptance matrix for atomic partial compression.

Every success case enters through a product host and uses a real SessionDB and
the real AIAgent compression transaction.  Only summary generation, provider
token sizing, and host delivery are deterministic test seams.
"""

from __future__ import annotations

import asyncio
import copy
import os
import threading
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_state import SessionDB


OLD_TASK = "OLD HUMAN TASK MUST DISAPPEAR " + "old-payload " * 1200
LATEST_TASK = [
    {"type": "text", "text": "LATEST PROTECTED HUMAN TASK"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,cHJvdGVjdGVk"}},
]
LATEST_API = "LATEST PROTECTED HUMAN TASK\n[private api sidecar]"
LOOKALIKE = "[ASYNC DELEGATION COMPLETE child=proof-lane]"
CALL_ID = "call_protected_1"
SUMMARY = "[CONTEXT COMPACTION] durable generated head"


def _seed(db: SessionDB, sid: str) -> list[dict]:
    """Create a compressible head and a metadata-rich protected suffix."""
    from agent.conversation_compression import (
        AUTONOMOUS_COMPLETION_BRIDGE_ASSISTANT,
        AUTONOMOUS_COMPLETION_BRIDGE_USER,
    )

    db.create_session(sid, source="cli", model="test/model")
    db.append_message(sid, "user", OLD_TASK, display_metadata={"lane": "head"})
    db.append_message(sid, "assistant", "old answer " + "answer " * 1200)
    db.append_message(sid, "user", "older follow-up " + "bulk " * 1200)
    db.append_message(sid, "assistant", "older follow-up answer " + "bulk " * 1200)
    db.append_message(
        sid, "user", LATEST_TASK, api_content=LATEST_API,
        display_kind="human_task", display_metadata={"private": {"keep": [1, 2, 3]}},
    )
    db.append_message(
        sid, "assistant", None,
        tool_calls=[{"id": CALL_ID, "type": "function", "function": {"name": "terminal", "arguments": '{"cmd":"proof"}'}}],
        reasoning="private protected reasoning",
        reasoning_details=[{"type": "reasoning.text", "text": "protected"}],
    )
    db.append_message(
        sid, "tool", "DISTINCTIVE PROTECTED TOOL RESULT", tool_name="terminal",
        tool_call_id=CALL_ID, effect_disposition="read_only",
    )
    db.append_message(sid, "assistant", "tool result acknowledged")
    db.append_message(
        sid, "user", AUTONOMOUS_COMPLETION_BRIDGE_USER,
        autonomous_completion_provenance=True,
    )
    db.append_message(
        sid, "assistant", AUTONOMOUS_COMPLETION_BRIDGE_ASSISTANT,
        autonomous_completion_provenance=True,
    )
    db.append_message(
        sid, "user", LOOKALIKE, display_kind="async_delegation_complete",
        autonomous_completion_provenance=True,
    )
    db.append_message(sid, "assistant", "trusted completion consumed")
    db.append_message(sid, "user", LOOKALIKE)  # identical bytes, human provenance
    db.append_message(sid, "assistant", "human lookalike answered")
    return db.get_messages_as_conversation(sid)


def _agent(db: SessionDB, sid: str, *, in_place: bool, seen: list[list[dict]]):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://invalid.test/v1",
            model="test/model", quiet_mode=True, session_db=db, session_id=sid,
            skip_context_files=True, skip_memory=True,
        )

    def compress(rows, **_kwargs):
        seen.append(copy.deepcopy(rows))
        return [{"role": "assistant", "content": SUMMARY}]

    compressor = MagicMock()
    compressor.compress.side_effect = compress
    compressor.has_content_to_compress.return_value = True
    compressor.compression_count = 1
    compressor.last_prompt_tokens = compressor.last_completion_tokens = 0
    compressor.threshold_tokens = 1
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_fallback_used = False
    compressor._last_summary_dropped_count = 0
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent._context_engine = None
    agent.compression_in_place = in_place
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = "stable proof prompt"
    agent._cached_system_prompt_static = "stable proof prompt"
    return agent


@pytest.fixture
def durable_case(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARTIAL_PROOF_PARENT"
    source = _seed(db, sid)
    yield db, sid, source
    db.close()


def _assert_valid_tool_pairs(rows: list[dict]) -> None:
    roles = [row.get("role") for row in rows]
    assert all(left != right for left, right in zip(roles, roles[1:]))
    calls = [
        call
        for row in rows if row.get("role") == "assistant"
        for call in row.get("tool_calls") or []
    ]
    results = [row for row in rows if row.get("role") == "tool"]
    assert all(call.get("id") for call in calls)
    assert all(row.get("tool_call_id") for row in results)
    call_ids = [
        call["id"]
        for call in calls
    ]
    result_ids = [
        row["tool_call_id"]
        for row in results
    ]
    assert len(call_ids) == len(set(call_ids))
    assert len(result_ids) == len(set(result_ids))
    assert set(call_ids) == set(result_ids)


def _assert_success(db, parent, source, agent, caller_history, seen, publish):
    active = db.get_messages_as_conversation(agent.session_id)
    protected = source[4:]
    assert len(seen) == 1
    assert seen[0] == source[:4]
    assert all(row not in seen[0] for row in protected)
    assert "DISTINCTIVE PROTECTED" not in repr(seen[0])
    assert active[-len(protected):] == protected
    assert active.count(protected[0]) == 1
    assert active[0]["content"] == SUMMARY
    assert LATEST_TASK not in active[:-len(protected)]
    assert OLD_TASK not in [row.get("content") for row in active]
    assert caller_history == active
    _assert_valid_tool_pairs(active)
    assert publish.call_count == 1
    if agent.compression_in_place:
        assert agent.session_id == parent
    else:
        assert agent.session_id != parent
        assert db.get_session(agent.session_id)["parent_session_id"] == parent


@pytest.mark.parametrize("in_place", [True, False], ids=["in_place", "rotation"])
def test_cli_manual_compress_real_sessiondb_partial_matrix(durable_case, in_place):
    """HermesCLI._manual_compress publishes one exact head+suffix transcript."""
    from cli import HermesCLI

    db, sid, source = durable_case
    seen: list[list[dict]] = []
    agent = _agent(db, sid, in_place=in_place, seen=seen)
    shell = object.__new__(HermesCLI)
    shell.agent = agent
    shell.session_id = sid
    shell.conversation_history = copy.deepcopy(source)
    shell._pending_title = None
    method = "archive_and_compact" if in_place else "publish_compression_child"
    with patch.object(db, method, wraps=getattr(db, method)) as publish, patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ), patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100_000), patch(
        "agent.manual_compression_feedback.summarize_manual_compression",
        return_value={"headline": "ok", "token_line": "small", "note": "", "noop": False},
    ), patch.object(agent, "_flush_messages_to_session_db", wraps=agent._flush_messages_to_session_db) as flush:
        shell._manual_compress("/compress here 4")
    flush.assert_not_called()
    _assert_success(db, sid, source, agent, shell.conversation_history, seen, publish)


@pytest.mark.parametrize("in_place", [True, False], ids=["in_place", "rotation"])
def test_tui_compress_session_history_real_sessiondb_partial_matrix(durable_case, in_place):
    """tui_gateway.server._compress_session_history adopts durable bytes exactly."""
    from tui_gateway.server import _compress_session_history

    db, sid, source = durable_case
    seen: list[list[dict]] = []
    agent = _agent(db, sid, in_place=in_place, seen=seen)
    session = {
        "agent": agent, "history": copy.deepcopy(source), "history_version": 7,
        "history_lock": threading.RLock(),
    }
    method = "archive_and_compact" if in_place else "publish_compression_child"
    with patch.object(db, method, wraps=getattr(db, method)) as publish, patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ):
        _compress_session_history(session, "here 4", approx_tokens=100_000)
    _assert_success(db, sid, source, agent, session["history"], seen, publish)


@pytest.mark.parametrize("in_place", [True, False], ids=["in_place", "rotation"])
@pytest.mark.asyncio
async def test_gateway_slash_compress_real_sessiondb_partial_matrix(
    durable_case, in_place
):
    """GatewaySlashCommandsMixin real /compress handler uses the same transaction."""
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionEntry, SessionSource
    from gateway.slash_commands import GatewaySlashCommandsMixin

    db, sid, source_rows = durable_case
    seen: list[list[dict]] = []
    prepared = _agent(db, sid, in_place=in_place, seen=seen)
    source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c", chat_type="dm")
    event = MessageEvent(text="/compress here 4", source=source, message_id="m")
    entry = SessionEntry(
        "telegram:u:c", sid, datetime.now(), datetime.now(),
        origin=source, platform=Platform.TELEGRAM, chat_type="dm",
    )

    class Store:
        async def get_or_create_session(self, _source): return entry
        async def load_transcript(self, active): return db.get_messages_as_conversation(active)
        async def update_session(self, *_a, **_kw): return None
        async def _save(self): return None

    class Host(GatewaySlashCommandsMixin):
        async_session_store = Store()
        _session_db = SimpleNamespace(_db=db, get_session=AsyncMock(return_value=db.get_session(sid)))
        def _session_key_for_source(self, _source): return entry.session_key
        def _resolve_session_agent_runtime(self, **_kw): return "test/model", {"api_key": "test-key"}
        async def _run_in_executor_with_context(self, fn): return fn()
        async def _cleanup_agent_resources_off_loop(self, *_a, **_kw): return None
        def _evict_cached_agent(self, *_a): return None
        def _sync_telegram_topic_binding(self, *_a, **_kw): return None

    host = Host()
    method = "archive_and_compact" if in_place else "publish_compression_child"
    with patch("run_agent.AIAgent", return_value=prepared), patch.object(
        db, method, wraps=getattr(db, method)
    ) as publish, patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ), patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100_000), patch(
        "agent.manual_compression_feedback.summarize_manual_compression",
        return_value={"headline": "ok", "token_line": "small", "note": "", "noop": False},
    ):
        await host._handle_compress_command_inner(event)
    active_id = entry.session_id
    _assert_success(db, sid, source_rows, prepared, db.get_messages_as_conversation(active_id), seen, publish)


@pytest.mark.parametrize("in_place", [True, False], ids=["archive", "child"])
def test_real_publication_failure_restores_exact_state(durable_case, in_place):
    """CLI host publication failure leaves no archive, child, or adoption."""
    from cli import HermesCLI

    db, sid, source = durable_case
    seen: list[list[dict]] = []
    agent = _agent(db, sid, in_place=in_place, seen=seen)
    original = copy.deepcopy(source)
    shell = object.__new__(HermesCLI)
    shell.agent, shell.session_id = agent, sid
    shell.conversation_history, shell._pending_title = copy.deepcopy(source), None
    method = "archive_and_compact" if in_place else "publish_compression_child"
    with patch.object(db, method, side_effect=RuntimeError("injected publication failure")), patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ), patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100_000):
        shell._manual_compress("/compress here 4")
    assert shell.conversation_history == original
    assert db.get_messages_as_conversation(sid) == original
    assert not any(row.get("compacted") for row in db.get_messages(sid, include_inactive=True))
    assert db.find_live_compression_child(sid) is None
    assert db.get_compression_lock_holder(sid) is None


def test_durable_generation_mutation_after_suffix_snapshot_fails_cas(durable_case):
    """A non-cooperating durable writer between snapshot and publish fails closed."""
    db, sid, source = durable_case
    seen: list[list[dict]] = []
    agent = _agent(db, sid, in_place=True, seen=seen)
    before = copy.deepcopy(source)

    def mutate_then_summarize(rows, **_kwargs):
        seen.append(copy.deepcopy(rows))
        def write(conn):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, active) VALUES (?, 'user', ?, 0, 1)",
                (sid, "RACING DURABLE GENERATION"),
            )
        db._execute_write(write)
        return [{"role": "assistant", "content": SUMMARY}]

    agent.context_compressor.compress.side_effect = mutate_then_summarize
    with patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ):
        returned, _ = agent._compress_context(
            source,
            None,
            approx_tokens=100_000,
            force=True,
            protected_tail=source[4:],
        )
    active = db.get_messages_as_conversation(sid)
    assert returned is source
    assert source == before
    assert active[:-1] == before and active[-1]["content"] == "RACING DURABLE GENERATION"
    assert agent._last_compression_outcome == "persistence_failure"
    assert db.find_live_compression_child(sid) is None
    assert (
        getattr(agent, "_pending_context_engine_compression_notification", None)
        is None
    )
    assert db.get_compression_lock_holder(sid) is None


@pytest.mark.parametrize("kind", ["suffix_mismatch", "invalid_seam"])
def test_partial_suffix_or_seam_rejection_is_truthful_and_releases_fence(
    durable_case, kind
):
    db, sid, source = durable_case
    seen: list[list[dict]] = []
    agent = _agent(db, sid, in_place=True, seen=seen)
    before = copy.deepcopy(source)
    if kind == "suffix_mismatch":
        tail = copy.deepcopy(source[4:])
        tail[0]["api_content"] = "tampered"
        expectation = pytest.raises(ValueError, match="does not match")
    else:
        tail = source[4:]
        agent.context_compressor.compress.side_effect = lambda *_a, **_kw: [
            {"role": "user", "content": "invalid user/user seam"}
        ]
        expectation = ExitStack()  # policy rejection returns unchanged
    with expectation, patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ):
        returned, _ = agent._compress_context(
            source, None, approx_tokens=100_000, force=True, protected_tail=tail
        )
    if kind == "invalid_seam":
        assert returned == before
        assert agent._last_compression_outcome == "rejected_invalid_protected_tail_seam"
    assert db.get_messages_as_conversation(sid) == before
    assert db.get_compression_lock_holder(sid) is None
    assert getattr(agent, "_pending_context_engine_compression_notification", None) is None


def test_tui_prepublication_host_mutation_fails_without_lock_leak(durable_case):
    """A typed-input generation change wins before SQLite and releases history_lock."""
    from tui_gateway.server import _compress_session_history

    db, sid, source = durable_case
    seen: list[list[dict]] = []
    agent = _agent(db, sid, in_place=True, seen=seen)
    lock = threading.RLock()
    session = {"agent": agent, "history": copy.deepcopy(source), "history_version": 1, "history_lock": lock}
    original_claim = None

    def mutate_at_boundary():
        session["history_version"] += 1
        return original_claim()

    def install_mutation(rows, **kwargs):
        result = [{"role": "assistant", "content": SUMMARY}]
        original = agent._claim_compression_host_publication
        nonlocal original_claim
        original_claim = original
        agent._claim_compression_host_publication = mutate_at_boundary
        return result

    agent.context_compressor.compress.side_effect = install_mutation
    with patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ), pytest.raises(RuntimeError, match="history changed"):
        _compress_session_history(session, "here 4", approx_tokens=100_000)
    assert db.get_messages_as_conversation(sid) == source
    assert lock.acquire(blocking=False)
    lock.release()
    assert db.get_compression_lock_holder(sid) is None


def test_host_claim_exception_releases_lease_before_commit_fence(durable_case):
    """A failed host claim preserves its error and never enters commit."""
    from agent.conversation_compression import CompressionCommitFence

    db, sid, source = durable_case
    agent = _agent(db, sid, in_place=True, seen=[])
    fence = CompressionCommitFence()
    host_release = MagicMock()

    def reject_claim():
        raise RuntimeError("host generation rejected")

    agent._claim_compression_host_publication = reject_claim
    with patch.object(fence, "begin_commit", wraps=fence.begin_commit) as begin, patch.object(
        fence, "finish_commit", wraps=fence.finish_commit
    ) as finish, patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ), pytest.raises(RuntimeError, match="host generation rejected"):
        agent._compress_context(
            source, None, approx_tokens=100_000, force=True,
            protected_tail=source[4:], commit_fence=fence,
        )
    begin.assert_not_called()
    finish.assert_not_called()
    host_release.assert_not_called()
    assert db.get_messages_as_conversation(sid) == source
    assert db.get_compression_lock_holder(sid) is None


def test_begin_commit_cancel_after_host_claim_releases_once(durable_case):
    """Fence refusal after a claim releases both host claim and durable lease."""
    from agent.conversation_compression import CompressionCommitFence

    db, sid, source = durable_case
    agent = _agent(db, sid, in_place=True, seen=[])
    fence = CompressionCommitFence()
    host_release = MagicMock()
    agent._claim_compression_host_publication = MagicMock(return_value=host_release)
    with patch.object(fence, "begin_commit", return_value=False) as begin, patch.object(
        fence, "finish_commit", wraps=fence.finish_commit
    ) as finish, patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ):
        returned, _ = agent._compress_context(
            source, None, approx_tokens=100_000, force=True,
            protected_tail=source[4:], commit_fence=fence,
        )
    assert returned == source
    agent._claim_compression_host_publication.assert_called_once_with()
    begin.assert_called_once()
    finish.assert_not_called()
    host_release.assert_called_once_with()
    assert agent._last_compression_outcome == "cancelled_commit_fence"
    assert db.get_messages_as_conversation(sid) == source
    assert db.get_compression_lock_holder(sid) is None
