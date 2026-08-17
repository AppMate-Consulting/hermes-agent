"""Executable acceptance matrix for atomic partial compression.

Every success case enters through a product host and uses a real SessionDB and
the real AIAgent compression transaction.  Only summary generation, provider
token sizing, and host delivery are deterministic test seams.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import threading
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_state import SessionDB
from agent.context_compressor import SUMMARY_PREFIX


OLD_TASK = "OLD HUMAN TASK MUST DISAPPEAR " + "old-payload " * 1200
LATEST_TASK = [
    {"type": "text", "text": "LATEST PROTECTED HUMAN TASK"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,cHJvdGVjdGVk"}},
]
LATEST_API = "LATEST PROTECTED HUMAN TASK\n[private api sidecar]"
LOOKALIKE = "[ASYNC DELEGATION COMPLETE child=proof-lane]"
CALL_ID = "call_protected_1"
SUMMARY = f"{SUMMARY_PREFIX}\ndurable generated head"


def test_suffix_task_authority_is_sequence_aware_for_autonomous_and_multimodal():
    """Autonomous provenance is not authority; structured human input is."""
    from agent.conversation_compression import (
        _latest_active_human_task_row,
        append_autonomous_completion_provenance,
    )

    autonomous_suffix = []
    append_autonomous_completion_provenance(autonomous_suffix)
    autonomous_suffix.append({
        "role": "user",
        "content": LOOKALIKE,
        "_autonomous_completion_bridge": True,
    })
    assert _latest_active_human_task_row(autonomous_suffix) is None

    multimodal = {"role": "user", "content": copy.deepcopy(LATEST_TASK)}
    assert _latest_active_human_task_row([multimodal]) == multimodal


def _select_transcript_dependent_context(
    request_rows: list[dict], *, conversation_messages=None, **_kwargs
) -> list[dict]:
    """Return a request clone whose system marker reflects its transcript."""
    selected = copy.deepcopy(request_rows)
    assert conversation_messages is not None
    marker = f"selected_conversation_count={len(conversation_messages)}"
    for row in selected:
        if row.get("role") == "system":
            assert isinstance(row.get("content"), str)
            row["content"] = f'{row["content"]}\n{marker}'
            break
    else:
        selected.insert(0, {"role": "system", "content": marker})
    return selected


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
    compressor.select_context.side_effect = lambda rows, **_kwargs: copy.deepcopy(rows)
    agent.context_compressor = compressor
    agent.context_engine = None
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


def _seed_here_one_boundary(
    db: SessionDB, sid: str, *, newer_human: bool
) -> tuple[list[dict], list[dict]]:
    """Seed the two security-sensitive ``/compress here 1`` suffix shapes."""
    from agent.conversation_compression import (
        AUTONOMOUS_COMPLETION_BRIDGE_ASSISTANT,
        AUTONOMOUS_COMPLETION_BRIDGE_USER,
    )
    from hermes_cli.partial_compress import split_history_for_partial_compress

    db.create_session(sid, source="gateway", model="test/model")
    db.append_message(sid, "user", OLD_TASK)
    db.append_message(sid, "assistant", "old answer " + "bulk " * 1200)
    db.append_message(sid, "user", "PRIOR GENUINE HUMAN TASK")
    db.append_message(sid, "assistant", "prior task work completed")
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
    db.append_message(
        sid, "assistant", None,
        tool_calls=[{
            "id": "call_autonomous_tail",
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"cmd":"status"}'},
        }],
    )
    db.append_message(
        sid, "tool", "AUTONOMOUS TAIL TOOL RESULT", tool_name="terminal",
        tool_call_id="call_autonomous_tail", effect_disposition="read_only",
    )
    db.append_message(sid, "assistant", "trusted completion consumed")
    if newer_human:
        db.append_message(
            sid,
            "user",
            copy.deepcopy(LATEST_TASK),
            api_content=LATEST_API,
            display_kind="human_task",
            display_metadata={"private": {"multimodal": True}},
        )
        db.append_message(
            sid, "assistant", None,
            tool_calls=[{
                "id": "call_new_human_tail",
                "type": "function",
                "function": {"name": "terminal", "arguments": '{"cmd":"proof"}'},
            }],
        )
        db.append_message(
            sid, "tool", "NEW HUMAN TAIL TOOL RESULT", tool_name="terminal",
            tool_call_id="call_new_human_tail", effect_disposition="read_only",
        )
        db.append_message(sid, "assistant", "newer task acknowledged")
    source = db.get_messages_as_conversation(sid)
    head, tail = split_history_for_partial_compress(source, 1)
    assert head + tail == source
    if newer_human:
        assert head == source[:-4]
        assert tail == source[-4:]
    else:
        # The two bridge rows and completion row are one provenance unit;
        # the consumer response remains adjacent as part of that exchange.
        assert head == source[:4]
        assert tail == source[4:]
    return source, tail


def _assert_here_one_boundary_result(
    db: SessionDB,
    parent: str,
    source: list[dict],
    protected_tail: list[dict],
    agent,
    host_history: list[dict],
    seen: list[list[dict]],
    publish,
    *,
    newer_human: bool,
) -> None:
    """Assert publication, provenance, classification, and exact host adoption."""
    from agent.context_compressor import ContextCompressor
    from agent.conversation_compression import (
        ACTIVE_TASK_CONTRACT_BRIDGE_AFTER,
        ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE,
        _latest_active_human_task_row,
    )

    active = db.get_messages_as_conversation(agent.session_id)
    assert len(seen) == 1
    assert seen[0] == source[:-len(protected_tail)]
    assert active[-len(protected_tail):] == protected_tail
    assert host_history == active
    if not agent.compression_in_place:
        assert source == db.get_messages_as_conversation(parent)
    assert publish.call_count == 1
    _assert_valid_tool_pairs(active)

    if not newer_human:
        completion = next(
            i for i, row in enumerate(active)
            if row.get("content") == LOOKALIKE
        )
        assert ContextCompressor._completion_has_durable_provenance(
            active, completion
        )
    active_human = _latest_active_human_task_row(active)
    assert active_human is not None
    expected_task = (
        LATEST_TASK if newer_human
        else "PRIOR GENUINE HUMAN TASK"
    )
    assert active_human["content"] == expected_task

    before_indexes = [
        i for i, row in enumerate(active)
        if row.get("content") == ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE
    ]
    after_indexes = [
        i for i, row in enumerate(active)
        if row.get("content") == ACTIVE_TASK_CONTRACT_BRIDGE_AFTER
    ]
    if newer_human:
        assert before_indexes == []
        assert after_indexes == []
        # Structured human authority suppresses head restoration even though
        # it cannot be represented by the text-only trusted contract.
        assert ContextCompressor._active_task_contract(active) is None
        assert active[:-len(protected_tail)].count(protected_tail[0]) == 0
        assert active.count(protected_tail[0]) == 1
    else:
        assert len(before_indexes) == len(after_indexes) == 1
        before = before_indexes[0]
        assert after_indexes == [before + 2]
        contract = ContextCompressor.parse_active_task_contract(
            active[before + 1], allow_projected=True
        )
        assert contract is not None
        assert contract["content"] == expected_task
        assert ContextCompressor._active_task_contract(active) == contract

    if agent.compression_in_place:
        assert agent.session_id == parent
    else:
        assert agent.session_id != parent
        assert db.get_session(agent.session_id)["parent_session_id"] == parent


def _assert_here_one_restart_replay(
    path: Path, session_id: str, expected: list[dict], *, newer_human: bool
) -> None:
    """A fresh SessionDB must replay the same authority and classification."""
    from agent.context_compressor import ContextCompressor
    from agent.conversation_compression import (
        ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE,
        _latest_active_human_task_row,
    )

    reopened = SessionDB(db_path=path)
    replay = reopened.get_messages_as_conversation(session_id)
    assert replay == expected
    if not newer_human:
        completion = next(
            i for i, row in enumerate(replay) if row.get("content") == LOOKALIKE
        )
        assert ContextCompressor._completion_has_durable_provenance(
            replay, completion
        )
    active_human = _latest_active_human_task_row(replay)
    assert active_human is not None
    assert active_human["content"] == (
        LATEST_TASK if newer_human
        else "PRIOR GENUINE HUMAN TASK"
    )
    contract_count = sum(
        row.get("content") == ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE for row in replay
    )
    assert contract_count == (0 if newer_human else 1)
    reopened.close()


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
def test_autonomous_only_suffix_restores_task_contract_across_restart(
    tmp_path: Path, in_place: bool
):
    """A /compress here 1 boundary keeps provenance and the prior human task."""
    from agent.context_compressor import ContextCompressor
    from agent.conversation_compression import (
        ACTIVE_TASK_CONTRACT_BRIDGE_AFTER,
        ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE,
        AUTONOMOUS_COMPLETION_BRIDGE_ASSISTANT,
        AUTONOMOUS_COMPLETION_BRIDGE_USER,
    )
    from hermes_cli.partial_compress import split_history_for_partial_compress

    path = tmp_path / "autonomous-replay.db"
    db = SessionDB(db_path=path)
    sid = "AUTONOMOUS_ONLY_SUFFIX"
    db.create_session(sid, source="cli", model="test/model")
    db.append_message(sid, "user", OLD_TASK)
    db.append_message(sid, "assistant", "old answer " + "bulk " * 1200)
    db.append_message(sid, "user", "LATEST GENUINE HUMAN TASK")
    db.append_message(sid, "assistant", "work completed")
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
    db.append_message(sid, "assistant", "completion consumed")
    source = db.get_messages_as_conversation(sid)
    head, tail = split_history_for_partial_compress(source, 1)
    assert head == source[:4]
    assert tail == source[4:]

    seen: list[list[dict]] = []
    agent = _agent(db, sid, in_place=in_place, seen=seen)
    with patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ):
        published, _ = agent._compress_context(
            source, None, approx_tokens=100_000, force=True,
            protected_tail=tail,
        )
    active_id = agent.session_id
    assert published[-len(tail):] == tail
    db.close()

    reopened = SessionDB(db_path=path)
    replay = reopened.get_messages_as_conversation(active_id)
    assert replay[-len(tail):] == tail
    completion_index = len(replay) - 2
    assert ContextCompressor._completion_has_durable_provenance(
        replay, completion_index
    )
    assert ContextCompressor._latest_user_task_snapshot(replay) is not None
    contract_indexes = [
        i for i, row in enumerate(replay)
        if row.get("content") == ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE
    ]
    assert len(contract_indexes) == 1
    before = contract_indexes[0]
    assert replay[before + 2]["content"] == ACTIVE_TASK_CONTRACT_BRIDGE_AFTER
    contract = ContextCompressor.parse_active_task_contract(
        replay[before + 1], allow_projected=True
    )
    assert contract is not None
    assert contract["content"] == "LATEST GENUINE HUMAN TASK"
    reopened.close()


@pytest.mark.parametrize("in_place", [True, False], ids=["in_place", "rotation"])
def test_push_wake_provenance_survives_restart_and_partial_compaction(
    tmp_path: Path, in_place: bool
):
    """The real push wake is trusted, durable, and never becomes task authority."""
    from agent.context_compressor import ContextCompressor
    from agent.conversation_compression import (
        ACTIVE_TASK_CONTRACT_BRIDGE_AFTER,
        ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE,
        AUTONOMOUS_COMPLETION_BRIDGE_ASSISTANT,
        AUTONOMOUS_COMPLETION_BRIDGE_USER,
        _latest_active_human_task_row,
    )
    from gateway.config import Platform
    from gateway.run import _event_conversation_forwarding_metadata
    from gateway.session import SessionSource
    from gateway.wake import deliver_wake
    from hermes_cli.partial_compress import split_history_for_partial_compress

    path = tmp_path / "push-wake-provenance.db"
    sid = "PUSH_WAKE_PROVENANCE"
    notification = "[ASYNC DELEGATION COMPLETE child=push-proof]"
    genuine_task = "PUSH PATH GENUINE HUMAN TASK"
    db = SessionDB(db_path=path)
    db.create_session(sid, source="telegram", model="test/model")
    db.append_message(sid, "user", OLD_TASK)
    db.append_message(sid, "assistant", "old answer " + "bulk " * 1200)
    db.append_message(sid, "user", genuine_task)
    db.append_message(sid, "assistant", "genuine task completed")

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), patch(
        "run_agent.get_tool_definitions", return_value=[]
    ), patch("run_agent.check_toolset_requirements", return_value={}), patch(
        "run_agent.OpenAI"
    ):
        from run_agent import AIAgent

        delivery_agent = AIAgent(
            api_key="test-key", base_url="https://invalid.test/v1",
            model="test/model", quiet_mode=True, session_db=db, session_id=sid,
            skip_context_files=True, skip_memory=True,
        )
    delivery_agent.compression_enabled = False
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="push completion consumed", tool_calls=None,
                reasoning_content=None, reasoning=None,
            ),
            finish_reason="stop",
        )],
        model="test/model", usage=None,
    )
    delivery_agent.client = MagicMock()
    delivery_agent.client.chat.completions.create.return_value = response

    class PersistingPushAdapter:
        handled = []

        async def handle_message(self, event):
            self.handled.append(event)
            display_kind, autonomous = _event_conversation_forwarding_metadata(event)
            delivery_agent.run_conversation(
                event.text,
                conversation_history=db.get_messages_as_conversation(sid),
                persist_user_display_kind=display_kind,
                persist_user_is_autonomous_completion=autonomous,
            )

    adapter = PersistingPushAdapter()
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="trusted-chat", chat_type="group"
    )
    asyncio.run(deliver_wake(adapter, text=notification, source=source))
    assert len(adapter.handled) == 1
    delivered_event = adapter.handled[0]
    assert delivered_event.internal is True
    assert delivered_event.autonomous_completion is True
    assert delivered_event.allow_gateway_control is True

    pushed = db.get_messages_as_conversation(sid)
    exact_suffix = pushed[-4:]
    assert [(row["role"], row.get("content")) for row in exact_suffix] == [
        ("user", AUTONOMOUS_COMPLETION_BRIDGE_USER),
        ("assistant", AUTONOMOUS_COMPLETION_BRIDGE_ASSISTANT),
        ("user", notification),
        ("assistant", "push completion consumed"),
    ]
    assert [
        row.get("_autonomous_completion_bridge") is True for row in exact_suffix
    ] == [True, True, True, False]
    assert exact_suffix[2]["display_kind"] == "internal_notification"
    assert ContextCompressor._completion_has_durable_provenance(
        pushed, len(pushed) - 2
    )
    assert _latest_active_human_task_row(pushed)["content"] == genuine_task

    db.close()
    db = SessionDB(db_path=path)
    restarted = db.get_messages_as_conversation(sid)
    assert restarted == pushed
    assert ContextCompressor._completion_has_durable_provenance(
        restarted, len(restarted) - 2
    )
    assert _latest_active_human_task_row(restarted)["content"] == genuine_task

    head, protected = split_history_for_partial_compress(restarted, 1)
    assert head == restarted[:-4]
    assert protected == exact_suffix
    seen: list[list[dict]] = []
    compression_agent = _agent(db, sid, in_place=in_place, seen=seen)
    with patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ):
        published, _ = compression_agent._compress_context(
            restarted, None, approx_tokens=100_000, force=True,
            protected_tail=protected,
        )
    active_sid = compression_agent.session_id
    assert len(seen) == 1
    assert seen[0] == head
    assert published[-4:] == exact_suffix
    durable = db.get_messages_as_conversation(active_sid)
    assert durable == published
    assert durable[-4:] == exact_suffix
    assert durable.count(exact_suffix[0]) == 1
    assert _latest_active_human_task_row(durable)["content"] == genuine_task
    before = [
        i for i, row in enumerate(durable)
        if row.get("content") == ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE
    ]
    after = [
        i for i, row in enumerate(durable)
        if row.get("content") == ACTIVE_TASK_CONTRACT_BRIDGE_AFTER
    ]
    assert len(before) == len(after) == 1
    assert after == [before[0] + 2]
    contract = ContextCompressor.parse_active_task_contract(
        durable[before[0] + 1], allow_projected=True
    )
    assert contract is not None
    assert contract["content"] == genuine_task
    assert ContextCompressor._active_task_contract(durable) == contract
    assert ContextCompressor._completion_has_durable_provenance(
        durable, len(durable) - 2
    )
    if in_place:
        assert active_sid == sid
    else:
        assert active_sid != sid
        assert db.get_session(active_sid)["parent_session_id"] == sid

    db.close()
    reopened = SessionDB(db_path=path)
    replay = reopened.get_messages_as_conversation(active_sid)
    assert replay == durable
    assert replay[-4:] == exact_suffix
    assert _latest_active_human_task_row(replay)["content"] == genuine_task
    assert ContextCompressor._completion_has_durable_provenance(
        replay, len(replay) - 2
    )
    reopened.close()


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
@pytest.mark.parametrize(
    "newer_human", [False, True],
    ids=["autonomous_suffix", "new_human_suffix"],
)
def test_tui_here_one_real_sessiondb_boundary_proof(
    tmp_path: Path, in_place: bool, newer_human: bool
):
    """The TUI's shared partial-compress entry preserves the trusted boundary."""
    from tui_gateway.server import _compress_session_history

    path = tmp_path / "tui-here-one.db"
    db = SessionDB(db_path=path)
    sid = "TUI_HERE_ONE"
    source, tail = _seed_here_one_boundary(db, sid, newer_human=newer_human)
    source_before = copy.deepcopy(source)
    seen: list[list[dict]] = []
    agent = _agent(db, sid, in_place=in_place, seen=seen)
    session = {
        "agent": agent,
        "history": copy.deepcopy(source),
        "history_version": 9,
        "history_lock": threading.RLock(),
        "session_key": sid,
    }
    method = "archive_and_compact" if in_place else "publish_compression_child"
    with patch.object(db, method, wraps=getattr(db, method)) as publish, patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ):
        _compress_session_history(session, "here 1", approx_tokens=100_000)

    assert source == source_before  # the host may adopt, never rewrite its input
    assert session["history_version"] == 10
    _assert_here_one_boundary_result(
        db, sid, source, tail, agent, session["history"], seen, publish,
        newer_human=newer_human,
    )
    active_id = agent.session_id
    expected_replay = copy.deepcopy(session["history"])
    db.close()
    _assert_here_one_restart_replay(
        path, active_id, expected_replay, newer_human=newer_human
    )


@pytest.mark.parametrize("in_place", [True, False], ids=["in_place", "rotation"])
@pytest.mark.asyncio
async def test_gateway_slash_compress_real_sessiondb_partial_matrix(
    durable_case, in_place
):
    """GatewaySlashCommandsMixin real /compress handler uses the same transaction."""
    from agent.conversation_loop import finalize_provider_request
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionEntry, SessionSource
    from gateway.slash_commands import GatewaySlashCommandsMixin

    db, sid, source_rows = durable_case
    seen: list[list[dict]] = []
    prepared = _agent(db, sid, in_place=in_place, seen=seen)
    prepared.provider = "openrouter"
    prepared._use_prompt_caching = False
    prepared._build_api_kwargs = lambda api_messages, tools_for_api=None: {
        "model": prepared.model,
        "messages": copy.deepcopy(api_messages),
        "tools": copy.deepcopy(tools_for_api if tools_for_api is not None else prepared.tools),
    }
    prepared._reapply_reasoning_echo_for_provider = lambda api_messages: 0
    prepared._sanitize_api_messages = lambda messages: copy.deepcopy(messages)
    prepared._drop_thinking_only_and_merge_users = lambda messages, **_kwargs: copy.deepcopy(messages)
    prepared.context_compressor.select_context.side_effect = (
        _select_transcript_dependent_context
    )
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
    actual_payloads = []
    real_finalize = finalize_provider_request

    def capture_finalize(*args, **kwargs):
        result = real_finalize(*args, **kwargs)
        actual_payloads.append(copy.deepcopy(result["payload"]))
        return result
    def identity_redecorate(_agent, api_messages, *, moa_prepared=None, tools_for_api=None, **_kwargs):
        return copy.deepcopy(api_messages), moa_prepared, copy.deepcopy(tools_for_api or [])
    expected_rows = [{"role": "assistant", "content": SUMMARY}] + copy.deepcopy(
        source_rows[4:]
    )
    prompt = prepared._cached_system_prompt
    tools = prepared.tools or []
    source_before = copy.deepcopy(source_rows)
    expected_before = copy.deepcopy(expected_rows)
    prompt_before = copy.deepcopy(prompt)
    tools_before = copy.deepcopy(tools)
    source_payload = finalize_provider_request(
        prepared,
        source_rows,
        system_message=prompt,
        tools=tools,
    )["payload"]
    expected_payload = finalize_provider_request(
        prepared,
        expected_rows,
        system_message=prompt,
        tools=tools,
    )["payload"]
    source_marker = f"selected_conversation_count={len(source_rows)}"
    expected_marker = f"selected_conversation_count={len(expected_rows)}"
    assert source_marker != expected_marker
    assert source_marker in repr(source_payload)
    assert expected_marker in repr(expected_payload)
    assert source_payload != expected_payload
    assert source_rows == source_before
    assert expected_rows == expected_before
    assert prompt == prompt_before
    assert tools == tools_before

    method = "archive_and_compact" if in_place else "publish_compression_child"
    with patch("run_agent.AIAgent", return_value=prepared), patch.object(
        db, method, wraps=getattr(db, method)
    ) as publish, patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ), patch(
        "agent.conversation_loop.finalize_provider_request",
        side_effect=capture_finalize,
    ), patch(
        "agent.conversation_loop._redecorate_prompt_cache_for_provider",
        side_effect=identity_redecorate,
    ), patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100_000), patch(
        "agent.manual_compression_feedback.summarize_manual_compression",
        return_value={"headline": "ok", "token_line": "small", "note": "", "noop": False},
    ):
        await host._handle_compress_command_inner(event)
    assert len(actual_payloads) == 2
    assert actual_payloads[0] != actual_payloads[1]
    active_id = entry.session_id
    _assert_success(db, sid, source_rows, prepared, db.get_messages_as_conversation(active_id), seen, publish)


@pytest.mark.parametrize("in_place", [True, False], ids=["in_place", "rotation"])
@pytest.mark.parametrize(
    "newer_human", [False, True],
    ids=["autonomous_suffix", "new_human_suffix"],
)
@pytest.mark.asyncio
async def test_gateway_slash_here_one_real_sessiondb_boundary_proof(
    tmp_path: Path, in_place: bool, newer_human: bool
):
    """The actual gateway slash route preserves and correctly classifies the suffix."""
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionEntry, SessionSource
    from gateway.slash_commands import GatewaySlashCommandsMixin

    path = tmp_path / "gateway-here-one.db"
    db = SessionDB(db_path=path)
    sid = "GATEWAY_HERE_ONE"
    source_rows, tail = _seed_here_one_boundary(
        db, sid, newer_human=newer_human
    )
    source_before = copy.deepcopy(source_rows)
    seen: list[list[dict]] = []
    prepared = _agent(db, sid, in_place=in_place, seen=seen)
    prepared.provider = "openrouter"
    prepared._use_prompt_caching = False
    prepared._build_api_kwargs = lambda api_messages, tools_for_api=None: {
        "model": prepared.model,
        "messages": copy.deepcopy(api_messages),
        "tools": copy.deepcopy(
            tools_for_api if tools_for_api is not None else prepared.tools
        ),
    }
    prepared._reapply_reasoning_echo_for_provider = lambda api_messages: 0
    prepared._sanitize_api_messages = lambda messages: copy.deepcopy(messages)
    prepared._drop_thinking_only_and_merge_users = (
        lambda messages, **_kwargs: copy.deepcopy(messages)
    )
    prepared.context_compressor.select_context.side_effect = (
        lambda rows, **_kwargs: copy.deepcopy(rows)
    )
    source = SessionSource(
        platform=Platform.TELEGRAM, user_id="proof-user", chat_id="proof-chat",
        chat_type="dm",
    )
    event = MessageEvent(text="/compress here 1", source=source, message_id="proof")
    entry = SessionEntry(
        "telegram:proof-user:proof-chat", sid, datetime.now(), datetime.now(),
        origin=source, platform=Platform.TELEGRAM, chat_type="dm",
    )
    routed: list[tuple] = []

    class Store:
        async def get_or_create_session(self, actual_source):
            routed.append(("lookup", actual_source))
            return entry
        async def load_transcript(self, active):
            routed.append(("load", active))
            return db.get_messages_as_conversation(active)
        async def update_session(self, active, **kwargs):
            routed.append(("update", active, kwargs))
        async def _save(self): return None

    class Host(GatewaySlashCommandsMixin):
        async_session_store = Store()
        _session_db = SimpleNamespace(
            _db=db, get_session=AsyncMock(return_value=db.get_session(sid))
        )
        def _session_key_for_source(self, actual_source):
            assert actual_source is source
            return entry.session_key
        def _resolve_session_agent_runtime(self, **kwargs):
            routed.append(("runtime", kwargs["source"], kwargs["session_key"]))
            return "test/model", {"api_key": "test-key"}
        async def _run_in_executor_with_context(self, fn): return fn()
        async def _cleanup_agent_resources_off_loop(self, *_a, **_kw): return None
        def _evict_cached_agent(self, *_a): return None
        def _sync_telegram_topic_binding(self, *_a, **_kw): return None

    host = Host()
    method = "archive_and_compact" if in_place else "publish_compression_child"

    def identity_redecorate(
        _agent, messages, *, moa_prepared, tools_for_api, **_kwargs
    ):
        return copy.deepcopy(messages), moa_prepared, copy.deepcopy(tools_for_api)

    with patch("run_agent.AIAgent", return_value=prepared), patch.object(
        db, method, wraps=getattr(db, method)
    ) as publish, patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ), patch(
        "agent.model_metadata.estimate_request_tokens_rough", return_value=100_000
    ), patch(
        "agent.manual_compression_feedback.summarize_manual_compression",
        return_value={"headline": "ok", "token_line": "small", "note": "", "noop": False},
    ), patch(
        "agent.conversation_loop._redecorate_prompt_cache_for_provider",
        side_effect=identity_redecorate,
    ):
        await host._handle_compress_command_inner(event)

    assert source_rows == source_before
    assert routed[0] == ("lookup", source)
    assert routed[1] == ("load", sid)
    assert ("runtime", source, entry.session_key) in routed
    assert entry.session_id == prepared.session_id
    _assert_here_one_boundary_result(
        db, sid, source_rows, tail, prepared,
        db.get_messages_as_conversation(entry.session_id), seen, publish,
        newer_human=newer_human,
    )
    active_id = entry.session_id
    expected_replay = db.get_messages_as_conversation(active_id)
    db.close()
    _assert_here_one_restart_replay(
        path, active_id, expected_replay, newer_human=newer_human
    )


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


def test_postcommit_host_adoption_failure_preserves_rotated_authority(durable_case):
    """A host failure after SQLite commit cannot resurrect the durable parent."""
    from agent.conversation_compression import (
        CompressionCommitFence,
        CompressionCommittedPostpublicationError,
    )

    db, parent, source = durable_case
    agent = _agent(db, parent, in_place=False, seen=[])
    fence = CompressionCommitFence()
    host_release = MagicMock()
    agent._claim_compression_host_publication = MagicMock(
        return_value=host_release
    )
    agent._adopt_compression_host_publication = MagicMock(
        side_effect=RuntimeError("injected postcommit adoption failure")
    )

    with patch.object(
        db, "publish_compression_child", wraps=db.publish_compression_child
    ) as publish, patch(
        "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
        side_effect=[100_000, 1_000],
    ), pytest.raises(
        CompressionCommittedPostpublicationError,
        match="postcommit adoption failure",
    ) as caught:
        agent._compress_context(
            source,
            None,
            approx_tokens=100_000,
            force=True,
            protected_tail=source[4:],
            commit_fence=fence,
        )

    child = agent.session_id
    assert caught.value.session_id == child
    assert caught.value.transcript == db.get_messages_as_conversation(child)
    assert caught.value.in_place is False
    assert isinstance(caught.value.cause, RuntimeError)
    assert child != parent
    assert agent._compression_durable_commit_occurred is True
    assert db.get_session(child)["parent_session_id"] == parent
    assert db.get_messages_as_conversation(child)
    assert db.get_session(parent)["end_reason"] == "compression"
    assert db.find_live_compression_child(parent)["id"] == child
    publish.assert_called_once()
    agent._adopt_compression_host_publication.assert_called_once()
    host_release.assert_called_once_with()
    assert fence.commit_in_flight is False
    assert db.get_compression_lock_holder(parent) is None


def test_postcommit_readback_failure_is_typed_and_retriable(durable_case):
    from agent.conversation_compression import (
        CompressionCommittedPostpublicationError,
    )

    db, sid, source = durable_case
    agent = _agent(db, sid, in_place=True, seen=[])
    real_read = db.get_messages_as_conversation
    calls = 0

    def fail_once(session_id, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected durable readback failure")
        return real_read(session_id, *args, **kwargs)

    with patch.object(db, "archive_and_compact", wraps=db.archive_and_compact) as publish, \
         patch.object(db, "get_messages_as_conversation", side_effect=fail_once), \
         patch(
             "agent.conversation_compression.estimate_finalized_payload_tokens_rough",
             side_effect=[100_000, 1_000],
         ), pytest.raises(CompressionCommittedPostpublicationError) as caught:
        agent._compress_context(source, None, approx_tokens=100_000, force=True)

    assert caught.value.in_place is True
    assert caught.value.session_id == sid
    assert caught.value.transcript is None
    assert caught.value.load_authoritative_transcript(agent) == real_read(sid)
    assert agent._last_compression_outcome == "committed_postpublication_sync_error"
    publish.assert_called_once()
    assert db.get_compression_lock_holder(sid) is None


def test_authoritative_reload_never_falls_back_to_candidate():
    from agent.conversation_compression import (
        CompressionCommittedPostpublicationError,
    )

    db = MagicMock()
    db.get_messages_as_conversation.side_effect = RuntimeError("read unavailable")
    agent = SimpleNamespace(_session_db=db)
    error = CompressionCommittedPostpublicationError(
        session_id="committed-child",
        transcript=None,
        candidate_transcript=[{"role": "user", "content": "speculative"}],
        in_place=False,
        cause=RuntimeError("initial read failed"),
    )

    with pytest.raises(RuntimeError, match="read unavailable"):
        error.load_authoritative_transcript(agent)
    db.get_messages_as_conversation.assert_called_once_with("committed-child")


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
