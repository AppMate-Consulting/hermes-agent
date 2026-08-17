"""Regression: preflight durable-snapshot adoption must not drop the live
turn's un-persisted user input.

``compress_context`` re-reads the durable parent after acquiring the
per-session compression lock.  When another writer (frontend commit,
background review, shared session_id) appended rows in that window,
``len(durable_parent) > len(messages)`` and preflight ADOPTS the snapshot
verbatim (conversation_compression.py, "grew before lease" path).

The in-memory transcript carries the CURRENT turn's un-persisted user
instruction — real user input anchored by ``_persist_user_message_idx`` that
exists ONLY in this agent's memory.  The durable snapshot does not contain it
yet, so a verbatim adoption silently drops it from the transcript that gets
summarized and rotated: the rotation-boundary flush (which runs afterwards,
on the adopted list) only sees rows that are already durable and skips them
by identity, so the live instruction never reaches state.db.

The fix persists the un-persisted tail through the normal flush path
(``conversation_history`` = the already-durable prefix, #68196 boundary)
BEFORE adopting, then re-reads the durable parent so the adopted snapshot
includes the tail.  If that flush fails, adoption is skipped entirely — the
in-memory transcript (which still carries the user's input) goes to the
summarizer instead.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


def _assert_role_alternation(rows):
    roles = [row.get("role") for row in rows]
    assert all(left != right for left, right in zip(roles, roles[1:]))


def _build_agent_with_db(
    db: SessionDB, session_id: str, *, in_place: bool | None = False
):
    """Build an AIAgent wired to ``db`` and pinned to ``session_id``.

    Mirrors the helper in ``test_rotation_flush_persisted_boundary_68196.py``:
    stub the compressor so it returns deterministic output without an LLM
    call, and pin ``compression_in_place=False`` so the legacy rotation path
    (which owns the "grew before lease" adoption) is exercised.
    """
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()

    def _compress(*_a, **_kw):
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    compressor.compress.side_effect = _compress
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    # One-time compression-model feasibility probe would resolve a REAL
    # auxiliary provider; mark it done like test_compression_concurrent_fork.
    agent._compression_feasibility_checked = True
    if in_place is not None:
        agent.compression_in_place = in_place
    return agent


def _contents(rows):
    return [r.get("content") for r in rows]


def _seed_drifted_session(db: SessionDB, session_id: str):
    """Seed the RED shape: in-memory transcript + a longer durable parent.

    Returns ``(agent, messages)`` where ``messages`` is the in-memory
    transcript the caller would hand preflight compression: the originally
    loaded durable rows as plain (unstamped) dicts plus one NEW live user
    instruction.  The DB meanwhile carries TWO extra rows written by a
    concurrent writer, so ``len(durable_parent) > len(messages)``.
    """
    db.create_session(session_id, source="desktop")
    db.append_message(session_id, "user", "persisted question")
    db.append_message(session_id, "assistant", "persisted answer")

    loaded = db.get_messages_as_conversation(session_id)
    messages = [*loaded, {"role": "user", "content": "LIVE USER INSTRUCTION"}]

    agent = _build_agent_with_db(db, session_id)
    # turn_context anchors this at the current-turn user message before
    # preflight compression runs; emulate that anchor.
    agent._persist_user_message_idx = len(messages) - 1

    # Concurrent writer commits rows while preflight waits on the lease.
    db.append_message(session_id, "assistant", "concurrent row 1")
    db.append_message(session_id, "assistant", "concurrent row 2")

    assert len(db.get_messages_as_conversation(session_id)) > len(messages)
    return agent, messages


def test_adoption_preserves_unpersisted_live_user_tail(tmp_path: Path) -> None:
    """Successful rotation carries a caller-only tail into the child once."""
    db = SessionDB(db_path=tmp_path / "state.db")
    agent, messages = _seed_drifted_session(db, "PREFLIGHT_ADOPT_PARENT")
    agent.context_compressor.compress.side_effect = lambda *_args, **_kwargs: [
        {"role": "assistant", "content": "[CONTEXT COMPACTION] summary"}
    ]

    with patch(
        "agent.conversation_compression.estimate_request_tokens_rough",
        side_effect=[120_000, 1_000],
    ):
        returned, _ = agent._compress_context(messages, "sys", approx_tokens=120_000)

    parent_rows = db.get_messages_as_conversation(
        "PREFLIGHT_ADOPT_PARENT", include_inactive=True
    )
    contents = _contents(parent_rows)

    assert contents == [
        "persisted question",
        "persisted answer",
        "concurrent row 1",
        "concurrent row 2",
    ]
    assert agent.session_id != "PREFLIGHT_ADOPT_PARENT"
    child = db.get_messages_as_conversation(agent.session_id)
    assert _contents(child).count("LIVE USER INSTRUCTION") == 1
    assert _contents(returned).count("LIVE USER INSTRUCTION") == 1
    assert agent._persist_user_message_idx == len(returned)


@pytest.mark.parametrize(
    "flush_failure",
    [
        pytest.param("exception", id="raises"),
        pytest.param(False, id="returns-false"),
        pytest.param(None, id="returns-none"),
    ],
)
def test_adoption_skipped_when_preflush_fails_keeps_live_input(
    flush_failure: object, tmp_path: Path
) -> None:
    """A caller-only tail reaches the engine without any parent preflush.

    The patched flush is deliberately unusable in each historical failure
    shape; transactional publication must not call it.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    agent, messages = _seed_drifted_session(db, "PREFLIGHT_ADOPT_FLUSH_FAIL")

    seen: list = []

    def _recording_compress(first_arg, **_kw):
        seen.append(first_arg)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _recording_compress

    def _failing_flush(*_a, **_kw):
        if flush_failure == "exception":
            raise RuntimeError("flush boom")
        return flush_failure

    with patch.object(
        agent, "_flush_messages_to_session_db", side_effect=_failing_flush
    ) as flush:
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert len(seen) == 1
    flush.assert_not_called()
    compress_input = seen[0]
    assert any(
        m.get("content") == "LIVE USER INSTRUCTION" for m in compress_input
    ), (
        "Compression ran on the adopted durable snapshot instead of the "
        "in-memory transcript after the pre-adoption flush failed "
        f"(flush returned {flush_failure!r}) — the live user input would be "
        "dropped from the summary. "
        f"Compress input contents: {_contents(compress_input)!r}"
    )


def test_adopted_parent_is_authoritative_for_engine_admission_task_and_memory(
    tmp_path: Path,
) -> None:
    from agent.conversation_compression import (
        AUTONOMOUS_COMPLETION_BRIDGE_ASSISTANT,
        AUTONOMOUS_COMPLETION_BRIDGE_USER,
    )

    db = SessionDB(db_path=tmp_path / "authoritative.db")
    sid = "AUTHORITATIVE_PARENT"
    db.create_session(sid, source="desktop")
    db.append_message(sid, "user", "stale caller task")
    db.append_message(sid, "assistant", "stale answer")
    caller = db.get_messages_as_conversation(sid)
    db.append_message(sid, "user", "NEWER AUTHORITATIVE TASK")
    db.append_message(sid, "assistant", "task running")
    db.append_message(
        sid,
        "user",
        AUTONOMOUS_COMPLETION_BRIDGE_USER,
        autonomous_completion_provenance=True,
    )
    db.append_message(
        sid,
        "assistant",
        AUTONOMOUS_COMPLETION_BRIDGE_ASSISTANT,
        autonomous_completion_provenance=True,
    )
    db.append_message(
        sid, "user", "[ASYNC DELEGATION COMPLETE child=durable]",
        display_kind="async_delegation_complete",
        autonomous_completion_provenance=True,
    )
    db.append_message(sid, "assistant", "runtime completion handled")

    # Do not override compression_in_place: the production default is the
    # contract under test here.
    agent = _build_agent_with_db(db, sid, in_place=None)
    assert agent.compression_in_place is True
    seen_engine = []
    agent.context_compressor.compress.side_effect = lambda rows, **_kw: (
        seen_engine.append(copy.deepcopy(rows))
        or [{"role": "assistant", "content": "[CONTEXT COMPACTION] summary"}]
    )
    memory_seen = []
    agent._memory_manager = MagicMock()
    agent._memory_manager.build_system_prompt.return_value = (
        "deterministic memory prompt"
    )
    agent._memory_manager.on_pre_compress.side_effect = (
        lambda rows: memory_seen.append(("pre", rows)) or ""
    )
    agent.commit_memory_session = lambda rows: memory_seen.append(("commit", rows))
    estimates = []

    def _estimate(rows, **_kw):
        estimates.append(copy.deepcopy(rows))
        return 20_000 if len(estimates) == 1 else 1_000

    with patch("agent.conversation_compression.estimate_request_tokens_rough", _estimate):
        compressed, _ = agent._compress_context(caller, "sys", approx_tokens=1)

    adopted = seen_engine[0]
    assert _contents(adopted)[-6:] == [
        "NEWER AUTHORITATIVE TASK", "task running",
        AUTONOMOUS_COMPLETION_BRIDGE_USER, AUTONOMOUS_COMPLETION_BRIDGE_ASSISTANT,
        "[ASYNC DELEGATION COMPLETE child=durable]", "runtime completion handled",
    ]
    assert estimates[0] == adopted
    assert all(snapshot == adopted for _, snapshot in memory_seen)
    assert all(snapshot is not adopted for _, snapshot in memory_seen)
    active = db.get_messages_as_conversation(sid)
    assert agent.session_id == sid
    assert _contents(active) == _contents(compressed)
    assert [row.get("role") for row in active] == [
        row.get("role") for row in compressed
    ]
    from agent.context_compressor import ContextCompressor
    contract = ContextCompressor._active_task_contract(active)
    assert contract is not None
    assert contract["content"] == "NEWER AUTHORITATIVE TASK"
    assert agent._persist_user_message_idx == len(adopted)
    _assert_role_alternation(active)


@pytest.mark.parametrize("in_place", [True, False], ids=["in-place", "rotation"])
def test_compress_context_rejects_commit_time_durable_drift(
    tmp_path: Path, in_place: bool
) -> None:
    """The summary hook races the real publication after identity capture.

    This deliberately does not call either SessionDB CAS API directly.  The
    production ``compress_context`` flow captures the authoritative identity,
    invokes the context engine, and then reaches the real publication method.
    Appending from the engine is therefore the exact post-read/pre-commit race.
    """
    from agent import relay_runtime
    from agent.conversation_compression import CompressionCommitFence

    db = SessionDB(db_path=tmp_path / f"drift-{in_place}.db")
    sid = f"COMMIT_DRIFT_{in_place}"
    db.create_session(sid, source="desktop")
    for index in range(8):
        db.append_message(
            sid,
            "user" if index % 2 == 0 else "assistant",
            f"durable row {index} " + ("payload " * 2_000),
        )
    caller = db.get_messages_as_conversation(sid)
    before_all = db.get_messages(sid, include_inactive=True)
    agent = _build_agent_with_db(db, sid, in_place=in_place)
    agent._cached_system_prompt = "stable prompt"
    agent._cached_system_prompt_static = "stable static"
    agent._memory_manager = MagicMock()
    agent._memory_manager.build_system_prompt.return_value = "memory prompt"
    agent.commit_memory_session = MagicMock()
    agent.event_callback = MagicMock()
    boundary = MagicMock()
    fence = CompressionCommitFence()

    def race_after_authoritative_read(rows, **_kwargs):
        assert _contents(rows) == _contents(caller)
        # Simulate a legacy/concurrent writer that does not participate in the
        # compression lease.  The normal append API must reject writes while
        # the lease is held, so insert transactionally beneath that guard.
        def _insert_late_row(conn):
            conn.execute(
                "INSERT INTO messages "
                "(session_id, role, content, timestamp, active) "
                "VALUES (?, ?, ?, ?, 1)",
                (sid, "user", "LATE DURABLE ROW", 0.0),
            )
            conn.execute(
                "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                (sid,),
            )
        db._execute_write(_insert_late_row)
        return [{"role": "assistant", "content": "[CONTEXT COMPACTION] summary"}]

    agent.context_compressor.compress.side_effect = race_after_authoritative_read
    with (
        patch.object(
            relay_runtime.SESSION_COORDINATOR,
            "notify_session_compacted",
            boundary,
        ),
        patch.object(
            db, "archive_and_compact", wraps=db.archive_and_compact
        ) as archive,
        patch.object(
            db, "publish_compression_child", wraps=db.publish_compression_child
        ) as publish,
        patch(
            "agent.conversation_compression.estimate_request_tokens_rough",
            side_effect=[100_000, 1_000],
        ),
    ):
        returned, prompt = agent._compress_context(
            caller, "sys", approx_tokens=100_000, commit_fence=fence
        )

    assert returned is caller
    assert prompt == "stable prompt"
    assert agent.session_id == sid
    assert agent._last_compression_outcome == "persistence_failure"
    assert agent._last_compaction_in_place is False
    assert fence.commit_in_flight is False
    assert db.get_compression_lock_holder(sid) is None
    active = db.get_messages_as_conversation(sid)
    assert _contents(active) == _contents(caller) + ["LATE DURABLE ROW"]
    after_all = db.get_messages(sid, include_inactive=True)
    assert len(after_all) == len(before_all) + 1
    assert all(row["active"] for row in after_all)
    assert db._conn.execute(
        "SELECT id FROM sessions WHERE parent_session_id = ?", (sid,)
    ).fetchall() == []
    assert (archive.call_count, publish.call_count) == (
        (1, 0) if in_place else (0, 1)
    )
    agent._memory_manager.on_pre_compress.assert_not_called()
    agent._memory_manager.on_session_switch.assert_not_called()
    agent.commit_memory_session.assert_not_called()
    agent.event_callback.assert_not_called()
    boundary.assert_not_called()


def test_rotation_publication_failure_keeps_parent_unchanged_and_live_tail_for_retry(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "publication-failure.db")
    sid = "PUBLICATION_FAILURE_PARENT"
    db.create_session(sid, source="desktop")
    db.append_message(sid, "user", "durable task")
    db.append_message(sid, "assistant", "durable answer")
    messages = db.get_messages_as_conversation(sid)
    messages.append({"role": "user", "content": "ordinary live tail"})
    original = copy.deepcopy(messages)
    agent = _build_agent_with_db(db, sid)
    agent._persist_user_message_idx = 2
    agent._cached_system_prompt = "cached prompt"
    agent._cached_system_prompt_static = "cached static"
    agent._memory_manager = MagicMock()
    agent._memory_manager.build_system_prompt.return_value = (
        "deterministic memory prompt"
    )
    agent.commit_memory_session = MagicMock()

    with (
        patch.object(
            db, "publish_compression_child", side_effect=RuntimeError("forced publish failure")
        ),
        patch(
            "agent.conversation_compression.estimate_request_tokens_rough",
            side_effect=[20_000, 1_000],
        ),
    ):
        returned, prompt = agent._compress_context(messages, "sys", approx_tokens=1)

    assert returned == original and messages == original
    assert prompt == "cached prompt"
    assert agent.session_id == sid
    assert _contents(db.get_messages_as_conversation(sid, include_inactive=True)) == [
        "durable task", "durable answer",
    ]
    assert _contents(returned)[-1] == "ordinary live tail"
    assert agent._persist_user_message_idx == 2
    assert agent._cached_system_prompt == "cached prompt"
    assert agent._cached_system_prompt_static == "cached static"
    agent._memory_manager.on_pre_compress.assert_not_called()
    agent.commit_memory_session.assert_not_called()
