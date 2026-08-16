"""Tests for in-place context compaction (config: compression.in_place, #38763).

When ``compression.in_place`` is True, ``compress_context()`` rewrites the
message list and rebuilds the system prompt but keeps the SAME ``session_id``:
no ``end_session``, no ``parent_session_id`` child row, no ``name #N`` title
renumber, no flush-cursor reset. This eliminates the session-rotation bug
cluster (#33618 /goal loss, #14238 lost response, #33907 orphans, #45117 search
gaps, #42228 null cwd). When the flag is False (default), rotation behaves
exactly as before.
"""

import os
import copy
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_agent(session_db, session_id, *, in_place):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = in_place
    # Mock the compressor to return a deterministic shrunk transcript so the
    # test exercises the DB-mutation path, not summarization quality.
    def _fake_compress(messages, current_tokens=None, focus_topic=None, force=False):
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"},
            {"role": "assistant", "content": "recent reply"},
        ]

    agent.context_compressor.compress = _fake_compress
    agent.context_compressor._last_compress_aborted = False
    agent.context_compressor._last_summary_error = None
    agent.context_compressor.compression_count = 1
    return agent


def _seed(db, sid, title, n=8):
    db.create_session(sid, "cli", model="test/model")
    db.set_session_title(sid, title)
    for i in range(n):
        db.append_message(
            session_id=sid,
            role="user" if i % 2 == 0 else "assistant",
            content=f"msg {i}",
        )


def _materially_compressible_messages(n=8):
    """Input whose like-for-like estimate can reclaim the 4K token floor."""
    return [
        {"role": "user", "content": f"m{i} " + ("payload " * 3_000)}
        for i in range(n)
    ]


class TestInPlaceCompaction:
    def test_in_place_keeps_same_session_id(self):
        """In-place mode: id unchanged, no child row, no rename, history kept."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_120000_aaaaaa"
            _seed(db, sid, "my-research")
            agent = _make_agent(db, sid, in_place=True)
            agent._last_flushed_db_idx = 5

            messages = _materially_compressible_messages()
            compressed, _sp = compress_context(
                agent, messages, approx_tokens=100_000, system_message="sys"
            )

            # Identity never moved.
            assert agent.session_id == sid
            # No continuation row forked.
            child = db._conn.execute(
                "SELECT id FROM sessions WHERE parent_session_id = ?", (sid,)
            ).fetchall()
            assert child == []
            # Session not ended; title untouched (no "#2").
            row = db.get_session(sid)
            assert row["end_reason"] is None
            assert row["title"] == "my-research"
            # DURABLE, NON-DESTRUCTIVE compaction (the core invariant, per
            # Teknium's review): the LIVE context is the compacted set, but the
            # pre-compaction turns are PRESERVED on disk (active=0), not deleted
            # — searchable + recoverable under the SAME id. A resume reloads the
            # compacted set so compaction actually shrinks the live session and
            # doesn't immediately re-compact (#38763).
            reloaded = db.get_messages_as_conversation(sid)
            assert len(reloaded) == 3
            assert [m.get("content") for m in reloaded] == [
                "[CONTEXT COMPACTION] summary of prior turns",
                "recent reply",
                messages[-1]["content"],
            ]
            assert row["message_count"] == 3  # live (active) count
            # NON-DESTRUCTIVE: the 8 seeded originals survive at active=0
            # alongside the 3 compacted rows — nothing was DELETEd.
            all_rows = db.get_messages(sid, include_inactive=True)
            assert len(all_rows) == 11
            archived = [m for m in all_rows if not m.get("active", 1)]
            assert len(archived) == 8
            # The originals remain FTS-searchable (active=0 is a content-
            # preserving UPDATE; the fts triggers don't key on active).
            hit = db._conn.execute(
                "SELECT 1 FROM messages_fts f JOIN messages m ON m.id = f.rowid "
                "WHERE m.session_id = ? AND messages_fts MATCH 'msg' AND m.active = 0 "
                "LIMIT 1",
                (sid,),
            ).fetchone()
            assert hit is not None
            # Flush identity/cursor reset so next-turn appends diff against the
            # compacted transcript (rebuilds the identity set on next flush).
            assert agent._last_flushed_db_idx == 0
            assert agent._flushed_db_message_ids == set()
            # Rotation-independent in-place signal set for the gateway.
            assert agent._last_compaction_in_place is True
            # Live transcript actually shrank.
            assert len(compressed) == 3

    def test_in_place_alternation_preserved(self):
        """The compacted list must not introduce consecutive same-role messages."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_120500_cccccc"
            _seed(db, sid, "alt")
            agent = _make_agent(db, sid, in_place=True)
            messages = _materially_compressible_messages()
            compressed, _ = compress_context(
                agent, messages, approx_tokens=100_000, system_message="sys"
            )
            roles = [m["role"] for m in compressed if m.get("role") != "system"]
            assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


    def test_rotation_still_preflushes(self):
        """Rotation MUST pre-flush so current-turn messages survive in the
        preserved old (parent) session before it is ended (#47202)."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            _seed(db, "rot_flush", "f")
            agent = _make_agent(db, "rot_flush", in_place=False)
            calls = {"n": 0}
            agent._flush_messages_to_session_db = lambda *a, **k: calls.__setitem__(
                "n", calls["n"] + 1
            )
            compress_context(
                agent, _materially_compressible_messages(),
                approx_tokens=100_000, system_message="sys",
            )
            assert calls["n"] == 1


class TestRotationFallbackWhenFlagOff:
    def test_rotation_when_flag_off(self):
        """Rotation is now the OPT-OUT fallback (default flipped to in-place in
        #38763). With in_place=False explicitly set, legacy rotation is
        unchanged — forks a renamed continuation session."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_130000_bbbbbb"
            _seed(db, sid, "my-research")
            agent = _make_agent(db, sid, in_place=False)
            agent._last_flushed_db_idx = 5

            messages = _materially_compressible_messages()
            compress_context(
                agent, messages, approx_tokens=100_000, system_message="sys"
            )

            # Identity rotated to a fresh id.
            assert agent.session_id != sid
            # Old session ended via compression; continuation forked and
            # carries the SAME name. Compression is an internal detail — the
            # conversation didn't change topic, so it must not be renumbered
            # into "my-research #2" and shown as a separate piece of work.
            assert db.get_session(sid)["end_reason"] == "compression"
            child = db._conn.execute(
                "SELECT id, title FROM sessions WHERE parent_session_id = ?", (sid,)
            ).fetchall()
            assert len(child) == 1
            assert child[0]["title"] == "my-research"
            # The compacted child is persisted atomically at the rotation
            # boundary, so a headless process killed before finalization can
            # still resume it without duplicating the three handoff messages.
            assert agent._last_flushed_db_idx == 3
            assert [m.get("content") for m in db.get_messages_as_conversation(agent.session_id)] == [
                "[CONTEXT COMPACTION] summary of prior turns",
                "recent reply",
                messages[-1]["content"],
            ]
            # Rotation mode does NOT set the in-place signal.
            assert getattr(agent, "_last_compaction_in_place", False) is False


class TestInPlaceSignalForGateway:
    """compress_context must expose a rotation-independent flag the gateway can
    read (instead of an id-change diff) to re-baseline transcript handling."""

    def test_signal_set_on_in_place_unset_on_rotation(self):
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            # in-place → flag True
            _seed(db, "s_ip", "ip")
            a_ip = _make_agent(db, "s_ip", in_place=True)
            compress_context(
                a_ip, _materially_compressible_messages(),
                approx_tokens=100_000, system_message="sys",
            )
            assert a_ip._last_compaction_in_place is True

            # rotation → flag False
            _seed(db, "s_rot", "rot")
            a_rot = _make_agent(db, "s_rot", in_place=False)
            compress_context(
                a_rot, _materially_compressible_messages(),
                approx_tokens=100_000, system_message="sys",
            )
            assert a_rot._last_compaction_in_place is False


class TestInPlaceConfigDefault:
    def test_flag_defaults_on(self):
        """In-place is the default as of #38763 (rotation is now opt-out via
        compression.in_place: false)."""
        from hermes_cli.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["compression"].get("in_place") is True


class TestInPlaceAntiGrowthGuard:
    """A compression whose result is LARGER than its input must never be
    persisted. In-place compaction commits inside compress_context() via
    archive_and_compact — BEFORE the gateway's rotation-only anti-growth
    guard (#83339) can inspect the result — so the guard must live at the
    commit site and cover the in-place path too. Observed failure: session
    hygiene "compressed" 426 -> 426 msgs, ~379,216 -> ~687,888 tokens and
    durably persisted the growth."""

    def test_in_place_refuses_growing_compression(self):
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_antigrow"
            _seed(db, sid, "grow")
            agent = _make_agent(db, sid, in_place=True)
            agent._last_flushed_db_idx = 5

            def _growing_compress(messages, current_tokens=None, focus_topic=None, force=False):
                # A "summary" bigger than the entire input transcript.
                return [
                    {"role": "user", "content": "X" * 200_000},
                    {"role": "assistant", "content": "tiny tail"},
                ]

            agent.context_compressor.compress = _growing_compress
            messages = _materially_compressible_messages()
            compressed, _sp = compress_context(
                agent, messages, approx_tokens=100_000, system_message="sys"
            )

            # Guard refused: the original transcript is returned untouched.
            assert compressed == messages
            # No in-place commit signal — nothing was persisted.
            assert getattr(agent, "_last_compaction_in_place", False) is False
            # Durable state is byte-for-byte the pre-compression live set:
            # nothing archived, nothing inserted.
            reloaded = db.get_messages_as_conversation(sid)
            assert [m["content"] for m in reloaded] == [f"msg {i}" for i in range(8)]
            all_rows = db.get_messages(sid, include_inactive=True)
            assert len(all_rows) == 8
            assert not any(not m.get("active", 1) for m in all_rows)
            # Session identity untouched.
            assert agent.session_id == sid
            assert db.get_session(sid)["end_reason"] is None

    def test_request_overhead_can_make_message_shrink_marginal(self):
        """Admission uses comparable full requests, never message-only size."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_request_estimator"
            _seed(db, sid, "request")
            agent = _make_agent(db, sid, in_place=True)
            messages = _materially_compressible_messages()
            before_rows = db.get_messages(sid, include_inactive=True)

            estimates = []

            def _request_estimate(candidate, *, system_prompt="", tools=None):
                # Message-only sizing shrinks dramatically, but the exact
                # request overhead leaves only 3K reclaim (< the 9.6K floor).
                estimates.append(copy.deepcopy(candidate))
                return (100_000, 97_000)[len(estimates) - 1]

            with patch(
                "agent.conversation_compression.estimate_request_tokens_rough",
                side_effect=_request_estimate,
            ):
                returned, _ = compress_context(
                    agent, messages, approx_tokens=100_000, system_message="sys"
                )

            assert returned is messages
            assert agent._last_compression_outcome == "rejected_below_minimum_reclaim"
            cooldown = db.get_compression_failure_cooldown(sid)
            assert cooldown is not None
            assert cooldown["error"] == "below_minimum_reclaim"
            assert len(cooldown["error"]) < 256
            fresh = _make_agent(db, sid, in_place=True)
            fresh.context_compressor.bind_session_state(db, sid)
            assert fresh.context_compressor.get_active_compression_failure_cooldown()
            after_rows = db.get_messages(sid, include_inactive=True)
            assert [(r["id"], r["active"]) for r in after_rows] == [
                (r["id"], r["active"]) for r in before_rows
            ]

    @pytest.mark.parametrize(
        ("request_out", "outcome"),
        [
            (101_000, "rejected_would_grow"),
            (100_000, "rejected_no_progress"),
            (97_000, "rejected_below_minimum_reclaim"),
        ],
    )
    def test_policy_rejection_precedes_every_persistence_side_effect(
        self, request_out, outcome
    ):
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = f"policy_{outcome}"
            _seed(db, sid, "policy")
            agent = _make_agent(db, sid, in_place=True)
            messages = _materially_compressible_messages()
            before = [(r["id"], r["active"]) for r in db.get_messages(
                sid, include_inactive=True
            )]
            agent.commit_memory_session = MagicMock()
            side_effects = [
                "archive_and_compact", "publish_compression_child",
                "end_session", "create_session",
            ]
            spies = {
                name: patch.object(db, name, wraps=getattr(db, name))
                for name in side_effects if hasattr(db, name)
            }
            started = {name: spy.start() for name, spy in spies.items()}
            agent._flush_messages_to_session_db = MagicMock()
            try:
                estimates = iter((100_000, request_out))
                with patch(
                    "agent.conversation_compression.estimate_request_tokens_rough",
                    side_effect=lambda candidate, **kwargs: next(estimates),
                ):
                    returned, _ = compress_context(
                        agent, messages, approx_tokens=100_000,
                        system_message="sys",
                    )
            finally:
                for spy in spies.values():
                    spy.stop()
            assert returned is messages
            assert agent._last_compression_outcome == outcome
            agent.commit_memory_session.assert_not_called()
            agent._flush_messages_to_session_db.assert_not_called()
            assert all(mock.call_count == 0 for mock in started.values())
            after = [(r["id"], r["active"]) for r in db.get_messages(
                sid, include_inactive=True
            )]
            assert after == before

    @pytest.mark.parametrize("rejection_cooldown_seconds", [60.0, None])
    def test_compressor_no_progress_has_no_persistence_side_effects(
        self, rejection_cooldown_seconds
    ):
        """Equivalent compressor output is rejected before request admission."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = f"compressor_no_progress_{rejection_cooldown_seconds}"
            _seed(db, sid, "no-progress")
            agent = _make_agent(db, sid, in_place=True)
            messages = _materially_compressible_messages()
            agent.context_compressor.compress = lambda current, **kwargs: current
            before = [(r["id"], r["active"]) for r in db.get_messages(
                sid, include_inactive=True
            )]
            agent.commit_memory_session = MagicMock()
            agent._flush_messages_to_session_db = MagicMock()
            agent._memory_manager = MagicMock()
            agent._memory_manager.build_system_prompt.return_value = ""
            agent.event_callback = MagicMock()
            with patch.object(db, "archive_and_compact", wraps=db.archive_and_compact) as archive, \
                 patch.object(db, "publish_compression_child", wraps=db.publish_compression_child) as publish, \
                 patch.object(db, "record_compression_failure_cooldown", wraps=db.record_compression_failure_cooldown) as cooldown:
                returned, _ = compress_context(
                    agent, messages, "sys", approx_tokens=100_000,
                    rejection_cooldown_seconds=rejection_cooldown_seconds,
                )
            assert returned is messages
            assert agent._last_compression_outcome == "rejected_no_progress"
            agent.commit_memory_session.assert_not_called()
            agent._flush_messages_to_session_db.assert_not_called()
            agent._memory_manager.on_session_switch.assert_not_called()
            agent.event_callback.assert_not_called()
            archive.assert_not_called()
            publish.assert_not_called()
            assert [(r["id"], r["active"]) for r in db.get_messages(
                sid, include_inactive=True
            )] == before
            if rejection_cooldown_seconds is None:
                cooldown.assert_not_called()
            else:
                cooldown.assert_called_once()
                assert db.get_compression_failure_cooldown(sid)["error"] == "no_progress"

    def test_uncached_admission_uses_built_input_prompt(self):
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "uncached_prompt"
            _seed(db, sid, "prompt")
            agent = _make_agent(db, sid, in_place=True)
            messages = _materially_compressible_messages()
            agent._cached_system_prompt = None
            agent._cached_system_prompt_static = None
            agent._build_system_prompt = MagicMock(return_value="EXACT BUILT PROMPT")
            seen = []
            estimates = iter((100_000, 80_000))

            def estimate(candidate, *, system_prompt, tools=None):
                seen.append((candidate, system_prompt, tools))
                return next(estimates)

            with patch(
                "agent.conversation_compression.estimate_request_tokens_rough",
                side_effect=estimate,
            ):
                compress_context(agent, messages, "sys", approx_tokens=100_000)
            assert len(seen) == 2
            assert seen[0][0] is not messages
            assert seen[0][0] == messages
            assert seen[0][1] == "EXACT BUILT PROMPT"
            assert seen[0][1] != ""
            assert seen[1][1] == "EXACT BUILT PROMPT"
            assert seen[0][2] is seen[1][2]

    @pytest.mark.parametrize(
        ("request_out", "outcome"),
        [(100_000, "rejected_no_progress"), (101_000, "rejected_would_grow"),
         (97_000, "rejected_below_minimum_reclaim")],
    )
    def test_in_place_mutating_engine_rolls_back_against_immutable_input(
        self, request_out, outcome, request
    ):
        """A plugin may mutate and return its exact input list (#remediation-7)."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            request.addfinalizer(db.close)
            sid = f"mutating-{outcome}"
            _seed(db, sid, "mutating")
            agent = _make_agent(db, sid, in_place=True)
            messages = _materially_compressible_messages()
            original = copy.deepcopy(messages)
            cached, static = "old prompt\x00", "old static\U0001f680"
            agent._cached_system_prompt = cached
            agent._cached_system_prompt_static = static
            agent.commit_memory_session = MagicMock()
            agent._flush_messages_to_session_db = MagicMock()
            agent._memory_manager = MagicMock()
            agent._memory_manager.build_system_prompt.return_value = (
                "deterministic external memory prompt"
            )
            agent.event_callback = MagicMock()

            def mutate(candidate, **_kwargs):
                candidate[:] = (
                    copy.deepcopy(original) if outcome == "rejected_no_progress"
                    else [{"role": "user", "content": "mutated candidate"}]
                )
                return candidate

            agent.context_compressor.compress = mutate
            estimates = []

            def estimate(candidate, **_kwargs):
                estimates.append(copy.deepcopy(candidate))
                return 100_000 if len(estimates) == 1 else request_out

            with patch(
                "agent.conversation_compression.estimate_request_tokens_rough",
                side_effect=estimate,
            ), patch.object(db, "archive_and_compact") as archive, patch.object(
                db, "publish_compression_child"
            ) as publish, patch.object(db, "end_session") as end, patch.object(
                db, "create_session"
            ) as create:
                returned, prompt = compress_context(
                    agent, messages, "sys", approx_tokens=100_000
                )

            if outcome == "rejected_no_progress":
                assert len(estimates) == 0
            else:
                assert len(estimates) == 2
                assert estimates[0] == original
                assert estimates[1] != original
                admitted_content = "\n".join(
                    str(message.get("content", "")) for message in estimates[1]
                )
                assert "mutated candidate" in admitted_content
                assert original[-1]["content"] in admitted_content
            assert returned is messages
            assert messages == original
            assert prompt == cached
            assert agent._cached_system_prompt == cached
            assert agent._cached_system_prompt_static == static
            assert agent._last_compression_outcome == outcome
            agent.commit_memory_session.assert_not_called()
            agent._flush_messages_to_session_db.assert_not_called()
            agent._memory_manager.on_session_switch.assert_not_called()
            agent.event_callback.assert_not_called()
            for side_effect in (archive, publish, end, create):
                side_effect.assert_not_called()

    def test_uncached_exact_noop_removes_prompt_cache_attributes(self):
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "uncached-noop"
            _seed(db, sid, "uncached")
            agent = _make_agent(db, sid, in_place=True)
            messages = _materially_compressible_messages()
            vars(agent).pop("_cached_system_prompt", None)
            vars(agent).pop("_cached_system_prompt_static", None)
            agent._build_system_prompt = MagicMock(return_value="logical built prompt")
            agent.context_compressor.compress = lambda current, **_kwargs: current
            agent.commit_memory_session = MagicMock()
            agent._flush_messages_to_session_db = MagicMock()
            agent.event_callback = MagicMock()

            with patch.object(db, "archive_and_compact") as archive, patch.object(
                db, "publish_compression_child"
            ) as publish:
                returned, prompt = compress_context(
                    agent, messages, "base prompt", approx_tokens=100_000
                )

            assert returned is messages
            assert prompt == "logical built prompt"
            assert "_cached_system_prompt" not in vars(agent)
            assert "_cached_system_prompt_static" not in vars(agent)
            assert agent._last_compression_outcome == "rejected_no_progress"
            agent.commit_memory_session.assert_not_called()
            agent._flush_messages_to_session_db.assert_not_called()
            agent.event_callback.assert_not_called()
            archive.assert_not_called()
            publish.assert_not_called()

    def test_rejection_restores_both_prompt_cache_tiers_byte_for_byte(self):
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "cached_prompt_restore"
            _seed(db, sid, "prompt")
            agent = _make_agent(db, sid, in_place=True)
            messages = _materially_compressible_messages()
            cached = "cached\x00prompt\U0001f642"
            static = "static\x00prefix\U0001f680"
            agent._cached_system_prompt = cached
            agent._cached_system_prompt_static = static
            estimates = iter((100_000, 100_000))
            with patch(
                "agent.conversation_compression.estimate_request_tokens_rough",
                side_effect=lambda candidate, **kwargs: next(estimates),
            ):
                returned, prompt = compress_context(
                    agent, messages, "sys", approx_tokens=100_000
                )
            assert returned is messages
            assert prompt == cached
            assert agent._cached_system_prompt == cached
            assert agent._cached_system_prompt_static == static

    def test_manual_rejection_does_not_write_automatic_cooldown(self):
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_manual_reject"
            _seed(db, sid, "manual")
            agent = _make_agent(db, sid, in_place=True)
            messages = _materially_compressible_messages()
            estimates = iter((100_000, 97_000))
            with patch(
                "agent.conversation_compression.estimate_request_tokens_rough",
                side_effect=lambda candidate, **kwargs: next(estimates),
            ):
                compress_context(
                    agent, messages, "sys", force=True, approx_tokens=100_000
                )
            assert agent._last_compression_outcome == "rejected_below_minimum_reclaim"
            assert db.get_compression_failure_cooldown(sid) is None

    def test_in_place_still_commits_shrinking_compression(self):
        """The guard must not block legitimate compressions — a result SMALLER
        than the input still commits in place (regression net for #83339)."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_shrink"
            _seed(db, sid, "shrink")
            agent = _make_agent(db, sid, in_place=True)
            agent._last_flushed_db_idx = 5

            messages = _materially_compressible_messages()
            compressed, _sp = compress_context(
                agent, messages, approx_tokens=100_000, system_message="sys"
            )

            # The fake compressor returns a small summary — commit happens.
            assert agent._last_compaction_in_place is True
            reloaded = db.get_messages_as_conversation(sid)
            assert [m.get("content") for m in reloaded] == [
                "[CONTEXT COMPACTION] summary of prior turns",
                "recent reply",
                messages[-1]["content"],
            ]


class TestCompactedTurnsStaySearchable:
    """Teknium's review hinges on the pre-compaction transcript staying
    DISCOVERABLE after in-place compaction. Compaction-archived rows
    (active=0, compacted=1) must surface in session_search by default, while
    rewind/undo rows (active=0, compacted=0) must stay hidden. The two share
    the active flag but are distinguished by the compacted flag."""

    def test_compacted_turns_found_by_default_search(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_search"
            db.create_session(sid, "cli", model="test/model")
            for r, c in [
                ("user", "configure the HMAC secret"),
                ("assistant", "set it in config.yaml"),
                ("user", "deploy returns 403"),
                ("assistant", "rotate the HMAC"),
                ("user", "works now"),
                ("assistant", "great"),
            ]:
                db.append_message(session_id=sid, role=r, content=c)

            before = db.search_messages("HMAC", role_filter=["user", "assistant"])
            assert len(before) == 2

            db.archive_and_compact(
                sid,
                [
                    {"role": "user", "content": "[SUMMARY] earlier setup"},
                    {"role": "assistant", "content": "ok"},
                ],
            )

            # The archived originals (active=0, compacted=1) are still found by
            # the DEFAULT search — this is the durability requirement.
            after = db.search_messages("HMAC", role_filter=["user", "assistant"])
            assert {m["id"] for m in after} == {1, 4}
            # Live context still excludes them.
            assert len(db.get_messages_as_conversation(sid)) == 2

    def test_rewound_turns_stay_hidden(self):
        """Rewind/undo (active=0, compacted=0) must NOT leak into default
        search — the distinction the compacted flag preserves."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_undo"
            db.create_session(sid, "cli", model="test/model")
            db.append_message(session_id=sid, role="user", content="ZEBRAWORD remember this")
            db.append_message(session_id=sid, role="assistant", content="noted")
            db.rewind_to_message(sid, db.get_messages(sid)[0]["id"])

            assert db.search_messages("ZEBRAWORD", role_filter=["user", "assistant"]) == []
            recovered = db.search_messages(
                "ZEBRAWORD", role_filter=["user", "assistant"], include_inactive=True
            )
            assert len(recovered) == 1
