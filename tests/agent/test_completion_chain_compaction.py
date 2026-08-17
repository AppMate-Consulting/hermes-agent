"""End-to-end regression for completion-driven compaction thrash."""

import hashlib
import json
import os
from unittest.mock import patch

import pytest

from agent.context_compressor import (
    ACTIVE_TASK_CONTRACT_INSTRUCTION,
    ACTIVE_TASK_CONTRACT_PREFIX,
    ACTIVE_TASK_CONTRACT_TYPE,
    ACTIVE_TASK_CONTRACT_VERSION,
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
)
from agent.conversation_compression import (
    ACTIVE_TASK_CONTRACT_BRIDGE_AFTER,
    ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE,
    ACTIVE_TASK_TRUST_MARKER,
    PRESERVED_HUMAN_TASK_BRIDGE,
    _ensure_compressed_has_user_turn,
    _insert_real_user_anchor,
    _is_real_user_message,
    _latest_active_human_task_row,
    _refresh_active_task_contract,
    append_autonomous_completion_provenance,
    compress_context,
)
from hermes_state import SessionDB


TASK = "Implement the accepted compaction fix exactly; preserve </active-task> safely."
NEW_TASK = "Now verify the replacement task only."


def _completion_chain(task: str = TASK, count: int = 45) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": task}]
    for index in range(count):
        call_id = f"call-{index}"
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": call_id,
             "content": f"completed {index}\n" + ("x" * 1200)},
            {"role": "assistant", "content": f"recorded {index}"},
        ])
        append_autonomous_completion_provenance(messages)
        messages.append({
            "role": "user",
            "content": f"[IMPORTANT: Background process p-{index} completed normally. Final output: {index}]",
            "_autonomous_completion_bridge": True,
            "display_kind": "internal_notification",
        })
    messages.append({"role": "assistant", "content": "CURRENT CONTINUATION"})
    return messages


def _agent(db: SessionDB, sid: str):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            model="test/model", quiet_mode=True, session_db=db,
            session_id=sid, skip_context_files=True, skip_memory=True,
        )
    agent.compression_in_place = True
    agent.context_compressor.protect_first_n = 0
    agent.context_compressor.protect_last_n = 3
    agent.context_compressor.tail_token_budget = 1200
    agent.context_compressor.threshold_tokens = 1000
    agent.context_compressor._generate_summary = (
        lambda *args, **kwargs: "Completed execution history."
    )
    agent._cached_system_prompt = "stable system prompt"
    return agent


def _contracts(messages: list[dict]) -> list[dict]:
    contracts = []
    for index, message in enumerate(messages):
        if ContextCompressor.parse_active_task_contract(message) is not None:
            contracts.append(message)
            continue
        if (
            index > 0
            and messages[index - 1].get("role") == "assistant"
            and messages[index - 1].get("content")
            == ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE
            and ContextCompressor.parse_active_task_contract(
                message, allow_projected=True
            ) is not None
        ):
            contracts.append(message)
    return contracts


def _contract_payload(messages: list[dict], message: dict) -> dict:
    index = next(i for i, candidate in enumerate(messages) if candidate is message)
    projected = (
        index > 0
        and messages[index - 1].get("role") == "assistant"
        and messages[index - 1].get("content")
        == ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE
    )
    return ContextCompressor.parse_active_task_contract(
        message, allow_projected=projected
    )


def test_contract_survives_db_resume_and_is_superseded_on_second_compaction(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "completion-chain"
    db.create_session(sid, source="gateway", model="test/model")
    original = _completion_chain()
    for message in original:
        db.append_message(
            sid, message["role"], message.get("content", ""),
            tool_calls=message.get("tool_calls"),
            tool_call_id=message.get("tool_call_id"),
            display_kind=message.get("display_kind"),
            autonomous_completion_provenance=(
                message.get("_autonomous_completion_bridge") is True
            ),
        )

    first, _ = compress_context(
        _agent(db, sid), original, "sys", approx_tokens=100_000,
    )
    resumed = db.get_messages_as_conversation(sid)
    rows_after_first = db.get_messages(sid, include_inactive=True)
    assert len([r for r in rows_after_first if not r.get("active", 1)]) == len(original)
    assert len([r for r in rows_after_first if r.get("active", 1)]) == len(first)

    assert not any(m.get("content") == TASK for m in first)
    assert not any(m.get("content") == TASK for m in resumed)
    assert len(_contracts(resumed)) == 1
    contract_message = _contracts(resumed)[0]
    parsed = _contract_payload(resumed, contract_message)
    assert parsed == {
        "content": TASK,
        "sha256": hashlib.sha256(TASK.encode()).hexdigest(),
    }
    assert contract_message["content"].startswith(ACTIVE_TASK_CONTRACT_PREFIX)
    assert _is_real_user_message(contract_message) is False
    assert "remains active until a later real human user message overrides it" in contract_message["content"]

    # A genuine later human turn becomes the sole contract source. The old
    # contract is synthetic, so it cannot drag the protected tail back to TASK.
    second_chain = _completion_chain(NEW_TASK, count=30)
    for message in second_chain:
        db.append_message(
            sid, message["role"], message.get("content", ""),
            tool_calls=message.get("tool_calls"),
            tool_call_id=message.get("tool_call_id"),
            display_kind=message.get("display_kind"),
            autonomous_completion_provenance=(
                message.get("_autonomous_completion_bridge") is True
            ),
        )
    resumed = db.get_messages_as_conversation(sid)
    archived_before_second = len([
        r for r in db.get_messages(sid, include_inactive=True)
        if not r.get("active", 1)
    ])
    active_before_second = len(resumed)
    second_agent = _agent(db, sid)
    second, _ = compress_context(
        second_agent, resumed, "sys", approx_tokens=100_000,
    )
    second_contracts = _contracts(second)
    assert len(second_contracts) == 1
    assert _contract_payload(second, second_contracts[0])["content"] == NEW_TASK
    assert TASK not in "\n".join(str(m.get("content", "")) for m in second)
    roles = [m["role"] for m in second]
    assert all(left != right for left, right in zip(roles, roles[1:]))
    call_ids = [
        call["id"] for message in second for call in message.get("tool_calls", [])
    ]
    assert len(call_ids) == len(set(call_ids))
    durable = db.get_messages_as_conversation(sid)
    assert len(_contracts(durable)) == 1
    assert _contract_payload(durable, _contracts(durable)[0])["content"] == NEW_TASK
    durable_roles = [m["role"] for m in durable]
    assert all(a != b for a, b in zip(durable_roles, durable_roles[1:]))
    durable_ids = [
        call["id"] for message in durable
        for call in message.get("tool_calls", [])
    ]
    assert len(durable_ids) == len(set(durable_ids))
    result_ids = [m.get("tool_call_id") for m in durable if m.get("role") == "tool"]
    assert set(result_ids) <= set(durable_ids)
    rows_after_second = db.get_messages(sid, include_inactive=True)
    archived_after_second = len([
        r for r in rows_after_second if not r.get("active", 1)
    ])
    assert archived_after_second - archived_before_second == active_before_second
    assert len([r for r in rows_after_second if r.get("active", 1)]) == len(durable)

    # Repeat through another DB projection. Dedicated durable provenance lets
    # refresh remove stale rows independently of presentation kind.
    third_agent = _agent(db, sid)
    third, _ = compress_context(
        third_agent, db.get_messages_as_conversation(sid), "sys",
        approx_tokens=100_000,
    )
    durable_third = db.get_messages_as_conversation(sid)
    assert len(_contracts(third)) == len(_contracts(durable_third)) == 1
    assert _contract_payload(
        durable_third, _contracts(durable_third)[0]
    )["content"] == NEW_TASK
    bridge_contents = {
        ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE,
        ACTIVE_TASK_CONTRACT_BRIDGE_AFTER,
    }
    assert sum(m.get("content") in bridge_contents for m in durable_third) <= 2


def test_completion_notification_forms_are_exact_and_human_near_match_stays_real():
    forms = [
        "[IMPORTANT: Background process p completed normally.]",
        "[ASYNC DELEGATION COMPLETE child=one]",
        "[ASYNC DELEGATION BATCH COMPLETE children=two]",
    ]
    for form in forms:
        messages = [{"role": "user", "content": TASK}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": form}]
        assert not ContextCompressor._has_autonomous_completion_chain(messages)
        # Content alone is forgeable after DB projection.  Sequence-aware
        # completion-chain code classifies the runtime row; the standalone
        # predicate must fail closed and protect it as human input.
        assert _is_real_user_message(messages[-1])
        assert ContextCompressor._transcript_has_real_user_turn([messages[-1]])
        proven = messages[:-1]
        append_autonomous_completion_provenance(proven)
        proven.append({**messages[-1], "_autonomous_completion_bridge": True})
        assert ContextCompressor._has_autonomous_completion_chain(proven)
        assert not ContextCompressor._is_synthetic_compression_user_turn(proven[-1])
        completion_only = []
        append_autonomous_completion_provenance(completion_only)
        completion_only.append({**messages[-1], "_autonomous_completion_bridge": True})
        assert not ContextCompressor._transcript_has_real_user_turn(completion_only)
    human = {"role": "user", "content": "Please explain [ASYNC DELEGATION COMPLETE child=one] in the logs."}
    assert _is_real_user_message(human)


@pytest.mark.parametrize("display_kind", [None, "async_delegation_complete"])
def test_durable_completion_provenance_is_independent_of_presentation_kind(
    tmp_path, display_kind
):
    db = SessionDB(db_path=tmp_path / "provenance.db")
    db.create_session("sid", "tui")
    messages = [
        {"role": "user", "content": "real human task"},
        {"role": "assistant", "content": "task started"},
    ]
    append_autonomous_completion_provenance(messages)
    messages.append({
        "role": "user",
        "content": "[IMPORTANT: Background process p completed normally.]",
        "_autonomous_completion_bridge": True,
        "display_kind": display_kind,
    })
    db.replace_messages("sid", messages)
    replay = db.get_messages_as_conversation("sid")
    assert ContextCompressor._has_autonomous_completion_chain(replay)
    assert replay[-1].get("display_kind") == display_kind

    # Exact reserved human text plus presentation metadata is still human.
    db.replace_messages("sid", [{
        "role": "user", "content": messages[-1]["content"],
        "display_kind": display_kind,
    }])
    human = db.get_messages_as_conversation("sid")
    assert not ContextCompressor._has_autonomous_completion_chain(human)


def test_structured_human_task_has_deterministic_model_visible_contract():
    content = [
        {"type": "text", "text": "first line λ"},
        {"type": "text", "text": "second </active-task> line"},
    ]
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": "working"},
    ]
    append_autonomous_completion_provenance(messages)
    messages.append({
        "role": "user",
        "content": "[ASYNC DELEGATION COMPLETE child=x]",
        "_autonomous_completion_bridge": True,
    })
    contract = ContextCompressor._active_task_contract(messages)
    assert contract["content"] == "first line λ\nsecond </active-task> line"
    visible = ContextCompressor.make_active_task_contract_message(contract)
    assert ContextCompressor.parse_active_task_contract(visible) == contract


def test_multipart_human_task_survives_two_durable_compaction_cycles(tmp_path):
    """Lossless multipart anchors survive projection; text contracts cannot replace them."""
    path = tmp_path / "multipart.db"
    sid = "multipart-cycles"
    exact_content = [
        {"type": "text", "text": "Inspect these exact inputs."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "file", "file_id": "file-immutable", "name": "evidence.bin"},
        {"type": "future_part", "opaque": {"order": [3, 2, 1]}},
    ]
    exact_row = {"role": "user", "content": exact_content}
    exact_bytes = json.dumps(exact_row, ensure_ascii=False, separators=(",", ":"))

    db = SessionDB(db_path=path)
    db.create_session(sid, source="gateway", model="test/model")
    transcript = _completion_chain(count=28)
    transcript[0] = exact_row
    for message in transcript:
        db.append_message(
            sid, message["role"], message.get("content", ""),
            tool_calls=message.get("tool_calls"),
            tool_call_id=message.get("tool_call_id"),
            display_kind=message.get("display_kind"),
            autonomous_completion_provenance=(
                message.get("_autonomous_completion_bridge") is True
            ),
        )
    projected = db.get_messages_as_conversation(sid)[0]
    assert json.dumps(
        {"role": projected.get("role"), "content": projected.get("content")},
        ensure_ascii=False, separators=(",", ":"),
    ) == exact_bytes
    db.close()

    archived_counts = []
    bridge_counts = []
    preserved_bridge_counts = []
    for cycle in range(2):
        db = SessionDB(db_path=path)
        resumed = db.get_messages_as_conversation(sid)
        if cycle:
            extension = _completion_chain("continue exact multipart task", count=20)[1:]
            for message in extension:
                db.append_message(
                    sid, message["role"], message.get("content", ""),
                    tool_calls=message.get("tool_calls"),
                    tool_call_id=message.get("tool_call_id"),
                    display_kind=message.get("display_kind"),
                    autonomous_completion_provenance=(
                        message.get("_autonomous_completion_bridge") is True
                    ),
                )
            resumed = db.get_messages_as_conversation(sid)
        cycle_agent = _agent(db, sid)
        protected_tail_bound = cycle_agent.context_compressor.protect_last_n
        # protect_last_n is a minimum floor, and provenance groups can expand
        # to several rows.  This deterministic-fixture ceiling admits those
        # groups while still catching the old whole-chain retention (56 rows).
        fixture_bridge_ceiling = max(16, protected_tail_bound * 4)
        compacted, _ = compress_context(
            cycle_agent, resumed, "sys", approx_tokens=100_000
        )
        durable = db.get_messages_as_conversation(sid)
        for persisted in (compacted, durable):
            exact = [m for m in persisted if json.dumps(
                {"role": m.get("role"), "content": m.get("content")},
                ensure_ascii=False, separators=(",", ":"),
            ) == exact_bytes]
            assert len(exact) == 1
        assert not _contracts(durable)
        for persisted in (compacted, durable):
            roles = [m["role"] for m in persisted]
            assert all(a != b for a, b in zip(roles, roles[1:]))
        calls = {
            call["id"] for m in durable for call in m.get("tool_calls", [])
        }
        results = {
            m.get("tool_call_id") for m in durable if m.get("role") == "tool"
        }
        assert all(
            m.get("tool_call_id") in calls
            for m in durable if m.get("role") == "tool"
        )
        assert calls <= results
        bridge_markers = [
            m for m in durable
            if (
                "HERMES_AUTONOMOUS_COMPLETION_BRIDGE"
                in str(m.get("content", ""))
                or m.get("content") == PRESERVED_HUMAN_TASK_BRIDGE
            )
        ]
        bridge_count = len(bridge_markers)
        assert bridge_count <= fixture_bridge_ceiling
        bridge_counts.append(bridge_count)
        preserved_bridge_counts.append(sum(
            m.get("content") == PRESERVED_HUMAN_TASK_BRIDGE for m in durable
        ))
        archived_counts.append(sum(
            not row.get("active", 1)
            for row in db.get_messages(sid, include_inactive=True)
        ))
        durable_fields = ("role", "content", "tool_calls", "tool_call_id", "name")

        def durable_shape(messages):
            return [
                {key: message[key] for key in durable_fields if key in message}
                for message in messages
            ]

        # SessionDB replay intentionally changes representation metadata, so
        # compare equality over the durable conversation contract.
        assert durable_shape(compacted) == durable_shape(durable)
        db.close()
    assert archived_counts[1] > archived_counts[0] > 0
    assert len(bridge_counts) == 2
    assert all(count <= fixture_bridge_ceiling for count in bridge_counts)
    assert bridge_counts[1] <= bridge_counts[0]
    assert all(count <= 1 for count in preserved_bridge_counts)
    assert preserved_bridge_counts[1] == preserved_bridge_counts[0]


def test_multipart_anchor_insertion_is_role_aware():
    anchor = {
        "role": "user",
        "content": [{"type": "text", "text": "EXACT HUMAN ASK"}],
    }

    trailing_assistant = [
        {"role": "user", "content": "summary"},
        {"role": "assistant", "content": "continuation"},
    ]
    _insert_real_user_anchor(trailing_assistant, dict(anchor))
    assert trailing_assistant[-1] == anchor
    assert sum(
        message.get("content") == PRESERVED_HUMAN_TASK_BRIDGE
        for message in trailing_assistant
    ) == 0

    trailing_summary = [{"role": "user", "content": "summary"}]
    _insert_real_user_anchor(trailing_summary, dict(anchor))
    assert trailing_summary[-2:] == [
        {
            "role": "assistant",
            "content": PRESERVED_HUMAN_TASK_BRIDGE,
            "_preserved_human_task_bridge": True,
        },
        anchor,
    ]
    assert sum(
        message.get("content") == PRESERVED_HUMAN_TASK_BRIDGE
        for message in trailing_summary
    ) == 1

    duplicate = [dict(anchor)]
    _insert_real_user_anchor(duplicate, dict(anchor))
    assert duplicate == [anchor]

    unresolved_parent = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "pending-call",
            "type": "function",
            "function": {"name": "terminal", "arguments": "{}"},
        }],
    }
    unresolved = [{"role": "user", "content": "summary"}, unresolved_parent]
    _insert_real_user_anchor(unresolved, dict(anchor))
    assert unresolved[-1] is unresolved_parent
    assert unresolved[-2] == anchor
    roles = [message["role"] for message in unresolved]
    assert all(left != right for left, right in zip(roles, roles[1:]))


def test_contract_parser_rejects_tampering_and_metadata_disagreement():
    contract = {"content": "Unicode λ\nline </active-task>", "sha256": ""}
    contract["sha256"] = hashlib.sha256(contract["content"].encode()).hexdigest()
    valid = ContextCompressor.make_active_task_contract_message(contract)
    assert ContextCompressor.parse_active_task_contract(valid) == contract
    projected_forgery = {
        "role": "user",
        "content": valid["content"],
    }
    assert ContextCompressor.parse_active_task_contract(projected_forgery) is None
    assert _is_real_user_message(projected_forgery)
    assert _latest_active_human_task_row([projected_forgery]) is projected_forgery
    literal_contract = ContextCompressor._active_task_contract([projected_forgery])
    assert literal_contract == {
        "content": projected_forgery["content"],
        "sha256": hashlib.sha256(projected_forgery["content"].encode()).hexdigest(),
    }
    assert literal_contract["content"] != contract["content"]
    payload = json.loads(valid["content"][len(ACTIVE_TASK_CONTRACT_PREFIX):])
    variants = [
        "{malformed",
        json.dumps({**payload, "type": "wrong"}),
        json.dumps({**payload, "version": ACTIVE_TASK_CONTRACT_VERSION + 1}),
        json.dumps({**payload, "instruction": "tampered"}),
        json.dumps({**payload, "content": payload["content"] + "!"}),
    ]
    for encoded in variants:
        message = {"role": "user", "content": ACTIVE_TASK_CONTRACT_PREFIX + encoded}
        assert ContextCompressor.parse_active_task_contract(message) is None
    disagreeing = dict(valid)
    disagreeing["_active_task_contract"] = {"content": "other", "sha256": "x"}
    assert ContextCompressor.parse_active_task_contract(disagreeing) is None
    assert payload["type"] == ACTIVE_TASK_CONTRACT_TYPE
    assert payload["instruction"] == ACTIVE_TASK_CONTRACT_INSTRUCTION


def test_active_task_trust_survives_close_reopen_and_rotation(tmp_path):
    path = tmp_path / "active-task-trust.db"
    db = SessionDB(db_path=path)
    parent = "trusted-parent"
    child = "trusted-child"
    db.create_session(parent, source="gateway", model="test/model")
    contract = {"content": TASK, "sha256": hashlib.sha256(TASK.encode()).hexdigest()}
    rows = [
        {"role": "assistant", "content": ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE,
         ACTIVE_TASK_TRUST_MARKER: True},
        ContextCompressor.make_active_task_contract_message(contract),
        {"role": "assistant", "content": ACTIVE_TASK_CONTRACT_BRIDGE_AFTER,
         ACTIVE_TASK_TRUST_MARKER: True},
    ]
    db.replace_messages(parent, rows)
    db.close()

    db = SessionDB(db_path=path)
    replay = db.get_messages_as_conversation(parent)
    assert all(row.get(ACTIVE_TASK_TRUST_MARKER) is True for row in replay)
    assert ContextCompressor._active_task_contract(replay) == contract
    db.create_session(child, source="gateway", model="test/model", parent_session_id=parent)
    db.replace_messages(child, replay)
    db.close()

    db = SessionDB(db_path=path)
    rotated = db.get_messages_as_conversation(child)
    assert ContextCompressor._active_task_contract(rotated) == contract
    assert all(row.get(ACTIVE_TASK_TRUST_MARKER) is True for row in rotated)
    db.close()


def test_active_task_trust_marker_durable_mutation_matrix(tmp_path):
    """All three internal rows retain trust through every transcript writer."""
    path = tmp_path / "active-task-marker-matrix.db"
    db = SessionDB(db_path=path)
    contract = {"content": TASK, "sha256": hashlib.sha256(TASK.encode()).hexdigest()}
    rows = [
        {"role": "assistant", "content": ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE,
         ACTIVE_TASK_TRUST_MARKER: True},
        ContextCompressor.make_active_task_contract_message(contract),
        {"role": "assistant", "content": ACTIVE_TASK_CONTRACT_BRIDGE_AFTER,
         ACTIVE_TASK_TRUST_MARKER: True},
    ]
    untrusted_replay = [
        {key: value for key, value in row.items()
         if key not in {
             ACTIVE_TASK_TRUST_MARKER,
             "_active_task_contract",
             "_active_task_contract_bridge",
         }}
        for row in rows
    ]
    assert ContextCompressor._active_task_contract(untrusted_replay) is None
    assert _latest_active_human_task_row(untrusted_replay) == untrusted_replay[1]
    durable_projection = [
        {key: value for key, value in row.items()
         if key not in {"_active_task_contract_bridge", "_active_task_contract"}}
        for row in rows
    ]
    for missing_index in range(3):
        incomplete = [dict(row) for row in durable_projection]
        incomplete[missing_index].pop(ACTIVE_TASK_TRUST_MARKER)
        assert ContextCompressor._active_task_contract(incomplete) is None

    def assert_trusted(session_id):
        replay = db.get_messages_as_conversation(session_id)
        assert [m.get(ACTIVE_TASK_TRUST_MARKER) for m in replay] == [True] * 3
        assert ContextCompressor._active_task_contract(replay) == contract
        return replay

    # Single-row append and the production batch writer use distinct bind paths.
    db.create_session("single", source="test", model="test/model")
    for row in rows:
        db.append_message(
            "single", row["role"], row["content"],
            active_task_contract_provenance=True,
        )
    assert_trusted("single")

    db.create_session("batch", source="test", model="test/model")
    db.append_messages_batch("batch", [dict(row) for row in rows])
    replay = assert_trusted("batch")

    db.replace_messages("batch", replay)
    replay = assert_trusted("batch")
    generation = db.get_active_transcript_identity("batch")
    db.archive_and_compact(
        "batch", replay, expected_active_identity=generation,
    )
    replay = assert_trusted("batch")

    # Rotation is a child publication through the same durable batch shape.
    db.create_session(
        "child", source="test", model="test/model", parent_session_id="batch",
    )
    db.append_messages_batch("child", replay)
    assert_trusted("child")
    db.close()

    db = SessionDB(db_path=path)
    assert_trusted("single")
    assert_trusted("batch")
    assert_trusted("child")
    db.close()


def test_active_task_provenance_update_advances_generation_and_fences_publish(tmp_path):
    db = SessionDB(db_path=tmp_path / "active-task-generation.db")
    db.create_session("s", source="test", model="test/model")
    row_id = db.append_message("s", "user", "ordinary")
    stale_generation = db.get_active_transcript_identity("s")

    # Simulate a legacy/concurrent writer changing only the trust column.
    with db._conn:
        db._conn.execute(
            "UPDATE messages SET active_task_contract_provenance = 1 WHERE id = ?",
            (row_id,),
        )
    assert db.get_active_transcript_identity("s") == stale_generation + 1
    with pytest.raises(RuntimeError, match="transcript changed"):
        db.archive_and_compact(
            "s", [{"role": "user", "content": "stale"}],
            expected_active_identity=stale_generation,
        )
    assert db.get_messages_as_conversation("s")[0]["content"] == "ordinary"
    db.close()


def test_pre_v27_database_reconciles_active_task_provenance_without_data_loss(tmp_path):
    """Exercise normal initialization against an actual v26-shaped SQLite file."""
    import sqlite3
    from hermes_state_common import SCHEMA_SQL

    path = tmp_path / "pre-v27.db"
    pre_v27_sql = SCHEMA_SQL.replace(
        ",\n    active_task_contract_provenance INTEGER NOT NULL DEFAULT 0\n", "\n"
    )
    raw = sqlite3.connect(path)
    raw.executescript(pre_v27_sql)
    raw.execute("INSERT INTO schema_version(version) VALUES (26)")
    raw.execute(
        "INSERT INTO sessions(id, source, model, started_at) "
        "VALUES ('legacy', 'test', 'test/model', 1)"
    )
    raw.execute(
        "INSERT INTO messages(session_id, role, content, timestamp) "
        "VALUES ('legacy', 'user', 'preserved legacy message', 1)"
    )
    raw.commit()
    raw.close()

    db = SessionDB(db_path=path)
    columns = {
        row["name"] for row in db._conn.execute("PRAGMA table_info(messages)")
    }
    assert "active_task_contract_provenance" in columns
    stored = db._conn.execute(
        "SELECT content, active_task_contract_provenance FROM messages "
        "WHERE session_id = 'legacy'"
    ).fetchone()
    assert tuple(stored) == ("preserved legacy message", 0)
    assert db.get_messages_as_conversation("legacy")[0]["content"] == "preserved legacy message"
    db.close()


def test_contract_refresh_requires_summary_and_removes_stale_bridges():
    original = [
        {"role": "user", "content": TASK},
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "[ASYNC DELEGATION COMPLETE child=x]"},
    ]
    compressed = [{"role": "assistant", "content": "tail"}]
    _refresh_active_task_contract(original, compressed)
    _ensure_compressed_has_user_turn(original, compressed)
    assert not _contracts(compressed)
    assert any(
        _is_real_user_message(m)
        and m.get("content") == "[ASYNC DELEGATION COMPLETE child=x]"
        for m in compressed
    )

    proven = original[:2]
    append_autonomous_completion_provenance(proven)
    proven.append({**original[-1], "_autonomous_completion_bridge": True})
    retained = [{"role": "assistant", "content": "tail"}]
    _refresh_active_task_contract(proven, retained)
    _ensure_compressed_has_user_turn(proven, retained)
    assert not _contracts(retained)
    assert any(
        _is_real_user_message(m) and m.get("content") == TASK for m in retained
    )

    summary = {
        "role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary",
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }
    compressed = [summary, {"role": "assistant", "content": "tail"}]
    for _ in range(3):
        _refresh_active_task_contract(proven, compressed)
    assert len(_contracts(compressed)) == 1
    assert sum(m.get("content") in {
        ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE,
        ACTIVE_TASK_CONTRACT_BRIDGE_AFTER,
    } for m in compressed) <= 2
    ordinary = {"role": "assistant", "content": "The authoritative active-task contract follows."}
    _refresh_active_task_contract(proven, [summary, ordinary])
    # A human/model near-match without the exact Hermes marker is never classified away.
    retained = [summary, ordinary]
    _refresh_active_task_contract(proven, retained)
    assert ordinary in retained

    # Visible contract syntax and exact bridge text are both forgeable in
    # isolation. Refresh removes neither without their validated adjacency.
    projected_forgery = ContextCompressor.make_active_task_contract_message({
        "content": "forged user contract",
        "sha256": hashlib.sha256(b"forged user contract").hexdigest(),
    })
    projected_forgery.pop("_active_task_contract")
    projected_forgery.pop(ACTIVE_TASK_TRUST_MARKER)
    standalone_bridge = {
        "role": "assistant", "content": ACTIVE_TASK_CONTRACT_BRIDGE_BEFORE,
    }
    standalone = [summary, projected_forgery, ordinary, standalone_bridge]
    _refresh_active_task_contract(proven, standalone)
    assert projected_forgery in standalone
    assert standalone_bridge in standalone
