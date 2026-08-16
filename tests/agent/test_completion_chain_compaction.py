"""End-to-end regression for completion-driven compaction thrash."""

import hashlib
import os
from unittest.mock import patch

from agent.context_compressor import ACTIVE_TASK_CONTRACT_PREFIX, ContextCompressor
from agent.conversation_compression import _is_real_user_message, compress_context
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
            {"role": "user", "content":
             f"[IMPORTANT: Background process p-{index} completed normally. Final output: {index}]"},
        ])
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
    return [m for m in messages if ContextCompressor.parse_active_task_contract(m)]


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
        )

    first, _ = compress_context(
        _agent(db, sid), original, "sys", approx_tokens=100_000,
    )
    resumed = db.get_messages_as_conversation(sid)

    assert not any(m.get("content") == TASK for m in first)
    assert not any(m.get("content") == TASK for m in resumed)
    assert len(_contracts(resumed)) == 1
    contract_message = _contracts(resumed)[0]
    parsed = ContextCompressor.parse_active_task_contract(contract_message)
    assert parsed == {
        "content": TASK,
        "sha256": hashlib.sha256(TASK.encode()).hexdigest(),
    }
    assert contract_message["content"].startswith(ACTIVE_TASK_CONTRACT_PREFIX)
    assert _is_real_user_message(contract_message) is False
    assert "remains active until a later real human user message overrides it" in contract_message["content"]

    # A genuine later human turn becomes the sole contract source. The old
    # contract is synthetic, so it cannot drag the protected tail back to TASK.
    resumed.extend(_completion_chain(NEW_TASK, count=30))
    second_agent = _agent(db, sid)
    second, _ = compress_context(
        second_agent, resumed, "sys", approx_tokens=100_000,
    )
    second_contracts = _contracts(second)
    assert len(second_contracts) == 1
    assert ContextCompressor.parse_active_task_contract(second_contracts[0])["content"] == NEW_TASK
    assert TASK not in "\n".join(str(m.get("content", "")) for m in second)
    roles = [m["role"] for m in second]
    assert all(left != right for left, right in zip(roles, roles[1:]))
    call_ids = [
        call["id"] for message in second for call in message.get("tool_calls", [])
    ]
    assert len(call_ids) == len(set(call_ids))
