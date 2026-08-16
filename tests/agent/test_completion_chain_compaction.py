"""Regression for completion-driven compaction thrash (Buzz 20260816)."""

import hashlib

from agent.context_compressor import (
    ACTIVE_TASK_CONTRACT_METADATA_KEY,
    ContextCompressor,
    estimate_messages_tokens_rough,
)


TASK = "Implement the accepted compaction fix exactly; preserve every stated invariant."


def _completion_chain(count: int = 100) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": TASK}]
    for index in range(count):
        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"completed work unit {index}\n" + ("x" * 800),
                },
                {"role": "assistant", "content": f"work unit {index} recorded"},
                {
                    "role": "user",
                    "content": (
                        f"[IMPORTANT: Background process proc-{index} completed "
                        f"normally. Final output: unit {index}]"
                    ),
                },
            ]
        )
    messages.append({"role": "assistant", "content": "CURRENT CONTINUATION VERBATIM"})
    return messages


def _compressor() -> ContextCompressor:
    compressor = ContextCompressor(
        model="test/model",
        quiet_mode=True,
        protect_first_n=0,
        protect_last_n=3,
        config_context_length=32_768,
    )
    compressor.threshold_tokens = 1_000
    compressor.tail_token_budget = 1_200
    compressor._generate_summary = lambda *args, **kwargs: "Completed execution history."
    return compressor


def test_completion_chain_preserves_contract_but_compresses_descendants():
    messages = _completion_chain()
    before = estimate_messages_tokens_rough(messages)
    latest_event = messages[-2].copy()
    latest_continuation = messages[-1].copy()

    compressed = _compressor().compress(messages)

    assert estimate_messages_tokens_rough(compressed) < before
    assert len(compressed) < len(messages) // 4
    assert latest_event in compressed
    assert latest_continuation in compressed
    assert not any(message.get("content") == TASK for message in compressed)

    markers = [
        message for message in compressed
        if message.get(ACTIVE_TASK_CONTRACT_METADATA_KEY)
    ]
    assert len(markers) == 1
    contract = markers[0][ACTIVE_TASK_CONTRACT_METADATA_KEY]
    assert contract == {
        "content": TASK,
        "sha256": hashlib.sha256(TASK.encode()).hexdigest(),
    }
    assert TASK in markers[0]["content"]


def test_repeated_completion_chain_compaction_refreshes_contract_marker():
    compressor = _compressor()
    first = compressor.compress(_completion_chain())
    # Reconstruct as persistence/resume does: underscore metadata is not
    # required because the durable visible marker is self-authenticating.
    resumed = [
        {key: value for key, value in message.items() if key != ACTIVE_TASK_CONTRACT_METADATA_KEY}
        for message in first
    ]
    resumed.extend(_completion_chain(30)[1:])

    second = compressor.compress(resumed)
    rendered = "\n".join(str(message.get("content", "")) for message in second)

    assert rendered.count("## Active Human Task Contract") == 1
    assert rendered.count(TASK) == 1
    assert second[-1]["content"] == "CURRENT CONTINUATION VERBATIM"
