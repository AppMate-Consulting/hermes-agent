"""Tests for /compress here [N] — boundary-aware partial compression.

Verifies the CLI handler (_manual_compress) splits the history, compresses
only the head, and re-appends the verbatim tail. Inspired by Claude Code's
Rewind "Summarize up to here" action (v2.1.139, May 2026).
"""

from unittest.mock import MagicMock, patch

from tests.cli.test_cli_init import _make_cli


def _make_history() -> list[dict[str, str]]:
    # 8 messages = 4 exchanges.
    h: list[dict[str, str]] = []
    for i in range(4):
        h.append({"role": "user", "content": f"u{i}"})
        h.append({"role": "assistant", "content": f"a{i}"})
    return h


def _wire_agent(shell, compressed_head):
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent.session_id = None
    shell.agent.tools = None
    shell.agent._compress_context.return_value = (compressed_head, "")
    shell.agent._compression_skipped_due_to_lock = False


def test_compress_here_transaction_receives_full_snapshot_and_protected_tail(capsys):
    """The transaction validates the suffix and gives only its head to the engine."""
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    # Pretend compression collapses the head into a single summary message.
    summary = [{"role": "user", "content": "[summary of earlier turns]"}]
    # _compress_context returns the already-published full candidate; the host
    # must adopt it without re-appending/reflushing the suffix.
    _wire_agent(shell, summary + history[4:])

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress("/compress here 2")

    # The atomic compression transaction receives the full authoritative
    # snapshot plus its exact protected suffix.  It, not the CLI host, is
    # responsible for passing only the head to the compressor and publishing
    # head+tail exactly once under the SessionDB generation CAS.
    shell.agent._compress_context.assert_called_once()
    call = shell.agent._compress_context.call_args
    assert call.args[0] == history
    assert call.kwargs["protected_tail"] == history[4:]
    # focus_topic must be None in partial mode (modes are exclusive).
    assert call.kwargs.get("focus_topic") is None


def test_compress_here_reappends_verbatim_tail(capsys):
    """The most recent exchanges are preserved verbatim after the summary."""
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    # Head compresses to an assistant-role summary so the seam
    # (assistant -> user tail) is already valid — tail rides along whole.
    summary = [{"role": "assistant", "content": "[summary]"}]
    _wire_agent(shell, summary + history[4:])

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress("/compress here 2")

    # Result = compressed head + verbatim tail (last 2 exchanges).
    assert shell.conversation_history == summary + history[4:]
    # Tail boundary keeps role alternation valid (tail starts on user).
    assert history[4]["role"] == "user"
    # No consecutive same-role user/assistant messages anywhere.
    roles = [m["role"] for m in shell.conversation_history
             if m["role"] in ("user", "assistant")]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


def test_compress_here_one_keeps_trusted_completion_sequence_whole():
    from agent.conversation_compression import (
        AUTONOMOUS_COMPLETION_BRIDGE_ASSISTANT,
        AUTONOMOUS_COMPLETION_BRIDGE_USER,
    )
    from hermes_cli.partial_compress import split_history_for_partial_compress

    history = _make_history() + [
        {
            "role": "user", "content": AUTONOMOUS_COMPLETION_BRIDGE_USER,
            "_autonomous_completion_bridge": True,
        },
        {
            "role": "assistant", "content": AUTONOMOUS_COMPLETION_BRIDGE_ASSISTANT,
            "_autonomous_completion_bridge": True,
        },
        {
            "role": "user", "content": "[ASYNC DELEGATION COMPLETE child=x]",
            "_autonomous_completion_bridge": True,
        },
        {"role": "assistant", "content": "consumed"},
    ]

    head, tail = split_history_for_partial_compress(history, 1)

    assert head == history[:-4]
    assert tail == history[-4:]


def test_compress_here_banner_mentions_summarizing_up_to_here(capsys):
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    _wire_agent(shell, [{"role": "user", "content": "[summary]"}])

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress("/compress here")

    out = capsys.readouterr().out
    assert "Summarizing up to here" in out
    assert "verbatim" in out


def test_bare_compress_still_full(capsys):
    """/compress with no args compresses the whole history (full mode)."""
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    _wire_agent(shell, list(history))

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress("/compress")

    call = shell.agent._compress_context.call_args
    # Full mode passes the entire history as the head.
    assert call.args[0] == history
    out = capsys.readouterr().out
    assert "Summarizing up to here" not in out


def test_focus_still_works(capsys):
    """/compress <focus> keeps the existing focus behavior."""
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    _wire_agent(shell, list(history))

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress("/compress database schema")

    call = shell.agent._compress_context.call_args
    assert call.args[0] == history
    assert call.kwargs.get("focus_topic") == "database schema"


def test_committed_postpublication_error_reconciles_cli_state(capsys):
    from agent.conversation_compression import (
        CompressionCommittedPostpublicationError,
    )

    shell = _make_cli()
    history = _make_history()
    authoritative = [{"role": "user", "content": "committed summary"}]
    _wire_agent(shell, authoritative)
    shell.conversation_history = history
    shell.session_id = "parent"
    shell.agent.session_id = "child"
    shell.agent._compress_context.side_effect = CompressionCommittedPostpublicationError(
        session_id="child",
        transcript=authoritative,
        in_place=False,
        cause=RuntimeError("host adoption failed"),
    )

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress("/compress")

    assert shell.session_id == "child"
    assert shell.agent.session_id == "child"
    assert shell.conversation_history == authoritative
    output = capsys.readouterr().out
    assert "Compression committed" in output
    assert "Compression failed" not in output


def test_committed_reload_failure_rebinds_and_blocks_cli(capsys):
    from agent.conversation_compression import CompressionCommittedPostpublicationError

    shell = _make_cli()
    shell.conversation_history = _make_history()
    shell.session_id = "parent"
    _wire_agent(shell, [])
    shell.agent.session_id = "parent"
    shell.agent._session_db = MagicMock()
    shell.agent._session_db.get_messages_as_conversation.side_effect = RuntimeError(
        "read still unavailable"
    )
    shell.agent._compress_context.side_effect = CompressionCommittedPostpublicationError(
        session_id="child",
        transcript=None,
        candidate_transcript=[{"role": "user", "content": "speculative"}],
        in_place=False,
        cause=RuntimeError("first read unavailable"),
    )

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress("/compress")

    assert shell.session_id == shell.agent.session_id == "child"
    assert shell.conversation_history == []
    shell.chat("must not run")
    shell.agent.run_conversation.assert_not_called()
    assert "Session blocked" in capsys.readouterr().out
