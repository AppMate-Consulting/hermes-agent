"""Tests for CLI manual compression messaging."""

from unittest.mock import MagicMock, patch

import pytest

from tests.cli.test_cli_init import _make_cli


def _make_history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def test_manual_compress_keeps_tui_composer_editable(capsys):
    """A follow-up can be drafted and queued while /compress runs."""
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent.tools = None
    shell.agent.session_id = shell.session_id

    observed = {}

    def compress(*_args, **_kwargs):
        # The classic TUI's TextArea consults this state for its read_only
        # condition. Compression must retain its status spinner without
        # preventing the user from drafting the next prompt.
        observed["running"] = shell._command_running
        observed["blocks_input"] = getattr(shell, "_command_blocks_input", shell._command_running)
        return list(history), ""

    shell.agent._compress_context.side_effect = compress

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress()

    assert observed == {"running": True, "blocks_input": False}


@pytest.mark.parametrize(
    "outcome",
    [
        "summary_failure",
        "rejected_no_progress",
        "rejected_would_grow",
        "rejected_below_minimum_reclaim",
    ],
)
def test_manual_compress_rejection_preserves_history_identity_and_notification(outcome):
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent.tools = None
    shell.agent.session_id = shell.session_id
    shell.agent._compression_skipped_due_to_lock = None
    shell.agent._last_compression_outcome = outcome
    shell.agent._compress_context.return_value = (list(history), "")

    with (
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
        patch(
            "agent.conversation_compression.finalize_context_engine_compression_notification"
        ) as finalize,
    ):
        shell._manual_compress()

    assert shell.conversation_history is history
    assert shell.conversation_history == _make_history()
    finalize.assert_called_once_with(shell.agent, committed=False)






def test_manual_compress_explains_when_token_estimate_rises(capsys):
    shell = _make_cli()
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "Dense summary that still counts as more tokens."},
        history[-1],
    ]
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent.tools = None
    shell.agent.session_id = shell.session_id  # no-op: no split
    shell.agent._compress_context.return_value = (compressed, "")
    shell.agent._compression_skipped_due_to_lock = False

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 120
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate):
        shell._manual_compress()

    output = capsys.readouterr().out
    assert "✅ Compressed: 4 → 3 messages" in output
    assert "Approx request size: ~100 → ~120 tokens" in output
    assert "denser summaries" in output


def test_manual_compress_syncs_session_id_after_split():
    """Regression for cli.session_id desync after /compress.

    _compress_context ends the parent session and creates a new child session,
    mutating agent.session_id. Without syncing, cli.session_id still points
    at the ended parent — causing /status, /resume, exit summary, and the
    next end_session() call (e.g. from /resume <id>) to target the wrong row.
    """
    shell = _make_cli()
    history = _make_history()
    old_id = shell.session_id
    new_child_id = "20260101_000000_child1"

    compressed = [
        {"role": "user", "content": "[summary]"},
        history[-1],
    ]
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent.tools = None
    # Simulate _compress_context mutating agent.session_id as a side effect.
    def _fake_compress(*args, **kwargs):
        shell.agent.session_id = new_child_id
        return (compressed, "")
    shell.agent._compress_context.side_effect = _fake_compress
    shell.agent._compression_skipped_due_to_lock = False
    shell.agent.session_id = old_id  # starts in sync
    shell._pending_title = "stale title"

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress()

    # CLI session_id must now point at the continuation child, not the parent.
    assert shell.session_id == new_child_id
    assert shell.session_id != old_id
    # Pending title must be cleared — titles belong to the parent lineage and
    # get regenerated for the continuation.
    assert shell._pending_title is None


def test_manual_compress_does_not_republish_transactional_child_history():
    """The compression transaction, not the CLI host, publishes the child."""
    shell = _make_cli()
    history = _make_history()
    old_id = shell.session_id
    new_child_id = "20260101_000000_child1"
    compressed = [
        {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] compacted"},
        history[-1],
    ]
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent.session_id = old_id

    def _fake_compress(*args, **kwargs):
        shell.agent.session_id = new_child_id
        return (compressed, "")

    shell.agent._compress_context.side_effect = _fake_compress
    shell.agent._compression_skipped_due_to_lock = False

    with patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=100):
        shell._manual_compress()

    assert shell.session_id == new_child_id
    assert shell.conversation_history == compressed
    shell.agent._flush_messages_to_session_db.assert_not_called()




def test_manual_compress_runs_when_auto_compaction_disabled(capsys):
    """compression.enabled: false disables *automatic* compaction only.

    Manual /compress must still work: the context-overflow error path
    (agent/conversation_loop.py) explicitly directs users to /compress when
    auto-compaction is off, and the gateway's /compress handler has never
    gated on the flag. Regression for the CLI refusing with "Compression is
    disabled in config."
    """
    shell = _make_cli()
    history = _make_history()
    compressed = [
        {"role": "user", "content": "[summary]"},
        history[-1],
    ]
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = False
    shell.agent._cached_system_prompt = ""
    shell.agent.tools = None
    shell.agent.session_id = shell.session_id
    shell.agent._compress_context.return_value = (compressed, "")
    # Explicit non-lock-skip: MagicMock getattr would return a truthy mock.
    shell.agent._compression_skipped_due_to_lock = False

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress()

    output = capsys.readouterr().out
    assert "Compression is disabled" not in output
    shell.agent._compress_context.assert_called_once()
    # Manual compression bypasses the summary-failure cooldown.
    assert shell.agent._compress_context.call_args.kwargs.get("force") is True
    assert shell.conversation_history == compressed




def test_manual_compress_shows_lock_skip_without_confirmed_holder(capsys):
    """When _compress_context skips due to the compression lock WITHOUT a
    confirmed holder (signal=True — acquisition failed but
    get_compression_lock_holder returned nothing, e.g. a SQLite error made
    try_acquire return False), _manual_compress must print the unconfirmed
    lock-skip wording, NOT claim another compression is definitely running,
    and NOT show the misleading "No changes from compression" no-op text."""
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent.tools = None
    shell.agent.session_id = shell.session_id

    # Simulate _compress_context setting the lock-skip signal and
    # returning unchanged messages.
    def _fake_compress(*args, **kwargs):
        shell.agent._compression_skipped_due_to_lock = True
        return (list(history), "")

    shell.agent._compress_context.side_effect = _fake_compress

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress()

    output = capsys.readouterr().out
    assert "Compression skipped" in output
    assert "could not acquire" in output
    # No confirmed holder → must not assert one is running.
    assert "already in progress" not in output
    assert "No changes from compression" not in output
    # Signal should be cleared after use.
    assert shell.agent._compression_skipped_due_to_lock is None


def test_manual_compress_shows_lock_in_progress_with_holder(capsys):
    """When the lock holder is a descriptive string, include it in the
    status message so the user knows which process to investigate."""
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent.tools = None
    shell.agent.session_id = shell.session_id

    def _fake_compress(*args, **kwargs):
        shell.agent._compression_skipped_due_to_lock = "pid=12345"
        return (list(history), "")

    shell.agent._compress_context.side_effect = _fake_compress

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress()

    output = capsys.readouterr().out
    assert "Compression already in progress" in output
    assert "pid=12345" in output
    assert "No changes from compression" not in output
