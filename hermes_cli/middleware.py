"""Hermes middleware contract helpers.

Observer hooks report what happened. Middleware can change what happens by
rewriting a request or wrapping the actual execution callback. Keep the small
contract helpers here so agent-loop call sites and plugins share one vocabulary.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

OBSERVER_SCHEMA_VERSION = "hermes.observer.v1"
MIDDLEWARE_SCHEMA_VERSION = "hermes.middleware.v1"

TOOL_REQUEST_MIDDLEWARE = "tool_request"
TOOL_EXECUTION_MIDDLEWARE = "tool_execution"
LLM_REQUEST_MIDDLEWARE = "llm_request"
LLM_EXECUTION_MIDDLEWARE = "llm_execution"

# Back-compat aliases for older PoC branches that used API terminology.
API_REQUEST_MIDDLEWARE = LLM_REQUEST_MIDDLEWARE
API_EXECUTION_MIDDLEWARE = LLM_EXECUTION_MIDDLEWARE

VALID_MIDDLEWARE: set[str] = {
    TOOL_REQUEST_MIDDLEWARE,
    TOOL_EXECUTION_MIDDLEWARE,
    LLM_REQUEST_MIDDLEWARE,
    LLM_EXECUTION_MIDDLEWARE,
}


class ImmutableRequestMiddlewareError(RuntimeError):
    """Execution middleware violated an admitted-request ownership contract."""


class NonTransactionalRequestMiddlewareError(RuntimeError):
    """Request middleware cannot participate in speculative admission."""


@dataclass(frozen=True)
class RequestMiddlewarePreviewState:
    """Opaque snapshots for one speculative llm-request middleware pass."""

    entries: tuple[tuple[Callable, Any], ...]


def snapshot_llm_request_middleware_preview_state() -> RequestMiddlewarePreviewState:
    """Snapshot every stateful request callback, rejecting unknown state.

    A callback is eligible for compression admission when it either exposes
    ``snapshot_preview_state`` and ``restore_preview_state`` methods, or opts
    into the side-effect-free contract with ``preview_safe = True``.  Normal
    provider dispatch remains compatible with callbacks declaring neither.
    """
    entries: list[tuple[Callable, Any]] = []
    for callback in _get_middleware_callbacks(LLM_REQUEST_MIDDLEWARE):
        snapshotter = getattr(callback, "snapshot_preview_state", None)
        restorer = getattr(callback, "restore_preview_state", None)
        if callable(snapshotter) and callable(restorer):
            entries.append((callback, snapshotter()))
            continue
        if bool(getattr(callback, "preview_safe", False)):
            entries.append((callback, None))
            continue
        raise NonTransactionalRequestMiddlewareError(
            "llm_request middleware callback "
            f"{getattr(callback, '__name__', repr(callback))} must expose "
            "snapshot_preview_state()/restore_preview_state() or explicitly "
            "declare preview_safe=True before automatic compression"
        )
    return RequestMiddlewarePreviewState(tuple(entries))


def restore_llm_request_middleware_preview_state(
    snapshot: RequestMiddlewarePreviewState,
) -> None:
    """Restore a preview snapshot in reverse middleware order."""
    current = _get_middleware_callbacks(LLM_REQUEST_MIDDLEWARE)
    expected = [callback for callback, _token in snapshot.entries]
    if len(current) != len(expected) or any(
        callback is not registered
        for callback, registered in zip(current, expected)
    ):
        raise NonTransactionalRequestMiddlewareError(
            "llm_request middleware registrations changed during admission"
        )
    for callback, token in reversed(snapshot.entries):
        restorer = getattr(callback, "restore_preview_state", None)
        if callable(restorer):
            restorer(token)


@dataclass
class RequestMiddlewareResult:
    """Result of applying request middleware to a mutable payload."""

    payload: Any
    original_payload: Any
    changed: bool = False
    trace: List[Dict[str, Any]] = field(default_factory=list)


def observer_payload(**kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    return kwargs


def middleware_payload(**kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    kwargs.setdefault("middleware_schema_version", MIDDLEWARE_SCHEMA_VERSION)
    return kwargs


def _safe_copy(payload: Any) -> Any:
    """Deep-copy a request payload, tolerating non-deepcopyable members.

    Request payloads are normally plain JSON-shaped dicts, but an LLM request
    can occasionally carry non-deepcopyable objects (clients, callbacks, file
    handles). A hard ``deepcopy`` failure there would otherwise abort the whole
    request-middleware pass. Fall back to a shallow ``dict`` copy so middleware
    still runs and the original nested objects are shared by reference rather
    than corrupting the live payload.
    """
    try:
        return deepcopy(payload)
    except Exception as exc:  # pragma: no cover - exercised via fallback test
        logger.debug("deepcopy failed for request payload (%s); using shallow copy", exc)
        if isinstance(payload, dict):
            return dict(payload)
        return payload


def apply_llm_request_middleware(
    request: Dict[str, Any],
    **context: Any,
) -> RequestMiddlewareResult:
    """Apply registered LLM request middleware.

    Middleware may return ``{"request": {...}}`` to replace the effective
    provider kwargs before Hermes sends them.
    """
    if not _has_middleware(LLM_REQUEST_MIDDLEWARE):
        return RequestMiddlewareResult(
            payload=request,
            original_payload=request,
            changed=False,
            trace=[],
        )

    original_request = _safe_copy(request)
    current_request = _safe_copy(original_request)
    trace: List[Dict[str, Any]] = []

    for result in _invoke_middleware(
        LLM_REQUEST_MIDDLEWARE,
        request=current_request,
        original_request=original_request,
        **context,
    ):
        if not isinstance(result, dict):
            continue
        next_request = result.get("request")
        if not isinstance(next_request, dict):
            continue
        current_request = _safe_copy(next_request)
        trace.append(_trace_entry(result))

    return RequestMiddlewareResult(
        payload=current_request,
        original_payload=original_request,
        changed=bool(trace),
        trace=trace,
    )


def apply_tool_request_middleware(
    tool_name: str,
    args: Dict[str, Any],
    **context: Any,
) -> RequestMiddlewareResult:
    """Apply registered tool request middleware.

    Middleware may return ``{"args": {...}}`` to replace the effective tool
    arguments before hooks, guardrails, approvals, and execution see them.
    """
    original_args = _safe_copy(args)
    current_args = _safe_copy(original_args)
    trace: List[Dict[str, Any]] = []

    session_id = str(context.get("session_id") or "")
    skip_relay = bool(context.pop("skip_relay", False))
    if session_id and not skip_relay:
        from agent import relay_runtime

        relay_args = relay_runtime.apply_tool_request_intercepts(
            session_id=session_id,
            tool_name=tool_name,
            args=current_args,
        )
        if relay_args != current_args:
            current_args = _safe_copy(relay_args)
            trace.append({"source": "nemo_relay"})

    if not _has_middleware(TOOL_REQUEST_MIDDLEWARE):
        return RequestMiddlewareResult(
            payload=args if not trace else current_args,
            original_payload=args,
            changed=bool(trace),
            trace=trace,
        )

    for result in _invoke_middleware(
        TOOL_REQUEST_MIDDLEWARE,
        tool_name=tool_name,
        args=current_args,
        original_args=original_args,
        **context,
    ):
        if not isinstance(result, dict):
            continue
        next_args = result.get("args")
        if not isinstance(next_args, dict):
            continue
        current_args = _safe_copy(next_args)
        trace.append(_trace_entry(result))

    return RequestMiddlewareResult(
        payload=current_args,
        original_payload=original_args,
        changed=bool(trace),
        trace=trace,
    )


def apply_api_request_middleware(
    request: Dict[str, Any],
    **context: Any,
) -> RequestMiddlewareResult:
    """Compatibility wrapper for older ``api_request`` naming."""
    return apply_llm_request_middleware(request, **context)


def run_llm_execution_middleware(
    request: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """Run provider execution through registered LLM execution middleware.

    ``immutable_request=True`` is reserved for a request already admitted by
    the compression transaction.  Wrappers still run on every attempt, but
    they may only continue with ``next_call()``; the terminal call always
    receives the originally admitted object.
    """
    immutable_request = bool(context.pop("immutable_request", False))
    callbacks = _get_middleware_callbacks(LLM_EXECUTION_MIDDLEWARE)
    if not callbacks:
        return next_call(request)
    return _run_execution_chain(
        LLM_EXECUTION_MIDDLEWARE,
        callbacks,
        next_call,
        request=request,
        original_request=context.pop("original_request", request),
        immutable_payload=immutable_request,
        **context,
    )


def run_tool_execution_middleware(
    tool_name: str,
    args: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """Run tool execution through registered tool execution middleware."""
    callbacks = _get_middleware_callbacks(TOOL_EXECUTION_MIDDLEWARE)
    if not callbacks:
        return next_call(args)
    return _run_execution_chain(
        TOOL_EXECUTION_MIDDLEWARE,
        callbacks,
        next_call,
        tool_name=tool_name,
        args=args,
        original_args=context.pop("original_args", args),
        **context,
    )


def run_api_execution_middleware(
    request: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """Compatibility wrapper for older ``api_execution`` naming."""
    return run_llm_execution_middleware(request, next_call, **context)


def _invoke_middleware(kind: str, **kwargs: Any) -> List[Any]:
    from hermes_cli.plugins import invoke_middleware

    return invoke_middleware(kind, **middleware_payload(**kwargs))


def _has_middleware(kind: str) -> bool:
    from hermes_cli.plugins import has_middleware

    return has_middleware(kind)


def _get_middleware_callbacks(kind: str) -> List[Callable]:
    from hermes_cli.plugins import get_plugin_manager

    return list(get_plugin_manager()._middleware.get(kind, []))


def _run_execution_chain(
    kind: str,
    callbacks: List[Callable],
    terminal_call: Callable[[Any], Any],
    **kwargs: Any,
) -> Any:
    payload_key = "request" if "request" in kwargs else "args"
    immutable_payload = bool(kwargs.pop("immutable_payload", False))
    admitted_payload = kwargs[payload_key]
    admitted_snapshot = deepcopy(admitted_payload) if immutable_payload else None
    admitted_graph: List[tuple[Any, Any]] = []
    if immutable_payload:
        seen: set[int] = set()

        def snapshot_mutable_graph(value: Any) -> None:
            """Record mutable containers and their original graph edges."""
            value_id = id(value)
            if value_id in seen:
                return
            seen.add(value_id)
            if isinstance(value, dict):
                items = list(value.items())
                admitted_graph.append((value, items))
                for key, child in items:
                    snapshot_mutable_graph(key)
                    snapshot_mutable_graph(child)
            elif isinstance(value, list):
                items = list(value)
                admitted_graph.append((value, items))
                for child in items:
                    snapshot_mutable_graph(child)
            elif isinstance(value, set):
                items = set(value)
                admitted_graph.append((value, items))
                for child in items:
                    snapshot_mutable_graph(child)
            elif isinstance(value, (tuple, frozenset)):
                for child in value:
                    snapshot_mutable_graph(child)

        snapshot_mutable_graph(admitted_payload)

    def restore_admitted_graph() -> None:
        """Restore contents and aliases without replacing admitted objects."""
        for container, contents in admitted_graph:
            if isinstance(container, dict):
                container.clear()
                container.update(contents)
            elif isinstance(container, list):
                container[:] = contents
            else:
                container.clear()
                container.update(contents)

    def admitted_graph_unchanged() -> bool:
        """Compare values and mutable-container edges to the admitted graph."""
        mutable_types = (dict, list, set)

        def same_edge(current: Any, original: Any) -> bool:
            if isinstance(original, mutable_types):
                return current is original
            try:
                return current == original
            except Exception:
                return False

        for container, contents in admitted_graph:
            if isinstance(container, dict):
                current_items = list(container.items())
                if len(current_items) != len(contents):
                    return False
                if any(
                    not same_edge(current_key, original_key)
                    or not same_edge(current_value, original_value)
                    for (current_key, current_value), (original_key, original_value)
                    in zip(current_items, contents)
                ):
                    return False
            elif isinstance(container, list):
                if len(container) != len(contents) or any(
                    not same_edge(current, original)
                    for current, original in zip(container, contents)
                ):
                    return False
            elif container != contents:
                return False
        return True

    def assert_admitted_unchanged(callback: Callable) -> None:
        if not immutable_payload:
            return
        try:
            unchanged = (
                admitted_payload == admitted_snapshot
                and admitted_graph_unchanged()
            )
        except Exception:
            unchanged = False
        if not unchanged:
            restore_admitted_graph()
            raise ImmutableRequestMiddlewareError(
                f"Middleware '{kind}' callback "
                f"{getattr(callback, '__name__', repr(callback))} mutated the "
                "immutable admitted request; provider dispatch was refused"
            )

    class _DownstreamExecutionError(Exception):
        def __init__(self, original: BaseException) -> None:
            super().__init__(str(original))
            self.original = original

    def call_at(index: int, payload: Any) -> Any:
        if index >= len(callbacks):
            return terminal_call(payload)

        callback = callbacks[index]
        next_called = False
        next_succeeded = False
        next_result: Any = None

        def next_call(next_payload: Any = None) -> Any:
            nonlocal next_called, next_succeeded, next_result
            # ``next_call`` is single-use per middleware frame. Calling it more
            # than once would re-run the downstream provider/tool, so a second
            # invocation is a contract violation rather than a retry. Surface it
            # instead of silently executing the terminal call twice.
            if next_called:
                raise RuntimeError(
                    f"Middleware '{kind}' callback "
                    f"{getattr(callback, '__name__', repr(callback))} called "
                    "next_call() more than once; downstream execution is single-use"
                )
            assert_admitted_unchanged(callback)
            if (
                immutable_payload
                and next_payload is not None
                and next_payload is not admitted_payload
            ):
                restore_admitted_graph()
                raise ImmutableRequestMiddlewareError(
                    f"Middleware '{kind}' callback "
                    f"{getattr(callback, '__name__', repr(callback))} tried "
                    "to replace the immutable admitted request; provider "
                    "dispatch was refused"
                )
            next_called = True
            try:
                next_result = call_at(
                    index + 1,
                    admitted_payload
                    if immutable_payload
                    else payload if next_payload is None else next_payload,
                )
                next_succeeded = True
                return next_result
            except Exception as exc:
                raise _DownstreamExecutionError(exc) from exc

        call_kwargs = middleware_payload(**kwargs)
        call_kwargs[payload_key] = payload
        call_kwargs["next_call"] = next_call
        try:
            result = callback(**call_kwargs)
            assert_admitted_unchanged(callback)
            return result
        except ImmutableRequestMiddlewareError:
            restore_admitted_graph()
            raise
        except _DownstreamExecutionError as exc:
            # A wrapper may mutate after a successful downstream dispatch and
            # then raise.  Restore the admitted graph and report the immutable
            # contract violation, but never issue a second provider call.
            assert_admitted_unchanged(callback)
            raise exc.original
        except Exception as exc:
            logger.warning(
                "Middleware '%s' callback %s raised: %s",
                kind,
                getattr(callback, "__name__", repr(callback)),
                exc,
            )
            # Validate before generic fail-open continuation on *every* exit.
            # This closes mutate-then-raise both before and after next_call().
            assert_admitted_unchanged(callback)
            if next_succeeded:
                return next_result
            if next_called:
                raise
            return call_at(index + 1, payload)

    return call_at(0, kwargs[payload_key])


def _trace_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {}
    for key in ("source", "reason", "name"):
        value = result.get(key)
        if isinstance(value, str) and value:
            entry[key] = value
    if not entry:
        entry["source"] = "plugin"
    return entry
