"""Regression coverage for Desktop loopback authentication boundaries."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth.ws_tickets import (
    _reset_for_tests,
    internal_ws_credential,
    mint_ticket,
)


class _QueryParams:
    def __init__(self, values):
        self._values = values

    def get(self, key, default=""):
        return self._values.get(key, default)


def _ws(query, *, client_host="127.0.0.1", headers=None):
    return SimpleNamespace(
        query_params=_QueryParams(query),
        client=SimpleNamespace(host=client_host),
        headers=headers or {},
        url=SimpleNamespace(path="/api/ws"),
    )


@pytest.fixture
def loopback_auth(monkeypatch):
    previous = {
        name: getattr(web_server.app.state, name, None)
        for name in ("bound_host", "bound_port", "auth_required")
    }
    monkeypatch.setenv("HERMES_DESKTOP_LOCAL_BACKEND", "1")
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 8080
    web_server.app.state.auth_required = False
    yield TestClient(web_server.app, base_url="http://127.0.0.1:8080")
    for name, value in previous.items():
        setattr(web_server.app.state, name, value)


def test_unauthenticated_desktop_loopback_http_is_rejected(loopback_auth):
    assert loopback_auth.get("/api/env").status_code == 401


def test_forwarded_header_lookalike_http_is_rejected(loopback_auth):
    response = loopback_auth.get(
        "/api/env",
        headers={
            "Forwarded": "for=127.0.0.1;host=localhost;proto=http",
            "X-Forwarded-For": "127.0.0.1",
            "X-Forwarded-Host": "localhost",
            "X-Forwarded-Proto": "http",
            "Origin": "http://localhost:8080",
        },
    )
    assert response.status_code == 401


def test_desktop_spawn_session_token_http_is_accepted(loopback_auth):
    response = loopback_auth.get(
        "/api/env",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
    )
    assert response.status_code == 200


def test_unauthenticated_desktop_loopback_ws_is_rejected(loopback_auth):
    assert web_server._ws_auth_reason(_ws({})) == ("no_credential", "none")


def test_forwarded_header_lookalike_ws_is_rejected(loopback_auth):
    socket = _ws(
        {},
        headers={
            "host": "127.0.0.1:8080",
            "origin": "http://localhost:8080",
            "forwarded": "for=127.0.0.1;host=localhost;proto=http",
            "x-forwarded-for": "127.0.0.1",
        },
    )
    assert web_server._ws_auth_ok(socket) is False


def test_desktop_spawn_session_token_ws_is_accepted(loopback_auth):
    socket = _ws({"token": web_server._SESSION_TOKEN})
    assert web_server._ws_auth_reason(socket) == (None, "token")


def test_gated_ws_ticket_and_internal_topologies_remain_valid(loopback_auth):
    _reset_for_tests()
    web_server.app.state.bound_host = "dashboard.example"
    web_server.app.state.auth_required = True
    try:
        ticket = mint_ticket(user_id="user", provider="test")
        assert web_server._ws_auth_reason(_ws({"ticket": ticket})) == (None, "ticket")
        assert web_server._ws_auth_reason(_ws({"ticket": ticket}))[0] == "ticket_invalid"

        internal = internal_ws_credential()
        assert web_server._ws_auth_reason(_ws({"internal": internal})) == (None, "internal")
        assert web_server._ws_auth_reason(_ws({"token": web_server._SESSION_TOKEN}))[0] == "no_credential"
    finally:
        _reset_for_tests()
