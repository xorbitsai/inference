# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import json
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from ...deploy import cmdline
from ..restful.restful_client import Client


@pytest.mark.parametrize("component", ["api", "supervisor"])
def test_client_keeps_credentials_and_encodes_operation(component):
    client = Client.__new__(Client)
    client.base_url = "http://localhost:9997"
    client._headers = {"Authorization": "Bearer token"}
    client.session = MagicMock()
    response = MagicMock(status_code=202)
    response.json.return_value = {"operation_id": "one"}
    client.session.post.return_value = response
    assert getattr(client, f"relaunch_{component}")("/new/python", "request") == {
        "operation_id": "one"
    }
    client.session.post.assert_called_once_with(
        f"http://localhost:9997/v1/{component}/relaunch",
        json={"python_executable": "/new/python", "request_id": "request"},
        headers=client._headers,
    )
    response.status_code = 200
    client.session.get.return_value = response
    getattr(client, f"get_{component}_relaunch")("operation/id")
    assert client.session.get.call_args.args[0].endswith(
        f"/v1/{component}-relaunches/operation%2Fid"
    )
    getattr(client, f"list_{component}_relaunches")()
    assert client.session.get.call_args.args[0].endswith(f"/v1/{component}-relaunches")


@pytest.mark.parametrize("component", ["api", "supervisor"])
def test_cli_uses_the_existing_login_session(monkeypatch, component):
    client = MagicMock()
    getattr(client, f"relaunch_{component}").return_value = {"operation_id": "one"}
    monkeypatch.setattr(cmdline, "RESTfulClient", lambda **kwargs: client)
    monkeypatch.setattr(cmdline, "get_stored_token", lambda *args: "session-token")
    result = CliRunner().invoke(
        cmdline.cli,
        [
            f"relaunch-{component}",
            "--endpoint",
            "http://localhost:9997",
            "--python-executable",
            "/new/python",
            "--request-id",
            "request",
        ],
    )
    assert result.exit_code == 0, result.output
    client._set_token.assert_called_once_with("session-token")
    getattr(client, f"relaunch_{component}").assert_called_once_with(
        "/new/python", "request"
    )
    assert json.loads(result.output) == {"operation_id": "one"}
