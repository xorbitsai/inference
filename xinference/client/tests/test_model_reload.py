# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from ..restful.async_restful_client import AsyncClient
from ..restful.restful_client import Client


def test_reload_client_keeps_auth_and_encodes_uid():
    client = Client.__new__(Client)
    client.base_url = "http://localhost:9997"
    client._headers = {"Authorization": "Bearer test"}
    client.session = MagicMock()
    response = MagicMock(status_code=202)
    response.json.return_value = {"operation_id": "one"}
    client.session.post.return_value = response
    assert client.reload_model("model/name", {"max_num_seqs": 8}) == {
        "operation_id": "one"
    }
    client.session.post.assert_called_once_with(
        "http://localhost:9997/v1/models/model%2Fname/reload",
        json={"model_config": {"max_num_seqs": 8}, "drain_timeout": 300},
        headers=client._headers,
    )
    response.status_code = 200
    client.session.get.return_value = response
    client.get_model_reload_status("test")
    client.get_model_reload_config("test")
    assert client.session.get.call_args.args[0].endswith("/test/reload/config")


@pytest.mark.asyncio
async def test_async_reload_client_releases_responses():
    client = AsyncClient.__new__(AsyncClient)
    client.base_url = "http://localhost:9997"
    client._headers = {"Authorization": "Bearer test"}
    client.session = MagicMock()
    client.session.closed = True
    response = MagicMock(status=202)
    response.wait_for_close = AsyncMock()
    response.json = AsyncMock(return_value={"operation_id": "one"})
    client.session.post = AsyncMock(return_value=response)
    client.session.get = AsyncMock(return_value=response)
    assert await client.reload_model("test", {"max_num_seqs": 8}) == {
        "operation_id": "one"
    }
    assert response.release.called
    response.status = 200
    await client.get_model_reload_status("test")
    await client.get_model_reload_config("test")
    assert client.session.get.call_args.args[0].endswith("/test/reload/config")


def test_reload_cli_preserves_json_types(monkeypatch):
    from click.testing import CliRunner

    from ...deploy import cmdline

    client = MagicMock()
    client.reload_model.return_value = {"operation_id": "one", "status": "reloading"}
    monkeypatch.setattr(cmdline, "RESTfulClient", lambda **kwargs: client)
    result = CliRunner().invoke(
        cmdline.cli,
        [
            "reload",
            "--model-uid",
            "test",
            "--api-key",
            "key",
            "--model-config",
            '{"max_num_seqs": 32, "enforce_eager": false}',
        ],
    )
    assert result.exit_code == 0, result.output
    client.reload_model.assert_called_once_with(
        "test", {"max_num_seqs": 32, "enforce_eager": False}, 300.0
    )
    assert "one" in result.output
