import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from xinference.api.routers import runtime_logs
from xinference.core import runtime_logs as log_reader


def _body(response):
    return json.loads(response.body)


def test_runtime_log_tail_incremental_read_and_multiple_rotations(
    tmp_path, monkeypatch
):
    path = tmp_path / "xinference.log"
    monkeypatch.setattr(log_reader, "get_log_file", lambda _: str(path))
    path.write_bytes(b"first\n")

    first = log_reader.read_runtime_log()
    assert first["text"] == "first\n"
    assert not first["has_more"]

    with path.open("ab") as stream:
        stream.write(b"second\n")
    second = log_reader.read_runtime_log(first["cursor"])
    assert second["text"] == "second\n"

    first_archive = tmp_path / "xinference.log.2026-09-27.1"
    path.rename(first_archive)
    with first_archive.open("ab") as stream:
        stream.write(b"last old line\n")
    path.write_bytes(b"intermediate line\n")

    second_archive = tmp_path / "xinference.log.2026-09-27.2"
    path.rename(second_archive)
    path.write_bytes(b"active line\n")

    old = log_reader.read_runtime_log(second["cursor"])
    assert old["text"] == "last old line\n"
    assert old["has_more"]
    intermediate = log_reader.read_runtime_log(old["cursor"])
    assert intermediate["text"] == "intermediate line\n"
    assert intermediate["has_more"]
    new = log_reader.read_runtime_log(intermediate["cursor"])
    assert new["text"] == "active line\n"
    assert not new["has_more"]


def test_runtime_log_read_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / "xinference.log"
    monkeypatch.setattr(log_reader, "get_log_file", lambda _: str(path))
    monkeypatch.setattr(log_reader, "MAX_RUNTIME_LOG_BYTES", 16)
    path.write_bytes(b"old\n" * 20 + b"latest\n")

    result = log_reader.read_runtime_log()
    assert result["text"].endswith("latest\n")
    assert len(result["text"].encode()) <= 16


@pytest.mark.asyncio
async def test_runtime_log_sources_include_remote_workers_with_local_node():
    api = MagicMock()
    api._supervisor_address = "127.0.0.1:9999"
    supervisor = AsyncMock()
    supervisor.get_status.return_value = {
        "workers": {
            "127.0.0.1:9999": {},
            "10.0.0.2:9999": {},
        }
    }
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)

    response = await runtime_logs.list_runtime_log_sources(api=api)
    assert _body(response) == {
        "sources": [
            {"id": "local", "label": "Local"},
            {"id": "10.0.0.2:9999", "label": "Worker 10.0.0.2:9999"},
        ]
    }

    with pytest.raises(HTTPException) as exc:
        await runtime_logs.read_runtime_logs(
            source="unregistered:1234", cursor="", api=api
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_runtime_log_reads_registered_worker(monkeypatch):
    api = MagicMock()
    api._supervisor_address = "127.0.0.1:9999"
    supervisor = AsyncMock()
    supervisor.get_status.return_value = {"workers": {"10.0.0.2:9999": {}}}
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    worker = AsyncMock()
    worker.read_runtime_logs.return_value = {
        "text": "worker error\n",
        "cursor": "1:2:13",
        "has_more": False,
        "reset": False,
    }
    actor_ref = AsyncMock(return_value=worker)
    monkeypatch.setattr(runtime_logs.xo, "actor_ref", actor_ref)

    response = await runtime_logs.read_runtime_logs(
        source="10.0.0.2:9999", cursor="1:2:0", api=api
    )
    assert _body(response)["text"] == "worker error\n"
    worker.read_runtime_logs.assert_awaited_once_with("1:2:0")
