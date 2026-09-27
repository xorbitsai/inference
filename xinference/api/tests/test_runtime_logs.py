import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from xinference.api.routers import runtime_logs
from xinference.core import runtime_logs as log_reader


def _body(response):
    return json.loads(response.body)


def test_runtime_log_tail_incremental_read_and_rotation(tmp_path, monkeypatch):
    path = tmp_path / "xinference.log"
    monkeypatch.setattr(log_reader, "get_log_file", lambda _: str(path))
    path.write_text("first\n", encoding="utf-8")

    first = log_reader.read_runtime_log()
    assert first["text"] == "first\n"
    assert not first["has_more"]

    with path.open("a", encoding="utf-8") as stream:
        stream.write("second\n")
    second = log_reader.read_runtime_log(first["cursor"])
    assert second["text"] == "second\n"

    archive = tmp_path / "xinference.log.2026-09-27"
    path.rename(archive)
    with archive.open("a", encoding="utf-8") as stream:
        stream.write("last old line\n")
    path.write_text("first new line\n", encoding="utf-8")

    old = log_reader.read_runtime_log(second["cursor"])
    assert old["text"] == "last old line\n"
    assert old["has_more"]
    new = log_reader.read_runtime_log(old["cursor"])
    assert new["text"] == "first new line\n"


def test_runtime_log_read_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / "xinference.log"
    monkeypatch.setattr(log_reader, "get_log_file", lambda _: str(path))
    monkeypatch.setattr(log_reader, "MAX_RUNTIME_LOG_BYTES", 16)
    path.write_text("old\n" * 20 + "latest\n", encoding="utf-8")

    result = log_reader.read_runtime_log()
    assert result["text"].endswith("latest\n")
    assert len(result["text"].encode()) <= 16


@pytest.mark.asyncio
async def test_runtime_log_sources_and_unregistered_worker():
    api = MagicMock()
    api._supervisor_address = "127.0.0.1:9999"
    supervisor = AsyncMock()
    supervisor.get_status.return_value = {"workers": {"127.0.0.1:9999": {}}}
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)

    response = await runtime_logs.list_runtime_log_sources(api=api)
    assert _body(response) == {"sources": [{"id": "local", "label": "Local"}]}

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
