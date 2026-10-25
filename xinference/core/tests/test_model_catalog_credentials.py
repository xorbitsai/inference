# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
import threading
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import requests
from xoscar.serialization import deserialize, serialize

from xinference import extensions
from xinference.core.exceptions import (
    ModelCatalogUnavailableError,
    ModelCatalogUpdateError,
)
from xinference.core.supervisor import SupervisorActor
from xinference.core.worker import WorkerActor
from xinference.extensions import ModelCatalogCredentials

TOKEN = "test-catalog-secret"


@pytest.fixture(autouse=True)
def reset_extensions(monkeypatch):
    monkeypatch.delenv("XINFERENCE_EXTENSIONS", raising=False)
    monkeypatch.setattr(extensions, "_loaded", None)


def providers(monkeypatch, *hooks):
    monkeypatch.setattr(
        extensions,
        "get_extensions",
        lambda: tuple(
            SimpleNamespace(name="catalog", model_catalog_credentials=hook)
            for hook in hooks
        ),
    )


def test_credentials_are_opaque_and_actor_serializable():
    credentials = ModelCatalogCredentials(TOKEN)
    assert TOKEN not in str({"credentials": credentials})
    header, buffers = serialize(credentials)
    restored = deserialize(header, buffers)
    assert restored == credentials
    assert restored.token == TOKEN
    assert TOKEN not in repr(restored)


@pytest.mark.parametrize(
    "token", ["", " padded", "padded ", "a\r\nb", "a\x00b", "a\x7fb", "中文", None]
)
def test_invalid_headers_are_rejected_without_echoing(token):
    with pytest.raises(ValueError, match="^Invalid model catalog credential$"):
        ModelCatalogCredentials(token)


def test_optional_provider_and_declining_providers(monkeypatch):
    request = object()
    assert extensions.resolve_model_catalog_credentials(request, "llm") is None
    monkeypatch.setattr(extensions, "get_extensions", lambda: (SimpleNamespace(),))
    assert extensions.resolve_model_catalog_credentials(request, "llm") is None
    hook = Mock(return_value=ModelCatalogCredentials(TOKEN))
    providers(monkeypatch, lambda *_: None, hook, lambda *_: None)
    assert extensions.resolve_model_catalog_credentials(request, "llm").token == TOKEN
    hook.assert_called_once_with(request, "llm")


@pytest.mark.parametrize("result", [TOKEN, {"token": TOKEN}, False])
def test_provider_must_return_the_opaque_type(monkeypatch, result):
    providers(monkeypatch, lambda *_: result)
    with pytest.raises(TypeError, match="must return ModelCatalogCredentials or None"):
        extensions.resolve_model_catalog_credentials(object(), "llm")


def test_conflicting_providers_are_rejected(monkeypatch):
    providers(monkeypatch, *[lambda *_: ModelCatalogCredentials(TOKEN)] * 2)
    with pytest.raises(ValueError, match="Multiple model catalog credential providers"):
        extensions.resolve_model_catalog_credentials(object(), "llm")


def test_provider_failure_never_falls_back(monkeypatch):
    following = Mock()

    def denied(*_):
        raise PermissionError("denied")

    providers(monkeypatch, denied, following)
    with pytest.raises(PermissionError, match="denied"):
        extensions.resolve_model_catalog_credentials(object(), "llm")
    following.assert_not_called()


def test_async_provider_is_rejected_and_closed(monkeypatch):
    async def resolve():
        return ModelCatalogCredentials(TOKEN)

    result = resolve()
    providers(monkeypatch, lambda *_: result)
    with pytest.raises(TypeError, match="synchronous"):
        extensions.resolve_model_catalog_credentials(object(), "llm")
    assert result.cr_frame is None


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr("xinference.model.llm.register_builtin_model", Mock())
    return SimpleNamespace(
        _custom_register_type_to_cls={"LLM": None},
        _store_complete_model_configurations=AsyncMock(),
    )


def response():
    return SimpleNamespace(
        status_code=200, raise_for_status=Mock(), json=lambda: {"models": []}
    )


@pytest.mark.asyncio
async def test_worker_uses_only_request_credentials_then_returns_to_anonymous(
    worker, monkeypatch
):
    download = Mock(return_value=response())
    monkeypatch.setattr(requests, "get", download)
    await WorkerActor.update_model_type(
        worker, "llm", credentials=ModelCatalogCredentials(TOKEN)
    )
    await WorkerActor.update_model_type(worker, "llm")
    first, second = download.call_args_list
    assert first.args == (
        "https://model.xinference.io/api/models/download?model_type=llm",
    )
    assert first.kwargs == {
        "timeout": 30,
        "allow_redirects": False,
        "headers": {"X-Models-Token": TOKEN},
    }
    assert second.kwargs == {"timeout": 30}
    assert worker._store_complete_model_configurations.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_worker_calls_do_not_share_credentials(worker, monkeypatch):
    barrier = threading.Barrier(2, timeout=10)
    observed = []
    environment = dict(os.environ)

    def download(url, **kwargs):
        barrier.wait()
        observed.append(kwargs["headers"]["X-Models-Token"])
        return response()

    monkeypatch.setattr(requests, "get", download)
    await asyncio.gather(
        *(
            WorkerActor.update_model_type(
                worker, "llm", credentials=ModelCatalogCredentials(token)
            )
            for token in ("user-a-token", "user-b-token")
        )
    )
    assert sorted(observed) == ["user-a-token", "user-b-token"]
    assert dict(os.environ) == environment
    assert not hasattr(worker, "credentials")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["401", "403", "500", "503", "network", "json", "redirect", "store"]
)
async def test_worker_failure_does_not_store_retry_or_expose_token(
    worker, monkeypatch, caplog, failure
):
    caplog.set_level(logging.DEBUG, logger="xinference.core.worker")
    result = response()
    download = Mock(return_value=result)
    error = requests.HTTPError(TOKEN)
    if failure == "network":
        download.side_effect = requests.ConnectionError(TOKEN)
    elif failure == "json":
        result.json = Mock(side_effect=ValueError(TOKEN))
    elif failure == "store":
        worker._store_complete_model_configurations.side_effect = RuntimeError(TOKEN)
    elif failure == "redirect":
        result.status_code = 302
    else:
        result.status_code = int(failure)
        result.raise_for_status.side_effect = error
    monkeypatch.setattr(requests, "get", download)
    with pytest.raises(
        ModelCatalogUpdateError, match="^Failed to update model configurations$"
    ) as caught:
        await WorkerActor.update_model_type(
            worker, "llm", credentials=ModelCatalogCredentials(TOKEN)
        )
    download.assert_called_once()
    if failure == "store":
        worker._store_complete_model_configurations.assert_awaited_once()
    else:
        worker._store_complete_model_configurations.assert_not_awaited()
    stage = {"json": "decode", "store": "store"}.get(failure, "download")
    status = None if failure == "network" else result.status_code
    error_type = {
        "network": "ConnectionError",
        "json": "ValueError",
        "redirect": "ValueError",
        "store": "RuntimeError",
    }.get(failure, "HTTPError")
    assert f"stage={stage}, error={error_type}, status={status}" in caplog.text
    assert TOKEN not in caplog.text
    assert TOKEN not in "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("credentialed", [False, True])
async def test_supervisor_fanout_preserves_anonymous_call_shape(credentialed):
    supervisor = SupervisorActor()
    first, second = AsyncMock(), AsyncMock()
    supervisor._worker_address_to_worker = {
        "first": SimpleNamespace(update_model_type=first),
        "second": SimpleNamespace(update_model_type=second),
    }
    kwargs = {"credentials": ModelCatalogCredentials(TOKEN)} if credentialed else {}
    await supervisor.update_model_type("llm", **kwargs)
    first.assert_awaited_once_with("llm", **kwargs)
    second.assert_awaited_once_with("llm", **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("workers", [0, 1, 2])
@pytest.mark.parametrize("credentialed", [False, True])
async def test_supervisor_reports_empty_or_partial_failure(
    workers, credentialed, caplog
):
    caplog.set_level(logging.DEBUG, logger="xinference.core.supervisor")
    supervisor = SupervisorActor()
    failed = AsyncMock(side_effect=ConnectionError(TOKEN))
    succeeded = AsyncMock()
    supervisor._worker_address_to_worker = {
        f"worker-{i}:1234": SimpleNamespace(update_model_type=method)
        for i, method in enumerate([failed, succeeded][:workers])
    }
    error_type = ModelCatalogUpdateError if workers else ModelCatalogUnavailableError
    message = (
        "One or more workers failed to update the model catalog"
        if workers
        else "No workers available to update the model catalog"
    )
    kwargs = {"credentials": ModelCatalogCredentials(TOKEN)} if credentialed else {}
    with pytest.raises(error_type, match=f"^{message}$") as caught:
        await supervisor.update_model_type("llm", **kwargs)
    if workers:
        failed.assert_awaited_once_with("llm", **kwargs)
        assert "worker-0:1234 (ConnectionError)" in caplog.text
    else:
        assert "No workers available" in caplog.text
    if workers == 2:
        succeeded.assert_awaited_once_with("llm", **kwargs)
        assert "worker-1:1234" not in caplog.text
    assert TOKEN not in caplog.text
    assert TOKEN not in "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )


@pytest.mark.asyncio
async def test_supervisor_waits_for_all_workers_even_if_call_raises_synchronously():
    supervisor = SupervisorActor()
    succeeded = AsyncMock()
    supervisor._worker_address_to_worker = {
        "healthy": SimpleNamespace(update_model_type=succeeded),
        "unavailable": SimpleNamespace(
            update_model_type=Mock(side_effect=ConnectionError(TOKEN))
        ),
    }
    with pytest.raises(ModelCatalogUpdateError):
        await supervisor.update_model_type("llm")
    succeeded.assert_awaited_once_with("llm")


@pytest.mark.asyncio
@pytest.mark.parametrize("credentialed", [False, True])
async def test_reload_keeps_exception_type_without_echoing_details(
    worker, monkeypatch, caplog, credentialed
):
    caplog.set_level(logging.DEBUG, logger="xinference.core.worker")
    monkeypatch.setattr(requests, "get", Mock(return_value=response()))
    monkeypatch.setattr(
        "xinference.model.llm.register_builtin_model",
        Mock(side_effect=RuntimeError(TOKEN)),
    )
    credentials = ModelCatalogCredentials(TOKEN) if credentialed else None
    await WorkerActor.update_model_type(worker, "llm", credentials=credentials)
    worker._store_complete_model_configurations.assert_awaited_once()
    assert "Error reloading built-in models (RuntimeError)" in caplog.text
    assert TOKEN not in caplog.text
