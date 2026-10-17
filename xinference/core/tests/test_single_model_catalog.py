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

"""Single-model refresh preserves other entries and request-local credentials."""

import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from xinference.core import model_catalog
from xinference.core.exceptions import ModelCatalogUpdateError
from xinference.core.model_catalog import store_model_catalog
from xinference.core.supervisor import SupervisorActor
from xinference.core.worker import WorkerActor
from xinference.extensions import ModelCatalogCredentials

TOKEN = "single-catalog-test-secret"


def test_single_model_replaces_all_its_variants_and_preserves_other_models(tmp_path):
    path = tmp_path / "llm_models.json"
    unrelated = {
        "model_name": "other",
        "updated_at": 7,
        "model_specs": [{"extra": [1, 2]}],
    }
    store_model_catalog(
        path,
        [
            unrelated,
            {"model_name": "wanted", "engine": "old-a"},
            {"model_name": "wanted", "engine": "old-b"},
        ],
    )
    variants = [
        {"model_name": "wanted", "engine": "new-a"},
        {"model_name": "wanted", "engine": "new-b"},
    ]
    store_model_catalog(path, variants, model_name="wanted")
    assert json.loads(path.read_text()) == [unrelated, *variants]
    store_model_catalog(
        path, {"model_name": "new", "readme": "原始说明"}, model_name="new"
    )
    assert json.loads(path.read_text()) == [
        unrelated,
        *variants,
        {"model_name": "new", "readme": "原始说明"},
    ]
    store_model_catalog(path, [])
    assert json.loads(path.read_text()) == []


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        "invalid",
        {},
        {"model_name": "other"},
        [{"model_name": "wanted"}, {"model_name": "other"}],
    ],
)
def test_invalid_single_response_never_changes_saved_catalog(tmp_path, data):
    path = tmp_path / "models.json"
    path.write_bytes(b'[{"model_name":"keep"}]')
    before = path.read_bytes()
    with pytest.raises(ValueError, match="single-model"):
        store_model_catalog(path, data, model_name="wanted")
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "raw", [b"broken JSON", b"{}", b'["not a model"]', b'[{"invalid":true}]']
)
def test_bad_existing_catalog_is_not_replaced_by_partial_data(tmp_path, raw):
    path = tmp_path / "models.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        store_model_catalog(path, {"model_name": "wanted"}, model_name="wanted")
    assert path.read_bytes() == raw


def test_concurrent_partial_updates_preserve_every_model(tmp_path):
    path = tmp_path / "models.json"

    def write(index):
        name = f"model-{index}"
        store_model_catalog(path, {"model_name": name, "index": index}, model_name=name)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(16)))
    assert sorted(json.loads(path.read_text()), key=lambda row: row["index"]) == [
        {"model_name": f"model-{i}", "index": i} for i in range(16)
    ]


@pytest.mark.parametrize("stage", ["dump", "replace"])
def test_failed_publication_preserves_previous_file_and_removes_temporary(
    tmp_path, monkeypatch, stage
):
    path = tmp_path / "models.json"
    before = b'[{"model_name":"keep"}]'
    path.write_bytes(before)

    def fail(*args, **kwargs):
        if stage == "dump":
            args[1].write("partial")
        raise OSError("disk failure")

    monkeypatch.setattr(
        model_catalog.json if stage == "dump" else model_catalog.os,
        "dump" if stage == "dump" else "replace",
        fail,
    )
    with pytest.raises(OSError):
        store_model_catalog(path, {"model_name": "wanted"}, model_name="wanted")
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".catalog-*"))


def test_readers_and_partial_writes_see_complete_full_refresh(tmp_path, monkeypatch):
    path = tmp_path / "models.json"
    store_model_catalog(path, [{"model_name": "old"}])
    writing, release, partial_started = Event(), Event(), Event()
    original = model_catalog.json.dump
    full = [{"model_name": "fresh"}]

    def dump(data, stream, **kwargs):
        if data == full:
            stream.write(" ")
            writing.set()
            assert release.wait(10)
        return original(data, stream, **kwargs)

    def write_partial():
        partial_started.set()
        store_model_catalog(path, {"model_name": "wanted"}, model_name="wanted")

    monkeypatch.setattr(model_catalog.json, "dump", dump)
    with ThreadPoolExecutor(max_workers=2) as pool:
        replace = pool.submit(store_model_catalog, path, full)
        try:
            assert writing.wait(10)
            assert json.loads(path.read_text()) == [{"model_name": "old"}]
            merge = pool.submit(write_partial)
            assert partial_started.wait(10)
        finally:
            release.set()
        replace.result()
        merge.result()
    assert json.loads(path.read_text()) == [*full, {"model_name": "wanted"}]


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.setattr("xinference.constants.XINFERENCE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("xinference.model.llm.register_builtin_model", Mock())
    worker = SimpleNamespace(_custom_register_type_to_cls={"LLM": None})
    worker._store_complete_model_configurations = partial(
        WorkerActor._store_complete_model_configurations, worker
    )
    return worker, tmp_path / "v2" / "builtin" / "llm" / "llm_models.json"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_type", ["llm", "embedding", "rerank", "image", "audio", "video", "world"]
)
@pytest.mark.parametrize("credentialed", [False, True])
async def test_worker_encodes_name_and_stores_only_requested_model(
    worker, monkeypatch, credentialed, model_type
):
    worker, path = worker
    worker._custom_register_type_to_cls[model_type] = None
    monkeypatch.setattr(f"xinference.model.{model_type}.register_builtin_model", Mock())
    path = path.parents[1] / model_type / f"{model_type}_models.json"
    name = "org/model & variant?x=1"
    store_model_catalog(path, [{"model_name": "unrelated"}])
    response = SimpleNamespace(
        status_code=200,
        raise_for_status=Mock(),
        json=lambda: {"model_name": name, "updated_at": 42},
    )
    download = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", download)
    kwargs = {"credentials": ModelCatalogCredentials(TOKEN)} if credentialed else {}
    await WorkerActor.update_model_type(worker, model_type, model_name=name, **kwargs)
    assert parse_qs(urlsplit(download.call_args.args[0]).query) == {
        "model_type": [model_type],
        "model_name": [name],
        "include_readme": ["true"],
    }
    assert download.call_args.kwargs == (
        {"timeout": 30, "allow_redirects": False, "headers": {"X-Models-Token": TOKEN}}
        if credentialed
        else {"timeout": 30}
    )
    assert json.loads(path.read_text()) == [
        {"model_name": "unrelated"},
        {"model_name": name, "updated_at": 42},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["401", "503", "redirect", "network", "json", "mismatch"]
)
async def test_single_refresh_failures_do_not_retry_write_or_expose_credentials(
    worker, monkeypatch, caplog, failure
):
    worker, path = worker
    store_model_catalog(path, [{"model_name": "keep"}])
    before = path.read_bytes()
    result = SimpleNamespace(
        status_code=200, raise_for_status=Mock(), json=lambda: {"model_name": TOKEN}
    )
    download = Mock(return_value=result)
    if failure == "network":
        download.side_effect = requests.ConnectionError(TOKEN)
    elif failure == "json":
        result.json = Mock(side_effect=ValueError(TOKEN))
    elif failure == "redirect":
        result.status_code = 302
    elif failure != "mismatch":
        result.status_code = int(failure)
        result.raise_for_status.side_effect = requests.HTTPError(TOKEN)
    monkeypatch.setattr(requests, "get", download)
    with pytest.raises(ModelCatalogUpdateError) as caught:
        await WorkerActor.update_model_type(
            worker,
            "llm",
            model_name="wanted",
            credentials=ModelCatalogCredentials(TOKEN),
        )
    assert download.call_count == 1
    assert path.read_bytes() == before
    assert TOKEN not in str(caught.value) + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("credentialed", [False, True])
async def test_supervisor_forwards_model_name_and_reports_partial_failures(
    monkeypatch, credentialed
):
    supervisor = SupervisorActor()
    first, second = AsyncMock(), AsyncMock(side_effect=RuntimeError(TOKEN))
    supervisor._worker_address_to_worker = {
        "a": SimpleNamespace(update_model_type=first),
        "b": SimpleNamespace(update_model_type=second),
    }
    options: dict = {"model_name": "wanted"}
    if credentialed:
        options["credentials"] = ModelCatalogCredentials(TOKEN)
    with pytest.raises(ModelCatalogUpdateError):
        await supervisor.update_model_type("llm", **options)
    first.assert_awaited_once_with("llm", **options)
    second.assert_awaited_once_with("llm", **options)
