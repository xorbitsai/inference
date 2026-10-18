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
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from importlib import import_module
from pathlib import Path
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


def catalog_record(model_type, name):
    directory = Path(__file__).parents[2] / "model" / model_type / "models"
    source = (
        directory / "LingBot-World-V2.json"
        if model_type == "world"
        else sorted(directory.glob("*.json"))[0]
    )
    record = json.loads(source.read_text(encoding="utf-8"))[0]
    if model_type == "world":
        record.pop("model_name", None)
        record["model_family"] = name
        for index, spec in enumerate(record["model_specs"]):
            spec["model_name"] = f"{name}-{index}"
    else:
        record["model_name"] = name
    return record


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
    assert json.loads(path.read_text(encoding="utf-8")) == [unrelated, *variants]
    store_model_catalog(
        path, {"model_name": "new", "readme": "原始说明"}, model_name="new"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == [
        unrelated,
        *variants,
        {"model_name": "new", "readme": "原始说明"},
    ]
    store_model_catalog(path, [])
    assert json.loads(path.read_text(encoding="utf-8")) == []


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
    assert sorted(
        json.loads(path.read_text(encoding="utf-8")), key=lambda row: row["index"]
    ) == [{"model_name": f"model-{i}", "index": i} for i in range(16)]


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
            assert json.loads(path.read_text(encoding="utf-8")) == [
                {"model_name": "old"}
            ]
            merge = pool.submit(write_partial)
            assert partial_started.wait(10)
        finally:
            release.set()
        replace.result()
        merge.result()
    assert json.loads(path.read_text(encoding="utf-8")) == [
        *full,
        {"model_name": "wanted"},
    ]


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
    unrelated = catalog_record(model_type, "unrelated")
    record = catalog_record(model_type, name)
    record["updated_at"] = 42
    store_model_catalog(path, [unrelated])
    response = SimpleNamespace(
        status_code=200,
        raise_for_status=Mock(),
        json=lambda: record,
    )
    download = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", download)
    reload = getattr(
        import_module(f"xinference.model.{model_type}"), "register_builtin_model"
    )
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
    assert json.loads(path.read_text(encoding="utf-8")) == [unrelated, record]
    reload.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["401", "404", "503", "redirect", "network", "json", "mismatch", "stored"],
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
    if failure == "stored":
        before = b"malformed local JSON"
        path.write_bytes(before)
        result.json = lambda: {"model_name": "wanted"}
    download = Mock(return_value=result)
    if failure == "network":
        download.side_effect = requests.ConnectionError(TOKEN)
    elif failure == "json":
        result.json = Mock(side_effect=ValueError(TOKEN))
    elif failure == "redirect":
        result.status_code = 302
    elif failure not in ("mismatch", "stored"):
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
    if failure in ("401", "404"):
        assert "stage=download" in caplog.text and f"status={failure}" in caplog.text
    elif failure in ("mismatch", "stored"):
        assert "stage=store" in caplog.text and "status=200" in caplog.text
        if failure == "stored":
            assert "JSONDecodeError" in caplog.text


@pytest.mark.parametrize("model_type", ["llm", "world"])
def test_single_update_loads_refreshed_and_unrelated_models_with_real_loader(
    tmp_path, monkeypatch, model_type
):
    path = tmp_path / "models.json"
    module = import_module(f"xinference.model.{model_type}")
    if model_type == "llm":
        metadata = Mock()
        monkeypatch.setattr(module, "_register_model_family_metadata", metadata)
    keep = catalog_record(model_type, "keep")
    old = catalog_record(model_type, "wanted")
    new = catalog_record(model_type, "wanted")
    new["updated_at"] = 42
    store_model_catalog(path, [old, keep])
    store_model_catalog(path, new, model_name="wanted", model_type=model_type)
    families = [] if model_type == "llm" else {}
    options = {"register_metadata": False} if model_type == "llm" else {}
    module.load_model_family_from_json(str(path), families, **options)
    if model_type == "llm":
        assert {family.model_name: family.updated_at for family in families} == {
            "wanted": 42,
            "keep": keep["updated_at"],
        }
        metadata.assert_not_called()
    else:
        assert set(families) == {
            spec["model_name"] for row in (keep, new) for spec in row["model_specs"]
        }
        assert all(
            family.updated_at == 42
            for name, variants in families.items()
            if name.startswith("wanted-")
            for family in variants
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_type", ["llm", "embedding", "rerank", "image", "audio", "video", "world"]
)
async def test_worker_rejects_invalid_family_before_publication(
    worker, monkeypatch, caplog, model_type
):
    worker, path = worker
    worker._custom_register_type_to_cls[model_type] = None
    reload = Mock()
    monkeypatch.setattr(f"xinference.model.{model_type}.register_builtin_model", reload)
    path = path.parents[1] / model_type / f"{model_type}_models.json"
    store_model_catalog(path, [catalog_record(model_type, "keep")])
    before = path.read_bytes()
    record = catalog_record(model_type, "wanted")
    record["version"] = 999
    monkeypatch.setattr(
        requests,
        "get",
        Mock(
            return_value=SimpleNamespace(
                status_code=200, raise_for_status=Mock(), json=lambda: record
            )
        ),
    )
    with pytest.raises(ModelCatalogUpdateError):
        await WorkerActor.update_model_type(worker, model_type, model_name="wanted")
    assert path.read_bytes() == before
    reload.assert_not_called()
    assert "stage=store" in caplog.text
    assert not list(path.parent.glob(".catalog-*"))


@pytest.mark.parametrize("raw", [b"broken JSON", b"{}", b'["bad row"]'])
def test_full_refresh_can_repair_a_malformed_catalog(tmp_path, raw):
    path = tmp_path / "models.json"
    path.write_bytes(raw)
    replacement = [{"model_name": "fresh"}]
    store_model_catalog(path, replacement)
    assert json.loads(path.read_text(encoding="utf-8")) == replacement


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("mode", [0o600, 0o640, 0o664])
def test_refresh_preserves_existing_permissions(tmp_path, single, mode):
    path = tmp_path / "models.json"
    path.write_bytes(b"[]")
    path.chmod(mode)
    store_model_catalog(
        path, [{"model_name": "wanted"}], model_name="wanted" if single else None
    )
    assert stat.S_IMODE(path.stat().st_mode) == mode


@pytest.mark.skipif(os.name == "nt", reason="POSIX umask")
@pytest.mark.parametrize("mask", [0o022, 0o027])
def test_new_catalog_obeys_umask_in_an_isolated_process(tmp_path, mask):
    path = tmp_path / "models.json"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sys; from pathlib import Path; "
            "from xinference.core.model_catalog import store_model_catalog; "
            "os.umask(int(sys.argv[2])); "
            "store_model_catalog(Path(sys.argv[1]), [])",
            str(path),
            str(mask),
        ],
        check=True,
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~mask


@pytest.mark.parametrize("permanent", [False, True])
def test_windows_busy_reader_retries_replacement_without_truncating(
    tmp_path, monkeypatch, permanent
):
    path = tmp_path / "models.json"
    store_model_catalog(path, [{"model_name": "keep"}])
    before = path.read_bytes()
    original = model_catalog.os.replace
    attempts = []

    def replace(source, target):
        attempts.append(source)
        assert path.read_bytes() == before
        if permanent or len(attempts) < 3:
            raise PermissionError("busy reader")
        original(source, target)

    monkeypatch.setattr(model_catalog.sys, "platform", "win32")
    monkeypatch.setattr(model_catalog.os, "replace", replace)
    monkeypatch.setattr(model_catalog.time, "sleep", Mock())
    if permanent:
        with pytest.raises(PermissionError):
            store_model_catalog(path, [])
        assert path.read_bytes() == before
        assert len(attempts) == 5
    else:
        store_model_catalog(path, [])
        assert json.loads(path.read_text(encoding="utf-8")) == []
        assert len(attempts) == 3
    assert not list(tmp_path.glob(".catalog-*"))


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
