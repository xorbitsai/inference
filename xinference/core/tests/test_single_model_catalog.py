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

# CI: linux

"""GitHub category and single-model refresh."""

import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import requests

from xinference.core import model_catalog
from xinference.core.exceptions import (
    ModelCatalogUnavailableError,
    ModelCatalogUpdateError,
)
from xinference.core.model_catalog import store_model_catalog
from xinference.core.supervisor import SupervisorActor
from xinference.core.worker import WorkerActor

COMMIT = "a" * 40


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


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", model_catalog.MODEL_KINDS)
async def test_worker_stores_and_reloads_github_catalog(
    tmp_path, monkeypatch, model_type
):
    monkeypatch.setattr("xinference.constants.XINFERENCE_MODEL_DIR", str(tmp_path))
    module = import_module(f"xinference.model.{model_type}")
    reload = Mock()
    monkeypatch.setattr(module, "register_builtin_model", reload)
    path = model_catalog.github_catalog_path(model_type)
    unrelated = catalog_record(model_type, "unrelated")
    updated = catalog_record(model_type, "wanted")
    worker = SimpleNamespace()

    await WorkerActor.update_model_type(worker, model_type, COMMIT, catalog=[unrelated])
    await WorkerActor.update_model_type(
        worker, model_type, COMMIT, catalog=[updated], model_name="wanted"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "last_update_commit": COMMIT,
        "complete": True,
        "models": [unrelated, updated],
    }
    assert reload.call_count == 2


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
@pytest.mark.parametrize("model_type", model_catalog.MODEL_KINDS)
@pytest.mark.parametrize("single", [False, True])
async def test_worker_rejects_invalid_family_before_publication(
    tmp_path, monkeypatch, model_type, single
):
    monkeypatch.setattr("xinference.constants.XINFERENCE_MODEL_DIR", str(tmp_path))
    path = model_catalog.github_catalog_path(model_type)
    reload = Mock()
    monkeypatch.setattr(f"xinference.model.{model_type}.register_builtin_model", reload)
    store_model_catalog(
        path, [catalog_record(model_type, "keep")], github_commit=COMMIT
    )
    before = path.read_bytes()
    record = catalog_record(model_type, "wanted")
    record["version"] = 999
    with pytest.raises(ModelCatalogUpdateError):
        await WorkerActor.update_model_type(
            SimpleNamespace(),
            model_type,
            COMMIT,
            catalog=[record],
            model_name="wanted" if single else None,
        )
    assert path.read_bytes() == before
    reload.assert_not_called()
    assert not list(path.parent.glob(".catalog-*"))


@pytest.mark.parametrize("has_previous", [False, True])
def test_failed_reload_restores_local_catalog(tmp_path, has_previous):
    path = tmp_path / "catalog.json"
    previous = [{"model_name": "previous"}]
    if has_previous:
        store_model_catalog(path, previous, github_commit=COMMIT)
    reload = Mock(side_effect=[RuntimeError("reload failed"), None])
    with pytest.raises(RuntimeError, match="reload failed"):
        store_model_catalog(
            path, [{"model_name": "new"}], github_commit=COMMIT, reload_catalog=reload
        )
    assert model_catalog.read_github_catalog(path) == (
        has_previous,
        previous if has_previous else [],
    )
    assert reload.call_count == 2


@pytest.mark.asyncio
async def test_image_github_refresh_overrides_without_timestamps_and_preserves_other_models(
    tmp_path, monkeypatch
):
    from xinference.model import image

    monkeypatch.setattr("xinference.constants.XINFERENCE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(image, "XINFERENCE_MODEL_DIR", str(tmp_path))
    worker = SimpleNamespace()
    name = "MiniCPM-OCR-1.0"
    (bundled,) = json.loads(
        (Path(image.__file__).parent / "models" / f"{name}.json").read_text()
    )
    changed = {**bundled, "updated_at": None, "model_specs": bundled["model_specs"][:1]}
    legacy = catalog_record("image", "legacy-test-model")
    legacy_path = tmp_path / "v2/builtin/image/image_models.json"
    store_model_catalog(legacy_path, [legacy])

    try:
        await WorkerActor.update_model_type(
            worker, "image", COMMIT, catalog=[changed], model_name=name
        )
        models = image.BUILTIN_IMAGE_MODELS
        assert "legacy-test-model" in models
        assert {spec.model_format for spec in models[name]} == {"pytorch"}
        assert not any(spec.is_builtin for spec in models[name])
        image.register_builtin_model()
        assert {spec.model_format for spec in models[name]} == {"pytorch"}

        await WorkerActor.update_model_type(worker, "image", COMMIT, catalog=[bundled])
        assert set(models) == {name}
        assert all(spec.is_builtin for spec in models[name])
    finally:
        model_catalog.github_catalog_path("image").unlink(missing_ok=True)
        legacy_path.unlink(missing_ok=True)
        image.register_builtin_model()


@pytest.mark.asyncio
async def test_world_single_refresh_removes_absent_family_variants(
    tmp_path, monkeypatch
):
    from xinference.model import world

    monkeypatch.setattr("xinference.constants.XINFERENCE_MODEL_DIR", str(tmp_path))
    path = Path(world.__file__).parent / "models" / "LingBot-World-V2.json"
    (family,) = json.loads(path.read_text(encoding="utf-8"))
    removed = family["model_specs"].pop()["model_name"]
    retained = {spec["model_name"] for spec in family["model_specs"]}
    other = "Astra"
    try:
        world.register_builtin_model()
        assert removed in world.BUILTIN_WORLD_MODELS

        await WorkerActor.update_model_type(
            SimpleNamespace(),
            "world",
            COMMIT,
            catalog=[family],
            model_name=family["model_family"],
        )

        for registry in (
            world.BUILTIN_WORLD_MODELS,
            world.WORLD_MODEL_DESCRIPTIONS,
            world.WORLD_ENGINES,
        ):
            assert removed not in registry
            assert retained | {other} <= registry.keys()
        assert all(
            model.is_builtin
            for name in retained
            for model in world.BUILTIN_WORLD_MODELS[name]
        )
    finally:
        model_catalog.github_catalog_path("world").unlink(missing_ok=True)
        world.register_builtin_model()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_type", "name"),
    [("embedding", "Qwen3-VL-Embedding-2B"), ("llm", "baichuan-2")],
)
async def test_exact_packaged_github_model_restores_trust_after_legacy_override(
    tmp_path, monkeypatch, model_type, name
):
    from xinference.model.utils import allow_trust_remote_code

    monkeypatch.setattr("xinference.constants.XINFERENCE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("xinference.constants.XINFERENCE_TRUST_REMOTE_CODE", False)
    module = import_module(f"xinference.model.{model_type}")
    path = Path(module.__file__).parent / "models" / f"{name}.json"
    (bundled,) = json.loads(path.read_text(encoding="utf-8"))
    legacy = {
        **bundled,
        "updated_at": bundled["updated_at"] + 1,
        "model_description": "Legacy catalog version",
    }
    legacy_path = tmp_path / "v2" / "builtin" / model_type / f"{model_type}_models.json"
    store_model_catalog(legacy_path, [legacy])

    def active_family():
        if model_type == "llm":
            return next(
                family
                for family in module.BUILTIN_LLM_FAMILIES
                if family.model_name == name
            )
        return module.BUILTIN_EMBEDDING_MODELS[name][0]

    worker = SimpleNamespace()
    try:
        module.register_builtin_model()
        assert active_family().model_description == legacy["model_description"]
        assert not allow_trust_remote_code(active_family())

        await WorkerActor.update_model_type(
            worker, model_type, COMMIT, catalog=[bundled], model_name=name
        )
        assert active_family().is_builtin is True
        assert allow_trust_remote_code(active_family())

        changed = {
            **bundled,
            "model_description": "Changed GitHub definition",
            "is_builtin": True,
        }
        await WorkerActor.update_model_type(
            worker, model_type, COMMIT, catalog=[changed], model_name=name
        )
        assert active_family().model_description == changed["model_description"]
        assert active_family().is_builtin is False
        assert not allow_trust_remote_code(active_family())
    finally:
        model_catalog.github_catalog_path(model_type).unlink(missing_ok=True)
        legacy_path.unlink(missing_ok=True)
        module.register_builtin_model()


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


def test_github_category_pins_directory_and_files_to_one_commit(monkeypatch):
    def get(url, *, timeout, params=None):
        assert timeout == 30
        if url.endswith("/commits/main"):
            payload = {"sha": COMMIT}
        elif url.endswith("/contents/xinference/model/image/models"):
            assert params == {"ref": COMMIT}
            payload = [
                {"name": "Zebra.json", "type": "file"},
                {"name": "alpha.json", "type": "file"},
                {"name": "README.md", "type": "file"},
            ]
        else:
            assert f"/{COMMIT}/xinference/model/image/models/" in url
            payload = [{"model_name": Path(url).stem}]
        return SimpleNamespace(raise_for_status=Mock(), json=lambda: payload)

    mock = Mock(side_effect=get)
    monkeypatch.setattr(requests, "get", mock)
    assert model_catalog.download_github_catalog("image") == (
        COMMIT,
        None,
        [{"model_name": "alpha"}, {"model_name": "Zebra"}],
    )
    assert mock.call_count == 4


def test_world_variant_updates_its_entire_family(monkeypatch):
    record = catalog_record("world", "family")
    responses = iter(
        [
            {"sha": COMMIT},
            [{"name": "family.json", "type": "file"}],
            [record],
        ]
    )
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **kw: SimpleNamespace(
            raise_for_status=Mock(), json=lambda: next(responses)
        ),
    )
    assert model_catalog.download_github_catalog(
        "world", record["model_specs"][0]["model_name"]
    ) == (COMMIT, "family", [record])


@pytest.mark.asyncio
async def test_supervisor_downloads_once_and_sends_same_snapshot_to_all_workers(
    monkeypatch,
):
    catalog = [{"model_name": "wanted"}]
    download = Mock(return_value=(COMMIT, "wanted", catalog))
    monkeypatch.setattr(model_catalog, "download_github_catalog", download)
    supervisor = SupervisorActor()
    first, second = AsyncMock(), AsyncMock(side_effect=RuntimeError("unavailable"))
    supervisor._worker_address_to_worker = {
        "a": SimpleNamespace(update_model_type=first),
        "b": SimpleNamespace(update_model_type=second),
    }
    with pytest.raises(ModelCatalogUpdateError):
        await supervisor.update_model_type("llm", model_name="wanted")
    download.assert_called_once_with("llm", "wanted")
    first.assert_awaited_once_with("llm", COMMIT, catalog=catalog, model_name="wanted")
    second.assert_awaited_once_with("llm", COMMIT, catalog=catalog, model_name="wanted")


@pytest.mark.asyncio
@pytest.mark.parametrize("has_worker", [False, True])
async def test_supervisor_download_failure_leaves_workers_untouched(
    monkeypatch, has_worker
):
    download = Mock(side_effect=requests.ConnectionError("GitHub unavailable"))
    monkeypatch.setattr(model_catalog, "download_github_catalog", download)
    supervisor = SupervisorActor()
    update = AsyncMock()
    if has_worker:
        supervisor._worker_address_to_worker = {
            "worker": SimpleNamespace(update_model_type=update)
        }
    with pytest.raises(
        ModelCatalogUpdateError if has_worker else ModelCatalogUnavailableError
    ):
        await supervisor.update_model_type("llm")
    assert download.call_count == int(has_worker)
    update.assert_not_awaited()
