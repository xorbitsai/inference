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

import asyncio
import os
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


class _ActorStub:
    pass


class _ProgressorStub:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _ModelFamilyStub:
    multimodal_projector = None

    @staticmethod
    def to_version_info():
        return {
            "model_version": "demo-v1",
            "model_file_location": "/tmp/demo-v1",
        }


class _ModelStub:
    model_family = _ModelFamilyStub()


def test_download_reporter_distinguishes_failure_from_completion():
    from xinference.core.worker import WorkerActor

    progressor = MagicMock()
    downloader = MagicMock(done=True, cancelled=False)
    succeeded_event = threading.Event()

    WorkerActor._upload_download_progress(
        progressor,
        downloader,
        completion_stage="completed",
        succeeded_event=succeeded_event,
    )
    assert progressor.set_progress.call_args.args[2]["stage"] == "failed"

    succeeded_event.set()
    WorkerActor._upload_download_progress(
        progressor,
        downloader,
        completion_stage="completed",
        succeeded_event=succeeded_event,
    )
    assert progressor.set_progress.call_args.args[2]["stage"] == "completed"


def test_download_reporter_never_reports_terminal_progress_while_active():
    from xinference.core.worker import WorkerActor

    progressor = MagicMock()
    downloader = MagicMock(cancelled=False)
    type(downloader).done = property(MagicMock(side_effect=[False, True, True]))
    downloader.get_progress.return_value = 1.0
    succeeded_event = threading.Event()
    succeeded_event.set()

    WorkerActor._upload_download_progress(
        progressor,
        downloader,
        completion_stage="completed",
        succeeded_event=succeeded_event,
    )

    assert progressor.set_progress.call_args_list[0].args[0] == 0.99
    assert progressor.set_progress.call_args_list[0].args[2]["stage"] == "downloading"
    assert progressor.set_progress.call_args_list[-1].args[0] == 1.0
    assert progressor.set_progress.call_args_list[-1].args[2]["stage"] == "completed"


@pytest.mark.asyncio
async def test_shared_download_helper_owns_cache_dispatch(monkeypatch):
    from xinference.core.worker import DownloadInfo, WorkerActor

    worker = _ActorStub()
    worker._upload_download_progress = MagicMock()
    worker.update_cache_status = AsyncMock()
    create_model_instance = MagicMock(return_value=_ModelStub())
    monkeypatch.setattr(
        "xinference.core.worker.create_model_instance", create_model_instance
    )
    monkeypatch.setenv("HF_HUB_DOWNLOAD_WORKERS", "17")

    model, downloader = await WorkerActor._download_model_files(
        worker,
        operation_uid="cache-1",
        request_id="caching-cache-1",
        download_info=DownloadInfo(),
        progressor=_ProgressorStub(),
        model_type="LLM",
        model_name="demo",
        model_engine="transformers",
        model_format="pytorch",
        model_size_in_billions=1,
        quantization="none",
        peft_model_config=None,
        download_hub="huggingface",
        model_path=None,
        model_kwargs={"multimodal_projector": "projector"},
        completion_stage="completed",
    )

    assert model.model_family.multimodal_projector == "projector"
    assert downloader.done
    create_model_instance.assert_called_once()
    worker.update_cache_status.assert_awaited_once_with(
        "demo", model.model_family.to_version_info()
    )
    assert os.environ["HF_HUB_DOWNLOAD_WORKERS"] == "17"


@pytest.mark.asyncio
async def test_cache_only_operation_skips_runtime_allocation():
    from xinference.core.worker import WorkerActor

    worker = _ActorStub()
    worker.address = "worker:1234"
    worker._cache_uid_to_download_info = {}
    worker._download_artifact_cleanup_lock = asyncio.Lock()
    worker._launch_semaphore = asyncio.Semaphore(1)
    worker._check_model_is_valid = MagicMock()
    worker._get_progressor = AsyncMock(return_value=_ProgressorStub())
    downloader = MagicMock(cancelled=False)
    worker._download_model_files = AsyncMock(return_value=(_ModelStub(), downloader))
    worker._allocate_subpool_devices = AsyncMock()
    worker._spawn_subpool = AsyncMock()
    worker.cache_builtin_model = WorkerActor.cache_builtin_model.__get__(worker)

    result = await worker.cache_builtin_model(
        cache_uid="cache-2",
        model_name="demo",
        model_size_in_billions=1,
        model_format="pytorch",
        quantization="none",
        model_engine="transformers",
    )

    assert result["cache_uid"] == "cache-2"
    assert result["worker_address"] == "worker:1234"
    assert "cache-2" not in worker._cache_uid_to_download_info
    worker._allocate_subpool_devices.assert_not_awaited()
    worker._spawn_subpool.assert_not_awaited()
    worker._download_model_files.assert_awaited_once()
    assert (
        worker._download_model_files.await_args.kwargs["completion_stage"]
        == "completed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("completion_stage", ["loading", "completed"])
@pytest.mark.parametrize("selected_python", [None, "/trusted/bin/python"])
async def test_audio_download_uses_trusted_startup_context(
    monkeypatch, completion_stage, selected_python
):
    from xinference.core.worker import DownloadInfo, WorkerActor

    worker = _ActorStub()
    worker._upload_download_progress = MagicMock()
    worker.update_cache_status = AsyncMock()
    factory = MagicMock(return_value=_ModelStub())
    monkeypatch.setattr("xinference.core.worker.create_model_instance", factory)
    await WorkerActor._download_model_files(
        worker,
        operation_uid="audio-1",
        request_id="audio-request",
        download_info=DownloadInfo(),
        progressor=_ProgressorStub(),
        model_type="audio",
        model_name="dots.tts.edit-v1.1",
        model_engine="PyTorch",
        model_format="pytorch",
        model_size_in_billions=None,
        quantization=None,
        peft_model_config=None,
        download_hub="modelscope",
        model_path=None,
        model_kwargs={
            "_xinference_cache_only": completion_stage != "completed",
            "_xinference_model_python_executable": "/untrusted/bin/python",
        },
        completion_stage=completion_stage,
        model_python_executable=selected_python,
    )
    factory.assert_called_once()
    assert factory.call_args.kwargs["_xinference_cache_only"] is (
        completion_stage == "completed"
    )
    assert (
        factory.call_args.kwargs["_xinference_model_python_executable"]
        == selected_python
    )


@pytest.mark.asyncio
async def test_cancel_cache_operation_uses_shared_download_cancellation():
    from xinference.core.worker import DownloadInfo, WorkerActor

    worker = _ActorStub()
    download_info = DownloadInfo()
    download_info.downloader = MagicMock()
    worker._cache_uid_to_download_info = {"cache-3": download_info}
    worker._cancel_download = WorkerActor._cancel_download
    worker.cancel_cache_model = WorkerActor.cancel_cache_model.__get__(worker)

    cancel_task = asyncio.create_task(worker.cancel_cache_model("cache-3"))
    await asyncio.sleep(0)
    assert download_info.cancel_event.is_set()
    download_info.downloader.cancel.assert_called_once_with()

    worker._cache_uid_to_download_info.pop("cache-3")
    await cancel_task


@pytest.mark.asyncio
async def test_cancel_cache_operation_reports_unwind_timeout(monkeypatch):
    from xinference.core import worker as worker_module
    from xinference.core.worker import DownloadInfo, WorkerActor

    worker = _ActorStub()
    download_info = DownloadInfo()
    download_info.downloader = MagicMock()
    worker._cache_uid_to_download_info = {"cache-3": download_info}
    worker._cancel_download = WorkerActor._cancel_download
    worker.cancel_cache_model = WorkerActor.cancel_cache_model.__get__(worker)
    monkeypatch.setattr(worker_module, "XINFERENCE_CANCEL_LAUNCH_TIMEOUT", 0)

    with pytest.raises(RuntimeError, match="still running"):
        await worker.cancel_cache_model("cache-3")

    assert download_info.cancel_event.is_set()
    download_info.downloader.cancel.assert_called_once_with()


def _modelscope_download_paths(model_id, root):
    file_download = pytest.importorskip("modelscope.hub.file_download")
    if hasattr(file_download, "create_temporary_directory_and_cache"):
        staging, sdk_cache = file_download.create_temporary_directory_and_cache(
            model_id, cache_dir=root
        )
    else:
        from modelscope.hub.utils.caching import ModelFileSystemCache

        owner, name = model_id.split("/")
        sdk_cache = ModelFileSystemCache(root, owner, name.replace(".", "___"))
        staging = root / "._____temp" / owner / name
        staging.mkdir(parents=True)
    return Path(sdk_cache.get_root_location()), Path(staging)


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["none", "cached", "active"])
async def test_cancelled_modelscope_download_cleans_sdk_repositories(
    monkeypatch, tmp_path, reference
):
    from xinference.core import worker as worker_module
    from xinference.core.worker import DownloadInfo, WorkerActor

    modelscope_root = tmp_path / "modelscope"
    monkeypatch.setenv("MODELSCOPE_CACHE", str(modelscope_root))
    cache_dir = tmp_path / "xinference-cache"
    monkeypatch.setattr(worker_module, "XINFERENCE_CACHE_DIR", str(cache_dir))

    model_ids = ("PhaiModelLab/Phai-M1", "Lightricks/LTX-2.5-Diffusers")
    repositories = []
    stages = []
    for model_id in model_ids:
        repository, staging = _modelscope_download_paths(
            model_id, modelscope_root / "models"
        )
        (repository / "ready.safetensors").write_bytes(b"complete weights")
        (staging / "temporary-file").write_bytes(b"partial download")
        repositories.append(repository)
        stages.append(staging)

    # SDKs expose the model ID as a compatibility link for dotted names.
    base_alias = modelscope_root / "models" / model_ids[1]
    base_alias.symlink_to(repositories[1], target_is_directory=True)
    unrelated, unrelated_stage = _modelscope_download_paths(
        "Lightricks/Other-Model", modelscope_root / "models"
    )
    repository_specs = [
        {"model_hub": "modelscope", "model_id": model_id} for model_id in model_ids
    ]
    payload = {"_download_repositories": repository_specs}
    worker = WorkerActor.__new__(WorkerActor)
    current = DownloadInfo(payload=payload)
    worker._cache_uid_to_download_info = {"cancelled": current}
    worker._model_uid_launching_guard = {}

    def finish_cancelled_download():
        worker._cache_uid_to_download_info.pop("cancelled")

    current.downloader = MagicMock()
    current.downloader.cancel.side_effect = finish_cancelled_download
    await worker.cancel_cache_model("cancelled")
    assert current.cancel_event.is_set()

    if reference == "active":
        worker._cache_uid_to_download_info["other"] = DownloadInfo(
            payload={"_download_repositories": [repository_specs[1]]}
        )
        with pytest.raises(RuntimeError, match="another download task"):
            await worker.delete_cache_model_artifacts(payload)
        assert all(path.is_dir() for path in repositories + stages)
        assert base_alias.is_symlink()
    else:
        if reference == "cached":
            cache_dir.mkdir()
            (cache_dir / "shared-weights").symlink_to(base_alias / "ready.safetensors")
        result = await worker.delete_cache_model_artifacts(payload)
        assert not any(stage.exists() for stage in stages)
        assert result["removed_bytes"] > 0
        retained = [repositories[1]] if reference == "cached" else []
        assert result["preserved_repositories"] == [str(path) for path in retained]
        assert [path for path in repositories if path.exists()] == retained
        assert base_alias.is_symlink() == bool(retained)
        if retained:
            assert (cache_dir / "shared-weights").read_bytes() == b"complete weights"

    assert unrelated.is_dir()
    assert unrelated_stage.is_dir()


def test_download_cleanup_refuses_symlinked_modelscope_owner(monkeypatch, tmp_path):
    from xinference.core.worker import WorkerActor

    modelscope_root = tmp_path / "modelscope"
    monkeypatch.setenv("MODELSCOPE_CACHE", str(modelscope_root))
    models = modelscope_root / "models"
    unrelated = models / "OtherOwner" / "Phai-M1"
    unrelated.mkdir(parents=True)
    (models / "PhaiModelLab").symlink_to(unrelated.parent, target_is_directory=True)
    payload = {
        "_download_repositories": [
            {"model_hub": "modelscope", "model_id": "PhaiModelLab/Phai-M1"}
        ]
    }

    with pytest.raises(ValueError, match="symlinked Hub repository parent"):
        WorkerActor._remove_download_repository_paths(
            WorkerActor._download_repository_paths(payload), set()
        )

    assert unrelated.is_dir()


def test_download_cleanup_removes_modelscope_snapshot_alias(monkeypatch, tmp_path):
    from xinference.core import worker as worker_module
    from xinference.core.worker import WorkerActor

    modelscope_root = tmp_path / "modelscope"
    monkeypatch.setenv("MODELSCOPE_CACHE", str(modelscope_root))
    monkeypatch.setattr(worker_module, "XINFERENCE_CACHE_DIR", str(tmp_path / "cache"))
    models = modelscope_root / "models"
    repository = models / "PhaiModelLab--Phai-M1"
    snapshot = repository / "snapshots" / "master"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors.incomplete").write_bytes(b"partial")
    alias = models / "PhaiModelLab" / "Phai-M1"
    alias.parent.mkdir()
    alias.symlink_to(snapshot, target_is_directory=True)
    payload = {
        "_download_repositories": [
            {"model_hub": "modelscope", "model_id": "PhaiModelLab/Phai-M1"}
        ]
    }

    result = WorkerActor._remove_download_repository_paths(
        WorkerActor._download_repository_paths(payload), set()
    )

    assert result["removed_repositories"] == [str(repository)]
    assert not repository.exists()
    assert not alias.is_symlink()


@pytest.mark.asyncio
async def test_operation_progress_aggregation_is_shared():
    from xinference.core.supervisor import SupervisorActor

    supervisor = _ActorStub()
    supervisor._progress_tracker = AsyncMock()
    supervisor._progress_tracker.get_progress_details = AsyncMock(
        side_effect=[
            (
                0.4,
                "Downloading",
                {
                    "stage": "downloading",
                    "download_files": [{"name": "a.bin"}],
                    "updated_at": 1,
                },
            ),
            (
                1.0,
                "Installing",
                {
                    "stage": "installing_dependencies",
                    "download_files": [],
                    "dependency_install_completed": 2,
                    "dependency_install_total": 5,
                    "dependency_install_plan": ["example==2.0", "child-pkg==3.0"],
                    "dependency_install_status": "performed",
                },
            ),
        ]
    )

    result = await SupervisorActor._get_operation_progress_details(
        supervisor,
        [
            ("launching-a", {"replica_id": 0, "replica_model_uid": "a"}),
            ("launching-b", {"replica_id": 1, "replica_model_uid": "b"}),
        ],
        target_key="replicas",
        default_stage="launching",
    )

    assert result["progress"] == pytest.approx(0.7)
    assert result["stage"] == "downloading"
    assert result["download_files"] == [
        {"name": "a.bin", "replica_id": 0, "replica_model_uid": "a"}
    ]
    assert len(result["replicas"]) == 2
    assert result["replicas"][1]["dependency_install_completed"] == 2
    assert result["replicas"][1]["dependency_install_total"] == 5
    assert result["replicas"][1]["dependency_install_plan"] == [
        "example==2.0",
        "child-pkg==3.0",
    ]
    assert result["replicas"][1]["dependency_install_status"] == "performed"
