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
import os
import threading
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
            (1.0, "Loaded", {"stage": "loading", "download_files": []}),
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
