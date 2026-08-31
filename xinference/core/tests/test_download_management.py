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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ..supervisor import SupervisorActor


@pytest.mark.asyncio
async def test_list_model_downloads_returns_only_active_downloads():
    downloading = SimpleNamespace(
        model_name="qwen2.5-instruct",
        model_uid="qwen-downloading",
        model_version="qwen2.5-instruct-pytorch-7b-none",
        status="CREATING",
        instance_created_ts=10,
        replica_statuses=[
            SimpleNamespace(
                replica_model_uid="qwen-downloading-0",
                worker_address="worker-0:9978",
            )
        ],
    )
    loading = SimpleNamespace(
        model_name="embedding",
        model_uid="embedding-loading",
        model_version="embedding",
        status="LOADING",
        instance_created_ts=20,
        replica_statuses=[],
    )
    ready = SimpleNamespace(
        model_name="ready",
        model_uid="ready",
        model_version="ready",
        status="READY",
        instance_created_ts=30,
        replica_statuses=[],
    )
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._download_task_store = MagicMock()
    supervisor._download_task_store.list_unfinished.return_value = []
    supervisor._cache_uid_to_worker = {}
    supervisor._status_guard_ref = SimpleNamespace(
        get_instance_info=AsyncMock(return_value=[ready, loading, downloading])
    )
    supervisor.get_launch_builtin_model_progress_details = AsyncMock(
        side_effect=[
            {
                "progress": 0.8,
                "stage": "loading",
                "download_files": [],
                "replicas": [],
            },
            {
                "progress": 0.4,
                "stage": "downloading",
                "download_files": [{"name": "model.safetensors"}],
                "replicas": [
                    {
                        "replica_id": 0,
                        "replica_model_uid": "qwen-downloading-0",
                        "progress": 0.4,
                        "stage": "downloading",
                        "download_files": [{"name": "model.safetensors"}],
                    }
                ],
            },
        ]
    )

    result = await SupervisorActor.list_model_downloads(supervisor)

    assert result == [
        {
            "kind": "launch",
            "cache_uid": None,
            "model_name": "qwen2.5-instruct",
            "model_uid": "qwen-downloading",
            "model_version": "qwen2.5-instruct-pytorch-7b-none",
            "status": "CREATING",
            "instance_created_ts": 10,
            "progress": 0.4,
            "stage": "downloading",
            "download_files": [
                {
                    "name": "model.safetensors",
                    "worker_address": "worker-0:9978",
                }
            ],
            "replicas": [
                {
                    "replica_id": 0,
                    "replica_model_uid": "qwen-downloading-0",
                    "progress": 0.4,
                    "stage": "downloading",
                    "download_files": [
                        {
                            "name": "model.safetensors",
                            "worker_address": "worker-0:9978",
                        }
                    ],
                    "worker_address": "worker-0:9978",
                }
            ],
            "error": None,
            "resumable": False,
        }
    ]
    assert supervisor.get_launch_builtin_model_progress_details.await_count == 2


@pytest.mark.asyncio
async def test_list_model_downloads_includes_paused_persistent_task():
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._status_guard_ref = SimpleNamespace(
        get_instance_info=AsyncMock(return_value=[])
    )
    supervisor._cache_uid_to_worker = {}
    supervisor._download_task_store = MagicMock()
    supervisor._download_task_store.list_unfinished.return_value = [
        {
            "cache_uid": "cache-qwen",
            "model_name": "qwen2.5-instruct",
            "model_type": "LLM",
            "model_engine": "transformers",
            "model_version": "qwen-v1",
            "worker_address": "worker-0:9978",
            "status": "paused",
            "progress": 0.45,
            "payload": {
                "model_size_in_billions": 7,
                "model_format": "pytorch",
                "quantization": "none",
            },
            "download_files": [{"name": "model.safetensors"}],
            "error": None,
            "created_at": 20,
            "updated_at": 30,
        }
    ]

    [result] = await SupervisorActor.list_model_downloads(supervisor)

    assert result["kind"] == "cache"
    assert result["cache_uid"] == "cache-qwen"
    assert result["stage"] == "paused"
    assert result["resumable"] is True
    assert result["progress"] == 0.45
    assert result["model_size_in_billions"] == 7
    assert result["model_format"] == "pytorch"
    assert result["quantization"] == "none"
    assert result["download_files"] == [
        {
            "name": "model.safetensors",
            "worker_address": "worker-0:9978",
        }
    ]


@pytest.mark.asyncio
async def test_persisted_interruption_wins_over_stale_progress():
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._cache_uid_to_worker = {}
    supervisor._download_task_store = MagicMock()
    supervisor._download_task_store.get.return_value = {
        "cache_uid": "cache-qwen",
        "status": "interrupted",
        "progress": 0.45,
        "worker_address": "worker-0:9978",
        "download_files": [{"name": "model.safetensors"}],
        "error": "Download interrupted by service restart",
        "updated_at": 30,
    }
    supervisor._get_operation_progress_details = AsyncMock()

    result = await SupervisorActor.get_cache_builtin_model_progress_details(
        supervisor, "cache-qwen"
    )

    assert result["stage"] == "interrupted"
    assert result["progress"] == 0.45
    supervisor._get_operation_progress_details.assert_not_awaited()


@pytest.mark.asyncio
async def test_monitor_does_not_replace_paused_progress_with_cancelled_progress():
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor.get_cache_builtin_model_progress_details = AsyncMock(
        return_value={
            "progress": 1.0,
            "stage": "cancelled",
            "download_files": [],
        }
    )
    supervisor._download_task_store = MagicMock()
    supervisor._download_task_store.get.return_value = {
        "cache_uid": "cache-qwen",
        "status": "pausing",
        "progress": 0.45,
        "download_files": [{"name": "model.safetensors"}],
    }

    await SupervisorActor._snapshot_cache_download(supervisor, "cache-qwen")

    supervisor._download_task_store.update.assert_not_called()


@pytest.mark.asyncio
async def test_pause_download_preserves_task_for_resume():
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._download_task_store = MagicMock()
    supervisor._download_task_store.get.side_effect = [
        {"cache_uid": "cache-qwen", "status": "downloading"},
        {"cache_uid": "cache-qwen", "status": "paused"},
    ]
    worker = SimpleNamespace(cancel_cache_model=AsyncMock())
    supervisor._cache_uid_to_worker = {"cache-qwen": worker}
    supervisor._cache_pause_requested = set()
    supervisor._snapshot_cache_download = AsyncMock()

    result = await SupervisorActor.pause_cache_builtin_model(supervisor, "cache-qwen")

    assert result["status"] == "paused"
    assert "cache-qwen" not in supervisor._cache_pause_requested
    supervisor._snapshot_cache_download.assert_awaited_once_with(
        "cache-qwen", preserve_status=True
    )
    worker.cancel_cache_model.assert_awaited_once_with("cache-qwen")
    assert supervisor._download_task_store.update.call_args_list[-1].kwargs == {
        "status": "paused",
        "error": None,
    }


@pytest.mark.asyncio
async def test_pause_worker_failure_marks_task_interrupted():
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._download_task_store = MagicMock()
    supervisor._download_task_store.get.return_value = {
        "cache_uid": "cache-qwen",
        "status": "downloading",
    }
    worker = SimpleNamespace(
        cancel_cache_model=AsyncMock(side_effect=RuntimeError("worker disconnected"))
    )
    supervisor._cache_uid_to_worker = {"cache-qwen": worker}
    supervisor._cache_pause_requested = set()
    supervisor._snapshot_cache_download = AsyncMock()

    with pytest.raises(RuntimeError, match="marked interrupted"):
        await SupervisorActor.pause_cache_builtin_model(supervisor, "cache-qwen")

    assert supervisor._download_task_store.update.call_args_list[-1].kwargs == {
        "status": "interrupted",
        "error": "worker disconnected",
    }
    assert "cache-qwen" not in supervisor._cache_pause_requested


@pytest.mark.asyncio
async def test_pause_resume_task_before_worker_assignment():
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._download_task_store = MagicMock()
    supervisor._download_task_store.get.side_effect = [
        {"cache_uid": "cache-qwen", "status": "resuming"},
        {"cache_uid": "cache-qwen", "status": "paused"},
    ]
    resume_task = asyncio.create_task(asyncio.Event().wait())
    supervisor._cache_uid_to_worker = {}
    supervisor._cache_uid_to_task = {"cache-qwen": resume_task}
    supervisor._cache_pause_requested = set()

    result = await SupervisorActor.pause_cache_builtin_model(supervisor, "cache-qwen")

    assert resume_task.cancelled()
    assert result["status"] == "paused"
    assert "cache-qwen" not in supervisor._cache_pause_requested


@pytest.mark.asyncio
async def test_cancel_resume_task_before_worker_assignment():
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._download_task_store = MagicMock()
    resume_task = asyncio.create_task(asyncio.Event().wait())
    supervisor._cache_uid_to_worker = {}
    supervisor._cache_uid_to_task = {"cache-qwen": resume_task}
    supervisor._cache_cancel_requested = set()

    await SupervisorActor.cancel_cache_builtin_model(supervisor, "cache-qwen")

    assert resume_task.cancelled()
    supervisor._download_task_store.delete.assert_called_once_with("cache-qwen")
    assert "cache-qwen" not in supervisor._cache_cancel_requested


@pytest.mark.asyncio
async def test_delete_paused_download_removes_artifacts_before_task():
    payload = {
        "model_name": "qwen2.5-instruct",
        "model_type": "LLM",
        "_download_repositories": [
            {"model_hub": "modelscope", "model_id": "Qwen/Qwen2.5-7B"}
        ],
    }
    task = {
        "cache_uid": "cache-qwen",
        "status": "paused",
        "worker_address": "worker-0:9978",
        "payload": payload,
    }
    protected_payload = {
        "model_name": "other",
        "_download_repositories": [
            {"model_hub": "modelscope", "model_id": "org/other"}
        ],
    }
    worker = SimpleNamespace(
        delete_cache_model_artifacts=AsyncMock(return_value={"removed_bytes": 1024})
    )
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._download_task_store = MagicMock()
    supervisor._download_task_store.get.return_value = task
    supervisor._download_task_store.list_unfinished.return_value = [
        task,
        {
            "cache_uid": "cache-other",
            "worker_address": "worker-0:9978",
            "payload": protected_payload,
        },
    ]
    supervisor._cache_uid_to_worker = {}
    supervisor._worker_address_to_worker = {"worker-0:9978": worker}

    result = await SupervisorActor.delete_cache_builtin_model(supervisor, "cache-qwen")

    assert result == {"removed_bytes": 1024}
    worker.delete_cache_model_artifacts.assert_awaited_once_with(
        payload, [protected_payload]
    )
    supervisor._download_task_store.delete.assert_called_once_with("cache-qwen")


@pytest.mark.asyncio
async def test_delete_download_keeps_task_when_artifact_cleanup_fails():
    task = {
        "cache_uid": "cache-qwen",
        "status": "paused",
        "worker_address": "worker-0:9978",
        "payload": {"model_name": "qwen2.5-instruct"},
    }
    worker = SimpleNamespace(
        delete_cache_model_artifacts=AsyncMock(
            side_effect=RuntimeError("repository still in use")
        )
    )
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._download_task_store = MagicMock()
    supervisor._download_task_store.get.return_value = task
    supervisor._download_task_store.list_unfinished.return_value = [task]
    supervisor._cache_uid_to_worker = {}
    supervisor._worker_address_to_worker = {"worker-0:9978": worker}

    with pytest.raises(RuntimeError, match="still in use"):
        await SupervisorActor.delete_cache_builtin_model(supervisor, "cache-qwen")

    supervisor._download_task_store.delete.assert_not_called()


@pytest.mark.asyncio
async def test_resume_persistent_download_reuses_saved_payload():
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._download_task_store = MagicMock()
    task = {
        "cache_uid": "cache-qwen",
        "model_name": "qwen2.5-instruct",
        "status": "interrupted",
        "worker_address": "worker-0:9978",
        "payload": {
            "cache_uid": "cache-qwen",
            "model_name": "qwen2.5-instruct",
            "model_size_in_billions": 7,
            "model_format": "pytorch",
            "quantization": "none",
            "model_engine": "transformers",
            "model_type": "LLM",
            "worker_ip": "worker-0:9978",
        },
    }
    supervisor._download_task_store.get.side_effect = [
        task,
        {**task, "status": "resuming"},
    ]
    supervisor._cache_uid_to_worker = {}
    supervisor._cache_uid_to_task = {}
    supervisor._worker_address_to_worker = {"worker-0:9978": object()}
    supervisor.cache_builtin_model = AsyncMock(return_value={})

    result = await SupervisorActor.resume_cache_builtin_model(supervisor, "cache-qwen")
    await asyncio.sleep(0)

    assert result["status"] == "resuming"
    supervisor._download_task_store.update.assert_called_once_with(
        "cache-qwen", status="resuming", error=None
    )
    supervisor.cache_builtin_model.assert_awaited_once_with(
        cache_uid="cache-qwen",
        model_name="qwen2.5-instruct",
        model_size_in_billions=7,
        model_format="pytorch",
        quantization="none",
        model_engine="transformers",
        model_type="LLM",
        worker_ip="worker-0:9978",
        _resume=True,
    )


@pytest.mark.asyncio
async def test_resume_worker_selection_failure_becomes_failed():
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._download_task_store = MagicMock()
    task = {
        "cache_uid": "cache-qwen",
        "model_name": "qwen2.5-instruct",
        "status": "interrupted",
        "worker_address": "worker-0:9978",
        "payload": {
            "cache_uid": "cache-qwen",
            "model_name": "qwen2.5-instruct",
            "model_engine": "transformers",
            "model_type": "LLM",
            "worker_ip": "worker-0:9978",
        },
    }
    resuming_task = {**task, "status": "resuming", "error": None}
    supervisor._download_task_store.get.side_effect = [
        task,
        resuming_task,
        resuming_task,
        resuming_task,
    ]
    supervisor._cache_uid_to_worker = {}
    supervisor._cache_uid_to_task = {}
    supervisor._cache_pause_requested = set()
    supervisor._cache_cancel_requested = set()
    supervisor._worker_address_to_worker = {"worker-0:9978": object()}
    supervisor.cache_builtin_model = AsyncMock(
        side_effect=RuntimeError("No available worker")
    )

    await SupervisorActor.resume_cache_builtin_model(supervisor, "cache-qwen")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert supervisor._download_task_store.update.call_args_list[-1].kwargs == {
        "status": "failed",
        "error": "No available worker",
    }
