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

from ..download_task_store import DownloadTaskStore


def _task(cache_uid: str, status: str):
    return {
        "cache_uid": cache_uid,
        "model_name": "qwen2.5-instruct",
        "model_type": "LLM",
        "model_engine": "transformers",
        "worker_address": "worker:9978",
        "status": status,
        "progress": 0.25,
        "payload": {"model_name": "qwen2.5-instruct"},
        "download_files": [{"name": "model.safetensors"}],
    }


def test_store_preserves_resumable_download_state(tmp_path):
    db_path = str(tmp_path / "downloads.db")
    store = DownloadTaskStore(db_path)
    store.upsert(_task("cache-1", "downloading"))

    store.update("cache-1", status="paused", progress=0.5, error=None)
    reopened_store = DownloadTaskStore(db_path)
    task = reopened_store.get("cache-1")

    assert task is not None
    assert task["status"] == "paused"
    assert task["progress"] == 0.5
    assert task["payload"] == {"model_name": "qwen2.5-instruct"}
    assert task["download_files"] == [{"name": "model.safetensors"}]
    assert [item["cache_uid"] for item in reopened_store.list_unfinished()] == [
        "cache-1"
    ]


def test_store_marks_active_tasks_interrupted_after_restart(tmp_path):
    store = DownloadTaskStore(str(tmp_path / "downloads.db"))
    for status in ("pending", "resuming", "downloading", "pausing"):
        store.upsert(_task(status, status))
    store.upsert(_task("paused", "paused"))
    store.upsert(_task("completed", "completed"))

    assert store.mark_active_interrupted() == 4

    unfinished = {item["cache_uid"]: item for item in store.list_unfinished()}
    assert unfinished["paused"]["status"] == "paused"
    assert "completed" not in unfinished
    for cache_uid in ("pending", "resuming", "downloading", "pausing"):
        assert unfinished[cache_uid]["status"] == "interrupted"
        assert "service restart" in unfinished[cache_uid]["error"]
