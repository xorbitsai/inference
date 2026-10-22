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

import pytest

from ..cache_tracker import CacheTrackerActor


@pytest.mark.parametrize("apple_first", [False, True])
def test_cache_tracker_merges_platform_specific_model_versions(apple_first):
    tracker = CacheTrackerActor.__new__(CacheTrackerActor)
    tracker._model_name_to_version_info = {}

    linux_versions = {
        "whisper-tiny": [
            {
                "model_version": "whisper-tiny",
                "cache_status": True,
                "model_file_location": "/linux/whisper-tiny",
            }
        ]
    }
    apple_versions = {
        "whisper-tiny": [
            {
                "model_version": "whisper-tiny",
                "cache_status": False,
                "model_file_location": None,
            },
            {
                "model_version": "whisper-tiny-mlx",
                "cache_status": True,
                "model_file_location": "/apple/whisper-tiny-mlx",
            },
        ]
    }
    worker_versions = (
        [(apple_versions, "apple-worker"), (linux_versions, "linux-worker")]
        if apple_first
        else [(linux_versions, "linux-worker"), (apple_versions, "apple-worker")]
    )

    for versions, address in worker_versions:
        tracker.record_model_version(versions, address)

    merged = {
        version["model_version"]: version
        for version in tracker.get_model_versions("whisper-tiny")
    }
    assert set(merged) == {"whisper-tiny", "whisper-tiny-mlx"}
    assert merged["whisper-tiny"]["model_file_location"] == {
        "linux-worker": "/linux/whisper-tiny"
    }
    assert merged["whisper-tiny-mlx"]["model_file_location"] == {
        "apple-worker": "/apple/whisper-tiny-mlx"
    }


@pytest.mark.parametrize("new_worker_cached", [False, True])
def test_cache_tracker_backfills_model_type_without_losing_locations(new_worker_cached):
    tracker = CacheTrackerActor.__new__(CacheTrackerActor)
    tracker._model_name_to_version_info = {}
    tracker.record_model_version(
        {
            "demo": [
                {
                    "model_version": "demo-v1",
                    "cache_status": True,
                    "model_file_location": "/old/demo",
                }
            ]
        },
        "old-worker",
    )
    tracker.record_model_version(
        {
            "demo": [
                {
                    "model_version": "demo-v1",
                    "cache_status": new_worker_cached,
                    "model_file_location": "/new/demo",
                }
            ]
        },
        "new-worker",
        model_type="embedding",
    )
    tracker.record_model_version(
        {
            "demo": [
                {
                    "model_version": "demo-v1",
                    "cache_status": False,
                    "model_file_location": None,
                }
            ]
        },
        "legacy-worker",
    )
    [version] = tracker.get_model_versions("demo")
    assert version["model_type"] == "embedding"
    expected = {"old-worker": "/old/demo"}
    if new_worker_cached:
        expected["new-worker"] = "/new/demo"
    assert version["model_file_location"] == expected
    assert tracker.list_cached_models("old-worker")[0]["model_type"] == "embedding"
