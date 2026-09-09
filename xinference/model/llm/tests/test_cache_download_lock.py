# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ..cache_manager import LLMCacheManager


def test_replica_downloads_share_repository_lock(tmp_path):
    start = threading.Barrier(4)
    downloaded = tmp_path / "downloaded"
    calls = []

    def cache_from_hub():
        if not downloaded.exists():
            calls.append(1)
            time.sleep(0.05)
            downloaded.write_text("complete")
        return str(downloaded)

    def launch_replica(i):
        manager = object.__new__(LLMCacheManager)
        manager._model_uri = None
        manager._model_hub = "modelscope"
        manager._model_id = "org/shared-model"
        manager._v2_cache_dir_prefix = str(tmp_path)
        manager.cache_from_modelscope = cache_from_hub
        start.wait(timeout=5)
        return manager.cache()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(launch_replica, range(4)))
    assert results == [str(downloaded)] * 4
    assert len(calls) == 1
