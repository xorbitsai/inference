# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from ..cache_manager import LLMCacheManager
from .test_llm_family import (
    _gguf_manager,
    _launch_gguf,
    _readme,
    _register_gguf,
    _source,
)
from .test_llm_family import fake_hub as fake_hub


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
        manager._model_name = "shared-model"
        manager._use_draft_model = False
        manager._cache_dir = str(tmp_path / "cache-dir")
        manager._v2_cache_dir_prefix = str(tmp_path)
        manager.cache_from_modelscope = cache_from_hub
        start.wait(timeout=5)
        return manager.cache()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(launch_replica, range(4)))
    assert results == [str(downloaded)] * 4
    assert len(calls) == 1


@pytest.mark.parametrize("existing_directory", [False, True], ids=["missing", "empty"])
@pytest.mark.parametrize("posix_normcase", [False, True], ids=["native", "posix"])
def test_gguf_concurrent_case_aliases_share_destination_lock_through_download(
    fake_hub, monkeypatch, existing_directory, posix_normcase
):
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    import huggingface_hub

    from ..custom import unregister_llm

    _register_gguf("my-llm", ("huggingface", "org/model-a", "model.gguf"))
    first = _gguf_manager("my-llm")
    cache_dir = first.get_cache_dir()
    parent = os.path.dirname(cache_dir)
    with tempfile.TemporaryDirectory(prefix="CaseProbe", dir=parent) as probe:
        shared = os.path.exists(
            os.path.join(parent, os.path.basename(probe).swapcase())
        )
    if existing_directory:
        os.makedirs(cache_dir)
    if posix_normcase:
        # macOS normcase preserves case even on a case-insensitive volume.
        monkeypatch.setattr(os.path, "normcase", lambda path: path)
    recorded = threading.Event()
    second_started = threading.Event()
    finish = threading.Event()
    download = huggingface_hub.hf_hub_download

    def pause_before_writing(*args, **kwargs):
        # Source recording has finished; keep the destination locked in the SDK.
        assert os.path.exists(first._source_record_path())
        recorded.set()
        assert finish.wait(10)
        return download(*args, **kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", pause_before_writing)

    def launch_second():
        second_started.set()
        return _launch_gguf("MY-LLM", "modelscope")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(_launch_gguf, "my-llm", "huggingface")
        try:
            assert recorded.wait(10)
            assert os.path.isdir(cache_dir) == existing_directory
            # Colliding registrations cannot coexist, but unregistering does not
            # cancel an in-flight launch or remove its downloaded directory.
            unregister_llm("my-llm")
            _register_gguf("MY-LLM", ("modelscope", "org-ms/model-b", "model.gguf"))
            second = _gguf_manager("MY-LLM", "modelscope")
            second_result = pool.submit(launch_second)
            assert second_started.wait(10)
            if shared:
                with pytest.raises(TimeoutError):
                    second_result.result(timeout=0.2)
            else:
                assert second_result.result(timeout=10) == second.get_cache_dir()
        finally:
            finish.set()
        assert first_result.result(timeout=10) == cache_dir
        if shared:
            with pytest.raises(
                ValueError, match="downloaded from huggingface:org/model-a"
            ):
                second_result.result(timeout=10)

    assert _readme(cache_dir, "model.gguf") == "org/model-a@None"
    with open(first._source_record_path(), encoding="utf-8") as fd:
        assert json.load(fd) == {"model.gguf": _source("huggingface", "org/model-a")}
    if shared:
        assert fake_hub.downloads == [("org/model-a", None)]
    else:
        assert _readme(second.get_cache_dir(), "model.gguf") == "org-ms/model-b@None"
        with open(second._source_record_path(), encoding="utf-8") as fd:
            assert json.load(fd) == {
                "model.gguf": _source("modelscope", "org-ms/model-b")
            }
        assert sorted(fake_hub.downloads) == [
            ("org-ms/model-b", None),
            ("org/model-a", None),
        ]
