# Copyright 2022-2026 XProbe Inc.
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
import shutil

import pytest
from tqdm.auto import tqdm

from ...utils import get_real_path
from ..utils import CancellableDownloader, parse_uri


def test_parse_uri():
    scheme, path = parse_uri("dir")
    assert scheme == "file"
    assert path == "dir"

    scheme, path = parse_uri("dir/file")
    assert scheme == "file"
    assert path == "dir/file"

    scheme, path = parse_uri("s3://bucket")
    assert scheme == "s3"
    assert path == "bucket"

    scheme, path = parse_uri("s3://bucket/dir")
    assert scheme == "s3"
    assert path == "bucket/dir"


def test_tqdm_patch():
    downloader = CancellableDownloader(cancel_error_cls=RuntimeError)

    with downloader:
        all_bar = tqdm(total=10)

        download_bars = [tqdm(total=300, unit="B") for _ in range(10)]

        for i in range(5):
            download_bars[i].update(300)

        all_bar.update(5)

        for i in range(5, 10):
            download_bars[i].update(150)

        expect = 0.5 + 0.5 * 1 / 2
        assert expect == downloader.get_progress()

        downloader.cancel()

        with pytest.raises(RuntimeError):
            all_bar.update(6)

    assert downloader.done


async def test_download_hugginface():
    import os

    # Skip network-intensive tests on CI to avoid timeout issues
    if os.environ.get("CI"):
        pytest.skip("Skip network-intensive download test on CI to avoid timeout")

    from ..llm import BUILTIN_LLM_FAMILIES
    from ..llm.cache_manager import LLMCacheManager as CacheManager

    cache_dir = None

    try:
        with CancellableDownloader() as downloader:
            family = next(
                f for f in BUILTIN_LLM_FAMILIES if f.model_name == "qwen2.5-instruct"
            ).copy()
            spec = next(
                s
                for s in family.model_specs
                if s.model_format == "pytorch"
                and s.model_size_in_billions == "0_5"
                and s.model_hub == "huggingface"
            )
            family.model_specs = [spec]

            async def check():
                last = None
                stagnant = 0
                while not done:
                    await asyncio.sleep(1)
                    progress = downloader.get_progress()
                    assert progress >= 0
                    if progress == last:
                        stagnant += 1
                        if stagnant > 60:  # no changes for 1 minute
                            raise TimeoutError("Download stuck")
                    else:
                        stagnant = 0
                    last = progress

            done = False
            check_task = asyncio.create_task(check())
            # download from huggingface
            cache_dir = await asyncio.to_thread(
                CacheManager(family).cache_from_huggingface
            )
            done = True

            await check_task
            assert downloader.get_progress() == 1.0
    finally:
        if cache_dir:
            shutil.rmtree(get_real_path(cache_dir))
            shutil.rmtree(cache_dir)


async def test_download_modelscope():
    import os

    # Skip network-intensive tests on CI to avoid timeout issues
    if os.environ.get("CI"):
        pytest.skip("Skip network-intensive download test on CI to avoid timeout")

    from ..llm import BUILTIN_LLM_FAMILIES
    from ..llm.cache_manager import LLMCacheManager as CacheManager

    cache_dir = None

    try:
        with CancellableDownloader() as downloader:
            family = next(
                f for f in BUILTIN_LLM_FAMILIES if f.model_name == "qwen2.5-instruct"
            ).copy()
            spec = next(
                s
                for s in family.model_specs
                if s.model_format == "pytorch"
                and s.model_size_in_billions == "0_5"
                and s.model_hub == "modelscope"
            )
            family.model_specs = [spec]

            async def check():
                last = None
                stagnant = 0
                while not done:
                    await asyncio.sleep(1)
                    progress = downloader.get_progress()
                    assert progress >= 0
                    if progress == last:
                        stagnant += 1
                        if stagnant > 60:  # no changes for 1 minute
                            raise TimeoutError("Download stuck")
                    else:
                        stagnant = 0
                    last = progress

            done = False
            check_task = asyncio.create_task(check())
            # download from huggingface
            cache_dir = await asyncio.to_thread(
                CacheManager(family).cache_from_modelscope
            )
            done = True

            await check_task
            assert downloader.get_progress() == 1.0
    finally:
        if cache_dir:
            shutil.rmtree(get_real_path(cache_dir))
            shutil.rmtree(cache_dir)


async def test_cancel():
    from ..llm import BUILTIN_LLM_FAMILIES
    from ..llm.cache_manager import LLMCacheManager as CacheManager

    with CancellableDownloader() as downloader:
        family = next(
            f for f in BUILTIN_LLM_FAMILIES if f.model_name == "qwen2.5-instruct"
        ).copy()
        spec = next(
            s
            for s in family.model_specs
            if s.model_format == "pytorch"
            and s.model_size_in_billions == "0_5"
            and s.model_hub == "modelscope"
        )
        family.model_specs = [spec]

        # download from huggingface
        cache_task = asyncio.create_task(
            asyncio.to_thread(CacheManager(family).cache_from_modelscope)
        )

        await asyncio.sleep(1)
        downloader.cancel()

        with pytest.raises(asyncio.CancelledError):
            await cache_task
        assert downloader.get_progress() == 1.0
