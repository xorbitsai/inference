# Copyright 2022-2025 XProbe Inc.
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
from ..utils import CancellableDownloader


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
    from ..llm import BUILTIN_LLM_FAMILIES
    from ..llm.llm_family import cache_from_huggingface

    cache_dir = None

    try:
        with CancellableDownloader() as downloader:
            family = next(
                f for f in BUILTIN_LLM_FAMILIES if f.model_name == "qwen2.5-instruct"
            )
            spec = next(
                s
                for s in family.model_specs
                if s.model_format == "pytorch" and s.model_size_in_billions == "0_5"
            )

            async def check():
                while not done:
                    await asyncio.sleep(1)
                    progress = downloader.get_progress()
                    assert progress >= 0

            done = False
            check_task = asyncio.create_task(check())
            # download from huggingface
            cache_dir = await asyncio.to_thread(cache_from_huggingface, family, spec)
            done = True

            await check_task
            assert downloader.get_progress() == 1.0
    finally:
        if cache_dir:
            shutil.rmtree(get_real_path(cache_dir))
            shutil.rmtree(cache_dir)


async def test_download_modelscope():
    from ..llm import BUILTIN_MODELSCOPE_LLM_FAMILIES
    from ..llm.llm_family import cache_from_modelscope

    cache_dir = None

    try:
        with CancellableDownloader() as downloader:
            family = next(
                f
                for f in BUILTIN_MODELSCOPE_LLM_FAMILIES
                if f.model_name == "qwen2.5-instruct"
            )
            spec = next(
                s
                for s in family.model_specs
                if s.model_format == "pytorch" and s.model_size_in_billions == "0_5"
            )

            async def check():
                while not done:
                    await asyncio.sleep(1)
                    progress = downloader.get_progress()
                    assert progress >= 0

            done = False
            check_task = asyncio.create_task(check())
            # download from huggingface
            cache_dir = await asyncio.to_thread(cache_from_modelscope, family, spec)
            done = True

            await check_task
            assert downloader.get_progress() == 1.0
    finally:
        if cache_dir:
            shutil.rmtree(get_real_path(cache_dir))
            shutil.rmtree(cache_dir)


async def test_cancel():
    from ..llm import BUILTIN_MODELSCOPE_LLM_FAMILIES
    from ..llm.llm_family import cache_from_modelscope

    with CancellableDownloader() as downloader:
        family = next(
            f
            for f in BUILTIN_MODELSCOPE_LLM_FAMILIES
            if f.model_name == "qwen2.5-instruct"
        )
        spec = next(
            s
            for s in family.model_specs
            if s.model_format == "pytorch" and s.model_size_in_billions == "0_5"
        )

        # download from huggingface
        cache_task = asyncio.create_task(
            asyncio.to_thread(cache_from_modelscope, family, spec)
        )

        await asyncio.sleep(1)
        downloader.cancel()

        with pytest.raises(asyncio.CancelledError):
            await cache_task
        assert downloader.get_progress() == 1.0
