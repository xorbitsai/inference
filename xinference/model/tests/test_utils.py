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
import shutil
import sys

import pytest
from tqdm.auto import tqdm

from ...utils import get_real_path
from ..utils import (
    CancellableDownloader,
    _apply_virtualenv_engine_overrides,
    _collect_virtualenv_engine_markers,
    _extract_engine_markers_from_packages,
    _force_virtualenv_engine_params,
    neutralize_broken_torchcodec,
    parse_uri,
)


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


def test_extract_engine_markers_from_packages():
    packages = [
        'vllm ; #engine# == "vllm"',
        "sglang ; #model_engine# == 'sglang'",
        "transformers>=4.51.0",
    ]
    assert _extract_engine_markers_from_packages(packages) == {"vllm", "sglang"}


def test_collect_virtualenv_engine_markers_platform_gating():
    class _VirtualEnv:
        packages = ['mlx-lm ; #engine# == "mlx"', 'vllm ; #engine# == "vllm"']

    class _Family:
        virtualenv = _VirtualEnv()
        model_specs = []

    engines = _collect_virtualenv_engine_markers(_Family())
    assert "vllm" in engines
    if sys.platform == "darwin":
        assert "mlx" in engines
    else:
        assert "mlx" not in engines


class _DummyEngineMissing:
    @staticmethod
    def check_lib():
        return False, "missing dependency"


class _DummyEngineOk:
    @staticmethod
    def check_lib():
        return True


class _DummyEngineMatchJson:
    @staticmethod
    def match_json(family, spec, quantization):
        return False


def test_force_virtualenv_engine_params_and_override():
    class _Spec:
        model_format = "pytorch"
        model_size_in_billions = "0_6"
        quantization = "none"

    class _Family:
        model_name = "qwen3"
        model_specs = [_Spec()]

    engine_params = {}
    available_params = {}
    supported = {"SGLang": [_DummyEngineMissing]}

    match_status = _force_virtualenv_engine_params(
        _Family(), supported, {"sglang"}, engine_params, available_params, False
    )

    assert "SGLang" in engine_params
    assert match_status["SGLang"] is False

    _apply_virtualenv_engine_overrides(
        engine_params, supported, {"sglang"}, True, match_status
    )
    assert engine_params["SGLang"][0]["virtualenv_required"] is True


def test_virtualenv_override_disabled_marks_unavailable():
    engine_params = {"SGLang": [{"model_name": "qwen3"}]}
    supported = {"SGLang": [_DummyEngineMissing]}

    _apply_virtualenv_engine_overrides(
        engine_params, supported, {"sglang"}, False, {"SGLang": False}
    )

    assert isinstance(engine_params["SGLang"], str)


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


def _clear_torchcodec_from_sys_modules():
    for name in [
        n for n in list(sys.modules) if n == "torchcodec" or n.startswith("torchcodec.")
    ]:
        del sys.modules[name]


def test_neutralize_broken_torchcodec_runtime_error(monkeypatch):
    """A torchcodec that raises RuntimeError on import (e.g. version-mismatched
    shared libs) is poisoned so importers see ImportError and can degrade."""
    _clear_torchcodec_from_sys_modules()
    import importlib as _importlib

    def fake_import(name, *args, **kwargs):
        if name == "torchcodec":
            raise RuntimeError("Could not load libtorchcodec")
        return _importlib.import_module(name, *args, **kwargs)

    monkeypatch.setattr("xinference.model.utils.importlib.import_module", fake_import)
    try:
        neutralize_broken_torchcodec()
        # torchcodec is now poisoned -> importing it raises ImportError, which is
        # exactly what sentence-transformers' guard tolerates.
        assert sys.modules.get("torchcodec", "missing") is None
        with pytest.raises(ImportError):
            import torchcodec  # noqa: F401
    finally:
        _clear_torchcodec_from_sys_modules()


def test_neutralize_broken_torchcodec_missing(monkeypatch):
    """A genuinely absent torchcodec is left as-is (its own ImportError stands)."""
    _clear_torchcodec_from_sys_modules()
    import importlib as _importlib

    def fake_import(name, *args, **kwargs):
        if name == "torchcodec":
            raise ModuleNotFoundError("No module named 'torchcodec'")
        return _importlib.import_module(name, *args, **kwargs)

    monkeypatch.setattr("xinference.model.utils.importlib.import_module", fake_import)
    neutralize_broken_torchcodec()
    # Not poisoned: absence is handled by the caller's own guard.
    assert "torchcodec" not in sys.modules


def test_neutralize_broken_torchcodec_healthy(monkeypatch):
    """A healthy torchcodec is left untouched."""
    _clear_torchcodec_from_sys_modules()
    import types

    fake_mod = types.ModuleType("torchcodec")
    sys.modules["torchcodec"] = fake_mod
    try:
        neutralize_broken_torchcodec()
        assert sys.modules["torchcodec"] is fake_mod
    finally:
        _clear_torchcodec_from_sys_modules()


def test_sentence_transformers_loaders_resolve_neutralizer():
    """The embedding and rerank loaders reference neutralize_broken_torchcodec
    via a relative import; verify that import path actually resolves (a wrong
    relative-import depth previously pointed at xinference.utils and broke load
    at runtime, see #5208)."""
    import ast
    import importlib
    import os

    here = os.path.dirname(__file__)
    core_files = {
        "xinference.model.embedding.sentence_transformers.core": os.path.join(
            here, "..", "embedding", "sentence_transformers", "core.py"
        ),
        "xinference.model.rerank.sentence_transformers.core": os.path.join(
            here, "..", "rerank", "sentence_transformers", "core.py"
        ),
    }

    for module_name, path in core_files.items():
        with open(path) as f:
            tree = ast.parse(f.read())
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "neutralize_broken_torchcodec" for alias in node.names
            ):
                found = True
                # Resolve the relative import against the core module's package.
                pkg = module_name.rsplit(".", 1)[0]
                base = pkg
                for _ in range(node.level - 1):
                    base = base.rsplit(".", 1)[0]
                target = f"{base}.{node.module}" if node.module else base
                mod = importlib.import_module(target)
                assert hasattr(mod, "neutralize_broken_torchcodec"), (
                    f"{module_name} imports neutralize_broken_torchcodec from "
                    f"{target!r}, which has no such symbol"
                )
        assert found, f"{module_name} no longer imports neutralize_broken_torchcodec"


def test_neutralize_broken_torchcodec_idempotent(monkeypatch):
    """After poisoning, a second call must be a no-op and must NOT re-import
    torchcodec (a stale re-import would raise+swallow ModuleNotFoundError on
    every subsequent model load)."""
    _clear_torchcodec_from_sys_modules()
    import importlib as _importlib

    calls = {"n": 0}
    real_import = _importlib.import_module

    def counting_import(name, *args, **kwargs):
        if name == "torchcodec":
            calls["n"] += 1
            raise RuntimeError("Could not load libtorchcodec")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        "xinference.model.utils.importlib.import_module", counting_import
    )
    try:
        neutralize_broken_torchcodec()
        neutralize_broken_torchcodec()
        neutralize_broken_torchcodec()
        # torchcodec import is attempted only on the first call; later calls
        # short-circuit on the sys.modules guard.
        assert calls["n"] == 1
        assert sys.modules.get("torchcodec", "missing") is None
    finally:
        _clear_torchcodec_from_sys_modules()
