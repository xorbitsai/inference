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
import io
import shutil
import sys
import threading
from contextvars import copy_context

import pytest
import tqdm as tqdm_module
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
    resolve_media_seed,
)


def test_resolve_media_seed():
    assert resolve_media_seed(0) == 0
    assert resolve_media_seed(42) == 42
    random_seed = resolve_media_seed(-1)
    assert random_seed is not None
    assert 0 <= random_seed <= 2**31 - 1

    with pytest.raises(ValueError, match="Seed must be an integer"):
        resolve_media_seed(True)
    with pytest.raises(ValueError, match="Seed must be -1"):
        resolve_media_seed(-2)


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


def test_tqdm_patch_uses_equal_file_weights():
    downloader = CancellableDownloader(cancel_error_cls=RuntimeError)

    with downloader:
        all_bar = tqdm(total=3, file=io.StringIO())
        large_file = tqdm(total=70, unit="B", file=io.StringIO())
        small_file = tqdm(total=30, unit="B", file=io.StringIO())

        # Their sizes must not determine their weight in the repository-level
        # progress. The repository bar's total is available before its first
        # update, so the downloader already knows that n == 3 here. The third
        # file has not started and therefore contributes zero.
        large_file.update(69)
        before_completion = downloader.get_progress()
        assert before_completion == pytest.approx((69 / 70) / 3)

        large_file.update(1)
        during_completion = downloader.get_progress()
        assert during_completion == pytest.approx(1 / 3)

        all_bar.update(1)
        after_completion = downloader.get_progress()
        assert after_completion == pytest.approx(1 / 3)
        assert after_completion == pytest.approx(during_completion)

        small_file.update(15)
        assert downloader.get_progress() == pytest.approx((1 + 15 / 30) / 3)


def test_tqdm_patch_preserves_completion_without_intermediate_poll():
    downloader = CancellableDownloader(cancel_error_cls=RuntimeError)

    with downloader:
        all_bar = tqdm(total=3, file=io.StringIO())
        file_bar = tqdm(total=70, unit="B", file=io.StringIO())

        # Complete the file before get_progress() has observed any partial
        # progress. Its contribution must remain visible until all_bar catches
        # up, rather than relying on a previously reported high-water mark.
        file_bar.update(70)
        assert downloader.get_progress() == pytest.approx(1 / 3)

        all_bar.update(1)
        assert downloader.get_progress() == pytest.approx(1 / 3)


def test_download_progress_details_include_completed_files():
    downloader = CancellableDownloader(cancel_error_cls=RuntimeError)

    with downloader:
        bar = tqdm(total=100, unit="B", desc="model.safetensors")
        bar.update(25)

        [downloading] = downloader.get_download_progress_details()
        assert downloading["name"] == "model.safetensors"
        assert downloading["downloaded_bytes"] == 25
        assert downloading["total_bytes"] == 100
        assert downloading["progress"] == 0.25
        assert downloading["status"] == "downloading"

        bar.update(75)

        [completed] = downloader.get_download_progress_details()
        assert completed["progress"] == 1.0
        assert completed["speed_bytes_per_second"] is None
        assert completed["eta_seconds"] == 0.0
        assert completed["status"] == "completed"
        bar.close()


async def test_concurrent_download_progress_details_are_isolated():
    """Each launch must only expose tqdm bars from its own task context."""
    d1 = CancellableDownloader(cancel_error_cls=RuntimeError)
    d2 = CancellableDownloader(cancel_error_cls=RuntimeError)
    barrier = threading.Barrier(2)

    async def collect_details(downloader, name):
        with downloader:

            def update_and_snapshot():
                bar = tqdm(total=100, unit="B", desc=name, file=io.StringIO())
                try:
                    bar.update(25)
                    barrier.wait(timeout=10)
                    return downloader.get_download_progress_details()
                finally:
                    bar.close()

            return await asyncio.to_thread(update_and_snapshot)

    details1, details2 = await asyncio.gather(
        collect_details(d1, "model-a.bin"),
        collect_details(d2, "model-b.bin"),
    )

    assert [item["name"] for item in details1] == ["model-a.bin"]
    assert [item["name"] for item in details2] == ["model-b.bin"]


def test_tqdm_owner_is_stable_across_threads():
    """A bar created for one downloader keeps that owner in a raw thread."""
    d1 = CancellableDownloader(cancel_error_cls=RuntimeError)
    d2 = CancellableDownloader(cancel_error_cls=RuntimeError)

    with d1:
        bar = tqdm(total=100, unit="B", desc="model-a.bin", file=io.StringIO())
        try:
            with d2:
                errors = []

                def update():
                    try:
                        bar.update(25)
                    except BaseException as error:
                        errors.append(error)

                thread = threading.Thread(target=update)
                thread.start()
                thread.join(timeout=10)

                assert not thread.is_alive()
                assert not errors
                assert [
                    item["name"] for item in d1.get_download_progress_details()
                ] == ["model-a.bin"]
                assert d2.get_download_progress_details() == []
        finally:
            bar.close()


def test_concurrent_progress_no_set_mutation():
    """Two concurrent downloaders race the progress sets: one thread creates
    new download bars and calls .update() (so patched_update grows
    _download_progresses), while another polls get_progress(). On main this
    raises "RuntimeError: Set changed size during iteration"; the per-instance
    _progress_lock + snapshot in get_progress makes it safe.

    Under CPython's default ~5ms switch interval the mutation rarely lands
    mid-iteration, so the crash is flaky on main (and can even pass 8/8). We
    tighten sys.setswitchinterval to force frequent GIL handoffs so the
    mutation coincides with the poller's iteration deterministically: red on
    main, green on the branch. The original interval is restored in finally.
    """
    d1 = CancellableDownloader(cancel_error_cls=RuntimeError)
    d2 = CancellableDownloader(cancel_error_cls=RuntimeError)
    errors = []

    orig_si = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    try:
        with d1, d2:
            stop = threading.Event()

            def updater():
                try:
                    # seed a few download bars first
                    for _ in range(5):
                        bar = tqdm(total=1000000, unit="B")
                        bar.update(1)
                    while not stop.is_set():
                        # a NEW bar each iteration -> add() grows the set size
                        bar = tqdm(total=1000000, unit="B")
                        bar.update(1)
                        bar.close()
                except RuntimeError as e:
                    errors.append(("updater", e))

            def poller():
                while not stop.is_set():
                    try:
                        d1.get_progress()
                        d2.get_progress()
                    except RuntimeError as e:
                        errors.append(("poller", e))
                        return

            # Propagate the logical downloader context into the raw thread so
            # this test still exercises concurrent set growth after tqdm bars
            # became owner-scoped.
            updater_context = copy_context()
            tu = threading.Thread(target=updater_context.run, args=(updater,))
            tp = threading.Thread(target=poller)
            tu.start()
            tp.start()
            # run the race for ~1s; a timeout + Event keeps CI from hanging
            tp.join(timeout=1.2)
            stop.set()
            tu.join(timeout=3.0)
            tp.join(timeout=3.0)
    finally:
        sys.setswitchinterval(orig_si)

    assert not errors, f"concurrent get_progress raised: {errors}"


def test_class_level_bookkeeping_no_per_instance_shadow():
    """qinxuye #5257 round-4 ask #1: _active_instances / _original_update /
    _original_update_plain must be routed through the class (type(self)), not
    ``self``. The old ``self._original_update = ...`` shadowed the class
    attribute, so two concurrent downloaders each saw the class-level None,
    patched tqdm independently, and the first to exit restored tqdm.update
    while the second was still active."""
    # Clean class state, independent of test ordering.
    CancellableDownloader._active_instances = 0
    CancellableDownloader._original_update = None
    CancellableDownloader._original_update_plain = None
    CancellableDownloader._original_init_plain = None
    CancellableDownloader._active_registry.clear()
    original_update = tqdm.update
    original_init_plain = tqdm_module.tqdm.__dict__["__init__"]

    d1 = CancellableDownloader(cancel_error_cls=RuntimeError)
    d2 = CancellableDownloader(cancel_error_cls=RuntimeError)
    d1.__enter__()
    d2.__enter__()
    try:
        # Shared class-level counter, no per-instance shadow.
        assert CancellableDownloader._active_instances == 2
        assert "_active_instances" not in vars(d1)
        assert "_active_instances" not in vars(d2)
        # Originals stored on the class, no per-instance shadow.
        assert CancellableDownloader._original_update is original_update
        assert CancellableDownloader._original_update_plain is not None
        assert CancellableDownloader._original_init_plain[0] is original_init_plain
        for d in (d1, d2):
            assert "_original_update" not in vars(d)
            assert "_original_update_plain" not in vars(d)
            assert "_original_init_plain" not in vars(d)
        # tqdm stays patched while any instance is active.
        assert tqdm.update is not original_update

        # d1 exits while d2 is still active: must NOT restore tqdm yet (the
        # concrete reproduction: "first downloader exiting while the second
        # remained active: tqdm.update was restored too early").
        d1.__exit__(None, None, None)
        assert CancellableDownloader._active_instances == 1
        assert (
            tqdm.update is not original_update
        ), "tqdm.update was restored while a downloader is still active"
    finally:
        if CancellableDownloader._active_instances > 0:
            d2.__exit__(None, None, None)
    # last instance out -> counter 0 and tqdm restored
    assert CancellableDownloader._active_instances == 0
    assert tqdm.update is original_update
    assert tqdm_module.tqdm.__dict__["__init__"] is original_init_plain


def test_reset_holds_lock_against_progress_poller():
    """qinxuye #5257 round-4 ask #2: reset() must take _progress_lock around
    both clears. During __exit__ the progress-upload thread can already have
    passed its done check and entered get_progress() while cleanup calls
    reset(); the unlocked clear raced the set iteration and raised
    "Set changed size during iteration"."""
    CancellableDownloader._active_instances = 0
    CancellableDownloader._original_update = None
    CancellableDownloader._original_update_plain = None
    CancellableDownloader._original_init_plain = None
    CancellableDownloader._active_registry.clear()
    d = CancellableDownloader(cancel_error_cls=RuntimeError)
    with d:
        for _ in range(20):
            bar = tqdm(total=1000000, unit="B")
            bar.update(1)
        errors = []
        orig_si = sys.getswitchinterval()
        sys.setswitchinterval(1e-4)
        try:
            stop = threading.Event()

            def poller():
                while not stop.is_set():
                    try:
                        d.get_progress()
                    except RuntimeError as e:
                        errors.append(e)
                        return

            def resetter():
                while not stop.is_set():
                    try:
                        bar = tqdm(total=1000000, unit="B")
                        bar.update(1)  # patched_update grows the set under the lock
                        d.reset()  # must clear under the same lock
                    except RuntimeError as e:
                        errors.append(e)
                        return

            tp = threading.Thread(target=poller)
            tr = threading.Thread(target=resetter)
            tp.start()
            tr.start()
            tp.join(timeout=1.2)
            stop.set()
            tr.join(timeout=3.0)
            tp.join(timeout=3.0)
        finally:
            sys.setswitchinterval(orig_si)

    assert not errors, f"reset/get_progress race raised: {errors}"


def test_active_registry_fallback_holds_global_lock():
    """The no-context, single-downloader fallback must hold the registry lock."""

    class LockCheckingSet(set):
        def __iter__(self):
            assert CancellableDownloader._global_lock.locked()
            return super().__iter__()

    original_registry = CancellableDownloader._active_registry
    CancellableDownloader._active_instances = 0
    CancellableDownloader._original_update = None
    CancellableDownloader._original_update_plain = None
    CancellableDownloader._original_init_plain = None
    CancellableDownloader._active_registry = LockCheckingSet()

    downloader = CancellableDownloader(cancel_error_cls=RuntimeError)
    try:
        with downloader:
            errors = []

            def update():
                try:
                    bar = tqdm(total=1, disable=True)
                    bar.update(1)
                    bar.close()
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=update)
            thread.start()
            thread.join(timeout=10)

            assert not thread.is_alive()
            assert not errors
    finally:
        CancellableDownloader._active_registry = original_registry


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


@pytest.fixture
def _reset_auto_hub_cache(monkeypatch):
    import xinference.model.utils as model_utils

    monkeypatch.setattr(model_utils, "_auto_detected_hub", None)
    monkeypatch.delenv("XINFERENCE_MODEL_SRC", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    for proxy_var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(proxy_var, raising=False)
    # urllib may also discover OS-level proxy settings (notably on macOS), so
    # isolate ordinary tests from the host network configuration.
    monkeypatch.setenv("NO_PROXY", "*")
    yield
    model_utils._auto_detected_hub = None


def test_auto_detect_download_hub_hf_reachable(monkeypatch, _reset_auto_hub_cache):
    import xinference.model.utils as model_utils

    monkeypatch.setattr(
        model_utils, "_is_hub_endpoint_reachable", lambda url, timeout: True
    )
    assert model_utils.auto_detect_download_hub() == "huggingface"


def test_auto_detect_download_hub_hf_unreachable(monkeypatch, _reset_auto_hub_cache):
    import xinference.model.utils as model_utils

    monkeypatch.setattr(
        model_utils, "_is_hub_endpoint_reachable", lambda url, timeout: False
    )
    assert model_utils.auto_detect_download_hub() == "modelscope"


def test_auto_detect_download_hub_uses_hf_endpoint(monkeypatch, _reset_auto_hub_cache):
    import xinference.model.utils as model_utils

    calls = []

    def probe(url, timeout):
        calls.append((url, timeout))
        return True

    endpoint = "https://hf-mirror.example.com"
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
    monkeypatch.setenv("NO_PROXY", "hf-mirror.example.com")
    monkeypatch.setattr(model_utils, "_is_hub_endpoint_reachable", probe)

    assert model_utils.auto_detect_download_hub() == "huggingface"
    assert calls == [(endpoint, model_utils.XINFERENCE_HUB_DETECT_TIMEOUT)]


def test_auto_detect_download_hub_avoids_proxy_for_hf_endpoint(
    monkeypatch, _reset_auto_hub_cache
):
    import xinference.model.utils as model_utils

    calls = {"n": 0}

    def probe(url, timeout):
        calls["n"] += 1
        return True

    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example.com")
    monkeypatch.delenv("NO_PROXY")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
    monkeypatch.setattr(model_utils, "_is_hub_endpoint_reachable", probe)

    assert model_utils.auto_detect_download_hub() == "modelscope"
    assert calls["n"] == 0


def test_auto_detect_download_hub_result_is_cached(monkeypatch, _reset_auto_hub_cache):
    import xinference.model.utils as model_utils

    calls = {"n": 0}

    def probe(url, timeout):
        calls["n"] += 1
        return True

    monkeypatch.setattr(model_utils, "_is_hub_endpoint_reachable", probe)
    assert model_utils.auto_detect_download_hub() == "huggingface"
    assert model_utils.auto_detect_download_hub() == "huggingface"
    assert calls["n"] == 1


def test_resolve_download_hub(monkeypatch, _reset_auto_hub_cache):
    import xinference.model.utils as model_utils

    monkeypatch.setattr(
        model_utils, "_is_hub_endpoint_reachable", lambda url, timeout: False
    )

    # explicit hubs are passed through untouched
    assert model_utils.resolve_download_hub("huggingface") == "huggingface"
    assert model_utils.resolve_download_hub("modelscope") == "modelscope"
    assert model_utils.resolve_download_hub("csghub") == "csghub"

    # "auto" always resolves via detection
    assert model_utils.resolve_download_hub("auto") == "modelscope"

    # unspecified hub goes through detection as well
    assert model_utils.resolve_download_hub(None) == "modelscope"

    # a local model path means no download, so no detection
    assert model_utils.resolve_download_hub(None, "/path/to/model") is None

    # a pinned XINFERENCE_MODEL_SRC resolves to that concrete hub
    monkeypatch.setenv("XINFERENCE_MODEL_SRC", "modelscope")
    assert model_utils.resolve_download_hub(None) == "modelscope"

    # XINFERENCE_MODEL_SRC="auto" resolves via detection
    monkeypatch.setenv("XINFERENCE_MODEL_SRC", "auto")
    assert model_utils.resolve_download_hub(None) == "modelscope"


def test_explicit_huggingface_bypasses_auto_proxy_avoidance(
    monkeypatch, _reset_auto_hub_cache
):
    import xinference.model.utils as model_utils

    monkeypatch.delenv("NO_PROXY")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
    assert model_utils._uses_environment_proxy("https://huggingface.co")

    # A per-launch selection is the most specific setting.
    assert model_utils.resolve_download_hub("huggingface") == "huggingface"

    # A service-level pin applies when the launch does not specify a hub.
    monkeypatch.setenv("XINFERENCE_MODEL_SRC", "huggingface")
    assert model_utils.resolve_download_hub(None) == "huggingface"

    # The per-launch setting also overrides a different service-level pin.
    monkeypatch.setenv("XINFERENCE_MODEL_SRC", "modelscope")
    assert model_utils.resolve_download_hub("huggingface") == "huggingface"


def test_download_from_modelscope_env_auto(monkeypatch, _reset_auto_hub_cache):
    import xinference.model.utils as model_utils

    monkeypatch.setenv("XINFERENCE_MODEL_SRC", "auto")
    monkeypatch.setattr(
        model_utils, "_is_hub_endpoint_reachable", lambda url, timeout: False
    )
    assert model_utils.download_from_modelscope() is True

    model_utils._auto_detected_hub = None
    monkeypatch.setattr(
        model_utils, "_is_hub_endpoint_reachable", lambda url, timeout: True
    )
    assert model_utils.download_from_modelscope() is False


def test_probe_bypasses_proxies_and_rejects_http_errors(
    monkeypatch, _reset_auto_hub_cache
):
    from types import SimpleNamespace

    import requests

    import xinference.model.utils as model_utils

    ca_bundle = "/private/corporate-ca.pem"
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", ca_bundle)

    def _head(status):
        def head(session, url, timeout=None, allow_redirects=None, proxies=None):
            assert session.trust_env is True
            assert proxies == {"http": None, "https": None, "all": None}
            settings = session.merge_environment_settings(
                url, proxies, stream=None, verify=None, cert=None
            )
            assert requests.utils.select_proxy(url, settings["proxies"]) is None
            assert settings["verify"] == ca_bundle
            return SimpleNamespace(status_code=status)

        return head

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.com:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.example.com:1080")
    monkeypatch.setattr(requests.Session, "head", _head(200))
    assert model_utils._is_hub_endpoint_reachable("https://huggingface.co", 1.0)

    # An error response means downloads would fail, so it must count as
    # unreachable.
    for status in (403, 407, 500, 503):
        monkeypatch.setattr(requests.Session, "head", _head(status))
        assert not model_utils._is_hub_endpoint_reachable("https://huggingface.co", 1.0)

    def head_raise(session, url, timeout=None, allow_redirects=None, proxies=None):
        assert session.trust_env is True
        assert proxies == {"http": None, "https": None, "all": None}
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests.Session, "head", head_raise)
    assert not model_utils._is_hub_endpoint_reachable("https://huggingface.co", 1.0)


def test_explicit_download_hub_overrides_model_src_env(
    monkeypatch, _reset_auto_hub_cache
):
    from ..embedding.embed_family import match_embedding
    from ..image.core import match_diffusion
    from ..rerank.rerank_family import match_rerank
    from ..video.core import match_diffusion as match_video_diffusion

    monkeypatch.setenv("XINFERENCE_MODEL_SRC", "modelscope")

    # an explicit hub must win over the XINFERENCE_MODEL_SRC fallback
    assert (
        match_diffusion("FLUX.1-schnell", download_hub="huggingface").model_hub
        == "huggingface"
    )
    assert (
        match_video_diffusion("CogVideoX-2b", download_hub="huggingface").model_hub
        == "huggingface"
    )
    assert (
        match_rerank("bge-reranker-large", download_hub="huggingface")
        .model_specs[0]
        .model_hub
        == "huggingface"
    )
    assert (
        match_embedding("bge-large-en", download_hub="huggingface")
        .model_specs[0]
        .model_hub
        == "huggingface"
    )

    # without an explicit hub, the env fallback still applies
    assert match_diffusion("FLUX.1-schnell").model_hub == "modelscope"
    assert match_video_diffusion("CogVideoX-2b").model_hub == "modelscope"
    assert match_rerank("bge-reranker-large").model_specs[0].model_hub == "modelscope"
    assert match_embedding("bge-large-en").model_specs[0].model_hub == "modelscope"


@pytest.mark.parametrize("offline_var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_auto_detect_honors_hf_offline_mode(
    monkeypatch, _reset_auto_hub_cache, offline_var
):
    import xinference.model.utils as model_utils

    calls = {"n": 0}

    def probe(url, timeout):
        calls["n"] += 1
        return False

    monkeypatch.setattr(model_utils, "_is_hub_endpoint_reachable", probe)
    monkeypatch.setenv(offline_var, "1")

    # Offline deployments read weights from a pre-populated local Hugging
    # Face cache: detection must pick huggingface without probing the
    # network instead of falling back to modelscope.
    assert model_utils.auto_detect_download_hub() == "huggingface"
    assert calls["n"] == 0
    assert model_utils.resolve_download_hub(None) == "huggingface"
    assert model_utils.resolve_download_hub("auto") == "huggingface"
