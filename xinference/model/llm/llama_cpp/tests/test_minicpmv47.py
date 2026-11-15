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
import asyncio
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from ..... import constants
from ....utils import (
    get_engine_params_by_name,
    get_engine_params_by_name_with_virtual_env,
    virtualenv_discovery_var,
)
from ... import generate_engine_config_by_model_family
from ...cache_manager import LLMCacheManager
from ...core import create_llm_model_instance
from ...llm_family import (
    LLM_ENGINES,
    check_engine_by_spec_parameters_with_virtual_env,
    match_llm,
)
from .. import core
from ..core import XllamaCppModel

QUANTIZATIONS = [
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
    "F16",
    "IQ3_M",
    "IQ3_XS",
    "IQ4_NL",
    "IQ4_XS",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
]
PROJECTORS = ["mmproj-model-f16.gguf", "mmproj-model-f16-4x.gguf"]


@pytest.mark.parametrize(
    "hub,owner,revision",
    [
        ("huggingface", "openbmb", "main"),
        ("modelscope", "OpenBMB", "master"),
    ],
)
@pytest.mark.parametrize("size,label", [(1, "1B"), (35, "35B-A3B")])
def test_official_gguf_engine_and_cache(monkeypatch, hub, owner, revision, size, label):
    monkeypatch.setattr(XllamaCppModel, "check_lib", classmethod(lambda cls: True))
    quantizations = QUANTIZATIONS + (["mix_q3gud_im"] if size == 35 else [])
    for quantization in quantizations:
        family = match_llm("MiniCPM-V-4.7", "ggufv2", size, quantization, hub)
        spec = family.model_specs[0]
        filename = f"MiniCPM-V-4_7-{label}-{quantization}.gguf"
        assert spec.model_id == f"{owner}/MiniCPM-V-4.7-{label}-gguf"
        assert spec.model_revision == revision
        assert spec.activated_size_in_billions == (3 if size == 35 else None)
        assert (
            spec.model_file_name_template.format(quantization=quantization) == filename
        )
        assert spec.multimodal_projectors == PROJECTORS
        assert (
            check_engine_by_spec_parameters_with_virtual_env(
                "llama.cpp",
                family.model_name,
                "ggufv2",
                size,
                quantization,
                llm_family=family,
            )
            is XllamaCppModel
        )
        for projector in PROJECTORS:
            cache = LLMCacheManager(family, projector)
            assert set(cache._source_files(downloading=True)) == {filename, projector}


@pytest.fixture
def native_runtime(monkeypatch):
    runtime = SimpleNamespace(
        __version__="2026.10.11870",
        CommonParams=lambda: SimpleNamespace(
            model="",
            mmproj=SimpleNamespace(path=""),
            cpuparams=SimpleNamespace(n_threads=1),
            cpuparams_batch=SimpleNamespace(n_threads=1),
        ),
        Server=Mock(),
    )
    monkeypatch.setitem(sys.modules, "xllamacpp", runtime)
    monkeypatch.setattr(core, "configure_gpu_layers", Mock())
    # Keep cached engine entries from a compatible runtime to test revalidation.
    engines = {}
    for name, size in [
        ("MiniCPM-V-4.7", 1),
        ("MiniCPM-V-4.7", 35),
        ("MiniCPM-V-4.6", 1),
    ]:
        family = match_llm(name, "ggufv2", size, "F16", "huggingface")
        generate_engine_config_by_model_family(family, target_engines=engines)
    for name, params in engines.items():
        monkeypatch.setitem(LLM_ENGINES, name, params)
    return runtime


def make_model(tmp_path, size=1, projector=PROJECTORS[0], single_file=False):
    family = match_llm("MiniCPM-V-4.7", "ggufv2", size, "Q4_K_M", "huggingface")
    filename = family.model_specs[0].model_file_name_template.format(
        quantization="Q4_K_M"
    )
    model_file = tmp_path / filename
    model_file.touch()
    (tmp_path / projector).touch()
    return XllamaCppModel(
        "minicpm-llamacpp-test",
        family,
        str(model_file if single_file else tmp_path),
        {
            "multimodal_projector": projector,
            "n_ctx": 1024,
            "n_parallel": 1,
            "n_threads": 1,
            "n_gpu_layers": 0,
            "enable_thinking": False,
        },
    )


@pytest.mark.parametrize("version", ["0.2.6", "2026.9.11063", "invalid"])
def test_old_runtime_rejected_before_loading(native_runtime, tmp_path, version):
    native_runtime.__version__ = version
    model = make_model(tmp_path)
    with pytest.raises(RuntimeError, match="2026.10.11870"):
        model.load()
    native_runtime.Server.assert_not_called()
    assert model._executor is None


@pytest.mark.parametrize(
    "discover", [get_engine_params_by_name, get_engine_params_by_name_with_virtual_env]
)
@pytest.mark.parametrize("virtualenv", [False, True])
@pytest.mark.parametrize("version", ["2026.7.10068", "2026.10.11870", "invalid"])
def test_runtime_version_discovery(
    native_runtime, monkeypatch, discover, virtualenv, version
):
    native_runtime.__version__ = version
    monkeypatch.setattr(constants, "XINFERENCE_ENABLE_VIRTUAL_ENV", not virtualenv)
    before = virtualenv_discovery_var.get()
    params = discover("LLM", "MiniCPM-V-4.7", enable_virtual_env=virtualenv)
    accepted = virtualenv or version == "2026.10.11870"
    assert isinstance(params.get("llama.cpp"), list) is accepted
    if accepted:
        assert {
            int(entry["model_size_in_billions"]) for entry in params["llama.cpp"]
        } == {1, 35}
    else:
        assert "2026.10.11870" in params["llama.cpp"]
    assert virtualenv_discovery_var.get() is before


@pytest.mark.parametrize("virtualenv", [False, True])
@pytest.mark.parametrize("version", ["2026.7.10068", "2026.10.11870", "invalid"])
@pytest.mark.parametrize("size", [1, 35])
@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
def test_factory_checks_runtime_before_caching(
    native_runtime, monkeypatch, virtualenv, version, size, hub
):
    native_runtime.__version__ = version
    monkeypatch.setattr(constants, "XINFERENCE_ENABLE_VIRTUAL_ENV", not virtualenv)
    cache = Mock(return_value="/models/test")
    monkeypatch.setattr(LLMCacheManager, "cache", cache)
    before = virtualenv_discovery_var.get()

    def launch():
        return create_llm_model_instance(
            "minicpm-preflight",
            "MiniCPM-V-4.7",
            "llama.cpp",
            "ggufv2",
            size,
            "F16",
            download_hub=hub,
            enable_virtual_env=virtualenv,
        )

    if virtualenv or version == "2026.10.11870":
        assert isinstance(launch(), XllamaCppModel)
        cache.assert_called_once_with()
    else:
        with pytest.raises(ValueError, match="2026.10.11870"):
            launch()
        cache.assert_not_called()
    assert virtualenv_discovery_var.get() is before


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1, 35])
@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
@pytest.mark.parametrize("cached_engine", [False, True])
async def test_worker_cache_only_allows_older_runtime(
    native_runtime, monkeypatch, size, hub, cached_engine
):
    from .....core.worker import WorkerActor

    native_runtime.__version__ = "2026.7.10068"
    monkeypatch.setattr(constants, "XINFERENCE_ENABLE_VIRTUAL_ENV", False)
    if not cached_engine:
        monkeypatch.setitem(LLM_ENGINES, "MiniCPM-V-4.7", {})
    cache = Mock(return_value="/models/test")
    monkeypatch.setattr(LLMCacheManager, "cache", cache)
    worker = SimpleNamespace(
        address="worker:1234",
        _cache_uid_to_download_info={},
        _download_artifact_cleanup_lock=asyncio.Lock(),
        _launch_semaphore=asyncio.Semaphore(1),
        _check_model_is_valid=Mock(),
        _get_progressor=AsyncMock(return_value=MagicMock()),
        _upload_download_progress=Mock(),
        update_cache_status=AsyncMock(),
        _allocate_subpool_devices=AsyncMock(),
        _spawn_subpool=AsyncMock(),
    )
    worker._download_model_files = WorkerActor._download_model_files.__get__(worker)
    worker.cache_builtin_model = WorkerActor.cache_builtin_model.__get__(worker)

    result = await worker.cache_builtin_model(
        cache_uid="minicpm-cache-only",
        model_name="MiniCPM-V-4.7",
        model_size_in_billions=size,
        model_format="ggufv2",
        quantization="F16",
        model_engine="llama.cpp",
        download_hub=hub,
        enable_virtual_env=False,
        multimodal_projector=PROJECTORS[0],
    )

    assert result["cache_uid"] == "minicpm-cache-only"
    cache.assert_called_once_with()
    worker.update_cache_status.assert_awaited_once()
    worker._allocate_subpool_devices.assert_not_awaited()
    worker._spawn_subpool.assert_not_awaited()
    native_runtime.Server.assert_not_called()
    assert not worker._cache_uid_to_download_info


def test_old_runtime_still_discovers_older_models(native_runtime):
    native_runtime.__version__ = "2026.7.10068"
    params = get_engine_params_by_name("LLM", "MiniCPM-V-4.6", enable_virtual_env=False)
    assert isinstance(params.get("llama.cpp"), list)


def test_release_common_params(monkeypatch, tmp_path):
    runtime = pytest.importorskip(
        "xllamacpp", minversion=core.MINICPMV47_MIN_XLLAMACPP_VERSION
    )
    server = Mock()
    monkeypatch.setattr(runtime, "Server", server)
    model = make_model(tmp_path)
    model.load()
    try:
        params = server.call_args.args[0]
        assert params.model.path.endswith("MiniCPM-V-4_7-1B-Q4_K_M.gguf")
        assert params.mmproj.path == str(tmp_path / PROJECTORS[0])
        assert params.n_ctx == 1024
        assert params.n_gpu_layers == 0
    finally:
        model._executor.shutdown()


@pytest.mark.parametrize("size", [1, 35])
@pytest.mark.parametrize("projector", PROJECTORS)
@pytest.mark.parametrize("single_file", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_load_and_image_chat(
    native_runtime, tmp_path, size, projector, single_file, stream
):
    model = make_model(tmp_path, size, projector, single_file)
    model.load()
    try:
        params = native_runtime.Server.call_args.args[0]
        assert params.mmproj.path == str(tmp_path / projector)
        assert params.use_jinja is True
        assert params.n_ctx == 1024
        # The official GGUF supplies its own chat template.
        assert not hasattr(params, "chat_template")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        original = deepcopy(messages)
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk" if stream else "chat.completion",
            "created": 0,
            "model": model.model_uid,
            "choices": [
                {
                    "index": 0,
                    "delta" if stream else "message": {
                        "role": "assistant",
                        "content": "A test image.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
        native_runtime.Server.return_value.handle_chat_completions.side_effect = (
            lambda request, callback: callback(response)
        )
        result = model.chat(messages, {"stream": stream, "max_tokens": 8})
        if stream:
            chunks = list(result)
            assert any(
                chunk["choices"][0]["delta"].get("content") == "A test image."
                for chunk in chunks
                if chunk.get("choices")
            )
        else:
            assert result["choices"][0]["message"]["content"] == "A test image."
        request = (
            native_runtime.Server.return_value.handle_chat_completions.call_args.args[0]
        )
        assert request["messages"] == original
        assert messages == original
        assert request["stream"] is stream
    finally:
        model._executor.shutdown()
