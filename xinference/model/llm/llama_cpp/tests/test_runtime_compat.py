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

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ...._llamacpp import configure_gpu_layers


@pytest.fixture(params=["legacy", "native"])
def runtime(request, monkeypatch):
    def common_params():
        return SimpleNamespace(
            model=SimpleNamespace(path=""),
            mmproj=SimpleNamespace(path=""),
            n_gpu_layers=-1,
            n_ctx=2048,
            n_batch=512,
            n_parallel=1,
            tensor_split=[0.0, 0.0],
            fit_params=True,
            fit_params_target=[1024**3],
            cpuparams=SimpleNamespace(n_threads=1),
            cpuparams_batch=SimpleNamespace(n_threads=1),
        )

    module = SimpleNamespace(
        CommonParams=common_params,
        Server=MagicMock(),
        # Include a development version: capability detection must work before
        # the new release is tagged, too.
        __version__="2026.10.11870+7.g90b9543",
        llama_pooling_type=SimpleNamespace(
            LLAMA_POOLING_TYPE_LAST=3,
            LLAMA_POOLING_TYPE_RANK=4,
            LLAMA_POOLING_TYPE_MEAN=1,
        ),
        ggml_type=SimpleNamespace(GGML_TYPE_F32=0, GGML_TYPE_BF16=30),
    )
    if request.param == "legacy":
        module.__version__ = "0.2.6"
        module.estimate_gpu_layers = MagicMock(
            return_value=SimpleNamespace(layers=7, tensor_split=[])
        )
        module.get_device_info = MagicMock(return_value=[{"type": 1}])
        module.ggml_backend_dev_type = SimpleNamespace(GGML_BACKEND_DEVICE_TYPE_GPU=1)
    monkeypatch.setitem(sys.modules, "xllamacpp", module)
    return module


@pytest.fixture(params=["llm", "embedding", "embeddinggemma2", "rerank"])
def model(request, tmp_path):
    spec = SimpleNamespace(
        model_file_name_template="model-{quantization}.gguf",
        model_file_name_split_template=None,
        quantization_parts=None,
        quantization="Q8_0",
        multimodal_projector=None,
    )
    config = {"n_ctx": 2048, "n_threads": 1, "n_gpu_layers": -1}
    if request.param == "llm":
        from ..core import XllamaCppModel

        result = object.__new__(XllamaCppModel)
        result.model_uid = "test-model"
        result.model_path = str(tmp_path)
        result.model_spec = spec
        result.model_family = SimpleNamespace(model_name="test", chat_template=None)
        result.prepare_parse_reasoning_content = MagicMock()
        result.prepare_parse_tool_calls = MagicMock()
        config["reasoning_content"] = False
    else:
        if request.param == "embedding":
            from ....embedding.llama_cpp.core import XllamaCppEmbeddingModel

            cls = XllamaCppEmbeddingModel
        elif request.param == "embeddinggemma2":
            from ....embedding.llama_cpp.embeddinggemma2 import (
                XllamaCppEmbeddingGemma2Model,
            )

            cls = XllamaCppEmbeddingGemma2Model
        else:
            from ....rerank.llama_cpp.core import XllamaCppRerankModel

            cls = XllamaCppRerankModel
        result = object.__new__(cls)
        result._model_path = str(tmp_path)
        result._model_spec = spec
        result._quantization = "Q8_0"
        result._model_name = "test"
        result._kwargs = {}
        result.model_family = SimpleNamespace(max_tokens=8192)
    result._llamacpp_model_config = config
    result._executor = None
    try:
        yield result
    finally:
        if result._executor is not None:
            result._executor.shutdown()


@pytest.mark.parametrize("layers", [-1, 0, 5, -2])
def test_load_with_legacy_and_native_placement(runtime, model, layers):
    model._llamacpp_model_config["n_gpu_layers"] = layers
    model.load()

    params = runtime.Server.call_args.args[0]
    expected = 7 if layers == -1 and hasattr(runtime, "estimate_gpu_layers") else layers
    assert params.n_gpu_layers == expected
    assert params.fit_params is True
    assert params.tensor_split == [0.0, 0.0]
    assert params.fit_params_target == [1024**3]
    if hasattr(runtime, "estimate_gpu_layers"):
        assert runtime.estimate_gpu_layers.call_count == (1 if layers == -1 else 0)


def test_native_load_preserves_explicit_fit_settings(runtime, model):
    if hasattr(runtime, "estimate_gpu_layers"):
        pytest.skip("Native fitting configuration")
    model._llamacpp_model_config.update(
        tensor_split=[0.25, 0.75], fit_params=False, fit_params_target=[2 * 1024**3]
    )
    model.load()

    params = runtime.Server.call_args.args[0]
    assert params.n_gpu_layers == -1
    assert params.tensor_split == [0.25, 0.75]
    assert params.fit_params is False
    assert params.fit_params_target == [2 * 1024**3]


def test_startup_oom_preserves_native_diagnostics(runtime, model):
    error = MemoryError("GPU allocation failed")
    error.memory_estimate = {"required_bytes": 1024**3}
    runtime.Server.side_effect = error

    with pytest.raises(MemoryError) as exc:
        model.load()

    assert exc.value is error
    assert exc.value.memory_estimate == {"required_bytes": 1024**3}


@pytest.mark.parametrize("outcome", ["layers", "split", "no_gpu", "error"])
def test_legacy_estimator_behavior(runtime, outcome):
    if not hasattr(runtime, "estimate_gpu_layers"):
        pytest.skip("Legacy estimator configuration")
    params = runtime.CommonParams()
    if outcome == "split":
        runtime.estimate_gpu_layers.return_value.tensor_split = [0.4, 0.6]
    elif outcome == "no_gpu":
        runtime.get_device_info.return_value = [{"type": 0}]
    elif outcome == "error":
        runtime.estimate_gpu_layers.side_effect = ValueError("cannot estimate")

    configure_gpu_layers(params, "/model.gguf", ["/projector.gguf"])

    assert params.n_gpu_layers == (7 if outcome == "layers" else 0x7FFFFFFF)
    assert params.tensor_split == ([0.4, 0.6] if outcome == "split" else [0.0, 0.0])
    if outcome == "no_gpu":
        runtime.estimate_gpu_layers.assert_not_called()
    else:
        runtime.estimate_gpu_layers.assert_called_once_with(
            gpus=[{"type": 1}],
            model_path="/model.gguf",
            projectors=["/projector.gguf"],
            context_length=2048,
            batch_size=512,
            num_parallel=1,
            kv_cache_type="",
        )


def test_load_with_installed_common_params(model, monkeypatch):
    xllamacpp = pytest.importorskip("xllamacpp")
    server = MagicMock()
    monkeypatch.setattr(xllamacpp, "Server", server)
    if hasattr(xllamacpp, "estimate_gpu_layers"):
        monkeypatch.setattr(xllamacpp, "get_device_info", lambda: [])
    model.load()

    params = server.call_args.args[0]
    expected = 0x7FFFFFFF if hasattr(xllamacpp, "estimate_gpu_layers") else -1
    assert params.n_gpu_layers == expected
