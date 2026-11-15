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
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ...cache_manager import LLMCacheManager
from ...llm_family import check_engine_by_spec_parameters_with_virtual_env, match_llm
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


def test_release_common_params(monkeypatch, tmp_path):
    runtime = pytest.importorskip("xllamacpp")
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
