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

# These quantizations are uploaded as two shards in a folder (same names on
# Hugging Face and ModelScope), not as one file.
CASES = [
    (
        "deepseek-r1-distill-qwen",
        "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "F16",
        [
            "DeepSeek-R1-Distill-Qwen-32B-F16/DeepSeek-R1-Distill-Qwen-32B-F16-00001-of-00002.gguf",
            "DeepSeek-R1-Distill-Qwen-32B-F16/DeepSeek-R1-Distill-Qwen-32B-F16-00002-of-00002.gguf",
        ],
    ),
    (
        "gemma-3-it",
        "bartowski/google_gemma-3-27b-it-GGUF",
        "bf16",
        [
            "google_gemma-3-27b-it-bf16/google_gemma-3-27b-it-bf16-00001-of-00002.gguf",
            "google_gemma-3-27b-it-bf16/google_gemma-3-27b-it-bf16-00002-of-00002.gguf",
        ],
    ),
]


@pytest.mark.parametrize("model_name, model_id, quantization, files", CASES)
def test_split_gguf_quantization_resolves_to_its_shards(
    model_name, model_id, quantization, files
):
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    specs = [
        s
        for s in get_llm_family_by_name(model_name).model_specs
        if s.model_format == "ggufv2"
        and s.model_id == model_id
        and s.quantization == quantization
    ]
    assert len(specs) == 2  # Hugging Face and ModelScope
    for spec in specs:
        names, _, need_merge = generate_model_file_names_with_quantization_parts(spec)
        assert (names, need_merge) == (files, True)


SPLIT_LAUNCHES = [
    ("deepseek-r1-distill-qwen", 32, "F16"),
    ("gemma-3-it", 27, "bf16"),
]


def _launch(monkeypatch, engine_cls, engine, model_name, size, quantization, **kwargs):
    from .. import llm_family
    from ..core import create_llm_model_instance

    monkeypatch.setattr(
        llm_family, "check_engine_by_spec_parameters", lambda *args: engine_cls
    )
    return create_llm_model_instance(
        "test-0",
        model_name,
        engine,
        model_format="ggufv2",
        model_size_in_billions=size,
        quantization=quantization,
        enable_virtual_env=False,
        **kwargs,
    )


def _fail_on_download(monkeypatch):
    from ..cache_manager import LLMCacheManager

    def cache(self):
        pytest.fail("the shards must not be downloaded")

    monkeypatch.setattr(LLMCacheManager, "cache", cache)


@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
@pytest.mark.parametrize("model_name, size, quantization", SPLIT_LAUNCHES)
def test_vllm_rejects_split_gguf_before_downloading(
    monkeypatch, model_name, size, quantization, hub
):
    from ..vllm.core import VLLMChatModel

    _fail_on_download(monkeypatch)
    with pytest.raises(ValueError, match="split GGUF.*`model_path`"):
        _launch(
            monkeypatch,
            VLLMChatModel,
            "vLLM",
            model_name,
            size,
            quantization,
            download_hub=hub,
        )


@pytest.mark.parametrize("model_name, size, quantization", SPLIT_LAUNCHES)
def test_vllm_still_accepts_an_explicit_merged_gguf(
    monkeypatch, model_name, size, quantization
):
    from ..vllm.core import VLLMChatModel

    class Engine(VLLMChatModel):
        def __init__(self, uid, family, path, config):
            self.path = path

    _fail_on_download(monkeypatch)
    model = _launch(
        monkeypatch,
        Engine,
        "vLLM",
        model_name,
        size,
        quantization,
        model_path="/merged.gguf",
    )
    assert model.path == "/merged.gguf"


@pytest.mark.parametrize("model_name, size, quantization", SPLIT_LAUNCHES)
def test_llama_cpp_still_downloads_split_gguf(
    monkeypatch, model_name, size, quantization
):
    from ..cache_manager import LLMCacheManager
    from ..llama_cpp.core import XllamaCppModel

    class Engine(XllamaCppModel):
        def __init__(self, uid, family, path, config):
            self.path = path

    monkeypatch.setattr(LLMCacheManager, "cache", lambda self: "/cached")
    model = _launch(monkeypatch, Engine, "llama.cpp", model_name, size, quantization)
    assert model.path == "/cached"
