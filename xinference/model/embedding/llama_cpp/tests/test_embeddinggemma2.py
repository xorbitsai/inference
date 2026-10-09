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

import copy
import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from xinference.model.utils import virtualenv_discovery_var

from ... import generate_engine_config_by_model_name
from ...cache_manager import EmbeddingCacheManager
from ...embed_family import match_embedding
from ..core import XllamaCppEmbeddingModel
from ..embeddinggemma2 import XllamaCppEmbeddingGemma2Model, prepare_inputs


class _InlineExecutor:
    def submit(self, fn):
        fn()


def _make_model(**kwargs):
    family = match_embedding("embeddinggemma-2", "ggufv2", "UD-Q4_K_XL", "huggingface")
    model = XllamaCppEmbeddingGemma2Model(
        "gemma", "/unused", family, quantization="UD-Q4_K_XL", **kwargs
    )
    model._llm = MagicMock()
    model._executor = _InlineExecutor()
    model._llm.handle_embeddings.return_value = {
        "object": "list",
        "model": "gemma",
        "data": [
            {"index": 0, "object": "embedding", "embedding": [3.0, 4.0] + [1.0] * 766}
        ],
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }
    return model


@pytest.mark.parametrize(
    "hub,revision", [("huggingface", "main"), ("modelscope", "master")]
)
@pytest.mark.parametrize(
    "quantization", ["BF16", "F16", "Q8_0", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL"]
)
def test_catalog_downloads_backbone_and_projector(hub, revision, quantization):
    family = match_embedding("embeddinggemma-2", "ggufv2", quantization, hub)
    spec = family.model_specs[0]
    assert spec.model_id == "unsloth/embeddinggemma-2-GGUF"
    assert spec.model_hub == hub
    assert spec.model_revision == revision
    cache = EmbeddingCacheManager(family)
    files, target, merge = cache.cache_helper._gguf_file_names()
    assert files == [f"embeddinggemma-2-{quantization}.gguf", "mmproj-BF16.gguf"]
    assert target == files[0]
    assert not merge
    engines = {}
    token = virtualenv_discovery_var.set(True)
    try:
        # Engine discovery uses HF specs; the same adapter serves both hubs.
        discovery_family = match_embedding(
            "embeddinggemma-2", "ggufv2", quantization, "huggingface"
        )
        generate_engine_config_by_model_name(discovery_family, engines)
    finally:
        virtualenv_discovery_var.reset(token)
    assert list(engines["embeddinggemma-2"]) == ["llama.cpp"]
    assert (
        engines["embeddinggemma-2"]["llama.cpp"][0]["embedding_class"]
        is XllamaCppEmbeddingGemma2Model
    )
    assert XllamaCppEmbeddingGemma2Model.match_json(family, spec, quantization) is True


def test_gguf_virtualenv_excludes_native_engines():
    from xinference.core.utils import filter_virtualenv_packages_by_markers

    family = match_embedding("embeddinggemma-2", "ggufv2", "Q8_0")
    packages = filter_virtualenv_packages_by_markers(
        family.virtualenv.packages, "llama.cpp", "13.0"
    )
    assert packages == ["pillow", "#llama_cpp_dependencies#"]


def test_multimodal_content_is_nested_and_ordered(tmp_path):
    audio = tmp_path / "sound.wav"
    audio.write_bytes(b"wave-data")
    item = {
        "text": "describe",
        "image": ["https://example.org/a.png", "data:image/png;base64,AA=="],
        "audio": audio,
        "video": "https://example.org/clip.mp4",
    }
    original = copy.deepcopy(item)
    inputs = prepare_inputs([item, {"image": b"png-data"}], prompt_name="SearchQuery")
    assert item == original
    content = inputs[0]["content"]
    assert [part["type"] for part in content] == [
        "text",
        "image_url",
        "image_url",
        "input_audio",
        "input_video",
    ]
    assert content[0]["text"] == "task: search result | query: describe"
    assert content[1]["image_url"]["url"] == item["image"][0]
    assert content[3]["input_audio"]["url"] == "data:audio/x-wav;base64,d2F2ZS1kYXRh"
    assert content[4]["input_video"]["url"] == item["video"]
    assert len(inputs[1]["content"]) == 1
    assert inputs[1]["content"][0]["type"] == "image_url"


def test_local_file_url_and_pil_image(tmp_path):
    from PIL import Image

    image = tmp_path / "test image.png"
    Image.new("RGB", (2, 2), "red").save(image)
    local = prepare_inputs({"image": image.as_uri()})
    pil = prepare_inputs({"image": Image.open(image)})
    assert local == pil
    assert local[0]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


@pytest.mark.parametrize("dimensions", [128, 256, 512, 768])
def test_dimension_truncation_precedes_normalization(dimensions):
    model = _make_model(dimensions=256)
    result = model._create_embedding(
        {"image": "https://example.org/a.png"}, dimensions=dimensions
    )
    vector = result["data"][0]["embedding"]
    assert len(vector) == dimensions
    assert math.isclose(sum(v * v for v in vector), 1.0)
    assert result["model_replica"] == "gemma"
    assert result["usage"]["total_tokens"] == 10
    model._llm.handle_embeddings.assert_called_once_with(
        {
            "input": [
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.org/a.png"},
                        }
                    ]
                }
            ]
        }
    )


def test_unnormalized_embeddings_and_empty_batch():
    model = _make_model(dimensions=128)
    result = model._create_embedding(
        "hello", normalize_embedding=False, model_uid="gemma"
    )
    assert result["data"][0]["embedding"] == [3.0, 4.0] + [1.0] * 126
    assert model._create_embedding([])["data"] == []
    assert model._llm.handle_embeddings.call_count == 1


@pytest.mark.parametrize(
    "kwargs", [{"dimensions": 64}, {"return_sparse": True}, {"prompt_name": "unknown"}]
)
def test_invalid_options_do_not_reach_backend(kwargs):
    model = _make_model()
    with pytest.raises(ValueError):
        model._create_embedding("hello", **kwargs)
    model._llm.handle_embeddings.assert_not_called()


@pytest.mark.parametrize("vector", [[1.0] * 512, [float("nan")] * 768, {"0": 1.0}])
def test_invalid_backend_embedding_is_rejected(vector):
    model = _make_model()
    model._llm.handle_embeddings.return_value["data"][0]["embedding"] = vector
    with pytest.raises(RuntimeError, match="invalid embeddings"):
        model._create_embedding("hello")


@pytest.fixture
def fake_xllamacpp(monkeypatch):
    module = SimpleNamespace(
        ggml_type=SimpleNamespace(GGML_TYPE_F32=0, GGML_TYPE_F16=1, GGML_TYPE_BF16=30),
        llama_pooling_type=SimpleNamespace(
            LLAMA_POOLING_TYPE_MEAN=1, LLAMA_POOLING_TYPE_LAST=3
        ),
        CommonParams=lambda: SimpleNamespace(
            model="",
            mmproj=SimpleNamespace(path=""),
            cpuparams=SimpleNamespace(n_threads=1),
            cpuparams_batch=SimpleNamespace(n_threads=1),
        ),
        Server=MagicMock(),
        __version__="2026.9.11063",
        estimate_gpu_layers=MagicMock(),
        get_device_info=lambda: [],
        ggml_backend_dev_type=SimpleNamespace(GGML_BACKEND_DEVICE_TYPE_GPU=1),
    )
    monkeypatch.setitem(sys.modules, "xllamacpp", module)
    return module


def test_load_uses_mean_pooling_and_noncausal_batches(fake_xllamacpp):
    model = _make_model(
        llamacpp_model_config={"n_ctx": 2048, "n_threads": 1, "n_gpu_layers": 0}
    )
    model.load()
    try:
        params = fake_xllamacpp.Server.call_args.args[0]
        assert params.model.endswith("embeddinggemma-2-UD-Q4_K_XL.gguf")
        assert params.mmproj.path == "/unused/mmproj-BF16.gguf"
        assert params.pooling_type == 1
        assert (params.n_ctx, params.n_batch, params.n_ubatch, params.n_parallel) == (
            2048,
            2048,
            2048,
            1,
        )
        assert params.cache_type_k == params.cache_type_v == 0
        assert params.embd_normalize == -1
    finally:
        model._executor.shutdown()


@pytest.mark.parametrize(
    "config",
    [
        {"n_ubatch": 512},
        {"n_batch": 512},
        {"n_parallel": 2},
        {"n_ctx": 8193},
        {"embedding": False},
        {"pooling_type": 3},
        {"cache_type_k": 1},
    ],
)
def test_invalid_runtime_config_is_rejected(fake_xllamacpp, config):
    model = _make_model(llamacpp_model_config=config)
    with pytest.raises(ValueError):
        model.load()
    fake_xllamacpp.Server.assert_not_called()


def test_old_runtime_failure_explains_architecture_requirement(fake_xllamacpp):
    fake_xllamacpp.Server.side_effect = AssertionError(
        "unknown model architecture: gemma-embedding2"
    )
    model = _make_model(llamacpp_model_config={"n_gpu_layers": 0})
    with pytest.raises(RuntimeError, match="llama.cpp#30054"):
        model.load()


def test_other_gguf_models_keep_generic_adapter(fake_xllamacpp):
    family = match_embedding("Qwen3-Embedding-0.6B", "ggufv2", "Q8_0")
    spec = family.model_specs[0]
    assert XllamaCppEmbeddingGemma2Model.match_json(family, spec, "Q8_0") is False
    assert XllamaCppEmbeddingModel.match_json(family, spec, "Q8_0") is True
    params = SimpleNamespace()
    XllamaCppEmbeddingModel.__new__(XllamaCppEmbeddingModel)._configure_params(params)
    assert params.pooling_type == 3
    assert params.n_parallel >= 1
    assert not hasattr(params, "embd_normalize")
