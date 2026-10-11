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
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xinference.model.utils import virtualenv_discovery_var

from .. import embeddinggemma2 as gemma
from ..embed_family import match_embedding
from ..sentence_transformers.core import SentenceTransformerEmbeddingModel
from ..vllm.core import VLLMEmbeddingModel


def _make_model(cls=gemma.SentenceTransformerEmbeddingGemma2Model, **kwargs):
    family = match_embedding("embeddinggemma-2", "pytorch", "none")
    model = cls("gemma", "/unused", family, **kwargs)
    model._model = MagicMock()
    model._prompts = {"SearchQuery": "task: search result | query: "}
    model._clean_cache_if_needed = lambda *args: None
    return model


def test_catalog_hubs_and_engine_discovery():
    from .. import generate_engine_config_by_model_name

    for hub, revision in (("huggingface", "main"), ("modelscope", "master")):
        family = match_embedding("embeddinggemma-2", "pytorch", "none", hub)
        assert family.model_specs[0].model_id == "google/embeddinggemma-2"
        assert family.model_specs[0].model_revision == revision
        assert family.model_specs[0].model_hub == hub
        assert family.model_ability == ["embed_vision", "embed_video", "embed_audio"]
    engines = {}
    token = virtualenv_discovery_var.set(True)
    try:
        family = match_embedding("embeddinggemma-2", "pytorch", "none")
        generate_engine_config_by_model_name(family, engines)
        assert set(engines["embeddinggemma-2"]) == {
            "sentence_transformers",
            "transformers",
            "vllm",
        }
        assert (
            engines["embeddinggemma-2"]["sentence_transformers"][0]["embedding_class"]
            is gemma.SentenceTransformerEmbeddingGemma2Model
        )
        assert (
            SentenceTransformerEmbeddingModel.match_json(
                family, family.model_specs[0], "none"
            )
            is not True
        )
        unsupported = SimpleNamespace(model_format="ggufv2")
        assert (
            gemma.TransformersEmbeddingGemma2Model.match_json(
                family, unsupported, "none"
            )
            is False
        )
        assert VLLMEmbeddingModel.match_json(family, unsupported, "none")[0] is False
    finally:
        virtualenv_discovery_var.reset(token)


@pytest.mark.parametrize("engine", ["sentence_transformers", "transformers", "vllm"])
def test_virtualenv_keeps_system_torch_pins_only_for_native_engines(engine):
    from xinference.core.utils import filter_virtualenv_packages_by_markers
    from xinference.core.virtual_env_manager import ensure_system_torch_pin

    family = match_embedding("embeddinggemma-2", "pytorch", "none")
    packages = filter_virtualenv_packages_by_markers(
        family.virtualenv.packages, engine, "13.0"
    )
    packages = ensure_system_torch_pin(packages)
    assert "pillow" in packages
    for pin in ("#system_torch#", "#system_torchvision#"):
        assert packages.count(pin) == (0 if engine == "vllm" else 1)
    assert any(pkg.startswith("sentence-transformers") for pkg in packages) == (
        engine == "sentence_transformers"
    )
    assert [pkg for pkg in packages if pkg.startswith("vllm")] == (
        ["vllm>=0.32.0"] if engine == "vllm" else []
    )


def test_interleaved_inputs_keep_media_and_do_not_mutate():
    item = {
        "text": "before <|image|> between <|image|> <|video|> <|audio|>",
        "image": ["a.png", "b.png"],
        "video": "clip.mp4",
        "audio": "sound.wav",
    }
    original = copy.deepcopy(item)
    messages = gemma.normalize_inputs(
        [item, {"audio": "only.wav"}],
        {"Document": "title: none | text: "},
        prompt_name="Document",
    )
    assert item == original
    parts = messages[0][0]["content"]
    assert [part["type"] for part in parts] == [
        "text",
        "image",
        "image",
        "video",
        "audio",
    ]
    assert parts[0]["text"] == "title: none | text: " + item["text"]
    assert messages[1][0]["content"] == [{"type": "audio", "audio": "only.wav"}]


@pytest.mark.parametrize("value", [{}, {"image_url": "x"}, {"text": 5}, [3]])
def test_invalid_inputs_are_rejected(value):
    with pytest.raises(ValueError):
        gemma.normalize_inputs(value, {})


@pytest.mark.parametrize("dimensions", [64, 1024, 128.0, True])
def test_invalid_dimensions_are_rejected(dimensions):
    with pytest.raises(ValueError, match="dimensions must be"):
        gemma.validate_dimensions(dimensions)


@pytest.mark.parametrize("dtype", ["float16", "fp16", torch.float16])
def test_float16_is_rejected(dtype):
    with pytest.raises(ValueError, match="float16 produces invalid"):
        gemma.resolve_dtype(dtype, "cpu")
    assert gemma.resolve_dtype(None, "cpu") == torch.float32


def test_sentence_transformers_uses_native_encode_and_normalizes_after_truncation():
    model = _make_model()
    callbacks = []
    model._model.register_forward_pre_hook.side_effect = (
        lambda callback: callbacks.append(callback)
        or model._model.register_forward_pre_hook.return_value
    )

    def encode(*args, **kwargs):
        callbacks[0](
            None, ({"attention_mask": torch.cat((torch.ones(283), torch.zeros(3)))},)
        )
        return torch.stack([torch.arange(1, 769, dtype=torch.float32), torch.ones(768)])

    model._model.encode.side_effect = encode
    result = model._create_embedding(
        ["query", {"image": "x.png"}],
        dimensions=128,
        prompt_name="SearchQuery",
        model_uid="public",
    )
    assert result["model"] == "public"
    assert result["model_replica"] == "gemma"
    assert [row["index"] for row in result["data"]] == [0, 1]
    assert result["usage"]["total_tokens"] == 283
    for row in result["data"]:
        assert len(row["embedding"]) == 128
        assert torch.linalg.vector_norm(
            torch.tensor(row["embedding"])
        ).item() == pytest.approx(1)
    args, kwargs = model._model.encode.call_args
    assert args[0][0][0]["content"][0]["text"].startswith(
        "task: search result | query: "
    )
    assert args[0][1][0]["content"] == [{"type": "image", "image": "x.png"}]
    assert kwargs["convert_to_tensor"] is True
    assert "output_value" not in kwargs
    model._model.register_forward_pre_hook.return_value.remove.assert_called_once()
    model._model.tokenize.assert_not_called()


def test_single_and_empty_inputs_have_correct_cardinality():
    model = _make_model(dimensions=256)
    model._model.encode.return_value = torch.ones(1, 768)
    assert (
        len(model._create_embedding({"audio": "x.wav"})["data"][0]["embedding"]) == 256
    )
    model._model.encode.reset_mock()
    assert model._create_embedding([])["data"] == []
    model._model.encode.assert_not_called()


def test_sentence_transformers_removes_token_hook_after_failure():
    model = _make_model()
    model._model.encode.side_effect = RuntimeError("encode failed")
    with pytest.raises(RuntimeError, match="encode failed"):
        model._create_embedding("text")
    model._model.register_forward_pre_hook.return_value.remove.assert_called_once()


def test_native_transformers_and_sentence_transformers_agree_on_mixed_batch(tmp_path):
    transformers = pytest.importorskip("transformers", minversion="5.19.0")
    st = pytest.importorskip("sentence_transformers", minversion="6.1.0")
    from PIL import Image
    from sentence_transformers.base.modules.normalize import Normalize
    from sentence_transformers.base.modules.transformer import Transformer
    from sentence_transformers.sentence_transformer.modules.pooling import Pooling
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<bos>": 2,
        "<unk>": 3,
        "hello": 4,
        "<|image|>": 5,
        "<|audio|>": 6,
        "<image_start>": 7,
        "<image_end>": 8,
        "<audio_start>": 9,
        "<audio_end>": 10,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 2), ("<eos>", 1)]
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=8192,
        model_specific_special_tokens={
            "image_token": "<|image|>",
            "audio_token": "<|audio|>",
            "boi_token": "<image_start>",
            "eoi_token": "<image_end>",
            "boa_token": "<audio_start>",
            "eoa_token": "<audio_end>",
        },
    )
    config = transformers.EmbeddingGemma2Config(
        text_config=transformers.EmbeddingGemma2TextConfig(
            hidden_size=8,
            hidden_size_per_layer_input=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            layer_types=["full_attention"],
            per_layer_config={},
        ),
        vision_config=transformers.Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=32,
            global_head_dim=32,
        ),
        audio_config=None,
        image_token_id=5,
        boi_token_id=7,
        eoi_token_id=8,
    )
    transformers.EmbeddingGemma2Model(config).save_pretrained(tmp_path)
    processor = transformers.EmbeddingGemma2Processor(
        transformers.Gemma4AudioFeatureExtractor(),
        transformers.Gemma4ImageProcessor(),
        tokenizer,
        transformers.Gemma4VideoProcessor(),
        chat_template="{% for message in messages %}{% for part in message['content'] %}"
        "{% if part['type'] == 'text' %}{{ part['text'] }}{% else %}<|image|>{% endif %}"
        "{% endfor %}{% endfor %}",
    )
    processor.save_pretrained(tmp_path)
    backbone = Transformer(
        str(tmp_path),
        transformer_task="feature-extraction",
        modality_config={
            key: {
                "method": "forward",
                "method_output_name": "last_hidden_state",
                **({"format": "structured"} if key == "message" else {}),
            }
            for key in ("text", "image", "message")
        },
        module_output_name="token_embeddings",
    )
    st.SentenceTransformer(
        modules=[backbone, Pooling(768, pooling_mode="mean"), Normalize()]
    ).save(str(tmp_path))
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32), (100, 50, 0)).save(image)
    results = []
    for cls in (
        gemma.TransformersEmbeddingGemma2Model,
        gemma.SentenceTransformerEmbeddingGemma2Model,
    ):
        model = cls(
            "native", str(tmp_path), match_embedding("embeddinggemma-2"), device="cpu"
        )
        model.load()
        # A single image in a two-item batch used to make output_value=None
        # index media tensors as if they had one row for every sample.
        result = model._create_embedding(
            ["hello", {"image": str(image)}], batch_size=2, dimensions=128
        )
        assert len(result["data"]) == 2
        results.append(result)
    assert results[0]["usage"] == results[1]["usage"]
    for left, right in zip(results[0]["data"], results[1]["data"]):
        torch.testing.assert_close(
            torch.tensor(left["embedding"]),
            torch.tensor(right["embedding"]),
            atol=1e-5,
            rtol=1e-4,
        )


def test_transformers_mean_pooling_masks_padding():
    model = _make_model(gemma.TransformersEmbeddingGemma2Model)
    features = {
        "input_ids": torch.tensor([[2, 3, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
    }
    batch = MagicMock()
    batch.to.return_value = features
    model._processor = MagicMock()
    model._processor.apply_chat_template.return_value = batch
    hidden = torch.ones(1, 3, 768)
    hidden[0, 0, 0] = 2
    hidden[0, 1, 0] = 4
    hidden[0, 2, 0] = 1000
    model._model.return_value = SimpleNamespace(last_hidden_state=hidden)
    result = model._create_embedding("text", normalize_embedding=False)
    assert result["data"][0]["embedding"][0] == 3
    assert result["usage"]["prompt_tokens"] == 2


def _fake_vllm(monkeypatch, supported=True):
    module = types.ModuleType("vllm")
    module.__version__ = "0.32.0"
    module.ModelRegistry = SimpleNamespace(
        get_supported_archs=lambda: ["EmbeddingGemma2Model"] if supported else []
    )
    module.PoolingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    module.LLM = MagicMock()
    monkeypatch.setitem(sys.modules, "vllm", module)
    utils = types.ModuleType("vllm.multimodal.utils")
    for modality in ("image", "video", "audio"):
        setattr(
            utils,
            f"fetch_{modality}",
            lambda value, modality=modality: f"{modality}:{value}",
        )
    monkeypatch.setitem(sys.modules, "vllm.multimodal.utils", utils)
    return module


def test_vllm_rejects_missing_architecture(monkeypatch):
    _fake_vllm(monkeypatch, supported=False)
    family = match_embedding("embeddinggemma-2", "pytorch", "none")
    token = virtualenv_discovery_var.set(False)
    try:
        matched = VLLMEmbeddingModel.match_json(family, family.model_specs[0], "none")
        assert matched[0] is False
        assert "vLLM>=0.32.0" in matched[1]
    finally:
        virtualenv_discovery_var.reset(token)


def test_vllm_load_sets_safe_dtype_and_matryoshka(monkeypatch, tmp_path):
    module = _fake_vllm(monkeypatch)
    (tmp_path / "chat_template.jinja").write_text("template")
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": {}})
    )
    model = VLLMEmbeddingModel(
        "gemma",
        str(tmp_path),
        match_embedding("embeddinggemma-2"),
        dimensions=256,
        hf_overrides={"vision_config": None},
        torch_dtype="bfloat16",
    )
    model.load()
    kwargs = module.LLM.call_args.kwargs
    assert kwargs["runner"] == "pooling"
    assert kwargs["max_model_len"] == 8192
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["hf_overrides"]["vision_config"] is None
    assert kwargs["hf_overrides"]["matryoshka_dimensions"] == [128, 256, 512, 768]
    assert "dimensions" not in kwargs
    assert model._dimensions == 256


def test_vllm_preserves_all_modalities_and_pooling_options(monkeypatch):
    _fake_vllm(monkeypatch)
    model = VLLMEmbeddingModel("gemma", "/unused", match_embedding("embeddinggemma-2"))
    model._model = MagicMock()
    model._tokenizer = MagicMock()
    model._tokenizer.apply_chat_template.return_value = "rendered"
    model._chat_template = "template"
    params = object()
    model._embed_embeddinggemma2(
        {
            "text": "caption",
            "image": ["a.png", "b.png"],
            "video": "clip.mp4",
            "audio": "sound.wav",
        },
        params,
    )
    item = model._model.embed.call_args.args[0][0]
    assert item["multi_modal_data"] == {
        "image": [
            f"image:file://{Path('a.png').absolute()}",
            f"image:file://{Path('b.png').absolute()}",
        ],
        "video": [f"video:file://{Path('clip.mp4').absolute()}"],
        "audio": [f"audio:file://{Path('sound.wav').absolute()}"],
    }
    assert model._model.embed.call_args.kwargs["pooling_params"] is params
    model._context_length = 8192
    model._dimensions = 128
    _, pool, _ = model._prepare_embedding(
        {"audio": "x.wav"}, normalize_embeddings=False
    )
    assert pool.dimensions == 128
    assert pool.use_activation is False
