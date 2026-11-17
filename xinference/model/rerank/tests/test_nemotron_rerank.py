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

import copy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xinference.core.utils import filter_virtualenv_packages_by_markers
from xinference.core.virtual_env_manager import expand_engine_dependency_placeholders
from xinference.model.rerank import _install
from xinference.model.rerank.nemotron import NEMOTRON_RERANK_MODEL_NAME as MODEL
from xinference.model.rerank.nemotron import nemotron_messages
from xinference.model.rerank.rerank_family import BUILTIN_RERANK_MODELS
from xinference.model.rerank.sentence_transformers.core import (
    SentenceTransformerRerankModel,
)
from xinference.model.rerank.transformers.core import NemotronTransformersRerankModel
from xinference.model.rerank.vllm.core import VLLMRerankModel
from xinference.model.utils import get_engine_params_by_name_with_virtual_env

IMAGE = "data:image/png;base64,aW1hZ2U="


def family():
    return BUILTIN_RERANK_MODELS[MODEL][0]


def make_model(cls):
    return cls("test", "/unused", family(), "none", device="cpu")


def test_catalog_and_virtualenv_engines():
    _install()
    spec = family()
    assert spec.model_ability == ["rerank_vision"]
    assert spec.max_tokens == 32768
    assert {(s.model_hub, s.model_id, s.model_revision) for s in spec.model_specs} == {
        ("huggingface", "nvidia/Nemotron-3.5-Rerank-8B-BF16", "main"),
        ("modelscope", "nv-community/Nemotron-3.5-Rerank-8B-BF16", "master"),
    }
    engines = get_engine_params_by_name_with_virtual_env(
        "rerank", MODEL, enable_virtual_env=True
    )
    for engine in ("sentence_transformers", "transformers", "vllm"):
        assert isinstance(engines[engine], list)
        packages = filter_virtualenv_packages_by_markers(
            expand_engine_dependency_placeholders(spec.virtualenv.packages, engine),
            engine,
            None,
        )
        assert not any("#engine#" in package for package in packages)
        if engine == "vllm":
            assert "vllm==0.32.0" in packages
            assert "transformers==5.19.0" in packages
            assert not any(p.startswith("sentence-transformers") for p in packages)
        else:
            assert "transformers==5.20.0" in packages
            assert not any(p.startswith("vllm") for p in packages)
            if engine == "sentence_transformers":
                assert "sentence-transformers[image]==6.1.0" in packages
            else:
                assert not any(p.startswith("sentence-transformers") for p in packages)
    assert not isinstance(engines.get("llama.cpp"), list)


@pytest.mark.parametrize(
    "document",
    [
        "text",
        {"text": "text"},
        {"image": IMAGE},
        {"text": "text", "image_url": {"url": IMAGE}},
        {
            "content": [
                {"type": "text", "text": "text"},
                {"type": "image_url", "image_url": {"url": IMAGE}},
            ]
        },
    ],
)
def test_input_normalization_preserves_original(document):
    original = copy.deepcopy(document)
    messages = nemotron_messages("query", document)
    assert [m["role"] for m in messages] == ["query", "document"]
    assert document == original
    assert all(b["type"] in ("text", "image") for b in messages[1]["content"])


@pytest.mark.parametrize(
    "query,document",
    [
        ({"image": IMAGE}, "text"),
        ("q", {}),
        ("q", {"content": "text"}),
        ("q", {"video": "video.mp4"}),
        ("q", {"content": [{"type": "audio", "audio": "audio.wav"}]}),
        ("q", {"text": 1}),
        ("q", {"image": IMAGE, "image_url": IMAGE}),
        ("q", {"image_url": {}}),
        ("q", {"image": "/tmp/local.png"}),
        ("q", {"content": [1]}),
    ],
)
def test_unsupported_inputs_rejected(query, document):
    with pytest.raises(ValueError, match="Nemotron"):
        nemotron_messages(query, document)


class Inputs(dict):
    def to(self, device):
        return self


def processor_inputs(length=4):
    return Inputs(
        input_ids=torch.ones((1, length), dtype=torch.int64),
        attention_mask=torch.ones((1, length), dtype=torch.int64),
        pixel_values=torch.ones((1, 3), dtype=torch.float32),
    )


async def test_native_transformers_raw_logits_and_multimodal_tokens():
    model = make_model(NemotronTransformersRerankModel)
    model._processor = SimpleNamespace(
        apply_chat_template=Mock(side_effect=[processor_inputs(4), processor_inputs(6)])
    )
    model._model = Mock(device="cpu", dtype=torch.bfloat16)
    model._model.side_effect = [
        SimpleNamespace(logits=torch.tensor([[-2.0]])),
        SimpleNamespace(logits=torch.tensor([[-1.0]])),
    ]
    documents = ["text", {"text": "page", "image": IMAGE}]
    result = await model.rerank(
        documents,
        "query",
        max_chunks_per_doc=None,
        return_documents=True,
        return_len=True,
        top_n=1,
    )
    assert result["results"] == [
        {"index": 1, "relevance_score": -1.0, "document": {"text": documents[1]}}
    ]
    assert result["meta"]["tokens"]["input_tokens"] == 10
    assert model._get_input_tokens_slice(1, 2) == [6]
    assert model._model.call_args.kwargs["pixel_values"].dtype == torch.bfloat16
    call = model._processor.apply_chat_template.call_args
    assert call.args[0][1]["content"][1] == {"type": "image", "image": IMAGE}
    assert call.kwargs["add_generation_prompt"] is False


def test_native_context_limit_and_empty_input():
    model = make_model(NemotronTransformersRerankModel)
    model._processor = SimpleNamespace(
        apply_chat_template=Mock(return_value=processor_inputs(32769))
    )
    model._model = Mock()
    with pytest.raises(ValueError, match="exceeds 32768"):
        model._rerank_nemotron(["doc"], ["q"], True)
    model._model.assert_not_called()
    assert model._rerank_nemotron([], [], False) == []


def test_sentence_transformers_predict_and_tokens():
    model = make_model(SentenceTransformerRerankModel)
    model._model = SimpleNamespace(
        predict=Mock(return_value=torch.tensor([-2.0, -1.0], dtype=torch.bfloat16))
    )
    model._nemotron_processor = SimpleNamespace(
        apply_chat_template=Mock(side_effect=[processor_inputs(3), processor_inputs(5)])
    )
    documents = ["text", {"text": "page", "image_url": {"url": IMAGE}}]
    scores = model._rerank(documents, ["q", "q"], None, None, True, True)
    assert scores == [-2.0, -1.0]
    assert model._get_n_tokens() == 8
    call = model._model.predict.call_args
    assert call.args[0] == [
        ["q", {"text": "text"}],
        ["q", {"text": "page", "image": IMAGE}],
    ]
    assert call.kwargs["batch_size"] == 1
    assert call.kwargs["convert_to_tensor"] is True


@pytest.mark.parametrize(
    "cls",
    [SentenceTransformerRerankModel, NemotronTransformersRerankModel, VLLMRerankModel],
)
def test_pair_count_validation(cls, monkeypatch):
    if cls is VLLMRerankModel:
        install_fake_vllm(monkeypatch)
    model = make_model(cls)
    with pytest.raises(ValueError, match="equal length"):
        (
            model._rerank_nemotron(["doc"], [], False)
            if cls is not VLLMRerankModel
            else model._rerank_nemotron(["doc"], [])
        )


def install_fake_vllm(monkeypatch, **attributes):
    module = ModuleType("vllm")
    module.__dict__.update(__version__="0.32.0", **attributes)
    monkeypatch.setitem(sys.modules, "vllm", module)
    output_module = ModuleType("vllm.outputs")
    output_module.ScoringRequestOutput = SimpleNamespace(
        from_base=lambda output: output
    )
    monkeypatch.setitem(sys.modules, "vllm.outputs", output_module)


@pytest.mark.parametrize("load_reporting", [False, True])
async def test_vllm_classification_template_and_images(monkeypatch, load_reporting):
    install_fake_vllm(monkeypatch)
    from transformers import image_utils

    image = Mock()
    image.convert.return_value = image
    monkeypatch.setattr(image_utils, "load_image", Mock(return_value=image))
    model = make_model(VLLMRerankModel)
    outputs = [
        SimpleNamespace(outputs=SimpleNamespace(score=-2.0), prompt_token_ids=[1, 2]),
        SimpleNamespace(
            outputs=SimpleNamespace(score=-1.0), prompt_token_ids=[1, 2, 3]
        ),
    ]
    progress_callbacks = []

    def encode(inputs, *, pooling_task, use_tqdm):
        if load_reporting:
            assert use_tqdm is model._pooling_load._sync_progress
            assert use_tqdm.count == len(inputs)
            progress_callbacks.append(use_tqdm)
            use_tqdm.update(len(inputs))
        return outputs

    model._model = SimpleNamespace(
        encode=Mock(side_effect=encode),
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_seqs=1)
            )
        ),
    )
    if load_reporting:
        model.enable_engine_load_reporting()
        model._pooling_load.configure_sync(model._model)
    model._tokenizer = SimpleNamespace(
        apply_chat_template=Mock(return_value="checkpoint prompt")
    )
    documents = ["text", {"text": "page", "image": IMAGE}]
    result = await model.rerank(
        documents, "q", return_documents=True, return_len=True, top_n=1
    )
    assert result["results"][0]["index"] == 1
    assert result["results"][0]["relevance_score"] == -1.0
    assert result["results"][0]["document"]["text"] == documents[1]
    assert result["meta"]["tokens"]["input_tokens"] == 5
    call = model._model.encode.call_args
    assert call.kwargs == {
        "pooling_task": "classify",
        "use_tqdm": progress_callbacks[0] if load_reporting else False,
    }
    if load_reporting:
        assert model._pooling_load._sync_progress is None
    assert "multi_modal_data" not in call.args[0][0]
    assert call.args[0][1]["multi_modal_data"] == {"image": image}
    assert model._tokenizer.apply_chat_template.call_args.args[0][1]["content"][1] == {
        "type": "image"
    }


def test_vllm_load_uses_mean_pooling_without_activation(monkeypatch):
    llm = Mock()
    install_fake_vllm(monkeypatch, LLM=llm)
    model = make_model(VLLMRerankModel)
    model._kwargs = {"pooler_config": {"use_activation": True}}
    model.load()
    call = llm.call_args
    assert call.kwargs["runner"] == "pooling"
    assert call.kwargs["max_model_len"] == 32768
    assert call.kwargs["pooler_config"] == {
        "seq_pooling_type": "MEAN",
        "use_activation": False,
    }
    assert model._native_pooling is False


@pytest.mark.parametrize(
    "engine,module_name,version",
    [
        ("vllm", "vllm", "0.31.0"),
        ("transformers", "transformers", "5.19.0"),
        ("sentence_transformers", "sentence_transformers", "6.0.0"),
    ],
)
def test_discovery_rejects_old_versions_without_virtualenv(
    monkeypatch, engine, module_name, version
):
    from xinference.model.rerank import nemotron

    monkeypatch.setattr(nemotron, "virtual_env_allows_missing_engine", lambda: False)
    monkeypatch.setattr(nemotron, "check_dependency_available", lambda *args: True)
    monkeypatch.setattr(
        nemotron.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            __version__=version if name == module_name else "99.0.0"
        ),
    )
    assert nemotron.check_nemotron_dependencies(engine)[0] is False


def test_sentence_transformers_load_keeps_raw_scores(monkeypatch):
    import transformers

    from xinference.model import utils

    monkeypatch.setattr(utils, "neutralize_broken_torchcodec", lambda: None)
    monkeypatch.setattr(transformers, "__version__", "5.20.0")
    constructor = Mock()
    module = ModuleType("sentence_transformers")
    module.__version__ = "6.1.0"
    module.CrossEncoder = constructor
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    model = make_model(SentenceTransformerRerankModel)
    model._kwargs = {"enable_flash_attn": True}
    model.load()
    arguments = constructor.call_args.kwargs
    assert arguments["trust_remote_code"] is True
    assert isinstance(arguments["activation_fn"], torch.nn.Identity)
    assert arguments["model_kwargs"] == {
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
    }
    assert arguments["max_length"] == 32768


async def test_native_requests_keep_queries_scores_and_token_counts_separate():
    import asyncio

    model = make_model(NemotronTransformersRerankModel)
    model._processor = SimpleNamespace(
        apply_chat_template=Mock(
            side_effect=[processor_inputs(3), processor_inputs(4), processor_inputs(5)]
        )
    )
    model._model = Mock(device="cpu", dtype=torch.bfloat16)
    model._model.side_effect = [
        SimpleNamespace(logits=torch.tensor([[score]])) for score in [-3.0, -1.0, -2.0]
    ]
    first, second = await asyncio.gather(
        model.rerank(["a", {"image": IMAGE}], "q1", None, None, True, True),
        model.rerank(["b"], "q2", None, None, True, True),
    )
    assert [r["index"] for r in first["results"]] == [1, 0]
    assert [r["relevance_score"] for r in second["results"]] == [-2.0]
    assert first["meta"]["tokens"]["input_tokens"] == 7
    assert second["meta"]["tokens"]["input_tokens"] == 5
    assert [
        call.args[0][0]["content"][0]["text"]
        for call in model._processor.apply_chat_template.call_args_list
    ] == ["q1", "q1", "q2"]


@pytest.mark.parametrize(
    "cls",
    [SentenceTransformerRerankModel, NemotronTransformersRerankModel, VLLMRerankModel],
)
def test_incompatible_format_rejected_even_with_virtualenv(cls, monkeypatch):
    from xinference.model.rerank import nemotron

    monkeypatch.setattr(nemotron, "virtual_env_allows_missing_engine", lambda: True)
    result = cls.match_json(family(), SimpleNamespace(model_format="ggufv2"), "none")
    assert result[0] is False


def test_native_transformers_does_not_claim_unrelated_models():
    other = family().copy(update={"model_name": "bge-reranker-base"})
    assert (
        NemotronTransformersRerankModel.match_json(other, other.model_specs[0], "none")[
            0
        ]
        is False
    )


def test_native_load_uses_sequence_classification_and_trust_policy(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.20.0")
    model = make_model(NemotronTransformersRerankModel)
    processor = Mock()
    loader = Mock()
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", processor)
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification, "from_pretrained", loader
    )
    model.load()
    processor.assert_called_once_with("/unused")
    assert loader.call_args.kwargs["trust_remote_code"] is True
    assert loader.call_args.kwargs["device_map"] == "cpu"
    assert model._model is loader.return_value.eval.return_value
