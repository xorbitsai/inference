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
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from xinference.model.rerank.vllm.core import VLLMRerankModel


@pytest.fixture(params=["0.19.0", "0.19.1", "0.20.0", "0.20.1"])
def native_model(request, monkeypatch):
    """Emulate both vLLM APIs without importing optional GPU libraries."""
    modern = request.param.startswith("0.20")
    model = object.__new__(VLLMRerankModel)
    model.model_family = SimpleNamespace(model_name="bge-reranker-base")
    config = SimpleNamespace(
        hf_config=SimpleNamespace(num_labels=1),
        is_multimodal_model=False,
        architecture="BertForSequenceClassification",
    )
    tok_kwargs = {"add_special_tokens": True, "truncation": False}
    renderer = SimpleNamespace(
        default_cmpl_tok_params=SimpleNamespace(get_encode_kwargs=lambda: tok_kwargs)
    )
    model._model = SimpleNamespace(
        model_config=config,
        vllm_config=SimpleNamespace(model_config=config),
        renderer=renderer,
    )
    model._tokenizer = object()
    calls = SimpleNamespace(
        validate=Mock(),
        prompt=Mock(),
        compress=Mock(return_value=2),
        constructor=Mock(),
        token_type_ids=[0, 0, 1, 1],
    )

    def validate(queries, documents):
        calls.validate(queries, documents)
        if not queries or not documents:
            raise ValueError("At least one element must be given")
        if len(queries) > 1 and len(queries) != len(documents):
            raise ValueError("Input lengths must be either 1:1, 1:N or N:N")
        # Changed values ensure callers use the validator's return value.
        return [q + "!" for q in queries], [d + "!" for d in documents]

    def prompt(query, document):
        calls.prompt(query, document)
        result = {"prompt_token_ids": [11, 12, 13, 14]}
        if calls.token_type_ids is not None:
            result["token_type_ids"] = list(calls.token_type_ids)
        return "unused", result

    class PoolingParams:
        def __init__(self, *, task):
            self.task = task
            self.extra_kwargs = None

    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)

    module("vllm", __version__=request.param, PoolingParams=PoolingParams)
    # The unused API must never be imported, even if a real vLLM is installed.
    old_path = "vllm.entrypoints.pooling.score.utils"
    new_path = "vllm.entrypoints.pooling.scoring.io_processor"
    monkeypatch.setitem(sys.modules, old_path if modern else new_path, None)
    if modern:

        class ChatTemplateConfig:
            pass

        class CrossEncoderIOProcessor:
            def __init__(self, vllm_config, renderer_arg, chat_config):
                assert vllm_config is model._model.vllm_config
                assert renderer_arg is renderer
                assert isinstance(chat_config, ChatTemplateConfig)
                calls.constructor()

            def valid_inputs(self, data_1, data_2):
                queries, documents = validate(data_1, data_2)
                return SimpleNamespace(data_1=queries, data_2=documents)

            def get_score_prompt(self, *, data_1, data_2, encode_kwargs):
                assert encode_kwargs is tok_kwargs
                return prompt(data_1, data_2)

        module("vllm.entrypoints.chat_utils", ChatTemplateConfig=ChatTemplateConfig)
        module(new_path, CrossEncoderIOProcessor=CrossEncoderIOProcessor)
        module(
            "vllm.entrypoints.pooling.scoring.utils",
            compress_token_type_ids=calls.compress,
        )
    else:

        def validate_score_input(data_1, data_2, *, is_multimodal_model, architecture):
            assert is_multimodal_model is False
            assert architecture == config.architecture
            return validate(data_1, data_2)

        def get_score_prompt(
            *, model_config, tokenizer, tokenization_kwargs, data_1, data_2
        ):
            assert model_config is config
            assert tokenizer is model._tokenizer
            assert tokenization_kwargs is tok_kwargs
            return prompt(data_1, data_2)

        module(
            old_path,
            validate_score_input=validate_score_input,
            get_score_prompt=get_score_prompt,
            compress_token_type_ids=calls.compress,
        )
    return model, calls


@pytest.mark.parametrize("query", ["q", ["q1", "q2"]])
def test_native_inputs_pairs_and_pooling_params(native_model, query):
    model, calls = native_model
    inputs = model._prepare_native_inputs(["d1", "d2"], query)
    expected_queries = ["q1!", "q2!"] if query == ["q1", "q2"] else ["q!", "q!"]
    assert calls.prompt.call_args_list == [
        ((q, d),) for q, d in zip(expected_queries, ["d1!", "d2!"])
    ]
    assert len(inputs) == 2
    assert inputs[0][1] is not inputs[1][1]
    for prompt, params in inputs:
        assert prompt == {"prompt_token_ids": [11, 12, 13, 14]}
        assert params.task == "classify"
        assert params.extra_kwargs == {"compressed_token_type_ids": 2}
    assert calls.compress.call_count == 2
    calls.compress.assert_called_with([0, 0, 1, 1])


@pytest.mark.parametrize("token_type_ids", [None, [], [0, 0]])
def test_native_inputs_optional_token_types(native_model, token_type_ids):
    model, calls = native_model
    calls.token_type_ids = token_type_ids
    inputs = model._prepare_native_inputs(["d"], "q")
    assert "token_type_ids" not in inputs[0][0]
    if token_type_ids:
        calls.compress.assert_called_once_with(token_type_ids)
    else:
        calls.compress.assert_not_called()
        assert inputs[0][1].extra_kwargs is None


@pytest.mark.parametrize("num_labels", [None, 0, 2])
def test_native_inputs_reject_non_scalar_scores(native_model, num_labels):
    model, calls = native_model
    if num_labels is None:
        del model._model.model_config.hf_config.num_labels
    else:
        model._model.model_config.hf_config.num_labels = num_labels
    with pytest.raises(ValueError, match="num_labels == 1"):
        model._prepare_native_inputs(["d"], "q")
    calls.validate.assert_not_called()
    calls.prompt.assert_not_called()


@pytest.mark.parametrize(
    "documents,query", [([], "q"), (["d"], []), (["d"], ["q1", "q2"])]
)
def test_native_inputs_propagate_validation_errors(native_model, documents, query):
    model, calls = native_model
    with pytest.raises(ValueError):
        model._prepare_native_inputs(documents, query)
    calls.prompt.assert_not_called()


def test_native_inputs_qwen_template_override(native_model):
    model, calls = native_model
    model.model_family.model_name = "Qwen3-Reranker-0.6B"
    model._prepare_native_inputs(["d"], "q", enable_qwen3_rerank_template=True)
    query, document = calls.validate.call_args.args
    assert "<Query>: q" in query[0]
    assert "<Document>: d" in document[0]
    model._prepare_native_inputs(["d"], "q", enable_qwen3_rerank_template=False)
    calls.validate.assert_called_with(["q"], ["d"])


def test_native_inputs_reject_unexpected_kwargs(native_model):
    model, calls = native_model
    with pytest.raises(RuntimeError, match="Unexpected keyword arguments"):
        model._prepare_native_inputs(["d"], "q", unexpected=True)
    calls.validate.assert_not_called()
