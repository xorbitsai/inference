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
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from xinference.model.rerank.vllm.core import VLLMRerankModel


@pytest.fixture
def model(monkeypatch):
    config = SimpleNamespace(
        score_type="cross-encoder",
        hf_config=SimpleNamespace(num_labels=1),
        is_multimodal_model=False,
        architecture="XLMRobertaForSequenceClassification",
    )
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.19.0"
    vllm.PoolingParams = SimpleNamespace
    vllm.LLM = Mock()
    args_module = ModuleType("vllm.engine.arg_utils")

    class EngineArgs(SimpleNamespace):
        def create_model_config(self):
            return config

    args_module.AsyncEngineArgs = EngineArgs
    engine = Mock()
    engine.model_config = config
    engine.renderer.default_cmpl_tok_params.get_encode_kwargs.return_value = {
        "add_special_tokens": True
    }
    engine_module = ModuleType("vllm.v1.engine.async_llm")
    engine_module.AsyncLLM = SimpleNamespace(from_engine_args=Mock(return_value=engine))
    outputs = ModuleType("vllm.outputs")
    outputs.ScoringRequestOutput = SimpleNamespace(from_base=lambda output: output)
    utils = ModuleType("vllm.entrypoints.pooling.score.utils")
    utils.validate_score_input = lambda queries, docs, **kw: (queries, docs)

    def get_prompt(**kw):
        assert kw["tokenization_kwargs"] == {"add_special_tokens": True}
        query, doc = kw["data_1"], kw["data_2"]
        return "", {
            "prompt_token_ids": [len(query), len(doc)],
            "token_type_ids": [0, 1],
        }

    utils.get_score_prompt = Mock(side_effect=get_prompt)
    utils.compress_token_type_ids = Mock(return_value=1)
    for name, module in {
        "vllm": vllm,
        "vllm.outputs": outputs,
        "vllm.engine.arg_utils": args_module,
        "vllm.v1.engine.async_llm": engine_module,
        "vllm.entrypoints.pooling.score.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        "xinference.model.rerank.vllm.core.is_vacc_available", lambda: False
    )
    monkeypatch.setattr("xinference.model.rerank.vllm.core.gc.collect", Mock())
    monkeypatch.setattr("xinference.model.rerank.vllm.core.empty_cache", Mock())
    family = SimpleNamespace(
        model_name="bge-reranker-v2-m3", model_specs=[None], type="normal"
    )
    model = VLLMRerankModel(
        "replica", "/model", family, "none", batch_size=8, batch_interval=0.1
    )
    model.load()
    return model


def output(prompt):
    return SimpleNamespace(
        outputs=SimpleNamespace(score=float(prompt["prompt_token_ids"][1])),
        prompt_token_ids=prompt["prompt_token_ids"],
    )


@pytest.mark.asyncio
async def test_overlap_sort_top_n_documents_and_independent_usage(model):
    started = []
    ready, release = asyncio.Event(), asyncio.Event()

    async def encode(prompt, params, request_id):
        started.append((prompt, params, request_id))
        if len(started) == 4:
            ready.set()
        await release.wait()
        assert "token_type_ids" not in prompt
        assert params.task == "classify"
        assert params.extra_kwargs == {"compressed_token_type_ids": 1}
        yield output(prompt)

    model._model.encode = encode
    tasks = [
        asyncio.create_task(
            model.rerank(["a", "long", "same"], "q", 2, None, True, True)
        ),
        asyncio.create_task(model.rerank(["bb"], "qq")),
    ]
    try:
        await asyncio.wait_for(ready.wait(), 2)
        assert model._process_batch_task is None
        release.set()
        one, two = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert [r["index"] for r in one["results"]] == [1, 2]
    assert [r["document"]["text"] for r in one["results"]] == ["long", "same"]
    assert [r["relevance_score"] for r in one["results"]] == [4, 4]
    assert one["meta"]["tokens"]["input_tokens"] == 6  # all inputs, before top_n
    assert two["results"][0]["index"] == 0 and two["results"][0]["document"] is None
    assert two["meta"]["tokens"] is None
    assert one["id"] != two["id"] and len({x[2] for x in started}) == 4
    assert model._counter == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_cancellation_and_failure_clean_up_siblings(model, fail):
    active, cancelled = set(), set()
    ready = asyncio.Event()

    async def encode(prompt, params, request_id):
        doc_len = prompt["prompt_token_ids"][1]
        active.add(doc_len)
        if len(active) == 2:
            ready.set()
        try:
            await ready.wait()
            if fail and doc_len == 1:
                raise ValueError("invalid input")
            await asyncio.Event().wait()
            yield output(prompt)
        except asyncio.CancelledError:
            cancelled.add(doc_len)
            raise
        finally:
            active.remove(doc_len)

    model._model.encode = encode
    task = asyncio.create_task(model.rerank(["a", "bb"], "query"))
    await asyncio.wait_for(ready.wait(), 2)
    if not fail:
        task.cancel()
    with pytest.raises(ValueError if fail else asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert not active and 2 in cancelled


def test_qwen_template_and_native_load_and_stop(model):
    model.model_family.model_name = "Qwen3-Reranker-0.6B"
    utils = sys.modules["vllm.entrypoints.pooling.score.utils"]
    model._prepare_native_inputs(["doc"], "query", enable_qwen3_rerank_template=True)
    args = utils.get_score_prompt.call_args.kwargs
    assert "<Query>: query" in args["data_1"] and "<Document>: doc" in args["data_2"]
    model._prepare_native_inputs(["doc"], "query", enable_qwen3_rerank_template=False)
    args = utils.get_score_prompt.call_args.kwargs
    assert (args["data_1"], args["data_2"]) == ("query", "doc")
    with pytest.raises(RuntimeError, match="Unexpected keyword"):
        model._prepare_native_inputs(["doc"], "query", unknown=True)
    args = sys.modules[
        "vllm.v1.engine.async_llm"
    ].AsyncLLM.from_engine_args.call_args.args[0]
    assert not hasattr(args, "batch_size") and not hasattr(args, "batch_interval")
    engine = model._model
    model.stop()
    engine.shutdown.assert_called_once()
    assert model._model is None


@pytest.mark.parametrize("fallback", ["old", "vacc", "vl", "bi-encoder", "multimodal"])
def test_legacy_load_fallback(model, monkeypatch, fallback):
    if fallback == "old":
        sys.modules["vllm"].__version__ = "0.18.0"
    elif fallback == "vacc":
        monkeypatch.setattr(
            "xinference.model.rerank.vllm.core.is_vacc_available", lambda: True
        )
        monkeypatch.setitem(sys.modules, "vllm_vacc", ModuleType("vllm_vacc"))
    elif fallback == "vl":
        model.model_family.model_name = "Qwen3-VL-Reranker-2B"
    elif fallback == "multimodal":
        model._model.model_config.is_multimodal_model = True
    else:
        model._model.model_config.score_type = "bi-encoder"
    legacy = VLLMRerankModel("legacy", "/model", model.model_family, "none")
    legacy.load()
    assert not legacy._native_pooling
    sys.modules["vllm"].LLM.assert_called_once()


def test_legacy_scoring_keeps_pair_preparation_and_cleanup(model):
    sys.modules["vllm"].__version__ = "0.18.0"
    legacy = VLLMRerankModel("legacy", "/model", model.model_family, "none")
    legacy.load()
    expected = [
        output({"prompt_token_ids": [1, 4]}),
        output({"prompt_token_ids": [1, 1]}),
    ]
    legacy._model.score.return_value = expected
    actual = legacy._rerank(["long", "a"], "q", top_n=1, return_len=True)
    assert actual == expected and legacy._counter == 1
    legacy._model.score.assert_called_once_with(
        ["q", "q"], ["long", "a"], use_tqdm=False
    )
    result = legacy._format_rerank_outputs(["long", "a"], actual, 1, True, True)
    assert result["results"][0]["document"]["text"] == "long"
    assert result["meta"]["tokens"]["input_tokens"] == 4
