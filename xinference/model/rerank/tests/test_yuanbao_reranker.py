# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

# CI: linux

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xinference._model_catalog import load_model_catalog
from xinference.model.utils import get_engine_params_by_name_with_virtual_env

from ..rerank_family import match_rerank
from ..sentence_transformers.core import SentenceTransformerRerankModel
from ..vllm.core import VLLMRerankModel

MODEL_NAMES = [f"Yuanbao-Reranker-{size}" for size in ("0.6B", "4B", "8B")]
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_yuanbao_catalog_and_engines(model_name):
    records = load_model_catalog(Path(__file__).parents[1] / "models")
    record = next(r for r in records if r["model_name"] == model_name)
    source = record["model_specs"][0]["model_src"]["huggingface"]
    assert source == {
        "model_id": f"Yuanbao-Search/{model_name}",
        "model_revision": "main",
        "quantizations": ["none"],
    }
    assert record["max_tokens"] == 8192
    assert record["type"] == "normal"
    family = match_rerank(model_name, "pytorch", "none", "huggingface")
    assert family.model_specs[0].model_revision == "main"
    engines = get_engine_params_by_name_with_virtual_env(
        "rerank", model_name, enable_virtual_env=True
    )
    supported = {
        engine: params
        for engine, params in engines.items()
        if isinstance(params, list) and params
    }
    assert set(supported) == {"sentence_transformers", "vllm"}
    for params in supported.values():
        assert params[0]["model_format"] == "pytorch"
        assert params[0]["quantization"] == "none"


def test_yuanbao_requires_transformers_only(monkeypatch):
    from ..sentence_transformers import core

    monkeypatch.setattr("xinference.constants.XINFERENCE_ENABLE_VIRTUAL_ENV", False)
    dependency = Mock(return_value=True)
    monkeypatch.setattr(core, "check_dependency_available", dependency)
    family = match_rerank(MODEL_NAMES[0], "pytorch", "none", "huggingface")
    spec = family.model_specs[0]
    assert SentenceTransformerRerankModel.match_json(family, spec, "none") is True
    dependency.assert_called_once_with("transformers", "transformers")
    assert VLLMRerankModel.match_json(family, spec, "none") is True
    for cls in (SentenceTransformerRerankModel, VLLMRerankModel):
        assert (
            cls.match_json(family, SimpleNamespace(model_format="ggufv2"), "Q8_0")[0]
            is False
        )


@pytest.fixture
def tiny_yuanbao(monkeypatch):
    import transformers
    from tokenizers import Tokenizer, models, pre_tokenizers

    pre_tokenizer = pre_tokenizers.Whitespace()
    sample = (
        PREFIX
        + SUFFIX
        + DEFAULT_INSTRUCTION
        + " Find matching code. capital of China Beijing Shanghai is a financial center "
        "北京是中国的首都 Instruct Query Document search long short longer document other "
        "first second empty"
    )
    vocab = {"[PAD]": 0, "[UNK]": 1, "yes": 2, "no": 3}
    for token, _ in pre_tokenizer.pre_tokenize_str(sample):
        if token not in vocab:
            vocab[token] = len(vocab)
    tokenizer_impl = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer_impl.pre_tokenizer = pre_tokenizer
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        pad_token="[PAD]",
        unk_token="[UNK]",
        padding_side="left",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        native = transformers.Qwen3ForCausalLM(
            transformers.Qwen3Config(
                vocab_size=len(vocab),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=256,
                pad_token_id=0,
            )
        ).eval()
    load_model = Mock(return_value=native)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer)
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", load_model
    )
    family = match_rerank(MODEL_NAMES[0], "pytorch", "none", "huggingface")
    family.max_tokens = 192
    model = SentenceTransformerRerankModel(
        "yuanbao", "/model", family, "none", device="cpu", enable_flash_attn=False
    )
    model.load()
    load_model.assert_called_once_with("/model", device_map="cpu", torch_dtype="auto")
    return model, tokenizer, native


@pytest.mark.parametrize("instruction", [None, "Find matching code.", ""])
def test_yuanbao_matches_reference_logits_and_prompt(tiny_yuanbao, instruction):
    model, tokenizer, native = tiny_yuanbao
    query = "capital of China"
    documents = ["Beijing", "Shanghai is a financial center", "北京是中国的首都"]
    task = DEFAULT_INSTRUCTION if instruction is None else instruction
    expected_prompts = [
        f"{PREFIX}<Instruct>: {task}\n\n<Query>: {query}\n\n<Document>: {doc}{SUFFIX}"
        for doc in documents
    ]
    captured = []
    process_inputs = model.process_inputs

    def process(pairs):
        assert [PREFIX + pair + SUFFIX for pair in pairs] == expected_prompts
        inputs = process_inputs(pairs)
        captured.append(inputs)
        return inputs

    model.process_inputs = process
    scores = model._rerank(
        documents, [query] * 3, None, None, True, True, instruction=instruction
    )
    inputs = captured[0]
    expected = tokenizer(
        expected_prompts, padding=True, add_special_tokens=False, return_tensors="pt"
    )
    assert torch.equal(inputs["input_ids"], expected["input_ids"])
    mask = inputs["attention_mask"]
    positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
    assert torch.equal(inputs["position_ids"], positions)
    with torch.inference_mode():
        logits = native(**expected, position_ids=positions, use_cache=False).logits
    expected_scores = logits[:, -1, [3, 2]].float().softmax(dim=-1)[:, 1].tolist()
    assert scores == pytest.approx(expected_scores)
    assert model._get_input_tokens() == mask.sum(dim=1).tolist()


def test_yuanbao_truncation_preserves_suffix(tiny_yuanbao):
    model, tokenizer, _ = tiny_yuanbao
    inputs = model.process_inputs(
        ["<Instruct>: search\n\n<Query>: query\n\n<Document>: " + "long " * 500]
    )
    tokens = inputs["input_ids"][0].tolist()
    assert len(tokens) == model.model_family.max_tokens
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    assert tokens[-len(suffix_ids) :] == suffix_ids


async def test_yuanbao_batched_results_and_empty_input(tiny_yuanbao):
    model, _, _ = tiny_yuanbao
    try:
        one, two, empty = await asyncio.gather(
            model.rerank(
                ["short", "longer document"], "first query", 1, None, True, True
            ),
            model.rerank(
                ["other"],
                "second query",
                None,
                None,
                False,
                True,
                instruction="Find code.",
            ),
            model.rerank([], "empty", None, None, True, True),
        )
    finally:
        if model._process_batch_task is not None:
            model._process_batch_task.cancel()
            await asyncio.gather(model._process_batch_task, return_exceptions=True)
    assert len(one["results"]) == len(two["results"]) == 1
    assert one["results"][0]["document"]["text"] in ("short", "longer document")
    assert two["results"][0]["index"] == 0
    assert two["results"][0]["document"] is None
    assert (
        one["meta"]["tokens"]["input_tokens"]
        > two["meta"]["tokens"]["input_tokens"]
        > 0
    )
    assert empty["results"] == []
    assert empty["meta"]["tokens"]["input_tokens"] == 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("version", ["0.12.0", "0.18.0"])
def test_yuanbao_vllm_load_and_score(monkeypatch, tiny_yuanbao, model_name, version):
    from ..vllm import core

    llm = Mock()
    llm.return_value.get_tokenizer.return_value = tiny_yuanbao[1]
    vllm = ModuleType("vllm")
    vllm.LLM = llm
    vllm.__version__ = version
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr(core, "is_vacc_available", lambda: False)
    monkeypatch.setattr(core, "QWEN3_RERANK_TEMPLATE", 0)
    family = match_rerank(model_name, "pytorch", "none", "huggingface")
    model = VLLMRerankModel(
        "yuanbao", "/model", family, "none", hf_overrides={"custom": True}
    )
    model.load()
    config = llm.call_args.kwargs
    assert config["hf_overrides"] == {
        "custom": True,
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    }
    assert config.get("task") == ("score" if version == "0.12.0" else None)
    model._rerank(["doc a", "doc b"], ["query a", "query b"], instruction="Find code.")
    queries, documents = llm.return_value.score.call_args.args
    assert queries == [
        f"{PREFIX}<Instruct>: Find code.\n\n<Query>: query {letter}\n\n"
        for letter in ("a", "b")
    ]
    assert documents == [f"<Document>: doc {letter}{SUFFIX}" for letter in ("a", "b")]
    queries, _ = model._prepare_pairs(["doc"], "query")
    assert f"<Instruct>: {DEFAULT_INSTRUCTION}\n\n<Query>: query\n\n" in queries[0]


@pytest.mark.parametrize("route", ["sync", "0.19.0", "0.20.2"])
@pytest.mark.parametrize("engine_limit", [4096, 8192, 16384])
@pytest.mark.parametrize("long_input", ["document", "query"])
def test_yuanbao_vllm_token_budget(
    monkeypatch, tiny_yuanbao, route, engine_limit, long_input
):
    _, tokenizer, _ = tiny_yuanbao
    family = match_rerank(MODEL_NAMES[0], "pytorch", "none", "huggingface")
    model = VLLMRerankModel("yuanbao-budget", "/model", family, "none")
    model._tokenizer = tokenizer
    config = SimpleNamespace(
        max_model_len=engine_limit,
        hf_config=SimpleNamespace(num_labels=1),
        is_multimodal_model=False,
        architecture="Qwen3ForSequenceClassification",
    )
    model._model = SimpleNamespace(
        model_config=config,
        vllm_config=SimpleNamespace(model_config=config),
        renderer=SimpleNamespace(
            default_cmpl_tok_params=SimpleNamespace(
                get_encode_kwargs=lambda: {"add_special_tokens": False}
            )
        ),
    )
    prompts = []

    def prompt(query, document):
        tokens = tokenizer.encode(query + document, add_special_tokens=False)
        prompts.append(tokens)
        return query + document, {"prompt_token_ids": tokens}

    if route == "sync":

        def score(queries, documents, **kwargs):
            return [prompt(q, d) for q, d in zip(queries, documents)]

        model._model.score = score
    else:
        vllm = ModuleType("vllm")
        vllm.__version__ = route
        vllm.PoolingParams = SimpleNamespace
        monkeypatch.setitem(sys.modules, "vllm", vllm)
        if route == "0.19.0":
            utils = ModuleType("vllm.entrypoints.pooling.score.utils")
            utils.validate_score_input = lambda queries, docs, **kw: (queries, docs)
            utils.get_score_prompt = lambda **kw: prompt(kw["data_1"], kw["data_2"])
            utils.compress_token_type_ids = Mock()
            monkeypatch.setitem(sys.modules, utils.__name__, utils)
        else:
            chat = ModuleType("vllm.entrypoints.chat_utils")
            chat.ChatTemplateConfig = SimpleNamespace
            monkeypatch.setitem(sys.modules, chat.__name__, chat)
            monkeypatch.setitem(sys.modules, "vllm.renderers.chat_utils", None)
            processor = ModuleType("vllm.entrypoints.pooling.scoring.io_processor")

            class CrossEncoderIOProcessor:
                def __init__(self, *args):
                    pass

                def valid_inputs(self, queries, documents):
                    return SimpleNamespace(data_1=queries, data_2=documents)

                def get_score_prompt(self, **kwargs):
                    return prompt(kwargs["data_1"], kwargs["data_2"])

            processor.CrossEncoderIOProcessor = CrossEncoderIOProcessor
            monkeypatch.setitem(sys.modules, processor.__name__, processor)
            utils = ModuleType("vllm.entrypoints.pooling.scoring.utils")
            utils.compress_token_type_ids = Mock()
            monkeypatch.setitem(sys.modules, utils.__name__, utils)

    query = "What is the capital of China?"
    document = " Beijing" * 10000
    if long_input == "query":
        query, document = document, query
    instruction = "Find matching passages."
    if route == "sync":
        model._rerank([document, "short"], query, instruction=instruction)
    else:
        inputs = model._prepare_native_inputs(
            [document, "short"], query, instruction=instruction
        )
        assert [p["prompt_token_ids"] for p, _ in inputs] == prompts
    limit = min(family.max_tokens, engine_limit)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    for tokens, doc in zip(prompts, [document, "short"]):
        body = f"{PREFIX}<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"
        expected = (
            tokenizer.encode(body, add_special_tokens=False)[: limit - len(suffix)]
            + suffix
        )
        assert tokens == expected
        assert len(tokens) <= limit
        assert tokens[-len(suffix) :] == suffix


@pytest.mark.parametrize("slow", [False, True])
def test_yuanbao_budget_keeps_pair_counts_and_rejects_too_small_limit(
    tiny_yuanbao, slow
):
    _, tokenizer, _ = tiny_yuanbao
    family = match_rerank(MODEL_NAMES[0], "pytorch", "none", "huggingface")
    model = VLLMRerankModel("yuanbao-pairs", "/model", family, "none")
    model._tokenizer = (
        SimpleNamespace(is_fast=False, encode=tokenizer.encode, decode=tokenizer.decode)
        if slow
        else tokenizer
    )
    model._model = SimpleNamespace(model_config=SimpleNamespace(max_model_len=192))
    queries, documents = model._prepare_pairs([" Beijing" * 1000, "short"], ["query"])
    assert len(queries) == len(documents) == 2
    for query, document in zip(queries, documents):
        assert len(tokenizer.encode(query + document, add_special_tokens=False)) <= 192
        assert document.endswith(SUFFIX)
    # Keep malformed lengths intact for vLLM's normal input validation.
    queries, documents = model._prepare_pairs(["doc"], ["q1", "q2"])
    assert (len(queries), len(documents)) == (2, 1)
    model._model.model_config.max_model_len = len(
        tokenizer.encode(SUFFIX, add_special_tokens=False)
    )
    with pytest.raises(ValueError, match="leave room for the scoring suffix"):
        model._prepare_pairs(["doc"], "query")
