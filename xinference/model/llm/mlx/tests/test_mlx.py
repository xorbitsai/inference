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
import base64
import concurrent.futures
import importlib
import json
import os
import platform
import sys
import threading
from types import SimpleNamespace

import pytest

from .....client import Client


def test_mlx_model_uses_model_sampling_defaults():
    from ..core import MLXModel

    model = object.__new__(MLXModel)
    model._model_generation_config = {}
    model._update_model_generation_config(
        {
            "generation_config": {
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 10,
            },
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        }
    )

    defaults = model._sanitize_generate_config({})
    assert defaults["temperature"] == 1.0
    assert defaults["top_p"] == 0.95
    assert defaults["top_k"] == 20

    explicit = model._sanitize_generate_config(
        {"temperature": 0.2, "top_p": 0.7, "top_k": 5}
    )
    assert explicit["temperature"] == 0.2
    assert explicit["top_p"] == 0.7
    assert explicit["top_k"] == 5

    model._update_model_generation_config(
        {
            "generation_config": {
                "temperature": 0.8,
                "top_p": None,
                "top_k": None,
            },
            "temperature": None,
            "top_p": 0.95,
        }
    )
    nullable_defaults = model._sanitize_generate_config({})
    assert nullable_defaults["temperature"] == 0.8
    assert nullable_defaults["top_p"] == 0.95
    assert nullable_defaults["top_k"] == 0

    float_top_k = model._sanitize_generate_config({"top_k": 20.0})
    assert float_top_k["top_k"] == 20


def test_mlx_batch_generator_normalizes_top_k():
    from ..core import MLXBatchModel

    model = object.__new__(MLXBatchModel)
    float_top_k_generator = {"generator": object()}
    none_top_k_generator = {"generator": object()}
    original_generators = MLXBatchModel._batch_generators
    try:
        MLXBatchModel._batch_generators = {
            (0.7, 0.9, 20): float_top_k_generator,
            (0.7, 0.9, 0): none_top_k_generator,
        }
        assert model._get_or_create_generator(0.7, 0.9, 20.0) is float_top_k_generator
        assert model._get_or_create_generator(0.7, 0.9, None) is none_top_k_generator
    finally:
        MLXBatchModel._batch_generators = original_generators


def test_mlx_batch_generator_reuses_prompt_cache(monkeypatch):
    from ..core import MLXBatchModel

    cached_state = [object()]

    class FakePromptCache:
        def fetch_nearest_cache(self, model_key, prompt_tokens):
            assert model_key == "model-key"
            assert prompt_tokens == [1, 2, 3, 4, 5]
            return cached_state, [4, 5]

    class FakeBatchGenerator:
        def __init__(self):
            self.calls = []

        def insert(self, prompts, **kwargs):
            self.calls.append((prompts, kwargs))
            return [7]

    model = object.__new__(MLXBatchModel)
    model._prompt_cache = FakePromptCache()
    model._prompt_cache_model_key = "model-key"
    monkeypatch.setattr(model, "_is_new_mlx_lm", lambda: True)
    generator = FakeBatchGenerator()

    assert model._insert_request(generator, [1, 2, 3, 4, 5], 32, None) == (
        7,
        3,
        None,
    )
    assert generator.calls == [
        (
            [[4, 5]],
            {
                "max_tokens": [32],
                "caches": [cached_state],
                "all_tokens": [[1, 2, 3]],
            },
        )
    ]


def test_mlx_batch_generator_checkpoints_reusable_prefix(monkeypatch):
    from ..core import MLXBatchModel

    cached_state = [object()]

    class FakePromptCache:
        def fetch_nearest_cache(self, model_key, prompt_tokens):
            return cached_state, prompt_tokens[3:]

    class FakeBatchGenerator:
        def __init__(self):
            self.calls = []

        def insert_segments(self, segments, **kwargs):
            self.calls.append((segments, kwargs))
            return [8]

    model = object.__new__(MLXBatchModel)
    model._prompt_cache = FakePromptCache()
    model._prompt_cache_model_key = "model-key"
    monkeypatch.setattr(model, "_is_new_mlx_lm", lambda: True)
    generator = FakeBatchGenerator()

    assert model._insert_request(generator, [1, 2, 3, 4, 5, 6], 32, 5) == (
        8,
        3,
        5,
    )
    assert generator.calls == [
        (
            [[[4, 5], [6]]],
            {
                "max_tokens": [32],
                "caches": [cached_state],
                "all_tokens": [[1, 2, 3]],
            },
        )
    ]


def test_mlx_batch_generator_stores_reusable_prefix():
    from ..core import MLXBatchModel

    class FakePromptCache:
        def __init__(self):
            self.inserted = []

        def insert_cache(self, *args, **kwargs):
            self.inserted.append((args, kwargs))

        def __len__(self):
            return len(self.inserted)

    cached_state = [object()]

    class FakeBatchGenerator:
        def extract_cache(self, uids):
            assert uids == [9]
            return {9: (cached_state, [1, 2, 3])}

    model = object.__new__(MLXBatchModel)
    model._prompt_cache = FakePromptCache()
    model._prompt_cache_model_key = "model-key"
    gen_dict = {"cache_boundaries": {9: 3}}

    model._store_segment_prompt_caches(
        FakeBatchGenerator(),
        [SimpleNamespace(uid=9, end_of_segment=True)],
        gen_dict,
    )

    assert gen_dict["cache_boundaries"] == {}
    assert model._prompt_cache.inserted == [
        (
            ("model-key", [1, 2, 3], cached_state),
            {"cache_type": "system"},
        )
    ]


@pytest.mark.asyncio
async def test_mlx_batch_generator_reports_cached_tokens(monkeypatch):
    from ..core import MLXBatchModel

    model = object.__new__(MLXBatchModel)
    gen_dict = {
        "generator": object(),
        "queues": {},
        "pending": {},
        "active": set(),
        "cache_boundaries": {},
    }
    monkeypatch.setattr(model, "_get_or_create_generator", lambda *args: gen_dict)
    monkeypatch.setattr(model, "_insert_request", lambda *args: (7, 3, None))

    async def feed_result():
        while 7 not in gen_dict["queues"]:
            await asyncio.sleep(0)
        await gen_dict["queues"][7].put(
            SimpleNamespace(uid=7, token=10, finish_reason="stop")
        )

    def ensure_background_worker(_gen_dict):
        asyncio.create_task(feed_result())

    monkeypatch.setattr(model, "_ensure_background_worker", ensure_background_worker)
    original_tokenizer = MLXBatchModel._tokenizer_ref
    MLXBatchModel._tokenizer_ref = SimpleNamespace(
        encode=lambda prompt: [1, 2, 3, 4, 5],
        decode=lambda tokens, **kwargs: "x",
    )
    try:
        chunks = [
            chunk
            async for chunk in model.generate_stream(
                "prompt", 1, prompt_cache_prefix_len=3
            )
        ]
    finally:
        MLXBatchModel._tokenizer_ref = original_tokenizer

    assert len(chunks) == 2
    assert all(
        chunk["usage"]["prompt_tokens_details"] == {"cached_tokens": 3}
        for chunk in chunks
    )


def test_mlx_vision_model_checkpoints_reusable_prompt_prefix(monkeypatch):
    from ..core import MLXVisionModel

    class FakePromptCache:
        def __init__(self):
            self.inserted = []

        def insert_cache(self, *args, **kwargs):
            self.inserted.append((args, kwargs))

        def __len__(self):
            return len(self.inserted)

    model = object.__new__(MLXVisionModel)
    model._reusable_prompt_cache = FakePromptCache()
    model._reusable_prompt_cache_model_key = "vision-model-key"
    initial_cache = [{"state": "initial"}]
    monkeypatch.setattr(
        model,
        "_fetch_reusable_prompt_cache",
        lambda prompt_tokens: (initial_cache, prompt_tokens, 0),
    )

    remaining_tokens, cache_kwargs, cached_tokens = (
        model._prepare_reusable_prompt_cache([1, 2, 3, 4, 5], 3)
    )

    assert remaining_tokens == [1, 2, 3, 4, 5]
    assert cached_tokens == 0
    assert cache_kwargs["prompt_cache"] is initial_cache
    assert cache_kwargs["prompt_cache_checkpoint_len"] == 3

    checkpoint_cache = [{"state": "prefix"}]
    cache_kwargs["prompt_cache_checkpoint"](3, checkpoint_cache)
    checkpoint_cache[0]["state"] = "mutated"

    assert model._reusable_prompt_cache.inserted == [
        (
            ("vision-model-key", [1, 2, 3], [{"state": "prefix"}]),
            {"cache_type": "system"},
        )
    ]


def test_mlx_generate_stream_passes_top_k():
    from ..core import MLXModel

    model = object.__new__(MLXModel)
    model.model_uid = "top-k-test"
    model._tokenizer = SimpleNamespace(eos_token_id=99)
    model._context_length = 128
    model._prompt_cache = None
    model._prepare_inputs = lambda prompt, kwargs: ([4, 5], 5, 3)

    captured = {}

    def fake_generate_stream_inner(**kwargs):
        captured.update(kwargs)
        yield SimpleNamespace(token=10, text="ok")

    model._generate_stream_inner = fake_generate_stream_inner
    results = list(
        model._generate_stream(
            "hello",
            {
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "repetition_penalty": None,
                "repetition_context_size": 20,
                "stop_token_ids": [],
                "stream": True,
            },
        )
    )

    assert captured["temperature"] == 1.0
    assert captured["top_p"] == 0.95
    assert captured["top_k"] == 20
    assert all(
        usage["prompt_tokens_details"] == {"cached_tokens": 3} for _, usage in results
    )
    assert all(
        chunk["usage"]["prompt_tokens_details"] == {"cached_tokens": 3}
        for chunk, _ in results
    )


def test_mlx_vision_chat_marks_reusable_text_prompt_prefix(monkeypatch):
    from ..core import MLXVisionModel

    class FakeTokenizer:
        @staticmethod
        def encode(prompt):
            return list(prompt.encode())

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=lambda messages: ([], None)),
    )
    model = object.__new__(MLXVisionModel)
    model.model_uid = "vision-prompt-prefix-test"
    model.model_family = SimpleNamespace(
        model_family="qwen3.8",
        model_name="qwen3.8",
        model_ability=["chat", "vision"],
        chat_template="test-template",
        stop=None,
        stop_token_ids=None,
    )
    model.reasoning_parser = None
    model._tokenizer = FakeTokenizer()
    model._transform_messages = lambda messages: messages

    def get_full_context(messages, *args, **kwargs):
        tool_names = ",".join(
            tool["function"]["name"] for tool in kwargs.get("tools", [])
        )
        return (
            f"tools:{tool_names}:stable-prefix:"
            + messages[-1]["content"]
            + ":generation-suffix"
        )

    model.get_full_context = get_full_context
    model._sanitize_generate_config = lambda config: config
    captured = {}

    def fake_generate(prompt, generate_config):
        captured["prompt"] = prompt
        captured["generate_config"] = generate_config
        return {"completion": "ok"}

    model.generate = fake_generate
    model._to_chat_completion = lambda completion, reasoning_parser: completion
    model._post_process_completion = lambda family, uid, completion: completion

    result = model.chat(
        [{"role": "user", "content": "dynamic question"}],
        {"tools": iter([{"type": "function", "function": {"name": "search"}}])},
    )

    assert result == {"completion": "ok"}
    assert captured["prompt"] == {
        "prompt": "tools:search:stable-prefix:dynamic question:generation-suffix"
    }
    assert captured["generate_config"]["prompt_cache_prefix_len"] == len(
        "tools:search:stable-prefix:"
    )


@pytest.mark.asyncio
async def test_mlx_chat_marks_reusable_prompt_prefix():
    from ..core import MLXChatModel

    class FakeTokenizer:
        chat_template = "test-template"

        @staticmethod
        def encode(prompt):
            return list(prompt.encode())

    model = object.__new__(MLXChatModel)
    model.model_uid = "prompt-prefix-test"
    model.model_family = SimpleNamespace(
        model_family="qwen3",
        model_name="qwen3",
        model_ability=["chat"],
        chat_template="test-template",
        stop=None,
        stop_token_ids=None,
    )
    model.reasoning_parser = None
    model._tokenizer = FakeTokenizer()

    def get_full_context(messages, *args, **kwargs):
        tool_names = ",".join(
            tool["function"]["name"] for tool in kwargs.get("tools", [])
        )
        return (
            f"tools:{tool_names}:stable-prefix:"
            + messages[-1]["content"]
            + ":generation-suffix"
        )

    model.get_full_context = get_full_context
    model._sanitize_generate_config = lambda config: config
    captured = {}

    async def fake_async_generate(prompt, generate_config):
        captured["prompt"] = prompt
        captured["generate_config"] = generate_config
        return {"completion": "ok"}

    model.async_generate = fake_async_generate
    model._to_chat_completion = lambda completion, reasoning_parser: completion
    model._post_process_completion = lambda family, uid, completion: completion

    result = await model.async_chat(
        [{"role": "user", "content": "dynamic question"}],
        {"tools": iter([{"type": "function", "function": {"name": "search"}}])},
    )

    assert result == {"completion": "ok"}
    assert captured["prompt"] == (
        "tools:search:stable-prefix:dynamic question:generation-suffix"
    )
    assert captured["generate_config"]["prompt_cache_prefix_len"] == len(
        "tools:search:stable-prefix:"
    )


@pytest.mark.asyncio
async def test_mlx_streaming_parses_multiple_qwen_tool_calls():
    from ...reasoning_parser import ReasoningParser
    from ...tool_parsers.qwen_tool_parser import QwenToolParser
    from ..core import MLXVisionModel

    model = object.__new__(MLXVisionModel)
    model.model_uid = "qwen3.8-mlx-test"
    model.model_family = SimpleNamespace(model_name="qwen3.8")
    model.reasoning_parser = ReasoningParser(
        reasoning_content=True,
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
        enable_thinking=True,
    )
    model.tool_parser = QwenToolParser()

    def chunk(text, finish_reason=None):
        return {
            "id": "task-835",
            "object": "text_completion",
            "created": 1,
            "model": model.model_uid,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }

    def make_chunks():
        return [
            chunk("<think>"),
            chunk("I should"),
            chunk(
                " search twice.</think>\n\n<tool_call>\n"
                "<function=web_search>\n"
                "<parameter=query>Dario"
            ),
            chunk(
                " Amodei recent news</parameter>\n"
                "<parameter=num_results>10</parameter>\n"
                "</function>\n"
            ),
            chunk("</tool_call>\n"),
            chunk("<tool_call>"),
            chunk(
                "\n<function=web_search>\n"
                "<parameter=query>Dario Amodei gossip</parameter>\n"
                "<parameter=num_results>10</parameter>\n"
                "</function>\n"
            ),
            chunk("</tool_call>"),
            chunk("", "stop"),
        ]

    def assert_results(results):
        tool_calls = [
            tool_call
            for result in results
            for choice in result["choices"]
            for tool_call in choice["delta"].get("tool_calls", [])
        ]

        assert [tool_call["index"] for tool_call in tool_calls] == [0, 1]
        assert [tool_call["function"]["name"] for tool_call in tool_calls] == [
            "web_search",
            "web_search",
        ]
        assert [
            json.loads(tool_call["function"]["arguments"])["query"]
            for tool_call in tool_calls
        ] == ["Dario Amodei recent news", "Dario Amodei gossip"]
        assert tool_calls[0]["id"] != tool_calls[1]["id"]
        assert results[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert (
            "".join(
                choice["delta"].get("reasoning_content") or ""
                for result in results
                for choice in result["choices"]
            )
            == "I should search twice."
        )
        assert all(
            "<tool_call>" not in (choice["delta"].get("content") or "")
            for result in results
            for choice in result["choices"]
        )

    assert_results(list(model._to_tool_completion_chunks(iter(make_chunks()))))

    async def async_chunks():
        for value in make_chunks():
            yield value

    async_results = [
        result
        async for result in model._async_to_tool_completion_chunks(async_chunks())
    ]
    assert_results(async_results)


def test_mlx_vision_model_stop_shuts_down_executor():
    from ..core import MLXVisionModel

    model = object.__new__(MLXVisionModel)
    model.model_uid = "shutdown-test"
    model._mlx_executor = None

    worker_name = model._run_on_mlx_thread(lambda: threading.current_thread().name)
    executor = model._mlx_executor

    assert worker_name.startswith("mlx-shutdown-test")
    assert executor is not None

    model.stop()

    assert model._mlx_executor is None
    with pytest.raises(RuntimeError, match="cannot schedule new futures"):
        executor.submit(lambda: None)

    # Teardown can safely be retried if initialization only partially completed.
    model.stop()


def test_mlx_vision_draft_generate_kwargs():
    from ..core import MLXVisionModel

    model = object.__new__(MLXVisionModel)
    model._draft_model = None
    model._draft_kind = None
    model._draft_block_size = None

    assert model._draft_generate_kwargs() == {}

    drafter = object()
    model._draft_model = drafter
    model._draft_kind = "mtp"

    # without an explicit size, the drafter's own block size is kept
    assert model._draft_generate_kwargs() == {
        "draft_model": drafter,
        "draft_kind": "mtp",
    }

    model._draft_block_size = 6
    assert model._draft_generate_kwargs() == {
        "draft_model": drafter,
        "draft_kind": "mtp",
        "draft_block_size": 6,
    }


def test_mlx_text_model_rejects_drafter():
    from ..core import MLXChatModel

    model = object.__new__(MLXChatModel)
    model._model_config = {
        "draft_model_path": "/tmp/gemma-4-assistant",
        "num_speculative_tokens": 3,
    }

    with pytest.raises(ValueError, match="only supported by the MLX vision engine"):
        model._load_model()

    # speculative options must not leak into the mlx_lm model config
    assert model._model_config == {}


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.processor() != "arm",
    reason="MLX only works for Apple silicon chip",
)
def test_mlx_vlm_generation_stream_is_thread_local(monkeypatch):
    import mlx.core as mx

    if not hasattr(mx, "ThreadLocalStream") or not hasattr(
        mx, "new_thread_local_stream"
    ):
        pytest.skip("MLX version does not support thread-local streams")

    from ..core import _ensure_mlx_vlm_thread_local_stream

    mlx_vlm_generate = importlib.import_module("mlx_vlm.generate")
    monkeypatch.setattr(
        mlx_vlm_generate,
        "generation_stream",
        mx.new_stream(mx.default_device()),
    )

    _ensure_mlx_vlm_thread_local_stream()

    assert isinstance(mlx_vlm_generate.generation_stream, mx.ThreadLocalStream)

    def generate_on_worker_thread():
        with mx.stream(mlx_vlm_generate.generation_stream):
            result = mx.arange(4) + 1
        mx.async_eval(result)
        return result.tolist()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(generate_on_worker_thread).result() == [1, 2, 3, 4]


class InferenceThread(threading.Thread):
    """Thread for running parallel inference requests."""

    def __init__(self, prompt, generate_config, model):
        super().__init__()
        self._prompt = [{"role": "user", "content": prompt}]
        self._generate_config = generate_config
        self._model = model
        self._ex = None
        self._result = None

    def run(self):
        try:
            if self._generate_config.get("stream", False):
                results = []
                for res in self._model.chat(
                    self._prompt, generate_config=self._generate_config
                ):
                    results.append(res)
                assert len(results) > 0
                self._result = results[-1]
            else:
                res = self._model.chat(
                    self._prompt, generate_config=self._generate_config
                )
                assert isinstance(res, dict)
                choices = res["choices"]
                assert isinstance(choices, list)
                choice = choices[0]["message"]
                assert isinstance(choice, dict)
                content = choice["content"]
                assert len(content) > 0
                self._result = res
        except BaseException as e:
            self._ex = e

    def join(self, timeout=None):
        super().join(timeout)
        if self._ex is not None:
            raise self._ex
        return self._result


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.processor() != "arm",
    reason="MLX only works for Apple silicon chip",
)
def test_load_mlx(setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="qwen2-instruct",
        model_engine="MLX",
        model_size_in_billions="0_5",
        model_format="mlx",
        quantization="4bit",
    )
    assert len(client.list_models()) == 1
    model = client.get_model(model_uid)
    messages = [{"role": "user", "content": "write a poem."}]
    completion = model.chat(messages)
    assert "content" in completion["choices"][0]["message"]
    assert len(completion["choices"][0]["message"]["content"]) != 0
    content = completion["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": "explain it"})
    completion = model.chat(messages)
    assert "content" in completion["choices"][0]["message"]
    assert len(completion["choices"][0]["message"]["content"]) != 0


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.processor() != "arm",
    reason="MLX only works for Apple silicon chip",
)
def test_load_mlx_vision(setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="qwen2-vl-instruct",
        model_engine="MLX",
        model_size_in_billions=2,
        model_format="mlx",
        quantization="4bit",
    )
    assert len(client.list_models()) == 1
    model = client.get_model(model_uid)

    path = os.path.join(os.path.dirname(__file__), "fish.png")
    with open(path, "rb") as f:
        content = f.read()
    b64_img = base64.b64encode(content).decode("utf-8")

    completion = model.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图中有几条鱼？"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}",
                        },
                    },
                ],
            }
        ],
        generate_config={"max_tokens": 100},
    )
    assert "content" in completion["choices"][0]["message"]
    assert len(completion["choices"][0]["message"]["content"]) != 0

    # test no image
    messages = [{"role": "user", "content": "write a poem."}]
    completion = model.chat(messages)
    assert "content" in completion["choices"][0]["message"]
    assert "content" in completion["choices"][0]["message"]
    assert len(completion["choices"][0]["message"]["content"]) != 0

    chunks = list(
        model.chat(
            messages,
            generate_config={
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": 4,
            },
        )
    )
    assert any(chunk.get("choices") for chunk in chunks)


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.processor() != "arm",
    reason="MLX only works for Apple silicon chip",
)
def test_mlx_parallel_inference(setup):
    """Test MLX continuous batching with parallel inference requests."""
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="qwen2-instruct",
        model_engine="MLX",
        model_size_in_billions="0_5",
        model_format="mlx",
        quantization="4bit",
    )
    assert len(client.list_models()) == 1
    model = client.get_model(model_uid)

    # Test parallel streaming and non-streaming requests
    thread1 = InferenceThread("1+1等于几？", {"stream": True}, model)
    thread2 = InferenceThread("中国的首都是哪里？", {"stream": False}, model)
    thread3 = InferenceThread("介绍一下Python。", {"stream": True}, model)

    # Start all threads
    thread1.start()
    thread2.start()
    thread3.start()

    # Wait for all to complete
    result1 = thread1.join()
    result2 = thread2.join()
    result3 = thread3.join()

    # Verify results
    assert result1 is not None
    assert result2 is not None
    assert result3 is not None

    # Check streaming results (should use 'delta' format)
    assert "choices" in result1
    assert len(result1["choices"]) > 0
    assert "delta" in result1["choices"][0]
    # Streaming can have empty content in last chunk (finish_reason only)
    assert result1["choices"][0]["finish_reason"] in ["stop", "length"]

    # Check non-streaming results (should use 'message' format)
    assert "choices" in result2
    assert len(result2["choices"]) > 0
    assert "message" in result2["choices"][0]
    assert "content" in result2["choices"][0]["message"]

    # Check second streaming results (should use 'delta' format)
    assert "choices" in result3
    assert len(result3["choices"]) > 0
    assert "delta" in result3["choices"][0]
    assert result3["choices"][0]["finish_reason"] in ["stop", "length"]
