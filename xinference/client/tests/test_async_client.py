# Copyright 2022-2026 XProbe Inc.
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
import os
import time

import psutil
import pytest

from ...constants import XINFERENCE_ENV_MODEL_SRC
from ..restful.async_restful_client import AsyncClient as AsyncRESTfulClient
from ..restful.async_restful_client import (
    AsyncRESTfulChatModelHandle,
    AsyncRESTfulEmbeddingModelHandle,
)


class _DummyAsyncResponse:
    status = 200

    async def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}

    def release(self):
        return None

    async def wait_for_close(self):
        return None


class _DummyAsyncSession:
    def __init__(self):
        self.last_json = None

    async def post(self, url, json=None, headers=None):
        self.last_json = json
        return _DummyAsyncResponse()

    async def close(self):
        return None


@pytest.mark.skipif(os.name == "nt", reason="Skip windows")
@pytest.mark.asyncio
async def test_async_RESTful_client(setup):
    endpoint, _ = setup
    client = AsyncRESTfulClient(endpoint)
    assert len(await client.list_models()) == 0

    model_uid = await client.launch_model(
        model_name="qwen1.5-chat",
        model_engine="llama.cpp",
        model_size_in_billions="0_5",
        quantization="q4_0",
    )
    assert len(await client.list_models()) == 1

    model = await client.get_model(model_uid=model_uid)
    assert isinstance(model, AsyncRESTfulChatModelHandle)

    with pytest.raises(RuntimeError):
        model = await client.get_model(model_uid="test")
        assert isinstance(model, AsyncRESTfulChatModelHandle)

    with pytest.raises(RuntimeError):
        completion = await model.generate({"max_tokens": 64})

    completion = await model.generate("Once upon a time, there was a very old computer")
    assert "text" in completion["choices"][0]

    completion = await model.generate(
        "Once upon a time, there was a very old computer", {"max_tokens": 64}
    )
    assert "text" in completion["choices"][0]

    streaming_response = await model.generate(
        "Once upon a time, there was a very old computer",
        {"max_tokens": 64, "stream": True},
    )

    async for chunk in streaming_response:
        assert "text" in chunk["choices"][0]

    with pytest.raises(RuntimeError):
        completion = await model.chat({"max_tokens": 64})

    messages = [{"role": "user", "content": "What is the capital of France?"}]
    completion = await model.chat(messages)
    assert "content" in completion["choices"][0]["message"]

    async def _check_stream():
        streaming_response = await model.chat(
            messages,
            generate_config={"stream": True, "max_tokens": 5},
        )
        async for chunk in streaming_response:
            if not chunk["choices"]:
                assert chunk["usage"]
                continue
            assert "finish_reason" in chunk["choices"][0]
            finish_reason = chunk["choices"][0]["finish_reason"]
            if finish_reason is None:
                assert ("content" in chunk["choices"][0]["delta"]) or (
                    "role" in chunk["choices"][0]["delta"]
                )
            else:
                assert chunk["choices"][0]["delta"] == {"content": ""}

    await _check_stream()

    tasks = [asyncio.create_task(_check_stream()) for _ in range(3)]
    await asyncio.gather(*tasks)

    await client.terminate_model(model_uid=model_uid)
    assert len(await client.list_models()) == 0

    model_uid = await client.launch_model(
        model_name="tiny-llama",
        model_engine="llama.cpp",
        model_size_in_billions=1,
        model_format="ggufv2",
        quantization="q2_K",
    )
    assert len(await client.list_models()) == 1

    # Test concurrent chat is OK.
    await model.close()
    model = await client.get_model(model_uid=model_uid)

    async def _check(stream=False):
        completion = await model.generate(
            "AI is going to", generate_config={"stream": stream, "max_tokens": 5}
        )
        if stream:
            count = 0
            has_text = False
            async for chunk in completion:
                assert "text" in chunk["choices"][0]
                if chunk["choices"][0]["text"]:
                    has_text = True
                count += 1
            assert has_text
            assert count > 2
        else:
            assert "text" in completion["choices"][0]
            assert len(completion["choices"][0]["text"]) > 0

    for stream in [True, False]:
        # Async concurrent execution of _check
        tasks = [asyncio.create_task(_check(stream=stream)) for _ in range(3)]
        await asyncio.gather(*tasks)

    await client.terminate_model(model_uid=model_uid)
    assert len(await client.list_models()) == 0

    with pytest.raises(RuntimeError):
        await client.terminate_model(model_uid=model_uid)

    model_uid2 = await client.launch_model(
        model_name="qwen1.5-chat",
        model_engine="llama.cpp",
        model_size_in_billions="0_5",
        quantization="q4_0",
    )
    await client.terminate_model(model_uid=model_uid2)
    assert len(await client.list_models()) == 0
    await model.close()
    await client.close()


@pytest.mark.asyncio
async def test_async_query_engines_by_name(setup):
    endpoint, _ = setup
    client = AsyncRESTfulClient(endpoint)

    assert len(await client.query_engine_by_model_name("qwen3")) > 0
    assert len(await client.query_engine_by_model_name("qwen3", model_type=None)) > 0
    assert (
        len(await client.query_engine_by_model_name("bge-m3", model_type="embedding"))
        > 0
    )
    await client.close()


@pytest.mark.skipif(os.name == "nt", reason="Skip windows")
@pytest.mark.asyncio
async def test_async_list_cached_models(setup):
    endpoint, _ = setup
    client = AsyncRESTfulClient(endpoint)
    res = await client.list_cached_models()
    assert len(res) > 0
    await client.close()


@pytest.mark.asyncio
async def test_async_RESTful_client_for_embedding(setup):
    endpoint, _ = setup
    client = AsyncRESTfulClient(endpoint)
    assert len(await client.list_models()) == 0

    model_uid = await client.launch_model(model_name="gte-base", model_type="embedding")
    assert len(await client.list_models()) == 1

    model = await client.get_model(model_uid=model_uid)
    assert isinstance(model, AsyncRESTfulEmbeddingModelHandle)

    completion = await model.create_embedding("write a poem.")
    assert len(completion["data"][0]["embedding"]) == 768

    await client.terminate_model(model_uid=model_uid)
    assert len(await client.list_models()) == 0
    await model.close()
    await client.close()


@pytest.mark.asyncio
async def test_async_RESTful_client_custom_model(setup):
    endpoint, _ = setup
    client = AsyncRESTfulClient(endpoint)

    model_regs = await client.list_model_registrations(model_type="LLM")
    assert len(model_regs) > 0
    for model_reg in model_regs:
        assert model_reg["is_builtin"]

    model = """{
  "version": 2,
  "context_length":2048,
  "model_name": "custom_model",
  "model_lang": [
    "en", "zh"
  ],
  "model_ability": [
    "chat"
  ],
  "model_family": "other",
  "model_specs": [
    {
      "model_format": "pytorch",
      "model_size_in_billions": 7,
      "quantization": "none",
      "model_id": "ziqingyang/chinese-alpaca-2-7b"
    }
  ],
  "chat_template": "xyz",
  "stop_token_ids": [],
  "stop": []
}"""
    await client.register_model(model_type="LLM", model=model, persist=False)

    data = await client.get_model_registration(
        model_type="LLM", model_name="custom_model"
    )
    assert "custom_model" in data["model_name"]

    new_model_regs = await client.list_model_registrations(model_type="LLM")
    assert len(new_model_regs) == len(model_regs) + 1
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom_model":
            custom_model_reg = model_reg
    assert custom_model_reg is not None

    await client.unregister_model(model_type="LLM", model_name="custom_model")
    new_model_regs = await client.list_model_registrations(model_type="LLM")
    assert len(new_model_regs) == len(model_regs)
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom_model":
            custom_model_reg = model_reg
    assert custom_model_reg is None

    # test register with chat_template using model_family
    model_with_prompt = """{
  "version": 2,
  "context_length":2048,
  "model_name": "custom_model",
  "model_lang": [
    "en", "zh"
  ],
  "model_ability": [
    "embed",
    "chat"
  ],
  "model_family": "other",
  "model_specs": [
    {
      "model_format": "pytorch",
      "model_size_in_billions": 7,
      "quantization": "none",
      "model_id": "ziqingyang/chinese-alpaca-2-7b"
    }
  ],
  "chat_template": "qwen-chat"
}"""
    await client.register_model(
        model_type="LLM", model=model_with_prompt, persist=False
    )
    await client.unregister_model(model_type="LLM", model_name="custom_model")

    model_with_vision = """{
      "version": 2,
      "context_length":2048,
      "model_name": "custom_model",
      "model_lang": [
        "en", "zh"
      ],
      "model_ability": [
        "chat",
        "vision"
      ],
      "model_family": "other",
      "model_specs": [
        {
          "model_format": "pytorch",
          "model_size_in_billions": 7,
          "quantization": "none",
          "model_id": "ziqingyang/chinese-alpaca-2-7b"
        }
      ],
      "chat_template": "xyz123"
    }"""
    with pytest.raises(RuntimeError):
        await client.register_model(
            model_type="LLM", model=model_with_vision, persist=False
        )

    model_with_tool_call = """{
          "version": 2,
          "context_length":2048,
          "model_name": "custom_model",
          "model_lang": [
            "en", "zh"
          ],
          "model_ability": [
            "chat",
            "tools"
          ],
          "model_family": "other",
          "model_specs": [
            {
              "model_format": "pytorch",
              "model_size_in_billions": 7,
              "quantization": "none",
              "model_id": "ziqingyang/chinese-alpaca-2-7b"
            }
          ],
          "chat_template": "xyz123"
        }"""
    with pytest.raises(RuntimeError):
        await client.register_model(
            model_type="LLM", model=model_with_tool_call, persist=False
        )
    await client.close()


@pytest.mark.asyncio
async def test_async_client_from_modelscope(setup):
    try:
        os.environ[XINFERENCE_ENV_MODEL_SRC] = "modelscope"

        endpoint, _ = setup
        client = AsyncRESTfulClient(endpoint)
        assert len(await client.list_models()) == 0

        model_uid = await client.launch_model(
            model_name="tiny-llama", model_engine="llama.cpp"
        )
        assert len(await client.list_models()) == 1
        model = await client.get_model(model_uid=model_uid)
        completion = await model.generate(
            "write a poem.", generate_config={"max_tokens": 5}
        )
        assert "text" in completion["choices"][0]
        assert len(completion["choices"][0]["text"]) > 0
    finally:
        os.environ.pop(XINFERENCE_ENV_MODEL_SRC)
    await model.close()
    await client.close()


@pytest.mark.asyncio
async def test_async_client_custom_embedding_model(setup):
    endpoint, _ = setup
    client = AsyncRESTfulClient(endpoint)

    model_regs = await client.list_model_registrations(model_type="embedding")
    assert len(model_regs) > 0
    for model_reg in model_regs:
        assert model_reg["is_builtin"]

    model = """{
  "model_name": "custom-bge-small-en",
  "dimensions": 1024,
  "max_tokens": 512,
  "language": ["en"],
  "model_specs": [
    {
      "model_format": "pytorch",
      "model_id": "Xorbits/bge-small-en",
      "quantization": "none"
    }
  ]
}"""
    await client.register_model(model_type="embedding", model=model, persist=False)

    data = await client.get_model_registration(
        model_type="embedding", model_name="custom-bge-small-en"
    )
    assert "custom-bge-small-en" in data["model_name"]

    new_model_regs = await client.list_model_registrations(model_type="embedding")
    assert len(new_model_regs) == len(model_regs) + 1
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom-bge-small-en":
            custom_model_reg = model_reg
    assert custom_model_reg is not None
    assert not custom_model_reg["is_builtin"]

    # unregister
    await client.unregister_model(
        model_type="embedding", model_name="custom-bge-small-en"
    )
    new_model_regs = await client.list_model_registrations(model_type="embedding")
    assert len(new_model_regs) == len(model_regs)
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom-bge-small-en":
            custom_model_reg = model_reg
    assert custom_model_reg is None
    await client.close()


@pytest.mark.asyncio
async def test_async_rerank(setup):
    endpoint, _ = setup
    client = AsyncRESTfulClient(endpoint)

    model_uid = await client.launch_model(
        model_name="bge-reranker-base", model_type="rerank"
    )
    assert len(await client.list_models()) == 1
    model = await client.get_model(model_uid)
    # We want to compute the similarity between the query sentence
    query = "A man is eating pasta."

    # With all sentences in the corpus
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]

    scores = await model.rerank(corpus, query, return_documents=True)
    assert scores["results"][0]["index"] == 0
    assert scores["results"][0]["document"]["text"] == corpus[0]

    scores = await model.rerank(corpus, query, top_n=3, return_documents=True)
    assert len(scores["results"]) == 3
    assert scores["results"][0]["index"] == 0
    assert scores["results"][0]["document"]["text"] == corpus[0]

    scores = await model.rerank(corpus, query, return_len=True)
    assert (
        scores["meta"]["tokens"]["input_tokens"]
        == scores["meta"]["tokens"]["output_tokens"]
    )

    scores = await model.rerank(corpus, query)
    assert scores["meta"]["tokens"] == None

    # testing long input
    corpus2 = corpus.copy()
    corpus2[-1] = corpus2[-1] * 50
    scores = await model.rerank(corpus2, query, top_n=3, return_documents=True)
    assert len(scores["results"]) == 3
    assert scores["results"][0]["index"] == 0
    assert scores["results"][0]["document"]["text"] == corpus2[0]

    kwargs = {
        "invalid": "invalid",
    }
    with pytest.raises(RuntimeError):
        await model.rerank(corpus, query, **kwargs)
    await model.close()
    await client.close()


@pytest.fixture
def set_auto_recover_limit():
    os.environ["XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT"] = "1"
    yield
    del os.environ["XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT"]


@pytest.fixture
def set_test_oom_error():
    os.environ["XINFERENCE_TEST_OUT_OF_MEMORY_ERROR"] = "1"
    yield
    del os.environ["XINFERENCE_TEST_OUT_OF_MEMORY_ERROR"]


@pytest.fixture
def setup_cluster():
    import xoscar as xo

    from ...api.restful_api import run_in_subprocess as restful_api_run_in_subprocess
    from ...conftest import TEST_FILE_LOGGING_CONF, TEST_LOGGING_CONF, api_health_check
    from ...deploy.local import health_check
    from ...deploy.local import run_in_subprocess as supervisor_run_in_subprocess

    supervisor_address = f"localhost:{xo.utils.get_next_port()}"
    local_cluster = supervisor_run_in_subprocess(
        supervisor_address, None, None, TEST_LOGGING_CONF
    )

    if not health_check(address=supervisor_address, max_attempts=20, sleep_interval=1):
        raise RuntimeError("Supervisor is not available after multiple attempts")

    try:
        port = xo.utils.get_next_port()
        restful_api_proc = restful_api_run_in_subprocess(
            supervisor_address,
            host="localhost",
            port=port,
            logging_conf=TEST_FILE_LOGGING_CONF,
        )
        endpoint = f"http://localhost:{port}"
        if not api_health_check(endpoint, max_attempts=10, sleep_interval=5):
            raise RuntimeError("Endpoint is not available after multiple attempts")

        yield f"http://localhost:{port}", supervisor_address
        restful_api_proc.kill()
    finally:
        local_cluster.kill()


@pytest.mark.asyncio
async def test_auto_recover(set_auto_recover_limit, setup_cluster):
    endpoint, _ = setup_cluster
    current_proc = psutil.Process()
    chilren_proc = set(current_proc.children(recursive=True))
    client = AsyncRESTfulClient(endpoint)

    model_uid = await client.launch_model(
        model_name="qwen1.5-chat",
        model_engine="llama.cpp",
        model_size_in_billions="0_5",
        quantization="q4_0",
    )
    new_children_proc = set(current_proc.children(recursive=True))
    model_proc = next(iter(new_children_proc - chilren_proc))
    assert len(await client.list_models()) == 1

    model = await client.get_model(model_uid=model_uid)
    assert isinstance(model, AsyncRESTfulChatModelHandle)

    completion = await model.generate("Once upon a time, there was a very old computer")
    assert "text" in completion["choices"][0]

    model_proc.kill()

    for _ in range(60):
        try:
            completion = await model.generate(
                "Once upon a time, there was a very old computer", {"max_tokens": 64}
            )
            assert "text" in completion["choices"][0]
            break
        except Exception:
            time.sleep(1)
    else:
        assert False
    await model.close()
    await client.close()


@pytest.mark.asyncio
async def test_model_error(set_test_oom_error, setup_cluster):
    endpoint, _ = setup_cluster
    client = AsyncRESTfulClient(endpoint)

    model_uid = await client.launch_model(
        model_name="qwen1.5-chat",
        model_engine="llama.cpp",
        model_size_in_billions="0_5",
        quantization="q4_0",
    )
    assert len(await client.list_models()) == 1

    model = await client.get_model(model_uid=model_uid)
    assert isinstance(model, AsyncRESTfulChatModelHandle)

    with pytest.raises(RuntimeError):
        await model.generate("Once upon a time, there was a very old computer")
    await model.close()
    await client.close()


@pytest.mark.asyncio
async def test_async_restful_chat_enable_thinking_injected():
    handle = AsyncRESTfulChatModelHandle.__new__(AsyncRESTfulChatModelHandle)
    handle._model_uid = "test-model"
    handle._base_url = "http://localhost"
    handle.auth_headers = {}
    handle.session = _DummyAsyncSession()

    messages = [{"role": "user", "content": "hi"}]
    await handle.chat(messages, enable_thinking=True)

    assert handle.session.last_json["chat_template_kwargs"] == {
        "enable_thinking": True,
        "thinking": True,
    }
    handle.session = None
