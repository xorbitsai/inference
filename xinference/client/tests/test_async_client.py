# Copyright 2022-2023 XProbe Inc.
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


@pytest.mark.skipif(os.name == "nt", reason="Skip windows")
@pytest.mark.asyncio
async def test_async_list_cached_models(setup):
    endpoint, _ = setup
    client = AsyncRESTfulClient(endpoint)
    res = await client.list_cached_models()
    assert len(res) > 0


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


@pytest.mark.asyncio
async def test_async_RESTful_client_custom_model(setup):
    endpoint, _ = setup
    client = AsyncRESTfulClient(endpoint)

    model_regs = await client.list_model_registrations(model_type="LLM")
    assert len(model_regs) > 0
    for model_reg in model_regs:
        assert model_reg["is_builtin"]

    model = """{
  "version": 1,
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
      "quantizations": [
        "4-bit",
        "8-bit",
        "none"
      ],
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
  "version": 1,
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
      "quantizations": [
        "4-bit",
        "8-bit",
        "none"
      ],
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
      "version": 1,
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
          "quantizations": [
            "4-bit",
            "8-bit",
            "none"
          ],
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
          "version": 1,
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
              "quantizations": [
                "4-bit",
                "8-bit",
                "none"
              ],
              "model_id": "ziqingyang/chinese-alpaca-2-7b"
            }
          ],
          "chat_template": "xyz123"
        }"""
    with pytest.raises(RuntimeError):
        await client.register_model(
            model_type="LLM", model=model_with_tool_call, persist=False
        )


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
  "model_id": "Xorbits/bge-small-en"
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


@pytest.mark.asyncio
async def test_async_auto_recover(set_auto_recover_limit, setup_cluster):
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


@pytest.mark.asyncio
async def test_async_model_error(set_test_oom_error, setup_cluster):
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

    with pytest.raises(RuntimeError, match="Model actor is out of memory"):
        await model.generate("Once upon a time, there was a very old computer")
