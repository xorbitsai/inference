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
import os
import os.path

import openai
import pytest
import requests
from packaging import version

from ....constants import XINFERENCE_ENV_MODEL_SRC


@pytest.mark.asyncio
@pytest.mark.skip(reason="Too large model to be tested")
async def test_openai_stream_options_llamacpp_chatglm(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_restful_api",
        "model_engine": "llama.cpp",
        "model_name": "chatglm3",
        "model_size_in_billions": "6",
        "quantization": "q4_0",
        "model_format": "ggmlv3",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    # chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi what can I help you?"},
        {"role": "user", "content": "法国的首都是哪里?"},
    ]

    if version.parse(openai.__version__) < version.parse("1.0"):
        openai.api_key = ""
        openai.api_base = f"{endpoint}/v1"
        openai_chat_completion = openai.ChatCompletion.acreate
        stream_chunk_type_name = "OpenAIObject"
        response_type_name = "OpenAIObject"
    else:
        client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
        openai_chat_completion = client.chat.completions.create
        stream_chunk_type_name = "ChatCompletionChunk"
        response_type_name = "ChatCompletion"

    result1 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": False,
        },
    ):
        result1.append(chunk)
    assert result1
    assert type(result1[0]).__name__ == stream_chunk_type_name
    # stop
    assert result1[-1].choices[0].finish_reason == "stop"

    result2 = await openai_chat_completion(
        messages=messages, stream=False, model=model_uid_res
    )

    assert result2
    assert type(result2).__name__ == response_type_name
    # stop
    assert result2.choices[0].finish_reason == "stop"

    result3 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": True,
        },
    ):
        result3.append(chunk)
    assert result3
    assert type(result3[0]).__name__ == stream_chunk_type_name
    # usage
    assert not result3[-1].choices
    assert result3[-1].usage


@pytest.mark.asyncio
@pytest.mark.skip(reason="Too large model to be tested")
async def test_openai_stream_options_llamacpp(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"
    # os.environ.get(XINFERENCE_ENV_MODEL_SRC, "modelscope")

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_restful_api",
        "model_engine": "llama.cpp",
        "model_name": "qwen1.5-chat",
        "model_size_in_billions": "0_5",
        "model_format": "ggufv2",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    # chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi what can I help you?"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    if version.parse(openai.__version__) < version.parse("1.0"):
        openai.api_key = ""
        openai.api_base = f"{endpoint}/v1"
        openai_chat_completion = openai.ChatCompletion.acreate
        stream_chunk_type_name = "OpenAIObject"
        response_type_name = "OpenAIObject"
    else:
        client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
        openai_chat_completion = client.chat.completions.create
        stream_chunk_type_name = "ChatCompletionChunk"
        response_type_name = "ChatCompletion"

    result1 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": False,
        },
    ):
        result1.append(chunk)
    assert result1
    assert type(result1[0]).__name__ == stream_chunk_type_name
    # stop
    assert result1[-1].choices[0].finish_reason == "stop"

    result2 = await openai_chat_completion(
        messages=messages, stream=False, model=model_uid_res
    )

    assert result2
    assert type(result2).__name__ == response_type_name
    # stop
    assert result2.choices[0].finish_reason == "stop"

    result3 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": True,
        },
    ):
        result3.append(chunk)
    assert result3
    assert type(result3[0]).__name__ == stream_chunk_type_name
    # usage
    assert not result3[-1].choices
    assert result3[-1].usage


@pytest.mark.asyncio
@pytest.mark.skip(reason="Too large model to be tested")
async def test_openai_stream_options_pytorch_chatglm(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"
    os.environ.get(XINFERENCE_ENV_MODEL_SRC, "modelscope")

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_restful_api",
        "model_engine": "Transformers",
        "model_name": "chatglm3",
        "model_size_in_billions": "6",
        "quantization": "none",
        "model_format": "pytorch",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    # chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi what can I help you?"},
        {"role": "user", "content": "法国的首都是哪里?"},
    ]

    if version.parse(openai.__version__) < version.parse("1.0"):
        openai.api_key = ""
        openai.api_base = f"{endpoint}/v1"
        openai_chat_completion = openai.ChatCompletion.acreate
        stream_chunk_type_name = "OpenAIObject"
        response_type_name = "OpenAIObject"
    else:
        client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
        openai_chat_completion = client.chat.completions.create
        stream_chunk_type_name = "ChatCompletionChunk"
        response_type_name = "ChatCompletion"

    result1 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": False,
        },
    ):
        result1.append(chunk)
    assert result1
    assert type(result1[0]).__name__ == stream_chunk_type_name
    # stop
    assert result1[-1].choices[0].finish_reason == "stop"

    result2 = await openai_chat_completion(
        messages=messages, stream=False, model=model_uid_res
    )

    assert result2
    assert type(result2).__name__ == response_type_name
    # stop
    assert result2.choices[0].finish_reason == "stop"

    result3 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": True,
        },
    ):
        result3.append(chunk)
    assert result3
    assert type(result3[0]).__name__ == stream_chunk_type_name
    # usage
    assert not result3[-1].choices
    assert result3[-1].usage


@pytest.mark.asyncio
@pytest.mark.skip(reason="Too large model to be tested")
async def test_openai_stream_options_pytorch(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"
    os.environ.get(XINFERENCE_ENV_MODEL_SRC, "modelscope")

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_restful_api",
        "model_engine": "Transformers",
        "model_name": "baichuan-2-chat",
        "model_size_in_billions": "7",
        "quantization": "none",
        "model_format": "pytorch",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    # chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi what can I help you?"},
        {"role": "user", "content": "法国的首都是哪里?"},
    ]

    if version.parse(openai.__version__) < version.parse("1.0"):
        openai.api_key = ""
        openai.api_base = f"{endpoint}/v1"
        openai_chat_completion = openai.ChatCompletion.acreate
        stream_chunk_type_name = "OpenAIObject"
        response_type_name = "OpenAIObject"
    else:
        client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
        openai_chat_completion = client.chat.completions.create
        stream_chunk_type_name = "ChatCompletionChunk"
        response_type_name = "ChatCompletion"

    result1 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": False,
        },
    ):
        result1.append(chunk)
    assert result1
    assert type(result1[0]).__name__ == stream_chunk_type_name
    # stop
    assert result1[-1].choices[0].finish_reason == "stop"

    result2 = await openai_chat_completion(
        messages=messages, stream=False, model=model_uid_res
    )

    assert result2
    assert type(result2).__name__ == response_type_name
    # stop
    assert result2.choices[0].finish_reason == "stop"

    result3 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": True,
        },
    ):
        result3.append(chunk)
    assert result3
    assert type(result3[0]).__name__ == stream_chunk_type_name
    # usage
    assert not result3[-1].choices
    assert result3[-1].usage


@pytest.mark.asyncio
@pytest.mark.skip(reason="Too large model to be tested")
async def test_openai_stream_options_pytorch_deepseek_vl(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"
    os.environ.get(XINFERENCE_ENV_MODEL_SRC, "modelscope")

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_restful_api",
        "model_engine": "Transformers",
        "model_name": "deepseek-vl-chat",
        "model_size_in_billions": "1_3",
        "quantization": "none",
        "model_format": "pytorch",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    # chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi what can I help you?"},
        {"role": "user", "content": "法国的首都是哪里?"},
    ]

    if version.parse(openai.__version__) < version.parse("1.0"):
        openai.api_key = ""
        openai.api_base = f"{endpoint}/v1"
        openai_chat_completion = openai.ChatCompletion.acreate
        stream_chunk_type_name = "OpenAIObject"
        response_type_name = "OpenAIObject"
    else:
        client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
        openai_chat_completion = client.chat.completions.create
        stream_chunk_type_name = "ChatCompletionChunk"
        response_type_name = "ChatCompletion"

    result1 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": False,
        },
    ):
        result1.append(chunk)
    assert result1
    assert type(result1[0]).__name__ == stream_chunk_type_name
    # stop
    assert result1[-1].choices[0].finish_reason == "stop"

    result2 = await openai_chat_completion(
        messages=messages, stream=False, model=model_uid_res
    )

    assert result2
    assert type(result2).__name__ == response_type_name
    # stop
    assert result2.choices[0].finish_reason == "stop"

    result3 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": True,
        },
    ):
        result3.append(chunk)
    assert result3
    assert type(result3[0]).__name__ == stream_chunk_type_name
    # usage
    assert not result3[-1].choices
    assert result3[-1].usage


@pytest.mark.asyncio
@pytest.mark.skip(reason="Too large model to be tested")
async def test_openai_stream_options_pytorch_internlm2(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"
    os.environ.get(XINFERENCE_ENV_MODEL_SRC, "modelscope")

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_restful_api",
        "model_engine": "Transformers",
        "model_name": "internlm2-chat",
        "model_size_in_billions": "7",
        "quantization": "none",
        "model_format": "pytorch",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    # chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi what can I help you?"},
        {"role": "user", "content": "法国的首都是哪里?"},
    ]

    if version.parse(openai.__version__) < version.parse("1.0"):
        openai.api_key = ""
        openai.api_base = f"{endpoint}/v1"
        openai_chat_completion = openai.ChatCompletion.acreate
        stream_chunk_type_name = "OpenAIObject"
        response_type_name = "OpenAIObject"
    else:
        client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
        openai_chat_completion = client.chat.completions.create
        stream_chunk_type_name = "ChatCompletionChunk"
        response_type_name = "ChatCompletion"

    result1 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": False,
        },
    ):
        result1.append(chunk)
    assert result1
    assert type(result1[0]).__name__ == stream_chunk_type_name
    # stop
    assert result1[-1].choices[0].finish_reason == "stop"

    result2 = await openai_chat_completion(
        messages=messages, stream=False, model=model_uid_res
    )

    assert result2
    assert type(result2).__name__ == response_type_name
    # stop
    assert result2.choices[0].finish_reason == "stop"

    result3 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": True,
        },
    ):
        result3.append(chunk)
    assert result3
    assert type(result3[0]).__name__ == stream_chunk_type_name
    # usage
    assert not result3[-1].choices
    assert result3[-1].usage


@pytest.mark.asyncio
@pytest.mark.skip(reason="Too large model to be tested")
async def test_openai_stream_options_pytorch_qwen_vl(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"
    os.environ.get(XINFERENCE_ENV_MODEL_SRC, "modelscope")

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_restful_api",
        "model_engine": "Transformers",
        "model_name": "qwen-vl-chat",
        "model_size_in_billions": "7",
        "quantization": "none",
        "model_format": "pytorch",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    # chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi what can I help you?"},
        {"role": "user", "content": "法国的首都是哪里?"},
    ]

    if version.parse(openai.__version__) < version.parse("1.0"):
        openai.api_key = ""
        openai.api_base = f"{endpoint}/v1"
        openai_chat_completion = openai.ChatCompletion.acreate
        stream_chunk_type_name = "OpenAIObject"
        response_type_name = "OpenAIObject"
    else:
        client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
        openai_chat_completion = client.chat.completions.create
        stream_chunk_type_name = "ChatCompletionChunk"
        response_type_name = "ChatCompletion"

    result1 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": False,
        },
    ):
        result1.append(chunk)
    assert result1
    assert type(result1[0]).__name__ == stream_chunk_type_name
    # stop
    assert result1[-1].choices[0].finish_reason == "stop"

    result2 = await openai_chat_completion(
        messages=messages, stream=False, model=model_uid_res
    )

    assert result2
    assert type(result2).__name__ == response_type_name
    # stop
    assert result2.choices[0].finish_reason == "stop"

    result3 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": True,
        },
    ):
        result3.append(chunk)
    assert result3
    assert type(result3[0]).__name__ == stream_chunk_type_name
    # usage
    assert not result3[-1].choices
    assert result3[-1].usage


@pytest.mark.asyncio
@pytest.mark.skip(reason="Too large model to be tested")
async def test_openai_stream_options_pytorch_yi_vl(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"
    os.environ.get(XINFERENCE_ENV_MODEL_SRC, "modelscope")

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_restful_api",
        "model_engine": "Transformers",
        "model_name": "yi-vl-chat",
        "model_size_in_billions": "6",
        "quantization": "none",
        "model_format": "pytorch",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    # chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi what can I help you?"},
        {"role": "user", "content": "法国的首都是哪里?"},
    ]

    if version.parse(openai.__version__) < version.parse("1.0"):
        openai.api_key = ""
        openai.api_base = f"{endpoint}/v1"
        openai_chat_completion = openai.ChatCompletion.acreate
        stream_chunk_type_name = "OpenAIObject"
        response_type_name = "OpenAIObject"
    else:
        client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
        openai_chat_completion = client.chat.completions.create
        stream_chunk_type_name = "ChatCompletionChunk"
        response_type_name = "ChatCompletion"

    result1 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": False,
        },
    ):
        result1.append(chunk)
    assert result1
    assert type(result1[0]).__name__ == stream_chunk_type_name
    # stop
    assert result1[-1].choices[0].finish_reason == "stop"

    result2 = await openai_chat_completion(
        messages=messages, stream=False, model=model_uid_res
    )

    assert result2
    assert type(result2).__name__ == response_type_name
    # stop
    assert result2.choices[0].finish_reason == "stop"

    result3 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": True,
        },
    ):
        result3.append(chunk)
    assert result3
    assert type(result3[0]).__name__ == stream_chunk_type_name
    # usage
    assert not result3[-1].choices
    assert result3[-1].usage


@pytest.mark.asyncio
@pytest.mark.skip(reason="Too large model to be tested")
async def test_openai_stream_options_sgalng(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"
    os.environ.get(XINFERENCE_ENV_MODEL_SRC, "modelscope")

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_restful_api",
        "model_engine": "SGLang",
        "model_name": "qwen-chat",
        "model_size_in_billions": "7",
        "quantization": "Int4",
        "model_format": "gptq",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    # chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi what can I help you?"},
        {"role": "user", "content": "法国的首都是哪里?"},
    ]

    if version.parse(openai.__version__) < version.parse("1.0"):
        openai.api_key = ""
        openai.api_base = f"{endpoint}/v1"
        openai_chat_completion = openai.ChatCompletion.acreate
        stream_chunk_type_name = "OpenAIObject"
        response_type_name = "OpenAIObject"
    else:
        client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
        openai_chat_completion = client.chat.completions.create
        stream_chunk_type_name = "ChatCompletionChunk"
        response_type_name = "ChatCompletion"

    result1 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": False,
        },
    ):
        result1.append(chunk)
    assert result1
    assert type(result1[0]).__name__ == stream_chunk_type_name
    # stop
    assert result1[-1].choices[0].finish_reason == "stop"

    result2 = await openai_chat_completion(
        messages=messages, stream=False, model=model_uid_res
    )

    assert result2
    assert type(result2).__name__ == response_type_name
    # stop
    assert result2.choices[0].finish_reason == "stop"

    result3 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": True,
        },
    ):
        result3.append(chunk)
    assert result3
    assert type(result3[0]).__name__ == stream_chunk_type_name
    # usage
    assert not result3[-1].choices
    assert result3[-1].usage


@pytest.mark.asyncio
@pytest.mark.skip(reason="Too large model to be tested")
async def test_openai_stream_options_vllm(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"
    os.environ.get(XINFERENCE_ENV_MODEL_SRC, "modelscope")

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_restful_api",
        "model_engine": "vLLM",
        "model_name": "qwen-chat",
        "model_size_in_billions": "7",
        "quantization": "none",
        "model_format": "pytorch",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    # chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi what can I help you?"},
        {"role": "user", "content": "法国的首都是哪里?"},
    ]

    if version.parse(openai.__version__) < version.parse("1.0"):
        openai.api_key = ""
        openai.api_base = f"{endpoint}/v1"
        openai_chat_completion = openai.ChatCompletion.acreate
        stream_chunk_type_name = "OpenAIObject"
        response_type_name = "OpenAIObject"
    else:
        client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
        openai_chat_completion = client.chat.completions.create
        stream_chunk_type_name = "ChatCompletionChunk"
        response_type_name = "ChatCompletion"

    result1 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": False,
        },
    ):
        result1.append(chunk)
    assert result1
    assert type(result1[0]).__name__ == stream_chunk_type_name
    # stop
    assert result1[-1].choices[0].finish_reason == "stop"

    result2 = await openai_chat_completion(
        messages=messages, stream=False, model=model_uid_res
    )

    assert result2
    assert type(result2).__name__ == response_type_name
    # stop
    assert result2.choices[0].finish_reason == "stop"

    result3 = []
    async for chunk in await openai_chat_completion(
        messages=messages,
        stream=True,
        model=model_uid_res,
        max_tokens=None,
        stream_options={
            "include_usage": True,
        },
    ):
        result3.append(chunk)
    assert result3
    assert type(result3[0]).__name__ == stream_chunk_type_name
    # usage
    assert not result3[-1].choices
    assert result3[-1].usage


@pytest.mark.asyncio
@pytest.mark.skip(reason="Too large model to be tested")
async def test_openai_stream_tools_vllm(setup):
    import json

    endpoint, _ = setup
    url = f"{endpoint}/v1/models"
    os.environ.get(XINFERENCE_ENV_MODEL_SRC, "modelscope")

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_restful_api",
        "model_engine": "vLLM",
        "model_name": "qwen-chat",
        "model_size_in_billions": "7",
        "quantization": "none",
        "model_format": "pytorch",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    def get_current_weather(location: str, unit="fahrenheit"):
        """Get the current weather in a given location"""
        if "tokyo" in location.lower():
            return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
        else:
            return json.dumps({"location": location, "temperature": "unknown"})

    def handle_tools_response(messages, tool_calls):
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = get_current_weather
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

    # create conversation with tool call
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in tokyo? Use tools to answer.",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    if version.parse(openai.__version__) < version.parse("1.0"):
        openai.api_key = ""
        openai.api_base = f"{endpoint}/v1"
        openai_chat_completion = openai.ChatCompletion.acreate
        stream_chunk_type_name = "OpenAIObject"
        response_type_name = "OpenAIObject"
    else:
        client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
        openai_chat_completion = client.chat.completions.create
        stream_chunk_type_name = "ChatCompletionChunk"
        response_type_name = "ChatCompletion"

    # Step 1: Get the initial response with stream=True
    result_stream_initial = []
    for chunk in openai_chat_completion(
        messages=messages,
        model=model_uid_res,
        stream=True,
        tools=tools,
    ):
        result_stream_initial.append(chunk)

    assert result_stream_initial
    assert type(result_stream_initial[0]).__name__ == stream_chunk_type_name

    # Simulate the tool calls for the initial stream response
    tool_calls = result_stream_initial[-1].choices[0].delta.tool_calls
    assert tool_calls
    handle_tools_response(messages, tool_calls)

    # Step 2: Get the final response after tool calls with stream=True
    result_stream_final = []
    for chunk in openai_chat_completion(
        messages=messages,
        model=model_uid_res,
        stream=True,
    ):
        result_stream_final.append(chunk)

    assert result_stream_final
    assert type(result_stream_final[0]).__name__ == stream_chunk_type_name

    # Combine streamed chunks into a single response
    final_response_stream = "".join(
        chunk.choices[0].delta.content for chunk in result_stream_final
    )
    assert "tokyo" in final_response_stream.lower()

    # Step 3: Get the initial response with stream=False
    messages = messages[:1]  # reset to initial messages
    initial_response = openai_chat_completion(
        messages=messages,  # reset to initial messages
        model=model_uid_res,
        tools=tools,
        stream=False,
    )

    assert initial_response
    assert type(initial_response).__name__ == response_type_name

    # Simulate the tool calls for the initial non-stream response
    tool_calls = initial_response.choices[0].message.tool_calls
    assert tool_calls is not None
    handle_tools_response(messages, tool_calls)

    # Step 4: Get the final response after tool calls with stream=False
    final_response = openai_chat_completion(
        messages=messages, model=model_uid_res, stream=False
    )

    assert final_response
    assert type(final_response).__name__ == response_type_name

    # Check the final response contains the weather info
    assert "tokyo" in final_response.choices[0].message.content.lower()
