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
