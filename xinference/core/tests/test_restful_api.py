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

import json
import os
import os.path
import sys
import time

import openai
import pytest
import requests
from packaging import version

from ...model.embedding import BUILTIN_EMBEDDING_MODELS


@pytest.mark.asyncio
async def test_restful_api(setup):
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
        "model_name": "orca",
        "quantization": "q4_0",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    # launch n_gpu error
    payload = {
        "model_uid": "test_restful_api",
        "model_name": "orca",
        "quantization": "q4_0",
        "n_gpu": -1,
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 400

    # same model uid
    payload = {
        "model_uid": "test_restful_api",
        "model_name": "orca",
        "quantization": "q4_0",
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 400

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 1

    # describe
    response = requests.get(f"{endpoint}/v1/models/test_restful_api")
    response_data = response.json()
    assert response_data["model_name"] == "orca"
    assert response_data["replica"] == 1

    response = requests.delete(f"{endpoint}/v1/models/bogus")
    assert response.status_code == 400

    # generate
    url = f"{endpoint}/v1/completions"
    payload = {
        "model": model_uid_res,
        "prompt": "Once upon a time, there was a very old computer.",
    }
    response = requests.post(url, json=payload)
    completion = response.json()
    assert "text" in completion["choices"][0]

    payload = {
        "model": "bogus",
        "prompt": "Once upon a time, there was a very old computer.",
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 400

    payload = {
        "prompt": "Once upon a time, there was a very old computer.",
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 500

    # chat without user messages
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model_uid_res,
        "messages": [
            {
                "role": "system",
                "content": "<任务> 识别用户输入的技术术语。请用{XXX} -> {XXX}的格式展示翻译前后的技术术语对应关系。\n<输入文本>\n今天天气\n<示例>\nTransformer -> Transformer\nToken -> Token\nZero Shot -> 零样本\nFew Shot -> 少样本\n<专有名词>",
            }
        ],
        "stop": ["\n"],
    }
    response = requests.post(url, json=payload)
    completion = response.json()
    assert "content" in completion["choices"][0]["message"]

    # chat
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model_uid_res,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi what can I help you?"},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "stop": ["\n"],
    }
    response = requests.post(url, json=payload)
    completion = response.json()
    assert "content" in completion["choices"][0]["message"]

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi what can I help you?"},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 500

    payload = {
        "model": "bogus",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi what can I help you?"},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 400

    payload = {
        "model": model_uid_res,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi what can I help you?"},
        ],
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 400

    # allow duplicate system messages
    payload = {
        "model": model_uid_res,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "You are not a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi what can I help you?"},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }
    response = requests.post(url, json=payload)
    completion = response.json()
    assert "content" in completion["choices"][0]["message"]

    # allow the first message is not system message.
    payload = {
        "model": model_uid_res,
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "Hi what can I help you?"},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }
    response = requests.post(url, json=payload)
    completion = response.json()
    assert "content" in completion["choices"][0]["message"]

    # delete
    url = f"{endpoint}/v1/models/test_restful_api"
    response = requests.delete(url)

    # list
    response = requests.get(f"{endpoint}/v1/models")
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # delete again
    url = f"{endpoint}/v1/models/test_restful_api"
    response = requests.delete(url)
    assert response.status_code == 400

    # list model registration

    url = f"{endpoint}/v1/model_registrations/LLM"

    response = requests.get(url)

    assert response.status_code == 200
    model_regs = response.json()
    assert len(model_regs) > 0
    for model_reg in model_regs:
        assert model_reg["is_builtin"]

    # register_model

    model = """{
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
  "prompt_style": {
    "style_name": "ADD_COLON_SINGLE",
    "system_prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    "roles": [
      "Instruction",
      "Response"
    ],
    "intra_message_sep": "\\n\\n### "
  }
}"""

    url = f"{endpoint}/v1/model_registrations/LLM"

    payload = {"model": model, "persist": False}

    response = requests.post(url, json=payload)
    assert response.status_code == 200

    # check model version info after registration
    url = f"{endpoint}/v1/models/LLM/custom_model/versions"
    response = requests.get(url)
    version_infos = response.json()
    assert len(version_infos) == 3  # three quantizations

    url = f"{endpoint}/v1/model_registrations/LLM"

    response = requests.get(url)

    assert response.status_code == 200
    new_model_regs = response.json()
    assert len(new_model_regs) == len(model_regs) + 1

    # get_model_registrations
    url = f"{endpoint}/v1/model_registrations/LLM/custom_model"
    response = requests.get(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "custom_model" in data["model_name"]

    # unregister_model
    url = f"{endpoint}/v1/model_registrations/LLM/custom_model"

    response = requests.delete(url, json=payload)
    assert response.status_code == 200

    # check model version info after unregister
    url = f"{endpoint}/v1/models/LLM/custom_model/versions"
    response = requests.get(url)
    version_infos = response.json()
    assert len(version_infos) == 0

    url = f"{endpoint}/v1/model_registrations/LLM"

    response = requests.get(url)
    assert response.status_code == 200
    new_model_regs = response.json()
    assert len(new_model_regs) == len(model_regs)
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom_model":
            custom_model_reg = model_reg
    assert custom_model_reg is None


def test_restful_api_for_embedding(setup):
    model_name = "gte-base"
    model_spec = BUILTIN_EMBEDDING_MODELS[model_name]

    endpoint, _ = setup
    url = f"{endpoint}/v1/models"

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_embedding",
        "model_name": model_name,
        "model_type": "embedding",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_embedding"

    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 1

    # test embedding
    url = f"{endpoint}/v1/embeddings"
    payload = {
        "model": "test_embedding",
        "input": "The food was delicious and the waiter...",
    }
    response = requests.post(url, json=payload)
    embedding_res = response.json()

    assert "embedding" in embedding_res["data"][0]
    assert len(embedding_res["data"][0]["embedding"]) == model_spec.dimensions

    # test multiple
    payload = {
        "model": "test_embedding",
        "input": [
            "The food was delicious and the waiter...",
            "how to implement quick sort in python?",
            "Beijing",
            "sorting algorithms",
        ],
    }
    response = requests.post(url, json=payload)
    embedding_res = response.json()

    assert len(embedding_res["data"]) == 4
    for data in embedding_res["data"]:
        assert len(data["embedding"]) == model_spec.dimensions

    # delete model
    url = f"{endpoint}/v1/models/test_embedding"
    response = requests.delete(url)
    assert response.status_code == 200

    response = requests.get(f"{endpoint}/v1/models")
    response_data = response.json()
    assert len(response_data["data"]) == 0


def _check_invalid_tool_calls(endpoint, model_uid_res):
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_exchange_rate",
                "description": "Get the exchange rate between two currencies",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "base_currency": {
                            "type": "string",
                            "description": "The currency to convert from",
                        },
                        "target_currency": {
                            "type": "string",
                            "description": "The currency to convert to",
                        },
                    },
                    "required": ["base_currency", "target_currency"],
                },
            },
        }
    ]

    completion = client.chat.completions.create(
        model=model_uid_res,
        messages=[
            {
                "content": "Can you book a flight for me from New York to London?",
                "role": "user",
            }
        ],
        tools=tools,
        max_tokens=200,
        temperature=0.1,
    )
    assert "stop" == completion.choices[0].finish_reason
    assert completion.choices[0].message.content
    assert len(completion.choices[0].message.tool_calls) == 0


@pytest.mark.parametrize(
    "model_format, quantization", [("ggmlv3", "q4_0"), ("pytorch", None)]
)
@pytest.mark.skip(reason="Cost too many resources.")
def test_restful_api_for_tool_calls(setup, model_format, quantization):
    model_name = "chatglm3"

    endpoint, _ = setup
    url = f"{endpoint}/v1/models"

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_tool",
        "model_engine": "transformers",
        "model_name": model_name,
        "model_size_in_billions": 6,
        "model_format": model_format,
        "quantization": quantization,
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_tool"

    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 1

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "获取当前天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "城市，例如北京"},
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "使用的温度单位。从所在的城市进行推断。",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        }
    ]

    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model_uid_res,
        "messages": [
            {"role": "system", "content": "你是一个有用的助手。不要对要函数调用的值做出假设。"},
            {"role": "user", "content": "上海现在的天气怎么样？"},
        ],
        "temperature": 0.7,
        "tools": tools,
        "stop": ["\n"],
    }
    response = requests.post(url, json=payload)
    completion = response.json()
    # glm4-chat fail response: 好的，请告诉我您希望使用的温度单位是摄氏度还是华氏度？

    assert (
        "get_current_weather"
        == completion["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    )
    arguments = completion["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    arg = json.loads(arguments)
    assert arg == {"location": "上海", "format": "celsius"}

    # tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "track",
                "description": "追踪指定股票的实时价格",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"description": "需要追踪的股票代码"}},
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "text-to-speech",
                "description": "将文本转换为语音",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"description": "需要转换成语音的文本"},
                        "voice": {"description": "要使用的语音类型（男声、女声等）"},
                        "speed": {"description": "语音的速度（快、中等、慢等）"},
                    },
                    "required": ["text"],
                },
            },
        },
    ]
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model_uid_res,
        "messages": [
            {"role": "user", "content": "帮我查询股票10111的价格"},
        ],
        "tools": tools,
        "stop": ["\n"],
    }
    response = requests.post(url, json=payload)
    completion = response.json()

    assert "content" in completion["choices"][0]["message"]
    assert "tool_calls" == completion["choices"][0]["finish_reason"]
    assert (
        "track"
        == completion["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    )
    arguments = completion["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    arg = json.loads(arguments)
    assert arg == {"symbol": "10111"}

    # Restful client
    from ...client import RESTfulClient

    client = RESTfulClient(endpoint)
    model = client.get_model(model_uid_res)
    completion = model.chat("帮我查询股票10111的价格", tools=tools)
    assert "content" in completion["choices"][0]["message"]
    assert "tool_calls" == completion["choices"][0]["finish_reason"]
    assert (
        "track"
        == completion["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    )
    arguments = completion["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    arg = json.loads(arguments)
    assert arg == {"symbol": "10111"}

    # openai client
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    completion = client.chat.completions.create(
        model=model_uid_res,
        messages=[{"role": "user", "content": "帮我查询股票10111的价格"}],
        tools=tools,
    )
    assert "tool_calls" == completion.choices[0].finish_reason
    assert "track" == completion.choices[0].message.tool_calls[0].function.name
    arguments = completion.choices[0].message.tool_calls[0].function.arguments
    arg = json.loads(arguments)
    assert arg == {"symbol": "10111"}

    assistant_message = completion.choices[0].message.model_dump()
    messages = [
        {"role": "user", "content": "帮我查询股票10111的价格"},
        assistant_message,
        {
            "role": "tool",
            "tool_call_id": assistant_message["tool_calls"][0]["id"],
            "name": assistant_message["tool_calls"][0]["function"]["name"],
            "content": str({"symbol": "10111", "price": 12345}),
        },
    ]

    # When kwargs is {}, the glm4-chat does not observe the output tool calls,
    # so the test will fail.
    for kwargs in [{"tools": tools}, {}]:
        completion = client.chat.completions.create(
            model=model_uid_res, messages=messages, **kwargs
        )
        assert completion.choices
        assert completion.choices[0].finish_reason == "stop"
        assert "10111" in completion.choices[0].message.content
        assert "12345" in completion.choices[0].message.content

    _check_invalid_tool_calls(endpoint, model_uid_res)


@pytest.mark.parametrize(
    "model_format, quantization", [("ggufv2", "Q4_K_S"), ("pytorch", None)]
)
@pytest.mark.skip(reason="Cost too many resources.")
def test_restful_api_for_gorilla_openfunctions_tool_calls(
    setup, model_format, quantization
):
    model_name = "gorilla-openfunctions-v1"

    endpoint, _ = setup
    url = f"{endpoint}/v1/models"

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_tool",
        "model_name": model_name,
        "model_size_in_billions": 7,
        "model_format": model_format,
        "quantization": quantization,
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_tool"

    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 1

    # tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "uber_ride",
                "description": "Find suitable ride for customers given the location, "
                "type of ride, and the amount of time the customer is "
                "willing to wait as parameters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "loc": {
                            "type": "int",
                            "description": "Location of the starting place of the Uber ride",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["plus", "comfort", "black"],
                            "description": "Types of Uber ride user is ordering",
                        },
                        "time": {
                            "type": "int",
                            "description": "The amount of time in minutes the customer is willing to wait",
                        },
                    },
                },
            },
        }
    ]
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model_uid_res,
        "messages": [
            {
                "role": "user",
                "content": 'Call me an Uber ride type "Plus" in Berkeley at zipcode 94704 in 10 minutes',
            },
        ],
        "tools": tools,
        "stop": ["\n"],
        "max_tokens": 200,
        "temperature": 0,
    }
    response = requests.post(url, json=payload)
    completion = response.json()

    assert "content" in completion["choices"][0]["message"]
    assert "tool_calls" == completion["choices"][0]["finish_reason"]
    assert (
        "uber_ride"
        == completion["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    )
    arguments = completion["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    arg = json.loads(arguments)
    assert arg == {"loc": 94704, "time": 10, "type": "plus"}

    _check_invalid_tool_calls(endpoint, model_uid_res)


@pytest.mark.parametrize(
    "model_format, quantization",
    [
        ("pytorch", None),
        ("ggufv2", "Q4_K_M"),
    ],
)
@pytest.mark.skip(reason="Cost too many resources.")
def test_restful_api_for_qwen_tool_calls(setup, model_format, quantization):
    model_name = "qwen1.5-chat"

    endpoint, _ = setup
    url = f"{endpoint}/v1/models"

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 0

    # launch
    payload = {
        "model_uid": "test_tool",
        "model_name": model_name,
        "model_engine": "transformers",
        "model_size_in_billions": 7,
        "model_format": model_format,
        "quantization": quantization,
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_tool"

    response = requests.get(url)
    response_data = response.json()
    assert len(response_data["data"]) == 1

    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model_uid_res,
        "messages": [
            {
                "role": "user",
                "content": "谁是周杰伦？",
            },
        ],
        "tools": [],
        "max_tokens": 2048,
        "temperature": 0,
    }
    response = requests.post(url, json=payload)
    completion = response.json()
    assert "stop" == completion["choices"][0]["finish_reason"]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "谷歌搜索是一个通用搜索引擎，搜索周杰伦。",
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "image_gen",
                "description": "文生图是一个AI绘画（图像生成）服务，画个周杰伦。",
            },
        },
    ]

    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model_uid_res,
        "messages": [
            {
                "role": "user",
                "content": "谁是周杰伦？",
            },
        ],
        "tools": tools,
        "max_tokens": 2048,
        "temperature": 0,
    }
    response = requests.post(url, json=payload)
    completion = response.json()
    assert "content" in completion["choices"][0]["message"]
    assert "tool_calls" == completion["choices"][0]["finish_reason"]
    assert (
        "google_search"
        == completion["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    )
    arguments = completion["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    assert json.loads(arguments)
    assert completion["usage"]
    assert completion["usage"]["prompt_tokens"] != -1

    # tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "搜索关键词或短语",
                        },
                    },
                    "required": ["search_query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "image_gen",
                "description": "文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "英文关键词，描述了希望图像具有什么内容",
                        },
                    },
                    "required": ["prompt"],
                },
            },
        },
    ]
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model_uid_res,
        "messages": [
            {
                "role": "user",
                "content": "谁是周杰伦？",
            },
        ],
        "tools": tools,
        "max_tokens": 2048,
        "temperature": 0,
    }
    response = requests.post(url, json=payload)
    completion = response.json()

    assert "content" in completion["choices"][0]["message"]
    assert "tool_calls" == completion["choices"][0]["finish_reason"]
    assert (
        "google_search"
        == completion["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    )
    arguments = completion["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    arg = json.loads(arguments)
    assert arg == {"search_query": "周杰伦"}

    # Check tool message.
    payload = {
        "model": model_uid_res,
        "messages": [
            {
                "role": "user",
                "content": "谁是周杰伦？",
            },
            completion["choices"][0]["message"],
            {
                "role": "tool",
                "content": "Jay Chou is a Taiwanese singer, songwriter, record producer, rapper, actor, television personality, and businessman.",
            },
        ],
        "tools": tools,
        "max_tokens": 2048,
        "temperature": 0,
    }
    response = requests.post(url, json=payload)
    completion2 = response.json()
    assert "stop" == completion2["choices"][0]["finish_reason"]
    assert "周杰伦" in completion2["choices"][0]["message"]["content"]
    # The content varies between gguf and torch model.
    # assert "歌" in completion2["choices"][0]["message"]["content"]

    # Check continue tool call.
    payload = {
        "model": model_uid_res,
        "messages": [
            {
                "role": "user",
                "content": "谁是周杰伦？",
            },
            completion["choices"][0]["message"],
            {
                "role": "tool",
                "content": "Jay Chou is a Taiwanese singer, songwriter, record producer, rapper, actor, television personality, and businessman.",
            },
            completion2["choices"][0]["message"],
            {"role": "user", "content": "画一个他的卡通形象出来"},
        ],
        "tools": tools,
        "max_tokens": 2048,
        "temperature": 0,
    }
    response = requests.post(url, json=payload)
    completion3 = response.json()
    assert "tool_calls" == completion3["choices"][0]["finish_reason"]
    assert (
        "image_gen"
        == completion3["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    )
    arguments = completion3["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    arg = json.loads(arguments)
    assert "Jay Chou" in arg["prompt"]

    # Qwen 1.5 4B can't pass the false tool call check.
    # _check_invalid_tool_calls(endpoint, model_uid_res)


def test_restful_api_with_request_limits(setup):
    model_name = "gte-base"

    endpoint, _ = setup
    url = f"{endpoint}/v1/models"

    # test embedding
    # launch
    payload = {
        "model_uid": "test_embedding",
        "model_name": model_name,
        "model_type": "embedding",
        "request_limits": 0,
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_embedding"

    # test embedding
    url = f"{endpoint}/v1/embeddings"
    payload = {
        "model": "test_embedding",
        "input": "The food was delicious and the waiter...",
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 429
    assert "Rate limit reached" in response.json()["detail"]

    # delete model
    url = f"{endpoint}/v1/models/test_embedding"
    response = requests.delete(url)
    assert response.status_code == 200

    # test llm
    url = f"{endpoint}/v1/models"
    payload = {
        "model_uid": "test_restful_api",
        "model_engine": "llama.cpp",
        "model_name": "orca",
        "quantization": "q4_0",
        "request_limits": 0,
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    # generate
    url = f"{endpoint}/v1/completions"
    payload = {
        "model": model_uid_res,
        "prompt": "Once upon a time, there was a very old computer.",
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 429
    assert "Rate limit reached" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform == "win32", reason="Window CI hangs after run this case."
)
async def test_openai(setup):
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
        "model_name": "orca",
        "quantization": "q4_0",
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

    result = []
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
    async for chunk in await openai_chat_completion(
        messages=messages, stream=True, model=model_uid_res, max_tokens=None
    ):
        if not hasattr(chunk, "choices") or len(chunk.choices) == 0:
            continue
        result.append(chunk)
    assert result
    assert type(result[0]).__name__ == stream_chunk_type_name

    result = await openai_chat_completion(
        messages=messages, stream=False, model=model_uid_res
    )

    assert result
    assert type(result).__name__ == response_type_name


def test_lang_chain(setup):
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
        "model_name": "orca",
        "quantization": "q4_0",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )
    from langchain.schema import AIMessage, HumanMessage, SystemMessage

    inference_server_url = f"{endpoint}/v1"

    chat = ChatOpenAI(
        model=model_uid_res,
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
        max_tokens=5,
        temperature=0,
    )

    messages = [
        SystemMessage(
            content="You are a helpful assistant that translates English to Italian."
        ),
        HumanMessage(
            content="Translate the following sentence from English to Italian: I love programming."
        ),
    ]
    r = chat(messages)
    assert type(r) == AIMessage
    assert r.content
    assert "amo" in r.content.lower()

    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    r = chat(
        chat_prompt.format_prompt(
            input_language="English",
            output_language="Italian",
            text="I love programming.",
        ).to_messages()
    )
    assert type(r) == AIMessage
    assert r.content


def test_launch_model_async(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models?wait_ready=false"

    payload = {
        "model_uid": "test_orca",
        "model_engine": "llama.cpp",
        "model_name": "orca",
        "quantization": "q4_0",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_orca"

    status_url = f"{endpoint}/v1/models/instances?model_uid=test_orca"
    while True:
        response = requests.get(status_url)
        response_data = response.json()
        assert len(response_data) == 1
        res = response_data[0]
        print(res)
        if res["status"] == "READY":
            break
        time.sleep(2)

    # delete again
    url = f"{endpoint}/v1/models/test_orca"
    requests.delete(url)

    response = requests.get(status_url)
    assert len(response.json()) == 0


def test_events(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"

    payload = {
        "model_uid": "test_orca",
        "model_engine": "llama.cpp",
        "model_name": "orca",
        "quantization": "q4_0",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_orca"

    events_url = f"{endpoint}/v1/models/test_orca/events"
    response = requests.get(events_url)
    response_data = response.json()
    # [{'event_type': 'INFO', 'event_ts': 1705896156, 'event_content': 'Launch model'}]
    assert len(response_data) == 1
    assert "Launch" in response_data[0]["event_content"]

    # delete again
    url = f"{endpoint}/v1/models/test_orca"
    requests.delete(url)

    response = requests.get(events_url)
    response_data = response.json()
    # [{'event_type': 'INFO', 'event_ts': 1705896215, 'event_content': 'Launch model'},
    #  {'event_type': 'INFO', 'event_ts': 1705896215, 'event_content': 'Terminate model'}]
    assert len(response_data) == 2
    assert "Terminate" in response_data[1]["event_content"]


def test_launch_model_by_version(setup):
    from ...model.llm import get_llm_model_descriptions

    endpoint, supervisor_addr = setup
    url = f"{endpoint}/v1/models/instance"

    version_info = get_llm_model_descriptions()["orca"][0]

    payload = {
        "model_uid": "test_orca",
        "model_engine": "llama.cpp",
        "model_type": "LLM",
        "model_version": version_info["model_version"],
    }
    response = requests.post(url, json=payload)
    assert response.json()["model_uid"] == "test_orca"

    url_version = f"{endpoint}/v1/models/LLM/orca/versions"
    response = requests.get(url_version)
    versions = response.json()

    has_version = False
    for info in versions:
        if info["model_version"] == version_info["model_version"]:
            has_version = True
            assert info["cache_status"] is True
            assert info["model_file_location"] is not None
            assert isinstance(info["model_file_location"], dict)
            assert supervisor_addr in info["model_file_location"]
            assert os.path.exists(info["model_file_location"][supervisor_addr])
            break
    assert has_version is True

    # delete again
    url = f"{endpoint}/v1/models/test_orca"
    requests.delete(url)


@pytest.mark.skipif(bool(os.environ.get("GITHUB_ACTIONS")), reason="Skip windows")
def test_cluster_info(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/cluster/info"

    response = requests.get(url)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["node_type"] == "Supervisor"
    assert result[0]["gpu_count"] == 0
    assert result[0]["gpu_vram_total"] == 0
    assert result[1]["node_type"] == "Worker"
    assert result[1]["gpu_count"] == 0
    assert result[1]["gpu_vram_total"] == 0
