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
import sys

import pytest


@pytest.mark.skip(reason="Cost too many resources.")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ["model_engine", "model_size_in_billions", "model_format", "quantization"],
    [
        ("vLLM", "1_5", "pytorch", None),
        ("Transformers", "1_5", "pytorch", None),
        ("SGLang", "1_5", "pytorch", None),
        ("llama.cpp", "1_5", "ggufv2", None),
        pytest.param(
            "MLX",
            "1_5",
            "mlx",
            None,
            marks=pytest.mark.skipif(
                sys.platform != "darwin", reason="only run at macOS"
            ),
        ),
    ],
)
async def test_restful_api_for_deepseek_with_reasoning(
    setup, model_engine, model_size_in_billions, model_format, quantization
):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="deepseek-r1",
        model_name="deepseek-r1-distill-qwen",
        model_engine=model_engine,
        model_size_in_billions=model_size_in_billions,
        model_format=model_format,
        quantization=quantization,
        max_model_len=62032,
        reasoning_content=True,
    )
    model = client.get_model(model_uid)
    messages = [{"role": "user", "content": "Hello! What can you do?"}]
    response = model.chat(messages)
    assert "reasoning_content" in response["choices"][0]["message"]

    # openai client
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    completion = client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "role": "user",
                "content": "Hello! What can you do?",
            }
        ],
    )
    assert "reasoning_content" in completion.choices[0].message.to_dict()

    client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
    result = []
    async for chunk in await client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "role": "user",
                "content": "Hello! What can you do?",
            }
        ],
        stream=True,
    ):
        result.append(chunk)
    assert result
    assert "reasoning_content" in result[0].choices[0].delta.to_dict()
    assert result[-1].choices[0].finish_reason == "stop"


@pytest.mark.skip(reason="Cost too many resources.")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ["model_size_in_billions", "model_format", "quantization"], [(7, "pytorch", None)]
)
async def test_restful_api_for_deepseek_without_reasoning(
    setup, model_size_in_billions, model_format, quantization
):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="deepseek-r1",
        model_name="deepseek-r1-distill-qwen",
        model_engine="vLLM",
        model_size_in_billions=model_size_in_billions,
        model_format=model_format,
        quantization=quantization,
        max_model_len=62032,
    )
    model = client.get_model(model_uid)
    messages = [{"role": "user", "content": "Hello! What can you do?"}]
    response = model.chat(messages)

    assert "reasoning_content" not in response["choices"][0]["message"]

    # openai client
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    completion = client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "role": "user",
                "content": "Hello! What can you do?",
            }
        ],
    )
    assert "reasoning_content" not in completion.choices[0].message.to_dict()

    client = openai.AsyncClient(api_key="not empty", base_url=f"{endpoint}/v1")
    result = []
    async for chunk in await client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "role": "user",
                "content": "Hello! What can you do?",
            }
        ],
        stream=True,
    ):
        result.append(chunk)
    assert result
    assert "reasoning_content" not in result[0].choices[0].delta.to_dict()
    assert result[-1].choices[0].finish_reason == "stop"
