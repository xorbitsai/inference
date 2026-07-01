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

import importlib.util
import sys

import pytest
import torch


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream, enable_thinking, reasoning_content",
    [
        (False, False, False),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, True, True),
        (True, True, False),
    ],
)
async def test_qwen3_with_thinking_params(
    setup, stream, enable_thinking, reasoning_content
):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="qwen3",
        model_name="qwen3",
        model_engine="Transformers",
        model_format="pytorch",
        model_size_in_billions="0_6",
        quantization="none",
        n_gpu="auto",
        replica=1,
        stream=stream,
        enable_thinking=enable_thinking,
        reasoning_content=reasoning_content,
    )
    model = client.get_model(model_uid)
    assert model is not None

    # openai client
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    completion = client.chat.completions.create(
        model=model_uid,
        messages=[{"role": "user", "content": "Hello"}],
        stream=stream,
    )

    if stream:
        full_content = ""
        full_reasoning = ""
        for chunk in completion:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                full_content += delta.content
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                full_reasoning += delta.reasoning_content
        if enable_thinking:
            if reasoning_content:
                assert full_reasoning
            else:
                assert not full_reasoning
                assert "<think>" in full_content
        else:
            assert not full_reasoning
            assert "<think>" not in full_content
    else:
        assert completion is not None
        assert completion.choices[0].message.content is not None
        if enable_thinking:
            if reasoning_content:
                assert completion.choices[0].message.reasoning_content is not None
            else:
                assert "<think>" in completion.choices[0].message.content
                assert "reasoning_content" not in completion.choices[0].message
        else:
            assert "<think>" not in completion.choices[0].message.content


@pytest.mark.asyncio
async def test_qwen3_with_tools(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    # Launch model
    model_uid = client.launch_model(
        model_uid="qwen3",
        model_name="qwen3",
        model_engine="Transformers",
        model_format="pytorch",
        model_size_in_billions="0_6",
        quantization="none",
        n_gpu="auto",
        replica=1,
        enable_thinking=True,
    )
    model = client.get_model(model_uid)
    assert model is not None

    # openai client
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    completion = client.chat.completions.create(
        model=model_uid,
        messages=[{"role": "user", "content": "你好"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询天气",
                    "parameters": {},
                },
            }
        ],
        max_tokens=100,
    )

    assert completion is not None
    assert completion.choices[0].message.content is not None
    assert completion.choices[0].message.tool_calls is not None

    completion = client.chat.completions.create(
        model=model_uid,
        messages=[{"role": "user", "content": "查询上海天气"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市或者地区，比如北京，上海",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        max_tokens=100,
    )
    assert completion is not None
    assert completion.choices[0].message.content is not None
    assert len(completion.choices[0].message.content) > 0

    # Check if tool_calls is present and is a list
    # Note: tool_calls might be None if the model didn't generate valid tool calls
    if completion.choices[0].message.tool_calls is not None:
        assert isinstance(completion.choices[0].message.tool_calls, list)

        # If tool_calls list is not empty, check the structure
        if len(completion.choices[0].message.tool_calls) > 0:
            tool_call = completion.choices[0].message.tool_calls[0]
            assert hasattr(tool_call, "id")
            assert hasattr(tool_call, "type")
            assert tool_call.type == "function"
            assert hasattr(tool_call, "function")
            assert hasattr(tool_call.function, "name")
            assert hasattr(tool_call.function, "arguments")
            # Check if arguments is a valid JSON string
            import json

            args = json.loads(tool_call.function.arguments)
            assert isinstance(args, dict)
            # Check specific parameters if expected
            if "location" in args:
                assert isinstance(args["location"], str)


ENGINES = ["Transformers"]
if importlib.util.find_spec("vllm") is not None:
    ENGINES.append("vLLM")


@pytest.fixture
def setup_cluster():
    import xoscar as xo

    from ....api.restful_api import run_in_subprocess as restful_api_run_in_subprocess
    from ....conftest import TEST_FILE_LOGGING_CONF, TEST_LOGGING_CONF, api_health_check
    from ....deploy.local import health_check
    from ....deploy.local import run_in_subprocess as supervisor_run_in_subprocess

    supervisor_address = f"localhost:{xo.utils.get_next_port()}"
    local_cluster = supervisor_run_in_subprocess(
        supervisor_address, None, None, TEST_LOGGING_CONF
    )

    if not health_check(address=supervisor_address, max_attempts=20, sleep_interval=1):
        raise RuntimeError("Supervisor is not available after multiple attempts")

    port = xo.utils.get_next_port()
    endpoint = f"http://localhost:{port}"
    try:
        restful_api_proc = restful_api_run_in_subprocess(
            supervisor_address,
            host="localhost",
            port=port,
            logging_conf=TEST_FILE_LOGGING_CONF,
        )
        if not api_health_check(endpoint, max_attempts=10, sleep_interval=5):
            raise RuntimeError("Endpoint is not available after multiple attempts")

        yield f"http://localhost:{port}", supervisor_address
        restful_api_proc.kill()
    finally:
        local_cluster.kill()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream, enable_model_thinking, enable_chat_thinking",
    [
        (False, True, False),
        (False, True, True),
        (True, True, True),
        (True, True, False),
    ],
)
@pytest.mark.parametrize("engine", ENGINES)
async def test_qwen3_enable_thinking(
    setup_cluster, stream, enable_model_thinking, enable_chat_thinking, engine
):
    endpoint, _ = setup_cluster
    from ....client import Client

    if (
        engine == "vLLM"
        and enable_model_thinking
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] < 8
    ):
        pytest.skip("vLLM reasoning requires compute capability >=8.0 (Ampere).")
    client = Client(endpoint)

    kwargs = {}
    if engine == "vLLM":
        total_mem_bytes = torch.cuda.get_device_properties(0).total_memory
        total_mem_gb = total_mem_bytes / (1024**3)
        # 3G gpu memory
        kwargs["gpu_memory_utilization"] = min(3 / total_mem_gb, 1.0)
        kwargs["max_model_len"] = 1000
        # enforce_eager to save launch time
        kwargs["enforce_eager"] = True

    model_uid = client.launch_model(
        model_uid="qwen3",
        model_name="qwen3",
        model_engine=engine,
        model_format="pytorch",
        model_size_in_billions="0_6",
        quantization="none",
        n_gpu="auto",
        replica=1,
        enable_thinking=enable_model_thinking,
        reasoning_content=False,
        **kwargs,
    )
    model = client.get_model(model_uid)
    assert model is not None

    completion = model.chat(
        messages=[{"role": "user", "content": "Hello"}],
        generate_config={
            "max_tokens": 25,
            "chat_template_kwargs": {"enable_thinking": enable_chat_thinking},
            "stream": stream,
        },
    )

    try:
        if stream:
            full_content = ""
            for chunk in completion:
                delta = chunk["choices"][0]["delta"]
                if delta.get("content"):
                    full_content += delta["content"]
            if enable_chat_thinking:
                assert "<think>" in full_content
            else:
                assert "<think>" not in full_content
        else:
            assert completion is not None
            assert completion["choices"][0]["message"]["content"] is not None
            if enable_chat_thinking:
                assert "<think>" in completion["choices"][0]["message"]["content"]
            else:
                assert "<think>" not in completion["choices"][0]["message"]["content"]
    finally:
        client.terminate_model(model_uid)
