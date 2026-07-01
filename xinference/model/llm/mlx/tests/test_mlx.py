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

import base64
import os
import platform
import sys
import threading

import pytest

from .....client import Client


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
    assert "图中" in completion["choices"][0]["message"]["content"]
    assert "鱼" in completion["choices"][0]["message"]["content"]

    # test no image
    messages = [{"role": "user", "content": "write a poem."}]
    completion = model.chat(messages)
    assert "content" in completion["choices"][0]["message"]
    assert "content" in completion["choices"][0]["message"]
    assert len(completion["choices"][0]["message"]["content"]) != 0


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
