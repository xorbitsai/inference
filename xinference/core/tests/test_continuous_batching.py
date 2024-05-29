# Copyright 2022-2024 XProbe Inc.
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
import threading

import pytest
import requests

from ...client.restful.restful_client import Client as RESTfulClient


class InferenceThread(threading.Thread):
    def __init__(self, prompt, generate_config, client, model):
        super().__init__()
        self._prompt = prompt
        self._generate_config = generate_config
        self._client = client
        self._model = model

    @property
    def stream(self):
        return self._generate_config.get("stream", False)

    def run(self):
        if self.stream:
            results = []
            for res in self._model.chat(
                self._prompt, generate_config=self._generate_config
            ):
                results.append(res)
            assert len(results) > 0
        else:
            res = self._model.chat(self._prompt, generate_config=self._generate_config)
            assert isinstance(res, dict)
            choices = res["choices"]
            assert isinstance(choices, list)
            choice = choices[0]
            assert isinstance(choice, str)
            assert len(choice) > 0


@pytest.fixture
def enable_batch():
    os.environ["XINFERENCE_TRANSFORMERS_ENABLE_BATCHING"] = "1"


def test_continuous_batching(enable_batch, setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"
    client = RESTfulClient(endpoint)

    # launch
    payload = {
        "model_engine": "transformers",
        "model_type": "LLM",
        "model_name": "qwen1.5-chat",
        "quantization": "none",
        "model_format": "pytorch",
        "model_size_in_billions": "0_5",
        # here note that device must be `cpu` for macOS,
        # since torch mps may have some issues for batch tensor calculation
        "device": "cpu",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "qwen1.5-chat"

    model = client.get_model(model_uid_res)

    # test correct
    thread1 = InferenceThread("1+1=3正确吗？", {"stream": True}, client, model)
    thread2 = InferenceThread("中国的首都是哪座城市？", {"stream": False}, client, model)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    # test error generate config
    with pytest.raises(RuntimeError):
        model.chat("你好", generate_config={"max_tokens": 99999999999999999})

    with pytest.raises(RuntimeError):
        model.chat("你好", generate_config={"stream_interval": 0})
