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
import sys
import threading
import time

import pytest
import requests

from ...client.restful.restful_client import Client as RESTfulClient
from ..scheduler import AbortRequestMessage


class BaseThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self._ex = None

    def run_internal(self):
        super().run()

    def run(self):
        try:
            self.run_internal()
        except BaseException as e:
            self._ex = e

    def join(self, timeout=None):
        super().join()
        if self._ex is not None:
            raise self._ex


class InferenceThread(BaseThread):
    def __init__(self, prompt, generate_config, client, model):
        super().__init__()
        self._prompt = prompt
        self._generate_config = generate_config
        self._client = client
        self._model = model
        self._ex = None

    @property
    def stream(self):
        return self._generate_config.get("stream", False)

    def run_internal(self):
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
            choice = choices[0]["text"]
            assert isinstance(choice, str)
            assert len(choice) > 0


class InferenceThreadWithError(InferenceThread):
    def __init__(self, prompt, generate_config, client, model, sleep=None):
        super().__init__(prompt, generate_config, client, model)
        self._sleep = sleep

    def run_internal(self):
        if self._sleep is not None:
            time.sleep(self._sleep)
        if self.stream:
            with pytest.raises(Exception):
                for res in self._model.chat(
                    self._prompt, generate_config=self._generate_config
                ):
                    print(res)
        else:
            with pytest.raises(Exception):
                self._model.chat(self._prompt, generate_config=self._generate_config)


class AbortThread(BaseThread):
    def __init__(self, client, model_uid, request_id, expected_res, sleep=None):
        super().__init__()
        self._client = client
        self._model_uid = model_uid
        self._request_id = request_id
        self._sleep = sleep
        self._expected_res = expected_res

    def run_internal(self):
        if self._sleep is not None:
            time.sleep(self._sleep)
        result = self._client.abort_request(self._model_uid, self._request_id)
        assert result["msg"] == self._expected_res


@pytest.fixture
def enable_batch():
    os.environ["XINFERENCE_TRANSFORMERS_ENABLE_BATCHING"] = "1"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="does not run on windows github CI due to its terrible runtime",
)
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
        # since torch mps may have some issues for batch tensor calculation,
        # see: https://github.com/pytorch/pytorch/issues/122030 and
        # https://github.com/pytorch/pytorch/issues/126862
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

    # test error with other correct requests
    thread1 = InferenceThread("1+1=3正确吗？", {"stream": True}, client, model)
    thread2 = InferenceThread("中国的首都是哪座城市？", {"stream": False}, client, model)
    thread3 = InferenceThreadWithError(
        "猫和狗有什么区别？", {"stream": True, "max_tokens": 99999999999999}, client, model
    )
    thread4 = InferenceThreadWithError(
        "简介篮球的发展历史。", {"stream": False, "stream_interval": 0}, client, model
    )

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

    # test same request ids
    thread1 = InferenceThread(
        "1+1=3正确吗？", {"stream": True, "request_id": "aaabbb"}, client, model
    )
    thread2 = InferenceThreadWithError(
        "中国的首都是哪座城市？", {"stream": False, "request_id": "aaabbb"}, client, model, 0.03
    )
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    # test abort request for stream
    thread1 = InferenceThread(
        "猫和狗有什么区别吗？", {"stream": True, "request_id": "aaabbb"}, client, model
    )
    thread2 = AbortThread(
        client, model_uid_res, "bbbaaa", AbortRequestMessage.NOT_FOUND.name
    )
    thread3 = AbortThread(client, "abcd", "aaabbb", AbortRequestMessage.NO_OP.name)
    thread4 = AbortThread(
        client, model_uid_res, "aaabbb", AbortRequestMessage.DONE.name, 0.01
    )
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    with pytest.raises(Exception):
        thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

    # test abort request for non-stream
    thread1 = InferenceThread("猫和狗有什么区别吗？", {"request_id": "aaabbb"}, client, model)
    thread2 = AbortThread(
        client, model_uid_res, "aaabbb", AbortRequestMessage.DONE.name, 0.01
    )
    thread1.start()
    thread2.start()
    with pytest.raises(Exception):
        thread1.join()
    thread2.join()

    # correctly terminate model
    client.terminate_model(model_uid_res)
