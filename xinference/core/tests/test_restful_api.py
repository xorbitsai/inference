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

import pytest
import requests


@pytest.mark.asyncio
async def test_restful_api(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data) == 0

    # launch
    payload = {"model_uid": "test", "model_name": "orca", "quantization": "q4_0"}

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test"

    # # embedding
    # url = f"{endpoint}/v1/embeddings"
    # payload = {
    #     "model": "test",
    #     "input": "The food was delicious and the waiter...",
    # }
    # response = requests.post(url, json=payload)
    # assert response.status_code == 500

    payload = {"model_uid": "test", "model_name": "orca", "quantization": "q4_0"}
    response = requests.post(url, json=payload)
    assert response.status_code == 400

    payload = {"model_name": "orca", "quantization": "q4_0"}
    response = requests.post(url, json=payload)
    assert response.status_code == 400

    # embedding
    url = f"{endpoint}/v1/embeddings"
    payload = {
        "model": "test",
        "input": "The food was delicious and the waiter...",
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 400

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data) == 1

    # describe
    response = requests.get(f"{endpoint}/v1/models/test")
    response_data = response.json()
    assert response_data["model_name"] == "orca"

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
    assert response.status_code == 422

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
    assert response.status_code == 422

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

    # delete
    url = f"{endpoint}/v1/models/test"
    response = requests.delete(url)

    # list
    response = requests.get(f"{endpoint}/v1/models")
    response_data = response.json()
    assert len(response_data) == 0

    # delete again
    url = f"{endpoint}/v1/models/test"
    response = requests.delete(url)
    assert response.status_code == 400

    # test for model that supports embedding
    url = f"{endpoint}/v1/models"

    payload = {
        "model_uid": "test2",
        "model_name": "orca",
        "quantization": "q4_0",
        "embedding": "True",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test2"

    url = f"{endpoint}/v1/embeddings"
    payload = {
        "model": "test2",
        "input": "The food was delicious and the waiter...",
    }
    response = requests.post(url, json=payload)
    embedding_res = response.json()

    assert "embedding" in embedding_res["data"][0]

    url = f"{endpoint}/v1/models/test2"
    response = requests.delete(url)
