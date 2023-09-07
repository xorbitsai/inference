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

from ...model.embedding import BUILTIN_EMBEDDING_MODELS


@pytest.mark.asyncio
async def test_restful_api(setup):
    endpoint, _ = setup
    url = f"{endpoint}/v1/models"

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data) == 0

    # launch
    payload = {
        "model_uid": "test_restful_api",
        "model_name": "orca",
        "quantization": "q4_0",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api"

    payload = {
        "model_uid": "test_restful_api",
        "model_name": "orca",
        "quantization": "q4_0",
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 400

    payload = {"model_name": "orca", "quantization": "q4_0"}
    response = requests.post(url, json=payload)
    assert response.status_code == 400

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data) == 1

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
    url = f"{endpoint}/v1/models/test_restful_api"
    response = requests.delete(url)

    # list
    response = requests.get(f"{endpoint}/v1/models")
    response_data = response.json()
    assert len(response_data) == 0

    # delete again
    url = f"{endpoint}/v1/models/test_restful_api"
    response = requests.delete(url)
    assert response.status_code == 400

    # test for model that supports embedding
    url = f"{endpoint}/v1/models"

    payload = {
        "model_uid": "test_restful_api2",
        "model_name": "orca",
        "quantization": "q4_0",
    }

    response = requests.post(url, json=payload)
    response_data = response.json()
    model_uid_res = response_data["model_uid"]
    assert model_uid_res == "test_restful_api2"

    url = f"{endpoint}/v1/embeddings"
    payload = {
        "model": "test_restful_api2",
        "input": "The food was delicious and the waiter...",
    }
    response = requests.post(url, json=payload)
    embedding_res = response.json()

    assert "embedding" in embedding_res["data"][0]

    url = f"{endpoint}/v1/models/test_restful_api2"
    response = requests.delete(url)

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
    print("Done!!!!!")


def test_restful_api_for_embedding(setup):
    model_name = "gte-base"
    model_spec = BUILTIN_EMBEDDING_MODELS[model_name]

    endpoint, _ = setup
    url = f"{endpoint}/v1/models"

    # list
    response = requests.get(url)
    response_data = response.json()
    assert len(response_data) == 0

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
    assert len(response_data) == 1

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
    assert len(response_data) == 0
