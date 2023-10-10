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

import pytest

from ..client import ActorClient, ChatModelHandle, EmbeddingModelHandle, RESTfulClient
from ..constants import XINFERENCE_ENV_MODEL_SRC


@pytest.mark.skipif(os.name == "nt", reason="Skip windows")
def test_client(setup):
    endpoint, _ = setup
    client = ActorClient(endpoint)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(
        model_name="orca", model_size_in_billions=3, quantization="q4_0"
    )
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, ChatModelHandle)

    completion = model.chat("write a poem.")
    assert "content" in completion["choices"][0]["message"]

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(
        model_name="orca",
        model_size_in_billions=3,
        quantization="q4_0",
    )

    model = client.get_model(model_uid=model_uid)

    embedding_res = model.create_embedding("The food was delicious and the waiter...")
    assert "embedding" in embedding_res["data"][0]

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0

    with pytest.raises(ValueError):
        client.launch_model(
            model_name="orca", model_size_in_billions=3, quantization="q4_0", n_gpu=100
        )

    with pytest.raises(ValueError):
        client.launch_model(
            model_name="orca",
            model_size_in_billions=3,
            quantization="q4_0",
            n_gpu="abcd",
        )


def test_client_for_embedding(setup):
    endpoint, _ = setup
    client = ActorClient(endpoint)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(model_name="gte-base", model_type="embedding")
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, EmbeddingModelHandle)

    completion = model.create_embedding("write a poem.")
    assert len(completion["data"][0]["embedding"]) == 768

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0


@pytest.mark.skipif(os.name == "nt", reason="Skip windows")
def test_replica_model(setup):
    endpoint, _ = setup
    client = ActorClient(endpoint)
    assert len(client.list_models()) == 0

    # Windows CI has limited resources, use replica 1
    replica = 1 if os.name == "nt" else 2
    model_uid = client.launch_model(
        model_name="orca",
        model_size_in_billions=3,
        quantization="q4_0",
        replica=replica,
    )
    # Only one model with 2 replica
    assert len(client.list_models()) == 1

    replica_uids = set()
    while len(replica_uids) != replica:
        model = client.get_model(model_uid=model_uid)
        replica_uids.add(model._model_ref.uid)

    embedding_res = model.create_embedding("The food was delicious and the waiter...")
    assert "embedding" in embedding_res["data"][0]

    client2 = RESTfulClient(endpoint)
    info = client2.describe_model(model_uid=model_uid)
    assert info["replica"] == replica

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0


def test_client_custom_model(setup):
    endpoint, _ = setup
    client = ActorClient(endpoint)

    model_regs = client.list_model_registrations(model_type="LLM")
    assert len(model_regs) > 0
    for model_reg in model_regs:
        assert model_reg["is_builtin"]

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
    client.register_model(model_type="LLM", model=model, persist=False)

    new_model_regs = client.list_model_registrations(model_type="LLM")
    assert len(new_model_regs) == len(model_regs) + 1
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom_model":
            custom_model_reg = model_reg
    assert custom_model_reg is not None

    client.unregister_model(model_type="LLM", model_name="custom_model")
    new_model_regs = client.list_model_registrations(model_type="LLM")
    assert len(new_model_regs) == len(model_regs)
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom_model":
            custom_model_reg = model_reg
    assert custom_model_reg is None


def test_client_from_modelscope(setup):
    try:
        os.environ[XINFERENCE_ENV_MODEL_SRC] = "modelscope"

        endpoint, _ = setup
        client = RESTfulClient(endpoint)
        assert len(client.list_models()) == 0

        model_uid = client.launch_model(model_name="tiny-llama")
        assert len(client.list_models()) == 1
        model = client.get_model(model_uid=model_uid)
        completion = model.generate("write a poem.", generate_config={"max_tokens": 5})
        assert "text" in completion["choices"][0]
        assert len(completion["choices"][0]["text"]) > 0
    finally:
        os.environ.pop(XINFERENCE_ENV_MODEL_SRC)
