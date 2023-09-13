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

from .....client import Client


def test_load_ggmlv3(setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="orca",
        model_size_in_billions=3,
        model_format="ggmlv3",
        quantization="q4_0",
    )
    assert len(client.list_models()) == 1
    model = client.get_model(model_uid)
    completion = model.chat("write a poem.")
    assert "content" in completion["choices"][0]["message"]
    assert len(completion["choices"][0]["message"]["content"]) != 0


def test_gguf(setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="TinyLlama",
        model_size_in_billions=1,
        model_format="ggufv1",
        quantization="Q2_K",
    )
    assert len(client.list_models()) == 1
    model = client.get_model(model_uid)
    completion = model.generate("AI is going to", generate_config={"max_tokens": 5})
    assert "id" in completion
    assert "text" in completion["choices"][0]
    assert len(completion["choices"][0]["text"]) > 0
