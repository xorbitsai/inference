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
import platform
import sys

import pytest

from .....client import Client


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
        quantization="4-bit",
    )
    assert len(client.list_models()) == 1
    model = client.get_model(model_uid)
    completion = model.chat("write a poem.")
    assert "content" in completion["choices"][0]["message"]
    assert len(completion["choices"][0]["message"]["content"]) != 0
