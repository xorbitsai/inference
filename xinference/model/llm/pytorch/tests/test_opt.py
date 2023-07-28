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

from .....client import Client, GenerateModelHandle


@pytest.mark.asyncio
@pytest.mark.parametrize("quantization", ["8-bit", "4-bit", "none"])
async def test_sync_client(setup, quantization):
    endpoint, _ = setup
    client = Client(endpoint)
    assert len(client.list_models()) == 0

    if quantization == "4-bit":
        with pytest.raises(ValueError):
            client.launch_model(
                model_name="opt",
                model_size_in_billions=1,
                model_format="pytorch",
                quantization=quantization,
                device="cpu",
            )
    else:
        model_uid = client.launch_model(
            model_name="opt",
            model_size_in_billions=1,
            model_format="pytorch",
            quantization=quantization,
            device="cpu",
        )
        assert len(client.list_models()) == 1

        model = client.get_model(model_uid=model_uid)
        assert isinstance(model, GenerateModelHandle)

        completion = model.generate("Once upon a time, there was a very old computer")
        assert "text" in completion["choices"][0]

        client.terminate_model(model_uid=model_uid)
        assert len(client.list_models()) == 0
