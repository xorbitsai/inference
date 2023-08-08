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
from pathlib import Path
from typing import Union

import pytest

from .....client import Client, GenerateModelHandle
from ... import BUILTIN_LLM_FAMILIES


@pytest.mark.asyncio
@pytest.mark.parametrize("quantization", ["8-bit", "4-bit", "none"])
async def test_opt_pytorch_model(setup, quantization):
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
        assert isinstance(completion, dict)
        assert "text" in completion["choices"][0]

        embedding_res = model.create_embedding(
            "The food was delicious and the waiter..."
        )
        assert "embedding" in embedding_res["data"][0]

        client.terminate_model(model_uid=model_uid)
        assert len(client.list_models()) == 0

        # check for cached revision
        home_address = str(Path.home())
        snapshot_address = (
            home_address
            + "/.cache/huggingface/hub/models--facebook--opt-125m/snapshots"
        )
        actual_revision = os.listdir(snapshot_address)
        model_name = "opt"
        expected_revision: Union[str, None] = ""

        for family in BUILTIN_LLM_FAMILIES:
            if model_name != family.model_name:
                continue
            for spec in family.model_specs:
                expected_revision = spec.model_revision

        assert expected_revision == actual_revision
