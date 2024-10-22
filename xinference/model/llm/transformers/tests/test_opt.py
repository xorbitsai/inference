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
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import pytest

from .....client import Client
from .....client.restful.restful_client import RESTfulGenerateModelHandle
from ... import BUILTIN_LLM_FAMILIES


@pytest.mark.asyncio
@pytest.mark.parametrize("quantization", ["none"])
async def test_opt_pytorch_model(setup, quantization):
    from .....constants import XINFERENCE_CACHE_DIR

    endpoint, _ = setup
    client = Client(endpoint)
    assert len(client.list_models()) == 0

    if quantization == "4-bit":
        with pytest.raises(ValueError):
            client.launch_model(
                model_name="opt",
                model_engine="transformers",
                model_size_in_billions=1,
                model_format="pytorch",
                quantization=quantization,
                device="cpu",
            )
    else:
        model_uid = client.launch_model(
            model_name="opt",
            model_engine="transformers",
            model_size_in_billions=1,
            model_format="pytorch",
            quantization=quantization,
            device="cpu",
        )
        assert len(client.list_models()) == 1

        model = client.get_model(model_uid=model_uid)
        assert isinstance(model, RESTfulGenerateModelHandle)

        # Test concurrent generate is OK.
        def _check():
            completion = model.generate(
                "Once upon a time, there was a very old computer"
            )
            assert isinstance(completion, dict)
            assert "text" in completion["choices"][0]

        results = []
        with ThreadPoolExecutor() as executor:
            for _ in range(3):
                r = executor.submit(_check)
                results.append(r)
        for r in results:
            r.result()

        client.terminate_model(model_uid=model_uid)
        assert len(client.list_models()) == 0

        # check for cached revision
        valid_file = os.path.join(
            XINFERENCE_CACHE_DIR, "opt-pytorch-1b", "__valid_download"
        )
        with open(valid_file, "r") as f:
            actual_revision = json.load(f)["revision"]
        model_name = "opt"
        expected_revision: Union[str, None] = ""  # type: ignore

        for family in BUILTIN_LLM_FAMILIES:
            if model_name != family.model_name:
                continue
            for spec in family.model_specs:
                expected_revision = spec.model_revision

        assert expected_revision == actual_revision
