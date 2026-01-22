# Copyright 2022-2026 XProbe Inc.
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

from concurrent.futures import ThreadPoolExecutor

import pytest

from .....client import Client
from .....client.restful.restful_client import RESTfulGenerateModelHandle


@pytest.mark.asyncio
@pytest.mark.parametrize("quantization", ["none"])
async def test_opt_pytorch_model(setup, quantization):
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
                "Once upon a time, there was a very old computer",
                generate_config={"max_tokens": 100},
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


@pytest.mark.asyncio
async def test_opt_fp4_model(setup):
    try:
        from transformers import FPQuantConfig  # noqa: F401
    except Exception:
        pytest.skip("FPQuantConfig is not available in transformers.")

    endpoint, _ = setup
    client = Client(endpoint)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(
        model_name="opt",
        model_engine="transformers",
        model_size_in_billions=1,
        model_format="fp4",
        quantization="mxfp4",
        device="cpu",
        quantization_config={
            "pseudoquantization": True,
            "forward_dtype": "mxfp4",
        },
        torch_dtype="bfloat16",
    )
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, RESTfulGenerateModelHandle)

    completion = model.generate(
        "Once upon a time, there was a very old computer",
        generate_config={"max_tokens": 32},
    )
    assert isinstance(completion, dict)
    assert "text" in completion["choices"][0]

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0
