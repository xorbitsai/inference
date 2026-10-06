# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
from ...llm_family import match_llm
from ..core import PytorchModel


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


@pytest.fixture
def opt_fp4_family():
    family = match_llm("opt", "pytorch", 1, "none").copy(deep=True)
    family.model_name = "test-opt-fp4"
    family.model_family = "opt"
    family.virtualenv = None
    family.model_specs[0].model_format = "fp4"
    family.model_specs[0].quantization = "mxfp4"
    return family


def test_opt_fp4_quantization_config(opt_fp4_family):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "FPQuantConfig"):
        pytest.skip("FPQuantConfig is not available in transformers.")
    model = PytorchModel("test-opt-fp4", opt_fp4_family, "/unused")
    config = model.apply_quantization_config({})["quantization_config"]
    assert isinstance(config, transformers.FPQuantConfig)
    assert config.pseudoquantization is True
    assert config.forward_dtype == "mxfp4"


@pytest.mark.asyncio
async def test_opt_fp4_model(request):
    try:
        from transformers import FPQuantConfig  # noqa: F401
    except Exception:
        pytest.skip("FPQuantConfig is not available in transformers.")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("FPQuant requires a CUDA GPU, including pseudoquantization.")
    pytest.importorskip("fp_quant")

    # Check optional support before starting the cluster/API subprocesses.
    opt_fp4_family = request.getfixturevalue("opt_fp4_family")
    endpoint, _ = request.getfixturevalue("setup")
    client = Client(endpoint)
    assert len(client.list_models()) == 0
    client.register_model("LLM", opt_fp4_family.json(), persist=False)

    model_uid = client.launch_model(
        model_name=opt_fp4_family.model_name,
        model_engine="transformers",
        model_size_in_billions=1,
        model_format="fp4",
        quantization="mxfp4",
        device="cuda",
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
    client.unregister_model("LLM", opt_fp4_family.model_name)
